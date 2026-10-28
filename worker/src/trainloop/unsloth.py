"""Trainer 3: Unsloth on Hugging Face Jobs (cloud GPUs, billed by the second).

Unsloth's library (Apache-2.0) does the training; jobs/unsloth_sft.py runs it on a GPU we rent from
Hugging Face and pushes the adapter straight to a private repo on the user's account. Unsloth Studio
(AGPL-3.0) is not used or copied.

Models are named `unsloth:<hugging face id>`, e.g. `unsloth:Qwen/Qwen3.5-4B`.

Unlike Tinker, the job keeps running in the cloud if this process stops: `trainloop status`
reattaches to it.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .tinker import expand_per_assistant_turn

PREFIX = "unsloth:"
SCRIPT = Path(__file__).with_name("jobs") / "unsloth_sft.py"

LORA_RANK = 32  # same as Tinker, so results compare
BATCH_SIZE = 4
LEARNING_RATE = 2e-4  # Unsloth's standard for LoRA with alpha = rank
MAX_SEQ_LENGTH = 32768  # the job script refuses (never cuts) examples longer than this
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TARGET_TOTAL_STEPS = 250
MAX_EPOCHS = 4

# 16-bit LoRA (Unsloth advises against 4-bit for MoE models, and it keeps quality level with Tinker).
# Weights at 2 bytes a parameter, plus room for the adapter, optimizer and activations.
WEIGHT_OVERHEAD = 1.3
ACTIVATION_GB = 6
# Modern bf16 GPUs only; T4 has no bf16 and L4 is too slow to be worth the saving.
GPU_MODELS = ("A10G", "L40S", "A100", "RTX PRO 6000", "H200")
# The job's time limit, which also caps the bill. Deliberately generous: setup (install, model
# download) plus training at a throughput well below what these GPUs manage.
SETUP_MINUTES = 20
ASSUMED_TOKENS_PER_SECOND = 300
MAX_HOURS = 24

TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}
Event = Callable[[dict], None]


class UnslothError(RuntimeError):
    pass


def hf_id(base_model: str) -> str:
    return base_model.removeprefix(PREFIX)


def _gb(text: str | None) -> float:
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return 0.0


def pick_gpu(params: int, hardware: list, bytes_per_param: float = 2) -> dict:
    """The cheapest single bf16 GPU that holds the model for LoRA training (16-bit, or 4-bit for QLoRA)."""
    weights_gb = params * bytes_per_param / 1e9
    need = weights_gb * WEIGHT_OVERHEAD + ACTIVATION_GB
    fits = [h for h in hardware
            if h.accelerator and h.accelerator.model in GPU_MODELS and str(h.accelerator.quantity) == "1"
            and _gb(h.accelerator.vram) >= need and _gb(h.ram) >= weights_gb * 1.2]
    if not fits:
        biggest = max((_gb(h.accelerator.vram) for h in hardware
                       if h.accelerator and str(h.accelerator.quantity) == "1"), default=0)
        raise UnslothError(
            f"This model needs about {need:.0f} GB of GPU memory to train, more than the largest single GPU "
            f"({biggest:.0f} GB). Multi-GPU training isn't wired up yet, so nothing was started or charged.")
    best = min(fits, key=lambda h: h.unit_cost_usd)
    per_hour = best.unit_cost_usd * (60 if best.unit_label == "minute" else 3600 if best.unit_label == "second" else 1)
    return {"flavor": best.name, "gpu": f"{best.accelerator.model} ({best.accelerator.vram})",
            "per_hour": round(per_hour, 2), "need_gb": round(need)}


def resolve_epochs(examples: int, batch_size: int = BATCH_SIZE) -> int:
    steps_per_epoch = max(1, math.ceil(examples / batch_size))
    return max(1, min(MAX_EPOCHS, math.ceil(TARGET_TOTAL_STEPS / steps_per_epoch)))


def time_limit_hours(tokens: int) -> float:
    hours = (SETUP_MINUTES * 60 + tokens / ASSUMED_TOKENS_PER_SECOND) / 3600
    return min(MAX_HOURS, math.ceil(hours * 2) / 2)  # round up to the half hour


def plan(train_jsonl: Path, work_dir: Path, base_model: str, est_tokens: int, token: str,
         s: dict | None = None) -> dict:
    """Everything decided before any money is spent: GPU, settings, and the most it can cost.

    `s` holds the Train page's settings (all of them apply here: we run the training script)."""
    from huggingface_hub import HfApi

    s = s or {}
    api = HfApi(token=token)
    repo = hf_id(base_model)
    try:
        info = api.model_info(repo, expand=["safetensors"])
    except Exception as e:  # noqa: BLE001 - gated, missing, or a typo
        raise UnslothError(f"Can't read {repo} on Hugging Face ({e}).") from e
    params = info.safetensors.total if info.safetensors else None
    if not params:
        raise UnslothError(f"{repo} doesn't publish its size (no safetensors), so no GPU can be sized for it.")
    qlora = s.get("method") == "qlora"
    gpu = pick_gpu(params, api.list_jobs_hardware(), 0.6 if qlora else 2)

    inputs = work_dir / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    examples = expand_per_assistant_turn(train_jsonl, inputs / "train.jsonl")
    batch, accum = s.get("batch_size") or BATCH_SIZE, s.get("grad_accum") or 1
    per_epoch = math.ceil(examples / (batch * accum))
    max_steps = s.get("max_steps")
    epochs = math.ceil(max_steps / per_epoch) if max_steps else s.get("epochs") or resolve_epochs(examples, batch * accum)
    steps = min(per_epoch * epochs, max_steps or per_epoch * epochs)
    hours = time_limit_hours(round(est_tokens * steps / per_epoch))
    rank = s.get("lora_rank") or LORA_RANK
    return {
        **gpu,
        "params": params,
        "examples": examples,
        "batch_size": batch,
        "grad_accum": accum,
        "epochs": epochs,
        "steps": steps,
        "max_steps": max_steps,
        "learning_rate": s.get("learning_rate") or LEARNING_RATE,
        "lora_rank": rank,
        "lora_alpha": s.get("lora_alpha") or rank,
        "lora_dropout": s.get("lora_dropout", 0.0),
        "target_modules": s.get("target_modules") or list(TARGET_MODULES),
        "lora_variant": s.get("lora_variant") or "lora",
        "method": "qlora" if qlora else "lora",
        "max_seq_length": s.get("context_length") or MAX_SEQ_LENGTH,
        "optimizer": s.get("optimizer") or "adamw_8bit",
        "lr_scheduler": s.get("lr_scheduler") or "linear",
        "weight_decay": s.get("weight_decay", 0.0),
        "warmup_steps": s.get("warmup_steps"),
        "seed": s.get("seed", 3407),
        "gradient_checkpointing": s.get("gradient_checkpointing") or "unsloth",
        "time_limit_hours": hours,
        "max_cost": round(hours * gpu["per_hour"], 2),
    }


def launch(work_dir: Path, base_model: str, p: dict, output_repo: str, token: str, run: str,
           held_out: Path | None = None):
    """Uploads the training data to the user's private Jobs storage and starts the GPU job."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=token)
    inputs = work_dir / "input"
    args = ["--data", "/input/train.jsonl", "--base-model", hf_id(base_model), "--output-repo", output_repo,
            "--epochs", str(p["epochs"]), "--learning-rate", str(p["learning_rate"]),
            "--lora-rank", str(p["lora_rank"]), "--batch-size", str(p["batch_size"])]
    # Settings from the Train page; older plans without them keep the script's defaults
    for key, flag in (("max_steps", "--max-steps"), ("grad_accum", "--grad-accum"), ("lora_alpha", "--lora-alpha"),
                      ("lora_dropout", "--lora-dropout"), ("max_seq_length", "--max-seq-length"),
                      ("optimizer", "--optim"), ("lr_scheduler", "--lr-scheduler"), ("weight_decay", "--weight-decay"),
                      ("warmup_steps", "--warmup-steps"), ("seed", "--seed"),
                      ("gradient_checkpointing", "--gradient-checkpointing")):
        if p.get(key) is not None:
            args += [flag, str(p[key])]
    if p.get("target_modules"):
        args += ["--target-modules", ",".join(p["target_modules"])]
    if p.get("lora_variant") == "rslora":
        args.append("--rslora")
    if p.get("method") == "qlora":
        args.append("--load-in-4bit")
    if held_out:
        shutil.copyfile(held_out, inputs / "held_out.jsonl")
        args += ["--eval", "/input/held_out.jsonl"]
    # /input, not /data: run_uv_job ships the script itself through a bucket mounted at /data.
    # Both live in the private `jobs-artifacts` bucket on the user's account.
    volume = api.sync_job_volume(inputs, "/input")
    try:
        return api.run_uv_job(
            str(SCRIPT), script_args=args, flavor=p["flavor"], timeout=f"{p['time_limit_hours']}h",
            secrets={"HF_TOKEN": token}, volumes=[volume], name=f"trainloop-{run}", labels={"trainloop-run": run},
        )
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 402:
            raise UnslothError("Your Hugging Face account has no prepaid credit for GPU jobs. Add some at "
                               "https://huggingface.co/settings/billing and start training again. "
                               "Nothing was charged.") from e
        raise UnslothError(f"Hugging Face wouldn't start the job: {e}. Nothing was charged.") from e


def follow(job_id: str, work_dir: Path, token: str, on_event: Event) -> dict:
    """Streams the job's log until it ends. Replays from the start, so it can reattach any time.

    Writes one line per training step to metrics.jsonl and the raw log to job.log. Returns
    {"stage", "pushed", "error", "eval"}.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    result: dict = {"stage": None, "pushed": None, "error": None, "eval": None}
    while True:
        with (work_dir / "metrics.jsonl").open("w") as metrics, (work_dir / "job.log").open("w") as log:
            try:
                for line in api.fetch_job_logs(job_id=job_id, follow=True):
                    log.write(line.rstrip("\n") + "\n")
                    log.flush()
                    event = _parse(line)
                    if not event:
                        continue
                    if event["event"] == "step":
                        metrics.write(json.dumps(event) + "\n")
                        metrics.flush()
                    elif event["event"] == "pushed":
                        result["pushed"] = event["repo"]
                    elif event["event"] == "error":
                        result["error"] = event["message"]
                    elif event["event"] == "eval":
                        result["eval"] = event
                    on_event(event)
            except Exception:  # noqa: BLE001 - a dropped stream; check the job and reattach
                pass
        job = api.inspect_job(job_id=job_id)
        stage = str(getattr(job.status.stage, "value", job.status.stage))
        if stage in TERMINAL:
            result["stage"] = stage
            if not result["error"] and stage != "COMPLETED":
                result["error"] = job.status.message or _last_lines(work_dir / "job.log")
            return result
        time.sleep(10)


def _parse(line: str) -> dict | None:
    i = line.find("TRAINLOOP ")
    if i < 0:
        return None
    try:
        event = json.loads(line[i + len("TRAINLOOP "):])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) and "event" in event else None


def _last_lines(log: Path, n: int = 5) -> str:
    try:
        lines = [l for l in log.read_text().splitlines() if l.strip()]
    except OSError:
        return "The job stopped without a message."
    return " / ".join(lines[-n:]) or "The job stopped without a message."


def download(repo: str, dest: Path, token: str) -> list[Path]:
    from huggingface_hub import snapshot_download

    if dest.exists():
        shutil.rmtree(dest)
    snapshot_download(repo_id=repo, local_dir=dest, token=token)
    return [f for f in dest.rglob("*") if f.is_file() and ".cache" not in f.parts]
