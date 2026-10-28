"""Trainer 2: LoRA fine-tuning on Tinker (Thinking Machines).

Ported from Post Trainer's tinker bridge, which trained Qwen3.5-4B end to end on a real
Tinker account (2026-08-20). The hard-won rules from that project are kept here:

- Training and publishing are separate steps. Training is the paid, irreversible part;
  publishing just moves bytes and must be retryable without retraining.
- Never train in a chat format we guessed. A wrong renderer doesn't error, it quietly
  produces a worse model, so an unverified model is refused before spending anything.
- Hidden settings are ours to get right: calibrated learning rate, epochs sized to the
  dataset, the model's full context, and a warning when examples will be truncated.

tinker / tinker-cookbook are imported lazily (they pull in torch). Install with:
    uv sync   (the "tinker" dependency group is on by default)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

LORA_RANK = 32  # room for domain knowledge, not just style; Post Trainer's rank 8 under-learned
BATCH_SIZE = 4
CHECKPOINT_EVERY_N_STEPS = 20
FALLBACK_LEARNING_RATE = 4.0e-4  # only for families Tinker hasn't calibrated
TARGET_TOTAL_STEPS = 250  # small datasets get several passes, large ones one
MAX_EPOCHS = 4
FALLBACK_SEQUENCE_LENGTH = 32_768  # lowest limit across Tinker's catalogue, so never over-asks

# Models Tinker trains but the installed cookbook's renderer table doesn't know yet.
# Each entry needs a note on how it was checked.
_RENDERER_OVERRIDES: dict[str, str] = {
    # Post Trainer, 2026-08: rendering the same conversation with "qwen3_5" for Qwen3.8-27B is
    # token-identical to Qwen3.6-27B, which the official table maps to "qwen3_5".
    "Qwen/Qwen3.8-27B": "qwen3_5",
}

CHECKPOINT_RECORD = "checkpoint.json"

Progress = Callable[[str], None]


class TinkerError(RuntimeError):
    pass


def ensure_ssl_certificates() -> None:
    """Point stdlib SSL at certifi's CA bundle when the system has none.

    python.org macOS builds ship without CA certs. httpx/requests vendor certifi so training
    works, then weights.download (plain urllib) fails at the very end, after the money is spent.
    """
    import ssl

    if os.environ.get("SSL_CERT_FILE") or ssl.get_default_verify_paths().cafile:
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def resolve_renderer(base_model: str) -> str:
    from tinker_cookbook import model_info

    try:
        return model_info.get_recommended_renderer_name(base_model)
    except Exception:  # noqa: BLE001 - missing from this cookbook version's table
        pass
    if base_model in _RENDERER_OVERRIDES:
        return _RENDERER_OVERRIDES[base_model]
    raise TinkerError(
        f"Can't train {base_model} on Tinker yet: no verified chat format for it. Training would look "
        "successful but produce a worse model, so nothing was started or charged."
    )


def resolve_learning_rate(base_model: str) -> float:
    from tinker_cookbook import hyperparam_utils

    try:
        return hyperparam_utils.get_lr(base_model, is_lora=True)
    except Exception:  # noqa: BLE001 - family not calibrated
        return FALLBACK_LEARNING_RATE


def resolve_epochs(example_count: int, batch_size: int = BATCH_SIZE) -> int:
    steps_per_epoch = max(1, example_count // batch_size)
    needed = -(-TARGET_TOTAL_STEPS // steps_per_epoch)  # ceil
    return max(1, min(MAX_EPOCHS, needed))


def resolve_sequence_length(base_model: str) -> int:
    import tinker

    try:
        caps = tinker.ServiceClient().get_server_capabilities()
        limits = [m.max_context_length for m in caps.supported_models
                  if m.model_name.split(":")[0] == base_model]
        if limits:
            return max(limits)
    except Exception:  # noqa: BLE001
        pass
    return FALLBACK_SEQUENCE_LENGTH


def expand_per_assistant_turn(src: Path, dst: Path) -> int:
    """One training example per assistant turn, each ending at that turn.

    Trained with LAST_ASSISTANT_MESSAGE, every reply is taught against exactly the context
    that precedes it at inference. Training whole conversations on all assistant turns is
    subtly wrong for renderers without the cookbook's "extension property" (qwen3_5,
    nemotron3, ...): it warns, completes, and yields a worse model. Single-turn data is
    unchanged. Streams line by line, so file size is bounded by disk, not memory.
    """
    count = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            for i, m in enumerate(messages):
                if m.get("role") == "assistant":
                    fout.write(json.dumps({"messages": messages[: i + 1]}, ensure_ascii=False) + "\n")
                    count += 1
    return count


def truncated_share(dataset: Path, base_model: str, max_length: int, sample: int = 300) -> float:
    """Share of sampled examples longer than max_length tokens (truncation is otherwise silent)."""
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    try:
        tokenizer = get_tokenizer(base_model)
    except Exception:  # noqa: BLE001
        return 0.0
    over = seen = 0
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            if seen >= sample:
                break
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            seen += 1
            if sum(len(tokenizer.encode(m["content"])) for m in messages) > max_length:
                over += 1
    return over / seen if seen else 0.0


def plan(train_jsonl: Path, work_dir: Path, base_model: str, s: dict | None = None) -> dict:
    """Everything decided before any money is spent. Also refuses unverified models.

    `s` holds the Train page's settings this trainer honours (settings.SUPPORTED["tinker"]);
    anything unset is sized for the data here.
    """
    s = s or {}
    ensure_ssl_certificates()
    renderer = resolve_renderer(base_model)
    work_dir.mkdir(parents=True, exist_ok=True)
    expanded = work_dir / "train.expanded.jsonl"
    examples = expand_per_assistant_turn(train_jsonl, expanded)
    batch = min(s.get("batch_size") or BATCH_SIZE, examples)  # the builder drops a trailing partial batch
    per_epoch = max(1, examples // batch)
    max_steps = s.get("max_steps")
    epochs = -(-max_steps // per_epoch) if max_steps else s.get("epochs") or resolve_epochs(examples, batch)
    max_length = min(s.get("context_length") or 1 << 30, resolve_sequence_length(base_model))
    return {
        "renderer": renderer,
        "dataset": str(expanded),
        "examples": examples,
        "batch_size": batch,
        "epochs": epochs,
        "steps": min(per_epoch * epochs, max_steps or per_epoch * epochs),
        "max_steps": max_steps,
        "learning_rate": s.get("learning_rate") or resolve_learning_rate(base_model),
        "lr_scheduler": s.get("lr_scheduler") or "linear",
        "lora_rank": s.get("lora_rank") or LORA_RANK,
        "max_length": max_length,
        "truncated_share": truncated_share(expanded, base_model, max_length),
    }


# Periodic checkpoints exist so a crashed run can resume; they cost $0.10/GB-month while they exist.
# A day is plenty to resume from. The final checkpoint is kept (the cookbook stores it without expiry).
PERIODIC_CHECKPOINT_TTL_SECONDS = 24 * 3600


def train(work_dir: Path, base_model: str, p: dict, on_progress: Progress = print, *,
          recipe_name: str = "trainloop_sft", on_step: Callable[[int], None] | None = None) -> str:
    """Runs the paid training loop and returns the checkpoint's sampler path.

    Re-running with the same `work_dir` resumes from the last saved checkpoint (the cookbook reads
    `work_dir/logs`), so a worker that dies mid-run can pick the job up again without starting over.
    `recipe_name` is attached to the Tinker session's user_metadata, which Tinker's billing usage
    reports per session: the hosted service passes its job id here to match cost to jobs.
    """
    import tinker
    from tinker_cookbook import checkpoint_utils, renderers
    from tinker_cookbook.supervised import train as sft
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    ensure_ssl_certificates()
    tinker.ServiceClient()  # fail on bad credentials before any training time

    builder = FromConversationFileBuilder(
        file_path=p["dataset"],
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=base_model,
            renderer_name=p["renderer"],
            max_length=p["max_length"],
            batch_size=p["batch_size"],
            train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        ),
    )
    # A real directory, never a temp one: if publishing fails, the checkpoint record must survive.
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config = sft.Config(
        log_path=str(log_dir),
        model_name=base_model,
        recipe_name=recipe_name,  # tags the session in Tinker's console and billing usage
        renderer_name=p["renderer"],
        dataset_builder=builder,
        learning_rate=p["learning_rate"],
        lr_schedule=p.get("lr_scheduler", "linear"),
        num_epochs=p["epochs"],
        max_steps=p.get("max_steps"),
        lora_rank=p["lora_rank"],
        save_every=CHECKPOINT_EVERY_N_STEPS,
        ttl_seconds=PERIODIC_CHECKPOINT_TTL_SECONDS,
        eval_every=0,
    )

    async def run() -> None:
        task = asyncio.create_task(sft.main(config))
        metrics = log_dir / "metrics.jsonl"  # the cookbook appends one line per completed step
        last = -1
        while not task.done():
            try:
                steps = sum(1 for line in metrics.open() if line.strip()) if metrics.exists() else 0
            except OSError:
                steps = last
            if steps != last:
                on_progress(f"step {steps} of ~{p['steps']}")
                if on_step:
                    on_step(steps)
                last = steps
            await asyncio.wait([task], timeout=3)
        await task  # surface the real exception, if any

    asyncio.run(run())

    record = checkpoint_utils.get_last_checkpoint(str(log_dir), required_key="sampler_path")
    if record is None or not record.sampler_path:
        raise TinkerError("Training finished but Tinker recorded no publishable checkpoint.")
    (work_dir / CHECKPOINT_RECORD).write_text(
        json.dumps({"sampler_path": record.sampler_path, "base_model": base_model}))
    return record.sampler_path


def export_adapter(work_dir: Path, dest: Path) -> Path:
    """Downloads the trained weights and converts them to a standard PEFT LoRA adapter.

    Re-runnable against a finished run without spending anything on training.
    """
    from tinker_cookbook import weights

    ensure_ssl_certificates()
    try:
        rec = json.loads((work_dir / CHECKPOINT_RECORD).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise TinkerError("No finished Tinker checkpoint recorded for this run.") from e

    raw = weights.download(tinker_path=rec["sampler_path"], output_dir=str(work_dir / "download"))
    if dest.exists():
        shutil.rmtree(dest)  # build_lora_adapter requires the output path not to exist
    try:
        weights.build_lora_adapter(base_model=rec["base_model"], adapter_path=raw, output_path=str(dest))
    except Exception as e:
        raise TinkerError(
            f"Training succeeded, but converting the adapter failed: {e}. The trained weights are safe at "
            f"{rec['sampler_path']}; run fetch again to retry without retraining."
        ) from e
    return dest
