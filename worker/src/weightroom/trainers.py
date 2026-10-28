"""The trainer behind the service: Tinker in production, a fake one that spends nothing for local runs and tests.

TinkerTrainer wraps trainloop.tinker, which keeps the hard-won rules: refuse models without a verified
chat format, Tinker's calibrated learning rate, epochs sized to the data, truncation warnings, and
resume from the last checkpoint when a run is restarted in the same work directory.
"""

import json
import random
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from . import config
from .errors import WRError

Answerer = Callable[[list[list[dict]]], list[str]]
VERIFIED_EXAMPLES = "Qwen/Qwen3.5-4B, Qwen/Qwen3-8B, Qwen/Qwen3.5-9B, moonshotai/Kimi-K2.6"


class TinkerTrainer:
    name = "tinker"

    def __init__(self):
        self._clients: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()

    def plan(self, train_path: Path, work_dir: Path, base_model: str, s: dict | None = None) -> dict:
        from trainloop import tinker

        try:
            return tinker.plan(train_path, work_dir, base_model, s or {})
        except tinker.TinkerError as e:
            raise WRError("model_not_supported", str(e),
                          f"Pick a model with a verified chat format, e.g. {VERIFIED_EXAMPLES}.") from e

    def train(self, work_dir: Path, base_model: str, p: dict, job_id: str, on_step: Callable[[int], None]) -> str:
        from trainloop import tinker

        return tinker.train(work_dir, base_model, p, on_progress=lambda _msg: None,
                            recipe_name=f"weightroom:{job_id}", on_step=on_step)

    def answerers(self, base_model: str, renderer: str, sampler_path: str | None, max_tokens: int,
                  work_dir: Path | None = None) -> dict[str, Answerer]:
        from trainloop.score import tinker_answerers

        return tinker_answerers({"base_model": base_model, "tinker_plan": {"renderer": renderer},
                                 "sampler_path": sampler_path}, max_tokens)

    def export(self, work_dir: Path, sampler_path: str, base_model: str, dest: Path) -> Path:
        from trainloop import tinker

        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / tinker.CHECKPOINT_RECORD).write_text(json.dumps({"sampler_path": sampler_path,
                                                                    "base_model": base_model}))
        return tinker.export_adapter(work_dir, dest)

    def _client(self, base_model: str, renderer_name: str, sampler_path: str):
        """Sampling clients are cached: building one loads a tokenizer."""
        with self._lock:
            if sampler_path in self._clients:
                self._clients.move_to_end(sampler_path)
                return self._clients[sampler_path]
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        from trainloop.tinker import ensure_ssl_certificates

        ensure_ssl_certificates()
        renderer = renderers.get_renderer(renderer_name, get_tokenizer(base_model), model_name=base_model)
        client = tinker.ServiceClient().create_sampling_client(model_path=sampler_path)
        with self._lock:
            self._clients[sampler_path] = (client, renderer)
            while len(self._clients) > 20:
                self._clients.popitem(last=False)
        return client, renderer

    def chat(self, base_model: str, renderer_name: str, sampler_path: str, messages: list[dict], max_tokens: int,
             temperature: float) -> tuple[str, int, int]:
        """One reply from the trained model, rendered exactly as it was trained (not a guessed chat template)."""
        import tinker

        from trainloop.playground import _text

        client, renderer = self._client(base_model, renderer_name, sampler_path)
        prompt = renderer.build_generation_prompt(messages)
        result = client.sample(prompt=prompt, num_samples=1, sampling_params=tinker.types.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, stop=renderer.get_stop_sequences())).result()
        tokens = result.sequences[0].tokens
        prompt_tokens = getattr(prompt, "length", None)
        if prompt_tokens is None:
            prompt_tokens = len(prompt.to_ints()) if hasattr(prompt, "to_ints") else 0
        return _text(renderer, tokens), int(prompt_tokens), len(tokens)


class FakeTrainer:
    """Behaves like Tinker without spending anything. Used locally and in tests."""
    name = "fake"
    fail_at_step: int | None = None  # tests: raise once at this step, to exercise restarts
    unsupported_prefix = "unsupported/"

    def plan(self, train_path: Path, work_dir: Path, base_model: str, s: dict | None = None) -> dict:
        s = s or {}
        if base_model.startswith(self.unsupported_prefix):
            raise WRError("model_not_supported", f"Can't train {base_model} on Tinker yet: no verified chat format.",
                          f"Pick a model with a verified chat format, e.g. {VERIFIED_EXAMPLES}.")
        work_dir.mkdir(parents=True, exist_ok=True)
        examples = sum(1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip())
        batch = min(s.get("batch_size") or 4, max(1, examples))
        per_epoch = max(1, examples // batch)
        epochs = s.get("epochs") or max(1, min(4, -(-250 // per_epoch)))
        return {"renderer": "fake", "dataset": str(train_path), "examples": examples, "batch_size": batch,
                "epochs": epochs, "steps": per_epoch * epochs, "max_steps": None,
                "learning_rate": s.get("learning_rate") or 2.5e-4, "lr_scheduler": "linear",
                "lora_rank": s.get("lora_rank") or 32, "max_length": 65536, "truncated_share": 0.0}

    def train(self, work_dir: Path, base_model: str, p: dict, job_id: str, on_step: Callable[[int], None]) -> str:
        progress = work_dir / "fake_progress.json"
        step = json.loads(progress.read_text())["step"] if progress.exists() else 0  # resume
        while step < p["steps"]:
            step += 1
            if self.fail_at_step is not None and step == self.fail_at_step:
                type(self).fail_at_step = None
                raise ConnectionError("fake trainer: connection dropped")
            time.sleep(config.get().fake_step_seconds)
            progress.write_text(json.dumps({"step": step}))
            on_step(step)
        return f"tinker://fake-{job_id}:train:0/sampler_weights/final"

    def answerers(self, base_model: str, renderer: str, sampler_path: str | None, max_tokens: int,
                  work_dir: Path | None = None) -> dict[str, Answerer]:
        references: dict[str, str] = {}
        if work_dir and (work_dir / "test.jsonl").exists():
            for line in (work_dir / "test.jsonl").read_text().splitlines():
                if line.strip():
                    msgs = json.loads(line)["messages"]
                    references[json.dumps(msgs[:-1])] = msgs[-1]["content"]
        labels = sorted(set(references.values())) or ["unknown"]

        def original(convs):
            rng = random.Random(1)
            return [references.get(json.dumps(c)) if rng.random() < 0.6 else rng.choice(labels) for c in convs]

        def trained(convs):
            rng = random.Random(2)
            return [references.get(json.dumps(c)) if rng.random() < 0.85 else rng.choice(labels) for c in convs]

        out = {"original": lambda convs: [a or "unknown" for a in original(convs)]}
        if sampler_path:
            out["trained"] = lambda convs: [a or "unknown" for a in trained(convs)]
        return out

    def export(self, work_dir: Path, sampler_path: str, base_model: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base_model, "r": 32,
                                                              "lora_alpha": 32, "peft_type": "LORA"}))
        (dest / "adapter_model.safetensors").write_bytes(b"fake adapter weights")
        return dest

    def chat(self, base_model: str, renderer_name: str, sampler_path: str, messages: list[dict], max_tokens: int,
             temperature: float) -> tuple[str, int, int]:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        reply = f"(fake model) you said: {last}"[: max_tokens * 4]
        return reply, sum(len(m["content"].split()) for m in messages), len(reply.split())


_trainer = None


def trainer():
    global _trainer
    if _trainer is None:
        _trainer = TinkerTrainer() if config.get().trainer == "tinker" else FakeTrainer()
    return _trainer


def reset() -> None:
    global _trainer
    _trainer = None
