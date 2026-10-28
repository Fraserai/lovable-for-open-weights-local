"""Ask the original model and the trained one the same question, side by side.

Tinker can sample from a finished checkpoint directly, so this needs no deployment and no
hosting bill. (Fireworks-trained models need a deployment first; not wired yet.)
"""

import json
import re
from pathlib import Path

from .tinker import ensure_ssl_certificates

MAX_TOKENS = 700
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _renderer(name: str, tokenizer, base_model: str):
    from tinker_cookbook import renderers

    return renderers.get_renderer(name, tokenizer, model_name=base_model)


def _text(renderer, tokens) -> str:
    message, _ = renderer.parse_response(tokens)
    content = message["content"]
    if not isinstance(content, str):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return _THINK.sub("", content).strip()


def compare(run_dir: Path, message: str, history: list[dict] | None = None) -> dict:
    """Returns {"original": ..., "trained": ...} for one user message."""
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    ensure_ssl_certificates()
    state = json.loads((run_dir / "run.json").read_text())
    if state.get("job") != "tinker" or not state.get("sampler_path"):
        raise RuntimeError("Talking to the model works for finished Tinker runs for now.")

    base = state["base_model"]
    tokenizer = get_tokenizer(base)
    trained_fmt = state["tinker_plan"]["renderer"]
    trained_renderer = _renderer(trained_fmt, tokenizer, base)
    # The trained model learned to answer directly. Ask the original for a direct answer too
    # (its "no thinking" chat format, where one exists), so the comparison is fair.
    try:
        original_renderer = _renderer(f"{trained_fmt}_disable_thinking", tokenizer, base)
    except Exception:  # noqa: BLE001 - this model family has no such variant
        original_renderer = trained_renderer

    messages = [{"role": "system", "content": state["system_prompt"]}, *(history or []),
                {"role": "user", "content": message}]
    service = tinker.ServiceClient()
    jobs = {
        "original": (service.create_sampling_client(base_model=base), original_renderer),
        "trained": (service.create_sampling_client(model_path=state["sampler_path"]), trained_renderer),
    }
    futures = {
        name: (client.sample(
            prompt=r.build_generation_prompt(messages), num_samples=1,
            sampling_params=tinker.types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.3,
                                                       stop=r.get_stop_sequences()),
        ), r)
        for name, (client, r) in jobs.items()
    }
    return {name: _text(r, fut.result().sequences[0].tokens) for name, (fut, r) in futures.items()}
