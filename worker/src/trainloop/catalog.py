"""One merged catalog: which open models can be fine-tuned on which trainer.

Sources (fetched live, never hard-coded):
  - Fireworks: docs.fireworks.ai/fine-tuning/models.md embeds a JS object
    `const TRAINING_SHAPES_CATALOG = {...}` whose "models" array lists every
    base model with its managed SFT/DPO LoRA flags, training context and params.
  - Tinker: tinker-docs.thinkingmachines.ai/tinker/models.json, a JSON array;
    every model on it is LoRA-trainable.
"""

import json
import re

import httpx

FIREWORKS_MODELS_URL = "https://docs.fireworks.ai/fine-tuning/models.md"
TINKER_MODELS_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
_CATALOG_MARKER = "const TRAINING_SHAPES_CATALOG = "
_CTX = re.compile(r"^(\d+)K$")


class CatalogError(RuntimeError):
    pass


def _get(url: str, transport: httpx.BaseTransport | None) -> str:
    with httpx.Client(transport=transport, timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _context_to_tokens(ctx) -> int | None:
    """Convert a context string like "64K" -> 65536, "256K" -> 262144; ints pass through."""
    if isinstance(ctx, bool):
        return None
    if isinstance(ctx, int):
        return ctx
    m = _CTX.match(str(ctx or "").strip())
    return int(m.group(1)) * 1024 if m else None


def _fireworks_models(text: str) -> list[dict]:
    """Extract the TRAINING_SHAPES_CATALOG object and return its "models" array."""
    i = text.find(_CATALOG_MARKER)
    if i < 0:
        raise CatalogError("Fireworks page format changed: no 'const TRAINING_SHAPES_CATALOG = {' found")
    start = i + len(_CATALOG_MARKER)
    if start >= len(text) or text[start] != "{":
        raise CatalogError("Fireworks page format changed: TRAINING_SHAPES_CATALOG does not start with '{'")

    depth = 0
    end = None
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end is None:
        raise CatalogError("Fireworks page format changed: no closing brace for TRAINING_SHAPES_CATALOG")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise CatalogError(f"Fireworks page format changed: TRAINING_SHAPES_CATALOG is no longer valid JSON ({e})") from e
    models = data.get("models")
    if not isinstance(models, list):
        raise CatalogError("Fireworks page format changed: TRAINING_SHAPES_CATALOG has no 'models' array")
    return models


def _hf_repo(url) -> str | None:
    """https://huggingface.co/thinkingmachines/Inkling -> thinkingmachines/Inkling."""
    prefix = "https://huggingface.co/"
    return url[len(prefix):] if isinstance(url, str) and url.startswith(prefix) else None


def _fireworks_entries(transport: httpx.BaseTransport | None) -> list[dict]:
    entries = []
    for m in _fireworks_models(_get(FIREWORKS_MODELS_URL, transport)):
        dpo = m.get("managedDpoLora")
        entries.append({
            "trainer": "fireworks",
            "trainer_model_id": f"accounts/fireworks/models/{m['id']}",
            "display_name": m.get("model") or m["id"],
            "hf_repo": None,
            "params": m.get("paramCount"),
            "train_context_tokens": m.get("trainCtx"),
            "sft_lora": bool(m.get("managedSftLora")),
            "dpo": dpo if isinstance(dpo, bool) else None,
            "train_price": None,
            "prefill_price": None,
            "sample_price": None,
        })
    return entries


def _tinker_entries(transport: httpx.BaseTransport | None) -> list[dict]:
    return [{
        "trainer": "tinker",
        "trainer_model_id": m["tinker_id"],
        "display_name": m["name"],
        "hf_repo": _hf_repo(m.get("url")),
        "params": m.get("params"),
        "train_context_tokens": _context_to_tokens(m.get("context")),
        "sft_lora": True,
        "dpo": None,  # per-model DPO support is not published in models.json
        "train_price": m.get("train"),
        "prefill_price": m.get("prefill"),
        "sample_price": m.get("sample"),
    } for m in json.loads(_get(TINKER_MODELS_URL, transport))]


def build_catalog(transport: httpx.BaseTransport | None = None, failures: dict[str, str] | None = None) -> list[dict]:
    """Fetch both trainer sources and return one entry per model per trainer.

    With `failures`, a source that can't be read is recorded there (trainer -> reason) and the other
    trainer's models are still returned; without it, the first failure raises.
    """
    entries: list[dict] = []
    for trainer, fetch in (("fireworks", _fireworks_entries), ("tinker", _tinker_entries)):
        try:
            entries += fetch(transport)
        except (httpx.HTTPError, CatalogError, ValueError, KeyError, TypeError) as e:
            if failures is None:
                raise
            failures[trainer] = str(e) or type(e).__name__
    return entries
