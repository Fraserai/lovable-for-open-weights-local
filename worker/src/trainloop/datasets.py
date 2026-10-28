"""Use a ready-made dataset instead of writing training examples from documents.

Two sources: a JSONL file the customer uploads, or a dataset on the Hugging Face Hub
(read through the datasets-server API, so no heavy `datasets` dependency).

Public datasets arrive in a handful of shapes. Rather than demand one, map the common
ones onto chat messages and say plainly when a row doesn't fit.
"""

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

ROWS_URL = "https://datasets-server.huggingface.co/rows"
SPLITS_URL = "https://datasets-server.huggingface.co/splits"
PAGE = 100  # datasets-server's maximum rows per request
RETRIES = 4  # the dataset viewer has brief 429/5xx hiccups; a big import makes many requests

# (user field, assistant field) pairs seen in public instruction datasets, matched case-insensitively.
# "issue"/"decision" covers ATO Interpretative Decisions and similar ruling datasets.
_PAIRS = [
    ("instruction", "output"), ("instruction", "response"), ("prompt", "response"),
    ("prompt", "completion"), ("question", "answer"), ("issue", "decision"),
    ("input", "output"), ("query", "answer"),
    # Classification datasets: a text column and its label
    ("text", "label"), ("text", "intent"), ("text", "category"), ("text", "class"),
    ("message", "label"), ("message", "intent"), ("message", "category"),
    ("sentence", "label"), ("input", "label"), ("content", "label"),
]
_ROLE_ALIASES = {"human": "user", "gpt": "assistant", "bot": "assistant", "assistant": "assistant",
                 "user": "user", "system": "system"}


class DatasetError(ValueError):
    pass


class _Blank(dict):
    """format_map helper: a missing or empty column renders as nothing instead of failing."""
    def __missing__(self, key):
        return ""


def render(template: str, row: dict) -> str:
    """Fill {Column} placeholders from a row (case-insensitive). Blank sections collapse."""
    values = _Blank({str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()})
    text = re.sub(r"\{([^{}]+)\}", lambda m: "{" + m.group(1).strip().lower() + "}", template).format_map(values)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clean(messages: list[dict]) -> list[dict] | None:
    out = []
    for m in messages:
        role = _ROLE_ALIASES.get(str(m.get("role") or m.get("from") or "").strip().lower())
        content = m.get("content") if m.get("content") is not None else m.get("value")
        if not role or not isinstance(content, str) or not content.strip():
            return None
        out.append({"role": role, "content": content.strip()})
    return out if any(m["role"] == "assistant" for m in out) else None


def to_messages(row: dict, fields: tuple[str, str] | None = None,
                templates: tuple[str, str] | None = None) -> list[dict] | None:
    """Map one dataset row onto chat messages, or None if it has no usable pair.

    `fields` names the (user, assistant) columns explicitly. `templates` builds each side from
    several columns, e.g. ("{Issue}\n\nFacts: {Facts}", "{Decision}\n\n{Reasons_for_Decision}"),
    so context that decides the answer isn't left out.
    """
    if templates:
        user, assistant = render(templates[0], row), render(templates[1], row)
        if user and assistant:
            return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        return None
    row = {str(k).strip().lower(): v for k, v in row.items()}
    pairs = [(fields[0].lower(), fields[1].lower())] if fields else _PAIRS

    if not fields:
        for key in ("messages", "conversations", "conversation"):
            if isinstance(row.get(key), list):
                return _clean(row[key])

    for user_key, assistant_key in pairs:
        user, assistant = row.get(user_key), row.get(assistant_key)
        if isinstance(assistant, (int, float)) and not isinstance(assistant, bool):
            assistant = str(assistant)
        if isinstance(user, str) and isinstance(assistant, str) and user.strip() and assistant.strip():
            # Alpaca-style datasets put extra material in "input" alongside "instruction".
            extra = row.get("input") if user_key == "instruction" else None
            if isinstance(extra, str) and extra.strip():
                user = f"{user.strip()}\n\n{extra.strip()}"
            return [{"role": "user", "content": user.strip()}, {"role": "assistant", "content": assistant.strip()}]
    return None


def label_names(features) -> dict[str, list[str]]:
    """Columns stored as numbered classes, with the names the dataset gives them (Hugging Face ClassLabel).

    Many classification datasets store "label": 3 and keep "card_arrival" in the dataset's metadata.
    Reading those names is reading the customer's data as written, not changing it."""
    out = {}
    for f in features or []:
        kind = (f or {}).get("type") or {}
        if kind.get("_type") == "ClassLabel" and isinstance(kind.get("names"), list):
            out[str(f["name"])] = [str(n) for n in kind["names"]]
    return out


def with_label_names(row: dict, names: dict[str, list[str]]) -> dict:
    if not names:
        return row
    out = dict(row)
    for col, labels in names.items():
        value = out.get(col)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(labels):
            out[col] = labels[value]
    return out


def from_jsonl(path: Path, fields: tuple[str, str] | None = None,
               templates: tuple[str, str] | None = None) -> Iterator[list[dict]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path.name} line {line_no} isn't valid JSON.") from e
            messages = to_messages(row, fields, templates) if isinstance(row, dict) else None
            if messages:
                yield messages


def _get(c: httpx.Client, url: str, params: dict, pause: float = 1.0) -> httpx.Response:
    """GET with retries on temporary failures (rate limits, 5xx, dropped connections), backing off 1-2-4-8s."""
    for attempt in range(RETRIES + 1):
        try:
            resp = c.get(url, params=params)
            if resp.status_code != 429 and resp.status_code < 500:
                return resp
        except httpx.TransportError:
            if attempt == RETRIES:
                raise
            resp = None
        if attempt < RETRIES:
            time.sleep(pause * 2 ** attempt)
    return resp


def hub_splits(repo: str, client: httpx.Client | None = None) -> list[tuple[str, str]]:
    c = client or httpx.Client(timeout=60)
    try:
        resp = _get(c, SPLITS_URL, {"dataset": repo})
        if resp.is_error:
            raise DatasetError(f"Hugging Face couldn't list splits for {repo}: {resp.status_code}")
        return [(s["config"], s["split"]) for s in resp.json().get("splits", [])]
    finally:
        if client is None:
            c.close()


def from_hub(repo: str, split: str | None = None, config: str | None = None, limit: int = 200,
             client: httpx.Client | None = None, fields: tuple[str, str] | None = None,
             templates: tuple[str, str] | None = None) -> Iterator[list[dict]]:
    """Stream rows from a public Hugging Face dataset, mapped to chat messages."""
    c = client or httpx.Client(timeout=120)
    try:
        available = hub_splits(repo, c)
        if not available:
            raise DatasetError(f"{repo} has no readable splits. It may be private or not yet indexed.")
        config, split = next(
            ((cfg, sp) for cfg, sp in available if (config is None or cfg == config) and (split is None or sp == split)),
            available[0],
        )
        sent = 0
        names: dict[str, list[str]] = {}
        for offset in range(0, limit, PAGE):
            resp = _get(c, ROWS_URL, {"dataset": repo, "config": config, "split": split,
                                      "offset": offset, "length": min(PAGE, limit - sent)})
            if resp.is_error:
                raise DatasetError(f"Hugging Face returned {resp.status_code} for {repo} ({config}/{split}).")
            page = resp.json()
            rows = page.get("rows", [])
            if not rows:
                return
            names = names or label_names(page.get("features"))
            for item in rows:
                messages = to_messages(with_label_names(item.get("row", {}), names), fields, templates)
                if messages:
                    yield messages
                    sent += 1
                    if sent >= limit:
                        return
    finally:
        if client is None:
            c.close()


def write_rows(rows: Iterator[list[dict]], out: Path, system_prompt: str | None) -> int:
    """Write chat rows as training JSONL, adding the system prompt where a row has none."""
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for messages in rows:
            if system_prompt and messages[0]["role"] != "system":
                messages = [{"role": "system", "content": system_prompt}, *messages]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise DatasetError(
            "No usable rows. Each row needs a chat conversation, or a question/answer style pair "
            "(messages, conversations, instruction+output, prompt+response, question+answer, issue+decision), "
            "or name the columns yourself with --user-field and --assistant-field."
        )
    return count
