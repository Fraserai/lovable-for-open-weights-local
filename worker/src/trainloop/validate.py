"""Catch malformed training rows before a paid run.

Rules follow the Fireworks chat SFT format (docs.fireworks.ai/fine-tuning/fine-tuning-models, Sept 2026):
one JSON object per line with a "messages" list; min 3 examples, max 3M.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant"}
MIN_EXAMPLES = 3
MAX_EXAMPLES = 3_000_000
CHARS_PER_TOKEN = 3.5  # conservative estimate; errs towards flagging long rows


@dataclass
class Report:
    rows: int = 0
    valid: int = 0
    est_tokens: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)
    warnings: list[tuple[int, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self, limit: int = 20) -> str:
        lines = [f"{self.valid}/{self.rows} rows valid, ~{self.est_tokens:,} tokens"]
        for label, items in (("ERROR", self.errors), ("warn", self.warnings)):
            for line_no, msg in items[:limit]:
                where = f"line {line_no}" if line_no else "file"
                lines.append(f"  {label} {where}: {msg}")
            if len(items) > limit:
                lines.append(f"  ... and {len(items) - limit} more {label.lower()}s")
        return "\n".join(lines)


def _content_text(content) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                return None
            parts.append(part["text"])
        return "".join(parts)
    return None


def check_row(row, max_tokens: int | None) -> tuple[list[str], list[str], int]:
    """Returns (errors, warnings, estimated_tokens) for one parsed row."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(row, dict):
        return ["row is not a JSON object"], [], 0
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ['missing or empty "messages" list'], [], 0

    chars = 0
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"message {i} is not an object")
            continue
        role = msg.get("role")
        if role not in VALID_ROLES:
            errors.append(f"message {i} has invalid role {role!r}")
        if role == "system" and i != 0:
            warnings.append(f"message {i}: system message not at the start")
        text = _content_text(msg.get("content"))
        if text is None:
            errors.append(f"message {i} content must be a string or a list of text parts")
        elif not text.strip():
            errors.append(f"message {i} ({role}) has empty content")
        else:
            chars += len(text)
        if "weight" in msg and msg["weight"] not in (0, 1):
            errors.append(f"message {i} weight must be 0 or 1")

    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if "assistant" not in roles:
        errors.append("no assistant message, so there is nothing to learn from this row")
    elif roles[-1] != "assistant":
        warnings.append("last message is not from the assistant")
    if "weight" in row and not isinstance(row["weight"], (int, float)):
        errors.append("row weight must be a number")

    tokens = int(chars / CHARS_PER_TOKEN)
    if max_tokens and tokens > max_tokens:
        errors.append(f"~{tokens:,} tokens exceeds the model's {max_tokens:,} training context")
    return errors, warnings, tokens


def validate_jsonl(path: Path, max_tokens: int | None = None) -> Report:
    report = Report()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            report.rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                report.errors.append((line_no, f"invalid JSON: {e.msg}"))
                continue
            errors, warnings, tokens = check_row(row, max_tokens)
            report.errors.extend((line_no, e) for e in errors)
            report.warnings.extend((line_no, w) for w in warnings)
            if not errors:
                report.valid += 1
                report.est_tokens += tokens
                key = json.dumps(row.get("messages"), sort_keys=True)
                if key in seen:
                    report.warnings.append((line_no, "duplicate of an earlier row"))
                seen.add(key)

    if report.valid < MIN_EXAMPLES:
        report.errors.append((0, f"need at least {MIN_EXAMPLES} valid examples, found {report.valid}"))
    if report.rows > MAX_EXAMPLES:
        report.errors.append((0, f"more than {MAX_EXAMPLES:,} examples"))
    return report
