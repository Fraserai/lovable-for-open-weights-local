"""Keep some examples back from training so the Scorecard has a fair test.

A random slice of train.jsonl moves to test.jsonl before anything trains. Any training example that
asks exactly what a held-back one asks is dropped too, so the answer can't leak in through a duplicate.
"""

import json
import random
from pathlib import Path

SHARE = 0.1       # of the examples
FLOOR, CAP = 10, 200
MIN_EXAMPLES = 30  # below this every example is needed for training; there's no held-back set
SEED = 3407


def _prompt_key(line: str) -> str:
    """Everything before the reply being taught: that's what a held-back example must not share."""
    return json.dumps(json.loads(line)["messages"][:-1], sort_keys=True)


def split(train: Path, test: Path, seed: int = SEED) -> tuple[int, int]:
    """Moves a held-back slice from `train` into `test`. Returns (examples kept for training, held back)."""
    lines = [line for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < MIN_EXAMPLES:
        test.unlink(missing_ok=True)
        return len(lines), 0
    order = list(range(len(lines)))
    random.Random(seed).shuffle(order)
    held = set(order[:min(CAP, max(FLOOR, round(len(lines) * SHARE)))])
    held_prompts = {_prompt_key(lines[i]) for i in held}
    kept = [line for i, line in enumerate(lines) if i not in held and _prompt_key(line) not in held_prompts]
    test.write_text("".join(lines[i] + "\n" for i in sorted(held)), encoding="utf-8")
    train.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    return len(kept), len(held)


def load(test: Path) -> list[list[dict]]:
    """The held-back conversations, each ending in the reference reply."""
    if not test.exists():
        return []
    return [json.loads(line)["messages"] for line in test.read_text(encoding="utf-8").splitlines() if line.strip()]
