"""Shared setup for the hosted-service tests: everything local, the fake trainer, no money spent."""

import csv
import io
import random

INTENTS = ["card_arrival", "lost_or_stolen_card", "exchange_rate", "top_up_failed", "pin_blocked"]

PRICES = {
    "Qwen/Qwen3.5-4B": {"name": "Qwen3.5-4B", "params": 4_000_000_000, "context": 65536,
                        "train": 0.737, "prefill": 0.33, "sample": 1.005},
    "moonshotai/Kimi-K2.6": {"name": "Kimi-K2.6", "params": 1_000_000_000_000, "context": 32768,
                             "train": 4.84, "prefill": 2.205, "sample": 5.49},
    "unsupported/Model-1B": {"name": "Model-1B", "params": 1_000_000_000, "context": 32768,
                             "train": 0.1, "prefill": 0.1, "sample": 0.1},
}


def banking_csv(rows: int = 300, columns: tuple[str, str] = ("text", "intent"), seed: int = 7) -> bytes:
    rng = random.Random(seed)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(columns)
    for i in range(rows):
        intent = INTENTS[i % len(INTENTS)]
        w.writerow([f"Message {i}: customer asks about {intent.replace('_', ' ')} (variant {rng.randint(0, 9999)})",
                    intent])
    return out.getvalue().encode()


def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}
