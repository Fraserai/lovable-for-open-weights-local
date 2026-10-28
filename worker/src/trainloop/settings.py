"""Training settings chosen on the Train page (or with `trainloop configure`), and which trainer honours which.

Every setting is optional: anything unset is sized for the data by the trainer module. A setting the
chosen trainer can't honour is reported back by `plan` as ignored rather than silently dropped.
The web page keeps the same table (web/src/lib/settings.ts) to grey out controls; keep them in step.
"""

from typing import Any

# name -> (type, allowed values or (min, max))
SCHEMA: dict[str, tuple[type, Any]] = {
    "project_name": (str, None),
    "method": (str, ("lora", "qlora")),
    "max_steps": (int, (1, 100_000)),
    "epochs": (int, (1, 10)),
    "context_length": (int, (256, 262_144)),
    "learning_rate": (float, (1e-7, 1e-1)),
    "lora_rank": (int, (1, 256)),
    "lora_alpha": (int, (1, 512)),
    "lora_dropout": (float, (0.0, 0.5)),
    "target_modules": (list, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")),
    "lora_variant": (str, ("lora", "rslora")),
    "optimizer": (str, ("adamw_8bit", "adamw_torch")),
    "lr_scheduler": (str, ("linear", "cosine", "constant")),
    "batch_size": (int, (1, 256)),
    "grad_accum": (int, (1, 64)),
    "weight_decay": (float, (0.0, 1.0)),
    "warmup_steps": (int, (0, 10_000)),
    "seed": (int, (0, 2**31 - 1)),
    "gradient_checkpointing": (str, ("unsloth", "true", "false")),
}

# Verified against each trainer's code/API (Sept 2026): tinker_cookbook supervised.Config,
# Fireworks supervisedFineTuningJobs, and our own Unsloth job script (jobs/unsloth_sft.py).
SUPPORTED: dict[str, set[str]] = {
    "tinker": {"project_name", "method", "max_steps", "epochs", "context_length", "learning_rate",
               "lora_rank", "lr_scheduler", "batch_size"},
    "fireworks": {"project_name", "method", "epochs", "context_length", "learning_rate", "lora_rank"},
    "unsloth": set(SCHEMA),
}
# QLoRA (4-bit base) only where we control the loader
METHODS: dict[str, set[str]] = {"tinker": {"lora"}, "fireworks": {"lora"}, "unsloth": {"lora", "qlora"}}
# Fireworks documents 32; 32 is the most our Tinker runs have used (no published cap found)
MAX_LORA_RANK = {"fireworks": 32, "tinker": 32, "unsloth": 256}


class SettingsError(ValueError):
    pass


def clean(raw: dict) -> dict:
    """Checks names, types and ranges. None / "" / "auto" mean unset and are dropped."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in SCHEMA:
            raise SettingsError(f"Unknown setting {key!r}.")
        if value is None or value == "" or value == "auto":
            continue
        kind, allowed = SCHEMA[key]
        try:
            if kind is list:
                value = [str(v) for v in value]
                bad = [v for v in value if v not in allowed]
                if bad or not value:
                    raise SettingsError(f"{key}: choose from {', '.join(allowed)}.")
            elif kind is str:
                value = str(value).strip()
                if allowed and value not in allowed:
                    raise SettingsError(f"{key}: choose from {', '.join(allowed)}.")
            else:
                value = kind(value)
                lo, hi = allowed
                if not lo <= value <= hi:
                    raise SettingsError(f"{key} must be between {lo} and {hi}.")
        except (TypeError, ValueError) as e:
            if isinstance(e, SettingsError):
                raise
            raise SettingsError(f"{key}: {value!r} isn't a valid {kind.__name__}.") from e
        out[key] = value
    return out


def load(state: dict) -> dict:
    """The run's settings, including ones saved before this module existed."""
    legacy = {"epochs": state.get("train_epochs"), "lora_rank": state.get("train_lora_rank"),
              "learning_rate": state.get("learning_rate")}
    return {**{k: v for k, v in legacy.items() if v}, **state.get("settings", {})}


def split(trainer: str, s: dict) -> tuple[dict, list[str]]:
    """(settings this trainer honours, names it will ignore)."""
    ok = SUPPORTED[trainer]
    used = {k: v for k, v in s.items() if k in ok}
    ignored = sorted(k for k in s if k not in ok)
    if used.get("method") and used["method"] not in METHODS[trainer]:
        ignored.append("method")
        used.pop("method")
    if used.get("lora_rank", 0) > MAX_LORA_RANK[trainer]:
        raise SettingsError(f"{trainer.title()} allows LoRA rank up to {MAX_LORA_RANK[trainer]}.")
    return used, ignored
