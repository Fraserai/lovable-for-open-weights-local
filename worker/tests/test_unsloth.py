import ast
from types import SimpleNamespace

import pytest

from trainloop import unsloth
from trainloop.config import trainer_for


def _hw(name, model, vram, ram, per_min, quantity="1"):
    acc = SimpleNamespace(model=model, vram=f"{vram} GB", quantity=quantity) if model else None
    return SimpleNamespace(name=name, accelerator=acc, ram=f"{ram} GB", unit_cost_usd=per_min, unit_label="minute")


# A slice of the live Hugging Face Jobs list (Sept 2026)
HARDWARE = [
    _hw("cpu-basic", None, 0, 16, 0.0002),
    _hw("t4-small", "T4", 16, 15, 0.0067),
    _hw("l4x1", "L4", 24, 30, 0.0133),
    _hw("a10g-small", "A10G", 24, 15, 0.0167),
    _hw("l40sx1", "L40S", 48, 62, 0.03),
    _hw("a100-large", "A100", 80, 142, 0.0417),
    _hw("h200", "H200", 141, 256, 0.0833),
    _hw("h200x8", "H200", 1128, 2048, 0.6667, quantity="8"),
]


def test_routing_by_model_name():
    assert trainer_for("unsloth:Qwen/Qwen3.5-4B") == "unsloth"
    assert trainer_for("Qwen/Qwen3.5-4B") == "tinker"
    assert unsloth.hf_id("unsloth:Qwen/Qwen3.5-4B") == "Qwen/Qwen3.5-4B"


def test_picks_the_cheapest_bf16_gpu_that_fits():
    small = unsloth.pick_gpu(4_659_865_088, HARDWARE)  # Qwen3.5-4B: ~18 GB, skips T4 and L4
    assert small["flavor"] == "a10g-small" and small["per_hour"] == pytest.approx(1.0, abs=0.01)
    assert unsloth.pick_gpu(27_000_000_000, HARDWARE)["flavor"] == "a100-large"  # ~76 GB
    assert unsloth.pick_gpu(45_000_000_000, HARDWARE)["flavor"] == "h200"


def test_refuses_models_that_need_more_than_one_gpu():
    with pytest.raises(unsloth.UnslothError, match="Multi-GPU"):
        unsloth.pick_gpu(321_323_031_390, HARDWARE)  # GLM-5.3-Flash in 16-bit


def test_epochs_and_time_limit():
    assert unsloth.resolve_epochs(1000) == 1 and unsloth.resolve_epochs(12) == 4
    assert unsloth.time_limit_hours(553_573) == 1.0  # Banking77: 20 min setup + ~31 min, rounded up
    assert unsloth.time_limit_hours(10**12) == unsloth.MAX_HOURS


def test_reads_progress_lines_from_the_job_log():
    line = '2026-09-23T10:00:00Z TRAINLOOP {"event": "step", "step": 3, "total": 250, "loss": 1.2}'
    assert unsloth._parse(line) == {"event": "step", "step": 3, "total": 250, "loss": 1.2}
    assert unsloth._parse("Downloading shards: 100%") is None
    assert unsloth._parse("TRAINLOOP {not json") is None


def test_job_script_is_standalone():
    """It runs in a bare container: it must not import trainloop, and pins its Unsloth version."""
    source = unsloth.SCRIPT.read_text()
    imports = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("trainloop" in (getattr(n, "module", "") or "") or
                   any("trainloop" in a.name for a in n.names) for n in imports)
    assert '"unsloth==' in source.split("# ///")[1]
