import json

import httpx
import pytest

from trainloop.catalog import CatalogError, _context_to_tokens, _fireworks_models, build_catalog

FW_PAGE = """\
## Models

export const ModelsCatalog = () => {
  const TRAINING_SHAPES_CATALOG = {
    "generatedAt": "2026-09-12 18:38 UTC",
    "models": [{
      "model": "GLM 5.3 Flash",
      "id": "glm-5p3-flash",
      "managedSftLora": true,
      "managedDpoLora": true,
      "trainCtx": 65536,
      "paramCount": 320000000000,
      "flags": {}
    }, {
      "model": "Tiny Test 1B",
      "id": "tiny-test-1b",
      "managedSftLora": false,
      "managedDpoLora": false,
      "trainCtx": 16384,
      "paramCount": 1000000000,
      "flags": {}
    }]
  };
  const models = TRAINING_SHAPES_CATALOG.models ?? [];
};
"""

TINKER_JSON = json.dumps([
    {
        "name": "GLM-5.3 (256K)",
        "url": "https://huggingface.co/zai-org/GLM-5.3",
        "tinker_id": "zai-org/GLM-5.3:peft:262144",
        "params": 753000000000,
        "active_params": 40000000000,
        "context": "256K",
        "train": "$14.58",
    },
    {
        "name": "Inkling",
        "url": "https://huggingface.co/thinkingmachines/Inkling",
        "tinker_id": "thinkingmachines/Inkling",
        "params": 975000000000,
        "active_params": 41000000000,
        "context": "64K",
        "train": "$5.61",
    },
])


def mock_transport(fw_text: str, tinker_text: str) -> httpx.MockTransport:
    def handler(request: httpx.Request):
        body = fw_text if "fireworks.ai" in request.url.host else tinker_text
        return httpx.Response(200, text=body)

    return httpx.MockTransport(handler)


def entries_by_trainer(entries, trainer):
    return (e for e in entries if e["trainer"] == trainer)


def test_fireworks_parsing():
    entries = build_catalog(mock_transport(FW_PAGE, TINKER_JSON))
    fw = list(entries_by_trainer(entries, "fireworks"))
    assert [e["display_name"] for e in fw] == ["GLM 5.3 Flash", "Tiny Test 1B"]

    glm = fw[0]
    assert glm["trainer_model_id"] == "accounts/fireworks/models/glm-5p3-flash"
    assert glm["sft_lora"] is True and glm["dpo"] is True
    assert glm["train_context_tokens"] == 65536 and glm["params"] == 320000000000
    assert glm["hf_repo"] is None and glm["train_price"] is None

    tiny = fw[1]
    assert tiny["sft_lora"] is False and tiny["dpo"] is False


def test_tinker_parsing():
    entries = build_catalog(mock_transport(FW_PAGE, TINKER_JSON))
    tk = list(entries_by_trainer(entries, "tinker"))
    assert len(tk) == 2

    glm = tk[0]
    assert glm["trainer_model_id"] == "zai-org/GLM-5.3:peft:262144"
    assert glm["hf_repo"] == "zai-org/GLM-5.3"
    assert glm["train_context_tokens"] == 262144
    assert glm["params"] == 753000000000
    assert glm["train_price"] == "$14.58"
    assert glm["sft_lora"] is True and glm["dpo"] is None

    inkling = tk[1]
    assert inkling["train_context_tokens"] == 65536
    assert inkling["hf_repo"] == "thinkingmachines/Inkling"


def test_context_conversion():
    assert _context_to_tokens("64K") == 65536
    assert _context_to_tokens("256K") == 262144
    assert _context_to_tokens("32K") == 32768
    assert _context_to_tokens(65536) == 65536
    assert _context_to_tokens("not-a-size") is None
    assert _context_to_tokens(None) is None


def test_changed_fireworks_format_raises_clear_error():
    for bad in ("no catalog here", "const TRAINING_SHAPES_CATALOG = [1, 2]", "const TRAINING_SHAPES_CATALOG = {"):
        with pytest.raises(CatalogError, match="Fireworks page format changed"):
            _fireworks_models(bad)


def test_two_trainers_merged():
    entries = build_catalog(mock_transport(FW_PAGE, TINKER_JSON))
    assert [e["trainer"] for e in entries] == ["fireworks", "fireworks", "tinker", "tinker"]

def test_one_unreadable_source_keeps_the_other():
    def handler(request: httpx.Request):
        if "fireworks.ai" in request.url.host:
            return httpx.Response(403, text="blocked")
        return httpx.Response(200, text=TINKER_JSON)

    failures: dict[str, str] = {}
    entries = build_catalog(httpx.MockTransport(handler), failures=failures)
    assert [e["trainer"] for e in entries] == ["tinker", "tinker"]
    assert list(failures) == ["fireworks"]
    # Without a failures dict the first failure still raises, as before
    with pytest.raises(httpx.HTTPStatusError):
        build_catalog(httpx.MockTransport(handler))


def test_catalog_command_keeps_last_known_models_for_an_unreadable_trainer(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from trainloop import catalog as catalog_mod
    from trainloop.cli import app

    out = tmp_path / "catalog.json"
    out.write_text(json.dumps([
        {"trainer": "fireworks", "trainer_model_id": "accounts/fireworks/models/glm-5p3-flash"},
        {"trainer": "tinker", "trainer_model_id": "old-tinker-model"},
    ]))

    def fake_build(transport=None, failures=None):
        failures["fireworks"] = "403 Forbidden"
        return [{"trainer": "tinker", "trainer_model_id": "new-tinker-model"}]

    monkeypatch.setattr(catalog_mod, "build_catalog", fake_build)
    result = CliRunner().invoke(app, ["catalog", "--json", str(out)])
    assert result.exit_code == 0, result.output
    ids = [e["trainer_model_id"] for e in json.loads(out.read_text())]
    assert ids == ["new-tinker-model", "accounts/fireworks/models/glm-5p3-flash"]
