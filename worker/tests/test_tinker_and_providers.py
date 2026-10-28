import asyncio
import json

import httpx

from trainloop.config import trainer_for
from trainloop.ingest import Chunk
from trainloop.synth import OPENAI_URL, OPENROUTER_URL, ZDR_PROVIDER, Generator, generate_rows
from trainloop.tinker import expand_per_assistant_turn, resolve_epochs


def test_small_datasets_get_more_epochs_capped_at_four():
    assert resolve_epochs(12) == 4       # 3 steps/epoch -> would need 84, capped
    assert resolve_epochs(1000) == 1     # 250 steps/epoch already hits the target
    assert resolve_epochs(600) == 2      # 150 steps/epoch -> 2 passes


def test_multi_turn_conversation_becomes_one_example_per_assistant_turn(tmp_path):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    convo = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
             {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
    single = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    src.write_text(json.dumps({"messages": convo}) + "\n" + json.dumps({"messages": single}) + "\n")

    assert expand_per_assistant_turn(src, dst) == 3
    rows = [json.loads(l)["messages"] for l in dst.read_text().splitlines()]
    assert rows == [convo[:2], convo, single]  # each ends at the assistant turn it teaches


def test_trainer_routing():
    assert trainer_for("accounts/fireworks/models/glm-5p3-flash") == "fireworks"
    assert trainer_for("Qwen/Qwen3.5-4B") == "tinker"


def _run(provider: str):
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append((str(request.url), body))
        name = body["response_format"]["json_schema"]["name"]
        content = {"system_prompt": "s"} if name == "system_prompt" else {"examples": [{"user": "q", "assistant": "a"}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})

    async def go():
        gen = Generator("k", "m", transport=httpx.MockTransport(handler), provider=provider)
        try:
            return await generate_rows(gen, [Chunk("f.pdf", "H", "text")], "goal", 1)
        finally:
            await gen.aclose()

    asyncio.run(go())
    return seen


def test_openrouter_is_always_pinned_to_zdr():
    for url, body in _run("openrouter"):
        assert url == OPENROUTER_URL and body["provider"] == ZDR_PROVIDER


def test_openai_uses_reasoning_effort_not_temperature():
    for url, body in _run("openai"):
        assert url == OPENAI_URL
        assert "provider" not in body and "temperature" not in body
        assert body["reasoning_effort"] == "low"


def test_replies_that_mention_source_material_are_caught():
    from trainloop.synth import leaks_source

    assert leaks_source("The excerpt doesn't say whether you're covered.")
    assert leaks_source("According to the document, fees are monthly.")
    assert not leaks_source("Fees are deducted monthly. For anything else, contact the fund.")
    assert not leaks_source("You can send documents to the fund by post.")


def test_dataset_rows_map_to_chat_messages():
    from trainloop.datasets import to_messages

    ato = to_messages({"Issue": "Is X taxable?", "Decision": "Yes.", "Facts": "..."})
    assert ato == [{"role": "user", "content": "Is X taxable?"}, {"role": "assistant", "content": "Yes."}]
    sharegpt = to_messages({"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]})
    assert sharegpt == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    alpaca = to_messages({"instruction": "Summarise", "input": "Long text", "output": "Short"})
    assert alpaca[0]["content"] == "Summarise\n\nLong text"
    assert to_messages({"title": "t", "body": "b"}, ("Title", "Body"))[1]["content"] == "b"
    assert to_messages({"nothing": "useful"}) is None
    assert to_messages({"question": "q", "answer": ""}) is None  # empty answer teaches nothing


def test_dataset_rows_get_the_goal_as_system_prompt(tmp_path):
    from trainloop.datasets import write_rows

    out = tmp_path / "train.jsonl"
    rows = iter([[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]])
    assert write_rows(rows, out, system_prompt="Be an expert") == 1
    msgs = json.loads(out.read_text())["messages"]
    assert msgs[0] == {"role": "system", "content": "Be an expert"}


def test_dataset_with_no_usable_rows_explains_what_is_needed(tmp_path):
    from trainloop.datasets import DatasetError, write_rows

    try:
        write_rows(iter([]), tmp_path / "out.jsonl", system_prompt=None)
    except DatasetError as e:
        assert "question+answer" in str(e) and "--user-field" in str(e)
    else:
        raise AssertionError("expected DatasetError")


def test_templates_build_examples_from_several_columns():
    from trainloop.datasets import to_messages

    row = {"Issue": "Is X deductible?", "Facts": "Bought a van.", "Decision": "Yes.", "Reasons_for_Decision": "s 8-1 applies."}
    msgs = to_messages(row, templates=("{Issue}\n\nFacts: {Facts}", "{Decision}\n\n{Reasons_for_Decision}"))
    assert msgs[0]["content"] == "Is X deductible?\n\nFacts: Bought a van."
    assert msgs[1]["content"] == "Yes.\n\ns 8-1 applies."
    # a missing column renders as nothing rather than failing the row
    partial = to_messages({"Issue": "Q?", "Decision": "A."}, templates=("{Issue}\n\n{Facts}", "{Decision}"))
    assert partial == [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A."}]
    assert to_messages({"Issue": ""}, templates=("{Issue}", "{Decision}")) is None
