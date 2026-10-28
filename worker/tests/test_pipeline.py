import asyncio
import json
from pathlib import Path

import httpx
import pytest

from trainloop.fireworks import Fireworks, FireworksError
from trainloop.ingest import Chunk, chunk_markdown
from trainloop.synth import ZDR_PROVIDER, Generator, generate_rows
from trainloop.validate import validate_jsonl


def write_jsonl(path: Path, rows: list) -> Path:
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def row(user="q", assistant="a"):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


# --- ingest ---

def test_chunks_split_on_headings_and_respect_size():
    para = "word " * 100  # ~500 chars
    md = "# Fees\n\n" + "\n\n".join([para] * 5) + "\n\n## Insurance\n\n" + para + "\n\n# Tiny\n\nx"
    chunks = chunk_markdown(md, "pds.pdf", max_chars=1200, min_chars=200)
    assert {c.heading for c in chunks} == {"Fees", "Insurance"}  # "Tiny" dropped as too short
    assert all(len(c.text) <= 1200 for c in chunks)
    assert sum(c.heading == "Fees" for c in chunks) == 3


def test_oversized_paragraph_is_split():
    chunks = chunk_markdown("# T\n\n" + "a" * 2500, "x.md", max_chars=1000, min_chars=1)
    assert [len(c.text) for c in chunks] == [1000, 1000, 500]


# --- validate ---

def test_valid_file(tmp_path):
    r = validate_jsonl(write_jsonl(tmp_path / "d.jsonl", [row("a"), row("b"), row("c")]))
    assert r.ok and r.valid == 3


def test_catches_malformed_rows(tmp_path):
    rows = [
        row("a"), row("b"), row("c"),
        "{not json",
        {"messages": []},
        {"messages": [{"role": "user", "content": "only a question"}]},
        {"messages": [{"role": "bot", "content": "x"}, {"role": "assistant", "content": "y"}]},
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "  "}]},
    ]
    r = validate_jsonl(write_jsonl(tmp_path / "d.jsonl", rows))
    assert not r.ok
    assert r.valid == 3
    assert {line for line, _ in r.errors} == {4, 5, 6, 7, 8}


def test_too_few_examples_and_too_long(tmp_path):
    r = validate_jsonl(write_jsonl(tmp_path / "d.jsonl", [row("a"), row("x" * 5000)]), max_tokens=100)
    msgs = " ".join(m for _, m in r.errors)
    assert "at least 3" in msgs and "training context" in msgs


def test_duplicates_warn(tmp_path):
    r = validate_jsonl(write_jsonl(tmp_path / "d.jsonl", [row("a"), row("a"), row("b")]))
    assert r.ok and any("duplicate" in m for _, m in r.warnings)


# --- synth ---

def test_generator_pins_zdr_and_builds_rows():
    bodies = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        bodies.append(body)
        name = body["response_format"]["json_schema"]["name"]
        content = ({"system_prompt": "You help members."} if name == "system_prompt"
                   else {"examples": [{"user": "What are the fees?", "assistant": "0.5% p.a."}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})

    async def go():
        gen = Generator("k", "z-ai/glm-5.3", transport=httpx.MockTransport(handler))
        try:
            return await generate_rows(gen, [Chunk("pds.pdf", "Fees", "Admin fee is 0.5% p.a.")], "answer member questions", 1)
        finally:
            await gen.aclose()

    system_prompt, rows, errors = asyncio.run(go())
    assert all(b["provider"] == ZDR_PROVIDER for b in bodies)
    assert errors == []
    assert rows == [{"messages": [
        {"role": "system", "content": "You help members."},
        {"role": "user", "content": "What are the fees?"},
        {"role": "assistant", "content": "0.5% p.a."},
    ]}]


def test_failing_chunk_is_reported_not_fatal():
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        if body["response_format"]["json_schema"]["name"] == "system_prompt":
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"system_prompt": "s"}'}}]})
        return httpx.Response(400, json={"error": "bad"})

    async def go():
        gen = Generator("k", "m", transport=httpx.MockTransport(handler))
        try:
            return await generate_rows(gen, [Chunk("a.pdf", "H", "text")], "goal", 2)
        finally:
            await gen.aclose()

    _, rows, errors = asyncio.run(go())
    assert rows == [] and len(errors) == 1 and "a.pdf / H" in errors[0]


# --- fireworks ---

def test_fireworks_upload_and_job_requests(tmp_path):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        path = request.url.path
        if path.endswith("/supervisedFineTuningJobs"):
            return httpx.Response(200, json={"name": "accounts/acct/supervisedFineTuningJobs/run1",
                                             "state": "JOB_STATE_CREATING"})
        if path.endswith("/datasets/run1-data") and request.method == "GET":
            return httpx.Response(200, json={"state": "READY"})
        return httpx.Response(200, json={})

    data = write_jsonl(tmp_path / "train.jsonl", [row("a"), row("b"), row("c")])
    fw = Fireworks("key", "acct", transport=httpx.MockTransport(handler), poll_seconds=0)
    ds = fw.upload_dataset("run1-data", data, 3)
    fw.wait_dataset_ready("run1-data")
    fw.create_sft_job("run1", ds, "accounts/fireworks/models/glm-5p3-flash", "run1", epochs=1, lora_rank=8)
    fw.close()

    create_ds, upload, get_ds, create_job = calls
    assert create_ds.url.path == "/v1/accounts/acct/datasets"
    assert json.loads(create_ds.content)["dataset"]["format"] == "CHAT"
    assert upload.url.path == "/v1/accounts/acct/datasets/run1-data:upload"
    assert b'filename="train.jsonl"' in upload.content
    assert create_job.url.params["supervisedFineTuningJobId"] == "run1"
    body = json.loads(create_job.content)
    assert body["dataset"] == "accounts/acct/datasets/run1-data"
    assert body["outputModel"] == "accounts/acct/models/run1"
    assert body["loraRank"] == 8 and body["evalAutoCarveout"] is True
    assert all(c.headers["authorization"] == "Bearer key" for c in calls)


def test_fireworks_errors_are_readable():
    fw = Fireworks("key", "acct", transport=httpx.MockTransport(lambda r: httpx.Response(403, text="no quota")))
    with pytest.raises(FireworksError, match="403: no quota"):
        fw.get_sft_job("x")
