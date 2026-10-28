"""The acceptance test, as an agent sees it: a real MCP client talks to a real server over HTTP, with only
an API key, and one-shots a training run. A worker runs alongside, like production. The fake trainer
stands in for Tinker, so nothing is spent."""

import json
import socket
import threading
import time

import anyio
import httpx
import pytest

from service_helpers import banking_csv


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(wr, monkeypatch):
    import uvicorn

    from weightroom import config, worker

    port = _free_port()
    monkeypatch.setenv("PUBLIC_BASE_URL", f"http://127.0.0.1:{port}")
    config.reload()
    from weightroom.api import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)

    stop = threading.Event()

    def work():
        while not stop.is_set():
            if not worker.run_once("test-worker"):
                stop.wait(0.05)
    worker_thread = threading.Thread(target=work, daemon=True)
    worker_thread.start()
    yield f"http://127.0.0.1:{port}"
    stop.set()
    server.should_exit = True
    thread.join(timeout=5)
    worker_thread.join(timeout=5)


def _payload(result) -> dict:
    if result.is_error:
        text = result.content[0].text  # the SDK prefixes "Error executing tool <name>: "
        return {"tool_error": json.loads(text[text.index("{"):])}
    if getattr(result, "structured_content", None):
        return result.structured_content
    return json.loads(result.content[0].text)


def test_agent_one_shots_a_training_run(live_server, tmp_path):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    base = live_server
    key = httpx.post(f"{base}/account/signup", json={"email": "agent@example.com",
                                                      "password": "correct horse battery"}).json()["api_key"]
    csv_path = tmp_path / "banking.csv"
    csv_path.write_bytes(banking_csv(columns=("message", "intent")))

    async def agent() -> dict:
        http = create_mcp_http_client(headers={"Authorization": f"Bearer {key}"})
        async with http, streamable_http_client(f"{base}/mcp", http_client=http) as (read, write), \
                ClientSession(read, write) as session:
            init = await session.initialize()
            assert "quote" in (init.instructions or "")
            tools = {t.name for t in (await session.list_tools()).tools}
            assert tools == {"upload_data", "check_data", "quote", "train", "status", "list_jobs", "deliver"}

            async def call(name, **args) -> dict:
                return _payload(await session.call_tool(name, args))

            up = await call("upload_data", filename="banking.csv")
            # the agent runs the curl command; here, the same PUT
            async with httpx.AsyncClient() as h:
                r = await h.put(up["upload_url"], content=csv_path.read_bytes())
                assert r.status_code == 200, r.text

            checked = await call("check_data", dataset_id=up["dataset_id"])
            assert checked["ready"] and checked["task"] == "labels"
            checked = await call("check_data", dataset_id=up["dataset_id"],
                                 instructions=checked["suggested_instructions"])

            q = await call("quote", dataset_id=up["dataset_id"])
            assert q["can_start"] and q["credits"] <= 10
            # (here the agent shows the user the price and gets a yes)

            job = await call("train", quote_id=q["quote_id"])
            retry = await call("train", quote_id=q["quote_id"])
            assert retry["job_id"] == job["job_id"] and retry["already_started"]

        # the agent's session ends mid-training; a new one finds the job
        http = create_mcp_http_client(headers={"Authorization": f"Bearer {key}"})
        async with http, streamable_http_client(f"{base}/mcp", http_client=http) as (read, write), \
                ClientSession(read, write) as session:
            await session.initialize()

            async def call(name, **args) -> dict:
                return _payload(await session.call_tool(name, args))

            jobs = (await call("list_jobs"))["jobs"]
            job_id = jobs[0]["job_id"]
            assert job_id == job["job_id"]
            for _ in range(200):
                st = await call("status", job_id=job_id)
                if st["state"] in ("done", "failed"):
                    break
                await anyio.sleep(0.05)
            assert st["state"] == "done", st

            for _ in range(200):
                d = await call("deliver", job_id=job_id)
                if all(x["state"] == "ready" for x in d.get("deliveries", [])):
                    break
                await anyio.sleep(0.05)
            err = await call("deliver", job_id="job_does_not_exist")
            no_key = None
        async with httpx.AsyncClient() as h:
            r = await h.post(f"{base}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                             json={"model": job_id, "messages": [{"role": "user", "content": "card not here"}]})
        return {"status": st, "deliver": d, "error": err, "chat": r.json(), "no_key": no_key}

    out = anyio.run(agent)
    card = out["status"]["scorecard"]
    assert card["trained"]["accuracy"] > card["original"]["accuracy"]
    assert out["deliver"]["deliveries"][0]["url"]
    assert out["deliver"]["endpoint"]["model"] == out["status"]["job_id"]
    err = out["error"]["tool_error"]["error"]
    assert err["code"] == "job_not_found" and err["fix"]
    assert out["chat"]["choices"][0]["message"]["content"]


def test_tools_without_a_key_say_how_to_get_one(live_server):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def agent():
        async with streamable_http_client(f"{live_server}/mcp") as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            return _payload(await session.call_tool("list_jobs", {}))

    err = anyio.run(agent)["tool_error"]["error"]
    assert err["code"] == "missing_api_key" and "/dashboard" in err["fix"]
