"""The agent's whole journey over REST, with the fake trainer: upload -> check -> quote -> train (retried)
-> worker -> scorecard -> deliver -> endpoint. Plus the ways it must refuse, refund and recover."""

from datetime import timedelta

import pytest
from sqlalchemy import select

from service_helpers import auth, banking_csv


def signup(client, email="founder@example.com") -> str:
    r = client.post("/account/signup", json={"email": email, "password": "correct horse battery"})
    assert r.status_code == 200, r.text
    return r.json()["api_key"]


def upload(client, key, body: bytes, filename="tickets.csv") -> str:
    r = client.post("/v1/uploads", json={"filename": filename}, headers=auth(key))
    assert r.status_code == 200, r.text
    up = r.json()
    assert up["command"].startswith("curl") and filename in up["command"]
    path = up["upload_url"].removeprefix("http://testserver")
    r = client.put(path, content=body)
    assert r.status_code == 200, r.text
    return up["dataset_id"]


def drain(worker_id="w1") -> int:
    from weightroom import worker

    done = 0
    while worker.run_once(worker_id):
        done += 1
    return done


def balance(client) -> int:
    return client.get("/account/me").json()["balance_credits"]


def add_credits(n: int, email="founder@example.com"):
    from weightroom import credits, db

    with db.session() as s:
        acct = s.scalar(select(db.Account).where(db.Account.email == email))
        credits.add(s, acct.id, n, "adjust", note="test")


def test_whole_journey(client):
    key = signup(client)
    assert balance(client) == 10
    ds = upload(client, key, banking_csv())

    r = client.post("/v1/datasets/check", json={"dataset_id": ds}, headers=auth(key))
    check = r.json()
    assert r.status_code == 200 and check["ready"] is True
    assert check["task"] == "labels" and check["distinct_labels"] == 5
    assert check["held_out_rows"] == 30 and check["train_rows"] == 270
    assert "suggested_instructions" in check  # no instructions: base model wouldn't know the labels

    r = client.post("/v1/datasets/check", json={"dataset_id": ds, "instructions": check["suggested_instructions"]},
                    headers=auth(key))
    assert "suggested_instructions" not in r.json()

    q = client.post("/v1/quotes", json={"dataset_id": ds}, headers=auth(key)).json()
    assert q["base_model"] == "Qwen/Qwen3.5-4B" and q["can_start"] is True
    assert 1 <= q["credits"] <= 10 and q["includes_scorecard"] is True
    assert q["quote_id"] in q["next"]

    first = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key)).json()
    again = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key)).json()
    assert first["job_id"] == again["job_id"]
    assert first["already_started"] is False and again["already_started"] is True
    assert balance(client) == 10 - q["credits"]  # charged once

    assert drain() == 1
    st = client.get(f"/v1/jobs/{first['job_id']}", headers=auth(key)).json()
    assert st["state"] == "done", st
    card = st["scorecard"]
    assert card["metric"] == "exact_match" and card["held_out_rows"] == 30
    assert card["trained"]["accuracy"] > card["original"]["accuracy"]
    assert "accuracy went from" in card["summary"] and card["per_label"]

    jobs = client.get("/v1/jobs", headers=auth(key)).json()["jobs"]
    assert [j["job_id"] for j in jobs] == [first["job_id"]] and jobs[0]["headline"]

    d = client.post(f"/v1/jobs/{first['job_id']}/deliver", json={}, headers=auth(key)).json()
    assert d["endpoint"]["model"] == first["job_id"] and d["check_back_in_seconds"]
    assert drain() == 1
    d = client.post(f"/v1/jobs/{first['job_id']}/deliver", json={}, headers=auth(key)).json()
    url = d["deliveries"][0]["url"]
    r = client.get(url.removeprefix("http://testserver"))
    assert r.status_code == 200 and r.content[:2] == b"\x1f\x8b"  # gzip

    add_credits(5)
    before = balance(client)
    r = client.post("/v1/chat/completions", headers=auth(key), json={
        "model": first["job_id"], "messages": [{"role": "user", "content": "my card hasn't arrived"}]})
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"]
    assert balance(client) <= before


def test_wrong_columns_are_explained_for_free(client):
    key = signup(client)
    ds = upload(client, key, banking_csv(columns=("msg_text", "tag")))
    err = client.post("/v1/datasets/check", json={"dataset_id": ds}, headers=auth(key)).json()["error"]
    assert err["code"] == "columns_not_recognised"
    assert "user_column='msg_text'" in err["fix"] and "assistant_column='tag'" in err["fix"]

    err = client.post("/v1/datasets/check", json={"dataset_id": ds, "user_column": "msg_text",
                                                  "assistant_column": "label"}, headers=auth(key)).json()["error"]
    assert err["code"] == "column_not_found"
    assert err["message"].startswith("No column called 'label'. Found 'msg_text', 'tag'")

    ok = client.post("/v1/datasets/check", json={"dataset_id": ds, "user_column": "msg_text",
                                                 "assistant_column": "tag"}, headers=auth(key)).json()
    assert ok["ready"] is True
    assert balance(client) == 10  # checking never costs anything


def test_bad_formats_and_keys(client):
    key = signup(client)
    r = client.post("/v1/uploads", json={"filename": "notes.pdf"}, headers=auth(key))
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_format"
    r = client.post("/v1/uploads", json={"filename": "x.csv"})
    assert r.status_code == 401 and "Authorization: Bearer" in r.json()["error"]["fix"]
    r = client.post("/v1/uploads", json={"filename": "x.csv"}, headers=auth("wr_nope"))
    assert r.json()["error"]["code"] == "invalid_api_key"
    r = client.put("/v1/uploads/ds_123?token=forged", content=b"a,b\n")
    assert r.status_code == 403


def _quote(client, key, model=None) -> dict:
    ds = upload(client, key, banking_csv())
    client.post("/v1/datasets/check", json={"dataset_id": ds}, headers=auth(key))
    return client.post("/v1/quotes", json={"dataset_id": ds, **({"base_model": model} if model else {})},
                       headers=auth(key)).json()


def test_caps_and_balance_block_spend(client):
    key = signup(client)
    big = _quote(client, key, "moonshotai/Kimi-K2.6")
    small = _quote(client, key)
    assert big["credits"] > small["credits"]  # a 1T model costs more than a 4B one on the same data

    add_credits(-10)  # spend the welcome credits elsewhere
    q = _quote(client, key)
    assert q["can_start"] is False and q["blocked"]["code"] == "insufficient_credits"
    assert "Upgrade to Pro" in q["next"]
    r = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key))
    assert r.status_code == 402 and r.json()["error"]["code"] == "insufficient_credits"
    assert balance(client) == 0

    add_credits(100)
    key_id = client.get("/account/me").json()["keys"][0]["id"]
    client.post(f"/account/keys/{key_id}/caps", json={"per_job_cap": big["credits"] - 1})
    r = client.post("/v1/jobs", json={"quote_id": big["quote_id"]}, headers=auth(key))
    assert r.status_code == 402 and r.json()["error"]["code"] == "over_job_cap"
    client.post(f"/account/keys/{key_id}/caps", json={"monthly_cap": small["credits"] + 1})
    assert client.post("/v1/jobs", json={"quote_id": small["quote_id"]}, headers=auth(key)).status_code == 200
    r = client.post("/v1/jobs", json={"quote_id": big["quote_id"]}, headers=auth(key))
    assert r.json()["error"]["code"] == "over_monthly_cap"
    assert balance(client) == 100 - small["credits"]


def test_unknown_and_unsupported_models(client):
    key = signup(client)
    ds = upload(client, key, banking_csv())
    client.post("/v1/datasets/check", json={"dataset_id": ds}, headers=auth(key))
    err = client.post("/v1/quotes", json={"dataset_id": ds, "base_model": "made/up"}, headers=auth(key)).json()["error"]
    assert err["code"] == "unknown_model" and "Qwen/Qwen3.5-4B" in err["fix"]
    err = client.post("/v1/quotes", json={"dataset_id": ds, "base_model": "unsupported/Model-1B"},
                      headers=auth(key)).json()["error"]
    assert err["code"] == "model_not_supported"


def test_expired_quote(client):
    from weightroom import db

    key = signup(client)
    q = _quote(client, key)
    with db.session() as s:
        s.get(db.Quote, q["quote_id"]).expires_at = db.now() - timedelta(minutes=1)
    err = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key)).json()["error"]
    assert err["code"] == "quote_expired" and "quote again" in err["fix"]


def test_dropped_connection_resumes_from_checkpoint(client):
    from weightroom import trainers

    key = signup(client)
    q = _quote(client, key)
    job = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key)).json()
    trainers.FakeTrainer.fail_at_step = 5
    drain()
    st = client.get(f"/v1/jobs/{job['job_id']}", headers=auth(key)).json()
    assert st["state"] == "done"
    from weightroom import db
    with db.session() as s:
        assert s.get(db.Job, job["job_id"]).attempts == 2


def test_failure_refunds_in_full(client, monkeypatch):
    from weightroom import trainers

    key = signup(client)
    q = _quote(client, key)
    job = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key)).json()
    assert balance(client) == 10 - q["credits"]

    def broken(*a, **k):
        raise ValueError("the data broke the trainer")
    monkeypatch.setattr(trainers.FakeTrainer, "train", broken)
    drain()
    st = client.get(f"/v1/jobs/{job['job_id']}", headers=auth(key)).json()
    assert st["state"] == "failed" and st["error"]["credits_refunded"] is True
    assert balance(client) == 10


def test_dead_worker_job_is_picked_up(client):
    from weightroom import db, worker

    key = signup(client)
    q = _quote(client, key)
    job = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key)).json()
    with db.session() as s:  # a worker claimed it, then died
        j = s.get(db.Job, job["job_id"])
        j.state, j.worker_id, j.attempts = "training", "dead-worker", 1
        j.heartbeat_at = db.now() - timedelta(minutes=10)
    assert worker.claim_job("w2") == job["job_id"]
    with db.session() as s:
        j = s.get(db.Job, job["job_id"])
        j.heartbeat_at = db.now()
    assert worker.claim_job("w3") is None  # alive again: nobody else takes it


def test_other_accounts_cant_see_your_work(client):
    key_a = signup(client, "a@example.com")
    q = _quote(client, key_a)
    job = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key_a)).json()
    client.post("/account/logout", json={})
    key_b = signup(client, "b@example.com")
    assert client.get(f"/v1/jobs/{job['job_id']}", headers=auth(key_b)).status_code == 404
    assert client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key_b)).status_code == 404
    assert client.get("/v1/jobs", headers=auth(key_b)).json()["jobs"] == []


def test_huggingface_delivery_needs_a_saved_token(client):
    key = signup(client)
    q = _quote(client, key)
    job = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key)).json()
    drain()
    err = client.post(f"/v1/jobs/{job['job_id']}/deliver", json={"hf_repo": "my-model"},
                      headers=auth(key)).json()["error"]
    assert err["code"] == "hf_token_missing" and "/dashboard" in err["fix"]
    assert client.post("/account/hf-token", json={"token": "nope"}).status_code == 400
    assert client.post("/account/hf-token", json={"token": "hf_abc123"}).status_code == 200
    assert client.get("/account/me").json()["has_hf_token"] is True


def test_pages_and_guide(client):
    for path in ("/", "/signup", "/login", "/dashboard", "/docs/agents"):
        assert client.get(path).status_code == 200
    guide = client.get("/llms.txt").text
    for tool in ("upload_data", "check_data", "quote", "train", "status", "list_jobs", "deliver"):
        assert tool in guide
    assert client.get("/account/me").status_code == 401


@pytest.mark.parametrize("bad", [{"stream": True}, {"messages": "hi"}])
def test_endpoint_rejects_bad_requests(client, bad):
    key = signup(client)
    body = {"model": "job_x", "messages": [{"role": "user", "content": "hi"}], **bad}
    r = client.post("/v1/chat/completions", json=body, headers=auth(key))
    assert r.status_code == 400


def test_rechecking_after_a_quote_needs_a_new_quote(client):
    key = signup(client)
    ds = upload(client, key, banking_csv())
    client.post("/v1/datasets/check", json={"dataset_id": ds}, headers=auth(key))
    q = client.post("/v1/quotes", json={"dataset_id": ds}, headers=auth(key)).json()
    client.post("/v1/datasets/check", json={"dataset_id": ds, "instructions": "Reply with the intent."},
                headers=auth(key))
    err = client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key)).json()["error"]
    assert err["code"] == "dataset_changed" and "quote again" in err["fix"]
    q2 = client.post("/v1/quotes", json={"dataset_id": ds}, headers=auth(key)).json()
    assert client.post("/v1/jobs", json={"quote_id": q2["quote_id"]}, headers=auth(key)).status_code == 200
    assert balance(client) == 10 - q2["credits"]
