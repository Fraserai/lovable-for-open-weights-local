"""Operator commands: retention reporting and manual credits."""

from datetime import timedelta

from sqlalchemy import select

from service_helpers import auth, banking_csv
from test_service_flow import drain, signup, upload


def test_retention_counts_models_still_in_use(client):
    from weightroom import admin, db

    key = signup(client)
    ids = []
    for _ in range(2):
        ds = upload(client, key, banking_csv())
        client.post("/v1/datasets/check", json={"dataset_id": ds}, headers=auth(key))
        q = client.post("/v1/quotes", json={"dataset_id": ds}, headers=auth(key)).json()
        ids.append(client.post("/v1/jobs", json={"quote_id": q["quote_id"]}, headers=auth(key)).json()["job_id"])
        admin.main(["grant", "founder@example.com", "10"])
    drain()
    r = client.post("/v1/chat/completions", headers=auth(key),
                    json={"model": ids[0], "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    with db.session() as s:
        for job_id in ids:  # pretend both were trained 40 days ago
            s.get(db.Job, job_id).finished_at = db.now() - timedelta(days=40)
        assert s.get(db.Job, ids[0]).endpoint_calls == 1
        report = admin.retention(s)
    assert "Models trained 30+ days ago: 2" in report
    assert "endpoint used in the last 7 days:  1/2 (50%)" in report
    assert "still on a paid plan:              0/1 (0%)" in report


def test_grant_and_set_plan(client, capsys):
    from weightroom import admin, db

    signup(client)
    admin.main(["grant", "founder@example.com", "25"])
    admin.main(["set-plan", "founder@example.com", "enterprise"])
    out = capsys.readouterr().out
    assert "balance 35 credits" in out and "plan enterprise" in out
    with db.session() as s:
        assert s.scalar(select(db.Account)).plan == "enterprise"
