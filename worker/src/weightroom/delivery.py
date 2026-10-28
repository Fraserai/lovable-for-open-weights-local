"""Handing the model back: a download of the LoRA adapter, a push to the customer's private Hugging Face
repo, and a live endpoint (OpenAI-compatible, served through Weightroom and metered in credits).

Downloads and repo pushes move a lot of bytes, so they run on the worker: deliver() queues them and
returns straight away; calling deliver() or status() again shows when each is ready.
"""

import re
import tarfile
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import config, db, security
from .errors import WRError
from .storage import store
from .trainers import trainer

DOWNLOAD_TTL = 24 * 3600
_REPO = re.compile(r"^(?:[A-Za-z0-9][\w.-]{0,95}/)?[A-Za-z0-9][\w.-]{0,95}$")


def download_url(d: db.Delivery) -> str:
    st = store()
    signed = st.presigned_get(d.result, DOWNLOAD_TTL, "adapter.tar.gz")
    if signed:
        return signed
    token = security.make_token("download", d.id, DOWNLOAD_TTL)
    return f"{config.get().public_base_url}/v1/downloads/{token}"


def endpoint_info(job: db.Job) -> dict:
    base = config.get().public_base_url
    return {
        "base_url": f"{base}/v1",
        "model": job.id,
        "auth": "Authorization: Bearer <your Weightroom API key>",
        "openai_compatible": True,
        "billing": "Metered in credits at Tinker's sampling rates.",
        "example_curl": (f'curl -sS {base}/v1/chat/completions -H "Authorization: Bearer $WEIGHTROOM_API_KEY" '
                         f'-H "Content-Type: application/json" -d \'{{"model": "{job.id}", "messages": '
                         f'[{{"role": "user", "content": "Hello"}}]}}\''),
        "example_python": (f'from openai import OpenAI\nclient = OpenAI(base_url="{base}/v1", '
                           f'api_key=os.environ["WEIGHTROOM_API_KEY"])\nprint(client.chat.completions.create('
                           f'model="{job.id}", messages=[{{"role": "user", "content": "Hello"}}])'
                           f'.choices[0].message.content)'),
        **({"system_prompt": "Applied automatically (the model was trained with it)."} if job.system_prompt else {}),
    }


def request(s: Session, account: db.Account, job: db.Job, *, download: bool = True, hf_repo: str | None = None,
            endpoint: bool = True) -> dict:
    if job.state != "done":
        raise WRError("job_not_done", f"Job {job.id} is {job.state}, not done.",
                      "Wait until status says done." if job.state != "failed" else "This job failed; train again.",
                      retryable=job.state != "failed", status=409)
    base = config.get().public_base_url
    if hf_repo:
        hf_repo = hf_repo.strip().removeprefix("https://huggingface.co/")
        if not _REPO.match(hf_repo):
            raise WRError("bad_repo_name", f"'{hf_repo}' isn't a valid Hugging Face repo name.",
                          "Use 'name' (goes under your account) or 'owner/name'.")
        if not account.hf_token_enc:
            raise WRError("hf_token_missing", "Weightroom doesn't have a Hugging Face token for this account yet.",
                          f"Ask the user to add a Hugging Face write token at {base}/dashboard, then call deliver again.",
                          status=409)
    wanted = ([("download", "")] if download else []) + ([("huggingface", hf_repo)] if hf_repo else [])
    for kind, target in wanted:
        d = s.scalar(select(db.Delivery).where(db.Delivery.job_id == job.id, db.Delivery.kind == kind,
                                               db.Delivery.target == target))
        if d is None:
            s.add(db.Delivery(job_id=job.id, account_id=account.id, kind=kind, target=target))
        elif d.state == "failed":  # asking again retries it
            d.state, d.error = "pending", None
    s.flush()
    from .jobs import delivery_view

    rows = s.scalars(select(db.Delivery).where(db.Delivery.job_id == job.id)).all()
    views = [delivery_view(d) for d in rows if (d.kind, d.target) in wanted]
    out: dict = {"job_id": job.id}
    if endpoint:
        out["endpoint"] = endpoint_info(job)
    if views:
        out["deliveries"] = views
    pending = any(v["state"] in ("pending", "working") for v in views)
    if pending:
        out["check_back_in_seconds"] = 30
        out["next"] = (f"Preparing. Call deliver again with job_id='{job.id}' (same arguments) in about 30 seconds "
                       "for the links.")
    else:
        out["next"] = "Everything requested is ready."
    return out


MODEL_CARD = """---
base_model: {base_model}
library_name: peft
tags: [lora, weightroom]
---

# {repo_id}

A LoRA adapter for `{base_model}`, trained with Tinker{via}.

Load it on top of `{base_model}`.{system}
{scorecard}{badge}"""


def _card(job: db.Job, repo_id: str, plan: str) -> str:
    system = (f"\n\nUse the same system prompt it was trained with:\n\n```\n{job.system_prompt}\n```"
              if job.system_prompt else "")
    sc = (job.scorecard or {}).get("summary")
    scorecard = f"\n## Scorecard\n\n{sc}\n" if sc else ""
    badge = f"\nTrained with [Weightroom]({config.get().public_base_url}).\n" if plan == "free" else ""
    via = "" if plan == "free" else " via Weightroom"
    return MODEL_CARD.format(base_model=job.base_model, repo_id=repo_id, via=via, system=system,
                             scorecard=scorecard, badge=badge)


def run(delivery_id: str) -> None:
    """Does one queued delivery. Called by the worker."""
    with db.session() as s:
        d = s.get(db.Delivery, delivery_id)
        job = s.get(db.Job, d.job_id)
        account = s.get(db.Account, d.account_id)
        kind, target, plan = d.kind, d.target, account.plan
        hf_token = security.decrypt(account.hf_token_enc) if kind == "huggingface" and account.hf_token_enc else None
        s.expunge(job)
    from .jobs import work_dir

    work = work_dir(job.id) / "deliver"
    adapter = work / "adapter"
    if not (adapter / "adapter_config.json").exists():
        trainer().export(work / "tinker", job.sampler_path, job.base_model, adapter)

    if kind == "download":
        archive = work / "adapter.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for f in sorted(adapter.rglob("*")):
                if f.is_file():
                    tar.add(f, arcname=f"adapter/{f.relative_to(adapter)}")
        key = f"deliveries/{job.id}/adapter.tar.gz"
        store().put(key, archive)
        _finish(delivery_id, key)
        return

    if not hf_token:
        raise WRError("hf_token_missing", "No Hugging Face token saved.", "Add one in the dashboard.")
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    repo_id = target if "/" in target else f"{api.whoami()['name']}/{target}"
    (adapter / "README.md").write_text(_card(job, repo_id, plan), encoding="utf-8")
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(adapter), repo_id=repo_id, commit_message=f"Adapter from Weightroom job {job.id}")
    _finish(delivery_id, f"https://huggingface.co/{repo_id}")


def _finish(delivery_id: str, result: str) -> None:
    with db.session() as s:
        d = s.get(db.Delivery, delivery_id)
        d.state, d.result, d.error = "ready", result, None


def fail(delivery_id: str, message: str) -> None:
    with db.session() as s:
        d = s.get(db.Delivery, delivery_id)
        d.state, d.error = "failed", message[:1000]
