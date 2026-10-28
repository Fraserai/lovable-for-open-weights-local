"""Quotes, training jobs, and the Scorecard.

The flow an agent follows: quote (free, exact price, expires in an hour) -> train (needs the quote;
calling it twice with the same quote returns the same job and never charges twice) -> status (poll;
says when to check back) -> the Scorecard when done.

Credits are reserved when training starts and refunded in full if the job fails. The quote is the price.
"""

import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from trainloop import holdout, score

from . import config, credits, data, db
from .errors import WRError, not_found
from .storage import store
from .trainers import trainer

LABEL_MAX_TOKENS = 32
WRITTEN_MAX_TOKENS = 512
SIDE_BY_SIDE_EXAMPLES = 5
MAX_ATTEMPTS = 3


def _seconds_per_step(params: int | None) -> float:
    """Rough Tinker pace, from our runs: Qwen3.5-4B ~2.6s/step, Kimi-K2.6 (1T MoE) ~13s/step."""
    if not params or params <= 12e9:
        return 2.6
    if params <= 40e9:
        return 5.0
    return 13.0


def quote(s: Session, account: db.Account, key: db.ApiKey, dataset_id: str, base_model: str | None = None,
          epochs: int | None = None) -> dict:
    ds = s.get(db.Dataset, dataset_id)
    if ds is None or ds.account_id != account.id:
        raise not_found("dataset", dataset_id)
    if ds.state != "checked":
        raise WRError("dataset_not_ready", "This dataset hasn't passed check_data yet.",
                      f"Call check_data with dataset_id='{ds.id}' and fix any errors it lists.")
    base_model = (base_model or config.get().default_base_model).strip()
    models = credits.tinker_models()
    if base_model not in models:
        some = ", ".join(list(models)[:12])
        raise WRError("unknown_model", f"{base_model} isn't a model Weightroom can train.",
                      f"Use a Tinker model id, e.g. {config.get().default_base_model}. Others: {some}.",
                      details={"available": list(models)})
    if epochs is not None and not 1 <= int(epochs) <= 10:
        raise WRError("bad_request", "epochs must be between 1 and 10.", "Leave epochs unset to size it to the data.")
    price = models[base_model]
    if price.get("train") is None:
        raise WRError("prices_unavailable", f"Tinker hasn't published a training price for {base_model}.",
                      "Pick another model.", details={"model": base_model})

    with tempfile.TemporaryDirectory() as tmp:
        train = store().get(data.train_key(ds.id), Path(tmp) / "train.jsonl")
        p = trainer().plan(train, Path(tmp) / "work", base_model, {"epochs": epochs} if epochs else {})

    per_epoch = max(1, p["examples"] // p["batch_size"])
    train_tokens = round((ds.est_tokens or 0) * p["steps"] / per_epoch)
    usd = train_tokens / 1e6 * price["train"]
    task = (ds.report or {}).get("task", {})
    score_tokens = LABEL_MAX_TOKENS if task.get("metric") == "exact_match" else WRITTEN_MAX_TOKENS
    held = ds.test_rows or 0
    if held:
        avg_prompt = (ds.est_tokens or 0) / max(1, ds.train_rows or 1)
        prefill, sample = price.get("prefill") or price["train"], price.get("sample") or price["train"]
        examples = held if task.get("metric") == "exact_match" or config.get().score_judge else SIDE_BY_SIDE_EXAMPLES
        usd += 2 * examples * (avg_prompt * prefill + score_tokens * sample) / 1e6
    cost = credits.usd_to_credits(usd)
    minutes = max(2, round((p["steps"] * _seconds_per_step(price.get("params")) + 120) / 60))

    plan = {"settings": {"epochs": p["epochs"], "batch_size": p["batch_size"], "learning_rate": p["learning_rate"],
                         "lora_rank": p["lora_rank"]},
            "steps": p["steps"], "examples": p["examples"], "renderer": p["renderer"], "train_tokens": train_tokens,
            "truncated_share": p.get("truncated_share", 0), "max_length": p.get("max_length"),
            "est_cost_usd": round(usd, 4), "score_max_tokens": score_tokens}
    q = db.Quote(account_id=account.id, api_key_id=key.id, dataset_id=ds.id, base_model=base_model, plan=plan,
                 credits=cost, usd_estimate=round(usd, 4), minutes_estimate=minutes,
                 expires_at=db.now() + timedelta(minutes=config.get().quote_ttl_minutes))
    s.add(q)
    s.flush()

    have = credits.balance(s, account.id)
    out = {
        "quote_id": q.id,
        "dataset_id": ds.id,
        "base_model": base_model,
        "credits": cost,
        "price_usd_at_top_up_rate": round(cost * credits.TOPUP_USD / credits.TOPUP_CREDITS, 2),
        "minutes_estimate": minutes,
        "training": {"examples": p["examples"], "epochs": p["epochs"], "steps": p["steps"],
                     "lora_rank": p["lora_rank"], "learning_rate": p["learning_rate"]},
        "includes_scorecard": bool(held),
        "balance_credits": have,
        "expires_in_minutes": config.get().quote_ttl_minutes,
    }
    notes = []
    if p.get("truncated_share", 0) > 0.02:
        notes.append(f"About {p['truncated_share']:.0%} of examples are longer than {p['max_length']:,} tokens and "
                     "will be cut short.")
    try:
        credits.check_spend(s, account, key, cost)
        out["can_start"] = True
        out["next"] = (f"Show the user this price ({cost} credits, ~{minutes} min) and get a yes, then call train "
                       f"with quote_id='{q.id}'.")
    except WRError as e:
        out["can_start"] = False
        out["blocked"] = e.to_dict()["error"]
        out["next"] = e.fix
    if notes:
        out["notes"] = notes
    return out


def start(s: Session, account: db.Account, key: db.ApiKey, quote_id: str) -> dict:
    q = s.get(db.Quote, quote_id)
    if q is None or q.account_id != account.id:
        raise not_found("quote", quote_id)
    existing = s.scalar(select(db.Job).where(db.Job.quote_id == q.id))
    if existing:  # a retry: same job, no second charge
        return {**status_of(s, existing), "already_started": True}
    if db.aware(q.expires_at) < db.now():
        raise WRError("quote_expired", "This quote has expired (quotes last an hour, as prices can change).",
                      f"Call quote again with dataset_id='{q.dataset_id}' and base_model='{q.base_model}'.")
    ds = s.get(db.Dataset, q.dataset_id)
    changed = (ds.report or {}).get("checked_at")
    changed = datetime.fromisoformat(changed) if changed else None
    if ds.state != "checked" or (changed and changed > db.aware(q.created_at)):
        raise WRError("dataset_changed", "The dataset was uploaded or checked again after this quote.",
                      f"Call check_data if needed, then quote again with dataset_id='{q.dataset_id}'.")
    if db.is_postgres():  # one spend at a time per account, so two trains can't both pass the balance check
        s.get(db.Account, account.id, with_for_update=True, populate_existing=True)
    credits.check_spend(s, account, key, q.credits)
    job = db.Job(account_id=account.id, api_key_id=key.id, quote_id=q.id, dataset_id=q.dataset_id,
                 base_model=q.base_model, plan=q.plan, credits=q.credits, total_steps=q.plan["steps"],
                 system_prompt=ds.instructions if ds else None)
    try:
        with s.begin_nested():
            s.add(job)
            s.flush()
            credits.add(s, account.id, -q.credits, "train", api_key_id=key.id, job_id=job.id,
                        note=f"Training {q.base_model}")
    except IntegrityError:  # a simultaneous retry won the race
        existing = s.scalar(select(db.Job).where(db.Job.quote_id == q.id))
        return {**status_of(s, existing), "already_started": True}
    return {**status_of(s, job), "already_started": False}


def get_job(s: Session, account: db.Account, job_id: str) -> db.Job:
    job = s.get(db.Job, job_id)
    if job is None or job.account_id != account.id:
        raise not_found("job", job_id)
    return job


def status_of(s: Session, job: db.Job) -> dict:
    out = {
        "job_id": job.id,
        "state": job.state,
        "base_model": job.base_model,
        "dataset_id": job.dataset_id,
        "progress": {"step": job.step, "total_steps": job.total_steps},
        "credits": job.credits,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }
    remaining = max(0, job.total_steps - job.step)
    if job.state == "queued":
        out["check_back_in_seconds"] = 30
        out["next"] = "Queued. Check status again shortly."
    elif job.state == "training":
        pace = _seconds_per_step((credits.tinker_models_cached().get(job.base_model) or {}).get("params"))
        out["check_back_in_seconds"] = int(max(30, min(300, remaining * pace / 3)))
        out["minutes_left_estimate"] = max(1, round((remaining * pace + 90) / 60))
        out["next"] = "Training. Check status again later; nothing else is needed."
    elif job.state == "scoring":
        out["check_back_in_seconds"] = 60
        out["next"] = "Training finished; scoring the trained model against the original on held-back rows."
    elif job.state == "done":
        out["scorecard"] = job.scorecard
        out["finished_at"] = job.finished_at.isoformat() if job.finished_at else None
        out["next"] = (f"Show the user the scorecard. Then call deliver with job_id='{job.id}' for the weights "
                       "(download), a private Hugging Face repo, or a live endpoint.")
    elif job.state == "failed":
        out["error"] = {"message": job.error, "credits_refunded": job.refunded, "retry_helps": job.error_retryable}
        out["next"] = ("Credits were refunded. " + ("Get a new quote and train again." if job.error_retryable
                       else "Fix the cause above before trying again."))
    deliveries = s.scalars(select(db.Delivery).where(db.Delivery.job_id == job.id)).all()
    if deliveries:
        out["deliveries"] = [delivery_view(d) for d in deliveries]
    return out


def delivery_view(d: db.Delivery) -> dict:
    from .delivery import download_url

    view = {"kind": d.kind, "state": d.state}
    if d.target:
        view["repo"] = d.target
    if d.state == "ready":
        view["url"] = download_url(d) if d.kind == "download" else d.result
    if d.state == "failed":
        view["error"] = d.error
    return view


def list_jobs(s: Session, account: db.Account, limit: int = 20) -> dict:
    rows = s.scalars(select(db.Job).where(db.Job.account_id == account.id)
                     .order_by(db.Job.created_at.desc()).limit(max(1, min(int(limit or 20), 100)))).all()
    return {"jobs": [{"job_id": j.id, "state": j.state, "base_model": j.base_model, "dataset_id": j.dataset_id,
                      "credits": j.credits, "created_at": j.created_at.isoformat() if j.created_at else None,
                      **({"headline": (j.scorecard or {}).get("summary")} if j.state == "done" else {})}
                     for j in rows]}


# Running a job (called by the worker) -------------------------------------------------------------

def work_dir(job_id: str) -> Path:
    return config.get().data_dir / "jobs" / job_id


def _update(job_id: str, **values) -> None:
    with db.session() as s:
        job = s.get(db.Job, job_id)
        for k, v in values.items():
            setattr(job, k, v)


def fail(job_id: str, message: str, retryable: bool) -> None:
    """Marks the job failed and refunds its credits (once)."""
    with db.session() as s:
        job = s.get(db.Job, job_id)
        job.state, job.error, job.error_retryable = "failed", message[:2000], retryable
        job.finished_at = db.now()
        if not job.refunded:
            credits.add(s, job.account_id, job.credits, "refund", api_key_id=job.api_key_id, job_id=job.id,
                        note="Refund: training failed")
            job.refunded = True


def run(job_id: str, worker_id: str) -> None:
    """Trains, scores and finishes one job. Safe to call again on a job a dead worker left behind:
    training resumes from Tinker's last checkpoint in the job's work directory."""
    with db.session() as s:
        job = s.get(db.Job, job_id)
        base, plan, sampler, dataset_id = job.base_model, job.plan, job.sampler_path, job.dataset_id
    work = work_dir(job_id)
    work.mkdir(parents=True, exist_ok=True)
    st = store()
    if not (work / "train.jsonl").exists():
        st.get(data.train_key(dataset_id), work / "train.jsonl")
    if st.exists(data.test_key(dataset_id)) and not (work / "test.jsonl").exists():
        st.get(data.test_key(dataset_id), work / "test.jsonl")

    t = trainer()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            try:
                _update(job_id, heartbeat_at=db.now())
            except Exception:  # noqa: BLE001 - next beat will try again
                pass

    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()
    try:
        if not sampler:
            p = t.plan(work / "train.jsonl", work / "tinker", base, plan["settings"])
            _update(job_id, state="training", total_steps=p["steps"], started_at=db.now(), heartbeat_at=db.now())
            last = {"t": 0.0}

            def on_step(step: int) -> None:
                import time
                if time.time() - last["t"] > 5 or step >= p["steps"]:
                    last["t"] = time.time()
                    _update(job_id, step=step, heartbeat_at=db.now())

            sampler = t.train(work / "tinker", base, p, job_id, on_step)
            _update(job_id, sampler_path=sampler, step=p["steps"], state="scoring", heartbeat_at=db.now())
        else:
            _update(job_id, state="scoring", heartbeat_at=db.now())
        scorecard = build_scorecard(t, base, plan.get("renderer") or "", sampler, work, plan)
        _update(job_id, scorecard=scorecard, state="done", finished_at=db.now(), error=None)
    finally:
        stop.set()


def build_scorecard(t, base_model: str, renderer: str, sampler: str, work: Path, plan: dict) -> dict:
    tests = holdout.load(work / "test.jsonl")
    if not tests:
        return {"available": False, "summary": "No scorecard: the dataset had too few rows to hold any back."}
    references = [m[-1]["content"] for m in tests]
    metric = score.metric_for(references)
    max_tokens = LABEL_MAX_TOKENS if metric == "exact" else WRITTEN_MAX_TOKENS
    answerers = t.answerers(base_model, renderer, sampler, max_tokens, work_dir=work)
    s = config.get()

    if metric == "exact" or (s.score_judge and s.openrouter_api_key):
        judge = score.judge_marker(score.DEFAULT_JUDGE, s.openrouter_api_key) if metric != "exact" else None
        result = score.score(tests, answerers, judge=judge, metric=metric)
        orig, trained = result["contenders"]["original"], result["contenders"]["trained"]
        card = {
            "available": True,
            "metric": "exact_match" if metric == "exact" else "judged_match",
            "held_out_rows": len(tests),
            "original": {"model": base_model, "accuracy": round(orig["score"], 4), "correct": orig["correct"]},
            "trained": {"accuracy": round(trained["score"], 4), "correct": trained["correct"]},
            "improvement_points": round((trained["score"] - orig["score"]) * 100, 1),
            "summary": (f"On {len(tests)} held-back rows the model never trained on, accuracy went from "
                        f"{orig['score']:.1%} (original {base_model}) to {trained['score']:.1%} (your model)."),
        }
        if metric == "exact":
            card["per_label"] = _per_label(references, orig["marks"], trained["marks"])
        fixed = [i for i in range(len(tests)) if trained["marks"][i] and not orig["marks"][i]]
        card["fixed_examples"] = [{"input": _last_user(tests[i]), "original_said": orig["answers"][i][:200],
                                   "trained_said": trained["answers"][i][:200]} for i in fixed[:3]]
        return card

    prompts = [t_[:-1] for t_ in tests[:SIDE_BY_SIDE_EXAMPLES]]
    orig_answers = answerers["original"](prompts)
    trained_answers = answerers["trained"](prompts)
    return {
        "available": True,
        "metric": "none",
        "held_out_rows": len(tests),
        "summary": ("These are written answers, so there's no automatic score. Compare the original and trained "
                    f"answers on {len(prompts)} held-back rows below."),
        "examples": [{"input": _last_user(tests[i]), "expected": tests[i][-1]["content"][:500],
                      "original_said": orig_answers[i][:500], "trained_said": trained_answers[i][:500]}
                     for i in range(len(prompts))],
    }


def _last_user(conversation: list[dict]) -> str:
    return next((m["content"] for m in reversed(conversation[:-1]) if m["role"] == "user"), "")[:300]


def _per_label(references: list[str], orig: list[bool], trained: list[bool]) -> list[dict]:
    by_label: dict[str, list[int]] = {}
    for i, r in enumerate(references):
        by_label.setdefault(score.normalise(r), []).append(i)
    biggest = sorted(by_label.items(), key=lambda kv: -len(kv[1]))[:5]
    return [{"label": label, "rows": len(idx),
             "original_accuracy": round(sum(orig[i] for i in idx) / len(idx), 3),
             "trained_accuracy": round(sum(trained[i] for i in idx) / len(idx), 3)} for label, idx in biggest]

