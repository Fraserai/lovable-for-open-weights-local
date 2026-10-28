"""The job worker: a separate process that trains, scores and delivers. Restarting the API never touches it.

Jobs are claimed from the database, so several workers can run side by side (Postgres hands each job to
one of them with SKIP LOCKED). A worker that dies leaves its job's heartbeat to go stale; the next worker
picks it up and training resumes from Tinker's last checkpoint. Run it with:

    python -m weightroom.worker
"""

import logging
import os
import signal
import socket
import threading
import time
from datetime import timedelta

from sqlalchemy import and_, or_, select, update

from . import config, db, delivery, jobs
from .errors import WRError

log = logging.getLogger("weightroom.worker")
STALE_AFTER = timedelta(minutes=3)
IDLE_SLEEP = 2.0


def _claim(model, active_states: tuple[str, ...], ready_state: str, worker_id: str, extra=None):
    """Takes the oldest ready row, or one whose worker stopped heart-beating. Returns its id or None."""
    stale = db.now() - STALE_AFTER
    claimable = or_(model.state == ready_state,
                    and_(model.state.in_(active_states), or_(model.heartbeat_at.is_(None), model.heartbeat_at < stale)))
    with db.session() as s:
        q = select(model.id).where(claimable).order_by(model.created_at).limit(1)
        if db.is_postgres():
            q = q.with_for_update(skip_locked=True)
        row_id = s.scalar(q)
        if row_id is None:
            return None
        values = {"worker_id": worker_id, "heartbeat_at": db.now()}
        if extra:
            values.update(extra)
        result = s.execute(update(model).where(model.id == row_id, claimable).values(**values))
        return row_id if result.rowcount == 1 else None


def claim_job(worker_id: str) -> str | None:
    return _claim(db.Job, ("training", "scoring"), "queued", worker_id, {"attempts": db.Job.attempts + 1})


def claim_delivery(worker_id: str) -> str | None:
    return _claim(db.Delivery, ("working",), "pending", worker_id, {"state": "working"})


def _retryable(e: Exception) -> bool:
    if isinstance(e, WRError):
        return e.retryable
    return not isinstance(e, (ValueError, KeyError, TypeError))


def process_job(job_id: str, worker_id: str) -> None:
    with db.session() as s:
        attempts = s.get(db.Job, job_id).attempts
    if attempts > jobs.MAX_ATTEMPTS:
        jobs.fail(job_id, "Training kept failing after several attempts.", retryable=True)
        return
    try:
        jobs.run(job_id, worker_id)
        log.info("job %s done", job_id)
    except Exception as e:  # noqa: BLE001 - every failure is recorded on the job
        message = e.message if isinstance(e, WRError) else f"{type(e).__name__}: {e}"
        if _retryable(e) and attempts < jobs.MAX_ATTEMPTS:
            log.warning("job %s attempt %s failed, will retry: %s", job_id, attempts, message)
            with db.session() as s:  # back in the queue; training resumes from the last checkpoint
                job = s.get(db.Job, job_id)
                job.state, job.heartbeat_at, job.worker_id = "queued", None, None
        else:
            log.exception("job %s failed", job_id)
            jobs.fail(job_id, message, retryable=_retryable(e))


def process_delivery(delivery_id: str) -> None:
    try:
        delivery.run(delivery_id)
    except Exception as e:  # noqa: BLE001
        log.exception("delivery %s failed", delivery_id)
        delivery.fail(delivery_id, e.message if isinstance(e, WRError) else f"{type(e).__name__}: {e}")


def run_once(worker_id: str) -> bool:
    """Does one unit of work if any is waiting. Returns whether it did something."""
    job_id = claim_job(worker_id)
    if job_id:
        process_job(job_id, worker_id)
        return True
    delivery_id = claim_delivery(worker_id)
    if delivery_id:
        process_delivery(delivery_id)
        return True
    return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config.require_safe_production()
    db.init()
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    log.info("worker %s started", worker_id)
    while not stopping.is_set():
        try:
            busy = run_once(worker_id)
        except Exception:  # noqa: BLE001 - keep the worker alive (e.g. the database blipped)
            log.exception("worker loop error")
            busy = False
        if not busy:
            stopping.wait(IDLE_SLEEP)
    log.info("worker %s stopping", worker_id)


if __name__ == "__main__":
    main()
