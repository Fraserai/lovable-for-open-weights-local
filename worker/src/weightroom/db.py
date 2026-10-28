"""Database: accounts, API keys, datasets, quotes, jobs, the credit ledger and deliveries.

Postgres in production (DATABASE_URL), SQLite for local runs and tests. Tables are created on start;
there's no migration tool yet, so add columns with care once real customers exist.
"""

import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
                        create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from . import config


def now() -> datetime:
    return datetime.now(timezone.utc)


def aware(dt: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; everything we store is UTC."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("acct"))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    plan: Mapped[str] = mapped_column(String(20), default="free")  # free | pro | business | enterprise
    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hf_token_enc: Mapped[str | None] = mapped_column(Text, nullable=True)
    free_grant_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    signup_ip: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    inference_carry_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("key"))
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    name: Mapped[str] = mapped_column(String(100), default="default")
    prefix: Mapped[str] = mapped_column(String(20))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    monthly_cap_credits: Mapped[int | None] = mapped_column(Integer, nullable=True)
    per_job_cap_credits: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ds"))
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True)
    state: Mapped[str] = mapped_column(String(20), default="awaiting_upload")  # awaiting_upload | uploaded | checked
    source: Mapped[str] = mapped_column(String(20), default="upload")  # upload | huggingface
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_format: Mapped[str | None] = mapped_column(String(10), nullable=True)  # jsonl | csv
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hf_dataset: Mapped[str | None] = mapped_column(String(255), nullable=True)
    upload_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    train_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    test_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    est_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Quote(Base):
    __tablename__ = "quotes"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("q"))
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    base_model: Mapped[str] = mapped_column(String(255))
    plan: Mapped[dict] = mapped_column(JSON)
    credits: Mapped[int] = mapped_column(Integer)
    usd_estimate: Mapped[float] = mapped_column(Float)
    minutes_estimate: Mapped[int] = mapped_column(Integer)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("quote_id", name="one_job_per_quote"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("job"))
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True, index=True)
    quote_id: Mapped[str] = mapped_column(ForeignKey("quotes.id"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    base_model: Mapped[str] = mapped_column(String(255))
    # queued | training | scoring | done | failed
    state: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    plan: Mapped[dict] = mapped_column(JSON)
    credits: Mapped[int] = mapped_column(Integer)
    refunded: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_retryable: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    scorecard: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sampler_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Endpoint use: the retention signal (is the customer still using the model they trained?)
    endpoint_calls: Mapped[int] = mapped_column(Integer, default=0)
    endpoint_last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CreditEntry(Base):
    """The ledger. Balances are sums of `delta`; nothing else stores a balance."""
    __tablename__ = "credit_ledger"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True, index=True)
    delta: Mapped[int] = mapped_column(Integer)
    # signup | free_monthly | plan | topup | train | refund | inference | adjust
    kind: Mapped[str] = mapped_column(String(20))
    job_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    note: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Makes Stripe webhook handling idempotent: the same invoice/session can't grant twice.
    external_ref: Mapped[str | None] = mapped_column(String(120), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)


class Delivery(Base):
    __tablename__ = "deliveries"
    __table_args__ = (UniqueConstraint("job_id", "kind", "target", name="one_delivery_per_target"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("dl"))
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    kind: Mapped[str] = mapped_column(String(20))  # download | huggingface
    target: Mapped[str] = mapped_column(String(255), default="")  # the HF repo, or "" for a download
    state: Mapped[str] = mapped_column(String(20), default="pending")  # pending | working | ready | failed
    result: Mapped[str | None] = mapped_column(String(500), nullable=True)  # storage key or repo URL
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


_engine = None
_Session: sessionmaker | None = None


def engine():
    global _engine, _Session
    if _engine is None:
        url = config.get().database_url
        if url.startswith("postgres://"):  # Railway/Heroku style
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(conn, _):  # noqa: ANN001
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def is_postgres() -> bool:
    return engine().dialect.name == "postgresql"


def init() -> None:
    Base.metadata.create_all(engine())


def reset_engine() -> None:
    """Forget the engine (tests point DATABASE_URL somewhere new)."""
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _Session = None


@contextmanager
def session() -> Iterator[Session]:
    engine()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
