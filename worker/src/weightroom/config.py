"""Settings for the hosted service, all from environment variables (see .env.example).

Defaults run everything locally with no outside services: SQLite, files on disk and a fake trainer
that spends nothing. Production sets DATABASE_URL, the S3 bucket, TRAINER=tinker and the Stripe keys.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value else default


def _int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "sqlite:///./weightroom.db"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./wr-data")).resolve())
    public_base_url: str = field(default_factory=lambda: _env("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/"))
    # Signs upload/download links and login cookies. Must be set (and kept) in production.
    secret_key: str = field(default_factory=lambda: _env("SECRET_KEY", "dev-only-not-secret"))
    # Fernet key for customers' Hugging Face tokens at rest. Generate: python -m weightroom.admin new-encryption-key
    encryption_key: str = field(default_factory=lambda: _env("ENCRYPTION_KEY"))

    storage: str = field(default_factory=lambda: _env("STORAGE", "local"))  # local | s3
    s3_endpoint: str = field(default_factory=lambda: _env("S3_ENDPOINT"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET"))
    s3_access_key_id: str = field(default_factory=lambda: _env("S3_ACCESS_KEY_ID"))
    s3_secret_access_key: str = field(default_factory=lambda: _env("S3_SECRET_ACCESS_KEY"))
    s3_region: str = field(default_factory=lambda: _env("S3_REGION", "auto"))

    trainer: str = field(default_factory=lambda: _env("TRAINER", "fake"))  # tinker | fake
    default_base_model: str = field(default_factory=lambda: _env("DEFAULT_BASE_MODEL", "Qwen/Qwen3.5-4B"))

    # 1 credit = this many dollars of our Tinker cost. Tinker raised prices on 17 July 2026 and may again:
    # change it here, not in code. Our 41c Banking77 run = 10 credits; the $4.12 Kimi run = 100.
    credit_value_usd: float = field(default_factory=lambda: _float("CREDIT_VALUE_USD", 0.041))
    free_monthly_credits: int = field(default_factory=lambda: _int("FREE_MONTHLY_CREDITS", 10))
    quote_ttl_minutes: int = field(default_factory=lambda: _int("QUOTE_TTL_MINUTES", 60))
    max_upload_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 200))

    # Written (open-ended) answers can be marked by a judge model on OpenRouter's zero-data-retention
    # hosts. Off by default: customer data then goes nowhere but our storage and Tinker.
    score_judge: bool = field(default_factory=lambda: _env("SCORE_JUDGE", "off").lower() in ("1", "on", "true"))
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))

    stripe_secret_key: str = field(default_factory=lambda: _env("STRIPE_SECRET_KEY"))
    stripe_webhook_secret: str = field(default_factory=lambda: _env("STRIPE_WEBHOOK_SECRET"))
    stripe_price_pro: str = field(default_factory=lambda: _env("STRIPE_PRICE_PRO"))
    stripe_price_business: str = field(default_factory=lambda: _env("STRIPE_PRICE_BUSINESS"))
    stripe_price_topup: str = field(default_factory=lambda: _env("STRIPE_PRICE_TOPUP"))
    enterprise_contact: str = field(default_factory=lambda: _env("ENTERPRISE_CONTACT", "mailto:caeseyfraser@gmail.com"))

    # Fake trainer pacing (tests set this to 0)
    fake_step_seconds: float = field(default_factory=lambda: _float("FAKE_STEP_SECONDS", 0.05))

    @property
    def is_production(self) -> bool:
        return self.public_base_url.startswith("https://")


def production_problems(s: "Settings") -> list[str]:
    """Settings that would be dangerous on a public server. The API and worker refuse to start with any."""
    if not s.is_production:
        return []
    problems = []
    if s.secret_key in ("", "dev-only-not-secret"):
        problems.append("SECRET_KEY isn't set: login cookies and upload/download links could be forged.")
    if not s.encryption_key:
        problems.append("ENCRYPTION_KEY isn't set: customers' Hugging Face tokens can't be stored.")
    if s.trainer != "tinker" and _env("ALLOW_FAKE_TRAINER") != "1":
        problems.append("TRAINER isn't 'tinker': customers would be charged for fake training runs.")
    if s.storage != "s3" and _env("ALLOW_LOCAL_STORAGE") != "1":
        problems.append("STORAGE isn't 's3': the web service and the worker can't share files on separate machines.")
    if s.database_url.startswith("sqlite") and _env("ALLOW_SQLITE") != "1":
        problems.append("DATABASE_URL isn't set: the web service and the worker need a shared Postgres database.")
    return problems


def require_safe_production() -> None:
    problems = production_problems(get())
    if problems:
        raise RuntimeError("Weightroom won't start on a public URL with these settings:\n- " + "\n- ".join(problems)
                           + "\nSee DEPLOY.md.")


settings = Settings()


def reload() -> Settings:
    """Re-read the environment (tests change it)."""
    global settings
    settings = Settings()
    return settings


def get() -> Settings:
    return settings
