"""Plans, credits and prices.

Lovable-style: a subscription gives monthly credits, top-ups add more. 1 credit is about $0.041 of our
Tinker cost (CREDIT_VALUE_USD), so a small Qwen3.5-4B run is ~10 credits and a 1T-parameter Kimi run ~100.
The ledger (db.CreditEntry) is the only record of credit; a balance is the sum of its rows.
"""

import math
import time
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import config, db
from .errors import WRError

PLANS = {
    "free": {"name": "Free", "usd_per_month": 0, "monthly_credits": None, "top_ups": False},
    "pro": {"name": "Pro", "usd_per_month": 49, "monthly_credits": 100, "top_ups": True},
    "business": {"name": "Business", "usd_per_month": 149, "monthly_credits": 300, "top_ups": True},
    "enterprise": {"name": "Enterprise", "usd_per_month": None, "monthly_credits": None, "top_ups": True},
}
# Top-ups cost a little more per credit than any plan (50c vs Pro 49c, Business ~50c), so the plan is
# always the better deal and top-ups are for running out mid-month, never a way around the plan.
TOPUP_CREDITS = 40
TOPUP_USD = 20


def usd_to_credits(usd: float) -> int:
    return max(1, math.ceil(round(usd / config.get().credit_value_usd, 6)))


def balance(s: Session, account_id: str) -> int:
    return int(s.scalar(select(func.coalesce(func.sum(db.CreditEntry.delta), 0))
                        .where(db.CreditEntry.account_id == account_id)) or 0)


def spent_this_month(s: Session, api_key_id: str) -> int:
    """Credits this key has spent in the last 30 days (training and inference, net of refunds)."""
    since = db.now() - timedelta(days=30)
    total = s.scalar(select(func.coalesce(func.sum(db.CreditEntry.delta), 0)).where(
        db.CreditEntry.api_key_id == api_key_id, db.CreditEntry.created_at >= since,
        db.CreditEntry.kind.in_(("train", "refund", "inference"))))
    return -int(total or 0)


def add(s: Session, account_id: str, delta: int, kind: str, *, api_key_id: str | None = None,
        job_id: str | None = None, note: str | None = None, external_ref: str | None = None) -> db.CreditEntry:
    entry = db.CreditEntry(account_id=account_id, api_key_id=api_key_id, delta=delta, kind=kind, job_id=job_id,
                           note=note, external_ref=external_ref)
    s.add(entry)
    s.flush()
    return entry


def has_ref(s: Session, external_ref: str) -> bool:
    return s.scalar(select(db.CreditEntry.id).where(db.CreditEntry.external_ref == external_ref)) is not None


def top_up_free(s: Session, account: db.Account) -> None:
    """Free accounts get their monthly credits back up to the allowance (no rollover)."""
    if account.plan != "free":
        return
    last = db.aware(account.free_grant_at)
    if last and db.now() - last < timedelta(days=30):
        return
    allowance = config.get().free_monthly_credits
    missing = allowance - balance(s, account.id)
    account.free_grant_at = db.now()
    if missing > 0:
        add(s, account.id, missing, "free_monthly", note="Free plan monthly credits")


def check_spend(s: Session, account: db.Account, key: db.ApiKey, credits: int) -> None:
    """Refuses spend over the key's caps or the account's balance, with the exact way out."""
    base = config.get().public_base_url
    if key.per_job_cap_credits is not None and credits > key.per_job_cap_credits:
        raise WRError("over_job_cap", f"This costs {credits} credits; this API key's per-job cap is "
                      f"{key.per_job_cap_credits}.", f"Pick a smaller model or dataset, or raise the cap at {base}/dashboard.",
                      status=402, details={"credits": credits, "per_job_cap": key.per_job_cap_credits})
    if key.monthly_cap_credits is not None:
        spent = spent_this_month(s, key.id)
        if spent + credits > key.monthly_cap_credits:
            raise WRError("over_monthly_cap", f"This costs {credits} credits; this key has {key.monthly_cap_credits - spent} "
                          f"left of its {key.monthly_cap_credits}-credit monthly cap.",
                          f"Raise the cap at {base}/dashboard, or wait for older spend to roll off.", status=402,
                          details={"credits": credits, "spent_last_30_days": spent, "monthly_cap": key.monthly_cap_credits})
    have = balance(s, account.id)
    if have < credits:
        plan = PLANS[account.plan]
        fix = (f"Buy a top-up ({TOPUP_CREDITS} credits for ${TOPUP_USD}) at {base}/dashboard." if plan["top_ups"]
               else f"Upgrade to Pro (${PLANS['pro']['usd_per_month']}/month, {PLANS['pro']['monthly_credits']} credits) "
                    f"at {base}/dashboard.")
        raise WRError("insufficient_credits", f"This costs {credits} credits; the account has {have}.", fix, status=402,
                      details={"credits": credits, "balance": have})


# Prices: Tinker's published per-million-token rates, from its live catalog (cached for an hour)
_PRICES: dict = {"at": 0.0, "models": {}}
_CACHE_SECONDS = 3600


def _money(value) -> float | None:
    try:
        return float(str(value).lstrip("$")) if value not in (None, "") else None
    except ValueError:
        return None


def tinker_models(refresh: bool = False) -> dict[str, dict]:
    """tinker id -> {train, prefill, sample, context, name, params}."""
    if refresh or not _PRICES["models"] or time.time() - _PRICES["at"] > _CACHE_SECONDS:
        from trainloop.catalog import CatalogError, build_catalog

        try:
            entries = build_catalog(failures={})
        except (CatalogError, Exception) as e:  # noqa: BLE001 - network
            if _PRICES["models"]:
                return _PRICES["models"]
            raise WRError("prices_unavailable", "Couldn't load Tinker's current prices.",
                          "Try again in a minute.", retryable=True, status=503) from e
        models = {}
        for e in entries:
            if e["trainer"] != "tinker":
                continue
            models[e["trainer_model_id"]] = {
                "name": e["display_name"], "params": e["params"], "context": e["train_context_tokens"],
                "train": _money(e.get("train_price")), "prefill": _money(e.get("prefill_price")),
                "sample": _money(e.get("sample_price")),
            }
        _PRICES.update(at=time.time(), models=models)
    return _PRICES["models"]


def tinker_models_cached() -> dict[str, dict]:
    """Whatever prices are already loaded; never fetches (for status pacing hints)."""
    return _PRICES["models"]


def set_prices_for_tests(models: dict[str, dict]) -> None:
    _PRICES.update(at=time.time() + 10**9, models=models)
