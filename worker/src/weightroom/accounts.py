"""Accounts for people: sign up, log in, manage API keys, spend caps and the Hugging Face token.

Agents never use these; they use an API key. The dashboard page calls these through /account/*.
"""

import re
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import credits, db, security
from .errors import WRError

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SESSION_TTL = 14 * 24 * 3600
SIGNUPS_PER_IP_PER_DAY = 3  # free credits are real compute; stop scripted farming


def signup(s: Session, email: str, password: str, ip: str | None = None) -> tuple[db.Account, str]:
    email = (email or "").strip().lower()
    if not _EMAIL.match(email):
        raise WRError("bad_email", "That doesn't look like an email address.", "Check it and try again.")
    if len(password or "") < 10:
        raise WRError("weak_password", "Passwords need at least 10 characters.", "Choose a longer password.")
    if s.scalar(select(db.Account.id).where(db.Account.email == email)):
        raise WRError("email_taken", "There's already an account with that email.", "Log in instead.", status=409)
    if ip:
        recent = s.scalars(select(db.Account.id).where(db.Account.signup_ip == ip,
                                                       db.Account.created_at >= db.now() - timedelta(days=1))).all()
        if len(recent) >= SIGNUPS_PER_IP_PER_DAY:
            raise WRError("too_many_signups", "Too many new accounts from this network today.",
                          "Log in to an existing account, or try again tomorrow.", status=429)
    account = db.Account(email=email, password_hash=security.hash_password(password), free_grant_at=db.now(),
                         signup_ip=ip)
    s.add(account)
    s.flush()
    from . import config

    credits.add(s, account.id, config.get().free_monthly_credits, "signup", note="Welcome credits")
    _, raw = security.create_api_key(s, account.id, "first key")
    return account, raw


def login(s: Session, email: str, password: str) -> db.Account:
    account = s.scalar(select(db.Account).where(db.Account.email == (email or "").strip().lower()))
    if account is None or not security.check_password(password or "", account.password_hash):
        raise WRError("bad_login", "Email or password is wrong.", "Try again.", status=401)
    return account


def session_token(account: db.Account) -> str:
    return security.make_token("session", account.id, SESSION_TTL)


def from_session(s: Session, token: str | None) -> db.Account:
    account_id = security.read_token("session", token or "")
    account = s.get(db.Account, account_id) if account_id else None
    if account is None:
        raise WRError("not_logged_in", "Please log in.", "Log in at /login.", status=401)
    return account


def overview(s: Session, account: db.Account) -> dict:
    credits.top_up_free(s, account)
    keys = s.scalars(select(db.ApiKey).where(db.ApiKey.account_id == account.id, db.ApiKey.revoked_at.is_(None))
                     .order_by(db.ApiKey.created_at)).all()
    job_rows = s.scalars(select(db.Job).where(db.Job.account_id == account.id)
                         .order_by(db.Job.created_at.desc()).limit(20)).all()
    plan = credits.PLANS[account.plan]
    return {
        "email": account.email,
        "plan": account.plan,
        "plan_name": plan["name"],
        "balance_credits": credits.balance(s, account.id),
        "can_top_up": plan["top_ups"],
        "top_up": {"credits": credits.TOPUP_CREDITS, "usd": credits.TOPUP_USD},
        "has_hf_token": bool(account.hf_token_enc),
        "keys": [{"id": k.id, "name": k.name, "prefix": k.prefix + "…", "monthly_cap_credits": k.monthly_cap_credits,
                  "per_job_cap_credits": k.per_job_cap_credits, "spent_last_30_days": credits.spent_this_month(s, k.id),
                  "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None} for k in keys],
        "jobs": [{"job_id": j.id, "state": j.state, "base_model": j.base_model, "credits": j.credits,
                  "created_at": j.created_at.isoformat() if j.created_at else None,
                  "summary": (j.scorecard or {}).get("summary")} for j in job_rows],
    }


def _cap(value) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        cap = int(value)
    except (TypeError, ValueError) as e:
        raise WRError("bad_cap", "Caps are whole numbers of credits.", "Enter a number, or leave it blank for no cap.") from e
    if cap < 1:
        raise WRError("bad_cap", "Caps must be at least 1 credit.", "Leave it blank for no cap.")
    return cap


def new_key(s: Session, account: db.Account, name: str, monthly_cap=None, per_job_cap=None) -> dict:
    row, raw = security.create_api_key(s, account.id, (name or "key").strip(), _cap(monthly_cap), _cap(per_job_cap))
    return {"id": row.id, "api_key": raw, "note": "Copy this key now; it won't be shown again."}


def _own_key(s: Session, account: db.Account, key_id: str) -> db.ApiKey:
    key = s.get(db.ApiKey, key_id)
    if key is None or key.account_id != account.id or key.revoked_at:
        raise WRError("key_not_found", "No such key.", "Refresh the page.", status=404)
    return key


def set_caps(s: Session, account: db.Account, key_id: str, monthly_cap=None, per_job_cap=None) -> None:
    key = _own_key(s, account, key_id)
    key.monthly_cap_credits, key.per_job_cap_credits = _cap(monthly_cap), _cap(per_job_cap)


def revoke(s: Session, account: db.Account, key_id: str) -> None:
    _own_key(s, account, key_id).revoked_at = db.now()


def save_hf_token(s: Session, account: db.Account, token: str | None) -> None:
    token = (token or "").strip()
    if not token:
        account.hf_token_enc = None
        return
    if not token.startswith("hf_"):
        raise WRError("bad_hf_token", "Hugging Face tokens start with 'hf_'.",
                      "Create a write token at https://huggingface.co/settings/tokens.")
    account.hf_token_enc = security.encrypt(token)
