"""API keys, passwords, signed links and login cookies, and encryption of stored tokens."""

import base64
import hashlib
import hmac
import secrets
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import config, db
from .errors import WRError

KEY_PREFIX = "wr_"


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_api_key(s: Session, account_id: str, name: str = "default", monthly_cap: int | None = None,
                   per_job_cap: int | None = None) -> tuple[db.ApiKey, str]:
    """Returns (row, raw key). The raw key is shown once and never stored."""
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    row = db.ApiKey(account_id=account_id, name=name[:100] or "default", prefix=raw[:10], key_hash=hash_key(raw),
                    monthly_cap_credits=monthly_cap, per_job_cap_credits=per_job_cap)
    s.add(row)
    s.flush()
    return row, raw


def bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def authenticate(s: Session, authorization: str | None) -> tuple[db.Account, db.ApiKey]:
    raw = bearer(authorization)
    if not raw:
        raise WRError("missing_api_key", "No API key was sent.",
                      f"Send 'Authorization: Bearer <your key>'. Get a key at {config.get().public_base_url}/dashboard.",
                      status=401)
    key = s.scalar(select(db.ApiKey).where(db.ApiKey.key_hash == hash_key(raw)))
    if key is None or key.revoked_at is not None:
        raise WRError("invalid_api_key", "That API key isn't valid (wrong, or revoked).",
                      f"Create a new key at {config.get().public_base_url}/dashboard.", status=401)
    account = s.get(db.Account, key.account_id)
    key.last_used_at = db.now()
    return account, key


# Passwords: scrypt, stored as "scrypt$<salt>$<hash>"
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def check_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, digest_b64 = stored.split("$")
        salt, digest = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
    except ValueError:
        return False
    test = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return hmac.compare_digest(test, digest)


# Signed tokens: "<payload>.<expiry>.<signature>" for upload/download links and login cookies
def _sign(message: str) -> str:
    mac = hmac.new(config.get().secret_key.encode(), message.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def make_token(purpose: str, subject: str, ttl_seconds: int) -> str:
    expiry = int(time.time()) + ttl_seconds
    return f"{subject}.{expiry}.{_sign(f'{purpose}:{subject}:{expiry}')}"


def read_token(purpose: str, token: str) -> str | None:
    """The subject, if the token is genuine and unexpired."""
    try:
        subject, expiry_s, sig = token.rsplit(".", 2)
        expiry = int(expiry_s)
    except ValueError:
        return None
    if expiry < time.time() or not hmac.compare_digest(sig, _sign(f"{purpose}:{subject}:{expiry}")):
        return None
    return subject


# Customers' Hugging Face tokens, encrypted at rest
def _fernet():
    from cryptography.fernet import Fernet

    key = config.get().encryption_key
    if not key:
        raise WRError("server_not_configured", "This server can't store tokens yet (ENCRYPTION_KEY isn't set).",
                      "Ask the Weightroom operator to set ENCRYPTION_KEY.", status=503)
    return Fernet(key.encode())


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt(value: str) -> str:
    return _fernet().decrypt(value.encode()).decode()
