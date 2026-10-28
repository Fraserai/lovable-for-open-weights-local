import pytest


@pytest.fixture
def wr(tmp_path, monkeypatch):
    """A fresh Weightroom: SQLite + local storage in tmp_path, the fake trainer, Tinker prices pinned."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'wr.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://testserver")
    monkeypatch.setenv("TRAINER", "fake")
    monkeypatch.setenv("FAKE_STEP_SECONDS", "0")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("STORAGE", "local")
    from cryptography.fernet import Fernet

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    for name in ("STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "SCORE_JUDGE"):
        monkeypatch.delenv(name, raising=False)

    from weightroom import config, credits, db, storage, trainers

    from service_helpers import PRICES

    config.reload()
    db.reset_engine()
    storage.reset()
    trainers.reset()
    trainers.FakeTrainer.fail_at_step = None
    credits.set_prices_for_tests(PRICES)
    db.init()
    yield
    db.reset_engine()


@pytest.fixture
def client(wr):
    from fastapi.testclient import TestClient

    from weightroom.api import create_app

    with TestClient(create_app()) as c:
        yield c
