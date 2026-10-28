"""Operator commands, run on the server (e.g. `railway run python -m weightroom.admin ...`).

    python -m weightroom.admin new-secrets            # values for SECRET_KEY and ENCRYPTION_KEY
    python -m weightroom.admin grant you@x.com 50     # add credits by hand (testing, goodwill)
    python -m weightroom.admin set-plan you@x.com pro # change a plan without Stripe (e.g. Enterprise deals)
    python -m weightroom.admin jobs                   # the latest jobs across all accounts
    python -m weightroom.admin retention              # are customers still using what they trained?
"""

import secrets
import sys
from datetime import timedelta

from sqlalchemy import select

from . import credits, db


def _account(s, email: str) -> db.Account:
    account = s.scalar(select(db.Account).where(db.Account.email == email.strip().lower()))
    if account is None:
        sys.exit(f"No account for {email}.")
    return account


def retention(s) -> str:
    """The churn question: of models trained at least 30 days ago, how many are still being used?"""
    now = db.now()
    old = s.scalars(select(db.Job).where(db.Job.state == "done", db.Job.finished_at <= now - timedelta(days=30))).all()
    recent = s.scalars(select(db.Job).where(db.Job.state == "done", db.Job.finished_at > now - timedelta(days=30))).all()

    def used_within(job, days):
        last = db.aware(job.endpoint_last_used_at)
        return last is not None and last >= now - timedelta(days=days)

    def pct(n, d):
        return f"{n}/{d} ({n / d:.0%})" if d else "0/0"

    lines = [f"Models trained 30+ days ago: {len(old)}",
             f"  endpoint used in the last 7 days:  {pct(sum(used_within(j, 7) for j in old), len(old))}",
             f"  endpoint used in the last 30 days: {pct(sum(used_within(j, 30) for j in old), len(old))}",
             f"  ever called the endpoint:          {pct(sum((j.endpoint_calls or 0) > 0 for j in old), len(old))}"]
    accounts = {j.account_id for j in old}
    if accounts:
        paid = s.scalars(select(db.Account).where(db.Account.id.in_(accounts),
                                                  db.Account.plan.in_(("pro", "business", "enterprise")))).all()
        retrained = {j.account_id for j in recent} & accounts
        lines += [f"Accounts behind them: {len(accounts)}",
                  f"  still on a paid plan:              {pct(len(paid), len(accounts))}",
                  f"  trained again in the last 30 days: {pct(len(retrained), len(accounts))}"]
    lines.append(f"Models trained in the last 30 days: {len(recent)}, "
                 f"{sum((j.endpoint_calls or 0) > 0 for j in recent)} already called through the endpoint")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, args = argv[0], argv[1:]
    if cmd == "new-secrets":
        from cryptography.fernet import Fernet

        print(f"SECRET_KEY={secrets.token_urlsafe(48)}\nENCRYPTION_KEY={Fernet.generate_key().decode()}")
        return
    db.init()
    with db.session() as s:
        if cmd == "grant" and len(args) == 2:
            account = _account(s, args[0])
            credits.add(s, account.id, int(args[1]), "adjust", note="Granted by operator")
            print(f"{account.email}: balance {credits.balance(s, account.id)} credits")
        elif cmd == "set-plan" and len(args) == 2 and args[1] in credits.PLANS:
            account = _account(s, args[0])
            account.plan = args[1]
            print(f"{account.email}: plan {account.plan}")
        elif cmd == "jobs":
            for j in s.scalars(select(db.Job).order_by(db.Job.created_at.desc()).limit(30)):
                print(f"{j.id}  {j.state:9} {j.step}/{j.total_steps:<5} {j.credits:>4} cr  {j.base_model}  "
                      f"{(j.error or (j.scorecard or {}).get('summary') or '')[:90]}")
        elif cmd == "retention":
            print(retention(s))
        else:
            sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
