"""Stripe: subscriptions (Pro, Business) and credit top-ups. Our own ledger is the record of credit.

Stripe tells us what was paid through webhooks; each payment grants credits once (the ledger's
external_ref makes a repeated webhook a no-op). Set the STRIPE_* variables (see DEPLOY.md).
"""

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import config, credits, db
from .errors import WRError


def _stripe():
    s = config.get()
    if not s.stripe_secret_key:
        raise WRError("billing_not_configured", "Billing isn't switched on for this server yet.",
                      "Ask the Weightroom operator to set the Stripe keys.", status=503)
    import stripe

    stripe.api_key = s.stripe_secret_key
    return stripe


def _price_for(item: str) -> str:
    s = config.get()
    price = {"pro": s.stripe_price_pro, "business": s.stripe_price_business, "topup": s.stripe_price_topup}.get(item)
    if not price:
        raise WRError("billing_not_configured", f"No Stripe price set for {item}.", "Set it in the server settings.", status=503)
    return price


def _plan_for_prices(price_ids: set[str]) -> str | None:
    s = config.get()
    if s.stripe_price_business and s.stripe_price_business in price_ids:
        return "business"
    if s.stripe_price_pro and s.stripe_price_pro in price_ids:
        return "pro"
    return None


def _ensure_customer(stripe, account: db.Account) -> str:
    if not account.stripe_customer_id:
        customer = stripe.Customer.create(email=account.email, metadata={"account_id": account.id})
        account.stripe_customer_id = customer["id"]
    return account.stripe_customer_id


def checkout(s: Session, account: db.Account, item: str) -> str:
    if item not in ("pro", "business", "topup"):
        raise WRError("bad_request", "Choose pro, business or topup.", "")
    if item == "topup" and not credits.PLANS[account.plan]["top_ups"]:
        raise WRError("plan_required", "Top-ups come with Pro and Business.", "Upgrade first.")
    if item in ("pro", "business") and account.plan != "free":
        raise WRError("already_subscribed", f"This account is already on {credits.PLANS[account.plan]['name']}.",
                      "Use Manage billing to change or cancel the plan.", status=409)
    stripe = _stripe()
    base = config.get().public_base_url
    params = {
        "customer": _ensure_customer(stripe, account),
        "line_items": [{"price": _price_for(item), "quantity": 1}],
        "mode": "payment" if item == "topup" else "subscription",
        "success_url": f"{base}/dashboard?paid={item}",
        "cancel_url": f"{base}/dashboard",
        "client_reference_id": account.id,
        "metadata": {"account_id": account.id, "item": item},
    }
    session = stripe.checkout.Session.create(**params)
    return session["url"]


def portal(s: Session, account: db.Account) -> str:
    stripe = _stripe()
    if not account.stripe_customer_id:
        raise WRError("no_billing", "There's no billing account yet.", "Choose a plan first.")
    return stripe.billing_portal.Session.create(customer=account.stripe_customer_id,
                                                return_url=f"{config.get().public_base_url}/dashboard")["url"]


def _strings(obj, prefix: str) -> set[str]:
    """Every string starting with `prefix` anywhere in a Stripe object (price ids move between API versions)."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            found |= _strings(v, prefix)
    elif isinstance(obj, list):
        for v in obj:
            found |= _strings(v, prefix)
    elif isinstance(obj, str) and obj.startswith(prefix):
        found.add(obj)
    return found


def _account_by_customer(s: Session, customer_id: str | None) -> db.Account | None:
    if not customer_id:
        return None
    return s.scalar(select(db.Account).where(db.Account.stripe_customer_id == customer_id))


def handle_webhook(s: Session, payload: bytes, signature: str | None) -> str:
    stripe = _stripe()
    try:
        stripe.Webhook.construct_event(payload, signature or "", config.get().stripe_webhook_secret)
    except Exception as e:  # noqa: BLE001 - bad signature or payload
        raise WRError("bad_signature", "Webhook signature check failed.", "", status=400) from e
    event = json.loads(payload)
    kind, obj = event.get("type"), (event.get("data") or {}).get("object") or {}

    if kind == "checkout.session.completed":
        meta = obj.get("metadata") or {}
        account = s.get(db.Account, meta.get("account_id") or obj.get("client_reference_id") or "")
        if account is None:
            return "ignored: no account"
        if obj.get("customer") and not account.stripe_customer_id:
            account.stripe_customer_id = obj["customer"]
        if meta.get("item") == "topup" and obj.get("payment_status") == "paid":
            ref = f"checkout:{obj['id']}"
            if not credits.has_ref(s, ref):
                credits.add(s, account.id, credits.TOPUP_CREDITS, "topup", note="Top-up", external_ref=ref)
            return "topup granted"
        if meta.get("item") in ("pro", "business"):
            account.plan = meta["item"]
            account.stripe_subscription_id = obj.get("subscription")
            return f"plan {meta['item']}"
        return "ignored"

    if kind in ("invoice.paid", "invoice.payment_succeeded"):
        if not str(obj.get("billing_reason") or "").startswith("subscription"):
            return "ignored: not a subscription invoice"  # e.g. an invoice for a one-off top-up
        account = _account_by_customer(s, obj.get("customer"))
        if account is None:
            return "ignored: unknown customer"
        plan = _plan_for_prices(_strings(obj, "price_")) or (account.plan if account.plan in ("pro", "business") else None)
        if plan is None:
            return "ignored: not a plan invoice"
        account.plan = plan
        ref = f"invoice:{obj['id']}"
        if not credits.has_ref(s, ref):
            credits.add(s, account.id, credits.PLANS[plan]["monthly_credits"], "plan",
                        note=f"{credits.PLANS[plan]['name']} monthly credits", external_ref=ref)
        return f"{plan} credits granted"

    if kind in ("customer.subscription.updated", "customer.subscription.deleted"):
        account = _account_by_customer(s, obj.get("customer"))
        if account is None:
            return "ignored: unknown customer"
        if kind.endswith("deleted") or obj.get("status") in ("canceled", "unpaid", "incomplete_expired"):
            account.plan, account.stripe_subscription_id = "free", None
            account.free_grant_at = None  # free credits resume from the next visit
            return "back to free"
        plan = _plan_for_prices(_strings(obj, "price_"))
        if plan:
            account.plan, account.stripe_subscription_id = plan, obj.get("id")
        return f"plan {account.plan}"
    return "ignored"
