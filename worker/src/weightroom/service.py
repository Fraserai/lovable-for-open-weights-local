"""The seven tools, shared by the REST API and the MCP server, plus the trained-model endpoint.

Each call authenticates the API key, runs in one database transaction (an error rolls everything back,
so a failed train never charges), and returns compact JSON.
"""

import math
import time

from . import config, credits, data, db, delivery, jobs, security
from .errors import WRError
from .trainers import trainer

# Used only if Tinker's price list can't be loaded: its highest published rates (Kimi-K2.6), per million tokens
FALLBACK_PREFILL_USD, FALLBACK_SAMPLE_USD = 2.205, 5.49


def _call(authorization: str | None, fn, *args, **kwargs) -> dict:
    with db.session() as s:
        account, key = security.authenticate(s, authorization)
        credits.top_up_free(s, account)
        return fn(s, account, key, *args, **kwargs)


def upload_data(authorization: str | None, filename: str) -> dict:
    return _call(authorization, data.create_upload, filename)


def check_data(authorization: str | None, **kwargs) -> dict:
    return _call(authorization, lambda s, a, k: data.check(s, a, k, **kwargs))


def quote(authorization: str | None, dataset_id: str, base_model: str | None = None, epochs: int | None = None) -> dict:
    return _call(authorization, jobs.quote, dataset_id, base_model, epochs)


def train(authorization: str | None, quote_id: str) -> dict:
    return _call(authorization, jobs.start, quote_id)


def status(authorization: str | None, job_id: str) -> dict:
    return _call(authorization, lambda s, a, k: jobs.status_of(s, jobs.get_job(s, a, job_id)))


def list_jobs(authorization: str | None, limit: int = 20) -> dict:
    return _call(authorization, lambda s, a, k: jobs.list_jobs(s, a, limit))


def deliver(authorization: str | None, job_id: str, download: bool = True, hf_repo: str | None = None,
            endpoint: bool = True) -> dict:
    return _call(authorization, lambda s, a, k: delivery.request(s, a, jobs.get_job(s, a, job_id), download=download,
                                                                 hf_repo=hf_repo, endpoint=endpoint))


def list_models(authorization: str | None) -> dict:
    def go(s, a, k):
        models = credits.tinker_models()
        return {"default": config.get().default_base_model,
                "models": [{"id": mid, "name": m["name"], "params": m["params"], "context": m["context"],
                            "train_usd_per_million_tokens": m["train"]} for mid, m in models.items()],
                "note": "quote gives the exact price in credits for your dataset."}
    return _call(authorization, go)


# The trained model, OpenAI-compatible -------------------------------------------------------------

def chat_completion(authorization: str | None, body: dict) -> dict:
    if body.get("stream"):
        raise WRError("stream_not_supported", "Streaming isn't supported yet.", "Send the request with stream=false.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or not all(
            isinstance(m, dict) and m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str)
            for m in messages):
        raise WRError("bad_request", "messages must be a list of {role, content} with text content.",
                      'Send e.g. [{"role": "user", "content": "Hello"}].')
    max_tokens = max(1, min(int(body.get("max_tokens") or body.get("max_completion_tokens") or 512), 4096))
    temperature = max(0.0, min(float(body.get("temperature", 0.7)), 2.0))

    with db.session() as s:
        account, key = security.authenticate(s, authorization)
        job = jobs.get_job(s, account, str(body.get("model") or ""))
        if job.state != "done" or not job.sampler_path:
            raise WRError("model_not_ready", f"Job {job.id} isn't finished training.", "Wait until status says done.",
                          status=409)
        if credits.balance(s, account.id) < 1:
            credits.check_spend(s, account, key, 1)
        if key.monthly_cap_credits is not None and credits.spent_this_month(s, key.id) >= key.monthly_cap_credits:
            credits.check_spend(s, account, key, 1)
        base, renderer, sampler, system = job.base_model, job.plan.get("renderer"), job.sampler_path, job.system_prompt
        account_id, key_id = account.id, key.id

    if system and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system}, *messages]
    text, prompt_tokens, completion_tokens = trainer().chat(base, renderer, sampler, messages, max_tokens, temperature)

    price = credits.tinker_models_cached().get(base) or credits.tinker_models().get(base) or {}
    if not price:  # never serve for free: bill at a conservative rate until prices load
        price = {"prefill": FALLBACK_PREFILL_USD, "sample": FALLBACK_SAMPLE_USD}
    usd = (prompt_tokens * (price.get("prefill") or price.get("train") or FALLBACK_PREFILL_USD)
           + completion_tokens * (price.get("sample") or price.get("train") or FALLBACK_SAMPLE_USD)) / 1e6
    with db.session() as s:
        used = s.get(db.Job, job.id)
        used.endpoint_calls = (used.endpoint_calls or 0) + 1
        used.endpoint_last_used_at = db.now()
        account = s.get(db.Account, account_id)
        account.inference_carry_usd = (account.inference_carry_usd or 0.0) + usd
        whole = math.floor(account.inference_carry_usd / config.get().credit_value_usd)
        if whole:
            credits.add(s, account_id, -whole, "inference", api_key_id=key_id, job_id=job.id,
                        note="Endpoint usage")
            account.inference_carry_usd -= whole * config.get().credit_value_usd

    return {
        "id": f"chatcmpl-{db.new_id('wr')}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": job.id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }
