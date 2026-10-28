"""The Scorecard: the original model, the trained one and a frontier model, on the held-back test set.

Each contender answers every held-back question (the conversation up to the reference reply). Short
references (labels, one-word decisions) are marked by exact match; longer ones by a judge model that
checks the answer says what the reference says. The proof that a model beats the frontier at one thing.

Held-back questions are the customer's data: the judge always runs on zero-data-retention hosts, and a
frontier model only gets them without that guarantee when the dataset is public on Hugging Face.
"""

import hashlib
import json
import re
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ZDR = {"zdr": True, "data_collection": "deny"}
DEFAULT_FRONTIER = "openai/gpt-6-astra"
DEFAULT_JUDGE = "z-ai/glm-5.3"
EXACT_MAX_WORDS = 6  # references this short are labels: mark by exact match
_THINK = re.compile(r"<think>.*?</think>", re.S)

Answerer = Callable[[list[list[dict]]], list[str]]  # conversations (without the reference) -> answers


class ScoreError(RuntimeError):
    pass


def normalise(s: str) -> str:
    return s.strip().strip("`'\".").split("\n")[0].strip().lower()


def metric_for(references: list[str]) -> str:
    """"exact" when the reference replies are labels, "judge" when they're written answers."""
    return "exact" if statistics.median(len(r.split()) for r in references) <= EXACT_MAX_WORDS else "judge"


def test_hash(tests: list[list[dict]]) -> str:
    return hashlib.sha256(json.dumps(tests, sort_keys=True).encode()).hexdigest()[:16]


def _post(client: httpx.Client, key: str, body: dict, retries: int = 4) -> dict:
    for attempt in range(retries + 1):
        resp = client.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"}, json=body)
        if resp.status_code == 402:
            raise ScoreError("OpenRouter is out of credit. Add some at https://openrouter.ai/settings/credits.")
        if resp.status_code == 404 and body.get("provider", {}).get("zdr"):
            raise ScoreError(f"{body['model']} has no zero-data-retention host on OpenRouter, so it can't see "
                             "this data. Pick another model.")
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
        if resp.is_error:
            raise ScoreError(f"OpenRouter answered {resp.status_code} for {body['model']}: {resp.text[:200]}")
        return resp.json()
    raise ScoreError(f"OpenRouter kept failing for {body['model']}.")


def openrouter_answerer(model: str, key: str, zdr: bool, max_tokens: int,
                        transport: httpx.BaseTransport | None = None, workers: int = 8) -> Answerer:
    def answer(conversations: list[list[dict]]) -> list[str]:
        with httpx.Client(timeout=180, transport=transport) as client:
            def one(messages: list[dict]) -> str:
                body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
                if zdr:
                    body["provider"] = ZDR
                out = _post(client, key, body)["choices"][0]["message"].get("content") or ""
                return _THINK.sub("", out).strip()
            with ThreadPoolExecutor(workers) as pool:
                return list(pool.map(one, conversations))
    return answer


JUDGE_PROMPT = """You are marking an answer against a reference answer.
Question:
{question}

Reference answer:
{reference}

Answer to mark:
{answer}

Does the answer give the same substance as the reference (same facts, decision or conclusion)? Wording, length
and extra correct detail don't matter; missing or contradicting the key point does. Reply with exactly YES or NO."""


def judge_marker(model: str, key: str, transport: httpx.BaseTransport | None = None, workers: int = 8):
    """Returns mark(questions, references, answers) -> [bool]. Always zero-data-retention."""
    def mark(questions: list[str], references: list[str], answers: list[str]) -> list[bool]:
        with httpx.Client(timeout=120, transport=transport) as client:
            def one(i: int) -> bool:
                if not answers[i].strip():
                    return False
                prompt = JUDGE_PROMPT.format(question=questions[i], reference=references[i], answer=answers[i])
                body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                        "max_tokens": 5, "provider": ZDR}
                out = _post(client, key, body)["choices"][0]["message"].get("content") or ""
                return _THINK.sub("", out).strip().upper().startswith("YES")
            with ThreadPoolExecutor(workers) as pool:
                return list(pool.map(one, range(len(answers))))
    return mark


def mark_answers(metric: str, tests: list[list[dict]], answers: list[str], judge=None) -> list[bool]:
    references = [t[-1]["content"] for t in tests]
    if metric == "exact":
        return [normalise(a) == normalise(r) for a, r in zip(answers, references)]
    if judge is None:
        raise ScoreError("Written answers need a judge model to mark them.")
    questions = ["\n".join(f"{m['role']}: {m['content']}" for m in t[:-1] if m["role"] != "system") for t in tests]
    return judge(questions, references, answers)


def score(tests: list[list[dict]], answerers: dict[str, Answerer], judge=None, metric: str | None = None,
          previous: dict | None = None, reuse: tuple[str, ...] = ()) -> dict:
    """Builds the scorecard. Contenders in `reuse` keep their earlier result when the test set is unchanged."""
    if not tests:
        raise ScoreError("There's no held-back test set to score on.")
    metric = metric or metric_for([t[-1]["content"] for t in tests])
    digest = test_hash(tests)
    same_test = bool(previous) and previous.get("test_hash") == digest and previous.get("metric") == metric
    contenders: dict[str, dict] = {}
    prompts = [t[:-1] for t in tests]
    for name, answer in answerers.items():
        if same_test and name in reuse and name in previous.get("contenders", {}):
            contenders[name] = {**previous["contenders"][name], "reused": True}
            continue
        answers = answer(prompts)
        marks = mark_answers(metric, tests, answers, judge)
        contenders[name] = {"score": sum(marks) / len(marks), "correct": sum(marks), "answers": answers, "marks": marks}
    return {"metric": metric, "examples": len(tests), "test_hash": digest,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "contenders": contenders}


def tinker_answerers(state: dict, max_tokens: int) -> dict[str, Answerer]:
    """The original model and the trained one, sampled through Tinker (as the side-by-side chat does)."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from .playground import _text
    from .tinker import ensure_ssl_certificates, resolve_renderer

    ensure_ssl_certificates()
    base = state["base_model"]
    tok = get_tokenizer(base)
    fmt = (state.get("tinker_plan") or {}).get("renderer") or resolve_renderer(base)
    trained_r = renderers.get_renderer(fmt, tok, model_name=base)
    try:  # ask the original for a direct answer too, so the comparison is fair
        original_r = renderers.get_renderer(f"{fmt}_disable_thinking", tok, model_name=base)
    except Exception:  # noqa: BLE001 - this family has no such variant
        original_r = trained_r
    service = tinker.ServiceClient()

    def sampler(client, r) -> Answerer:
        def answer(conversations: list[list[dict]]) -> list[str]:
            params = tinker.types.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=r.get_stop_sequences())
            futures = [client.sample(prompt=r.build_generation_prompt(c), num_samples=1, sampling_params=params)
                       for c in conversations]
            return [_text(r, f.result().sequences[0].tokens) for f in futures]
        return answer

    out = {"original": sampler(service.create_sampling_client(base_model=base), original_r)}
    if state.get("sampler_path"):
        out["trained"] = sampler(service.create_sampling_client(model_path=state["sampler_path"]), trained_r)
    return out


def estimate_frontier_cost(tests: list[list[dict]], model: str, max_tokens: int,
                           transport: httpx.BaseTransport | None = None) -> float | None:
    """Rough dollars for the frontier model to answer the test set (4 characters ≈ 1 token)."""
    try:
        with httpx.Client(timeout=30, transport=transport) as client:
            data = client.get("https://openrouter.ai/api/v1/models").json()["data"]
        price = next(m["pricing"] for m in data if m["id"] == model)
    except Exception:  # noqa: BLE001 - no estimate rather than no scorecard
        return None
    prompt_tokens = sum(len(json.dumps(t[:-1])) for t in tests) / 4
    return prompt_tokens * float(price["prompt"]) + len(tests) * max_tokens * float(price["completion"])


def load_previous(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
