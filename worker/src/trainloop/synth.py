"""Write chat training examples from document chunks + the customer's plain-English goal.

Chunks are customer content, so by default every request goes to OpenRouter pinned to
zero-data-retention providers. OpenAI is an explicit opt-in (GENERATOR_PROVIDER=openai)
for test or public documents only, since its API retains data unless under a ZDR agreement.
"""

import asyncio
import json
import re

import httpx

from .ingest import Chunk

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ZDR_PROVIDER = {"zdr": True, "data_collection": "deny", "require_parameters": True}

_EXAMPLES_SCHEMA = {
    "name": "training_examples",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
                    "required": ["user", "assistant"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    },
}

_SYSTEM_PROMPT_SCHEMA = {
    "name": "system_prompt",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"system_prompt": {"type": "string"}},
        "required": ["system_prompt"],
        "additionalProperties": False,
    },
}

_EXAMPLES_INSTRUCTIONS = """You create supervised fine-tuning data.

The organisation's goal for their model:
<goal>
{goal}
</goal>

The knowledge below is what the model will know after training. Write {n} realistic conversations
that serve the goal, using only facts from that knowledge.
- "user": something a real user of this model would plausibly send. Vary phrasing, length and intent.
- "assistant": the ideal reply, spoken by the model as its own knowledge, like an expert who simply
  knows this. Never invent facts, figures or policies it doesn't contain; simple arithmetic on its
  figures is fine. Match the tone the goal implies. When a question goes beyond what is known, say so
  plainly and suggest who to contact. The user cannot see any reference material, so the reply must
  never mention one.
- If the knowledge has little useful content for the goal (e.g. a table of contents, legal boilerplate),
  return fewer conversations or an empty list.

<knowledge from="{source}" topic="{heading}">
{text}
</knowledge>"""

# Replies that talk about the material they were written from teach the model to reference
# documents the user never sees. The prompt avoids this; this catches what slips through.
_LEAK = re.compile(r"\b(the|this|that|provided|given)\s+(excerpt|source|document|passage|knowledge|text provided|reference material)\b"
                   r"|\bexcerpt\b", re.IGNORECASE)


def leaks_source(reply: str) -> bool:
    return bool(_LEAK.search(reply))


_SYSTEM_PROMPT_INSTRUCTIONS = """Write a concise system prompt (2-4 sentences) for a model fine-tuned to meet this goal.
It will be used in every training example and at inference time. Do not mention training or fine-tuning.

<goal>
{goal}
</goal>"""


class Generator:
    def __init__(self, api_key: str, model: str, transport: httpx.AsyncBaseTransport | None = None,
                 concurrency: int = 8, max_retries: int = 4, provider: str = "openrouter"):
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=180, transport=transport
        )
        self._model = model
        self._provider = provider
        self._sem = asyncio.Semaphore(concurrency)
        self._max_retries = max_retries

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _complete(self, prompt: str, schema: dict) -> dict:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": schema},
        }
        if self._provider == "openai":
            # GPT-5 models are reasoning models: they take reasoning_effort, not temperature.
            body["reasoning_effort"] = "low"
            url = OPENAI_URL
        else:
            body["provider"] = ZDR_PROVIDER
            body["temperature"] = 0.7
            url = OPENROUTER_URL
        async with self._sem:
            for attempt in range(self._max_retries + 1):
                resp = await self._client.post(url, json=body)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(content)
        raise RuntimeError("unreachable")

    async def system_prompt(self, goal: str) -> str:
        out = await self._complete(_SYSTEM_PROMPT_INSTRUCTIONS.format(goal=goal), _SYSTEM_PROMPT_SCHEMA)
        return out["system_prompt"].strip()

    async def examples_for_chunk(self, chunk: Chunk, goal: str, n: int) -> list[dict]:
        prompt = _EXAMPLES_INSTRUCTIONS.format(
            goal=goal, n=n, source=chunk.source, heading=chunk.heading, text=chunk.text
        )
        out = await self._complete(prompt, _EXAMPLES_SCHEMA)
        return out.get("examples", [])


def to_row(system_prompt: str, user: str, assistant: str) -> dict:
    """One line of Fireworks chat-format JSONL."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


async def generate_rows(gen: Generator, chunks: list[Chunk], goal: str, per_chunk: int,
                        on_progress=None) -> tuple[str, list[dict], list[str]]:
    """Returns (system_prompt, rows, errors). A failing chunk is reported, not fatal."""
    system_prompt = await gen.system_prompt(goal)
    rows: list[dict] = []
    errors: list[str] = []

    async def one(chunk: Chunk):
        try:
            return chunk, await gen.examples_for_chunk(chunk, goal, per_chunk), None
        except Exception as e:  # noqa: BLE001 - surface per-chunk failures to the user
            return chunk, [], f"{chunk.source} / {chunk.heading or '(no heading)'}: {e}"

    dropped = 0
    for fut in asyncio.as_completed([one(c) for c in chunks]):
        chunk, examples, err = await fut
        if err:
            errors.append(err)
        for ex in examples:
            if leaks_source(ex["assistant"]):
                dropped += 1
                continue
            rows.append(to_row(system_prompt, ex["user"], ex["assistant"]))
        if on_progress:
            on_progress()
    if dropped:
        errors.append(f"dropped {dropped} repl{'y' if dropped == 1 else 'ies'} that mentioned the source material")
    return system_prompt, rows, errors
