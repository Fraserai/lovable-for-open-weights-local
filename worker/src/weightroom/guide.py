"""The agent guide: one page an agent can read once (served as /llms.txt and /docs/agents)."""

from . import config, credits


def text() -> str:
    base = config.get().public_base_url
    p = credits.PLANS
    return f"""# Weightroom

> Train a custom model on your own data and get weights you own. Weightroom is the tool a coding agent
> calls to fine-tune an open model (on Tinker), prove it beats the original on held-back data, and hand
> back the weights, a private Hugging Face repo or a live endpoint. No ML team.

The user brings training-ready data (JSONL, CSV, or a public Hugging Face dataset). You drive the tools.
Every error is JSON with `code`, `message`, a concrete `fix`, and `retryable`.

## Connect

MCP (recommended): `{base}/mcp` with header `Authorization: Bearer <Weightroom API key>`.

Claude Code:

```
claude mcp add --transport http weightroom {base}/mcp --header "Authorization: Bearer $WEIGHTROOM_API_KEY"
```

Get a key (with free credits) at {base}/signup.

## The tools, in order

1. `upload_data(filename)` - for a local .jsonl or .csv. Returns `dataset_id` and a one-line `curl` command;
   run it from the folder holding the file. Free. Skip this for a public Hugging Face dataset.
2. `check_data(dataset_id | hf_dataset, user_column?, assistant_column?, instructions?)` - free and instant.
   Detects the input and answer columns, counts rows and labels, flags lopsided labels, and lists errors with
   the fix. Holds back ~10% of rows for the scorecard. For label tasks it returns `suggested_instructions`:
   pass them back as `instructions` so the original model knows the labels and the comparison is fair.
3. `quote(dataset_id, base_model?, epochs?)` - free. The exact price in credits and minutes, and a
   `quote_id` valid for an hour. Default model: {config.get().default_base_model}.
   **Show the user the price and get a yes.**
4. `train(quote_id)` - spends the credits and returns a `job_id` immediately. Safe to retry: the same
   `quote_id` always returns the same job and never charges twice. Failed jobs are refunded in full.
5. `status(job_id)` - state, progress and `check_back_in_seconds`. When `done`, it carries the scorecard.
   `list_jobs()` finds jobs started in earlier sessions.
6. `deliver(job_id, download?, hf_repo?, endpoint?)` - the weights (LoRA adapter download), a push to the
   user's private Hugging Face repo (they save a Hugging Face write token in the dashboard first), and a live
   OpenAI-compatible endpoint. Call it again with the same arguments until the links are ready.

## The scorecard

Before training, ~10% of rows are held back. After training, the original and the trained model answer
them. Short answers (labels, decisions) are scored by exact match: "accuracy went from 67.5% (original) to
82.5% (your model) on 200 held-back rows". Written answers get side-by-side examples, not a made-up number.

## Data formats

- JSONL with a `messages` list per row (system/user/assistant), or question/answer style fields
  (instruction+output, prompt+response, question+answer, text+label, text+intent, ...).
- CSV with a header row. Name the columns with `user_column` and `assistant_column` if not detected.
- Public Hugging Face datasets by id (`hf_dataset='owner/name'`, optional `hf_config`, `hf_split`, `limit`).

## The live endpoint

```
POST {base}/v1/chat/completions
Authorization: Bearer <API key>
{{"model": "<job_id>", "messages": [{{"role": "user", "content": "..."}}]}}
```

OpenAI-compatible (no streaming yet). The system prompt the model was trained with is added automatically.
Metered in credits at Tinker's sampling rates.

## Credits and plans

Credits buy training and endpoint use: a 4B model on ~1,000 examples is about 10 credits; a 1T-parameter model
about 100. `quote` gives the exact number for your data before anything is spent. Plans: Free (10 credits/month), Pro (${p['pro']['usd_per_month']}/month, {p['pro']['monthly_credits']} credits),
Business (${p['business']['usd_per_month']}/month, {p['business']['monthly_credits']} credits, per-key spend caps),
Enterprise (custom). Top-ups: {credits.TOPUP_CREDITS} credits for ${credits.TOPUP_USD}. Each API key can have a monthly and a
per-job spend cap; `train` refuses anything over them.

## REST equivalents

All under `{base}/v1`, with the same bearer key and the same JSON:
`POST /uploads` {{filename}}, `PUT /uploads/<id>?token=...` (the file), `POST /datasets/check`,
`POST /quotes`, `POST /jobs` {{quote_id}}, `GET /jobs`, `GET /jobs/<id>`, `POST /jobs/<id>/deliver`,
`GET /models`, `POST /chat/completions`.
"""
