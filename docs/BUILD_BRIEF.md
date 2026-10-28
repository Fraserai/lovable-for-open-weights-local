# Build brief: Weightroom goes live

Read this whole file before writing any code. It replaces earlier direction where they conflict.

## What we are building

**Weightroom is a tool any coding agent can call to train an open model and hand back weights the
business owns.** The customer brings the data. The agent (Claude Code, or any other) drives the tool.
It's cheap, it can later run inside an enterprise's own cloud, and nobody needs an ML team.

The agent is the user. Every design choice below follows from that: few choices, structured answers,
jobs that run on their own, and no way to spend money by accident.

Today's goal: **live, with a paywall, and a coding agent can one-shot a training run end to end.**

## Where things stand (read the code before changing it)

Keep and reuse:
- `worker/src/trainloop/tinker.py`: proven Tinker LoRA training. Keep its rules: refuse models with no
  verified chat format, calibrated learning rate, epochs sized to the data, truncation warning,
  checkpoint every 20 steps, training and export kept separate so export can be retried for free.
- `worker/src/trainloop/validate.py`: row checks before a paid run.
- `worker/src/trainloop/datasets.py`: maps common dataset shapes onto chat messages (JSONL + Hugging Face).
- `cli.py: plan_for()`: cost estimate from Tinker's live price list.
- `worker/scripts/banking77_score.py`: the scoring approach (exact match on held-out rows) to generalise.

Problems to fix:
- Tinker training blocks the process that starts it (11 to 55 minutes). Stopping the server kills it.
- No database, no users. State is `runs/<name>/run.json` on one machine.
- The web app runs Python by spawning `uv run trainloop`. The new service calls Python functions directly.
- Defaults point at Fireworks GLM-5.3-Flash, which has never trained live.
- No CSV support. No automatic before/after scoring on Tinker runs.

## Tinker has changed since this code was written (checked 20 Oct 2026)

Read Tinker's docs through their `llms.txt` (tinker-docs.thinkingmachines.ai) and the cookbook's own
`AGENTS.md` and skills files before touching `tinker.py`. What matters for this build:

- **SDK upgrade first.** `pyproject.toml` pins `tinker>=0.30`. SDK 0.32.0 (2 Oct) changed `ServiceClient`
  to take `base_url`, `api_key` and `default_headers_overrides` as explicit keyword arguments, and 0.27.1
  removed subprocess-isolated sampling. Upgrade `tinker` and `tinker-cookbook` to current, run the tests,
  and do one cheap live check that planning and sampling still work before building on top.
- **Resuming after a crash is supported, but not automatic.** Save full training state with `save_state`
  and resume with `create_training_client_from_state_with_optimizer(path)`. Use that for the worker
  restart path; don't assume the cookbook resumes on its own.
- **Checkpoints cost money forever unless they expire.** Storage is $0.10/GB-month and a checkpoint with
  no TTL is kept until deleted. The cookbook saves every 20 steps. Give intermediate checkpoints a short
  `ttl_seconds` (minimum 1 hour). Keep only each job's final sampler weights long-term, and charge for
  that storage or expire it on a published schedule.
- **Real cost per job is now available.** The billing API reports estimated dollar cost per session
  (SDK 0.30.2+), and sessions carry `user_metadata`. Tag every Tinker session with our `job_id` and settle
  each job's credits from Tinker's reported cost, not only our estimate. Call `ServiceClient.close()` to
  record whether a session succeeded, errored or was interrupted.
- **Inference endpoint (beta).** Base URL `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`
  with the Tinker API key; the model name is the checkpoint's sampler path (`tinker://.../sampler_weights/...`).
  `/chat/completions` applies the model's default Hugging Face chat template. **We train with the cookbook's
  renderer, which can differ from that template.** For each supported model, confirm the two produce the
  same tokens. Where they don't, our endpoint must render with our renderer and call the native
  `SamplingClient` instead of the compatible API. Tinker labels this inference beta, for testing and
  internal tools, with a waitlist for higher-throughput production serving. Ask the founder to join it.
- **Downloads.** Sampler checkpoints download as a LoRA adapter archive (`adapter_model.safetensors` +
  `adapter_config.json`), matching what `export_adapter` already builds.
- **Later, for Enterprise:** Tinker now has projects with membership-gated access, SSO, role-based access,
  an audit-log API and private VPC connectivity. Not needed today, but design jobs so each account could
  map to its own Tinker project later.
- **Full fine-tuning** is in beta. Stay on LoRA.

## Step 0: housekeeping (do first)

1. Ask the founder to move the project from `~/Desktop/CODEBASES/LOVABLEFOROPENWEIGHTS` to `~/Code/weightroom`.
   The Desktop syncs to iCloud, which makes files cloud-only and is the likely cause of the "hidden flag"
   problem noted in `worker/pyproject.toml`. Do not move it yourself without a yes.
2. `git init` if needed, commit the current state, work on a branch.
3. Move `OPENWAITS/` and `REPLIT FOR POST TRAINING/` into `archive/` and leave them alone.
4. Rewrite the top of `CLAUDE.md` so it matches this brief (mission, product sentence, scope below).
   Remove "convert content into training data" from what we do, and remove "not an eval product":
   the before/after scorecard is now core. Keep the APRA / enterprise notes and working principles.
5. Change the default base model to a Tinker model (`Qwen/Qwen3.5-4B`).

## Scope

In scope today:
- A hosted Python service: REST API + remote MCP server + background job worker, one codebase.
- Postgres, object storage for datasets and adapters, Stripe billing.
- Tinker as the only trainer.
- Data checking (not data preparation), quotes, training, before/after scoring, delivery.
- Sign-up, API keys, spend caps, credits, a landing page, an agent-readable guide.

Parked (do not delete the code, do not wire it in): the Configure / Train web screens, Fireworks, Unsloth,
Together, `trainloop prepare` (documents to training data), voice, database connectors.

The customer brings training-ready data. We check it; we do not write or clean it for them.

## Architecture

- **Service:** Python (FastAPI) in `service/`, importing `trainloop` modules directly. Exposes `/v1/...`
  REST and a remote MCP endpoint at `/mcp` (Streamable HTTP, official MCP Python SDK; verify the current
  SDK and transport against live docs before coding).
- **Worker:** a separate process from the same codebase that pulls jobs from Postgres
  (`SELECT ... FOR UPDATE SKIP LOCKED`). Restarting the API never touches a running job.
- **Storage:** S3-compatible bucket (Cloudflare R2 by default) for uploads, held-out splits, logs and
  exported adapters. Tinker log/checkpoint directories go on a persistent volume, never a temp dir.
- **Database:** Postgres. Tables at minimum: `accounts`, `api_keys`, `datasets`, `quotes`, `jobs`,
  `credit_ledger`, `deliveries`.
- **Web:** keep Next.js only for the landing page, sign-up, API key management, billing and a simple
  job list. No training UI.
- **Hosting:** Railway (or Fly). The worker needs a long-running container: the Tinker SDK pulls in torch,
  and jobs run for up to an hour, so serverless will not work.

New environment variables (add to `.env.example`, never commit real values): `DATABASE_URL`,
`S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `STRIPE_SECRET_KEY`,
`STRIPE_WEBHOOK_SECRET`, `ENCRYPTION_KEY` (for stored customer Hugging Face tokens), `PUBLIC_BASE_URL`.

## The tools (MCP and REST share them)

All responses are compact JSON with explicit units (`usd`, `credits`, `seconds`). No progress bars, no
log dumps. Every error carries `code`, a plain-English `message`, a concrete `fix`, and `retryable: true|false`.
Good: `"No column called 'label'. Found 'text' and 'intent'. Pass label_column='intent'."`

1. **`upload_data`**: a hosted server cannot see the agent's local files. Returns a short-lived signed
   upload URL and a `dataset_id`, plus a one-line `curl` command the agent can run. Accepts JSONL and CSV.
2. **`check_data`** (free, instant): takes a `dataset_id` or a Hugging Face dataset id (+ config/split).
   Returns row count, detected layout, which columns map to user/assistant, label distribution and a
   lopsidedness warning, rows that will be cut short, duplicates, and errors that block training. Reuse
   `datasets.py` and `validate.py`; add CSV. Here it also carves out the held-out set (see Scorecard) and
   stores it, so training never sees those rows.
3. **`quote`**: takes `dataset_id`, `base_model` (default `Qwen/Qwen3.5-4B`), optional `goal`. Returns
   `quote_id`, cost in credits and USD, estimated minutes, model, steps, and what scoring will cost.
   Quotes expire after 60 minutes. Spends nothing. Uses `plan_for` logic plus the Tinker price list.
4. **`train`**: requires a `quote_id`. Returns a `job_id` immediately. **Idempotent:** calling it again with
   the same `quote_id` returns the same job and never starts or charges a second run. Refuses with a clear
   error if the quote is expired, credits are short, or the quote exceeds the key's spend cap.
   Reserves credits at start; settles to actual cost when the job ends and refunds the difference.
5. **`status`**: takes `job_id`. Returns state (`queued|training|scoring|done|failed`), step x of y,
   `check_back_in_seconds`, and when done the scorecard and cost. On failure: what happened, whether
   credits were refunded, and whether retrying will help.
6. **`list_jobs`**: every job for this API key, newest first, so a fresh agent session can find work
   started by an earlier one.
7. **`deliver`**: takes `job_id` and what the customer wants: a signed download URL for the LoRA adapter
   (reuse `tinker.export_adapter`), a push to the customer's private Hugging Face repo (customer's own HF
   token, stored encrypted; reuse `handback.py`, fix its model card, which still says Fireworks), and a
   live endpoint served behind our API key and metered in credits (see the Tinker section for the
   chat-template check). Return a ready-to-run example call.

Tool descriptions must use the words people actually use: "train or fine-tune a custom model on your own
data and get weights you own". Say plainly in `train`'s description that it costs money and needs a quote.

## The scorecard (proof at the end)

- At `check_data`, hold back 10% of rows (minimum 20, maximum 300), stratified by label when there are labels.
- After training, run the base model and the trained model on the held-out rows (same system prompt,
  temperature 0). Generalise `banking77_score.py`.
- **Labelled tasks** (assistant answers come from a small fixed set, e.g. at most 100 distinct short
  answers): report exact-match accuracy for both, plus per-label accuracy for the 5 biggest labels and
  3 examples the training fixed.
- **Open-ended tasks:** do not invent a number. Return 5 side-by-side examples (prompt, base answer,
  trained answer, expected answer) and say that no automatic score applies.
- Include the scoring cost in the quote. The scorecard is part of `status` once the job is `done`.

## Jobs that survive

- Job state lives in Postgres, not in a process. The worker records each step as it goes.
- If the worker restarts mid-job, resume from the last saved training state with
  `create_training_client_from_state_with_optimizer`. If resuming isn't possible, mark the job failed and
  refund its credits automatically.
- One test must kill the worker mid-job and show the job resumes or is refunded, never silently lost.

## Billing (Lovable-style: subscription plus credits)

- 1 credit is about $0.041 of our Tinker cost. Keep this in config, not code: Tinker raised prices on
  17 July 2026 and may again. Credits charged = ceil(total Tinker cost incl. scoring / credit value), minimum 1.
  Reference points: our 41c Qwen3.5-4B Banking77 run = 10 credits; our $4.12 Kimi-K2.6 run = 100 credits.
- Plans:
  - **Free:** $0, 10 credits a month, no top-ups, "Trained with Weightroom" line on the Hugging Face model card.
  - **Pro:** $49/month, 100 credits a month, top-ups.
  - **Business:** $149/month, 300 credits a month, team members, per-key spend caps, top-ups.
  - **Enterprise:** custom (from $5,000/month); a "contact us" button for now.
- Top-ups: $20 per 100 credits (Pro and Business).
- Stripe Checkout for subscriptions and top-ups, webhooks into our own `credit_ledger`. The ledger is the
  source of truth; balances are derived from it. Every train reservation, settlement and refund is a row.
- Per-API-key spend caps (monthly and per job), set in the dashboard. `train` refuses anything over the cap.

## Safety and privacy

- Customer data goes to our storage and to Tinker only. Never log dataset contents. Never send customer
  data to OpenRouter, OpenAI or any other model API.
- Signed URLs expire. API keys are hashed at rest. Customer HF tokens are encrypted at rest.
- **Never spend real money in tests.** Mock Tinker in the test suite. Before any live paid run, show the
  founder the quote and wait for a yes.

## Landing page and agent guide

- Landing page: name, one line ("Train a model on your data. Own the weights."), the brand lines from
  `BRAND.md` used as written, the Banking77 proof (Qwen3.5-4B 67.5% to 82.5% for about 41c; Kimi-K2.6
  76% to 92% for about $4.12), pricing, and "connect your agent" instructions.
- Connect instructions: the hosted MCP URL plus an `Authorization: Bearer <api key>` header, with the exact
  command for Claude Code (verify the current `claude mcp add` syntax for remote HTTP servers).
- `/llms.txt` and one agent guide page: what each tool does, what it costs, the order to call them,
  and a worked example.
- Sign-up gives 10 free credits so an agent can complete a real small run before any card is needed.

## Build order (commit after each step, tests passing)

1. Step 0 housekeeping, then the Tinker SDK upgrade and a cheap live check.
2. Service skeleton, Postgres schema and migrations, API keys, error format.
3. `upload_data` and `check_data` (with CSV and the held-out split).
4. `quote` and the credit ledger.
5. Job worker, `train`, `status`, `list_jobs`, the restart test.
6. Scorecard.
7. `deliver` (download, HF push, endpoint).
8. MCP server over the same functions.
9. Stripe, sign-up, dashboard, spend caps.
10. Deploy, landing page, `llms.txt`.
11. Acceptance test (below).

If time runs short, stop after step 8 with a working local service rather than half-finishing billing.
Tell the founder exactly where you stopped.

## Acceptance test (the definition of done)

From a fresh Claude Code session connected only to the hosted MCP URL with a $10 spend cap, give one sentence:

> "Train a model on banking77_train.csv to classify customer messages by intent, keep it private. Budget $10."

It passes when the agent, without help:
1. uploads and checks the data,
2. shows a quote and waits for approval (the only human step),
3. trains, and survives the agent's session ending and a new one finding the job via `list_jobs`,
4. returns a scorecard showing base vs trained accuracy on held-out rows, in line with our earlier
   Banking77 result (about 67.5% to 82.5% on Qwen3.5-4B), and the actual cost,
5. delivers a private Hugging Face repo and a working endpoint, with an example call that works,
6. never double-charges when a call is retried.

Then run the same test with a CSV in the wrong layout and confirm the error tells the agent exactly how
to fix it, for free, before anything is charged.

## Working principles

- The founder's vision is fixed. Build it; don't redesign it or add features outside this scope.
- Lean. Fewer, better tools beat more tools.
- Verify current API details (Tinker, MCP SDK, Stripe, Claude Code) against live docs before coding.
- Report progress in plain language: what works, what's left, what needs the founder.
