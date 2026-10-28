# Weightroom

**We're the shovel that lets anyone, and any business, build and own their own intelligence.**
From Models to Moats. See [BRAND.md](BRAND.md). Product brief: [CLAUDE.md](CLAUDE.md).

Pick any open model, bring a dataset or your documents, and train it on a cloud trainer (Tinker,
Fireworks, or a GPU on your Hugging Face account). You see what training costs first, and you keep the weights.

## The hosted service (what customers and agents use)

`worker/src/weightroom/` is the product: a REST API and a remote MCP server any coding agent connects to,
plus a job worker. An agent checks the data (free), gets a quote, trains on Tinker, shows the scorecard
(original vs trained on held-back rows) and delivers the weights, a private Hugging Face repo or a live
endpoint. Agent guide: `/llms.txt` on the running server. Deploying: [DEPLOY.md](DEPLOY.md).

Run it locally (no keys needed: SQLite, local files and a fake trainer that spends nothing):

```bash
cd worker && uv sync --no-group documents
PYTHONPATH=src uv run uvicorn weightroom.api:app --reload   # http://localhost:8000
PYTHONPATH=src uv run python -m weightroom.worker          # in a second terminal
uv run pytest                                              # includes a real MCP client one-shotting a run
```

Keep the project out of iCloud-synced folders (Desktop, Documents). With "Optimise Mac Storage" on,
macOS offloads `node_modules` and `.venv` files to the cloud and everything stalls while they download.

## The CLI and the parked web app

```bash
cp .env.example .env            # fill in the keys you have (never commit .env; it's git-ignored)
cd worker && uv sync            # includes Tinker's SDK and Docling, both large
cd ../web && pnpm install && pnpm dev   # the earlier Train screen, http://localhost:3000 (parked)
```

## How it fits together

```
web/     Next.js app: the one-page Train screen (Configure · Current Run · History)
worker/  Python CLI `trainloop`: imports or writes training data, plans cost, trains, hands the model back
runs/    one folder per run (uploads, training data, logs). Not in git.
data/    cached model catalog. Not in git.
```

The page calls the worker's commands for each step, so everything it does can also be run from the
terminal (see [worker/README.md](worker/README.md)).

| Step | Worker command | Where it runs |
|---|---|---|
| Use a ready-made dataset | `trainloop dataset` | Your `.jsonl`, or any Hugging Face dataset (free) |
| Or turn documents into examples | `trainloop prepare` | Docling + OpenRouter (zero data retention) |
| Choose model, trainer and settings | `trainloop configure` | Free |
| See what training will cost | `trainloop plan` | Free |
| Train | `trainloop train` | Tinker (Hugging Face ids), Fireworks (`accounts/...`), or a Hugging Face GPU running Unsloth (`unsloth:<id>`) |
| Take the model home | `trainloop fetch --hf-repo you/name` | Download + private Hugging Face repo |
| Prove it: the Scorecard | `trainloop score` | Original vs trained vs a frontier model on the ~10% of examples held back from training |

Which trainer honours which setting lives in `worker/src/trainloop/settings.py` (mirrored in
`web/src/lib/settings.ts`). Controls a trainer can't honour are greyed out on the page.

## Status (2026-09-24)

- **Proof it improves models:** on Banking77 (77-way intent classification, 200 held-out messages),
  Qwen3.5-4B went from 67.5% to 82.5% after 1,000 examples (~41¢, ~11 min on Tinker), and
  **Kimi-K2.6 (1T parameters) went from 76.0% to 92.0%** after 1,540 examples (~$4.12, ~55 min).
  Reproduce: `worker/scripts/banking77_data.py`, `trainloop dataset --no-hold-out` + `train`, then
  `worker/scripts/banking77_score.py <run>`. The Kimi run's result is also on its Scorecard in the app.
- **Trainers:** Tinker is proven end to end. Fireworks and Hugging Face GPU jobs are wired and their
  plans verified; neither has trained live yet. **All three accounts are currently out of credit.**
- **Hosted service (2026-10-20):** accounts, API keys with spend caps, quotes, a job worker that resumes
  after crashes, the scorecard, delivery, Stripe plans and top-ups, and the MCP server, all tested end to
  end with a fake trainer. Not yet run against live Tinker or Stripe: that's the first step in DEPLOY.md.
