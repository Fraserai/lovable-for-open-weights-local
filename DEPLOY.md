# Deploying Weightroom

One Docker image, two Railway services (the website/API and the job worker), a Postgres database and
an S3-compatible bucket. About 30 minutes the first time. Nothing here costs money until a real training
run, and every run shows a quote first.

## What you need

| Account | For | Notes |
|---|---|---|
| Railway | Hosting both services and Postgres | Any plan with persistent volumes |
| Cloudflare R2 (or any S3) | Datasets and trained adapters | Create a bucket, e.g. `weightroom`, and an API token with read/write on it |
| Tinker | Training | Your existing key, with credit loaded |
| Stripe | Plans and top-ups | Start in test mode |
| Hugging Face | Nothing server-side | Customers add their own token in the dashboard |

## 1. Stripe (test mode first)

1. Products → create three prices:
   - **Weightroom Pro**: $49, recurring monthly → copy its price id (`price_...`) into `STRIPE_PRICE_PRO`
   - **Weightroom Business**: $149, recurring monthly → `STRIPE_PRICE_BUSINESS`
   - **40 credits**: $20, one-off → `STRIPE_PRICE_TOPUP`
2. Developers → Webhooks → add an endpoint at `https://<your domain>/stripe/webhook` with these events:
   `checkout.session.completed`, `invoice.paid`, `customer.subscription.updated`,
   `customer.subscription.deleted`. Copy its signing secret into `STRIPE_WEBHOOK_SECRET`.
3. Settings → Billing → Customer portal: turn it on (the dashboard's "Manage billing" button uses it).

## 2. Railway

1. New project → **Deploy from GitHub repo** → `Fraserai/lovable-for-open-weights`. Railway finds the `Dockerfile`.
2. Add **Postgres** to the project. `DATABASE_URL` is shared to services as a reference variable.
3. Service **web** (the one created from the repo):
   - Settings → Networking → generate a domain (or add your own). Put it in `PUBLIC_BASE_URL` as `https://...`.
   - Settings → Healthcheck path: `/healthz`
   - Variables: everything in the table below.
4. Service **worker**: add another service from the same repo.
   - Settings → Deploy → **Custom start command**: `python -m weightroom.worker`
   - Add a **volume** mounted at `/data`. Training checkpoints and logs live here, so a restarted worker
     resumes a run instead of starting over.
   - Variables: the same as web (Railway can share them).
   - No domain needed.

Keep the worker at one replica to start. More replicas are safe (each job goes to one worker) but each
needs its own volume, and a job resumes fastest on the worker that started it.

### Variables

| Variable | Value |
|---|---|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` |
| `PUBLIC_BASE_URL` | `https://<your domain>` (no trailing slash) |
| `SECRET_KEY`, `ENCRYPTION_KEY` | From `python -m weightroom.admin new-secrets` (run it locally once). **Never change them later**: logins, links and saved tokens depend on them. |
| `TRAINER` | `tinker` |
| `TINKER_API_KEY` | Your Tinker key |
| `STORAGE` | `s3` (required: web and worker run on different machines) |
| `S3_ENDPOINT` | `https://<account id>.r2.cloudflarestorage.com` |
| `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | From R2 |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` | From Stripe |
| `STRIPE_PRICE_PRO`, `STRIPE_PRICE_BUSINESS`, `STRIPE_PRICE_TOPUP` | The three price ids |
| `ENTERPRISE_CONTACT` | Optional, e.g. `mailto:you@yourdomain.com` |
| `CREDIT_VALUE_USD` | Optional. Default `0.041` (1 credit ≈ 4.1c of Tinker cost). Raise it if Tinker raises prices. |

**If a service won't start**, its log says exactly which setting is missing. On a public `https://` URL
Weightroom refuses to run without `SECRET_KEY`, `ENCRYPTION_KEY`, `TRAINER=tinker`, `STORAGE=s3` and a
Postgres `DATABASE_URL`: any of those wrong would leak links, lose files, or charge customers for fake runs.
If agents connect through a second domain, add it to `EXTRA_HOSTS` (comma-separated).

**If the image build fails at the torch step**, the PyTorch CPU index may be unreachable from the builder.
Delete the `&& uv pip install ... torch ...` line and both `--no-install-package torch` flags in the
`Dockerfile`. The image gets much bigger, but it works.

## 3. First live check (about 10 credits, ~15 minutes)

1. Open your domain → **Get an API key**. You start with 10 free credits. To test past that without
   Stripe: `railway run --service web python -m weightroom.admin grant you@example.com 100`.
2. Connect Claude Code with the command shown after sign-up.
3. Ask it, in one go:

   > Use Weightroom to train a model on the Hugging Face dataset mteb/banking77 (columns text and
   > label_text, 1,000 rows) that classifies bank customer messages by intent. Keep it private.
   > Show me the quote before training.

4. It should check the data, pass the suggested instructions, show you a quote and wait, then train.
   When it's done the scorecard should be in the same range as our earlier run (Qwen3.5-4B: about 67.5%
   to 82.5%). Then ask it to deliver: a download link and a live endpoint.
5. Check the worker's logs on Railway and `python -m weightroom.admin jobs` if anything looks off.

## Notes

- The free tier refills to 10 credits every 30 days; paid plans add credits on each Stripe invoice.
- New accounts are limited to three per network per day, so free credits can't be farmed.
- `python -m weightroom.admin retention` answers the churn question: of models trained 30+ days ago, how
  many are still called through their endpoint, and how many of their accounts still pay.
- The quote is the price. A job's credits are reserved at start and refunded in full if it fails.
  Tinker's billing usage is tagged per job (`weightroom:<job id>`) if you want to compare our quotes
  with real cost.
- Tinker keeps each job's final checkpoint (it powers the live endpoint). Intermediate checkpoints
  expire after a day.
- Tinker's OpenAI-compatible endpoint is beta. Weightroom serves models through Tinker's native sampling
  with the exact chat format each model was trained with, so answers match what training taught.
