# Project: Lovable for Open Models

## Mission
**We're the shovel that lets anyone, and any business, build and own their own intelligence.**
Not one use case: whatever people build, the job is that anyone can make a model and walk away owning it.
We focus on building the best tool in this space, opening it up to users, builders and businesses.

Brand lines (see [BRAND.md](BRAND.md), use as written): **From Models to Moats. · Build Intelligence. Own It. ·
Train Models. Build Moats. · Your Models. Your Intelligence.**

## What we're building (direction set 20 Oct 2026; see docs/BUILD_BRIEF.md)
**Weightroom is a tool any coding agent can call to train an open model and hand back weights the
business owns.** The customer brings the data. The agent (Claude Code, or any other) drives the tool.
It's cheap, it can later run inside an enterprise's own cloud, and nobody needs an ML team.

The agent is the user: few choices, compact JSON answers, jobs that run on their own, a quote before
any spend, retries that never double-charge, and proof at the end (base vs trained on held-back data).

## Scope (do not expand without the founder's say-so)
**Customer brings:** training-ready data (JSONL, CSV or a Hugging Face dataset) and what the model
should do. They own what goes in, its quality, and their rights to use it. We check data; we do not
write or clean it for them.

**We do the rest:**
1. Take the upload and check it (free): layout, columns, rows, balance, errors with the exact fix
2. Quote the exact cost before anything is spent
3. Pick sensible training settings and train on Tinker (LoRA)
4. Score the trained model against the original on held-back rows (the Scorecard)
5. Hand back the model: download, private Hugging Face repo, or a live endpoint

The hosted service lives in `worker/src/weightroom/` (REST API + remote MCP server + job worker),
built on the `trainloop` modules. Billing: subscription plus credits (Free / Pro / Business / Enterprise).

**Parked (keep the code, don't wire it in):** the Train web screens in `web/`, Fireworks, Unsloth,
Together, `trainloop prepare` (documents to training data), voice, database connectors.

## Proof so far
Banking77 (77 intents, 200 held-out messages): Qwen3.5-4B 67.5% to 82.5% for ~41c on Tinker;
Kimi-K2.6 76.0% to 92.0% for ~$4.12.

## Core APIs
| Role | Service | Notes (verified Sept 2026) |
|---|---|---|
| Platform intelligence / agent | OpenRouter | GPT-6 Astra, Claude Fable 5.1, open models. Fable 5.1 on OpenRouter does NOT allow zero data retention: never send raw customer data to it. Pin ZDR providers for sensitive data. |
| Models, weights, storage | Hugging Face | Model catalog, private repos for finished weights, Storage Buckets (S3-compatible, mutable) for uploads and checkpoints, Inference Endpoints. |
| Trainer 1 | Fireworks | SFT, DPO, RFT, LoRA. GLM-5.3-Flash is fine-tunable here. Build SDK: `Dataset.from_file()` + `create_supervised_fine_tuning_job()`. REST job API exposes epochs, LR, LoRA rank (max 32), eval carve-out. No region field on training jobs and early stop is deprecated (checked live docs 2026-09-22); the Australian region AP_NEWSOUTHWALES_1 (quota via sales) applies to deployments, so confirm where training runs. Multi-LoRA: up to 100 adapters per deployment. Base models show a Tunable flag. |
| Trainer 2 | Tinker (Thinking Machines) | LoRA, full control of the training loop: SFT, RL (GRPO/PPO), DPO, distillation. Models include GLM-5.3 (256K), Kimi K2.6, Qwen 3.5/3.6/3.8, Nemotron 3, gpt-oss, DeepSeek V3.1, Inkling. Not GLM-5.3-Flash (yet). Machine-readable model list: https://tinker-docs.thinkingmachines.ai/tinker/models.json . Checkpoints downloadable, can publish to HF Hub. Retires models regularly. |
| Trainer 3 | Together AI | LoRA AND full fine-tuning, up to 1T params. SFT + DPO. LoRA on MoE expert layers. Bring-your-own HF model. Early stopping with refunds. Dedicated endpoint hosting is costly (~$6.49/hr H100), so make download / switch-off easy. Check if GLM-5.3-Flash is supported. |
| Fast classifier | Jev (TypeSafe) | Choice / true-false / score only, no text generation. 64K context. ~$0.042/M input, output unmetered. Use for tagging uploads, routing chat intents. |

## Supporting stack
- **Chat UI:** Vercel AI SDK + AI Elements, or assistant-ui
- **Voice:** ElevenLabs Scribe or Deepgram (API); Whisper self-hosted for sensitive customers
- **File parsing:** Docling (structure, tables; MIT) and MarkItDown (fast; MIT)
- **DB connectors:** dlt (open-source Python, reads any SQL database)
- **Content -> training data:** build our own converter. Kiln's Python library (MIT) can help.
  Easy Dataset is a good blueprint but is AGPL-3.0: do not embed it in the hosted service.
- **Long-running jobs:** Trigger.dev (open source, TypeScript, self-hostable); Inngest as lighter alternative
- **Basics:** auth, Postgres, Stripe billing

## Trainer routing
Build a catalog that merges the Hugging Face model list with each trainer's
support list (Fireworks Tunable flag, Tinker models.json, Together supported models).
User sees any model; platform labels what's trainable where and routes the job.

## Australian enterprise notes
- APRA's 30 April 2026 AI letter: AI managed under CPS 230 / CPS 234; expects model
  inventories, third-party model assurance, mapped foundation-model dependencies and
  credible exit/substitution. Open weights the customer owns = strong answer.
- We'd likely be a material service provider under CPS 230: map every data hop.
- Design a "sovereign mode" showing what stays onshore (Fireworks NSW region for
  serving; confirm training locations for each trainer).
- GLM is from Zhipu (China). Self/Fireworks-hosted MIT weights send no data to Zhipu,
  but have that answer ready for boards.

## Kiln: reuse, don't copy the product
Kiln (github.com/Kiln-AI/Kiln) does many things for developers: evals, RAG, agents,
fine-tuning. We are a different product: hosted, chat-first, outcome-focused,
open-weights-first. Do not change the plan because of it.

We CAN reuse pieces of it:
- **OK to use:** the `kiln-ai` Python library and REST server (MIT licensed).
  Useful for synthetic data generation (content -> training examples), dataset
  handling/export formats, and its existing fine-tune integrations with Fireworks
  and Together.
- **Do not copy:** the Kiln desktop app code. It is source-available under a
  fair-code model, not MIT.
- Check the current LICENSE files before vendoring anything.

## Working principles
- Founder's vision is fixed. Don't redesign the product; build it.
- Lean and focused. Resist adding features outside the scope above.
- Verify current API details against live docs before coding. These change often.
