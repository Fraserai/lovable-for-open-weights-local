# worker

The `trainloop` command: documents or a dataset + a goal → training data → fine-tune on Fireworks,
Tinker or Unsloth (on Hugging Face GPUs) → model back (download or private Hugging Face repo). The web app calls these same commands.

## Setup

```bash
cp ../.env.example ../.env   # fill in the keys you need
uv sync                      # includes the Tinker SDK (large: torch). Skip it: uv sync --no-group tinker
```

Keys by step: `OPENROUTER_API_KEY` (or `OPENAI_API_KEY` with `GENERATOR_PROVIDER=openai`, test docs only)
to write examples; `TINKER_API_KEY` or `FIREWORKS_API_KEY` + `FIREWORKS_ACCOUNT_ID` to train, or `HF_TOKEN` (write, with
prepaid Jobs credit on the account) for Unsloth; `HF_TOKEN` (write) to publish.

## Use

```bash
# 1a. From documents: parse (free with --preview), then write examples
uv run trainloop prepare ./docs --goal "Answer member questions about fees" --run acme-faq --preview
uv run trainloop prepare ./docs --goal "..." --run acme-faq --base-model Qwen/Qwen3.5-4B --max-chunks 20

# 1b. Or from a ready-made dataset (no example writing, no cost)
uv run trainloop dataset --run ato --hf simplelex/ATO-Australian-Tax-Rulings-and-Guidance \
    --config ato_interpretative_decisions --limit 300 --base-model Qwen/Qwen3.5-4B --goal "..."
uv run trainloop dataset --run mine --file data.jsonl --goal "..."

# 2. Optionally change the model or settings (free; 0 = sized for the data), see the plan and cost
#    (free), then train (paid; asks first). The web page's Train screen calls these same commands.
uv run trainloop configure --run ato --base-model unsloth:Qwen/Qwen3.5-4B --epochs 2 --lora-rank 16
uv run trainloop configure --run ato --json '{"max_steps": 60, "lora_alpha": 64, "lr_scheduler": "cosine"}'  # any of settings.py
uv run trainloop plan --run ato
uv run trainloop train --run ato

# 3. Progress (Fireworks jobs run remotely; Tinker runs in the foreground of `train`)
uv run trainloop status --run ato

# 4. Take it home
uv run trainloop fetch --run ato --hf-repo you/ato-tax-model
```

Try a finished Tinker model against the original: `uv run trainloop chat --run ato --message "..."`.

Other commands: `trainloop validate file.jsonl` checks any dataset; `trainloop catalog` lists what each
trainer can fine-tune (live).

Everything for a run lives in `../runs/<name>/`, with `run.json` recording progress so steps can be
re-run or resumed.

## Which trainer

The base model decides: `accounts/fireworks/models/...` trains on Fireworks, a Hugging Face id
(`Qwen/Qwen3.5-4B`) trains on Tinker, and `unsloth:<Hugging Face id>` trains with Unsloth on a GPU
rented from Hugging Face Jobs. Unsloth takes almost any open model on the Hub: the plan picks the
cheapest single GPU that fits it in 16-bit, sets a time limit that caps the bill, and refuses models
that need more than one GPU (not wired yet). The job runs in the cloud, so it survives this process
stopping (`trainloop status` reattaches), and it pushes the adapter to a private repo on your account.
Pass `--held-out file.jsonl` to `train` to score the original and trained model on examples kept out
of training. Uses Unsloth's Apache-2.0 library only; Unsloth Studio is AGPL and is not used. Settings are chosen for you: LoRA rank 32, epochs sized to the
dataset, and on Tinker its calibrated learning rate and the model's own chat format (unverified
formats are refused before anything is charged).

## Tests

```bash
uv run pytest
```
