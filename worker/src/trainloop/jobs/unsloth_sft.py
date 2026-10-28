# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["unsloth==2026.9.10"]
# ///
"""Runs on a Hugging Face Jobs GPU: LoRA fine-tune with Unsloth, push the adapter to a private repo.

Launched by trainloop (worker/src/trainloop/unsloth.py). Standalone on purpose: it runs in a fresh
container and must not import trainloop. It reports back through `TRAINLOOP {json}` lines in its
log, which the worker follows.

Same rules as the Tinker trainer:
- Train on the last assistant reply only, rendered with the model's own chat template, and taught
  against exactly the prompt the model sees at inference. A model whose template can't be checked
  that way is refused before training starts (setup time only, no training).
- Push the adapter before anything optional (the held-out check), so a late failure can't lose it.
"""

import argparse
import json
import os
import re
import sys

MIN_USABLE_SHARE = 0.9  # below this the chat template is not behaving as checked; refuse
EVAL_BATCH = 16
_THINK = re.compile(r"<think>.*?</think>", re.S)


def emit(event: str, **data) -> None:
    print("TRAINLOOP " + json.dumps({"event": event, **data}), flush=True)


def fail(message: str) -> None:
    emit("error", message=message)
    sys.exit(1)


def read_jsonl(path: str) -> list[list[dict]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["messages"] for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-repo", required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=32768)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lora-alpha", type=int, help="Default: the rank")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--rslora", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA: 4-bit base weights")
    p.add_argument("--optim", default="adamw_8bit")
    p.add_argument("--lr-scheduler", default="linear")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, help="Default: 5%% of steps, 1-10")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--gradient-checkpointing", default="unsloth", choices=["unsloth", "true", "false"])
    p.add_argument("--eval", help="Optional held-out JSONL: messages ending in the expected assistant reply")
    a = p.parse_args()
    token = os.environ["HF_TOKEN"]

    import unsloth  # noqa: F401 - must be imported before transformers to patch it
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments
    from unsloth import FastLanguageModel

    emit("loading", model=a.base_model, gpu=torch.cuda.get_device_name(0))
    model, processor = FastLanguageModel.from_pretrained(
        model_name=a.base_model, max_seq_length=a.max_seq_length, load_in_4bit=a.load_in_4bit,
        load_in_16bit=not a.load_in_4bit,
        full_finetuning=False, token=token,
    )
    # Multimodal checkpoints (e.g. Qwen3.5) come with a processor; text training uses its tokenizer.
    tok = getattr(processor, "tokenizer", processor)
    templater = tok if getattr(tok, "chat_template", None) else processor

    def render(messages: list[dict], prompt: bool) -> str:
        # enable_thinking=False: train and ask for a direct answer (templates without it ignore it)
        return templater.apply_chat_template(messages, tokenize=False, add_generation_prompt=prompt,
                                             enable_thinking=False)

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    rows, unusable, too_long, longest = [], 0, 0, 0
    for messages in read_jsonl(a.data):
        if not messages or messages[-1].get("role") != "assistant":
            unusable += 1
            continue
        prompt_ids = encode(render(messages[:-1], prompt=True))
        full_ids = encode(render(messages, prompt=False))
        # The reply must be taught against exactly the prompt the model gets at inference
        if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
            unusable += 1
            continue
        if len(full_ids) > a.max_seq_length:  # cutting it would cut the answer
            too_long += 1
            continue
        longest = max(longest, len(full_ids))
        rows.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
                     "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]})
    total = len(rows) + unusable + too_long
    emit("data", rows=len(rows), unusable=unusable, too_long=too_long, longest=longest)
    if not rows or len(rows) / total < MIN_USABLE_SHARE:
        fail(f"Only {len(rows)} of {total} examples fit {a.base_model}'s chat format as checked "
             f"({unusable} didn't match, {too_long} too long). Nothing was trained.")

    checkpointing = {"unsloth": "unsloth", "true": True, "false": False}[a.gradient_checkpointing]
    model = FastLanguageModel.get_peft_model(
        model, r=a.lora_rank, lora_alpha=a.lora_alpha or a.lora_rank, lora_dropout=a.lora_dropout, bias="none",
        target_modules=a.target_modules.split(","), use_rslora=a.rslora,
        use_gradient_checkpointing=checkpointing, random_state=a.seed, max_seq_length=a.max_seq_length,
    )

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                emit("step", step=state.global_step, total=state.max_steps, loss=logs["loss"],
                     learning_rate=logs.get("learning_rate"), epoch=logs.get("epoch"))

    steps = -(-len(rows) // (a.batch_size * a.grad_accum)) * a.epochs
    if a.max_steps > 0:
        steps = min(steps, a.max_steps)
    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        callbacks=[Progress()],
        args=TrainingArguments(
            output_dir="/tmp/out", per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
            num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.learning_rate,
            lr_scheduler_type=a.lr_scheduler,
            warmup_steps=a.warmup_steps if a.warmup_steps is not None else max(1, min(10, steps // 20)),
            optim=a.optim, weight_decay=a.weight_decay,
            bf16=True, logging_steps=1, save_strategy="no", report_to="none", seed=a.seed,
            remove_unused_columns=False,
        ),
    )
    result = trainer.train()
    emit("trained", steps=result.global_step, loss=result.training_loss)

    api = HfApi(token=token)
    api.create_repo(a.output_repo, private=True, exist_ok=True)
    model.push_to_hub(a.output_repo, token=token, private=True)
    tok.push_to_hub(a.output_repo, token=token, private=True)
    api.upload_file(path_or_fileobj=json.dumps(trainer.state.log_history, indent=1).encode(),
                    path_in_repo="training_log.json", repo_id=a.output_repo,
                    commit_message="Training metrics")
    emit("pushed", repo=a.output_repo)

    if a.eval:
        held_out = read_jsonl(a.eval)
        FastLanguageModel.for_inference(model)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        prompts = [render(m[:-1], prompt=True) for m in held_out]
        expected = [m[-1]["content"] for m in held_out]

        def answers() -> list[str]:
            out = []
            for i in range(0, len(prompts), EVAL_BATCH):
                batch = tok(prompts[i:i + EVAL_BATCH], return_tensors="pt", padding=True,
                            add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**batch, max_new_tokens=64, do_sample=False)
                for seq in gen[:, batch["input_ids"].shape[1]:]:
                    out.append(_THINK.sub("", tok.decode(seq, skip_special_tokens=True)).strip())
            return out

        def score(got: list[str]) -> float:
            norm = lambda s: s.strip().strip("`'\".").split("\n")[0].strip().lower()  # noqa: E731
            return sum(norm(g) == norm(e) for g, e in zip(got, expected)) / len(expected)

        trained = answers()
        with model.disable_adapter():
            original = answers()
        report = {"examples": len(expected), "original": {"accuracy": score(original), "answers": original},
                  "trained": {"accuracy": score(trained), "answers": trained}}
        api.upload_file(path_or_fileobj=json.dumps(report, indent=1).encode(), path_in_repo="held_out_check.json",
                        repo_id=a.output_repo, commit_message="Held-out check")
        emit("eval", examples=len(expected), original=report["original"]["accuracy"],
             trained=report["trained"]["accuracy"])

    emit("done", repo=a.output_repo)


if __name__ == "__main__":
    main()
