"""Hand the trained model back to the customer via a private Hugging Face repo."""

from pathlib import Path

from huggingface_hub import HfApi

MODEL_CARD = """---
base_model: {hf_base}
library_name: peft
tags: [lora]
---

# {repo_id}

LoRA adapter fine-tuned from `{base_model}` on Fireworks (job `{job}`).

**Goal:** {goal}

This is an adapter, not full weights. Load it on top of the exact base checkpoint it was
trained against. Fireworks notes that its training base may differ from the public
Hugging Face weights; confirm before serving it elsewhere.

The system prompt used in every training example (use the same one at inference):

```
{system_prompt}
```
"""


def push_to_hub(folder: Path, repo_id: str, token: str, *, base_model: str, hf_base: str, job: str,
                goal: str, system_prompt: str) -> str:
    api = HfApi(token=token)
    api.create_repo(repo_id, private=True, exist_ok=True)
    readme = folder / "README.md"
    if not readme.exists():
        readme.write_text(MODEL_CARD.format(repo_id=repo_id, base_model=base_model, hf_base=hf_base, job=job,
                                            goal=goal, system_prompt=system_prompt))
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, commit_message=f"Adapter from {job}")
    return f"https://huggingface.co/{repo_id}"
