"""Score the original model and (if trained) the fine-tuned one on held-out Banking77 messages.

Exact label match on messages neither model trained on. Internal proof tooling.
Usage: python scripts/banking77_score.py <run-name>
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "worker" / "src"))


def normalise(s: str) -> str:
    return s.strip().strip("`'\".").split("\n")[0].strip().lower()


def main(run: str) -> None:
    import tinker
    from dotenv import load_dotenv
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from trainloop.playground import _text
    from trainloop.tinker import ensure_ssl_certificates, resolve_renderer

    load_dotenv(ROOT / ".env")
    ensure_ssl_certificates()
    state = json.loads((ROOT / "runs" / run / "run.json").read_text())
    test = json.loads((ROOT / "data" / "banking77" / "test.json").read_text())
    base = state["base_model"]
    tok = get_tokenizer(base)
    fmt = resolve_renderer(base)
    trained_r = renderers.get_renderer(fmt, tok, model_name=base)
    try:  # ask the original for a direct answer, as the playground does
        original_r = renderers.get_renderer(f"{fmt}_disable_thinking", tok, model_name=base)
    except Exception:  # noqa: BLE001
        original_r = trained_r

    svc = tinker.ServiceClient()
    models = {"original": (svc.create_sampling_client(base_model=base), original_r)}
    if state.get("sampler_path"):
        models["trained"] = (svc.create_sampling_client(model_path=state["sampler_path"]), trained_r)

    out = ROOT / "runs" / run / "banking77_scores.json"
    # The original model's score doesn't change after training; reuse it rather than pay twice.
    results = json.loads(out.read_text()) if out.exists() else {}
    for name, (client, r) in models.items():
        if name in results:
            print(f"{name:9} accuracy {results[name]['accuracy']:.1%}   (reused from the earlier run)")
            continue
        params = tinker.types.SamplingParams(max_tokens=24, temperature=0.0, stop=r.get_stop_sequences())
        futures = [client.sample(prompt=r.build_generation_prompt([
            {"role": "system", "content": test["system"]}, {"role": "user", "content": row["text"]}]),
            num_samples=1, sampling_params=params) for row in test["rows"]]
        answers = [normalise(_text(r, f.result().sequences[0].tokens)) for f in futures]
        hits = [a == row["label_text"].lower() for a, row in zip(answers, test["rows"])]
        valid = sum(a in {l.lower() for l in test["labels"]} for a in answers)
        results[name] = {"accuracy": sum(hits) / len(hits), "valid_label": valid / len(hits), "answers": answers}
        print(f"{name:9} accuracy {sum(hits)}/{len(hits)} = {sum(hits)/len(hits):.1%}   "
              f"(answered with a real label {valid}/{len(hits)})")

    out.write_text(json.dumps(results, indent=2))
    if "trained" in results:
        fixed = [(row["text"], results["original"]["answers"][i], row["label_text"])
                 for i, row in enumerate(test["rows"])
                 if results["trained"]["answers"][i] == row["label_text"].lower()
                 and results["original"]["answers"][i] != row["label_text"].lower()]
        print(f"\nfixed by training: {len(fixed)} messages, e.g.")
        for text, was, right in fixed[:4]:
            print(f"  '{text}'\n     original said: {was!r}   correct: {right!r}")


if __name__ == "__main__":
    main(sys.argv[1])
