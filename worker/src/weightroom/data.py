"""Getting the customer's data in and checking it, free, before anything is spent.

We check data; we don't write or clean it. When something's wrong the answer says exactly what to
change ("No column called 'label'. Found 'text' and 'intent'. Pass assistant_column='intent'.").

check_data also holds back ~10% of rows (trainloop.holdout) so the Scorecard tests the trained model
on examples it never saw.
"""

import collections
import csv
import io
import json
import statistics
import tempfile
from datetime import timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from trainloop import datasets as tl_datasets
from trainloop import holdout, score
from trainloop.validate import validate_jsonl

from . import config, db, security
from .errors import WRError, not_found
from .storage import store

FORMATS = {".jsonl": "jsonl", ".csv": "csv"}
UPLOAD_TTL = 3600
MAX_HF_ROWS = 20_000
MAX_LABELS_LISTED = 100


def _raw_key(ds: db.Dataset) -> str:
    return f"datasets/{ds.id}/raw.{ds.file_format}"


def train_key(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/train.jsonl"


def test_key(dataset_id: str) -> str:
    return f"datasets/{dataset_id}/test.jsonl"


def create_upload(s: Session, account: db.Account, key: db.ApiKey, filename: str) -> dict:
    name = Path(filename or "").name
    fmt = FORMATS.get(Path(name).suffix.lower())
    if not fmt:
        raise WRError("unsupported_format", f"{name or 'That file'} isn't a format Weightroom reads.",
                      "Upload a .jsonl file (one JSON object per line) or a .csv file with a header row. "
                      "Or skip the upload and pass a Hugging Face dataset id to check_data.")
    ds = db.Dataset(account_id=account.id, api_key_id=key.id, filename=name, file_format=fmt,
                    upload_expires_at=db.now() + timedelta(seconds=UPLOAD_TTL))
    s.add(ds)
    s.flush()
    url = f"{config.get().public_base_url}/v1/uploads/{ds.id}?token={security.make_token('upload', ds.id, UPLOAD_TTL)}"
    return {
        "dataset_id": ds.id,
        "upload_url": url,
        "method": "PUT",
        "expires_in_seconds": UPLOAD_TTL,
        "max_size_mb": config.get().max_upload_mb,
        "command": f'curl -sS --fail -T "{name}" "{url}"',
        "next": f"Run the command from the folder holding {name}, then call check_data with dataset_id='{ds.id}'.",
    }


def save_upload(s: Session, dataset_id: str, token: str, tmp: Path, size: int) -> dict:
    if security.read_token("upload", token) != dataset_id:
        raise WRError("upload_link_invalid", "This upload link is invalid or has expired.",
                      "Call upload_data again for a fresh link.", status=403)
    ds = s.get(db.Dataset, dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    if size == 0:
        raise WRError("empty_upload", "The upload was empty.", "Check the file path in the curl command and run it again.")
    store().put(_raw_key(ds), tmp)
    ds.size_bytes = size
    ds.state = "uploaded"  # re-uploading means checking (and quoting) again
    ds.report = {**(ds.report or {}), "checked_at": db.now().isoformat()}
    return {"dataset_id": ds.id, "bytes": size,
            "next": f"Call check_data with dataset_id='{ds.id}'."}


def _read_rows(path: Path, fmt: str) -> tuple[list[dict], list[str], list[str]]:
    """(raw rows, column names seen, problems reading the file)."""
    rows: list[dict] = []
    problems: list[str] = []
    if fmt == "csv":
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        columns = [c for c in (reader.fieldnames or []) if c is not None]
        for row in reader:
            rows.append({k: v for k, v in row.items() if k is not None})
        if not columns:
            problems.append("The CSV has no header row.")
        return rows, columns, problems
    columns_seen: dict[str, None] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                if len(problems) < 5:
                    problems.append(f"line {line_no} isn't valid JSON ({e.msg}).")
                continue
            if not isinstance(row, dict):
                if len(problems) < 5:
                    problems.append(f"line {line_no} isn't a JSON object.")
                continue
            rows.append(row)
            if len(columns_seen) < 200:
                columns_seen.update(dict.fromkeys(row))
    return rows, list(columns_seen), problems


def _suggest_columns(rows: list[dict], columns: list[str]) -> tuple[str, str] | None:
    """Best guess at (user column, assistant column): long free text in, short repeated values out."""
    sample = rows[:500]
    stats = {}
    for c in columns:
        values = [str(r.get(c) or "").strip() for r in sample]
        values = [v for v in values if v]
        if not values:
            continue
        stats[c] = (statistics.mean(len(v) for v in values), len(set(values)) / len(values))
    if len(stats) < 2:
        return None
    user = max(stats, key=lambda c: stats[c][0])
    rest = [c for c in stats if c != user]
    assistant = min(rest, key=lambda c: (stats[c][1], stats[c][0]))
    return user, assistant


def _columns_error(columns: list[str], wanted: tuple[str, str] | None, rows: list[dict]) -> WRError:
    found = ", ".join(f"'{c}'" for c in columns[:20]) or "none"
    if wanted:
        lower = {c.lower() for c in columns}
        missing = [w for w in wanted if w.lower() not in lower]
        if missing:
            guess = _suggest_columns(rows, columns)
            fix = (f"Pass user_column='{guess[0]}' and assistant_column='{guess[1]}'." if guess
                   else "Pass user_column and assistant_column naming columns that exist.")
            return WRError("column_not_found", f"No column called '{missing[0]}'. Found {found}.", fix,
                           details={"columns": columns[:50]})
        return WRError("no_usable_rows",
                       f"Columns '{wanted[0]}' and '{wanted[1]}' exist but no row has text in both.",
                       "Check those columns hold text, or name different ones.", details={"columns": columns[:50]})
    guess = _suggest_columns(rows, columns)
    fix = (f"Name the columns: pass user_column='{guess[0]}' (what goes in) and assistant_column='{guess[1]}' "
           f"(what the model should answer)." if guess else
           "Use rows with a 'messages' list, or name the input and answer columns with user_column and assistant_column.")
    return WRError("columns_not_recognised", f"Couldn't tell which columns are the input and the answer. Found {found}.",
                   fix, details={"columns": columns[:50]})


def _task_summary(references: list[str]) -> dict:
    if not references:
        return {"task": "unknown"}
    metric = score.metric_for(references)
    if metric != "exact":
        return {"task": "written_answers", "metric": "side_by_side"}
    counts = collections.Counter(score.normalise(r) for r in references)
    out = {"task": "labels", "metric": "exact_match", "distinct_labels": len(counts),
           "biggest_labels": [{"label": k, "rows": v} for k, v in counts.most_common(5)]}
    biggest, smallest = counts.most_common()[0][1], counts.most_common()[-1][1]
    out["smallest_label_rows"] = smallest
    out["lopsided"] = len(counts) > 1 and (biggest / len(references) > 0.5 or biggest / max(1, smallest) > 10)
    return out


def check(s: Session, account: db.Account, key: db.ApiKey, *, dataset_id: str | None = None,
          hf_dataset: str | None = None, hf_config: str | None = None, hf_split: str | None = None,
          limit: int = 5000, user_column: str | None = None, assistant_column: str | None = None,
          instructions: str | None = None) -> dict:
    if bool(dataset_id) == bool(hf_dataset):
        raise WRError("bad_request", "Give exactly one of dataset_id (an upload) or hf_dataset (a Hugging Face id).",
                      "For a local file call upload_data first. For a public dataset pass hf_dataset='owner/name'.")
    if bool(user_column) != bool(assistant_column):
        raise WRError("bad_request", "user_column and assistant_column go together.",
                      "Pass both, or neither to let Weightroom detect them.")
    fields = (user_column, assistant_column) if user_column else None
    instructions = (instructions or "").strip() or None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        out = tmp / "train.jsonl"
        columns: list[str] = []
        problems: list[str] = []
        raw_count = 0
        if dataset_id:
            ds = s.get(db.Dataset, dataset_id)
            if ds is None or ds.account_id != account.id:
                raise not_found("dataset", dataset_id)
            if ds.source == "upload" and ds.state == "awaiting_upload":
                raise WRError("not_uploaded", "Nothing has been uploaded to this dataset yet.",
                              "Run the curl command from upload_data, then call check_data again.")
            if ds.source == "upload":
                raw = store().get(_raw_key(ds), tmp / f"raw.{ds.file_format}")
                rows, columns, problems = _read_rows(raw, ds.file_format)
                raw_count = len(rows)
                messages = [m for m in (tl_datasets.to_messages(r, fields) for r in rows) if m]
                if not messages:
                    if problems and not rows:
                        raise WRError("unreadable_file", f"Couldn't read {ds.filename}: {problems[0]}",
                                      "Fix the file and upload it again with a new upload_data link.")
                    raise _columns_error(columns, fields, rows)
                count = tl_datasets.write_rows(iter(messages), out, system_prompt=instructions)
            else:
                count = _from_hub(ds.hf_dataset, (ds.report or {}).get("hf", {}), limit, fields, out, instructions)
                raw_count = count
        else:
            if "/" not in hf_dataset:
                raise WRError("bad_request", f"'{hf_dataset}' isn't a Hugging Face dataset id.",
                              "Use the owner/name form, e.g. hf_dataset='PolyAI/banking77'.")
            ds = db.Dataset(account_id=account.id, api_key_id=key.id, source="huggingface", hf_dataset=hf_dataset,
                            state="uploaded", report={"hf": {"config": hf_config, "split": hf_split}})
            s.add(ds)
            s.flush()
            count = _from_hub(hf_dataset, {"config": hf_config, "split": hf_split}, limit, fields, out, instructions)
            raw_count = count

        test = tmp / "test.jsonl"
        kept, held = holdout.split(out, test)
        report = validate_jsonl(out)
        references = [m[-1]["content"] for m in holdout.load(test)] + \
                     [json.loads(line)["messages"][-1]["content"] for line in out.read_text().splitlines()[:5000] if line.strip()]
        task = _task_summary(references)

        warnings = [w for _, w in report.warnings]
        dupes = sum(1 for w in warnings if "duplicate" in w)
        warnings = [w for w in warnings if "duplicate" not in w]
        notes: list[str] = []
        if raw_count and count < raw_count:
            notes.append(f"{raw_count - count:,} of {raw_count:,} rows had no usable input/answer pair and were skipped.")
        notes.extend(problems[:3])
        if dupes:
            notes.append(f"{dupes:,} rows repeat an earlier row exactly.")
        if task.get("lopsided"):
            notes.append(f"Labels are lopsided (smallest label has {task['smallest_label_rows']} rows). "
                         "The model may learn to over-predict the big labels; consider balancing.")
        suggested = None
        has_system = (out.read_text(encoding="utf-8").lstrip()[:200].find('"role": "system"') >= 0)
        if task.get("task") == "labels" and not instructions and not has_system:
            labels = sorted({score.normalise(r) for r in references})
            if len(labels) <= MAX_LABELS_LISTED:
                suggested = "Reply with exactly one of these labels and nothing else: " + ", ".join(labels)
            notes.append("No instructions: the original model won't know which labels exist, so the before/after "
                         "comparison is unfair to it. Pass instructions (see suggested_instructions) and check again.")
        if held == 0:
            notes.append(f"Fewer than {holdout.MIN_EXAMPLES} rows, so none are held back: there'll be no Scorecard.")
        notes.extend(sorted(set(warnings))[:3])

        errors = [{"line": line, "problem": msg} for line, msg in report.errors[:10]]
        ok = report.ok
        if ok:
            store().put(train_key(ds.id), out)
            if held:
                store().put(test_key(ds.id), test)
        ds.state = "checked" if ok else "uploaded"
        ds.train_rows, ds.test_rows, ds.est_tokens = kept, held, report.est_tokens
        ds.instructions = instructions
        ds.report = {**(ds.report or {}), "task": task, "columns": columns[:50], "ok": ok,
                     "checked_at": db.now().isoformat()}

    result = {
        "dataset_id": ds.id,
        "ready": ok,
        "rows": count,
        "train_rows": kept,
        "held_out_rows": held,
        "est_training_tokens_per_epoch": report.est_tokens,
        **({"columns": columns[:50]} if columns else {}),
        **task,
        "scorecard": (f"exact-match accuracy on {held} held-back rows" if held and task.get("metric") == "exact_match"
                      else f"side-by-side answers on {held} held-back rows" if held else "none (too few rows)"),
        "notes": notes[:8],
    }
    if suggested:
        result["suggested_instructions"] = suggested
    if errors:
        result["errors"] = errors
        result["fix"] = ("Fix the rows listed in errors (line numbers are in the converted data, in file order), "
                         "upload again, and check again.")
    result["next"] = (f"Call quote with dataset_id='{ds.id}' (default model {config.get().default_base_model})."
                      if ok else "Fix the errors, then check again.")
    return result


def _from_hub(hf_id: str, opts: dict, limit: int, fields, out: Path, instructions: str | None) -> int:
    limit = max(10, min(int(limit or 5000), MAX_HF_ROWS))
    try:
        rows = tl_datasets.from_hub(hf_id, split=opts.get("split"), config=opts.get("config"), limit=limit,
                                    fields=fields)
        return tl_datasets.write_rows(rows, out, system_prompt=instructions)
    except tl_datasets.DatasetError as e:
        raise WRError("dataset_unreadable", str(e),
                      "Check the id, config and split on huggingface.co, or name the columns with user_column and "
                      "assistant_column. Private datasets aren't supported yet: download it and use upload_data.") from e
