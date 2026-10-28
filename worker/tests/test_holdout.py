import json

from trainloop import holdout


def _write(path, n, dup_of=None):
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(n)]
    if dup_of is not None:  # the same question again, with a different answer
        rows.append({"messages": [{"role": "user", "content": f"q{dup_of}"}, {"role": "assistant", "content": "other"}]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _questions(path):
    return {json.loads(line)["messages"][0]["content"] for line in path.read_text().splitlines()}


def test_holds_back_ten_percent_and_never_shares_a_question(tmp_path):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    _write(train, 300)
    kept, held = holdout.split(train, test)
    assert (kept, held) == (270, 30)
    assert not _questions(train) & _questions(test)


def test_a_duplicate_of_a_held_back_question_leaves_training(tmp_path):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    _write(train, 100)
    holdout.split(train, test)
    held_q = sorted(_questions(test))[0]
    # Rebuild with a duplicate of one held-back question; it must not end up in training
    _write(train, 100, dup_of=int(held_q[1:]))
    kept, held = holdout.split(train, test)
    assert held_q not in _questions(train)
    assert kept == 100 + 1 - held - 1


def test_small_datasets_keep_everything_for_training(tmp_path):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    test.write_text("stale\n")
    _write(train, 20)
    assert holdout.split(train, test) == (20, 0)
    assert not test.exists()


def test_the_split_is_the_same_every_time(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        _write(d / "train.jsonl", 200)
        holdout.split(d / "train.jsonl", d / "test.jsonl")
    assert (a / "test.jsonl").read_text() == (b / "test.jsonl").read_text()
