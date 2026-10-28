import json

import httpx
import pytest

from trainloop import score as sc


def conv(q, a, system="Classify."):
    return [{"role": "system", "content": system}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]


TESTS = [conv("lost my card", "lost_card"), conv("where is my money", "pending_transfer"), conv("hi", "greeting")]


def test_labels_are_marked_by_exact_match_and_prose_by_a_judge():
    assert sc.metric_for(["lost_card", "greeting"]) == "exact"
    assert sc.metric_for(["You can withdraw your super early only on severe financial hardship grounds."] * 2) == "judge"


def test_scorecard_marks_each_contender():
    card = sc.score(TESTS, {
        "original": lambda prompts: ["Lost_Card.", "no idea", "hello"],
        "trained": lambda prompts: ["lost_card", "pending_transfer", "greeting"],
    })
    assert card["metric"] == "exact" and card["examples"] == 3
    assert card["contenders"]["original"]["correct"] == 1
    assert card["contenders"]["trained"]["score"] == 1.0


def test_contenders_only_see_the_conversation_before_the_reference():
    seen = []
    sc.score(TESTS, {"trained": lambda prompts: seen.extend(prompts) or ["x"] * len(prompts)})
    assert all(p[-1]["role"] == "user" for p in seen)


def test_unchanged_scores_are_reused_not_paid_for_twice():
    first = sc.score(TESTS, {"original": lambda p: ["lost_card", "x", "x"]})
    card = sc.score(TESTS, {"original": lambda p: pytest.fail("should reuse")}, previous=first, reuse=("original",))
    assert card["contenders"]["original"]["reused"] and card["contenders"]["original"]["correct"] == 1


def test_written_answers_go_to_the_judge():
    tests = [conv("Can I get my super early?", "Only on severe financial hardship or compassionate grounds, via the ATO.")] * 2
    calls = []
    judge = lambda qs, refs, answers: calls.append(answers) or [True, False]  # noqa: E731
    card = sc.score(tests, {"trained": lambda p: ["yes if hardship", "anytime"]}, judge=judge)
    assert card["metric"] == "judge" and card["contenders"]["trained"]["correct"] == 1 and calls


def _router(handler):
    return httpx.MockTransport(handler)


def test_frontier_calls_pin_zero_retention_when_the_data_is_private():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hm</think> lost_card"}}]})

    answers = sc.openrouter_answerer("openai/gpt-6-astra", "k", zdr=True, max_tokens=64, transport=_router(handler))(
        [TESTS[0][:-1]])
    assert answers == ["lost_card"]
    assert bodies[0]["provider"] == sc.ZDR and bodies[0]["temperature"] == 0


def test_no_zero_retention_host_is_a_clear_error_not_a_fallback():
    handler = lambda r: httpx.Response(404, json={"error": {"message": "No endpoints found matching your data policy"}})  # noqa: E731
    with pytest.raises(sc.ScoreError, match="no zero-data-retention host"):
        sc.openrouter_answerer("anthropic/claude-fable-5.1", "k", zdr=True, max_tokens=64, transport=_router(handler))(
            [TESTS[0][:-1]])


def test_out_of_credit_says_so():
    handler = lambda r: httpx.Response(402, json={"error": {"message": "Insufficient credits"}})  # noqa: E731
    with pytest.raises(sc.ScoreError, match="out of credit"):
        sc.openrouter_answerer("m", "k", zdr=False, max_tokens=64, transport=_router(handler))([TESTS[0][:-1]])


def test_judge_reads_yes_and_no_and_always_pins_zero_retention():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        verdict = "YES" if "hardship" in body["messages"][0]["content"].split("Answer to mark:")[1] else "NO"
        return httpx.Response(200, json={"choices": [{"message": {"content": verdict}}]})

    mark = sc.judge_marker("z-ai/glm-5.3", "k", transport=_router(handler))
    assert mark(["q", "q", "q"], ["ref", "ref", "ref"], ["on hardship grounds", "never", ""]) == [True, False, False]
    assert all(b["provider"] == sc.ZDR for b in bodies)
