"""Argument validation and answer decoding. No checkpoint needed."""

import pytest

from laya_mcp_server.questions import (
    BOOL_LABELS,
    DEFAULT_SCORE_LEVELS,
    QUESTION_ID,
    DecisionInputError,
    build_request,
    decode_answer,
    parse_state,
    validate_model,
)


def test_choice_from_label_list():
    request = build_request("text", "choice", "Which team?", ["billing", "technical"])
    assert request.question == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": None, "technical": None},
    }
    assert request.labels == ("billing", "technical")
    assert request.questions == {QUESTION_ID: request.question}


def test_choice_from_label_descriptions():
    request = build_request("text", "choice", "Which team?", {"billing": "refunds ", "other": ""})
    assert request.question["criteria"] == {"billing": "refunds", "other": None}


@pytest.mark.parametrize(
    "options, message",
    [
        (None, "choice needs options"),
        (["only"], "at least 2 options"),
        (["a", "a"], "must be unique"),
        (["a", " "], "non-empty string"),
        ("a,b", "list of strings"),
    ],
)
def test_choice_rejects_bad_options(options, message):
    with pytest.raises(DecisionInputError, match=message):
        build_request("text", "choice", "Which?", options)


def test_many_choice_options_warn():
    request = build_request("text", "choice", "Which?", ["o%d" % i for i in range(25)])
    assert any("25 options" in note for note in request.notes)


def test_score_uses_rubric_in_order():
    request = build_request("text", "score", "How urgent?", ["low", "high"])
    assert request.question == {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]}
    assert request.labels == ("low", "high")


def test_score_defaults_rubric_and_says_so():
    request = build_request("text", "score", "How urgent?")
    assert request.labels == DEFAULT_SCORE_LEVELS
    assert request.notes and "default rubric" in request.notes[0]


def test_score_rejects_label_descriptions():
    with pytest.raises(DecisionInputError, match="rubric levels"):
        build_request("text", "score", "How urgent?", {"low": "x", "high": "y"})


def test_bool_maps_to_noul_with_neutral_labels():
    request = build_request("text", "bool", "Is it positive?")
    assert request.question == {
        "type": "noul",
        "instructions": "Is it positive?",
        "criteria": {"true": "yes", "false": "no"},
        "labels": BOOL_LABELS,
    }
    assert request.labels == ("true", "false")


def test_bool_takes_yes_no_descriptions():
    request = build_request("text", "bool", "Positive?", ["yes, it is positive", "no, it is negative"])
    assert request.question["criteria"] == {"true": "yes, it is positive", "false": "no, it is negative"}


def test_bool_rejects_wrong_option_count():
    with pytest.raises(DecisionInputError, match="yes_description, no_description"):
        build_request("text", "bool", "Positive?", ["yes"])


@pytest.mark.parametrize("instructions", ["", "   ", None, 3])
def test_instructions_required(instructions):
    with pytest.raises(DecisionInputError, match="instructions"):
        build_request("text", "bool", instructions)


def test_unknown_question_type():
    with pytest.raises(DecisionInputError, match="question_type"):
        build_request("text", "maybe", "Q?")


def test_state_json_string_is_parsed():
    assert parse_state('{"subject": "hi", "body": "there"}') == {"subject": "hi", "body": "there"}
    assert parse_state('[{"role": "user", "content": "hi"}]') == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("text", ["{not json", "[1, 2", "{}", "plain text", " 42 "])
def test_state_other_text_is_kept(text):
    assert parse_state(text) == text


@pytest.mark.parametrize("state", ["", "  ", {}, [], None, 7])
def test_state_rejects_empty_or_scalar(state):
    with pytest.raises(DecisionInputError, match="state"):
        parse_state(state)


def test_validate_model():
    assert validate_model(None) == "auto"
    assert validate_model("multilingual") == "multilingual"
    with pytest.raises(DecisionInputError, match="model"):
        validate_model("gpt")


def _answer(**fields):
    return {"answer_confidence": 0.7, **fields}


def test_decode_choice():
    request = build_request("t", "choice", "Which?", ["a", "b"])
    out = decode_answer(request, _answer(choice="b", probabilities={"a": 0.3, "b": 0.7}))
    assert out == {"answer": "b", "probabilities": {"a": 0.3, "b": 0.7}, "confidence": 0.7}


def test_decode_score_keys_by_level_text():
    request = build_request("t", "score", "How?", ["low", "mid", "high"])
    raw = _answer(score=1.5, probabilities={"0": 0.1, "1": 0.3, "2": 0.6})
    out = decode_answer(request, raw)
    assert out["probabilities"] == {"low": 0.1, "mid": 0.3, "high": 0.6}
    assert out["answer"] == "high"
    assert out["expected_score"] == 1.5
    assert out["expected_score_normalized"] == 0.75


def test_decode_bool():
    request = build_request("t", "bool", "Yes?")
    out = decode_answer(request, _answer(noul=0.25))
    assert out["probabilities"] == {"true": 0.25, "false": 0.75}
    assert out["answer"] == "false"
    assert out["probability_true"] == 0.25


def test_input_limits():
    from laya_mcp_server.questions import MAX_OPTIONS, MAX_STATE_CHARS

    with pytest.raises(DecisionInputError, match="limit"):
        parse_state("x" * (MAX_STATE_CHARS + 1))
    with pytest.raises(DecisionInputError, match="limit"):
        parse_state({"body": "x" * MAX_STATE_CHARS})
    # Deeply nested "JSON" is kept as text instead of crashing the parser.
    assert parse_state("[" * 50_000) == "[" * 50_000
    with pytest.raises(DecisionInputError, match="instructions are"):
        build_request("t", "bool", "q" * 4001)
    with pytest.raises(DecisionInputError, match="at most %d options" % MAX_OPTIONS):
        build_request("t", "choice", "Which?", ["o%d" % i for i in range(MAX_OPTIONS + 1)])
    with pytest.raises(DecisionInputError, match="longer than"):
        build_request("t", "choice", "Which?", ["a", "b" * 1001])
    with pytest.raises(DecisionInputError, match="description is longer"):
        build_request("t", "choice", "Which?", {"a": "x", "b": "y" * 1001})
    with pytest.raises(DecisionInputError, match="between 2 and 20"):
        build_request("t", "score", "How?", ["level %d" % i for i in range(21)])


def test_revision_is_validated_without_loading_anything():
    from laya_mcp_server.engine import PINNED_REVISION, LayaEngine

    assert LayaEngine().revision == PINNED_REVISION
    assert LayaEngine(revision="main").revision == "main"
    for bad in ["../../etc", "", "a b", "x" * 101]:
        with pytest.raises(ValueError, match="LAYA_MCP_REVISION"):
            LayaEngine(revision=bad)


def test_oversized_option_lists_are_rejected_before_any_work():
    import time

    from laya_mcp_server.questions import MAX_OPTIONS

    started = time.perf_counter()
    for options in (["o%d" % i for i in range(50_000)], {"o%d" % i: "" for i in range(50_000)}):
        with pytest.raises(DecisionInputError, match="at most %d options" % MAX_OPTIONS):
            build_request("t", "choice", "Which?", options)
    with pytest.raises(DecisionInputError, match="between 2 and 20"):
        build_request("t", "score", "How?", ["l%d" % i for i in range(50_000)])
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize("text", ["[NaN]", '{"x": Infinity}', "[-Infinity]"])
def test_non_json_constants_stay_text(text):
    assert parse_state(text) == text
