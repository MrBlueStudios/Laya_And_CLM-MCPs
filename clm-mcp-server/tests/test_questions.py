"""Argument validation and answer decoding. No network."""

import json

import pytest

from clm_mcp_server.questions import (
    MAX_CHOICE_OPTIONS,
    MAX_RANK_CANDIDATES,
    MAX_STATE_CHARS,
    QUESTION_ID,
    DecisionInputError,
    ResponseError,
    build_rank_request,
    build_request,
    decode_answer,
    decode_ranking,
    parse_state,
)


def answers(answer):
    return {"model": "clm-latest", "answers": {QUESTION_ID: answer}, "usage": {"input_tokens": 3, "output_tokens": 0}}


# ---------------------------------------------------------------- requests

def test_choice_from_labels_and_descriptions():
    labels = build_request("t", "choice", "Which team?", ["billing", "technical"])
    assert labels.question == {"type": "choice", "instructions": "Which team?",
                               "criteria": {"billing": None, "technical": None}}
    described = build_request("t", "choice", "Which team?", {"billing": " Charges, refunds ", "other": ""})
    assert described.question["criteria"] == {"billing": "Charges, refunds", "other": None}
    assert described.labels == ("billing", "other")


@pytest.mark.parametrize(
    "options, message",
    [
        (None, "choice needs options"),
        (["only"], "at least 2 options"),
        (["a", "a"], "option labels must be unique"),
        ({" a": "x", "a": "y"}, "option labels must be unique"),
        ({"a": "same text", "b": "same text"}, "option texts"),
        ({"a": "b", "b": ""}, "option texts"),  # label b is embedded as "b", like a's description
        (["o%d" % i for i in range(MAX_CHOICE_OPTIONS + 1)], "rank_clm_candidates"),
        ("a,b", "list of strings"),
        ({"a": 3, "b": "x"}, "must be a string"),
    ],
)
def test_choice_rejects_bad_options(options, message):
    with pytest.raises(DecisionInputError, match=message):
        build_request("t", "choice", "Which?", options)


def test_score_and_default_rubric():
    rubric = build_request("t", "score", "How urgent?", ["Can wait.", "Needs action now."])
    assert rubric.question == {"type": "score", "instructions": "How urgent?",
                               "criteria": ["Can wait.", "Needs action now."]}
    default = build_request("t", "score", "How urgent?")
    assert len(default.labels) == 5 and "default rubric" in default.notes[0]
    with pytest.raises(DecisionInputError, match="rubric levels must be unique"):
        build_request("t", "score", "How urgent?", ["low", "low"])


def test_bool_is_native_noul():
    plain = build_request("t", "bool", "Is it positive?")
    assert plain.question == {"type": "noul", "instructions": "Is it positive?"}
    described = build_request("t", "bool", "Positive?", ["Yes, it praises the product.", "No, it complains."])
    assert described.question["criteria"] == {"true": "Yes, it praises the product.", "false": "No, it complains."}
    with pytest.raises(DecisionInputError, match="yes_description, no_description"):
        build_request("t", "bool", "Positive?", ["only one"])


@pytest.mark.parametrize("instructions", [None, "", "  ", 7, "x" * 4001])
def test_instructions_required_and_bounded(instructions):
    with pytest.raises(DecisionInputError, match="instructions"):
        build_request("t", "bool", instructions)


def test_state_parsing_and_limits():
    assert parse_state('{"subject": "hi"}') == {"subject": "hi"}
    assert parse_state("[not json") == "[not json"
    assert parse_state("[" * 100_000) == "[" * 100_000  # deep nesting stays text instead of crashing
    for bad in ["", "   ", {}, [], None, 3]:
        with pytest.raises(DecisionInputError):
            parse_state(bad)
    with pytest.raises(DecisionInputError, match="limit"):
        parse_state("x" * (MAX_STATE_CHARS + 1))
    with pytest.raises(DecisionInputError, match="limit"):
        parse_state({"body": "x" * MAX_STATE_CHARS})


def test_rank_request_keys_candidates_by_index():
    request = build_rank_request("Which tool?", ["Book a table.", "Find flights."], top_k=5)
    assert request.top_k == 2
    assert request.questions == {QUESTION_ID: {"type": "choice", "instructions": None,
                                               "criteria": {"0": "Book a table.", "1": "Find flights."}}}


@pytest.mark.parametrize(
    "candidates, top_k, message",
    [
        (["one"], 1, "between 2 and"),
        (["c%d" % i for i in range(MAX_RANK_CANDIDATES + 1)], 1, "between 2 and"),
        (["a", " a "], 1, "candidates must be unique"),
        (["a", ""], 1, "non-empty string"),
        (["a", "b"], 0, "top_k"),
        (["a", "b"], 51, "top_k"),
        (["a", "b"], True, "top_k"),
        ("a b", 1, "list of strings"),
    ],
)
def test_rank_rejects_bad_arguments(candidates, top_k, message):
    with pytest.raises(DecisionInputError, match=message):
        build_rank_request("state", candidates, top_k=top_k)


# ---------------------------------------------------------------- responses

def test_decode_choice_recomputes_the_label():
    request = build_request("t", "choice", "Which?", ["a", "b"])
    # A server that names a label we never sent must not get that string into the output.
    out = decode_answer(request, answers({"type": "choice", "choice": "<script>", "confidence": 0.2,
                                          "probabilities": {"a": 0.4, "b": 0.6}}))
    assert out == {"answer": "b", "probabilities": {"a": 0.4, "b": 0.6}, "confidence": 0.6}


def test_decode_score_maps_levels():
    request = build_request("t", "score", "How?", ["low", "mid", "high"])
    out = decode_answer(request, answers({"type": "score", "score": 1.5, "confidence": 0.1,
                                          "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}}))
    assert out["probabilities"] == {"low": 0.1, "mid": 0.3, "high": 0.6}
    assert out["answer"] == "high"
    assert out["expected_score"] == 1.5 and out["expected_score_normalized"] == 0.75


def test_decode_bool():
    request = build_request("t", "bool", "Yes?")
    out = decode_answer(request, answers({"type": "noul", "noul": 0.25}))
    assert out == {"answer": "false", "probabilities": {"true": 0.25, "false": 0.75},
                   "confidence": 0.75, "probability_true": 0.25}


@pytest.mark.parametrize(
    "answer, message",
    [
        ({"type": "choice", "probabilities": {"a": 0.5}}, "exactly the requested options"),
        ({"type": "choice", "probabilities": {"a": 0.5, "b": 0.5, "c": 0.0}}, "exactly the requested options"),
        ({"type": "choice", "probabilities": {"a": 0.9, "b": 0.9}}, "sum to"),
        ({"type": "choice", "probabilities": {"a": float("nan"), "b": 0.5}}, "outside"),
        ({"type": "choice", "probabilities": {"a": "0.5", "b": 0.5}}, "not a number"),
        ({"type": "choice", "probabilities": {"a": True, "b": 0.0}}, "not a number"),
        ({"type": "score", "probabilities": {"a": 0.5, "b": 0.5}}, "does not answer"),
    ],
)
def test_decode_rejects_invalid_answers(answer, message):
    request = build_request("t", "choice", "Which?", ["a", "b"])
    with pytest.raises(ResponseError, match=message):
        decode_answer(request, answers(answer))


@pytest.mark.parametrize("body", [None, [], {"answers": []}, {"answers": {}}, {"answers": {"other": {}}}])
def test_decode_rejects_malformed_bodies(body):
    with pytest.raises(ResponseError):
        decode_answer(build_request("t", "bool", "Yes?"), body)


def test_decode_ranking():
    request = build_rank_request("q", ["alpha", "beta", "gamma", "delta"], top_k=2)
    probs = {"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1}
    out = decode_ranking(request, answers({"type": "choice", "choice": "1", "probabilities": probs}))
    assert [r["index"] for r in out["ranking"]] == [1, 2]
    assert out["ranking"][0] == {"rank": 1, "index": 1, "candidate_preview": "beta", "probability": 0.6, "lift": 2.4}
    assert out["candidates_total"] == 4 and out["uniform_probability"] == 0.25
    assert out["top_margin"] == pytest.approx(0.4)


def test_ranking_previews_are_truncated():
    long = "word " * 100
    request = build_rank_request("q", [long, "short"], top_k=2)
    out = decode_ranking(request, answers({"type": "choice", "probabilities": {"0": 0.7, "1": 0.3}}))
    assert len(out["ranking"][0]["candidate_preview"]) == 200
    assert json.dumps(out)  # plain JSON-serialisable


@pytest.mark.parametrize("text", ["[NaN]", '{"x": Infinity}'])
def test_non_json_constants_stay_text(text):
    assert parse_state(text) == text
