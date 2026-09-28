"""Translate `evaluate_laya_decision` arguments into one Laya typed question, and its answer back.

Pure Python (no torch), so every validation rule is testable without loading a checkpoint.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Union

QuestionType = Literal["choice", "score", "bool"]
ModelName = Literal["auto", "english", "multilingual", "typed-decisions"]
State = Union[str, dict, list]

QUESTION_TYPES = ("choice", "score", "bool")
MODEL_NAMES = ("auto", "english", "multilingual", "typed-decisions")

# Key the single question is sent under; Laya answers a dict of named questions.
QUESTION_ID = "decision"

DEFAULT_SCORE_LEVELS = ("very low", "low", "moderate", "high", "very high")

# `bool` is Laya's `noul` type. Its stock option labels render as "false: ..." / "true: ...", and on
# the English checkpoint that label pair can outweigh the state and pin a clearly positive input at
# P(true) = 0 (upstream issue #156). Neutral labels with yes/no descriptions avoid that: on our spot
# check of 8 unambiguous yes/no questions they scored 8/8, the stock labels 7/8.
BOOL_LABELS = {"true": "A", "false": "B"}
DEFAULT_BOOL_CRITERIA = {"true": "yes", "false": "no"}

# Options share a fixed head budget (192 tokens on the English checkpoint), so accuracy falls off
# past ~20 options (Banking77: 0.425 at 77 labels).
MANY_OPTIONS_WARNING_THRESHOLD = 20

# Bounds on one call. Laya reads 512-1,024 tokens, so longer input only costs tokenising time;
# the caps stop a runaway call from tying up the model.
MAX_STATE_CHARS = 100_000
MAX_INSTRUCTIONS_CHARS = 4_000
MAX_OPTIONS = 64
MAX_SCORE_LEVELS = 20
MAX_OPTION_CHARS = 1_000


class DecisionInputError(ValueError):
    """An argument the caller can fix. The message is safe to return to the model."""


@dataclass(frozen=True)
class DecisionRequest:
    """One validated decision, ready for `Router.predict`."""

    state: State
    question_type: QuestionType
    question: dict[str, Any]
    # Keys of the returned `probabilities`, in option order.
    labels: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())

    @property
    def questions(self) -> dict[str, dict[str, Any]]:
        return {QUESTION_ID: self.question}


def _reject_constant(name: str) -> Any:
    raise ValueError("%s is not valid JSON" % name)


def parse_state(state: Any) -> State:
    """Accept text, a JSON object/array, or a string holding one.

    A string that parses as a JSON object or array is passed on parsed: Laya serializes structured
    states itself and reads an array as a chronological conversation (truncating the oldest turns
    first), which it cannot do for the same JSON left as flat text.
    """
    if isinstance(state, str):
        text = state.strip()
        if not text:
            raise DecisionInputError("state must not be empty")
        if len(state) > MAX_STATE_CHARS:
            raise DecisionInputError("state is %d characters; the limit is %d" % (len(state), MAX_STATE_CHARS))
        if text[0] in "{[":
            try:
                parsed = json.loads(text, parse_constant=_reject_constant)
            except (ValueError, RecursionError):
                return state
            if isinstance(parsed, (dict, list)) and parsed:
                return parsed
        return state
    if isinstance(state, (dict, list)):
        if not state:
            raise DecisionInputError("state must not be an empty object or array")
        try:
            size = len(json.dumps(state, ensure_ascii=False))
        except (TypeError, ValueError, RecursionError) as exc:
            raise DecisionInputError("state is not serialisable as JSON: %s" % exc) from exc
        if size > MAX_STATE_CHARS:
            raise DecisionInputError("state is %d characters as JSON; the limit is %d" % (size, MAX_STATE_CHARS))
        return state
    raise DecisionInputError(
        "state must be text or a JSON object/array, got %s" % type(state).__name__
    )


def _clean_labels(options: list[Any], what: str) -> list[str]:
    labels = []
    for i, option in enumerate(options):
        if not isinstance(option, str) or not option.strip():
            raise DecisionInputError("%s[%d] must be a non-empty string" % (what, i))
        if len(option.strip()) > MAX_OPTION_CHARS:
            raise DecisionInputError("%s[%d] is longer than %d characters" % (what, i, MAX_OPTION_CHARS))
        labels.append(option.strip())
    duplicates = sorted(label for label, n in Counter(labels).items() if n > 1)
    if duplicates:
        raise DecisionInputError("%s must be unique; repeated: %s" % (what, ", ".join(duplicates)))
    return labels


def _choice_criteria(options: Any) -> dict[str, str | None]:
    if isinstance(options, (dict, list)) and len(options) > MAX_OPTIONS:
        # Checked before cleaning, so an oversized list costs nothing to reject.
        raise DecisionInputError(
            "choice takes at most %d options (Laya is accurate up to ~20); narrow the list first" % MAX_OPTIONS
        )
    if options is None:
        raise DecisionInputError(
            "choice needs options: a list of labels, or an object mapping each label to a description"
        )
    if isinstance(options, dict):
        labels = _clean_labels(list(options.keys()), "options keys")
        criteria: dict[str, str | None] = {}
        for label, description in zip(labels, options.values()):
            if description is not None and not isinstance(description, str):
                raise DecisionInputError("options[%r] description must be a string" % label)
            if description and len(description.strip()) > MAX_OPTION_CHARS:
                raise DecisionInputError("options[%r] description is longer than %d characters" % (label, MAX_OPTION_CHARS))
            criteria[label] = description.strip() if description and description.strip() else None
    elif isinstance(options, list):
        criteria = {label: None for label in _clean_labels(options, "options")}
    else:
        raise DecisionInputError("options must be a list of strings or an object of label -> description")
    if len(criteria) < 2:
        raise DecisionInputError("choice needs at least 2 options to form a distribution")
    if len(criteria) > MAX_OPTIONS:
        raise DecisionInputError(
            "choice takes at most %d options (Laya is accurate up to ~20); narrow the list first" % MAX_OPTIONS
        )
    return criteria


def build_request(
    state: Any,
    question_type: Any,
    instructions: Any,
    options: Any = None,
) -> DecisionRequest:
    """Validate tool arguments and build the Laya question they describe."""
    parsed_state = parse_state(state)
    if not isinstance(instructions, str) or not instructions.strip():
        raise DecisionInputError("instructions must be a non-empty string: the question to answer")
    instructions = instructions.strip()
    if len(instructions) > MAX_INSTRUCTIONS_CHARS:
        raise DecisionInputError(
            "instructions are %d characters; the limit is %d" % (len(instructions), MAX_INSTRUCTIONS_CHARS)
        )
    notes: list[str] = []

    if question_type == "choice":
        criteria = _choice_criteria(options)
        labels = tuple(criteria)
        question = {"type": "choice", "instructions": instructions, "criteria": criteria}
        if len(labels) > MANY_OPTIONS_WARNING_THRESHOLD:
            notes.append(
                "%d options share a fixed token budget; Laya accuracy drops past ~%d options. "
                "Consider a coarse-to-fine two-step choice."
                % (len(labels), MANY_OPTIONS_WARNING_THRESHOLD)
            )
    elif question_type == "score":
        if options is None:
            levels = list(DEFAULT_SCORE_LEVELS)
            notes.append("no options given; used the default rubric %s" % " < ".join(levels))
        elif isinstance(options, list):
            if len(options) > MAX_SCORE_LEVELS:
                raise DecisionInputError("score needs between 2 and %d rubric levels" % MAX_SCORE_LEVELS)
            levels = _clean_labels(options, "options")
        else:
            raise DecisionInputError("score options must be a list of rubric levels, lowest first")
        if not 2 <= len(levels) <= MAX_SCORE_LEVELS:
            raise DecisionInputError("score needs between 2 and %d rubric levels" % MAX_SCORE_LEVELS)
        labels = tuple(levels)
        question = {"type": "score", "instructions": instructions, "criteria": levels}
    elif question_type == "bool":
        criteria_bool = dict(DEFAULT_BOOL_CRITERIA)
        if options is not None:
            if not isinstance(options, list) or len(options) != 2:
                raise DecisionInputError(
                    "bool options are optional; when given they must be exactly "
                    "[yes_description, no_description]"
                )
            yes, no = _clean_labels(options, "options")
            criteria_bool = {"true": yes, "false": no}
        labels = ("true", "false")
        question = {
            "type": "noul",
            "instructions": instructions,
            "criteria": criteria_bool,
            "labels": dict(BOOL_LABELS),
        }
    else:
        raise DecisionInputError(
            "question_type must be one of %s, got %r" % (", ".join(QUESTION_TYPES), question_type)
        )

    return DecisionRequest(
        state=parsed_state,
        question_type=question_type,
        question=question,
        labels=labels,
        notes=tuple(notes),
    )


def validate_model(model: Any) -> ModelName:
    if model is None:
        return "auto"
    if model not in MODEL_NAMES:
        raise DecisionInputError("model must be one of %s, got %r" % (", ".join(MODEL_NAMES), model))
    return model


def decode_answer(request: DecisionRequest, answer: dict[str, Any]) -> dict[str, Any]:
    """Turn Laya's typed answer into one shape for every question type.

    `probabilities` is always the full distribution over `request.labels` and `answer` is always its
    arg-max key, so a caller can read any decision the same way.
    """
    if request.question_type == "choice":
        probabilities = {label: float(answer["probabilities"][label]) for label in request.labels}
        extra: dict[str, Any] = {}
    elif request.question_type == "score":
        raw = answer["probabilities"]
        probabilities = {label: float(raw[str(i)]) for i, label in enumerate(request.labels)}
        expected = float(answer["score"])
        top = len(request.labels) - 1
        extra = {
            "expected_score": round(expected, 4),
            "expected_score_normalized": round(expected / top, 4),
        }
    else:
        p_true = float(answer["noul"])
        probabilities = {"true": p_true, "false": round(1.0 - p_true, 4)}
        extra = {"probability_true": p_true}

    if request.question_type == "choice":
        # Laya takes the arg-max before rounding; re-deriving it from rounded values could split a tie.
        best = answer["choice"]
    else:
        best = max(probabilities, key=probabilities.__getitem__)
    return {
        "answer": best,
        "probabilities": probabilities,
        "confidence": float(answer["answer_confidence"]),
        **extra,
    }
