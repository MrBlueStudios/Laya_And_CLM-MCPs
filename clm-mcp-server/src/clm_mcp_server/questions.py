"""Translate tool arguments into CLM System One questions, and CLM answers back.

Pure Python, no network: every validation rule is testable on its own.

CLM scores a state against each candidate text with a dual encoder. The text it embeds for an
option is the option's description when one is given, otherwise the label, so two options with
the same text would be indistinguishable; that is rejected here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Union

QuestionType = Literal["choice", "score", "bool"]
State = Union[str, dict, list]

QUESTION_TYPES = ("choice", "score", "bool")
QUESTION_ID = "decision"
DEFAULT_SCORE_LEVELS = ("very low", "low", "moderate", "high", "very high")

# Bounds on one call. CLM truncates every text at its encoder's token limit (2,048 by default),
# so longer input only costs bandwidth; the caps stop a runaway call from shipping megabytes.
MAX_STATE_CHARS = 100_000
MAX_INSTRUCTIONS_CHARS = 4_000
MAX_OPTION_CHARS = 2_000
MAX_CHOICE_OPTIONS = 100
MAX_SCORE_LEVELS = 20
MAX_RANK_CANDIDATES = 1_000
MAX_CANDIDATE_CHARS = 4_000
MAX_RANK_TOTAL_CHARS = 1_000_000
MAX_TOP_K = 50
PREVIEW_CHARS = 200
# A distribution that sums further than this from 1 did not come from a softmax.
SUM_TOLERANCE = 1e-3


class DecisionInputError(ValueError):
    """An argument the caller can fix. The message is safe to return to the model."""


class ResponseError(ValueError):
    """The CLM server answered with something that is not a valid answer to the request."""


@dataclass(frozen=True)
class DecisionRequest:
    """One validated decision, ready for `POST /v1/systemone`."""

    state: State
    question_type: QuestionType
    question: dict[str, Any]
    # Keys of the returned `probabilities`, in option order.
    labels: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())

    @property
    def questions(self) -> dict[str, dict[str, Any]]:
        return {QUESTION_ID: self.question}


@dataclass(frozen=True)
class RankRequest:
    """A ranking of candidate texts, sent as one `choice` keyed by candidate index."""

    state: State
    instructions: str | None
    candidates: tuple[str, ...]
    top_k: int

    @property
    def questions(self) -> dict[str, dict[str, Any]]:
        criteria = {str(i): text for i, text in enumerate(self.candidates)}
        return {QUESTION_ID: {"type": "choice", "instructions": self.instructions, "criteria": criteria}}


def _reject_constant(name: str) -> Any:
    raise ValueError("%s is not valid JSON" % name)


def parse_state(state: Any) -> State:
    """Accept text, a JSON object/array, or a string holding one.

    CLM renders a structured state as `key: value` prose, the layout its heads were trained on,
    so a string that parses as a JSON object or array is passed on parsed.
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
    raise DecisionInputError("state must be text or a JSON object/array, got %s" % type(state).__name__)


def _clean_texts(items: list[Any], what: str, max_chars: int) -> list[str]:
    texts = []
    for i, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise DecisionInputError("%s[%d] must be a non-empty string" % (what, i))
        text = item.strip()
        if len(text) > max_chars:
            raise DecisionInputError("%s[%d] is %d characters; the limit is %d" % (what, i, len(text), max_chars))
        texts.append(text)
    return texts


def _reject_duplicates(texts: list[str], what: str) -> None:
    seen: set[str] = set()
    duplicates = []
    for text in texts:
        if text in seen and text not in duplicates:
            duplicates.append(text)
        seen.add(text)
    if duplicates:
        shown = ", ".join(repr(d[:60]) for d in duplicates[:5])
        raise DecisionInputError("%s must be unique; repeated: %s" % (what, shown))


def _instructions(instructions: Any, required: bool) -> str | None:
    if instructions is None and not required:
        return None
    if not isinstance(instructions, str) or not instructions.strip():
        if required:
            raise DecisionInputError("instructions must be a non-empty string: the question to answer")
        return None
    text = instructions.strip()
    if len(text) > MAX_INSTRUCTIONS_CHARS:
        raise DecisionInputError("instructions are %d characters; the limit is %d" % (len(text), MAX_INSTRUCTIONS_CHARS))
    return text


def _choice_criteria(options: Any) -> dict[str, str | None]:
    if options is None:
        raise DecisionInputError(
            "choice needs options: a list of labels, or an object mapping each label to a description"
        )
    if isinstance(options, (dict, list)) and len(options) > MAX_CHOICE_OPTIONS:
        raise DecisionInputError(
            "choice takes at most %d options; use rank_clm_candidates for longer lists" % MAX_CHOICE_OPTIONS
        )
    if isinstance(options, dict):
        labels = _clean_texts(list(options.keys()), "options keys", MAX_OPTION_CHARS)
        # Keys that differ only in surrounding spaces would merge once stripped.
        _reject_duplicates(labels, "option labels")
        criteria: dict[str, str | None] = {}
        for label, description in zip(labels, options.values()):
            if description is not None and not isinstance(description, str):
                raise DecisionInputError("options[%r] description must be a string" % label)
            description = description.strip() if description else ""
            if len(description) > MAX_OPTION_CHARS:
                raise DecisionInputError(
                    "options[%r] description is %d characters; the limit is %d"
                    % (label, len(description), MAX_OPTION_CHARS)
                )
            criteria[label] = description or None
    elif isinstance(options, list):
        labels = _clean_texts(options, "options", MAX_OPTION_CHARS)
        # Check before building the dict, which would silently merge repeats.
        _reject_duplicates(labels, "option labels")
        criteria = {label: None for label in labels}
    else:
        raise DecisionInputError("options must be a list of strings or an object of label -> description")
    if len(criteria) < 2:
        raise DecisionInputError("choice needs at least 2 options to form a distribution")
    if len(criteria) > MAX_CHOICE_OPTIONS:
        raise DecisionInputError(
            "choice takes at most %d options; use rank_clm_candidates for longer lists" % MAX_CHOICE_OPTIONS
        )
    # CLM embeds the description, or the label when there is none: those texts must differ.
    _reject_duplicates([d or label for label, d in criteria.items()], "option texts (descriptions, or labels)")
    return criteria


def build_request(state: Any, question_type: Any, instructions: Any, options: Any = None) -> DecisionRequest:
    """Validate `evaluate_clm_decision` arguments and build the CLM question they describe."""
    parsed_state = parse_state(state)
    text = _instructions(instructions, required=True)
    notes: list[str] = []

    if question_type == "choice":
        criteria = _choice_criteria(options)
        labels = tuple(criteria)
        question: dict[str, Any] = {"type": "choice", "instructions": text, "criteria": criteria}
    elif question_type == "score":
        if options is None:
            levels = list(DEFAULT_SCORE_LEVELS)
            notes.append("no options given; used the default rubric %s" % " < ".join(levels))
        elif isinstance(options, list):
            if len(options) > MAX_SCORE_LEVELS:
                raise DecisionInputError("score needs between 2 and %d rubric levels" % MAX_SCORE_LEVELS)
            levels = _clean_texts(options, "options", MAX_OPTION_CHARS)
        else:
            raise DecisionInputError("score options must be a list of rubric levels, lowest first")
        _reject_duplicates(levels, "rubric levels")
        if not 2 <= len(levels) <= MAX_SCORE_LEVELS:
            raise DecisionInputError("score needs between 2 and %d rubric levels" % MAX_SCORE_LEVELS)
        labels = tuple(levels)
        question = {"type": "score", "instructions": text, "criteria": levels}
    elif question_type == "bool":
        question = {"type": "noul", "instructions": text}
        if options is not None:
            if not isinstance(options, list) or len(options) != 2:
                raise DecisionInputError(
                    "bool options are optional; when given they must be exactly [yes_description, no_description]"
                )
            yes, no = _clean_texts(options, "options", MAX_OPTION_CHARS)
            _reject_duplicates([yes, no], "bool descriptions")
            question["criteria"] = {"true": yes, "false": no}
        labels = ("true", "false")
    else:
        raise DecisionInputError(
            "question_type must be one of %s, got %r" % (", ".join(QUESTION_TYPES), question_type)
        )
    return DecisionRequest(
        state=parsed_state, question_type=question_type, question=question, labels=labels, notes=tuple(notes)
    )


def build_rank_request(state: Any, candidates: Any, instructions: Any = None, top_k: Any = 10) -> RankRequest:
    """Validate `rank_clm_candidates` arguments."""
    parsed_state = parse_state(state)
    text = _instructions(instructions, required=False)
    if not isinstance(candidates, list):
        raise DecisionInputError("candidates must be a list of strings")
    if not 2 <= len(candidates) <= MAX_RANK_CANDIDATES:
        raise DecisionInputError("candidates must hold between 2 and %d texts" % MAX_RANK_CANDIDATES)
    texts = _clean_texts(candidates, "candidates", MAX_CANDIDATE_CHARS)
    _reject_duplicates(texts, "candidates")
    total = sum(len(t) for t in texts)
    if total > MAX_RANK_TOTAL_CHARS:
        raise DecisionInputError("candidates total %d characters; the limit is %d" % (total, MAX_RANK_TOTAL_CHARS))
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise DecisionInputError("top_k must be an integer from 1 to %d" % MAX_TOP_K)
    return RankRequest(state=parsed_state, instructions=text, candidates=tuple(texts), top_k=min(top_k, len(texts)))


def _probability(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResponseError("%s is not a number" % where)
    p = float(value)
    if not math.isfinite(p) or not -1e-6 <= p <= 1 + 1e-6:
        raise ResponseError("%s is %r, outside [0, 1]" % (where, value))
    return min(1.0, max(0.0, p))


def _distribution(answer: dict[str, Any], keys: list[str]) -> list[float]:
    """The answer's probabilities in `keys` order, checked to be a distribution over exactly those keys."""
    raw = answer.get("probabilities")
    if not isinstance(raw, dict) or set(raw) != set(keys):
        raise ResponseError("probabilities do not cover exactly the requested options")
    probs = [_probability(raw[k], "probability of %r" % k) for k in keys]
    if abs(sum(probs) - 1.0) > SUM_TOLERANCE:
        raise ResponseError("probabilities sum to %.4f, not 1" % sum(probs))
    return probs


def _answer_of(result: Any, expected_type: str) -> dict[str, Any]:
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise ResponseError("response has no answers object")
    answer = result["answers"].get(QUESTION_ID)
    if not isinstance(answer, dict) or answer.get("type") != expected_type:
        raise ResponseError("response does not answer the %s question" % expected_type)
    return answer


def decode_answer(request: DecisionRequest, result: Any) -> dict[str, Any]:
    """Validate CLM's answer and reshape it like evaluate_laya_decision's result.

    Only numbers are taken from the response. The answer label is recomputed from the
    distribution, so every string in the output comes from the request.
    """
    if request.question_type == "bool":
        answer = _answer_of(result, "noul")
        p_true = _probability(answer.get("noul"), "noul")
        probabilities = {"true": round(p_true, 4), "false": round(1.0 - p_true, 4)}
        extra: dict[str, Any] = {"probability_true": round(p_true, 4)}
        best = "true" if p_true >= 0.5 else "false"
        confidence = max(p_true, 1.0 - p_true)
    else:
        laya_type = request.question_type  # "choice" and "score" share their names with CLM's
        answer = _answer_of(result, laya_type)
        keys = list(request.labels) if laya_type == "choice" else [str(i) for i in range(len(request.labels))]
        probs = _distribution(answer, keys)
        probabilities = {label: round(p, 4) for label, p in zip(request.labels, probs)}
        top = max(range(len(probs)), key=probs.__getitem__)
        best = request.labels[top]
        confidence = probs[top]
        extra = {}
        if laya_type == "score":
            expected = sum(i * p for i, p in enumerate(probs))
            extra = {
                "expected_score": round(expected, 4),
                "expected_score_normalized": round(expected / (len(probs) - 1), 4),
            }
    return {"answer": best, "probabilities": probabilities, "confidence": round(confidence, 4), **extra}


def decode_ranking(request: RankRequest, result: Any) -> dict[str, Any]:
    """Top-k candidates by probability, with lift over a uniform guess and the margin at the top."""
    answer = _answer_of(result, "choice")
    n = len(request.candidates)
    probs = _distribution(answer, [str(i) for i in range(n)])
    order = sorted(range(n), key=lambda i: (-probs[i], i))
    ranking = []
    for rank, index in enumerate(order[: request.top_k], start=1):
        text = request.candidates[index]
        ranking.append({
            "rank": rank,
            "index": index,
            "candidate_preview": text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…",
            "probability": round(probs[index], 6),
            "lift": round(probs[index] * n, 2),
        })
    margin = probs[order[0]] - probs[order[1]]
    return {
        "ranking": ranking,
        "candidates_total": n,
        "uniform_probability": round(1.0 / n, 6),
        "top_margin": round(margin, 6),
    }
