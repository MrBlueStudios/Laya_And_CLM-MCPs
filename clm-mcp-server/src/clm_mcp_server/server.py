"""MCP stdio server exposing CLM-8B (a running clm-serve) as decision and ranking tools.

Environment:
  CLM_BASE_URL             clm-serve address (default http://127.0.0.1:8700). Plain http only to
                           this machine; use https or an SSH tunnel for anything else.
  CLM_API_KEY              bearer key, if clm-serve was started with CLM_API_KEY
  CLM_MODEL                head to use (default clm-latest)
  CLM_TIMEOUT              seconds per request (default 120; the first call on new texts is slowest)
  CLM_TEMPERATURE          divides the logits before the softmax (default 1)
  CLM_ALLOW_INSECURE_HTTP  1 to allow plain http to another host on a private, encrypted network
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field
from typing_extensions import NotRequired, TypedDict

from . import __version__
from .backend import (
    CLMBackend,
    CLMBackendError,
    CLMConfigError,
    CLMRequestRejected,
    CLMSettings,
    validate_model_name,
)
from .questions import (
    MAX_RANK_CANDIDATES,
    MAX_TOP_K,
    DecisionInputError,
    ResponseError,
    build_rank_request,
    build_request,
    decode_answer,
    decode_ranking,
)

SERVER_NAME = "clm-mcp-server"

INSTRUCTIONS = """\
CLM-8B (Contrastive-LM, Apache-2.0): a System One model that embeds a state and each candidate \
with a frozen Qwen3-8B encoder plus two small trained heads, and returns a probability \
distribution over the candidates. It never generates text. It runs behind a clm-serve backend \
(GPU for the encoder, or llama.cpp on CPU, which is slow).

Use rank_clm_candidates to pick the best of many candidates (up to 1,000): tools, retrieved \
passages, draft answers, next actions. CLM caches candidate embeddings, so re-ranking the same \
list against new states is fast. Use evaluate_clm_decision for a typed choice/score/bool question, \
the same shape as evaluate_laya_decision.

CLM's probabilities are relative to the candidates you give it and are not calibrated: it always \
ranks something first, even if nothing fits. For an absolute yes/no ("is any passage relevant?") \
or calibrated triage of a few labels, prefer the Laya tools. English only.
"""

EVALUATE_DESCRIPTION = """\
Answer one typed decision about `state` with CLM-8B. Returns the probability distribution over the \
options and its arg-max `answer`, with no text generation. Probabilities are relative to the \
options given and not calibrated: compare them, and take thresholds from your own data.

question_type:
- choice: pick one of `options` (a list of labels, or an object of label -> description; 2-100).
- score: ordinal rating; `options` are rubric levels, lowest first (default: very low .. very high).
- bool: yes/no; also returns probability_true. `options` is optional: [yes_description, no_description].

CLM compares meanings, and reads each option's description when one is given, otherwise its label. \
Write options as short plain sentences describing that answer ("Critical: an outage that needs \
action now"), not bare labels or identifiers: bare levels like "not urgent" / "blocking" can score \
badly. `instructions` is the question; CLM reads the state, then the question. For more than ~20 \
candidates use rank_clm_candidates. Needs a running clm-serve: call clm_server_status if a call fails.
"""

RANK_DESCRIPTION = """\
Rank up to 1,000 candidate texts by how well each fits `state` (and the optional `instructions` \
question), e.g. which tool to call, which retrieved passage answers the question, which draft \
answer is best, which action to take next. Returns the top_k candidates by index.

Probabilities sum to 1 over ALL candidates, so they shrink as the list grows. Judge them with \
`lift` (probability x number of candidates; 1 means no better than chance) and `top_margin` (lead \
of the best over the second). CLM always ranks something first even if no candidate fits: check \
the winner with a bool question when that matters. Describe each candidate in plain words ("Book \
a table at a restaurant."), not as an identifier ("reserve_restaurant"), keep all candidates in one \
form, and map the returned indices back to your own IDs. Needs a running clm-serve.
"""


class Calibration(TypedDict):
    method: Annotated[str, Field(description="'none', or 'fixed_temperature' when CLM_TEMPERATURE is set")]
    temperature: float
    calibrated: Annotated[bool, Field(description="Always false: CLM probabilities are relative, not calibrated")]


class DecisionResult(TypedDict):
    answer: Annotated[str, Field(description="Key of `probabilities` with the highest probability")]
    probabilities: Annotated[dict[str, float], Field(description="Distribution over every option, in option order")]
    confidence: Annotated[float, Field(description="Probability of `answer`, relative to these options")]
    expected_score: NotRequired[Annotated[float, Field(description="score only: expected level index, 0 = first")]]
    expected_score_normalized: NotRequired[Annotated[float, Field(description="score only: scaled to 0..1")]]
    probability_true: NotRequired[Annotated[float, Field(description="bool only: P(yes)")]]
    question_type: Literal["choice", "score", "bool"]
    model: str
    calibration: Calibration
    server_ms: Annotated[float | None, Field(description="Time clm-serve reported for the request")]
    round_trip_ms: float
    input_tokens: Annotated[int, Field(description="Encoder tokens spent; 0 when every text was cached")]
    generated_tokens: Annotated[int, Field(description="Always 0: CLM scores candidates instead of generating")]
    warnings: list[str]


class RankedCandidate(TypedDict):
    rank: int
    index: Annotated[int, Field(description="Position of the candidate in the `candidates` you passed")]
    candidate_preview: Annotated[str, Field(description="Start of the candidate text (200 characters)")]
    probability: Annotated[float, Field(description="Share of probability over all candidates")]
    lift: Annotated[float, Field(description="probability x candidates_total; 1.0 = chance")]


class RankResult(TypedDict):
    ranking: list[RankedCandidate]
    candidates_total: int
    uniform_probability: float
    top_margin: Annotated[float, Field(description="Probability of the best minus the second best")]
    model: str
    server_ms: float | None
    round_trip_ms: float
    input_tokens: int
    generated_tokens: int
    warnings: list[str]


class CLMService:
    """Settings, the HTTP client, and the two tool bodies, independent of MCP for testing."""

    def __init__(self, settings: CLMSettings | None = None, config_error: str | None = None, transport=None):
        self.settings = settings
        self.config_error = config_error
        self._backend = CLMBackend(settings, transport=transport) if settings is not None else None

    @classmethod
    def from_env(cls) -> CLMService:
        try:
            return cls(CLMSettings.from_env())
        except CLMConfigError as exc:
            # Start anyway and explain on every call, so Claude can tell the user what to fix.
            return cls(config_error=str(exc))

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    def _require_backend(self) -> CLMBackend:
        if self._backend is None:
            raise CLMBackendError("clm-mcp-server is misconfigured: %s" % self.config_error)
        return self._backend

    def _model(self, model: str | None) -> str:
        if model is None:
            return self.settings.model
        try:
            return validate_model_name(model)
        except ValueError as exc:
            raise DecisionInputError(str(exc)) from exc

    def _calibration(self) -> dict[str, Any]:
        t = self.settings.temperature
        return {"method": "none" if t == 1.0 else "fixed_temperature", "temperature": t, "calibrated": False}

    def evaluate(self, state, question_type, instructions, options=None, model=None) -> dict[str, Any]:
        backend = self._require_backend()
        request = build_request(state, question_type, instructions, options)
        model_name = self._model(model)
        response = backend.systemone(request.state, request.questions, model_name)
        return {
            **decode_answer(request, response.body),
            "question_type": request.question_type,
            "model": model_name,
            "calibration": self._calibration(),
            "server_ms": response.server_ms,
            "round_trip_ms": round(response.round_trip_ms, 1),
            **_usage(response.body),
            "warnings": list(request.notes),
        }

    def rank(self, state, candidates, instructions=None, top_k=10, model=None) -> dict[str, Any]:
        backend = self._require_backend()
        request = build_rank_request(state, candidates, instructions, top_k)
        model_name = self._model(model)
        response = backend.systemone(request.state, request.questions, model_name)
        result = decode_ranking(request, response.body)
        warnings = []
        top = result["ranking"][0]
        if top["lift"] < 2.0:
            warnings.append(
                "no clear winner: the best candidate is only %.1fx a uniform guess" % top["lift"]
            )
        return {
            **result,
            "model": model_name,
            "server_ms": response.server_ms,
            "round_trip_ms": round(response.round_trip_ms, 1),
            **_usage(response.body),
            "warnings": warnings,
        }

    def status(self) -> dict[str, Any]:
        base = {"server_version": __version__}
        if self._backend is None:
            return {**base, "state": "misconfigured", "error": self.config_error}
        settings = self.settings
        base.update({
            "base_url": settings.display_url,
            "tls": settings.uses_tls,
            "api_key_configured": settings.api_key is not None,
            "default_model": settings.model,
            "temperature": settings.temperature,
        })
        try:
            health = self._backend.health().body
        except CLMBackendError as exc:
            return {**base, "state": "unreachable", "error": str(exc)}
        health = health if isinstance(health, dict) else {}
        out = {
            **base,
            "state": "ready" if health.get("ok") is True and health.get("embedder") is True else "degraded",
            "encoder_reachable": health.get("embedder") is True,
            "mock_encoder": health.get("mock") is True,
            "cache_hit_rate": _cache_hit_rate(health.get("cache")),
        }
        try:
            models = self._backend.models().body
            names = [m.get("name") for m in models.get("models", []) if isinstance(m, dict)]
            out["models"] = [n for n in names if isinstance(n, str) and len(n) <= 64][:20]
        except (CLMBackendError, AttributeError) as exc:
            out["models_error"] = str(exc)
        return out


def _usage(body: Any) -> dict[str, int]:
    usage = body.get("usage") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}

    def count(key: str) -> int:
        value = usage.get(key, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    return {"input_tokens": count("input_tokens"), "generated_tokens": count("output_tokens")}


def _cache_hit_rate(cache: Any) -> float | None:
    rate = cache.get("hit_rate") if isinstance(cache, dict) else None
    return round(float(rate), 3) if isinstance(rate, (int, float)) and not isinstance(rate, bool) else None


def _tool_errors(fn):
    """Run a tool body; turn every failure into a ToolError the model can read."""
    try:
        return fn()
    except (DecisionInputError, CLMRequestRejected) as exc:
        raise ToolError("invalid request: %s" % exc) from exc
    except CLMBackendError as exc:
        raise ToolError(str(exc)) from exc
    except ResponseError as exc:
        raise ToolError("the CLM server returned an invalid answer: %s" % exc) from exc
    except Exception as exc:
        raise ToolError("CLM call failed: %s: %s" % (type(exc).__name__, exc)) from exc


def build_server(service: CLMService) -> MCPServer:
    @asynccontextmanager
    async def lifespan(_server: MCPServer):
        try:
            yield {}
        finally:
            service.close()

    server = MCPServer(
        SERVER_NAME, version=__version__, instructions=INSTRUCTIONS, lifespan=lifespan, log_level="WARNING"
    )
    read_only = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

    @server.tool(name="evaluate_clm_decision", title="CLM typed decision",
                 description=EVALUATE_DESCRIPTION, annotations=read_only)
    def evaluate_clm_decision(
        *,
        state: Annotated[
            str | dict[str, Any] | list[Any],
            Field(description="Context: text, or a JSON object/array (rendered as key: value prose). "
                  "A string holding JSON is parsed."),
        ],
        question_type: Annotated[Literal["choice", "score", "bool"], Field(description="choice, score or bool")],
        options: Annotated[
            list[str] | dict[str, str] | None,
            Field(description="choice: labels, or {label: description} (required). score: rubric levels, "
                  "lowest first (optional). bool: [yes_description, no_description] (optional)."),
        ] = None,
        instructions: Annotated[str, Field(description="The question to answer about the state")],
        model: Annotated[str | None, Field(description="CLM head to use; default from CLM_MODEL (clm-latest)")] = None,
    ) -> DecisionResult:
        return _tool_errors(lambda: service.evaluate(state, question_type, instructions, options, model))

    @server.tool(name="rank_clm_candidates", title="CLM candidate ranking",
                 description=RANK_DESCRIPTION, annotations=read_only)
    def rank_clm_candidates(
        *,
        state: Annotated[
            str | dict[str, Any] | list[Any],
            Field(description="What the candidates are ranked against: a query, task, or situation"),
        ],
        candidates: Annotated[
            list[str],
            Field(description="2 to %d distinct candidate texts, in one consistent form" % MAX_RANK_CANDIDATES),
        ],
        instructions: Annotated[
            str | None,
            Field(description="Optional question appended to the state, e.g. 'Which tool should handle this?'"),
        ] = None,
        top_k: Annotated[int, Field(description="How many of the best candidates to return (1-%d)" % MAX_TOP_K)] = 10,
        model: Annotated[str | None, Field(description="CLM head to use; default from CLM_MODEL (clm-latest)")] = None,
    ) -> RankResult:
        return _tool_errors(lambda: service.rank(state, candidates, instructions, top_k, model))

    @server.tool(name="clm_server_status", title="CLM backend status",
                 description="Whether the CLM backend (clm-serve and its encoder) is reachable and ready, "
                 "which heads it serves, and how this MCP server is configured (never the API key). "
                 "Call it when a CLM call fails.", annotations=read_only)
    def clm_server_status() -> dict[str, Any]:
        return service.status()

    return server


def _check(service: CLMService) -> int:
    """Print the backend status and one sample ranking; exit non-zero when it isn't usable."""
    status = service.status()
    print(json.dumps(status, indent=2))
    if status.get("state") != "ready":
        return 1
    sample = service.rank(
        "What causes tides on Earth?",
        ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."],
        top_k=3,
    )
    print(json.dumps(sample, indent=2))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="clm-decision-mcp", description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="print the backend status and one sample ranking, then exit")
    args = parser.parse_args(argv)
    service = CLMService.from_env()
    if args.check:
        try:
            sys.exit(_check(service))
        except (CLMBackendError, DecisionInputError, ResponseError) as exc:
            print("error: %s" % exc, file=sys.stderr)
            sys.exit(1)
        finally:
            service.close()
    build_server(service).run()


if __name__ == "__main__":
    main()
