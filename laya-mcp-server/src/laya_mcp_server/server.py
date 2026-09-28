"""MCP stdio server exposing the Laya decision model as `evaluate_laya_decision`.

Environment:
  LAYA_DEVICE       torch device (cpu, cuda, cuda:1, mps); default: best available
  LAYA_MCP_MODEL    default checkpoint: auto (route by language), english, multilingual, typed-decisions
  LAYA_MCP_PRELOAD  checkpoints to load at startup, comma separated, or "none"; default follows
                    LAYA_MCP_MODEL (auto -> english)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field
from typing_extensions import NotRequired, TypedDict

from . import __version__
from .engine import LayaEngine
from .questions import DecisionInputError, build_request

SERVER_NAME = "laya-mcp-server"

INSTRUCTIONS = """\
Local Laya decision model (convaiinnovations/laya, Apache-2.0). A non-autoregressive encoder answers \
one typed question about a text or JSON state with a calibrated probability distribution in a single \
forward pass (~33 ms on a GPU, ~0.1-0.3 s on CPU). It never generates text, so there is nothing to \
parse or hallucinate.

Use evaluate_laya_decision for fast classification, routing, triage, moderation or guardrail checks, \
yes/no checks and ordinal scoring whenever the answer is one of a fixed set of options. Do not use it \
for open-ended questions, summarisation, code, or multi-step reasoning.

Read `confidence` (calibrated probability of the returned answer) and treat values below ~0.6 as \
uncertain. The checkpoints are zero-shot: they can be confidently wrong on unfamiliar tasks, so \
confirm high-stakes decisions another way.
"""

TOOL_DESCRIPTION = """\
Answer one typed decision about `state` with the local Laya model. Returns the full calibrated \
probability distribution over the options and its arg-max `answer`, in one forward pass with no text \
generation.

question_type:
- choice: pick one of `options` (a list of labels, or an object of label -> description; 2-20 work best).
- score: ordinal rating; `options` are rubric levels, lowest first (default: very low .. very high). \
Also returns expected_score.
- bool: yes/no; also returns probability_true. `options` is optional: [yes_description, no_description].

`instructions` is the question itself, phrased for the options, e.g. "Which team should handle this \
ticket?". The English checkpoint reads about 320 tokens of state; longer input is truncated and \
reported in `warnings`.
"""


class Calibration(TypedDict):
    method: str
    bucket: Annotated[str, Field(description="Laya temperature bucket: question type and option count")]
    temperature: Annotated[float, Field(description="Temperature the logits were divided by before softmax")]
    calibrated: Annotated[
        bool, Field(description="False when the checkpoint has no valid fitted temperature for this bucket")
    ]


class DecisionResult(TypedDict):
    answer: Annotated[str, Field(description="Key of `probabilities` with the highest probability")]
    probabilities: Annotated[
        dict[str, float],
        Field(description="Calibrated distribution over every option, in option order; sums to 1"),
    ]
    confidence: Annotated[float, Field(description="Calibrated probability of `answer` (max of the distribution)")]
    expected_score: NotRequired[
        Annotated[float, Field(description="score only: expected rubric index, 0 = first level")]
    ]
    expected_score_normalized: NotRequired[
        Annotated[float, Field(description="score only: expected_score scaled to 0..1")]
    ]
    probability_true: NotRequired[Annotated[float, Field(description="bool only: P(yes)")]]
    question_type: Literal["choice", "score", "bool"]
    model: Annotated[str, Field(description="Laya checkpoint that answered")]
    routing_reason: str | None
    device: str | None
    calibration: Calibration | None
    latency_ms: Annotated[float, Field(description="Model time: routing, tokenising, one forward pass, decoding")]
    input_tokens: int
    generated_tokens: Annotated[int, Field(description="Always 0: Laya scores options instead of generating")]
    warnings: list[str]


def build_server(engine: LayaEngine, *, warm_up: bool = True) -> MCPServer:
    """An MCP server bound to `engine`. Tests pass a shared, already warm engine."""

    @asynccontextmanager
    async def lifespan(_server: MCPServer):
        # Started after the stdio transport is up, so the handshake never waits on a model load.
        if warm_up and engine.preload:
            engine.start_warm_up()
        yield {}

    server = MCPServer(
        SERVER_NAME, version=__version__, instructions=INSTRUCTIONS, lifespan=lifespan, log_level="WARNING"
    )

    @server.tool(
        name="evaluate_laya_decision",
        title="Laya calibrated decision",
        description=TOOL_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    def evaluate_laya_decision(
        *,
        state: Annotated[
            str | dict[str, Any] | list[Any],
            Field(
                description="Context to evaluate: plain text, or a JSON object (e.g. an email with "
                "subject/body) or array (a conversation, oldest turn first). A string holding JSON is parsed."
            ),
        ],
        question_type: Annotated[
            Literal["choice", "score", "bool"],
            Field(description="choice = one of the options, score = ordinal rubric, bool = yes/no"),
        ],
        options: Annotated[
            list[str] | dict[str, str] | None,
            Field(
                description="choice: labels, or {label: description} (required). score: rubric levels, "
                "lowest first (optional). bool: [yes_description, no_description] (optional)."
            ),
        ] = None,
        instructions: Annotated[
            str,
            Field(description="The question to answer about the state, and any guidelines for answering it"),
        ],
        model: Annotated[
            Literal["auto", "english", "multilingual", "typed-decisions"] | None,
            Field(
                description="Checkpoint override. auto (default) routes English text to 'english' and other "
                "languages to 'multilingual'; 'multilingual' is ~2x faster and reads 1,024 tokens."
            ),
        ] = None,
    ) -> DecisionResult:
        try:
            request = build_request(state, question_type, instructions, options)
            return engine.evaluate(request, model)
        except DecisionInputError as exc:
            raise ToolError("invalid request: %s" % exc) from exc
        except Exception as exc:
            raise ToolError("Laya inference failed: %s: %s" % (type(exc).__name__, exc)) from exc

    @server.tool(
        name="laya_server_status",
        title="Laya server status",
        description="Whether the Laya checkpoints are loaded (state: cold, loading, ready, error), on which "
        "device, and the server's default model. Call it if a decision is slow or fails.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    )
    def laya_server_status() -> dict[str, Any]:
        return {**engine.status(), "server_version": __version__, "laya_version": _laya_version()}

    return server


def _laya_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("laya")
    except Exception:
        return None


def _prefetch(engine: LayaEngine) -> None:
    """Download and warm the preload checkpoints, answer one sample decision, and print it."""
    # laya prints warnings to stdout; keep stdout for the JSON result alone.
    with contextlib.redirect_stdout(sys.stderr):
        seconds = engine.warm_up()
    request = build_request(
        {"subject": "Duplicate charge", "body": "We were billed twice for March. Please refund the duplicate."},
        "choice",
        "Which team should handle this?",
        {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"},
    )
    with contextlib.redirect_stdout(sys.stderr):
        result = engine.evaluate(request)
    print("warmed %s in %.1fs" % (", ".join(engine.preload) or "nothing", seconds), file=sys.stderr)
    print(json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="laya-decision-mcp", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="download and load the preload checkpoints, run one sample decision, print it, and exit",
    )
    args = parser.parse_args(argv)

    # transformers probes for TensorFlow at import, and a present TF can deadlock model construction.
    os.environ.setdefault("USE_TF", "0")
    if os.name == "nt":
        # Windows without Developer Mode cannot create the Hugging Face cache's symlinks.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    engine = LayaEngine.from_env()
    if args.prefetch:
        _prefetch(engine)
        return
    # stdio: while serving, the SDK points fd 1 at stderr, so stray prints from laya/torch cannot
    # corrupt the JSON-RPC stream.
    build_server(engine).run()


if __name__ == "__main__":
    main()
