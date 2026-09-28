"""Owns the Laya Router: builds it once, warms it in the background, and answers one decision."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from typing import Any, Sequence

from .questions import (
    DecisionInputError,
    DecisionRequest,
    ModelName,
    decode_answer,
    validate_model,
)

CHECKPOINTS = ("english", "multilingual", "typed-decisions")
# Laya's own type names, keyed by the tool's question_type.
LAYA_TYPES = {"choice": "choice", "score": "score", "bool": "noul"}

# All three checkpoints live in one Hugging Face repo, the English one at its root.
LAYA_REPO = "convaiinnovations/laya"
SUBFOLDERS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
# The revision this server was tested against (25 Sept 2026). Pinned, so a change upstream cannot
# silently swap the weights under a running tool; LAYA_MCP_REVISION=main follows the latest.
PINNED_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")


def _checkpoint_list(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if names in (["none"], ["0"]):
        return []
    unknown = [name for name in names if name not in CHECKPOINTS]
    if unknown:
        raise ValueError(
            "LAYA_MCP_PRELOAD: unknown checkpoint(s) %s; use %s or 'none'"
            % (", ".join(unknown), ", ".join(CHECKPOINTS))
        )
    return names


def default_preload(model: ModelName) -> list[str]:
    # Automatic routing sends English text to the English checkpoint, so that is the one to warm.
    return ["english"] if model == "auto" else [model]


class LayaEngine:
    """Thread-safe wrapper around one `laya.Router`.

    Inference is serialised: Laya answers a whole question set in one forward pass, so there is
    nothing to gain from overlapping calls on one device, and serialising keeps latency predictable.
    """

    def __init__(
        self,
        device: str | None = None,
        default_model: ModelName = "auto",
        preload: Sequence[str] | None = None,
        revision: str = PINNED_REVISION,
    ):
        self.device = device
        self.default_model = validate_model(default_model)
        self.preload = list(preload) if preload is not None else default_preload(self.default_model)
        if not REVISION_PATTERN.match(revision):
            raise ValueError("LAYA_MCP_REVISION must be a commit hash, branch or tag, got %r" % revision)
        self.revision = revision
        self._router: Any = None
        self._fetched: set[str] = set()
        self._lock = threading.Lock()
        self._warming = False
        self._warm_error: str | None = None
        self._warm_seconds: float | None = None

    @classmethod
    def from_env(cls) -> LayaEngine:
        """LAYA_DEVICE, LAYA_MCP_MODEL, LAYA_MCP_PRELOAD (comma list), LAYA_MCP_REVISION."""
        model = validate_model(os.environ.get("LAYA_MCP_MODEL", "").strip() or "auto")
        raw_preload = os.environ.get("LAYA_MCP_PRELOAD", "").strip()
        preload = _checkpoint_list(raw_preload) if raw_preload else None
        device = os.environ.get("LAYA_DEVICE", "").strip() or None
        revision = os.environ.get("LAYA_MCP_REVISION", "").strip() or PINNED_REVISION
        return cls(device=device, default_model=model, preload=preload, revision=revision)

    def _router_locked(self) -> Any:
        if self._router is None:
            from laya import Router

            self._router = Router(device=self.device)
        return self._router

    def _fetch_locked(self, checkpoint: str) -> None:
        """Download one checkpoint at the pinned revision and point the Router at that snapshot.

        The Router would otherwise download from the repo's main branch. Only this checkpoint's
        files are fetched; a snapshot already in the Hugging Face cache is reused.
        """
        if checkpoint in self._fetched:
            return
        from huggingface_hub import snapshot_download

        sub = SUBFOLDERS[checkpoint]
        prefix = sub + "/" if sub else ""
        path = snapshot_download(
            LAYA_REPO,
            revision=self.revision,
            allow_patterns=[prefix + name for name in CHECKPOINT_FILES],
            token=os.environ.get("HF_TOKEN") or None,
        )
        self._router_locked().models[checkpoint] = (path, sub)
        self._fetched.add(checkpoint)

    def agent(self, checkpoint: str) -> Any:
        """The loaded agent for a checkpoint, or None. Never triggers a load."""
        from laya.mcp.device import router_agent

        return router_agent(self._router, checkpoint)

    def warm_up(self) -> float:
        """Load every preload checkpoint and run one forward pass on each; returns seconds taken.

        The first forward pass on a fresh model is several times slower than the rest, so doing it
        here keeps that cost off the first real decision.
        """
        with self._lock:
            started = time.perf_counter()
            self._warming = True
            try:
                router = self._router_locked()
                for name in self.preload:
                    self._fetch_locked(name)
                    router.preload([name])
                    router.predict(
                        "Warm-up request.",
                        {"warm": {"type": "noul", "instructions": "Is this a warm-up request?"}},
                        model=name,
                    )
            except Exception as exc:
                self._warm_error = "%s: %s" % (type(exc).__name__, exc)
                raise
            finally:
                self._warming = False
            self._warm_error = None
            self._warm_seconds = time.perf_counter() - started
            return self._warm_seconds

    def start_warm_up(self) -> threading.Thread:
        """Warm up on a daemon thread; a decision that arrives first waits for it."""

        def run() -> None:
            try:
                seconds = self.warm_up()
                print(
                    "[laya-mcp-server] ready in %.1fs (%s)" % (seconds, ", ".join(self.preload) or "lazy"),
                    file=sys.stderr,
                )
            except Exception as exc:
                print("[laya-mcp-server] warm-up failed, will retry on demand: %s" % exc, file=sys.stderr)

        thread = threading.Thread(target=run, name="laya-warm-up", daemon=True)
        thread.start()
        return thread

    def evaluate(self, request: DecisionRequest, model: str | None = None) -> dict[str, Any]:
        """Answer one decision in a single forward pass and describe how it was answered."""
        model_name = self.default_model if model is None else validate_model(model)
        with self._lock:
            router = self._router_locked()
            target = model_name
            if model_name == "auto":
                # Routing is pure Python and takes well under a millisecond.
                target = router.route(request.state, request.questions)["model"]
            self._fetch_locked(target)
            started = time.perf_counter()
            try:
                result = router.predict(
                    request.state,
                    request.questions,
                    model=None if model_name == "auto" else model_name,
                )
            except ValueError as exc:
                # Laya raises ValueError for requests it cannot encode, e.g. options that overflow
                # the head token budget; those are the caller's to fix.
                raise DecisionInputError(str(exc)) from exc
            latency_ms = (time.perf_counter() - started) * 1000.0

        routing = result["routing"]
        checkpoint = routing["model"]
        answer = result["answers"][next(iter(request.questions))]
        usage = result.get("usage", {})
        agent = self.agent(checkpoint)

        calibration = self._calibration(agent, request)
        warnings = list(request.notes)
        if calibration is not None and not calibration["calibrated"]:
            warnings.append(
                "this checkpoint ships no usable temperature for %s questions, so their probabilities "
                "are not calibrated; do not gate on `confidence`" % calibration["bucket"]
            )
        max_len = agent.cfg.get("max_len", 512) if agent is not None else None
        if max_len and usage.get("input_tokens", 0) >= max_len:
            warnings.append(
                "the request filled the %s checkpoint's %d-token window, so part of the state was "
                "cut off; shorten it, or use model='multilingual' (1,024 tokens)" % (checkpoint, max_len)
            )

        return {
            **decode_answer(request, answer),
            "question_type": request.question_type,
            "model": checkpoint,
            "routing_reason": routing.get("reason"),
            "device": agent.device.type if agent is not None else None,
            "calibration": calibration,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "generated_tokens": int(usage.get("output_tokens", 0)),
            "warnings": warnings,
        }

    @staticmethod
    def _calibration(agent: Any, request: DecisionRequest) -> dict[str, Any] | None:
        """The temperature Laya divided the logits by, looked up exactly as `Agent._decode_answers` does.

        Laya clamps fitted temperatures to [0.5, 5]; a clamped one (the English checkpoint's
        `choice:11+` bucket ships 0.10) is not a fitted value, so that bucket is not calibrated.
        """
        if agent is None:
            return None
        from laya.common import QTYPES, temp_bucket

        qtype = QTYPES[LAYA_TYPES[request.question_type]]
        bucket = temp_bucket(qtype, len(request.labels))
        if bucket in agent.temperature_by_options:
            applied, fitted = agent.temperature_by_options[bucket], agent.temperature_by_options_raw[bucket]
        else:
            applied, fitted = agent.temperature[qtype], agent.temperature_raw[qtype]
        try:
            calibrated = float(fitted) == applied
        except (TypeError, ValueError):
            calibrated = False
        return {
            "method": "temperature_scaling",
            "bucket": bucket,
            "temperature": round(float(applied), 4),
            "calibrated": calibrated,
        }

    def status(self) -> dict[str, Any]:
        router = self._router
        loaded = list(router.loaded) if router is not None else []
        devices = {}
        for name in loaded:
            agent = self.agent(name)
            if agent is not None:
                devices[name] = agent.device.type
        if self._warming:
            state = "loading"
        elif loaded:
            state = "ready"
        elif self._warm_error:
            state = "error"
        else:
            state = "cold"
        return {
            "state": state,
            "default_model": self.default_model,
            "preload": list(self.preload),
            "loaded_checkpoints": devices,
            "requested_device": self.device or "auto",
            "revision": self.revision,
            "warm_up_seconds": round(self._warm_seconds, 1) if self._warm_seconds is not None else None,
            "last_error": self._warm_error,
        }
