"""HTTP client for a running `clm-serve`, with the network rules the MCP server relies on.

- The server address and API key come from the environment only, never from tool arguments.
- Plain http is allowed only to this machine (loopback). Anything else needs https, unless
  CLM_ALLOW_INSECURE_HTTP=1 says the link is already private (an SSH tunnel, WireGuard, Tailscale).
- Redirects are never followed, so the API key only ever goes to the configured origin.
- Responses are size-capped and parsed as JSON; `questions.py` then validates their content.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from . import __version__

DEFAULT_BASE_URL = "http://127.0.0.1:8700"
DEFAULT_MODEL = "clm-latest"
DEFAULT_TIMEOUT_S = 120.0
CONNECT_TIMEOUT_S = 5.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_DETAIL_CHARS = 300
MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class CLMConfigError(ValueError):
    """The environment configures the CLM connection in a way we refuse to use."""


class CLMBackendError(RuntimeError):
    """The CLM server could not answer. The message is safe to return to the model."""


class CLMRequestRejected(CLMBackendError):
    """The CLM server rejected the request itself (HTTP 422): the caller can fix it."""


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_model_name(name: str) -> str:
    if not MODEL_NAME.match(name):
        raise ValueError("model names are letters, digits, '.', '_' and '-' (at most 64), got %r" % name)
    return name


@dataclass(frozen=True)
class CLMSettings:
    base_url: str
    api_key: str | None
    model: str
    timeout_s: float
    temperature: float
    allow_insecure_http: bool

    @property
    def display_url(self) -> str:
        return self.base_url

    @property
    def uses_tls(self) -> bool:
        return self.base_url.startswith("https://")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> CLMSettings:
        """CLM_BASE_URL, CLM_API_KEY, CLM_MODEL, CLM_TIMEOUT, CLM_TEMPERATURE, CLM_ALLOW_INSECURE_HTTP."""
        allow_insecure = environ.get("CLM_ALLOW_INSECURE_HTTP", "").strip().lower() in ("1", "true", "yes")
        base_url = (environ.get("CLM_BASE_URL", "").strip() or DEFAULT_BASE_URL).rstrip("/")
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise CLMConfigError("CLM_BASE_URL must be an http:// or https:// URL, got %r" % base_url)
        if parts.username or parts.password:
            raise CLMConfigError("CLM_BASE_URL must not contain credentials; put the key in CLM_API_KEY")
        if parts.query or parts.fragment:
            raise CLMConfigError("CLM_BASE_URL must not have a query string or fragment")
        if parts.scheme == "http" and not _is_loopback(parts.hostname) and not allow_insecure:
            raise CLMConfigError(
                "CLM_BASE_URL sends decisions over plain http to %s. Use https, tunnel the server to "
                "127.0.0.1 (ssh -L 8700:127.0.0.1:8700 host), or set CLM_ALLOW_INSECURE_HTTP=1 if the "
                "network is already private and encrypted." % parts.hostname
            )

        api_key = environ.get("CLM_API_KEY", "").strip() or None
        if api_key is not None and any(c in api_key for c in "\r\n"):
            raise CLMConfigError("CLM_API_KEY must be a single line")

        model = environ.get("CLM_MODEL", "").strip() or DEFAULT_MODEL
        try:
            validate_model_name(model)
        except ValueError as exc:
            raise CLMConfigError("CLM_MODEL: %s" % exc) from exc

        timeout_s = _number(environ, "CLM_TIMEOUT", DEFAULT_TIMEOUT_S, low=1.0, high=3600.0)
        temperature = _number(environ, "CLM_TEMPERATURE", 1.0, low=0.01, high=100.0)
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            temperature=temperature,
            allow_insecure_http=allow_insecure,
        )


def _number(environ: Mapping[str, str], name: str, default: float, *, low: float, high: float) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise CLMConfigError("%s must be a number, got %r" % (name, raw)) from exc
    if not (math.isfinite(value) and low <= value <= high):
        raise CLMConfigError("%s must be between %g and %g, got %r" % (name, low, high, raw))
    return value


@dataclass(frozen=True)
class CLMResponse:
    body: Any
    server_ms: float | None
    round_trip_ms: float


class CLMBackend:
    """A small, thread-safe client for `clm-serve`'s HTTP API."""

    def __init__(self, settings: CLMSettings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        headers = {"Accept": "application/json", "User-Agent": "clm-mcp-server/%s" % __version__}
        if settings.api_key:
            headers["Authorization"] = "Bearer %s" % settings.api_key
        self._client = httpx.Client(
            base_url=settings.base_url,
            headers=headers,
            timeout=httpx.Timeout(settings.timeout_s, connect=CONNECT_TIMEOUT_S),
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def systemone(self, state: Any, questions: dict[str, Any], model: str | None = None) -> CLMResponse:
        body = {
            "state": state,
            "questions": questions,
            "model": model or self.settings.model,
            "temperature": self.settings.temperature,
        }
        return self._request("POST", "/v1/systemone", body)

    def health(self) -> CLMResponse:
        return self._request("GET", "/health")

    def models(self) -> CLMResponse:
        return self._request("GET", "/v1/models")

    def _request(self, method: str, path: str, body: Any = None) -> CLMResponse:
        where = self.settings.display_url
        started = time.perf_counter()
        try:
            with self._client.stream(method, path, json=body) as response:
                raw = _read_capped(response)
                status, headers = response.status_code, response.headers
        except httpx.TimeoutException as exc:
            raise CLMBackendError(
                "the CLM server at %s did not answer within %gs (CLM_TIMEOUT)" % (where, self.settings.timeout_s)
            ) from exc
        except httpx.TransportError as exc:
            raise CLMBackendError(
                "can't reach the CLM server at %s (%s). Start clm-serve (see the clm-mcp-server README) "
                "or point CLM_BASE_URL at it." % (where, type(exc).__name__)
            ) from exc
        round_trip_ms = (time.perf_counter() - started) * 1000.0

        if 300 <= status < 400:
            raise CLMBackendError(
                "the CLM server at %s answered with a redirect (%d). Redirects are not followed, so the API "
                "key cannot leak to another address: point CLM_BASE_URL at the final address." % (where, status)
            )
        if status in (401, 403):
            raise CLMBackendError(
                "the CLM server at %s rejected the API key (%d). Set CLM_API_KEY to the key clm-serve was "
                "started with." % (where, status)
            )
        if status == 422:
            raise CLMRequestRejected("the CLM server rejected the request: %s" % _detail(raw))
        if status == 502:
            raise CLMBackendError(
                "the CLM server's encoder (vLLM or llama.cpp) failed or is unreachable: %s" % _detail(raw)
            )
        if status != 200:
            raise CLMBackendError("the CLM server returned HTTP %d: %s" % (status, _detail(raw)))
        try:
            parsed = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise CLMBackendError("the CLM server returned a body that is not JSON") from exc
        return CLMResponse(body=parsed, server_ms=_latency_header(headers), round_trip_ms=round_trip_ms)


def _read_capped(response: httpx.Response) -> bytes:
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
        raise CLMBackendError("the CLM server's response is %s bytes; the limit is %d" % (declared, MAX_RESPONSE_BYTES))
    chunks, size = [], 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise CLMBackendError("the CLM server's response exceeded %d bytes" % MAX_RESPONSE_BYTES)
        chunks.append(chunk)
    return b"".join(chunks)


def _detail(raw: bytes) -> str:
    """FastAPI's `{"detail": ...}`, or the start of the body, flattened to one short line."""
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "detail" in parsed:
            detail = parsed["detail"]
            text = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    except (ValueError, RecursionError):
        pass
    text = " ".join(text.split())
    return text if len(text) <= MAX_DETAIL_CHARS else text[: MAX_DETAIL_CHARS - 1] + "…"


def _latency_header(headers: httpx.Headers) -> float | None:
    try:
        value = float(headers.get("x-clm-latency-ms", ""))
    except ValueError:
        return None
    return round(value, 1) if math.isfinite(value) and 0 <= value < 3.6e6 else None
