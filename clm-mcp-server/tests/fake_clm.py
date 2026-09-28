"""A stand-in for clm-serve over real HTTP, for tests that must not need a GPU.

It follows clm-serve 0.1.0's wire format (routes, auth, status codes, answer shapes, and how a
question becomes candidate texts) but scores candidates by word overlap instead of embeddings, so
answers are deterministic and roughly sensible. `behaviour` switches on failure modes.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODELS = ["clm-latest", "clm-raw"]


STOPWORDS = {"a", "an", "and", "the", "is", "it", "this", "that", "to", "of", "my", "i", "you", "for", "on", "in"}


def _words(text: str) -> set[str]:
    """Crude stems (first five letters) of the non-stopwords, so 'charged' meets 'charges'."""
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def _to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return "\n".join("%s: %s" % (k, _to_text(v)) for k, v in x.items())
    if isinstance(x, list):
        return "\n".join("- %s" % _to_text(v) for v in x)
    return "" if x is None else str(x)


def _candidates(q: dict) -> tuple[list[str], list[str]]:
    """clm.schema.candidates: option keys and the text embedded for each."""
    t, crit, ins = q.get("type"), q.get("criteria"), _to_text(q.get("instructions")).strip()
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise ValueError("choice question needs a non-empty 'criteria' object")
        keys = list(crit)
        return keys, [crit[k] if crit[k] not in (None, "") else k for k in keys]
    if t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError("score question needs 'criteria' as an ordered list of >= 2 levels")
        return [str(i) for i in range(len(crit))], [_to_text(c) for c in crit]
    if t == "noul":
        crit = crit or {}
        texts = []
        for k in ("false", "true"):
            d = crit.get(k) or ("Yes. This is true: %s" % ins if k == "true" else "No. This is false: %s" % ins)
            texts.append("%s: %s" % (k, d))
        return ["false", "true"], texts
    raise ValueError("unknown question type %r" % t)


def _answer(state: Any, q: dict, temperature: float) -> dict:
    keys, texts = _candidates(q)
    context = _words(_to_text(state) + " " + _to_text(q.get("instructions")))
    logits = [20.0 * len(context & _words(t)) / math.sqrt(len(_words(t)) or 1) / temperature for t in texts]
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    probs = [e / sum(exps) for e in exps]
    dist = dict(zip(keys, probs))
    if q["type"] == "noul":
        return {"type": "noul", "noul": dist["true"]}
    j = max(range(len(probs)), key=probs.__getitem__)
    rest = [p for i, p in enumerate(probs) if i != j]
    conf = max(0.0, probs[j] - sum(rest) / len(rest))
    if q["type"] == "choice":
        return {"type": "choice", "choice": keys[j], "confidence": conf, "probabilities": dist}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": conf,
            "legend": {str(i): c for i, c in enumerate(q["criteria"])}, "probabilities": dist}


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # Clients that stop reading early (size caps, timeouts) are expected in these tests.
        pass


class FakeCLM:
    """`with FakeCLM(api_key="k") as fake: fake.url` serves on a free local port."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.behaviour: dict[str, Any] = {}
        self.requests: list[dict[str, Any]] = []
        self._server = _QuietServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return "http://127.0.0.1:%d" % self._server.server_address[1]

    def __enter__(self) -> FakeCLM:
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def _send(self, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _intercept(self) -> bool:
                """Apply a configured failure; True when the request is already answered."""
                b = fake.behaviour
                fake.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers)})
                if b.get("delay"):
                    time.sleep(b["delay"])
                if b.get("redirect_to"):
                    self.send_response(307)
                    self.send_header("Location", b["redirect_to"] + self.path)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return True
                if b.get("status"):
                    self._send(b["status"], {"detail": b.get("detail", "forced failure")})
                    return True
                if b.get("oversize"):
                    self._send(200, b"[" + b"0," * (5 * 1024 * 1024) + b"0]")
                    return True
                if b.get("raw_body") is not None:
                    self._send(200, b["raw_body"])
                    return True
                return False

            def _authorised(self) -> bool:
                if fake.api_key and self.headers.get("Authorization") != "Bearer %s" % fake.api_key:
                    self._send(401, {"detail": "invalid API key"})
                    return False
                return True

            def do_GET(self) -> None:
                if self._intercept():
                    return
                if self.path == "/health":
                    self._send(200, {"ok": True, "embedder": True, "models": MODELS,
                                     "cache": {"hit_rate": 0.5}})
                elif self.path == "/v1/models":
                    if self._authorised():
                        self._send(200, {"models": [{"name": m, "description": "", "release_date": ""} for m in MODELS]})
                else:
                    self._send(404, {"detail": "Not Found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                if self._intercept():
                    return
                if self.path != "/v1/systemone":
                    self._send(404, {"detail": "Not Found"})
                    return
                if not self._authorised():
                    return
                body = json.loads(raw)
                fake.requests[-1]["body"] = body
                model = body.get("model") or "clm-latest"
                if model not in MODELS:
                    self._send(422, {"detail": "unknown model %r; available: %s" % (model, MODELS)})
                    return
                try:
                    answers = {qid: _answer(body["state"], q, float(body.get("temperature", 1.0)))
                               for qid, q in body["questions"].items()}
                except (ValueError, KeyError, TypeError) as exc:
                    self._send(422, {"detail": "invalid request: %s" % exc})
                    return
                out = {"model": model, "answers": answers,
                       "usage": {"billing_units": len(answers), "input_tokens": 17, "output_tokens": 0}}
                if fake.behaviour.get("mutate"):
                    fake.behaviour["mutate"](out)
                self._send(200, out, {"X-CLM-Latency-Ms": "12.5"})

        return Handler
