"""A CPU stand-in for CLM's vLLM encoder: Qwen3-8B embeddings from a GGUF via llama.cpp.

CLM's heads read Qwen3-8B last-token-pooled embeddings from an OpenAI-compatible
`/v1/embeddings` endpoint, normally `vllm serve Qwen/Qwen3-8B --runner pooling` on a GPU. This
serves the same endpoint from a quantised GGUF on CPU (or an iGPU with a Vulkan/SYCL build of
llama.cpp), so CLM can run on a machine without an NVIDIA GPU. It is much slower (seconds per
new text on CPU instead of milliseconds), and a quantised encoder is not bit-identical to the
bf16 one the heads were trained on; see the README for the measured agreement.

    pip install llama-cpp-python numpy
    python llamacpp_embeddings.py --gguf Qwen3-8B-Q8_0.gguf --port 8090
    clm-serve --host 127.0.0.1 --emb-url http://127.0.0.1:8090/v1/embeddings ...

It binds 127.0.0.1 only and takes no API key: keep it on the same machine as clm-serve.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_TEXTS = 256            # clm-serve sends batches of 32
MAX_TEXT_CHARS = 64_000    # well past 2,048 tokens; the rest would be truncated anyway
READ_TIMEOUT_S = 30


def build_handler(llm, model_name: str, max_tokens: int):
    lock = threading.Lock()  # one llama.cpp context: embed one request at a time

    class Handler(BaseHTTPRequestHandler):
        timeout = READ_TIMEOUT_S  # a client that stops sending mid-request is dropped

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:  # keep request text out of the logs
            pass

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/embeddings":
                self._send(404, {"error": "not found"})
                return
            # Browsers can POST text/plain to localhost from any page, but can't set JSON content
            # types or a foreign Host without a preflight this server never answers.
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if host not in ("127.0.0.1", "localhost") or content_type != "application/json":
                self._send(403, {"error": "local JSON requests only"})
                return
            raw_length = self.headers.get("Content-Length") or ""
            if not raw_length.isdigit():
                self._send(411, {"error": "Content-Length required"})
                return
            length = int(raw_length)
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": "request too large"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                texts = body["input"]
                texts = [texts] if isinstance(texts, str) else list(texts)
                if not all(isinstance(t, str) and len(t) <= MAX_TEXT_CHARS for t in texts):
                    raise ValueError("input must be strings of at most %d characters" % MAX_TEXT_CHARS)
                if not 1 <= len(texts) <= MAX_TEXTS:
                    raise ValueError("send 1 to %d texts per request" % MAX_TEXTS)
            except (ValueError, KeyError, TypeError, RecursionError) as exc:
                self._send(400, {"error": str(exc)[:200]})
                return
            use_b64 = body.get("encoding_format") == "base64"
            data, tokens = [], 0
            with lock:
                for i, text in enumerate(texts):
                    # One text per call: embed() truncates to the context (max_tokens) and pools
                    # the last token, as vLLM's pooling runner does for Qwen3-8B.
                    values, count = llm.embed(text, normalize=False, truncate=True, return_count=True)
                    tokens += count
                    vector = np.asarray(values, dtype=np.float32)
                    encoded = base64.b64encode(vector.tobytes()).decode("ascii")
                    data.append({"object": "embedding", "index": i,
                                 "embedding": encoded if use_b64 else vector.tolist()})
            self._send(200, {"object": "list", "data": data, "model": model_name,
                             "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gguf", required=True, help="path to a Qwen3-8B GGUF (Q8_0 recommended)")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model-name", default="qwen3-8b", help="name reported at /v1/models")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    import llama_cpp

    llm = llama_cpp.Llama(
        model_path=args.gguf,
        embedding=True,
        pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
        n_ctx=args.max_tokens,
        n_batch=args.max_tokens,
        n_ubatch=args.max_tokens,
        n_threads=args.threads,
        verbose=False,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), build_handler(llm, args.model_name, args.max_tokens))
    print("[embeddings] %s on http://127.0.0.1:%d/v1/embeddings" % (args.gguf, args.port), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
