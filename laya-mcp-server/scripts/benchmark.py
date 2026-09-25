"""Measure evaluate_laya_decision latency per checkpoint and question type on this machine.

    uv run --frozen python scripts/benchmark.py                    # english + multilingual
    uv run --frozen python scripts/benchmark.py english --runs 50 --json out.json

Laya publishes ~33 ms per decision on a T4 GPU; run this on your own hardware to see where you land.
"""

import argparse
import json
import os
import statistics
import warnings

os.environ.setdefault("USE_TF", "0")

from laya_mcp_server.engine import CHECKPOINTS, LayaEngine  # noqa: E402
from laya_mcp_server.questions import build_request  # noqa: E402

STATE = {
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
DECISIONS = {
    "choice": ("Which department should handle this?", ["billing", "technical", "sales", "other"]),
    "score": ("How urgent is this?", ["not urgent", "soon", "blocking"]),
    "bool": ("Does the customer threaten to cancel?", None),
}


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoints", nargs="*", help="any of %s (default: english multilingual)" % ", ".join(CHECKPOINTS))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--json", help="also write the rows to this file")
    args = parser.parse_args()
    checkpoints = args.checkpoints or ["english", "multilingual"]
    for name in checkpoints:
        if name not in CHECKPOINTS:
            parser.error("unknown checkpoint %r" % name)

    rows = []
    for checkpoint in checkpoints:
        engine = LayaEngine(device=os.environ.get("LAYA_DEVICE") or None, default_model=checkpoint)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_s = engine.warm_up()
        for question_type, (instructions, options) in DECISIONS.items():
            request = build_request(STATE, question_type, instructions, options)
            for _ in range(3):
                engine.evaluate(request)
            results = [engine.evaluate(request) for _ in range(args.runs)]
            latencies = [r["latency_ms"] for r in results]
            rows.append({
                "checkpoint": checkpoint,
                "device": results[0]["device"],
                "question_type": question_type,
                "answer": results[0]["answer"],
                "confidence": results[0]["confidence"],
                "p50_ms": round(statistics.median(latencies), 1),
                "p95_ms": round(percentile(latencies, 0.95), 1),
                "min_ms": round(min(latencies), 1),
                "generated_tokens": max(r["generated_tokens"] for r in results),
                "load_s": round(load_s, 1),
            })
        del engine

    print("| checkpoint | device | type | answer (confidence) | p50 ms | p95 ms | min ms | generated tokens |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %(checkpoint)s | %(device)s | %(question_type)s | %(answer)s (%(confidence).3f) | %(p50_ms)s "
              "| %(p95_ms)s | %(min_ms)s | %(generated_tokens)s |" % r)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
