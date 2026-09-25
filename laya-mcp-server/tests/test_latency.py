"""Latency of one decision through MCP, against a device-dependent budget.

Laya's published figure is ~33 ms per decision on a T4 GPU (39.5 ms for the English checkpoint,
32.8 ms for multilingual). CPU is roughly 5-10x slower, so the default budget follows the device:

  LAYA_LATENCY_BUDGET_MS   override the p50 budget (ms) for the model time
  LAYA_LATENCY_REPORT      write the measured numbers to this JSON file
"""

import json
import os
import statistics
import time

import pytest

from conftest import call

pytestmark = [pytest.mark.model, pytest.mark.anyio]

DEFAULT_BUDGET_MS = {"cuda": 50.0, "mps": 120.0, "cpu": 500.0}
RUNS = 20


def budget_ms(device):
    override = os.environ.get("LAYA_LATENCY_BUDGET_MS")
    return float(override) if override else DEFAULT_BUDGET_MS.get(device, DEFAULT_BUDGET_MS["cpu"])


async def test_single_decision_latency(client, engine):
    arguments = dict(
        state="The checkout page returns a 500 error for every customer since this morning.",
        question_type="choice",
        options={"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"},
        instructions="Which department should handle this?",
    )
    for _ in range(3):
        await call(client, **arguments)

    model_ms, round_trip_ms = [], []
    for _ in range(RUNS):
        started = time.perf_counter()
        result = await call(client, **arguments)
        round_trip_ms.append((time.perf_counter() - started) * 1000.0)
        model_ms.append(result["latency_ms"])

    device = result["device"]
    report = {
        "device": device,
        "checkpoint": result["model"],
        "torch_threads": __import__("torch").get_num_threads(),
        "runs": RUNS,
        "model_p50_ms": round(statistics.median(model_ms), 1),
        "model_min_ms": round(min(model_ms), 1),
        "round_trip_p50_ms": round(statistics.median(round_trip_ms), 1),
        "mcp_overhead_p50_ms": round(statistics.median(r - m for r, m in zip(round_trip_ms, model_ms)), 1),
        "budget_ms": budget_ms(device),
    }
    print("\nlatency:", json.dumps(report))
    if os.environ.get("LAYA_LATENCY_REPORT"):
        with open(os.environ["LAYA_LATENCY_REPORT"], "w") as f:
            json.dump(report, f, indent=2)

    assert report["model_p50_ms"] <= report["budget_ms"]
    # The MCP layer adds validation and JSON only; the model dominates.
    assert report["mcp_overhead_p50_ms"] < 25.0
