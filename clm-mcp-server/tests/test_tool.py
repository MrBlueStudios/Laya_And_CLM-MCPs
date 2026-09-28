"""The MCP tools over an in-process client, against the fake clm-serve."""

import math

import pytest

from clm_mcp_server.server import CLMService, build_server
from conftest import call, call_error, settings_for

pytestmark = pytest.mark.anyio

TICKET = "Customer: my invoice was charged twice and nobody answers the phone!"


async def test_tools_are_listed_with_their_schemas(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"evaluate_clm_decision", "rank_clm_candidates", "clm_server_status"}
    evaluate = tools["evaluate_clm_decision"].input_schema
    assert list(evaluate["properties"]) == ["state", "question_type", "options", "instructions", "model"]
    assert set(evaluate["required"]) == {"state", "question_type", "instructions"}
    rank = tools["rank_clm_candidates"].input_schema
    assert list(rank["properties"]) == ["state", "candidates", "instructions", "top_k", "model"]
    assert set(rank["required"]) == {"state", "candidates"}
    for tool in tools.values():
        assert tool.annotations.read_only_hint is True
        assert len(tool.description) < 2048  # Claude Code truncates longer descriptions


def assert_distribution(result, labels):
    probabilities = result["probabilities"]
    assert list(probabilities) == list(labels)
    assert math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3)
    assert result["answer"] == max(probabilities, key=probabilities.get)
    assert result["confidence"] == pytest.approx(max(probabilities.values()), abs=1e-4)
    assert result["generated_tokens"] == 0
    assert result["calibration"] == {"method": "none", "temperature": 1.0, "calibrated": False}


async def test_choice(client, fake):
    options = {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}
    result = await call(client, "evaluate_clm_decision", state=TICKET, question_type="choice",
                        options=options, instructions="Which team should handle this?")
    assert_distribution(result, options)
    assert result["answer"] == "billing"
    assert result["server_ms"] == 12.5 and result["input_tokens"] == 17
    assert fake.requests[-1]["body"]["questions"]["decision"]["criteria"] == options


async def test_score_and_bool(client):
    levels = ["It can wait a few days.", "An outage that needs action right now."]
    score = await call(client, "evaluate_clm_decision", state="Production outage right now for everyone",
                       question_type="score", options=levels, instructions="How urgent is this?")
    assert_distribution(score, levels)
    assert 0 <= score["expected_score_normalized"] <= 1
    yes_no = await call(client, "evaluate_clm_decision", state=TICKET, question_type="bool",
                        instructions="Is the invoice charged twice?")
    assert_distribution(yes_no, ["true", "false"])
    assert yes_no["probability_true"] == yes_no["probabilities"]["true"]


async def test_rank(client, fake):
    candidates = ["Find and book flights.", "Book a table at a restaurant.", "Get the weather forecast."]
    result = await call(client, "rank_clm_candidates", state="Book a restaurant table for two tonight",
                        candidates=candidates, top_k=2)
    assert [r["rank"] for r in result["ranking"]] == [1, 2]
    assert result["ranking"][0]["index"] == 1
    assert result["ranking"][0]["candidate_preview"] == candidates[1]
    assert result["candidates_total"] == 3
    sent = fake.requests[-1]["body"]["questions"]["decision"]
    assert sent == {"type": "choice", "instructions": None, "criteria": {"0": candidates[0], "1": candidates[1],
                                                                          "2": candidates[2]}}


async def test_rank_warns_without_a_clear_winner(client):
    result = await call(client, "rank_clm_candidates", state="zzz", candidates=["aaa", "bbb", "ccc"])
    assert result["ranking"][0]["lift"] == pytest.approx(1.0)
    assert result["warnings"] and "no clear winner" in result["warnings"][0]


@pytest.mark.parametrize("arguments, message", [
    ({"question_type": "choice", "options": ["only"]}, "at least 2 options"),
    ({"question_type": "bool", "model": "missing-head"}, "unknown model 'missing-head'"),
    ({"question_type": "bool", "model": "../x"}, "model names"),
    ({"question_type": "maybe"}, "question_type"),
])
async def test_bad_arguments(client, arguments, message):
    error = await call_error(client, "evaluate_clm_decision", state="text", instructions="Question?", **arguments)
    assert message in error


async def test_an_invalid_server_answer_is_rejected(client, fake):
    def break_distribution(out):
        out["answers"]["decision"]["probabilities"] = {"true": 0.9, "false": 0.9}

    def inject_label(out):
        out["answers"]["decision"]["noul"] = 1.7

    fake.behaviour = {"mutate": inject_label}
    error = await call_error(client, "evaluate_clm_decision", state="text", question_type="bool", instructions="Q?")
    assert "invalid answer" in error
    fake.behaviour = {"mutate": break_distribution}
    error = await call_error(client, "evaluate_clm_decision", state="text", question_type="choice",
                             options=["true", "false"], instructions="Q?")
    assert "invalid answer" in error


async def test_backend_failures_reach_the_model_as_readable_errors(client, fake):
    fake.behaviour = {"status": 502, "detail": "embedder unreachable at http://127.0.0.1:8090"}
    error = await call_error(client, "rank_clm_candidates", state="q", candidates=["a", "b"])
    assert "encoder" in error and "8090" in error


async def test_status(client):
    status = (await client.call_tool("clm_server_status", {})).structured_content
    assert status["state"] == "ready"
    assert status["models"] == ["clm-latest", "clm-raw"]
    assert status["api_key_configured"] is True
    assert "test-key" not in str(status)  # never echo the key


async def test_misconfigured_server_still_starts_and_explains():
    from mcp import Client

    service = CLMService(config_error="CLM_BASE_URL sends decisions over plain http to 10.0.0.5")
    async with Client(build_server(service)) as client:
        error = await call_error(client, "rank_clm_candidates", state="q", candidates=["a", "b"])
        assert "misconfigured" in error and "plain http" in error
        status = (await client.call_tool("clm_server_status", {})).structured_content
        assert status["state"] == "misconfigured"


async def test_unreachable_backend_status():
    from mcp import Client

    service = CLMService(settings_for("http://127.0.0.1:9"))
    async with Client(build_server(service)) as client:
        status = (await client.call_tool("clm_server_status", {})).structured_content
        assert status["state"] == "unreachable" and "can't reach" in status["error"]
