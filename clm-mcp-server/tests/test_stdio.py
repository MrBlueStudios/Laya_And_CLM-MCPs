"""Launch the server from .mcp.json over stdio, the way Claude Code does."""

import json
import os
import re
import shutil
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from conftest import API_KEY

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(shutil.which("uv") is None, reason=".mcp.json launches with uv")]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def expand(value, env):
    """Claude Code's `${VAR}` / `${VAR:-default}` expansion for .mcp.json command, args and env."""
    return re.sub(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}",
        lambda m: env.get(m.group(1)) or (m.group(2) if m.group(2) is not None else ""),
        value,
    )


def server_from_mcp_json(user_env):
    config = json.loads((PROJECT_ROOT / ".mcp.json").read_text())["mcpServers"]["clm-mcp-server"]
    assert config["type"] == "stdio"
    # The key must come from the user's environment, never from the committed file.
    assert "${CLM_API_KEY" in config["env"]["CLM_API_KEY"]
    claude_env = {**{k: v for k, v in os.environ.items() if not k.startswith("CLM_")}, **user_env}
    return StdioServerParameters(
        command=expand(config["command"], claude_env),
        args=[expand(arg, claude_env) for arg in config["args"]],
        env={**claude_env, **{k: expand(v, claude_env) for k, v in config["env"].items()},
             "CLAUDE_PROJECT_DIR": str(PROJECT_ROOT)},
        cwd=PROJECT_ROOT,
    )


async def test_mcp_json_launches_the_server(fake):
    params = server_from_mcp_json({"CLM_BASE_URL": fake.url, "CLM_API_KEY": API_KEY})
    async with Client(params, read_timeout_seconds=120) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert tools == {"evaluate_clm_decision", "rank_clm_candidates", "clm_server_status"}
        status = (await client.call_tool("clm_server_status", {})).structured_content
        assert status["state"] == "ready" and status["api_key_configured"] is True
        result = await client.call_tool("rank_clm_candidates", {
            "state": "Book a restaurant table for two tonight",
            "candidates": ["Find and book flights.", "Book a table at a restaurant."],
        })
        assert not result.is_error, result.content[0].text
        assert result.structured_content["ranking"][0]["index"] == 1
    assert fake.requests[-1]["headers"]["Authorization"] == "Bearer %s" % API_KEY


async def test_mcp_json_defaults_to_local_and_keyless(fake):
    # With nothing set, the entry expands to http://127.0.0.1:8700 and no key.
    params = server_from_mcp_json({})
    assert params.env["CLM_BASE_URL"] == "http://127.0.0.1:8700"
    assert params.env["CLM_API_KEY"] == ""
