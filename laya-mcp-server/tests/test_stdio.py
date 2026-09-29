"""Launch the server as a subprocess over stdio, the way Claude Code does."""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest
import torch
from mcp import Client
from mcp.client.stdio import StdioServerParameters

pytestmark = [pytest.mark.model, pytest.mark.anyio]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCP_JSON = PROJECT_ROOT / ".mcp.json"
DECISION = {
    "state": "Hi, we were billed twice for March. Please refund the duplicate.",
    "question_type": "choice",
    "options": ["billing", "technical", "other"],
    "instructions": "Which department should handle this?",
}


def expand(value, env):
    """Claude Code's `${VAR}` / `${VAR:-default}` expansion for .mcp.json command, args and env."""

    def substitute(match):
        name, default = match.group(1), match.group(2)
        return env.get(name) or (default if default is not None else "")

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", substitute, value)


def server_from_mcp_json():
    config = json.loads(MCP_JSON.read_text())["mcpServers"]["laya-mcp-server"]
    assert config["type"] == "stdio"
    # CLAUDE_PROJECT_DIR is only set in the *server's* environment, so .mcp.json expansion sees it
    # unset and falls back to the default, resolved against the project directory.
    claude_env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PROJECT_DIR"}
    env = {k: expand(v, claude_env) for k, v in config.get("env", {}).items()}
    return StdioServerParameters(
        command=expand(config["command"], claude_env),
        args=[expand(arg, claude_env) for arg in config["args"]],
        env={**os.environ, **env, "CLAUDE_PROJECT_DIR": str(PROJECT_ROOT)},
        cwd=PROJECT_ROOT,
    )


async def decide(params):
    async with Client(params, read_timeout_seconds=600) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        first_status = (await client.call_tool("laya_server_status", {})).structured_content
        result = await client.call_tool("evaluate_laya_decision", DECISION)
        assert not result.is_error, result.content[0].text
        final_status = (await client.call_tool("laya_server_status", {})).structured_content
        return tools, first_status, result.structured_content, final_status


@pytest.mark.skipif(shutil.which("uv") is None, reason=".mcp.json launches the server with uv")
async def test_mcp_json_launches_the_server():
    tools, first_status, result, final_status = await decide(server_from_mcp_json())

    assert tools == {"evaluate_laya_decision", "laya_server_status"}
    # The handshake does not wait for the model: warm-up runs in the background.
    assert first_status["state"] in ("cold", "loading", "ready")
    assert result["answer"] == "billing"
    assert result["generated_tokens"] == 0
    assert final_status["state"] == "ready"
    assert final_status["preload"] == ["english"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a host without CUDA to trigger laya's fallback print")
async def test_stray_stdout_does_not_corrupt_the_protocol():
    # Asking for CUDA on a host without it makes laya print its CPU-fallback warning to stdout
    # while the server is live. The stdio transport must keep that off the JSON-RPC stream.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "laya_mcp_server"],
        env={**os.environ, "LAYA_DEVICE": "cuda", "LAYA_MCP_PRELOAD": "english"},
    )
    _tools, _first, result, final_status = await decide(params)

    assert result["answer"] == "billing"
    assert result["device"] == "cpu"
    assert final_status["requested_device"] == "cuda"
