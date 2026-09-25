import os
import warnings

import pytest

os.environ.setdefault("USE_TF", "0")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def engine():
    """One warm English checkpoint shared by every model test (loading it takes ~10-20 s)."""
    from laya_mcp_server.engine import LayaEngine

    engine = LayaEngine(device=os.environ.get("LAYA_DEVICE") or None, preload=["english"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.warm_up()
    return engine


@pytest.fixture
async def client(engine):
    """An MCP client connected in-process to a server bound to the shared engine."""
    from mcp import Client

    from laya_mcp_server.server import build_server

    async with Client(build_server(engine, warm_up=False)) as client:
        yield client


async def call(client, **arguments):
    """Call evaluate_laya_decision and return its structured result, failing on a tool error."""
    result = await client.call_tool("evaluate_laya_decision", arguments)
    assert not result.is_error, result.content[0].text
    return result.structured_content
