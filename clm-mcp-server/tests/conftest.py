import pytest

from clm_mcp_server.backend import CLMSettings
from clm_mcp_server.server import CLMService, build_server
from fake_clm import FakeCLM

API_KEY = "test-key-7f3a"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fake():
    with FakeCLM(api_key=API_KEY) as server:
        yield server


def settings_for(url: str, **env: str) -> CLMSettings:
    return CLMSettings.from_env({"CLM_BASE_URL": url, "CLM_API_KEY": API_KEY, **env})


@pytest.fixture
def service(fake):
    service = CLMService(settings_for(fake.url, CLM_TIMEOUT="5"))
    yield service
    service.close()


@pytest.fixture
async def client(service):
    from mcp import Client

    async with Client(build_server(service)) as client:
        yield client


async def call(client, tool, **arguments):
    """Call a tool and return its structured result, failing on a tool error."""
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content[0].text
    return result.structured_content


async def call_error(client, tool, **arguments) -> str:
    """Call a tool that must fail and return its error text."""
    result = await client.call_tool(tool, arguments)
    assert result.is_error, result.structured_content
    return result.content[0].text
