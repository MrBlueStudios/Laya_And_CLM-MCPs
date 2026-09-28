"""How the HTTP client behaves on the wire: auth, errors, redirects, limits."""

import socket

import httpx
import pytest

from clm_mcp_server.backend import CLMBackend, CLMBackendError, CLMRequestRejected
from conftest import API_KEY, settings_for
from fake_clm import FakeCLM

QUESTIONS = {"decision": {"type": "noul", "instructions": "Is it?"}}


def backend(url, **env):
    return CLMBackend(settings_for(url, **env))


def test_sends_the_key_and_reads_latency(fake):
    response = backend(fake.url).systemone("state", QUESTIONS)
    assert response.server_ms == 12.5 and response.round_trip_ms > 0
    sent = fake.requests[-1]
    assert sent["headers"]["Authorization"] == "Bearer %s" % API_KEY
    assert sent["body"] == {"state": "state", "questions": QUESTIONS, "model": "clm-latest", "temperature": 1.0}


def test_wrong_key_is_explained(fake):
    client = CLMBackend(settings_for(fake.url, CLM_API_KEY="wrong"))
    with pytest.raises(CLMBackendError, match="rejected the API key"):
        client.systemone("state", QUESTIONS)


@pytest.mark.parametrize("status, error, message", [
    (422, CLMRequestRejected, "rejected the request: forced failure"),
    (502, CLMBackendError, "encoder"),
    (500, CLMBackendError, "HTTP 500"),
])
def test_status_codes(fake, status, error, message):
    fake.behaviour = {"status": status}
    with pytest.raises(error, match=message):
        backend(fake.url).systemone("state", QUESTIONS)


def test_redirects_are_not_followed_so_the_key_stays_put(fake):
    with FakeCLM() as elsewhere:
        fake.behaviour = {"redirect_to": elsewhere.url}
        with pytest.raises(CLMBackendError, match="redirect"):
            backend(fake.url).systemone("state", QUESTIONS)
        assert elsewhere.requests == []


def test_proxy_variables_never_see_local_traffic(fake, monkeypatch):
    # A proxy in the environment must not receive the key or the decisions for a local server.
    with FakeCLM() as proxy:
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, proxy.url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        backend(fake.url).systemone("state", QUESTIONS)
        assert proxy.requests == []
    assert fake.requests[-1]["headers"]["Authorization"] == "Bearer %s" % API_KEY


def test_unreachable_server():
    with socket.socket() as sock:  # a port nothing listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(CLMBackendError, match="can't reach the CLM server"):
        backend("http://127.0.0.1:%d" % port).health()


def test_timeout(fake):
    fake.behaviour = {"delay": 2.0}
    with pytest.raises(CLMBackendError, match="did not answer within 1s"):
        backend(fake.url, CLM_TIMEOUT="1").systemone("state", QUESTIONS)


def test_oversized_response_declared(fake):
    fake.behaviour = {"oversize": True}
    with pytest.raises(CLMBackendError, match="limit is"):
        backend(fake.url).systemone("state", QUESTIONS)


def test_oversized_response_streamed():
    def stream():
        for _ in range(9):
            yield b"x" * (1024 * 1024)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=stream()))
    client = CLMBackend(settings_for("http://127.0.0.1:1"), transport=transport)
    with pytest.raises(CLMBackendError, match="exceeded"):
        client.health()


def test_non_json_body(fake):
    fake.behaviour = {"raw_body": b"<html>proxy error</html>"}
    with pytest.raises(CLMBackendError, match="not JSON"):
        backend(fake.url).systemone("state", QUESTIONS)


def test_long_error_details_are_cut():
    transport = httpx.MockTransport(lambda request: httpx.Response(500, json={"detail": "y" * 5000}))
    client = CLMBackend(settings_for("http://127.0.0.1:1"), transport=transport)
    with pytest.raises(CLMBackendError) as info:
        client.health()
    assert len(str(info.value)) < 400
