"""Where the server may connect, and how it reads its configuration."""

import pytest

from clm_mcp_server.backend import DEFAULT_BASE_URL, CLMConfigError, CLMSettings


def test_defaults():
    s = CLMSettings.from_env({})
    assert (s.base_url, s.api_key, s.model, s.timeout_s, s.temperature) == (DEFAULT_BASE_URL, None, "clm-latest", 120.0, 1.0)
    assert not s.uses_tls


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8700", "http://localhost:8700/", "http://[::1]:8700", "http://127.8.9.10:1",
    "https://gpu.example.com", "https://gpu.example.com/clm/",
])
def test_allowed_urls(url):
    assert CLMSettings.from_env({"CLM_BASE_URL": url}).base_url == url.rstrip("/")


@pytest.mark.parametrize("url, message", [
    ("http://gpu.example.com:8700", "plain http"),
    ("http://10.0.0.5:8700", "plain http"),
    ("http://localhost.evil.com", "plain http"),
    ("ftp://127.0.0.1", "http:// or https://"),
    ("127.0.0.1:8700", "http:// or https://"),
    ("https://user:pw@gpu.example.com", "credentials"),
    ("https://gpu.example.com/?key=1", "query string"),
])
def test_refused_urls(url, message):
    with pytest.raises(CLMConfigError, match=message):
        CLMSettings.from_env({"CLM_BASE_URL": url})


def test_insecure_http_needs_an_explicit_opt_in():
    s = CLMSettings.from_env({"CLM_BASE_URL": "http://100.64.0.7:8700", "CLM_ALLOW_INSECURE_HTTP": "1"})
    assert s.base_url == "http://100.64.0.7:8700" and s.allow_insecure_http


def test_api_key_handling():
    assert CLMSettings.from_env({"CLM_API_KEY": "  "}).api_key is None  # ${CLM_API_KEY:-} expands to ""
    assert CLMSettings.from_env({"CLM_API_KEY": " k "}).api_key == "k"
    with pytest.raises(CLMConfigError, match="single line"):
        CLMSettings.from_env({"CLM_API_KEY": "k\r\nX-Evil: 1"})


@pytest.mark.parametrize("env, message", [
    ({"CLM_MODEL": "../etc"}, "CLM_MODEL"),
    ({"CLM_TIMEOUT": "fast"}, "CLM_TIMEOUT"),
    ({"CLM_TIMEOUT": "0"}, "CLM_TIMEOUT"),
    ({"CLM_TEMPERATURE": "nan"}, "CLM_TEMPERATURE"),
    ({"CLM_TEMPERATURE": "500"}, "CLM_TEMPERATURE"),
])
def test_bad_values(env, message):
    with pytest.raises(CLMConfigError, match=message):
        CLMSettings.from_env(env)
