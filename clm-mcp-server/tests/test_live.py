"""Against a real clm-serve (GPU vLLM, or the llama.cpp CPU encoder in scripts/).

    CLM_TEST_URL=http://127.0.0.1:8700 CLM_TEST_API_KEY=... uv run --frozen pytest -m live

The expectations are the CLM-8B model card's own examples plus phrasing checks this repo relies
on in its tool descriptions. Skipped when CLM_TEST_URL is unset.
"""

import os

import pytest

from clm_mcp_server.backend import CLMSettings
from clm_mcp_server.server import CLMService

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("CLM_TEST_URL"), reason="set CLM_TEST_URL to a running clm-serve"),
]


@pytest.fixture(scope="module")
def live():
    service = CLMService(CLMSettings.from_env({
        "CLM_BASE_URL": os.environ["CLM_TEST_URL"],
        "CLM_API_KEY": os.environ.get("CLM_TEST_API_KEY", ""),
        "CLM_TIMEOUT": "600",
    }))
    yield service
    service.close()


def test_backend_is_real(live):
    status = live.status()
    assert status["state"] == "ready", status
    assert status["mock_encoder"] is False


def test_model_card_ranking(live):
    result = live.rank("What causes tides on Earth?",
                       ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."])
    assert result["ranking"][0]["index"] == 0
    assert result["ranking"][0]["probability"] > 0.9  # model card: 0.993


def test_model_card_choice(live):
    result = live.evaluate("Customer: my invoice was charged twice and nobody answers the phone!", "choice",
                           "Which team should handle this?",
                           {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"})
    assert result["answer"] == "billing"  # model card: 0.939


def test_descriptive_rubric_levels(live):
    levels = ["Not urgent: it can wait a few days.", "Soon: it should be handled today.",
              "Critical: an outage that needs action right now."]
    outage = live.evaluate("Production has been down for every customer since 10am!", "score",
                           "How urgent is this?", levels)
    calm = live.evaluate("Could you update my invoice address when you have a moment? No rush.", "score",
                         "How urgent is this?", levels)
    assert outage["answer"] == levels[2]
    assert calm["answer"] == levels[0]


def test_plain_descriptions_route_tools(live):
    tools = ["Find and book flights.", "Book a table at a restaurant.", "Get the weather forecast for a city.",
             "Send an email to a contact.", "Add an event to the calendar."]
    result = live.rank("Book a table for two at an Italian place tonight", tools)
    assert result["ranking"][0]["index"] == 1
