"""What happens inside a decision: one non-autoregressive forward pass, temperature-calibrated output."""

import numpy as np
import pytest
import torch
from laya.common import clamp_temperature
from transformers.generation.utils import GenerationMixin

from conftest import call

pytestmark = [pytest.mark.model, pytest.mark.anyio]

CASES = [
    dict(
        state="Hi, we were billed twice for March. Please refund the duplicate or we will cancel.",
        question_type="choice",
        options={"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"},
        instructions="Which department should handle this?",
    ),
    dict(
        state="The dashboard is a little slow to load in the mornings.",
        question_type="score",
        options=["not urgent", "soon", "blocking"],
        instructions="How urgent is this?",
    ),
    dict(state="The checkout page returns a 500 error.", question_type="bool", instructions="Is this a bug report?"),
]
IDS = [case["question_type"] for case in CASES]


@pytest.fixture
def forward_calls(engine):
    """Record every forward pass of the English checkpoint: its logits, one entry per call."""
    calls = []

    def hook(_module, _inputs, output):
        logits, _act = output
        calls.append(logits.detach().float().cpu())

    handle = engine.agent("english").model.register_forward_hook(hook)
    yield calls
    handle.remove()


@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_one_forward_pass_and_no_generation(client, forward_calls, monkeypatch, case):
    def no_generate(*_args, **_kwargs):
        raise AssertionError("evaluate_laya_decision must not generate text")

    monkeypatch.setattr(GenerationMixin, "generate", no_generate)
    result = await call(client, **case)

    assert len(forward_calls) == 1, "a decision is exactly one forward pass"
    assert result["generated_tokens"] == 0
    assert result["answer"] in result["probabilities"]


def test_model_is_an_encoder_with_no_generation_head(engine):
    model = engine.agent("english").model
    assert not isinstance(model, GenerationMixin)
    assert not isinstance(model.encoder, GenerationMixin)
    assert not hasattr(model, "generate") and not hasattr(model, "lm_head")
    # The English checkpoint's backbone is ModernBERT-large: a bidirectional encoder, not a decoder.
    assert type(model.encoder).__name__ == "ModernBertModel"


@pytest.mark.parametrize("case", CASES, ids=IDS)
async def test_probabilities_are_temperature_calibrated_logits(client, engine, forward_calls, case):
    """The returned distribution is softmax(logits / T), T being the checkpoint's fitted temperature."""
    result = await call(client, **case)
    labels = list(result["probabilities"])
    k = len(labels)
    logits = forward_calls[0][0, :k].double()

    agent = engine.agent("english")
    bucket = result["calibration"]["bucket"]
    fitted = clamp_temperature(agent.cfg["temperature_by_options"][bucket])
    temperature = result["calibration"]["temperature"]
    assert temperature == pytest.approx(fitted, abs=1e-4)
    assert temperature != pytest.approx(1.0), "calibration must actually rescale the logits"

    calibrated = torch.softmax(logits / fitted, dim=0).numpy()
    raw = torch.softmax(logits, dim=0).numpy()
    returned = [result["probabilities"][label] for label in labels]
    if result["question_type"] == "bool":
        # Laya orders a bool's options [false, true]; the tool reports them [true, false].
        calibrated, raw = calibrated[::-1], raw[::-1]
    np.testing.assert_allclose(returned, calibrated, atol=1e-4)
    # Every fitted temperature is > 1: calibration softens the raw, over-confident distribution.
    assert max(returned) < max(raw)
