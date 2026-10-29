"""Claude model config, selector list, pricing, and the model_choice wiring."""
import pytest

import app
from info import API_PRICING, CLAUDE_MODELS, DEFAULT_MODEL, MODEL_CONFIGS
from utils import estimate_api_cost, fetch_available_models, get_model_config


def test_default_model_is_claude_and_selectable():
    assert DEFAULT_MODEL == "claude-opus-5-5"
    assert DEFAULT_MODEL in CLAUDE_MODELS
    assert CLAUDE_MODELS == ["claude-opus-5-5", "claude-sonnet-5-5"]


def test_get_model_config_opus():
    cfg = get_model_config("claude-opus-5-5")
    assert cfg["family"] == "opus"
    assert cfg["max_output_tokens"] == 128000


def test_get_model_config_unknown_falls_back_to_default():
    assert get_model_config("not-a-model") == MODEL_CONFIGS[DEFAULT_MODEL]


def test_available_models_are_static_claude_list():
    models = fetch_available_models()
    assert list(models) == CLAUDE_MODELS
    assert all({"name", "quality", "speed", "cost"} <= set(v) for v in models.values())


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
def test_cost_uses_claude_pricing(model):
    p = API_PRICING[model]
    assert estimate_api_cost(1_000_000, 1_000_000, model) == pytest.approx(p["prompt"] + p["completion"])


def test_opus_pricing_values_and_unknown_model_prices_as_default():
    assert estimate_api_cost(1_000_000, 1_000_000, "claude-opus-5-5") == pytest.approx(24.0)
    assert estimate_api_cost(1_000_000, 0, "claude-sonnet-5-5") == pytest.approx(2.0)
    assert estimate_api_cost(1_000_000, 0, "not-a-model") == pytest.approx(4.0)


def test_model_choice_drives_matrix_and_reasoning_model(monkeypatch):
    monkeypatch.setitem(app.st.session_state, "model_choice", "claude-sonnet-5-5")
    assert app._model_choice(app.MATRIX_MODEL) == "claude-sonnet-5-5"
    assert app._model_choice(app.REASONING_MODEL) == "claude-sonnet-5-5"


def test_anthropic_key_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert app._anthropic_key() == "sk-test"
