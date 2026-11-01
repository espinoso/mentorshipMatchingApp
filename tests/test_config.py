"""Claude model config, selector list, and the model_choice wiring."""
import app
from info import CLAUDE_MODELS, DEFAULT_MODEL, MODEL_CONFIGS
from utils import fetch_available_models, get_model_config


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


def test_model_is_fixed_in_code(monkeypatch):
    # No user-facing selector anymore: _model_choice always returns the code default,
    # ignoring any stray model_choice left in session_state.
    monkeypatch.setitem(app.st.session_state, "model_choice", "claude-sonnet-5-5")
    assert app._model_choice(app.MATRIX_MODEL) == app.MATRIX_MODEL
    assert app._model_choice(app.REASONING_MODEL) == app.REASONING_MODEL


def test_anthropic_key_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert app._anthropic_key() == "sk-test"
