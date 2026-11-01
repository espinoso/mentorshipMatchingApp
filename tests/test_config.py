"""Model wiring: the model is fixed in code (no selector), and the Anthropic key reads env."""
import app


def test_model_is_fixed_in_code(monkeypatch):
    # No user-facing selector: _model_choice always returns the code default,
    # ignoring any stray model_choice left in session_state.
    monkeypatch.setitem(app.st.session_state, "model_choice", "claude-sonnet-5-5")
    assert app._model_choice(app.REASONING_MODEL) == app.REASONING_MODEL
    assert app._model_choice("claude-sonnet-5-5") == "claude-sonnet-5-5"


def test_anthropic_key_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert app._anthropic_key() == "sk-test"
