from typing import ClassVar

import anthropic
import pytest

from llm_client import ClaudeClient, LLMError, parse_json

JSON_INSTRUCTION = "Respond with a single valid JSON object and nothing else."


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ThinkingBlock:
    type = "thinking"
    text = "ignored"


class _FinalMessage:
    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.content = [_ThinkingBlock(), _Block('{"ok": '), _Block("true}")]


class _Stream:
    def __init__(self, stop_reason="end_turn"):
        self._stop_reason = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return _FinalMessage(self._stop_reason)


class _FakeAnthropic:
    last_kwargs: ClassVar[dict] = {}
    stop_reason = "end_turn"

    def __init__(self, api_key=None):
        self.messages = self

    def stream(self, **kwargs):
        _FakeAnthropic.last_kwargs = kwargs
        return _Stream(self.stop_reason)


def test_complete_json_streams_without_temperature(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    out = ClaudeClient("claude-opus-5-5", "k").complete_json("sys", "usr", max_tokens=1234)
    kw = _FakeAnthropic.last_kwargs
    assert out == '{"ok": true}'
    assert "temperature" not in kw
    assert "top_p" not in kw
    assert kw["model"] == "claude-opus-5-5"
    assert kw["max_tokens"] == 1234
    assert kw["system"].startswith("sys")
    assert kw["system"].endswith(JSON_INSTRUCTION)
    assert kw["messages"] == [{"role": "user", "content": "usr"}]


def test_truncated_response_raises_llmerror(monkeypatch):
    class Truncated(_FakeAnthropic):
        stop_reason = "max_tokens"

    monkeypatch.setattr(anthropic, "Anthropic", Truncated)
    with pytest.raises(LLMError, match="cut off"):
        ClaudeClient("m", "k").complete_json("sys", "usr", max_tokens=10)


def test_empty_key_raises_llmerror_naming_env_var():
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        ClaudeClient("m", "")


def test_request_failure_wrapped(monkeypatch):
    class Boom(_FakeAnthropic):
        def stream(self, **kwargs):
            raise RuntimeError("400 bad model")

    monkeypatch.setattr(anthropic, "Anthropic", Boom)
    with pytest.raises(LLMError, match="400 bad model"):
        ClaudeClient("m", "k").complete_json("s", "u", max_tokens=10)


def test_cache_system_sends_block_with_cache_control(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    ClaudeClient("m", "k").complete_json("sys", "usr", max_tokens=10, cache_system=True)
    system = _FakeAnthropic.last_kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["type"] == "text"
    assert system[0]["text"].startswith("sys")
    assert system[0]["text"].endswith(JSON_INSTRUCTION)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_no_cache_control_by_default(monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    ClaudeClient("m", "k").complete_json("sys", "usr", max_tokens=10)
    assert isinstance(_FakeAnthropic.last_kwargs["system"], str)


def test_parse_json_clean():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_fenced():
    assert parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_parse_json_surrounding_prose():
    assert parse_json('Here you go: {"a": {"b": 2}} Hope that helps!') == {"a": {"b": 2}}


@pytest.mark.parametrize("bad", ["no json here", "{not: valid}", ""])
def test_parse_json_garbage_raises(bad):
    with pytest.raises(LLMError):
        parse_json(bad)
