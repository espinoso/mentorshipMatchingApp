"""Tests for app.get_reasoning_for_assignments (Claude call, graceful {} on any failure)."""

import json

import pandas as pd
import pytest

import app
from llm_client import LLMError


class _Fake:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def __call__(self, model, api_key):
        self.model = model
        return self

    def complete_json(self, system, user, *, max_tokens, cache_system=False):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def run(monkeypatch):
    def _run(fake):
        monkeypatch.setattr(app, "ClaudeClient", fake)
        monkeypatch.setattr(app, "_anthropic_key", lambda: "k")
        mentees = pd.DataFrame({"Mentee ID": [1], "Field": ["Bio"]})
        mentors = pd.DataFrame({"Mentor ID": [2], "Field": ["Bio"]})
        return app.get_reasoning_for_assignments([("1", "2", 90)], mentees, mentors)

    return _run


def test_valid_response_builds_keyed_dict(run):
    fake = _Fake(json.dumps({"reasonings": [{"mentee_id": "1", "mentor_id": "2", "reasoning": "Both biology."}]}))
    assert run(fake) == {("1", "2"): "Both biology."}
    assert fake.model == app.REASONING_MODEL
    # Budget scales with the assignment count and leaves room for Opus thinking (regression:
    # a flat 8000 truncated large-cohort reasoning once thinking is counted).
    assert fake.calls[0]["max_tokens"] >= app.THINKING_HEADROOM
    assert "1 → 2" in fake.calls[0]["user"]


@pytest.mark.parametrize("response", ["", "not json", "[1, 2]", '"str"', '{"reasonings": "x"}', '{"reasonings": [1, null]}', "{}"])
def test_malformed_response_returns_empty(run, response):
    assert run(_Fake(response)) == {}


def test_llm_error_returns_empty(run):
    assert run(_Fake(error=LLMError("boom"))) == {}


def test_reasoning_for_pair_returns_text_and_runs_quiet(monkeypatch):
    seen = {}

    def fake(assignments, ee, mo, *, quiet=False):
        seen["quiet"], seen["assignments"] = quiet, assignments
        return {("1", "2"): "Because the fields align."}

    monkeypatch.setattr(app, "get_reasoning_for_assignments", fake)
    assert app.reasoning_for_pair("1", "2", 90, None, None) == "Because the fields align."
    assert seen["quiet"] is True  # single-pair lookups don't emit toasts
    assert seen["assignments"] == [("1", "2", 90)]


def test_reasoning_for_pair_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(app, "get_reasoning_for_assignments", lambda *a, **k: {})
    out = app.reasoning_for_pair("1", "2", 73, None, None)
    assert "73%" in out
