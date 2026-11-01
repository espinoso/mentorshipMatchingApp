"""Tests for the Claude matrix-scoring path (app._score_mentor_batch), the benchmark baseline."""

import json
from typing import ClassVar

import pandas as pd
import pytest

import app
from llm_client import LLMError
from matching_core import parse_matrix_response
from utils import estimate_output_tokens


def _mentees(ids):
    return pd.DataFrame({"Mentee ID": ids, "Field": ["Bio"] * len(ids)})


def _mentors(ids):
    return pd.DataFrame({"Mentor ID": ids, "Field": ["Bio"] * len(ids)})


class _FakeClient:
    """Records the last call and returns a canned response (or raises)."""

    calls: ClassVar[list] = []

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def __call__(self, model, api_key):  # app does ClaudeClient(MATRIX_MODEL, key)
        self.model = model
        self.api_key = api_key
        return self

    def complete_json(self, system, user, *, max_tokens, cache_system=False):
        _FakeClient.calls.append(
            {"system": system, "user": user, "max_tokens": max_tokens, "cache_system": cache_system}
        )
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def _reset_calls():
    _FakeClient.calls = []


def test_wellformed_response_builds_expected_matrix(monkeypatch):
    """A well-formed matrix response feeds straight into parse_matrix_response."""
    monkeypatch.setattr(app, "get_prompt_for_api", lambda *a, **k: "RUBRIC")
    payload = json.dumps(
        {
            "matrix": [
                {"mentor_id": "M1", "scores": [
                    {"mentee_id": "E1", "percentage": 90},
                    {"mentee_id": "E2", "percentage": 40},
                ]},
            ]
        }
    )
    monkeypatch.setattr(app, "ClaudeClient", _FakeClient(response=payload))

    text = app._score_mentor_batch(_mentees(["E1", "E2"]), _mentors(["M1"]))
    report = parse_matrix_response(text, ["M1"], ["E1", "E2"])

    assert report.error is None
    assert report.df is not None
    assert report.df.loc["M1", "E1"] == 90
    assert report.df.loc["M1", "E2"] == 40
    # No pair was omitted, so nothing was filled with the poor-match default.
    assert report.missing_mentees_filled == []


def test_budget_and_cache_and_invariant_system(monkeypatch):
    """max_tokens is budgeted & capped, caching is on, and the system prompt is batch-invariant."""
    monkeypatch.setattr(app, "get_prompt_for_api", lambda *a, **k: "RUBRIC")
    monkeypatch.setattr(app, "ClaudeClient", _FakeClient(response='{"matrix": []}'))

    mentees = _mentees(["E1", "E2", "E3"])
    app._score_mentor_batch(mentees, _mentors(["M1", "M2"]))
    app._score_mentor_batch(mentees, _mentors(["M3"]))  # different batch, same mentees

    first, second = _FakeClient.calls
    assert first["cache_system"] is True
    # The cached system prompt must be byte-identical across batches for cache hits.
    assert first["system"] == second["system"]
    # ...and the mentor batch (user) must differ.
    assert first["user"] != second["user"]
    # Budget = estimate_output_tokens * 1.5 PLUS a thinking allowance (Opus spends thinking
    # tokens before the JSON), floored at that allowance and capped under the 128K output cap.
    expected = int(estimate_output_tokens(2, 3) * 1.5) + app.THINKING_HEADROOM
    assert first["max_tokens"] == max(app.THINKING_HEADROOM, min(expected, app.MATRIX_MAX_OUTPUT_TOKENS))
    assert first["max_tokens"] <= app.MATRIX_MAX_OUTPUT_TOKENS
    # Regression guard: a small matrix must still leave room for thinking (the bug: budget was 4096).
    assert first["max_tokens"] >= app.THINKING_HEADROOM


def test_cutoff_raises_not_filled(monkeypatch):
    """A cut-off (LLMError) propagates; it is NOT swallowed into a 30-filled matrix."""
    monkeypatch.setattr(app, "get_prompt_for_api", lambda *a, **k: "RUBRIC")
    monkeypatch.setattr(app, "ClaudeClient", _FakeClient(error=LLMError("cut off")))

    with pytest.raises(LLMError):
        app._score_mentor_batch(_mentees(["E1"]), _mentors(["M1"]))
