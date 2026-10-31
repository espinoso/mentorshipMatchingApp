"""Tests for the Claude matrix-scoring path (app._score_mentor_batch + generate_matrix_in_batches)
and the output-token batch sizing (utils.calculate_optimal_batch_size)."""

import json
from typing import ClassVar
from unittest.mock import MagicMock

import pandas as pd
import pytest

import app
from llm_client import LLMError
from matching_core import parse_matrix_response
from utils import MATRIX_OUTPUT_CAP, calculate_optimal_batch_size, estimate_output_tokens


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


def test_driver_marks_batch_error_on_cutoff(monkeypatch):
    """generate_matrix_in_batches reports an errored batch instead of 30-filling it."""
    monkeypatch.setattr(app, "get_prompt_for_api", lambda *a, **k: "RUBRIC")
    monkeypatch.setattr(app, "ClaudeClient", _FakeClient(error=LLMError("cut off")))
    monkeypatch.setattr(app, "st", MagicMock())

    import progress_ui

    update = MagicMock()
    monkeypatch.setattr(progress_ui, "initialize_progress_tracking", MagicMock())
    monkeypatch.setattr(progress_ui, "update_batch_progress", update)
    monkeypatch.setattr(progress_ui, "render_batch_card", MagicMock())
    monkeypatch.setattr(progress_ui, "render_progress_overview", MagicMock())

    result = app.generate_matrix_in_batches(_mentees(["E1"]), _mentors(["M1"]), batch_size=1)

    # No batch produced a matrix, so the driver returns None (not a 30-filled DataFrame).
    assert result is None
    statuses = [c.args[1] for c in update.call_args_list if len(c.args) >= 2]
    assert "error" in statuses
    assert "complete" not in statuses


def test_driver_assembles_all_batches_in_parallel(monkeypatch):
    """generate_matrix_in_batches scores batches concurrently and assembles the full matrix,
    regardless of the order futures complete."""
    fake_st = MagicMock()
    # make st.session_state.get honor its default (country_restrictions -> ["spain"])
    fake_st.session_state.get = lambda key, default=None: default
    monkeypatch.setattr(app, "st", fake_st)

    import progress_ui
    for fn in ("initialize_progress_tracking", "update_batch_progress",
               "render_batch_card", "render_progress_overview"):
        monkeypatch.setattr(progress_ui, fn, MagicMock())

    def fake_score(mentees_df, mentor_batch_df, api_key=None, model=None):
        rows = [{"mentor_id": str(m), "scores": [{"mentee_id": "E1", "percentage": 70}]}
                for m in mentor_batch_df["Mentor ID"]]
        return json.dumps({"matrix": rows})

    monkeypatch.setattr(app, "_score_mentor_batch", fake_score)

    matrix = app.generate_matrix_in_batches(_mentees(["E1"]),
                                            _mentors(["M1", "M2", "M3", "M4", "M5"]),
                                            batch_size=2)  # 3 batches, scored in parallel

    assert matrix is not None
    assert sorted(matrix.index) == ["M1", "M2", "M3", "M4", "M5"]  # every batch landed
    assert (matrix["E1"] == 70).all()


def test_batch_size_output_never_exceeds_cap():
    """calculate_optimal_batch_size never returns a batch whose estimated output exceeds the cap."""
    for num_mentees in (1, 10, 62, 200, 500, 2000):
        for total_mentors in (1, 5, 15, 60):
            bs = calculate_optimal_batch_size(total_mentors, num_mentees)
            assert 1 <= bs <= min(30, total_mentors) or bs == 1
            assert int(estimate_output_tokens(bs, num_mentees) * 1.5) <= MATRIX_OUTPUT_CAP
