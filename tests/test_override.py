"""Guarded-swap override: _apply_override reassigns a mentee and keeps all state consistent."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import ui_processing as u


class _FakeState(dict):
    """Dict with attribute access, like streamlit's session_state."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_apply_override_moves_assignment_updates_match_and_clears_cache(monkeypatch):
    # mentors M1,M2 x mentee E1; E1 is currently on M1 (score 50), M2 scores 88.
    matrix = pd.DataFrame({"E1": {"M1": 50, "M2": 88}})
    state = _FakeState(
        matrix_df=matrix,
        assignments=[("E1", "M1", 50)],
        matches=[{"mentee_id": "E1", "matches": [
            {"mentor_id": "M1", "match_percentage": 50, "match_quality": "x", "reasoning": "old"}]}],
        reasoning_cache={("E1", "M1"): "cached", ("E2", "M9"): "other"},
        mentees_df="ee", mentors_df="mo",
    )
    monkeypatch.setattr(u, "st", type("S", (), {"session_state": state}))
    monkeypatch.setattr(u, "validate_and_flag_matches", lambda *a, **k: ({"E1": []}, {"mean": 88}))

    u._apply_override("E1", "M2")

    assert state["assignments"] == [("E1", "M2", 88)]        # assignment moved to M2 at its matrix score
    top = state["matches"][0]["matches"][0]
    assert top["mentor_id"] == "M2" and top["match_percentage"] == 88
    assert top["reasoning"] is None                          # now-stale reasoning dropped
    assert ("E1", "M1") not in state["reasoning_cache"]      # this mentee's cached reasoning gone
    assert ("E2", "M9") in state["reasoning_cache"]          # another mentee's cache untouched
    assert state["flagged_matches"] == {"E1": []}            # flags/stats recomputed from validation
    assert state["score_stats"] == {"mean": 88}
