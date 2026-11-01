"""The sidebar country rule (default-on Spain↔Spain) is enforced in the live Generate flow as a
SOFT priority: it deprioritizes same-country pairs (avoids them when a decent cross-country option
exists) but still allows a strong same-country match, and never leaves a mentee unmatched over it.
Regression guard: the toggle was once inert on the Jev path, and once a hard zero."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from streamlit.testing.v1 import AppTest

import app
from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from matching_core import SAME_COUNTRY_PENALTY

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(monkeypatch, matrix):
    """Drive the live Generate flow with a stubbed scorer; return the AppTest after the run.
    E1/M1 are Spanish (the same-country pair); E2 Iran, M2 Sri Lanka."""
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    monkeypatch.setenv("ALLOW_NO_AUTH", "1")  # dev opt-out past the fail-closed auth gate
    mentees = pd.DataFrame({
        MENTEE_COLUMNS['id']: ["E1", "E2"],
        MENTEE_COLUMNS['field']: ["Bio", "CS"],
        MENTEE_COLUMNS['country']: ["Spain", "Iran"],
    })
    mentors = pd.DataFrame({
        MENTOR_COLUMNS['id']: ["M1", "M2"],
        MENTOR_COLUMNS['field']: ["Bio", "CS"],
        MENTOR_COLUMNS['country']: ["Spain", "Sri Lanka"],
    })
    monkeypatch.setattr(app, "generate_matrix_jev", lambda ee, mo: matrix)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.session_state["api_key"] = "x"
    at.session_state["mentees_df"] = mentees
    at.session_state["mentors_df"] = mentors
    at.session_state["country_restrictions"] = ["spain"]
    at.session_state["start_processing"] = True
    at.run()
    return at


def test_same_country_is_penalized_and_avoided_when_cross_country_is_competitive(monkeypatch):
    # M1×E1 (Spain↔Spain) is the top raw score, but E1 also has a decent cross-country option (M2=70).
    matrix = pd.DataFrame({"E1": {"M1": 95, "M2": 70}, "E2": {"M1": 10, "M2": 90}})
    at = _run(monkeypatch, matrix)
    # The cell is penalized (not zeroed), and the soft rule steered E1 away from the Spanish mentor.
    assert at.session_state["matrix_df"].loc["M1", "E1"] == max(0.0, 95 - SAME_COUNTRY_PENALTY)
    assert ("E1", "M1") not in [(e, m) for e, m, _ in at.session_state["assignments"]]


def test_strong_same_country_still_wins_when_cross_country_is_weak(monkeypatch):
    # Same Spanish pair, but now E1's only cross-country option (M2=20) is far worse than the
    # penalized same-country score (95-40=55) — so the soft rule still allows the same-country match.
    matrix = pd.DataFrame({"E1": {"M1": 95, "M2": 20}, "E2": {"M1": 10, "M2": 90}})
    at = _run(monkeypatch, matrix)
    assert ("E1", "M1") in [(e, m) for e, m, _ in at.session_state["assignments"]]
