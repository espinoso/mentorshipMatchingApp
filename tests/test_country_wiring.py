"""The sidebar country rule (default-on Spain↔Spain) must be enforced in the live Generate flow,
not just available as a tested helper. Regression guard: the toggle was inert on the Jev path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from streamlit.testing.v1 import AppTest

import app
from info import MENTEE_COLUMNS, MENTOR_COLUMNS

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_country_rule_blocks_same_country_pair_in_live_flow(monkeypatch):
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
    # The tempting pick: Spanish mentor M1 × Spanish mentee E1 scores highest.
    matrix = pd.DataFrame({"E1": {"M1": 95, "M2": 20}, "E2": {"M1": 10, "M2": 90}})
    monkeypatch.setattr(app, "generate_matrix_jev", lambda ee, mo: matrix)

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.session_state["api_key"] = "x"
    at.session_state["mentees_df"] = mentees
    at.session_state["mentors_df"] = mentors
    at.session_state["country_restrictions"] = ["spain"]
    at.session_state["start_processing"] = True
    at.run()

    # The same-country cell is zeroed before assignment, so M1 is never matched to E1.
    assert at.session_state["matrix_df"].loc["M1", "E1"] == 0
    assert ("E1", "M1") not in [(e, m) for e, m, _ in at.session_state["assignments"]]
