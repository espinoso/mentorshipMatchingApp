"""The sidebar toggle maps to countries=["spain"] (on) or [] (off) on the real function."""

import pandas as pd

from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from matching import apply_country_restrictions
from matching_core import SAME_COUNTRY_PENALTY


def _frames():
    mentors = pd.DataFrame(
        {MENTOR_COLUMNS["id"]: ["1", "2"], MENTOR_COLUMNS["country"]: ["Spain", "France"]}
    )
    mentees = pd.DataFrame(
        {MENTEE_COLUMNS["id"]: ["101", "102"], MENTEE_COLUMNS["country"]: ["Spain", "Spain"]}
    )
    matrix = pd.DataFrame(50.0, index=["1", "2"], columns=["101", "102"])
    return matrix, mentees, mentors


def test_toggle_on_penalizes_spain_spain_cells():
    matrix, mentees, mentors = _frames()
    out, filtered = apply_country_restrictions(matrix, mentees, mentors, countries=["spain"])
    assert out.loc["1", "101"] == max(0.0, 50.0 - SAME_COUNTRY_PENALTY)
    assert out.loc["2", "101"] == 50.0  # non-Spanish mentor untouched
    assert len(filtered) == 2


def test_toggle_off_leaves_matrix_unchanged():
    matrix, mentees, mentors = _frames()
    out, filtered = apply_country_restrictions(matrix, mentees, mentors, countries=[])
    assert out.equals(matrix)
    assert filtered == []
