import pandas as pd

from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from matching_core import apply_country_restrictions

MC, EC = MENTOR_COLUMNS["country"], MENTEE_COLUMNS["country"]


def _frames(mentor_countries, mentee_countries):
    mentors = pd.DataFrame(
        {
            MENTOR_COLUMNS["id"]: [str(i + 1) for i in range(len(mentor_countries))],
            MC: mentor_countries,
        }
    )
    mentees = pd.DataFrame(
        {
            MENTEE_COLUMNS["id"]: [str(101 + i) for i in range(len(mentee_countries))],
            EC: mentee_countries,
        }
    )
    matrix = pd.DataFrame(
        50.0, index=mentors[MENTOR_COLUMNS["id"]], columns=mentees[MENTEE_COLUMNS["id"]]
    )
    return matrix, mentees, mentors


def test_default_is_no_restriction():
    matrix, mentees, mentors = _frames(["Spain"], ["Spain"])
    out, filtered = apply_country_restrictions(matrix, mentees, mentors)
    assert out.equals(matrix)
    assert filtered == []


def test_avoid_same_country_zeroes_equal_non_empty():
    matrix, mentees, mentors = _frames(["Portugal", " SPAIN ", None], ["portugal", "Spain", None])
    out, filtered = apply_country_restrictions(matrix, mentees, mentors, avoid_same_country=True)
    assert out.loc["1", "101"] == 0 and out.loc["2", "102"] == 0
    assert out.loc["1", "102"] == 50 and out.loc["3", "103"] == 50  # empty != empty
    assert {(f["mentor_id"], f["mentee_id"]) for f in filtered} == {("1", "101"), ("2", "102")}
    assert all(f["original_score"] == 50.0 and f["reason"] for f in filtered)


def test_countries_spain_only_leaves_portugal():
    matrix, mentees, mentors = _frames(["Spain", "Portugal"], ["Spain", "Portugal"])
    out, filtered = apply_country_restrictions(matrix, mentees, mentors, countries=["spain"])
    assert out.loc["1", "101"] == 0
    assert out.loc["2", "102"] == 50
    assert [(f["mentor_id"], f["mentee_id"]) for f in filtered] == [("1", "101")]


def test_countries_takes_precedence_over_avoid_same_country():
    matrix, mentees, mentors = _frames(["Spain", "Portugal"], ["Spain", "Portugal"])
    out, _ = apply_country_restrictions(
        matrix, mentees, mentors, avoid_same_country=True, countries=["spain"]
    )
    assert out.loc["1", "101"] == 0 and out.loc["2", "102"] == 50


def test_missing_column_returns_unchanged_with_marker():
    matrix, mentees, mentors = _frames(["Spain"], ["Spain"])
    out, filtered = apply_country_restrictions(
        matrix, mentees, mentors.drop(columns=[MC]), avoid_same_country=True
    )
    assert out.equals(matrix)
    assert filtered == [{"skipped": "mentor country column missing"}]
    _, filtered = apply_country_restrictions(
        matrix, mentees.drop(columns=[EC]), mentors, avoid_same_country=True
    )
    assert filtered == [{"skipped": "mentee country column missing"}]


def test_zero_score_cells_zeroed_but_not_reported():
    matrix, mentees, mentors = _frames(["Spain"], ["Spain"])
    matrix.loc["1", "101"] = 0.0
    out, filtered = apply_country_restrictions(matrix, mentees, mentors, avoid_same_country=True)
    assert out.loc["1", "101"] == 0 and filtered == []


def test_input_matrix_not_mutated():
    matrix, mentees, mentors = _frames(["Spain"], ["Spain"])
    apply_country_restrictions(matrix, mentees, mentors, avoid_same_country=True)
    assert matrix.loc["1", "101"] == 50


def test_empty_countries_list_does_not_disable_avoid_flag():
    # Hardening: countries=[] must NOT silently turn the rule off when avoiding same-country.
    matrix, mentees, mentors = _frames(["Spain"], ["Spain"])
    out, filtered = apply_country_restrictions(
        matrix, mentees, mentors, avoid_same_country=True, countries=[]
    )
    assert out.loc["1", "101"] == 0
    assert [(f["mentor_id"], f["mentee_id"]) for f in filtered] == [("1", "101")]


def test_no_restriction_returns_empty_before_column_check():
    # Hardening: the no-restriction default returns [] even when a country column is missing,
    # never a skip-marker.
    matrix, mentees, mentors = _frames(["Spain"], ["Spain"])
    out, filtered = apply_country_restrictions(matrix, mentees, mentors.drop(columns=[MC]))
    assert out.equals(matrix)
    assert filtered == []
