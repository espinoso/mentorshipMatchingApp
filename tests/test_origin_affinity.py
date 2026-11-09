"""Origin affinity: a small positive when the mentor's country of ORIGIN matches the student's
affiliation country — and it never overrides the shared-AFFILIATION penalty."""

import pandas as pd

from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from matching_core import (
    ORIGIN_AFFINITY_BONUS as BONUS,
)
from matching_core import (
    SAME_COUNTRY_PENALTY,
    apply_country_restrictions,
    apply_origin_affinity,
)

MID, EID = MENTOR_COLUMNS["id"], MENTEE_COLUMNS["id"]
M_AFF, M_ORIG = MENTOR_COLUMNS["country"], MENTOR_COLUMNS["country_of_origin"]
E_AFF = MENTEE_COLUMNS["country"]


def _frames(mentors, mentees, start=50.0):
    """mentors/mentees: list of (affiliation, origin) / affiliation."""
    mo = pd.DataFrame({MID: [str(10 + i) for i in range(len(mentors))],
                       M_AFF: [a for a, _ in mentors], M_ORIG: [o for _, o in mentors]})
    ee = pd.DataFrame({EID: [str(100 + i) for i in range(len(mentees))], E_AFF: mentees})
    matrix = pd.DataFrame(start, index=mo[MID], columns=ee[EID])
    return matrix, ee, mo


def test_origin_match_boosts_only_matching_cells():
    matrix, ee, mo = _frames([("USA", "Spain")], ["Spain", "Portugal"])
    out, boosted = apply_origin_affinity(matrix, ee, mo)
    assert out.loc["10", "100"] == 50.0 + BONUS   # mentor origin Spain ~ Spanish student
    assert out.loc["10", "101"] == 50.0           # Portuguese student untouched
    assert boosted == 1


def test_boost_clips_at_100():
    matrix, ee, mo = _frames([("USA", "Spain")], ["Spain"], start=95.0)
    out, _ = apply_origin_affinity(matrix, ee, mo)
    assert out.loc["10", "100"] == 100.0


def test_no_origin_column_is_noop():
    matrix, ee, mo = _frames([("USA", "Spain")], ["Spain"])
    out, boosted = apply_origin_affinity(matrix, ee, mo.drop(columns=[M_ORIG]))
    assert boosted == 0 and out.equals(matrix)


def test_origin_bonus_never_overrides_affiliation_penalty():
    # A Spain-born mentor working ABROAD (US) and one working IN Spain, both for a Spanish student.
    matrix, ee, mo = _frames([("USA", "Spain"), ("Spain", "Spain")], ["Spain"])
    m1, _ = apply_country_restrictions(matrix, ee, mo, countries=["spain"])  # penalize shared affiliation
    m2, _ = apply_origin_affinity(m1, ee, mo)                                # then the origin boost
    abroad = m2.loc["10", "100"]   # US affiliation: no penalty, +origin bonus
    in_spain = m2.loc["11", "100"] # Spain affiliation: penalized, +origin bonus
    assert abroad == 50.0 + BONUS
    assert in_spain == max(0.0, 50.0 - SAME_COUNTRY_PENALTY) + BONUS
    assert in_spain < 50.0 < abroad   # the abroad Spanish mentor wins; in-Spain stays deprioritized
