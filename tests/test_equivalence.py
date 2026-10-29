"""Golden equivalence test for the headless Hungarian assignment.

There is no API key in CI, so we cannot run a live before/after match. Instead we pin a
FIXED synthetic score matrix and assert that `matching_core.hungarian_assignment` produces
exactly the assignment the pre-refactor solver produced for it. Both implementations use the
same slot-expansion + `scipy.linear_sum_assignment` + numeric mentee sort, so a fixed matrix
with a unique optimum is a deterministic stand-in for a live equivalence run. If the solver's
behavior ever drifts, this golden assertion fails.
"""

import pandas as pd

from matching_core import hungarian_assignment

# 3 mentors x 5 mentees, cap 2 (6 slots >= 5 mentees). Scores chosen so the optimum is unique:
# m1 clearly wins mentees 1&2, m2 wins 3&4, m3 wins 5.
FIXED_MATRIX = pd.DataFrame(
    {
        "1": [95, 10, 10],
        "2": [90, 10, 10],
        "3": [10, 95, 10],
        "4": [10, 90, 10],
        "5": [10, 10, 95],
    },
    index=["m1", "m2", "m3"],
)

# The assignment the pre-refactor logic produced for FIXED_MATRIX (sorted by numeric mentee id).
EXPECTED_ASSIGNMENTS = [
    ("1", "m1", 95),
    ("2", "m1", 90),
    ("3", "m2", 95),
    ("4", "m2", 90),
    ("5", "m3", 95),
]


def test_fixed_matrix_matches_pre_refactor_assignment():
    res = hungarian_assignment(FIXED_MATRIX, max_mentees_per_mentor=2)
    assert res.assignments == EXPECTED_ASSIGNMENTS
    assert res.unmatched_mentees == []
    assert res.cleaning == {"nan_filled": 0, "inf_replaced": 0, "clipped": 0}


def test_input_matrix_not_mutated():
    before = FIXED_MATRIX.copy()
    hungarian_assignment(FIXED_MATRIX, max_mentees_per_mentor=2)
    assert FIXED_MATRIX.equals(before)
