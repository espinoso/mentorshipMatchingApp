import pandas as pd

from matching_core import AssignmentResult, check_mentor_conflicts, hungarian_assignment


def test_optimal_assignment_prefers_high_scores():
    df = pd.DataFrame({"1": [90, 10], "2": [20, 80]}, index=["m1", "m2"])  # mentors x mentees
    res = hungarian_assignment(df, max_mentees_per_mentor=1)
    assert isinstance(res, AssignmentResult)
    pairs = {mentee: mentor for mentee, mentor, _ in res.assignments}
    assert pairs == {"1": "m1", "2": "m2"}
    assert res.unmatched_mentees == []
    assert res.cleaning == {"nan_filled": 0, "inf_replaced": 0, "clipped": 0}


def test_cap_limits_mentees_per_mentor():
    df = pd.DataFrame({"1": [90, 1], "2": [85, 1], "3": [80, 1]}, index=["m1", "m2"])
    res = hungarian_assignment(df, max_mentees_per_mentor=2)
    counts = {}
    for _, mentor, _ in res.assignments:
        counts[mentor] = counts.get(mentor, 0) + 1
    assert all(c <= 2 for c in counts.values())


def test_more_mentees_than_capacity_reports_unmatched():
    df = pd.DataFrame({"1": [90], "2": [80], "3": [70]}, index=["m1"])  # 1 mentor, cap 2, 3 mentees
    res = hungarian_assignment(df, max_mentees_per_mentor=2)
    assert len(res.assignments) == 2 and len(res.unmatched_mentees) == 1
    assert res.unmatched_mentees == ["3"]


def test_nan_filled_and_recorded():
    df = pd.DataFrame({"1": [float("nan")], "2": [80.0]}, index=["m1"])
    res = hungarian_assignment(df, max_mentees_per_mentor=2)
    assert res.cleaning["nan_filled"] == 1
    assert {m: s for m, _, s in res.assignments}["1"] == 30


def test_inf_and_out_of_range_cleaned_and_recorded():
    df = pd.DataFrame({"1": [float("inf"), 150.0], "2": [-5.0, 60.0]}, index=["m1", "m2"])
    res = hungarian_assignment(df, max_mentees_per_mentor=1)
    assert res.cleaning["inf_replaced"] == 1
    assert res.cleaning["clipped"] == 2
    assert all(0 <= s <= 100 for _, _, s in res.assignments)


def test_assignments_sorted_numerically_by_mentee_id():
    df = pd.DataFrame({"10": [90, 1], "2": [1, 90], "1": [50, 50]}, index=["m1", "m2"])
    res = hungarian_assignment(df, max_mentees_per_mentor=2)
    assert [m for m, _, _ in res.assignments] == ["1", "2", "10"]


def test_check_mentor_conflicts_flags_over_cap():
    def m(mentee, mentor):
        return {
            "mentee_id": mentee,
            "matches": [{"rank": 1, "mentor_id": mentor, "match_percentage": 90}],
        }

    matches = [m("1", "a"), m("2", "a"), m("3", "a"), m("4", "b")]
    conflicts = check_mentor_conflicts(matches, max_mentees_per_mentor=2)
    assert [(c["mentor_id"], c["count"]) for c in conflicts] == [("a", 3)]
    assert check_mentor_conflicts([]) == []
