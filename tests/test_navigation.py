import app


def test_reachable_no_data():
    # Both entity steps (Mentees, Mentors) are always reachable; nothing downstream.
    assert app._reachable(has_data=False, has_results=False) == {0, 1}


def test_reachable_with_data():
    # Both files present unlocks Generate (2), but not Results.
    assert app._reachable(has_data=True, has_results=False) == {0, 1, 2}


def test_reachable_with_results():
    # Results (3) and Review (4) both unlock once there are matches.
    assert app._reachable(has_data=True, has_results=True) == {0, 1, 2, 3, 4}


def test_results_unreachable_without_data():
    assert 3 not in app._reachable(has_data=False, has_results=False)
    assert 4 not in app._reachable(has_data=False, has_results=False)


def test_five_steps():
    assert app._STEPS == ["Mentees", "Mentors", "Generate", "Results", "Review"]
