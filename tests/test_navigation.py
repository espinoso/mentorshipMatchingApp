import app


def test_reachable_no_data():
    assert app._reachable(has_data=False, has_results=False) == {0, 1, 2}


def test_reachable_with_data():
    assert app._reachable(has_data=True, has_results=False) == {0, 1, 2, 3, 4}


def test_reachable_with_results():
    assert app._reachable(has_data=True, has_results=True) == {0, 1, 2, 3, 4, 5}


def test_results_unreachable_without_data():
    # results can't exist without data, but guard the invariant anyway
    assert 5 not in app._reachable(has_data=False, has_results=False)
