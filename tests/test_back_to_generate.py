from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_missing_api_key_stops_with_error(monkeypatch):
    """No API key → main() shows a clear error and stops, before any step/rail renders."""
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.session_state["api_key"] = ""
    at.run()
    assert any("Missing API key" in str(e.value) for e in at.error)
    assert not [b for b in at.button if b.key and b.key.startswith("rail_")]


def test_dead_end_run_shows_enabled_back_button(monkeypatch):
    """A run that can't proceed (here: start_processing set with no data) renders the locked
    rail plus an enabled '← Back to Generate' escape, so the user is never trapped."""
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.session_state["api_key"] = "x"          # pass the key gate
    at.session_state["mentees_df"] = None        # not ready -> dead-end
    at.session_state["mentors_df"] = None
    at.session_state["start_processing"] = True
    at.run()
    rail = [b for b in at.button if b.key and b.key.startswith("rail_")]
    assert rail and all(b.disabled for b in rail)
    back = [b for b in at.button if b.key == "back_to_generate"]
    assert len(back) == 1 and not back[0].disabled
    back[0].click()
    at.run()
    assert not at.session_state["start_processing"]
    # The user is freed: Upload (rail_0) is reachable again.
    assert not any(b.disabled for b in at.button if b.key == "rail_0")
