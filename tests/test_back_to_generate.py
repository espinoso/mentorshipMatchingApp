from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_no_api_key_run_shows_enabled_back_button(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.session_state["mentees_df"] = pd.DataFrame({"a": [1]})
    at.session_state["mentors_df"] = pd.DataFrame({"b": [1]})
    at.session_state["api_key"] = ""
    at.session_state["step"] = 4
    at.session_state["start_processing"] = True
    at.run()
    rail = [b for b in at.button if b.key and b.key.startswith("rail_")]
    assert rail and all(b.disabled for b in rail)
    back = [b for b in at.button if b.key == "back_to_generate"]
    assert len(back) == 1 and not back[0].disabled
    back[0].click()
    at.run()
    assert at.session_state["step"] == 4 and not at.session_state["start_processing"]
    assert not any(b.disabled for b in at.button if b.key == "rail_0")
