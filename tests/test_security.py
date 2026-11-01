"""Security-fix regression tests: spreadsheet formula neutralization + fail-closed auth."""

import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import sanitize_spreadsheet_df

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_sanitize_neutralizes_formula_cells():
    df = pd.DataFrame({
        "text": ["=HYPERLINK('x')", "normal", "-1", "@cmd", "+2", "\tTab"],
        "num": [5, 10, 15, 20, 25, 30],
    })
    out = sanitize_spreadsheet_df(df)
    assert out.loc[0, "text"] == "'=HYPERLINK('x')"
    assert out.loc[1, "text"] == "normal"            # plain text untouched
    assert out.loc[2, "text"] == "'-1"
    assert out.loc[3, "text"] == "'@cmd"
    assert out.loc[4, "text"] == "'+2"
    assert out.loc[5, "text"] == "'\tTab"
    assert list(out["num"]) == [5, 10, 15, 20, 25, 30]  # numbers never touched


def test_app_fails_closed_without_password(monkeypatch):
    """No APP_ACCESS_PASSWORD and no ALLOW_NO_AUTH → the app refuses to serve (fail closed)."""
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    monkeypatch.delenv("ALLOW_NO_AUTH", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("not configured" in str(e.value).lower() for e in at.error)
    assert not [b for b in at.button if b.key and b.key.startswith("rail_")]  # app did not load


def test_app_allows_dev_opt_out(monkeypatch):
    """ALLOW_NO_AUTH=1 is the explicit local/dev bypass."""
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    monkeypatch.setenv("ALLOW_NO_AUTH", "1")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.session_state["authenticated"] is True
