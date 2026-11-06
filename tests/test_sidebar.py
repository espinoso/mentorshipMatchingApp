"""The baremos sidebar: six weight sliders + Spain/Portugal toggles, wired to session state."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_sidebar_exposes_weights_and_country_rules(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
    monkeypatch.setenv("ALLOW_NO_AUTH", "1")
    at = AppTest.from_file(APP, default_timeout=30).run()

    labels = [s.label for s in at.sidebar.slider]
    assert labels == ["Field & specialization", "Goals & mentoring fit", "Skills overlap",
                      "Level & seniority", "Availability / time zone", "Style fit"]
    assert {"Avoid Spain ↔ Spain", "Avoid Portugal ↔ Portugal"} <= {c.label for c in at.sidebar.checkbox}

    # Defaults reach session state, and both countries are avoided out of the box.
    assert at.session_state["match_weights"]["field_specialization"] == 30
    assert at.session_state["country_restrictions"] == ["spain", "portugal"]
