"""The exported Excel is de-anonymized locally from the session identity map; without a map it
stays ID-only (and never errors)."""

import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import ui_processing as u
from info import MENTEE_COLUMNS, MENTOR_COLUMNS

MATCHES = [{"mentee_id": "E1", "matches": [
    {"rank": 1, "mentor_id": "M1", "match_percentage": 88, "match_quality": "Excellent"}]}]
MENTEES = pd.DataFrame({MENTEE_COLUMNS["id"]: ["E1"], MENTEE_COLUMNS["field"]: ["Bio"]})
MENTORS = pd.DataFrame({MENTOR_COLUMNS["id"]: ["M1"], MENTOR_COLUMNS["field"]: ["Bio"]})
ASSIGN = [("E1", "M1", 88)]
IDENT = {
    "mentee": {"E1": {"name": "Ada Lovelace", "email": "ada@x.org"}},
    "mentor": {"M1": {"name": "Alan Turing", "email": "alan@y.org", "linkedin": "linkedin.com/in/turing"}},
}


def _sheets(data):
    return pd.read_excel(BytesIO(data), sheet_name=None)


def test_export_is_deanonymized_with_identities():
    sheets = _sheets(u._matches_report_bytes(MATCHES, MENTEES, MENTORS, ASSIGN, identities=IDENT))
    matches = sheets["Matches"].iloc[0]
    assert matches["Mentee_Name"] == "Ada Lovelace"
    assert matches["Mentor_Email"] == "alan@y.org"
    assert matches["Mentor_LinkedIn"] == "linkedin.com/in/turing"
    # the per-entity sheets gain Name columns too
    assert "Alan Turing" in set(sheets["Mentors"]["Name"])
    assert "Ada Lovelace" in set(sheets["Mentees"]["Name"])


def test_export_without_identities_stays_id_only():
    sheets = _sheets(u._matches_report_bytes(MATCHES, MENTEES, MENTORS, ASSIGN))
    matches = sheets["Matches"].iloc[0]
    assert matches["Mentee_ID"] == "E1" and matches["Mentor_ID"] == "M1"
    assert pd.isna(matches["Mentee_Name"]) or not str(matches["Mentee_Name"]).strip()  # no PII invented
    assert "Name" not in sheets["Mentors"].columns    # per-entity sheets gain no identity columns
