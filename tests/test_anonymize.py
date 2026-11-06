"""PII capture/restore: the identity map is exact, empty for CODED files, and PII-only."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from anonymize import capture_identities, detect_pii_columns, display_name


def test_detect_pii_columns_matches_2026_headers():
    cols = ["Timestamp", "Full name ", "Email ", "Affiliation (e.g., Harvard University)",
            "Field of expertise (e.g., Biomedicine)", "Please share the link to your LinkedIn profile here."]
    found = detect_pii_columns(cols)
    assert found["name"] == "Full name "
    assert found["email"] == "Email "
    assert found["linkedin"] == "Please share the link to your LinkedIn profile here."


def test_capture_is_exact_by_assigned_id():
    raw = pd.DataFrame({
        "Full name ": ["Ada Lovelace", "Alan Turing"],
        "Email ": ["ada@x.org", "alan@y.org"],
        "Field of expertise": ["Mathematics", "Computer Science"],
    })
    ids = range(1, len(raw) + 1)  # how the normalizer assigns IDs (positional)
    idmap = capture_identities(raw, ids)
    assert idmap["1"] == {"name": "Ada Lovelace", "email": "ada@x.org"}
    assert idmap["2"]["name"] == "Alan Turing"
    assert "linkedin" not in idmap["1"]  # no LinkedIn column -> key omitted


def test_coded_file_without_pii_yields_empty_map():
    coded = pd.DataFrame({
        "Mentor ID": [1, 2],
        "Mentor Field of expertise": ["Bio", "CS"],
        "Mentor Country of affiliated institution": ["Spain", "UK"],
    })
    assert capture_identities(coded, coded["Mentor ID"]) == {}


def test_blank_pii_cells_skipped():
    raw = pd.DataFrame({"Full name ": ["", None], "Email ": ["a@b.c", None]})
    idmap = capture_identities(raw, range(1, 3))
    assert idmap == {"1": {"email": "a@b.c"}}  # row 1 had only an email; row 2 all blank -> omitted


def test_display_name_falls_back_to_id_label():
    idmap = {"1": {"name": "Ada Lovelace"}, "2": {"email": "x@y.z"}}
    assert display_name(idmap, 1, "Mentor ") == "Ada Lovelace"
    assert display_name(idmap, 2, "Mentor ") == "Mentor 2"   # no name -> id label
    assert display_name({}, 7, "Mentee ") == "Mentee 7"      # no map at all
