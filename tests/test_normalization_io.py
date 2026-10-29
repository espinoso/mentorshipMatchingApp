"""Tests for normalization file I/O robustness (header sniffing)."""
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Normalization"))
from shared_utils import read_input_sheet


def test_read_input_sheet_sniffs_header_past_blank_rows():
    """The sample files carry two blank leading rows; read_input_sheet must find the real
    header (header=1 would land on a blank row and yield Unnamed columns)."""
    path = os.path.join(tempfile.mkdtemp(), "sheet.xlsx")
    rows = [
        [None, None, None],
        [None, None, None],
        ["Code Mentee", "Mentee Field of expertise", "Mentee Specialization"],
        ["Mentee 1", "Biology", "genetics"],
        ["Mentee 2", "Chemistry", "electrochemistry"],
    ]
    pd.DataFrame(rows).to_excel(path, index=False, header=False)

    df = read_input_sheet(path)
    assert list(df.columns) == ["Code Mentee", "Mentee Field of expertise", "Mentee Specialization"]
    assert len(df) == 2
    assert str(df.iloc[0]["Code Mentee"]) == "Mentee 1"
