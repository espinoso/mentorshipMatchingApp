"""create_heatmap: field grouping, no in-cell numbers, enriched hover, assignment outlines."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualization import create_heatmap


def test_groups_by_field_and_has_no_cell_text():
    # mentors A,B (rows) x mentees 1,2 (cols)
    m = pd.DataFrame({'1': [10, 20], '2': [30, 40]}, index=['A', 'B'])
    fig = create_heatmap(
        m, assignments=[('1', 'B', 55)],
        mentee_fields={'1': 'Bio', '2': 'CS'}, mentor_fields={'A': 'CS', 'B': 'Bio'},
    )
    hm = fig.data[0]
    assert hm.texttemplate in (None, "")  # no printed % in every cell
    assert list(hm.x) == ['1', '2']  # mentees grouped by field (Bio < CS)
    assert list(hm.y) == ['B', 'A']  # mentors grouped by field (Bio 'B' < CS 'A')
    assert hm.customdata is not None  # fields for the hover
    assert len(fig.layout.shapes) == 1  # the assignment is outlined


def test_works_without_field_maps():
    m = pd.DataFrame({'1': [10]}, index=['A'])
    fig = create_heatmap(m)  # no fields, no assignments
    assert list(fig.data[0].x) == ['1'] and list(fig.data[0].y) == ['A']
