"""
Visualization functions for the Mentorship Matching System
Contains functions for creating charts, graphs, and visual representations of data
"""

import re

import plotly.graph_objects as go


def _num(value):
    """Leading integer in an id, for a stable numeric tiebreak; large default sorts unknowns last."""
    m = re.search(r'\d+', str(value))
    return int(m.group()) if m else 10**9


def create_heatmap(matrix_df, assignments=None, mentee_fields=None, mentor_fields=None):
    """Interactive compatibility heatmap (mentors × mentees).

    When field maps (id -> field) are given, rows and columns are grouped by field so same-field
    pairs form visible blocks (a dense grid of printed numbers is unreadable, so there are none —
    color carries the value and hover carries the detail). assignments are outlined.
    """
    df = matrix_df
    if mentor_fields:
        df = df.loc[sorted(df.index, key=lambda i: (str(mentor_fields.get(str(i), '~')).lower(), _num(i)))]
    if mentee_fields:
        df = df[sorted(df.columns, key=lambda c: (str(mentee_fields.get(str(c), '~')).lower(), _num(c)))]

    row_field = [str(mentor_fields.get(str(i), '')) if mentor_fields else '' for i in df.index]
    col_field = [str(mentee_fields.get(str(c), '')) if mentee_fields else '' for c in df.columns]
    customdata = [[[rf, cf] for cf in col_field] for rf in row_field]

    # Ids as strings so both axes are categorical — a mixed str/int pair makes plotly tile the grid wrong.
    fig = go.Figure(data=go.Heatmap(
        z=df.values, x=[str(c) for c in df.columns], y=[str(i) for i in df.index],
        colorscale='RdYlGn', zmin=0, zmax=100,
        colorbar={'title': "Match %"},
        customdata=customdata,
        hovertemplate=(
            'Mentor %{y} (%{customdata[0]}) → Mentee %{x} (%{customdata[1]})'
            '<br>Match: %{z}%<extra></extra>'
        ),
        hoverongaps=False,
    ))

    if assignments:
        col_pos = {str(c): i for i, c in enumerate(df.columns)}
        row_pos = {str(r): i for i, r in enumerate(df.index)}
        shapes = []
        for mentee, mentor, _score in assignments:
            x, y = col_pos.get(str(mentee)), row_pos.get(str(mentor))
            if x is not None and y is not None:
                shapes.append({
                    'type': "rect", 'x0': x - 0.5, 'y0': y - 0.5, 'x1': x + 0.5, 'y1': y + 0.5,
                    'line': {'color': "#1565C0", 'width': 2},
                })
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        title="Compatibility — grouped by field · blue = chosen match · hover for detail",
        xaxis_title="Mentees", yaxis_title="Mentors",
        height=max(560, len(df) * 13),
        xaxis={'type': 'category', 'tickangle': -45, 'showticklabels': len(df.columns) <= 80},
        yaxis={'type': 'category', 'showticklabels': len(df) <= 80},
        font={'size': 9},
    )
    return fig

