"""
Visualization functions for the Mentorship Matching System
Contains functions for creating charts, graphs, and visual representations of data
"""

import plotly.graph_objects as go
import pandas as pd


def create_heatmap(matrix_df, assignments=None):
    """Create interactive heatmap visualization of compatibility matrix
    
    Args:
        matrix_df: DataFrame with mentors as rows, mentees as columns
        assignments: Optional list of (mentee, mentor, score) tuples to highlight
    
    Returns:
        plotly.graph_objects.Figure: Interactive heatmap figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns,
        y=matrix_df.index,
        colorscale='RdYlGn',  # Red-Yellow-Green
        text=matrix_df.values,
        texttemplate='%{text}%',
        textfont={"size": 8},
        colorbar=dict(title="Match %"),
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> → <b>%{x}</b><br>Match: %{z}%<extra></extra>'
    ))
    
    if assignments:
        assignment_dict = {(mentee, mentor): score for mentee, mentor, score in assignments}
        
        shapes = []
        for mentee_idx, mentee in enumerate(matrix_df.columns):
            for mentor_idx, mentor in enumerate(matrix_df.index):
                if (mentee, mentor) in assignment_dict:
                    shapes.append(dict(
                        type="rect",
                        x0=mentee_idx - 0.5,
                        y0=mentor_idx - 0.5,
                        x1=mentee_idx + 0.5,
                        y1=mentor_idx + 0.5,
                        line=dict(color="blue", width=3),
                    ))
        
        fig.update_layout(shapes=shapes)
    
    fig.update_layout(
        title="Mentee-Mentor Compatibility Matrix",
        xaxis_title="Mentees",
        yaxis_title="Mentors",
        height=max(600, len(matrix_df) * 25),  # Dynamic height
        width=max(800, len(matrix_df.columns) * 30),  # Dynamic width
        xaxis=dict(tickangle=-45),
        font=dict(size=10)
    )
    
    return fig

