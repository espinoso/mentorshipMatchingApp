"""Backwards-compatible re-exports.

The matching logic lives in `matching_core` (Streamlit-free). This module keeps the
old `from matching import ...` call sites working with a single implementation.
"""

from matching_core import (
    AssignmentResult,
    MatrixParseReport,
    apply_country_restrictions,
    check_mentor_conflicts,
    hungarian_assignment,
    parse_matrix_response,
)

__all__ = [
    "AssignmentResult",
    "MatrixParseReport",
    "apply_country_restrictions",
    "check_mentor_conflicts",
    "hungarian_assignment",
    "parse_matrix_response",
]
