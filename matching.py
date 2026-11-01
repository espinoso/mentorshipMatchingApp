"""Backwards-compatible re-exports.

The matching logic lives in `matching_core` (Streamlit-free). This module keeps the
old `from matching import ...` call sites working with a single implementation.
"""

from matching_core import (
    AssignmentResult,
    apply_country_restrictions,
    check_mentor_conflicts,
    hungarian_assignment,
)

__all__ = [
    "AssignmentResult",
    "apply_country_restrictions",
    "check_mentor_conflicts",
    "hungarian_assignment",
]
