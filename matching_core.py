"""Headless matching logic: no Streamlit, no I/O. Callers decide how to display a report."""

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from info import MENTEE_COLUMNS, MENTOR_COLUMNS

MISSING_MENTEE_SCORE = 30  # AI omitted the pair: assume a poor match


@dataclass
class AssignmentResult:
    assignments: list[tuple[str, str, int]]  # (mentee, mentor, score), sorted by mentee id
    unmatched_mentees: list[str] = field(default_factory=list)  # no mentor capacity left
    cleaning: dict = field(default_factory=dict)  # nan_filled / inf_replaced / clipped counts


def hungarian_assignment(
    matrix_df: pd.DataFrame, max_mentees_per_mentor: int = 2
) -> AssignmentResult:
    """Optimal mentee-mentor assignment over a mentors x mentees score matrix (0-100)."""
    nan_filled = int(matrix_df.isna().sum().sum())
    matrix_df = matrix_df.fillna(30)  # AI omitted the pair: treat as a poor match

    inf_mask = np.isinf(matrix_df.to_numpy(dtype=float, na_value=np.nan))
    inf_replaced = int(inf_mask.sum())
    matrix_df = matrix_df.replace([np.inf, -np.inf], 50)

    try:
        matrix_df = matrix_df.astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Matrix must contain only numeric values: {e}") from e

    clipped = int(((matrix_df < 0) | (matrix_df > 100)).sum().sum())
    matrix_df = matrix_df.clip(0, 100)
    cleaning = {"nan_filled": nan_filled, "inf_replaced": inf_replaced, "clipped": clipped}

    mentors, mentees = list(matrix_df.index), list(matrix_df.columns)
    # Clone each mentor column once per slot so the solver enforces the cap.
    cost = -np.repeat(matrix_df.to_numpy().T, max_mentees_per_mentor, axis=1)
    mentee_idx, slot_idx = linear_sum_assignment(cost)

    assignments = [
        (
            mentees[i],
            mentors[j // max_mentees_per_mentor],
            int(matrix_df.iloc[j // max_mentees_per_mentor, i]),
        )
        for i, j in zip(mentee_idx, slot_idx, strict=True)
    ]

    def numeric_sort_key(a: tuple) -> int:
        m = re.search(r"\d+", str(a[0]))
        return int(m.group()) if m else 0

    assignments.sort(key=numeric_sort_key)
    matched = {a[0] for a in assignments}
    unmatched = [m for m in mentees if m not in matched]
    return AssignmentResult(assignments, unmatched, cleaning)


def check_mentor_conflicts(matches: list[dict], max_mentees_per_mentor: int = 2) -> list[dict]:
    """Mentors given more than `max_mentees_per_mentor` mentees at rank 1."""
    if not matches:
        return []

    mentor_assignments: dict = {}
    for match_data in matches:
        if match_data["matches"]:
            rank1_match = match_data["matches"][0]
            if rank1_match["rank"] == 1:
                mentor_assignments.setdefault(rank1_match["mentor_id"], []).append(
                    (match_data["mentee_id"], rank1_match["match_percentage"])
                )

    return [
        {"mentor_id": mentor_id, "assignments": assigned, "count": len(assigned)}
        for mentor_id, assigned in mentor_assignments.items()
        if len(assigned) > max_mentees_per_mentor
    ]


def _countries_by_id(df: pd.DataFrame, cols: dict) -> dict[str, str]:
    """Map string ID -> country normalized with strip().lower() ("" when missing)."""
    return {
        str(i): "" if pd.isna(c) else str(c).strip().lower()
        for i, c in zip(df[cols["id"]], df[cols["country"]], strict=True)
    }


# Same-country pairing is a soft priority, not a ban: a shared-country cell loses this many
# points so the solver avoids it when a better cross-country option exists, yet a clearly
# stronger same-country match can still win and nobody is left unmatched when it's unavoidable.
# Tune higher to avoid same-country harder, lower to weigh it more like any other pair.
SAME_COUNTRY_PENALTY = 40.0


def apply_country_restrictions(
    matrix_df: pd.DataFrame,
    mentees_df: pd.DataFrame,
    mentors_df: pd.DataFrame,
    avoid_same_country: bool = False,
    countries: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Penalize mentor x mentee cells that share a country; returns (matrix copy, penalized cells).

    Same-country pairs are deprioritized by SAME_COUNTRY_PENALTY (clipped at 0), not zeroed — a
    soft priority, so a strong same-country match can still beat a weak cross-country one and the
    assignment never leaves a mentee unmatched just to avoid a same-country pairing.
    `countries` (restrict only within those countries) wins over `avoid_same_country`;
    an empty `countries` list is treated as "no named list" (falls back to the flag).
    With no restriction requested the matrix is returned unchanged with `[]` (checked before
    column presence, so a default call never returns a skip-marker). A missing country column
    when a restriction IS requested returns the matrix unchanged, with `[{"skipped": ...}]` as
    the report. Penalized cells are listed only when their original score was > 0.
    """
    out = matrix_df.copy()
    named = {c.strip().lower() for c in countries} or None if countries else None
    if named is None and not avoid_same_country:
        return out, []
    if MENTOR_COLUMNS["country"] not in mentors_df.columns:
        return out, [{"skipped": "mentor country column missing"}]
    if MENTEE_COLUMNS["country"] not in mentees_df.columns:
        return out, [{"skipped": "mentee country column missing"}]
    mentor_c = _countries_by_id(mentors_df, MENTOR_COLUMNS)
    mentee_c = _countries_by_id(mentees_df, MENTEE_COLUMNS)
    reason = (
        f"Same-country restriction ({', '.join(sorted(named))})"
        if named is not None
        else "Same-country restriction"
    )

    filtered: list[dict] = []
    for mentor_id in out.index:
        mc = mentor_c.get(str(mentor_id), "")
        if not mc or (named is not None and mc not in named):
            continue
        for mentee_id in out.columns:
            if mentee_c.get(str(mentee_id), "") != mc:
                continue
            score = out.loc[mentor_id, mentee_id]
            if pd.isna(score):
                continue
            out.loc[mentor_id, mentee_id] = max(0.0, float(score) - SAME_COUNTRY_PENALTY)
            if score > 0:
                filtered.append(
                    {
                        "mentor_id": str(mentor_id),
                        "mentee_id": str(mentee_id),
                        "original_score": float(score),
                        "reason": reason,
                    }
                )
    return out, filtered
