"""Headless matching logic: no Streamlit, no I/O. Callers decide how to display a report."""

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from info import MENTEE_COLUMNS, MENTOR_COLUMNS

MISSING_MENTEE_SCORE = 30  # AI omitted the pair: assume a poor match


@dataclass
class MatrixParseReport:
    """Outcome of parsing one batch's matrix response; `df` is None when unusable."""

    df: pd.DataFrame | None = None
    wrong_mentors: list[str] = field(default_factory=list)
    wrong_mentees: list[str] = field(default_factory=list)
    missing_mentors: list[str] = field(default_factory=list)
    missing_mentees_filled: list[str] = field(default_factory=list)
    completion_rate: float = 0.0  # percent of expected mentor x mentee scores received
    error: str | None = None
    notes: list[str] = field(default_factory=list)  # non-fatal observations
    raw_excerpt: str = ""  # first 2000 chars of the cleaned response, for debugging


def _as_text(response_text: object) -> str:
    """Unwrap SDK objects that carry the text under `.text` (possibly nested)."""
    if isinstance(response_text, str):
        return response_text
    text_obj = getattr(response_text, "text", response_text)
    return str(getattr(text_obj, "text", text_obj))


def parse_matrix_response(
    response_text: object, expected_mentors: list, expected_mentees: list
) -> MatrixParseReport:
    """Parse the model's matrix JSON and validate it against the requested batch IDs."""
    report = MatrixParseReport()
    text = _as_text(response_text).strip()
    report.raw_excerpt = text[:2000]

    # A repr of a config object means a reasoning model returned no JSON content.
    if "ResponseTextConfig" in text or "ResponseFormatText" in text:
        report.error = "Received a configuration object instead of JSON content"
        return report

    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or first >= last:
        report.error = "Could not find JSON object boundaries in response"
        return report
    before, after = text[:first].strip(), text[last + 1 :].strip()
    if before or after:
        report.notes.append(
            f"Model added text around the JSON (before: {before[:500]!r}, after: {after[:500]!r})"
        )
    text = text[first : last + 1]

    mentor_set = {str(x) for x in expected_mentors}
    mentee_set = {str(x) for x in expected_mentees}
    matrix: dict[str, dict[str, object]] = {}
    seen_mentees: set[str] = set()
    try:
        data = json.loads(text)
        if "matrix" not in data:
            raise ValueError("Response should contain 'matrix' key")
        if not isinstance(data["matrix"], list):
            raise TypeError("Matrix should be a list")
        for entry in data["matrix"]:
            mentor_id = str(entry["mentor_id"])
            if mentor_id not in mentor_set:
                report.wrong_mentors.append(mentor_id)
                continue
            scores = {}
            for score in entry["scores"]:
                mentee_id = str(score["mentee_id"])
                if mentee_id not in mentee_set:
                    if mentee_id not in report.wrong_mentees:
                        report.wrong_mentees.append(mentee_id)
                    continue
                scores[mentee_id] = score["percentage"]
                seen_mentees.add(mentee_id)
            if scores:
                matrix[mentor_id] = scores
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        report.error = f"Error parsing matrix response: {e!s}"
        return report

    expected_total = len(expected_mentors) * len(expected_mentees)
    actual_total = sum(len(s) for s in matrix.values())
    report.completion_rate = actual_total / expected_total * 100 if expected_total else 0.0
    report.missing_mentors = sorted(mentor_set - set(matrix))
    if not matrix:
        report.error = "No valid mentor data after filtering"
        return report

    df = pd.DataFrame(matrix).T
    report.missing_mentees_filled = [str(m) for m in expected_mentees if str(m) not in df.columns]
    for mentee in report.missing_mentees_filled:
        df[mentee] = MISSING_MENTEE_SCORE
    report.df = df[[str(m) for m in sorted(expected_mentees)]]
    return report


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


def apply_country_restrictions(
    matrix_df: pd.DataFrame,
    mentees_df: pd.DataFrame,
    mentors_df: pd.DataFrame,
    avoid_same_country: bool = False,
    countries: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Zero mentor x mentee cells that share a country; returns (matrix copy, filtered cells).

    `countries` (restrict only within those countries) wins over `avoid_same_country`.
    With neither set the matrix is returned unchanged. A missing country column also returns
    the matrix unchanged, with `[{"skipped": ...}]` as the report. Filtered cells are listed
    only when their original score was > 0.
    """
    out = matrix_df.copy()
    if MENTOR_COLUMNS["country"] not in mentors_df.columns:
        return out, [{"skipped": "mentor country column missing"}]
    if MENTEE_COLUMNS["country"] not in mentees_df.columns:
        return out, [{"skipped": "mentee country column missing"}]
    if countries is None and not avoid_same_country:
        return out, []

    named = {c.strip().lower() for c in countries} if countries is not None else None
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
            out.loc[mentor_id, mentee_id] = 0
            if pd.notna(score) and score > 0:
                filtered.append(
                    {
                        "mentor_id": str(mentor_id),
                        "mentee_id": str(mentee_id),
                        "original_score": float(score),
                        "reason": reason,
                    }
                )
    return out, filtered
