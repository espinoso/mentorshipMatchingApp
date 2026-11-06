"""In-app PII handling for the matcher.

This year's raw survey exports carry name/email/LinkedIn; last year's "CODED" files did not.
The AIs (Jev scoring, Claude reasoning) must only ever see IDs + work fields, so these helpers
capture the PII into a map keyed by the assigned ID — kept in Streamlit session memory only,
never persisted or sent anywhere — so results can be de-anonymized locally for display/export.
"""
import re

import pandas as pd

# logical PII key -> header patterns (case-insensitive substring match on raw survey column names)
_PII_PATTERNS = {
    "name": (r"full\s*name", r"^name$", r"\bfull name\b"),
    "email": (r"e-?mail",),
    "linkedin": (r"linkedin",),
}


def detect_pii_columns(columns) -> dict:
    """Map each PII key to the first raw column whose header matches it, or None if absent."""
    found = {}
    for key, patterns in _PII_PATTERNS.items():
        match = None
        for col in columns:
            text = str(col).strip().lower()
            if any(re.search(p, text) for p in patterns):
                match = col
                break
        found[key] = match
    return found


def capture_identities(raw_df: pd.DataFrame, id_values, columns=None) -> dict:
    """Build ``{str(id): {name?, email?, linkedin?}}`` from raw_df's PII columns.

    ``id_values[i]`` is the ID the normalizer assigned to ``raw_df`` row ``i`` (assignment is
    positional), so the map is exact. Blank PII cells are skipped; a row with no PII is omitted.
    Returns ``{}`` when the file has no PII columns at all (already-anonymized CODED files).
    """
    pii = detect_pii_columns(columns if columns is not None else raw_df.columns)
    present = {key: col for key, col in pii.items() if col is not None}
    if not present:
        return {}
    identities: dict = {}
    for i, rid in enumerate(id_values):
        record = {}
        for key, col in present.items():
            val = raw_df.iloc[i][col]
            if pd.notna(val) and str(val).strip():
                record[key] = str(val).strip()
        if record:
            identities[str(rid)] = record
    return identities


def display_name(identity_map: dict, entity_id, fallback_prefix: str = "") -> str:
    """The person's name if captured, else a stable id label like ``Mentor 12`` (prefix + id)."""
    record = identity_map.get(str(entity_id)) if identity_map else None
    if record and record.get("name"):
        return record["name"]
    return f"{fallback_prefix}{entity_id}".strip()
