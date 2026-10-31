"""Prompt templates for the Mentorship Matching System.

The scoring rubric is intentionally short. Modern Claude judges multidisciplinary
compatibility better than the old hand-tuned arithmetic + curated field-compatibility matrix
did (which demonstrably capped good cross-field matches), so the rubric names the dimensions
and a calibration scale in plain language and lets the model judge. Determinism — the output
schema, completeness, the truncation guard, and hard rules like the country restriction and the
mentees-per-mentor cap — lives in code, not here.
"""

import streamlit as st

from info import JSON_FOOTER

MATRIX_RUBRIC = """Return ONLY the JSON described at the end of this message.

You are scoring mentor-mentee compatibility for IMFAHE's International Mentor Program, a
multidisciplinary program that pairs mentees with mentors for international career acceleration.

Give every mentee-mentor pair a 0-100 compatibility score. Score EVERY pair, including weak
ones - never omit a pair.

Judge compatibility on these dimensions, in rough order of importance:
1. Field / discipline alignment - same or closely related academic/professional areas.
2. Specialization & skills overlap - do the mentee's specialization and hard skills genuinely
   intersect the mentor's expertise?
3. Fit to the mentee's needs - does the mentor's expertise, position and experience address the
   mentee's stated guidance needs and 2-year career goals?
4. Seniority - is the mentor experienced/educated enough to guide this mentee?

CALIBRATION - use the FULL 0-100 range and be discriminating. In a diverse, multidisciplinary
pool MOST pairs are mediocre; reserve high scores for genuinely strong fits:
- 85-100: excellent - same or adjacent field AND clear specialization/skills overlap AND directly
  serves the mentee's stated goals. Rare.
- 65-80: good - same/related field with real but partial overlap, or strong overlap in one dimension.
- 45-60: moderate - related field OR some transferable overlap; the mentor can help but it is not targeted.
- 25-40: weak - different fields with only loose, thematic ("could be relevant") connection. Most
  cross-discipline pairs land here.
- 5-20: poor - essentially unrelated; no field, specialization, or needs overlap.

Be strict about the top band: "the mentor could probably help somehow" is a 45-60, NOT an 80. A
different field with no concrete specialization overlap cannot exceed ~55, however senior the mentor.

Judge relatedness and overlap using your own understanding of the disciplines - do NOT require exact
keyword matches, and judge meaning, not shared words."""


def create_default_prompt():
    """The editable scoring rubric (shown in the sidebar's Advanced section)."""
    return MATRIX_RUBRIC


def get_prompt_for_api(include_json_instructions=True):
    """Full system rubric sent to the model: the (possibly customized) rubric + the output schema.

    JSON robustness lives in the client (streamed, with a stop_reason guard) and the parser
    (fence-stripping), so we no longer prepend the old 'RETURN ONLY JSON' coercion block — just
    the schema footer, which names the exact output structure.
    """
    rubric = st.session_state.get("custom_prompt") or MATRIX_RUBRIC
    if not include_json_instructions:
        return rubric
    return rubric + "\n\n" + JSON_FOOTER


def create_matrix_prompt(include_json_instructions=True):
    """The full default matrix prompt (rubric + schema). Kept for callers and tests."""
    if not include_json_instructions:
        return MATRIX_RUBRIC
    return MATRIX_RUBRIC + "\n\n" + JSON_FOOTER
