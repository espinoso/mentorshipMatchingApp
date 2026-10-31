"""Prompt templates for the Mentorship Matching System.

The division of labor, redone for a capable model:
- CODE owns the plumbing — the output schema, the "score every pair" completeness rule, the
  truncation guard (in the client), and hard rules like the country restriction and the
  mentees-per-mentor cap (in the matcher).
- The RUBRIC below is pure scoring policy — the dimensions and the calibration scale — stated
  in plain language. It is what a person sees and may edit; modern Claude judges multidisciplinary
  compatibility better than the old hand-tuned arithmetic + curated field matrix did.
"""

import streamlit as st

from info import JSON_FOOTER

# Code-owned instruction wrapped around the rubric (not part of the editable rubric itself).
_SCORING_INSTRUCTION = (
    "Score the compatibility of EVERY mentee-mentor pair from 0 to 100 using the rubric below. "
    "Return a score for every pair — never omit one, even obviously weak matches (give those low scores)."
)

# The editable scoring policy: dimensions + calibration only. No JSON/format talk, no plumbing.
MATRIX_RUBRIC = """You are scoring mentor-mentee compatibility for IMFAHE's International Mentor
Program, a multidisciplinary program that pairs mentees with mentors for international career acceleration.

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
    """The editable scoring rubric (shown in the sidebar's Advanced section) — policy only."""
    return MATRIX_RUBRIC


def get_prompt_for_api(include_json_instructions=True):
    """Assemble the full system prompt: the code-owned scoring instruction + the (possibly
    customized) rubric + the output schema. JSON robustness lives in the client (streamed, with a
    stop_reason guard) and the parser (fence-stripping), so there is no separate 'RETURN ONLY JSON'
    coercion block — the schema footer names the exact structure.
    """
    rubric = st.session_state.get("custom_prompt") or MATRIX_RUBRIC
    prompt = f"{_SCORING_INSTRUCTION}\n\n{rubric}"
    if include_json_instructions:
        prompt += "\n\n" + JSON_FOOTER
    return prompt
