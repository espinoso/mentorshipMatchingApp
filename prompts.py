"""
Prompt templates for the Mentorship Matching System
Contains all prompt generation functions for the matching algorithm
"""

import streamlit as st
from info import MENTEE_COLUMNS, MENTOR_COLUMNS, JSON_FOOTER, FIELD_COMPATIBILITY_MATRIX_TEXT

# ============================================================================
# PROMPT CONSTANTS - Single source of truth for JSON instructions
# ============================================================================

# JSON Header - Instructions for models that need explicit JSON format guidance
JSON_HEADER = """⚠️ CRITICAL: RETURN ONLY JSON - NO EXPLANATIONS ⚠️

You MUST return ONLY valid JSON in the exact format specified below.
DO NOT include any text before the JSON.
DO NOT include any text after the JSON.
DO NOT explain your process.
DO NOT use markdown code blocks.
DO NOT say "Here's the JSON" or similar.

START your response with: {{"matrix": [
END your response with: }}

═══════════════════════════════════════════════════════════════

"""



def create_matrix_prompt(use_file_search=False, include_json_instructions=True):
    """Create the prompt for generating compatibility matrix with STRICT SCORING and COMPLETENESS
    
    Args:
        use_file_search: Whether to include file search instructions
        include_json_instructions: If False, omit JSON format instructions (for models with structured output)
    """
    
    file_search_instruction = """
TRAINING DATA REFERENCE:
Use the training files to identify patterns of successful matches.
Pay attention to what makes a "good" vs "poor" match in historical data.
""" if use_file_search else ""
    
    json_header = JSON_HEADER if include_json_instructions else ""
    
    prompt = f"""{json_header}You are evaluating mentor-mentee compatibility for IMFAHE's International Mentor Program (Multidisciplinary).

MISSION: Match mentees with mentors across ALL academic and professional disciplines 
for international career acceleration through expert guidance.

Program Coverage: Life Sciences, Engineering, Computer Science, Business, Economics, 
Social Sciences, Humanities, Design, Environmental Sciences, and all professional fields

{file_search_instruction}

═══════════════════════════════════════════════════════════════
⚠️ CRITICAL REQUIREMENTS - READ FIRST ⚠️
═══════════════════════════════════════════════════════════════

1. COMPLETENESS: You MUST return scores for EVERY mentee-mentor combination
   - If you receive 50 mentors and 100 mentees, you MUST return 50 × 100 = 5,000 scores
   - NEVER skip a combination, even if it's a terrible match
   - Bad matches should score 5-40%, NOT be omitted
   - Missing combinations will be filled with 30% (poor match), which defeats our purpose

2. CALCULATION METHOD: Follow this sequence for EACH pair:
   Step 1: Calculate field alignment points (0-40) using the matrix below
   Step 2: Count specialization keyword matches (0-30)
   Step 3: Evaluate expertise relevance (0-20)
   Step 4: Check education qualification (0-10)
   Step 5: Sum the points = subtotal
   Step 6: Apply caps if needed (different fields → max 40%, no keywords → max 55%)
   Step 7: Final score = min(subtotal, applicable_cap)

3. FIELD MATCHING IS STRICT: Use ONLY the field compatibility matrix provided
   - Do NOT invent relationships between fields
   - If two fields are not listed as related → 0 points
   - "Can benefit from" or "might be relevant" = 0 points (we need direct matches)

4. KEYWORD MATCHING IS LITERAL: Count actual word overlaps
   - "protein crystallography" vs "protein structure" = 1 match ("protein")
   - "electrochemistry" vs "immunotherapies" = 0 matches
   - Ignore common words (and, the, in, of, research, analysis, study, development)
   - Only count technical/domain-specific terms

{FIELD_COMPATIBILITY_MATRIX_TEXT}
═══════════════════════════════════════════════════════════════
SCORING RUBRIC (0-100 scale) - APPLY MATHEMATICALLY
═══════════════════════════════════════════════════════════════

A. FIELD ALIGNMENT (40 points) - USE THE MATRIX ABOVE
───────────────────────────────────────────────────────────────
Compare {MENTEE_COLUMNS['field']} vs {MENTOR_COLUMNS['field']}

⚠️ DO NOT use reasoning like "can benefit from" or "might be relevant"
⚠️ ONLY award points if the pairing is EXPLICITLY in the matrix above
⚠️ If not in the matrix → 0 points

Example:
✅ Chemistry vs Chemistry → 40 points (identical)
✅ Chemistry vs Chemical Engineering → 30 points (in matrix)
✅ Chemistry vs Biology → 10 points (adjacent via Biochemistry)
❌ Chemistry vs Molecular Life Sciences → 0 points (NOT in matrix)
❌ Chemistry vs Immunology → 0 points (NOT in matrix)

⚠️ MANDATORY CAP: If Field Alignment = 0 points → FINAL SCORE CANNOT EXCEED 40%
   Even if other factors sum to 60 points, cap the final result at 40%

═══════════════════════════════════════════════════════════════
B. SPECIALIZATION MATCH (30 points) - COUNT KEYWORDS LITERALLY
═══════════════════════════════════════════════════════════════
Extract technical keywords from both specialization fields and COUNT overlaps:
{MENTEE_COLUMNS['specialization']} keywords vs {MENTOR_COLUMNS['specialization']} keywords

Steps:
1. Remove common words: and, the, in, of, to, for, with, research, analysis, study, development, using, based, methods
2. Extract remaining technical terms (usually 3-8 words per specialization)
3. Count exact or near-exact matches (e.g., "protein" = "proteins", "crystallography" = "crystallographic")
4. Award points based on count:
   • 5+ matching keywords: 30 points
   • 3-4 matching keywords: 20 points
   • 1-2 matching keywords: 10 points
   • 0 matching keywords: 0 points

⚠️ DO NOT award points for "thematically similar" or "could be related"
⚠️ ONLY count actual word/term overlaps

Examples:
✅ "protein crystallography X-ray diffraction" vs "crystallography protein structure analysis"
   Matches: protein, crystallography → 2 keywords → 10 points

❌ "electrochemistry battery materials" vs "immunotherapies cancer treatment"
   Matches: (none) → 0 keywords → 0 points

⚠️ MANDATORY CAP: If Specialization Match = 0 points → FINAL SCORE CANNOT EXCEED 55%

═══════════════════════════════════════════════════════════════
C. MENTOR EXPERTISE → MENTEE NEEDS (20 points)
───────────────────────────────────────────────────────────────
Does {MENTOR_COLUMNS['specialization']} address {MENTEE_COLUMNS['guidance_areas']} and {MENTEE_COLUMNS['career_goals']}?

Consider:
• Mentor's technical specialization vs mentee's guidance needs
• Mentor's {MENTOR_COLUMNS['current_position']} relevance to mentee's career goals
• Any additional context from {MENTEE_COLUMNS['other_info']}

Scoring:
• Mentor's expertise directly addresses 80%+ of mentee's needs: 20 points
• Addresses 50-80% of needs: 12 points
• Addresses 30-50% of needs: 6 points
• Addresses <30% of needs: 0 points

⚠️ "Direct addressing" means technical expertise overlap, not general mentorship
⚠️ Current position helps assess if mentor can guide on specific career paths

═══════════════════════════════════════════════════════════════
D. EDUCATION & EXPERIENCE (10 points)
───────────────────────────────────────────────────────────────
• Mentor's {MENTOR_COLUMNS['education_level']} ≥ Mentee's {MENTEE_COLUMNS['education_level']}: 5 points
• Mentor's {MENTOR_COLUMNS['experience_years']} ≥ 5 years: 5 points

═══════════════════════════════════════════════════════════════
CALIBRATION EXAMPLES (FOLLOW THESE PATTERNS)
═══════════════════════════════════════════════════════════════

🟢 EXAMPLE 1: EXCELLENT MATCH (85-95%)
Mentee: Biology | Specialization: "protein folding, molecular dynamics"
Mentor: Molecular Biology | Specialization: "protein structure, folding dynamics"

CALCULATION:
Field: Biology vs Molecular Biology = HIGHLY RELATED = 30 points
Specialization: "protein folding molecular dynamics" vs "protein structure folding dynamics"
  → Matches: protein, folding, dynamics = 3 keywords = 20 points
Expertise: Directly addresses = 20 points
Education: Qualified = 10 points
TOTAL: 30 + 20 + 20 + 10 = 80 points
Caps: None apply
FINAL SCORE: 80%

═══════════════════════════════════════════════════════════════

🟡 EXAMPLE 2: GOOD MATCH (60-75%) - BASED ON YOUR ACTUAL DATA
Mentee: Bioinformatics | Specialization: "type 2 diabetes, computational modeling, personalized medicine"
Mentor: Medicine (Cardiorenal Physiology) | Specialization: "diabetes, cardiovascular disease, patient outcomes"

CALCULATION:
Field: Bioinformatics vs Medicine = RELATED (see matrix) = 20 points
Specialization: "diabetes" overlaps + "medicine"/"patient" themes = 2 keywords = 10 points
Expertise: Diabetes research directly addresses mentee's diabetes focus = 20 points
Education: Qualified = 10 points
TOTAL: 20 + 10 + 20 + 10 = 60 points
Caps: None apply (has keyword matches)
FINAL SCORE: 60%

⚠️ NOTE: This should score 60-70%, NOT 30% like in your test. If still scoring low, field pairing may not be in matrix.

═══════════════════════════════════════════════════════════════

🔴 EXAMPLE 3: POOR MATCH (5-20%) - BASED ON YOUR ACTUAL DATA
Mentee: Environmental Sciences | Specialization: "conservation, biodiversity, climate change"
Mentor: Pharmacy (Industrial) | Specialization: "drug manufacturing, quality control, pharmaceutical production"

CALCULATION:
Field: Environmental Sciences vs Pharmacy = NOT IN MATRIX = 0 points
Specialization: No keyword overlap ("conservation" ≠ "pharmaceutical") = 0 points
Expertise: Cannot address conservation/climate needs = 0 points
Education: Qualified = 10 points
TOTAL: 0 + 0 + 0 + 10 = 10 points
Caps: Different fields → max 40% cap (but we're already at 10%)
FINAL SCORE: 10%

⚠️ IMPORTANT: This is a bad match, but we STILL INCLUDE IT in the output with 10-15%
⚠️ DO NOT omit this combination from your response
⚠️ This matches your bad match example and should score 10-20%, NOT 30%

═══════════════════════════════════════════════════════════════

🔴 EXAMPLE 4: TERRIBLE MATCH (5-15%) - STILL INCLUDE THIS SCORE
Mentee: Computer Science | Specialization: "machine learning, neural networks"
Mentor: Medicine | Specialization: "cardiology, clinical trials"

CALCULATION:
Field: Computer Science vs Medicine = DIFFERENT = 0 points
Specialization: No overlap = 0 points
Expertise: Cannot address technical CS needs = 0 points
Education: Has PhD but in wrong field = 5 points
TOTAL: 0 + 0 + 0 + 5 = 5 points
Caps: Different fields → max 40% cap (but we're already at 5%)
FINAL SCORE: 5%

⚠️ IMPORTANT: Include this score. Do NOT skip it. Bad matches need low scores, not omission.
"""
    
    json_footer = JSON_FOOTER if include_json_instructions else ""
    
    full_prompt = prompt + json_footer
    return full_prompt


def get_prompt_for_api(include_json_instructions=True):
    """
    Assemble the full prompt to send to the API.
    Wraps the custom rubric from session state with JSON instructions and output format.
    
    Args:
        include_json_instructions: If False, omit JSON format instructions (for models with structured output)
    """
    custom_rubric = st.session_state.get('custom_prompt', '')
    training_file_ids = st.session_state.get('training_file_ids', [])
    use_file_search = len(training_file_ids) > 0
    
    json_header = JSON_HEADER if include_json_instructions else ""
    json_footer = JSON_FOOTER if include_json_instructions else ""
    
    full_prompt = json_header + custom_rubric + json_footer
    return full_prompt


def create_default_prompt():
    """Extract only the EDITABLE rubric section (middle part of the full prompt)"""
    training_file_ids = st.session_state.get('training_file_ids', [])
    full_prompt = create_matrix_prompt(use_file_search=len(training_file_ids) > 0)
    
    start_marker = "You are evaluating mentor-mentee"
    end_marker = "OUTPUT FORMAT - MUST INCLUDE"
    
    start_idx = full_prompt.find(start_marker)
    end_idx = full_prompt.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        editable_section = full_prompt[start_idx:end_idx].strip()
        return editable_section
    
    return full_prompt

