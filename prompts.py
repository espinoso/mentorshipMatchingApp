"""
Prompt templates for the Mentorship Matching System
Contains all prompt generation functions for the matching algorithm
"""

import streamlit as st
from info import MENTEE_COLUMNS, MENTOR_COLUMNS


def create_matrix_prompt(use_file_search=False, specialty=None):
    """Create the prompt for generating compatibility matrix with STRICT SCORING and COMPLETENESS"""
    
    # Get specialty from session state if not provided (now always multidisciplinary)
    if specialty is None:
        specialty = st.session_state.get('imfahe_specialty', 
            "IMFAHE International Mentor Program (Multidisciplinary - All Fields)")
    
    # Extract specialty name
    specialty_name = "IMFAHE International Mentor Program (Multidisciplinary)"
    specialty_fields = "All academic and professional fields"
    
    file_search_instruction = """
TRAINING DATA REFERENCE:
Use the training files to identify patterns of successful matches.
Pay attention to what makes a "good" vs "poor" match in historical data.
""" if use_file_search else ""
    
    prompt = f"""⚠️ CRITICAL: RETURN ONLY JSON - NO EXPLANATIONS ⚠️

You MUST return ONLY valid JSON in the exact format specified below.
DO NOT include any text before the JSON.
DO NOT include any text after the JSON.
DO NOT explain your process.
DO NOT use markdown code blocks.
DO NOT say "Here's the JSON" or similar.

START your response with: {{"matrix": [
END your response with: ]}}

═══════════════════════════════════════════════════════════════

You are evaluating mentor-mentee compatibility for IMFAHE's International Mentor Program (Multidisciplinary).

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

═══════════════════════════════════════════════════════════════
FIELD COMPATIBILITY MATRIX (AUTHORITATIVE - USE THIS EXACTLY)
═══════════════════════════════════════════════════════════════

⚠️ WILDCARD RULE: If mentee field == mentor field (case-insensitive) → 40 POINTS
   This covers ALL exact matches, including fields not explicitly listed below.

40 POINTS - IDENTICAL FIELDS (explicit list + wildcard above):
• Accounting = Accounting
• Agricultural Sciences = Agricultural Sciences
• Architecture = Architecture
• Behavioral Science = Behavioral Science
• Bioengineering = Bioengineering
• Biomedical Engineering = Biomedical Engineering
• Biomedicine = Biomedicine
• Biotechnology = Biotechnology
• Business Informatics = Business Informatics
• Chemical Engineering = Chemical Engineering
• Chemistry = Chemistry
• Communication Studies = Communication Studies
• Computer Science = Computer Science
• Economics = Economics
• Economics/Entrepreneurship = Economics/Entrepreneurship
• Education = Education
• Electrical Engineering = Electrical Engineering
• Energy Engineering = Energy Engineering
• Engineering = Engineering
• Environmental Science = Environmental Science
• Fashion Design = Fashion Design
• Health Sciences = Health Sciences
• Interior Design = Interior Design
• Law = Law
• Linguistics = Linguistics
• Literature Studies = Literature Studies
• Mathematics = Mathematics
• Mechanical Engineering = Mechanical Engineering
• Pharmaceutical Sciences = Pharmaceutical Sciences
• Pharmacy = Pharmacy
• Physics = Physics
• Psychological Neuroscience = Psychological Neuroscience
• Psychology = Psychology
• Renewable Energy = Renewable Energy
• Sociology = Sociology
• Sports Science = Sports Science
• Telecommunications Engineering = Telecommunications Engineering
• Advanced Chemistry = Advanced Chemistry
• Agrifood and Biosystems Science and Engineering = Agrifood and Biosystems Science and Engineering
• Analytical Chemistry = Analytical Chemistry
• Biochemistry = Biochemistry
• Biochemistry and Molecular Biology = Biochemistry and Molecular Biology
• Biodiversity = Biodiversity
• Bioengineering = Bioengineering
• Bioinformatics = Bioinformatics
• Biology = Biology
• Biomedical Engineering = Biomedical Engineering
• Biomedical Sciences = Biomedical Sciences
• Biomedicine = Biomedicine
• Biomedicine and Biotechnology = Biomedicine and Biotechnology
• Biophysics = Biophysics
• Biotechnology = Biotechnology
• Cancer Research = Cancer Research
• Cell Biology = Cell Biology
• Chemistry = Chemistry
• Clinical Psychology = Clinical Psychology
• Computational Biology = Computational Biology
• Data Science = Data Science
• Drug Discovery = Drug Discovery
• Ecology = Ecology
• Environmental Engineering = Environmental Engineering
• Environmental Sciences = Environmental Sciences
• Epidemiology = Epidemiology
• Genetics = Genetics
• Genomics = Genomics
• Health Sciences = Health Sciences
• Immunology = Immunology
• Material Science = Material Science
• Medical Sciences = Medical Sciences
• Medical Technology = Medical Technology
• Medicine = Medicine
• Microbiology = Microbiology
• Molecular Biology = Molecular Biology
• Molecular Life Sciences = Molecular Life Sciences
• Neuroscience = Neuroscience
• Neuroscience and Neurosurgery = Neuroscience and Neurosurgery
• Nutrition and Dietetics = Nutrition and Dietetics
• Oncology = Oncology
• Pharmaceutical Sciences = Pharmaceutical Sciences
• Pharmacology = Pharmacology
• Pharmacology and Physiology = Pharmacology and Physiology
• Pharmacy = Pharmacy
• Pharmacy and Biomedical Sciences = Pharmacy and Biomedical Sciences
• Physiology = Physiology
• Psychiatry = Psychiatry
• Public Health = Public Health
• Statistical Physics and Complex Systems = Statistical Physics and Complex Systems
• Translational Medicine = Translational Medicine

30 POINTS - HIGHLY RELATED (within same cluster, highly overlapping):

CLUSTER A: Molecular, Cellular & Genetic Sciences
• Biochemistry ↔ Biochemistry and Molecular Biology
• Biochemistry ↔ Molecular Biology
• Biochemistry ↔ Molecular Life Sciences
• Molecular Biology ↔ Molecular Life Sciences
• Molecular Biology ↔ Cell Biology
• Molecular Biology ↔ Biochemistry and Molecular Biology
• Cell Biology ↔ Molecular Life Sciences
• Genetics ↔ Genomics
• Genetics ↔ Molecular Biology
• Genomics ↔ Bioinformatics
• Bioinformatics ↔ Computational Biology
• Biophysics ↔ Molecular Biology
• Biophysics ↔ Biochemistry

CLUSTER B: Biomedical & Medical Sciences  
• Biomedical Sciences ↔ Biomedicine
• Biomedical Sciences ↔ Medicine
• Biomedicine ↔ Medicine
• Biomedicine ↔ Biomedicine and Biotechnology
• Medical Sciences ↔ Medicine
• Medical Sciences ↔ Biomedical Sciences
• Medicine ↔ Translational Medicine
• Medicine ↔ Medical Technology
• Oncology ↔ Cancer Research
• Neuroscience ↔ Neuroscience and Neurosurgery
• Biomedical Engineering ↔ Biomedical Sciences
• Public Health ↔ Epidemiology
• Clinical Psychology ↔ Psychiatry

CLUSTER C: Pharmaceutical & Drug Development
• Pharmacology ↔ Pharmacy
• Pharmacology ↔ Pharmaceutical Sciences
• Pharmacy ↔ Pharmaceutical Sciences
• Pharmacy ↔ Pharmacy and Biomedical Sciences
• Pharmacology ↔ Pharmacology and Physiology
• Drug Discovery ↔ Pharmaceutical Sciences
• Drug Discovery ↔ Pharmacology

CLUSTER D: Environmental & Agricultural Sciences
• Ecology ↔ Biodiversity
• Ecology ↔ Environmental Sciences
• Environmental Sciences ↔ Environmental Engineering
• Environmental Engineering ↔ Agrifood and Biosystems Science and Engineering

CLUSTER E: General Biology & Applied Life Sciences
• Biology ↔ Microbiology
• Biology ↔ Biotechnology
• Health Sciences ↔ Nutrition and Dietetics
• Biotechnology ↔ Biomedicine and Biotechnology

CLUSTER F: Chemistry & Materials
• Chemistry ↔ Analytical Chemistry
• Chemistry ↔ Advanced Chemistry
• Chemistry ↔ Material Science

CLUSTER G: Engineering Fields (NEW)
• Engineering ↔ Mechanical Engineering
• Engineering ↔ Electrical Engineering
• Mechanical Engineering ↔ Electrical Engineering
• Electrical Engineering ↔ Telecommunications Engineering
• Chemical Engineering ↔ Chemistry
• Energy Engineering ↔ Renewable Energy
• Energy Engineering ↔ Electrical Engineering

CLUSTER H: Psychology & Behavioral Sciences (NEW)
• Psychology ↔ Psychological Neuroscience
• Psychology ↔ Behavioral Science
• Psychological Neuroscience ↔ Behavioral Science
• Behavioral Science ↔ Sociology

CLUSTER I: Economics & Business (NEW)
• Economics ↔ Economics/Entrepreneurship
• Economics ↔ Accounting
• Economics/Entrepreneurship ↔ Accounting

CLUSTER J: Design & Architecture (NEW)
• Architecture ↔ Interior Design
• Fashion Design ↔ Interior Design

CLUSTER K: Language & Communication (NEW)
• Linguistics ↔ Literature Studies
• Linguistics ↔ Communication Studies
• Literature Studies ↔ Communication Studies

20 POINTS - RELATED (cross-cluster connections, overlapping disciplines):

Cross Molecular-Biomedical:
• Molecular Biology ↔ Biomedical Sciences
• Molecular Biology ↔ Medicine
• Biochemistry ↔ Biomedical Sciences
• Genetics ↔ Medicine
• Genomics ↔ Biomedical Sciences
• Bioinformatics ↔ Biomedical Sciences
• Bioinformatics ↔ Medicine
• Computational Biology ↔ Biomedical Sciences
• Cell Biology ↔ Biomedical Sciences
• Molecular Life Sciences ↔ Medicine

Cross Biomedical-Pharma:
• Biomedicine ↔ Pharmacology
• Biomedicine ↔ Pharmacy
• Biomedical Sciences ↔ Pharmaceutical Sciences
• Medicine ↔ Drug Discovery
• Oncology ↔ Pharmacology
• Translational Medicine ↔ Drug Discovery
• Physiology ↔ Pharmacology and Physiology
• Pharmacy and Biomedical Sciences ↔ Biomedical Sciences

Cross Molecular-Pharma:
• Molecular Biology ↔ Drug Discovery
• Biochemistry ↔ Pharmacology
• Genetics ↔ Pharmacology

Cross Biology-Medical:
• Biology ↔ Medicine
• Biology ↔ Biomedical Sciences
• Biology ↔ Health Sciences
• Microbiology ↔ Immunology
• Microbiology ↔ Medicine
• Biology ↔ Biomedicine
• Biotechnology ↔ Biomedical Sciences
• Biotechnology ↔ Bioengineering

Cross Medical-Specialized:
• Medicine ↔ Oncology
• Medicine ↔ Physiology
• Medicine ↔ Neuroscience
• Medicine ↔ Public Health
• Biomedical Sciences ↔ Cancer Research
• Epidemiology ↔ Public Health

Cross Environmental-Biology:
• Environmental Sciences ↔ Biology
• Ecology ↔ Biology
• Biodiversity ↔ Biology

Cross Chemistry-Biology:
• Chemistry ↔ Biochemistry
• Chemistry ↔ Biology
• Analytical Chemistry ↔ Biochemistry

Cross Data-Biology:
• Data Science ↔ Bioinformatics
• Data Science ↔ Computational Biology
• Data Science ↔ Genomics

Cross Life Sciences-Engineering (NEW):
• Biomedicine ↔ Chemical Engineering
• Biotechnology ↔ Chemical Engineering
• Chemistry ↔ Chemical Engineering
• Pharmaceutical Sciences ↔ Chemical Engineering
• Biomedical Engineering ↔ Mechanical Engineering
• Biomedical Engineering ↔ Electrical Engineering

Cross Life Sciences-Computer Science (NEW):
• Biomedicine ↔ Computer Science (bioinformatics)
• Biotechnology ↔ Computer Science (biotech data)
• Health Sciences ↔ Computer Science (health informatics)

Cross Engineering-Computer Science (NEW):
• Engineering ↔ Computer Science (software engineering)
• Electrical Engineering ↔ Computer Science (hardware-software)
• Telecommunications Engineering ↔ Computer Science (network engineering)
• Computer Science ↔ Business Informatics

Cross Engineering-Physics/Math (NEW):
• Engineering ↔ Physics
• Mechanical Engineering ↔ Physics
• Electrical Engineering ↔ Physics
• Engineering ↔ Mathematics
• Computer Science ↔ Mathematics

Cross Psychology-Health (NEW):
• Psychology ↔ Health Sciences (mental health)
• Psychological Neuroscience ↔ Biomedicine (neuroscience)
• Behavioral Science ↔ Health Sciences (behavioral health)

Cross Social Sciences-Humanities (NEW):
• Sociology ↔ Psychology
• Education ↔ Psychology
• Sociology ↔ Communication Studies
• Law ↔ Sociology
• Law ↔ Economics

Cross Business-Technology (NEW):
• Economics/Entrepreneurship ↔ Business Informatics
• Economics ↔ Business Informatics
• Economics/Entrepreneurship ↔ Computer Science (tech entrepreneurship)

Cross Business-Design (NEW):
• Fashion Design ↔ Economics/Entrepreneurship (fashion business)
• Architecture ↔ Economics/Entrepreneurship (real estate, construction)

Cross Design-Engineering (NEW):
• Architecture ↔ Engineering (structural, civil)
• Architecture ↔ Mechanical Engineering (HVAC, building systems)
• Interior Design ↔ Architecture

Cross Environmental Sciences (NEW):
• Environmental Science ↔ Agricultural Sciences
• Environmental Science ↔ Chemistry (environmental chemistry)
• Agricultural Sciences ↔ Biotechnology (agribiotechnology)

Cross Health-Sports (NEW):
• Sports Science ↔ Health Sciences
• Sports Science ↔ Psychology (sports psychology)
• Sports Science ↔ Biomedicine (sports medicine)

Cross Education Connections (NEW):
• Education ↔ Linguistics (language education)
• Education ↔ Mathematics (mathematics education)
• Education ↔ Psychology (educational psychology)
• Education ↔ Sociology (sociology of education)

10 POINTS - ADJACENT (minimal overlap, bridged by intermediate field):
• Environmental Sciences ↔ Chemistry (via environmental chemistry)
• Environmental Sciences ↔ Health Sciences (via environmental health)
• Chemistry ↔ Medicine (via pharmaceutical sciences)
• Chemistry ↔ Pharmacology (via medicinal chemistry)
• Chemistry ↔ Pharmacy (via medicinal chemistry)
• Physics ↔ Chemistry (via physical chemistry)
• Mathematics ↔ Economics (via econometrics)
• Mathematics ↔ Physics (via mathematical physics)
• Data Science ↔ Epidemiology (via biostatistics)
• Statistical Physics and Complex Systems ↔ Computational Biology (via systems modeling)
• Bioengineering ↔ Medicine (via biomedical engineering)
• Material Science ↔ Biomedical Engineering (via biomaterials)
• Nutrition and Dietetics ↔ Medicine (via clinical nutrition)
• Public Health ↔ Medicine (via preventive medicine)
• Law ↔ Business Informatics (via legal tech, data privacy)
• Communication Studies ↔ Business Informatics (via digital media)
• Sociology ↔ Economics (via socioeconomics)
• Literature Studies ↔ Education (via literature education)
• Fashion Design ↔ Communication Studies (via visual communication)
• Architecture ↔ Environmental Science (via sustainable design)
• Energy Engineering ↔ Environmental Science (via renewable energy)
• Renewable Energy ↔ Environmental Science
• Telecommunications Engineering ↔ Business Informatics (via IT infrastructure)
• Accounting ↔ Law (via tax law, corporate law)
• Health Sciences ↔ Sociology (via public health, health equity)
• Agricultural Sciences ↔ Economics (via agricultural economics)

0 POINTS - DIFFERENT (everything else - NO connection):
Examples of COMPLETELY unrelated pairings:
• Environmental Sciences ↔ Pharmacology → 0 POINTS
• Environmental Sciences ↔ Drug Discovery → 0 POINTS
• Ecology ↔ Pharmaceutical Sciences → 0 POINTS
• Chemistry ↔ Neuroscience → 0 POINTS (no bridge)
• Data Science ↔ Ecology → 0 POINTS (no connection)
• Clinical Psychology ↔ Environmental Engineering → 0 POINTS
• Statistical Physics ↔ Pharmacy → 0 POINTS
• Material Science ↔ Psychiatry → 0 POINTS
• Fashion Design ↔ Biomedicine → 0 POINTS (NEW FIELDS)
• Literature Studies ↔ Engineering → 0 POINTS (NEW FIELDS)
• Law ↔ Chemistry → 0 POINTS (NEW FIELDS)
• Sports Science ↔ Linguistics → 0 POINTS (NEW FIELDS)
• Accounting ↔ Pharmaceutical Sciences → 0 POINTS (NEW FIELDS)
• Interior Design ↔ Physics → 0 POINTS (NEW FIELDS)
• Telecommunications Engineering ↔ Psychology → 0 POINTS (NEW FIELDS)

⚠️ ANY PAIRING NOT EXPLICITLY LISTED ABOVE → 0 POINTS
When in doubt, assign 0 points. Do NOT invent relationships.

✅ IMPORTANT: The matrix above now covers ALL 37 fields in your data, including:
   Life Sciences, Engineering, Computer Science, Business, Social Sciences,
   Humanities, Design, Environmental Sciences, and more.

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

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT - MUST INCLUDE ALL COMBINATIONS
═══════════════════════════════════════════════════════════════

Return JSON with this EXACT structure (no markdown, no extra text):

  {{
  "matrix": [
      {{
      "mentor_id": "EXACT_CODE_FROM_{MENTOR_COLUMNS['id']}",
      "scores": [
        {{"mentee_id": "EXACT_CODE", "percentage": 82}},
        {{"mentee_id": "EXACT_CODE", "percentage": 45}},
        {{"mentee_id": "EXACT_CODE", "percentage": 15}},
        {{"mentee_id": "EXACT_CODE", "percentage": 68}},
        ... (MUST include ALL mentees, even with low scores)
      ]
    }},
    ... (MUST include ALL mentors)
  ]
}}

CRITICAL RULES:
1. Every mentor MUST have scores for ALL mentees
2. If mentor has 100 mentees, scores array MUST have 100 entries
3. Use actual calculated scores - even if 5%, 10%, 15%
4. DO NOT skip combinations
5. DO NOT use null, undefined, or omit entries
6. Low scores (5-40%) are expected and correct for bad matches

VERIFICATION BEFORE SENDING RESPONSE:
✓ Did I include ALL mentors in the response?
✓ Does each mentor have ALL mentee scores?
✓ Did I include low scores (5-30%) for obviously bad matches?
✓ Did I apply the field alignment matrix strictly (not creatively)?
✓ Did I count actual keyword overlaps (not thematic similarity)?
✓ Did I apply the 40% cap for different fields?
✓ Did I apply the 55% cap for zero keyword overlap?

═══════════════════════════════════════════════════════════════
⚠️ FINAL REMINDER: JSON ONLY - NO TEXT ⚠️
═══════════════════════════════════════════════════════════════

Your ENTIRE response must be ONLY the JSON object.

❌ BAD - Do NOT do this:
"I will now calculate the scores. Here's the JSON: {{"matrix": [...]}}"

❌ BAD - Do NOT do this:
"To generate matches, I will use the scoring criteria... Let me compute..."

✅ GOOD - Do this (ONLY JSON, nothing else):
{{"matrix": [{{"mentor_id": "M001", "scores": [{{"mentee_id": "E001", "percentage": 75}}]}}]}}

Remember:
• First character of response: {{
• Last character of response: }}
• No text before or after
• No explanations
• No markdown
• No process descriptions

BEGIN YOUR RESPONSE NOW WITH: {{"matrix": ["""
    return prompt


def create_reasoning_prompt(assignments, specialty=None):
    """Create prompt to get reasoning for specific assignments"""
    
    # Get specialty from session state if not provided
    if specialty is None:
        specialty = st.session_state.get('imfahe_specialty', 
            "IMP-Biomedicine (Biology, Medicine, Pharmacy, Biotechnology or related areas)")
    
    specialty_name = specialty.split("(")[0].strip()
    
    assignment_list = "\n".join([f"- {mentee} → {mentor} (score: {score}%)" 
                                  for mentee, mentor, score in assignments])
    
    prompt = f"""You are explaining matches for IMFAHE's International Mentor Program ({specialty_name}).

CONTEXT:
- IMFAHE Mission: Global career acceleration through international mentorship
- Focus: Helping mentees develop international careers through practical guidance
- These assignments were optimally selected considering all mentees and mentors

ASSIGNMENTS TO EXPLAIN:
{assignment_list}

For each assignment, provide reasoning in 2-3 sentences that explains:

1. FIELD & SPECIALIZATION MATCH (Most Important)
   - Mention if fields are the same/aligned
   - Highlight specific keyword overlaps in specializations
   - Example: "Both work in biomedicine, with overlapping expertise in molecular biology and genetics"

2. MENTOR SKILLS → MENTEE NEEDS
   - Connect mentor's specialization to mentee's guidance areas and career goals
   - Example: "Mentor's experience in grant writing directly addresses mentee's need for research funding guidance"

3. ADDITIONAL FACTORS (if relevant)
   - Education compatibility (mentor qualified to guide at mentee's level)
   - Institution type match (if mentee seeks industry/academia transition)
   - Alma mater connection (if mentor studied where mentee currently is)
   - Example: "Mentor's PhD from mentee's current university provides valuable institutional insights"

NOTE: Do NOT mention country or language factors (these are for internal use only)

OUTPUT FORMAT:
Return ONLY a JSON array (no markdown, no extra text):
[
  {{
    "mentee_id": "EXACT_CODE",
    "mentor_id": "EXACT_CODE",
    "reasoning": "2-3 sentences explaining field/specialization alignment, how mentor's skills address mentee's needs, and any relevant bonuses."
  }},
  ...
]"""
    return prompt


def get_prompt_for_api():
    """
    Assemble the full prompt to send to the API.
    Wraps the custom rubric from session state with JSON instructions and output format.
    """
    # Get the custom rubric from session state
    custom_rubric = st.session_state.get('custom_prompt', '')
    training_file_ids = st.session_state.get('training_file_ids', [])
    use_file_search = len(training_file_ids) > 0
    
    # JSON Header (non-editable)
    json_header = """⚠️ CRITICAL: RETURN ONLY JSON - NO EXPLANATIONS ⚠️

You MUST return ONLY valid JSON in the exact format specified below.
DO NOT include any text before the JSON.
DO NOT include any text after the JSON.
DO NOT explain your process.
DO NOT use markdown code blocks.
DO NOT say "Here's the JSON" or similar.

START your response with: {"matrix": [
END your response with: ]}

═══════════════════════════════════════════════════════════════

"""
    
    # Output Format Footer (non-editable)
    output_footer = f"""
═══════════════════════════════════════════════════════════════
OUTPUT FORMAT - MUST INCLUDE ALL COMBINATIONS
═══════════════════════════════════════════════════════════════

Return JSON with this EXACT structure (no markdown, no extra text):

  {{
  "matrix": [
      {{
      "mentor_id": "EXACT_CODE_FROM_{MENTOR_COLUMNS['id']}",
      "scores": [
        {{"mentee_id": "EXACT_CODE", "percentage": 82}},
        {{"mentee_id": "EXACT_CODE", "percentage": 45}},
        {{"mentee_id": "EXACT_CODE", "percentage": 15}},
        {{"mentee_id": "EXACT_CODE", "percentage": 68}},
        ... (MUST include ALL mentees, even with low scores)
      ]
    }},
    ... (MUST include ALL mentors)
  ]
}}

CRITICAL RULES:
1. Every mentor MUST have scores for ALL mentees
2. If mentor has 100 mentees, scores array MUST have 100 entries
3. Use actual calculated scores - even if 5%, 10%, 15%
4. DO NOT skip combinations
5. DO NOT use null, undefined, or omit entries
6. Low scores (5-40%) are expected and correct for bad matches

VERIFICATION BEFORE SENDING RESPONSE:
✓ Did I include ALL mentors in the response?
✓ Does each mentor have ALL mentee scores?
✓ Did I include low scores (5-30%) for obviously bad matches?
✓ Did I apply the field alignment matrix strictly (not creatively)?
✓ Did I count actual keyword overlaps (not thematic similarity)?
✓ Did I apply the 40% cap for different fields?
✓ Did I apply the 55% cap for zero keyword overlap?

═══════════════════════════════════════════════════════════════
⚠️ FINAL REMINDER: JSON ONLY - NO TEXT ⚠️
═══════════════════════════════════════════════════════════════

Your ENTIRE response must be ONLY the JSON object.

❌ BAD - Do NOT do this:
"I will now calculate the scores. Here's the JSON: {{"matrix": [...]}}"

❌ BAD - Do NOT do this:
"To generate matches, I will use the scoring criteria... Let me compute..."

✅ GOOD - Do this (ONLY JSON, nothing else):
{{"matrix": [{{"mentor_id": "M001", "scores": [{{"mentee_id": "E001", "percentage": 75}}]}}]}}

Remember:
• First character of response: {{
• Last character of response: }}
• No text before or after
• No explanations
• No markdown
• No process descriptions

BEGIN YOUR RESPONSE NOW WITH: {{"matrix": ["""
    
    # Assemble the full prompt
    full_prompt = json_header + custom_rubric + output_footer
    return full_prompt


def create_default_prompt():
    """Extract only the EDITABLE rubric section (middle part of the full prompt)"""
    # Get the full prompt
    training_file_ids = st.session_state.get('training_file_ids', [])
    full_prompt = create_matrix_prompt(use_file_search=len(training_file_ids) > 0)
    
    # Extract only the editable section (between JSON header and output format footer)
    # Start marker: "You are evaluating mentor-mentee"
    # End marker: "OUTPUT FORMAT - MUST INCLUDE"
    start_marker = "You are evaluating mentor-mentee"
    end_marker = "OUTPUT FORMAT - MUST INCLUDE"
    
    start_idx = full_prompt.find(start_marker)
    end_idx = full_prompt.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # Extract the editable rubric (everything between markers)
        editable_section = full_prompt[start_idx:end_idx].strip()
        return editable_section
    
    # Fallback if markers not found
    return full_prompt

