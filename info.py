# =====================================================================
# COLUMN CONFIGURATION
# Define all column names used in mentee and mentor data files
# =====================================================================

MENTOR_COLUMNS = {
    'id': 'Mentor ID',
    'affiliation': 'Mentor Affiliation',
    'country': 'Mentor Country of affiliated institution',
    'current_position': 'Mentor Current position',
    'institution_type': 'Mentor Type of Institution',
    'country_of_origin': 'Mentor Country of origin',
    'alma_mater': 'Mentor Alma Mater',
    'education_level': 'Mentor Highest Educational Level',
    'field': 'Mentor Field of expertise',
    'specialization': 'Mentor Specialization',
    'experience_years': 'Mentor Years of Professional Experience in his her Field',
    'career_dev_experience': 'Mentor experience in career development'
}

MENTEE_COLUMNS = {
    'id': 'Mentee ID',
    'affiliation': 'Mentee Affiliation',
    'country': 'Mentee Country of your affiliated university',
    'education_level': 'Mentee Highest educational level completed',
    'current_program': 'Mentee Current education program',
    'field': 'Mentee Field of expertise',
    'specialization': 'Mentee Specialization',
    'languages': 'Mentee Languages spoken fluently',
    'guidance_areas': 'Mentee Areas where guidance is needed',
    'career_goals': 'Mentee Career goals for the next 2 years',
    'other_info': 'Mentee other relevant info'
}

# =====================================================================
# MODEL CONFIGURATION
# Configuration for different OpenAI model families
# =====================================================================

MODEL_CONFIGS = {
    'gpt-5.1': {
        'supports_temperature': True,
        'supports_top_p': True,
        'supports_structured_output': True,
        'supports_stateful': False,  # Disabled per user request (high rate limits)
        'token_param': 'max_output_tokens',
        'family': 'gpt-5-full',
        'tpm_limit': 30000,  # Tokens per minute
        'rpm_limit': 500,    # Requests per minute
        'tpd_limit': 900000  # Tokens per day
    },
    'gpt-5-mini': {
        'supports_temperature': False,
        'supports_top_p': False,
        'supports_structured_output': False,  # Reasoning model - no structured JSON schema support
        'supports_stateful': True,
        'is_reasoning_model': True,  # Returns ResponseReasoningItem, not standard text
        'token_param': 'max_output_tokens',
        'family': 'gpt-5-mini',
        'tpm_limit': 500000,
        'rpm_limit': 500,
        'tpd_limit': 5000000
    },
    'gpt-5-nano': {
        'supports_temperature': False,
        'supports_top_p': False,
        'supports_structured_output': False,  # Reasoning model - no structured JSON schema support
        'supports_stateful': True,
        'is_reasoning_model': True,  # Returns ResponseReasoningItem, not standard text
        'token_param': 'max_output_tokens',
        'family': 'gpt-5-nano',
        'tpm_limit': 200000,
        'rpm_limit': 500,
        'tpd_limit': 2000000
    },
    'gpt-4.1': {
        'supports_temperature': True,
        'supports_top_p': True,
        'supports_structured_output': True,
        'supports_stateful': False,  # Disabled per user request (high rate limits)
        'token_param': 'max_output_tokens',
        'family': 'gpt-4.1-full',
        'tpm_limit': 30000,
        'rpm_limit': 500,
        'tpd_limit': 900000
    },
    'gpt-4.1-mini': {
        'supports_temperature': True,
        'supports_top_p': True,
        'supports_structured_output': True,
        'supports_stateful': True,
        'token_param': 'max_output_tokens',
        'family': 'gpt-4.1-mini',
        'tpm_limit': 200000,
        'rpm_limit': 500,
        'tpd_limit': 2000000
    },
    'gpt-4.1-nano': {
        'supports_temperature': True,
        'supports_top_p': True,
        'supports_structured_output': True,
        'supports_stateful': True,
        'token_param': 'max_output_tokens',
        'family': 'gpt-4.1-nano',
        'tpm_limit': 200000,
        'rpm_limit': 500,
        'tpd_limit': 2000000
    },
    'default': {
        'supports_temperature': False,  # Conservative default
        'supports_top_p': False,  # Conservative default
        'supports_structured_output': True,
        'supports_stateful': True,
        'token_param': 'max_output_tokens',
        'family': 'unknown',
        'tpm_limit': 30000,  # Conservative default
        'rpm_limit': 500,
        'tpd_limit': 900000
    }
}

# =====================================================================
# MODEL TOKEN LIMITS
# Maximum tokens per model for validation
# =====================================================================

MODEL_TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "gpt-5": 200000,
    "gpt-5.1": 200000,
    "gpt-5.1-mini": 200000,
    "gpt-5-mini": 200000,
    "gpt-5-nano": 128000,
    "gpt-5.1-reasoning": 200000,
    "chatgpt-4o-latest": 128000,
}

# =====================================================================
# API PRICING
# Pricing per 1M tokens (source: https://openai.com/api/pricing/)
# =====================================================================

API_PRICING = {
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "o1-preview": {"prompt": 15.00, "completion": 60.00},
    "o1-mini": {"prompt": 3.00, "completion": 12.00},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-4-turbo-preview": {"prompt": 10.00, "completion": 30.00},
    "gpt-4": {"prompt": 30.00, "completion": 60.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "gpt-5": {"prompt": 20.00, "completion": 60.00},
    "gpt-5.1": {"prompt": 20.00, "completion": 60.00},
}

# =====================================================================
# FIELD COMPATIBILITY
# Related field groups for validation
# =====================================================================

RELATED_FIELD_GROUPS = [
    {'biology', 'molecular biology', 'biochemistry', 'biophysics', 'cell biology', 'structural biology'},
    {'computer science', 'software engineering', 'data science', 'information technology', 'artificial intelligence'},
    {'physics', 'applied physics', 'quantum physics', 'astrophysics'},
    {'chemistry', 'organic chemistry', 'analytical chemistry', 'physical chemistry', 'inorganic chemistry'},
    {'medicine', 'clinical medicine', 'medical sciences', 'biomedical sciences'},
    {'engineering', 'mechanical engineering', 'electrical engineering', 'civil engineering'},
    {'biotechnology', 'bioengineering', 'biomedical engineering'},
    {'mathematics', 'applied mathematics', 'statistics', 'computational mathematics'},
    {'neuroscience', 'cognitive science', 'behavioral neuroscience'},
    {'genetics', 'genomics', 'molecular genetics'},
    {'immunology', 'microbiology', 'virology'},
    {'pharmacology', 'pharmacy', 'pharmaceutical sciences'},
]

# =====================================================================
# EXCLUDED MODEL PREFIXES
# Model prefixes to filter out (non-chat models)
# =====================================================================

EXCLUDED_MODEL_PREFIXES = (
    'text-embedding-', 'embedding-', 
    'tts-', 'whisper-', 
    'dall-e-', 'davinci-', 'curie-', 'babbage-', 'ada-'
)

# =====================================================================
# PROMPT TEMPLATES
# JSON format instructions and field compatibility matrix
# =====================================================================

# JSON Footer - Output format specifications for models that need explicit JSON format guidance
JSON_FOOTER = """
═══════════════════════════════════════════════════════════════
OUTPUT FORMAT - MUST INCLUDE ALL COMBINATIONS
═══════════════════════════════════════════════════════════════

Return JSON with this EXACT structure (no markdown, no extra text):

  {{
  "matrix": [
      {{
      "mentor_id": "EXACT_CODE_FROM_Mentor ID",
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

# Field Compatibility Matrix - Authoritative field matching rules
FIELD_COMPATIBILITY_MATRIX_TEXT = """
═══════════════════════════════════════════════════════════════
FIELD COMPATIBILITY MATRIX (AUTHORITATIVE - USE THIS EXACTLY)
═══════════════════════════════════════════════════════════════

⚠️ WILDCARD RULE: If mentee field == mentor field (case-insensitive) → 40 POINTS
   This covers ALL exact matches, including fields not explicitly listed below.

40 POINTS - IDENTICAL FIELDS (explicit list + wildcard above):

=== CORE BROAD CATEGORIES ===
• Accounting = Accounting
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

=== LIFE SCIENCES - MOLECULAR & CELLULAR ===
• Advanced Chemistry = Advanced Chemistry
• Analytical Chemistry = Analytical Chemistry
• Biochemistry = Biochemistry
• Biochemistry and Molecular Biology = Biochemistry and Molecular Biology
• Biodiversity = Biodiversity
• Bioinformatics = Bioinformatics
• Biology = Biology
• Biophysics = Biophysics
• Cell Biology = Cell Biology
• Computational Biology = Computational Biology
• Data Science = Data Science
• Ecology = Ecology
• Environmental Sciences = Environmental Sciences
• Genetics = Genetics
• Genomics = Genomics
• Molecular Biology = Molecular Biology
• Molecular Life Sciences = Molecular Life Sciences

=== LIFE SCIENCES - BIOMEDICAL & MEDICAL ===
• Biomedical Sciences = Biomedical Sciences
• Biomedicine and Biotechnology = Biomedicine and Biotechnology
• Cancer Research = Cancer Research
• Clinical Psychology = Clinical Psychology
• Drug Discovery = Drug Discovery
• Epidemiology = Epidemiology
• Immunology = Immunology
• Material Science = Material Science
• Medical Sciences = Medical Sciences
• Medical Technology = Medical Technology
• Medicine = Medicine
• Microbiology = Microbiology
• Neuroscience = Neuroscience
• Neuroscience and Neurosurgery = Neuroscience and Neurosurgery
• Nutrition and Dietetics = Nutrition and Dietetics
• Oncology = Oncology
• Pharmacology = Pharmacology
• Pharmacology and Physiology = Pharmacology and Physiology
• Pharmacy and Biomedical Sciences = Pharmacy and Biomedical Sciences
• Physiology = Physiology
• Psychiatry = Psychiatry
• Public Health = Public Health
• Translational Medicine = Translational Medicine

=== ENGINEERING SPECIALIZATIONS ===
• Aeronautics = Aeronautics (maps to Engineering cluster)
• Applied Linguistics = Applied Linguistics (maps to Linguistics)
• Biological Engineering = Biological Engineering (maps to Bioengineering)
• Bioprocess Engineering = Bioprocess Engineering (maps to Engineering cluster)
• Civil Engineering = Civil Engineering (maps to Engineering cluster)
• Electronic Engineering = Electronic Engineering (maps to Electrical Engineering)
• Environmental Engineering = Environmental Engineering
• Forest Engineering = Forest Engineering (maps to Engineering cluster)
• Software Engineering = Software Engineering (maps to Computer Science)
• Telecom Engineering = Telecom Engineering (maps to Telecommunications Engineering)

=== OTHER SPECIALIZED FIELDS ===
• Agricultural Sciences = Agricultural Sciences
• Agrifood and Biosystems Science and Engineering = Agrifood and Biosystems Science and Engineering
• Statistical Physics and Complex Systems = Statistical Physics and Complex Systems

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

CLUSTER G: Engineering Fields
• Engineering ↔ Mechanical Engineering
• Engineering ↔ Electrical Engineering
• Mechanical Engineering ↔ Electrical Engineering
• Electrical Engineering ↔ Telecommunications Engineering
• Chemical Engineering ↔ Chemistry
• Energy Engineering ↔ Renewable Energy
• Energy Engineering ↔ Electrical Engineering
• Engineering ↔ Civil Engineering
• Engineering ↔ Electronic Engineering
• Engineering ↔ Aeronautics
• Engineering ↔ Bioprocess Engineering
• Engineering ↔ Forest Engineering

CLUSTER H: Psychology & Behavioral Sciences
• Psychology ↔ Psychological Neuroscience
• Psychology ↔ Behavioral Science
• Psychological Neuroscience ↔ Behavioral Science
• Behavioral Science ↔ Sociology

CLUSTER I: Economics & Business
• Economics ↔ Economics/Entrepreneurship
• Economics ↔ Accounting
• Economics/Entrepreneurship ↔ Accounting

CLUSTER J: Design & Architecture
• Architecture ↔ Interior Design
• Fashion Design ↔ Interior Design

CLUSTER K: Language & Communication
• Linguistics ↔ Literature Studies
• Linguistics ↔ Communication Studies
• Linguistics ↔ Applied Linguistics
• Literature Studies ↔ Communication Studies

CLUSTER L: Computer Science & Software
• Computer Science ↔ Software Engineering
• Computer Science ↔ Data Science
• Computer Science ↔ Business Informatics

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

Cross Life Sciences-Engineering:
• Biomedicine ↔ Chemical Engineering
• Biotechnology ↔ Chemical Engineering
• Chemistry ↔ Chemical Engineering
• Pharmaceutical Sciences ↔ Chemical Engineering
• Biomedical Engineering ↔ Mechanical Engineering
• Biomedical Engineering ↔ Electrical Engineering
• Bioengineering ↔ Biological Engineering
• Bioengineering ↔ Bioprocess Engineering

Cross Life Sciences-Computer Science:
• Biomedicine ↔ Computer Science (bioinformatics)
• Biotechnology ↔ Computer Science (biotech data)
• Health Sciences ↔ Computer Science (health informatics)

Cross Engineering-Computer Science:
• Engineering ↔ Computer Science (software engineering)
• Electrical Engineering ↔ Computer Science (hardware-software)
• Telecommunications Engineering ↔ Computer Science (network engineering)
• Computer Science ↔ Business Informatics
• Software Engineering ↔ Computer Science

Cross Engineering-Physics/Math:
• Engineering ↔ Physics
• Mechanical Engineering ↔ Physics
• Electrical Engineering ↔ Physics
• Engineering ↔ Mathematics
• Computer Science ↔ Mathematics

Cross Psychology-Health:
• Psychology ↔ Health Sciences (mental health)
• Psychological Neuroscience ↔ Biomedicine (neuroscience)
• Behavioral Science ↔ Health Sciences (behavioral health)

Cross Social Sciences-Humanities:
• Sociology ↔ Psychology
• Education ↔ Psychology
• Sociology ↔ Communication Studies
• Law ↔ Sociology
• Law ↔ Economics

Cross Business-Technology:
• Economics/Entrepreneurship ↔ Business Informatics
• Economics ↔ Business Informatics
• Economics/Entrepreneurship ↔ Computer Science (tech entrepreneurship)

Cross Business-Design:
• Fashion Design ↔ Economics/Entrepreneurship (fashion business)
• Architecture ↔ Economics/Entrepreneurship (real estate, construction)

Cross Design-Engineering:
• Architecture ↔ Engineering (structural, civil)
• Architecture ↔ Mechanical Engineering (HVAC, building systems)
• Interior Design ↔ Architecture

Cross Environmental Sciences:
• Environmental Science ↔ Agricultural Sciences
• Environmental Science ↔ Chemistry (environmental chemistry)
• Agricultural Sciences ↔ Biotechnology (agribiotechnology)

Cross Health-Sports:
• Sports Science ↔ Health Sciences
• Sports Science ↔ Psychology (sports psychology)
• Sports Science ↔ Biomedicine (sports medicine)

Cross Education Connections:
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
• Fashion Design ↔ Biomedicine → 0 POINTS
• Literature Studies ↔ Engineering → 0 POINTS
• Law ↔ Chemistry → 0 POINTS
• Sports Science ↔ Linguistics → 0 POINTS
• Accounting ↔ Pharmaceutical Sciences → 0 POINTS
• Interior Design ↔ Physics → 0 POINTS
• Telecommunications Engineering ↔ Psychology → 0 POINTS

⚠️ ANY PAIRING NOT EXPLICITLY LISTED ABOVE → 0 POINTS
When in doubt, assign 0 points. Do NOT invent relationships.

✅ IMPORTANT: The matrix above now covers ALL 37 fields in your data, including:
   Life Sciences, Engineering, Computer Science, Business, Social Sciences,
   Humanities, Design, Environmental Sciences, and more.
"""

