# =====================================================================
# COLUMN CONFIGURATION
# Define all column names used in mentee and mentor data files
# =====================================================================

MENTOR_COLUMNS = {
    'id': 'Mentor ID',
    'affiliation': 'Mentor Affiliation',
    'country': 'Mentor Country of affiliated institution',
    'current_position': 'Mentor Current position',  # NEW: Current job title/role
    'institution_type': 'Mentor Type of Institution',
    'country_of_origin': 'Mentor Country of origin',  # NEW: Nationality/birth country
    'alma_mater': 'Mentor Alma Mater',
    'education_level': 'Mentor Highest Educational Level',
    'field': 'Mentor Field of expertise',
    'specialization': 'Mentor Specialization',
    'experience_years': 'Mentor Years of Professional Experience in his her Field',
    'career_dev_experience': 'Mentor experience in career development'  # NEW: Mentoring expertise (not used in matching per user request)
}

# Training file specific columns (not in regular mentor data)
TRAINING_COLUMNS = {
    'good_match': 'Mentee considered good mentor assigned'  # Only in training data
}

MENTEE_COLUMNS = {
    'id': 'Mentee ID',
    'affiliation': 'Mentee Affiliation',
    'country': 'Mentee Country of your affiliated institution',
    'education_level': 'Mentee Highest educational level completed',
    'current_program': 'Mentee Current education program',
    'field': 'Mentee Field of expertise',
    'specialization': 'Mentee Specialization',
    'languages': 'Mentee Languages spoken fluently',
    'guidance_areas': 'Mentee Areas where guidance is needed',
    'career_goals': 'Mentee Career goals for the next 2 years',
    'other_info': 'Mentee other relevant info'  # NEW: Additional relevant information
}

# Helper functions for UI improvements (Improvements: 15, 16, 20, 23, 24)
def get_prompt_presets():
    """Get configuration presets for matching (Improvement 16)"""
    presets = {
        "Quick Start ⚡": {
            "description": "Fast & cheap matching with good quality",
            "model": "gpt-4o-mini",
            "batch_size": 20,
            "prompt_type": "standard"
        },
        "Balanced ⭐": {
            "description": "Best balance of quality and speed (Recommended)",
            "model": "gpt-4o",
            "batch_size": 15,
            "prompt_type": "standard"
        },
        "High Quality 🎯": {
            "description": "Advanced reasoning for complex matching logic",
            "model": "o1-mini",
            "batch_size": 10,
            "prompt_type": "detailed"
        },
        "Maximum Quality 💎": {
            "description": "Best possible matches (slowest, most expensive)",
            "model": "o1-preview",
            "batch_size": 10,
            "prompt_type": "detailed"
        }
    }
    return presets

