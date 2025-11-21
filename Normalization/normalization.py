"""
Normalization utilities for cleaning and standardizing data.
Includes dictionaries and pattern matching for common transformations.
"""

import re
import unicodedata
from typing import Tuple, List, Optional

# ============================================================================
# PATTERN DICTIONARY - Common transformations
# ============================================================================

EDUCATION_LEVEL_PATTERNS = {
    # PhD variations
    r'\bph\.?\s*d\.?\b': 'PhD',
    r'\bphd\b': 'PhD',
    r'\bp\.?\s*h\.?\s*d\.?\b': 'PhD',
    r'\bdoctorate\b': 'PhD',
    r'\bdoctor\b': 'PhD',
    r'\bdr\.?\b': 'PhD',
    r'^p$': 'PhD',  # Single "P" as mentioned by user
    
    # Masters variations
    r'\bmaster\'?s\b': 'Masters',
    r'\bm\.?\s*s\.?\s*c\.?\b': 'Masters',
    r'\bmsc\b': 'Masters',
    r'\bma\b': 'Masters',
    r'\bms\b': 'Masters',
    r'\bm\.?\s*a\.?\b': 'Masters',
    r'\bm\.?\s*s\.?\b': 'Masters',
    
    # Bachelors variations
    r'\bbachelor\'?s\b': 'Bachelors',
    r'\bb\.?\s*a\.?\b': 'Bachelors',
    r'\bb\.?\s*s\.?\s*c\.?\b': 'Bachelors',
    r'\bbsc\b': 'Bachelors',
    r'\bba\b': 'Bachelors',
    r'\bbs\b': 'Bachelors',
    r'^b$': 'Bachelors',  # Single "B" as mentioned by user
    
    # High School
    r'\bhigh\s+school\b': 'High School',
    r'\bsecondary\s+school\b': 'High School',
}

# ============================================================================
# UNIVERSITY NAME DICTIONARY
# ============================================================================

UNIVERSITY_NAME_MAPPING = {
    # European University variations
    'european university': 'European University of Madrid',
    'european unifversity of madrid': 'European University of Madrid',  # Typo
    'universidad europea de madrid': 'European University of Madrid',
    'universidad europea madrid': 'European University of Madrid',
    'universidad europea': 'European University of Madrid',
    'universidad europea ': 'European University of Madrid',
    'european university of madrid (universidad europea de madrid)': 'European University of Madrid',
    'universidad europea de madrid ': 'European University of Madrid',
    'universidad europea de madrid': 'European University of Madrid',
    
    # University of Seville
    'university of seville': 'University of Seville',
    'university of seville (universidad de sevilla)': 'University of Seville',
    'universidad de sevilla': 'University of Seville',
    
    # University of Valladolid
    'university of valladolid': 'University of Valladolid',
    'universidad de valladolid': 'University of Valladolid',
    'university of valladolid ': 'University of Valladolid',
    
    # University of Jaen (remove accent)
    'university of jaen': 'University of Jaen',
    'university of jaén': 'University of Jaen',
    'universidad de jaen': 'University of Jaen',
    'universidad de jaén': 'University of Jaen',
    
    # University of Malaga (remove accent)
    'university of malaga': 'University of Malaga',
    'university of málaga': 'University of Malaga',
    'university of malaga ': 'University of Malaga',
    'universidad de malaga': 'University of Malaga',
    'universidad de málaga': 'University of Malaga',
    
    # University of Porto
    'university of porto': 'University of Porto',
    'universidade do porto': 'University of Porto',
    'bsc at university of porto': 'University of Porto',
    'fcup, porto, portugal': 'University of Porto',  # Faculty of Sciences, University of Porto
    
    # University of Valladolid (many variations)
    'university of valladolid': 'University of Valladolid',
    'universidad de valladolid': 'University of Valladolid',
    'universidad de valladolid - spain': 'University of Valladolid',
    'universidad de valladolid, spain': 'University of Valladolid',
    'universidad de valladolid - escuela tecnica superior de arquitectura': 'University of Valladolid',
    'univeristy of valladolid': 'University of Valladolid',  # Typo
    'universidad de valladolid - spain\nie business school - spain': 'University of Valladolid',  # Multiple, take first
    
    # Complutense University of Madrid
    'complutense university': 'Complutense University of Madrid',
    'complutense university of madrid': 'Complutense University of Madrid',
    'complutense university of madrid (spain)': 'Complutense University of Madrid',
    'universidad complutense de madrid': 'Complutense University of Madrid',
    'universidad complutense de madrid (spain)': 'Complutense University of Madrid',
    
    # Autonomous University of Madrid
    'carlos iii university madrid': 'Autonomous University of Madrid',  # Carlos III is different, but mapping for now
    'universidad autonoma de madrid': 'Autonomous University of Madrid',
    'universidad autónoma de madrid': 'Autonomous University of Madrid',
    'universidad autonoma de madrid, spain': 'Autonomous University of Madrid',
    'uam spain': 'Autonomous University of Madrid',
    
    # University of Seville
    'university of seville': 'University of Seville',
    'universidad de sevilla': 'University of Seville',
    'universidad de sevilla, spain': 'University of Seville',
    'university of sevilla': 'University of Seville',
    
    # University of Valencia
    'universidad de valencia': 'University of Valencia',
    'universidad de valencia, spain': 'University of Valencia',
    'universitat de valencia': 'University of Valencia',
    'universitat de valencia - spain': 'University of Valencia',
    'universidad de valencia (spain)': 'University of Valencia',
    'university of valencia, spain': 'University of Valencia',
    
    # University of Oviedo
    'university of oviedo': 'University of Oviedo',
    'universidad de oviedo': 'University of Oviedo',
    'universidad de oviedo, spain': 'University of Oviedo',
    
    # University of La Laguna
    'universidad de la laguna': 'University of La Laguna',
    'universidad de la laguna (spain)': 'University of La Laguna',
    'universidad de la laguna, spain': 'University of La Laguna',
    
    # University of Malaga
    'universidad de malaga': 'University of Malaga',
    'universidad de málaga': 'University of Malaga',
    'universidad de málaga (spain)': 'University of Malaga',
    
    # University of Navarra
    'university of navarra': 'University of Navarra',
    'universidad de navarra': 'University of Navarra',
    'university of navarra spain': 'University of Navarra',
    
    # University of Zaragoza
    'universidad de zaragoza': 'University of Zaragoza',
    
    # University of Leon
    'universidad de leon': 'University of Leon',
    'universidad de leon, spain': 'University of Leon',
    
    # University of Jaen
    'university of jaén': 'University of Jaen',
    'university of jaen (spain)': 'University of Jaen',
    
    # University of Santiago de Compostela
    'universidade de santiago de compostela': 'University of Santiago de Compostela',
    'universidade de santiago de compostela, spain': 'University of Santiago de Compostela',
    'university of santiago de compostela, spain': 'University of Santiago de Compostela',
    
    # University of Vigo
    'universidade de vigo': 'University of Vigo',
    'universidade de vigo, spain': 'University of Vigo',
    
    # University of Minho (Portugal)
    'university of minho': 'University of Minho',
    'universidade do minho': 'University of Minho',
    'university of minho, portugal': 'University of Minho',
    'universidade do minho, portugal': 'University of Minho',
    
    # University of Coimbra (Portugal)
    'university of coimbra': 'University of Coimbra',
    'university of coimbra, portugal': 'University of Coimbra',
    
    # University of Lisbon (Portugal)
    'university of lisbon': 'University of Lisbon',
    'university of lisbon (portugal)': 'University of Lisbon',
    'university of lisbon, portugal': 'University of Lisbon',
    
    # University of Aveiro (Portugal)
    'university of aveiro': 'University of Aveiro',
    'university of aveiro, portugal': 'University of Aveiro',
    
    # Other Spanish universities
    'university pompeu fabra': 'Pompeu Fabra University',
    'university of cantabria': 'University of Cantabria',
    'uned': 'UNED',
    'uned&spain': 'UNED',
    
    # Portuguese universities
    'instituto superior técnico': 'Technical Superior Institute',
    'instituto superior técnico - portugal': 'Technical Superior Institute',
    'instituto superior tecnico': 'Technical Superior Institute',  # Without accent
    'institute superior tecnico': 'Technical Superior Institute',  # After translation
    'instituto superior de agronomia': 'Instituto Superior de Agronomia',
    'portuguese catholic university': 'Portuguese Catholic University',
    'universidade católica portuguesa': 'Portuguese Catholic University',
    'biotechnology school (esb)': 'Portuguese Catholic University',  # ESB is part of Portuguese Catholic University
    'biotechnology school (esb), portuguese catholic university': 'Portuguese Catholic University',
    'biotechnology school (esb), portuguese catholic university (universidade católica portuguesa)': 'Portuguese Catholic University',
    'ispa - instituto universitário': 'ISPA - Instituto Universitário',
    'ispa - instituto universitario': 'ISPA - Instituto Universitário',  # Without accent
    'ispa - institute universitario': 'ISPA - Instituto Universitário',  # After translation
    'universidade lusófona de humanidades e tecnologias': 'Lusófona University',
    'universidade lusofona de humanidades e tecnologias': 'Lusófona University',  # Without accent
    'universidade lusofona of humanidades e tecnologias': 'Lusófona University',  # After translation
    'utad': 'Universidade de Trás-os-Montes e Alto Douro',
    'portugal, utad': 'Universidade de Trás-os-Montes e Alto Douro',
    
    # Spanish universities - fix translation issues
    'universitat de barcelona': 'University of Barcelona',
    'universit de barcelona': 'University of Barcelona',  # Missing 'at' - fix
    'universit of barcelona': 'University of Barcelona',  # After incorrect translation
    'universitat de valencia': 'University of Valencia',
    'universit de valencia': 'University of Valencia',  # Missing 'at' - fix
    'universit of valencia': 'University of Valencia',  # After incorrect translation
    'universidad politecnica de madrid': 'Polytechnic University of Madrid',
    'universidad politecnica de madrid (upm)': 'Polytechnic University of Madrid',
    'university politecnica of madrid (upm)': 'Polytechnic University of Madrid',  # After translation
    'univerity of huelva': 'University of Huelva',  # Typo fix
    'university of huelva': 'University of Huelva',
    
    # International universities
    'the institute of cancer research': 'The Institute of Cancer Research',
    'institute of cancer research': 'The Institute of Cancer Research',
    'uepa - state university of para': 'State University of Pará',
    'state university of para': 'State University of Pará',
    'st cloud state university': 'St Cloud State University',
    'master\'s st cloud state university': 'St Cloud State University',
    'medical college of wisconsin': 'Medical College of Wisconsin',
    'phd the medical college of wisconsin': 'Medical College of Wisconsin',
    'wright state university': 'Wright State University',
    'wright state u': 'Wright State University',  # Abbreviation
    'university of zagreb': 'University of Zagreb',
    'masters in croatia- university of zagreb': 'University of Zagreb',
    'university of leeds': 'University of Leeds',
    'u. of leeds': 'University of Leeds',  # Abbreviation
    'pontifical university of salamanca': 'Pontifical University of Salamanca',
    'indian institute of science education and research': 'Indian Institute of Science Education and Research',
    'indian institute of science education': 'Indian Institute of Science Education and Research',  # Incomplete
    'from indian institute of science education': 'Indian Institute of Science Education and Research',  # Fragment
    'iiser mohali': 'IISER Mohali',
    'research mohave (iiser mohali)': 'IISER Mohali',
    'universita del studi di cagliari': 'University of Cagliari',  # Already in dict but fix translation
    'universita of the studi di cagliari': 'University of Cagliari',  # After incorrect translation
    
    # International universities (add more as needed)
    'delft university of technology': 'Delft University of Technology',
    'radboud university': 'Radboud University',
    'dalhousie university': 'Dalhousie University',
    'central south university': 'Central South University',
    'zhejiang university': 'Zhejiang University',
    'smith college': 'Smith College',
    'university of east anglia': 'University of East Anglia',
    'university of glasgow': 'University of Glasgow',
    'university  of glasgow, uk': 'University of Glasgow',
    'westminster college oxford': 'Westminster College Oxford',
    'politecnico di milano': 'Politecnico di Milano',
    'università del studi di cagliari': 'University of Cagliari',
    'university politehnica of bucharest': 'University Politehnica of Bucharest',
    'universitat autònoma de barcelona': 'Autonomous University of Barcelona',
    'uab': 'Autonomous University of Barcelona',
    'upc': 'Polytechnic University of Catalonia',
    
    # Multiple institutions (take first one)
    'maastricth univeristy, netherlands & university of seville, spain': 'University of Seville',  # Typo in first, take second
    'libre university (colombia). university of valladolid (spain)': 'University of Valladolid',
    
    # Politecnical Institute of Braganza (user specified name)
    'polytechnic institute of bragança': 'Politecnical Institute of Braganza',
    'polytechnic institute of braganca': 'Politecnical Institute of Braganza',
    'polytechnic institute of braganza': 'Politecnical Institute of Braganza',
    'politecnical institute of braganza': 'Politecnical Institute of Braganza',
    'instituto politécnico de bragança': 'Politecnical Institute of Braganza',
    'instituto politécnico de braganca': 'Politecnical Institute of Braganza',
    'instituto politécnico de braganza': 'Politecnical Institute of Braganza',
    'instituto politénico de bragança': 'Politecnical Institute of Braganza',  # Typo
    'instituto politénico de braganca': 'Politecnical Institute of Braganza',  # Typo
    'instituto politénico de braganza': 'Politecnical Institute of Braganza',  # Typo
    'institute politecnico of braganca': 'Politecnical Institute of Braganza',  # After translation
    'institute politecnico of braganza': 'Politecnical Institute of Braganza',  # After translation
    'institute politecnico of bragança': 'Politecnical Institute of Braganza',  # After translation with accent
    'institute politenico of braganca': 'Politecnical Institute of Braganza',  # After translation with typo preserved
    'institute politenico of braganza': 'Politecnical Institute of Braganza',  # After translation with typo preserved
    'institute politenico of bragança': 'Politecnical Institute of Braganza',  # After translation with typo preserved
    'polytechnic institute of bragança/ faculty of engineering of the university of porto': 'Politecnical Institute of Braganza',
    'polytechnic institute of bragança - ipb': 'Politecnical Institute of Braganza',
    'instituto politécnico de bragança ': 'Politecnical Institute of Braganza',
    'institute politecnico of bragança ': 'Politecnical Institute of Braganza',  # With trailing space
}

# ============================================================================
# COUNTRY NAME DICTIONARY
# ============================================================================

COUNTRY_NAME_MAPPING = {
    'españa': 'Spain',
    'espana': 'Spain',
    'spain': 'Spain',
    'portugal': 'Portugal',
}

# Comprehensive country name mapping (for mentors - all countries)
MENTOR_COUNTRY_MAPPING = {
    # USA variations
    'united states of america': 'USA',
    'united states': 'USA',
    'usa': 'USA',
    'usa ': 'USA',
    'usa,': 'USA',
    
    # UK variations
    'united kingdom': 'UK',
    'uk': 'UK',
    'uk,': 'UK',
    
    # Netherlands variations
    'the netherlands': 'Netherlands',
    'netherlands': 'Netherlands',
    
    # Spain variations
    'spain': 'Spain',
    'españa': 'Spain',
    'espana': 'Spain',
    'spain ': 'Spain',
    'spain,': 'Spain',
    
    # Portugal variations
    'portugal': 'Portugal',
    'portugal ': 'Portugal',
    'portugal,': 'Portugal',
    
    # Italy variations
    'italy': 'Italy',
    'italia': 'Italy',
    'italy,': 'Italy',
    
    # Germany variations
    'germany': 'Germany',
    'germany ': 'Germany',
    'germany,': 'Germany',
    
    # Other countries (standardize capitalization)
    'austria': 'Austria',
    'belgium': 'Belgium',
    'brazil': 'Brazil',
    'canada': 'Canada',
    'colombia': 'Colombia',
    'denmark': 'Denmark',
    'estonia': 'Estonia',
    'france': 'France',
    'india': 'India',
    'ireland': 'Ireland',
    'luxembourg': 'Luxembourg',
    'norway': 'Norway',
    'qatar': 'Qatar',
    'singapore': 'Singapore',
    'sweden': 'Sweden',
    'switzerland': 'Switzerland',
    'switzerland ': 'Switzerland',
    'taiwan': 'Taiwan',
    'czech republic': 'Czech Republic',
    'armenia': 'Armenia',
    'sri lanka': 'Sri Lanka',
    'croatia': 'Croatia',
    'iran': 'Iran',
    'nigeria': 'Nigeria',
    'romania': 'Romania',
    'china': 'China',
    
    # Special cases
    'no affiliation': 'N/A',
    'none': 'N/A',
    'n/a': 'N/A',
}

# Common city-country patterns to remove
COUNTRY_CITY_PATTERNS = [
    r'^[^,]+,\s*(spain|portugal|españa|espana)$',  # "City, Country"
    r'^[^,]+,\s*(spain|portugal|españa|espana)\s*$',  # With trailing space
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def remove_accents(text: str) -> str:
    """Remove accents and special characters from text."""
    if not text or pd.isna(text):
        return text
    
    # Normalize to NFD (decomposed form) and remove combining characters
    text = unicodedata.normalize('NFD', str(text))
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace - trim and collapse multiple spaces."""
    if not text or pd.isna(text):
        return text
    
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    return text


def translate_spanish_to_english(text: str) -> str:
    """Translate common Spanish/Catalan words to English."""
    if not text or pd.isna(text):
        return text
    
    text = str(text)
    # Translate "Universitat" (Catalan) to "University" - must come before "Universidad"
    text = re.sub(r'\buniversitat\b', 'University', text, flags=re.IGNORECASE)
    # Translate "Universidad" (Spanish) to "University"
    text = re.sub(r'\buniversidad\b', 'University', text, flags=re.IGNORECASE)
    text = re.sub(r'\buniversidades\b', 'Universities', text, flags=re.IGNORECASE)
    text = re.sub(r'\binstituto\b', 'Institute', text, flags=re.IGNORECASE)
    text = re.sub(r'\binstitutos\b', 'Institutes', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfacultad\b', 'Faculty', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfacultades\b', 'Faculties', text, flags=re.IGNORECASE)
    text = re.sub(r'\bde\b', 'of', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdel\b', 'of the', text, flags=re.IGNORECASE)
    text = re.sub(r'\bla\b', 'the', text, flags=re.IGNORECASE)
    text = re.sub(r'\bel\b', 'the', text, flags=re.IGNORECASE)
    
    return text


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_university_name(university: str, log_unmapped: List[str] = None) -> str:
    """
    Normalize university name to standard English format.
    
    Args:
        university: Original university name
        log_unmapped: Optional list to log unmapped values
    
    Returns:
        Normalized university name
    """
    
    if pd.isna(university) or not str(university).strip():
        return "N/A"
    
    # Step 1: Remove accents
    normalized = remove_accents(str(university))
    
    # Step 2: Normalize whitespace
    normalized = normalize_whitespace(normalized)
    
    # Step 3: Convert to lowercase for matching
    normalized_lower = normalized.lower()
    
    # Step 4: Check exact match in dictionary
    if normalized_lower in UNIVERSITY_NAME_MAPPING:
        return UNIVERSITY_NAME_MAPPING[normalized_lower]
    
    # Step 5: Try partial matching (contains)
    for key, value in UNIVERSITY_NAME_MAPPING.items():
        if key in normalized_lower or normalized_lower in key:
            return value
    
    # Step 5b: Try keyword matching for Polytechnic/Braganza (including typo "politenico")
    if ('politecnico' in normalized_lower or 'politecnical' in normalized_lower or 
        'polytechnic' in normalized_lower or 'politenico' in normalized_lower):  # Typo variant
        if 'braganza' in normalized_lower or 'braganca' in normalized_lower or 'bragança' in normalized_lower:
            return 'Politecnical Institute of Braganza'
    
    # Step 6: Translate Spanish to English
    normalized = translate_spanish_to_english(normalized)
    
    # Step 7: Remove accents again after translation
    normalized = remove_accents(normalized)
    
    # Step 8: Normalize whitespace again
    normalized = normalize_whitespace(normalized)
    
    # Step 9: Check again after translation
    normalized_lower = normalized.lower()
    if normalized_lower in UNIVERSITY_NAME_MAPPING:
        return UNIVERSITY_NAME_MAPPING[normalized_lower]
    
    # Step 10: Try partial matching again
    for key, value in UNIVERSITY_NAME_MAPPING.items():
        if key in normalized_lower or normalized_lower in key:
            return value
    
    # Step 10b: Try keyword matching for Polytechnic/Braganza (after translation, including typo "politenico")
    if ('politecnico' in normalized_lower or 'politecnical' in normalized_lower or 
        'polytechnic' in normalized_lower or 'politenico' in normalized_lower):  # Typo variant
        if 'braganza' in normalized_lower or 'braganca' in normalized_lower or 'bragança' in normalized_lower:
            return 'Politecnical Institute of Braganza'
    
    # Step 11: If still not found, log and return cleaned version
    if log_unmapped is not None:
        log_unmapped.append(f"'{university}' → '{normalized}' (not in dictionary)")
    
    # Return cleaned version with proper capitalization
    return normalized.title()


def normalize_country_name(country: str, log_unmapped: List[str] = None) -> str:
    """
    Normalize country name - extract only country, remove cities.
    Only allows: Spain, Portugal, or N/A
    
    Args:
        country: Original country value (may include city)
        log_unmapped: Optional list to log unmapped values
    
    Returns:
        Normalized country name (Spain, Portugal, or N/A)
    """
    
    if pd.isna(country) or not str(country).strip():
        return "N/A"
    
    original = str(country).strip()
    normalized = original
    
    # Step 1: Remove accents
    normalized = remove_accents(normalized)
    
    # Step 2: Normalize whitespace
    normalized = normalize_whitespace(normalized)
    
    # Step 3: Convert to lowercase for matching
    normalized_lower = normalized.lower()
    
    # Step 4: Remove city patterns (e.g., "Madrid, Spain" → "Spain")
    for pattern in COUNTRY_CITY_PATTERNS:
        match = re.search(pattern, normalized_lower, re.IGNORECASE)
        if match:
            # Extract country part
            if 'spain' in match.group(1).lower() or 'españa' in match.group(1).lower() or 'espana' in match.group(1).lower():
                normalized = 'Spain'
                break
            elif 'portugal' in match.group(1).lower():
                normalized = 'Portugal'
                break
    
    # Step 5: Check exact match in dictionary
    normalized_lower = normalized.lower()
    if normalized_lower in COUNTRY_NAME_MAPPING:
        return COUNTRY_NAME_MAPPING[normalized_lower]
    
    # Step 6: Try partial matching
    if 'spain' in normalized_lower or 'españa' in normalized_lower or 'espana' in normalized_lower:
        return 'Spain'
    elif 'portugal' in normalized_lower:
        return 'Portugal'
    
    # Step 7: If not recognized, log and return N/A
    if log_unmapped is not None:
        log_unmapped.append(f"'{original}' → 'N/A' (not recognized as Spain or Portugal)")
    
    return "N/A"


def normalize_education_level_with_patterns(level: str, log_unmapped: List[str] = None) -> str:
    """
    Normalize education level using pattern matching.
    Only allows: Doctorate, PhD, Masters, Bachelors, High School, or N/A
    
    Args:
        level: Original education level
        log_unmapped: Optional list to log unmapped values
    
    Returns:
        Normalized education level
    """
    
    if pd.isna(level) or not str(level).strip():
        return "N/A"
    
    original = str(level).strip()
    normalized = original
    
    # Step 1: Normalize whitespace
    normalized = normalize_whitespace(normalized)
    
    # Step 2: Convert to title case for matching
    normalized_title = normalized.title()
    
    # Step 3: Check if already valid
    valid_values = ['Doctorate', 'PhD', 'Masters', 'Bachelors', 'High School', 'N/A']
    if normalized_title in valid_values:
        return normalized_title
    
    # Step 4: Apply pattern matching
    normalized_lower = normalized.lower()
    for pattern, replacement in EDUCATION_LEVEL_PATTERNS.items():
        if re.search(pattern, normalized_lower, re.IGNORECASE):
            normalized = replacement
            break
    
    # Step 5: Check if result is valid
    if normalized in valid_values:
        return normalized
    
    # Step 6: If still not valid, log and return N/A
    if log_unmapped is not None:
        log_unmapped.append(f"'{original}' → 'N/A' (not recognized, patterns didn't match)")
    
    return "N/A"


def normalize_text_general(text: str) -> str:
    """
    General text normalization: remove accents, normalize whitespace.
    
    Args:
        text: Original text
    
    Returns:
        Normalized text
    """
    
    if pd.isna(text) or not str(text).strip():
        return text if not pd.isna(text) else "N/A"
    
    # Remove accents
    normalized = remove_accents(str(text))
    
    # Normalize whitespace
    normalized = normalize_whitespace(normalized)
    
    return normalized


# ============================================================================
# LANGUAGE NAME DICTIONARY
# ============================================================================

LANGUAGE_NAME_MAPPING = {
    # Standard language names (normalize variations)
    'english': 'English',
    'englisch': 'English',  # German spelling typo
    'spanish': 'Spanish',
    'portuguese': 'Portuguese',
    'portughese': 'Portuguese',  # Typo
    'french': 'French',
    'german': 'German',
    'italian': 'Italian',
    'chinese': 'Chinese',
    'arabic': 'Arabic',
    'greek': 'Greek',
    'persian': 'Persian',
    'hungarian': 'Hungarian',
    'serbian': 'Serbian',
}

# Language qualifiers to remove (regex patterns)
LANGUAGE_QUALIFIERS = [
    r'\s*\([^)]*\)',  # Remove anything in parentheses
    r'\s*\([^)]*$',  # Remove unclosed parentheses
    r'\s*native\s*',  # Remove "native"
    r'\s*mother\s*tongue\s*',  # Remove "mother tongue"
    r'\s*fluent\s*',  # Remove "fluent"
    r'\s*advanced\s*',  # Remove "advanced"
    r'\s*intermediate\s*',  # Remove "intermediate"
    r'\s*moderately\s*',  # Remove "moderately"
    r'\s*level\s*',  # Remove "level"
    r'\s*\([^)]*level[^)]*\)',  # Remove "level" in parentheses
    r'\s*\([^)]*c1[^)]*\)',  # Remove C1 qualifiers
    r'\s*\([^)]*c2[^)]*\)',  # Remove C2 qualifiers
    r'\s*\([^)]*b1[^)]*\)',  # Remove B1 qualifiers
    r'\s*\([^)]*b2[^)]*\)',  # Remove B2 qualifiers
    r'\s*\([^)]*a1[^)]*\)',  # Remove A1 qualifiers
    r'\s*\([^)]*a2[^)]*\)',  # Remove A2 qualifiers
    r'\s*\([^)]*ielts[^)]*\)',  # Remove IELTS qualifiers
    r'\s*\([^)]*ise[^)]*\)',  # Remove ISE qualifiers
    r'\s*\([^)]*trinity[^)]*\)',  # Remove Trinity qualifiers
    r'\s*\([^)]*partially[^)]*\)',  # Remove "partially" qualifiers
    r'\s*\([^)]*basic[^)]*\)',  # Remove "basic" qualifiers
    r'\s*\([^)]*quite\s*rusty[^)]*\)',  # Remove "quite rusty" qualifiers
    r'\s*\([^)]*need\s*more\s*practice[^)]*\)',  # Remove practice qualifiers
    r'\s*\([^)]*currently\s*learning[^)]*\)',  # Remove learning qualifiers
    r'\s*\([^)]*don.*feel.*confident[^)]*\)',  # Remove confidence qualifiers
    r'\s*\([^)]*also[^)]*\)',  # Remove "also" qualifiers
    r'\s*\([^)]*abroad[^)]*\)',  # Remove "abroad" qualifiers
    r'\s*\([^)]*prc[^)]*\)',  # Remove PRC qualifiers
    r'\s*\([^)]*year[^)]*\)',  # Remove year qualifiers
]

# Common separators to split on
LANGUAGE_SEPARATORS = [
    r'\s*,\s*',  # Comma
    r'\s+and\s+',  # "and"
    r'\s*\+\s*',  # Plus sign
    r'\s+',  # Multiple spaces (fallback)
]


def normalize_language_name(lang: str) -> str:
    """
    Normalize a single language name.
    
    Args:
        lang: Language name (may have qualifiers, wrong case, etc.)
    
    Returns:
        Normalized language name
    """
    if not lang or pd.isna(lang):
        return ""
    
    # Remove accents
    normalized = remove_accents(str(lang).strip())
    
    # Convert to lowercase for matching
    normalized_lower = normalized.lower()
    
    # Check dictionary
    if normalized_lower in LANGUAGE_NAME_MAPPING:
        return LANGUAGE_NAME_MAPPING[normalized_lower]
    
    # Try partial matching
    for key, value in LANGUAGE_NAME_MAPPING.items():
        if key in normalized_lower or normalized_lower in key:
            return value
    
    # If not found, capitalize properly
    return normalized.title().strip()


def normalize_languages(languages: str, log_unmapped: List[str] = None) -> str:
    """
    Normalize languages string to comma-separated list.
    Removes qualifiers, normalizes language names, handles different separators.
    
    Args:
        languages: Original languages string
        log_unmapped: Optional list to log unmapped language names
    
    Returns:
        Comma-separated list of normalized language names
    """
    if pd.isna(languages) or not str(languages).strip():
        return "N/A"
    
    original = str(languages).strip()
    normalized = original
    
    # Step 1: Remove accents
    normalized = remove_accents(normalized)
    
    # Step 2: Remove common prefixes like "Fluent in"
    normalized = re.sub(r'^fluent\s+in\s+', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^speaking\s+', '', normalized, flags=re.IGNORECASE)
    
    # Step 3: Remove qualifiers (parentheses, level indicators, etc.)
    for pattern in LANGUAGE_QUALIFIERS:
        normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
    
    # Step 4: Remove extra text after periods or long descriptions
    # Split on period and take first part if it's a long sentence
    if len(normalized) > 100 or '.' in normalized:
        # Try to extract just the language list part
        # Look for patterns like "Language1, Language2, Language3"
        match = re.search(r'^([^.]*(?:and|,|\+)[^.]*)', normalized, re.IGNORECASE)
        if match:
            normalized = match.group(1)
    
    # Step 4b: Remove phrases like "also I know", "don't feel confident", etc.
    normalized = re.sub(r'\s*also\s+I\s+know\s+', ', ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*but\s+I\s+don.*', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*but\s+I.*feel.*confident.*', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*currently\s+learning\s+[^,)]*', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*after\s+[^,)]*', '', normalized, flags=re.IGNORECASE)
    
    # Step 4b: Remove phrases like "also I know", "don't feel confident", etc.
    normalized = re.sub(r'\s*also\s+I\s+know\s+', ', ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*but\s+I\s+don.*', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*but\s+I.*feel.*confident.*', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*currently\s+learning\s+[^,)]*', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*after\s+[^,)]*', '', normalized, flags=re.IGNORECASE)
    
    # Step 5: Split on separators
    # Handle mixed separators by normalizing to commas first
    normalized = re.sub(r'\s+and\s+', ', ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*\+\s*', ', ', normalized)
    
    # Now split on commas
    if ',' in normalized:
        parts = re.split(r'\s*,\s*', normalized)
    # Otherwise split on multiple spaces
    else:
        parts = re.split(r'\s+', normalized)
    
    # Step 6: Normalize each language name
    normalized_languages = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Remove trailing punctuation
        part = re.sub(r'[.,;:]+$', '', part)
        
        # Normalize the language name
        lang = normalize_language_name(part)
        
        if lang:
            normalized_languages.append(lang)
        elif log_unmapped is not None:
            log_unmapped.append(f"'{part}' (from '{original}')")
    
    # Step 7: Remove duplicates while preserving order
    seen = set()
    unique_languages = []
    for lang in normalized_languages:
        lang_lower = lang.lower()
        if lang_lower not in seen:
            seen.add(lang_lower)
            unique_languages.append(lang)
    
    # Step 8: Return comma-separated list
    if not unique_languages:
        return "N/A"
    
    return ", ".join(unique_languages)


def normalize_mentor_country(country: str, log_unmapped: List[str] = None) -> str:
    """
    Normalize mentor country name(s).
    Handles multiple countries, standardizes names, removes separators like "and" and "/".
    
    Args:
        country: Original country value (may include multiple countries, typos, etc.)
        log_unmapped: Optional list to log unmapped country names
    
    Returns:
        Comma-separated list of normalized country names, or N/A if not recognized
    """
    if pd.isna(country) or not str(country).strip():
        return "N/A"
    
    original = str(country).strip()
    normalized = original
    
    # Step 1: Remove accents
    normalized = remove_accents(normalized)
    
    # Step 2: Normalize whitespace
    normalized = normalize_whitespace(normalized)
    
    # Step 3: Handle special cases
    normalized_lower = normalized.lower()
    if normalized_lower in ['no affiliation', 'none', 'n/a', '']:
        return "N/A"
    
    # Step 4: Normalize separators - replace "and", "/", "&" with commas
    normalized = re.sub(r'\s+and\s+', ', ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*/\s*', ', ', normalized)
    normalized = re.sub(r'\s*&\s*', ', ', normalized)
    normalized = re.sub(r'\s*,\s*and\s+', ', ', normalized, flags=re.IGNORECASE)
    
    # Step 5: Split on commas
    parts = re.split(r'\s*,\s*', normalized)
    
    # Step 6: Normalize each country name
    normalized_countries = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Remove trailing punctuation
        part = re.sub(r'[.,;:]+$', '', part)
        part = part.strip()
        
        if not part:
            continue
        
        # Convert to lowercase for matching
        part_lower = part.lower()
        
        # Check dictionary
        if part_lower in MENTOR_COUNTRY_MAPPING:
            country_name = MENTOR_COUNTRY_MAPPING[part_lower]
            if country_name != 'N/A':  # Don't add N/A to the list
                normalized_countries.append(country_name)
        else:
            # Try partial matching (case-insensitive)
            found = False
            for key, value in MENTOR_COUNTRY_MAPPING.items():
                if key in part_lower or part_lower in key:
                    if value != 'N/A':
                        normalized_countries.append(value)
                    found = True
                    break
            
            if not found:
                # Not recognized - log for review
                if log_unmapped is not None:
                    log_unmapped.append(f"'{part}' (from '{original}')")
                # Try to capitalize properly as fallback
                normalized_countries.append(part.title())
    
    # Step 7: Remove duplicates while preserving order
    seen = set()
    unique_countries = []
    for country in normalized_countries:
        country_lower = country.lower()
        if country_lower not in seen:
            seen.add(country_lower)
            unique_countries.append(country)
    
    # Step 8: Return comma-separated list or N/A
    if not unique_countries:
        return "N/A"
    
    return ", ".join(unique_countries)


def normalize_mentor_country_of_origin(origin: str, log_unmapped: List[str] = None) -> str:
    """
    Normalize mentor country of origin.
    Extracts country names from phrases like "Born in X", "Dual citizen of X and Y", etc.
    
    Args:
        origin: Original country of origin value (may include "Born in", "Dual citizen", etc.)
        log_unmapped: Optional list to log unmapped country names
    
    Returns:
        Comma-separated list of normalized country names, or N/A if not recognized
    """
    if pd.isna(origin) or not str(origin).strip():
        return "N/A"
    
    original = str(origin).strip()
    normalized = original
    
    # Step 1: Remove accents
    normalized = remove_accents(normalized)
    
    # Step 2: Normalize whitespace
    normalized = normalize_whitespace(normalized)
    
    # Step 3: Remove common prefixes
    normalized = re.sub(r'^born\s+and\s+raised\s+in\s+', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^born\s+in\s+', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'^born\s+', '', normalized, flags=re.IGNORECASE)
    
    # Step 4: Handle "dual citizen" - extract both countries
    # Pattern: "Country1 (dual citizen of Country2)" or "Country1 (dual citizen Country2)"
    dual_citizen_match = re.search(r'\(dual\s+citizen(?:\s+of)?\s+([^)]+)\)', normalized, re.IGNORECASE)
    if dual_citizen_match:
        # Extract the main country (before parentheses)
        main_country = re.sub(r'\s*\(.*$', '', normalized).strip()
        # Extract the dual citizen country
        dual_country = dual_citizen_match.group(1).strip()
        # Remove "of" if present
        dual_country = re.sub(r'^of\s+', '', dual_country, flags=re.IGNORECASE).strip()
        # Combine them
        normalized = f"{main_country}, {dual_country}"
    # Step 5: Remove any remaining parentheses content (other qualifiers)
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)
    
    # Step 6: Normalize separators - replace "and" with commas
    normalized = re.sub(r'\s+and\s+', ', ', normalized, flags=re.IGNORECASE)
    
    # Step 7: Split on commas
    parts = re.split(r'\s*,\s*', normalized)
    
    # Step 8: Normalize each country name using the same mapping as affiliated country
    normalized_countries = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Remove trailing punctuation
        part = re.sub(r'[.,;:]+$', '', part)
        part = part.strip()
        
        if not part:
            continue
        
        # Convert to lowercase for matching
        part_lower = part.lower()
        
        # Check dictionary
        if part_lower in MENTOR_COUNTRY_MAPPING:
            country_name = MENTOR_COUNTRY_MAPPING[part_lower]
            if country_name != 'N/A':  # Don't add N/A to the list
                normalized_countries.append(country_name)
        else:
            # Try partial matching (case-insensitive)
            found = False
            for key, value in MENTOR_COUNTRY_MAPPING.items():
                if key in part_lower or part_lower in key:
                    if value != 'N/A':
                        normalized_countries.append(value)
                    found = True
                    break
            
            if not found:
                # Not recognized - log for review
                if log_unmapped is not None:
                    log_unmapped.append(f"'{part}' (from '{original}')")
                # Try to capitalize properly as fallback
                normalized_countries.append(part.title())
    
    # Step 9: Remove duplicates while preserving order
    seen = set()
    unique_countries = []
    for country in normalized_countries:
        country_lower = country.lower()
        if country_lower not in seen:
            seen.add(country_lower)
            unique_countries.append(country)
    
    # Step 10: Return comma-separated list or N/A
    if not unique_countries:
        return "N/A"
    
    return ", ".join(unique_countries)


def normalize_mentor_alma_mater(alma_mater: str, log_unmapped: List[str] = None) -> str:
    """
    Normalize mentor alma mater (institution where bachelor's degree was completed).
    - If only a country is listed, return N/A
    - If an institution is listed, normalize using the same university mapping as mentees
    - Handles multiple institutions (comma-separated)
    
    Args:
        alma_mater: Original alma mater value (may include country, multiple institutions, etc.)
        log_unmapped: Optional list to log unmapped values
    
    Returns:
        Normalized institution name(s) or N/A if only country or not found
    """
    if pd.isna(alma_mater) or not str(alma_mater).strip():
        return "N/A"
    
    original = str(alma_mater).strip()
    normalized = original
    
    # Step 1: Remove accents
    normalized = remove_accents(normalized)
    
    # Step 2: Normalize whitespace
    normalized = normalize_whitespace(normalized)
    
    # Step 2b: Fix common typos
    normalized = re.sub(r'\buniverity\b', 'university', normalized, flags=re.IGNORECASE)  # Typo: univerity -> university
    
    # Step 2c: Expand abbreviations
    normalized = re.sub(r'\bU\.\s+of\s+', 'University of ', normalized, flags=re.IGNORECASE)  # "U. of" -> "University of"
    normalized = re.sub(r'\bU\s+of\s+', 'University of ', normalized, flags=re.IGNORECASE)  # "U of" -> "University of"
    normalized = re.sub(r'\bState\s+U\b', 'State University', normalized, flags=re.IGNORECASE)  # "State U" -> "State University"
    normalized = re.sub(r'\bWright\s+State\s+U\b', 'Wright State University', normalized, flags=re.IGNORECASE)
    
    # Step 2d: Remove degree-only prefixes/suffixes that indicate no institution
    # If the entire string is just a degree, return N/A
    degree_only_pattern = r'^(bachelor\'?s?|master\'?s?|phd|ph\.?\s*d\.?|bsc|msc|degree)$'
    if re.match(degree_only_pattern, normalized, re.IGNORECASE):
        return "N/A"
    
    # Step 2e: Extract institution from fragments
    # Remove prefixes like "from", "PhD", "Master's", "Bachelor's" at the start
    normalized = re.sub(r'^(from|phd|ph\.?\s*d\.?|master\'?s?|bachelor\'?s?|bsc|msc)\s+', '', normalized, flags=re.IGNORECASE)
    # Remove "in [Country]" patterns that come before institution
    normalized = re.sub(r'^(masters?|bachelor\'?s?|phd|ph\.?\s*d\.?)\s+in\s+[^,]+-\s*', '', normalized, flags=re.IGNORECASE)
    # Remove trailing fragments in parentheses that are just research areas
    normalized = re.sub(r'\s*\([^)]*research[^)]*\)$', '', normalized, flags=re.IGNORECASE)
    # Extract from patterns like "Research X (IISER Y)" -> "IISER Y"
    research_match = re.search(r'\(([^)]+)\)', normalized)
    if research_match and ('iiser' in research_match.group(1).lower() or 'institute' in research_match.group(1).lower()):
        normalized = research_match.group(1).strip()
    
    # Step 3: Check if it's just a country name (common countries)
    # Define country patterns here (used later too)
    country_only_patterns = [
        r'^(spain|portugal|usa|uk|united kingdom|united states|india|iran|nigeria|brazil|china|colombia|croatia|romania|netherlands|germany|france|italy|belgium|austria|sweden|switzerland|denmark|norway|ireland|singapore|qatar|taiwan|estonia|luxembourg|canada)$',
        r'^(spain|portugal|usa|uk|united kingdom|united states|india|iran|nigeria|brazil|china|colombia|croatia|romania|netherlands|germany|france|italy|belgium|austria|sweden|switzerland|denmark|norway|ireland|singapore|qatar|taiwan|estonia|luxembourg|canada)\s*$',
    ]
    
    normalized_lower = normalized.lower()
    for pattern in country_only_patterns:
        if re.match(pattern, normalized_lower, re.IGNORECASE):
            # It's just a country, return N/A
            return "N/A"
    
    # Step 4: Extract institution name(s) - remove country names and qualifiers
    # Remove country names in parentheses or after commas/dashes
    countries_pattern = r'(?:spain|portugal|usa|uk|united\s+kingdom|united\s+states|india|iran|nigeria|brazil|china|colombia|croatia|romania|netherlands|germany|france|italy|belgium|austria|sweden|switzerland|denmark|norway|ireland|singapore|qatar|taiwan|estonia|luxembourg|canada)'
    
    # Remove country in parentheses
    normalized = re.sub(r'\s*\([^)]*' + countries_pattern + r'[^)]*\)', '', normalized, flags=re.IGNORECASE)
    # Remove country after comma
    normalized = re.sub(r',\s*' + countries_pattern + r'\s*$', '', normalized, flags=re.IGNORECASE)
    # Remove country after dash
    normalized = re.sub(r'\s*-\s*' + countries_pattern + r'\s*$', '', normalized, flags=re.IGNORECASE)
    # Remove country after & or and
    normalized = re.sub(r'\s*[&]\s*' + countries_pattern + r'\s*$', '', normalized, flags=re.IGNORECASE)
    
    # Step 5: Remove degree qualifiers in parentheses
    normalized = re.sub(r'\s*\([^)]*(?:bachelor|master|phd|degree|bsc|msc|ph\.?\s*d\.?)[^)]*\)', '', normalized, flags=re.IGNORECASE)
    
    # Step 5b: Handle abbreviations in parentheses that are part of institution name
    # If we have "Biotechnology School (ESB)" and ESB is part of Portuguese Catholic University, extract the full name
    if 'biotechnology school' in normalized_lower and 'esb' in normalized_lower:
        # ESB is Escola Superior de Biotecnologia, part of Portuguese Catholic University
        normalized = re.sub(r'biotechnology\s+school\s*\([^)]*esb[^)]*\)', 'Portuguese Catholic University', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'biotechnology\s+school\s*\([^)]*esb[^)]*\),?\s*', '', normalized, flags=re.IGNORECASE)
    
    # Step 6: Remove other qualifiers
    normalized = re.sub(r'\s*-\s*(?:bachelor|master|phd|degree|bsc|msc|ph\.?\s*d\.?)[^)]*$', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*at\s+', ' ', normalized, flags=re.IGNORECASE)  # "BSc at University" -> "BSc University"
    normalized = re.sub(r'^\s*(?:bsc|msc|phd|bachelor|master)\s+', '', normalized, flags=re.IGNORECASE)  # Remove degree prefixes
    
    # Step 7: Handle multiple institutions (separated by ".", "&", "and", ";", ",")
    # Split on common separators (but be careful with commas - only if it looks like multiple institutions)
    # Check for patterns like "Institution1, Institution2" or "Institution1 and Institution2"
    has_multiple = False
    if ' and ' in normalized_lower or ' & ' in normalized_lower or ';' in normalized:
        has_multiple = True
    # Also check for comma-separated institutions (but not "City, Country" patterns)
    elif ',' in normalized:
        # Check if comma separates two potential institutions (not country)
        parts_by_comma = [p.strip() for p in normalized.split(',')]
        if len(parts_by_comma) == 2:
            # Check if second part is NOT a country
            second_part = parts_by_comma[1].lower()
            is_country = any(re.match(pattern, second_part, re.IGNORECASE) for pattern in country_only_patterns)
            if not is_country and len(second_part) > 3:  # Likely an institution, not a country
                has_multiple = True
    
    if has_multiple:
        # Multiple institutions - split and process each
        if ',' in normalized and (' and ' not in normalized_lower and ' & ' not in normalized_lower):
            parts = [p.strip() for p in normalized.split(',')]
        else:
            parts = re.split(r'\s*(?:and|&|;|,)\s+', normalized, flags=re.IGNORECASE)
        normalized_institutions = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Remove trailing punctuation
            part = re.sub(r'[.,;:]+$', '', part).strip()
            
            # Check if it's just a country after cleaning
            part_lower = part.lower()
            is_country_only = False
            for pattern in country_only_patterns:
                if re.match(pattern, part_lower, re.IGNORECASE):
                    is_country_only = True
                    break
            
            if not is_country_only and part:
                # Normalize using same function as mentees
                inst = normalize_university_name(part, log_unmapped)
                if inst and inst != "N/A":
                    normalized_institutions.append(inst)
        
        if normalized_institutions:
            # Remove duplicates
            seen = set()
            unique = []
            for inst in normalized_institutions:
                if inst.lower() not in seen:
                    seen.add(inst.lower())
                    unique.append(inst)
            return ", ".join(unique) if unique else "N/A"
        else:
            return "N/A"
    else:
        # Single institution
        # Check again if it's just a country after cleaning
        normalized_lower = normalized.lower()
        is_country_only = False
        for pattern in country_only_patterns:
            if re.match(pattern, normalized_lower, re.IGNORECASE):
                is_country_only = True
                break
        
        if is_country_only:
            return "N/A"
        
        # Normalize using same function as mentees
        result = normalize_university_name(normalized, log_unmapped)
        return result if result and result != "N/A" else "N/A"


# ============================================================================
# FIELD OF EXPERTISE MAPPING
# Maps specific fields to broad categories (for matching)
# Format: {specific_field: (broad_category, terms_to_add_to_specialization)}
# ============================================================================

# Broad categories from Expertise Guide (40-point matches)
BROAD_FIELD_CATEGORIES = {
    'accounting', 'agricultural sciences', 'architecture', 'behavioral science',
    'bioengineering', 'biomedical engineering', 'biomedicine', 'biotechnology',
    'business informatics', 'chemical engineering', 'chemistry', 'communication studies',
    'computer science', 'economics', 'economics/entrepreneurship', 'education',
    'electrical engineering', 'energy engineering', 'engineering', 'environmental science',
    'fashion design', 'health sciences', 'interior design', 'law', 'linguistics',
    'literature studies', 'mathematics', 'mechanical engineering', 'pharmaceutical sciences',
    'pharmacy', 'physics', 'psychological neuroscience', 'psychology', 'renewable energy',
    'sociology', 'sports science', 'telecommunications engineering', 'advanced chemistry',
    'analytical chemistry', 'biochemistry', 'biochemistry and molecular biology',
    'biodiversity', 'bioinformatics', 'biology', 'biomedical sciences',
    'biomedicine and biotechnology', 'biophysics', 'cancer research', 'cell biology',
    'clinical psychology', 'computational biology', 'data science', 'drug discovery',
    'ecology', 'environmental engineering', 'environmental sciences', 'epidemiology',
    'genetics', 'genomics', 'immunology', 'material science', 'medical sciences',
    'medical technology', 'medicine', 'microbiology', 'molecular biology',
    'molecular life sciences', 'neuroscience', 'neuroscience and neurosurgery',
    'nutrition and dietetics', 'oncology', 'pharmacology', 'pharmacology and physiology',
    'pharmacy and biomedical sciences', 'physiology', 'psychiatry', 'public health',
    'statistical physics and complex systems', 'translational medicine'
}

# Field mapping: specific → (broad_category, terms_for_specialization)
FIELD_TO_BROAD_CATEGORY = {
    # Direct matches (already broad) - no mapping needed, just normalize case
    'biomedicine': ('Biomedicine', []),
    'biomedicine ': ('Biomedicine', []),
    'engineering': ('Engineering', []),
    'psychology': ('Psychology', []),
    'economics': ('Economics', []),
    'computer science': ('Computer Science', []),
    'chemistry': ('Chemistry', []),
    'physics': ('Physics', []),
    'architecture': ('Architecture', []),
    'architecture ': ('Architecture', []),
    'law': ('Law', []),
    'education': ('Education', []),
    'fashion design': ('Fashion Design', []),
    'interior design': ('Interior Design', []),
    'interior design ': ('Interior Design', []),
    'behavioral science': ('Behavioral Science', []),
    'psychological neuroscience': ('Psychological Neuroscience', []),
    'accounting': ('Accounting', []),
    
    # Typos and variations
    'bioemedicine': ('Biomedicine', []),  # Typo
    'economis': ('Economics', []),  # Typo
    'biomedicina, biotechnology': ('Biomedicine', ['biotechnology']),
    'biomedicina': ('Biomedicine', []),
    
    # Molecular/Cellular → Biomedicine
    'cell biology': ('Biomedicine', ['cell biology']),
    'molecular biology': ('Biomedicine', ['molecular biology']),
    'biochemistry': ('Biomedicine', ['biochemistry']),
    'genetics': ('Biomedicine', ['genetics']),
    'genetics and developmental biology': ('Biomedicine', ['genetics', 'developmental biology']),
    'genomics': ('Biomedicine', ['genomics']),
    'bioinformatics': ('Biomedicine', ['bioinformatics']),
    'computational biology': ('Biomedicine', ['computational biology']),
    'molecular life sciences': ('Biomedicine', ['molecular life sciences']),
    'biophysics': ('Biomedicine', ['biophysics']),
    
    # Medical Specializations → Biomedicine or Medicine
    'cancer biology': ('Biomedicine', ['cancer biology']),
    'cardiovascular research': ('Biomedicine', ['cardiovascular research']),
    'cognitive neuroscience': ('Biomedicine', ['cognitive neuroscience']),
    'neurology': ('Medicine', ['neurology']),
    'oncology': ('Biomedicine', ['oncology']),
    'microbiology': ('Biomedicine', ['microbiology']),
    'microbiology, oncology': ('Biomedicine', ['microbiology', 'oncology']),
    'toxicology': ('Biomedicine', ['toxicology']),
    'toxicology, biomedical science': ('Biomedicine', ['toxicology', 'biomedical science']),
    'immunology': ('Biomedicine', ['immunology']),
    'physiology': ('Biomedicine', ['physiology']),
    'renal and cardiovascular physiology': ('Biomedicine', ['renal physiology', 'cardiovascular physiology']),
    'epidemiology': ('Biomedicine', ['epidemiology']),
    'public health': ('Public Health', []),
    'medicine': ('Medicine', []),
    'medicine, pulmonology': ('Medicine', ['pulmonology']),
    'translational medicine': ('Translational Medicine', []),
    'medical sciences': ('Medical Sciences', []),
    'medical technology': ('Medical Technology', []),
    
    # Engineering Specializations → Engineering
    'aeronautics': ('Engineering', ['aeronautics']),
    'aeronautics ': ('Engineering', ['aeronautics']),
    'civil engeneering': ('Engineering', ['civil engineering']),  # Typo
    'civil engineering': ('Engineering', ['civil engineering']),
    'electronic engineering': ('Engineering', ['electronic engineering']),
    'electrical and computer engineering': ('Engineering', ['electrical engineering', 'computer engineering']),
    'telecom engineering': ('Telecommunications Engineering', ['telecom engineering']),
    'telecos': ('Telecommunications Engineering', ['telecommunications']),
    'materials engineering': ('Engineering', ['materials engineering']),
    'forest engineering': ('Engineering', ['forest engineering']),
    'forestry engineering': ('Engineering', ['forestry engineering']),
    'industrial and manufacturing engineering & operations management': ('Engineering', ['industrial engineering', 'manufacturing engineering', 'operations management']),
    'software engineering': ('Computer Science', ['software engineering']),
    'biomedical / mechanical / aerospace engineering': ('Biomedical Engineering', ['mechanical engineering', 'aerospace engineering']),
    
    # Bioengineering variations
    'bioengineering': ('Bioengineering', []),
    'bioengineering ': ('Bioengineering', []),
    'bio engineering, green nanotechnology': ('Bioengineering', ['green nanotechnology']),
    'bioengineering - biomaterials': ('Bioengineering', ['biomaterials']),
    'biological engineering': ('Bioengineering', ['biological engineering']),
    'bioprocess engineering, industrial biotechnology': ('Bioengineering', ['bioprocess engineering', 'industrial biotechnology']),
    'public health / bioengineering': ('Bioengineering', ['public health']),
    
    # Biomedical Engineering
    'biomedical engineering': ('Biomedical Engineering', []),
    'biomedical engineering ': ('Biomedical Engineering', []),
    
    # Chemistry Specializations → Chemistry
    'chemistry – sustainable chemistry and catalysis': ('Chemistry', ['sustainable chemistry', 'catalysis']),
    'analytical chemistry': ('Chemistry', ['analytical chemistry']),
    'advanced chemistry': ('Advanced Chemistry', []),
    
    # Physics Specializations → Physics
    'physics - material science': ('Physics', ['material science']),
    'physics, data, sustainability': ('Physics', ['data science', 'sustainability']),
    'statistical physics and complex systems': ('Statistical Physics and Complex Systems', []),
    
    # Biology variations
    'biology': ('Biology', []),
    'experimental biology': ('Biology', ['experimental biology']),
    'biology and pharmacy': ('Biomedicine', ['biology', 'pharmacy']),
    
    # Biotechnology
    'biotechnology': ('Biotechnology', []),
    'biotech': ('Biotechnology', []),
    'biotech ': ('Biotechnology', []),
    
    # Combined Biomedical fields
    'biomedicine, pharmacy': ('Biomedicine', ['pharmacy']),
    'neuroscience, biomedicine': ('Biomedicine', ['neuroscience']),
    'cancer and neuroscience': ('Biomedicine', ['cancer research', 'neuroscience']),
    'pharmaceutics, tissue engineering, biomedicine': ('Biomedicine', ['pharmaceutics', 'tissue engineering']),
    
    # Business/Economics
    'business': ('Economics', ['business']),
    'business, economics, law.': ('Economics', ['business', 'law']),
    'economics and business': ('Economics', ['business']),
    'economics, law and pharmaeconomics': ('Economics', ['law', 'pharmacoeconomics']),
    'finance and treasury': ('Economics', ['finance', 'treasury']),
    
    # Psychology variations
    'negotiation/hrm/psychology': ('Psychology', ['negotiation', 'human resource management']),
    'pyschology/hrm/negotiation': ('Psychology', ['human resource management', 'negotiation']),  # Typo
    
    # Linguistics
    'applied linguistics': ('Linguistics', ['applied linguistics']),
    'philosophy, linguistics, sustainability': ('Linguistics', ['philosophy', 'sustainability']),
    
    # Engineering combinations
    'engineering, design, innovation': ('Engineering', ['design', 'innovation']),
    'engineering and consultancy': ('Engineering', ['consultancy']),
    
    # Other
    'geology, mining and manufacturing': ('Engineering', ['geology', 'mining', 'manufacturing']),
    'radiation detection for nuclear medicine': ('Biomedicine', ['radiation detection', 'nuclear medicine']),
}

# Fields that should be added to the guide (for future reference)
FIELDS_TO_ADD_TO_GUIDE = [
    'Aeronautics',  # Maps to Engineering
    'Applied Linguistics',  # Maps to Linguistics
    'Biological Engineering',  # Maps to Bioengineering
    'Bioprocess Engineering',  # Maps to Engineering
    'Civil Engineering',  # Maps to Engineering
    'Electronic Engineering',  # Maps to Electrical Engineering
    'Forest Engineering',  # Maps to Engineering
    'Software Engineering',  # Maps to Computer Science
    'Telecom Engineering',  # Maps to Telecommunications Engineering
]


def normalize_mentor_field_of_expertise(field: str, client=None, cache=None, 
                                       row_num: int=0, log_unmapped: List[str]=None) -> Tuple[str, List[str]]:
    """
    Normalize mentor field of expertise to broad category.
    Maps specific fields to broad categories (for matching) and extracts specific terms.
    
    Args:
        field: Original field of expertise value
        client: OpenAI client (for AI fallback)
        cache: AI cache instance
        row_num: Row number for logging
        log_unmapped: Optional list to log unmapped fields
    
    Returns:
        Tuple of (broad_category, list_of_terms_for_specialization)
    """
    if pd.isna(field) or not str(field).strip():
        return ("N/A", [])
    
    original = str(field).strip()
    normalized = original
    
    # Step 1: Remove accents
    normalized = remove_accents(normalized)
    
    # Step 2: Normalize whitespace
    normalized = normalize_whitespace(normalized)
    
    # Step 3: Fix common typos
    normalized = re.sub(r'\bbioemedicine\b', 'biomedicine', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\beconomis\b', 'economics', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bcivil\s+engeneering\b', 'civil engineering', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bpyschology\b', 'psychology', normalized, flags=re.IGNORECASE)
    
    # Step 4: Convert to lowercase for matching
    normalized_lower = normalized.lower()
    
    # Step 5: Check dictionary mapping FIRST (prioritize specific mappings)
    if normalized_lower in FIELD_TO_BROAD_CATEGORY:
        broad_category, spec_terms = FIELD_TO_BROAD_CATEGORY[normalized_lower]
        return (broad_category, spec_terms)
    
    # Step 6: Check if it's already a broad category (and not in dictionary)
    if normalized_lower in BROAD_FIELD_CATEGORIES:
        # Already broad, just normalize capitalization
        return (normalized.title(), [])
    
    # Step 7: Handle combined fields (comma, "and", "/", "&")
    # Check if it contains multiple fields
    separators = [r',\s*', r'\s+and\s+', r'\s*/\s*', r'\s*&\s*']
    parts = []
    for sep in separators:
        if re.search(sep, normalized, re.IGNORECASE):
            parts = re.split(sep, normalized, flags=re.IGNORECASE)
            break
    
    if len(parts) > 1:
        # Multiple fields - process each
        mapped_parts = []
        all_spec_terms = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_lower = part.lower()
            
            # Check if it's a broad category
            if part_lower in BROAD_FIELD_CATEGORIES:
                mapped_parts.append((part.title(), True))  # True = in guide
            # Check dictionary
            elif part_lower in FIELD_TO_BROAD_CATEGORY:
                broad, spec = FIELD_TO_BROAD_CATEGORY[part_lower]
                mapped_parts.append((broad, True))
                all_spec_terms.extend(spec)
            else:
                # Not in dictionary - will need AI or pattern matching
                mapped_parts.append((part, False))
        
        # Select primary category: most common in guide, or first if multiple
        guide_categories = [cat for cat, in_guide in mapped_parts if in_guide]
        if guide_categories:
            # Use first one that's in guide (most common approach)
            primary_category = guide_categories[0]
            # Add other parts as specialization terms
            for part, in_guide in mapped_parts:
                if part != primary_category and not in_guide:
                    all_spec_terms.append(part.lower())
            return (primary_category, all_spec_terms)
        else:
            # None in guide - use first one, will need AI
            primary_category = mapped_parts[0][0]
            for part, _ in mapped_parts[1:]:
                all_spec_terms.append(part.lower())
            # Continue to AI classification below
    
    # Step 8: Pattern-based extraction for long descriptions
    # Look for known guide terms in the description
    found_guide_term = None
    for guide_term in sorted(BROAD_FIELD_CATEGORIES, key=len, reverse=True):  # Longest first
        if guide_term in normalized_lower:
            found_guide_term = guide_term.title()
            # Extract other terms as specialization
            # Remove the found term and clean up
            remaining = re.sub(r'\b' + re.escape(guide_term) + r'\b', '', normalized_lower, flags=re.IGNORECASE)
            remaining = normalize_whitespace(remaining)
            # Split remaining into terms
            remaining_terms = [t.strip() for t in re.split(r'[,\s]+', remaining) if t.strip() and len(t.strip()) > 2]
            return (found_guide_term, remaining_terms)
    
    # Step 9: AI classification (same logic as mentees, for consistency)
    if client and cache:
        try:
            # Use same AI prompt as mentees for consistency
            prompt = f"""You are an academic field classifier. Based on the information provided, determine the BROAD field of expertise category.

Field of Expertise: {normalized}

Your task:
- Identify the BROAD academic/professional field category
- Examples of broad categories: Biomedicine, Mechanical Engineering, Computer Science, Business, Law, Psychology, Physics, Chemistry, etc.
- Return ONLY the broad category name (1-3 words maximum), nothing else
- Capitalize properly (e.g., "Computer Science" not "computer science")

Return only the broad field category:"""

            # Check cache
            cache_input = f"field_classify|{normalized}"
            cache_key = cache.get_cache_key(cache_input)
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                broad_category = cached_result
            else:
                import openai
                MAX_RETRIES = 3
                RETRY_DELAY = 2
                MODEL = "gpt-4o-mini"
                
                for attempt in range(MAX_RETRIES):
                    try:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": "You only return a single broad academic/professional field category. Maximum 3 words. Proper capitalization."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                            max_tokens=20
                        )
                        
                        broad_category = response.choices[0].message.content.strip()
                        broad_category = broad_category.strip('."\'')
                        cache.set(cache_key, broad_category)
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            import time
                            time.sleep(RETRY_DELAY)
                        else:
                            raise e
            
            # Extract specific terms (everything except the broad category)
            spec_terms = []
            # Always extract terms from the original normalized field (before AI classification)
            # Split on common separators to get individual terms
            spec_terms_raw = re.split(r'[/,\s]+', normalized.lower())
            spec_terms = [t.strip() for t in spec_terms_raw if t.strip() and len(t.strip()) > 1]
            
            # Remove the broad category from terms if it appears
            spec_terms = [t for t in spec_terms if broad_category.lower() not in t and t not in broad_category.lower()]
            
            # Normalize common abbreviations
            normalized_terms = []
            for term in spec_terms:
                term_lower = term.lower()
                if term_lower == 'ai':
                    normalized_terms.append('artificial intelligence')
                elif term_lower == 'ml':
                    normalized_terms.append('machine learning')
                elif term_lower == 'software':
                    normalized_terms.append('software engineering')
                elif term_lower == 'semiconductors':
                    normalized_terms.append('semiconductors')
                else:
                    normalized_terms.append(term)
            spec_terms = normalized_terms
            
            if log_unmapped is not None:
                log_unmapped.append(f"'{original}' → '{broad_category}' (AI classified, terms: {spec_terms})")
            
            return (broad_category, spec_terms)
        except Exception as e:
            if log_unmapped is not None:
                log_unmapped.append(f"'{original}' → 'N/A' (AI classification failed: {str(e)})")
            return ("N/A", [])
    else:
        # No AI available - log and return original
        if log_unmapped is not None:
            log_unmapped.append(f"'{original}' → '{normalized.title()}' (not in dictionary, no AI)")
        return (normalized.title(), [])


# Import pandas for pd.isna checks
import pandas as pd
from typing import Tuple, List

