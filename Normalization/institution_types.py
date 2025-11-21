"""
Institution type mapping dictionary for mentor processing.
Maps various institution type values to standardized values.
"""

import pandas as pd
from typing import Tuple

# Institution type mapping dictionary
# Format: {original_value: (normalized_value, add_to_specialization)}
# If add_to_specialization is True, the original value should be added to specialization

INSTITUTION_TYPE_MAPPING = {
    # Direct matches - no transformation needed
    'Industry': ('Industry', False),
    'Industry ': ('Industry', False),
    'Academia': ('Academia', False),
    'Academia ': ('Academia', False),
    'ACADEMIA': ('Academia', False),
    
    # Academia variations - map to Academia, original goes to specialization
    'Academic/Teaching': ('Academia', True),
    'Acamedia': ('Academia', True),  # Typo, but map to Academia
    'Acadèmia, startups ': ('Academia', True),  # Has startups, but primarily academia
    'Education': ('Academia', True),
    'Research': ('Academia', True),
    'Research Support': ('Academia', True),
    
    # Industry variations - map to Industry, original goes to specialization
    'Non-profit': ('Industry', True),
    'Non-Profit': ('Industry', True),
    'Technology': ('Industry', True),
    'Tech': ('Industry', True),
    'Startup': ('Industry', True),
    'Consulting': ('Industry', True),
    'Corporate': ('Industry', True),
    'Banking': ('Industry', True),
    'Financial Sector': ('Industry', True),
    'Healthcare Industry': ('Industry', True),
    'Biotech company ': ('Industry', True),
    'Medical Device Industry ; Start-up': ('Industry', True),
    'Oil and Gas': ('Industry', True),
    'Logistics and Supply Chain': ('Industry', True),
    'Energy transition ': ('Industry', True),
    'Architecture ': ('Industry', True),  # Professional practice
    'Civil service': ('Industry', True),
    'Clinical. Applied Research': ('Industry', True),
    'IT Consulting': ('Industry', True),
    'Interior Design Consultancy': ('Industry', True),
    'Consulting on EU grants': ('Industry', True),
    
    # Combined types - both words present
    'Academia with Industry Experience': ('Industry & Academia', False),
    'R&D, Academia and Industry': ('Industry & Academia', False),
    
    # Multiple types separated by commas
    'Academia, Consulting': ('Industry & Academia', True),  # Has both, add original
    'Industry & Consulting': ('Industry', True),  # Industry with consulting
    'Industry / start up': ('Industry', True),
    'Industry, Manufacturing, Automotiv': ('Industry', True),
    'Industry - Central Lab for Clinical Trials ': ('Industry', True),
    'Industry - Healtcare': ('Industry', True),
    'Industry - Healthcare': ('Industry', True),
    'Academia, previous experience in biotech': ('Academia', True),
    'Research & Consulting': ('Industry & Academia', True),
    'Startup, fintech, sustainability ': ('Industry', True),
    'Professor, start up and treasurer': ('Industry & Academia', True),
    'I work as a clinical neurologist and as a researcher': ('Industry & Academia', True),
}


def normalize_institution_type(original_value: str, log_transformations: list = None) -> Tuple[str, bool]:
    """
    Normalize institution type to one of: Industry, Academia, Industry & Academia.
    
    Args:
        original_value: Original institution type value
        log_transformations: Optional list to log transformations for review
    
    Returns:
        Tuple of (normalized_value, should_add_to_specialization)
    """
    if not original_value or pd.isna(original_value):
        return ("N/A", False)
    
    original_str = str(original_value).strip()
    
    # Check for combined types first (both words present)
    original_lower = original_str.lower()
    has_industry = 'industry' in original_lower or 'corporate' in original_lower or 'business' in original_lower
    has_academia = 'academia' in original_lower or 'academic' in original_lower or 'research' in original_lower or 'university' in original_lower
    
    if has_industry and has_academia:
        # Both words present - map to Industry & Academia
        normalized = "Industry & Academia"
        # Check if exact match in dictionary
        if original_str in INSTITUTION_TYPE_MAPPING:
            normalized, add_to_spec = INSTITUTION_TYPE_MAPPING[original_str]
            if log_transformations is not None and add_to_spec:
                log_transformations.append(f"'{original_str}' → '{normalized}' (original added to specialization)")
            return (normalized, add_to_spec)
        else:
            # Not in dictionary but has both words
            if log_transformations is not None:
                log_transformations.append(f"'{original_str}' → 'Industry & Academia' (detected both words, original added to specialization)")
            return ("Industry & Academia", True)
    
    # Check exact match in dictionary
    if original_str in INSTITUTION_TYPE_MAPPING:
        normalized, add_to_spec = INSTITUTION_TYPE_MAPPING[original_str]
        if log_transformations is not None and add_to_spec:
            log_transformations.append(f"'{original_str}' → '{normalized}' (original added to specialization)")
        elif log_transformations is not None and original_str not in ['Industry', 'Industry ', 'Academia', 'Academia ', 'ACADEMIA']:
            log_transformations.append(f"'{original_str}' → '{normalized}' (exact match)")
        return (normalized, add_to_spec)
    
    # Case-insensitive match
    for key, (normalized, add_to_spec) in INSTITUTION_TYPE_MAPPING.items():
        if key.lower() == original_str.lower():
            if log_transformations is not None:
                if add_to_spec:
                    log_transformations.append(f"'{original_str}' → '{normalized}' (case-insensitive match, original added to specialization)")
                else:
                    log_transformations.append(f"'{original_str}' → '{normalized}' (case-insensitive match)")
            return (normalized, add_to_spec)
    
    # Default: try to infer from content
    if has_industry:
        if log_transformations is not None:
            log_transformations.append(f"'{original_str}' → 'Industry' (inferred, original added to specialization)")
        return ("Industry", True)
    elif has_academia:
        if log_transformations is not None:
            log_transformations.append(f"'{original_str}' → 'Academia' (inferred, original added to specialization)")
        return ("Academia", True)
    else:
        # Unknown - default to Industry and add original to specialization
        if log_transformations is not None:
            log_transformations.append(f"'{original_str}' → 'Industry' (unknown type, original added to specialization)")
        return ("Industry", True)

