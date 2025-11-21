"""
Enhanced Mentor Skills Merger and Normalizer
Transforms PRE format to POST format with full automation.
"""

import pandas as pd
import openai
import os
import time
import sys
import argparse
from typing import Tuple
import json

# Import shared utilities
from shared_utils import (
    AICache, find_column_flexible, generate_output_filename,
    clean_dataframe, find_missing_data, validate_required_columns,
    TransformationReport, write_error_log
)

# Import institution type normalization
from institution_types import normalize_institution_type

# Import normalization utilities
from normalization import (
    normalize_mentor_country, normalize_mentor_country_of_origin, 
    normalize_mentor_alma_mater, normalize_mentor_field_of_expertise
)

# Configuration
MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Required output columns (exact names from POST format)
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

# PRE column mapping (what to look for in PRE files)
PRE_COLUMN_MAPPING = {
    'affiliation': ['affiliation'],
    'country': ['country of your affiliated institution'],
    'current_position': ['current position'],
    'institution_type': ['type of work'],
    'country_of_origin': ['country of origin'],
    'education_level': ['highest educational level', 'highest education level'],
    'alma_mater': ['institution & country where bachelor', 'bachelor\'s degree'],
    'field': ['field of expertise'],
    'specialization': ['specialization'],
    'hard_skills': ['specific hard skills', 'professional competencies'],
    'experience_years': ['years of professional experience'],
    'career_dev_experience': ['areas where you feel comfortable mentoring', 'comfortable mentoring']
}


def setup_openai():
    """Setup OpenAI API key from environment variable"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
        os.environ['OPENAI_API_KEY'] = api_key
    
    openai.api_key = api_key
    return openai


def extract_education_level_and_field(education_text: str, client, cache: AICache, row_num: int) -> Tuple[str, str]:
    """
    Extract education level and field from combined education description.
    Similar to mentee program splitting.
    Returns: (level, field) where level is one of: PhD, Masters, Bachelors
    With caching support.
    """
    if pd.isna(education_text) or not str(education_text).strip():
        return "N/A", "N/A"
    
    edu_text = str(education_text).strip()
    
    # Check cache
    cache_key = cache.get_cache_key(edu_text)
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Row {row_num}: Education extraction (cached): Level='{cached_result[0]}', Field='{cached_result[1]}'")
        return cached_result[0], cached_result[1]
    
    prompt = f"""You are an education level parser. Your task is to extract two pieces of information from an education description.

Input: "{edu_text}"

Extract:
1. LEVEL: Must be exactly one of: PhD, Masters, Bachelors (normalize any variations, extract the HIGHEST level mentioned)
2. FIELD: The field(s) of study mentioned

Return ONLY in this exact JSON format:
{{"level": "PhD", "field": "Computer Science"}}

If there are multiple degrees, extract the HIGHEST level and all fields mentioned.
If there is NO field mentioned, return:
{{"level": "PhD", "field": "ERROR: No field found"}}

Examples:
- "Double degree: Law, Business Management; Master's in Management" → {{"level": "Masters", "field": "Law, Business Management, Management"}}
- "PhD in Computer Science" → {{"level": "PhD", "field": "Computer Science"}}
- "PhD" → {{"level": "PhD", "field": "ERROR: No field found"}}
- "BSc and BA" → {{"level": "Bachelors", "field": "ERROR: No field found"}}
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You only return valid JSON with 'level' and 'field' keys. Level must be PhD, Masters, or Bachelors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                parsed = json.loads(result)
                level = parsed.get("level", "ERROR")
                field = parsed.get("field", "ERROR: No field found")
                
                # Cache result
                cache.set(cache_key, (level, field))
                
                print(f"Row {row_num}: Extracted Level='{level}', Field='{field}'")
                return level, field
                
            except json.JSONDecodeError:
                print(f"Row {row_num}: Failed to parse JSON response: {result}")
                if attempt < MAX_RETRIES - 1:
                    continue
                else:
                    return "ERROR", f"ERROR: Could not parse '{edu_text}'"
                
        except Exception as e:
            print(f"Row {row_num}: Education extraction attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Row {row_num}: All retries failed.")
                return "N/A", "N/A"
    
    return "N/A", "N/A"


def merge_and_normalize_skills(extracted_field: str, original_institution_type: str,
                               specialization: str, hard_skills: str, client, cache, row_num: int) -> str:
    """
    Use OpenAI to merge and normalize skills from four sources:
    1. Extracted field from education level
    2. Original institution type (if it was transformed)
    3. Specialization column
    4. Hard skills column
    With caching support.
    """
    # Handle empty values
    field_text = str(extracted_field).strip() if pd.notna(extracted_field) and "ERROR" not in str(extracted_field) else ""
    inst_text = str(original_institution_type).strip() if pd.notna(original_institution_type) else ""
    spec_text = str(specialization).strip() if pd.notna(specialization) else ""
    skills_text = str(hard_skills).strip() if pd.notna(hard_skills) else ""
    
    # If all are empty, return N/A
    if not field_text and not inst_text and not spec_text and not skills_text:
        return "N/A"
    
    # Check cache (use combined input as key)
    cache_input = f"{field_text}|{inst_text}|{spec_text}|{skills_text}"
    cache_key = cache.get_cache_key(cache_input)
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Row {row_num}: Skills merged (cached)")
        return cached_result
    
    # Prepare the prompt
    # If specialization contains "|", it means it's already combined from multiple sources
    if "|" in spec_text:
        # Already combined - treat as single field
        prompt = f"""You are a skills normalization expert. I need you to merge and normalize multiple fields into a single list of keywords.

Combined Sources (separated by |): {spec_text}

Your task:
1. Extract all relevant keywords from all sources
2. Normalize them by:
   - Converting to lowercase
   - Standardizing terminology (e.g., "AI" → "artificial intelligence", "ML" → "machine learning")
   - Removing exact duplicates (but keep related concepts even if they overlap)
3. Sort them from BROADER concepts to MORE SPECIFIC skills
4. Return ONLY a comma-separated list of keywords, nothing else

Example output format: strategic leadership, business development, project management, machine learning, python programming, data visualization

Now process the sources above and return only the normalized comma-separated list:"""
    else:
        # Original format - four separate fields
        prompt = f"""You are a skills normalization expert. I need you to merge and normalize four fields into a single list of keywords.

Field 1 (Field from education level): {field_text}
Field 2 (Original institution type - if transformed): {inst_text}
Field 3 (Specialization): {spec_text}
Field 4 (Specific Hard Skills): {skills_text}

Your task:
1. Extract all relevant keywords from all four fields
2. Normalize them by:
   - Converting to lowercase
   - Standardizing terminology (e.g., "AI" → "artificial intelligence", "ML" → "machine learning")
   - Removing exact duplicates (but keep related concepts even if they overlap)
3. Sort them from BROADER concepts to MORE SPECIFIC skills
4. Return ONLY a comma-separated list of keywords, nothing else

Example output format: strategic leadership, business development, project management, machine learning, python programming, data visualization

Now process the fields above and return only the normalized comma-separated list:"""

    # Make API call with retry logic
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise skills normalization assistant. You only return comma-separated keyword lists, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up the result (remove any extra formatting)
            result = result.replace('\n', ' ').replace('  ', ' ').strip()
            
            # Remove any leading/trailing punctuation or quotes
            result = result.strip('."\'')
            
            # Cache result
            cache.set(cache_key, result)
            
            print(f"Row {row_num}: Skills merged successfully")
            return result
            
        except Exception as e:
            print(f"Row {row_num}: Skills merge attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Row {row_num}: All retries failed. Using combined original data.")
                # Fallback: just combine and clean the original data
                combined = []
                for text in [field_text, inst_text, spec_text, skills_text]:
                    if text:
                        combined.append(text)
                return ", ".join(combined).lower()


def find_pre_columns(df: pd.DataFrame) -> dict:
    """Find required columns in PRE file using flexible matching."""
    found_cols = {}
    
    for key, search_terms in PRE_COLUMN_MAPPING.items():
        col = find_column_flexible(df, search_terms)
        if col:
            found_cols[key] = col
        else:
            found_cols[key] = None
    
    return found_cols


def process_mentor_file(input_file: str, output_file: str = None, dry_run: bool = False, 
                       test_rows: int = None, client=None, cache=None, report=None):
    """
    Process mentor file from PRE to POST format.
    
    Args:
        input_file: Path to input PRE file
        output_file: Path to output POST file (auto-generated if None)
        dry_run: If True, don't write output file
        test_rows: If set, only process first N rows
        client: OpenAI client
        cache: AI cache instance
        report: Transformation report instance
    """
    if client is None:
        client = setup_openai()
    if cache is None:
        cache = AICache()
    if report is None:
        report = TransformationReport("Mentor Skills Merger")
    
    # Track transformations for logging
    institution_type_transformations = []
    unmapped_countries = []
    unmapped_countries_of_origin = []
    unmapped_alma_maters = []
    unmapped_fields = []
    
    # Read the Excel file (handle header row)
    print("\n1. Reading input file...")
    df = pd.read_excel(input_file, header=1)  # Skip first row (usually metadata)
    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   ✓ Columns found: {len(df.columns)}")
    
    # Limit rows if test mode
    if test_rows:
        df = df.head(test_rows)
        print(f"   ⚠ TEST MODE: Processing only first {test_rows} rows")
    
    report.stats['total_rows'] = len(df)
    
    # Find required columns
    print("\n2. Finding required columns...")
    pre_cols = find_pre_columns(df)
    
    # Verify critical columns found
    critical_cols = ['education_level', 'field', 'specialization', 'hard_skills', 'institution_type']
    missing_critical = [key for key in critical_cols if pre_cols[key] is None]
    
    if missing_critical:
        print(f"   ✗ Error: Could not find required columns: {missing_critical}")
        print(f"   Available columns: {df.columns.tolist()}")
        return None
    
    for key, col_name in pre_cols.items():
        if col_name:
            print(f"   ✓ Found: {key} → '{col_name}'")
        else:
            print(f"   ⚠ Not found: {key} (will use N/A)")
    
    # Create output dataframe
    df_output = pd.DataFrame()
    
    # Generate IDs
    print("\n3. Generating Mentor IDs...")
    df_output[MENTOR_COLUMNS['id']] = range(1, len(df) + 1)
    print(f"   ✓ Generated {len(df)} IDs")
    
    # Copy simple columns (with prefix)
    print("\n4. Copying and renaming columns...")
    if pre_cols['affiliation']:
        df_output[MENTOR_COLUMNS['affiliation']] = df[pre_cols['affiliation']].fillna("N/A")
    else:
        df_output[MENTOR_COLUMNS['affiliation']] = "N/A"
    
    if pre_cols['country']:
        # Normalize country names (handle multiple countries, standardize names)
        df_output[MENTOR_COLUMNS['country']] = df[pre_cols['country']].apply(
            lambda x: normalize_mentor_country(x, unmapped_countries) if pd.notna(x) else "N/A"
        )
    else:
        df_output[MENTOR_COLUMNS['country']] = "N/A"
    
    if pre_cols['current_position']:
        df_output[MENTOR_COLUMNS['current_position']] = df[pre_cols['current_position']].fillna("N/A")
    else:
        df_output[MENTOR_COLUMNS['current_position']] = "N/A"
    
    if pre_cols['country_of_origin']:
        # Normalize country of origin (extract from "Born in", "Dual citizen", etc.)
        df_output[MENTOR_COLUMNS['country_of_origin']] = df[pre_cols['country_of_origin']].apply(
            lambda x: normalize_mentor_country_of_origin(x, unmapped_countries_of_origin) if pd.notna(x) else "N/A"
        )
    else:
        df_output[MENTOR_COLUMNS['country_of_origin']] = "N/A"
    
    if pre_cols['alma_mater']:
        # Normalize alma mater (extract institution, use same mapping as mentees)
        df_output[MENTOR_COLUMNS['alma_mater']] = df[pre_cols['alma_mater']].apply(
            lambda x: normalize_mentor_alma_mater(x, unmapped_alma_maters) if pd.notna(x) else "N/A"
        )
    else:
        df_output[MENTOR_COLUMNS['alma_mater']] = "N/A"
    
    # Store original field for specialization merge (before we normalize it)
    original_field_of_expertise = df[pre_cols['field']].copy() if pre_cols['field'] else pd.Series(["N/A"] * len(df))
    
    # Normalize field of expertise (map to broad category, extract specific terms)
    print("\n5. Normalizing fields of expertise...")
    print("-" * 80)
    normalized_fields = []
    extracted_field_terms = []  # Terms to add to specialization
    
    for idx, row in df.iterrows():
        field = row[pre_cols['field']] if pre_cols['field'] else "N/A"
        progress = f"[{idx + 1}/{len(df)}]"
        print(f"{progress}", end=" ", flush=True)
        
        try:
            broad_category, spec_terms = normalize_mentor_field_of_expertise(
                field,
                client,
                cache,
                idx + 1,
                unmapped_fields
            )
            normalized_fields.append(broad_category)
            extracted_field_terms.append(spec_terms)
            report.record_operation('fields_classified')
        except Exception as e:
            print(f"\nRow {idx + 1}: Error normalizing field: {str(e)}")
            normalized_fields.append("N/A")
            extracted_field_terms.append([])
            report.record_failure(idx + 1, "Field normalization", str(e))
        
        time.sleep(0.1)
    
    df_output[MENTOR_COLUMNS['field']] = normalized_fields
    print("\n" + "-" * 80)
    print(f"   ✓ Normalized all fields of expertise")
    
    if pre_cols['experience_years']:
        df_output[MENTOR_COLUMNS['experience_years']] = df[pre_cols['experience_years']].fillna("N/A")
    else:
        df_output[MENTOR_COLUMNS['experience_years']] = "N/A"
    
    if pre_cols['career_dev_experience']:
        df_output[MENTOR_COLUMNS['career_dev_experience']] = df[pre_cols['career_dev_experience']].fillna("N/A")
    else:
        df_output[MENTOR_COLUMNS['career_dev_experience']] = "N/A"
    
    print("   ✓ Copied simple columns")
    
    # STEP 1: Normalize institution types
    print("\n5. Normalizing institution types...")
    print("-" * 80)
    normalized_institution_types = []
    original_institution_types_for_spec = []  # Store originals that should go to specialization
    
    for idx, row in df.iterrows():
        institution_type = row[pre_cols['institution_type']] if pre_cols['institution_type'] else "N/A"
        progress = f"[{idx + 1}/{len(df)}]"
        print(f"{progress}", end=" ", flush=True)
        
        try:
            normalized, add_to_spec = normalize_institution_type(
                institution_type,
                institution_type_transformations
            )
            normalized_institution_types.append(normalized)
            
            # Store original if it should be added to specialization
            if add_to_spec and pd.notna(institution_type) and str(institution_type).strip():
                original_institution_types_for_spec.append(str(institution_type).strip())
            else:
                original_institution_types_for_spec.append("")
            
            report.record_operation('institution_types_normalized')
        except Exception as e:
            print(f"\nRow {idx + 1}: Error normalizing institution type: {str(e)}")
            normalized_institution_types.append("N/A")
            original_institution_types_for_spec.append("")
            report.record_failure(idx + 1, "Institution type normalization", str(e))
        
        time.sleep(0.05)  # Shorter delay for dictionary lookup
    
    df_output[MENTOR_COLUMNS['institution_type']] = normalized_institution_types
    print("\n" + "-" * 80)
    print(f"   ✓ Normalized all institution types")
    
    # Show transformation log
    if institution_type_transformations:
        print(f"\n   Institution Type Transformations (for review):")
        for trans in institution_type_transformations:
            print(f"     {trans}")
    
    # STEP 2: Extract education level and field
    print("\n6. Extracting education levels and fields...")
    print("-" * 80)
    extracted_levels = []
    extracted_fields = []
    
    for idx, row in df.iterrows():
        education_text = row[pre_cols['education_level']] if pre_cols['education_level'] else "N/A"
        progress = f"[{idx + 1}/{len(df)}]"
        print(f"{progress}", end=" ", flush=True)
        
        try:
            level, field = extract_education_level_and_field(education_text, client, cache, idx + 1)
            extracted_levels.append(level)
            extracted_fields.append(field)
            report.record_operation('education_levels_normalized')
        except Exception as e:
            print(f"\nRow {idx + 1}: Error extracting education: {str(e)}")
            extracted_levels.append("N/A")
            extracted_fields.append("N/A")
            report.record_failure(idx + 1, "Education extraction", str(e))
        
        time.sleep(0.1)
    
    df_output[MENTOR_COLUMNS['education_level']] = extracted_levels
    print("\n" + "-" * 80)
    print(f"   ✓ Extracted all education levels and fields")
    
    # STEP 3: Merge and normalize skills
    print("\n7. Merging and normalizing specializations and skills...")
    print("-" * 80)
    merged_skills = []
    
    for idx, row in df.iterrows():
        extracted_field = extracted_fields[idx] if idx < len(extracted_fields) else "N/A"
        original_inst_type = original_institution_types_for_spec[idx] if idx < len(original_institution_types_for_spec) else ""
        specialization = row[pre_cols['specialization']] if pre_cols['specialization'] else "N/A"
        hard_skills = row[pre_cols['hard_skills']] if pre_cols['hard_skills'] else "N/A"
        
        # Add extracted specific terms from field of expertise
        extracted_field_terms_str = ""
        if idx < len(extracted_field_terms) and extracted_field_terms[idx]:
            extracted_field_terms_str = ", ".join(extracted_field_terms[idx])
        
        # Combine all sources: extracted field terms (from field normalization) + extracted education field + 
        # original institution type + specialization + hard skills
        # Order: broader → specific (extracted field terms are broader concepts)
        all_sources = []
        if extracted_field_terms_str:
            all_sources.append(str(extracted_field_terms_str))
        if extracted_field and extracted_field != "N/A" and "ERROR" not in str(extracted_field) and pd.notna(extracted_field):
            all_sources.append(str(extracted_field))
        if original_inst_type and pd.notna(original_inst_type) and str(original_inst_type).strip():
            all_sources.append(str(original_inst_type))
        if specialization and pd.notna(specialization) and specialization != "N/A" and str(specialization).strip():
            all_sources.append(str(specialization))
        if hard_skills and pd.notna(hard_skills) and hard_skills != "N/A" and str(hard_skills).strip():
            all_sources.append(str(hard_skills))
        
        # Combine into single string for AI processing (use | separator to indicate combined)
        # Filter out empty strings and ensure all are strings
        all_sources = [s.strip() for s in all_sources if s and str(s).strip() and str(s).strip() != "N/A"]
        combined_sources = " | ".join(all_sources) if all_sources else "N/A"
        
        progress = f"[{idx + 1}/{len(df)}]"
        print(f"{progress}", end=" ", flush=True)
        
        try:
            # Pass combined sources as specialization field (with | separator)
            normalized = merge_and_normalize_skills(
                "",  # Empty - not used when combined
                "",  # Empty - not used when combined
                combined_sources,  # Pass all combined sources as specialization (with | separator)
                "",  # Empty - not used when combined
                client,
                cache,
                idx + 1
            )
            merged_skills.append(normalized)
            report.record_operation('skills_merged')
        except Exception as e:
            print(f"\nRow {idx + 1}: Error merging skills: {str(e)}")
            # Fallback: just combine the sources
            fallback = ", ".join([s for s in all_sources if s]).lower()
            merged_skills.append(fallback if fallback else "N/A")
            report.record_failure(idx + 1, "Skills merging", str(e))
        
        time.sleep(0.1)
    
    df_output[MENTOR_COLUMNS['specialization']] = merged_skills
    print("\n" + "-" * 80)
    print(f"   ✓ Merged and normalized all skills")
    
    # Clean data (NaN → N/A, trim whitespace)
    print("\n8. Cleaning data...")
    df_output = clean_dataframe(df_output)
    
    # Find missing data
    missing = find_missing_data(df_output, MENTOR_COLUMNS)
    report.set_missing_data(missing)
    
    # Reorder columns to match POST format
    print("\n9. Reordering columns...")
    column_order = [MENTOR_COLUMNS[key] for key in ['id', 'affiliation', 'country', 'current_position',
                                                     'institution_type', 'country_of_origin', 'alma_mater',
                                                     'education_level', 'field', 'specialization',
                                                     'experience_years', 'career_dev_experience']]
    df_output = df_output[column_order]
    print("   ✓ Columns reordered")
    
    # Validate required columns
    print("\n10. Validating output...")
    is_valid, missing_cols = validate_required_columns(df_output, MENTOR_COLUMNS)
    if not is_valid:
        print(f"   ⚠ Warning: Missing columns in output: {missing_cols}")
    else:
        print("   ✓ All required columns present")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = generate_output_filename(input_file)
    
    # Save to output file (unless dry-run)
    if not dry_run:
        print(f"\n11. Saving output file...")
        df_output.to_excel(output_file, index=False)
        print(f"   ✓ Output saved to: {output_file}")
    else:
        print(f"\n11. DRY-RUN MODE: Skipping file write")
        print(f"   Would save to: {output_file}")
    
    # Show sample results
    print("\n12. Sample results (first 3 rows):")
    print("=" * 80)
    for idx in range(min(3, len(df_output))):
        print(f"\nRow {idx + 1}:")
        for col in column_order[:6]:  # Show first 6 columns
            print(f"  {col}: {df_output.iloc[idx][col]}")
    
    return df_output, output_file, institution_type_transformations, unmapped_countries, unmapped_countries_of_origin, unmapped_alma_maters, unmapped_fields


def main():
    parser = argparse.ArgumentParser(description='Enhanced Mentor Skills Merger and Normalizer')
    parser.add_argument('--input', '-i', type=str, help='Input PRE file path')
    parser.add_argument('--output', '-o', type=str, help='Output POST file path (auto-generated if not provided)')
    parser.add_argument('--dry-run', action='store_true', help='Dry-run mode (don\'t write output file)')
    parser.add_argument('--test-rows', type=int, help='Process only first N rows (for testing)')
    
    args = parser.parse_args()
    
    # Determine input file
    if args.input:
        input_file = args.input
    else:
        # Default: look for PRE file in Files directory
        files_dir = os.path.join(os.path.dirname(__file__), 'Files')
        pre_files = [f for f in os.listdir(files_dir) if 'Mentor' in f and 'PRE' in f and f.endswith('.xlsx')]
        if pre_files:
            input_file = os.path.join(files_dir, pre_files[0])
            print(f"Using default input file: {input_file}")
        else:
            print("Error: No input file specified and no PRE file found in Files directory")
            print("Usage: python merge_mentor_skills_enhanced.py --input <file> [--output <file>] [--dry-run] [--test-rows N]")
            return
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print("=" * 80)
    print("Enhanced Mentor Skills Merger and Normalizer")
    print("=" * 80)
    
    if args.dry_run:
        print("\n⚠ DRY-RUN MODE: No files will be written")
    if args.test_rows:
        print(f"\n⚠ TEST MODE: Processing only first {args.test_rows} rows")
    
    # Setup
    print("\n0. Setting up OpenAI API...")
    client = setup_openai()
    cache = AICache()
    report = TransformationReport("Mentor Skills Merger")
    print("   ✓ API key configured")
    print("   ✓ Cache initialized")
    
    try:
        # Process file
        result = process_mentor_file(
            input_file=input_file,
            output_file=args.output,
            dry_run=args.dry_run,
            test_rows=args.test_rows,
            client=client,
            cache=cache,
            report=report
        )
        
        if result is None:
            print("\n✗ Processing failed")
            return
        
        df_output, output_file, transformations, unmapped_countries, unmapped_countries_of_origin, unmapped_alma_maters, unmapped_fields = result
        
        # Generate and display summary
        print("\n" + "=" * 80)
        summary = report.generate_summary(output_file, cache.get_stats())
        print(summary)
        
        # Show unmapped countries for review
        if unmapped_countries:
            print("\n" + "=" * 80)
            print("Unmapped Countries (for review):")
            print("=" * 80)
            for val in unmapped_countries:
                print(f"  {val}")
        else:
            print("\n✓ All countries normalized successfully")
        
        # Show unmapped countries of origin for review
        if unmapped_countries_of_origin:
            print("\n" + "=" * 80)
            print("Unmapped Countries of Origin (for review):")
            print("=" * 80)
            for val in unmapped_countries_of_origin:
                print(f"  {val}")
        else:
            print("\n✓ All countries of origin normalized successfully")
        
        # Show unmapped alma maters for review
        if unmapped_alma_maters:
            print("\n" + "=" * 80)
            print("Unmapped Alma Maters (for review):")
            print("=" * 80)
            for val in unmapped_alma_maters:
                print(f"  {val}")
        else:
            print("\n✓ All alma maters normalized successfully")
        
        # Show unmapped fields for review
        if unmapped_fields:
            print("\n" + "=" * 80)
            print("Unmapped Fields of Expertise (for review):")
            print("=" * 80)
            for val in unmapped_fields:
                print(f"  {val}")
        else:
            print("\n✓ All fields of expertise normalized successfully")
        
        # Write error log if there are errors
        if report.errors:
            error_log_path = write_error_log(
                report.errors,
                os.path.dirname(output_file) if output_file else os.path.dirname(input_file),
                "mentor"
            )
            if error_log_path:
                print(f"\nError log written to: {error_log_path}")
        
        print("\n" + "=" * 80)
        print("✓ Processing complete!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

