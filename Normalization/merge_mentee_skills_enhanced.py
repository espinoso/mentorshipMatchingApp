"""
Enhanced Mentee Skills Merger and Normalizer
Transforms PRE format to POST format with full automation.
"""

import argparse
import os
import sys
import time

import pandas as pd

# Import shared utilities
from shared_utils import (
    AICache,
    TransformationReport,
    classify_education_level,
    classify_level_llm,
    clean_dataframe,
    extract_program_specialization,
    find_column_flexible,
    find_missing_data,
    generate_output_filename,
    merge_skill_sources,
    read_input_sheet,
    setup_claude,
    validate_required_columns,
    write_error_log,
)

# Import normalization utilities
from normalization import (
    normalize_country_name,
    normalize_education_level_with_patterns,
    normalize_languages,
    normalize_text_general,
    normalize_university_name,
)

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Required output columns (exact names from POST format)
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

# PRE column mapping (what to look for in PRE files)
PRE_COLUMN_MAPPING = {
    'affiliation': ['university where you are currently studying', 'affiliation'],
    'country': ['affiliated university', 'affiliated institution'],
    'education_level': ['highest educational level', 'highest education level'],
    'current_program': ['current education program'],
    'field': ['field of expertise'],
    'specialization': ['specialization'],
    'hard_skills': ['specific hard skills'],
    'languages': ['languages spoken fluently'],
    'guidance_areas': ['areas where guidance is needed'],
    'career_goals': ['career goals for the next 2-3 years', 'career goals'],
    'other_info': ['is there anything else', 'anything else']
}


def _clean(value) -> str:
    """Stringify a raw cell; NaN and 'N/A' become ''."""
    text = "" if pd.isna(value) else str(value).strip()
    return "" if text.upper() == "N/A" else text


def classify_mentee_field(original_field, current_program, client, cache: AICache, row_num: int) -> str:
    """
    The only per-row LLM call for mentees: BROAD field of expertise (1-3 words).
    Level normalization and the skills merge are deterministic (see shared_utils).
    With caching support.
    """
    field, program = _clean(original_field), _clean(current_program)
    if not field and not program:
        return "N/A"

    cache_key = cache.get_cache_key(f"mentee_field|{field}|{program}")
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Row {row_num}: Field of expertise (cached): '{cached_result}'")
        return cached_result

    prompt = f"""You are an academic field classifier. Based on the information provided, determine the BROAD field of expertise category.

Original Field of Expertise: {field}
Current Education Program: {program}

Your task:
- Identify the BROAD academic/professional field category
- Examples of broad categories: Biomedicine, Mechanical Engineering, Computer Science, Business, Law, Psychology, Physics, Chemistry, etc.
- Return ONLY the broad category name (1-3 words maximum), nothing else
- Capitalize properly (e.g., "Computer Science" not "computer science")

Return only the broad field category:"""

    for attempt in range(MAX_RETRIES):
        try:
            result = client.complete_text(
                "You only return a single broad academic/professional field category. Maximum 3 words. Proper capitalization.",
                prompt,
                max_tokens=50,  # a few words of field label + headroom so it isn't cut off
            ).strip('."\' \n')
            if not result:
                raise ValueError("empty response")
            cache.set(cache_key, result)
            print(f"Row {row_num}: Field of expertise determined as '{result}'")
            return result
        except Exception as e:
            print(f"Row {row_num}: Field determination attempt {attempt + 1} failed: {e!s}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    print(f"Row {row_num}: All retries failed. Using original field.")
    return field or "N/A"


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


def process_mentee_file(input_file: str, output_file: str = None, dry_run: bool = False, 
                       test_rows: int = None, client=None, cache=None, report=None):
    """
    Process mentee file from PRE to POST format.
    
    Args:
        input_file: Path to input PRE file
        output_file: Path to output POST file (auto-generated if None)
        dry_run: If True, don't write output file
        test_rows: If set, only process first N rows
        client: ClaudeClient (see shared_utils.setup_claude)
        cache: AI cache instance
        report: Transformation report instance
    """
    if client is None:
        client = setup_claude()
    if cache is None:
        cache = AICache()
    if report is None:
        report = TransformationReport("Mentee Skills Merger")
    
    # Lists to track unmapped values for review
    unmapped_universities = []
    unmapped_countries = []
    unmapped_education_levels = []
    unmapped_program_levels = []
    unmapped_languages = []
    
    # Read the Excel file (sniff the header row — files may have blank leading rows)
    print("\n1. Reading input file...")
    df = read_input_sheet(input_file)
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
    critical_cols = ['education_level', 'current_program', 'field', 'specialization', 'hard_skills']
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
    print("\n3. Generating Mentee IDs...")
    df_output[MENTEE_COLUMNS['id']] = range(1, len(df) + 1)
    print(f"   ✓ Generated {len(df)} IDs")
    
    # Copy and normalize simple columns (with prefix)
    print("\n4. Copying, normalizing and renaming columns...")
    if pre_cols['affiliation']:
        # Normalize university names
        df_output[MENTEE_COLUMNS['affiliation']] = df[pre_cols['affiliation']].apply(
            lambda x: normalize_university_name(x, unmapped_universities)
        )
    else:
        df_output[MENTEE_COLUMNS['affiliation']] = "N/A"
    
    if pre_cols['country']:
        # Normalize country names (extract only country, remove cities)
        df_output[MENTEE_COLUMNS['country']] = df[pre_cols['country']].apply(
            lambda x: normalize_country_name(x, unmapped_countries)
        )
    else:
        df_output[MENTEE_COLUMNS['country']] = "N/A"
    
    if pre_cols['languages']:
        # Normalize languages to comma-separated list
        df_output[MENTEE_COLUMNS['languages']] = df[pre_cols['languages']].apply(
            lambda x: normalize_languages(x, unmapped_languages) if pd.notna(x) else "N/A"
        )
    else:
        df_output[MENTEE_COLUMNS['languages']] = "N/A"
    
    if pre_cols['guidance_areas']:
        df_output[MENTEE_COLUMNS['guidance_areas']] = df[pre_cols['guidance_areas']].apply(
            lambda x: normalize_text_general(x) if pd.notna(x) else "N/A"
        )
    else:
        df_output[MENTEE_COLUMNS['guidance_areas']] = "N/A"
    
    if pre_cols['career_goals']:
        df_output[MENTEE_COLUMNS['career_goals']] = df[pre_cols['career_goals']].apply(
            lambda x: normalize_text_general(x) if pd.notna(x) else "N/A"
        )
    else:
        df_output[MENTEE_COLUMNS['career_goals']] = "N/A"
    
    if pre_cols['other_info']:
        df_output[MENTEE_COLUMNS['other_info']] = df[pre_cols['other_info']].apply(
            lambda x: normalize_text_general(x) if pd.notna(x) else "N/A"
        )
    else:
        df_output[MENTEE_COLUMNS['other_info']] = "N/A"
    
    print("   ✓ Copied simple columns")
    
    # Save original Field of expertise for merging (before we overwrite it)
    original_field_of_expertise = df[pre_cols['field']].copy() if pre_cols['field'] else pd.Series(["N/A"] * len(df))
    
    # Levels and skills are computed in code; the one LLM call per row classifies the broad field
    print("\n5. Normalizing profiles (fields, skills, education)...")
    print("-" * 80)
    total_rows = len(df)
    normalized_education, split_levels, broad_fields, merged_skills = [], [], [], []

    for idx, row in df.iterrows():
        education_level = row[pre_cols['education_level']] if pre_cols['education_level'] else "N/A"
        current_program = row[pre_cols['current_program']] if pre_cols['current_program'] else "N/A"
        original_field = original_field_of_expertise.iloc[idx] if len(original_field_of_expertise) > idx else "N/A"
        specialization = row[pre_cols['specialization']] if pre_cols['specialization'] else "N/A"
        hard_skills = row[pre_cols['hard_skills']] if pre_cols['hard_skills'] else "N/A"

        print(f"[{idx + 1}/{total_rows}]", end=" ", flush=True)

        try:
            education = classify_education_level(education_level)
            if education == "N/A":  # keyword miss: try the pattern table (also logs for review)
                education = normalize_education_level_with_patterns(education_level, unmapped_education_levels)
            if education == "N/A":  # still unknown and present: LLM fallback
                education = classify_level_llm(education_level, client, cache)
            level = classify_education_level(current_program)
            if level == "N/A":
                level = normalize_education_level_with_patterns(current_program, unmapped_program_levels)
            if level == "N/A":
                level = classify_level_llm(current_program, client, cache)
            skills = merge_skill_sources(
                original_field, extract_program_specialization(current_program), specialization, hard_skills
            )
            broad_field = classify_mentee_field(original_field, current_program, client, cache, idx + 1)

            normalized_education.append(education)
            split_levels.append(level)
            broad_fields.append(broad_field)
            merged_skills.append(skills)
            for op in ('education_levels_normalized', 'education_programs_split',
                       'fields_classified', 'skills_merged'):
                report.record_operation(op)
            report.record_success(idx + 1)
        except Exception as e:
            print(f"\nRow {idx + 1}: Error normalizing profile: {e!s}")
            normalized_education.append("N/A")
            split_levels.append("N/A")
            broad_fields.append("N/A")
            merged_skills.append("N/A")
            report.record_failure(idx + 1, "Profile normalization", str(e))

        time.sleep(0.1)

    df_output[MENTEE_COLUMNS['education_level']] = normalized_education
    df_output[MENTEE_COLUMNS['current_program']] = split_levels
    # Normalize field and specialization text (remove accents, trim whitespace)
    df_output[MENTEE_COLUMNS['field']] = [normalize_text_general(f) if pd.notna(f) else "N/A" for f in broad_fields]
    df_output[MENTEE_COLUMNS['specialization']] = [normalize_text_general(s) if pd.notna(s) else "N/A" for s in merged_skills]
    print("\n" + "-" * 80)
    print("   ✓ Normalized all profiles")

    # Clean data (NaN → N/A, trim whitespace)
    print("\n6. Cleaning data...")
    df_output = clean_dataframe(df_output)
    
    # Find missing data
    missing = find_missing_data(df_output, MENTEE_COLUMNS)
    report.set_missing_data(missing)
    
    # Reorder columns to match POST format
    print("\n7. Reordering columns...")
    column_order = [MENTEE_COLUMNS[key] for key in ['id', 'affiliation', 'country', 'education_level', 
                                                     'current_program', 'field', 'specialization', 
                                                     'languages', 'guidance_areas', 'career_goals', 'other_info']]
    df_output = df_output[column_order]
    print("   ✓ Columns reordered")
    
    # Validate required columns
    print("\n8. Validating output...")
    is_valid, missing_cols = validate_required_columns(df_output, MENTEE_COLUMNS)
    if not is_valid:
        print(f"   ⚠ Warning: Missing columns in output: {missing_cols}")
    else:
        print("   ✓ All required columns present")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = generate_output_filename(input_file)
    
    # Save to output file (unless dry-run)
    if not dry_run:
        print("\n9. Saving output file...")
        df_output.to_excel(output_file, index=False)
        print(f"   ✓ Output saved to: {output_file}")
    else:
        print("\n9. DRY-RUN MODE: Skipping file write")
        print(f"   Would save to: {output_file}")
    
    # Show sample results
    print("\n10. Sample results (first 3 rows):")
    print("=" * 80)
    for idx in range(min(3, len(df_output))):
        print(f"\nRow {idx + 1}:")
        for col in column_order[:5]:  # Show first 5 columns
            print(f"  {col}: {df_output.iloc[idx][col]}")
    
    # Store unmapped values for reporting
    unmapped_values = {
        'universities': unmapped_universities,
        'countries': unmapped_countries,
        'education_levels': unmapped_education_levels,
        'program_levels': unmapped_program_levels,
        'languages': unmapped_languages
    }
    
    return df_output, output_file, unmapped_values


def main():
    parser = argparse.ArgumentParser(description='Enhanced Mentee Skills Merger and Normalizer')
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
        pre_files = [f for f in os.listdir(files_dir) if 'Mentee' in f and 'PRE' in f and f.endswith('.xlsx')]
        if pre_files:
            input_file = os.path.join(files_dir, pre_files[0])
            print(f"Using default input file: {input_file}")
        else:
            print("Error: No input file specified and no PRE file found in Files directory")
            print("Usage: python merge_mentee_skills_enhanced.py --input <file> [--output <file>] [--dry-run] [--test-rows N]")
            return
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print("=" * 80)
    print("Enhanced Mentee Skills Merger and Normalizer")
    print("=" * 80)
    
    if args.dry_run:
        print("\n⚠ DRY-RUN MODE: No files will be written")
    if args.test_rows:
        print(f"\n⚠ TEST MODE: Processing only first {args.test_rows} rows")
    
    # Setup
    print("\n0. Setting up Claude API...")
    client = setup_claude()
    cache = AICache()
    report = TransformationReport("Mentee Skills Merger")
    print("   ✓ API key configured")
    print("   ✓ Cache initialized")
    
    try:
        # Process file
        result = process_mentee_file(
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
        
        df_output, output_file, unmapped_values = result
        
        # Generate and display summary
        print("\n" + "=" * 80)
        summary = report.generate_summary(output_file, cache.get_stats())
        print(summary)
        
        # Show unmapped values for review
        print("\n" + "=" * 80)
        print("Unmapped Values (for review):")
        print("=" * 80)
        
        if unmapped_values['universities']:
            print("\nUniversities (not in dictionary):")
            for val in unmapped_values['universities']:
                print(f"  {val}")
        else:
            print("\n✓ All universities mapped successfully")
        
        if unmapped_values['countries']:
            print("\nCountries (not recognized as Spain/Portugal):")
            for val in unmapped_values['countries']:
                print(f"  {val}")
        else:
            print("\n✓ All countries normalized successfully")
        
        if unmapped_values['education_levels']:
            print("\nEducation Levels (patterns didn't match):")
            for val in unmapped_values['education_levels']:
                print(f"  {val}")
        else:
            print("\n✓ All education levels normalized successfully")
        
        if unmapped_values['program_levels']:
            print("\nProgram Levels (patterns didn't match):")
            for val in unmapped_values['program_levels']:
                print(f"  {val}")
        else:
            print("\n✓ All program levels normalized successfully")
        
        if unmapped_values['languages']:
            print("\nLanguages (not recognized):")
            for val in unmapped_values['languages']:
                print(f"  {val}")
        else:
            print("\n✓ All languages normalized successfully")
        
        print("=" * 80)
        
        # Write error log if there are errors
        if report.errors:
            error_log_path = write_error_log(
                report.errors,
                os.path.dirname(output_file) if output_file else os.path.dirname(input_file),
                "mentee"
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
        print(f"\n\n✗ Fatal error: {e!s}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

