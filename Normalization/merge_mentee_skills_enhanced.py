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
    clean_dataframe,
    find_column_flexible,
    find_missing_data,
    generate_output_filename,
    read_input_sheet,
    setup_claude,
    validate_required_columns,
    write_error_log,
)

from llm_client import parse_json

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
    'affiliation': ['university where you are currently studying'],
    'country': ['country of your affiliated university'],
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


def normalize_mentee_row(education_level, current_program, original_field, specialization,
                         hard_skills, client, cache: AICache, row_num: int) -> dict:
    """
    ONE Claude call that normalizes a mentee's education level, program level,
    broad field of expertise and merged skills (replaces four sequential calls).
    Returns: {"education_level", "program_level", "field_of_expertise", "skills"}
    With caching support.
    """
    edu, prog, field, spec, skills = (
        _clean(v) for v in (education_level, current_program, original_field, specialization, hard_skills)
    )
    if not (edu or prog or field or spec or skills):
        return {"education_level": "N/A", "program_level": "N/A",
                "field_of_expertise": "N/A", "skills": "N/A"}

    cache_key = cache.get_cache_key(f"mentee_row|{edu}|{prog}|{field}|{spec}|{skills}")
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Row {row_num}: Profile normalized (cached)")
        return dict(cached_result)

    prompt = f"""You are normalizing one mentee's profile for a mentorship matching program. From the raw fields below, produce a clean, structured profile.

Raw fields:
- Highest education level: "{edu}"
- Current education program: "{prog}"
- Original field of expertise: "{field}"
- Specialization: "{spec}"
- Hard skills: "{skills}"

Return ONLY a JSON object with exactly these keys:
{{
  "education_level": one of exactly "Masters", "PhD", "High School", "Bachelors" — map the highest education level (closest match if unclear).
  "program_level": one of exactly "PhD", "Masters", "Bachelors" — the level of the CURRENT program; "N/A" if none stated.
  "field_of_expertise": the BROAD academic/professional field category, 1-3 words, properly capitalized (e.g. "Biomedicine", "Computer Science"), inferred from the original field and the current program.
  "specialization_extracted": the field of study from the current program (short phrase), or "N/A".
  "skills": a SINGLE comma-separated list of normalized keywords merged from the original field, the program specialization, the specialization field, and the hard skills. Lowercase; standardize terminology (AI→artificial intelligence, ML→machine learning); remove exact duplicates; order broader→more specific.
}}"""

    for attempt in range(MAX_RETRIES):
        try:
            raw = client.complete_text(
                "You return only a JSON object with the requested keys.", prompt, max_tokens=600
            )
            parsed = parse_json(raw)

            education = str(parsed.get("education_level", "")).strip()
            if education not in ("Masters", "PhD", "High School", "Bachelors"):
                education = edu or "N/A"  # unexpected value: keep original, as before
            field_out = str(parsed.get("field_of_expertise", "")).strip('."\' ') or field or "N/A"
            skills_out = parsed.get("skills", "")
            if isinstance(skills_out, list):
                skills_out = ", ".join(str(s) for s in skills_out)
            skills_out = str(skills_out).replace("\n", " ").replace("  ", " ").strip('."\' ') or "N/A"

            result = {
                "education_level": education,
                "program_level": str(parsed.get("program_level", "ERROR")).strip(),
                "field_of_expertise": field_out,
                "skills": skills_out,
            }
            cache.set(cache_key, result)
            print(f"Row {row_num}: Profile normalized")
            return dict(result)
        except Exception as e:  # LLMError (transport or unparseable JSON) or malformed shape
            print(f"Row {row_num}: Profile normalization attempt {attempt + 1} failed: {e!s}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    print(f"Row {row_num}: All retries failed. Keeping original data.")
    return {
        "education_level": edu or "N/A",
        "program_level": prog or "N/A",  # original text; pattern step flags it for review
        "field_of_expertise": field or "N/A",
        "skills": ", ".join(t for t in (field, spec, skills) if t).lower() or "N/A",
    }


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
    
    # Normalize education, program level, broad field and skills in ONE pass (one LLM call per row)
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
            profile = normalize_mentee_row(
                education_level, current_program, original_field, specialization, hard_skills,
                client, cache, idx + 1
            )
            # Pattern normalization catches any variations the model returns
            education = normalize_education_level_with_patterns(profile["education_level"], unmapped_education_levels)
            level = profile["program_level"]
            # Handle "P", "B", "ERROR: No level found", etc.
            if "ERROR" in str(level) or "No level found" in str(level):
                level = "N/A"
            else:
                level = normalize_education_level_with_patterns(level, unmapped_program_levels)
            normalized_education.append(education)
            split_levels.append(level)
            broad_fields.append(profile["field_of_expertise"])
            merged_skills.append(profile["skills"])
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

