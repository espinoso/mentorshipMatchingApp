"""
Enhanced Mentor Skills Merger and Normalizer
Transforms PRE format to POST format with full automation.
"""

import argparse
import os
import sys
import time

import pandas as pd

# Import institution type normalization
from institution_types import normalize_institution_type

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
    sanitize_spreadsheet_df,
    setup_claude,
    validate_required_columns,
    write_error_log,
)

# Import normalization utilities
from normalization import (
    normalize_education_level_with_patterns,
    normalize_mentor_alma_mater,
    normalize_mentor_country,
    normalize_mentor_country_of_origin,
    normalize_mentor_field_of_expertise,
)

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
    'country': ['affiliated institution', 'affiliated university'],
    'current_position': ['current position'],
    'institution_type': ['type of institution', 'institution type'],
    'country_of_origin': ['country of origin'],
    'education_level': ['highest educational level', 'highest education level'],
    'alma_mater': ['institution & country where bachelor', 'bachelor\'s degree'],
    'field': ['field of expertise'],
    'specialization': ['specialization'],
    'hard_skills': ['specific hard skills', 'professional competencies'],
    'experience_years': ['years of professional experience'],
    'career_dev_experience': ['areas where you feel comfortable mentoring', 'comfortable mentoring']
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


def process_mentor_file(input_file: str, output_file: str = None, dry_run: bool = False, 
                       test_rows: int = None, client=None, cache=None, report=None):
    """
    Process mentor file from PRE to POST format.
    
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
        report = TransformationReport("Mentor Skills Merger")
    
    # Track transformations for logging
    institution_type_transformations = []
    unmapped_countries = []
    unmapped_countries_of_origin = []
    unmapped_alma_maters = []
    unmapped_fields = []
    
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
    critical_cols = ['education_level', 'field', 'specialization', 'hard_skills']  # institution_type is optional (not in every survey)
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
    
    if pre_cols['experience_years']:
        df_output[MENTOR_COLUMNS['experience_years']] = df[pre_cols['experience_years']].fillna("N/A")
    else:
        df_output[MENTOR_COLUMNS['experience_years']] = "N/A"
    
    if pre_cols['career_dev_experience']:
        df_output[MENTOR_COLUMNS['career_dev_experience']] = df[pre_cols['career_dev_experience']].fillna("N/A")
    else:
        df_output[MENTOR_COLUMNS['career_dev_experience']] = "N/A"
    
    print("   ✓ Copied simple columns")
    
    # Normalize institution types (dictionary lookup, no LLM, so no per-row progress output)
    print("\n5. Normalizing institution types...")
    normalized_institution_types = []
    original_institution_types_for_spec = []  # Store originals that should go to specialization
    
    for idx, row in df.iterrows():
        institution_type = row[pre_cols['institution_type']] if pre_cols['institution_type'] else "N/A"
        
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
            print(f"\nRow {idx + 1}: Error normalizing institution type: {e!s}")
            normalized_institution_types.append("N/A")
            original_institution_types_for_spec.append("")
            report.record_failure(idx + 1, "Institution type normalization", str(e))
    
    df_output[MENTOR_COLUMNS['institution_type']] = normalized_institution_types
    print("   ✓ Normalized all institution types")
    
    # Show transformation log
    if institution_type_transformations:
        print("\n   Institution Type Transformations (for review):")
        for trans in institution_type_transformations:
            print(f"     {trans}")
    
    # Education level and skills are computed in code; the field mapper only calls the LLM for
    # fields the dictionary/patterns can't map (one call per row at most)
    print("\n6. Normalizing profiles (fields, skills, education)...")
    print("-" * 80)
    normalized_fields, extracted_levels, merged_skills = [], [], []
    
    for idx, row in df.iterrows():
        field = row[pre_cols['field']] if pre_cols['field'] else "N/A"
        education_text = row[pre_cols['education_level']] if pre_cols['education_level'] else "N/A"
        specialization = row[pre_cols['specialization']] if pre_cols['specialization'] else "N/A"
        hard_skills = row[pre_cols['hard_skills']] if pre_cols['hard_skills'] else "N/A"
        
        print(f"[{idx + 1}/{len(df)}]", end=" ", flush=True)
        
        try:
            broad_category, field_terms = normalize_mentor_field_of_expertise(
                field, client, cache, idx + 1, unmapped_fields
            )
            level = classify_education_level(education_text)
            if level == "N/A":  # keyword miss: try the pattern table
                level = normalize_education_level_with_patterns(education_text)
            if level == "N/A":  # still unknown and present: LLM fallback
                level = classify_level_llm(education_text, client, cache)
            # Order: broader -> specific (field terms, education field, institution type, specialization, skills)
            skills = merge_skill_sources(
                ", ".join(field_terms),
                extract_program_specialization(education_text),
                original_institution_types_for_spec[idx],
                specialization,
                hard_skills,
            )
            
            normalized_fields.append(broad_category)
            extracted_levels.append(level)
            merged_skills.append(skills)
            for op in ('fields_classified', 'education_levels_normalized', 'skills_merged'):
                report.record_operation(op)
        except Exception as e:
            print(f"\nRow {idx + 1}: Error normalizing profile: {e!s}")
            normalized_fields.append("N/A")
            extracted_levels.append("N/A")
            merged_skills.append("N/A")
            report.record_failure(idx + 1, "Profile normalization", str(e))
        
        time.sleep(0.1)
    
    df_output[MENTOR_COLUMNS['field']] = normalized_fields
    df_output[MENTOR_COLUMNS['education_level']] = extracted_levels
    df_output[MENTOR_COLUMNS['specialization']] = merged_skills
    print("\n" + "-" * 80)
    print("   ✓ Normalized all profiles")
    
    # Clean data (NaN → N/A, trim whitespace)
    print("\n7. Cleaning data...")
    df_output = clean_dataframe(df_output)
    
    # Find missing data
    missing = find_missing_data(df_output, MENTOR_COLUMNS)
    report.set_missing_data(missing)
    
    # Reorder columns to match POST format
    print("\n8. Reordering columns...")
    column_order = [MENTOR_COLUMNS[key] for key in ['id', 'affiliation', 'country', 'current_position',
                                                     'institution_type', 'country_of_origin', 'alma_mater',
                                                     'education_level', 'field', 'specialization',
                                                     'experience_years', 'career_dev_experience']]
    df_output = df_output[column_order]
    print("   ✓ Columns reordered")
    
    # Validate required columns
    print("\n9. Validating output...")
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
        print("\n10. Saving output file...")
        sanitize_spreadsheet_df(df_output).to_excel(output_file, index=False)
        print(f"   ✓ Output saved to: {output_file}")
    else:
        print("\n10. DRY-RUN MODE: Skipping file write")
        print(f"   Would save to: {output_file}")
    
    # Show sample results
    print("\n11. Sample results (first 3 rows):")
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
    print("\n0. Setting up Claude API...")
    client = setup_claude()
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
        print(f"\n\n✗ Fatal error: {e!s}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

