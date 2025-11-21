"""
Enhanced Mentee Skills Merger and Normalizer
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

# Import normalization utilities
from normalization import (
    normalize_university_name, normalize_country_name,
    normalize_education_level_with_patterns, normalize_text_general,
    normalize_languages
)

# Configuration
MODEL = "gpt-4o-mini"
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


def setup_openai():
    """Setup OpenAI API key from environment variable"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
        os.environ['OPENAI_API_KEY'] = api_key
    
    openai.api_key = api_key
    return openai


def normalize_education_level(education_level: str, client, cache: AICache, row_num: int) -> str:
    """
    Use OpenAI to normalize education level to one of: Masters, PhD, High School, Bachelors
    With caching support.
    """
    if pd.isna(education_level):
        return "N/A"
    
    education_text = str(education_level).strip()
    if not education_text:
        return "N/A"
    
    # Check cache
    cache_key = cache.get_cache_key(education_text)
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Row {row_num}: Education level (cached): '{cached_result}'")
        return cached_result
    
    prompt = f"""You are an education level normalizer. Your task is to map any education level description to exactly ONE of these standardized values:
- Masters
- PhD
- High School
- Bachelors

Input: "{education_text}"

Return ONLY one of the four standardized values above, nothing else. If it's a Master's degree (any variation), return "Masters". If it's a PhD/Doctorate (any variation), return "PhD". If it's a Bachelor's degree (any variation), return "Bachelors". If it's High School (any variation), return "High School"."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You only return one of these exact words: Masters, PhD, High School, Bachelors. Nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate the result
            valid_values = ["Masters", "PhD", "High School", "Bachelors"]
            if result in valid_values:
                cache.set(cache_key, result)
                print(f"Row {row_num}: Education level normalized to '{result}'")
                return result
            else:
                print(f"Row {row_num}: Unexpected result '{result}', keeping original")
                return education_text
                
        except Exception as e:
            print(f"Row {row_num}: Education normalization attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Row {row_num}: All retries failed. Keeping original.")
                return education_text


def split_current_education_program(program: str, client, cache: AICache, row_num: int) -> Tuple[str, str]:
    """
    Use OpenAI to split current education program into level and specialization
    Returns: (level, specialization) where level is one of: PhD, Masters, Bachelors
    With caching support.
    """
    if pd.isna(program):
        return "N/A", "N/A"
    
    program_text = str(program).strip()
    if not program_text:
        return "N/A", "N/A"
    
    # Check cache
    cache_key = cache.get_cache_key(program_text)
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Row {row_num}: Program split (cached): Level='{cached_result[0]}', Specialization='{cached_result[1]}'")
        return cached_result[0], cached_result[1]
    
    prompt = f"""You are an education program parser. Your task is to extract two pieces of information from an education program description:

Input: "{program_text}"

Extract:
1. LEVEL: Must be exactly one of: PhD, Masters, Bachelors (normalize any variations)
2. SPECIALIZATION: The field of study or program name

Return ONLY in this exact JSON format:
{{"level": "PhD", "specialization": "Computer Science"}}

If there is NO specialization mentioned in the input, return:
{{"level": "PhD", "specialization": "ERROR: No specialization found"}}

Examples:
- "Ph.D. in Computer Science" → {{"level": "PhD", "specialization": "Computer Science"}}
- "Master's program in Data Science" → {{"level": "Masters", "specialization": "Data Science"}}
- "PhD" → {{"level": "PhD", "specialization": "ERROR: No specialization found"}}
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You only return valid JSON with 'level' and 'specialization' keys. Level must be PhD, Masters, or Bachelors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                parsed = json.loads(result)
                level = parsed.get("level", "ERROR")
                specialization = parsed.get("specialization", "ERROR: No specialization found")
                
                # Cache result
                cache.set(cache_key, (level, specialization))
                
                print(f"Row {row_num}: Split program into Level='{level}', Specialization='{specialization}'")
                return level, specialization
                
            except json.JSONDecodeError:
                print(f"Row {row_num}: Failed to parse JSON response: {result}")
                if attempt < MAX_RETRIES - 1:
                    continue
                else:
                    return "ERROR", f"ERROR: Could not parse '{program_text}'"
                
        except Exception as e:
            print(f"Row {row_num}: Program split attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Row {row_num}: All retries failed. Keeping original.")
                return program_text, "ERROR: Processing failed"
    
    return "ERROR", "ERROR: Processing failed"


def determine_broad_field_of_expertise(original_field: str, extracted_specialization: str, 
                                       client, cache: AICache, row_num: int) -> str:
    """
    Use OpenAI to determine a broad field of expertise category
    With caching support.
    """
    # Handle empty values
    original_text = str(original_field).strip() if pd.notna(original_field) else ""
    extracted_text = str(extracted_specialization).strip() if pd.notna(extracted_specialization) else ""
    
    # Skip ERROR values
    if "ERROR" in extracted_text:
        extracted_text = ""
    
    # If both are empty, return N/A
    if not original_text and not extracted_text:
        return "N/A"
    
    # Check cache (use combined input as key)
    cache_input = f"{original_text}|{extracted_text}"
    cache_key = cache.get_cache_key(cache_input)
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Row {row_num}: Field of expertise (cached): '{cached_result}'")
        return cached_result
    
    prompt = f"""You are an academic field classifier. Based on the information provided, determine the BROAD field of expertise category.

Original Field of Expertise: {original_text}
Specialization from Current Program: {extracted_text}

Your task:
- Identify the BROAD academic/professional field category
- Examples of broad categories: Biomedicine, Mechanical Engineering, Computer Science, Business, Law, Psychology, Physics, Chemistry, etc.
- Return ONLY the broad category name (1-3 words maximum), nothing else
- Capitalize properly (e.g., "Computer Science" not "computer science")

Return only the broad field category:"""

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
            
            result = response.choices[0].message.content.strip()
            result = result.strip('."\'')
            
            # Cache result
            cache.set(cache_key, result)
            
            print(f"Row {row_num}: Field of expertise determined as '{result}'")
            return result
            
        except Exception as e:
            print(f"Row {row_num}: Field determination attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Row {row_num}: All retries failed. Using original field.")
                return original_text if original_text else "N/A"


def merge_and_normalize_skills(field_of_expertise: str, extracted_specialization: str, 
                               specialization: str, hard_skills: str, client, cache: AICache, row_num: int) -> str:
    """
    Use OpenAI to merge and normalize skills from all four sources
    With caching support.
    """
    # Handle empty values
    field_text = str(field_of_expertise).strip() if pd.notna(field_of_expertise) else ""
    extracted_text = str(extracted_specialization).strip() if pd.notna(extracted_specialization) else ""
    spec_text = str(specialization).strip() if pd.notna(specialization) else ""
    skills_text = str(hard_skills).strip() if pd.notna(hard_skills) else ""
    
    # Skip ERROR values from previous step
    if "ERROR" in extracted_text:
        extracted_text = ""
    
    # If all are empty, return N/A
    if not field_text and not extracted_text and not spec_text and not skills_text:
        return "N/A"
    
    # Check cache (use combined input as key)
    cache_input = f"{field_text}|{extracted_text}|{spec_text}|{skills_text}"
    cache_key = cache.get_cache_key(cache_input)
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Row {row_num}: Skills merged (cached)")
        return cached_result
    
    # Prepare the prompt
    prompt = f"""You are a skills normalization expert. I need you to merge and normalize four fields into a single list of keywords.

Field 1 (Original Field of expertise): {field_text}
Field 2 (Specialization from education program): {extracted_text}
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

Example output format: biomedical engineering, molecular biology, biomedicine, pcr, cell culture, crispr, python, r programming

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
                for text in [field_text, extracted_text, spec_text, skills_text]:
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


def process_mentee_file(input_file: str, output_file: str = None, dry_run: bool = False, 
                       test_rows: int = None, client=None, cache=None, report=None):
    """
    Process mentee file from PRE to POST format.
    
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
        report = TransformationReport("Mentee Skills Merger")
    
    # Lists to track unmapped values for review
    unmapped_universities = []
    unmapped_countries = []
    unmapped_education_levels = []
    unmapped_program_levels = []
    unmapped_languages = []
    
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
    
    # STEP 1: Normalize education levels
    print("\n5. Normalizing education levels...")
    print("-" * 80)
    normalized_education = []
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        education_level = row[pre_cols['education_level']] if pre_cols['education_level'] else "N/A"
        progress = f"[{idx + 1}/{total_rows}]"
        print(f"{progress}", end=" ", flush=True)
        
        try:
            normalized = normalize_education_level(education_level, client, cache, idx + 1)
            # Apply pattern normalization to catch any variations
            normalized = normalize_education_level_with_patterns(normalized, unmapped_education_levels)
            normalized_education.append(normalized)
            report.record_operation('education_levels_normalized')
            report.record_success(idx + 1)
        except Exception as e:
            print(f"\nRow {idx + 1}: Error normalizing education level: {str(e)}")
            normalized_education.append("N/A")
            report.record_failure(idx + 1, "Education normalization", str(e))
        
        time.sleep(0.1)
    
    df_output[MENTEE_COLUMNS['education_level']] = normalized_education
    print("\n" + "-" * 80)
    print(f"   ✓ Normalized all education levels")
    
    # STEP 2: Split current education program
    print("\n6. Splitting current education programs...")
    print("-" * 80)
    split_levels = []
    extracted_specializations = []
    
    for idx, row in df.iterrows():
        current_program = row[pre_cols['current_program']] if pre_cols['current_program'] else "N/A"
        progress = f"[{idx + 1}/{total_rows}]"
        print(f"{progress}", end=" ", flush=True)
        
        try:
            level, specialization = split_current_education_program(current_program, client, cache, idx + 1)
            # Normalize level using patterns (handle "P", "B", "ERROR: No level found", etc.)
            if level == "ERROR" or "ERROR" in str(level) or "No level found" in str(level):
                level = "N/A"
            else:
                level = normalize_education_level_with_patterns(level, unmapped_program_levels)
            split_levels.append(level)
            extracted_specializations.append(specialization)
            report.record_operation('education_programs_split')
        except Exception as e:
            print(f"\nRow {idx + 1}: Error splitting program: {str(e)}")
            split_levels.append("N/A")
            extracted_specializations.append("N/A")
            report.record_failure(idx + 1, "Program splitting", str(e))
        
        time.sleep(0.1)
    
    df_output[MENTEE_COLUMNS['current_program']] = split_levels
    print("\n" + "-" * 80)
    print(f"   ✓ Split all education programs")
    
    # STEP 3: Determine broad Field of Expertise
    print("\n7. Determining broad Field of Expertise categories...")
    print("-" * 80)
    broad_fields = []
    
    for idx, row in df.iterrows():
        original_field = original_field_of_expertise.iloc[idx] if len(original_field_of_expertise) > idx else "N/A"
        extracted_spec = extracted_specializations[idx] if idx < len(extracted_specializations) else "N/A"
        
        progress = f"[{idx + 1}/{total_rows}]"
        print(f"{progress}", end=" ", flush=True)
        
        try:
            broad_field = determine_broad_field_of_expertise(
                original_field,
                extracted_spec,
                client,
                cache,
                idx + 1
            )
            broad_fields.append(broad_field)
            report.record_operation('fields_classified')
        except Exception as e:
            print(f"\nRow {idx + 1}: Error classifying field: {str(e)}")
            broad_fields.append("N/A")
            report.record_failure(idx + 1, "Field classification", str(e))
        
        time.sleep(0.1)
    
    # Normalize field names (remove accents, trim whitespace)
    broad_fields_normalized = [normalize_text_general(field) if pd.notna(field) else "N/A" for field in broad_fields]
    df_output[MENTEE_COLUMNS['field']] = broad_fields_normalized
    print("\n" + "-" * 80)
    print(f"   ✓ Determined broad field categories")
    
    # STEP 4: Merge and normalize all specialization/skills fields
    print("\n8. Merging and normalizing specializations and skills...")
    print("-" * 80)
    merged_skills = []
    
    for idx, row in df.iterrows():
        original_field = original_field_of_expertise.iloc[idx] if len(original_field_of_expertise) > idx else "N/A"
        extracted_spec = extracted_specializations[idx] if idx < len(extracted_specializations) else "N/A"
        specialization = row[pre_cols['specialization']] if pre_cols['specialization'] else "N/A"
        hard_skills = row[pre_cols['hard_skills']] if pre_cols['hard_skills'] else "N/A"
        
        progress = f"[{idx + 1}/{total_rows}]"
        print(f"{progress}", end=" ", flush=True)
        
        try:
            normalized = merge_and_normalize_skills(
                original_field,
                extracted_spec,
                specialization,
                hard_skills,
                client,
                cache,
                idx + 1
            )
            merged_skills.append(normalized)
            report.record_operation('skills_merged')
        except Exception as e:
            print(f"\nRow {idx + 1}: Error merging skills: {str(e)}")
            merged_skills.append("N/A")
            report.record_failure(idx + 1, "Skills merging", str(e))
        
        time.sleep(0.1)
    
    # Normalize specialization (remove accents, trim whitespace)
    merged_skills_normalized = [normalize_text_general(skill) if pd.notna(skill) else "N/A" for skill in merged_skills]
    df_output[MENTEE_COLUMNS['specialization']] = merged_skills_normalized
    print("\n" + "-" * 80)
    print(f"   ✓ Merged and normalized all skills")
    
    # Clean data (NaN → N/A, trim whitespace)
    print("\n9. Cleaning data...")
    df_output = clean_dataframe(df_output)
    
    # Find missing data
    missing = find_missing_data(df_output, MENTEE_COLUMNS)
    report.set_missing_data(missing)
    
    # Reorder columns to match POST format
    print("\n10. Reordering columns...")
    column_order = [MENTEE_COLUMNS[key] for key in ['id', 'affiliation', 'country', 'education_level', 
                                                     'current_program', 'field', 'specialization', 
                                                     'languages', 'guidance_areas', 'career_goals', 'other_info']]
    df_output = df_output[column_order]
    print("   ✓ Columns reordered")
    
    # Validate required columns
    print("\n11. Validating output...")
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
        print(f"\n12. Saving output file...")
        df_output.to_excel(output_file, index=False)
        print(f"   ✓ Output saved to: {output_file}")
    else:
        print(f"\n12. DRY-RUN MODE: Skipping file write")
        print(f"   Would save to: {output_file}")
    
    # Show sample results
    print("\n13. Sample results (first 3 rows):")
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
    print("\n0. Setting up OpenAI API...")
    client = setup_openai()
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
        print(f"\n\n✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

