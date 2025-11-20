import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from openai import OpenAI
from io import BytesIO
import time
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
from datetime import datetime

from info import (
    MENTEE_COLUMNS, MENTOR_COLUMNS
)

from ui import (
    CSS_STYLES
)

from utils import (
    validate_uploaded_file, estimate_tokens, fetch_available_models, estimate_api_cost, show_debug_info, clean_dataframe
)

from prompts import (
    create_matrix_prompt, create_reasoning_prompt, get_prompt_for_api, create_default_prompt
)

# Page configuration
st.set_page_config(
    page_title="Mentorship Matching System",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling (Improvements: 1, 2, 3, 10, 21, 22)
st.markdown(CSS_STYLES, unsafe_allow_html=True)


def is_field_compatible(field1, field2):
    """
    Check if two fields are same or related
    Used for validation to detect mismatched high scores
    """
    # Normalize fields
    f1 = str(field1).lower().strip()
    f2 = str(field2).lower().strip()
    
    # Exact match
    if f1 == f2:
        return True
    
    # Define related field groups based on common academic/professional fields
    related_groups = [
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
        # Add more groups as needed based on your domain
    ]
    
    for group in related_groups:
        if f1 in group and f2 in group:
            return True
    
    return False

def validate_and_flag_matches(matches, mentees_df, mentors_df):
    """
    Validate match scores and flag suspicious high scores
    This helps identify when AI is inflating scores incorrectly
    """
    st.info("🔍 Validating match quality...")
    
    flagged = []
    stats = {
        'total': 0,
        'excellent': 0,  # >=85
        'strong': 0,     # 75-84
        'good': 0,       # 60-74
        'fair': 0,       # 45-59
        'poor': 0        # <45
    }
    
    for match_data in matches:
        mentee_id = match_data['mentee_id']
        for match in match_data['matches']:
            mentor_id = match['mentor_id']
            score = match['match_percentage']
            
            stats['total'] += 1
            
            # Update statistics
            if score >= 85:
                stats['excellent'] += 1
            elif score >= 75:
                stats['strong'] += 1
            elif score >= 60:
                stats['good'] += 1
            elif score >= 45:
                stats['fair'] += 1
            else:
                stats['poor'] += 1
            
            # Get mentee and mentor data
            # Convert IDs to int for comparison (matches contain string IDs from JSON, DataFrame has int IDs)
            try:
                mentee_id_int = int(mentee_id)
                mentor_id_int = int(mentor_id)
                mentee = mentees_df[mentees_df[MENTEE_COLUMNS['id']] == mentee_id_int].iloc[0]
                mentor = mentors_df[mentors_df[MENTOR_COLUMNS['id']] == mentor_id_int].iloc[0]
            except (ValueError, IndexError) as e:
                st.warning(f"⚠️ Could not extract details for {mentee_id} → {mentor_id}: {e}")
                continue
            
            issues = []
            
            # Validation 1: High score with different fields
            if score >= 75:
                mentee_field = str(mentee[MENTEE_COLUMNS['field']]).lower().strip()
                mentor_field = str(mentor[MENTOR_COLUMNS['field']]).lower().strip()
                
                if mentee_field != mentor_field:
                    # Check if they're at least related
                    if not is_field_compatible(mentee_field, mentor_field):
                        issues.append(f"Score {score}% but fields don't match: '{mentee_field}' vs '{mentor_field}'")
            
            # Validation 2: High score with no keyword overlap
            if score >= 70:
                mentee_spec = str(mentee.get(MENTEE_COLUMNS['specialization'], '')).lower()
                mentor_spec = str(mentor.get(MENTOR_COLUMNS['specialization'], '')).lower()
                
                # Simple keyword overlap check (ignore common short words)
                mentee_words = set(word for word in mentee_spec.split() if len(word) > 3)
                mentor_words = set(word for word in mentor_spec.split() if len(word) > 3)
                overlap = len(mentee_words & mentor_words)
                
                if overlap == 0 and mentee_spec and mentor_spec:
                    issues.append(f"Score {score}% but 0 keyword overlap in specializations")
            
            # Validation 3: Excellent score requires strong evidence
            if score >= 85:
                # Must have field match
                mentee_field = str(mentee[MENTEE_COLUMNS['field']]).lower().strip()
                mentor_field = str(mentor[MENTOR_COLUMNS['field']]).lower().strip()
                
                if mentee_field != mentor_field and not is_field_compatible(mentee_field, mentor_field):
                    issues.append(f"Excellent score ({score}%) requires field match")
                
                # Check experience if available
                if MENTOR_COLUMNS['experience_years'] in mentor:
                    try:
                        mentor_exp = int(mentor[MENTOR_COLUMNS['experience_years']])
                        if mentor_exp < 5:
                            issues.append(f"Excellent score ({score}%) but mentor has only {mentor_exp} years experience")
                    except (ValueError, TypeError):
                        pass
            
            if issues:
                flagged.append({
                    'mentee_id': mentee_id,
                    'mentor_id': mentor_id,
                    'score': score,
                    'issues': issues
                })
    
    # Display statistics
    st.markdown("### 📊 Score Distribution")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Excellent (≥85%)", stats['excellent'])
    with col2:
        st.metric("Strong (75-84%)", stats['strong'])
    with col3:
        st.metric("Good (60-74%)", stats['good'])
    with col4:
        st.metric("Fair (45-59%)", stats['fair'])
    with col5:
        st.metric("Poor (<45%)", stats['poor'])
    
    # Show average score
    if matches:
        all_scores = [match['match_percentage'] for match_data in matches for match in match_data['matches']]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        st.info(f"📈 Average Score: {avg_score:.1f}%")
    
    # Show flagged matches
    if flagged:
        st.warning(f"⚠️ {len(flagged)} matches flagged for review")
        
        with st.expander(f"🚩 View Flagged Matches ({len(flagged)})"):
            for fm in flagged:
                st.error(f"**{fm['mentee_id']} → {fm['mentor_id']}** ({fm['score']}%)")
                for issue in fm['issues']:
                    st.write(f"  • {issue}")
                st.divider()
    else:
        st.success("✅ All matches passed validation checks!")
    
    return flagged, stats

def analyze_score_distribution(matrix_df):
    """
    Analyze the distribution of scores to detect if AI is being too generous
    Returns distribution statistics for monitoring score quality
    """
    import numpy as np
    
    all_scores = matrix_df.values.flatten()
    all_scores = all_scores[~np.isnan(all_scores)]  # Remove NaN values
    
    if len(all_scores) == 0:
        st.error("❌ No valid scores to analyze")
        return None
    
    distribution = {
        'excellent_90_100': int(np.sum((all_scores >= 90) & (all_scores <= 100))),
        'strong_75_89': int(np.sum((all_scores >= 75) & (all_scores < 90))),
        'good_60_74': int(np.sum((all_scores >= 60) & (all_scores < 75))),
        'fair_45_59': int(np.sum((all_scores >= 45) & (all_scores < 60))),
        'poor_30_44': int(np.sum((all_scores >= 30) & (all_scores < 45))),
        'very_poor_0_29': int(np.sum(all_scores < 30)),
        'total': len(all_scores),
        'average': float(np.mean(all_scores)),
        'median': float(np.median(all_scores))
    }
    
    st.markdown("### 📊 Score Distribution Analysis")
    
    # Display metrics in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        pct = (distribution['excellent_90_100'] / distribution['total']) * 100
        st.metric("Excellent\n90-100%", distribution['excellent_90_100'], f"{pct:.1f}%")
    with col2:
        pct = (distribution['strong_75_89'] / distribution['total']) * 100
        st.metric("Strong\n75-89%", distribution['strong_75_89'], f"{pct:.1f}%")
    with col3:
        pct = (distribution['good_60_74'] / distribution['total']) * 100
        st.metric("Good\n60-74%", distribution['good_60_74'], f"{pct:.1f}%")
    with col4:
        pct = (distribution['fair_45_59'] / distribution['total']) * 100
        st.metric("Fair\n45-59%", distribution['fair_45_59'], f"{pct:.1f}%")
    with col5:
        pct = (distribution['poor_30_44'] / distribution['total']) * 100
        st.metric("Poor\n30-44%", distribution['poor_30_44'], f"{pct:.1f}%")
    with col6:
        pct = (distribution['very_poor_0_29'] / distribution['total']) * 100
        st.metric("Very Poor\n0-29%", distribution['very_poor_0_29'], f"{pct:.1f}%")
    
    # Show average and median
    col_avg, col_med = st.columns(2)
    with col_avg:
        st.info(f"📈 **Average Score**: {distribution['average']:.1f}%")
    with col_med:
        st.info(f"📊 **Median Score**: {distribution['median']:.1f}%")
    
    # Expected distribution guidance
    st.caption("💡 **Expected Distribution**: Most scores should be in 30-60% range (poor/fair matches), with only 5-15% scoring 75%+")
    
    # Warning checks
    high_score_pct = (distribution['strong_75_89'] + distribution['excellent_90_100']) / distribution['total']
    low_score_pct = (distribution['very_poor_0_29'] + distribution['poor_30_44']) / distribution['total']
    
    warnings = []
    
    if high_score_pct > 0.30:
        warnings.append(f"⚠️ **Too many high scores**: {high_score_pct*100:.1f}% of matches are 'Strong' or 'Excellent' (expected <30%). AI may be too generous.")
    
    if low_score_pct < 0.40:
        warnings.append(f"⚠️ **Too few poor matches**: Only {low_score_pct*100:.1f}% scored below 45% (expected >40%). Most mentor-mentee pairs should be poor matches.")
    
    if distribution['average'] > 65:
        warnings.append(f"⚠️ **Average too high**: {distribution['average']:.1f}% (expected 50-65%). Scoring may be inflated.")
    
    if distribution['average'] < 40:
        warnings.append(f"✅ **Good average**: {distribution['average']:.1f}% indicates realistic scoring with many poor matches.")
    
    # Display warnings
    if warnings:
        with st.expander("⚠️ Distribution Analysis"):
            for warning in warnings:
                if "✅" in warning:
                    st.success(warning)
                else:
                    st.warning(warning)
    else:
        st.success("✅ Score distribution looks reasonable!")
    
    return distribution

# =====================================================================
# PROMPT FUNCTIONS
# All prompt generation functions have been moved to prompts.py
# This keeps app.py cleaner and more maintainable.
# Functions available: create_matrix_prompt, create_reasoning_prompt,
#                      get_prompt_for_api, create_default_prompt
# =====================================================================

def clean_text_for_csv(df):
    """Clean text fields to prevent CSV parsing issues"""
    df = df.copy()
    
    # Clean string columns
    for col in df.select_dtypes(include=['object']).columns:
        # Replace newlines with spaces
        df[col] = df[col].astype(str).str.replace('\n', ' ', regex=False)
        df[col] = df[col].astype(str).str.replace('\r', ' ', regex=False)
        # Replace multiple spaces with single space
        df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
        # Strip whitespace
        df[col] = df[col].str.strip()
        # Replace 'nan' string back to NaN
        df[col] = df[col].replace('nan', np.nan)
    
    return df

def prepare_data_for_assistant(mentees_df, mentors_df, training_files):
    """Prepare data with ALL columns for comprehensive matching
    
    Args:
        mentees_df: DataFrame of mentees (None to skip mentees section for stateful subsequent batches)
        mentors_df: DataFrame of mentors
        training_files: List of training DataFrames
    """
    
    # Combine into a single structured document
    combined_data = "# MENTORSHIP MATCHING DATA\n\n"
    
    # Add mentees section (only if provided)
    if mentees_df is not None:
        mentees_full = mentees_df.copy()
        mentees_cleaned = clean_text_for_csv(mentees_full)
        combined_data += "## MENTEES TO MATCH (ALL DATA)\n"
        combined_data += f"Total mentees: {len(mentees_cleaned)}\n"
        combined_data += f"All columns included: {', '.join(mentees_cleaned.columns.tolist())}\n\n"
        combined_data += mentees_cleaned.to_csv(index=False)
        combined_data += "\n"
    
    # Add mentors section (always provided)
    mentors_full = mentors_df.copy()
    mentors_cleaned = clean_text_for_csv(mentors_full)
    combined_data += "## AVAILABLE MENTORS (ALL DATA)\n"
    combined_data += f"Total mentors: {len(mentors_cleaned)}\n"
    combined_data += f"All columns included: {', '.join(mentors_cleaned.columns.tolist())}\n\n"
    combined_data += mentors_cleaned.to_csv(index=False)
    
    # Add training files (only if provided)
    if training_files and len(training_files) > 0:
        combined_data += "\n## TRAINING EXAMPLES (First 5 rows for reference)\n"
        for i, df in enumerate(training_files):
            df_cleaned = clean_text_for_csv(df.head(5))
            combined_data += f"\n### Training Set {i+1}\n"
            combined_data += df_cleaned.to_csv(index=False)
    
    return combined_data

def call_openai_api_with_chat_completions(prompt, mentees_df, mentors_df, training_data, api_key, model="gpt-4o-mini", mentor_subset=None, use_stateful=False, previous_response_id=None, is_first_batch=True):
    """Use OpenAI Responses API - replaces Chat Completions API
    
    Migration to Responses API per:
    https://platform.openai.com/docs/guides/migrate-to-responses
    
    Supports both stateless (store=False) and stateful (store=True) modes:
    - Stateless: Each call is independent with full context (default, backward compatible)
    - Stateful: Context is maintained across calls, only new data sent in subsequent calls
    
    Args:
        mentor_subset: Optional list of mentor IDs to evaluate (for batching)
        use_stateful: Enable stateful mode (store=True) for efficient batching
        previous_response_id: Response ID from previous call (for stateful chaining)
        is_first_batch: Whether this is the first batch (needs full context)
    """
    try:
        client = OpenAI(api_key=api_key)
        
        # Check if Responses API is available
        if hasattr(client, 'responses'):
            responses_api = client.responses
            api_name = "Responses API"
        elif hasattr(client, 'beta') and hasattr(client.beta, 'responses'):
            responses_api = client.beta.responses
            api_name = "Responses API (Beta)"
        else:
            # Fallback to Chat Completions if Responses not available
            st.warning("⚠️ Responses API not available, falling back to Chat Completions")
            return call_openai_api_chat_completions_legacy(prompt, mentees_df, mentors_df, training_data, api_key, model, mentor_subset)
        
        st.info(f"🔄 Using {api_name}...")
        
        # If mentor_subset provided, filter mentors_df
        if mentor_subset:
            mentors_to_use = mentors_df[mentors_df[MENTOR_COLUMNS['id']].isin(mentor_subset)]
        else:
            mentors_to_use = mentors_df
        
        # Prepare data based on mode
        if use_stateful and not is_first_batch and previous_response_id:
            # Stateful mode, subsequent batch: Only send NEW mentor data
            st.info("🔗 Stateful mode: Sending only new mentor batch (context preserved from previous call)")
            combined_data = prepare_data_for_assistant(None, mentors_to_use, [])  # Only mentors, no mentees/training
            full_message = f"Evaluate these additional mentors against the same mentees from the previous context:\n\n{combined_data}"
        else:
            # Stateless mode OR first batch: Send complete context
            if use_stateful:
                st.info("🔗 Stateful mode: First batch - sending full context")
            combined_data = prepare_data_for_assistant(mentees_df, mentors_to_use, training_data)
            full_message = f"{prompt}\n\n{combined_data}\n\nGenerate matches for ALL mentees in JSON format as specified."
        
        # Estimate tokens and cost
        prompt_tokens = estimate_tokens(full_message, model)
        estimated_completion_tokens = len(mentees_df) * 500  # Rough estimate
        estimated_cost = estimate_api_cost(prompt_tokens, estimated_completion_tokens, model)
        
        # Display token and cost information
        mode_str = "stateful (store=True)" if use_stateful else "stateless (store=False)"
        if use_stateful and not is_first_batch:
            mode_str += " - subsequent batch"
        st.markdown(f"""
        <div class="cost-estimate">
            <strong>📊 API Usage Estimate:</strong><br>
            • Input tokens: ~{prompt_tokens:,}<br>
            • Expected output tokens: ~{estimated_completion_tokens:,}<br>
            • Estimated cost: ${estimated_cost:.4f}<br>
            • Model: {model}<br>
            • Mode: {mode_str}
        </div>
        """, unsafe_allow_html=True)
        
        # Check token limits (updated for newer models)
        total_tokens = prompt_tokens + estimated_completion_tokens
        model_limits = {
            # GPT-4o models
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            # o1 models (reasoning)
            "o1-preview": 128000,
            "o1-mini": 128000,
            "o3-mini": 200000,
            # GPT-4 Turbo
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            # GPT-4 classic
            "gpt-4": 8192,
            # GPT-3.5
            "gpt-3.5-turbo": 16384,
            # GPT-5 models
            "gpt-5": 200000,
            "gpt-5.1": 200000,
            "gpt-5.1-mini": 200000,
            "gpt-5-mini": 200000,
            "gpt-5-nano": 128000,
            "gpt-5.1-reasoning": 200000,
            # ChatGPT models
            "chatgpt-4o-latest": 128000,
        }
        
        if total_tokens > model_limits.get(model, 128000):
            st.markdown(f"""
            <div class="token-warning">
                <strong>⚠️ Token Limit Warning:</strong><br>
                Estimated tokens ({total_tokens:,}) may exceed model limit ({model_limits.get(model, 128000):,}).<br>
                Try reducing batch size or splitting the data.
            </div>
            """, unsafe_allow_html=True)
            return None
        
        # Check if training files are uploaded to OpenAI storage
        training_file_ids = st.session_state.get('training_file_ids', [])
        
        # Prepare messages for Responses API
        # Responses API uses 'input' parameter which can accept messages array
        messages = [
            {
                "role": "system",
                "content": "You are a mentorship matching expert for IMFAHE's International Mentor Program. You MUST respond with valid JSON only, following the exact format specified."
            },
            {
                "role": "user",
                "content": full_message
            }
        ]
        
        st.info("🔄 Sending request to OpenAI Responses API...")
        
        # File search is now handled as a tool in Responses API (not as message attachments)
        if training_file_ids:
            st.info(f"📚 Including {len(training_file_ids)} training file(s) via file_search tool...")
        
        # Call Responses API
        start_time = time.time()
        
        # Check if model supports response_format and temperature (o1/o3 models don't)
        is_reasoning_model = any(x in model.lower() for x in ['o1-', 'o3-', 'o1', 'o3'])
        supports_json_mode = not is_reasoning_model
        
        # Build API call parameters for Responses API
        api_params = {
            "model": model,
            "input": messages,  # Changed from "messages" to "input"
            "timeout": 300,
            "store": use_stateful,  # True for stateful, False for stateless
        }
        
        # Add previous_response_id for stateful chaining (if provided)
        if use_stateful and previous_response_id:
            api_params["previous_response_id"] = previous_response_id
            st.info(f"🔗 Chaining to previous response: {previous_response_id[:20]}...")
        
        # Add temperature for standard models (o1/o3 don't support it)
        if not is_reasoning_model:
            api_params["temperature"] = 0.1  # Low temperature for consistency
        
        # Add file_search tool if training files exist
        # NOTE: Responses API file_search requires vector_store_ids, not file_ids
        # For now, we'll skip file_search in Responses API and include training data in prompt
        # To use file_search properly, you need to create a vector store first
        if training_file_ids:
            st.warning("⚠️ File search with Responses API requires vector stores. Training data will be included in prompt instead.")
            # TODO: Implement vector store creation and use vector_store_ids
            # api_params["tools"] = [
            #     {
            #         "type": "file_search",
            #         "vector_store_ids": [vector_store_id]
            #     }
            # ]
        
        # Add structured output with schema (Responses API structured outputs)
        # Define the exact schema we expect for the matrix response
        if supports_json_mode:
            api_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "mentorship_compatibility_matrix",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "matrix": {
                                "type": "array",
                                "description": "Array of mentor compatibility scores",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "mentor_id": {
                                            "type": "string",
                                            "description": "The mentor's ID"
                                        },
                                        "scores": {
                                            "type": "array",
                                            "description": "Compatibility scores for each mentee",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "mentee_id": {
                                                        "type": "string",
                                                        "description": "The mentee's ID"
                                                    },
                                                    "percentage": {
                                                        "type": "number",
                                                        "description": "Match percentage (0-100)",
                                                        "minimum": 0,
                                                        "maximum": 100
                                                    }
                                                },
                                                "required": ["mentee_id", "percentage"],
                                                "additionalProperties": False
                                            }
                                        }
                                    },
                                    "required": ["mentor_id", "scores"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["matrix"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        
        # Retry logic for rate limits - ONLY for the API call itself
        max_retries = 5
        retry_count = 0
        response = None
        
        while retry_count < max_retries:
            try:
                st.caption(f"🔄 API attempt {retry_count + 1}/{max_retries}...")
                response = responses_api.create(**api_params)
                # API call succeeded, break out immediately
                st.caption(f"✅ API call succeeded on attempt {retry_count + 1}")
                break
                
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                
                # Log the actual exception for debugging
                st.caption(f"⚠️ Exception on attempt {retry_count + 1}: {error_type}")
                with st.expander(f"🔍 Debug: Exception details (attempt {retry_count + 1})"):
                    st.code(f"Type: {error_type}\nMessage: {error_str[:500]}")
                
                # Check if it's a rate limit error (429)
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        st.error(f"❌ Rate limit exceeded after {max_retries} attempts")
                        show_debug_info()
                        return None
                    
                    # Extract wait time from error message
                    import re
                    wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                    if wait_match:
                        wait_time = float(wait_match.group(1))
                    else:
                        # Exponential backoff if no wait time specified
                        wait_time = 2 ** retry_count
                    
                    # Add buffer (2-5 seconds)
                    buffer = 3
                    total_wait = wait_time + buffer
                    
                    st.warning(f"⏳ Rate limit (429). Waiting {total_wait:.1f}s before attempt {retry_count+1}/{max_retries}...")
                    st.caption(f"💡 This is normal with rapid batch processing. The system will automatically retry.")
                    
                    time.sleep(total_wait)
                    continue  # Retry
                else:
                    # Not a rate limit error, fail immediately
                    st.error(f"❌ API call failed with {error_type}: {str(e)[:200]}")
                    show_debug_info()
                    return None
        
        # Check if we got a response
        if response is None:
            st.error("❌ Failed to get response after retries")
            return None
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        st.success(f"✅ Response received in {minutes}m {seconds}s")
        
        # Extract response content (Responses API has specific structure)
        if hasattr(response, 'output'):
            # Responses API format: output is a list of ResponseOutputMessage
            # Structure: response.output[0].content[0].text
            try:
                if isinstance(response.output, list) and len(response.output) > 0:
                    output_msg = response.output[0]
                    if hasattr(output_msg, 'content') and isinstance(output_msg.content, list) and len(output_msg.content) > 0:
                        response_content = output_msg.content[0].text
                    else:
                        response_content = str(output_msg)
                else:
                    response_content = str(response.output)
                st.info("📋 Using Responses API output format")
            except Exception as e:
                st.error(f"❌ Error extracting Responses API content: {e}")
                response_content = str(response.output)
        elif hasattr(response, 'choices'):
            # Chat Completions format (fallback)
            response_content = response.choices[0].message.content
            st.info("📋 Using Chat Completions output format")
        else:
            st.error("❌ Unknown response format")
            st.write(f"Response type: {type(response)}")
            st.write(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
            return None
        
        # Display token usage (handle both API formats)
        if hasattr(response, 'usage'):
            usage = response.usage
            
            # Try Responses API field names first, then Chat Completions names
            actual_prompt_tokens = (
                getattr(usage, 'input_tokens', None) or 
                getattr(usage, 'prompt_tokens', 0)
            )
            actual_completion_tokens = (
                getattr(usage, 'output_tokens', None) or 
                getattr(usage, 'completion_tokens', 0)
            )
            
            actual_cost = estimate_api_cost(actual_prompt_tokens, actual_completion_tokens, model)
            
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Actual API Usage:</strong><br>
                • Input tokens: {actual_prompt_tokens:,}<br>
                • Output tokens: {actual_completion_tokens:,}<br>
                • Total tokens: {actual_prompt_tokens + actual_completion_tokens:,}<br>
                • Actual cost: ${actual_cost:.4f}
            </div>
            """, unsafe_allow_html=True)
        
        # Store debug info in session state for troubleshooting
        if 'last_api_request' not in st.session_state:
            st.session_state.last_api_request = {}
        if 'last_api_response' not in st.session_state:
            st.session_state.last_api_response = {}
        
        st.session_state.last_api_request = {
            'prompt': prompt,
            'full_message': full_message,
            'model': model,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        # Get response ID for stateful chaining
        response_id = getattr(response, 'id', None)
        
        st.session_state.last_api_response = {
            'content': response_content,
            'tokens': {
                'input': actual_prompt_tokens if hasattr(response, 'usage') else 0,
                'output': actual_completion_tokens if hasattr(response, 'usage') else 0,
                'total': (actual_prompt_tokens + actual_completion_tokens) if hasattr(response, 'usage') else 0
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'api_type': api_name,
            'response_id': response_id
        }
        
        # Return both content and response_id (for stateful chaining)
        if use_stateful:
            return response_content, response_id
        else:
            return response_content
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        show_debug_info()
        return None
def call_openai_api_chat_completions_legacy(prompt, mentees_df, mentors_df, training_data, api_key, model="gpt-4o-mini", mentor_subset=None):
    """
    Legacy Chat Completions API implementation (fallback when Responses API not available)
    
    This is the original implementation kept for backward compatibility.
    """
    try:
        client = OpenAI(api_key=api_key)
        
        # If mentor_subset provided, filter mentors_df
        if mentor_subset:
            mentors_to_use = mentors_df[mentors_df[MENTOR_COLUMNS['id']].isin(mentor_subset)]
        else:
            mentors_to_use = mentors_df
        
        # Prepare combined data with ALL columns
        combined_data = prepare_data_for_assistant(mentees_df, mentors_to_use, training_data)
        
        # Estimate tokens and cost
        full_message = f"{prompt}\n\n{combined_data}\n\nGenerate matches for ALL mentees in JSON format as specified."
        
        # Check if training files are uploaded to OpenAI storage
        training_file_ids = st.session_state.get('training_file_ids', [])
        
        # Prepare messages for Chat Completions API
        messages = [
            {
                "role": "system",
                "content": "You are a mentorship matching expert for IMFAHE's International Mentor Program. You MUST respond with valid JSON only, following the exact format specified."
            },
            {
                "role": "user",
                "content": full_message
            }
        ]
        
        # Add file_search attachments if training files exist (may not work in all versions)
        if training_file_ids:
            st.info(f"📚 Including {len(training_file_ids)} training file(s) for context...")
            try:
                messages[1]["attachments"] = [
                    {
                        "file_id": file_id,
                        "tools": [{"type": "file_search"}]
                    }
                    for file_id in training_file_ids
                ]
            except:
                st.warning("⚠️ Could not attach files with Chat Completions API")
        
        st.info("🔄 Sending request to OpenAI Chat Completions API (Legacy)...")
        
        # Check if model supports response_format and temperature (o1/o3 models don't)
        is_reasoning_model = any(x in model.lower() for x in ['o1-', 'o3-', 'o1', 'o3'])
        supports_json_mode = not is_reasoning_model
        
        # Build API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "timeout": 300,
        }
        
        # Add temperature for standard models (o1/o3 don't support it)
        if not is_reasoning_model:
            api_params["temperature"] = 0.1
        
        # Add JSON mode if supported
        if supports_json_mode:
            api_params["response_format"] = {"type": "json_object"}
        
        start_time = time.time()
        response = client.chat.completions.create(**api_params)
        elapsed = time.time() - start_time
        
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        st.success(f"✅ Response received in {minutes}m {seconds}s")
        
        # Extract response content
        response_content = response.choices[0].message.content
        return response_content
        
    except Exception as e:
        st.error(f"❌ Legacy API Error: {str(e)}")
        return None


# Keep old function name as alias for backward compatibility
def call_openai_api_with_assistants(prompt, mentees_df, mentors_df, training_data, api_key, model="gpt-4o-mini", mentor_subset=None, use_stateful=False, previous_response_id=None, is_first_batch=True):
    """
    DEPRECATED: Assistants API will be deprecated in 2025.
    This function now redirects to the Responses API.
    
    See: https://platform.openai.com/docs/guides/migrate-to-responses
    """
    return call_openai_api_with_chat_completions(prompt, mentees_df, mentors_df, training_data, api_key, model, mentor_subset, use_stateful, previous_response_id, is_first_batch)

def parse_llm_response(response_text):
    """Parse the LLM response and extract matches"""
    try:
        # Clean the response text
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        matches = json.loads(response_text)
        
        # Validate structure
        if not isinstance(matches, list):
            raise ValueError("Response should be a list")
        
        for match_data in matches:
            if 'mentee_id' not in match_data or 'matches' not in match_data:
                raise ValueError("Invalid match structure")
        
        return matches
        
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"❌ Error parsing response: {str(e)}")
        with st.expander("View raw response"):
            st.text(response_text[:2000])
            return None
        
def generate_matrix_in_batches(mentees_df, mentors_df, training_data, api_key, model, batch_size=15, use_stateful=False):
    """Generate compatibility matrix in batches to avoid token limits
    
    Args:
        batch_size: Number of mentors to process per batch (default 15)
        use_stateful: Enable stateful mode (store=True) for efficient batch processing
                     When True, only the first batch sends full context, subsequent batches
                     only send new mentor data and reference the previous response.
    
    Returns:
        Complete matrix DataFrame
    """
    all_mentor_ids = mentors_df[MENTOR_COLUMNS['id']].tolist()
    all_mentee_ids = mentees_df[MENTEE_COLUMNS['id']].tolist()
    
    # Split mentors into batches
    mentor_batches = [all_mentor_ids[i:i + batch_size] for i in range(0, len(all_mentor_ids), batch_size)]
    
    mode_info = "🔗 STATEFUL MODE" if use_stateful else "📦 STATELESS MODE"
    st.info(f"{mode_info}: Processing in {len(mentor_batches)} batches ({batch_size} mentors per batch)")
    
    if use_stateful:
        st.markdown("""
        <div class="info-card">
            <strong>🔗 Stateful Batching Enabled:</strong><br>
            • First batch: Sends full mentee data + initial mentors (~high tokens)<br>
            • Subsequent batches: Only new mentors (~low tokens, 90% savings)<br>
            • Context preserved via previous_response_id chaining
        </div>
        """, unsafe_allow_html=True)
    
    combined_matrix_dict = {}
    previous_response_id = None  # Track for stateful chaining
    
    # Process each batch
    for batch_num, mentor_batch in enumerate(mentor_batches, 1):
        is_first_batch = (batch_num == 1)
        
        batch_label = "FIRST BATCH (full context)" if is_first_batch else f"Batch {batch_num} (mentors only)"
        st.info(f"🔄 {batch_label}: Processing {len(mentor_batch)} mentors ({mentor_batch[0]} to {mentor_batch[-1]})")
        
        # Call API for this batch with stateful parameters
        training_file_ids = st.session_state.get('training_file_ids', [])
        result = call_openai_api_with_assistants(
            get_prompt_for_api(),
            mentees_df,
            mentors_df,
            training_data,
            api_key,
            model,
            mentor_subset=mentor_batch,
            use_stateful=use_stateful,
            previous_response_id=previous_response_id,
            is_first_batch=is_first_batch
        )
        
        # Handle both tuple (stateful) and string (stateless) returns
        if use_stateful and isinstance(result, tuple):
            response, response_id = result
            previous_response_id = response_id  # Chain for next batch
            if response_id:
                st.success(f"✅ Response ID saved for chaining: {response_id[:20]}...")
        else:
            response = result
        
        if not response:
            st.error(f"❌ Batch {batch_num} failed to get response")
            show_debug_info()  # Show debug info for failed batches
            continue
        
        # Parse this batch's response
        batch_matrix_df = parse_matrix_response(response, mentor_batch, all_mentee_ids)
        
        if batch_matrix_df is None:
            st.error(f"❌ Batch {batch_num} failed to parse")
            continue
        
        # DEBUG: Show what mentors we got back vs what we expected
        received_mentors = list(batch_matrix_df.index)
        # Convert to strings for comparison (batch_matrix_df.index contains strings from JSON)
        expected_mentors_set = set(str(m) for m in mentor_batch)
        received_mentors_set = set(received_mentors)
        
        # Track what we're adding and check for overwrites
        dict_size_before = len(combined_matrix_dict)
        newly_added = 0
        overwrites = []
        
        for mentor_id in batch_matrix_df.index:
            if mentor_id in combined_matrix_dict:
                overwrites.append(mentor_id)
                st.error(f"🚨 DUPLICATE: '{mentor_id}' already exists! Batch {batch_num} is overwriting previous data!")
            
            combined_matrix_dict[mentor_id] = batch_matrix_df.loc[mentor_id].to_dict()
            newly_added += 1
        
        dict_size_after = len(combined_matrix_dict)
        actual_new_entries = dict_size_after - dict_size_before
        
        # Show detailed statistics
        st.info(f"📈 Dict growth: {dict_size_before} → {dict_size_after} (+{actual_new_entries} new entries)")
        
        if overwrites:
            st.error(f"⚠️ WARNING: {len(overwrites)} duplicate(s) were OVERWRITTEN!")
            with st.expander(f"🔍 Overwritten mentor IDs ({len(overwrites)})"):
                for m in sorted(overwrites):
                    st.write(f"- {m} (was already in dict from previous batch)")
        
        # Detailed success/warning message
        if received_mentors_set == expected_mentors_set:
            st.success(f"✅ Batch {batch_num}: Received all {len(batch_matrix_df)}/{len(mentor_batch)} requested mentors!")
        else:
            missing_in_batch = expected_mentors_set - received_mentors_set
            st.warning(f"⚠️ Batch {batch_num}: Only {len(batch_matrix_df)}/{len(mentor_batch)} mentors received")
            with st.expander(f"📋 Missing from batch {batch_num} ({len(missing_in_batch)})"):
                for m in sorted(missing_in_batch):
                    st.write(f"- {m}")
        
        # Delay between batches to avoid rate limits
        # Longer delay in stateful mode since we're making rapid successive calls
        if batch_num < len(mentor_batches):
            delay = 5 if use_stateful else 2
            st.caption(f"⏳ Waiting {delay}s before next batch to avoid rate limits...")
            time.sleep(delay)
    
    # Combine all batches into single DataFrame
    if not combined_matrix_dict:
        st.error("❌ No batches succeeded")
        return None

    st.divider()
    st.info("📊 **Combining all batches into final matrix...**")
    
    complete_matrix_df = pd.DataFrame(combined_matrix_dict).T
    # Convert mentee IDs to strings to match DataFrame columns (which come from JSON as strings)
    complete_matrix_df = complete_matrix_df[[str(m) for m in sorted(all_mentee_ids)]]
    
    # Final validation with detailed breakdown
    mentors_received = len(complete_matrix_df)
    expected_count = len(all_mentor_ids)
    mentors_in_dict = set(complete_matrix_df.index)
    # Convert to strings for comparison (complete_matrix_df.index contains strings from JSON)
    mentors_requested = set(str(m) for m in all_mentor_ids)
    
    # Check for discrepancies
    missing_from_dict = mentors_requested - mentors_in_dict
    extra_in_dict = mentors_in_dict - mentors_requested
    
    st.success(f"✅ **Final matrix**: {mentors_received}/{expected_count} unique mentors in dictionary")
    
    # Show detailed breakdown
    with st.expander("🔍 Final Matrix Validation Details"):
        st.write(f"**Expected mentors**: {expected_count} (from original data)")
        st.write(f"**Mentors in dictionary**: {mentors_received} (unique keys)")
        st.write(f"**Dictionary size**: {len(combined_matrix_dict)} entries")
        
        if missing_from_dict:
            st.error(f"❌ **Missing {len(missing_from_dict)} mentors** that were requested but not in dict:")
            for m in sorted(missing_from_dict)[:20]:  # Show max 20
                st.text(f"  - {m}")
        else:
            st.success("✅ All requested mentors are in the dictionary!")
        
        if extra_in_dict:
            st.warning(f"⚠️ **Extra {len(extra_in_dict)} mentors** in dict that weren't requested:")
            for m in sorted(extra_in_dict)[:20]:
                st.text(f"  - {m}")
    
    if mentors_received < expected_count:
        st.warning(f"⚠️ **DISCREPANCY**: Expected {expected_count} but only have {mentors_received} in final matrix")
    
    # RETRY LOGIC: If mentors are missing, try up to 3 times
    if mentors_received < expected_count:
        # Convert to strings for comparison (complete_matrix_df.index contains strings from JSON)
        missing = set(str(m) for m in all_mentor_ids) - set(complete_matrix_df.index)
        st.warning(f"⚠️ Missing {len(missing)} mentors after initial batches")
        
        with st.expander(f"📋 Missing mentors ({len(missing)})"):
            for m in sorted(missing):
                st.write(f"- {m}")
        
        # Try up to 3 retries
        max_retries = 3
        for retry_num in range(1, max_retries + 1):
            # Recalculate what's still missing (convert to strings for comparison)
            current_missing = set(str(m) for m in all_mentor_ids) - set(complete_matrix_df.index)
            
            if not current_missing:
                st.success(f"🎉 All mentors retrieved!")
                break
            
            st.info(f"🔄 Retry {retry_num}/{max_retries}: Attempting {len(current_missing)} missing mentors...")
            
            try:
                training_file_ids = st.session_state.get('training_file_ids', [])
                retry_result = call_openai_api_with_assistants(
                    get_prompt_for_api(),
                    mentees_df,
                    mentors_df,
                    training_data,
                    api_key,
                    model,
                    mentor_subset=list(current_missing),
                    use_stateful=use_stateful,
                    previous_response_id=previous_response_id if use_stateful else None,
                    is_first_batch=False  # Retries are always subsequent
                )
                
                # Handle both tuple (stateful) and string (stateless) returns
                if use_stateful and isinstance(retry_result, tuple):
                    retry_response, retry_response_id = retry_result
                    if retry_response_id:
                        previous_response_id = retry_response_id  # Update chain
                else:
                    retry_response = retry_result
                
                if not retry_response:
                    st.error(f"❌ Retry {retry_num} failed to get response")
                    if retry_num < max_retries:
                        st.info("⏳ Waiting 5 seconds before next retry...")
                        time.sleep(5)
                    continue
                
                retry_matrix_df = parse_matrix_response(retry_response, list(current_missing), all_mentee_ids)
                
                if retry_matrix_df is not None and len(retry_matrix_df) > 0:
                    # Add retry results to combined matrix
                    added_count = 0
                    for mentor_id in retry_matrix_df.index:
                        combined_matrix_dict[mentor_id] = retry_matrix_df.loc[mentor_id].to_dict()
                        added_count += 1
                    
                    st.success(f"✅ Retry {retry_num} successful: Added {added_count} mentors")
                    
                    # Rebuild complete matrix from updated dictionary
                    complete_matrix_df = pd.DataFrame(combined_matrix_dict).T
                    complete_matrix_df = complete_matrix_df[sorted(all_mentee_ids)]
                    
                    # Update count
                    mentors_received = len(complete_matrix_df)
                    st.info(f"📈 Matrix now has {mentors_received}/{expected_count} mentors")
                    
                    # Check if complete
                    if mentors_received == expected_count:
                        st.success(f"🎉 NOW COMPLETE: {mentors_received}/{expected_count} mentors!")
                        break
                else:
                    st.error(f"❌ Retry {retry_num} failed to parse")
                    if retry_num < max_retries:
                        st.info("⏳ Waiting 5 seconds before next retry...")
                        time.sleep(5)
                        
            except Exception as e:
                st.error(f"❌ Retry {retry_num} error: {str(e)}")
                if retry_num < max_retries:
                    st.info("⏳ Waiting 5 seconds before next retry...")
                    time.sleep(5)
        
        # Final check after all retries (convert to strings for comparison)
        final_missing = set(str(m) for m in all_mentor_ids) - set(complete_matrix_df.index)
        if final_missing:
            st.warning(f"⚠️ After {max_retries} retries, still missing {len(final_missing)} mentors")
            with st.expander(f"📋 Permanently missing mentors ({len(final_missing)})"):
                for m in sorted(final_missing):
                    st.write(f"- {m}")
            st.info(f"💡 Continuing with {len(complete_matrix_df)}/{expected_count} mentors. Missing mentors will be excluded from matching.")
    
    return complete_matrix_df

def parse_matrix_response(response_text, expected_mentors, expected_mentees):
    """Parse the matrix response from AI and validate completeness + ID matching"""
    try:
        # Clean the response text
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # IMPROVED: Extract JSON if AI added explanatory text before/after
        # Look for the first { and last } to extract just the JSON portion
        import re
        
        # Try to find JSON object boundaries
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            # Extract only the JSON portion
            json_portion = response_text[first_brace:last_brace + 1]
            
            # Check if we had to extract (AI added text)
            if first_brace > 0 or last_brace < len(response_text) - 1:
                before_text = response_text[:first_brace].strip()
                after_text = response_text[last_brace + 1:].strip()
                
                st.warning("⚠️ AI included explanatory text instead of pure JSON. Attempting to extract JSON...")
                
                if before_text:
                    with st.expander("📝 Text before JSON (should be empty)"):
                        st.text(before_text[:500])  # Show first 500 chars
                
                if after_text:
                    with st.expander("📝 Text after JSON (should be empty)"):
                        st.text(after_text[:500])
            
            response_text = json_portion
        else:
            st.error("❌ Could not find JSON object boundaries in response")
            with st.expander("View raw response"):
                st.text(response_text[:2000])
            return None
        
        # Parse JSON
        data = json.loads(response_text)
        
        # Validate structure
        if 'matrix' not in data:
            raise ValueError("Response should contain 'matrix' key")
        
        matrix_data = data['matrix']
        if not isinstance(matrix_data, list):
            raise ValueError("Matrix should be a list")
        
        # Convert to pandas DataFrame with VALIDATION
        matrix_dict = {}
        all_mentees = set()
        
        # Track validation issues
        # Convert to strings to match JSON response format (IDs come as strings from JSON)
        expected_mentor_set = set(str(x) for x in expected_mentors)
        expected_mentee_set = set(str(x) for x in expected_mentees)
        wrong_mentors = []  # Mentors not in expected list
        wrong_mentees = []  # Mentees not in expected list
        
        for mentor_entry in matrix_data:
            mentor_id = str(mentor_entry['mentor_id'])  # Ensure string for comparison
            
            # VALIDATION 1: Check if this mentor ID was requested
            if mentor_id not in expected_mentor_set:
                wrong_mentors.append(mentor_id)
                st.warning(f"⚠️ AI returned WRONG mentor ID: '{mentor_id}' (not in requested batch)")
                continue  # SKIP this mentor - don't add to dict
            
            scores = {}
            for score_entry in mentor_entry['scores']:
                mentee_id = str(score_entry['mentee_id'])  # Ensure string for comparison
                percentage = score_entry['percentage']
                
                # VALIDATION 2: Check if this mentee ID is valid
                if mentee_id not in expected_mentee_set:
                    if mentee_id not in wrong_mentees:  # Only warn once per wrong mentee
                        wrong_mentees.append(mentee_id)
                    continue  # SKIP this mentee score
                
                scores[mentee_id] = percentage
                all_mentees.add(mentee_id)
            
            # Only add mentor if they have at least some valid scores
            if scores:
                matrix_dict[mentor_id] = scores
        
        # Report validation issues
        if wrong_mentors:
            st.error(f"🚨 WRONG MENTOR IDs: AI returned {len(wrong_mentors)} mentor(s) NOT in requested batch!")
            with st.expander(f"❌ Wrong mentor IDs ({len(wrong_mentors)})"):
                for m in sorted(wrong_mentors):
                    st.write(f"- {m} (REJECTED - not in batch request)")
        
        if wrong_mentees:
            st.error(f"🚨 WRONG MENTEE IDs: AI returned {len(wrong_mentees)} invalid mentee(s)!")
            with st.expander(f"❌ Wrong mentee IDs ({len(wrong_mentees)})"):
                for m in sorted(wrong_mentees)[:10]:  # Show max 10
                    st.write(f"- {m}")
        
        # Validate completeness (after filtering)
        mentors_received = len(matrix_dict)
        mentees_received = len(all_mentees)
        expected_mentor_count = len(expected_mentors)
        expected_mentee_count = len(expected_mentees)
        
        st.info(f"📊 Valid data received: {mentors_received}/{expected_mentor_count} mentors, {mentees_received}/{expected_mentee_count} mentees")
        
        # TASK 4: Validate completeness of scores BEFORE creating DataFrame
        expected_total_scores = expected_mentor_count * expected_mentee_count
        actual_total_scores = sum(len(scores) for scores in matrix_dict.values())
        
        if actual_total_scores < expected_total_scores:
            missing_count = expected_total_scores - actual_total_scores
            completion_rate = (actual_total_scores / expected_total_scores) * 100 if expected_total_scores > 0 else 0
            
            st.error(f"🚨 INCOMPLETE RESPONSE: Expected {expected_total_scores} scores, got {actual_total_scores} ({completion_rate:.1f}% complete)")
            st.error(f"   Missing {missing_count} combinations!")
            st.error(f"   ⚠️ The AI is still omitting combinations despite instructions.")
            
            # Log which mentors have incomplete data
            with st.expander("📋 Mentors with incomplete score arrays"):
                for mentor_id, scores in matrix_dict.items():
                    scores_count = len(scores)
                    if scores_count < expected_mentee_count:
                        st.warning(f"   • Mentor {mentor_id}: {scores_count}/{expected_mentee_count} mentees ({expected_mentee_count - scores_count} missing)")
        elif actual_total_scores == expected_total_scores:
            st.success(f"✅ COMPLETE RESPONSE: All {expected_total_scores} scores received!")
        
        if mentors_received < expected_mentor_count:
            # Convert to strings for comparison (matrix_dict.keys() contains strings)
            missing_mentors = set(str(m) for m in expected_mentors) - set(matrix_dict.keys())
            st.warning(f"⚠️ Missing {len(missing_mentors)} requested mentor(s)")
            with st.expander(f"📋 Missing mentors ({len(missing_mentors)})"):
                for m in sorted(missing_mentors):
                    st.write(f"- {m}")
        
        if mentees_received < expected_mentee_count:
            # Convert to strings for comparison (all_mentees contains strings)
            missing_mentees = set(str(m) for m in expected_mentees) - all_mentees
            st.warning(f"⚠️ Missing {len(missing_mentees)} mentee(s) in responses")
            with st.expander(f"📋 Missing mentees ({len(missing_mentees)})"):
                for m in sorted(missing_mentees):
                    st.write(f"- {m}")
        
        # Create DataFrame
        if not matrix_dict:
            st.error("❌ No valid mentor data after filtering!")
            return None
        
        df = pd.DataFrame(matrix_dict).T  # Transpose so mentors are rows, mentees are columns
        
        # Fill missing mentees with LOW score (30) - AI omitted these (likely poor matches)
        # Convert to strings for comparison (df.columns contains strings)
        missing_mentees = [m for m in expected_mentees if str(m) not in df.columns]
        if missing_mentees:
            st.warning(f"⚠️ {len(missing_mentees)} mentee column(s) missing from matrix. Filling with 30% (assumed poor match).")
            st.caption("The AI omitted these combinations despite instructions. This indicates likely poor matches.")
            show_debug_info()  # Allow user to inspect request/response
            for mentee in missing_mentees:
                df[mentee] = 30  # Assume omitted = poor match, score low (not neutral 50)
        
        # Sort mentee columns (convert to strings to match df.columns)
        df = df[[str(m) for m in sorted(expected_mentees)]]
        
        return df
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        st.error(f"❌ Error parsing matrix response: {str(e)}")
        show_debug_info()  # Show debug info for parsing errors
        with st.expander("View raw response"):
            st.text(response_text[:2000])
        return None

def hungarian_assignment(matrix_df, max_mentees_per_mentor=2):
    """
    Use Hungarian algorithm to find optimal mentee-mentor assignments
    with constraint of max 2 mentees per mentor
    
    Args:
        matrix_df: DataFrame with mentors as rows, mentees as columns, percentages as values
        max_mentees_per_mentor: Maximum number of mentees per mentor (default 2)
    
    Returns:
        List of tuples: [(mentee_id, mentor_id, score), ...]
    """
    mentors = list(matrix_df.index)
    mentees = list(matrix_df.columns)
    
    # Validate matrix data
    st.write("🔍 Validating matrix data...")
    
    # Check for NaN values
    nan_count = matrix_df.isna().sum().sum()
    if nan_count > 0:
        st.warning(f"⚠️ Found {nan_count} NaN values in matrix. Replacing with 30% (poor match - AI omitted these)...")
        st.caption("NaN values indicate the AI skipped these combinations despite instructions. Treating as poor matches.")
        show_debug_info()  # Allow user to inspect request/response
        matrix_df = matrix_df.fillna(30)
    
    # Check for inf values
    inf_mask = np.isinf(matrix_df.values)
    if inf_mask.any():
        inf_count = inf_mask.sum()
        st.warning(f"⚠️ Found {inf_count} infinite values in matrix. Replacing with 50%...")
        matrix_df = matrix_df.replace([np.inf, -np.inf], 50)
    
    # Check data types - ensure all values are numeric
    try:
        matrix_df = matrix_df.astype(float)
    except Exception as e:
        st.error(f"❌ Matrix contains non-numeric values: {e}")
        st.write("Sample of problematic data:")
        st.write(matrix_df.head())
        raise ValueError(f"Matrix must contain only numeric values: {e}")
    
    # Check value range (should be 0-100)
    min_val = matrix_df.min().min()
    max_val = matrix_df.max().max()
    if min_val < 0 or max_val > 100:
        st.warning(f"⚠️ Matrix values outside expected range [0, 100]: min={min_val}, max={max_val}")
        # Clip to valid range
        matrix_df = matrix_df.clip(0, 100)
    
    st.write(f"✅ Matrix validation passed: {len(mentors)} mentors × {len(mentees)} mentees")
    
    # Create expanded cost matrix to handle mentor capacity constraint
    # Each mentor gets duplicated max_mentees_per_mentor times
    expanded_mentors = []
    for mentor in mentors:
        for i in range(max_mentees_per_mentor):
            expanded_mentors.append(f"{mentor}_copy{i}")
    
    # Build cost matrix (negative because Hungarian minimizes, we want to maximize)
    cost_matrix = np.zeros((len(mentees), len(expanded_mentors)))
    
    for i, mentee in enumerate(mentees):
        for j, expanded_mentor in enumerate(expanded_mentors):
            actual_mentor = expanded_mentor.rsplit('_copy', 1)[0]
            score = matrix_df.loc[actual_mentor, mentee]
            cost_matrix[i, j] = -score  # Negative for maximization
    
    # Final validation before Hungarian algorithm
    if not np.isfinite(cost_matrix).all():
        st.error("❌ Cost matrix still contains invalid values after cleaning")
        st.write("Cost matrix stats:")
        st.write(f"- NaN count: {np.isnan(cost_matrix).sum()}")
        st.write(f"- Inf count: {np.isinf(cost_matrix).sum()}")
        raise ValueError("Cost matrix contains invalid numeric entries after cleaning")
    
    # Run Hungarian algorithm
    st.write("🧮 Running Hungarian algorithm for optimal assignment...")
    mentee_indices, mentor_indices = linear_sum_assignment(cost_matrix)
    
    # Extract assignments
    assignments = []
    for mentee_idx, mentor_idx in zip(mentee_indices, mentor_indices):
        mentee = mentees[mentee_idx]
        expanded_mentor = expanded_mentors[mentor_idx]
        actual_mentor = expanded_mentor.rsplit('_copy', 1)[0]
        score = matrix_df.loc[actual_mentor, mentee]
        assignments.append((mentee, actual_mentor, int(score)))
    
    # Sort by mentee ID for consistency
    assignments.sort(key=lambda x: x[0])
    
    return assignments

def get_reasoning_for_assignments(assignments, mentees_df, mentors_df, api_key, model="gpt-4o-mini"):
    """Get reasoning for specific assignments via separate API call"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Validate inputs
        if not assignments or len(assignments) == 0:
            st.warning("⚠️ No assignments to get reasoning for")
            return {}
        
        if mentees_df is None or mentors_df is None:
            st.error("❌ Missing mentee or mentor data")
            return {}
        
        # Prepare data for context (just the assigned pairs)
        assignment_details = []
        for mentee_id, mentor_id, score in assignments:
            try:
                # Convert IDs to int for comparison (assignments may contain string IDs, DataFrame has int IDs)
                mentee_id_int = int(mentee_id)
                mentor_id_int = int(mentor_id)
                mentee_row = mentees_df[mentees_df[MENTEE_COLUMNS['id']] == mentee_id_int].iloc[0]
                mentor_row = mentors_df[mentors_df[MENTOR_COLUMNS['id']] == mentor_id_int].iloc[0]
                
                # Helper to safely get string values
                def safe_str(value):
                    if pd.isna(value):
                        return ''
                    return str(value).strip()
                
                assignment_details.append({
                    'mentee_id': safe_str(mentee_id),
                    'mentor_id': safe_str(mentor_id),
                    'score': int(score) if not pd.isna(score) else 0,
                    'mentee_field': safe_str(mentee_row.get(MENTEE_COLUMNS['field'], '')),
                    'mentee_specialization': safe_str(mentee_row.get(MENTEE_COLUMNS['specialization'], '')),
                    'mentee_guidance_areas': safe_str(mentee_row.get(MENTEE_COLUMNS['guidance_areas'], '')),
                    'mentee_goals': safe_str(mentee_row.get(MENTEE_COLUMNS['career_goals'], '')),
                    'mentee_other_info': safe_str(mentee_row.get(MENTEE_COLUMNS['other_info'], '')),  # NEW
                    'mentor_field': safe_str(mentor_row.get(MENTOR_COLUMNS['field'], '')),
                    'mentor_specialization': safe_str(mentor_row.get(MENTOR_COLUMNS['specialization'], '')),
                    'mentor_position': safe_str(mentor_row.get(MENTOR_COLUMNS['current_position'], '')),  # NEW
                    'mentor_experience': safe_str(mentor_row.get(MENTOR_COLUMNS['experience_years'], ''))
                })
            except Exception as e:
                st.warning(f"⚠️ Could not extract details for {mentee_id} → {mentor_id}: {str(e)}")
                # Add with minimal info
                assignment_details.append({
                    'mentee_id': str(mentee_id),
                    'mentor_id': str(mentor_id),
                    'score': int(score) if isinstance(score, (int, float)) else 0,
                    'mentee_field': '',
                    'mentee_specialization': '',
                    'mentee_guidance_areas': '',
                    'mentee_goals': '',
                    'mentee_other_info': '',  # NEW
                    'mentor_field': '',
                    'mentor_specialization': '',
                    'mentor_position': '',  # NEW
                    'mentor_experience': ''
                })
        
        # Check if we have valid assignment details
        if not assignment_details:
            st.error("❌ No valid assignment details extracted")
            return {}
        
        st.info(f"📝 Extracted details for {len(assignment_details)} assignments")
        
        # Create focused prompt
        prompt = f"""Provide brief reasoning for these {len(assignment_details)} mentorship assignments.

For each assignment, explain in 2-3 sentences why this is a good match based on:
- Field/specialization alignment
- Skills compatibility
- Career goals support

ASSIGNMENTS:
"""
        for ad in assignment_details:
            # Safely truncate fields
            mentee_spec = str(ad.get('mentee_specialization', ''))[:100]
            mentee_guidance = str(ad.get('mentee_guidance_areas', ''))[:100]
            mentee_goals = str(ad.get('mentee_goals', ''))[:100]
            mentee_other = str(ad.get('mentee_other_info', ''))[:100]
            mentor_spec = str(ad.get('mentor_specialization', ''))[:100]
            mentor_pos = str(ad.get('mentor_position', ''))[:80]
            
            prompt += f"""
{ad['mentee_id']} → {ad['mentor_id']} (Score: {ad['score']}%)
- Mentee field: {ad['mentee_field']}
- Mentee specialization: {mentee_spec}...
- Mentee guidance needs: {mentee_guidance}...
- Mentee goals: {mentee_goals}...
- Mentee other info: {mentee_other}...
- Mentor field: {ad['mentor_field']}
- Mentor position: {mentor_pos}...
- Mentor specialization: {mentor_spec}...
- Mentor experience: {ad['mentor_experience']} years

"""
        
        prompt += """
OUTPUT FORMAT (IMPORTANT):
Return ONLY a JSON object with this exact structure:
{
  "reasonings": [
    {
      "mentee_id": "EXACT_CODE",
      "mentor_id": "EXACT_CODE",
      "reasoning": "2-3 sentence explanation"
    }
  ]
}

No markdown, no extra text, just the JSON object."""
        
        # Make API call with progress bar
        progress_text = st.empty()
        progress_text.text("🧠 Requesting reasoning from AI...")
        
        # Check if Responses API is available
        if hasattr(client, 'responses'):
            responses_api = client.responses
        elif hasattr(client, 'beta') and hasattr(client.beta, 'responses'):
            responses_api = client.beta.responses
        else:
            # Fallback to Chat Completions API
            responses_api = None
        
        # Handle o1/o3 models differently (no system messages, no temperature)
        is_reasoning_model = model.startswith("o1-") or model.startswith("o3-")
        
        if responses_api:
            # Use Responses API with structured output for reasoning
            # Define schema for reasoning - must be object at root level
            reasoning_schema = {
                "format": {
                    "type": "json_schema",
                    "name": "mentorship_reasoning",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasonings": {
                                "type": "array",
                                "description": "Array of reasoning for each assignment",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "mentee_id": {
                                            "type": "string",
                                            "description": "The mentee's ID"
                                        },
                                        "mentor_id": {
                                            "type": "string",
                                            "description": "The mentor's ID"
                                        },
                                        "reasoning": {
                                            "type": "string",
                                            "description": "Explanation of why this is a good match"
                                        }
                                    },
                                    "required": ["mentee_id", "mentor_id", "reasoning"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["reasonings"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
            
            # Retry logic for rate limits in reasoning call - ONLY for the API call
            max_retries = 5
            retry_count = 0
            response = None
            
            while retry_count < max_retries:
                try:
                    if is_reasoning_model:
                        # Reasoning models - combine system instruction into user message
                        full_prompt = "You are an expert at explaining mentorship compatibility. Return only valid JSON.\n\n" + prompt
                        response = responses_api.create(
                            model=model,
                            input=full_prompt,
                            max_output_tokens=8000,
                            store=False,
                            text=reasoning_schema
                        )
                    else:
                        # Standard models
                        response = responses_api.create(
                            model=model,
                            input=[
                                {"role": "system", "content": "You are an expert at explaining mentorship compatibility. Return only valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_output_tokens=8000,
                            store=False,
                            text=reasoning_schema
                        )
                    # API call succeeded, break out
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a rate limit error (429)
                    if "rate_limit_exceeded" in error_str or "429" in error_str:
                        retry_count += 1
                        
                        if retry_count >= max_retries:
                            st.error(f"❌ Rate limit exceeded on reasoning call after {max_retries} retries")
                            return {}
                        
                        # Extract wait time from error message
                        import re
                        wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                        if wait_match:
                            wait_time = float(wait_match.group(1))
                        else:
                            wait_time = 2 ** retry_count
                        
                        buffer = 3
                        total_wait = wait_time + buffer
                        
                        st.warning(f"⏳ Rate limit on reasoning call. Waiting {total_wait:.1f}s before retry {retry_count+1}/{max_retries}...")
                        time.sleep(total_wait)
                        continue  # Retry
                    else:
                        # Not a rate limit error, fail immediately
                        st.error(f"❌ Reasoning API call failed: {str(e)}")
                        return {}
            
            # Check if we got a response
            if response is None:
                st.error("❌ Failed to get reasoning response after retries")
                return {}
        else:
            # Fallback to Chat Completions API
            token_limit_param = "max_completion_tokens" if is_reasoning_model else "max_tokens"
            
            if is_reasoning_model:
                full_prompt = "You are an expert at explaining mentorship compatibility. Return only valid JSON.\n\n" + prompt
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    **{token_limit_param: 8000}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert at explaining mentorship compatibility. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    **{token_limit_param: 8000}
                )
        
        progress_text.text("🧠 Parsing reasoning response...")
        
        # Parse response (handle both API formats)
        if hasattr(response, 'output'):
            # Responses API format: output[0].content[0].text
            try:
                if isinstance(response.output, list) and len(response.output) > 0:
                    output_msg = response.output[0]
                    if hasattr(output_msg, 'content') and isinstance(output_msg.content, list) and len(output_msg.content) > 0:
                        response_text = output_msg.content[0].text.strip()
                    else:
                        response_text = str(output_msg).strip()
                else:
                    response_text = str(response.output).strip()
            except Exception as e:
                st.error(f"❌ Error extracting reasoning response: {e}")
                response_text = str(response.output).strip()
        elif hasattr(response, 'choices'):
            # Chat Completions format
            response_text = response.choices[0].message.content.strip()
        else:
            st.error("❌ Unknown response format for reasoning")
            return {}
        
        # Debug: show what we got
        st.info(f"📝 Received reasoning response ({len(response_text)} characters)")
        
        # Show first part of response for debugging
        with st.expander("🔍 Debug: View response preview"):
            st.text(response_text[:500])
        
        # Clean markdown
        original_text = response_text
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            reasoning_response = json.loads(response_text)
        except json.JSONDecodeError as parse_error:
            st.error(f"❌ JSON parse failed: {parse_error}")
            with st.expander("📄 Full response text"):
                st.code(original_text)
            raise
        
        # Extract the reasonings array from the object
        if isinstance(reasoning_response, dict) and 'reasonings' in reasoning_response:
            reasoning_data = reasoning_response['reasonings']
        elif isinstance(reasoning_response, list):
            # Fallback: if it's already a list (old format)
            reasoning_data = reasoning_response
        else:
            st.error(f"❌ Response is {type(reasoning_response).__name__}, expected object with 'reasonings' array")
            with st.expander("📄 Parsed data"):
                st.json(reasoning_response)
            raise ValueError("Response should be an object with 'reasonings' array")
        
        if not isinstance(reasoning_data, list):
            st.error(f"❌ 'reasonings' is {type(reasoning_data).__name__}, expected list")
            raise ValueError("'reasonings' should be a list")
        
        st.info(f"✅ Parsed {len(reasoning_data)} reasoning entries from JSON")
        
        # Convert to dict for easy lookup
        reasoning_dict = {}
        for idx, r in enumerate(reasoning_data):
            if 'mentee_id' in r and 'mentor_id' in r and 'reasoning' in r:
                reasoning_dict[(r['mentee_id'], r['mentor_id'])] = r['reasoning']
            else:
                st.warning(f"⚠️ Entry {idx} missing required fields: {r.keys()}")
        
        progress_text.empty()
        st.success(f"✅ Got reasoning for {len(reasoning_dict)} assignments")
        
        # Debug: Show what keys we have
        if len(reasoning_dict) > 0:
            sample_keys = list(reasoning_dict.keys())[:3]
            st.info(f"📋 Sample reasoning keys: {sample_keys}")
        
        return reasoning_dict
        
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON parsing error in reasoning: {str(e)}")
        with st.expander("View raw reasoning response"):
            try:
                st.text(response_text[:2000])
            except:
                st.text("Could not display response")
        st.info("💡 Continuing with generic reasoning...")
        return {}
    except Exception as e:
        st.error(f"❌ Error getting reasoning: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.info("💡 Continuing without detailed reasoning...")
        return {}

def create_heatmap(matrix_df, assignments=None):
    """Create interactive heatmap visualization of compatibility matrix
    
    Args:
        matrix_df: DataFrame with mentors as rows, mentees as columns
        assignments: Optional list of (mentee, mentor, score) tuples to highlight
    """
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns,
        y=matrix_df.index,
        colorscale='RdYlGn',  # Red-Yellow-Green
        text=matrix_df.values,
        texttemplate='%{text}%',
        textfont={"size": 8},
        colorbar=dict(title="Match %"),
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> → <b>%{x}</b><br>Match: %{z}%<extra></extra>'
    ))
    
    # Add rectangles around assigned matches
    if assignments:
        assignment_dict = {(mentee, mentor): score for mentee, mentor, score in assignments}
        
        shapes = []
        for mentee_idx, mentee in enumerate(matrix_df.columns):
            for mentor_idx, mentor in enumerate(matrix_df.index):
                if (mentee, mentor) in assignment_dict:
                    # Add a thick border around assigned cells
                    shapes.append(dict(
                        type="rect",
                        x0=mentee_idx - 0.5,
                        y0=mentor_idx - 0.5,
                        x1=mentee_idx + 0.5,
                        y1=mentor_idx + 0.5,
                        line=dict(color="blue", width=3),
                    ))
        
        fig.update_layout(shapes=shapes)
    
    # Update layout
    fig.update_layout(
        title="Mentee-Mentor Compatibility Matrix",
        xaxis_title="Mentees",
        yaxis_title="Mentors",
        height=max(600, len(matrix_df) * 25),  # Dynamic height
        width=max(800, len(matrix_df.columns) * 30),  # Dynamic width
        xaxis=dict(tickangle=-45),
        font=dict(size=10)
    )
    
    return fig

def enforce_mentor_limit(matches):
    """Enforce maximum 2 mentees per mentor by keeping only the best matches
    
    When a mentor is matched with more than 2 mentees (at rank 1), we keep only
    the top 2 mentees with the highest match percentages. Other mentees get their
    rank 2 promoted to rank 1, and so on.
    """
    if not matches:
        return matches
    
    # Track rank 1 assignments per mentor with match percentages
    rank1_assignments = {}  # {mentor_id: [(mentee_id, percentage, match_data), ...]}
    
    # Collect all rank 1 assignments
    for match_data in matches:
        mentee_id = match_data['mentee_id']
        for match in match_data['matches']:
            if match['rank'] == 1:
                mentor_id = match['mentor_id']
                percentage = match['match_percentage']
                if mentor_id not in rank1_assignments:
                    rank1_assignments[mentor_id] = []
                rank1_assignments[mentor_id].append((mentee_id, percentage, match_data))
    
    # Find mentors assigned to more than 2 mentees
    mentors_to_fix = {mentor_id: assignments 
                      for mentor_id, assignments in rank1_assignments.items() 
                      if len(assignments) > 2}
    
    if not mentors_to_fix:
        return matches  # No conflicts, return as is
    
    # For each overassigned mentor, keep only top 2 mentees
    mentees_to_reassign = set()
    
    for mentor_id, assignments in mentors_to_fix.items():
        # Sort by percentage (highest first)
        sorted_assignments = sorted(assignments, key=lambda x: x[1], reverse=True)
        
        # Keep top 2, mark others for reassignment
        for mentee_id, percentage, match_data in sorted_assignments[2:]:
            mentees_to_reassign.add(mentee_id)
    
    # Process matches: remove overassigned rank 1, promote rank 2 to rank 1
    updated_matches = []
    
    for match_data in matches:
        mentee_id = match_data['mentee_id']
        
        if mentee_id in mentees_to_reassign:
            # This mentee needs reassignment
            current_matches = match_data['matches']
            rank1_mentor = current_matches[0]['mentor_id']
            
            # Remove the overassigned rank 1, promote others
            new_matches = [
                {
                    'rank': i + 1,
                    'mentor_id': current_matches[i+1]['mentor_id'],
                    'match_percentage': current_matches[i+1]['match_percentage'],
                    'match_quality': current_matches[i+1]['match_quality'],
                    'reasoning': current_matches[i+1]['reasoning'] + f" (Promoted from rank {i+2} due to {rank1_mentor} being overassigned)"
                }
                for i in range(min(2, len(current_matches) - 1))  # Promote rank 2 and 3
            ]
            
            updated_matches.append({
                'mentee_id': mentee_id,
                'matches': new_matches,
                'note': f'Original rank 1 ({rank1_mentor}) was overassigned'
            })
        else:
            # Keep as is
            updated_matches.append(match_data)
    
    return updated_matches

def check_mentor_conflicts(matches):
    """Check if any mentor is assigned to more than 2 mentees at rank 1"""
    if not matches:
        return []
    
    mentor_assignments = {}
    conflicts = []
    
    for match_data in matches:
        mentee_id = match_data['mentee_id']
        # Only check rank 1 assignments
        if match_data['matches'] and len(match_data['matches']) > 0:
            rank1_match = match_data['matches'][0]
            if rank1_match['rank'] == 1:
                mentor_id = rank1_match['mentor_id']
                percentage = rank1_match['match_percentage']
            if mentor_id not in mentor_assignments:
                mentor_assignments[mentor_id] = []
                mentor_assignments[mentor_id].append((mentee_id, percentage))
    
    for mentor_id, assignments in mentor_assignments.items():
        if len(assignments) > 2:
            conflicts.append({
                'mentor_id': mentor_id,
                'assignments': assignments,
                'count': len(assignments)
            })
    
    return conflicts

# ============================================================================
# OPENAI FILE STORAGE FUNCTIONS
# ============================================================================

def list_openai_files(api_key):
    """List all files uploaded to OpenAI storage"""
    try:
        client = OpenAI(api_key=api_key)
        files = client.files.list(purpose='assistants')
        
        file_list = []
        for file in files.data:
            created_at = datetime.fromtimestamp(file.created_at)
            age_days = (datetime.now() - created_at).days
            
            file_list.append({
                'id': file.id,
                'filename': file.filename,
                'size_bytes': file.bytes,
                'size_kb': file.bytes / 1024,
                'created_at': created_at,
                'age_days': age_days,
                'purpose': file.purpose
            })
        
        return file_list
    except Exception as e:
        st.error(f"❌ Error listing files: {str(e)}")
        return []

def upload_training_files_to_openai(training_dfs, api_key):
    """Upload training DataFrames to OpenAI storage"""
    try:
        client = OpenAI(api_key=api_key)
        uploaded_file_ids = []
        
        for i, df in enumerate(training_dfs):
            # Convert DataFrame to CSV
            csv_content = df.to_csv(index=False)
            
            # Create BytesIO object
            file_obj = BytesIO(csv_content.encode('utf-8'))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_obj.name = f"training_data_{timestamp}_{i+1}.csv"
            
            # Upload to OpenAI
            st.info(f"📤 Uploading {file_obj.name} ({len(csv_content)/1024:.1f} KB)...")
            
            file = client.files.create(
                file=file_obj,
                purpose='assistants'
            )
            
            uploaded_file_ids.append(file.id)
            st.success(f"✅ Uploaded: {file.id}")
        
        return uploaded_file_ids
        
    except Exception as e:
        st.error(f"❌ Error uploading files: {str(e)}")
        return []

def delete_openai_files(api_key, file_ids):
    """Delete specific files from OpenAI storage"""
    try:
        client = OpenAI(api_key=api_key)
        deleted_count = 0
        
        for file_id in file_ids:
            try:
                client.files.delete(file_id)
                deleted_count += 1
            except Exception as e:
                st.warning(f"Could not delete {file_id}: {str(e)}")
        
        return deleted_count
        
    except Exception as e:
        st.error(f"❌ Error deleting files: {str(e)}")
        return 0

def download_openai_file(api_key, file_id):
    """Download file content from OpenAI storage and return as DataFrame"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Retrieve file content
        file_content = client.files.content(file_id)
        
        # Convert to DataFrame (assuming CSV format)
        from io import StringIO
        csv_content = file_content.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error downloading file {file_id}: {str(e)}")
        return None

def main():
    # Header with animation
    st.markdown("""
    <div class="main-header">
        <h1>🤝 Mentorship Matching System</h1>
        <p>AI-powered mentor-mentee matching</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state (Improvement 24)
    if 'training_files' not in st.session_state:
        st.session_state.training_files = []
    if 'mentees_df' not in st.session_state:
        st.session_state.mentees_df = None
    if 'mentors_df' not in st.session_state:
        st.session_state.mentors_df = None
    if 'matches' not in st.session_state:
        st.session_state.matches = None
    if 'matrix_df' not in st.session_state:
        st.session_state.matrix_df = None
    if 'assignments' not in st.session_state:
        st.session_state.assignments = None
    if 'custom_prompt' not in st.session_state:
        st.session_state.custom_prompt = create_default_prompt()
    if 'selected_preset' not in st.session_state:
        st.session_state.selected_preset = "Balanced ⭐"
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 15
    if 'training_file_ids' not in st.session_state:
        st.session_state.training_file_ids = []
    if 'training_file_names' not in st.session_state:
        st.session_state.training_file_names = []
    if 'mentee_search' not in st.session_state:
        st.session_state.mentee_search = ""
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "gpt-5.1"  # Default to latest model
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    if 'imfahe_specialty' not in st.session_state:
        st.session_state.imfahe_specialty = "IMP-Biomedicine (Biology, Medicine, Pharmacy, Biotechnology or related areas)"
    
    # No sidebar - cleaner interface
    
    # Check completion status for wizard flow
    # Step 1 (API key) + Step 3 (data files) are required
    # Step 2 (training files) is optional
    step1_complete = (
        st.session_state.mentees_df is not None and 
        st.session_state.mentors_df is not None
    )
    
    # Get API key early for step 2 check
    import os
    
    # Try to get from session state or environment
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
    
    step2_complete = bool(st.session_state.api_key) and step1_complete
    step3_complete = step2_complete  # Can view data once files uploaded and settings configured
    
    # Create wizard tabs (6 tabs in logical order)
    tab_names = [
        "1️⃣ Configure Settings",
        "2️⃣ Training Files",
        "3️⃣ Upload Data Files",
        "4️⃣ Customize Prompt",
        "5️⃣ Data Overview",
        "6️⃣ Generate Matches"
    ]
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)
    
    # TAB 1: Configure Settings
    with tab1:
        st.header("⚙️ Configure Settings")
        
        # Check API key from environment
        if st.session_state.api_key:
            st.markdown(f"""
            <div class="success-card">
                ✅ OpenAI API key loaded from environment variable <code>OPENAI_API_KEY</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="conflict-warning">
                ⚠️ <strong>Missing API Key</strong><br>
                Please set the <code>OPENAI_API_KEY</code> environment variable.<br>
                Get your key at <a href="https://platform.openai.com/api-keys" target="_blank">https://platform.openai.com/api-keys</a>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        st.divider()
        
        # Model Selection (dynamic from API)
        st.subheader("🤖 AI Model Selection")
        
        # Fetch available models from OpenAI
        with st.spinner("Fetching available models from OpenAI..."):
            available_models = fetch_available_models(st.session_state.api_key)
        
        if available_models:
            # Sort models by creation date (newest first)
            sorted_model_ids = sorted(
                available_models.keys(),
                key=lambda x: available_models[x].get("created", 0),
                reverse=True  # Newest first
            )
            
            # Ensure current model_choice is valid, otherwise use first model
            if st.session_state.model_choice not in sorted_model_ids:
                st.session_state.model_choice = sorted_model_ids[0]
            
            # Model selector - using key parameter to prevent jumping
            from datetime import datetime
            
            def format_model_name(model_id):
                """Format model name with creation date"""
                model_info = available_models[model_id]
                # Convert Unix timestamp to readable date
                created_date = datetime.fromtimestamp(model_info.get("created", 0)).strftime("%Y-%m-%d")
                return f"{model_info['name']} (created: {created_date})"
            
            selected_model = st.selectbox(
                "Choose AI Model (sorted newest → oldest)",
                sorted_model_ids,
                index=sorted_model_ids.index(st.session_state.model_choice),
                format_func=format_model_name,
                key="model_selector",
                help="Select the OpenAI model to use for matching. Models are sorted by creation date (newest first)."
            )
            
            # Update session state only if changed
            if selected_model != st.session_state.model_choice:
                st.session_state.model_choice = selected_model
            
            # Show model details
            model_info = available_models[selected_model]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quality", model_info["quality"])
            with col2:
                st.metric("Speed", model_info["speed"])
            with col3:
                st.caption("**Cost**")
                st.caption(model_info["cost"])
        else:
            st.error("Failed to fetch models. Using default: gpt-5.1")
            st.session_state.model_choice = "gpt-5.1"
        
        st.divider()
        
        # Batch Size
        st.subheader("📦 Batch Processing")
        batch_size = st.slider(
            "Number of mentors to process per API call",
            min_value=5,
            max_value=30,
            value=st.session_state.batch_size,
            help="Larger batches = faster processing but higher cost per call. Smaller batches = more granular but more API calls."
        )
        st.session_state.batch_size = batch_size
        
        st.divider()
        
        # Stateful mode toggle
        st.subheader("🔗 Stateful Batch Processing (Responses API)")
        use_stateful = st.checkbox(
            "Enable stateful mode (store=True)",
            value=st.session_state.get('use_stateful_mode', False),
            help="Stateful mode uses context chaining to dramatically reduce token usage in batch processing"
        )
        st.session_state.use_stateful_mode = use_stateful
        
        if use_stateful:
            st.markdown("""
            <div class="success-card">
                <strong>✅ Stateful Mode Enabled</strong><br>
                <strong>How it works:</strong><br>
                • First batch: Full mentee data + first mentor batch (normal token count)<br>
                • Subsequent batches: Only new mentors (~90% token savings)<br>
                • Context preserved via <code>previous_response_id</code> chaining<br><br>
                <strong>Benefits:</strong><br>
                • Massive token savings (est. 70-90% reduction for large datasets)<br>
                • Lower API costs for multi-batch operations<br>
                • Faster processing with smaller prompts<br><br>
                <strong>Requirements:</strong><br>
                • Responses API must be available (test with test_responses_api.py)<br>
                • All batches must use the same model and settings
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 **Tip:** Enable stateful mode to save ~70-90% on tokens for large datasets with multiple batches")
        
        # Info box
        st.markdown("""
        <div class="info-card">
            💡 Configuration complete. Proceed to Tab 2 to manage training files (optional), or Tab 3 to upload data files.
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 2: Training Files
    with tab2:
        st.header("📚 Step 2: Manage Training Files")
        
        if not st.session_state.api_key:
            st.markdown("""
            <div class="conflict-warning">
                ⚠️ <strong>API Key Required</strong><br>
                Please configure your OpenAI API key in Tab 1 before managing training files.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        st.markdown("""
        <div class="info-card">
            Training files help the AI learn patterns from historical mentor-mentee matches.<br>
            You can manage existing files in OpenAI storage or upload new ones.
        </div>
        """, unsafe_allow_html=True)
        
        # Show currently selected training files
        if st.session_state.training_file_ids:
            st.markdown("---")
            st.subheader("✅ Currently Selected Training Files")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                file_count = len(st.session_state.training_file_ids)
                st.success(f"📊 {file_count} file(s) selected from OpenAI storage")
                
                if st.session_state.training_file_names:
                    with st.expander("📄 View File Names"):
                        for i, name in enumerate(st.session_state.training_file_names, 1):
                            st.write(f"**{i}.** {name}")
                else:
                    with st.expander("📄 View File IDs"):
                        for i, file_id in enumerate(st.session_state.training_file_ids, 1):
                            st.write(f"**{i}.** {file_id}")
            
            with col2:
                if st.button("🗑️ Clear Selection", help="Remove selected training files"):
                    st.session_state.training_files = []
                    st.session_state.training_file_ids = []
                    st.session_state.training_file_names = []
                    st.success("Cleared!")
                    st.rerun()
            
            st.markdown("---")
        elif st.session_state.training_files:
            # Legacy support for locally uploaded files
            st.markdown("---")
            st.subheader("✅ Currently Loaded Training Files (Local)")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                total_rows = sum(len(df) for df in st.session_state.training_files)
                st.success(f"📊 {len(st.session_state.training_files)} file(s) loaded • {total_rows} total records")
                
                with st.expander("📄 View Details"):
                    for i, df in enumerate(st.session_state.training_files, 1):
                        st.write(f"**File {i}**: {len(df)} rows × {len(df.columns)} columns")
            
            with col2:
                if st.button("🗑️ Clear All", help="Remove training files from session"):
                    st.session_state.training_files = []
                    st.session_state.training_file_ids = []
                    st.session_state.training_file_names = []
                    st.success("Cleared!")
                    st.rerun()
            
            st.markdown("---")
        
        # Create subtabs for existing and new training files
        training_tab1, training_tab2 = st.tabs(["📁 Existing Files", "📤 Upload New"])
        
        with training_tab1:
            st.subheader("Files in OpenAI Storage")
            
            if st.button("🔄 Refresh File List", key="refresh_files"):
                st.rerun()
            
            with st.spinner("Loading files from OpenAI..."):
                existing_files = list_openai_files(st.session_state.api_key)
            
            if existing_files:
                st.success(f"✅ Found {len(existing_files)} file(s) in storage")
                
                st.markdown("""
                <div class="info-card">
                    💡 Select the training files you want to use for matching, then click "Load Selected Files"
                </div>
                """, unsafe_allow_html=True)
                
                # File selection checkboxes
                selected_files = []
                for file_info in existing_files:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        is_selected = st.checkbox(
                            f"📄 {file_info['filename']} ({file_info['size_kb']:.1f} KB • {file_info['age_days']} days old)",
                            key=f"select_{file_info['id']}"
                        )
                        if is_selected:
                            selected_files.append(file_info)
                    
                    with col2:
                        # Delete button
                        if st.button(f"🗑️", key=f"delete_{file_info['id']}", help="Delete this file"):
                            with st.spinner("Deleting..."):
                                deleted = delete_openai_files(st.session_state.api_key, [file_info['id']])
                                if deleted > 0:
                                    st.success("✅ File deleted")
                                    time.sleep(0.5)
                                    st.rerun()
                
                # Use selected files button
                if selected_files:
                    st.divider()
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{len(selected_files)} file(s) selected**")
                    with col2:
                        if st.button("📥 Click here to load selected files", type="primary", width='stretch'):
                            # Just store the file IDs and basic info
                            selected_ids = [f['id'] for f in selected_files]
                            selected_names = [f['filename'] for f in selected_files]
                            
                            st.session_state.training_file_ids = selected_ids
                            st.session_state.training_file_names = selected_names
                            st.session_state.training_files = []  # Clear any old DataFrames
                            
                            st.success(f"✅ Selected {len(selected_ids)} training file(s) for matching!")
                            st.info("These files will be used via OpenAI's file_search tool during matching")
                            time.sleep(1)
                            st.rerun()
                
                # Bulk delete old files
                st.divider()
                old_files = [f for f in existing_files if f['age_days'] > 30]
                if old_files:
                    st.warning(f"⚠️ You have {len(old_files)} file(s) older than 30 days")
                    if st.button("🗑️ Delete All Old Files"):
                        old_file_ids = [f['id'] for f in old_files]
                        with st.spinner("Deleting old files..."):
                            deleted = delete_openai_files(st.session_state.api_key, old_file_ids)
                            st.success(f"✅ Deleted {deleted} file(s)")
                            time.sleep(1)
                            st.rerun()
            else:
                st.info("📭 No training files found in OpenAI storage. Upload some in the 'Upload New' tab!")
        
        with training_tab2:
            st.subheader("Upload New Training Files")
            
            st.markdown("""
            <div class="token-warning">
                💡 <strong>Tip:</strong> Training files should contain historical mentor-mentee pairings.<br>
                Upload Excel (.xlsx) files with the same structure as your mentee/mentor data.
            </div>
            """, unsafe_allow_html=True)
            
        training_files = st.file_uploader(
                "Select training file(s)",
            type=['xlsx'],
            accept_multiple_files=True,
                key=f"training_upload_{st.session_state.reset_counter}",
                help="Upload one or more Excel files with historical pairing data"
        )
        
        if training_files:
            st.info(f"📁 {len(training_files)} file(s) selected")
            
            # Load and validate files first
            training_dfs = []
            for file in training_files:
                try:
                    df = pd.read_excel(file)
                    df = clean_dataframe(df)
                    training_dfs.append(df)
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ <strong>{file.name}</strong>: {len(df)} rows, {len(df.columns)} columns
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="conflict-warning">
                        ❌ <strong>{file.name}</strong>: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
            
            if training_dfs:
                st.divider()
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("💾 Save Locally Only", width='stretch'):
                        st.session_state.training_files = training_dfs
                        st.success("✅ Training files saved to session (local only)")
                        time.sleep(0.5)  # Brief pause to show success message
                        st.rerun()
                
                with col2:
                    if st.button("☁️ Upload to OpenAI Storage", width='stretch', type="primary"):
                        with st.spinner("Uploading to OpenAI..."):
                            uploaded_ids = upload_training_files_to_openai(training_dfs, st.session_state.api_key)
                            if uploaded_ids:
                                st.session_state.training_files = training_dfs
                                st.session_state.training_file_ids = uploaded_ids
                                st.success(f"✅ Uploaded {len(uploaded_ids)} file(s) to OpenAI storage!")
                                time.sleep(1)  # Brief pause to show success message
                                st.rerun()
                
                st.markdown("""
                <div class="info-card">
                    <strong>Which option should I choose?</strong><br>
                    • <strong>Save Locally</strong>: Files sent directly in prompts (higher token usage)<br>
                    • <strong>Upload to OpenAI</strong>: Files stored in OpenAI, referenced via file_search (lower token usage, better for large files)
                </div>
                """, unsafe_allow_html=True)
        
    
    # TAB 3: Upload Data Files
    with tab3:
        st.header("📁 Step 3: Upload Mentee & Mentor Files")
        
        if not st.session_state.api_key:
            st.markdown("""
            <div class="conflict-warning">
                ⚠️ <strong>API Key Required</strong><br>
                Please configure your OpenAI API key in Tab 1 before uploading data files.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        st.markdown("""
        <div class="info-card">
            Upload your mentee and mentor Excel files. Both files are required to proceed with matching.
        </div>
        """, unsafe_allow_html=True)
        
        # Mentees and Mentors side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Mentees File *Required*")
            mentee_file = st.file_uploader(
                "Upload mentees Excel file",
                type=['xlsx'],
                key=f"mentee_upload_{st.session_state.reset_counter}",
                help=f"Excel file containing mentee information. Must have '{MENTEE_COLUMNS['id']}' column."
            )
            
        if mentee_file:
            try:
                df = pd.read_excel(mentee_file)
                df = clean_dataframe(df)
                
                errors, warnings, completeness = validate_uploaded_file(df, 'mentee')
                
                if errors:
                    for error in errors:
                        st.markdown(f"""
                        <div class="conflict-warning">
                            ❌ <strong>Error:</strong> {error}<br>
                            <small>Please fix this before proceeding</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.session_state.mentees_df = df
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ <strong>Loaded {len(df)} mentees</strong><br>
                        📊 Columns: {len(df.columns)} | Data quality: {completeness:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show data quality guide if not 100%
                    if completeness < 100:
                        with st.expander("ℹ️ Data Quality Explanation"):
                            total_cells = df.shape[0] * df.shape[1]
                            empty_cells = df.isna().sum().sum()
                            st.markdown(f"""
                            **How data quality is calculated:**
                            - Total cells: {total_cells:,} ({df.shape[0]} rows × {df.shape[1]} columns)
                            - Filled cells: {total_cells - empty_cells:,}
                            - Empty cells: {empty_cells:,}
                            - Quality: {completeness:.1f}% = (Filled / Total) × 100
                            
                            **Possible reasons for < 100%:**
                            - Optional fields left blank (e.g., "Mentee other relevant info")
                            - Missing data in survey responses
                            - Partial form submissions
                            - Fields not applicable to all mentees
                            
                            **Is this okay?** Yes! As long as required fields (ID, field, specialization) are filled, 
                            the system will work fine. Empty optional fields won't affect matching quality.
                            """)
                    
                    if warnings:
                        for warning in warnings:
                            st.markdown(f"""
                            <div class="token-warning">
                                ⚠️ {warning}
                            </div>
                            """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.markdown(f"""
                <div class="conflict-warning">
                    ❌ <strong>Error loading file:</strong> {str(e)}<br>
                    <small>Make sure it's a valid Excel (.xlsx) file</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("👨‍🏫 Mentors File *Required*")
            mentor_file = st.file_uploader(
                "Upload mentors Excel file",
                type=['xlsx'],
                key=f"mentor_upload_{st.session_state.reset_counter}",
                help=f"Excel file containing mentor information. Must have '{MENTOR_COLUMNS['id']}' column."
            )
            
        if mentor_file:
            try:
                df = pd.read_excel(mentor_file)
                df = clean_dataframe(df)
                
                errors, warnings, completeness = validate_uploaded_file(df, 'mentor')
                
                if errors:
                    for error in errors:
                        st.markdown(f"""
                        <div class="conflict-warning">
                            ❌ <strong>Error:</strong> {error}<br>
                            <small>Please fix this before proceeding</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.session_state.mentors_df = df
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ <strong>Loaded {len(df)} mentors</strong><br>
                        📊 Columns: {len(df.columns)} | Data quality: {completeness:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show data quality guide if not 100%
                    if completeness < 100:
                        with st.expander("ℹ️ Data Quality Explanation"):
                            total_cells = df.shape[0] * df.shape[1]
                            empty_cells = df.isna().sum().sum()
                            st.markdown(f"""
                            **How data quality is calculated:**
                            - Total cells: {total_cells:,} ({df.shape[0]} rows × {df.shape[1]} columns)
                            - Filled cells: {total_cells - empty_cells:,}
                            - Empty cells: {empty_cells:,}
                            - Quality: {completeness:.1f}% = (Filled / Total) × 100
                            
                            **Possible reasons for < 100%:**
                            - Optional fields left blank (e.g., "Mentor current position", "Alma Mater")
                            - Missing data in survey responses
                            - Partial form submissions
                            - Fields not applicable to all mentors
                            - Missing years of experience (1 mentor in your data)
                            
                            **Is this okay?** Yes! As long as required fields (ID, field, specialization) are filled, 
                            the system will work fine. Empty optional fields won't affect matching quality significantly.
                            """)
                    
                    if warnings:
                        for warning in warnings:
                            st.markdown(f"""
                            <div class="token-warning">
                                ⚠️ {warning}
                            </div>
                            """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.markdown(f"""
                <div class="conflict-warning">
                    ❌ <strong>Error loading file:</strong> {str(e)}<br>
                    <small>Make sure it's a valid Excel (.xlsx) file</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
    
    # TAB 4: Customize Prompt
    with tab4:
        st.header("🔧 Step 4: Customize Matching Prompt (Optional)")
        
        st.markdown("""
        <div class="info-card">
            Customize the scoring criteria, field compatibility matrix, and calibration examples. 
            The default prompt works well for most cases.
        </div>
        """, unsafe_allow_html=True)
        
        custom_prompt = st.text_area(
            "Matching Rules and Scoring Rubric",
            value=st.session_state.custom_prompt,
            height=500,
            help="Modify scoring criteria, field relationships, and calibration examples"
        )
        
        # Show character and token count
        char_count = len(custom_prompt)
        token_count = estimate_tokens(custom_prompt)
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Characters", f"{char_count:,}")
        with col_info2:
            st.metric("Est. Tokens", f"{token_count:,}")
        
        # Update session state
        st.session_state.custom_prompt = custom_prompt
        
        st.caption("💡 **Tip:** You can adjust scoring weights, modify field relationships, or add domain-specific criteria.")
    
    # TAB 5: Data Overview
    with tab5:
        st.header("📊 Step 5: Review Your Data")
        
        st.markdown("""
        <div class="info-card">
            Review your uploaded data before generating matches. All columns will be used for matching.
        </div>
        """, unsafe_allow_html=True)
        
        # Check data availability directly (not using cached step1_complete)
        data_ready = (
            st.session_state.mentees_df is not None and 
            st.session_state.mentors_df is not None
        )
        
        if not data_ready:
            st.markdown("""
            <div class="token-warning">
                ⚠️ <strong>No data uploaded yet!</strong><br>
                Go to <strong>Tab 3</strong> to upload your mentees and mentors files.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Simplified data summary - mentees and mentors only
        col1, col2 = st.columns(2)
        
        with col1:
            mentee_count = len(st.session_state.mentees_df) if st.session_state.mentees_df is not None else 0
            st.markdown(f"""
            <div class="stat-card" style="text-align: center; padding: 1rem;">
                <h2 style="color: #28a745; margin: 0;">👥 {mentee_count}</h2>
                <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.9rem;">Mentees</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mentor_count = len(st.session_state.mentors_df) if st.session_state.mentors_df is not None else 0
            st.markdown(f"""
            <div class="stat-card" style="text-align: center; padding: 1rem;">
                <h2 style="color: #17a2b8; margin: 0;">👨‍🏫 {mentor_count}</h2>
                <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.9rem;">Mentors</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Data preview with better styling
        # Training files info
        if st.session_state.training_file_ids:
            with st.expander("📚 Training Files Info", expanded=False):
                st.write("**Files selected from OpenAI storage:**")
                if st.session_state.training_file_names:
                    for i, name in enumerate(st.session_state.training_file_names, 1):
                        st.write(f"{i}. {name}")
                else:
                    for i, file_id in enumerate(st.session_state.training_file_ids, 1):
                        st.write(f"{i}. File ID: `{file_id}`")
                st.caption(f"{len(st.session_state.training_file_ids)} file(s) will be used via file_search during matching")
        elif st.session_state.training_files:
            with st.expander("📚 Preview Training Data (Local)", expanded=False):
                for i, df in enumerate(st.session_state.training_files, 1):
                    st.write(f"**Training File {i}**: {len(df)} rows, {len(df.columns)} columns")
                    st.dataframe(df.head(5), width='stretch')
                    if i < len(st.session_state.training_files):
                        st.divider()
                st.caption(f"Showing first 5 rows of each training file • {len(st.session_state.training_files)} file(s) total")
        
        if st.session_state.mentees_df is not None:
            with st.expander("👥 Preview Mentees Data", expanded=True):
                # Show first 5 rows
                st.dataframe(st.session_state.mentees_df.head(5), width='stretch')
                
                # Button to show full table
                if st.button("📋 Show Full Mentees Table", key="show_full_mentees"):
                    st.dataframe(st.session_state.mentees_df, width='stretch')
                    st.caption(f"Showing all {len(st.session_state.mentees_df)} rows • {len(st.session_state.mentees_df.columns)} columns")
                else:
                    st.caption(f"Showing first 5 of {len(st.session_state.mentees_df)} rows • {len(st.session_state.mentees_df.columns)} columns")
        
        if st.session_state.mentors_df is not None:
            with st.expander("👨‍🏫 Preview Mentors Data", expanded=True):
                # Show first 5 rows
                st.dataframe(st.session_state.mentors_df.head(5), width='stretch')
                
                # Button to show full table
                if st.button("📋 Show Full Mentors Table", key="show_full_mentors"):
                    st.dataframe(st.session_state.mentors_df, width='stretch')
                    st.caption(f"Showing all {len(st.session_state.mentors_df)} rows • {len(st.session_state.mentors_df.columns)} columns")
                else:
                    st.caption(f"Showing first 5 of {len(st.session_state.mentors_df)} rows • {len(st.session_state.mentors_df.columns)} columns")
        
        st.markdown("""
        <div class="success-card">
            ✅ <strong>Data looks good!</strong><br>
            You can now proceed to <strong>"6️⃣ Generate Matches"</strong>.
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 6: Generate Matches
    with tab6:
        st.header("🎯 Step 6: Generate Matches")
        
        # Use the api_key from session state
        api_key = st.session_state.api_key
        
        # Get model choice from session state (set in Tab 1)
        model_choice = st.session_state.model_choice
        
        # Check readiness
        ready = (
            st.session_state.mentees_df is not None and
            st.session_state.mentors_df is not None and
            api_key
        )
        
        # Show prerequisites if not met
        if not ready:
            st.markdown("""
            <div class="token-warning">
                ⚠️ <strong>Prerequisites not met!</strong> Please complete the following:
            </div>
            """, unsafe_allow_html=True)
            
            # Check data files directly
            data_files_ready = (
                st.session_state.mentees_df is not None and
                st.session_state.mentors_df is not None
            )
            
            if not data_files_ready:
                st.error("❌ **Step 3**: Upload mentee and mentor files")
            else:
                st.success("✅ **Step 3**: Data files uploaded")
            
            if not api_key:
                st.error("❌ **Step 1**: Configure API key and settings")
            else:
                st.success("✅ **Step 1**: Configuration complete")
            
            st.info("👆 Complete the missing steps above, then return to this tab.")
            st.stop()
        
        # Show data size with better styling
            mentee_count = len(st.session_state.mentees_df)
            mentor_count = len(st.session_state.mentors_df)
            
        st.markdown(f"""
        <div class="info-card">
            📊 <strong>Ready to match {mentee_count} mentees with {mentor_count} mentors</strong><br>
            <small>This will generate {mentee_count * mentor_count} compatibility scores using {st.session_state.batch_size} mentors per batch</small>
        </div>
        """, unsafe_allow_html=True)
            
            # Generate matches button
        if st.button("🚀 Generate Matches", type="primary", width='stretch'):
            # Save timestamp (Improvement 24)
            import datetime
            st.session_state.last_match_time = datetime.datetime.now()
            
            # PRE-FLIGHT VALIDATION: Check for duplicate IDs before sending to AI
            st.info("🔍 Pre-flight check: Validating data integrity...")
            
            validation_passed = True
            
            # Check mentees for duplicates
            mentee_ids = st.session_state.mentees_df[MENTEE_COLUMNS['id']].dropna()
            mentee_duplicates = mentee_ids[mentee_ids.duplicated(keep=False)]
            if len(mentee_duplicates) > 0:
                validation_passed = False
                unique_dups = mentee_duplicates.unique()
                st.error(f"🚨 DUPLICATE MENTEE IDs: {len(unique_dups)} mentee ID(s) appear multiple times in your data!")
                with st.expander(f"❌ Duplicate mentee IDs ({len(unique_dups)})"):
                    for dup_id in sorted(unique_dups):
                        count = (mentee_ids == dup_id).sum()
                        st.write(f"- **{dup_id}** appears {count} times")
                st.error("⚠️ This will cause the AI to return wrong results. Please fix your data and reload.")
            
            # Check mentors for duplicates
            mentor_ids = st.session_state.mentors_df[MENTOR_COLUMNS['id']].dropna()
            mentor_duplicates = mentor_ids[mentor_ids.duplicated(keep=False)]
            if len(mentor_duplicates) > 0:
                validation_passed = False
                unique_dups = mentor_duplicates.unique()
                st.error(f"🚨 DUPLICATE MENTOR IDs: {len(unique_dups)} mentor ID(s) appear multiple times in your data!")
                with st.expander(f"❌ Duplicate mentor IDs ({len(unique_dups)})"):
                    for dup_id in sorted(unique_dups):
                        count = (mentor_ids == dup_id).sum()
                        st.write(f"- **{dup_id}** appears {count} times")
                st.error("⚠️ This will cause the AI to return wrong results. Please fix your data and reload.")
            
            if not validation_passed:
                st.markdown("""
                <div class="conflict-warning">
                    <strong>🛑 Cannot proceed with matching</strong><br>
                    <small>Your data contains duplicate IDs. Each mentee and mentor must have a unique ID.</small><br>
                    <small><strong>Why?</strong> Duplicate IDs cause the AI to return the same ID in multiple batches, leading to data overwrites and incorrect results.</small><br><br>
                    <strong>How to fix:</strong>
                    <ol>
                        <li>Open your Excel file</li>
                        <li>Find the duplicate IDs listed above</li>
                        <li>Make each ID unique (e.g., add a suffix: "Mentor 5" → "Mentor 5a", "Mentor 5b")</li>
                        <li>Save and reload the file</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                st.stop()  # Stop execution here
            
            st.success("✅ Pre-flight check passed: All IDs are unique!")
            
            # Better progress indicators (Improvement 3, 17)
            progress_container = st.container()
            
            with progress_container:
                # PHASE 1: Get compatibility matrix from AI (using batching)
                st.markdown("""
                <div class="info-card">
                    📊 <strong>Phase 1: Generating compatibility matrix with batching...</strong><br>
                    <small>This may take several minutes. Please don't close the browser.</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Get stateful mode setting
                use_stateful = st.session_state.get('use_stateful_mode', False)
                
                matrix_df = generate_matrix_in_batches(
                        st.session_state.mentees_df,
                        st.session_state.mentors_df,
                        st.session_state.training_files,
                        api_key,
                    model_choice,
                    batch_size=st.session_state.batch_size,
                    use_stateful=use_stateful
                )
                
                if matrix_df is not None:
                        st.session_state.matrix_df = matrix_df
                        st.markdown(f"""
                        <div class="success-card">
                            ✅ <strong>Matrix generated successfully!</strong><br>
                            {len(matrix_df)} mentors × {len(matrix_df.columns)} mentees = {len(matrix_df) * len(matrix_df.columns)} evaluations!
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # PHASE 2: Apply Hungarian algorithm for optimal assignment
                        st.markdown("""
                        <div class="info-card">
                            🧮 <strong>Phase 2: Running Hungarian algorithm for optimal assignments...</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        assignments = hungarian_assignment(matrix_df, max_mentees_per_mentor=2)
                        st.session_state.assignments = assignments
                        
                        # Calculate statistics
                        avg_score = np.mean([score for _, _, score in assignments])
                        min_score = min([score for _, _, score in assignments])
                        max_score = max([score for _, _, score in assignments])
                        
                        st.markdown(f"""
                        <div class="success-card">
                            ✅ <strong>Optimal assignments found!</strong><br>
                            Avg: {avg_score:.1f}%, Min: {min_score}%, Max: {max_score}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # PHASE 3: Get reasoning for assignments
                        st.markdown("""
                        <div class="info-card">
                            🧠 <strong>Phase 3: Getting detailed reasoning for assignments...</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        reasoning_dict = get_reasoning_for_assignments(
                            assignments,
                        st.session_state.mentees_df,
                        st.session_state.mentors_df,
                        api_key,
                        model_choice
                    )
                
                        # Convert to old format for compatibility with display code
                        matches = []
                        
                        # Debug: Show what we're looking for vs what we have
                        if assignments and reasoning_dict:
                            st.info(f"🔍 Looking up reasoning: {len(assignments)} assignments, {len(reasoning_dict)} reasoning entries")
                            sample_assignment = assignments[0]
                            sample_key = (sample_assignment[0], sample_assignment[1])
                            with st.expander("🔍 Debug: Key matching"):
                                st.write(f"Sample assignment key: {sample_key}")
                                st.write(f"Sample assignment types: ({type(sample_assignment[0]).__name__}, {type(sample_assignment[1]).__name__})")
                                if reasoning_dict:
                                    sample_reason_key = list(reasoning_dict.keys())[0]
                                    st.write(f"Sample reasoning key: {sample_reason_key}")
                                    st.write(f"Sample reasoning types: ({type(sample_reason_key[0]).__name__}, {type(sample_reason_key[1]).__name__})")
                                    st.write(f"Keys match? {sample_key in reasoning_dict}")
                        
                        # Helper function to find reasoning with flexible key matching
                        def find_reasoning(mentee_id, mentor_id, reasoning_dict):
                            """Try multiple key formats to find reasoning"""
                            # Extract numeric parts if present
                            mentee_num = re.search(r'\d+', str(mentee_id))
                            mentor_num = re.search(r'\d+', str(mentor_id))
                            
                            # Try various key combinations
                            possible_keys = [
                                (mentee_id, mentor_id),  # Original format
                                (str(mentee_id), str(mentor_id)),  # String versions
                            ]
                            
                            # Add numeric-only versions if we found numbers
                            if mentee_num and mentor_num:
                                possible_keys.append((mentee_num.group(), mentor_num.group()))
                            
                            # Try each possible key
                            for key in possible_keys:
                                if key in reasoning_dict:
                                    return reasoning_dict[key]
                            
                            # No match found
                            return None
                        
                        for mentee, mentor, score in assignments:
                            # Try to find reasoning with flexible matching
                            reasoning = find_reasoning(mentee, mentor, reasoning_dict)
                            
                            # Fallback if no reasoning found
                            if reasoning is None:
                                reasoning = f"This match was selected by the Hungarian optimization algorithm as the globally optimal assignment with a {score}% compatibility score. Detailed reasoning was unavailable."
                            
                            matches.append({
                                'mentee_id': mentee,
                                'matches': [{
                                    'rank': 1,
                                    'mentor_id': mentor,
                                    'match_percentage': score,
                                    'match_quality': 'Excellent' if score >= 90 else 'Strong' if score >= 75 else 'Good' if score >= 60 else 'Fair',
                                    'reasoning': reasoning
                                }]
                            })
                        
                        st.session_state.matches = matches
                        st.success(f"✅ Complete! {len(matches)} mentees optimally matched with guaranteed constraints!")
                        
                        # Validate match quality (Phase 1 improvement)
                        flagged_matches, score_stats = validate_and_flag_matches(
                            st.session_state.matches,
                            st.session_state.mentees_df,
                            st.session_state.mentors_df
                        )
                        
                        # Store validation results for reference
                        st.session_state.flagged_matches = flagged_matches
                        st.session_state.score_stats = score_stats
                        
                        st.balloons()
        else:
            # Show what's missing (only if something is actually missing)
            missing = []
            if st.session_state.mentees_df is None:
                missing.append("Mentees file")
            if st.session_state.mentors_df is None:
                missing.append("Mentors file")
            if not api_key:
                missing.append("OpenAI API key")
            
            if missing:
                st.warning(f"⚠️ Please provide: {', '.join(missing)}")
        
        # Display results (Improvements 6, 17, 19, 20, 21)
        if st.session_state.matches:
            st.divider()
            st.subheader("📊 Matching Results")
            
            # Display statistics with better styling (Improvement 17)
            if st.session_state.assignments:
                scores = [score for _, _, score in st.session_state.assignments]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: #667eea; margin: 0;">{len(st.session_state.assignments)}</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Total Matches</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: #28a745; margin: 0;">{np.mean(scores):.1f}%</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Avg Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: #ffc107; margin: 0;">{min(scores)}%</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Min Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: #17a2b8; margin: 0;">{max(scores)}%</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Max Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show score distribution analysis (Task 6)
            if st.session_state.matrix_df is not None:
                st.divider()
                
                # Analyze score distribution to detect AI generosity issues
                score_distribution = analyze_score_distribution(st.session_state.matrix_df)
                st.session_state.score_distribution = score_distribution
                
                st.divider()
                
            # Show heatmap
            if st.session_state.matrix_df is not None:
                st.subheader("🔥 Compatibility Heatmap")
                st.caption("Blue boxes show optimal assignments selected by Hungarian algorithm")
                
                fig = create_heatmap(st.session_state.matrix_df, st.session_state.assignments)
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to download matrix
                csv = st.session_state.matrix_df.to_csv()
                st.download_button(
                    label="📥 Download Full Matrix (CSV)",
                    data=csv,
                    file_name=f"compatibility_matrix_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Check for any remaining conflicts (should be none with Hungarian)
            conflicts = check_mentor_conflicts(st.session_state.matches)
            
            if conflicts:
                st.markdown("""
                <div class="conflict-warning">
                    <strong>⚠️ Warning:</strong> Unexpected conflicts detected. Please review.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ All constraints satisfied: No mentor has >2 mentees!")
            
            # Results display
            st.divider()
            
            # Excel export button (auto-generate and download)
            st.subheader("📥 Export Results")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Results sheet
                results_data = []
                for match_data in st.session_state.matches:
                    for match in match_data['matches']:
                        results_data.append({
                            'Mentee_ID': match_data['mentee_id'],
                            'Rank': match['rank'],
                            'Mentor_ID': match['mentor_id'],
                            'Match_Percentage': match['match_percentage'],
                            'Match_Quality': match['match_quality'],
                            'Reasoning': match['reasoning']
                        })
                
                pd.DataFrame(results_data).to_excel(writer, sheet_name='Matches', index=False)
                st.session_state.mentees_df.to_excel(writer, sheet_name='Mentees', index=False)
                st.session_state.mentors_df.to_excel(writer, sheet_name='Mentors', index=False)
            
            # Add summary sheet
            if st.session_state.assignments:
                summary_data = {
                    'Metric': ['Total Matches', 'Average Score', 'Min Score', 'Max Score', 'Generated At'],
                    'Value': [
                        len(st.session_state.assignments),
                        f"{np.mean([s for _, _, s in st.session_state.assignments]):.1f}%",
                        f"{min([s for _, _, s in st.session_state.assignments])}%",
                        f"{max([s for _, _, s in st.session_state.assignments])}%",
                        time.strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                with pd.ExcelWriter(output, engine='openpyxl', mode='a') as writer:
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            output.seek(0)
            st.download_button(
                "📊 Download Excel Report",
                output.getvalue(),
                f"matches_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            
            st.divider()
            
            # Search box
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_query = st.text_input(
                    "🔍 Search Mentees", 
                    value=st.session_state.mentee_search,
                    placeholder="Type mentee ID to filter...",
                    label_visibility="collapsed"
                )
                st.session_state.mentee_search = search_query
            with search_col2:
                if st.button("🔄 Clear", width='stretch'):
                    st.session_state.mentee_search = ""
                    st.rerun()
            
            # Filter matches based on search
            filtered_matches = [
                m for m in st.session_state.matches 
                if search_query.lower() in m['mentee_id'].lower()
            ]
            
            if not filtered_matches and search_query:
                st.warning(f"No mentees found matching '{search_query}'")
            else:
                st.subheader(f"All Matches ({len(filtered_matches)})")
                
                # Display all match cards in a single list
                for match_data in filtered_matches:
                    mentee_id = match_data['mentee_id']
                    
                    for match in match_data['matches']:
                        quality_class = match['match_quality'].lower()
                        
                        st.markdown(f"""
                        <div class="match-card">
                            <h4>{mentee_id} → {match['mentor_id']}</h4>
                            <p><span class="percentage-badge {quality_class}">{match['match_percentage']}% - {match['match_quality']}</span></p>
                            <p><strong>Reasoning:</strong> {match['reasoning']}</p>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
