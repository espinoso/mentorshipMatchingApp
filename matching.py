"""
Matching algorithms for the Mentorship Matching System
Contains functions for optimal assignment, conflict detection, and response parsing
"""

import json
import re
import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import linear_sum_assignment
from utils import show_debug_info
from info import MENTEE_COLUMNS, MENTOR_COLUMNS


def parse_matrix_response(response_text, expected_mentors, expected_mentees):
    """Parse the matrix response from AI and validate completeness + ID matching"""
    try:
        # Ensure response_text is a string (handle ResponseTextConfig objects)
        if not isinstance(response_text, str):
            # Try to extract text from ResponseTextConfig or other objects
            if hasattr(response_text, 'text'):
                # Check if it's nested (ResponseTextConfig.text.text)
                text_obj = response_text.text
                if hasattr(text_obj, 'text'):
                    response_text = str(text_obj.text)
                elif isinstance(text_obj, str):
                    response_text = text_obj
                else:
                    response_text = str(text_obj)
            else:
                # Convert to string as fallback
                response_text = str(response_text)
        
        # Clean the response text
        response_text = response_text.strip()
        
        # Check if we got a ResponseTextConfig string representation (invalid - from reasoning models)
        if response_text.startswith("ResponseTextConfig") or "ResponseFormatText" in response_text or "ResponseTextConfig" in response_text:
            st.error("❌ **Invalid response: Received configuration object instead of JSON content**")
            st.warning("""
            This happens when using reasoning models (gpt-5-mini, gpt-5-nano) that cannot produce structured JSON.
            
            **Solution:** Please use a compatible model:
            - ✅ **gpt-4.1-mini** (recommended)
            - ✅ **gpt-4.1-nano** (fastest)
            - ✅ **gpt-5.1** or **gpt-4.1** (highest quality)
            """)
            return None
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        import re
        
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_portion = response_text[first_brace:last_brace + 1]
            
            if first_brace > 0 or last_brace < len(response_text) - 1:
                before_text = response_text[:first_brace].strip()
                after_text = response_text[last_brace + 1:].strip()
                
                st.warning("⚠️ AI included explanatory text instead of pure JSON. Attempting to extract JSON...")
                
                if before_text:
                    with st.expander("📝 Text before JSON (should be empty)"):
                        st.text(before_text[:500])
                
                if after_text:
                    with st.expander("📝 Text after JSON (should be empty)"):
                        st.text(after_text[:500])
            
            response_text = json_portion
        else:
            st.error("❌ Could not find JSON object boundaries in response")
            with st.expander("View raw response"):
                st.text(response_text[:2000])
            return None
        
        data = json.loads(response_text)
        
        if 'matrix' not in data:
            raise ValueError("Response should contain 'matrix' key")
        
        matrix_data = data['matrix']
        if not isinstance(matrix_data, list):
            raise ValueError("Matrix should be a list")
        
        matrix_dict = {}
        all_mentees = set()
        
        expected_mentor_set = set(str(x) for x in expected_mentors)
        expected_mentee_set = set(str(x) for x in expected_mentees)
        wrong_mentors = []
        wrong_mentees = []
        
        for mentor_entry in matrix_data:
            mentor_id = str(mentor_entry['mentor_id'])
            
            if mentor_id not in expected_mentor_set:
                wrong_mentors.append(mentor_id)
                st.warning(f"⚠️ AI returned WRONG mentor ID: '{mentor_id}' (not in requested batch)")
                continue
            
            scores = {}
            for score_entry in mentor_entry['scores']:
                mentee_id = str(score_entry['mentee_id'])
                percentage = score_entry['percentage']
                
                if mentee_id not in expected_mentee_set:
                    if mentee_id not in wrong_mentees:
                        wrong_mentees.append(mentee_id)
                    continue
                
                scores[mentee_id] = percentage
                all_mentees.add(mentee_id)
            
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
                for m in sorted(wrong_mentees)[:10]:
                    st.write(f"- {m}")
        
        mentors_received = len(matrix_dict)
        mentees_received = len(all_mentees)
        expected_mentor_count = len(expected_mentors)
        expected_mentee_count = len(expected_mentees)
        
        # Suppress these messages - progress UI shows batch status instead
        # Only show errors and debug info
        from utils import is_debug_mode
        
        if is_debug_mode():
            st.caption(f"📊 Valid data: {mentors_received}/{expected_mentor_count} mentors, {mentees_received}/{expected_mentee_count} mentees")
        
        expected_total_scores = expected_mentor_count * expected_mentee_count
        actual_total_scores = sum(len(scores) for scores in matrix_dict.values())
        
        if actual_total_scores < expected_total_scores:
            missing_count = expected_total_scores - actual_total_scores
            completion_rate = (actual_total_scores / expected_total_scores) * 100 if expected_total_scores > 0 else 0
            
            st.error(f"🚨 INCOMPLETE RESPONSE: Expected {expected_total_scores} scores, got {actual_total_scores} ({completion_rate:.1f}% complete)")
            st.error(f"   Missing {missing_count} combinations!")
            st.error(f"   ⚠️ The AI is still omitting combinations despite instructions.")
            
            if is_debug_mode():
                with st.expander("📋 Mentors with incomplete score arrays"):
                    for mentor_id, scores in matrix_dict.items():
                        scores_count = len(scores)
                        if scores_count < expected_mentee_count:
                            st.warning(f"   • Mentor {mentor_id}: {scores_count}/{expected_mentee_count} mentees ({expected_mentee_count - scores_count} missing)")
        elif actual_total_scores == expected_total_scores:
            # Only show success in debug mode
            if is_debug_mode():
                st.caption(f"✅ COMPLETE: All {expected_total_scores} scores received")
        
        if mentors_received < expected_mentor_count:
            missing_mentors = set(str(m) for m in expected_mentors) - set(matrix_dict.keys())
            st.warning(f"⚠️ Missing {len(missing_mentors)} requested mentor(s)")
            with st.expander(f"📋 Missing mentors ({len(missing_mentors)})"):
                for m in sorted(missing_mentors):
                    st.write(f"- {m}")
        
        if mentees_received < expected_mentee_count:
            missing_mentees = set(str(m) for m in expected_mentees) - all_mentees
            st.warning(f"⚠️ Missing {len(missing_mentees)} mentee(s) in responses")
            with st.expander(f"📋 Missing mentees ({len(missing_mentees)})"):
                for m in sorted(missing_mentees):
                    st.write(f"- {m}")
        
        if not matrix_dict:
            st.error("❌ No valid mentor data after filtering!")
            return None
        
        df = pd.DataFrame(matrix_dict).T
        
        # Fill missing mentees with LOW score (30) - AI omitted these (likely poor matches)
        missing_mentees = [m for m in expected_mentees if str(m) not in df.columns]
        if missing_mentees:
            st.warning(f"⚠️ {len(missing_mentees)} mentee column(s) missing from matrix. Filling with 30% (assumed poor match).")
            st.caption("The AI omitted these combinations despite instructions. This indicates likely poor matches.")
            show_debug_info()
            for mentee in missing_mentees:
                df[mentee] = 30
        
        df = df[[str(m) for m in sorted(expected_mentees)]]
        
        return df
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        st.error(f"❌ Error parsing matrix response: {str(e)}")
        show_debug_info()
        with st.expander("View raw response"):
            st.text(response_text[:2000])
        return None


def hungarian_assignment(matrix_df, max_mentees_per_mentor=2):
    """
    Use Hungarian algorithm to find optimal mentee-mentor assignments
    
    Args:
        matrix_df: DataFrame with mentors as rows, mentees as columns, percentages as values
        max_mentees_per_mentor: Maximum number of mentees per mentor (default 2)
    
    Returns:
        List of tuples: [(mentee_id, mentor_id, score), ...]
    """
    mentors = list(matrix_df.index)
    mentees = list(matrix_df.columns)
    
    st.write("🔍 Validating matrix data...")
    
    nan_count = matrix_df.isna().sum().sum()
    if nan_count > 0:
        st.warning(f"⚠️ Found {nan_count} NaN values in matrix. Replacing with 30% (poor match - AI omitted these)...")
        st.caption("NaN values indicate the AI skipped these combinations despite instructions. Treating as poor matches.")
        show_debug_info()
        matrix_df = matrix_df.fillna(30)
    
    inf_mask = np.isinf(matrix_df.values)
    if inf_mask.any():
        inf_count = inf_mask.sum()
        st.warning(f"⚠️ Found {inf_count} infinite values in matrix. Replacing with 50%...")
        matrix_df = matrix_df.replace([np.inf, -np.inf], 50)
    
    try:
        matrix_df = matrix_df.astype(float)
    except Exception as e:
        st.error(f"❌ Matrix contains non-numeric values: {e}")
        st.write("Sample of problematic data:")
        st.write(matrix_df.head())
        raise ValueError(f"Matrix must contain only numeric values: {e}")
    
    min_val = matrix_df.min().min()
    max_val = matrix_df.max().max()
    if min_val < 0 or max_val > 100:
        st.warning(f"⚠️ Matrix values outside expected range [0, 100]: min={min_val}, max={max_val}")
        matrix_df = matrix_df.clip(0, 100)
    
    st.write(f"✅ Matrix validation passed: {len(mentors)} mentors × {len(mentees)} mentees")
    
    expanded_mentors = []
    for mentor in mentors:
        for i in range(max_mentees_per_mentor):
            expanded_mentors.append(f"{mentor}_copy{i}")
    
    cost_matrix = np.zeros((len(mentees), len(expanded_mentors)))
    
    for i, mentee in enumerate(mentees):
        for j, expanded_mentor in enumerate(expanded_mentors):
            actual_mentor = expanded_mentor.rsplit('_copy', 1)[0]
            score = matrix_df.loc[actual_mentor, mentee]
            cost_matrix[i, j] = -score
    
    if not np.isfinite(cost_matrix).all():
        st.error("❌ Cost matrix still contains invalid values after cleaning")
        st.write("Cost matrix stats:")
        st.write(f"- NaN count: {np.isnan(cost_matrix).sum()}")
        st.write(f"- Inf count: {np.isinf(cost_matrix).sum()}")
        raise ValueError("Cost matrix contains invalid numeric entries after cleaning")
    
    st.write("🧮 Running Hungarian algorithm for optimal assignment...")
    mentee_indices, mentor_indices = linear_sum_assignment(cost_matrix)
    
    assignments = []
    for mentee_idx, mentor_idx in zip(mentee_indices, mentor_indices):
        mentee = mentees[mentee_idx]
        expanded_mentor = expanded_mentors[mentor_idx]
        actual_mentor = expanded_mentor.rsplit('_copy', 1)[0]
        score = matrix_df.loc[actual_mentor, mentee]
        assignments.append((mentee, actual_mentor, int(score)))
    
    # Sort numerically by mentee_id (handle string IDs like "1", "2", "10", "11")
    def numeric_sort_key(x):
        mentee_id = str(x[0])
        # Extract numeric part from mentee_id
        match = re.search(r'\d+', mentee_id)
        if match:
            return int(match.group())
        return 0
    
    assignments.sort(key=numeric_sort_key)
    
    return assignments


def check_mentor_conflicts(matches):
    """Check if any mentor is assigned to more than 2 mentees at rank 1"""
    if not matches:
        return []
    
    mentor_assignments = {}
    conflicts = []
    
    for match_data in matches:
        mentee_id = match_data['mentee_id']
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


def apply_country_restrictions(matrix_df, mentees_df, mentors_df):
    """
    Apply country-based restrictions to the compatibility matrix.
    
    Rule: Spanish mentors cannot mentor Spanish mentees.
    Spanish mentors can only mentor mentees from other countries (e.g., Portugal).
    
    Args:
        matrix_df: DataFrame with mentors as rows, mentees as columns, scores as values
        mentees_df: DataFrame containing mentee data with country information
        mentors_df: DataFrame containing mentor data with country information
        
    Returns:
        tuple: (filtered_matrix_df, filtered_matches_list)
            - filtered_matrix_df: Matrix with restricted matches set to 0
            - filtered_matches_list: List of dicts with info about filtered matches
                [{"mentor_id": "...", "mentee_id": "...", "original_score": 85, "reason": "Spain-Spain restriction"}]
    """
    filtered_matrix_df = matrix_df.copy()
    filtered_matches = []
    
    # Get country columns
    mentor_country_col = MENTOR_COLUMNS['country']
    mentee_country_col = MENTEE_COLUMNS['country']
    
    # Verify columns exist
    if mentor_country_col not in mentors_df.columns:
        st.warning(f"⚠️ Mentor country column '{mentor_country_col}' not found. Skipping country restrictions.")
        return filtered_matrix_df, filtered_matches
    
    if mentee_country_col not in mentees_df.columns:
        st.warning(f"⚠️ Mentee country column '{mentee_country_col}' not found. Skipping country restrictions.")
        return filtered_matrix_df, filtered_matches
    
    # Create mapping of IDs to countries (normalized to lowercase)
    mentor_countries = {}
    for idx, row in mentors_df.iterrows():
        mentor_id = str(row[MENTOR_COLUMNS['id']])
        country = str(row[mentor_country_col]).strip().lower() if pd.notna(row[mentor_country_col]) else ""
        mentor_countries[mentor_id] = country
    
    mentee_countries = {}
    for idx, row in mentees_df.iterrows():
        mentee_id = str(row[MENTEE_COLUMNS['id']])
        country = str(row[mentee_country_col]).strip().lower() if pd.notna(row[mentee_country_col]) else ""
        mentee_countries[mentee_id] = country
    
    # Apply restriction: Spain-Spain matches = 0
    spain_mentors = [mid for mid, country in mentor_countries.items() if country == 'spain']
    spain_mentees = [eid for eid, country in mentee_countries.items() if country == 'spain']
    
    if spain_mentors and spain_mentees:
        st.info(f"🇪🇸 Found {len(spain_mentors)} Spanish mentor(s) and {len(spain_mentees)} Spanish mentee(s)")
    
    for mentor_id in filtered_matrix_df.index:
        mentor_country = mentor_countries.get(str(mentor_id), "")
        
        if mentor_country == 'spain':
            for mentee_id in filtered_matrix_df.columns:
                mentee_id_str = str(mentee_id)
                mentee_country = mentee_countries.get(mentee_id_str, "")
                
                if mentee_country == 'spain':
                    # Get original score before filtering
                    original_score = filtered_matrix_df.loc[mentor_id, mentee_id]
                    
                    # Set to 0 (invalid match)
                    filtered_matrix_df.loc[mentor_id, mentee_id] = 0
                    
                    # Track this filtered match
                    if pd.notna(original_score) and original_score > 0:
                        filtered_matches.append({
                            'mentor_id': str(mentor_id),
                            'mentee_id': mentee_id_str,
                            'original_score': float(original_score),
                            'reason': 'Spain-Spain restriction: Spanish mentors cannot mentor Spanish mentees'
                        })
    
    return filtered_matrix_df, filtered_matches

