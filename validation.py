"""
Data validation and quality analysis functions for the Mentorship Matching System
Contains functions for validating matches, analyzing score distributions, and preparing data
"""

import streamlit as st
import pandas as pd
import numpy as np
from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from utils import is_field_compatible


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
            
            try:
                mentee_id_int = int(mentee_id)
                mentor_id_int = int(mentor_id)
                mentee = mentees_df[mentees_df[MENTEE_COLUMNS['id']] == mentee_id_int].iloc[0]
                mentor = mentors_df[mentors_df[MENTOR_COLUMNS['id']] == mentor_id_int].iloc[0]
            except (ValueError, IndexError) as e:
                st.warning(f"⚠️ Could not extract details for {mentee_id} → {mentor_id}: {e}")
                continue
            
            issues = []
            
            if score >= 75:
                mentee_field = str(mentee[MENTEE_COLUMNS['field']]).lower().strip()
                mentor_field = str(mentor[MENTOR_COLUMNS['field']]).lower().strip()
                
                if mentee_field != mentor_field:
                    if not is_field_compatible(mentee_field, mentor_field):
                        issues.append(f"Score {score}% but fields don't match: '{mentee_field}' vs '{mentor_field}'")
            
            if score >= 70:
                mentee_spec = str(mentee.get(MENTEE_COLUMNS['specialization'], '')).lower()
                mentor_spec = str(mentor.get(MENTOR_COLUMNS['specialization'], '')).lower()
                
                mentee_words = set(word for word in mentee_spec.split() if len(word) > 3)
                mentor_words = set(word for word in mentor_spec.split() if len(word) > 3)
                overlap = len(mentee_words & mentor_words)
                
                if overlap == 0 and mentee_spec and mentor_spec:
                    issues.append(f"Score {score}% but 0 keyword overlap in specializations")
            
            if score >= 85:
                mentee_field = str(mentee[MENTEE_COLUMNS['field']]).lower().strip()
                mentor_field = str(mentor[MENTOR_COLUMNS['field']]).lower().strip()
                
                if mentee_field != mentor_field and not is_field_compatible(mentee_field, mentor_field):
                    issues.append(f"Excellent score ({score}%) requires field match")
                
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
    
    if matches:
        all_scores = [match['match_percentage'] for match_data in matches for match in match_data['matches']]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        st.info(f"📈 Average Score: {avg_score:.1f}%")
    
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
    all_scores = matrix_df.values.flatten()
    all_scores = all_scores[~np.isnan(all_scores)]
    
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
    
    col_avg, col_med = st.columns(2)
    with col_avg:
        st.info(f"📈 **Average Score**: {distribution['average']:.1f}%")
    with col_med:
        st.info(f"📊 **Median Score**: {distribution['median']:.1f}%")
    
    st.caption("💡 **Expected Distribution**: Most scores should be in 30-60% range (poor/fair matches), with only 5-15% scoring 75%+")
    
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


def clean_text_for_csv(df):
    """Clean text fields to prevent CSV parsing issues"""
    df = df.copy()
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace('\n', ' ', regex=False)
        df[col] = df[col].astype(str).str.replace('\r', ' ', regex=False)
        df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
        df[col] = df[col].str.strip()
        df[col] = df[col].replace('nan', np.nan)
    
    return df


def prepare_data_for_assistant(mentees_df, mentors_df):
    """Prepare data with ALL columns for comprehensive matching
    
    Args:
        mentees_df: DataFrame of mentees (None to skip mentees section for stateful subsequent batches)
        mentors_df: DataFrame of mentors
    """
    
    combined_data = "# MENTORSHIP MATCHING DATA\n\n"
    
    if mentees_df is not None:
        mentees_full = mentees_df.copy()
        mentees_cleaned = clean_text_for_csv(mentees_full)
        combined_data += "## MENTEES TO MATCH (ALL DATA)\n"
        combined_data += f"Total mentees: {len(mentees_cleaned)}\n"
        combined_data += f"All columns included: {', '.join(mentees_cleaned.columns.tolist())}\n\n"
        combined_data += mentees_cleaned.to_csv(index=False)
        combined_data += "\n"
    
    mentors_full = mentors_df.copy()
    mentors_cleaned = clean_text_for_csv(mentors_full)
    combined_data += "## AVAILABLE MENTORS (ALL DATA)\n"
    combined_data += f"Total mentors: {len(mentors_cleaned)}\n"
    combined_data += f"All columns included: {', '.join(mentors_cleaned.columns.tolist())}\n\n"
    combined_data += mentors_cleaned.to_csv(index=False)
    
    return combined_data

