"""
UI Component for Tab 7: Processing & Results
Handles the matching generation process, results display, and export functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from io import BytesIO
from datetime import datetime
from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from matching import hungarian_assignment, check_mentor_conflicts
from validation import validate_and_flag_matches, analyze_score_distribution
from visualization import create_heatmap
from utils import is_debug_mode
from progress_ui import render_processing_summary


def get_country_flag(country_name):
    """Convert country name to flag emoji (basic mapping for common countries)"""
    if not country_name or pd.isna(country_name):
        return "🌍"
    
    # Normalize country name: strip, lowercase, and handle common variations
    country_normalized = str(country_name).strip().lower()
    
    # Handle common country name variations (e.g., "España" -> "Spain")
    country_variations = {
        'españa': 'spain', 'espana': 'spain',  # Spanish name variations
        'portugal': 'portugal',  # Already in English
        'reino unido': 'united kingdom', 'reino unido de gran bretaña': 'united kingdom',
        'estados unidos': 'united states', 'ee.uu.': 'united states', 'eeuu': 'united states',
        'méxico': 'mexico', 'mexico': 'mexico',
        'brasil': 'brazil',
    }
    
    # Check if it's a known variation first
    if country_normalized in country_variations:
        country_normalized = country_variations[country_normalized]
    
    # Common country to flag emoji mapping
    country_flags = {
        'spain': '🇪🇸', 'united states': '🇺🇸', 'usa': '🇺🇸', 'united kingdom': '🇬🇧', 'uk': '🇬🇧',
        'france': '🇫🇷', 'germany': '🇩🇪', 'italy': '🇮🇹', 'portugal': '🇵🇹', 'netherlands': '🇳🇱',
        'belgium': '🇧🇪', 'switzerland': '🇨🇭', 'austria': '🇦🇹', 'sweden': '🇸🇪', 'norway': '🇳🇴',
        'denmark': '🇩🇰', 'finland': '🇫🇮', 'poland': '🇵🇱', 'greece': '🇬🇷', 'ireland': '🇮🇪',
        'canada': '🇨🇦', 'mexico': '🇲🇽', 'brazil': '🇧🇷', 'argentina': '🇦🇷', 'chile': '🇨🇱',
        'colombia': '🇨🇴', 'peru': '🇵🇪', 'venezuela': '🇻🇪', 'ecuador': '🇪🇨', 'uruguay': '🇺🇾',
        'japan': '🇯🇵', 'china': '🇨🇳', 'india': '🇮🇳', 'south korea': '🇰🇷', 'australia': '🇦🇺',
        'new zealand': '🇳🇿', 'south africa': '🇿🇦', 'egypt': '🇪🇬', 'turkey': '🇹🇷', 'israel': '🇮🇱',
        'russia': '🇷🇺', 'ukraine': '🇺🇦', 'czech republic': '🇨🇿', 'hungary': '🇭🇺', 'romania': '🇷🇴'
    }
    
    # Try exact match first
    if country_normalized in country_flags:
        return country_flags[country_normalized]
    
    # Try partial match (e.g., "spain" in "spain country" or vice versa)
    for country, flag in country_flags.items():
        if country in country_normalized or country_normalized in country:
            return flag
    
    # Default to globe if not found
    return "🌍"

# Note: These functions will be moved to api_client.py in the next phase
# Importing here to avoid circular dependency - they're defined in app.py
# TODO: Move to api_client.py and update import


def render_processing_tab():
    """Render Tab 7: Processing & Results"""
    
    # Use the api_key from session state
    api_key = st.session_state.api_key
    model_choice = st.session_state.model_choice
    
    ready = (
        st.session_state.mentees_df is not None and
        st.session_state.mentors_df is not None and
        api_key
    )
    
    if not ready:
        st.error("❌ Prerequisites not met. Please go back to Tab 6 to configure settings and upload data.")
        st.stop()
    
    # Check if we should start processing (triggered from Tab 6)
    if st.session_state.get('start_processing', False):
        st.session_state.start_processing = False
        st.session_state.last_match_time = datetime.now()
        
        # PRE-FLIGHT VALIDATION: Check for duplicate IDs before sending to AI
        if is_debug_mode():
            st.info("🔍 Pre-flight check: Validating data integrity...")
        
        validation_passed = True
        
        mentee_ids = st.session_state.mentees_df[MENTEE_COLUMNS['id']].dropna()
        mentee_duplicates = mentee_ids[mentee_ids.duplicated(keep=False)]
        if len(mentee_duplicates) > 0:
            validation_passed = False
            unique_dups = mentee_duplicates.unique()
            st.error(f"🚨 DUPLICATE MENTEE IDs: {len(unique_dups)} mentee ID(s) appear multiple times in your data!")
            if is_debug_mode():
                with st.expander(f"❌ Duplicate mentee IDs ({len(unique_dups)})"):
                    for dup_id in sorted(unique_dups):
                        count = (mentee_ids == dup_id).sum()
                        st.write(f"- **{dup_id}** appears {count} times")
            st.error("⚠️ This will cause incorrect results. Please fix your data and reload.")
        
        mentor_ids = st.session_state.mentors_df[MENTOR_COLUMNS['id']].dropna()
        mentor_duplicates = mentor_ids[mentor_ids.duplicated(keep=False)]
        if len(mentor_duplicates) > 0:
            validation_passed = False
            unique_dups = mentor_duplicates.unique()
            st.error(f"🚨 DUPLICATE MENTOR IDs: {len(unique_dups)} mentor ID(s) appear multiple times in your data!")
            if is_debug_mode():
                with st.expander(f"❌ Duplicate mentor IDs ({len(unique_dups)})"):
                    for dup_id in sorted(unique_dups):
                        count = (mentor_ids == dup_id).sum()
                        st.write(f"- **{dup_id}** appears {count} times")
            st.error("⚠️ This will cause incorrect results. Please fix your data and reload.")
        
        if not validation_passed:
            if is_debug_mode():
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
            else:
                st.error("🛑 Cannot proceed: Duplicate IDs found. Please fix your data and reload.")
            st.stop()
        
        # Only show validation success in debug mode
        if is_debug_mode():
            st.success("✅ Validation passed: All IDs are unique!")
        
        # Better progress indicators
        progress_container = st.container()
        
        with progress_container:
            # PHASE 1: Get compatibility matrix from AI (using batching)
            # Progress UI will show this, only show detailed message in debug mode
            if is_debug_mode():
                st.markdown("""
                <div class="info-card">
                    📊 <strong>Phase 1: Generating compatibility matrix with batching...</strong><br>
                    <small>This may take several minutes. Please don't close the browser.</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Get stateful mode setting
            use_stateful = st.session_state.get('use_stateful_mode', False)
            
            # Import API functions from app.py (will be moved to api_client.py later)
            from app import generate_matrix_in_batches, get_reasoning_for_assignments
            
            # Use auto-calculated batch size (will be calculated if not already set)
            batch_size = st.session_state.get('batch_size', None)
            
            matrix_df = generate_matrix_in_batches(
                st.session_state.mentees_df,
                st.session_state.mentors_df,
                st.session_state.training_files,
                api_key,
                model_choice,
                batch_size=batch_size,  # None = auto-calculate
                use_stateful=use_stateful
            )
            
            if matrix_df is not None:
                st.session_state.matrix_df = matrix_df
                if is_debug_mode():
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ <strong>Matrix generated successfully!</strong><br>
                        {len(matrix_df)} mentors × {len(matrix_df.columns)} mentees = {len(matrix_df) * len(matrix_df.columns)} evaluations!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ Matrix generated: {len(matrix_df)} mentors × {len(matrix_df.columns)} mentees")
                
                # PHASE 2: Apply Hungarian algorithm for optimal assignment
                if is_debug_mode():
                    st.markdown("""
                    <div class="info-card">
                        🧮 <strong>Phase 2: Running Hungarian algorithm for optimal assignments...</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("🧮 Finding optimal assignments...")
                
                assignments = hungarian_assignment(matrix_df, max_mentees_per_mentor=2)
                st.session_state.assignments = assignments
                
                # Calculate statistics
                avg_score = np.mean([score for _, _, score in assignments])
                min_score = min([score for _, _, score in assignments])
                max_score = max([score for _, _, score in assignments])
                
                if is_debug_mode():
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ <strong>Optimal assignments found!</strong><br>
                        Avg: {avg_score:.1f}%, Min: {min_score}%, Max: {max_score}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"✅ Optimal assignments found! Average score: {avg_score:.1f}%")
                
                # PHASE 3: Get reasoning for assignments
                if is_debug_mode():
                    st.markdown("""
                    <div class="info-card">
                        🧠 <strong>Phase 3: Getting detailed reasoning for assignments...</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("🧠 Getting detailed reasoning...")
                
                reasoning_dict = get_reasoning_for_assignments(
                    assignments,
                    st.session_state.mentees_df,
                    st.session_state.mentors_df,
                    api_key,
                    model_choice
                )
            
                # Convert to old format for compatibility with display code
                matches = []
                
                # Debug: Show what we're looking for vs what we have (only in debug mode)
                if is_debug_mode() and assignments and reasoning_dict:
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
                st.success(f"✅ Complete! {len(matches)} mentees optimally matched!")
                
                # Validate match quality
                flagged_matches, score_stats = validate_and_flag_matches(
                    st.session_state.matches,
                    st.session_state.mentees_df,
                    st.session_state.mentors_df
                )
                
                # Store validation results for reference
                st.session_state.flagged_matches = flagged_matches
                st.session_state.score_stats = score_stats
                
                st.balloons()
    
    # Show processing summary if processing is complete
    if st.session_state.get('matches') and 'batch_progress' in st.session_state:
        render_processing_summary()
        st.divider()
    
    # Display results
    if st.session_state.matches:
        st.divider()
        st.subheader("📊 Matching Results")
        
        # Display statistics
        if st.session_state.assignments:
            scores = [score for _, _, score in st.session_state.assignments]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Matches", len(st.session_state.assignments))
            with col2:
                st.metric("Avg Score", f"{np.mean(scores):.1f}%")
            with col3:
                st.metric("Min Score", f"{min(scores)}%")
            with col4:
                st.metric("Max Score", f"{max(scores)}%")
        
        # Show score distribution analysis (only in debug mode)
        if is_debug_mode() and st.session_state.matrix_df is not None:
            st.divider()
            score_distribution = analyze_score_distribution(st.session_state.matrix_df)
            st.session_state.score_distribution = score_distribution
            st.divider()
        
        # Show heatmap
        if st.session_state.matrix_df is not None:
            st.subheader("🔥 Compatibility Heatmap")
            st.caption("Blue boxes show optimal assignments selected by Hungarian algorithm")
            
            fig = create_heatmap(st.session_state.matrix_df, st.session_state.assignments)
            # Pass config parameter explicitly to avoid deprecation warning
            # This tells Streamlit/Plotly to use config dict instead of keyword arguments
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                config={
                    'displayModeBar': True,
                    'displaylogo': False
                }
            )
            
            # Option to download matrix
            csv = st.session_state.matrix_df.to_csv()
            st.download_button(
                label="📥 Download Full Matrix (CSV)",
                data=csv,
                file_name=f"compatibility_matrix_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Check for conflicts
        conflicts = check_mentor_conflicts(st.session_state.matches)
        
        if conflicts:
            if is_debug_mode():
                st.markdown("""
                <div class="conflict-warning">
                    <strong>⚠️ Warning:</strong> Unexpected conflicts detected. Please review.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Some conflicts detected. Please review.")
        else:
            st.success("✅ All constraints satisfied: No mentor has >2 mentees!")
        
        # Results display
        st.divider()
        
        # Excel export
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
            
            # Sort matches numerically by mentee_id
            def numeric_sort_key(m):
                mentee_id = str(m['mentee_id'])
                match = re.search(r'\d+', mentee_id)
                if match:
                    return int(match.group())
                return 0
            
            sorted_filtered_matches = sorted(filtered_matches, key=numeric_sort_key)
            
            # Display all match cards
            for match_data in sorted_filtered_matches:
                mentee_id = match_data['mentee_id']
                
                # Get mentee data
                mentee_row = st.session_state.mentees_df[
                    st.session_state.mentees_df[MENTEE_COLUMNS['id']] == mentee_id
                ]
                
                # Safely get country - check if column exists and handle NaN
                mentee_country = None
                if not mentee_row.empty and MENTEE_COLUMNS['country'] in mentee_row.columns:
                    country_val = mentee_row[MENTEE_COLUMNS['country']].iloc[0]
                    if pd.notna(country_val) and str(country_val).strip():
                        # Normalize: trim whitespace and lowercase for consistent matching
                        mentee_country = str(country_val).strip().lower()
                        # Debug: log country value if in debug mode
                        if is_debug_mode():
                            st.caption(f"🔍 Debug: Mentee {mentee_id} country = '{mentee_country}' (original: '{country_val}')")
                
                mentee_field = mentee_row[MENTEE_COLUMNS['field']].iloc[0] if not mentee_row.empty else None
                mentee_flag = get_country_flag(mentee_country)
                
                for match in match_data['matches']:
                    mentor_id = match['mentor_id']
                    quality_class = match['match_quality'].lower()
                    
                    # Get mentor data
                    mentor_row = st.session_state.mentors_df[
                        st.session_state.mentors_df[MENTOR_COLUMNS['id']] == mentor_id
                    ]
                    
                    # Safely get country - check if column exists and handle NaN
                    mentor_country = None
                    if not mentor_row.empty and MENTOR_COLUMNS['country'] in mentor_row.columns:
                        country_val = mentor_row[MENTOR_COLUMNS['country']].iloc[0]
                        if pd.notna(country_val) and str(country_val).strip():
                            # Normalize: trim whitespace and lowercase for consistent matching
                            mentor_country = str(country_val).strip().lower()
                            # Debug: log country value if in debug mode
                            if is_debug_mode():
                                st.caption(f"🔍 Debug: Mentor {mentor_id} country = '{mentor_country}' (original: '{country_val}')")
                    
                    mentor_field = mentor_row[MENTOR_COLUMNS['field']].iloc[0] if not mentor_row.empty else None
                    mentor_flag = get_country_flag(mentor_country)
                    
                    # Build field labels
                    mentee_field_label = f'<span style="background: #F5F5F7; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.85rem; margin-left: 0.5rem;">{mentee_field or "N/A"}</span>' if mentee_field else ''
                    mentor_field_label = f'<span style="background: #F5F5F7; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.85rem; margin-left: 0.5rem;">{mentor_field or "N/A"}</span>' if mentor_field else ''
                    
                    st.markdown(f"""
                    <div class="match-card">
                        <h4>Mentee {mentee_id} {mentee_flag} {mentee_field_label} → Mentor {mentor_id} {mentor_flag} {mentor_field_label}</h4>
                        <p><span class="percentage-badge {quality_class}">{match['match_percentage']}% - {match['match_quality']}</span></p>
                        <p><strong>Reasoning:</strong> {match['reasoning']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Show waiting message if no matches yet
        st.info("👆 Click 'Generate Matches' in Tab 6 to start the matching process.")

