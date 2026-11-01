"""
UI Component for Tab 7: Processing & Results
Handles the matching generation process, results display, and export functionality
"""

import re
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from matching import check_mentor_conflicts, hungarian_assignment
from progress_ui import render_processing_summary
from utils import is_debug_mode, show_debug_info
from validation import analyze_score_distribution, validate_and_flag_matches
from visualization import create_heatmap


def _matches_report_bytes(matches, mentees_df, mentors_df, assignments, reasoning_by_pair=None):
    """Build the Excel report (Matches/Mentees/Mentors/Summary). reasoning_by_pair, when given,
    is {(str mentee_id, str mentor_id): text} and fills the Reasoning column; otherwise it's blank."""
    rows = []
    for md in matches:
        for m in md['matches']:
            reason = ""
            if reasoning_by_pair:
                reason = reasoning_by_pair.get((str(md['mentee_id']), str(m['mentor_id'])), "")
            rows.append({
                'Mentee_ID': md['mentee_id'], 'Rank': m['rank'], 'Mentor_ID': m['mentor_id'],
                'Match_Percentage': m['match_percentage'], 'Match_Quality': m['match_quality'],
                'Reasoning': reason,
            })
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='Matches', index=False)
        mentees_df.to_excel(writer, sheet_name='Mentees', index=False)
        mentors_df.to_excel(writer, sheet_name='Mentors', index=False)
        if assignments:
            pd.DataFrame({
                'Metric': ['Total Matches', 'Average Score', 'Min Score', 'Max Score', 'Generated At'],
                'Value': [
                    len(assignments),
                    f"{np.mean([s for _, _, s in assignments]):.1f}%",
                    f"{min(s for _, _, s in assignments)}%",
                    f"{max(s for _, _, s in assignments)}%",
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                ],
            }).to_excel(writer, sheet_name='Summary', index=False)
    return output.getvalue()


_BANDS = [
    ("Excellent", "excellent", "#2E7D32"),
    ("Strong", "strong", "#66BB6A"),
    ("Good", "good", "#FFB300"),
    ("Fair", "fair", "#FB8C00"),
    ("Poor", "poor", "#E53935"),
]


def _distribution_bar(stats):
    """One slim stacked bar (Excellent→Poor) + legend, replacing the five big metric tiles."""
    counts = [(label, int(stats.get(key, 0)), color) for label, key, color in _BANDS]
    total = sum(c for _, c, _ in counts) or 1
    segs = "".join(
        f'<div title="{label}: {c}" style="width:{100 * c / total:.1f}%;background:{color};"></div>'
        for label, c, color in counts if c
    )
    legend = " &nbsp; ".join(
        f'<span style="color:{color};">■</span> {label} {c}' for label, c, color in counts
    )
    return (
        f'<div style="display:flex;height:14px;border-radius:7px;overflow:hidden;margin:0.35rem 0;">{segs}</div>'
        f'<div style="font-size:0.8rem;color:#777;">{legend}</div>'
    )


def _render_results_header(assignments, stats, flagged):
    """Compact results header: one summary line + a slim distribution bar + the flagged review list."""
    scores = [s for _, _, s in assignments]
    avg = np.mean(scores) if scores else 0
    review = f"⚠️ {len(flagged)} to review" if flagged else "✅ all clear"
    st.markdown(f"#### ✅ {len(assignments)} matched &nbsp;·&nbsp; avg {avg:.0f}% &nbsp;·&nbsp; {review}")
    if stats:
        st.markdown(_distribution_bar(stats), unsafe_allow_html=True)
    if flagged:
        with st.expander(f"🚩 Review {len(flagged)} flagged match(es)"):
            for fm in flagged:
                st.markdown(f"**Mentee {fm['mentee_id']} → Mentor {fm['mentor_id']}** ({fm['score']}%)")
                for issue in fm['issues']:
                    st.caption(f"• {issue}")


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


def _back_to_generate():
    """Enabled escape hatch shown on any dead-end run, so the locked rail can't trap the user."""
    if st.button("← Back to Generate", key="back_to_generate"):
        st.session_state["start_processing"] = False
        st.session_state["step"] = 2  # Generate
        st.rerun()


def render_processing_tab():
    """Render Tab 7: Processing & Results"""
    
    # Use the api_key from session state
    api_key = st.session_state.api_key
    
    ready = (
        st.session_state.mentees_df is not None and
        st.session_state.mentors_df is not None and
        api_key
    )
    
    if not ready:
        st.session_state.start_processing = False  # don't leave the rail locked
        st.error("❌ Prerequisites not met. Please go back to the earlier steps to configure settings and upload data.")
        _back_to_generate()
        st.stop()
    
    # Check if we should start processing (triggered from the Generate step)
    if st.session_state.get('start_processing', False):
        st.session_state.start_processing = False
        st.session_state.last_match_time = datetime.now()
        # Drop any prior results so a failed/empty re-run can't re-display stale matches as if fresh.
        st.session_state.matrix_df = None
        st.session_state.assignments = None
        st.session_state.matches = None
        
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
            _back_to_generate()
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
            
            # Import API functions from app.py (will be moved to api_client.py later)
            from app import generate_matrix_jev

            # Score the compatibility matrix with Jev (System One); assignment + reasoning follow.
            matrix_df = generate_matrix_jev(
                st.session_state.mentees_df,
                st.session_state.mentors_df,
            )
            
            if matrix_df is not None:
                st.session_state.matrix_df = matrix_df
                max_mentees_per_mentor = st.session_state.get('max_mentees_per_mentor', 2)
                result = hungarian_assignment(matrix_df, max_mentees_per_mentor=max_mentees_per_mentor)

                # Render the matrix-cleaning notes the old (st-coupled) solver used to emit.
                cleaning = result.cleaning
                if cleaning['nan_filled']:
                    st.warning(f"⚠️ Found {cleaning['nan_filled']} NaN values in matrix. Replacing with 30% (poor match - AI omitted these)...")
                    st.caption("NaN values indicate the AI skipped these combinations despite instructions. Treating as poor matches.")
                    show_debug_info()
                if cleaning['inf_replaced']:
                    st.warning(f"⚠️ Found {cleaning['inf_replaced']} infinite values in matrix. Replacing with 50%...")
                if cleaning['clipped']:
                    st.warning(f"⚠️ Clipped {cleaning['clipped']} matrix value(s) outside the expected range [0, 100].")
                if result.unmatched_mentees:
                    st.warning(f"⚠️ {len(result.unmatched_mentees)} mentee(s) could not be matched (no mentor capacity left): {', '.join(map(str, result.unmatched_mentees))}")

                assignments = result.assignments
                st.session_state.assignments = assignments

                # Reasoning is fetched on demand (per match, via the "Why this match?" button),
                # not up front — the bulk call was the slowest, priciest step and most of it went unread.
                matches = [
                    {
                        'mentee_id': mentee,
                        'matches': [{
                            'rank': 1,
                            'mentor_id': mentor,
                            'match_percentage': score,
                            'match_quality': (
                                'Excellent' if score >= 90 else 'Strong' if score >= 75
                                else 'Good' if score >= 60 else 'Fair'
                            ),
                            'reasoning': None,
                        }],
                    }
                    for mentee, mentor, score in assignments
                ]
                st.session_state.matches = matches

                # Validate match quality (compute only; the compact header renders the summary).
                flagged_matches, score_stats = validate_and_flag_matches(
                    st.session_state.matches,
                    st.session_state.mentees_df,
                    st.session_state.mentors_df,
                    render=False,
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
        if st.session_state.assignments:
            _render_results_header(
                st.session_state.assignments,
                st.session_state.get('score_stats'),
                st.session_state.get('flagged_matches') or [],
            )
        
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
        max_mentees_per_mentor = st.session_state.get('max_mentees_per_mentor', 2)
        conflicts = check_mentor_conflicts(st.session_state.matches, max_mentees_per_mentor=max_mentees_per_mentor)
        
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
            st.success(f"✅ All constraints satisfied: No mentor has >{max_mentees_per_mentor} mentee(s)!")
        
        # Results display
        st.divider()
        
        # Export — plain is instant; the reasoning version is generated on demand (it costs tokens).
        st.subheader("📥 Export Results")
        stamp = time.strftime('%Y%m%d_%H%M%S')
        xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        col_plain, col_reason = st.columns(2)
        with col_plain:
            st.download_button(
                "📥 Download matches",
                _matches_report_bytes(st.session_state.matches, st.session_state.mentees_df,
                                      st.session_state.mentors_df, st.session_state.assignments),
                f"matches_{stamp}.xlsx", xlsx_mime, width='stretch',
            )
        with col_reason:
            if st.button("🧠 Prepare matches + reasonings", width='stretch',
                         help="Generates a Claude explanation for every match, then offers the download."):
                from app import get_reasoning_for_assignments
                with st.spinner(f"Generating reasonings for {len(st.session_state.assignments)} matches…"):
                    rdict = get_reasoning_for_assignments(
                        st.session_state.assignments, st.session_state.mentees_df, st.session_state.mentors_df)
                    rby = {(str(k[0]), str(k[1])): v for k, v in rdict.items()}
                    st.session_state.export_with_reasoning = _matches_report_bytes(
                        st.session_state.matches, st.session_state.mentees_df,
                        st.session_state.mentors_df, st.session_state.assignments, rby)
            if st.session_state.get("export_with_reasoning"):
                st.download_button(
                    "📥 Download matches + reasonings",
                    st.session_state.export_with_reasoning,
                    f"matches_reasonings_{stamp}.xlsx", xlsx_mime, width='stretch', type="primary",
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
                    
                    with st.container(border=True):
                        st.markdown(
                            f'<h4>Mentee {mentee_id} {mentee_flag} {mentee_field_label} → '
                            f'Mentor {mentor_id} {mentor_flag} {mentor_field_label}</h4>'
                            f'<p><span class="percentage-badge {quality_class}">'
                            f'{match["match_percentage"]}% - {match["match_quality"]}</span></p>',
                            unsafe_allow_html=True,
                        )
                        # Reasoning is fetched on demand and cached, so we only pay for matches opened.
                        cache = st.session_state.setdefault("reasoning_cache", {})
                        ck = (str(mentee_id), str(mentor_id))
                        if st.button("💡 Why this match?", key=f"why_{mentee_id}_{mentor_id}") and ck not in cache:
                            from app import reasoning_for_pair
                            with st.spinner("Asking Claude…"):
                                cache[ck] = reasoning_for_pair(
                                    mentee_id, mentor_id, match['match_percentage'],
                                    st.session_state.mentees_df, st.session_state.mentors_df,
                                )
                        if ck in cache:
                            st.markdown(f"**Reasoning:** {cache[ck]}")
    else:
        # Show waiting message if no matches yet
        st.info("👆 Click 'Generate Matches' in the Generate step to start the matching process.")
        _back_to_generate()

