import os
import time

import pandas as pd
import streamlit as st

from info import DEFAULT_MODEL, MENTEE_COLUMNS, MENTOR_COLUMNS
from llm_client import ClaudeClient, LLMError, parse_json
from matching import apply_country_restrictions, parse_matrix_response
from prompts import create_default_prompt, get_prompt_for_api
from ui import CSS_STYLES
from ui_config import render_config_tab
from ui_matches import render_matches_tab
from ui_overview import render_overview_tab
from ui_processing import render_processing_tab
from ui_prompt import render_prompt_tab
from ui_upload import render_upload_tab
from utils import (
    estimate_output_tokens,
    is_debug_mode,
    show_debug_info,
)
from validation import prepare_data_for_assistant

st.set_page_config(
    page_title="Mentorship Matching System",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)


def _render_parse_report(report, expected_mentors, expected_mentees):
    """Render a MatrixParseReport with the same st.* messages the old parser emitted.

    Returns the parsed DataFrame, or None when the batch is unusable, so callers keep the
    old `parse_matrix_response` contract (df-or-None). Debug gating matches the original.
    """
    # Fatal errors the old parser returned on immediately, before any validation output.
    if report.df is None and report.error and report.error != "No valid mentor data after filtering":
        if report.error == "Received a configuration object instead of JSON content":
            st.error("❌ **Invalid response: Received configuration object instead of JSON content**")
            st.warning("""
            The model returned a configuration object instead of the match scores.

            **Solution:** Re-run the match, or pick a different Claude model in the sidebar.
            """)
        elif report.error == "Could not find JSON object boundaries in response":
            st.error("❌ Could not find JSON object boundaries in response")
            with st.expander("View raw response"):
                st.text(report.raw_excerpt)
        else:
            st.error(f"❌ {report.error}")
            show_debug_info()
            with st.expander("View raw response"):
                st.text(report.raw_excerpt)
        return None

    # Non-fatal: the model wrapped the JSON in extra prose.
    if report.notes:
        st.warning("⚠️ AI included explanatory text instead of pure JSON. Attempting to extract JSON...")
        for note in report.notes:
            with st.expander("📝 Text around JSON (should be empty)"):
                st.text(note)

    # Wrong IDs the AI returned (one warning per mentor, then the summaries).
    for mentor_id in report.wrong_mentors:
        st.warning(f"⚠️ AI returned WRONG mentor ID: '{mentor_id}' (not in requested batch)")
    if report.wrong_mentors:
        st.error(f"🚨 WRONG MENTOR IDs: AI returned {len(report.wrong_mentors)} mentor(s) NOT in requested batch!")
        with st.expander(f"❌ Wrong mentor IDs ({len(report.wrong_mentors)})"):
            for m in sorted(report.wrong_mentors):
                st.write(f"- {m} (REJECTED - not in batch request)")

    if report.wrong_mentees:
        st.error(f"🚨 WRONG MENTEE IDs: AI returned {len(report.wrong_mentees)} invalid mentee(s)!")
        with st.expander(f"❌ Wrong mentee IDs ({len(report.wrong_mentees)})"):
            for m in sorted(report.wrong_mentees)[:10]:
                st.write(f"- {m}")

    mentors_received = len(expected_mentors) - len(report.missing_mentors)
    mentees_received = len(expected_mentees) - len(report.missing_mentees_filled)
    if is_debug_mode():
        st.caption(
            f"📊 Valid data: {mentors_received}/{len(expected_mentors)} mentors, "
            f"{mentees_received}/{len(expected_mentees)} mentees"
        )

    expected_total = len(expected_mentors) * len(expected_mentees)
    actual_total = round(report.completion_rate / 100 * expected_total)
    if report.completion_rate < 100:
        st.error(
            f"🚨 INCOMPLETE RESPONSE: Expected {expected_total} scores, got {actual_total} "
            f"({report.completion_rate:.1f}% complete)"
        )
        st.error(f"   Missing {expected_total - actual_total} combinations!")
        st.error("   ⚠️ The AI is still omitting combinations despite instructions.")
    elif is_debug_mode():
        st.caption(f"✅ COMPLETE: All {expected_total} scores received")

    if report.missing_mentors:
        st.warning(f"⚠️ Missing {len(report.missing_mentors)} requested mentor(s)")
        with st.expander(f"📋 Missing mentors ({len(report.missing_mentors)})"):
            for m in sorted(report.missing_mentors):
                st.write(f"- {m}")

    if report.missing_mentees_filled:
        st.warning(f"⚠️ Missing {len(report.missing_mentees_filled)} mentee(s) in responses")
        with st.expander(f"📋 Missing mentees ({len(report.missing_mentees_filled)})"):
            for m in sorted(report.missing_mentees_filled):
                st.write(f"- {m}")

    if report.df is None:
        st.error("❌ No valid mentor data after filtering!")
        return None

    if report.missing_mentees_filled:
        st.warning(
            f"⚠️ {len(report.missing_mentees_filled)} mentee column(s) missing from matrix. "
            "Filling with 30% (assumed poor match)."
        )
        st.caption("The AI omitted these combinations despite instructions. This indicates likely poor matches.")
        show_debug_info()

    return report.df


MATRIX_MODEL = DEFAULT_MODEL  # defaults; the UI's st.session_state.model_choice overrides (see _model_choice)
REASONING_MODEL = DEFAULT_MODEL
# Opus's adaptive thinking is spent BEFORE the JSON and counts against max_tokens, so
# every budget must leave generous room for it on top of the expected output — otherwise
# a real-size matrix is truncated and the whole batch fails (caught live, not by mocks).
THINKING_HEADROOM = 20000
REASONING_MAX_TOKENS = 8000  # JSON floor; the actual reasoning budget scales with assignments
MATRIX_MAX_OUTPUT_TOKENS = 120000  # headroom under Claude's 128K output cap


def _model_choice(default: str) -> str:
    """The user's selected Claude model (Configure tab), else the given default."""
    return st.session_state.get("model_choice", default)


def _anthropic_key() -> str:
    """Anthropic API key from the environment, then Streamlit secrets. The only place it is read."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return ""


def _score_mentor_batch(mentees_df, mentor_batch_df) -> str:
    """Score one mentor batch against all mentees with Claude; return the raw JSON text.

    The system prompt (scoring rubric + every mentee) is byte-identical across batches, so
    prompt caching hits; only the user prompt (this batch's mentors) varies. max_tokens is
    budgeted from the expected output so the JSON isn't truncated. Raises LLMError when the
    response is cut off, so the caller marks the batch an error instead of filling 30s.
    """
    system = f"{get_prompt_for_api()}\n\n{prepare_data_for_assistant(mentees_df, None)}"
    user = (
        f"{prepare_data_for_assistant(None, mentor_batch_df)}\n\n"
        "Generate matches for ALL mentees in JSON format as specified."
    )
    budget = int(estimate_output_tokens(len(mentor_batch_df), len(mentees_df)) * 1.5) + THINKING_HEADROOM
    budget = max(THINKING_HEADROOM, min(budget, MATRIX_MAX_OUTPUT_TOKENS))
    client = ClaudeClient(_model_choice(MATRIX_MODEL), _anthropic_key())
    return client.complete_json(system, user, max_tokens=budget, cache_system=True)


def generate_matrix_in_batches(mentees_df, mentors_df, batch_size=None):
    """Generate the compatibility matrix one mentor batch at a time (Claude + prompt caching).

    Args:
        batch_size: Mentors per batch. If None, sized so each batch's estimated output stays
                    under the 128K output cap.

    The model is st.session_state.model_choice (default MATRIX_MODEL); the key comes from _anthropic_key().

    Returns:
        Complete matrix DataFrame, or None if no batch succeeded.
    """
    from progress_ui import (
        initialize_progress_tracking,
        render_batch_card,
        render_progress_overview,
        update_batch_progress,
    )
    from utils import calculate_optimal_batch_size

    all_mentor_ids = mentors_df[MENTOR_COLUMNS['id']].tolist()
    all_mentee_ids = mentees_df[MENTEE_COLUMNS['id']].tolist()
    total_mentors = len(all_mentor_ids)
    total_mentees = len(all_mentee_ids)

    # Auto-calculate batch size if not provided
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(total_mentors, total_mentees)
        if is_debug_mode():
            total_batches = int((total_mentors + batch_size - 1) / batch_size) if batch_size else 0
            st.info(f"📊 **Auto-calculated batch size**: {batch_size} mentors per batch")
            st.caption(
                f"Sized by estimated output tokens (cap under Claude's 128K). "
                f"Total batches: {total_batches}"
            )

    mentor_batches = [all_mentor_ids[i:i + batch_size] for i in range(0, len(all_mentor_ids), batch_size)]
    total_batches = len(mentor_batches)

    # Initialize progress tracking
    initialize_progress_tracking(total_batches)

    # Initialize all batches as pending
    for batch_num in range(1, total_batches + 1):
        update_batch_progress(batch_num, 'pending')

    if is_debug_mode():
        st.info(f"📦 Processing in {total_batches} batches ({batch_size} mentors per batch)")

    # Create empty container for progress UI (to update in place)
    progress_placeholder = st.empty()

    # Create a container for completed batch cards (they accumulate)
    completed_batches_container = st.container()

    # Create empty placeholder for currently processing batch
    current_batch_placeholder = st.empty()

    # Render initial progress overview
    with progress_placeholder.container():
        render_progress_overview()

    combined_matrix_dict = {}

    for batch_num, mentor_batch in enumerate(mentor_batches, 1):
        batch_start_time = time.time()

        # Update progress: mark as processing
        update_batch_progress(batch_num, 'processing')

        # Update progress overview in placeholder (replaces previous content)
        with progress_placeholder.container():
            render_progress_overview()

        # Show current batch status in placeholder (replaces previous)
        with current_batch_placeholder.container():
            render_batch_card(batch_num, 'processing', is_current=True)

        # Batch status is shown in progress UI, suppress these messages
        # Only show detailed info in debug mode
        if is_debug_mode():
            st.caption(f"🔄 Batch {batch_num}: {len(mentor_batch)} mentors ({mentor_batch[0]} to {mentor_batch[-1]})")

        mentor_batch_df = mentors_df[mentors_df[MENTOR_COLUMNS['id']].isin(mentor_batch)]

        try:
            response = _score_mentor_batch(mentees_df, mentor_batch_df)
        except LLMError as e:
            # A cut-off or failed batch: mark it an error. Do NOT fall through to a 30-fill.
            update_batch_progress(batch_num, 'error', {'error': str(e)})
            st.error(f"❌ Batch {batch_num} failed: {e!s}")
            continue

        batch_time = time.time() - batch_start_time

        batch_report = parse_matrix_response(response, mentor_batch, all_mentee_ids)
        batch_matrix_df = _render_parse_report(batch_report, mentor_batch, all_mentee_ids)

        if batch_matrix_df is None:
            update_batch_progress(batch_num, 'error', {'error': 'Failed to parse response'})
            st.error(f"❌ Batch {batch_num} failed to parse")
            continue

        received_mentors = list(batch_matrix_df.index)
        expected_mentors_set = {str(m) for m in mentor_batch}
        received_mentors_set = set(received_mentors)

        overwrites = []

        for mentor_id in batch_matrix_df.index:
            if mentor_id in combined_matrix_dict:
                overwrites.append(mentor_id)
                if is_debug_mode():
                    st.caption(f"⚠️ DUPLICATE: '{mentor_id}' already exists!")

            combined_matrix_dict[mentor_id] = batch_matrix_df.loc[mentor_id].to_dict()

        # Prepare batch details (token/cost telemetry dropped)
        batch_details = {
            'mentors': len(batch_matrix_df),
            'mentees': len(all_mentee_ids),
            'scores': len(batch_matrix_df) * len(all_mentee_ids),
            'time': batch_time,
            'tokens': None,
            'cost': None,
        }

        # Update progress: mark as complete
        update_batch_progress(batch_num, 'complete', batch_details)

        # Update progress overview in placeholder (replaces previous content)
        with progress_placeholder.container():
            render_progress_overview()

        # Clear current batch placeholder and move completed batch to container
        current_batch_placeholder.empty()

        # Render completed batch card in the completed batches container
        with completed_batches_container:
            render_batch_card(batch_num, 'complete', batch_details)

        if is_debug_mode():
            if overwrites:
                st.caption(f"⚠️ {len(overwrites)} duplicate(s) detected")
            if received_mentors_set == expected_mentors_set:
                st.caption(f"✅ Batch {batch_num}: Received all {len(batch_matrix_df)}/{len(mentor_batch)} mentors")
            else:
                missing_in_batch = expected_mentors_set - received_mentors_set
                st.caption(f"⚠️ Batch {batch_num}: Missing {len(missing_in_batch)} mentor(s)")

    if not combined_matrix_dict:
        st.error("❌ No batches succeeded")
        return None

    st.divider()
    if is_debug_mode():
        st.info("📊 **Combining all batches into final matrix...**")
    
    complete_matrix_df = pd.DataFrame(combined_matrix_dict).T
    complete_matrix_df = complete_matrix_df[[str(m) for m in sorted(all_mentee_ids)]]
    
    # Apply country restrictions (Spain-Spain rule)
    st.divider()
    st.subheader("🌍 Applying Country Restrictions")
    st.info("🔍 Checking for country-based restrictions (Spain-Spain rule)...")
    mentor_country_col = MENTOR_COLUMNS['country']
    mentee_country_col = MENTEE_COLUMNS['country']
    if mentor_country_col in mentors_df.columns and mentee_country_col in mentees_df.columns:
        spain_mentors = (mentors_df[mentor_country_col].astype(str).str.strip().str.lower() == 'spain').sum()
        spain_mentees = (mentees_df[mentee_country_col].astype(str).str.strip().str.lower() == 'spain').sum()
        if spain_mentors and spain_mentees:
            st.info(f"🇪🇸 Found {spain_mentors} Spanish mentor(s) and {spain_mentees} Spanish mentee(s)")

    # countries=["spain"] preserves the legacy hardcoded Spain-Spain restriction (see Task 6).
    complete_matrix_df, filtered_matches = apply_country_restrictions(
        complete_matrix_df, mentees_df, mentors_df, countries=["spain"]
    )

    # A missing country column yields a skip-marker; render it like the old warning, no filtering.
    if filtered_matches and "skipped" in filtered_matches[0]:
        st.warning(f"⚠️ {filtered_matches[0]['skipped']}. Skipping country restrictions.")
        filtered_matches = []

    # Display filtered matches information
    if filtered_matches:
        st.warning(f"🚫 **Country Restriction Applied**: {len(filtered_matches)} match(es) filtered")
        with st.expander(f"📋 Filtered Matches Details ({len(filtered_matches)})"):
            st.markdown("**Rule**: Spanish mentors cannot mentor Spanish mentees")
            st.markdown("**Action**: Scores set to 0 (invalid match)")
            st.markdown("---")
            
            # Sort by original score (highest first) to show most impactful filters
            sorted_matches = sorted(filtered_matches, key=lambda x: x['original_score'], reverse=True)
            
            # Display as table for better readability
            matches_data = []
            for match in sorted_matches:
                matches_data.append({
                    'Mentor ID': match['mentor_id'],
                    'Mentee ID': match['mentee_id'],
                    'Original Score': f"{match['original_score']:.1f}%",
                    'Reason': 'Spain-Spain restriction'
                })
            
            if matches_data:
                matches_df = pd.DataFrame(matches_data)
                st.dataframe(matches_df, width='stretch', hide_index=True)
                
                # Summary statistics
                avg_score = sum(m['original_score'] for m in sorted_matches) / len(sorted_matches)
                max_score = sorted_matches[0]['original_score']
                st.caption(f"📊 Average filtered score: {avg_score:.1f}% | Highest filtered: {max_score:.1f}%")
    else:
        st.success("✅ No matches filtered by country restrictions")
    
    mentors_received = len(complete_matrix_df)
    expected_count = len(all_mentor_ids)
    mentors_in_dict = set(complete_matrix_df.index)
    mentors_requested = {str(m) for m in all_mentor_ids}
    
    missing_from_dict = mentors_requested - mentors_in_dict
    extra_in_dict = mentors_in_dict - mentors_requested
    
    st.success(f"✅ Final matrix: {mentors_received}/{expected_count} mentors")
    
    if is_debug_mode():
        with st.expander("🔍 Final Matrix Validation Details"):
            st.write(f"**Expected mentors**: {expected_count} (from original data)")
            st.write(f"**Mentors in dictionary**: {mentors_received} (unique keys)")
            st.write(f"**Dictionary size**: {len(combined_matrix_dict)} entries")
            
            if missing_from_dict:
                st.error(f"❌ **Missing {len(missing_from_dict)} mentors** that were requested but not in dict:")
                for m in sorted(missing_from_dict)[:20]:
                    st.text(f"  - {m}")
            else:
                st.success("✅ All requested mentors are in the dictionary!")
            
            if extra_in_dict:
                st.warning(f"⚠️ **Extra {len(extra_in_dict)} mentors** in dict that weren't requested:")
                for m in sorted(extra_in_dict)[:20]:
                    st.text(f"  - {m}")
    
    if mentors_received < expected_count:
        st.warning(f"⚠️ **DISCREPANCY**: Expected {expected_count} but only have {mentors_received} in final matrix")
    
    if mentors_received < expected_count:
        missing = {str(m) for m in all_mentor_ids} - set(complete_matrix_df.index)
        st.warning(f"⚠️ Missing {len(missing)} mentors after initial batches")
        
        with st.expander(f"📋 Missing mentors ({len(missing)})"):
            for m in sorted(missing):
                st.write(f"- {m}")
        
        max_retries = 3
        for retry_num in range(1, max_retries + 1):
            # Recalculate what's still missing (convert to strings for comparison)
            current_missing = {str(m) for m in all_mentor_ids} - set(complete_matrix_df.index)
            
            if not current_missing:
                st.success("🎉 All mentors retrieved!")
                break
            
            st.info(f"🔄 Retry {retry_num}/{max_retries}: Attempting {len(current_missing)} missing mentors...")
            
            try:
                retry_batch_df = mentors_df[
                    mentors_df[MENTOR_COLUMNS['id']].astype(str).isin(current_missing)
                ]
                try:
                    retry_response = _score_mentor_batch(mentees_df, retry_batch_df)
                except LLMError as e:
                    st.error(f"❌ Retry {retry_num} failed to get response: {e!s}")
                    if retry_num < max_retries:
                        st.info("⏳ Waiting 5 seconds before next retry...")
                        time.sleep(5)
                    continue

                retry_report = parse_matrix_response(retry_response, list(current_missing), all_mentee_ids)
                retry_matrix_df = _render_parse_report(retry_report, list(current_missing), all_mentee_ids)

                if retry_matrix_df is not None and len(retry_matrix_df) > 0:
                    added_count = 0
                    for mentor_id in retry_matrix_df.index:
                        combined_matrix_dict[mentor_id] = retry_matrix_df.loc[mentor_id].to_dict()
                        added_count += 1
                    
                    st.success(f"✅ Retry {retry_num} successful: Added {added_count} mentors")
                    
                    # Rebuild complete matrix from updated dictionary
                    complete_matrix_df = pd.DataFrame(combined_matrix_dict).T
                    complete_matrix_df = complete_matrix_df[[str(m) for m in sorted(all_mentee_ids)]]
                    
                    mentors_received = len(complete_matrix_df)
                    st.info(f"📈 Matrix now has {mentors_received}/{expected_count} mentors")
                    
                    if mentors_received == expected_count:
                        st.success(f"🎉 NOW COMPLETE: {mentors_received}/{expected_count} mentors!")
                        break
                else:
                    st.error(f"❌ Retry {retry_num} failed to parse")
                    if retry_num < max_retries:
                        st.info("⏳ Waiting 5 seconds before next retry...")
                        time.sleep(5)
                        
            except Exception as e:
                st.error(f"❌ Retry {retry_num} error: {e!s}")
                if retry_num < max_retries:
                    st.info("⏳ Waiting 5 seconds before next retry...")
                    time.sleep(5)
        
        final_missing = {str(m) for m in all_mentor_ids} - set(complete_matrix_df.index)
        if final_missing:
            st.warning(f"⚠️ After {max_retries} retries, still missing {len(final_missing)} mentors")
            with st.expander(f"📋 Permanently missing mentors ({len(final_missing)})"):
                for m in sorted(final_missing):
                    st.write(f"- {m}")
            st.info(f"💡 Continuing with {len(complete_matrix_df)}/{expected_count} mentors. Missing mentors will be excluded from matching.")
    
    return complete_matrix_df


def get_reasoning_for_assignments(assignments, mentees_df, mentors_df):
    """Get a short reasoning per assignment from Claude; {} on any failure (non-critical).

    The model is st.session_state.model_choice (default REASONING_MODEL); the key comes from _anthropic_key().
    """
    try:
        
        if not assignments or len(assignments) == 0:
            st.warning("⚠️ No assignments to get reasoning for")
            return {}
        
        if mentees_df is None or mentors_df is None:
            st.error("❌ Missing mentee or mentor data")
            return {}
        
        assignment_details = []
        for mentee_id, mentor_id, score in assignments:
            try:
                mentee_id_int = int(mentee_id)
                mentor_id_int = int(mentor_id)
                mentee_row = mentees_df[mentees_df[MENTEE_COLUMNS['id']] == mentee_id_int].iloc[0]
                mentor_row = mentors_df[mentors_df[MENTOR_COLUMNS['id']] == mentor_id_int].iloc[0]
                
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
                    'mentee_other_info': safe_str(mentee_row.get(MENTEE_COLUMNS['other_info'], '')),
                    'mentor_field': safe_str(mentor_row.get(MENTOR_COLUMNS['field'], '')),
                    'mentor_specialization': safe_str(mentor_row.get(MENTOR_COLUMNS['specialization'], '')),
                    'mentor_position': safe_str(mentor_row.get(MENTOR_COLUMNS['current_position'], '')),
                    'mentor_experience': safe_str(mentor_row.get(MENTOR_COLUMNS['experience_years'], ''))
                })
            except Exception as e:
                if is_debug_mode():
                    st.warning(f"⚠️ Could not extract details for {mentee_id} → {mentor_id}: {e!s}")
                assignment_details.append({
                    'mentee_id': str(mentee_id),
                    'mentor_id': str(mentor_id),
                    'score': int(score) if isinstance(score, (int, float)) else 0,
                    'mentee_field': '',
                    'mentee_specialization': '',
                    'mentee_guidance_areas': '',
                    'mentee_goals': '',
                    'mentee_other_info': '',
                    'mentor_field': '',
                    'mentor_specialization': '',
                    'mentor_position': '',
                    'mentor_experience': ''
                })
        
        if not assignment_details:
            st.error("❌ No valid assignment details extracted")
            return {}
        
        if is_debug_mode():
            st.info(f"📝 Extracted details for {len(assignment_details)} assignments")
        
        system = (
            "You are an expert at explaining mentorship compatibility. For each assignment, "
            "explain in 2-3 sentences why it is a good match based on field/specialization "
            "alignment, skills compatibility and career goals support.\n\n"
            'Return a JSON object: {"reasonings": [{"mentee_id": "EXACT_CODE", '
            '"mentor_id": "EXACT_CODE", "reasoning": "2-3 sentence explanation"}]} '
            "with one entry per assignment, using the IDs exactly as given."
        )
        user = f"ASSIGNMENTS ({len(assignment_details)}):\n"
        for ad in assignment_details:
            user += f"""
{ad['mentee_id']} → {ad['mentor_id']} (Score: {ad['score']}%)
- Mentee field: {ad['mentee_field']}
- Mentee specialization: {ad['mentee_specialization'][:100]}...
- Mentee guidance needs: {ad['mentee_guidance_areas'][:100]}...
- Mentee goals: {ad['mentee_goals'][:100]}...
- Mentee other info: {ad['mentee_other_info'][:100]}...
- Mentor field: {ad['mentor_field']}
- Mentor position: {ad['mentor_position'][:80]}...
- Mentor specialization: {ad['mentor_specialization'][:100]}...
- Mentor experience: {ad['mentor_experience']} years
"""

        reasoning_budget = min(
            MATRIX_MAX_OUTPUT_TOKENS,
            max(REASONING_MAX_TOKENS, len(assignments) * 150) + THINKING_HEADROOM,
        )
        text = ClaudeClient(_model_choice(REASONING_MODEL), _anthropic_key()).complete_json(
            system, user, max_tokens=reasoning_budget
        )
        data = parse_json(text)
        entries = data.get("reasonings") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise LLMError("Reasoning response had no 'reasonings' list.")
        reasoning_dict = {
            (r["mentee_id"], r["mentor_id"]): r["reasoning"]
            for r in entries
            if isinstance(r, dict) and {"mentee_id", "mentor_id", "reasoning"} <= r.keys()
        }
        st.success(f"✅ Got reasoning for {len(reasoning_dict)} assignments")
        return reasoning_dict

    except Exception as e:
        st.error("❌ Error getting reasoning")
        if is_debug_mode():
            st.error(f"❌ Error getting reasoning: {e!s}")
        st.info("💡 Continuing without detailed reasoning...")
        return {}


def main():
    # Password authentication check - must be first thing
    import os
    
    # Initialize authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'password_attempts' not in st.session_state:
        st.session_state.password_attempts = 0
    if 'password_locked' not in st.session_state:
        st.session_state.password_locked = False
    
    # Get password from environment variable
    required_password = os.getenv('APP_ACCESS_PASSWORD', '')
    
    # If no password is set in env, skip authentication (for development)
    if not required_password:
        st.session_state.authenticated = True
    # If already authenticated, continue
    elif st.session_state.authenticated:
        pass  # Continue to app
    # If locked due to too many attempts
    elif st.session_state.password_locked:
        st.error("🔒 **Access Locked**: Too many failed password attempts. Please refresh the page to try again.")
        st.stop()
    # Show password form
    else:
        # Centered container with all elements together
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h2>🔐 Access Required</h2>
                <p>Please enter the access password to continue.</p>
            </div>
            """, unsafe_allow_html=True)
            
            password_input = st.text_input(
                "Access Password",
                type="password",
                key="password_input",
                label_visibility="visible"
            )
            
            if st.button("🔓 Unlock", type="primary", use_container_width=True):
                if password_input == required_password:
                    st.session_state.authenticated = True
                    st.session_state.password_attempts = 0
                    st.rerun()
                else:
                    st.session_state.password_attempts += 1
                    remaining_attempts = 5 - st.session_state.password_attempts
                    
                    if st.session_state.password_attempts >= 5:
                        st.session_state.password_locked = True
                        st.error("🔒 **Access Locked**: Too many failed attempts. Please refresh the page.")
                        st.stop()
                    else:
                        st.error(f"❌ Incorrect password. {remaining_attempts} attempt(s) remaining.")
            
            if st.session_state.password_attempts > 0:
                st.warning(f"⚠️ Failed attempts: {st.session_state.password_attempts}/5")
        
        st.stop()  # Prevent app from loading until authenticated
    
    # Header with animation
    st.markdown("""
    <div class="main-header">
        <h1>🤝 Mentorship Matching System</h1>
        <p>AI-powered mentor-mentee matching</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
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
    if 'max_mentees_per_mentor' not in st.session_state:
        st.session_state.max_mentees_per_mentor = 2
    if 'mentee_search' not in st.session_state:
        st.session_state.mentee_search = ""
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = DEFAULT_MODEL
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    
    # Initialize debug mode from environment variable
    if 'debug_mode' not in st.session_state:
        import os
        debug_env = os.getenv('DEBUG_MODE', '').upper().strip()
        if debug_env == 'FALSE':
            st.session_state.debug_mode = False
        elif debug_env == 'TRUE':
            st.session_state.debug_mode = True
        else:
            # Default: ON if env var not present or invalid
            st.session_state.debug_mode = True
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = _anthropic_key()
    
    tab_names = [
        "1️⃣ Configure Settings",
        "2️⃣ Upload Data Files",
        "3️⃣ Customize Prompt",
        "4️⃣ Data Overview",
        "5️⃣ Generate Matches",
        "6️⃣ Processing & Results"
    ]
    
    # WORKAROUND: Wrap tabs in columns to fix rendering issue (known Streamlit bug)
    # This fixes the issue where tabs after position 4 don't render
    cols = st.columns([0.999, 0.001])
    with cols[0]:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)
        
        # TAB 1: Configure Settings
        with tab1:
            render_config_tab()
        
        # TAB 2: Upload Data Files
        with tab2:
            render_upload_tab()
        
        # TAB 3: Customize Prompt
        with tab3:
            render_prompt_tab()
        
        # TAB 4: Data Overview
        with tab4:
            render_overview_tab()
        
        # TAB 5: Generate Matches
        with tab5:
            render_matches_tab()
        
        # TAB 6: Processing & Results
        with tab6:
            render_processing_tab()

if __name__ == "__main__":
    main()
