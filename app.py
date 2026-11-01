import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from info import (
    DEFAULT_MODEL,
    MATRIX_BATCH_SIZE,
    MATRIX_MAX_WORKERS,
    MENTEE_COLUMNS,
    MENTOR_COLUMNS,
)
from llm_client import ClaudeClient, LLMError, parse_json
from matching import apply_country_restrictions, parse_matrix_response
from prompts import create_default_prompt, get_prompt_for_api
from ui import CSS_STYLES
from ui_matches import render_matches_tab
from ui_processing import render_processing_tab, render_review_tab
from ui_upload import render_entity_step
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
    initial_sidebar_state="expanded"
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


MATRIX_MODEL = DEFAULT_MODEL  # the model is fixed in code (no user-facing selector)
REASONING_MODEL = DEFAULT_MODEL
# Opus's adaptive thinking is spent BEFORE the JSON and counts against max_tokens, so
# every budget must leave generous room for it on top of the expected output — otherwise
# a real-size matrix is truncated and the whole batch fails (caught live, not by mocks).
THINKING_HEADROOM = 20000
REASONING_MAX_TOKENS = 8000  # JSON floor; the actual reasoning budget scales with assignments
MATRIX_MAX_OUTPUT_TOKENS = 120000  # headroom under Claude's 128K output cap


def _model_choice(default: str) -> str:
    """The model to use — fixed in code (DEFAULT_MODEL); there is no user-facing selector."""
    return default


def _anthropic_key() -> str:
    """Anthropic API key from the environment, then Streamlit secrets. The only place it is read."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return ""


def _typesafe_key() -> str:
    """TypeSafe/Jev API key from the environment, then Streamlit secrets (TYPESAFE_API_KEY or JEV_API_KEY)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["TYPESAFE_API_KEY"]
    except Exception:
        pass
    try:
        return st.secrets["JEV_API_KEY"]
    except Exception:
        return ""


def generate_matrix_jev(mentees_df, mentors_df):
    """Score the compatibility matrix with Jev (TypeSafe System One), with a live progress bar.

    Returns the mentors×mentees matrix DataFrame, or None (after an st.error) if the key is
    missing or Jev fails — so the caller skips assignment instead of matching on a broken matrix.
    """
    from jev_scoring import JevError, score_matrix_jev

    key = _typesafe_key()
    if not key:
        st.error("❌ No Jev API key. Add TYPESAFE_API_KEY (or JEV_API_KEY) to your secrets.")
        return None

    total = len(mentees_df) * len(mentors_df)
    bar = st.progress(0.0, text=f"Scoring {total} mentee×mentor pairs with Jev…")
    seen = {"frac": 0.0}

    def on_progress(done, total_pairs):
        frac = done / total_pairs
        if frac - seen["frac"] >= 0.02 or done == total_pairs:  # throttle st reruns to ~50 updates
            seen["frac"] = frac
            bar.progress(frac, text=f"Scoring with Jev… {done}/{total_pairs} pairs")

    try:
        matrix = score_matrix_jev(mentees_df, mentors_df, key, progress_cb=on_progress)
    except JevError as e:
        bar.empty()
        st.error(f"❌ Jev scoring failed: {e}")
        return None
    bar.empty()
    return matrix


def _score_mentor_batch(mentees_df, mentor_batch_df, api_key=None, model=None) -> str:
    """Score one mentor batch against all mentees with Claude; return the raw JSON text.

    The system prompt (scoring rubric + every mentee) is byte-identical across batches, so
    prompt caching hits; only the user prompt (this batch's mentors) varies. max_tokens is
    budgeted from the expected output so the JSON isn't truncated. Raises LLMError when the
    response is cut off, so the caller marks the batch an error instead of filling 30s.

    api_key/model are resolved by the caller when this runs on a worker thread (st.secrets and
    st.session_state aren't thread-safe); they default to the normal lookups for direct callers.
    """
    system = f"{get_prompt_for_api()}\n\n{prepare_data_for_assistant(mentees_df, None)}"
    user = (
        f"{prepare_data_for_assistant(None, mentor_batch_df)}\n\n"
        "Generate matches for ALL mentees in JSON format as specified."
    )
    budget = int(estimate_output_tokens(len(mentor_batch_df), len(mentees_df)) * 1.5) + THINKING_HEADROOM
    budget = max(THINKING_HEADROOM, min(budget, MATRIX_MAX_OUTPUT_TOKENS))
    client = ClaudeClient(model or _model_choice(MATRIX_MODEL), api_key or _anthropic_key())
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

    all_mentor_ids = mentors_df[MENTOR_COLUMNS['id']].tolist()
    all_mentee_ids = mentees_df[MENTEE_COLUMNS['id']].tolist()

    if batch_size is None:
        batch_size = MATRIX_BATCH_SIZE

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

    combined_matrix_dict = {}

    # Resolve the key and model once, on the main thread: st.secrets / st.session_state are not
    # thread-safe, so the worker threads below must never touch them.
    api_key = _anthropic_key()
    model = _model_choice(MATRIX_MODEL)

    def _fetch(item):
        """Worker thread: Claude call + pure parse only — NO st.* (Streamlit isn't thread-safe)."""
        batch_num, mentor_batch = item
        mentor_batch_df = mentors_df[mentors_df[MENTOR_COLUMNS['id']].isin(mentor_batch)]
        started = time.time()
        try:
            response = _score_mentor_batch(mentees_df, mentor_batch_df, api_key, model)
        except LLMError as e:
            # A cut-off or failed batch: report it. Do NOT fall through to a 30-fill.
            return batch_num, mentor_batch, None, str(e), time.time() - started
        report = parse_matrix_response(response, mentor_batch, all_mentee_ids)
        return batch_num, mentor_batch, report, None, time.time() - started

    # Every batch is in flight at once — reflect that before any result lands.
    for batch_num in range(1, total_batches + 1):
        update_batch_progress(batch_num, 'processing')
    with progress_placeholder.container():
        render_progress_overview()

    # Score the batches concurrently; render + merge each result on the main thread as it lands.
    with ThreadPoolExecutor(max_workers=min(MATRIX_MAX_WORKERS, total_batches)) as pool:
        futures = [pool.submit(_fetch, (bn, mb)) for bn, mb in enumerate(mentor_batches, 1)]
        for future in as_completed(futures):
            batch_num, mentor_batch, batch_report, err, batch_time = future.result()

            if err is not None:
                update_batch_progress(batch_num, 'error', {'error': err})
                st.error(f"❌ Batch {batch_num} failed: {err}")
                continue

            # _render_parse_report emits the validation messages, so it must run here (main thread).
            batch_matrix_df = _render_parse_report(batch_report, mentor_batch, all_mentee_ids)
            if batch_matrix_df is None:
                update_batch_progress(batch_num, 'error', {'error': 'Failed to parse response'})
                st.error(f"❌ Batch {batch_num} failed to parse")
                continue

            # Mentor ids partition the batches, so no mentor can appear twice.
            for mentor_id in batch_matrix_df.index:
                combined_matrix_dict[mentor_id] = batch_matrix_df.loc[mentor_id].to_dict()

            batch_details = {
                'mentors': len(batch_matrix_df),
                'mentees': len(all_mentee_ids),
                'scores': len(batch_matrix_df) * len(all_mentee_ids),
                'time': batch_time,
                'tokens': None,
                'cost': None,
            }
            update_batch_progress(batch_num, 'complete', batch_details)
            with progress_placeholder.container():
                render_progress_overview()
            with completed_batches_container:
                render_batch_card(batch_num, 'complete', batch_details)

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

    # Organizer toggle (Configure step); default ["spain"] preserves the legacy Spain-Spain rule.
    complete_matrix_df, filtered_matches = apply_country_restrictions(
        complete_matrix_df, mentees_df, mentors_df,
        countries=st.session_state.get("country_restrictions", ["spain"]),
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


def reasoning_for_pair(mentee_id, mentor_id, score, mentees_df, mentors_df):
    """One assignment's reasoning, for on-demand display. Returns the text, or a plain fallback
    if Claude is unavailable. Callers cache it (reasoning is generated only when a match is opened)."""
    d = get_reasoning_for_assignments([(mentee_id, mentor_id, score)], mentees_df, mentors_df, quiet=True)
    texts = list(d.values())
    return texts[0] if texts else (
        f"Selected by the optimizer as the best available match ({score}%). "
        "A detailed explanation isn't available right now."
    )


def get_reasoning_for_assignments(assignments, mentees_df, mentors_df, *, quiet=False):
    """Get a short reasoning per assignment from Claude; {} on any failure (non-critical).

    quiet suppresses the success/failure toasts (used for single-pair, on-demand lookups).
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
        if not quiet:
            st.success(f"✅ Got reasoning for {len(reasoning_dict)} assignments")
        return reasoning_dict

    except Exception as e:
        if not quiet:
            st.error("❌ Error getting reasoning")
            if is_debug_mode():
                st.error(f"❌ Error getting reasoning: {e!s}")
            st.info("💡 Continuing without detailed reasoning...")
        return {}


# --------------------------------------------------------------------------- navigation + rail

_STEPS = ["Mentees", "Mentors", "Generate", "Results", "Review"]


def _reachable(has_data: bool, has_results: bool) -> set[int]:
    # Both entity steps are always reachable (upload in any order); `has_data` means BOTH files.
    s = {0, 1}  # Mentees, Mentors
    if has_data:
        s.add(2)  # Generate
    if has_results:
        s.update({3, 4})  # Results (overview), Review (matches list)
    return s


def _resolve_step(has_data: bool, has_results: bool) -> int:
    """Stored step clamped into what's reachable (e.g. removing a file clears results)."""
    step = st.session_state.get("step", 0)
    if step not in _reachable(has_data, has_results):
        step = 2 if has_data else 0
        st.session_state["step"] = step
    return step


def _nav_to(step: int) -> None:
    st.session_state["step"] = step


def render_rail(current: int, has_data: bool, has_results: bool, clickable: bool = True) -> None:
    """Step rail as real buttons: they rerun in place, unlike anchors which reset the session."""
    reachable = _reachable(has_data, has_results) if clickable else set()
    cols = st.columns(len(_STEPS))
    for i, label in enumerate(_STEPS):
        enabled = i in reachable
        cols[i].button(
            f"{'✓' if i < current else i + 1}  {label}",
            key=f"rail_{i}",
            type="primary" if i == current else "secondary",
            use_container_width=True,
            disabled=not enabled,
            on_click=_nav_to if enabled else None,
            args=(i,) if enabled else None,
        )


def render_sidebar() -> None:
    """Settings live here, not as wizard steps: the few knobs an organizer touches. Scoring runs
    on Jev (fixed rubric), so there is no model or rubric control."""
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.max_mentees_per_mentor = st.number_input(
            "Max mentees per mentor",
            min_value=1,
            max_value=3,
            value=st.session_state.get("max_mentees_per_mentor", 2),
            step=1,
            help="The most mentees a single mentor can be assigned. Enforced during optimal assignment.",
        )
        restrict_spain = st.checkbox(
            "Spain ↔ Spain restriction",
            value="spain" in st.session_state.get("country_restrictions", ["spain"]),
            help="Don't match Spanish mentors to Spanish mentees. Needs a country column in the data.",
        )
        st.session_state.country_restrictions = ["spain"] if restrict_spain else []


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
        st.session_state.batch_size = MATRIX_BATCH_SIZE
    if 'max_mentees_per_mentor' not in st.session_state:
        st.session_state.max_mentees_per_mentor = 2
    if 'country_restrictions' not in st.session_state:
        st.session_state.country_restrictions = ["spain"]
    if 'mentee_search' not in st.session_state:
        st.session_state.mentee_search = ""
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    
    # Initialize debug mode from environment variable
    if 'debug_mode' not in st.session_state:
        import os
        debug_env = os.getenv('DEBUG_MODE', '').upper().strip()
        if debug_env == 'TRUE':
            st.session_state.debug_mode = True
        else:
            # Default: OFF (production). Set DEBUG_MODE=TRUE to expose token/cost internals.
            st.session_state.debug_mode = False
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = _anthropic_key()
    
    if not st.session_state.api_key:
        st.error(
            "❌ **Missing API key.** Set `ANTHROPIC_API_KEY` (environment variable or Streamlit "
            "secrets) to use the app."
        )
        st.stop()

    render_sidebar()

    has_data = (
        st.session_state.get("mentees_df") is not None
        and st.session_state.get("mentors_df") is not None
    )
    has_results = st.session_state.get("matches") is not None
    if st.session_state.get("start_processing"):
        render_rail(3, has_data, has_results, clickable=False)  # locked: a stray click can't derail the run
        render_processing_tab()  # runs the pipeline this rerun, then clears start_processing
        return
    step = _resolve_step(has_data, has_results)
    render_rail(step, has_data, has_results)

    if step == 0:
        render_entity_step("mentee")
    elif step == 1:
        render_entity_step("mentor")
    elif step == 2:
        render_matches_tab()
    elif step == 3:
        render_processing_tab()  # Results: overview + heatmap
    else:
        render_review_tab()  # Review: the matches list you act on


if __name__ == "__main__":
    main()
