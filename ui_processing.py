"""
UI Component for Tab 7: Processing & Results
Handles the matching generation process, results display, and export functionality
"""

import html
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
        f'<div style="display:flex;height:14px;border-radius:7px;overflow:hidden;margin:0.5rem 0 0.8rem;">{segs}</div>'
        f'<div style="font-size:0.8rem;color:#777;margin-bottom:0.4rem;">{legend}</div>'
    )


def _render_results_header(assignments, stats):
    """Compact results header: one summary line + a slim distribution bar. Flagged matches are
    acted on in the Review step (via its 'Needs review only' filter), not a read-only list here."""
    scores = [s for _, _, s in assignments]
    avg = np.mean(scores) if scores else 0
    flagged_n = len(st.session_state.get('flagged_matches') or [])
    review = f"⚠️ {flagged_n} to review" if flagged_n else "✅ all clear"
    st.markdown(f"#### ✅ {len(assignments)} matched &nbsp;·&nbsp; avg {avg:.0f}% &nbsp;·&nbsp; {review}")
    if stats:
        st.markdown(_distribution_bar(stats), unsafe_allow_html=True)


def get_country_flag(country_name):
    """Country name -> flag emoji, or "" when unknown (caller shows the name, not a globe)."""
    if not country_name or pd.isna(country_name):
        return ""
    
    # Normalize country name: strip, lowercase, and handle common variations
    country_normalized = str(country_name).strip().lower()
    
    # Handle common country name variations (e.g., "España" -> "Spain")
    country_variations = {
        'españa': 'spain', 'espana': 'spain',
        'reino unido': 'united kingdom', 'reino unido de gran bretaña': 'united kingdom',
        'estados unidos': 'united states', 'ee.uu.': 'united states', 'eeuu': 'united states',
        'méxico': 'mexico', 'brasil': 'brazil', 'irán': 'iran',
        'países bajos': 'netherlands', 'paises bajos': 'netherlands', 'holanda': 'netherlands',
        'alemania': 'germany', 'francia': 'france', 'italia': 'italy', 'reino unido ': 'united kingdom',
        'emiratos árabes unidos': 'united arab emirates', 'arabia saudí': 'saudi arabia',
        'corea del sur': 'south korea', 'sudáfrica': 'south africa',
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
        'russia': '🇷🇺', 'ukraine': '🇺🇦', 'czech republic': '🇨🇿', 'hungary': '🇭🇺', 'romania': '🇷🇴',
        'iran': '🇮🇷', 'sri lanka': '🇱🇰', 'pakistan': '🇵🇰', 'bangladesh': '🇧🇩', 'indonesia': '🇮🇩',
        'vietnam': '🇻🇳', 'thailand': '🇹🇭', 'philippines': '🇵🇭', 'taiwan': '🇹🇼', 'singapore': '🇸🇬',
        'malaysia': '🇲🇾', 'hong kong': '🇭🇰', 'nigeria': '🇳🇬', 'kenya': '🇰🇪', 'ghana': '🇬🇭',
        'morocco': '🇲🇦', 'tunisia': '🇹🇳', 'algeria': '🇩🇿', 'lebanon': '🇱🇧', 'jordan': '🇯🇴',
        'iraq': '🇮🇶', 'saudi arabia': '🇸🇦', 'united arab emirates': '🇦🇪', 'qatar': '🇶🇦',
        'bolivia': '🇧🇴', 'paraguay': '🇵🇾', 'costa rica': '🇨🇷', 'panama': '🇵🇦', 'cuba': '🇨🇺',
        'dominican republic': '🇩🇴', 'guatemala': '🇬🇹', 'el salvador': '🇸🇻', 'honduras': '🇭🇳',
        'croatia': '🇭🇷', 'serbia': '🇷🇸', 'bulgaria': '🇧🇬', 'slovakia': '🇸🇰', 'slovenia': '🇸🇮',
        'lithuania': '🇱🇹', 'latvia': '🇱🇻', 'estonia': '🇪🇪', 'iceland': '🇮🇸', 'luxembourg': '🇱🇺',
        'cyprus': '🇨🇾', 'malta': '🇲🇹',
    }
    
    # Try exact match first
    if country_normalized in country_flags:
        return country_flags[country_normalized]
    
    # Unknown country: no flag — the caller shows the country name instead of a meaningless globe.
    return ""


def _country_token(country):
    """Compact country marker for a card: flag emoji if known, else the name, else ""."""
    if not country or pd.isna(country):
        return ""
    name = str(country).strip()
    return get_country_flag(name) or html.escape(name.title())


def _field_label(field):
    """Small grey field pill, or "" when the field is missing."""
    if not field or str(field).strip().lower() in ("", "nan", "n/a", "none"):
        return ""
    return (
        '<span style="background:#F5F5F7;padding:0.2rem 0.5rem;border-radius:4px;'
        f'font-size:0.85rem;margin-left:0.4rem;">{html.escape(str(field))}</span>'
    )


def _col_val(row, col):
    """First non-blank value of a column in a one-row frame, or None."""
    if not row.empty and col in row.columns:
        v = row[col].iloc[0]
        if pd.notna(v) and str(v).strip():
            return str(v).strip()
    return None


def _numeric(value):
    """Leading integer in an id, for numeric sort; 0 if none."""
    m = re.search(r'\d+', str(value))
    return int(m.group()) if m else 0


def _match_rows(matches, mentees_df, mentors_df, flagged_pairs):
    """Flatten matches into rows carrying the fields the list filters/sorts/renders on.
    Mentee country = affiliated-university country; mentor country = country of origin."""
    # Match ids as strings — assignment ids are strings while the df id column may be int.
    mentee_ids = mentees_df[MENTEE_COLUMNS['id']].astype(str)
    mentor_ids = mentors_df[MENTOR_COLUMNS['id']].astype(str)
    rows = []
    for md in matches:
        mid = md['mentee_id']
        mrow = mentees_df[mentee_ids == str(mid)]
        mentee_field = _col_val(mrow, MENTEE_COLUMNS['field']) or ''
        mentee_country = _col_val(mrow, MENTEE_COLUMNS['country'])
        for m in md['matches']:
            tid = m['mentor_id']
            trow = mentors_df[mentor_ids == str(tid)]
            rows.append({
                'mentee_id': mid, 'mentor_id': tid,
                'score': m['match_percentage'], 'quality': m['match_quality'],
                'mentee_field': mentee_field, 'mentor_field': _col_val(trow, MENTOR_COLUMNS['field']) or '',
                'mentee_country': mentee_country,
                'mentor_country': _col_val(trow, MENTOR_COLUMNS['country_of_origin']),
                'flagged': (str(mid), str(tid)) in flagged_pairs,
            })
    return rows


_QUALITY_ORDER = {'Excellent': 0, 'Strong': 1, 'Good': 2, 'Fair': 3, 'Poor': 4}


def _match_quality(score):
    """Quality band for a match badge — same 5 bands as the distribution (Excellent≥85 … Poor<45)."""
    return ('Excellent' if score >= 85 else 'Strong' if score >= 75
            else 'Good' if score >= 60 else 'Fair' if score >= 45 else 'Poor')


def _alternative_mentors(mentee_id, current_mentor_id, n=6):
    """Top-N mentors for this mentee by matrix score, excluding the current one:
    (id, field, score, current_load)."""
    matrix = st.session_state.get('matrix_df')
    if matrix is None or str(mentee_id) not in matrix.columns:
        return []
    mentors_df = st.session_state.mentors_df
    mentor_ids = mentors_df[MENTOR_COLUMNS['id']].astype(str)
    load = {}
    for _m, mt, _s in st.session_state.get('assignments', []):
        load[str(mt)] = load.get(str(mt), 0) + 1
    out = []
    for mid, score in matrix[str(mentee_id)].sort_values(ascending=False).items():
        mid = str(mid)
        if mid == str(current_mentor_id):
            continue
        field = _col_val(mentors_df[mentor_ids == mid], MENTOR_COLUMNS['field']) or ''
        out.append((mid, field, round(float(score)), load.get(mid, 0)))
        if len(out) >= n:
            break
    return out


def _apply_override(mentee_id, new_mentor_id):
    """Guarded swap: reassign a mentee to a new mentor. Updates assignments + matches, clears the
    now-stale cached reasoning, and recomputes flags/stats. No global re-optimization."""
    matrix = st.session_state.matrix_df
    score = round(float(matrix.loc[str(new_mentor_id), str(mentee_id)]))
    st.session_state.assignments = [
        (m, new_mentor_id, score) if str(m) == str(mentee_id) else (m, mt, s)
        for m, mt, s in st.session_state.assignments
    ]
    for md in st.session_state.matches:
        if str(md['mentee_id']) == str(mentee_id):
            md['matches'][0].update({
                'mentor_id': new_mentor_id, 'match_percentage': score,
                'match_quality': _match_quality(score), 'reasoning': None,
            })
    cache = st.session_state.get('reasoning_cache', {})
    for key in [k for k in cache if str(k[0]) == str(mentee_id)]:
        cache.pop(key, None)
    flagged, stats = validate_and_flag_matches(
        st.session_state.matches, st.session_state.mentees_df, st.session_state.mentors_df, render=False)
    st.session_state.flagged_matches = flagged
    st.session_state.score_stats = stats


def _render_match_card(r):
    """One match card: header (ids, country, field) + badge + on-demand reasoning."""
    # Survey-derived text is escaped and the CSS class is allowlisted (values render via unsafe HTML).
    cls = r["quality"].lower() if r["quality"] in _QUALITY_ORDER else "fair"
    mentee_id = html.escape(str(r["mentee_id"]))
    mentor_id = html.escape(str(r["mentor_id"]))
    quality = html.escape(str(r["quality"]))
    header = (
        f'<h4>Mentee {mentee_id} {_country_token(r["mentee_country"])} {_field_label(r["mentee_field"])}'
        f' → Mentor {mentor_id} {_country_token(r["mentor_country"])} {_field_label(r["mentor_field"])}</h4>'
        f'<p><span class="percentage-badge {cls}">{int(r["score"])}% - {quality}</span>'
        f'{" 🚩" if r["flagged"] else ""}</p>'
    )
    with st.container(border=True):
        st.markdown(header, unsafe_allow_html=True)
        cache = st.session_state.setdefault("reasoning_cache", {})
        ck = (str(r['mentee_id']), str(r['mentor_id']))
        if st.button("💡 Why this match?", key=f"why_{r['mentee_id']}_{r['mentor_id']}") and ck not in cache:
            from app import reasoning_for_pair
            with st.spinner("Asking Claude…"):
                cache[ck] = reasoning_for_pair(r['mentee_id'], r['mentor_id'], r['score'],
                                               st.session_state.mentees_df, st.session_state.mentors_df)
        if ck in cache:
            st.markdown(f"**Reasoning:** {cache[ck]}")

        # Guarded override: reassign this mentee to a different mentor (warns if over capacity).
        with st.expander("🔀 Change mentor"):
            alts = _alternative_mentors(r['mentee_id'], r['mentor_id'])
            if not alts:
                st.caption("No alternative mentors available.")
            else:
                cap = st.session_state.get('max_mentees_per_mentor', 2)
                labels = {
                    f"Mentor {mid} · {field or 'N/A'} · {score}% · {ld}/{cap} mentees": mid
                    for mid, field, score, ld in alts
                }
                pick = st.selectbox("Alternatives (by score)", list(labels),
                                    key=f"alt_{r['mentee_id']}", label_visibility="collapsed")
                if st.button("Use this mentor", key=f"swap_{r['mentee_id']}"):
                    new_mentor = labels[pick]
                    load = sum(1 for _, mt, _ in st.session_state.assignments if str(mt) == str(new_mentor))
                    _apply_override(r['mentee_id'], new_mentor)
                    if load >= cap:
                        st.toast(f"⚠️ Mentor {new_mentor} now has {load + 1} mentees (cap {cap})", icon="⚠️")
                    else:
                        st.toast(f"✅ Reassigned mentee {r['mentee_id']} → mentor {new_mentor}")
                    st.rerun()


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
                            'match_quality': _match_quality(score),
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
    
    # Results step: the at-a-glance overview + heatmap. The matches list lives in the Review step.
    if not st.session_state.matches:
        st.info("👆 Generate matches in the Generate step to see results.")
        _back_to_generate()
        return

    st.divider()
    st.subheader("🎯 Results")
    if st.session_state.assignments:
        _render_results_header(st.session_state.assignments, st.session_state.get('score_stats'))

    # Capacity banner — up top with the summary.
    max_mentees_per_mentor = st.session_state.get('max_mentees_per_mentor', 2)
    if check_mentor_conflicts(st.session_state.matches, max_mentees_per_mentor=max_mentees_per_mentor):
        st.warning(f"⚠️ Some mentors are over the cap of {max_mentees_per_mentor} — see the Review step.")
    else:
        st.success(f"✅ Every mentor is within the cap (≤{max_mentees_per_mentor} mentees).")

    # Jump into the Review step, optionally filtered to the ones needing a look.
    flagged_n = len(st.session_state.get('flagged_matches') or [])
    go_all, go_flagged = st.columns(2)
    with go_all:
        if st.button("📋 Review all matches →", width='stretch', type="primary"):
            st.session_state.review_only = False
            st.session_state["step"] = 4
            st.rerun()
    with go_flagged:
        if st.button(f"⚠️ Review {flagged_n} flagged →", width='stretch', disabled=flagged_n == 0):
            st.session_state.review_only = True
            st.session_state["step"] = 4
            st.rerun()

    st.divider()

    # Export — up with the results (plain is instant; the reasoning version costs tokens, on demand).
    st.subheader("📥 Export")
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

    # Heatmap — the full-matrix download sits next to the title so it isn't lost below the chart.
    if st.session_state.matrix_df is not None:
        hm_title, hm_dl = st.columns([3, 1])
        with hm_title:
            st.subheader("🔥 Compatibility Heatmap")
        with hm_dl:
            st.download_button("📥 Full matrix (CSV)", st.session_state.matrix_df.to_csv(),
                               f"compatibility_matrix_{stamp}.csv", "text/csv", width='stretch')
        st.caption("Grouped by field so same-field matches cluster · blue outline = chosen match · hover a cell for detail")
        mentee_fields = {
            str(r[MENTEE_COLUMNS['id']]): r[MENTEE_COLUMNS['field']]
            for _, r in st.session_state.mentees_df.iterrows()
        }
        mentor_fields = {
            str(r[MENTOR_COLUMNS['id']]): r[MENTOR_COLUMNS['field']]
            for _, r in st.session_state.mentors_df.iterrows()
        }
        fig = create_heatmap(st.session_state.matrix_df, st.session_state.assignments,
                             mentee_fields, mentor_fields)
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': True, 'displaylogo': False})

    if is_debug_mode() and st.session_state.matrix_df is not None:
        st.session_state.score_distribution = analyze_score_distribution(st.session_state.matrix_df)


def render_review_tab():
    """Review step: the matches list where organizers search, filter, and override."""
    if not st.session_state.get('matches'):
        st.info("👆 Generate matches first.")
        _back_to_generate()
        return

    st.subheader("🔍 Review matches")
    flagged_pairs = {
        (str(f['mentee_id']), str(f['mentor_id']))
        for f in (st.session_state.get('flagged_matches') or [])
    }
    rows = _match_rows(st.session_state.matches, st.session_state.mentees_df,
                       st.session_state.mentors_df, flagged_pairs)

    ctrl_search, ctrl_sort, ctrl_review = st.columns([3, 2, 2])
    with ctrl_search:
        query = st.text_input(
            "Search matches", value=st.session_state.get('mentee_search', ''),
            placeholder="Search by mentee/mentor ID or field…", label_visibility="collapsed",
        ).strip().lower()
        st.session_state.mentee_search = query
    with ctrl_sort:
        sort_by = st.selectbox(
            "Sort by", ["Mentee ID", "Score: high → low", "Score: low → high", "Quality"],
            label_visibility="collapsed",
        )
    with ctrl_review:
        review_only = st.checkbox(
            f"⚠️ Needs review only ({len(flagged_pairs)})",
            value=st.session_state.get('review_only', False),
        )
        st.session_state.review_only = review_only

    def _hit(r):
        if not query:
            return True
        return query in f"{r['mentee_id']} {r['mentor_id']} {r['mentee_field']} {r['mentor_field']}".lower()

    shown = [r for r in rows if _hit(r) and (not review_only or r['flagged'])]
    if sort_by == "Score: high → low":
        shown.sort(key=lambda r: -r['score'])
    elif sort_by == "Score: low → high":
        shown.sort(key=lambda r: r['score'])
    elif sort_by == "Quality":
        shown.sort(key=lambda r: _QUALITY_ORDER.get(r['quality'], 9))
    else:
        shown.sort(key=lambda r: _numeric(r['mentee_id']))

    st.caption(f"Showing {len(shown)} of {len(rows)} matches")
    if not shown:
        st.info("No matches for these filters.")
    for r in shown:
        _render_match_card(r)

