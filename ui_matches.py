"""
UI Component for the Generate step: Matches Summary
Shows a summary of all settings and data before generating matches
"""

import streamlit as st



def render_matches_tab():
    """Render the Generate step: Matches Summary"""
    
    try:
        data_ready = (
            st.session_state.get('mentees_df') is not None and 
            st.session_state.get('mentors_df') is not None
        )
        
        if not data_ready:
            st.markdown("""
            <div class="info-card">
                Review your configuration and generate compatibility matches. Upload data files in the <strong>Upload</strong> step to begin.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="token-warning">
                ⚠️ <strong>No data uploaded yet!</strong><br>
                Go to the <strong>Upload</strong> step to upload your mentees and mentors files.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
            
        from utils import is_debug_mode

        mentees_df = st.session_state.get('mentees_df')
        mentors_df = st.session_state.get('mentors_df')
        mentee_count = len(mentees_df) if mentees_df is not None else 0
        mentor_count = len(mentors_df) if mentors_df is not None else 0

        st.subheader("⚡ Generate")
        st.markdown("""
        <div class="info-card">
            <strong>Ready to generate matches.</strong><br>
            Jev scores every mentee×mentor pair, scipy finds the optimal assignment,
            then Claude explains each match.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Mentees", mentee_count)
        with col2:
            st.metric("👨‍🏫 Mentors", mentor_count)
        with col3:
            st.metric("⚖️ Max per Mentor", st.session_state.get('max_mentees_per_mentor', 2))

        if is_debug_mode():
            api_key = st.session_state.get('api_key')
            st.caption(f"Anthropic key: {'✅ Configured' if api_key else '❌ Missing'}")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Start the run and jump to Results; main() locks the rail while it runs
        if st.button("🚀 Generate Matches", type="primary", width='stretch'):
            st.session_state.start_processing = True
            st.session_state["step"] = 3  # Results
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error in Generate step: {e!s}")
        st.exception(e)
        import traceback
        with st.expander("🔍 Full Error Details", expanded=False):
            st.code(traceback.format_exc())
