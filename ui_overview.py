"""
UI Component for Tab 5: Data Overview
Handles displaying and reviewing uploaded data before generating matches
"""

import streamlit as st


def render_overview_tab():
    """Render Tab 5: Review Your Data"""
    
    try:
        st.markdown("""
        <div class="info-card">
            Review your uploaded data before generating matches. All columns will be used for matching.
        </div>
        """, unsafe_allow_html=True)
        
        data_ready = (
            st.session_state.get('mentees_df') is not None and 
            st.session_state.get('mentors_df') is not None
        )
        
        if not data_ready:
            st.markdown("""
            <div class="token-warning">
                ⚠️ <strong>No data uploaded yet!</strong><br>
                Go to <strong>Tab 2</strong> to upload your mentees and mentors files.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Data preview with better styling
        # Use .get() for safe access
        mentees_df = st.session_state.get('mentees_df')
        mentors_df = st.session_state.get('mentors_df')
        
        if mentees_df is not None:
            with st.expander("👥 Preview Mentees Data", expanded=False):  # Changed to False
                st.dataframe(mentees_df.head(5))  # Removed width='stretch'
                
                # Button to show full table
                if st.button("📋 Show Full Mentees Table", key="show_full_mentees"):
                    st.dataframe(mentees_df)  # Removed width='stretch'
                    st.caption(f"Showing all {len(mentees_df)} rows • {len(mentees_df.columns)} columns")
                else:
                    st.caption(f"Showing first 5 of {len(mentees_df)} rows • {len(mentees_df.columns)} columns")
        
        if mentors_df is not None:
            with st.expander("👨‍🏫 Preview Mentors Data", expanded=False):  # Changed to False
                st.dataframe(mentors_df.head(5))  # Removed width='stretch'
                
                # Button to show full table
                if st.button("📋 Show Full Mentors Table", key="show_full_mentors"):
                    st.dataframe(mentors_df)  # Removed width='stretch'
                    st.caption(f"Showing all {len(mentors_df)} rows • {len(mentors_df.columns)} columns")
                else:
                    st.caption(f"Showing first 5 of {len(mentors_df)} rows • {len(mentors_df.columns)} columns")
        
        st.markdown("""
        <div class="success-card">
            ✅ <strong>Data looks good!</strong><br>
            You can now proceed to <strong>"5️⃣ Generate Matches"</strong>.
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error in Tab 4 (Data Overview): {str(e)}")
        st.exception(e)
        import traceback
        with st.expander("🔍 Full Error Details", expanded=False):
            st.code(traceback.format_exc())

