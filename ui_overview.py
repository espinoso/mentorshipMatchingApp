"""
UI Component for Tab 5: Data Overview
Handles displaying and reviewing uploaded data before generating matches
"""

import streamlit as st


def render_overview_tab():
    """Render Tab 5: Review Your Data"""
    
    st.markdown("""
    <div class="info-card">
        Review your uploaded data before generating matches. All columns will be used for matching.
    </div>
    """, unsafe_allow_html=True)
    
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
            st.dataframe(st.session_state.mentees_df.head(5), width='stretch')
            
            # Button to show full table
            if st.button("📋 Show Full Mentees Table", key="show_full_mentees"):
                st.dataframe(st.session_state.mentees_df, width='stretch')
                st.caption(f"Showing all {len(st.session_state.mentees_df)} rows • {len(st.session_state.mentees_df.columns)} columns")
            else:
                st.caption(f"Showing first 5 of {len(st.session_state.mentees_df)} rows • {len(st.session_state.mentees_df.columns)} columns")
    
    if st.session_state.mentors_df is not None:
        with st.expander("👨‍🏫 Preview Mentors Data", expanded=True):
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

