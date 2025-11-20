"""
UI Component for Tab 3: Upload Data Files
Handles uploading and validating mentee and mentor Excel files
"""

import streamlit as st
import pandas as pd
from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from utils import clean_dataframe, validate_uploaded_file


def render_upload_tab():
    """Render Tab 3: Upload Mentee & Mentor Files"""
    
    if not st.session_state.api_key:
        st.markdown("""
        <div class="conflict-warning">
            ⚠️ <strong>API Key Required</strong><br>
            Please configure your OpenAI API key in Tab 1 before uploading data files.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    st.markdown("""
    <div class="info-card">
        Upload your mentee and mentor Excel files. Both files are required to proceed with matching.
    </div>
    """, unsafe_allow_html=True)
    
    # Mentees and Mentors side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Mentees File *Required*")
        mentee_file = st.file_uploader(
            "Upload mentees Excel file",
            type=['xlsx'],
            key=f"mentee_upload_{st.session_state.reset_counter}",
            help=f"Excel file containing mentee information. Must have '{MENTEE_COLUMNS['id']}' column."
        )
        
    if mentee_file:
        try:
            df = pd.read_excel(mentee_file)
            df = clean_dataframe(df)
            
            errors, warnings, completeness = validate_uploaded_file(df, 'mentee')
            
            if errors:
                for error in errors:
                    st.markdown(f"""
                    <div class="conflict-warning">
                        ❌ <strong>Error:</strong> {error}<br>
                        <small>Please fix this before proceeding</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.session_state.mentees_df = df
                st.markdown(f"""
                <div class="success-card">
                    ✅ <strong>Loaded {len(df)} mentees</strong><br>
                    📊 Columns: {len(df.columns)} | Data quality: {completeness:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                if completeness < 100:
                    with st.expander("ℹ️ Data Quality Explanation"):
                        total_cells = df.shape[0] * df.shape[1]
                        empty_cells = df.isna().sum().sum()
                        st.markdown(f"""
                        **How data quality is calculated:**
                        - Total cells: {total_cells:,} ({df.shape[0]} rows × {df.shape[1]} columns)
                        - Filled cells: {total_cells - empty_cells:,}
                        - Empty cells: {empty_cells:,}
                        - Quality: {completeness:.1f}% = (Filled / Total) × 100
                        
                        **Possible reasons for < 100%:**
                        - Optional fields left blank (e.g., "Mentee other relevant info")
                        - Missing data in survey responses
                        - Partial form submissions
                        - Fields not applicable to all mentees
                        
                        **Is this okay?** Yes! As long as required fields (ID, field, specialization) are filled, 
                        the system will work fine. Empty optional fields won't affect matching quality.
                        """)
                
                if warnings:
                    for warning in warnings:
                        st.markdown(f"""
                        <div class="token-warning">
                            ⚠️ {warning}
                        </div>
                        """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.markdown(f"""
            <div class="conflict-warning">
                ❌ <strong>Error loading file:</strong> {str(e)}<br>
                <small>Make sure it's a valid Excel (.xlsx) file</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("👨‍🏫 Mentors File *Required*")
        mentor_file = st.file_uploader(
            "Upload mentors Excel file",
            type=['xlsx'],
            key=f"mentor_upload_{st.session_state.reset_counter}",
            help=f"Excel file containing mentor information. Must have '{MENTOR_COLUMNS['id']}' column."
        )
        
    if mentor_file:
        try:
            df = pd.read_excel(mentor_file)
            df = clean_dataframe(df)
            
            errors, warnings, completeness = validate_uploaded_file(df, 'mentor')
            
            if errors:
                for error in errors:
                    st.markdown(f"""
                    <div class="conflict-warning">
                        ❌ <strong>Error:</strong> {error}<br>
                        <small>Please fix this before proceeding</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.session_state.mentors_df = df
                st.markdown(f"""
                <div class="success-card">
                    ✅ <strong>Loaded {len(df)} mentors</strong><br>
                    📊 Columns: {len(df.columns)} | Data quality: {completeness:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                if completeness < 100:
                    with st.expander("ℹ️ Data Quality Explanation"):
                        total_cells = df.shape[0] * df.shape[1]
                        empty_cells = df.isna().sum().sum()
                        st.markdown(f"""
                        **How data quality is calculated:**
                        - Total cells: {total_cells:,} ({df.shape[0]} rows × {df.shape[1]} columns)
                        - Filled cells: {total_cells - empty_cells:,}
                        - Empty cells: {empty_cells:,}
                        - Quality: {completeness:.1f}% = (Filled / Total) × 100
                        
                        **Possible reasons for < 100%:**
                        - Optional fields left blank (e.g., "Mentor current position", "Alma Mater")
                        - Missing data in survey responses
                        - Partial form submissions
                        - Fields not applicable to all mentors
                        - Missing years of experience (1 mentor in your data)
                        
                        **Is this okay?** Yes! As long as required fields (ID, field, specialization) are filled, 
                        the system will work fine. Empty optional fields won't affect matching quality significantly.
                        """)
                
                if warnings:
                    for warning in warnings:
                        st.markdown(f"""
                        <div class="token-warning">
                            ⚠️ {warning}
                        </div>
                        """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.markdown(f"""
            <div class="conflict-warning">
                ❌ <strong>Error loading file:</strong> {str(e)}<br>
                <small>Make sure it's a valid Excel (.xlsx) file</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()

