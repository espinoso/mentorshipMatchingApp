"""
UI Component for Tab 2: Training Files
Handles training file management, including viewing existing files in OpenAI storage and uploading new ones
"""

import streamlit as st
import pandas as pd
import time
from utils import clean_dataframe
from openai_files import list_openai_files, upload_training_files_to_openai, delete_openai_files


def render_training_tab():
    """Render Tab 2: Manage Training Files"""
    
    if not st.session_state.api_key:
        st.markdown("""
        <div class="conflict-warning">
            ⚠️ <strong>API Key Required</strong><br>
            Please configure your OpenAI API key in Tab 1 before managing training files.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    st.markdown("""
    <div class="info-card">
        Training files help the AI learn patterns from historical mentor-mentee matches.<br>
        You can manage existing files in OpenAI storage or upload new ones.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.training_file_ids:
        st.markdown("---")
        st.subheader("✅ Currently Selected Training Files")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            file_count = len(st.session_state.training_file_ids)
            st.success(f"📊 {file_count} file(s) selected from OpenAI storage")
            
            if st.session_state.training_file_names:
                with st.expander("📄 View File Names"):
                    for i, name in enumerate(st.session_state.training_file_names, 1):
                        st.write(f"**{i}.** {name}")
            else:
                with st.expander("📄 View File IDs"):
                    for i, file_id in enumerate(st.session_state.training_file_ids, 1):
                        st.write(f"**{i}.** {file_id}")
        
        with col2:
            if st.button("🗑️ Clear Selection", help="Remove selected training files"):
                st.session_state.training_files = []
                st.session_state.training_file_ids = []
                st.session_state.training_file_names = []
                st.success("Cleared!")
                st.rerun()
        
        st.markdown("---")
    elif st.session_state.training_files:
        # Legacy support for locally uploaded files
        st.markdown("---")
        st.subheader("✅ Currently Loaded Training Files (Local)")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            total_rows = sum(len(df) for df in st.session_state.training_files)
            st.success(f"📊 {len(st.session_state.training_files)} file(s) loaded • {total_rows} total records")
            
            with st.expander("📄 View Details"):
                for i, df in enumerate(st.session_state.training_files, 1):
                    st.write(f"**File {i}**: {len(df)} rows × {len(df.columns)} columns")
        
        with col2:
            if st.button("🗑️ Clear All", help="Remove training files from session"):
                st.session_state.training_files = []
                st.session_state.training_file_ids = []
                st.session_state.training_file_names = []
                st.success("Cleared!")
                st.rerun()
        
        st.markdown("---")
    
    training_tab1, training_tab2 = st.tabs(["📁 Existing Files", "📤 Upload New"])
    
    with training_tab1:
        st.subheader("Files in OpenAI Storage")
        
        if st.button("🔄 Refresh File List", key="refresh_files"):
            st.rerun()
        
        with st.spinner("Loading files from OpenAI..."):
            existing_files = list_openai_files(st.session_state.api_key)
        
        if existing_files:
            st.success(f"✅ Found {len(existing_files)} file(s) in storage")
            
            st.markdown("""
            <div class="info-card">
                💡 Select the training files you want to use for matching, then click "Load Selected Files"
            </div>
            """, unsafe_allow_html=True)
            
            # File selection checkboxes
            selected_files = []
            for file_info in existing_files:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    is_selected = st.checkbox(
                        f"📄 {file_info['filename']} ({file_info['size_kb']:.1f} KB • {file_info['age_days']} days old)",
                        key=f"select_{file_info['id']}"
                    )
                    if is_selected:
                        selected_files.append(file_info)
                
                with col2:
                    if st.button(f"🗑️", key=f"delete_{file_info['id']}", help="Delete this file"):
                        with st.spinner("Deleting..."):
                            deleted = delete_openai_files(st.session_state.api_key, [file_info['id']])
                            if deleted > 0:
                                st.success("✅ File deleted")
                                time.sleep(0.5)
                                st.rerun()
            
            # Use selected files button
            if selected_files:
                st.divider()
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{len(selected_files)} file(s) selected**")
                with col2:
                    if st.button("📥 Click here to load selected files", type="primary", width='stretch'):
                        # Just store the file IDs and basic info
                        selected_ids = [f['id'] for f in selected_files]
                        selected_names = [f['filename'] for f in selected_files]
                        
                        st.session_state.training_file_ids = selected_ids
                        st.session_state.training_file_names = selected_names
                        st.session_state.training_files = []
                        
                        st.success(f"✅ Selected {len(selected_ids)} training file(s) for matching!")
                        st.info("These files will be used via OpenAI's file_search tool during matching")
                        time.sleep(1)
                        st.rerun()
            
            # Bulk delete old files
            st.divider()
            old_files = [f for f in existing_files if f['age_days'] > 30]
            if old_files:
                st.warning(f"⚠️ You have {len(old_files)} file(s) older than 30 days")
                if st.button("🗑️ Delete All Old Files"):
                    old_file_ids = [f['id'] for f in old_files]
                    with st.spinner("Deleting old files..."):
                        deleted = delete_openai_files(st.session_state.api_key, old_file_ids)
                        st.success(f"✅ Deleted {deleted} file(s)")
                        time.sleep(1)
                        st.rerun()
        else:
            st.info("📭 No training files found in OpenAI storage. Upload some in the 'Upload New' tab!")
    
    with training_tab2:
        st.subheader("Upload New Training Files")
        
        st.markdown("""
        <div class="token-warning">
            💡 <strong>Tip:</strong> Training files should contain historical mentor-mentee pairings.<br>
            Upload Excel (.xlsx) files with the same structure as your mentee/mentor data.
        </div>
        """, unsafe_allow_html=True)
        
        training_files = st.file_uploader(
            "Select training file(s)",
            type=['xlsx'],
            accept_multiple_files=True,
            key=f"training_upload_{st.session_state.reset_counter}",
            help="Upload one or more Excel files with historical pairing data"
        )
        
        if training_files:
            st.info(f"📁 {len(training_files)} file(s) selected")
            
            training_dfs = []
            for file in training_files:
                try:
                    df = pd.read_excel(file)
                    df = clean_dataframe(df)
                    training_dfs.append(df)
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ <strong>{file.name}</strong>: {len(df)} rows, {len(df.columns)} columns
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="conflict-warning">
                        ❌ <strong>{file.name}</strong>: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
            
            if training_dfs:
                st.divider()
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("💾 Save Locally Only", width='stretch'):
                        st.session_state.training_files = training_dfs
                        st.success("✅ Training files saved to session (local only)")
                        time.sleep(0.5)  # Brief pause to show success message
                        st.rerun()
                
                with col2:
                    if st.button("☁️ Upload to OpenAI Storage", width='stretch', type="primary"):
                        with st.spinner("Uploading to OpenAI..."):
                            uploaded_ids = upload_training_files_to_openai(training_dfs, st.session_state.api_key)
                            if uploaded_ids:
                                st.session_state.training_files = training_dfs
                                st.session_state.training_file_ids = uploaded_ids
                                st.success(f"✅ Uploaded {len(uploaded_ids)} file(s) to OpenAI storage!")
                                time.sleep(1)  # Brief pause to show success message
                                st.rerun()
                
                st.markdown("""
                <div class="info-card">
                    <strong>Which option should I choose?</strong><br>
                    • <strong>Save Locally</strong>: Files sent directly in prompts (higher token usage)<br>
                    • <strong>Upload to OpenAI</strong>: Files stored in OpenAI, referenced via file_search (lower token usage, better for large files)
                </div>
                """, unsafe_allow_html=True)

