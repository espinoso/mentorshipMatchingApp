"""
UI Component for Tab 3: Upload Data Files
Handles uploading and validating mentee and mentor Excel files

Refactored version with:
- Extracted shared file handling into reusable functions
- Consolidated session state initialization
- Helper functions for state management
- Removed code duplication (~380 lines saved)
"""

import html
import time
from io import BytesIO
from typing import Literal, Optional, Tuple, Dict, Any, List

import streamlit as st
import pandas as pd

from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from utils import clean_dataframe, validate_uploaded_file
from normalization_utils import (
    detect_file_format,
    process_file_with_progress,
    create_download_bytes,
    validate_pre_format_columns
)


# =============================================================================
# TYPE ALIASES
# =============================================================================
FileType = Literal['mentee', 'mentor']


# =============================================================================
# CONSTANTS
# =============================================================================
LOG_CONTAINER_CSS = """
<style>
.log-container {
    height: 400px;
    overflow-y: auto;
    overflow-x: auto;
    background-color: #1e1e1e;
    border: 1px solid #3e3e3e;
    border-radius: 5px;
    padding: 10px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    white-space: pre;
    color: #d4d4d4;
}
</style>
"""

FILE_TYPE_CONFIG = {
    'mentee': {
        'icon': '👥',
        'title': 'Mentees File',
        'columns': MENTEE_COLUMNS,
        'df_key': 'mentees_df',
        'quality_note': 'Optional fields left blank (e.g., "Mentee other relevant info")',
    },
    'mentor': {
        'icon': '👨‍🏫',
        'title': 'Mentors File',
        'columns': MENTOR_COLUMNS,
        'df_key': 'mentors_df',
        'quality_note': 'Optional fields left blank (e.g., "Mentor current position", "Alma Mater")',
    }
}

# Session state defaults - centralized for easy maintenance
SESSION_STATE_DEFAULTS = {
    'processing_mentee': False,
    'processing_mentor': False,
    'mentee_processed_df': None,
    'mentor_processed_df': None,
    'mentee_processing_info': None,
    'mentor_processing_info': None,
    'mentee_cancel_flag': {'cancelled': False},
    'mentor_cancel_flag': {'cancelled': False},
    'mentee_processing_start_time': None,
    'mentor_processing_start_time': None,
    'mentee_progress_state': {'messages': [], 'current_status': '', 'processed_rows': 0, 'total_rows': 0},
    'mentor_progress_state': {'messages': [], 'current_status': '', 'processed_rows': 0, 'total_rows': 0},
    'mentee_processing_complete': False,
    'mentor_processing_complete': False,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def init_session_state():
    """Initialize all session state variables with defaults."""
    for key, default in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            # Deep copy mutable defaults to avoid shared state
            if isinstance(default, dict):
                st.session_state[key] = default.copy()
            else:
                st.session_state[key] = default


def get_state_key(file_type: FileType, suffix: str) -> str:
    """Generate session state key for a file type."""
    return f"{file_type}_{suffix}"


def get_processing_status(file_type: FileType) -> Literal['idle', 'processing', 'complete', 'cancelled']:
    """Determine the current processing status for a file type."""
    is_processing = st.session_state.get(get_state_key(file_type, 'processing'), False)
    is_complete = st.session_state.get(get_state_key(file_type, 'processing_complete'), False)
    cancel_flag = st.session_state.get(get_state_key(file_type, 'cancel_flag'), {})
    processing_info = st.session_state.get(get_state_key(file_type, 'processing_info'))
    
    actually_cancelled = cancel_flag.get('cancelled', False)
    if processing_info:
        actually_cancelled = actually_cancelled or processing_info.get('cancelled', False)
    
    if is_processing and not is_complete:
        return 'processing'
    elif is_complete and not actually_cancelled:
        return 'complete'
    elif actually_cancelled and not is_processing:
        return 'cancelled'
    return 'idle'


def get_progress_messages(file_type: FileType) -> List[str]:
    """Get progress messages from processing info or progress state."""
    messages = []
    
    # Try processing_info first (most reliable after completion)
    processing_info = st.session_state.get(get_state_key(file_type, 'processing_info'))
    if processing_info:
        messages = processing_info.get('messages', [])
        if messages:
            # Update progress_state for consistency
            progress_state = st.session_state.get(get_state_key(file_type, 'progress_state'), {})
            progress_state['messages'] = messages.copy()
            st.session_state[get_state_key(file_type, 'progress_state')] = progress_state
    
    # Fall back to progress_state
    if not messages:
        progress_state = st.session_state.get(get_state_key(file_type, 'progress_state'), {})
        messages = progress_state.get('messages', [])
    
    return messages


def render_log_container(messages: List[str], show_header: bool = True):
    """Render a scrollable log container with messages. Does nothing if no messages."""
    if not messages:
        return
    
    if show_header:
        st.markdown("**📋 Processing Log:**")
    
    log_text = html.escape('\n'.join(messages))
    st.markdown(f'<div class="log-container">{log_text}</div>', unsafe_allow_html=True)
    st.caption(f"📊 Total log entries: {len(messages)}")


def render_html_card(content: str, card_class: str):
    """Render an HTML card with the specified class."""
    st.markdown(f'<div class="{card_class}">{content}</div>', unsafe_allow_html=True)


def start_processing(file_type: FileType):
    """Initialize processing state for a file type."""
    st.session_state[get_state_key(file_type, 'processing')] = True
    st.session_state[get_state_key(file_type, 'processing_start_time')] = time.time()
    st.session_state[get_state_key(file_type, 'cancel_flag')] = {'cancelled': False}
    st.session_state[get_state_key(file_type, 'progress_state')] = {
        'messages': [], 'current_status': 'Initializing...', 'processed_rows': 0, 'total_rows': 0
    }
    st.session_state[get_state_key(file_type, 'processing_complete')] = False
    st.rerun()


def cancel_processing(file_type: FileType):
    """Cancel processing for a file type."""
    st.session_state[get_state_key(file_type, 'cancel_flag')]['cancelled'] = True
    st.session_state[get_state_key(file_type, 'processing')] = False
    st.session_state[get_state_key(file_type, 'processing_start_time')] = None
    st.session_state[get_state_key(file_type, 'processing_complete')] = False
    st.info("ℹ️ Processing cancelled. You can click 'Normalize Data' again to retry.")
    st.rerun()


# =============================================================================
# PROCESSING LOGIC
# =============================================================================
def run_file_processing(file_type: FileType):
    """Execute file processing for the given file type."""
    original_df = st.session_state.get(get_state_key(file_type, 'original_df'))
    if original_df is not None:
        total_rows = len(original_df)
        progress_state = st.session_state.get(get_state_key(file_type, 'progress_state'), {})
        progress_state['total_rows'] = total_rows
        st.session_state[get_state_key(file_type, 'progress_state')] = progress_state
    
    # Create placeholders for real-time updates
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Recreate file object from saved bytes
    file_bytes = st.session_state.get(get_state_key(file_type, 'original_file_bytes'))
    file_name = st.session_state.get(get_state_key(file_type, 'original_file_name'))
    file_obj = BytesIO(file_bytes)
    file_obj.name = file_name
    
    cancel_flag = st.session_state.get(get_state_key(file_type, 'cancel_flag'), {})
    
    try:
        processed_df, output_file, processing_info = process_file_with_progress(
            file_obj,
            file_type,
            st.session_state.api_key,
            progress_container=progress_placeholder,
            status_container=status_placeholder,
            cancel_flag=cancel_flag
        )
    except Exception as e:
        import traceback
        progress_state = st.session_state.get(get_state_key(file_type, 'progress_state'), {})
        processing_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'messages': progress_state.get('messages', []),
            'progress_state': progress_state.copy()
        }
        processed_df = None
        output_file = None
    
    # Update progress state from processing info
    if processing_info:
        progress_state = st.session_state.get(get_state_key(file_type, 'progress_state'), {})
        if 'progress_state' in processing_info:
            progress_state.update(processing_info['progress_state'])
        if 'messages' in processing_info:
            progress_state['messages'] = processing_info['messages']
            if processing_info['messages']:
                progress_state['current_status'] = processing_info['messages'][-1]
        st.session_state[get_state_key(file_type, 'progress_state')] = progress_state
    
    # Check if cancelled
    if cancel_flag.get('cancelled', False):
        progress_placeholder.empty()
        status_placeholder.empty()
        cancel_processing(file_type)
        return
    
    # Reset processing flag
    st.session_state[get_state_key(file_type, 'processing')] = False
    
    if processed_df is not None:
        config = FILE_TYPE_CONFIG[file_type]
        st.session_state[get_state_key(file_type, 'processed_df')] = processed_df
        st.session_state[config['df_key']] = processed_df
        st.session_state[get_state_key(file_type, 'processing_info')] = processing_info
        
        if processing_info and 'messages' in processing_info:
            progress_state = st.session_state.get(get_state_key(file_type, 'progress_state'), {})
            progress_state['messages'] = processing_info['messages']
            st.session_state[get_state_key(file_type, 'progress_state')] = progress_state
        
        st.session_state[get_state_key(file_type, 'cancel_flag')]['cancelled'] = False
        st.session_state[get_state_key(file_type, 'processing_complete')] = True
        # Clear placeholders before rerun
        progress_placeholder.empty()
        status_placeholder.empty()
        st.rerun()
    else:
        error_msg = processing_info.get('error', 'Unknown error') if processing_info else 'Unknown error'
        st.session_state[get_state_key(file_type, 'processing_start_time')] = None
        
        if processing_info and processing_info.get('messages'):
            st.warning("Processing log before error:")
            with st.expander("View Processing Log", expanded=True):
                st.code('\n'.join(processing_info['messages']), language=None)
        
        st.error(f"❌ Normalization failed: {error_msg}")
        
        if processing_info and 'traceback' in processing_info:
            with st.expander("Error Details"):
                st.code(processing_info['traceback'])
        
        # Clear placeholders before rerun
        progress_placeholder.empty()
        status_placeholder.empty()
        st.rerun()


# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_data_quality_expander(df: pd.DataFrame, file_type: FileType, completeness: float):
    """Render the data quality explanation expander."""
    config = FILE_TYPE_CONFIG[file_type]
    total_cells = df.shape[0] * df.shape[1]
    empty_cells = df.isna().sum().sum()
    
    with st.expander("ℹ️ Data Quality Explanation"):
        st.markdown(f"""
**How data quality is calculated:**
- Total cells: {total_cells:,} ({df.shape[0]} rows × {df.shape[1]} columns)
- Filled cells: {total_cells - empty_cells:,}
- Empty cells: {empty_cells:,}
- Quality: {completeness:.1f}% = (Filled / Total) × 100

**Possible reasons for < 100%:**
- {config['quality_note']}
- Missing data in survey responses
- Partial form submissions
- Fields not applicable to all {file_type}s

**Is this okay?** Yes! As long as required fields (ID, field, specialization) are filled, 
the system will work fine. Empty optional fields won't affect matching quality.
""")


def render_processing_ui(file_type: FileType, messages: List[str]):
    """Render the processing status UI based on current state."""
    status = get_processing_status(file_type)
    config = FILE_TYPE_CONFIG[file_type]
    
    if status == 'complete':
        render_log_container(messages)
    
    elif status == 'cancelled':
        st.warning("⚠️ Processing was cancelled")
        render_log_container(messages)
    
    elif status == 'processing':
        st.markdown(f"### 🔄 Processing {config['title'].replace(' File', '')} File")
        render_log_container(messages)
        
        if st.button("❌ Cancel", key=f"cancel_{file_type}", type="secondary", use_container_width=True):
            cancel_processing(file_type)


def render_download_section(file_type: FileType):
    """Render the download button and processing statistics."""
    config = FILE_TYPE_CONFIG[file_type]
    processed_df = st.session_state.get(get_state_key(file_type, 'processed_df'))
    
    if processed_df is None:
        return
    
    st.markdown("---")
    st.markdown("**📥 Download Normalized File**")
    
    download_bytes = create_download_bytes(processed_df)
    st.download_button(
        label=f"⬇️ Download Normalized {config['title'].replace(' File', '')}s File",
        data=download_bytes,
        file_name=f"{file_type}s_normalized.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{file_type}_processed"
    )
    
    processing_info = st.session_state.get(get_state_key(file_type, 'processing_info'))
    if processing_info and 'report' in processing_info:
        report = processing_info['report']
        with st.expander("📊 Processing Statistics"):
            st.text(f"Total rows: {report.stats.get('total_rows', 0)}")
            st.text(f"Successful: {report.stats.get('successful_rows', 0)}")
            st.text(f"Failed: {report.stats.get('failed_rows', 0)}")
            
            if 'cache_stats' in processing_info:
                cache_stats = processing_info['cache_stats']
                st.text(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.1f}%")
            
            if report.errors:
                st.warning(f"⚠️ {len(report.errors)} errors encountered during processing")
                with st.expander("View Errors"):
                    for error in report.errors[:10]:
                        st.text(error)


def render_file_upload_section(file_type: FileType, any_processing: bool, other_type_processing: bool):
    """
    Render the complete file upload section for either mentee or mentor.
    
    Args:
        file_type: 'mentee' or 'mentor'
        any_processing: True if any file is currently being processed
        other_type_processing: True if the OTHER file type is being processed
    """
    config = FILE_TYPE_CONFIG[file_type]
    
    st.subheader(f"{config['icon']} {config['title']} *Required*")
    
    # Check if THIS file type is currently processing
    is_current_type_processing = st.session_state.get(get_state_key(file_type, 'processing'), False)
    is_current_type_complete = st.session_state.get(get_state_key(file_type, 'processing_complete'), False)
    
    if other_type_processing:
        other_type = 'mentor' if file_type == 'mentee' else 'mentee'
        st.info(f"⏳ Please wait for {other_type} processing to complete before uploading {file_type} files.")
    
    if is_current_type_processing:
        st.info(f"⏳ Processing in progress. Please wait for normalization to complete.")
    
    # File uploader - disabled if any processing OR if this type is processing
    uploaded_file = st.file_uploader(
        f"Upload {file_type}s Excel file",
        type=['xlsx'],
        key=f"{file_type}_upload_{st.session_state.reset_counter}",
        help=f"Excel file containing {file_type} information. Must have '{config['columns']['id']}' column.",
        disabled=any_processing or is_current_type_processing
    )
    
    # Check for previously uploaded file (e.g., after cancel)
    if not uploaded_file:
        saved_bytes = st.session_state.get(get_state_key(file_type, 'original_file_bytes'))
        if saved_bytes:
            saved_name = st.session_state.get(get_state_key(file_type, 'original_file_name'), f'{file_type}s.xlsx')
            st.info(f"📄 File loaded: {saved_name}")
            
            uploaded_file = BytesIO(saved_bytes)
            uploaded_file.name = saved_name
            
            saved_df = st.session_state.get(get_state_key(file_type, 'original_df'))
            if saved_df is not None:
                df = saved_df
            else:
                df = pd.read_excel(uploaded_file)
                df = clean_dataframe(df)
                st.session_state[get_state_key(file_type, 'original_df')] = df
    
    # If THIS file type is currently processing, show processing UI and return early (block all file interactions)
    if is_current_type_processing:
        status = get_processing_status(file_type)
        messages = get_progress_messages(file_type)
        has_messages = len(messages) > 0
        
        if status != 'idle' or has_messages:
            st.markdown("---")
            render_processing_ui(file_type, messages)
        
        # Run processing if flag is set
        if is_current_type_processing and not is_current_type_complete:
            run_file_processing(file_type)
        
        return
    
    if not uploaded_file:
        return
    
    try:
        # Load and clean dataframe (if not already loaded from saved state)
        if 'df' not in dir() or df is None:
            df = pd.read_excel(uploaded_file)
            df = clean_dataframe(df)
        
        # Detect file format
        file_format = detect_file_format(df, file_type)
        
        # Validate based on format
        if file_format == 'POST':
            errors, warnings, completeness = validate_uploaded_file(df, file_type)
        else:
            is_valid_pre, missing_pre_cols = validate_pre_format_columns(df, file_type)
            if is_valid_pre:
                errors = []
                warnings = []
                total_cells = df.shape[0] * df.shape[1]
                empty_cells = df.isna().sum().sum()
                completeness = ((total_cells - empty_cells) / total_cells) * 100 if total_cells > 0 else 100.0
            else:
                errors = [f"Missing required PRE format column: {col}" for col in missing_pre_cols]
                warnings = []
                completeness = 0.0
        
        # Show errors if any
        if errors:
            for error in errors:
                render_html_card(
                    f"❌ <strong>Error:</strong> {error}<br>"
                    "<small>Please upload a file with the correct columns (PRE or POST format)</small>",
                    "conflict-warning"
                )
            return
        
        # Check if already processed
        already_processed = st.session_state.get(get_state_key(file_type, 'processed_df')) is not None
        
        # Show format detection result
        if file_format == 'POST':
            render_html_card(
                "✅ <strong>POST Format Detected</strong><br>"
                "File is already in the correct format and ready to use.",
                "success-card"
            )
            st.session_state[config['df_key']] = df
            st.session_state[get_state_key(file_type, 'processed_df')] = None
            st.info(f"📄 **Using POST format file** ({len(df)} {file_type}s)")
        elif file_format == 'PRE' and already_processed:
            # PRE format file that has been normalized
            processed_df = st.session_state.get(get_state_key(file_type, 'processed_df'))
            render_html_card(
                "✅ <strong>POST Format Generated</strong><br>"
                "File has been normalized and is ready to use.",
                "success-card"
            )
            st.info(f"📄 **Using normalized file** ({len(processed_df)} {file_type}s)")
            
            # Show download section right after "Using normalized file" message
            render_download_section(file_type)
        else:
            # PRE format file that needs normalization
            render_html_card(
                "📋 <strong>PRE Format Detected</strong><br>"
                "File needs normalization before use. Click \"Normalize Data\" to process.",
                "token-warning"
            )
            # Store original file for processing
            if hasattr(uploaded_file, 'getvalue'):
                st.session_state[get_state_key(file_type, 'original_file_bytes')] = uploaded_file.getvalue()
                st.session_state[get_state_key(file_type, 'original_file_name')] = uploaded_file.name
            st.session_state[get_state_key(file_type, 'original_df')] = df
            # Don't clear processed_df here - only clear on new file upload
            if not already_processed:
                st.session_state[get_state_key(file_type, 'processed_df')] = None
                st.session_state[get_state_key(file_type, 'processing_complete')] = False
        
        # Show loaded file info (only for files not yet processed - processed files show count in format card)
        if not already_processed:
            render_html_card(
                f"✅ <strong>Loaded {len(df)} {file_type}s</strong><br>"
                f"📊 Columns: {len(df.columns)} | Data quality: {completeness:.1f}%",
                "success-card"
            )
            
            # Show data quality explanation if not 100%
            if completeness < 100:
                render_data_quality_expander(df, file_type, completeness)
        
        # Show warnings
        for warning in warnings:
            render_html_card(f"⚠️ {warning}", "token-warning")
        
        # Normalize button for PRE format (only show if not already processed and not currently processing)
        is_current_type_processing = st.session_state.get(get_state_key(file_type, 'processing'), False)
        if file_format == 'PRE' and not any_processing and not already_processed and not is_current_type_processing:
            if st.button("🔄 Normalize Data", key=f"normalize_{file_type}", type="primary", use_container_width=True):
                start_processing(file_type)
        
        # Show processing UI if applicable
        status = get_processing_status(file_type)
        messages = get_progress_messages(file_type)
        has_messages = len(messages) > 0
        
        if status != 'idle' or has_messages:
            st.markdown("---")
            render_processing_ui(file_type, messages)
        
        # Run processing if flag is set
        is_processing = st.session_state.get(get_state_key(file_type, 'processing'), False)
        is_complete = st.session_state.get(get_state_key(file_type, 'processing_complete'), False)
        
        if is_processing and not is_complete:
            run_file_processing(file_type)
        
        # Show download section (only if not already shown after "Using normalized file" message)
        # Only show here if file is POST format or if we haven't shown it yet
        if file_format == 'POST' or not already_processed:
            render_download_section(file_type)
    
    except Exception as e:
        render_html_card(
            f"❌ <strong>Error loading file:</strong> {str(e)}<br>"
            "<small>Make sure it's a valid Excel (.xlsx) file</small>",
            "conflict-warning"
        )


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================
def render_upload_tab():
    """Render Tab 3: Upload Mentee & Mentor Files"""
    
    # Check for API key
    if not st.session_state.api_key:
        render_html_card(
            "⚠️ <strong>API Key Required</strong><br>"
            "Please configure your OpenAI API key in Tab 1 before uploading data files.",
            "conflict-warning"
        )
        st.stop()
    
    # Initialize session state
    init_session_state()
    
    # Inject CSS
    st.markdown(LOG_CONTAINER_CSS, unsafe_allow_html=True)
    
    # Info card
    render_html_card(
        "Upload your mentee and mentor Excel files. Both files are required to proceed with matching."
        "<br><small>Files in PRE format will need normalization. Files in POST format can be used directly.</small>",
        "info-card"
    )
    
    # Check processing states
    any_processing = st.session_state.processing_mentee or st.session_state.processing_mentor
    
    # Render side-by-side columns
    col1, col2 = st.columns(2)
    
    with col1:
        render_file_upload_section(
            file_type='mentee',
            any_processing=any_processing,
            other_type_processing=st.session_state.processing_mentor
        )
    
    with col2:
        render_file_upload_section(
            file_type='mentor',
            any_processing=any_processing,
            other_type_processing=st.session_state.processing_mentee
        )
    
    st.divider()