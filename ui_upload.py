"""
Per-entity upload + normalization step (one step for Mentees, one for Mentors).

One file per entity: upload -> auto schema check -> (if a raw PRE file) normalize with a
progress bar -> ready, with a preview folded in. Removing the file resets the step. The
normalization engine (run_file_processing / process_file_with_progress) is unchanged; this
module is the calm presentation around it.
"""

import time
from io import BytesIO
from typing import Literal

import pandas as pd
import streamlit as st

from info import MENTEE_COLUMNS, MENTOR_COLUMNS
from normalization_utils import (
    detect_file_format,
    process_file_with_progress,
    validate_pre_format_columns,
)
from utils import clean_dataframe, validate_uploaded_file

FileType = Literal['mentee', 'mentor']


def _store_df(df_key, df):
    """Store an uploaded dataframe; a changed one invalidates any earlier results."""
    old = st.session_state.get(df_key)
    if old is None or not old.equals(df):  # POST files are re-stored on every rerun
        for k in ('matrix_df', 'assignments', 'matches'):
            st.session_state[k] = None
    st.session_state[df_key] = df


FILE_TYPE_CONFIG = {
    'mentee': {
        'icon': '👥',
        'title': 'Mentees File',
        'columns': MENTEE_COLUMNS,
        'df_key': 'mentees_df',
    },
    'mentor': {
        'icon': '👨‍🏫',
        'title': 'Mentors File',
        'columns': MENTOR_COLUMNS,
        'df_key': 'mentors_df',
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
# STATE HELPERS
# =============================================================================
def init_session_state():
    """Initialize all session state variables with defaults."""
    for key, default in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
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


def get_progress_messages(file_type: FileType) -> list[str]:
    """Get progress messages from processing info or progress state."""
    messages = []
    processing_info = st.session_state.get(get_state_key(file_type, 'processing_info'))
    if processing_info:
        messages = processing_info.get('messages', [])
    if not messages:
        progress_state = st.session_state.get(get_state_key(file_type, 'progress_state'), {})
        messages = progress_state.get('messages', [])
    return messages


def start_processing(file_type: FileType):
    """Initialize processing state for a file type and rerun into the processing view."""
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
    st.rerun()


def _reset_entity(file_type: FileType):
    """Remove everything for this entity — removing the file is starting from scratch."""
    for suffix in ('processed_df', 'processing_info', 'original_df', 'original_file_bytes', 'original_file_name'):
        st.session_state[get_state_key(file_type, suffix)] = None
    st.session_state[get_state_key(file_type, 'processing_complete')] = False
    st.session_state[get_state_key(file_type, 'processing')] = False
    st.session_state[FILE_TYPE_CONFIG[file_type]['df_key']] = None
    for k in ('matrix_df', 'assignments', 'matches'):  # a removed entity invalidates any results
        st.session_state[k] = None


# =============================================================================
# PROCESSING ENGINE (unchanged)
# =============================================================================
def run_file_processing(file_type: FileType):
    """Execute file processing for the given file type."""
    original_df = st.session_state.get(get_state_key(file_type, 'original_df'))
    if original_df is not None:
        total_rows = len(original_df)
        progress_state = st.session_state.get(get_state_key(file_type, 'progress_state'), {})
        progress_state['total_rows'] = total_rows
        st.session_state[get_state_key(file_type, 'progress_state')] = progress_state

    # Create placeholders for real-time updates (the handler renders a progress bar into these)
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
        _store_df(config['df_key'], processed_df)
        st.session_state[get_state_key(file_type, 'processing_info')] = processing_info
        st.session_state[get_state_key(file_type, 'cancel_flag')]['cancelled'] = False
        st.session_state[get_state_key(file_type, 'processing_complete')] = True
        progress_placeholder.empty()
        status_placeholder.empty()
        st.rerun()
    else:
        error_msg = processing_info.get('error', 'Unknown error') if processing_info else 'Unknown error'
        st.session_state[get_state_key(file_type, 'processing_start_time')] = None
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error(f"❌ Normalization failed: {error_msg}")
        if processing_info and processing_info.get('messages'):
            with st.expander("View processing log"):
                st.code('\n'.join(processing_info['messages']), language=None)
        if processing_info and 'traceback' in processing_info:
            with st.expander("Error details"):
                st.code(processing_info['traceback'])


# =============================================================================
# CALM PRESENTATION
# =============================================================================
def _preview(df: pd.DataFrame, file_type: FileType):
    """Compact data preview, folded in from the old Data Overview step."""
    st.caption(f"{len(df)} {file_type}s · {len(df.columns)} columns")
    show_all = st.toggle("Show all rows", key=f"showall_{file_type}", value=False)
    st.dataframe(df if show_all else df.head(5), use_container_width=True, hide_index=True)


def _failed_rows(file_type: FileType) -> int:
    info = st.session_state.get(get_state_key(file_type, 'processing_info'))
    if info and 'report' in info:
        return info['report'].stats.get('failed_rows', 0)
    return 0


def render_entity_step(file_type: FileType):
    """The per-entity step: upload -> check -> (normalize) -> ready + preview.

    Completed data is read from the stored dataframe (`df_key`), which survives navigation —
    the file_uploader widget is unmounted when you switch steps, so we never key "is there data"
    off it. Removing/replacing a file is an explicit button, not an inferred empty uploader.
    """
    init_session_state()
    config = FILE_TYPE_CONFIG[file_type]
    name = config['title'].replace(' File', '')  # "Mentees" / "Mentors"
    st.subheader(f"{config['icon']} {name}")

    # Mid-normalization: show a progress bar and run the engine (it reruns on completion).
    if st.session_state.get(get_state_key(file_type, 'processing'), False):
        st.markdown(f"**Normalizing {name.lower()}…** This can take a minute; please keep the tab open.")
        run_file_processing(file_type)
        return

    # Already have usable data for this entity → show it. Persists across step navigation.
    ready_df = st.session_state.get(config['df_key'])
    if ready_df is not None:
        failed = _failed_rows(file_type)
        if failed:
            st.warning(f"⚠️ {failed} row(s) had issues during normalization.")
        st.success(f"✓ {len(ready_df)} {file_type}s ready.")
        _preview(ready_df, file_type)
        if st.button("Replace file", key=f"replace_{file_type}"):
            _reset_entity(file_type)
            nonce_key = get_state_key(file_type, 'upload_nonce')
            st.session_state[nonce_key] = st.session_state.get(nonce_key, 0) + 1
            st.rerun()
        return

    # No data yet → show the uploader.
    nonce = st.session_state.get(get_state_key(file_type, 'upload_nonce'), 0)
    uploaded = st.file_uploader(
        f"Upload the {file_type}s Excel file",
        type=['xlsx'],
        key=f"{file_type}_upload_{nonce}",
        help=f"Must include a '{config['columns']['id']}' column. A raw survey export or an already-normalized file both work.",
    )
    if uploaded is None:
        st.info(f"Upload the {file_type}s file to begin. It will be checked and, if needed, normalized.")
        return

    # A file is present: read, detect format, validate the schema.
    try:
        df = clean_dataframe(pd.read_excel(uploaded))
    except Exception as e:
        st.error(f"Couldn't read that file: {e!s}. Make sure it's a valid Excel (.xlsx).")
        return

    file_format = detect_file_format(df, file_type)
    if file_format == 'POST':
        errors, _warnings, _completeness = validate_uploaded_file(df, file_type)
    else:
        is_valid_pre, missing = validate_pre_format_columns(df, file_type)
        errors = [] if is_valid_pre else [f"Missing required column: {c}" for c in missing]

    if errors:
        for err in errors:
            st.error(f"❌ {err}")
        st.caption("Upload a file with the required columns — a raw survey export or an already-normalized file.")
        return

    # Valid. Already-normalized (POST) files are ready immediately (df_key set -> ready view).
    if file_format == 'POST':
        _store_df(config['df_key'], df)
        st.rerun()

    # Raw (PRE) file: offer normalization.
    st.success(f"✓ File looks valid — {len(df)} rows. It needs normalizing before matching.")
    st.session_state[get_state_key(file_type, 'original_df')] = df
    st.session_state[get_state_key(file_type, 'original_file_bytes')] = uploaded.getvalue()
    st.session_state[get_state_key(file_type, 'original_file_name')] = uploaded.name
    if st.button(f"Normalize {name.lower()}", type="primary", use_container_width=True):
        start_processing(file_type)
