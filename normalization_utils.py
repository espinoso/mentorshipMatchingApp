"""
Utility functions for file format detection and normalization integration.
"""

import pandas as pd
import streamlit as st
from typing import Tuple, Optional, Dict, Any, Callable
from io import BytesIO
import tempfile
import os
import sys
from pathlib import Path
import importlib.util
import time

from info import MENTEE_COLUMNS, MENTOR_COLUMNS


def detect_file_format(df: pd.DataFrame, file_type: str) -> str:
    """
    Detect if uploaded file is in PRE or POST format.
    
    Args:
        df: DataFrame to check
        file_type: 'mentee' or 'mentor'
    
    Returns:
        'POST' if file has expected POST format columns, 'PRE' otherwise
    """
    if file_type == 'mentee':
        required_columns = MENTEE_COLUMNS
    else:
        required_columns = MENTOR_COLUMNS
    
    # Check if all required POST format columns are present
    # We check for the key columns: id, field, specialization
    key_columns = ['id', 'field', 'specialization']
    
    for key in key_columns:
        expected_col = required_columns[key]
        # Check exact match
        if expected_col not in df.columns:
            # Check case-insensitive match
            found = False
            for col in df.columns:
                if str(col).strip().lower() == expected_col.strip().lower():
                    found = True
                    break
            if not found:
                return 'PRE'
    
    # If all key columns are present, it's POST format
    return 'POST'


def save_uploaded_file_to_temp(uploaded_file) -> str:
    """
    Save Streamlit uploaded file to temporary file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Path to temporary file
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    temp_path = temp_file.name
    temp_file.close()
    
    # Write uploaded file content to temp file
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    return temp_path


def load_normalization_modules():
    """Load normalization modules dynamically."""
    normalization_dir = Path(__file__).parent / "Normalization"
    
    # Add Normalization directory to sys.path so relative imports work
    if str(normalization_dir) not in sys.path:
        sys.path.insert(0, str(normalization_dir))
    
    # Load shared_utils
    shared_utils_path = normalization_dir / "shared_utils.py"
    spec = importlib.util.spec_from_file_location("shared_utils", shared_utils_path)
    shared_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared_utils)
    
    # Make it available as 'shared_utils' module for relative imports
    sys.modules['shared_utils'] = shared_utils
    
    # Load normalization
    normalization_path = normalization_dir / "normalization.py"
    spec_norm = importlib.util.spec_from_file_location("normalization", normalization_path)
    normalization_module = importlib.util.module_from_spec(spec_norm)
    spec_norm.loader.exec_module(normalization_module)
    
    # Make it available as 'normalization' module for relative imports
    sys.modules['normalization'] = normalization_module
    
    # Load institution_types
    institution_types_path = normalization_dir / "institution_types.py"
    spec_inst = importlib.util.spec_from_file_location("institution_types", institution_types_path)
    institution_types = importlib.util.module_from_spec(spec_inst)
    spec_inst.loader.exec_module(institution_types)
    
    sys.modules['institution_types'] = institution_types
    
    return shared_utils, normalization_module, institution_types


def validate_pre_format_columns(df: pd.DataFrame, file_type: str) -> Tuple[bool, list]:
    """
    Validate that file has required PRE format columns.
    
    Args:
        df: DataFrame to validate
        file_type: 'mentee' or 'mentor'
    
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    # Load shared_utils to use find_column_flexible (same as normalization scripts)
    normalization_dir = Path(__file__).parent / "Normalization"
    shared_utils_path = normalization_dir / "shared_utils.py"
    spec = importlib.util.spec_from_file_location("shared_utils", shared_utils_path)
    shared_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared_utils)
    find_column_flexible = shared_utils.find_column_flexible
    
    if file_type == 'mentee':
        # Mentee PRE column mapping (from merge_mentee_skills_enhanced.py)
        PRE_COLUMN_MAPPING = {
            'affiliation': ['university where you are currently studying'],
            'country': ['country of your affiliated university'],
            'education_level': ['highest educational level', 'highest education level'],
            'current_program': ['current education program'],
            'field': ['field of expertise'],
            'specialization': ['specialization'],
            'hard_skills': ['specific hard skills'],
            'languages': ['languages spoken fluently'],
            'guidance_areas': ['areas where guidance is needed'],
            'career_goals': ['career goals for the next 2-3 years', 'career goals'],
            'other_info': ['is there anything else', 'anything else']
        }
        # Critical columns for mentees (must be present)
        critical_columns = ['education_level', 'current_program', 'field', 'specialization', 'hard_skills']
    else:
        # Mentor PRE column mapping (from merge_mentor_skills_enhanced.py)
        PRE_COLUMN_MAPPING = {
            'affiliation': ['affiliation'],
            'country': ['country of your affiliated institution'],
            'current_position': ['current position'],
            'institution_type': ['type of work'],
            'country_of_origin': ['country of origin'],
            'education_level': ['highest educational level', 'highest education level'],
            'alma_mater': ['institution & country where bachelor', 'bachelor\'s degree'],
            'field': ['field of expertise'],
            'specialization': ['specialization'],
            'hard_skills': ['specific hard skills', 'professional competencies'],
            'experience_years': ['years of professional experience'],
            'career_dev_experience': ['areas where you feel comfortable mentoring', 'comfortable mentoring']
        }
        # Critical columns for mentors (must be present)
        critical_columns = ['education_level', 'field', 'specialization', 'hard_skills', 'institution_type']
    
    # Use find_column_flexible to check for columns (same logic as normalization scripts)
    missing_columns = []
    
    for key in critical_columns:
        if key in PRE_COLUMN_MAPPING:
            search_terms = PRE_COLUMN_MAPPING[key]
            found_col = find_column_flexible(df, search_terms)
            if found_col is None:
                missing_columns.append(f"{key} (looking for: {', '.join(search_terms)})")
    
    is_valid = len(missing_columns) == 0
    return is_valid, missing_columns


def create_streamlit_progress_handler(progress_placeholder, status_placeholder, progress_state=None, session_state_key=None):
    """
    Create a progress handler that updates Streamlit UI and tracks progress.
    
    Args:
        progress_placeholder: Streamlit placeholder for progress messages
        status_placeholder: Streamlit placeholder for status
        progress_state: Dictionary to store progress state (messages, current_status, processed_rows)
        session_state_key: Key in session_state to update (e.g., 'mentee_progress_state')
    
    Returns:
        Function that can be used to replace print() calls
    """
    messages = []
    if progress_state is None:
        progress_state = {'messages': [], 'current_status': '', 'processed_rows': 0, 'total_rows': 0}
    
    def progress_handler(*args, **kwargs):
        """Capture print statements and update Streamlit"""
        # Handle print with end=" " (same line) - check if end is in kwargs
        end_char = kwargs.get('end', '\n')
        flush = kwargs.get('flush', False)
        
        # Join all arguments into message
        message = ' '.join(str(arg) for arg in args)
        
        # If end is not newline, append to last message (same line printing)
        if end_char != '\n' and messages:
            # Append to last message (same line)
            messages[-1] = messages[-1] + message
            # Update current status with combined message
            progress_state['current_status'] = messages[-1]
        else:
            # New line - add as new message
            messages.append(message)
            progress_state['current_status'] = message
        
        progress_state['messages'] = messages
        
        # Extract row number from messages like "[1/100]" or "Row 1:"
        import re
        # Check last message for row progress
        last_msg = messages[-1] if messages else ""
        row_match = re.search(r'\[(\d+)/(\d+)\]', last_msg)
        if row_match:
            progress_state['processed_rows'] = int(row_match.group(1))
            progress_state['total_rows'] = int(row_match.group(2))
        else:
            row_match = re.search(r'Row\s+(\d+)', last_msg, re.IGNORECASE)
            if row_match and progress_state['total_rows'] == 0:
                # Try to estimate total from context
                pass
        
        # Update session state if key provided (for modal updates)
        if session_state_key and hasattr(st, 'session_state'):
            if session_state_key in st.session_state:
                st.session_state[session_state_key] = progress_state.copy()
        
        # Update status placeholder with latest message
        if status_placeholder and messages:
            # Show a cleaner status (extract step name if possible)
            status_display = messages[-1]
            # Clean up status display
            if status_display.startswith('['):
                # It's a progress indicator, show it as-is
                status_placeholder.text(f"Status: {status_display}")
            else:
                status_placeholder.text(f"Status: {status_display[:100]}...")
        
        # Update progress placeholder with recent messages
        if progress_placeholder:
            # Show last 30 messages for better visibility
            recent_messages = messages[-30:] if len(messages) > 30 else messages
            with progress_placeholder.container():
                # Show as code block for terminal-like appearance
                log_text = '\n'.join(recent_messages)
                st.code(log_text, language=None)
    
    return progress_handler, messages, progress_state


def process_file_with_progress(
    uploaded_file,
    file_type: str,
    api_key: str,
    progress_container=None,
    status_container=None,
    cancel_flag=None
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[dict]]:
    """
    Process uploaded file with Streamlit progress tracking.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        file_type: 'mentee' or 'mentor'
        api_key: OpenAI API key
        progress_container: Streamlit container for progress messages
        status_container: Streamlit container for current status
        cancel_flag: Dictionary with 'cancelled' key to check for cancellation
    
    Returns:
        Tuple of (processed_df, output_file_path, processing_info)
        Returns (None, None, None) on error or cancellation
    """
    import time
    start_time = time.time()
    
    try:
        # Load normalization modules first (this sets up sys.path and sys.modules)
        normalization_dir = Path(__file__).parent / "Normalization"
        shared_utils, normalization_module, institution_types = load_normalization_modules()
        
        # Save uploaded file to temp location
        temp_input_path = save_uploaded_file_to_temp(uploaded_file)
        
        try:
            # Setup OpenAI client (using the old API style for compatibility)
            import openai
            openai.api_key = api_key
            
            # Initialize cache and report
            cache = shared_utils.AICache()
            report = shared_utils.TransformationReport(f"{file_type.capitalize()} Normalization")
            
            # Create temp output file
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
            temp_output_path = temp_output.name
            temp_output.close()
            
            # Use provided placeholders or create new ones if not provided
            if progress_container is None:
                progress_placeholder = None
            else:
                progress_placeholder = progress_container  # Use the provided placeholder
            
            if status_container is None:
                status_placeholder = None
            else:
                status_placeholder = status_container  # Use the provided placeholder
            
            # Initialize progress state
            progress_state = {'messages': [], 'current_status': '', 'processed_rows': 0, 'total_rows': 0}
            
            # Determine session state key for progress updates
            session_state_key = f'{file_type}_progress_state'
            
            # Create progress handler
            progress_handler, messages, progress_state = create_streamlit_progress_handler(
                progress_placeholder, status_placeholder, progress_state, session_state_key
            )
            
            # Add initial message to show processing has started
            initial_msg = f"Starting {file_type} file normalization..."
            messages.append(initial_msg)
            progress_state['messages'] = messages
            progress_state['current_status'] = initial_msg
            if session_state_key and hasattr(st, 'session_state'):
                st.session_state[session_state_key] = progress_state.copy()
            
            # Temporarily replace print to capture progress
            import builtins
            original_print = builtins.print
            builtins.print = progress_handler
            
            try:
                # Import the processing function (after modules are loaded)
                if file_type == 'mentee':
                    # Import from the Normalization directory
                    mentee_module_path = normalization_dir / "merge_mentee_skills_enhanced.py"
                    spec_mentee = importlib.util.spec_from_file_location("merge_mentee_skills_enhanced", mentee_module_path)
                    mentee_module = importlib.util.module_from_spec(spec_mentee)
                    spec_mentee.loader.exec_module(mentee_module)
                    process_mentee_file = mentee_module.process_mentee_file
                    
                    result = process_mentee_file(
                        input_file=temp_input_path,
                        output_file=temp_output_path,
                        dry_run=False,
                        test_rows=None,
                        client=openai,
                        cache=cache,
                        report=report
                    )
                else:
                    # Import from the Normalization directory
                    mentor_module_path = normalization_dir / "merge_mentor_skills_enhanced.py"
                    spec_mentor = importlib.util.spec_from_file_location("merge_mentor_skills_enhanced", mentor_module_path)
                    mentor_module = importlib.util.module_from_spec(spec_mentor)
                    spec_mentor.loader.exec_module(mentor_module)
                    process_mentor_file = mentor_module.process_mentor_file
                    
                    result = process_mentor_file(
                        input_file=temp_input_path,
                        output_file=temp_output_path,
                        dry_run=False,
                        test_rows=None,
                        client=openai,
                        cache=cache,
                        report=report
                    )
                
                # Restore original print
                builtins.print = original_print
                
                # Add completion message
                completion_msg = f"Processing completed. Total messages captured: {len(messages)}"
                messages.append(completion_msg)
                progress_state['messages'] = messages
                progress_state['current_status'] = completion_msg
                if session_state_key and hasattr(st, 'session_state'):
                    st.session_state[session_state_key] = progress_state.copy()
                
                if result is None:
                    return None, None, {
                        'error': 'Processing failed',
                        'messages': messages,
                        'progress_state': progress_state
                    }
                
                # Check for cancellation
                if cancel_flag and cancel_flag.get('cancelled', False):
                    # Restore original print
                    builtins.print = original_print
                    return None, None, {
                        'error': 'Processing cancelled by user',
                        'cancelled': True,
                        'messages': messages,
                        'progress_state': progress_state
                    }
                
                if file_type == 'mentee':
                    df_output, output_file, unmapped_values = result
                    processing_info = {
                        'unmapped_values': unmapped_values,
                        'report': report,
                        'cache_stats': cache.get_stats(),
                        'messages': messages.copy(),  # Make a copy to ensure it's captured
                        'start_time': start_time,
                        'elapsed_time': time.time() - start_time,
                        'progress_state': progress_state.copy()  # Make a copy
                    }
                else:
                    df_output, output_file, transformations, unmapped_countries, unmapped_countries_of_origin, unmapped_alma_maters, unmapped_fields = result
                    processing_info = {
                        'transformations': transformations,
                        'unmapped_countries': unmapped_countries,
                        'unmapped_countries_of_origin': unmapped_countries_of_origin,
                        'unmapped_alma_maters': unmapped_alma_maters,
                        'unmapped_fields': unmapped_fields,
                        'report': report,
                        'cache_stats': cache.get_stats(),
                        'messages': messages,
                        'start_time': start_time,
                        'elapsed_time': time.time() - start_time,
                        'progress_state': progress_state
                    }
                
                # Read the output file into DataFrame
                processed_df = pd.read_excel(temp_output_path)
                
                # Clean up temp files
                try:
                    os.unlink(temp_input_path)
                    os.unlink(temp_output_path)
                except:
                    pass
                
                return processed_df, output_file, processing_info
                
            except Exception as e:
                # Restore original print
                builtins.print = original_print
                raise e
                
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_input_path)
            except:
                pass
            raise e
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, None, {'error': str(e), 'traceback': error_details}


def create_download_bytes(df: pd.DataFrame) -> BytesIO:
    """
    Create BytesIO object with Excel file content for download.
    
    Args:
        df: DataFrame to download
    
    Returns:
        BytesIO object with Excel file content
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output
