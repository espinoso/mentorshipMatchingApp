"""
Modal overlay component for file processing.
Blocks UI and shows progress with time estimates.
"""

import streamlit as st
import time
from typing import Optional, Callable
from datetime import datetime, timedelta


def render_processing_modal(
    file_type: str,
    is_processing: bool,
    on_cancel: Optional[Callable] = None,
    status_message: str = "",
    progress_messages: list = None,
    start_time: Optional[float] = None,
    total_rows: int = 0,
    processed_rows: int = 0
):
    """
    Render a full-screen modal overlay for processing.
    
    Args:
        file_type: 'mentee' or 'mentor'
        is_processing: Whether processing is active
        on_cancel: Callback function for cancel button
        status_message: Current status message
        progress_messages: List of progress messages
        start_time: Processing start time (timestamp)
        total_rows: Total rows to process
        processed_rows: Rows processed so far
    """
    if not is_processing:
        return
    
    # Calculate elapsed time and ETA
    elapsed_time = None
    eta = None
    if start_time:
        elapsed_seconds = time.time() - start_time
        elapsed_time = format_time(elapsed_seconds)
        
        if processed_rows > 0 and total_rows > 0:
            # Estimate time per row
            time_per_row = elapsed_seconds / processed_rows
            remaining_rows = total_rows - processed_rows
            eta_seconds = time_per_row * remaining_rows
            eta = format_time(eta_seconds)
    
    # Modal overlay with CSS
    st.markdown("""
    <style>
    .processing-modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .processing-modal-content {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        max-width: 600px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .processing-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .processing-status {
        font-size: 1.1rem;
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .processing-time-info {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #e8f4f8;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .processing-messages {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        max-height: 300px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.85rem;
        margin: 1rem 0;
    }
    .processing-message {
        padding: 0.25rem 0;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Modal content
    st.markdown("""
    <div class="processing-modal-overlay">
        <div class="processing-modal-content">
    """, unsafe_allow_html=True)
    
    # Header
    entity_name = "Mentee" if file_type == 'mentee' else "Mentor"
    st.markdown(f"""
    <div class="processing-header">
        🔄 Processing {entity_name} File
    </div>
    """, unsafe_allow_html=True)
    
    # Status message
    if status_message:
        st.markdown(f"""
        <div class="processing-status">
            <strong>Status:</strong> {status_message}
        </div>
        """, unsafe_allow_html=True)
    
    # Time information
    if elapsed_time:
        time_info = f"⏱️ <strong>Elapsed:</strong> {elapsed_time}"
        if eta:
            time_info += f" | <strong>ETA:</strong> {eta}"
        if processed_rows > 0 and total_rows > 0:
            progress_pct = (processed_rows / total_rows) * 100
            time_info += f" | <strong>Progress:</strong> {processed_rows}/{total_rows} ({progress_pct:.1f}%)"
        
        st.markdown(f"""
        <div class="processing-time-info">
            {time_info}
        </div>
        """, unsafe_allow_html=True)
    
    # Progress messages
    if progress_messages:
        st.markdown("""
        <div class="processing-messages">
        """, unsafe_allow_html=True)
        
        # Show last 15 messages
        recent_messages = progress_messages[-15:] if len(progress_messages) > 15 else progress_messages
        for msg in recent_messages:
            st.markdown(f'<div class="processing-message">{msg}</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Cancel button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("❌ Cancel", key=f"cancel_{file_type}", use_container_width=True):
            if on_cancel:
                on_cancel()
            st.rerun()
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., "2m 34s", "1h 15m")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    if minutes < 60:
        if secs > 0:
            return f"{minutes}m {secs}s"
        return f"{minutes}m"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    if mins > 0:
        return f"{hours}h {mins}m"
    return f"{hours}h"

