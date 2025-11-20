"""
Progress UI Components for Tab 7 Processing
Provides visual progress indicators, batch status cards, and time tracking
"""

import streamlit as st
import time
from utils import format_time, calculate_eta, is_debug_mode


def initialize_progress_tracking(total_batches):
    """Initialize progress tracking in session state
    
    Args:
        total_batches: Total number of batches to process
    """
    st.session_state.batch_progress = {
        'total_batches': total_batches,
        'completed_batches': 0,
        'current_batch': 0,
        'start_time': time.time(),
        'batch_times': [],
        'batch_statuses': {},  # {batch_num: 'complete'|'processing'|'pending'|'error'}
        'batch_details': {}  # {batch_num: {mentors, mentees, scores, tokens, cost, etc.}}
    }


def render_progress_overview(container=None):
    """Render overall progress overview card at the top
    
    Args:
        container: Optional Streamlit container to render into (for updating in place)
    """
    if 'batch_progress' not in st.session_state:
        return
    
    progress = st.session_state.batch_progress
    completed = progress['completed_batches']
    total = progress['total_batches']
    current = progress['current_batch']
    progress_pct = (completed / total) * 100 if total > 0 else 0
    
    elapsed = time.time() - progress['start_time']
    avg_time = sum(progress['batch_times']) / len(progress['batch_times']) if progress['batch_times'] else None
    eta = calculate_eta(completed, total, progress['batch_times']) if avg_time and completed < total else None
    
    # Use container if provided, otherwise use st
    render_target = container if container else st
    
    # Progress overview card
    render_target.markdown("""
    <div style="background: white; border: 1px solid #E0E0E0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
        <h3 style="margin-top: 0; color: #E6122C; font-size: 1.2rem;">📊 Processing Progress</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar and metrics
    col1, col2, col3 = render_target.columns([2, 1, 1])
    
    with col1:
        render_target.progress(progress_pct / 100)
        status_text = f"Batch {completed}/{total} complete ({progress_pct:.0f}%)"
        if current > completed:
            status_text += f" | Current: Batch {current}/{total}"
        render_target.caption(status_text)
    
    with col2:
        render_target.metric("⏱️ Elapsed", format_time(elapsed))
    
    with col3:
        if eta:
            render_target.metric("⏳ ETA", format_time(eta))
        elif completed >= total:
            render_target.metric("✅ Complete", format_time(elapsed))
        else:
            render_target.metric("⏳ ETA", "Calculating...")
    
    # Statistics dashboard
    if progress['batch_times']:
        avg_time = sum(progress['batch_times']) / len(progress['batch_times'])
        total_mentors = sum(d.get('mentors', 0) for d in progress['batch_details'].values())
        total_scores = sum(d.get('scores', 0) for d in progress['batch_details'].values())
        
        col1, col2, col3, col4 = render_target.columns(4)
        with col1:
            render_target.metric("Batches", f"{completed}/{total}")
        with col2:
            render_target.metric("Mentors", f"{total_mentors}")
        with col3:
            render_target.metric("Scores", f"{total_scores:,}")
        with col4:
            render_target.metric("Avg Time", format_time(avg_time))
    
    render_target.divider()


def render_batch_card(batch_num, status, batch_details=None, is_current=False):
    """Render a batch status card
    
    Args:
        batch_num: Batch number (1-indexed)
        status: 'complete', 'processing', 'pending', or 'error'
        batch_details: Dict with batch information (mentors, mentees, scores, time, etc.)
        is_current: Whether this is the currently processing batch
    """
    status_icons = {
        'complete': '✅',
        'processing': '🔄',
        'pending': '⏳',
        'error': '❌'
    }
    
    status_colors = {
        'complete': '#E8F5E9',  # Light green
        'processing': '#E3F2FD',  # Light blue
        'pending': '#F5F5F5',  # Light gray
        'error': '#FFEBEE'  # Light red
    }
    
    icon = status_icons.get(status, '⏳')
    bg_color = status_colors.get(status, '#F5F5F5')
    
    # Build card content
    if status == 'complete' and batch_details:
        mentors = batch_details.get('mentors', 0)
        mentees = batch_details.get('mentees', 0)
        scores = batch_details.get('scores', 0)
        batch_time = batch_details.get('time', 0)
        
        # Collapsed view for completed batches (user mode)
        if not is_debug_mode():
            with st.expander(f"{icon} Batch {batch_num}/{st.session_state.batch_progress['total_batches']} - Complete", expanded=False):
                st.markdown(f"""
                **Metrics:** {mentors} mentors | {mentees} mentees | {scores:,} scores  
                **Time:** {format_time(batch_time)}
                """)
        else:
            # Expanded view for debug mode
            st.markdown(f"""
            <div style="background: {bg_color}; border: 1px solid #E0E0E0; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;">
                <h4 style="margin-top: 0; color: #3A3A3A;">{icon} Batch {batch_num}/{st.session_state.batch_progress['total_batches']} - Complete</h4>
                <p><strong>Metrics:</strong> {mentors} mentors | {mentees} mentees | {scores:,} scores</p>
                <p><strong>Time:</strong> {format_time(batch_time)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Debug details
            if batch_details.get('tokens'):
                st.caption(f"Tokens: {batch_details['tokens']:,} | Cost: ${batch_details.get('cost', 0):.4f}")
            if batch_details.get('response_id'):
                st.caption(f"Response ID: {batch_details['response_id'][:30]}...")
    
    elif status == 'processing':
        # Current batch - always visible
        st.markdown(f"""
        <div style="background: {bg_color}; border: 2px solid #2196F3; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;">
            <h4 style="margin-top: 0; color: #3A3A3A;">{icon} Batch {batch_num}/{st.session_state.batch_progress['total_batches']} - Processing...</h4>
        </div>
        """, unsafe_allow_html=True)
    
    elif status == 'pending':
        # Pending batches - minimal display
        if is_debug_mode():
            st.markdown(f"""
            <div style="background: {bg_color}; border: 1px solid #E0E0E0; border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem; opacity: 0.6;">
                <p style="margin: 0; color: #9E9E9E;">{icon} Batch {batch_num}/{st.session_state.batch_progress['total_batches']} - Pending</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif status == 'error':
        st.markdown(f"""
        <div style="background: {bg_color}; border: 1px solid #F44336; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;">
            <h4 style="margin-top: 0; color: #C62828;">{icon} Batch {batch_num}/{st.session_state.batch_progress['total_batches']} - Error</h4>
            <p style="color: #C62828;">{batch_details.get('error', 'Unknown error') if batch_details else 'Unknown error'}</p>
        </div>
        """, unsafe_allow_html=True)


def update_batch_progress(batch_num, status, batch_details=None):
    """Update progress tracking for a batch
    
    Args:
        batch_num: Batch number (1-indexed)
        status: 'complete', 'processing', 'pending', or 'error'
        batch_details: Dict with batch information
    """
    if 'batch_progress' not in st.session_state:
        return
    
    progress = st.session_state.batch_progress
    progress['batch_statuses'][batch_num] = status
    progress['current_batch'] = batch_num
    
    if status == 'complete' and batch_details:
        progress['batch_details'][batch_num] = batch_details
        if 'time' in batch_details:
            progress['batch_times'].append(batch_details['time'])
        progress['completed_batches'] = len([s for s in progress['batch_statuses'].values() if s == 'complete'])
    
    elif status == 'processing':
        progress['current_batch'] = batch_num


def render_all_batches():
    """Render all batch cards in order"""
    if 'batch_progress' not in st.session_state:
        return
    
    progress = st.session_state.batch_progress
    total = progress['total_batches']
    current = progress['current_batch']
    
    for batch_num in range(1, total + 1):
        status = progress['batch_statuses'].get(batch_num, 'pending')
        batch_details = progress['batch_details'].get(batch_num)
        is_current = (batch_num == current)
        
        render_batch_card(batch_num, status, batch_details, is_current)


def render_processing_summary():
    """Render collapsed summary after processing completes"""
    if 'batch_progress' not in st.session_state:
        return
    
    progress = st.session_state.batch_progress
    total = progress['total_batches']
    completed = progress['completed_batches']
    elapsed = time.time() - progress['start_time']
    
    total_mentors = sum(d.get('mentors', 0) for d in progress['batch_details'].values())
    total_mentees = progress['batch_details'].get(1, {}).get('mentees', 0) if progress['batch_details'] else 0
    
    with st.expander("📊 Processing Summary", expanded=False):
        st.markdown(f"""
        ✅ **Processing Complete**
        - {completed}/{total} batches processed
        - Total time: {format_time(elapsed)}
        - {total_mentors} mentors × {total_mentees} mentees
        """)
        
        if is_debug_mode():
            st.divider()
            render_all_batches()

