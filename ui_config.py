"""
UI Component for Tab 1: Configure Settings
Handles API key configuration, model selection, batch size, and stateful mode settings
"""

import streamlit as st
from datetime import datetime
from utils import fetch_available_models, get_model_config, is_debug_mode


def render_config_tab():
    """Render Tab 1: Configure Settings"""
    
    # API Key status (only show in debug mode)
    if is_debug_mode():
        if st.session_state.api_key:
            st.markdown(f"""
            <div class="success-card">
                ✅ OpenAI API key loaded from environment variable <code>OPENAI_API_KEY</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="conflict-warning">
                ⚠️ <strong>Missing API Key</strong><br>
                Please set the <code>OPENAI_API_KEY</code> environment variable.<br>
                Get your key at <a href="https://platform.openai.com/api-keys" target="_blank">https://platform.openai.com/api-keys</a>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
    else:
        # In user mode, still check for API key but don't show message
        if not st.session_state.api_key:
            st.error("⚠️ API key not configured. Please set the OPENAI_API_KEY environment variable.")
            st.stop()
    
    st.markdown("""
    <div class="info-card">
        Configure your AI model settings, batch processing options, and stateful mode. These settings control how the matching algorithm processes your data.
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selection (dynamic from API)
    st.subheader("🤖 AI Model Selection")
    
    # Fetch available models from OpenAI
    with st.spinner("Fetching available models from OpenAI..."):
        available_models = fetch_available_models(st.session_state.api_key)
    
    allowed_models = [
        'gpt-5.1',
        'gpt-5-mini', 
        'gpt-5-nano',
        'gpt-4.1',
        'gpt-4.1-mini',
        'gpt-4.1-nano'
    ]
    
    if available_models:
        filtered_models = {k: v for k, v in available_models.items() if k in allowed_models}
        
        if not filtered_models:
            st.warning(f"⚠️ None of the recommended models are available. Using all models.")
            filtered_models = available_models
        
        sorted_model_ids = sorted(
            filtered_models.keys(),
            key=lambda x: filtered_models[x].get("created", 0),
            reverse=True
        )
        
        if st.session_state.model_choice not in sorted_model_ids:
            st.session_state.model_choice = sorted_model_ids[0]
        
        available_models = filtered_models
        
        # Model selector - using key parameter to prevent jumping
        def format_model_name(model_id):
            """Format model name with creation date"""
            model_info = available_models[model_id]
            created_date = datetime.fromtimestamp(model_info.get("created", 0)).strftime("%Y-%m-%d")
            return f"{model_info['name']} (created: {created_date})"
        
        selected_model = st.selectbox(
            "Choose AI Model (sorted newest → oldest)",
            sorted_model_ids,
            index=sorted_model_ids.index(st.session_state.model_choice),
            format_func=format_model_name,
            key="model_selector",
            help="Select the OpenAI model to use for matching. Models are sorted by creation date (newest first)."
        )
        
        if selected_model != st.session_state.model_choice:
            st.session_state.model_choice = selected_model
        
        model_info = available_models[selected_model]
        
        if is_debug_mode():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quality", model_info["quality"])
            with col2:
                st.metric("Speed", model_info["speed"])
            with col3:
                st.caption("**Cost**")
                st.caption(model_info["cost"])
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Quality", model_info["quality"])
            with col2:
                st.metric("Speed", model_info["speed"])
        
        selected_config = get_model_config(selected_model)
        # Reasoning model warning (only show in debug mode)
        if is_debug_mode():
            if selected_config.get('is_reasoning_model', False) and not selected_config.get('supports_structured_output', True):
                st.error(f"""
                ⚠️ **{selected_model} is a reasoning model that does not support structured JSON output.**
                
                This model may fail or return incomplete responses for the matching system.
                
                **Recommended alternatives:**
                - **gpt-4.1-mini** (best balance of speed, cost, and reliability)
                - **gpt-4.1-nano** (fastest, cheapest)
                - **gpt-5.1** or **gpt-4.1** (highest quality, but rate limits may require stateless mode)
                """)
    else:
        st.error("Failed to fetch models. Using default: gpt-5.1")
        st.session_state.model_choice = "gpt-5.1"
    
    st.divider()
    
    # Stateful mode toggle
    st.subheader("🔗 Stateful Batch Processing (Responses API)")
    
    # Disable stateful mode for full models (gpt-5.1, gpt-4.1)
    selected_model = st.session_state.model_choice
    is_full_model = selected_model in ['gpt-5.1', 'gpt-4.1']
    
    if is_full_model:
        # Show disabled checkbox with explanation
        use_stateful = st.checkbox(
            "Enable stateful mode (store=True)",
            value=False,
            disabled=True,
            help=f"Stateful mode is not available for {selected_model} (high rate limits, not needed)"
        )
        st.session_state.use_stateful_mode = False
        if is_debug_mode():
            st.info(f"ℹ️ Stateful mode is disabled for {selected_model} (high rate limits, not needed)")
    else:
        use_stateful = st.checkbox(
            "Enable stateful mode (store=True)",
            value=st.session_state.get('use_stateful_mode', False),
            help="Stateful mode uses context chaining to dramatically reduce token usage in batch processing"
        )
        st.session_state.use_stateful_mode = use_stateful
    
    # Stateful mode detailed explanation (only show in debug mode)
    if is_debug_mode():
        if use_stateful:
            st.markdown("""
            <div class="success-card">
                <strong>✅ Stateful Mode Enabled</strong><br>
                <strong>How it works:</strong><br>
                • First batch: Full mentee data + first mentor batch (normal token count)<br>
                • Subsequent batches: Only new mentors (~90% token savings)<br>
                • Context preserved via <code>previous_response_id</code> chaining<br><br>
                <strong>Benefits:</strong><br>
                • Massive token savings (est. 70-90% reduction for large datasets)<br>
                • Lower API costs for multi-batch operations<br>
                • Faster processing with smaller prompts<br><br>
                <strong>Requirements:</strong><br>
                • Responses API must be available (test with test_responses_api.py)<br>
                • All batches must use the same model and settings
            </div>
            """, unsafe_allow_html=True)
        elif not is_full_model:
            st.info("💡 **Tip:** Enable stateful mode to save ~70-90% on tokens for large datasets with multiple batches")
    
    st.divider()
    
    # Matching Constraints
    st.subheader("⚖️ Matching Constraints")
    
    # Initialize max_mentees_per_mentor if not set
    if 'max_mentees_per_mentor' not in st.session_state:
        st.session_state.max_mentees_per_mentor = 2
    
    max_mentees = st.number_input(
        "Maximum Mentees per Mentor",
        min_value=1,
        max_value=3,
        value=st.session_state.max_mentees_per_mentor,
        step=1,
        help="Set the maximum number of mentees that can be assigned to a single mentor. Range: 1-3."
    )
    
    st.session_state.max_mentees_per_mentor = max_mentees
    
    if is_debug_mode():
        st.info(f"ℹ️ Each mentor can be assigned to a maximum of {max_mentees} mentee(s). This constraint is enforced by the Hungarian algorithm during optimal assignment.")

