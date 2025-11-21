"""
UI Component for Tab 6: Generate Matches Summary
Shows a summary of all settings and data before generating matches
"""

import streamlit as st


def render_matches_tab():
    """Render Tab 6: Generate Matches Summary"""
    
    try:
        data_ready = (
            st.session_state.get('mentees_df') is not None and 
            st.session_state.get('mentors_df') is not None
        )
        
        if not data_ready:
            st.markdown("""
            <div class="info-card">
                Review your configuration and generate compatibility matches. Upload data files in <strong>Tab 2</strong> to begin.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="token-warning">
                ⚠️ <strong>No data uploaded yet!</strong><br>
                Go to <strong>Tab 2</strong> to upload your mentees and mentors files.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
            
        # Import after checks (to avoid any import issues blocking the message)
        from utils import is_debug_mode
        from prompts import create_default_prompt
        
        mentees_df = st.session_state.get('mentees_df')
        mentors_df = st.session_state.get('mentors_df')

        # Ready message at the top
        mentee_count = len(mentees_df) if mentees_df is not None else 0
        mentor_count = len(mentors_df) if mentors_df is not None else 0
        total_evaluations = mentee_count * mentor_count
        
        # Calculate optimal batch size if we have all data
        batch_size = st.session_state.get('batch_size', 15)
        batch_info = None
        if mentees_df is not None and mentors_df is not None and st.session_state.get('api_key'):
            from utils import calculate_optimal_batch_size, estimate_batch_tokens, get_model_config
            model = st.session_state.get('model_choice', 'gpt-5.1')
            use_stateful = st.session_state.get('use_stateful_mode', False)
            batch_size = calculate_optimal_batch_size(
                total_mentors=mentor_count,
                num_mentees=mentee_count,
                model=model,
                use_stateful=use_stateful
            )
            st.session_state.batch_size = batch_size
            
            # Calculate batch details for display
            model_config = get_model_config(model)
            tpm_limit = model_config.get('tpm_limit', 30000)
            input_tokens, output_tokens = estimate_batch_tokens(
                batch_size, mentee_count, model, use_stateful, is_first_batch=True
            )
            total_tokens = input_tokens + output_tokens
            num_batches = int((mentor_count + batch_size - 1) / batch_size)
            
            batch_info = {
                'batch_size': batch_size,
                'num_batches': num_batches,
                'tpm_limit': tpm_limit,
                'usable_tpm': int(tpm_limit * 0.90),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'mode': 'Stateful' if use_stateful else 'Stateless'
            }
        
        st.markdown(f"""
        <div class="info-card">
            <strong>Ready to generate matches!</strong><br>
            This will process {total_evaluations:,} compatibility evaluations using {batch_size} mentors per batch (auto-optimized).
        </div>
        """, unsafe_allow_html=True)
        
        # Compact summary layout
        use_stateful = st.session_state.get('use_stateful_mode', False)
        stateful_status = "Enabled" if use_stateful else "Disabled"
        default_prompt = create_default_prompt()
        custom_prompt = st.session_state.get('custom_prompt', default_prompt)
        is_custom = custom_prompt != default_prompt
        
        # Compact summary in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🤖 Model", st.session_state.get('model_choice', 'gpt-5.1'))
        with col2:
            st.metric("📦 Batch Size", batch_size)
        with col3:
            st.metric("👥 Mentees", mentee_count)
        with col4:
            st.metric("👨‍🏫 Mentors", mentor_count)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Evaluations", f"{total_evaluations:,}")
        with col2:
            st.metric("🔗 Stateful", stateful_status)
        with col3:
            prompt_status = "Custom" if is_custom else "Default"
            st.metric("🔧 Prompt", prompt_status)
        with col4:
            max_mentees = st.session_state.get('max_mentees_per_mentor', 2)
            st.metric("⚖️ Max per Mentor", max_mentees)
        
        # Batch Processing Details
        if batch_info:
            st.divider()
            st.subheader("📦 Batch Processing Configuration")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Batch Size", f"{batch_info['batch_size']} mentors/batch")
            with col2:
                st.metric("Total Batches", f"{batch_info['num_batches']} batches")
            with col3:
                st.metric("Tokens/Batch", f"{batch_info['total_tokens']:,}")
            
            # Only show detailed calculation in debug mode
            if is_debug_mode():
                with st.expander("ℹ️ Batch Size Calculation Details", expanded=False):
                    st.markdown(f"""
                    **Auto-optimized batch size** based on model rate limits:
                    
                    - **Model TPM Limit**: {batch_info['tpm_limit']:,} tokens/minute
                    - **Safety Margin**: 90% of limit = {batch_info['usable_tpm']:,} usable tokens
                    - **Estimated Tokens per Batch**: 
                      - Input: {batch_info['input_tokens']:,} tokens
                      - Output: {batch_info['output_tokens']:,} tokens
                      - **Total**: {batch_info['total_tokens']:,} tokens
                    - **Processing Mode**: {batch_info['mode']}
                    
                    This ensures batches fit within rate limits while maximizing efficiency.
                    """)
        elif mentees_df is not None and mentors_df is not None:
            st.divider()
            st.info("📦 **Batch size**: Will be auto-calculated when you click 'Generate Matches'")
        
        # Show debug info if in debug mode
        if is_debug_mode():
            api_key = st.session_state.get('api_key')
            st.caption(f"API Key: {'✅ Configured' if api_key else '❌ Missing'}")
            if is_custom:
                with st.expander("📝 Preview Custom Prompt", expanded=False):
                    st.text_area("Custom Prompt", custom_prompt, height=200, disabled=True)
            elif st.session_state.get('selected_preset'):
                st.caption(f"Preset: {st.session_state.get('selected_preset', 'Balanced ⭐')}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Button to navigate to processing tab
        if st.button("🚀 Generate Matches", type="primary", width='stretch'):
            # Set flag to start processing in Tab 6
            st.session_state.start_processing = True
            st.success("✅ Ready to process! Please navigate to **Tab 6: Processing & Results** to start.")
            st.info("💡 The processing will begin automatically when you open Tab 6.")
    
    except Exception as e:
        st.error(f"❌ Error in Tab 5 (Generate Matches): {str(e)}")
        st.exception(e)
        import traceback
        with st.expander("🔍 Full Error Details", expanded=False):
            st.code(traceback.format_exc())
