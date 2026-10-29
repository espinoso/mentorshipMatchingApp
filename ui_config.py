"""
UI Component for Tab 1: Configure Settings
Handles API key status, model selection, and matching constraints
"""

import streamlit as st

from info import DEFAULT_MODEL
from utils import fetch_available_models, is_debug_mode


def render_config_tab():
    """Render Tab 1: Configure Settings"""
    
    # API Key status (key is read once at startup by app._anthropic_key into session_state.api_key)
    if not st.session_state.api_key:
        st.markdown("""
        <div class="conflict-warning">
            ⚠️ <strong>Missing API Key</strong><br>
            Please set <code>ANTHROPIC_API_KEY</code> (environment variable or Streamlit secrets).<br>
            Get your key at <a href="https://console.anthropic.com/settings/keys" target="_blank">https://console.anthropic.com/settings/keys</a>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    if is_debug_mode():
        st.markdown("""
        <div class="success-card">
            ✅ Anthropic API key loaded from <code>ANTHROPIC_API_KEY</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        Configure your AI model and matching constraints. These settings control how the matching algorithm processes your data.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🤖 AI Model Selection")

    available_models = fetch_available_models()
    model_ids = list(available_models)
    if st.session_state.model_choice not in model_ids:
        st.session_state.model_choice = DEFAULT_MODEL

    selected_model = st.selectbox(
        "Choose AI Model",
        model_ids,
        index=model_ids.index(st.session_state.model_choice),
        key="model_selector",
        help="Claude model used to score compatibility and write the match reasoning.",
    )
    st.session_state.model_choice = selected_model

    model_info = available_models[selected_model]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Quality", model_info["quality"])
    with col2:
        st.metric("Speed", model_info["speed"])
    st.caption(f"Pricing: {model_info['cost']}")

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

    restrict_spain = st.checkbox(
        "Apply the Spain ↔ Spain mentoring restriction (Spanish mentors are not matched to Spanish mentees)",
        value="spain" in st.session_state.get("country_restrictions", ["spain"]),
    )
    st.session_state.country_restrictions = ["spain"] if restrict_spain else []
    
    if is_debug_mode():
        st.info(f"ℹ️ Each mentor can be assigned to a maximum of {max_mentees} mentee(s). This constraint is enforced by the Hungarian algorithm during optimal assignment.")

