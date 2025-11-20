"""
UI Component for Tab 4: Customize Prompt
Handles customization of the matching prompt and scoring criteria
"""

import streamlit as st
from utils import estimate_tokens


def render_prompt_tab():
    """Render Tab 4: Customize Matching Prompt (Optional)"""
    
    st.markdown("""
    <div class="info-card">
        Customize the scoring criteria, field compatibility matrix, and calibration examples. 
        The default prompt works well for most cases.
    </div>
    """, unsafe_allow_html=True)
    
    custom_prompt = st.text_area(
        "Matching Rules and Scoring Rubric",
        value=st.session_state.custom_prompt,
        height=350,
        help="Modify scoring criteria, field relationships, and calibration examples"
    )
    
    char_count = len(custom_prompt)
    token_count = estimate_tokens(custom_prompt)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Characters", f"{char_count:,}")
    with col_info2:
        st.metric("Est. Tokens", f"{token_count:,}")
    
    st.session_state.custom_prompt = custom_prompt
    
    st.caption("💡 **Tip:** You can adjust scoring weights, modify field relationships, or add domain-specific criteria.")

