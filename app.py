import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from openai import OpenAI
from io import BytesIO
import time
import tiktoken
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Mentorship Matching System",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling (Improvements: 1, 2, 3, 10, 21, 22)
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Improved main header with animation */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: fadeInDown 0.6s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-header h1 {
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        font-weight: 300;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Sticky header */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        padding: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: -1rem -1rem 1rem -1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Enhanced mentee card */
    .mentee-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.6rem 0;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    
    .mentee-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: 0;
    }
    
    .mentee-card:hover {
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
        border-left-width: 6px;
    }
    
    .mentee-card.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-left: 6px solid #5a6fd8 !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4) !important;
        transform: translateX(12px) scale(1.03);
    }
    
    .mentee-card * {
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced match card */
    .match-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
        padding: 1.8rem;
        border-radius: 16px;
        border: 2px solid #e9ecef;
        margin: 1.2rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        color: #212529 !important;
        transition: all 0.3s ease;
        animation: slideInRight 0.4s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .match-card:hover {
        box-shadow: 0 8px 28px rgba(0,0,0,0.12);
        transform: translateY(-4px);
        border-color: #667eea;
    }
    
    .match-card h4 {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.01em !important;
    }
    
    .match-card p {
        color: #4a5568 !important;
        margin-bottom: 0.75rem !important;
        line-height: 1.7 !important;
        font-size: 0.95rem !important;
    }
    
    .match-card strong {
        color: #1a202c !important;
        font-weight: 600 !important;
    }
    
    /* Enhanced percentage badge with animation */
    .percentage-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 24px;
        font-weight: 600;
        color: white;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .excellent { 
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    .strong { 
        background: linear-gradient(135deg, #20c997 0%, #17a2b8 100%);
    }
    .good { 
        background: linear-gradient(135deg, #17a2b8 0%, #20c997 100%);
    }
    .fair { 
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: #212529;
    }
    
    /* Enhanced alert boxes */
    .conflict-warning {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
        animation: shake 0.5s ease;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    .cost-estimate {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
    }
    
    .token-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        color: #856404;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
        animation: bounce 1s ease;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Wizard steps */
    .wizard-step {
        display: flex;
        align-items: center;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: #f8f9fa;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .wizard-step.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #5a6fd8;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .wizard-step.completed {
        background: #d4edda;
        border-color: #28a745;
    }
    
    .wizard-step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #667eea;
        color: white;
        font-weight: 600;
        margin-right: 0.8rem;
        font-size: 0.9rem;
    }
    
    .wizard-step.completed .wizard-step-number {
        background: #28a745;
    }
    
    .wizard-step.active .wizard-step-number {
        background: white;
        color: #667eea;
    }
    
    /* Info card */
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.15);
    }
    
    /* Success card */
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.15);
        animation: fadeInUp 0.5s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading skeleton */
    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 8px;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Tooltip */
    .tooltip-icon {
        display: inline-block;
        width: 18px;
        height: 18px;
        background: #667eea;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 18px;
        font-size: 12px;
        font-weight: bold;
        cursor: help;
        margin-left: 0.3rem;
    }
    
    /* Better button styles */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    /* Search box */
    .search-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .search-box:focus {
        outline: none;
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Copy button */
    .copy-button {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background: #667eea;
        color: white;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        border: none;
        font-weight: 500;
    }
    
    .copy-button:hover {
        background: #5a6fd8;
        transform: scale(1.05);
    }
    
    /* Preset badge */
    .preset-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        background: #ffc107;
        color: #212529;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    /* Accessibility - High contrast mode support */
    @media (prefers-contrast: high) {
        .mentee-card, .match-card {
            border-width: 3px;
        }
    }
    
    /* Accessibility - Reduced motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* Better focus indicators for accessibility */
    button:focus, input:focus, textarea:focus {
        outline: 3px solid #667eea;
        outline-offset: 2px;
    }
    
    /* Stats card */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    /* Help panel */
    .help-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    
    /* Session info */
    .session-info {
        background: #f8f9fa;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #6c757d;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for UI improvements (Improvements: 15, 16, 20, 23, 24)
def get_prompt_presets():
    """Get configuration presets for matching (Improvement 16)"""
    presets = {
        "Quick Start ⚡": {
            "description": "Fast matching with standard criteria",
            "model": "gpt-4o-mini",
            "batch_size": 20,
            "prompt_type": "standard"
        },
        "Balanced ⭐": {
            "description": "Best balance of quality and speed (Recommended)",
            "model": "gpt-4o-mini",
            "batch_size": 15,
            "prompt_type": "standard"
        },
        "Thorough 🎯": {
            "description": "Deep analysis with comprehensive matching",
            "model": "o1-mini",
            "batch_size": 10,
            "prompt_type": "detailed"
        },
        "Maximum Quality 💎": {
            "description": "Best possible matches (slower, more expensive)",
            "model": "o1-preview",
            "batch_size": 10,
            "prompt_type": "detailed"
        }
    }
    return presets

def show_help_documentation():
    """Display help documentation (Improvement 23)"""
    st.markdown("""
    <div class="help-panel">
        <h3>📚 Quick Help Guide</h3>
        <p><strong>Getting Started:</strong></p>
        <ol>
            <li>Upload your mentee and mentor Excel files</li>
            <li>Enter your OpenAI API key</li>
            <li>Select a configuration preset or customize settings</li>
            <li>Click "Generate Matches" and wait for results</li>
        </ol>
        
        <p><strong>File Format Requirements:</strong></p>
        <ul>
            <li>Mentee file must have a "Code Mentee" column</li>
            <li>Mentor file must have a "Code mentor" column</li>
            <li>Additional columns improve matching quality</li>
        </ul>
        
        <p><strong>Tips for Best Results:</strong></p>
        <ul>
            <li>Include detailed information in your data files</li>
            <li>Use the "Balanced" preset for most cases</li>
            <li>Check the cost estimate before running</li>
            <li>Review the heatmap to understand all possible matches</li>
        </ul>
        
        <p><strong>Troubleshooting:</strong></p>
        <ul>
            <li><em>Token limit warnings:</em> Try reducing batch size or use fewer mentors/mentees</li>
            <li><em>Timeout errors:</em> Switch from Assistants API to Direct API or reduce data size</li>
            <li><em>Missing matches:</em> The system will automatically retry failed batches</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def show_session_info():
    """Display session information (Improvement 24)"""
    import datetime
    if 'session_start' not in st.session_state:
        st.session_state.session_start = datetime.datetime.now()
    
    if 'last_match_time' in st.session_state:
        time_info = f"Last match: {st.session_state.last_match_time.strftime('%I:%M %p')}"
    else:
        time_info = f"Session started: {st.session_state.session_start.strftime('%I:%M %p')}"
    
    st.markdown(f"""
    <div class="session-info">
        ⏰ {time_info}
    </div>
    """, unsafe_allow_html=True)

def copy_to_clipboard_js(text, button_id):
    """Generate JavaScript for copy to clipboard (Improvement 20)"""
    return f"""
    <button class="copy-button" onclick="
        navigator.clipboard.writeText(`{text}`).then(() => {{
            this.innerHTML = '✓ Copied!';
            setTimeout(() => {{ this.innerHTML = '📋 Copy'; }}, 2000);
        }});
    " id="{button_id}">📋 Copy</button>
    """

def show_wizard_steps(step1_done, step2_done, step3_done, current_step):
    """Display wizard-style progress steps (Improvement 7)"""
    steps = [
        {"num": 1, "title": "Upload Data", "done": step1_done},
        {"num": 2, "title": "Configure Settings", "done": step2_done},
        {"num": 3, "title": "Generate Matches", "done": step3_done}
    ]
    
    st.markdown("### 🧭 Progress")
    
    for step in steps:
        step_class = ""
        if step["done"]:
            step_class = "completed"
            icon = "✓"
        elif step["num"] == current_step:
            step_class = "active"
            icon = step["num"]
        else:
            icon = step["num"]
        
        st.markdown(f"""
        <div class="wizard-step {step_class}" role="status" aria-label="Step {step['num']}: {step['title']}">
            <div class="wizard-step-number">{icon}</div>
            <div><strong>{step['title']}</strong></div>
        </div>
        """, unsafe_allow_html=True)

def validate_uploaded_file(df, file_type):
    """Validate uploaded files and show warnings (Improvement 18)"""
    warnings = []
    errors = []
    
    # Check for required columns
    if file_type == 'mentee':
        if 'Code Mentee' not in df.columns:
            errors.append("Missing required column: 'Code Mentee'")
    elif file_type == 'mentor':
        if 'Code mentor' not in df.columns:
            errors.append("Missing required column: 'Code mentor'")
    
    # Check for empty values
    if file_type == 'mentee' and 'Code Mentee' in df.columns:
        empty_count = df['Code Mentee'].isna().sum()
        if empty_count > 0:
            warnings.append(f"{empty_count} rows have empty 'Code Mentee' values")
    
    if file_type == 'mentor' and 'Code mentor' in df.columns:
        empty_count = df['Code mentor'].isna().sum()
        if empty_count > 0:
            warnings.append(f"{empty_count} rows have empty 'Code mentor' values")
    
    # CRITICAL: Check for duplicate IDs
    if file_type == 'mentee' and 'Code Mentee' in df.columns:
        id_col = df['Code Mentee'].dropna()  # Remove NaN values
        duplicates = id_col[id_col.duplicated(keep=False)]
        if len(duplicates) > 0:
            unique_dups = duplicates.unique()
            errors.append(f"DUPLICATE IDs FOUND: {len(unique_dups)} mentee ID(s) appear multiple times!")
            errors.append(f"Duplicate mentee IDs: {', '.join(map(str, unique_dups[:10]))}" + (" ..." if len(unique_dups) > 10 else ""))
    
    if file_type == 'mentor' and 'Code mentor' in df.columns:
        id_col = df['Code mentor'].dropna()  # Remove NaN values
        duplicates = id_col[id_col.duplicated(keep=False)]
        if len(duplicates) > 0:
            unique_dups = duplicates.unique()
            errors.append(f"DUPLICATE IDs FOUND: {len(unique_dups)} mentor ID(s) appear multiple times!")
            errors.append(f"Duplicate mentor IDs: {', '.join(map(str, unique_dups[:10]))}" + (" ..." if len(unique_dups) > 10 else ""))
    
    # Check data quality
    total_cells = df.shape[0] * df.shape[1]
    empty_cells = df.isna().sum().sum()
    completeness = ((total_cells - empty_cells) / total_cells) * 100
    
    if completeness < 50:
        warnings.append(f"Low data completeness: {completeness:.1f}%")
    
    return errors, warnings, completeness

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in a text string"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding
    
    return len(encoding.encode(text))

def estimate_api_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Estimate the cost of an API call based on token usage"""
    # Pricing as of 2024 (per 1M tokens)
    pricing = {
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},  # per 1M tokens
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4": {"prompt": 30.00, "completion": 60.00},
        "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50}
    }
    
    if model not in pricing:
        model = "gpt-4o-mini"
    
    prompt_cost = (prompt_tokens / 1_000_000) * pricing[model]["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing[model]["completion"]
    
    return prompt_cost + completion_cost

def clean_dataframe(df):
    """Clean and validate dataframe"""
    # Remove empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Check if we have "Unnamed" columns (indicates empty header rows)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        # Find the first row that has actual data
        for idx in range(len(df)):
            row = df.iloc[idx]
            non_empty_values = [val for val in row if pd.notna(val) and str(val).strip() != '']
            if len(non_empty_values) >= 3:
                df.columns = df.iloc[idx].values
                df = df.iloc[idx+1:].reset_index(drop=True)
                break
    
    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)
    
    # Clean string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', '')
        df[col] = df[col].replace('', np.nan)
    
    return df

def validate_file_structure(df, file_type):
    """Validate that uploaded files have minimum required columns"""
    # Only check for the essential ID column - all other columns are optional
    # This allows flexibility while ensuring we can identify mentees/mentors
    
    if file_type == 'mentee':
        # Just need to identify mentees
        required_cols = ['Code Mentee']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if not missing_cols:
            # Show info about what columns we found
            return []
        return missing_cols
    elif file_type == 'mentor':
        # Just need to identify mentors
        required_cols = ['Code mentor']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if not missing_cols:
            return []
        return missing_cols
    else:
        return []

def generate_sample_data():
    """Generate sample data for testing"""
    mentees_data = {
        'Code Mentee': ['Mentee_A1', 'Mentee_B2', 'Mentee_C3', 'Mentee_D4'],
        'Mentee Affiliation': ['University Alpha', 'Institute Beta', 'College Gamma', 'University Delta'],
        'Mentee Country of your affiliated institution': ['Spain', 'France', 'Italy', 'Germany'],
        'Mentee Highest educational level completed': ['Master', 'Bachelor', 'Master', 'Bachelor'],
        'Mentee Current education program': ['PhD', 'Master', 'PhD', 'PhD'],
        'Mentee Field of expertise': ['Biomedicine', 'Computer Science', 'Biomedicine', 'Data Science'],
        'Mentee Specialization': ['Cancer Research', 'Machine Learning', 'Genetics', 'AI Ethics'],
        'Mentee Specific hard skills that you have': [
            'Cell culture, PCR, Western blot',
            'Python, TensorFlow, Deep Learning',
            'GWAS, R programming, Statistics',
            'Python, Ethics frameworks, Policy analysis'
        ],
        'Mentee Areas where guidance is needed': [
            'Research methodology, grant writing',
            'Industry transition, networking',
            'International collaboration, publication',
            'Career planning, academic vs industry'
        ],
        'Mentee Career goals for the next 2 years': [
            'Postdoc in top research institute',
            'Senior data scientist role',
            'International research collaboration',
            'PhD completion and industry transition'
        ]
    }
    
    mentors_data = {
        'Code mentor': ['Mentor_X1', 'Mentor_Y2', 'Mentor_Z3', 'Mentor_W4', 'Mentor_V5'],
        'Mentor Affiliation': ['Harvard Medical', 'Google AI', 'Max Planck Institute', 'MIT', 'Stanford'],
        'Mentor Country of affiliated institution': ['USA', 'USA', 'Germany', 'USA', 'USA'],
        'Mentor Type of Institution': ['Academia', 'Industry', 'Academia', 'Academia', 'Academia'],
        'Field of expertise': ['Biomedicine', 'Computer Science', 'Biomedicine', 'Data Science', 'Biomedicine'],
        'Mentor Specialization': [
            'Oncology research, immunotherapy',
            'Machine Learning, Computer Vision',
            'Genomics, population genetics',
            'AI Ethics, responsible AI',
            'Molecular biology, drug discovery'
        ],
        'Mentor Specific Hard Skills and Professional Competencies has Mastered': [
            'Clinical trials, biomarker discovery, immunology',
            'TensorFlow, PyTorch, MLOps, team leadership',
            'Population genomics, GWAS, statistical genetics',
            'AI governance, policy development, stakeholder engagement',
            'High-throughput screening, medicinal chemistry'
        ],
        'Mentor Years of Professional Experience in his her Field': [15, 8, 12, 6, 20]
    }
    
    return pd.DataFrame(mentees_data), pd.DataFrame(mentors_data)

def create_matrix_prompt(use_file_search=False):
    """Create the prompt for generating compatibility matrix"""
    
    file_search_note = """
TRAINING DATA:
- Historical matching data is available via the file_search tool
- Use file_search to learn patterns from successful past matches
- Apply insights from historical data to inform your evaluation
- Look for patterns in field alignment, skill matches, and career goal compatibility

""" if use_file_search else ""
    
    prompt = f"""You are an expert mentorship coordinator. Evaluate compatibility between ALL mentees and ALL mentors.

{file_search_note}CRITICAL REQUIREMENTS:
- You MUST evaluate EVERY SINGLE mentor with EVERY SINGLE mentee
- The matrix must be COMPLETE - missing mentors or mentees will break the system
- Count the mentors and mentees in the data and ensure your output has ALL of them

TASK: Generate a complete compatibility matrix
- Evaluate EVERY mentee with EVERY mentor (no exceptions!)
- Return a percentage score (0-100) for each combination
- Base scores on ALL provided data fields

EVALUATION CRITERIA:
- Field of expertise and specialization alignment (30%)
- Hard skills compatibility (25%)
- Career goals alignment with mentor strengths (20%)
- Language compatibility for effective communication (10%)
- Geographic/country compatibility for collaboration (5%)
- Academic vs Industry experience match (5%)
- Experience level appropriateness (5%)

SCORING GUIDELINES:
- 90-100%: Exceptional match - highly aligned on all major factors
- 75-89%: Strong match - aligned on most critical factors
- 60-74%: Good match - aligned on several important factors
- 45-59%: Fair match - some alignment, workable
- 0-44%: Poor match - minimal alignment

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown, no extra text):
  {{
  "matrix": [
      {{
        "mentor_id": "EXACT_CODE_FROM_FILE", 
      "scores": [
        {{"mentee_id": "EXACT_CODE_FROM_FILE", "percentage": 85}},
        {{"mentee_id": "EXACT_CODE_FROM_FILE", "percentage": 72}},
        ... (ALL mentees - check the data for complete list!)
      ]
    }},
    ... (ALL mentors - check the data for complete list!)
  ]
}}

VERIFICATION BEFORE RESPONDING:
1. Count how many mentors are in the provided data
2. Count how many mentees are in the provided data
3. Ensure your response has exactly that many mentor entries
4. Ensure each mentor entry has exactly that many mentee scores

IMPORTANT: 
- Use EXACT codes from the provided data (Code mentor, Code Mentee columns)
- Include every single mentee-mentor combination
- DO NOT skip any mentors or mentees
- DO NOT include reasoning (that comes later)
- Just percentages for now"""
    return prompt

def create_reasoning_prompt(assignments):
    """Create prompt to get reasoning for specific assignments"""
    assignment_list = "\n".join([f"- {mentee} → {mentor} (score: {score}%)" 
                                  for mentee, mentor, score in assignments])
    
    prompt = f"""You previously evaluated mentee-mentor compatibility. Now provide brief reasoning for these specific assignments:

{assignment_list}

For each assignment, explain in 2-3 sentences why this is a good match, mentioning:
- Key skill/expertise alignments
- Career goal compatibility
- Any language/geographic advantages
- Other relevant factors

OUTPUT FORMAT:
Return ONLY a JSON array (no markdown, no extra text):
[
  {{
    "mentee_id": "EXACT_CODE",
    "mentor_id": "EXACT_CODE",
    "reasoning": "Brief 2-3 sentence explanation of why this match works well."
  }},
  ...
]"""
    return prompt

def create_default_prompt():
    """Legacy prompt - now using matrix approach"""
    training_file_ids = st.session_state.get('training_file_ids', [])
    return create_matrix_prompt(use_file_search=len(training_file_ids) > 0)

def clean_text_for_csv(df):
    """Clean text fields to prevent CSV parsing issues"""
    df = df.copy()
    
    # Clean string columns
    for col in df.select_dtypes(include=['object']).columns:
        # Replace newlines with spaces
        df[col] = df[col].astype(str).str.replace('\n', ' ', regex=False)
        df[col] = df[col].astype(str).str.replace('\r', ' ', regex=False)
        # Replace multiple spaces with single space
        df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
        # Strip whitespace
        df[col] = df[col].str.strip()
        # Replace 'nan' string back to NaN
        df[col] = df[col].replace('nan', np.nan)
    
    return df

def prepare_data_for_assistant(mentees_df, mentors_df, training_files):
    """Prepare data with ALL columns for comprehensive matching"""
    
    # Use ALL columns - no filtering
    mentees_full = mentees_df.copy()
    mentors_full = mentors_df.copy()
    
    # Clean text fields to prevent CSV parsing issues (removes embedded newlines)
    mentees_cleaned = clean_text_for_csv(mentees_full)
    mentors_cleaned = clean_text_for_csv(mentors_full)
    
    # Combine into a single structured document
    combined_data = "# MENTORSHIP MATCHING DATA\n\n"
    combined_data += "## MENTEES TO MATCH (ALL DATA)\n"
    combined_data += f"Total mentees: {len(mentees_cleaned)}\n"
    combined_data += f"All columns included: {', '.join(mentees_cleaned.columns.tolist())}\n\n"
    combined_data += mentees_cleaned.to_csv(index=False)
    combined_data += "\n## AVAILABLE MENTORS (ALL DATA)\n"
    combined_data += f"Total mentors: {len(mentors_cleaned)}\n"
    combined_data += f"All columns included: {', '.join(mentors_cleaned.columns.tolist())}\n\n"
    combined_data += mentors_cleaned.to_csv(index=False)
    
    if training_files:
        combined_data += "\n## TRAINING EXAMPLES (First 5 rows for reference)\n"
        for i, df in enumerate(training_files):
            df_cleaned = clean_text_for_csv(df.head(5))
            combined_data += f"\n### Training Set {i+1}\n"
            combined_data += df_cleaned.to_csv(index=False)
    
    return combined_data

def call_openai_api_with_assistants(prompt, mentees_df, mentors_df, training_data, api_key, model="gpt-4o-mini", mentor_subset=None):
    """Use OpenAI Assistants API - sends ALL data directly in message
    
    Args:
        mentor_subset: Optional list of mentor IDs to evaluate (for batching)
    """
    try:
        client = OpenAI(api_key=api_key)
        
        # If mentor_subset provided, filter mentors_df
        if mentor_subset:
            mentors_to_use = mentors_df[mentors_df['Code mentor'].isin(mentor_subset)]
        else:
            mentors_to_use = mentors_df
        
        # Prepare combined data with ALL columns
        combined_data = prepare_data_for_assistant(mentees_df, mentors_to_use, training_data)
        
        # Estimate tokens and cost
        full_message = f"{prompt}\n\n{combined_data}\n\nGenerate matches for ALL mentees in JSON format as specified."
        prompt_tokens = estimate_tokens(full_message, model)
        estimated_completion_tokens = len(mentees_df) * 500  # Rough estimate
        estimated_cost = estimate_api_cost(prompt_tokens, estimated_completion_tokens, model)
        
        # Display token and cost information
        st.markdown(f"""
        <div class="cost-estimate">
            <strong>📊 API Usage Estimate:</strong><br>
            • Prompt tokens: ~{prompt_tokens:,}<br>
            • Expected completion tokens: ~{estimated_completion_tokens:,}<br>
            • Estimated cost: ${estimated_cost:.4f}<br>
            • Model: {model}
        </div>
        """, unsafe_allow_html=True)
        
        # Check token limits
        total_tokens = prompt_tokens + estimated_completion_tokens
        model_limits = {
            "gpt-4o-mini": 128000,
            "gpt-4o": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16384
        }
        
        if total_tokens > model_limits.get(model, 128000):
            st.markdown(f"""
            <div class="token-warning">
                <strong>⚠️ Token Limit Warning:</strong><br>
                Estimated tokens ({total_tokens:,}) may exceed model limit ({model_limits.get(model, 128000):,}).<br>
                Try Direct API with fewer mentees or use a different approach.
            </div>
            """, unsafe_allow_html=True)
            return None
        
        # Create the assistant (with file_search if training files uploaded to OpenAI)
        st.info("🤖 Creating AI assistant for matching...")
        
        # Check if training files are uploaded to OpenAI storage
        training_file_ids = st.session_state.get('training_file_ids', [])
        vector_store = None  # Initialize for cleanup
        
        if training_file_ids:
            st.info(f"📚 Enabling file_search for {len(training_file_ids)} training file(s)...")
            # Create a vector store with the training files
            vector_store = client.beta.vector_stores.create(
                name=f"Training Data {time.strftime('%Y%m%d_%H%M%S')}",
                file_ids=training_file_ids
            )
            
            assistant = client.beta.assistants.create(
                name="Mentorship Matcher",
                instructions=prompt,
                model=model,
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
            )
        else:
            assistant = client.beta.assistants.create(
                name="Mentorship Matcher",
                instructions=prompt,
                model=model
            )
        
        # Create a thread and send data directly in message
        st.info("🔄 Sending data and processing matches...")
        thread = client.beta.threads.create()
        
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=full_message
        )
        
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Wait for completion with timeout
        max_wait_time = 600  # 10 minutes timeout (Assistants API can be slow)
        start_time = time.time()
        
        progress_placeholder = st.empty()
        
        while run.status in ['queued', 'in_progress', 'cancelling']:
            time.sleep(3)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            if elapsed > max_wait_time:
                progress_placeholder.error("⏱️ Processing timeout after 10 minutes. Try using 'Direct API' instead or reduce data size.")
                # Cleanup
                try:
                    client.beta.assistants.delete(assistant.id)
                    if vector_store:
                        client.beta.vector_stores.delete(vector_store.id)
                except:
                    pass
                return None
            
            # Show detailed progress
            if run.status == 'queued':
                progress_placeholder.warning(f"⏳ Queued - waiting for OpenAI resources... ({minutes}m {seconds}s)")
            else:
                progress_placeholder.info(f"🤖 Processing matches... ({minutes}m {seconds}s)")
        
        if run.status == 'completed':
            # Get the messages
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            # Extract the assistant's response
            for msg in messages.data:
                if msg.role == 'assistant':
                    response_content = msg.content[0].text.value
                    
                    # Clean up (delete assistant and vector store)
                    try:
                        client.beta.assistants.delete(assistant.id)
                        if vector_store:
                            client.beta.vector_stores.delete(vector_store.id)
                    except:
                        pass
                    
                    return response_content
        else:
            st.error(f"❌ Processing failed with status: {run.status}")
            if hasattr(run, 'last_error'):
                st.error(f"Error details: {run.last_error}")
        
        # Cleanup on failure
        try:
            client.beta.assistants.delete(assistant.id)
            if vector_store:
                client.beta.vector_stores.delete(vector_store.id)
        except:
            pass
        return None
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def parse_llm_response(response_text):
    """Parse the LLM response and extract matches"""
    try:
        # Clean the response text
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        matches = json.loads(response_text)
        
        # Validate structure
        if not isinstance(matches, list):
            raise ValueError("Response should be a list")
        
        for match_data in matches:
            if 'mentee_id' not in match_data or 'matches' not in match_data:
                raise ValueError("Invalid match structure")
        
        return matches
        
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"❌ Error parsing response: {str(e)}")
        with st.expander("View raw response"):
            st.text(response_text[:2000])
            return None
        
def generate_matrix_in_batches(mentees_df, mentors_df, training_data, api_key, model, batch_size=15):
    """Generate compatibility matrix in batches to avoid token limits
    
    Args:
        batch_size: Number of mentors to process per batch (default 15)
    
    Returns:
        Complete matrix DataFrame
    """
    all_mentor_ids = mentors_df['Code mentor'].tolist()
    all_mentee_ids = mentees_df['Code Mentee'].tolist()
    
    # Split mentors into batches
    mentor_batches = [all_mentor_ids[i:i + batch_size] for i in range(0, len(all_mentor_ids), batch_size)]
    
    st.info(f"📦 Processing in {len(mentor_batches)} batches ({batch_size} mentors per batch)")
    
    combined_matrix_dict = {}
    
    # Process each batch
    for batch_num, mentor_batch in enumerate(mentor_batches, 1):
        st.info(f"🔄 Batch {batch_num}/{len(mentor_batches)}: Processing {len(mentor_batch)} mentors ({mentor_batch[0]} to {mentor_batch[-1]})")
        
        # Call API for this batch
        training_file_ids = st.session_state.get('training_file_ids', [])
        response = call_openai_api_with_assistants(
            create_matrix_prompt(use_file_search=len(training_file_ids) > 0),
            mentees_df,
            mentors_df,
            training_data,
            api_key,
            model,
            mentor_subset=mentor_batch
        )
        
        if not response:
            st.error(f"❌ Batch {batch_num} failed to get response")
            continue
        
        # Parse this batch's response
        batch_matrix_df = parse_matrix_response(response, mentor_batch, all_mentee_ids)
        
        if batch_matrix_df is None:
            st.error(f"❌ Batch {batch_num} failed to parse")
            continue
        
        # DEBUG: Show what mentors we got back vs what we expected
        received_mentors = list(batch_matrix_df.index)
        expected_mentors_set = set(mentor_batch)
        received_mentors_set = set(received_mentors)
        
        # Track what we're adding and check for overwrites
        dict_size_before = len(combined_matrix_dict)
        newly_added = 0
        overwrites = []
        
        for mentor_id in batch_matrix_df.index:
            if mentor_id in combined_matrix_dict:
                overwrites.append(mentor_id)
                st.error(f"🚨 DUPLICATE: '{mentor_id}' already exists! Batch {batch_num} is overwriting previous data!")
            
            combined_matrix_dict[mentor_id] = batch_matrix_df.loc[mentor_id].to_dict()
            newly_added += 1
        
        dict_size_after = len(combined_matrix_dict)
        actual_new_entries = dict_size_after - dict_size_before
        
        # Show detailed statistics
        st.info(f"📈 Dict growth: {dict_size_before} → {dict_size_after} (+{actual_new_entries} new entries)")
        
        if overwrites:
            st.error(f"⚠️ WARNING: {len(overwrites)} duplicate(s) were OVERWRITTEN!")
            with st.expander(f"🔍 Overwritten mentor IDs ({len(overwrites)})"):
                for m in sorted(overwrites):
                    st.write(f"- {m} (was already in dict from previous batch)")
        
        # Detailed success/warning message
        if received_mentors_set == expected_mentors_set:
            st.success(f"✅ Batch {batch_num}: Received all {len(batch_matrix_df)}/{len(mentor_batch)} requested mentors!")
        else:
            missing_in_batch = expected_mentors_set - received_mentors_set
            st.warning(f"⚠️ Batch {batch_num}: Only {len(batch_matrix_df)}/{len(mentor_batch)} mentors received")
            with st.expander(f"📋 Missing from batch {batch_num} ({len(missing_in_batch)})"):
                for m in sorted(missing_in_batch):
                    st.write(f"- {m}")
        
        # Small delay between batches to avoid rate limits
        if batch_num < len(mentor_batches):
            time.sleep(2)
    
    # Combine all batches into single DataFrame
    if not combined_matrix_dict:
        st.error("❌ No batches succeeded")
        return None

    st.divider()
    st.info("📊 **Combining all batches into final matrix...**")
    
    complete_matrix_df = pd.DataFrame(combined_matrix_dict).T
    complete_matrix_df = complete_matrix_df[sorted(all_mentee_ids)]
    
    # Final validation with detailed breakdown
    mentors_received = len(complete_matrix_df)
    expected_count = len(all_mentor_ids)
    mentors_in_dict = set(complete_matrix_df.index)
    mentors_requested = set(all_mentor_ids)
    
    # Check for discrepancies
    missing_from_dict = mentors_requested - mentors_in_dict
    extra_in_dict = mentors_in_dict - mentors_requested
    
    st.success(f"✅ **Final matrix**: {mentors_received}/{expected_count} unique mentors in dictionary")
    
    # Show detailed breakdown
    with st.expander("🔍 Final Matrix Validation Details"):
        st.write(f"**Expected mentors**: {expected_count} (from original data)")
        st.write(f"**Mentors in dictionary**: {mentors_received} (unique keys)")
        st.write(f"**Dictionary size**: {len(combined_matrix_dict)} entries")
        
        if missing_from_dict:
            st.error(f"❌ **Missing {len(missing_from_dict)} mentors** that were requested but not in dict:")
            for m in sorted(missing_from_dict)[:20]:  # Show max 20
                st.text(f"  - {m}")
        else:
            st.success("✅ All requested mentors are in the dictionary!")
        
        if extra_in_dict:
            st.warning(f"⚠️ **Extra {len(extra_in_dict)} mentors** in dict that weren't requested:")
            for m in sorted(extra_in_dict)[:20]:
                st.text(f"  - {m}")
    
    if mentors_received < expected_count:
        st.warning(f"⚠️ **DISCREPANCY**: Expected {expected_count} but only have {mentors_received} in final matrix")
    
    # RETRY LOGIC: If mentors are missing, try up to 3 times
    if mentors_received < expected_count:
        missing = set(all_mentor_ids) - set(complete_matrix_df.index)
        st.warning(f"⚠️ Missing {len(missing)} mentors after initial batches")
        
        with st.expander(f"📋 Missing mentors ({len(missing)})"):
            for m in sorted(missing):
                st.write(f"- {m}")
        
        # Try up to 3 retries
        max_retries = 3
        for retry_num in range(1, max_retries + 1):
            # Recalculate what's still missing
            current_missing = set(all_mentor_ids) - set(complete_matrix_df.index)
            
            if not current_missing:
                st.success(f"🎉 All mentors retrieved!")
                break
            
            st.info(f"🔄 Retry {retry_num}/{max_retries}: Attempting {len(current_missing)} missing mentors...")
            
            try:
                training_file_ids = st.session_state.get('training_file_ids', [])
                retry_response = call_openai_api_with_assistants(
                    create_matrix_prompt(use_file_search=len(training_file_ids) > 0),
                    mentees_df,
                    mentors_df,
                    training_data,
                    api_key,
                    model,
                    mentor_subset=list(current_missing)
                )
                
                if not retry_response:
                    st.error(f"❌ Retry {retry_num} failed to get response")
                    if retry_num < max_retries:
                        st.info("⏳ Waiting 5 seconds before next retry...")
                        time.sleep(5)
                    continue
                
                retry_matrix_df = parse_matrix_response(retry_response, list(current_missing), all_mentee_ids)
                
                if retry_matrix_df is not None and len(retry_matrix_df) > 0:
                    # Add retry results to combined matrix
                    added_count = 0
                    for mentor_id in retry_matrix_df.index:
                        combined_matrix_dict[mentor_id] = retry_matrix_df.loc[mentor_id].to_dict()
                        added_count += 1
                    
                    st.success(f"✅ Retry {retry_num} successful: Added {added_count} mentors")
                    
                    # Rebuild complete matrix from updated dictionary
                    complete_matrix_df = pd.DataFrame(combined_matrix_dict).T
                    complete_matrix_df = complete_matrix_df[sorted(all_mentee_ids)]
                    
                    # Update count
                    mentors_received = len(complete_matrix_df)
                    st.info(f"📈 Matrix now has {mentors_received}/{expected_count} mentors")
                    
                    # Check if complete
                    if mentors_received == expected_count:
                        st.success(f"🎉 NOW COMPLETE: {mentors_received}/{expected_count} mentors!")
                        break
                else:
                    st.error(f"❌ Retry {retry_num} failed to parse")
                    if retry_num < max_retries:
                        st.info("⏳ Waiting 5 seconds before next retry...")
                        time.sleep(5)
                        
            except Exception as e:
                st.error(f"❌ Retry {retry_num} error: {str(e)}")
                if retry_num < max_retries:
                    st.info("⏳ Waiting 5 seconds before next retry...")
                    time.sleep(5)
        
        # Final check after all retries
        final_missing = set(all_mentor_ids) - set(complete_matrix_df.index)
        if final_missing:
            st.warning(f"⚠️ After {max_retries} retries, still missing {len(final_missing)} mentors")
            with st.expander(f"📋 Permanently missing mentors ({len(final_missing)})"):
                for m in sorted(final_missing):
                    st.write(f"- {m}")
            st.info(f"💡 Continuing with {len(complete_matrix_df)}/{expected_count} mentors. Missing mentors will be excluded from matching.")
    
    return complete_matrix_df

def parse_matrix_response(response_text, expected_mentors, expected_mentees):
    """Parse the matrix response from AI and validate completeness + ID matching"""
    try:
        # Clean the response text
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        data = json.loads(response_text)
        
        # Validate structure
        if 'matrix' not in data:
            raise ValueError("Response should contain 'matrix' key")
        
        matrix_data = data['matrix']
        if not isinstance(matrix_data, list):
            raise ValueError("Matrix should be a list")
        
        # Convert to pandas DataFrame with VALIDATION
        matrix_dict = {}
        all_mentees = set()
        
        # Track validation issues
        expected_mentor_set = set(expected_mentors)
        expected_mentee_set = set(expected_mentees)
        wrong_mentors = []  # Mentors not in expected list
        wrong_mentees = []  # Mentees not in expected list
        
        for mentor_entry in matrix_data:
            mentor_id = mentor_entry['mentor_id']
            
            # VALIDATION 1: Check if this mentor ID was requested
            if mentor_id not in expected_mentor_set:
                wrong_mentors.append(mentor_id)
                st.warning(f"⚠️ AI returned WRONG mentor ID: '{mentor_id}' (not in requested batch)")
                continue  # SKIP this mentor - don't add to dict
            
            scores = {}
            for score_entry in mentor_entry['scores']:
                mentee_id = score_entry['mentee_id']
                percentage = score_entry['percentage']
                
                # VALIDATION 2: Check if this mentee ID is valid
                if mentee_id not in expected_mentee_set:
                    if mentee_id not in wrong_mentees:  # Only warn once per wrong mentee
                        wrong_mentees.append(mentee_id)
                    continue  # SKIP this mentee score
                
                scores[mentee_id] = percentage
                all_mentees.add(mentee_id)
            
            # Only add mentor if they have at least some valid scores
            if scores:
                matrix_dict[mentor_id] = scores
        
        # Report validation issues
        if wrong_mentors:
            st.error(f"🚨 WRONG MENTOR IDs: AI returned {len(wrong_mentors)} mentor(s) NOT in requested batch!")
            with st.expander(f"❌ Wrong mentor IDs ({len(wrong_mentors)})"):
                for m in sorted(wrong_mentors):
                    st.write(f"- {m} (REJECTED - not in batch request)")
        
        if wrong_mentees:
            st.error(f"🚨 WRONG MENTEE IDs: AI returned {len(wrong_mentees)} invalid mentee(s)!")
            with st.expander(f"❌ Wrong mentee IDs ({len(wrong_mentees)})"):
                for m in sorted(wrong_mentees)[:10]:  # Show max 10
                    st.write(f"- {m}")
        
        # Validate completeness (after filtering)
        mentors_received = len(matrix_dict)
        mentees_received = len(all_mentees)
        expected_mentor_count = len(expected_mentors)
        expected_mentee_count = len(expected_mentees)
        
        st.info(f"📊 Valid data received: {mentors_received}/{expected_mentor_count} mentors, {mentees_received}/{expected_mentee_count} mentees")
        
        if mentors_received < expected_mentor_count:
            missing_mentors = set(expected_mentors) - set(matrix_dict.keys())
            st.warning(f"⚠️ Missing {len(missing_mentors)} requested mentor(s)")
            with st.expander(f"📋 Missing mentors ({len(missing_mentors)})"):
                for m in sorted(missing_mentors):
                    st.write(f"- {m}")
        
        if mentees_received < expected_mentee_count:
            missing_mentees = set(expected_mentees) - all_mentees
            st.warning(f"⚠️ Missing {len(missing_mentees)} mentee(s) in responses")
            with st.expander(f"📋 Missing mentees ({len(missing_mentees)})"):
                for m in sorted(missing_mentees):
                    st.write(f"- {m}")
        
        # Create DataFrame
        if not matrix_dict:
            st.error("❌ No valid mentor data after filtering!")
            return None
        
        df = pd.DataFrame(matrix_dict).T  # Transpose so mentors are rows, mentees are columns
        
        # Fill missing mentees with median score (50) if any are missing
        for mentee in expected_mentees:
            if mentee not in df.columns:
                df[mentee] = 50  # Default score for missing data
        
        df = df[sorted(expected_mentees)]  # Sort mentee columns
        
        return df
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        st.error(f"❌ Error parsing matrix response: {str(e)}")
        with st.expander("View raw response"):
            st.text(response_text[:2000])
        return None

def hungarian_assignment(matrix_df, max_mentees_per_mentor=2):
    """
    Use Hungarian algorithm to find optimal mentee-mentor assignments
    with constraint of max 2 mentees per mentor
    
    Args:
        matrix_df: DataFrame with mentors as rows, mentees as columns, percentages as values
        max_mentees_per_mentor: Maximum number of mentees per mentor (default 2)
    
    Returns:
        List of tuples: [(mentee_id, mentor_id, score), ...]
    """
    mentors = list(matrix_df.index)
    mentees = list(matrix_df.columns)
    
    # Create expanded cost matrix to handle mentor capacity constraint
    # Each mentor gets duplicated max_mentees_per_mentor times
    expanded_mentors = []
    for mentor in mentors:
        for i in range(max_mentees_per_mentor):
            expanded_mentors.append(f"{mentor}_copy{i}")
    
    # Build cost matrix (negative because Hungarian minimizes, we want to maximize)
    cost_matrix = np.zeros((len(mentees), len(expanded_mentors)))
    
    for i, mentee in enumerate(mentees):
        for j, expanded_mentor in enumerate(expanded_mentors):
            actual_mentor = expanded_mentor.rsplit('_copy', 1)[0]
            score = matrix_df.loc[actual_mentor, mentee]
            cost_matrix[i, j] = -score  # Negative for maximization
    
    # Run Hungarian algorithm
    mentee_indices, mentor_indices = linear_sum_assignment(cost_matrix)
    
    # Extract assignments
    assignments = []
    for mentee_idx, mentor_idx in zip(mentee_indices, mentor_indices):
        mentee = mentees[mentee_idx]
        expanded_mentor = expanded_mentors[mentor_idx]
        actual_mentor = expanded_mentor.rsplit('_copy', 1)[0]
        score = matrix_df.loc[actual_mentor, mentee]
        assignments.append((mentee, actual_mentor, int(score)))
    
    # Sort by mentee ID for consistency
    assignments.sort(key=lambda x: x[0])
    
    return assignments

def get_reasoning_for_assignments(assignments, mentees_df, mentors_df, api_key, model="gpt-4o-mini"):
    """Get reasoning for specific assignments via separate API call"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Validate inputs
        if not assignments or len(assignments) == 0:
            st.warning("⚠️ No assignments to get reasoning for")
            return {}
        
        if mentees_df is None or mentors_df is None:
            st.error("❌ Missing mentee or mentor data")
            return {}
        
        # Prepare data for context (just the assigned pairs)
        assignment_details = []
        for mentee_id, mentor_id, score in assignments:
            try:
                mentee_row = mentees_df[mentees_df['Code Mentee'] == mentee_id].iloc[0]
                mentor_row = mentors_df[mentors_df['Code mentor'] == mentor_id].iloc[0]
                
                # Helper to safely get string values
                def safe_str(value):
                    if pd.isna(value):
                        return ''
                    return str(value).strip()
                
                assignment_details.append({
                    'mentee_id': safe_str(mentee_id),
                    'mentor_id': safe_str(mentor_id),
                    'score': int(score) if not pd.isna(score) else 0,
                    'mentee_field': safe_str(mentee_row.get('Mentee Field of expertise', '')),
                    'mentee_skills': safe_str(mentee_row.get('Mentee Specific hard skills that you have', '')),
                    'mentee_goals': safe_str(mentee_row.get('Mentee Career goals for the next 2 years', '')),
                    'mentor_field': safe_str(mentor_row.get('Field of expertise', '')),
                    'mentor_skills': safe_str(mentor_row.get('Mentor Specific Hard Skills and Professional Competencies has Mastered', '')),
                    'mentor_experience': safe_str(mentor_row.get('Mentor Years of Professional Experience in his her Field', ''))
                })
            except Exception as e:
                st.warning(f"⚠️ Could not extract details for {mentee_id} → {mentor_id}: {str(e)}")
                # Add with minimal info
                assignment_details.append({
                    'mentee_id': str(mentee_id),
                    'mentor_id': str(mentor_id),
                    'score': int(score) if isinstance(score, (int, float)) else 0,
                    'mentee_field': '',
                    'mentee_skills': '',
                    'mentee_goals': '',
                    'mentor_field': '',
                    'mentor_skills': '',
                    'mentor_experience': ''
                })
        
        # Check if we have valid assignment details
        if not assignment_details:
            st.error("❌ No valid assignment details extracted")
            return {}
        
        st.info(f"📝 Extracted details for {len(assignment_details)} assignments")
        
        # Create focused prompt
        prompt = f"""Provide brief reasoning for these {len(assignment_details)} mentorship assignments.

For each assignment, explain in 2-3 sentences why this is a good match based on:
- Field/specialization alignment
- Skills compatibility
- Career goals support

ASSIGNMENTS:
"""
        for ad in assignment_details:
            prompt += f"""
{ad['mentee_id']} → {ad['mentor_id']} (Score: {ad['score']}%)
- Mentee field: {ad['mentee_field']}
- Mentee skills: {ad['mentee_skills'][:100]}...
- Mentee goals: {ad['mentee_goals'][:100]}...
- Mentor field: {ad['mentor_field']}
- Mentor skills: {ad['mentor_skills'][:100]}...
- Mentor experience: {ad['mentor_experience']} years

"""
        
        prompt += """
OUTPUT FORMAT (IMPORTANT):
Return ONLY a JSON array with this exact structure:
[
  {
    "mentee_id": "EXACT_CODE",
    "mentor_id": "EXACT_CODE",
    "reasoning": "2-3 sentence explanation"
  }
]

No markdown, no extra text, just the JSON array."""
        
        # Make API call with progress bar
        progress_text = st.empty()
        progress_text.text("🧠 Requesting reasoning from AI...")
        
        # Handle o1 models differently (no system messages, no temperature)
        if model.startswith("o1-"):
            # o1 models don't support system messages - combine into user message
            full_prompt = "You are an expert at explaining mentorship compatibility. Return only valid JSON.\n\n" + prompt
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=8000
            )
        else:
            # Standard models
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at explaining mentorship compatibility. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=8000
            )
        
        progress_text.text("🧠 Parsing reasoning response...")
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Debug: show what we got
        st.info(f"📝 Received reasoning response ({len(response_text)} characters)")
        
        # Show first part of response for debugging
        with st.expander("🔍 Debug: View response preview"):
            st.text(response_text[:500])
        
        # Clean markdown
        original_text = response_text
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            reasoning_data = json.loads(response_text)
        except json.JSONDecodeError as parse_error:
            st.error(f"❌ JSON parse failed: {parse_error}")
            with st.expander("📄 Full response text"):
                st.code(original_text)
            raise
        
        if not isinstance(reasoning_data, list):
            st.error(f"❌ Response is {type(reasoning_data).__name__}, expected list")
            with st.expander("📄 Parsed data"):
                st.json(reasoning_data)
            raise ValueError("Response should be a list of reasoning objects")
        
        st.info(f"✅ Parsed {len(reasoning_data)} reasoning entries from JSON")
        
        # Convert to dict for easy lookup
        reasoning_dict = {}
        for idx, r in enumerate(reasoning_data):
            if 'mentee_id' in r and 'mentor_id' in r and 'reasoning' in r:
                reasoning_dict[(r['mentee_id'], r['mentor_id'])] = r['reasoning']
            else:
                st.warning(f"⚠️ Entry {idx} missing required fields: {r.keys()}")
        
        progress_text.empty()
        st.success(f"✅ Got reasoning for {len(reasoning_dict)} assignments")
        
        # Debug: Show what keys we have
        if len(reasoning_dict) > 0:
            sample_keys = list(reasoning_dict.keys())[:3]
            st.info(f"📋 Sample reasoning keys: {sample_keys}")
        
        return reasoning_dict
        
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON parsing error in reasoning: {str(e)}")
        with st.expander("View raw reasoning response"):
            try:
                st.text(response_text[:2000])
            except:
                st.text("Could not display response")
        st.info("💡 Continuing with generic reasoning...")
        return {}
    except Exception as e:
        st.error(f"❌ Error getting reasoning: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        st.info("💡 Continuing without detailed reasoning...")
        return {}

def create_heatmap(matrix_df, assignments=None):
    """Create interactive heatmap visualization of compatibility matrix
    
    Args:
        matrix_df: DataFrame with mentors as rows, mentees as columns
        assignments: Optional list of (mentee, mentor, score) tuples to highlight
    """
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns,
        y=matrix_df.index,
        colorscale='RdYlGn',  # Red-Yellow-Green
        text=matrix_df.values,
        texttemplate='%{text}%',
        textfont={"size": 8},
        colorbar=dict(title="Match %"),
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> → <b>%{x}</b><br>Match: %{z}%<extra></extra>'
    ))
    
    # Add rectangles around assigned matches
    if assignments:
        assignment_dict = {(mentee, mentor): score for mentee, mentor, score in assignments}
        
        shapes = []
        for mentee_idx, mentee in enumerate(matrix_df.columns):
            for mentor_idx, mentor in enumerate(matrix_df.index):
                if (mentee, mentor) in assignment_dict:
                    # Add a thick border around assigned cells
                    shapes.append(dict(
                        type="rect",
                        x0=mentee_idx - 0.5,
                        y0=mentor_idx - 0.5,
                        x1=mentee_idx + 0.5,
                        y1=mentor_idx + 0.5,
                        line=dict(color="blue", width=3),
                    ))
        
        fig.update_layout(shapes=shapes)
    
    # Update layout
    fig.update_layout(
        title="Mentee-Mentor Compatibility Matrix",
        xaxis_title="Mentees",
        yaxis_title="Mentors",
        height=max(600, len(matrix_df) * 25),  # Dynamic height
        width=max(800, len(matrix_df.columns) * 30),  # Dynamic width
        xaxis=dict(tickangle=-45),
        font=dict(size=10)
    )
    
    return fig

def enforce_mentor_limit(matches):
    """Enforce maximum 2 mentees per mentor by keeping only the best matches
    
    When a mentor is matched with more than 2 mentees (at rank 1), we keep only
    the top 2 mentees with the highest match percentages. Other mentees get their
    rank 2 promoted to rank 1, and so on.
    """
    if not matches:
        return matches
    
    # Track rank 1 assignments per mentor with match percentages
    rank1_assignments = {}  # {mentor_id: [(mentee_id, percentage, match_data), ...]}
    
    # Collect all rank 1 assignments
    for match_data in matches:
        mentee_id = match_data['mentee_id']
        for match in match_data['matches']:
            if match['rank'] == 1:
                mentor_id = match['mentor_id']
                percentage = match['match_percentage']
                if mentor_id not in rank1_assignments:
                    rank1_assignments[mentor_id] = []
                rank1_assignments[mentor_id].append((mentee_id, percentage, match_data))
    
    # Find mentors assigned to more than 2 mentees
    mentors_to_fix = {mentor_id: assignments 
                      for mentor_id, assignments in rank1_assignments.items() 
                      if len(assignments) > 2}
    
    if not mentors_to_fix:
        return matches  # No conflicts, return as is
    
    # For each overassigned mentor, keep only top 2 mentees
    mentees_to_reassign = set()
    
    for mentor_id, assignments in mentors_to_fix.items():
        # Sort by percentage (highest first)
        sorted_assignments = sorted(assignments, key=lambda x: x[1], reverse=True)
        
        # Keep top 2, mark others for reassignment
        for mentee_id, percentage, match_data in sorted_assignments[2:]:
            mentees_to_reassign.add(mentee_id)
    
    # Process matches: remove overassigned rank 1, promote rank 2 to rank 1
    updated_matches = []
    
    for match_data in matches:
        mentee_id = match_data['mentee_id']
        
        if mentee_id in mentees_to_reassign:
            # This mentee needs reassignment
            current_matches = match_data['matches']
            rank1_mentor = current_matches[0]['mentor_id']
            
            # Remove the overassigned rank 1, promote others
            new_matches = [
                {
                    'rank': i + 1,
                    'mentor_id': current_matches[i+1]['mentor_id'],
                    'match_percentage': current_matches[i+1]['match_percentage'],
                    'match_quality': current_matches[i+1]['match_quality'],
                    'reasoning': current_matches[i+1]['reasoning'] + f" (Promoted from rank {i+2} due to {rank1_mentor} being overassigned)"
                }
                for i in range(min(2, len(current_matches) - 1))  # Promote rank 2 and 3
            ]
            
            updated_matches.append({
                'mentee_id': mentee_id,
                'matches': new_matches,
                'note': f'Original rank 1 ({rank1_mentor}) was overassigned'
            })
        else:
            # Keep as is
            updated_matches.append(match_data)
    
    return updated_matches

def check_mentor_conflicts(matches):
    """Check if any mentor is assigned to more than 2 mentees at rank 1"""
    if not matches:
        return []
    
    mentor_assignments = {}
    conflicts = []
    
    for match_data in matches:
        mentee_id = match_data['mentee_id']
        # Only check rank 1 assignments
        if match_data['matches'] and len(match_data['matches']) > 0:
            rank1_match = match_data['matches'][0]
            if rank1_match['rank'] == 1:
                mentor_id = rank1_match['mentor_id']
                percentage = rank1_match['match_percentage']
            if mentor_id not in mentor_assignments:
                mentor_assignments[mentor_id] = []
                mentor_assignments[mentor_id].append((mentee_id, percentage))
    
    for mentor_id, assignments in mentor_assignments.items():
        if len(assignments) > 2:
            conflicts.append({
                'mentor_id': mentor_id,
                'assignments': assignments,
                'count': len(assignments)
            })
    
    return conflicts

# ============================================================================
# OPENAI FILE STORAGE FUNCTIONS
# ============================================================================

def list_openai_files(api_key):
    """List all files uploaded to OpenAI storage"""
    try:
        client = OpenAI(api_key=api_key)
        files = client.files.list(purpose='assistants')
        
        file_list = []
        for file in files.data:
            created_at = datetime.fromtimestamp(file.created_at)
            age_days = (datetime.now() - created_at).days
            
            file_list.append({
                'id': file.id,
                'filename': file.filename,
                'size_bytes': file.bytes,
                'size_kb': file.bytes / 1024,
                'created_at': created_at,
                'age_days': age_days,
                'purpose': file.purpose
            })
        
        return file_list
    except Exception as e:
        st.error(f"❌ Error listing files: {str(e)}")
        return []

def upload_training_files_to_openai(training_dfs, api_key):
    """Upload training DataFrames to OpenAI storage"""
    try:
        client = OpenAI(api_key=api_key)
        uploaded_file_ids = []
        
        for i, df in enumerate(training_dfs):
            # Convert DataFrame to CSV
            csv_content = df.to_csv(index=False)
            
            # Create BytesIO object
            file_obj = BytesIO(csv_content.encode('utf-8'))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_obj.name = f"training_data_{timestamp}_{i+1}.csv"
            
            # Upload to OpenAI
            st.info(f"📤 Uploading {file_obj.name} ({len(csv_content)/1024:.1f} KB)...")
            
            file = client.files.create(
                file=file_obj,
                purpose='assistants'
            )
            
            uploaded_file_ids.append(file.id)
            st.success(f"✅ Uploaded: {file.id}")
        
        return uploaded_file_ids
        
    except Exception as e:
        st.error(f"❌ Error uploading files: {str(e)}")
        return []

def delete_openai_files(api_key, file_ids):
    """Delete specific files from OpenAI storage"""
    try:
        client = OpenAI(api_key=api_key)
        deleted_count = 0
        
        for file_id in file_ids:
            try:
                client.files.delete(file_id)
                deleted_count += 1
            except Exception as e:
                st.warning(f"Could not delete {file_id}: {str(e)}")
        
        return deleted_count
        
    except Exception as e:
        st.error(f"❌ Error deleting files: {str(e)}")
        return 0

def download_openai_file(api_key, file_id):
    """Download file content from OpenAI storage and return as DataFrame"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Retrieve file content
        file_content = client.files.content(file_id)
        
        # Convert to DataFrame (assuming CSV format)
        from io import StringIO
        csv_content = file_content.read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error downloading file {file_id}: {str(e)}")
        return None

def main():
    # Header with animation
    st.markdown("""
    <div class="main-header">
        <h1>🤝 Mentorship Matching System</h1>
        <p>AI-powered mentor-mentee matching with optimized token usage</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state (Improvement 24)
    if 'training_files' not in st.session_state:
        st.session_state.training_files = []
    if 'mentees_df' not in st.session_state:
        st.session_state.mentees_df = None
    if 'mentors_df' not in st.session_state:
        st.session_state.mentors_df = None
    if 'matches' not in st.session_state:
        st.session_state.matches = None
    if 'matrix_df' not in st.session_state:
        st.session_state.matrix_df = None
    if 'assignments' not in st.session_state:
        st.session_state.assignments = None
    if 'custom_prompt' not in st.session_state:
        st.session_state.custom_prompt = create_default_prompt()
    if 'selected_preset' not in st.session_state:
        st.session_state.selected_preset = "Balanced ⭐"
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 15
    if 'training_file_ids' not in st.session_state:
        st.session_state.training_file_ids = []
    if 'training_file_names' not in st.session_state:
        st.session_state.training_file_names = []
    if 'mentee_search' not in st.session_state:
        st.session_state.mentee_search = ""
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "gpt-4o-mini"
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    
    # No sidebar - cleaner interface
    
    # Check completion status for wizard flow
    # Step 1 (API key) + Step 3 (data files) are required
    # Step 2 (training files) is optional
    step1_complete = (
        st.session_state.mentees_df is not None and 
        st.session_state.mentors_df is not None
    )
    
    # Get API key early for step 2 check
    import os
    
    # Try to get from session state or environment
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
    
    step2_complete = bool(st.session_state.api_key) and step1_complete
    step3_complete = step2_complete  # Can view data once files uploaded and settings configured
    
    # Create wizard tabs (6 tabs in logical order)
    tab_names = [
        "1️⃣ Configure Settings",
        "2️⃣ Training Files",
        "3️⃣ Upload Data Files",
        "4️⃣ Customize Prompt",
        "5️⃣ Data Overview",
        "6️⃣ Generate Matches"
    ]
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)
    
    # TAB 1: Configure Settings
    with tab1:
        st.header("⚙️ Step 1: Configure OpenAI API")
        
        st.markdown("""
        <div class="info-card">
            Start by entering your OpenAI API key. This is required before uploading any files.
        </div>
        """, unsafe_allow_html=True)
        
        # API Key input
        st.subheader("🔑 OpenAI API Key")
        api_key_input = st.text_input(
            "Enter your OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Your OpenAI API key is required to generate matches. Get one at https://platform.openai.com/api-keys"
        )
        
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
        
        if api_key_input:
            st.markdown("""
            <div class="success-card">
                ✅ API key configured
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="token-warning">
                ⚠️ Please enter your API key to continue
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Configuration Presets
        st.subheader("🎯 Matching Configuration Presets")
        presets = get_prompt_presets()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_preset = st.selectbox(
                "Choose a configuration preset",
                list(presets.keys()),
                index=list(presets.keys()).index(st.session_state.selected_preset) if st.session_state.selected_preset in presets else 1,
                help="Select a preset based on your needs for speed vs quality"
            )
            
            if selected_preset != st.session_state.selected_preset:
                st.session_state.selected_preset = selected_preset
        
        with col2:
            preset_config = presets[selected_preset]
            st.metric("Model", preset_config["model"])
            st.metric("Batch Size", preset_config["batch_size"])
        
        # Show preset description
        if "Recommended" in preset_config["description"]:
            st.markdown(f"""
            <div class="info-card">
                ⭐ <strong>{selected_preset}</strong><br>
                {preset_config["description"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"**{selected_preset}**: {preset_config['description']}")
        
        st.divider()
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings (Optional)"):
            st.write("Override preset settings if needed:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_choice_override = st.selectbox(
                    "AI Model",
                    [
                        "gpt-4o-mini",
                        "gpt-4o",
                        "o1-mini",
                        "o1-preview",
                        "gpt-3.5-turbo"
                    ],
                    index=["gpt-4o-mini", "gpt-4o", "o1-mini", "o1-preview", "gpt-3.5-turbo"].index(preset_config["model"]),
                    help="Choose the AI model for matching",
                    key="model_override"
                )
                st.session_state.model_choice = model_choice_override
            
            with col2:
                batch_size = st.slider(
                    "Batch Size",
                    min_value=5,
                    max_value=30,
                    value=preset_config["batch_size"],
                    help="Number of mentors to process per batch"
                )
                st.session_state.batch_size = batch_size
        
        # Info box
        st.markdown("""
        <div class="info-card">
            💡 After configuring your API key, proceed to Tab 2 to manage training files
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 2: Training Files
    with tab2:
        st.header("📚 Step 2: Manage Training Files")
        
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
        
        # Show currently selected training files
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
        
        # Create subtabs for existing and new training files
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
                        # Delete button
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
                        if st.button("✅ Use Selected Files", type="primary", width='stretch'):
                            # Just store the file IDs and basic info
                            selected_ids = [f['id'] for f in selected_files]
                            selected_names = [f['filename'] for f in selected_files]
                            
                            st.session_state.training_file_ids = selected_ids
                            st.session_state.training_file_names = selected_names
                            st.session_state.training_files = []  # Clear any old DataFrames
                            
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
            
            # Load and validate files first
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
        
    
    # TAB 3: Upload Data Files
    with tab3:
        st.header("📁 Step 3: Upload Mentee & Mentor Files")
        
        if not st.session_state.api_key:
            st.markdown("""
            <div class="conflict-warning">
                ⚠️ <strong>API Key Required</strong><br>
                Please configure your OpenAI API key in Tab 1 before uploading data files.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        st.markdown("""
        <div class="info-card">
            Upload your mentee and mentor Excel files. Both files are required to proceed with matching.
        </div>
        """, unsafe_allow_html=True)
        
        # Mentees and Mentors side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Mentees File *Required*")
            mentee_file = st.file_uploader(
                "Upload mentees Excel file",
                type=['xlsx'],
                key=f"mentee_upload_{st.session_state.reset_counter}",
                help="Excel file containing mentee information. Must have 'Code Mentee' column."
            )
            
        if mentee_file:
            try:
                df = pd.read_excel(mentee_file)
                df = clean_dataframe(df)
                
                errors, warnings, completeness = validate_uploaded_file(df, 'mentee')
                
                if errors:
                    for error in errors:
                        st.markdown(f"""
                        <div class="conflict-warning">
                            ❌ <strong>Error:</strong> {error}<br>
                            <small>Please fix this before proceeding</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.session_state.mentees_df = df
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ <strong>Loaded {len(df)} mentees</strong><br>
                        📊 Columns: {len(df.columns)} | Data quality: {completeness:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if warnings:
                        for warning in warnings:
                            st.markdown(f"""
                            <div class="token-warning">
                                ⚠️ {warning}
                            </div>
                            """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.markdown(f"""
                <div class="conflict-warning">
                    ❌ <strong>Error loading file:</strong> {str(e)}<br>
                    <small>Make sure it's a valid Excel (.xlsx) file</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("👨‍🏫 Mentors File *Required*")
            mentor_file = st.file_uploader(
                "Upload mentors Excel file",
                type=['xlsx'],
                key=f"mentor_upload_{st.session_state.reset_counter}",
                help="Excel file containing mentor information. Must have 'Code mentor' column."
            )
            
        if mentor_file:
            try:
                df = pd.read_excel(mentor_file)
                df = clean_dataframe(df)
                
                errors, warnings, completeness = validate_uploaded_file(df, 'mentor')
                
                if errors:
                    for error in errors:
                        st.markdown(f"""
                        <div class="conflict-warning">
                            ❌ <strong>Error:</strong> {error}<br>
                            <small>Please fix this before proceeding</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.session_state.mentors_df = df
                    st.markdown(f"""
                    <div class="success-card">
                        ✅ <strong>Loaded {len(df)} mentors</strong><br>
                        📊 Columns: {len(df.columns)} | Data quality: {completeness:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if warnings:
                        for warning in warnings:
                            st.markdown(f"""
                            <div class="token-warning">
                                ⚠️ {warning}
                            </div>
                            """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.markdown(f"""
                <div class="conflict-warning">
                    ❌ <strong>Error loading file:</strong> {str(e)}<br>
                    <small>Make sure it's a valid Excel (.xlsx) file</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧪 Load Sample Data", width='stretch'):
                sample_mentees, sample_mentors = generate_sample_data()
                st.session_state.mentees_df = sample_mentees
                st.session_state.mentors_df = sample_mentors
                st.success("✅ Sample data loaded!")
                st.rerun()

    
    # TAB 4: Customize Prompt
    with tab4:
        st.header("🔧 Step 4: Customize Matching Prompt (Optional)")
        
        st.markdown("""
        <div class="info-card">
            Optionally customize how the AI evaluates mentee-mentor compatibility. The default prompt works well for most cases.
        </div>
        """, unsafe_allow_html=True)
        
        # Character counter
        custom_prompt = st.text_area(
            "Matching Prompt",
            value=st.session_state.custom_prompt,
            height=400,
            help="Modify how the AI evaluates compatibility"
        )
        
        # Show character and token count
        char_count = len(custom_prompt)
        token_count = estimate_tokens(custom_prompt)
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Characters", f"{char_count:,}")
        with col_info2:
            st.metric("Est. Tokens", f"{token_count:,}")
        with col_info3:
            st.metric("Lines", custom_prompt.count('\n') + 1)
        
        # Update session state whenever the text area changes
        st.session_state.custom_prompt = custom_prompt
        
        st.divider()
        
        # Prompt tips
        with st.expander("💡 Prompt Customization Tips"):
            st.markdown("""
            **Tips for customizing the prompt:**
            - Adjust the percentage weights for different criteria
            - Add specific domain expertise requirements
            - Include language or timezone preferences
            - Specify industry-specific matching factors
            
            **Example customizations:**
            - Increase "Field of expertise" weight to 40% for technical roles
            - Add "Time zone overlap" as a 10% factor for remote mentoring
            - Emphasize "Publications" for academic mentorship programs
            """)
    
    # TAB 5: Data Overview
    with tab5:
        st.header("📊 Step 5: Review Your Data")
        
        st.markdown("""
        <div class="info-card">
            Review your uploaded data before generating matches. All columns will be used for matching.
        </div>
        """, unsafe_allow_html=True)
        
        # Check data availability directly (not using cached step1_complete)
        data_ready = (
            st.session_state.mentees_df is not None and 
            st.session_state.mentors_df is not None
        )
        
        if not data_ready:
            st.markdown("""
            <div class="token-warning">
                ⚠️ <strong>No data uploaded yet!</strong><br>
                Go to <strong>Tab 3</strong> to upload your mentees and mentors files.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Better metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Count training files from IDs (preferred) or DataFrames (legacy)
            training_count = len(st.session_state.training_file_ids) if st.session_state.training_file_ids else len(st.session_state.training_files) if st.session_state.training_files else 0
            
            # Show different subtitle based on source
            if st.session_state.training_file_ids:
                subtitle = '<small>From OpenAI storage</small>'
            elif st.session_state.training_files:
                subtitle = f'<small>{sum(len(df) for df in st.session_state.training_files)} total records</small>'
            else:
                subtitle = '<small>No files selected</small>'
            
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="color: #667eea; margin: 0;">📚 {training_count}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Training Files</p>
                {subtitle}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mentee_count = len(st.session_state.mentees_df) if st.session_state.mentees_df is not None else 0
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="color: #28a745; margin: 0;">👥 {mentee_count}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Mentees</p>
                {f'<small>{len(st.session_state.mentees_df.columns)} columns</small>' if st.session_state.mentees_df is not None else ''}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            mentor_count = len(st.session_state.mentors_df) if st.session_state.mentors_df is not None else 0
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="color: #17a2b8; margin: 0;">👨‍🏫 {mentor_count}</h3>
                <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Mentors</p>
                {f'<small>{len(st.session_state.mentors_df.columns)} columns</small>' if st.session_state.mentors_df is not None else ''}
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Data preview with better styling
        # Training files info
        if st.session_state.training_file_ids:
            with st.expander("📚 Training Files Info", expanded=False):
                st.write("**Files selected from OpenAI storage:**")
                if st.session_state.training_file_names:
                    for i, name in enumerate(st.session_state.training_file_names, 1):
                        st.write(f"{i}. {name}")
                else:
                    for i, file_id in enumerate(st.session_state.training_file_ids, 1):
                        st.write(f"{i}. File ID: `{file_id}`")
                st.caption(f"{len(st.session_state.training_file_ids)} file(s) will be used via file_search during matching")
        elif st.session_state.training_files:
            with st.expander("📚 Preview Training Data (Local)", expanded=False):
                for i, df in enumerate(st.session_state.training_files, 1):
                    st.write(f"**Training File {i}**: {len(df)} rows, {len(df.columns)} columns")
                    st.dataframe(df.head(5), width='stretch')
                    if i < len(st.session_state.training_files):
                        st.divider()
                st.caption(f"Showing first 5 rows of each training file • {len(st.session_state.training_files)} file(s) total")
        
        if st.session_state.mentees_df is not None:
            with st.expander("👥 Preview Mentees Data", expanded=False):
                st.dataframe(st.session_state.mentees_df.head(10), width='stretch')
                st.caption(f"Showing 10 of {len(st.session_state.mentees_df)} rows • {len(st.session_state.mentees_df.columns)} columns")
        
        if st.session_state.mentors_df is not None:
            with st.expander("👨‍🏫 Preview Mentors Data", expanded=False):
                st.dataframe(st.session_state.mentors_df.head(10), width='stretch')
                st.caption(f"Showing 10 of {len(st.session_state.mentors_df)} rows • {len(st.session_state.mentors_df.columns)} columns")
        
        st.markdown("""
        <div class="success-card">
            ✅ <strong>Data looks good!</strong><br>
            You can now proceed to <strong>"6️⃣ Generate Matches"</strong>.
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 6: Generate Matches
    with tab6:
        st.header("🎯 Step 6: Generate Matches")
        
        # Use the api_key from session state
        api_key = st.session_state.api_key
        
        # Get model choice from session state (set in Tab 1)
        model_choice = st.session_state.model_choice
        
        # Check readiness
        ready = (
            st.session_state.mentees_df is not None and
            st.session_state.mentors_df is not None and
            api_key
        )
        
        # Show prerequisites if not met
        if not ready:
            st.markdown("""
            <div class="token-warning">
                ⚠️ <strong>Prerequisites not met!</strong> Please complete the following:
            </div>
            """, unsafe_allow_html=True)
            
            # Check data files directly
            data_files_ready = (
                st.session_state.mentees_df is not None and
                st.session_state.mentors_df is not None
            )
            
            if not data_files_ready:
                st.error("❌ **Step 3**: Upload mentee and mentor files")
            else:
                st.success("✅ **Step 3**: Data files uploaded")
            
            if not api_key:
                st.error("❌ **Step 1**: Configure API key and settings")
            else:
                st.success("✅ **Step 1**: Configuration complete")
            
            st.info("👆 Complete the missing steps above, then return to this tab.")
            st.stop()
        
        # Show data size with better styling
            mentee_count = len(st.session_state.mentees_df)
            mentor_count = len(st.session_state.mentors_df)
            
        st.markdown(f"""
        <div class="info-card">
            📊 <strong>Ready to match {mentee_count} mentees with {mentor_count} mentors</strong><br>
            <small>This will generate {mentee_count * mentor_count} compatibility scores using {st.session_state.batch_size} mentors per batch</small>
        </div>
        """, unsafe_allow_html=True)
            
            # Generate matches button
        if st.button("🚀 Generate Matches", type="primary", width='stretch'):
            # Save timestamp (Improvement 24)
            import datetime
            st.session_state.last_match_time = datetime.datetime.now()
            
            # PRE-FLIGHT VALIDATION: Check for duplicate IDs before sending to AI
            st.info("🔍 Pre-flight check: Validating data integrity...")
            
            validation_passed = True
            
            # Check mentees for duplicates
            mentee_ids = st.session_state.mentees_df['Code Mentee'].dropna()
            mentee_duplicates = mentee_ids[mentee_ids.duplicated(keep=False)]
            if len(mentee_duplicates) > 0:
                validation_passed = False
                unique_dups = mentee_duplicates.unique()
                st.error(f"🚨 DUPLICATE MENTEE IDs: {len(unique_dups)} mentee ID(s) appear multiple times in your data!")
                with st.expander(f"❌ Duplicate mentee IDs ({len(unique_dups)})"):
                    for dup_id in sorted(unique_dups):
                        count = (mentee_ids == dup_id).sum()
                        st.write(f"- **{dup_id}** appears {count} times")
                st.error("⚠️ This will cause the AI to return wrong results. Please fix your data and reload.")
            
            # Check mentors for duplicates
            mentor_ids = st.session_state.mentors_df['Code mentor'].dropna()
            mentor_duplicates = mentor_ids[mentor_ids.duplicated(keep=False)]
            if len(mentor_duplicates) > 0:
                validation_passed = False
                unique_dups = mentor_duplicates.unique()
                st.error(f"🚨 DUPLICATE MENTOR IDs: {len(unique_dups)} mentor ID(s) appear multiple times in your data!")
                with st.expander(f"❌ Duplicate mentor IDs ({len(unique_dups)})"):
                    for dup_id in sorted(unique_dups):
                        count = (mentor_ids == dup_id).sum()
                        st.write(f"- **{dup_id}** appears {count} times")
                st.error("⚠️ This will cause the AI to return wrong results. Please fix your data and reload.")
            
            if not validation_passed:
                st.markdown("""
                <div class="conflict-warning">
                    <strong>🛑 Cannot proceed with matching</strong><br>
                    <small>Your data contains duplicate IDs. Each mentee and mentor must have a unique ID.</small><br>
                    <small><strong>Why?</strong> Duplicate IDs cause the AI to return the same ID in multiple batches, leading to data overwrites and incorrect results.</small><br><br>
                    <strong>How to fix:</strong>
                    <ol>
                        <li>Open your Excel file</li>
                        <li>Find the duplicate IDs listed above</li>
                        <li>Make each ID unique (e.g., add a suffix: "Mentor 5" → "Mentor 5a", "Mentor 5b")</li>
                        <li>Save and reload the file</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                st.stop()  # Stop execution here
            
            st.success("✅ Pre-flight check passed: All IDs are unique!")
            
            # Better progress indicators (Improvement 3, 17)
            progress_container = st.container()
            
            with progress_container:
                # PHASE 1: Get compatibility matrix from AI (using batching)
                st.markdown("""
                <div class="info-card">
                    📊 <strong>Phase 1: Generating compatibility matrix with batching...</strong><br>
                    <small>This may take several minutes. Please don't close the browser.</small>
                </div>
                """, unsafe_allow_html=True)
                
                matrix_df = generate_matrix_in_batches(
                        st.session_state.mentees_df,
                        st.session_state.mentors_df,
                        st.session_state.training_files,
                        api_key,
                    model_choice,
                    batch_size=st.session_state.batch_size
                )
                
                if matrix_df is not None:
                        st.session_state.matrix_df = matrix_df
                        st.markdown(f"""
                        <div class="success-card">
                            ✅ <strong>Matrix generated successfully!</strong><br>
                            {len(matrix_df)} mentors × {len(matrix_df.columns)} mentees = {len(matrix_df) * len(matrix_df.columns)} evaluations!
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # PHASE 2: Apply Hungarian algorithm for optimal assignment
                        st.markdown("""
                        <div class="info-card">
                            🧮 <strong>Phase 2: Running Hungarian algorithm for optimal assignments...</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        assignments = hungarian_assignment(matrix_df, max_mentees_per_mentor=2)
                        st.session_state.assignments = assignments
                        
                        # Calculate statistics
                        avg_score = np.mean([score for _, _, score in assignments])
                        min_score = min([score for _, _, score in assignments])
                        max_score = max([score for _, _, score in assignments])
                        
                        st.markdown(f"""
                        <div class="success-card">
                            ✅ <strong>Optimal assignments found!</strong><br>
                            Avg: {avg_score:.1f}%, Min: {min_score}%, Max: {max_score}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # PHASE 3: Get reasoning for assignments
                        st.markdown("""
                        <div class="info-card">
                            🧠 <strong>Phase 3: Getting detailed reasoning for assignments...</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        reasoning_dict = get_reasoning_for_assignments(
                            assignments,
                        st.session_state.mentees_df,
                        st.session_state.mentors_df,
                        api_key,
                        model_choice
                    )
                
                        # Convert to old format for compatibility with display code
                        matches = []
                        
                        # Debug: Show what we're looking for vs what we have
                        if assignments and reasoning_dict:
                            st.info(f"🔍 Looking up reasoning: {len(assignments)} assignments, {len(reasoning_dict)} reasoning entries")
                            sample_assignment = assignments[0]
                            sample_key = (sample_assignment[0], sample_assignment[1])
                            with st.expander("🔍 Debug: Key matching"):
                                st.write(f"Sample assignment key: {sample_key}")
                                st.write(f"Sample assignment types: ({type(sample_assignment[0]).__name__}, {type(sample_assignment[1]).__name__})")
                                if reasoning_dict:
                                    sample_reason_key = list(reasoning_dict.keys())[0]
                                    st.write(f"Sample reasoning key: {sample_reason_key}")
                                    st.write(f"Sample reasoning types: ({type(sample_reason_key[0]).__name__}, {type(sample_reason_key[1]).__name__})")
                                    st.write(f"Keys match? {sample_key in reasoning_dict}")
                        
                        # Helper function to find reasoning with flexible key matching
                        def find_reasoning(mentee_id, mentor_id, reasoning_dict):
                            """Try multiple key formats to find reasoning"""
                            # Extract numeric parts if present
                            mentee_num = re.search(r'\d+', str(mentee_id))
                            mentor_num = re.search(r'\d+', str(mentor_id))
                            
                            # Try various key combinations
                            possible_keys = [
                                (mentee_id, mentor_id),  # Original format
                                (str(mentee_id), str(mentor_id)),  # String versions
                            ]
                            
                            # Add numeric-only versions if we found numbers
                            if mentee_num and mentor_num:
                                possible_keys.append((mentee_num.group(), mentor_num.group()))
                            
                            # Try each possible key
                            for key in possible_keys:
                                if key in reasoning_dict:
                                    return reasoning_dict[key]
                            
                            # No match found
                            return None
                        
                        for mentee, mentor, score in assignments:
                            # Try to find reasoning with flexible matching
                            reasoning = find_reasoning(mentee, mentor, reasoning_dict)
                            
                            # Fallback if no reasoning found
                            if reasoning is None:
                                reasoning = f"This match was selected by the Hungarian optimization algorithm as the globally optimal assignment with a {score}% compatibility score. Detailed reasoning was unavailable."
                            
                            matches.append({
                                'mentee_id': mentee,
                                'matches': [{
                                    'rank': 1,
                                    'mentor_id': mentor,
                                    'match_percentage': score,
                                    'match_quality': 'Excellent' if score >= 90 else 'Strong' if score >= 75 else 'Good' if score >= 60 else 'Fair',
                                    'reasoning': reasoning
                                }]
                            })
                        
                        st.session_state.matches = matches
                        st.success(f"✅ Complete! {len(matches)} mentees optimally matched with guaranteed constraints!")
                        st.balloons()
        else:
            # Show what's missing (only if something is actually missing)
            missing = []
            if st.session_state.mentees_df is None:
                missing.append("Mentees file")
            if st.session_state.mentors_df is None:
                missing.append("Mentors file")
            if not api_key:
                missing.append("OpenAI API key")
            
            if missing:
                st.warning(f"⚠️ Please provide: {', '.join(missing)}")
        
        # Display results (Improvements 6, 17, 19, 20, 21)
        if st.session_state.matches:
            st.divider()
            st.subheader("📊 Matching Results")
            
            # Display statistics with better styling (Improvement 17)
            if st.session_state.assignments:
                scores = [score for _, _, score in st.session_state.assignments]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: #667eea; margin: 0;">{len(st.session_state.assignments)}</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Total Matches</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: #28a745; margin: 0;">{np.mean(scores):.1f}%</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Avg Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: #ffc107; margin: 0;">{min(scores)}%</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Min Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: #17a2b8; margin: 0;">{max(scores)}%</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">Max Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show heatmap
            if st.session_state.matrix_df is not None:
                st.divider()
                st.subheader("🔥 Compatibility Heatmap")
                st.caption("Blue boxes show optimal assignments selected by Hungarian algorithm")
                
                fig = create_heatmap(st.session_state.matrix_df, st.session_state.assignments)
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to download matrix
                csv = st.session_state.matrix_df.to_csv()
                st.download_button(
                    label="📥 Download Full Matrix (CSV)",
                    data=csv,
                    file_name=f"compatibility_matrix_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Check for any remaining conflicts (should be none with Hungarian)
            conflicts = check_mentor_conflicts(st.session_state.matches)
            
            if conflicts:
                st.markdown("""
                <div class="conflict-warning">
                    <strong>⚠️ Warning:</strong> Unexpected conflicts detected. Please review.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ All constraints satisfied: No mentor has >2 mentees!")
            
            # Results display
            st.divider()
            
            # Excel export button (auto-generate and download)
            st.subheader("📥 Export Results")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Results sheet
                results_data = []
                for match_data in st.session_state.matches:
                    for match in match_data['matches']:
                        results_data.append({
                            'Mentee_ID': match_data['mentee_id'],
                            'Rank': match['rank'],
                            'Mentor_ID': match['mentor_id'],
                            'Match_Percentage': match['match_percentage'],
                            'Match_Quality': match['match_quality'],
                            'Reasoning': match['reasoning']
                        })
                
                pd.DataFrame(results_data).to_excel(writer, sheet_name='Matches', index=False)
                st.session_state.mentees_df.to_excel(writer, sheet_name='Mentees', index=False)
                st.session_state.mentors_df.to_excel(writer, sheet_name='Mentors', index=False)
            
            # Add summary sheet
            if st.session_state.assignments:
                summary_data = {
                    'Metric': ['Total Matches', 'Average Score', 'Min Score', 'Max Score', 'Generated At'],
                    'Value': [
                        len(st.session_state.assignments),
                        f"{np.mean([s for _, _, s in st.session_state.assignments]):.1f}%",
                        f"{min([s for _, _, s in st.session_state.assignments])}%",
                        f"{max([s for _, _, s in st.session_state.assignments])}%",
                        time.strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                with pd.ExcelWriter(output, engine='openpyxl', mode='a') as writer:
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            output.seek(0)
            st.download_button(
                "📊 Download Excel Report",
                output.getvalue(),
                f"matches_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            
            st.divider()
            
            # Search box
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_query = st.text_input(
                    "🔍 Search Mentees", 
                    value=st.session_state.mentee_search,
                    placeholder="Type mentee ID to filter...",
                    label_visibility="collapsed"
                )
                st.session_state.mentee_search = search_query
            with search_col2:
                if st.button("🔄 Clear", width='stretch'):
                    st.session_state.mentee_search = ""
                    st.rerun()
            
            # Filter matches based on search
            filtered_matches = [
                m for m in st.session_state.matches 
                if search_query.lower() in m['mentee_id'].lower()
            ]
            
            if not filtered_matches and search_query:
                st.warning(f"No mentees found matching '{search_query}'")
            else:
                st.subheader(f"All Matches ({len(filtered_matches)})")
                
                # Display all match cards in a single list
                for match_data in filtered_matches:
                    mentee_id = match_data['mentee_id']
                    
                    for match in match_data['matches']:
                        quality_class = match['match_quality'].lower()
                        
                        st.markdown(f"""
                        <div class="match-card">
                            <h4>{mentee_id} → {match['mentor_id']}</h4>
                            <p><span class="percentage-badge {quality_class}">{match['match_percentage']}% - {match['match_quality']}</span></p>
                            <p><strong>Reasoning:</strong> {match['reasoning']}</p>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
