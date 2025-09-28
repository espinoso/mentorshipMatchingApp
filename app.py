import streamlit as st
import pandas as pd
import numpy as np
import json
import openai
from io import BytesIO
import re
from typing import List, Dict, Any
import time

# Page configuration
st.set_page_config(
    page_title="Mentorship Matching System",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .mentee-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .mentee-card:hover {
        background-color: #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .match-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .percentage-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        color: white;
    }
    .excellent { background-color: #28a745; }
    .good { background-color: #17a2b8; }
    .fair { background-color: #ffc107; color: #212529; }
    .conflict-warning {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def clean_dataframe(df):
    """Clean and validate dataframe"""
    # Remove empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Clean column names - remove extra spaces and special characters
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
    """Validate that uploaded files have required columns"""
    required_mentee_cols = [
        'Code Mentee', 'Mentee Field of expertise', 'Mentee Specialization',
        'Mentee Specific hard skills that you have', 'Mentee Areas where guidance is needed',
        'Mentee Career goals for the next 2 years'
    ]
    
    required_mentor_cols = [
        'Code mentor', 'Field of expertise', 'Mentor Specialization',
        'Mentor Specific Hard Skills and Professional Competencies has Mastered',
        'Mentor Years of Professional Experience in his her Field'
    ]
    
    if file_type == 'mentee':
        missing_cols = [col for col in required_mentee_cols if col not in df.columns]
    elif file_type == 'mentor':
        missing_cols = [col for col in required_mentor_cols if col not in df.columns]
    else:  # training file
        # Training files should have both mentee and mentor columns
        missing_cols = []
        
    return missing_cols

def generate_sample_data():
    """Generate sample data for testing"""
    # Sample mentee data
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
    
    # Sample mentor data
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

def create_default_prompt():
    """Create the default prompt for the LLM"""
    prompt = """You are an expert mentorship coordinator with deep understanding of academic and professional development. Your task is to match mentees with mentors based on the provided data.

TRAINING DATA CONTEXT:
The training files contain historical examples of successful mentorship pairings. Use these examples to understand what constitutes effective matches in this domain.

MATCHING CRITERIA:
Analyze the following factors to determine compatibility:
- Field of expertise alignment and complementarity
- Specialization overlap and learning opportunities  
- Hard skills compatibility and knowledge transfer potential
- Career goals alignment with mentor's experience
- Guidance needs matching mentor's strengths
- Professional experience level appropriateness

CONSTRAINTS:
- Each mentor can be assigned to a maximum of 2 mentees
- Provide exactly 3 mentor recommendations per mentee, ranked by fit quality
- If the same mentor appears in multiple top recommendations, prioritize based on overall match quality

OUTPUT FORMAT:
For each mentee, provide a JSON object with:
{
  "mentee_id": "mentee_code",
  "matches": [
    {
      "rank": 1,
      "mentor_id": "mentor_code", 
      "match_percentage": 85,
      "match_quality": "Excellent",
      "reasoning": "Detailed explanation of why this mentor is the best fit..."
    },
    {
      "rank": 2,
      "mentor_id": "mentor_code",
      "match_percentage": 78,
      "match_quality": "Good", 
      "reasoning": "Explanation of the match strengths and considerations..."
    },
    {
      "rank": 3,
      "mentor_id": "mentor_code",
      "match_percentage": 65,
      "match_quality": "Fair",
      "reasoning": "Explanation of why this is a viable but less optimal match..."
    }
  ]
}

Provide ONLY valid JSON output with no additional text or explanations outside the JSON structure."""

    return prompt

def call_openai_api(prompt, mentees_df, mentors_df, training_data, api_key):
    """Call OpenAI API to generate matches"""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare data for the prompt
        mentees_info = mentees_df.to_string(index=False)
        mentors_info = mentors_df.to_string(index=False)
        training_info = ""
        
        if training_data:
            for i, df in enumerate(training_data):
                training_info += f"\n\nTraining File {i+1}:\n{df.head(10).to_string(index=False)}\n"
        
        full_prompt = f"""{prompt}

CURRENT MENTEES TO MATCH:
{mentees_info}

AVAILABLE MENTORS:
{mentors_info}

TRAINING EXAMPLES:
{training_info}

Generate matches for all mentees listed above."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert mentorship coordinator. Respond only with valid JSON."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None

def parse_llm_response(response_text):
    """Parse the LLM response and extract matches"""
    try:
        # Clean the response text
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Try to parse as JSON
        matches = json.loads(response_text)
        return matches
    except json.JSONDecodeError as e:
        st.error(f"Error parsing LLM response: {str(e)}")
        st.text("Raw response:")
        st.text(response_text)
        return None

def check_mentor_conflicts(matches):
    """Check if any mentor is assigned to more than 2 mentees"""
    mentor_assignments = {}
    conflicts = []
    
    for match_data in matches:
        mentee_id = match_data['mentee_id']
        for match in match_data['matches']:
            mentor_id = match['mentor_id']
            if mentor_id not in mentor_assignments:
                mentor_assignments[mentor_id] = []
            mentor_assignments[mentor_id].append((mentee_id, match['rank']))
    
    for mentor_id, assignments in mentor_assignments.items():
        if len(assignments) > 2:
            conflicts.append({
                'mentor_id': mentor_id,
                'assignments': assignments,
                'count': len(assignments)
            })
    
    return conflicts

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤝 Mentorship Matching System</h1>
        <p>AI-powered mentor-mentee matching for academic and professional development</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'training_files' not in st.session_state:
        st.session_state.training_files = []
    if 'mentees_df' not in st.session_state:
        st.session_state.mentees_df = None
    if 'mentors_df' not in st.session_state:
        st.session_state.mentors_df = None
    if 'matches' not in st.session_state:
        st.session_state.matches = None
    if 'custom_prompt' not in st.session_state:
        st.session_state.custom_prompt = create_default_prompt()
    
    # Sidebar for file uploads and settings
    with st.sidebar:
        st.header("📁 File Uploads")
        
        # Training files upload
        st.subheader("1. Training Files")
        st.write("Upload historical mentorship data (Excel files)")
        training_files = st.file_uploader(
            "Choose training files", 
            type=['xlsx'], 
            accept_multiple_files=True,
            key="training_upload"
        )
        
        if training_files and len(training_files) != len(st.session_state.training_files):
            st.session_state.training_files = []
            for file in training_files:
                try:
                    df = pd.read_excel(file)
                    df = clean_dataframe(df)
                    st.session_state.training_files.append(df)
                    st.success(f"✅ {file.name} loaded successfully")
                except Exception as e:
                    st.error(f"❌ Error loading {file.name}: {str(e)}")
        
        # Mentee file upload
        st.subheader("2. Mentees File")
        mentee_file = st.file_uploader("Choose mentees file", type=['xlsx'], key="mentee_upload")
        
        if mentee_file:
            try:
                df = pd.read_excel(mentee_file)
                df = clean_dataframe(df)
                missing_cols = validate_file_structure(df, 'mentee')
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.session_state.mentees_df = df
                    st.success(f"✅ {mentee_file.name} loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading mentee file: {str(e)}")
        
        # Mentor file upload  
        st.subheader("3. Mentors File")
        mentor_file = st.file_uploader("Choose mentors file", type=['xlsx'], key="mentor_upload")
        
        if mentor_file:
            try:
                df = pd.read_excel(mentor_file)
                df = clean_dataframe(df)
                missing_cols = validate_file_structure(df, 'mentor')
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.session_state.mentors_df = df
                    st.success(f"✅ {mentor_file.name} loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading mentor file: {str(e)}")
        
        # API Key input
        st.subheader("4. OpenAI API Key")
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        
        # Use sample data button
        st.subheader("🧪 Testing")
        if st.button("Use Sample Data"):
            sample_mentees, sample_mentors = generate_sample_data()
            st.session_state.mentees_df = sample_mentees
            st.session_state.mentors_df = sample_mentors
            st.success("✅ Sample data loaded!")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "🔧 Customize Prompt", "🎯 Results"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Training Files")
            if st.session_state.training_files:
                for i, df in enumerate(st.session_state.training_files):
                    st.write(f"**File {i+1}:** {len(df)} records")
                    with st.expander(f"Preview Training File {i+1}"):
                        st.dataframe(df.head())
            else:
                st.info("No training files uploaded yet")
            
            st.subheader("👥 Mentees")
            if st.session_state.mentees_df is not None:
                st.write(f"**Total mentees:** {len(st.session_state.mentees_df)}")
                with st.expander("Preview Mentees Data"):
                    st.dataframe(st.session_state.mentees_df)
            else:
                st.info("No mentee file uploaded yet")
        
        with col2:
            st.subheader("👨‍🏫 Mentors")
            if st.session_state.mentors_df is not None:
                st.write(f"**Total mentors:** {len(st.session_state.mentors_df)}")
                with st.expander("Preview Mentors Data"):
                    st.dataframe(st.session_state.mentors_df)
            else:
                st.info("No mentor file uploaded yet")
    
    with tab2:
        st.header("Customize LLM Prompt")
        st.write("Modify the prompt sent to the AI to adjust matching criteria and behavior:")
        
        custom_prompt = st.text_area(
            "LLM Prompt",
            value=st.session_state.custom_prompt,
            height=400,
            help="Edit this prompt to change how the AI evaluates mentor-mentee compatibility"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes"):
                st.session_state.custom_prompt = custom_prompt
                st.success("Prompt updated!")
        
        with col2:
            if st.button("🔄 Reset to Default"):
                st.session_state.custom_prompt = create_default_prompt()
                st.success("Prompt reset to default!")
                st.rerun()
    
    with tab3:
        st.header("Matching Results")
        
        # Check if all required data is available
        ready_to_match = (
            st.session_state.mentees_df is not None and 
            st.session_state.mentors_df is not None and 
            api_key
        )
        
        if ready_to_match:
            if st.button("🚀 Generate Matches", type="primary"):
                with st.spinner("🤖 AI is analyzing and generating matches..."):
                    response = call_openai_api(
                        st.session_state.custom_prompt,
                        st.session_state.mentees_df,
                        st.session_state.mentors_df,
                        st.session_state.training_files,
                        api_key
                    )
                    
                    if response:
                        matches = parse_llm_response(response)
                        if matches:
                            st.session_state.matches = matches
                            st.success("✅ Matches generated successfully!")
                        else:
                            st.error("Failed to parse AI response")
                    else:
                        st.error("Failed to get response from AI")
        else:
            missing_items = []
            if st.session_state.mentees_df is None:
                missing_items.append("Mentees file")
            if st.session_state.mentors_df is None:
                missing_items.append("Mentors file") 
            if not api_key:
                missing_items.append("OpenAI API key")
            
            st.warning(f"⚠️ Please provide: {', '.join(missing_items)}")
        
        # Display results
        if st.session_state.matches:
            # Check for conflicts
            conflicts = check_mentor_conflicts(st.session_state.matches)
            if conflicts:
                st.markdown("""
                <div class="conflict-warning">
                    <strong>⚠️ Mentor Assignment Conflicts Detected:</strong>
                    Some mentors are recommended for more than 2 mentees. Please review the assignments below.
                </div>
                """, unsafe_allow_html=True)
                
                for conflict in conflicts:
                    st.error(f"Mentor {conflict['mentor_id']} assigned to {conflict['count']} mentees: {[f'{a[0]} (rank {a[1]})' for a in conflict['assignments']]}")
            
            # Interactive results display
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("👥 Select Mentee")
                selected_mentee = None
                
                for match_data in st.session_state.matches:
                    mentee_id = match_data['mentee_id']
                    if st.button(f"📋 {mentee_id}", key=f"btn_{mentee_id}"):
                        st.session_state.selected_mentee = mentee_id
                
                if 'selected_mentee' in st.session_state:
                    selected_mentee = st.session_state.selected_mentee
            
            with col2:
                if 'selected_mentee' in st.session_state:
                    # Find the selected mentee's matches
                    mentee_matches = None
                    for match_data in st.session_state.matches:
                        if match_data['mentee_id'] == st.session_state.selected_mentee:
                            mentee_matches = match_data['matches']
                            break
                    
                    if mentee_matches:
                        st.subheader(f"🎯 Matches for {st.session_state.selected_mentee}")
                        
                        for match in mentee_matches:
                            quality_class = match['match_quality'].lower()
                            
                            st.markdown(f"""
                            <div class="match-card">
                                <h4>#{match['rank']} - {match['mentor_id']}</h4>
                                <p>
                                    <span class="percentage-badge {quality_class}">
                                        {match['match_percentage']}% - {match['match_quality']}
                                    </span>
                                </p>
                                <p><strong>Reasoning:</strong></p>
                                <p>{match['reasoning']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("👆 Select a mentee from the left to view their matches")
            
            # Download results
            st.subheader("📥 Download Results")
            if st.button("📊 Generate Excel Report"):
                # Create downloadable Excel file
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = []
                    for match_data in st.session_state.matches:
                        mentee_id = match_data['mentee_id']
                        for match in match_data['matches']:
                            summary_data.append({
                                'Mentee_ID': mentee_id,
                                'Mentor_Rank': match['rank'],
                                'Mentor_ID': match['mentor_id'],
                                'Match_Percentage': match['match_percentage'],
                                'Match_Quality': match['match_quality'],
                                'Reasoning': match['reasoning']
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Matching_Results', index=False)
                    
                    # Original data sheets
                    st.session_state.mentees_df.to_excel(writer, sheet_name='Mentees_Data', index=False)
                    st.session_state.mentors_df.to_excel(writer, sheet_name='Mentors_Data', index=False)
                
                output.seek(0)
                st.download_button(
                    label="📥 Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"mentorship_matches_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()