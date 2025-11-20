import tiktoken
from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
from info import (
    MENTEE_COLUMNS, MENTOR_COLUMNS
)

def validate_uploaded_file(df, file_type):
    """Validate uploaded files and show warnings (Improvement 18)"""
    warnings = []
    errors = []
    
    # Get the appropriate ID column name
    id_col_name = MENTEE_COLUMNS['id'] if file_type == 'mentee' else MENTOR_COLUMNS['id']
    
    # Check for required columns
    if file_type == 'mentee':
        if MENTEE_COLUMNS['id'] not in df.columns:
            errors.append(f"Missing required column: '{MENTEE_COLUMNS['id']}'")
    elif file_type == 'mentor':
        if MENTOR_COLUMNS['id'] not in df.columns:
            errors.append(f"Missing required column: '{MENTOR_COLUMNS['id']}'")
    
    # Check for empty values
    if id_col_name in df.columns:
        empty_count = df[id_col_name].isna().sum()
        if empty_count > 0:
            warnings.append(f"{empty_count} rows have empty '{id_col_name}' values")
    
    # CRITICAL: Check for duplicate IDs
    if id_col_name in df.columns:
        id_col = df[id_col_name].dropna()  # Remove NaN values
        duplicates = id_col[id_col.duplicated(keep=False)]
        if len(duplicates) > 0:
            unique_dups = duplicates.unique()
            entity_name = "mentee" if file_type == 'mentee' else "mentor"
            errors.append(f"DUPLICATE IDs FOUND: {len(unique_dups)} {entity_name} ID(s) appear multiple times!")
            errors.append(f"Duplicate {entity_name} IDs: {', '.join(map(str, unique_dups[:10]))}" + (" ..." if len(unique_dups) > 10 else ""))
    
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

def fetch_available_models(api_key: str):
    """
    Fetch available models from OpenAI API dynamically
    Returns a dict of model_id -> model_info
    """
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        
        # Filter OUT non-chat models (embeddings, TTS, whisper, etc.)
        # Include ALL chat/completion models (gpt-*, o1-*, o3-*, chatgpt-*, etc.)
        available_models = {}
        
        # Exclude these model types (not suitable for chat/completion)
        excluded_prefixes = (
            'text-embedding-', 'embedding-', 
            'tts-', 'whisper-', 
            'dall-e-', 'davinci-', 'curie-', 'babbage-', 'ada-'
        )
        
        for model in models.data:
            model_id = model.id
            
            # Skip models that are clearly not for chat/completion
            if any(model_id.startswith(prefix) for prefix in excluded_prefixes):
                continue
            
            # Skip if model ID suggests it's not a chat model
            if any(keyword in model_id.lower() for keyword in ['embed', 'whisper', 'tts', 'dall-e']):
                continue
            
            # Categorize model quality based on name patterns
            quality = "Good"
            speed = "Medium"
            cost_estimate = "Variable"
            
            # GPT-5 series
            if model_id.startswith('gpt-5'):
                if 'reasoning' in model_id:
                    quality = "Best"
                    speed = "Slow"
                    cost_estimate = "$5-40 per 1k pairs"
                elif 'nano' in model_id:
                    quality = "Basic"
                    speed = "Fastest"
                    cost_estimate = "$0.05-0.40 per 1k pairs"
                elif 'mini' in model_id:
                    quality = "Good"
                    speed = "Fast"
                    cost_estimate = "$0.25-3 per 1k pairs"
                else:
                    quality = "Excellent"
                    speed = "Medium"
                    cost_estimate = "$1.25-10 per 1k pairs"
            
            # o-series (reasoning models)
            elif model_id.startswith('o1-') or model_id.startswith('o3-'):
                if 'mini' in model_id:
                    quality = "Excellent"
                    speed = "Slow"
                    cost_estimate = "$3-12 per 1k pairs"
                else:
                    quality = "Exceptional"
                    speed = "Very Slow"
                    cost_estimate = "$15-60 per 1k pairs"
            
            # GPT-4o series
            elif 'gpt-4o' in model_id:
                if 'mini' in model_id:
                    quality = "Good"
                    speed = "Fast"
                    cost_estimate = "$0.15-0.60 per 1k pairs"
                else:
                    quality = "Excellent"
                    speed = "Fast"
                    cost_estimate = "$1.50-10 per 1k pairs"
            
            # GPT-4 series
            elif model_id.startswith('gpt-4'):
                quality = "Very Good"
                speed = "Medium"
                cost_estimate = "$5-30 per 1k pairs"
            
            # GPT-3.5 series
            elif model_id.startswith('gpt-3.5'):
                quality = "Basic"
                speed = "Very Fast"
                cost_estimate = "$0.05-1.50 per 1k pairs"
            
            # ChatGPT series
            elif model_id.startswith('chatgpt'):
                quality = "Good"
                speed = "Fast"
                cost_estimate = "$0.50-5 per 1k pairs"
            
            # Add to available models (include created timestamp for sorting)
            available_models[model_id] = {
                "name": model_id,
                "quality": quality,
                "speed": speed,
                "cost": cost_estimate,
                "created": model.created  # Unix timestamp from API
            }
        
        # If no models found, something went wrong
        if not available_models:
            raise Exception("No chat models found in API response")
        
        return available_models
        
    except Exception as e:
        st.error(f"Failed to fetch models from OpenAI: {str(e)}")
        # Return fallback models with estimated timestamps
        import time
        current_time = int(time.time())
        return {
            "gpt-4o": {"name": "gpt-4o", "quality": "Excellent", "speed": "Fast", "cost": "$1.50-10 per 1k pairs", "created": current_time - 100},
            "gpt-4o-mini": {"name": "gpt-4o-mini", "quality": "Good", "speed": "Fast", "cost": "$0.15-0.60 per 1k pairs", "created": current_time - 200},
            "o1-mini": {"name": "o1-mini", "quality": "Excellent", "speed": "Slow", "cost": "$3-12 per 1k pairs", "created": current_time - 300},
        }

def estimate_api_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Estimate the cost of an API call based on token usage"""
    # Pricing as of November 2024 (per 1M tokens)
    # Source: https://openai.com/api/pricing/
    pricing = {
        # GPT-4o models (latest)
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        
        # o1 reasoning models
        "o1-preview": {"prompt": 15.00, "completion": 60.00},
        "o1-mini": {"prompt": 3.00, "completion": 12.00},
        
        # GPT-4 Turbo models
        "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
        "gpt-4-turbo-preview": {"prompt": 10.00, "completion": 30.00},
        
        # GPT-4 classic
        "gpt-4": {"prompt": 30.00, "completion": 60.00},
        
        # GPT-3.5
        "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
        
        # Future models (placeholder pricing)
        "gpt-5": {"prompt": 20.00, "completion": 60.00},  # Placeholder
        "gpt-5.1": {"prompt": 20.00, "completion": 60.00},  # Placeholder
    }
    
    # Default to gpt-4o-mini if model not found
    if model not in pricing:
        model = "gpt-4o-mini"
    
    prompt_cost = (prompt_tokens / 1_000_000) * pricing[model]["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing[model]["completion"]
    
    return prompt_cost + completion_cost

def normalize_column_names(df):
    """
    Strip whitespace from column names to handle files with trailing spaces.
    Many Excel files have column names like 'Mentee Affiliation ' (with space).
    """
    df.columns = df.columns.str.strip()
    return df

def show_debug_info():
    """
    Display debug information (last API request/response) in an expandable section.
    Useful for troubleshooting warnings and errors.
    """
    if 'last_api_request' in st.session_state and 'last_api_response' in st.session_state:
        with st.expander("🔍 Debug Info - View Request & Response", expanded=False):
            st.markdown("**📤 Last API Request**")
            st.caption(f"Model: {st.session_state.last_api_request.get('model', 'N/A')} | Sent: {st.session_state.last_api_request.get('timestamp', 'N/A')}")
            # Use st.code() instead of st.text_area() - no keys needed, better for display
            st.code(
                st.session_state.last_api_request.get('full_message', 'No request data available'),
                language=None,
                line_numbers=False
            )
            
            st.divider()
            
            st.markdown("**📥 Last AI Response**")
            tokens = st.session_state.last_api_response.get('tokens', {})
            st.caption(f"Tokens: Prompt={tokens.get('prompt', 0):,} | Completion={tokens.get('completion', 0):,} | Total={tokens.get('total', 0):,} | Received: {st.session_state.last_api_response.get('timestamp', 'N/A')}")
            # Use st.code() instead of st.text_area() - no keys needed, better for display
            st.code(
                st.session_state.last_api_response.get('content', 'No response data available'),
                language=None,
                line_numbers=False
            )
    else:
        st.caption("💡 Debug info will appear here after your first API call")

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
    
    # Normalize column names (strip whitespace only - preserve special characters)
    df = normalize_column_names(df)
    
    # Clean string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', '')
        df[col] = df[col].replace('', np.nan)
    
    return df