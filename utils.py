import tiktoken
from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
from info import (
    MENTEE_COLUMNS, MENTOR_COLUMNS, API_PRICING, EXCLUDED_MODEL_PREFIXES, MODEL_CONFIGS, RELATED_FIELD_GROUPS
)

def validate_uploaded_file(df, file_type):
    """Validate uploaded files and show warnings"""
    warnings = []
    errors = []
    
    # Get column definitions based on file type
    if file_type == 'mentee':
        columns = MENTEE_COLUMNS
        entity_name = "mentee"
    else:
        columns = MENTOR_COLUMNS
        entity_name = "mentor"
    
    # REQUIRED COLUMNS (errors if missing)
    required_columns = {
        'id': f"Missing required column: '{columns['id']}' - Needed for identification",
        'field': f"Missing required column: '{columns['field']}' - Critical for matching",
        'specialization': f"Missing required column: '{columns['specialization']}' - Critical for matching"
    }
    
    for col_key, error_msg in required_columns.items():
        if columns[col_key] not in df.columns:
            errors.append(error_msg)
    
    # IMPORTANT COLUMNS (warnings if missing - needed for certain features)
    important_columns = {
        'country': f"Missing important column: '{columns['country']}' - Needed for country-based restrictions"
    }
    
    for col_key, warning_msg in important_columns.items():
        if columns[col_key] not in df.columns:
            warnings.append(warning_msg)
    
    # Validate ID column (if present)
    id_col_name = columns['id']
    if id_col_name in df.columns:
        empty_count = df[id_col_name].isna().sum()
        if empty_count > 0:
            warnings.append(f"{empty_count} rows have empty '{id_col_name}' values")
        
        # CRITICAL: Check for duplicate IDs
        id_col = df[id_col_name].dropna()  # Remove NaN values
        duplicates = id_col[id_col.duplicated(keep=False)]
        if len(duplicates) > 0:
            unique_dups = duplicates.unique()
            errors.append(f"DUPLICATE IDs FOUND: {len(unique_dups)} {entity_name} ID(s) appear multiple times!")
            errors.append(f"Duplicate {entity_name} IDs: {', '.join(map(str, unique_dups[:10]))}" + (" ..." if len(unique_dups) > 10 else ""))
    
    # Validate Field column (if present)
    if columns['field'] in df.columns:
        empty_field_count = df[columns['field']].isna().sum()
        if empty_field_count > 0:
            warnings.append(f"{empty_field_count} rows have empty '{columns['field']}' values - matching may be incomplete")
    
    # Validate Specialization column (if present)
    if columns['specialization'] in df.columns:
        empty_spec_count = df[columns['specialization']].isna().sum()
        if empty_spec_count > 0:
            warnings.append(f"{empty_spec_count} rows have empty '{columns['specialization']}' values - matching quality may be reduced")
    
    # Overall data completeness
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
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))    

def fetch_available_models(api_key: str):
    """
    Fetch available models from OpenAI API dynamically
    Returns a dict of model_id -> model_info
    """
    try:
        client = OpenAI(api_key=api_key)
        # Try to list models - this may fail with Responses API, so we have fallback
        try:
            models = client.models.list()
        except Exception as list_error:
            # If models.list() fails (e.g., Responses API doesn't support it), use fallback
            raise Exception(f"Models API not available: {str(list_error)}")
        
        # Filter out non-chat models (embeddings, TTS, whisper, etc.)
        # Include all chat/completion models (gpt-*, o1-*, o3-*, chatgpt-*, etc.)
        available_models = {}
        
        for model in models.data:
            model_id = model.id
            
            if any(model_id.startswith(prefix) for prefix in EXCLUDED_MODEL_PREFIXES):
                continue
            
            if any(keyword in model_id.lower() for keyword in ['embed', 'whisper', 'tts', 'dall-e']):
                continue
            
            quality = "Good"
            speed = "Medium"
            cost_estimate = "Variable"
            
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
            
            elif model_id.startswith('o1-') or model_id.startswith('o3-'):
                if 'mini' in model_id:
                    quality = "Excellent"
                    speed = "Slow"
                    cost_estimate = "$3-12 per 1k pairs"
                else:
                    quality = "Exceptional"
                    speed = "Very Slow"
                    cost_estimate = "$15-60 per 1k pairs"
            
            elif 'gpt-4o' in model_id:
                if 'mini' in model_id:
                    quality = "Good"
                    speed = "Fast"
                    cost_estimate = "$0.15-0.60 per 1k pairs"
                else:
                    quality = "Excellent"
                    speed = "Fast"
                    cost_estimate = "$1.50-10 per 1k pairs"
            
            elif model_id.startswith('gpt-4'):
                quality = "Very Good"
                speed = "Medium"
                cost_estimate = "$5-30 per 1k pairs"
            
            elif model_id.startswith('gpt-3.5'):
                quality = "Basic"
                speed = "Very Fast"
                cost_estimate = "$0.05-1.50 per 1k pairs"
            
            elif model_id.startswith('chatgpt'):
                quality = "Good"
                speed = "Fast"
                cost_estimate = "$0.50-5 per 1k pairs"
            
            available_models[model_id] = {
                "name": model_id,
                "quality": quality,
                "speed": speed,
                "cost": cost_estimate,
                "created": model.created
            }
        
        if not available_models:
            raise Exception("No chat models found in API response")
        
        return available_models
        
    except Exception as e:
        # Fallback to hardcoded models if API call fails (e.g., Responses API doesn't support models.list())
        import time
        current_time = int(time.time())
        # Return the models we actually use, sorted by preference
        fallback_models = {
            "gpt-5.1": {"name": "gpt-5.1", "quality": "Excellent", "speed": "Medium", "cost": "$1.25-10 per 1k pairs", "created": current_time - 100},
            "gpt-5-mini": {"name": "gpt-5-mini", "quality": "Good", "speed": "Fast", "cost": "$0.25-3 per 1k pairs", "created": current_time - 200},
            "gpt-5-nano": {"name": "gpt-5-nano", "quality": "Basic", "speed": "Fastest", "cost": "$0.05-0.40 per 1k pairs", "created": current_time - 300},
            "gpt-4.1": {"name": "gpt-4.1", "quality": "Excellent", "speed": "Medium", "cost": "$1.25-10 per 1k pairs", "created": current_time - 400},
            "gpt-4.1-mini": {"name": "gpt-4.1-mini", "quality": "Good", "speed": "Fast", "cost": "$0.25-3 per 1k pairs", "created": current_time - 500},
            "gpt-4.1-nano": {"name": "gpt-4.1-nano", "quality": "Basic", "speed": "Fastest", "cost": "$0.05-0.40 per 1k pairs", "created": current_time - 600},
        }
        # Only show error in debug mode, otherwise silently use fallback
        if is_debug_mode():
            st.warning(f"⚠️ Could not fetch models from API (using fallback): {str(e)}")
        return fallback_models

def estimate_api_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Estimate the cost of an API call based on token usage"""
    if model not in API_PRICING:
        model = "gpt-4o-mini"
    
    prompt_cost = (prompt_tokens / 1_000_000) * API_PRICING[model]["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * API_PRICING[model]["completion"]
    
    return prompt_cost + completion_cost

def estimate_input_tokens_stateless(num_mentors: int, num_mentees: int, base_prompt_tokens: int = 3500) -> int:
    """
    Estimate input tokens for stateless mode (store=False).
    
    Based on actual API usage: 14,900 tokens for 15 mentors × 62 mentees.
    Reverse-engineered: ~600 tokens per mentor, ~50 tokens per mentee, ~3,500 base.
    
    Args:
        num_mentors: Number of mentors in this batch
        num_mentees: Total number of mentees (always sent in stateless)
        base_prompt_tokens: Base prompt overhead (instructions, rubric, etc.)
    
    Returns:
        Estimated input tokens
    """
    # Base prompt (instructions, rubric, JSON schema, etc.) - ~3,500 tokens
    # ~600 tokens per mentor (mentor data) - based on actual: (14,900 - 3,500 - 3,100) / 15 ≈ 600
    # ~50 tokens per mentee (mentee data)
    mentor_tokens = num_mentors * 600
    mentee_tokens = num_mentees * 50
    return base_prompt_tokens + mentor_tokens + mentee_tokens

def estimate_input_tokens_stateful_first(num_mentors: int, num_mentees: int, base_prompt_tokens: int = 3500) -> int:
    """
    Estimate input tokens for first batch in stateful mode (full context).
    Same as stateless for the first batch.
    """
    return estimate_input_tokens_stateless(num_mentors, num_mentees, base_prompt_tokens)

def estimate_input_tokens_stateful_subsequent(num_mentors: int, base_overhead: int = 2000) -> int:
    """
    Estimate input tokens for subsequent batches in stateful mode.
    
    Based on user observation: ~5k tokens for subsequent batches.
    Only new mentors are sent, mentees are preserved in conversation context.
    
    Args:
        num_mentors: Number of NEW mentors in this batch
        base_overhead: Base overhead for API call structure
    
    Returns:
        Estimated input tokens
    """
    # Much lower - only new mentors (~500 tokens per new mentor)
    return base_overhead + (num_mentors * 500)

def estimate_output_tokens(num_mentors: int, num_mentees: int) -> int:
    """
    Estimate output tokens for compatibility matrix.
    
    Based on actual API usage: 9,458 tokens for 15 mentors × 62 mentees.
    That's approximately 10 tokens per mentee-mentor pair (9,458 / 930 = 10.17).
    
    Args:
        num_mentors: Number of mentors in batch
        num_mentees: Number of mentees
    
    Returns:
        Estimated output tokens
    """
    # Base formula: 10 tokens per mentee-mentor pair
    # Based on actual usage: 9,458 / (15 × 62) = 10.17 tokens per pair
    return 10 * num_mentees * num_mentors

def calculate_optimal_batch_size(
    total_mentors: int,
    num_mentees: int,
    model: str,
    use_stateful: bool = False,
    safety_margin: float = 0.90
) -> int:
    """
    Calculate optimal batch size based on TPM (Tokens Per Minute) limits.
    
    Strategy:
    1. Calculate tokens per batch (input + output)
    2. Ensure batch fits within TPM limit (with safety margin)
    3. Maximize batch size for efficiency (fewer API calls)
    
    Args:
        total_mentors: Total number of mentors to process
        num_mentees: Total number of mentees
        model: Model name (e.g., 'gpt-5.1')
        use_stateful: Whether stateful mode is enabled
        safety_margin: Fraction of TPM to use (0.90 = 90% of limit, allows up to ~27k tokens for 30k TPM)
    
    Returns:
        Optimal batch size (number of mentors per batch)
    """
    model_config = get_model_config(model)
    tpm_limit = model_config.get('tpm_limit', 30000)  # Default fallback
    usable_tpm = int(tpm_limit * safety_margin)
    
    # Try different batch sizes, starting from high and going down
    # This maximizes efficiency (fewer API calls)
    for batch_size in range(min(30, total_mentors), 4, -1):
        # Estimate tokens for this batch
        if use_stateful:
            # First batch is expensive (full context)
            # Subsequent batches are cheap (only new mentors)
            # We'll optimize for subsequent batches (more common)
            input_tokens = estimate_input_tokens_stateful_subsequent(batch_size)
        else:
            input_tokens = estimate_input_tokens_stateless(batch_size, num_mentees)
        
        output_tokens = estimate_output_tokens(batch_size, num_mentees)
        total_tokens = input_tokens + output_tokens
        
        # Check if this batch size fits within TPM limit
        if total_tokens <= usable_tpm:
            return batch_size
    
    # Fallback to minimum safe batch size
    return 5

def estimate_batch_tokens(
    num_mentors: int,
    num_mentees: int,
    model: str,
    use_stateful: bool = False,
    is_first_batch: bool = True
) -> tuple[int, int]:
    """
    Estimate input and output tokens for a batch.
    
    Args:
        num_mentors: Number of mentors in this batch
        num_mentees: Number of mentees
        model: Model name
        use_stateful: Whether stateful mode is enabled
        is_first_batch: Whether this is the first batch (for stateful mode)
    
    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    if use_stateful and not is_first_batch:
        input_tokens = estimate_input_tokens_stateful_subsequent(num_mentors)
    else:
        input_tokens = estimate_input_tokens_stateless(num_mentors, num_mentees)
    
    output_tokens = estimate_output_tokens(num_mentors, num_mentees)
    
    return input_tokens, output_tokens

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
            st.code(
                st.session_state.last_api_request.get('full_message', 'No request data available'),
                language=None,
                line_numbers=False
            )
            
            st.divider()
            
            st.markdown("**📥 Last AI Response**")
            tokens = st.session_state.last_api_response.get('tokens', {})
            st.caption(f"Tokens: Prompt={tokens.get('prompt', 0):,} | Completion={tokens.get('completion', 0):,} | Total={tokens.get('total', 0):,} | Received: {st.session_state.last_api_response.get('timestamp', 'N/A')}")
            st.code(
                st.session_state.last_api_response.get('content', 'No response data available'),
                language=None,
                line_numbers=False
            )
    else:
        st.caption("💡 Debug info will appear here after your first API call")

def clean_dataframe(df):
    """Clean and validate dataframe"""
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Check for "Unnamed" columns (indicates empty header rows)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        for idx in range(len(df)):
            row = df.iloc[idx]
            non_empty_values = [val for val in row if pd.notna(val) and str(val).strip() != '']
            if len(non_empty_values) >= 3:
                df.columns = df.iloc[idx].values
                df = df.iloc[idx+1:].reset_index(drop=True)
                break
    
    df = normalize_column_names(df)
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', '')
        df[col] = df[col].replace('', np.nan)
    
    return df

def get_model_config(model_name):
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['default'])

def is_field_compatible(field1, field2):
    """
    Check if two fields are same or related
    Used for validation to detect mismatched high scores
    """
    f1 = str(field1).lower().strip()
    f2 = str(field2).lower().strip()
    
    if f1 == f2:
        return True
    
    for group in RELATED_FIELD_GROUPS:
        if f1 in group and f2 in group:
            return True
    
    return False

def is_debug_mode():
    """
    Check if debug mode is enabled based on DEBUG_MODE environment variable.
    
    Logic:
    - If DEBUG_MODE env var not present → debug_mode = True (ON by default)
    - If DEBUG_MODE = "TRUE" → debug_mode = True
    - If DEBUG_MODE = "FALSE" → debug_mode = False
    - Otherwise → defaults to True
    
    Returns:
        bool: True if debug mode is enabled, False otherwise
    """
    import os
    
    # Check session state first (if already initialized)
    if 'debug_mode' in st.session_state:
        return st.session_state.debug_mode
    
    # Read from environment variable
    debug_env = os.getenv('DEBUG_MODE', '').upper().strip()
    
    if debug_env == 'FALSE':
        return False
    elif debug_env == 'TRUE':
        return True
    else:
        # Default: ON if env var not present or invalid
        return True


def extract_openai_answer(response):
    """
    Universal extractor for OpenAI Responses API.
    
    Handles reasoning models (o1, o3, gpt-5-mini, gpt-5-nano)
    and standard models (gpt-5.1, gpt-4.1, etc.).
    
    For reasoning models: Looks for type == "message" items (ignores type == "reasoning")
    For non-reasoning models: Uses response.output_text directly
    
    Args:
        response: OpenAI Responses API response object
        
    Returns:
        answer_text (str): The extracted text content
        
    Raises:
        ValueError: If no valid answer text is found
    """
    
    # 1. Fast path for non-reasoning models (gpt-4.1, gpt-5.1)
    # These responses always contain .output_text
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)
    
    # 2. Fallback to scanning output items for assistant messages
    if hasattr(response, "output"):
        for item in response.output:
            # Skip reasoning objects - they're metadata, not the answer
            if getattr(item, "type", None) == "reasoning":
                continue
            
            # Look for message items (this is where the actual answer is for reasoning models)
            if getattr(item, "type", None) == "message":
                # message.content is a list of content parts (text, images, etc.)
                if hasattr(item, "content") and isinstance(item.content, list):
                    for part in item.content:
                        # Look for output_text parts
                        if hasattr(part, "type") and getattr(part, "type", None) == "output_text":
                            if hasattr(part, "text"):
                                return str(part.text)
                        # Fallback: if part has text attribute directly
                        elif hasattr(part, "text"):
                            text_value = part.text
                            if isinstance(text_value, str):
                                return text_value
                            elif hasattr(text_value, "text"):
                                return str(text_value.text)
    
    # 3. Fallback for future formats: raw JSON dictionaries
    if isinstance(response, dict):
        # Reasoning or non-reasoning, same approach
        output = response.get("output") or []
        for item in output:
            # Skip reasoning objects
            if item.get("type") == "reasoning":
                continue
            
            if item.get("type") == "message":
                parts = item.get("content", [])
                for p in parts:
                    if p.get("type") == "output_text":
                        text = p.get("text")
                        if text:
                            return str(text)
        
        # Standard model shortcut
        if "output_text" in response:
            text = response["output_text"]
            if text:
                return str(text)
    
    # 4. If we got here, nothing was extracted
    raise ValueError(
        "Could not extract assistant response text. "
        "Response format may be unsupported or empty. "
        "For reasoning models, ensure there's a message item (not just reasoning objects)."
    )


def format_time(seconds):
    """Format seconds to human-readable time string
    
    Args:
        seconds: Time in seconds (float or int)
        
    Returns:
        str: Formatted time string (e.g., "23s", "2m 34s", "1h 15m")
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


def calculate_eta(completed, total, elapsed_times):
    """Calculate estimated time remaining based on average batch time
    
    Args:
        completed: Number of completed batches (int)
        total: Total number of batches (int)
        elapsed_times: List of elapsed times for completed batches (list of floats)
        
    Returns:
        float or None: Estimated seconds remaining, or None if cannot calculate
    """
    if not elapsed_times or completed >= total:
        return None
    
    avg_time = sum(elapsed_times) / len(elapsed_times)
    remaining = total - completed
    return avg_time * remaining