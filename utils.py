import numpy as np
import pandas as pd
import streamlit as st
import tiktoken

from info import (
    API_PRICING,
    CLAUDE_MODELS,
    DEFAULT_MODEL,
    MENTEE_COLUMNS,
    MENTOR_COLUMNS,
    MODEL_CONFIGS,
    RELATED_FIELD_GROUPS,
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

def fetch_available_models(api_key=None):
    """Static list of the selectable Claude models: model_id -> display info (api_key is unused)."""
    return {
        m: {
            "name": m,
            "quality": MODEL_CONFIGS[m]["quality"],
            "speed": MODEL_CONFIGS[m]["speed"],
            "cost": f"${API_PRICING[m]['prompt']:.2f} in / ${API_PRICING[m]['completion']:.2f} out per 1M tokens",
        }
        for m in CLAUDE_MODELS
    }

def estimate_api_cost(prompt_tokens: int, completion_tokens: int, model: str = DEFAULT_MODEL) -> float:
    """Estimate the cost (USD) of an API call from token usage; unknown models price as the default."""
    if model not in API_PRICING:
        model = DEFAULT_MODEL
    
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

# Safe cap on estimated OUTPUT tokens per batch, with headroom under Claude's 128K output cap.
MATRIX_OUTPUT_CAP = 100000


def calculate_optimal_batch_size(
    total_mentors: int,
    num_mentees: int,
) -> int:
    """
    Largest mentor batch whose estimated OUTPUT tokens stay under MATRIX_OUTPUT_CAP.

    With Claude, input is no longer the constraint (huge context + prompt caching); the 128K
    output cap is. Estimated output is ~15 tokens per mentor-mentee pair (the 10/pair of
    estimate_output_tokens plus the 1.5x headroom the matrix call budgets with).

    Args:
        total_mentors: Total number of mentors to process
        num_mentees: Total number of mentees

    Returns:
        Optimal batch size (number of mentors per batch)
    """
    if num_mentees <= 0:
        return min(30, max(total_mentors, 1))

    # Try from high to low; return the largest batch whose estimated output fits.
    for batch_size in range(min(30, total_mentors), 0, -1):
        if int(estimate_output_tokens(batch_size, num_mentees) * 1.5) <= MATRIX_OUTPUT_CAP:
            return batch_size

    # Even one mentor overflows the cap (enormous cohort): one mentor per batch is the floor.
    return 1

def estimate_batch_tokens(
    num_mentors: int,
    num_mentees: int,
) -> tuple[int, int]:
    """
    Estimate input and output tokens for a batch.
    
    Args:
        num_mentors: Number of mentors in this batch
        num_mentees: Number of mentees
    
    Returns:
        Tuple of (input_tokens, output_tokens)
    """
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
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS[DEFAULT_MODEL])

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