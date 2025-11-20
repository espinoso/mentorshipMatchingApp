import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import time

from info import (
    MENTEE_COLUMNS, MENTOR_COLUMNS, MODEL_TOKEN_LIMITS
)

from ui import (
    CSS_STYLES
)

from utils import (
    estimate_tokens, estimate_api_cost, show_debug_info,
    get_model_config, extract_openai_answer, is_debug_mode,
    format_time, calculate_eta
)

from prompts import (
    get_prompt_for_api, create_default_prompt
)

from matching import (
    parse_matrix_response, apply_country_restrictions
)

from validation import (
    prepare_data_for_assistant
)

from ui_config import (
    render_config_tab
)

from ui_training import (
    render_training_tab
)

from ui_upload import (
    render_upload_tab
)

from ui_prompt import (
    render_prompt_tab
)

from ui_overview import (
    render_overview_tab
)

from ui_matches import (
    render_matches_tab
)

from ui_processing import (
    render_processing_tab
)

st.set_page_config(
    page_title="Mentorship Matching System",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)


def call_openai_api_with_chat_completions(prompt, mentees_df, mentors_df, training_data, api_key, model="gpt-4o-mini", mentor_subset=None, use_stateful=False, previous_response_id=None, is_first_batch=True):
    """Use OpenAI Responses API - replaces Chat Completions API
    
    Migration to Responses API per:
    https://platform.openai.com/docs/guides/migrate-to-responses
    
    Supports both stateless (store=False) and stateful (store=True) modes:
    - Stateless: Each call is independent with full context (default, backward compatible)
    - Stateful: Context is maintained across calls, only new data sent in subsequent calls
    
    Args:
        mentor_subset: Optional list of mentor IDs to evaluate (for batching)
        use_stateful: Enable stateful mode (store=True) for efficient batching
        previous_response_id: Response ID from previous call (for stateful chaining)
        is_first_batch: Whether this is the first batch (needs full context)
    """
    try:
        client = OpenAI(api_key=api_key)
        
        if hasattr(client, 'responses'):
            responses_api = client.responses
            api_name = "Responses API"
        elif hasattr(client, 'beta') and hasattr(client.beta, 'responses'):
            responses_api = client.beta.responses
            api_name = "Responses API (Beta)"
        else:
            if is_debug_mode():
                st.warning("⚠️ Responses API not available, falling back to Chat Completions")
            return call_openai_api_chat_completions_legacy(prompt, mentees_df, mentors_df, training_data, api_key, model, mentor_subset)
        
        if is_debug_mode():
            st.info(f"🔄 Using {api_name}...")
        
        if mentor_subset:
            mentors_to_use = mentors_df[mentors_df[MENTOR_COLUMNS['id']].isin(mentor_subset)]
        else:
            mentors_to_use = mentors_df
        
        if use_stateful and not is_first_batch and previous_response_id:
            # Stateful mode, subsequent batch: Only send new mentor data (context preserved)
            if is_debug_mode():
                st.info("🔗 Stateful mode: Sending only new mentor batch (context preserved from previous call)")
            combined_data = prepare_data_for_assistant(None, mentors_to_use, [])
            full_message = f"Evaluate these additional mentors against the same mentees from the previous context:\n\n{combined_data}"
        else:
            # Stateless mode OR first batch: Send complete context
            if is_debug_mode() and use_stateful:
                st.info("🔗 Stateful mode: First batch - sending full context")
            combined_data = prepare_data_for_assistant(mentees_df, mentors_to_use, training_data)
            full_message = f"{prompt}\n\n{combined_data}\n\nGenerate matches for ALL mentees in JSON format as specified."
        
        # API usage estimates using improved formulas
        from utils import estimate_batch_tokens
        
        num_mentors_in_batch = len(mentor_subset) if mentor_subset else len(mentors_df)
        num_mentees = len(mentees_df)
        
        # Use improved estimation functions
        estimated_input_tokens, estimated_output_tokens = estimate_batch_tokens(
            num_mentors=num_mentors_in_batch,
            num_mentees=num_mentees,
            model=model,
            use_stateful=use_stateful,
            is_first_batch=is_first_batch
        )
        
        estimated_cost = estimate_api_cost(estimated_input_tokens, estimated_output_tokens, model)
        total_tokens = estimated_input_tokens + estimated_output_tokens
        
        # Show estimates in debug mode
        if is_debug_mode():
            mode_str = "stateful (store=True)" if use_stateful else "stateless (store=False)"
            if use_stateful and not is_first_batch:
                mode_str += " - subsequent batch"
            st.markdown(f"""
            <div class="cost-estimate">
                <strong>📊 API Usage Estimate:</strong><br>
                • Input tokens: ~{estimated_input_tokens:,}<br>
                • Expected output tokens: ~{estimated_output_tokens:,}<br>
                • Total tokens: ~{total_tokens:,}<br>
                • Estimated cost: ${estimated_cost:.4f}<br>
                • Model: {model}<br>
                • Mode: {mode_str}
            </div>
            """, unsafe_allow_html=True)
        
        if total_tokens > MODEL_TOKEN_LIMITS.get(model, 128000):
            if is_debug_mode():
                st.markdown(f"""
                <div class="token-warning">
                    <strong>⚠️ Token Limit Warning:</strong><br>
                    Estimated tokens ({total_tokens:,}) may exceed model limit ({MODEL_TOKEN_LIMITS.get(model, 128000):,}).<br>
                    Try reducing batch size or splitting the data.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ Token limit exceeded. Try reducing batch size or splitting the data.")
            return None
        
        training_file_ids = st.session_state.get('training_file_ids', [])
        
        messages = [
            {
                "role": "system",
                "content": "You are a mentorship matching expert for IMFAHE's International Mentor Program. You MUST respond with valid JSON only, following the exact format specified."
            },
            {
                "role": "user",
                "content": full_message
            }
        ]
        
        # Suppress these messages - progress UI shows batch status
        if is_debug_mode():
            st.caption("🔄 Sending request to OpenAI Responses API...")
            if training_file_ids:
                st.caption(f"📚 Including {len(training_file_ids)} training file(s)...")
        
        start_time = time.time()
        
        model_config = get_model_config(model)
        
        is_reasoning_model = model_config.get('is_reasoning_model', False) or any(x in model.lower() for x in ['o1-', 'o3-', 'o1', 'o3'])
        supports_json_mode = not is_reasoning_model
        
        if is_debug_mode() and is_reasoning_model:
            st.warning(f"⚠️ {model} is a reasoning model. These models may not support structured JSON output and could return incomplete responses.")
        
        api_params = {
            "model": model,
            "input": messages,
            "timeout": 300,
            "store": use_stateful,
        }
        
        if use_stateful and previous_response_id:
            api_params["previous_response_id"] = previous_response_id
            if is_debug_mode():
                st.info(f"🔗 Chaining to previous response: {previous_response_id[:20]}...")
        
        if model_config['supports_temperature'] and not is_reasoning_model:
            api_params["temperature"] = 0.1
        
        if model_config.get('supports_top_p', False) and not is_reasoning_model:
            api_params["top_p"] = 0.95
        
        # Responses API file_search requires vector_store_ids, not file_ids
        if is_debug_mode() and training_file_ids:
            st.warning("⚠️ File search with Responses API requires vector stores. Training data will be included in prompt instead.")
        
        if supports_json_mode and model_config['supports_structured_output']:
            api_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "mentorship_compatibility_matrix",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "matrix": {
                                "type": "array",
                                "description": "Array of mentor compatibility scores",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "mentor_id": {
                                            "type": "string",
                                            "description": "The mentor's ID"
                                        },
                                        "scores": {
                                            "type": "array",
                                            "description": "Compatibility scores for each mentee",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "mentee_id": {
                                                        "type": "string",
                                                        "description": "The mentee's ID"
                                                    },
                                                    "percentage": {
                                                        "type": "number",
                                                        "description": "Match percentage (0-100)",
                                                        "minimum": 0,
                                                        "maximum": 100
                                                    }
                                                },
                                                "required": ["mentee_id", "percentage"],
                                                "additionalProperties": False
                                            }
                                        }
                                    },
                                    "required": ["mentor_id", "scores"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["matrix"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        
        # Retry logic for rate limits
        max_retries = 5
        retry_count = 0
        response = None
        
        while retry_count < max_retries:
            try:
                if is_debug_mode():
                    st.caption(f"🔄 API attempt {retry_count + 1}/{max_retries}...")
                response = responses_api.create(**api_params)
                if is_debug_mode():
                    st.caption(f"✅ API call succeeded on attempt {retry_count + 1}")
                break
                
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                
                if is_debug_mode():
                    st.caption(f"⚠️ Exception on attempt {retry_count + 1}: {error_type}")
                    with st.expander(f"🔍 Debug: Exception details (attempt {retry_count + 1})"):
                        st.code(f"Type: {error_type}\nMessage: {error_str[:500]}")
                
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        st.error(f"❌ Rate limit exceeded after {max_retries} attempts")
                        if is_debug_mode():
                            show_debug_info()
                        return None
                    
                    import re
                    wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                    
                    if wait_match:
                        wait_time = float(wait_match.group(1))
                        buffer = 3
                        total_wait = wait_time + buffer
                        if is_debug_mode():
                            st.warning(f"⏳ Rate limit (burst). Waiting {total_wait:.1f}s before attempt {retry_count+1}/{max_retries}...")
                        else:
                            st.info(f"⏳ Rate limit reached. Waiting {total_wait:.1f}s...")
                    else:
                        # Wait 65 seconds to let the 1-minute window reset
                        total_wait = 65
                        if is_debug_mode():
                            st.warning(f"⏳ Rate limit (TPM - Tokens Per Minute limit exceeded). Waiting {total_wait}s before attempt {retry_count+1}/{max_retries}...")
                            
                            if "Request too large" in error_str or "Requested" in error_str:
                                tokens_match = re.search(r'Requested (\d+)', error_str)
                                if tokens_match:
                                    requested = int(tokens_match.group(1))
                                    st.warning(f"⚠️ Single request uses {requested:,} tokens, which may exceed your per-minute limit")
                                
                                st.markdown("""
                                <div class="token-warning">
                                    <strong>💡 To avoid this:</strong><br>
                                    • Reduce batch size in Tab 1 (try 10 or 5 mentors per batch)<br>
                                    • Enable stateful mode (reduces subsequent batches to ~5k tokens)<br>
                                    • Use a model with higher rate limits
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info(f"⏳ Rate limit reached. Waiting {total_wait}s...")
                    
                    if is_debug_mode():
                        st.caption(f"💡 This is normal with rate-limited APIs. The system will automatically retry after waiting.")
                    time.sleep(total_wait)
                    continue
                else:
                    st.error(f"❌ API call failed: {error_type}")
                    if is_debug_mode():
                        st.error(f"❌ API call failed with {error_type}: {str(e)[:200]}")
                        show_debug_info()
                    return None
        
        if response is None:
            st.error("❌ Failed to get response after retries")
            return None
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        # Suppress this message - progress UI shows it instead
        if is_debug_mode():
            st.caption(f"✅ Response received in {minutes}m {seconds}s")
        
        # Use universal extractor (handles both reasoning and non-reasoning models)
        try:
            response_content = extract_openai_answer(response)
            if is_debug_mode():
                st.info("📋 Successfully extracted response content")
        except ValueError as e:
            # Check if it's a Responses API response or Chat Completions
            if hasattr(response, 'choices'):
                # Chat Completions format (fallback)
                content = response.choices[0].message.content
                response_content = str(content) if content else ""
                if is_debug_mode():
                    st.info("📋 Using Chat Completions output format")
            else:
                # Could not extract from Responses API
                st.error(f"❌ Could not extract response content")
                if is_debug_mode():
                    st.error(f"❌ Could not extract response content: {str(e)}")
                    st.write("Debug - Response structure:", {
                        "has_output": hasattr(response, 'output'),
                        "output_type": type(getattr(response, 'output', None)),
                        "output_length": len(getattr(response, 'output', [])) if hasattr(response, 'output') and isinstance(response.output, list) else None,
                        "has_output_text": hasattr(response, 'output_text'),
                        "output_text": getattr(response, 'output_text', None),
                    })
                    
                    # Show what's in the output array
                    if hasattr(response, 'output') and isinstance(response.output, list):
                        st.write("Output items:")
                        for idx, item in enumerate(response.output):
                            item_type = getattr(item, 'type', 'unknown')
                            st.write(f"  [{idx}] type={item_type}")
                            if item_type == 'reasoning':
                                st.write(f"      (reasoning object - metadata, not the answer)")
                            elif item_type == 'message':
                                st.write(f"      (message object - this should contain the answer)")
                    
                    st.warning("""
                    **What to check:**
                    - For reasoning models: Look for `type == "message"` items in `response.output[]`
                    - For non-reasoning models: Check `response.output_text`
                    - Reasoning objects (`type == "reasoning"`) are metadata, not the answer
                    
                    **If only reasoning objects found:**
                    The model may not support structured JSON output. Try:
                    - ✅ **gpt-4.1-mini** (recommended)
                    - ✅ **gpt-4.1-nano** (fastest)
                    - ✅ **gpt-5.1** or **gpt-4.1** (highest quality)
                    """)
                return None
        except Exception as e:
            st.error(f"❌ Error extracting response")
            if is_debug_mode():
                st.error(f"❌ Unexpected error extracting response: {str(e)}")
                st.write("Debug - Full response:", response)
            return None
        
        if hasattr(response, 'usage'):
            usage = response.usage
            
            actual_prompt_tokens = (
                getattr(usage, 'input_tokens', None) or 
                getattr(usage, 'prompt_tokens', 0)
            )
            actual_completion_tokens = (
                getattr(usage, 'output_tokens', None) or 
                getattr(usage, 'completion_tokens', 0)
            )
            
            actual_cost = estimate_api_cost(actual_prompt_tokens, actual_completion_tokens, model)
            
            if is_debug_mode():
                st.markdown(f"""
                <div class="success-card">
                    <strong>✅ Actual API Usage:</strong><br>
                    • Input tokens: {actual_prompt_tokens:,}<br>
                    • Output tokens: {actual_completion_tokens:,}<br>
                    • Total tokens: {actual_prompt_tokens + actual_completion_tokens:,}<br>
                    • Actual cost: ${actual_cost:.4f}
                </div>
                """, unsafe_allow_html=True)
        
        if 'last_api_request' not in st.session_state:
            st.session_state.last_api_request = {}
        if 'last_api_response' not in st.session_state:
            st.session_state.last_api_response = {}
        
        st.session_state.last_api_request = {
            'prompt': prompt,
            'full_message': full_message,
            'model': model,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        response_id = getattr(response, 'id', None)
        
        st.session_state.last_api_response = {
            'content': response_content,
            'tokens': {
                'input': actual_prompt_tokens if hasattr(response, 'usage') else 0,
                'output': actual_completion_tokens if hasattr(response, 'usage') else 0,
                'total': (actual_prompt_tokens + actual_completion_tokens) if hasattr(response, 'usage') else 0
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'api_type': api_name,
            'response_id': response_id
        }
        
        if use_stateful:
            return response_content, response_id
        else:
            return response_content
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        show_debug_info()
        return None
def call_openai_api_chat_completions_legacy(prompt, mentees_df, mentors_df, training_data, api_key, model="gpt-4o-mini", mentor_subset=None):
    """
    Legacy Chat Completions API implementation (fallback when Responses API not available)
    
    This is the original implementation kept for backward compatibility.
    """
    try:
        client = OpenAI(api_key=api_key)
        
        if mentor_subset:
            mentors_to_use = mentors_df[mentors_df[MENTOR_COLUMNS['id']].isin(mentor_subset)]
        else:
            mentors_to_use = mentors_df
        
        combined_data = prepare_data_for_assistant(mentees_df, mentors_to_use, training_data)
        
        full_message = f"{prompt}\n\n{combined_data}\n\nGenerate matches for ALL mentees in JSON format as specified."
        
        training_file_ids = st.session_state.get('training_file_ids', [])
        
        messages = [
            {
                "role": "system",
                "content": "You are a mentorship matching expert for IMFAHE's International Mentor Program. You MUST respond with valid JSON only, following the exact format specified."
            },
            {
                "role": "user",
                "content": full_message
            }
        ]
        
        if training_file_ids:
            st.info(f"📚 Including {len(training_file_ids)} training file(s) for context...")
            try:
                messages[1]["attachments"] = [
                    {
                        "file_id": file_id,
                        "tools": [{"type": "file_search"}]
                    }
                    for file_id in training_file_ids
                ]
            except:
                st.warning("⚠️ Could not attach files with Chat Completions API")
        
        st.info("🔄 Sending request to OpenAI Chat Completions API (Legacy)...")
        
        model_config = get_model_config(model)
        
        is_reasoning_model = model_config.get('is_reasoning_model', False) or any(x in model.lower() for x in ['o1-', 'o3-', 'o1', 'o3'])
        supports_json_mode = not is_reasoning_model
        
        api_params = {
            "model": model,
            "messages": messages,
            "timeout": 300,
        }
        
        if not is_reasoning_model:
            api_params["temperature"] = 0.1
        
        if supports_json_mode:
            api_params["response_format"] = {"type": "json_object"}
        
        start_time = time.time()
        response = client.chat.completions.create(**api_params)
        elapsed = time.time() - start_time
        
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        # Suppress this message - progress UI shows it instead
        if is_debug_mode():
            st.caption(f"✅ Response received in {minutes}m {seconds}s")
        
        response_content = response.choices[0].message.content
        return response_content
        
    except Exception as e:
        st.error("❌ API Error")
        if is_debug_mode():
            st.error(f"❌ Legacy API Error: {str(e)}")
        return None


# Keep old function name as alias for backward compatibility
def call_openai_api_with_assistants(prompt, mentees_df, mentors_df, training_data, api_key, model="gpt-4o-mini", mentor_subset=None, use_stateful=False, previous_response_id=None, is_first_batch=True):
    """
    DEPRECATED: Assistants API will be deprecated in 2025.
    This function now redirects to the Responses API.
    
    See: https://platform.openai.com/docs/guides/migrate-to-responses
    """
    return call_openai_api_with_chat_completions(prompt, mentees_df, mentors_df, training_data, api_key, model, mentor_subset, use_stateful, previous_response_id, is_first_batch)

def generate_matrix_in_batches(mentees_df, mentors_df, training_data, api_key, model, batch_size=None, use_stateful=False):
    """Generate compatibility matrix in batches to avoid token limits
    
    Args:
        batch_size: Number of mentors to process per batch. If None, auto-calculated based on TPM limits.
        use_stateful: Enable stateful mode (store=True) for efficient batch processing
                     When True, only the first batch sends full context, subsequent batches
                     only send new mentor data and reference the previous response.
    
    Returns:
        Complete matrix DataFrame
    """
    from utils import calculate_optimal_batch_size, estimate_batch_tokens, get_model_config
    from progress_ui import initialize_progress_tracking, render_progress_overview, update_batch_progress, render_batch_card
    
    all_mentor_ids = mentors_df[MENTOR_COLUMNS['id']].tolist()
    all_mentee_ids = mentees_df[MENTEE_COLUMNS['id']].tolist()
    total_mentors = len(all_mentor_ids)
    total_mentees = len(all_mentee_ids)
    
    # Auto-calculate batch size if not provided
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(
            total_mentors=total_mentors,
            num_mentees=total_mentees,
            model=model,
            use_stateful=use_stateful
        )
        
        # Show calculation details only in debug mode
        if is_debug_mode():
            model_config = get_model_config(model)
            tpm_limit = model_config.get('tpm_limit', 30000)
            input_tokens, output_tokens = estimate_batch_tokens(
                batch_size, total_mentees, model, use_stateful, is_first_batch=True
            )
            total_tokens = input_tokens + output_tokens
            
            st.info(f"📊 **Auto-calculated batch size**: {batch_size} mentors per batch")
            with st.expander("ℹ️ Batch size calculation details"):
                st.markdown(f"""
                **Model**: {model}  
                **TPM Limit**: {tpm_limit:,} tokens/minute  
                **Safety Margin**: 90% of limit = {int(tpm_limit * 0.90):,} usable tokens  
                **Estimated tokens per batch**: {total_tokens:,} (input: {input_tokens:,} + output: {output_tokens:,})  
                **Mode**: {'Stateful' if use_stateful else 'Stateless'}  
                **Total batches**: {int((total_mentors + batch_size - 1) / batch_size)}
                """)
    
    mentor_batches = [all_mentor_ids[i:i + batch_size] for i in range(0, len(all_mentor_ids), batch_size)]
    total_batches = len(mentor_batches)
    
    # Initialize progress tracking
    initialize_progress_tracking(total_batches)
    
    # Initialize all batches as pending
    for batch_num in range(1, total_batches + 1):
        update_batch_progress(batch_num, 'pending')
    
    # Show mode info (only in debug mode)
    if is_debug_mode():
        mode_info = "🔗 STATEFUL MODE" if use_stateful else "📦 STATELESS MODE"
        st.info(f"{mode_info}: Processing in {total_batches} batches ({batch_size} mentors per batch)")
        
        if use_stateful:
            st.markdown("""
            <div class="info-card">
                <strong>🔗 Stateful Batching Enabled:</strong><br>
                • First batch: Sends full mentee data + initial mentors (~high tokens)<br>
                • Subsequent batches: Only new mentors (~low tokens, 90% savings)<br>
                • Context preserved via previous_response_id chaining
            </div>
            """, unsafe_allow_html=True)
    
    # Create empty container for progress UI (to update in place)
    progress_placeholder = st.empty()
    
    # Create a container for completed batch cards (they accumulate)
    completed_batches_container = st.container()
    
    # Create empty placeholder for currently processing batch
    current_batch_placeholder = st.empty()
    
    # Render initial progress overview
    with progress_placeholder.container():
        render_progress_overview()
    
    combined_matrix_dict = {}
    previous_response_id = None
    
    for batch_num, mentor_batch in enumerate(mentor_batches, 1):
        is_first_batch = (batch_num == 1)
        batch_start_time = time.time()
        
        # Update progress: mark as processing
        update_batch_progress(batch_num, 'processing')
        
        # Update progress overview in placeholder (replaces previous content)
        with progress_placeholder.container():
            render_progress_overview()
        
        # Show current batch status in placeholder (replaces previous)
        with current_batch_placeholder.container():
            render_batch_card(batch_num, 'processing', is_current=True)
        
        # Batch status is shown in progress UI, suppress these messages
        # Only show detailed info in debug mode
        if is_debug_mode():
            batch_label = "FIRST BATCH (full context)" if is_first_batch else f"Batch {batch_num} (mentors only)"
            st.caption(f"🔄 {batch_label}: {len(mentor_batch)} mentors ({mentor_batch[0]} to {mentor_batch[-1]})")
        
        model_config = get_model_config(model)
        include_json_instructions = not model_config.get('supports_structured_output', True)
        
        training_file_ids = st.session_state.get('training_file_ids', [])
        result = call_openai_api_with_assistants(
            get_prompt_for_api(include_json_instructions=include_json_instructions),
            mentees_df,
            mentors_df,
            training_data,
            api_key,
            model,
            mentor_subset=mentor_batch,
            use_stateful=use_stateful,
            previous_response_id=previous_response_id,
            is_first_batch=is_first_batch
        )
        
        batch_time = time.time() - batch_start_time
        
        if use_stateful and isinstance(result, tuple):
            response, response_id = result
            previous_response_id = response_id
        else:
            response = result
            response_id = None
        
        if not response:
            update_batch_progress(batch_num, 'error', {'error': 'Failed to get response'})
            if is_debug_mode():
                st.error(f"❌ Batch {batch_num} failed to get response")
                show_debug_info()
            continue
        
        batch_matrix_df = parse_matrix_response(response, mentor_batch, all_mentee_ids)
        
        if batch_matrix_df is None:
            update_batch_progress(batch_num, 'error', {'error': 'Failed to parse response'})
            st.error(f"❌ Batch {batch_num} failed to parse")
            continue
        
        received_mentors = list(batch_matrix_df.index)
        expected_mentors_set = set(str(m) for m in mentor_batch)
        received_mentors_set = set(received_mentors)
        
        dict_size_before = len(combined_matrix_dict)
        overwrites = []
        
        for mentor_id in batch_matrix_df.index:
            if mentor_id in combined_matrix_dict:
                overwrites.append(mentor_id)
                if is_debug_mode():
                    st.caption(f"⚠️ DUPLICATE: '{mentor_id}' already exists!")
            
            combined_matrix_dict[mentor_id] = batch_matrix_df.loc[mentor_id].to_dict()
        
        dict_size_after = len(combined_matrix_dict)
        actual_new_entries = dict_size_after - dict_size_before
        
        # Get token usage and cost if available (from last API response)
        tokens = None
        cost = None
        if hasattr(st.session_state, 'last_api_response'):
            api_response = st.session_state.last_api_response
            if 'tokens' in api_response:
                tokens = api_response['tokens'].get('total', 0)
                # Estimate cost if we have token info
                if tokens > 0:
                    from utils import estimate_api_cost
                    input_tokens = api_response['tokens'].get('input', 0)
                    output_tokens = api_response['tokens'].get('output', 0)
                    cost = estimate_api_cost(input_tokens, output_tokens, model)
        
        # Prepare batch details
        batch_details = {
            'mentors': len(batch_matrix_df),
            'mentees': len(all_mentee_ids),
            'scores': len(batch_matrix_df) * len(all_mentee_ids),
            'time': batch_time,
            'tokens': tokens,
            'cost': cost,
            'response_id': response_id[:30] + '...' if response_id else None
        }
        
        # Update progress: mark as complete
        update_batch_progress(batch_num, 'complete', batch_details)
        
        # Update progress overview in placeholder (replaces previous content)
        with progress_placeholder.container():
            render_progress_overview()
        
        # Clear current batch placeholder and move completed batch to container
        current_batch_placeholder.empty()
        
        # Render completed batch card in the completed batches container
        with completed_batches_container:
            render_batch_card(batch_num, 'complete', batch_details)
        
        if is_debug_mode():
            if overwrites:
                st.caption(f"⚠️ {len(overwrites)} duplicate(s) detected")
            if received_mentors_set == expected_mentors_set:
                st.caption(f"✅ Batch {batch_num}: Received all {len(batch_matrix_df)}/{len(mentor_batch)} mentors")
            else:
                missing_in_batch = expected_mentors_set - received_mentors_set
                st.caption(f"⚠️ Batch {batch_num}: Missing {len(missing_in_batch)} mentor(s)")
        
        if batch_num < len(mentor_batches):
            delay = 5 if use_stateful else 2
            if is_debug_mode():
                st.caption(f"⏳ Waiting {delay}s before next batch...")
            time.sleep(delay)
    
    if not combined_matrix_dict:
        st.error("❌ No batches succeeded")
        return None

    st.divider()
    if is_debug_mode():
        st.info("📊 **Combining all batches into final matrix...**")
    
    complete_matrix_df = pd.DataFrame(combined_matrix_dict).T
    complete_matrix_df = complete_matrix_df[[str(m) for m in sorted(all_mentee_ids)]]
    
    # Apply country restrictions (Spain-Spain rule)
    st.divider()
    st.subheader("🌍 Applying Country Restrictions")
    st.info("🔍 Checking for country-based restrictions (Spain-Spain rule)...")
    complete_matrix_df, filtered_matches = apply_country_restrictions(
        complete_matrix_df, mentees_df, mentors_df
    )
    
    # Display filtered matches information
    if filtered_matches:
        st.warning(f"🚫 **Country Restriction Applied**: {len(filtered_matches)} match(es) filtered")
        with st.expander(f"📋 Filtered Matches Details ({len(filtered_matches)})"):
            st.markdown("**Rule**: Spanish mentors cannot mentor Spanish mentees")
            st.markdown("**Action**: Scores set to 0 (invalid match)")
            st.markdown("---")
            
            # Sort by original score (highest first) to show most impactful filters
            sorted_matches = sorted(filtered_matches, key=lambda x: x['original_score'], reverse=True)
            
            # Display as table for better readability
            matches_data = []
            for match in sorted_matches:
                matches_data.append({
                    'Mentor ID': match['mentor_id'],
                    'Mentee ID': match['mentee_id'],
                    'Original Score': f"{match['original_score']:.1f}%",
                    'Reason': 'Spain-Spain restriction'
                })
            
            if matches_data:
                matches_df = pd.DataFrame(matches_data)
                st.dataframe(matches_df, width='stretch', hide_index=True)
                
                # Summary statistics
                avg_score = sum(m['original_score'] for m in sorted_matches) / len(sorted_matches)
                max_score = sorted_matches[0]['original_score']
                st.caption(f"📊 Average filtered score: {avg_score:.1f}% | Highest filtered: {max_score:.1f}%")
    else:
        st.success("✅ No matches filtered by country restrictions")
    
    mentors_received = len(complete_matrix_df)
    expected_count = len(all_mentor_ids)
    mentors_in_dict = set(complete_matrix_df.index)
    mentors_requested = set(str(m) for m in all_mentor_ids)
    
    missing_from_dict = mentors_requested - mentors_in_dict
    extra_in_dict = mentors_in_dict - mentors_requested
    
    st.success(f"✅ Final matrix: {mentors_received}/{expected_count} mentors")
    
    if is_debug_mode():
        with st.expander("🔍 Final Matrix Validation Details"):
            st.write(f"**Expected mentors**: {expected_count} (from original data)")
            st.write(f"**Mentors in dictionary**: {mentors_received} (unique keys)")
            st.write(f"**Dictionary size**: {len(combined_matrix_dict)} entries")
            
            if missing_from_dict:
                st.error(f"❌ **Missing {len(missing_from_dict)} mentors** that were requested but not in dict:")
                for m in sorted(missing_from_dict)[:20]:
                    st.text(f"  - {m}")
            else:
                st.success("✅ All requested mentors are in the dictionary!")
            
            if extra_in_dict:
                st.warning(f"⚠️ **Extra {len(extra_in_dict)} mentors** in dict that weren't requested:")
                for m in sorted(extra_in_dict)[:20]:
                    st.text(f"  - {m}")
    
    if mentors_received < expected_count:
        st.warning(f"⚠️ **DISCREPANCY**: Expected {expected_count} but only have {mentors_received} in final matrix")
    
    if mentors_received < expected_count:
        missing = set(str(m) for m in all_mentor_ids) - set(complete_matrix_df.index)
        st.warning(f"⚠️ Missing {len(missing)} mentors after initial batches")
        
        with st.expander(f"📋 Missing mentors ({len(missing)})"):
            for m in sorted(missing):
                st.write(f"- {m}")
        
        max_retries = 3
        for retry_num in range(1, max_retries + 1):
            # Recalculate what's still missing (convert to strings for comparison)
            current_missing = set(str(m) for m in all_mentor_ids) - set(complete_matrix_df.index)
            
            if not current_missing:
                st.success(f"🎉 All mentors retrieved!")
                break
            
            st.info(f"🔄 Retry {retry_num}/{max_retries}: Attempting {len(current_missing)} missing mentors...")
            
            try:
                model_config = get_model_config(model)
                include_json_instructions = not model_config.get('supports_structured_output', True)
                
                training_file_ids = st.session_state.get('training_file_ids', [])
                retry_result = call_openai_api_with_assistants(
                    get_prompt_for_api(include_json_instructions=include_json_instructions),
                    mentees_df,
                    mentors_df,
                    training_data,
                    api_key,
                    model,
                    mentor_subset=list(current_missing),
                    use_stateful=use_stateful,
                    previous_response_id=previous_response_id if use_stateful else None,
                    is_first_batch=False  # Retries are always subsequent
                )
                
                if use_stateful and isinstance(retry_result, tuple):
                    retry_response, retry_response_id = retry_result
                    if retry_response_id:
                        previous_response_id = retry_response_id
                else:
                    retry_response = retry_result
                
                if not retry_response:
                    st.error(f"❌ Retry {retry_num} failed to get response")
                    if retry_num < max_retries:
                        st.info("⏳ Waiting 5 seconds before next retry...")
                        time.sleep(5)
                    continue
                
                retry_matrix_df = parse_matrix_response(retry_response, list(current_missing), all_mentee_ids)
                
                if retry_matrix_df is not None and len(retry_matrix_df) > 0:
                    added_count = 0
                    for mentor_id in retry_matrix_df.index:
                        combined_matrix_dict[mentor_id] = retry_matrix_df.loc[mentor_id].to_dict()
                        added_count += 1
                    
                    st.success(f"✅ Retry {retry_num} successful: Added {added_count} mentors")
                    
                    # Rebuild complete matrix from updated dictionary
                    complete_matrix_df = pd.DataFrame(combined_matrix_dict).T
                    complete_matrix_df = complete_matrix_df[sorted(all_mentee_ids)]
                    
                    mentors_received = len(complete_matrix_df)
                    st.info(f"📈 Matrix now has {mentors_received}/{expected_count} mentors")
                    
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
        
        final_missing = set(str(m) for m in all_mentor_ids) - set(complete_matrix_df.index)
        if final_missing:
            st.warning(f"⚠️ After {max_retries} retries, still missing {len(final_missing)} mentors")
            with st.expander(f"📋 Permanently missing mentors ({len(final_missing)})"):
                for m in sorted(final_missing):
                    st.write(f"- {m}")
            st.info(f"💡 Continuing with {len(complete_matrix_df)}/{expected_count} mentors. Missing mentors will be excluded from matching.")
    
    return complete_matrix_df


def get_reasoning_for_assignments(assignments, mentees_df, mentors_df, api_key, model="gpt-4o-mini"):
    """Get reasoning for specific assignments via separate API call"""
    try:
        client = OpenAI(api_key=api_key)
        
        if not assignments or len(assignments) == 0:
            st.warning("⚠️ No assignments to get reasoning for")
            return {}
        
        if mentees_df is None or mentors_df is None:
            st.error("❌ Missing mentee or mentor data")
            return {}
        
        assignment_details = []
        for mentee_id, mentor_id, score in assignments:
            try:
                mentee_id_int = int(mentee_id)
                mentor_id_int = int(mentor_id)
                mentee_row = mentees_df[mentees_df[MENTEE_COLUMNS['id']] == mentee_id_int].iloc[0]
                mentor_row = mentors_df[mentors_df[MENTOR_COLUMNS['id']] == mentor_id_int].iloc[0]
                
                def safe_str(value):
                    if pd.isna(value):
                        return ''
                    return str(value).strip()
                
                assignment_details.append({
                    'mentee_id': safe_str(mentee_id),
                    'mentor_id': safe_str(mentor_id),
                    'score': int(score) if not pd.isna(score) else 0,
                    'mentee_field': safe_str(mentee_row.get(MENTEE_COLUMNS['field'], '')),
                    'mentee_specialization': safe_str(mentee_row.get(MENTEE_COLUMNS['specialization'], '')),
                    'mentee_guidance_areas': safe_str(mentee_row.get(MENTEE_COLUMNS['guidance_areas'], '')),
                    'mentee_goals': safe_str(mentee_row.get(MENTEE_COLUMNS['career_goals'], '')),
                    'mentee_other_info': safe_str(mentee_row.get(MENTEE_COLUMNS['other_info'], '')),
                    'mentor_field': safe_str(mentor_row.get(MENTOR_COLUMNS['field'], '')),
                    'mentor_specialization': safe_str(mentor_row.get(MENTOR_COLUMNS['specialization'], '')),
                    'mentor_position': safe_str(mentor_row.get(MENTOR_COLUMNS['current_position'], '')),
                    'mentor_experience': safe_str(mentor_row.get(MENTOR_COLUMNS['experience_years'], ''))
                })
            except Exception as e:
                if is_debug_mode():
                    st.warning(f"⚠️ Could not extract details for {mentee_id} → {mentor_id}: {str(e)}")
                assignment_details.append({
                    'mentee_id': str(mentee_id),
                    'mentor_id': str(mentor_id),
                    'score': int(score) if isinstance(score, (int, float)) else 0,
                    'mentee_field': '',
                    'mentee_specialization': '',
                    'mentee_guidance_areas': '',
                    'mentee_goals': '',
                    'mentee_other_info': '',
                    'mentor_field': '',
                    'mentor_specialization': '',
                    'mentor_position': '',
                    'mentor_experience': ''
                })
        
        if not assignment_details:
            st.error("❌ No valid assignment details extracted")
            return {}
        
        if is_debug_mode():
            st.info(f"📝 Extracted details for {len(assignment_details)} assignments")
        
        prompt = f"""Provide brief reasoning for these {len(assignment_details)} mentorship assignments.

For each assignment, explain in 2-3 sentences why this is a good match based on:
- Field/specialization alignment
- Skills compatibility
- Career goals support

ASSIGNMENTS:
"""
        for ad in assignment_details:
            mentee_spec = str(ad.get('mentee_specialization', ''))[:100]
            mentee_guidance = str(ad.get('mentee_guidance_areas', ''))[:100]
            mentee_goals = str(ad.get('mentee_goals', ''))[:100]
            mentee_other = str(ad.get('mentee_other_info', ''))[:100]
            mentor_spec = str(ad.get('mentor_specialization', ''))[:100]
            mentor_pos = str(ad.get('mentor_position', ''))[:80]
            
            prompt += f"""
{ad['mentee_id']} → {ad['mentor_id']} (Score: {ad['score']}%)
- Mentee field: {ad['mentee_field']}
- Mentee specialization: {mentee_spec}...
- Mentee guidance needs: {mentee_guidance}...
- Mentee goals: {mentee_goals}...
- Mentee other info: {mentee_other}...
- Mentor field: {ad['mentor_field']}
- Mentor position: {mentor_pos}...
- Mentor specialization: {mentor_spec}...
- Mentor experience: {ad['mentor_experience']} years

"""
        
        prompt += """
OUTPUT FORMAT (IMPORTANT):
Return ONLY a JSON object with this exact structure:
{
  "reasonings": [
    {
      "mentee_id": "EXACT_CODE",
      "mentor_id": "EXACT_CODE",
      "reasoning": "2-3 sentence explanation"
    }
  ]
}

No markdown, no extra text, just the JSON object."""
        
        progress_text = st.empty()
        progress_text.text("🧠 Requesting reasoning from AI...")
        
        model_config = get_model_config(model)
        
        # Check if Responses API is available
        if hasattr(client, 'responses'):
            responses_api = client.responses
        elif hasattr(client, 'beta') and hasattr(client.beta, 'responses'):
            responses_api = client.beta.responses
        else:
            # Fallback to Chat Completions API
            responses_api = None
        
        model_config = get_model_config(model)
        
        # Handle o1/o3/gpt-5-mini/gpt-5-nano models differently
        is_reasoning_model = model_config.get('is_reasoning_model', False) or model.startswith("o1-") or model.startswith("o3-")
        
        if responses_api:
            # Define schema for reasoning - must be object at root level
            reasoning_schema = {
                "format": {
                    "type": "json_schema",
                    "name": "mentorship_reasoning",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasonings": {
                                "type": "array",
                                "description": "Array of reasoning for each assignment",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "mentee_id": {
                                            "type": "string",
                                            "description": "The mentee's ID"
                                        },
                                        "mentor_id": {
                                            "type": "string",
                                            "description": "The mentor's ID"
                                        },
                                        "reasoning": {
                                            "type": "string",
                                            "description": "Explanation of why this is a good match"
                                        }
                                    },
                                    "required": ["mentee_id", "mentor_id", "reasoning"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["reasonings"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
            
            # Retry logic for rate limits
            max_retries = 5
            retry_count = 0
            response = None
            
            while retry_count < max_retries:
                try:
                    if is_reasoning_model:
                        full_prompt = "You are an expert at explaining mentorship compatibility. Return only valid JSON.\n\n" + prompt
                        response = responses_api.create(
                            model=model,
                            input=full_prompt,
                            max_output_tokens=8000,
                            store=False,
                            text=reasoning_schema
                        )
                    else:
                        reasoning_params = {
                            "model": model,
                            "input": [
                                {"role": "system", "content": "You are an expert at explaining mentorship compatibility. Return only valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            "max_output_tokens": 8000,
                            "store": False,
                            "text": reasoning_schema
                        }
                        
                        # Add temperature if model supports it
                        if model_config['supports_temperature']:
                            reasoning_params["temperature"] = 0.3
                        
                        # Add top_p if model supports it
                        if model_config.get('supports_top_p', False):
                            reasoning_params["top_p"] = 0.95
                        
                        response = responses_api.create(**reasoning_params)
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    
                    if "rate_limit_exceeded" in error_str or "429" in error_str:
                        retry_count += 1
                        
                        if retry_count >= max_retries:
                            st.error("❌ Rate limit exceeded on reasoning call")
                            if is_debug_mode():
                                st.error(f"❌ Rate limit exceeded on reasoning call after {max_retries} attempts")
                            return {}
                        
                        import re
                        wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                        
                        if wait_match:
                            wait_time = float(wait_match.group(1))
                            buffer = 3
                            total_wait = wait_time + buffer
                        else:
                            total_wait = 65
                        
                        if is_debug_mode():
                            st.warning(f"⏳ Rate limit on reasoning call. Waiting {total_wait:.1f}s before attempt {retry_count+1}/{max_retries}...")
                        else:
                            st.info(f"⏳ Rate limit reached. Waiting {total_wait:.1f}s...")
                        time.sleep(total_wait)
                        continue
                    else:
                        st.error("❌ Reasoning API call failed")
                        if is_debug_mode():
                            st.error(f"❌ Reasoning API call failed: {str(e)}")
                        return {}
            
            if response is None:
                st.error("❌ Failed to get reasoning response")
                if is_debug_mode():
                    st.error("❌ Failed to get reasoning response after retries")
                return {}
        else:
            # Fallback to Chat Completions API
            token_limit_param = "max_completion_tokens" if is_reasoning_model else "max_tokens"
            
            if is_reasoning_model:
                full_prompt = "You are an expert at explaining mentorship compatibility. Return only valid JSON.\n\n" + prompt
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    **{token_limit_param: 8000}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert at explaining mentorship compatibility. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    **{token_limit_param: 8000}
                )
        
        progress_text.text("🧠 Parsing reasoning response...")
        
        # Use universal extractor (handles both reasoning and non-reasoning models)
        try:
            response_text = extract_openai_answer(response).strip()
        except ValueError as e:
            # Fallback to Chat Completions format
            if hasattr(response, 'choices'):
                content = response.choices[0].message.content
                response_text = str(content).strip() if content else ""
            else:
                st.error("❌ Could not extract reasoning response")
                if is_debug_mode():
                    st.error(f"❌ Could not extract reasoning response: {str(e)}")
                return {}
        except Exception as e:
            st.error("❌ Error extracting reasoning response")
            if is_debug_mode():
                st.error(f"❌ Unexpected error extracting reasoning response: {str(e)}")
            return {}
        
        if is_debug_mode():
            st.info(f"📝 Received reasoning response ({len(response_text)} characters)")
            with st.expander("🔍 Debug: View response preview"):
                st.text(response_text[:500])
        
        original_text = response_text
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            reasoning_response = json.loads(response_text)
        except json.JSONDecodeError as parse_error:
            st.error("❌ JSON parse failed")
            if is_debug_mode():
                st.error(f"❌ JSON parse failed: {parse_error}")
                with st.expander("📄 Full response text"):
                    st.code(original_text)
            raise
        
        if isinstance(reasoning_response, dict) and 'reasonings' in reasoning_response:
            reasoning_data = reasoning_response['reasonings']
        elif isinstance(reasoning_response, list):
            reasoning_data = reasoning_response
        else:
            st.error("❌ Invalid response format")
            if is_debug_mode():
                st.error(f"❌ Response is {type(reasoning_response).__name__}, expected object with 'reasonings' array")
                with st.expander("📄 Parsed data"):
                    st.json(reasoning_response)
            raise ValueError("Response should be an object with 'reasonings' array")
        
        if not isinstance(reasoning_data, list):
            st.error("❌ Invalid reasoning format")
            if is_debug_mode():
                st.error(f"❌ 'reasonings' is {type(reasoning_data).__name__}, expected list")
            raise ValueError("'reasonings' should be a list")
        
        if is_debug_mode():
            st.info(f"✅ Parsed {len(reasoning_data)} reasoning entries from JSON")
        
        reasoning_dict = {}
        for idx, r in enumerate(reasoning_data):
            if 'mentee_id' in r and 'mentor_id' in r and 'reasoning' in r:
                reasoning_dict[(r['mentee_id'], r['mentor_id'])] = r['reasoning']
            else:
                if is_debug_mode():
                    st.warning(f"⚠️ Entry {idx} missing required fields: {r.keys()}")
        
        progress_text.empty()
        st.success(f"✅ Got reasoning for {len(reasoning_dict)} assignments")
        
        if is_debug_mode() and len(reasoning_dict) > 0:
            sample_keys = list(reasoning_dict.keys())[:3]
            st.info(f"📋 Sample reasoning keys: {sample_keys}")
        
        return reasoning_dict
        
    except json.JSONDecodeError as e:
        st.error("❌ JSON parsing error in reasoning")
        if is_debug_mode():
            st.error(f"❌ JSON parsing error in reasoning: {str(e)}")
            with st.expander("View raw reasoning response"):
                try:
                    st.text(response_text[:2000])
                except:
                    st.text("Could not display response")
        st.info("💡 Continuing with generic reasoning...")
        return {}
    except Exception as e:
        st.error("❌ Error getting reasoning")
        if is_debug_mode():
            st.error(f"❌ Error getting reasoning: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
        st.info("💡 Continuing without detailed reasoning...")
        return {}


def main():
    # Password authentication check - must be first thing
    import os
    
    # Initialize authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'password_attempts' not in st.session_state:
        st.session_state.password_attempts = 0
    if 'password_locked' not in st.session_state:
        st.session_state.password_locked = False
    
    # Get password from environment variable
    required_password = os.getenv('APP_ACCESS_PASSWORD', '')
    
    # If no password is set in env, skip authentication (for development)
    if not required_password:
        st.session_state.authenticated = True
    # If already authenticated, continue
    elif st.session_state.authenticated:
        pass  # Continue to app
    # If locked due to too many attempts
    elif st.session_state.password_locked:
        st.error("🔒 **Access Locked**: Too many failed password attempts. Please refresh the page to try again.")
        st.stop()
    # Show password form
    else:
        # Centered container with all elements together
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h2>🔐 Access Required</h2>
                <p>Please enter the access password to continue.</p>
            </div>
            """, unsafe_allow_html=True)
            
            password_input = st.text_input(
                "Access Password",
                type="password",
                key="password_input",
                label_visibility="visible"
            )
            
            if st.button("🔓 Unlock", type="primary", use_container_width=True):
                if password_input == required_password:
                    st.session_state.authenticated = True
                    st.session_state.password_attempts = 0
                    st.rerun()
                else:
                    st.session_state.password_attempts += 1
                    remaining_attempts = 5 - st.session_state.password_attempts
                    
                    if st.session_state.password_attempts >= 5:
                        st.session_state.password_locked = True
                        st.error("🔒 **Access Locked**: Too many failed attempts. Please refresh the page.")
                        st.stop()
                    else:
                        st.error(f"❌ Incorrect password. {remaining_attempts} attempt(s) remaining.")
            
            if st.session_state.password_attempts > 0:
                st.warning(f"⚠️ Failed attempts: {st.session_state.password_attempts}/5")
        
        st.stop()  # Prevent app from loading until authenticated
    
    # Header with animation
    st.markdown("""
    <div class="main-header">
        <h1>🤝 Mentorship Matching System</h1>
        <p>AI-powered mentor-mentee matching</p>
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
        st.session_state.model_choice = "gpt-5.1"  # Default to latest model
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    
    # Initialize debug mode from environment variable
    if 'debug_mode' not in st.session_state:
        import os
        debug_env = os.getenv('DEBUG_MODE', '').upper().strip()
        if debug_env == 'FALSE':
            st.session_state.debug_mode = False
        elif debug_env == 'TRUE':
            st.session_state.debug_mode = True
        else:
            # Default: ON if env var not present or invalid
            st.session_state.debug_mode = True
    
    # No sidebar - cleaner interface
    
    # Step 1 (API key) + Step 3 (data files) are required
    # Step 2 (training files) is optional
    step1_complete = (
        st.session_state.mentees_df is not None and 
        st.session_state.mentors_df is not None
    )
    
    import os
    
    # Try to get from session state or environment
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')
    
    step2_complete = bool(st.session_state.api_key) and step1_complete
    step3_complete = step2_complete  # Can view data once files uploaded and settings configured
    
    tab_names = [
        "1️⃣ Configure Settings",
        "2️⃣ Training Files",
        "3️⃣ Upload Data Files",
        "4️⃣ Customize Prompt",
        "5️⃣ Data Overview",
        "6️⃣ Generate Matches",
        "7️⃣ Processing & Results"
    ]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)
    
    # TAB 1: Configure Settings
    with tab1:
        render_config_tab()
    
    # TAB 2: Training Files
    with tab2:
        render_training_tab()
    
    # TAB 3: Upload Data Files
    with tab3:
        render_upload_tab()
    
    # TAB 4: Customize Prompt
    with tab4:
        render_prompt_tab()
    
    # TAB 5: Data Overview
    with tab5:
        render_overview_tab()
    
    # TAB 6: Generate Matches
    with tab6:
        render_matches_tab()
    
    # TAB 7: Processing & Results
    with tab7:
        render_processing_tab()

if __name__ == "__main__":
    main()
