# Mentorship Matching Revamp — Phase 2: Claude Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every LLM call (matrix scoring, per-match reasoning, and the normalization pipeline) off OpenAI onto Anthropic's Claude, with streaming + a real output-token budget + a `stop_reason` guard that makes a cut-off response an honest error instead of the silent fill-30 corruption. One provider, one key — matching Peer Mentoring and the job-search tool.

**Architecture:** Phase 1's headless `matching_core.py` (parse → Hungarian → assignment) is **unchanged**; this phase replaces the *client layer* that feeds it. A new `llm_client.py` (an Anthropic wrapper modeled on Peer Mentoring's `matching/llm.py`) is the single place LLM calls go through. "Stateful mode" (`previous_response_id`) is replaced by prompt caching; the hand-rolled rate-limit loops are replaced by the SDK's auto-retry; `temperature` is dropped (current Claude models reject it).

**Tech Stack:** Python 3.11, Streamlit, pandas, numpy, scipy, **anthropic** (new), pytest, ruff. `openai` + `tiktoken` removed.

**Spec:** `docs/revamp-plan.md`. Prior phase: `docs/superpowers/plans/2026-10-28-mentorship-revamp-phase1.md` (its `matching_core` is reused as-is). Reference implementation: Peer Mentoring's `matching/llm.py` (sibling repo) — streaming `client.messages.stream` + `get_final_message()` + `stop_reason == "max_tokens"` guard, no temperature.

## Global Constraints

- **Provider is Claude.** Use the official `anthropic` SDK (`anthropic.Anthropic`). No OpenAI, no OpenAI-compatible shims. Default model **`claude-opus-5-5`** for matrix scoring + reasoning; **`claude-haiku-5-5`** for the high-volume normalization classification/merge calls (cheap, simple). Model IDs are the bare strings — never date-suffixed.
- **No `temperature` / `top_p`** on any Claude call — current models return 400 if sent. Drop them entirely (do not port the 0.1 / 0.3 values).
- **Stream every matrix/reasoning call** (`client.messages.stream(...).get_final_message()`) — large `max_tokens` otherwise hits the HTTP timeout. Set an explicit `max_tokens` budget on every call, and **raise a clear error when `resp.stop_reason == "max_tokens"`** (never let a truncated matrix fall through to the fill-30 path).
- **This is NOT a no-behavior-change phase.** Exact compatibility scores will differ (different model) — that is expected. What must be preserved: the pipeline shape (LLM scores a 0–100 matrix → `matching_core` parses → Hungarian assigns), the strict output schema (`{"matrix":[{"mentor_id","scores":[{"mentee_id","percentage"}]}]}` and the reasoning shape), and `matching_core.py` itself (do not modify it).
- **Secrets:** read `ANTHROPIC_API_KEY` from env and `st.secrets`. Remove all `OPENAI_API_KEY` reads.
- **Prompt caching replaces stateful mode:** cache the stable prefix (system rubric + the mentees CSV, which is constant across mentor batches) with `cache_control: {type: "ephemeral"}`; vary only the mentor batch after the breakpoint. Verify with `usage.cache_read_input_tokens`.
- Keep the app runnable at every task boundary; `ruff check .` exit 0; all tests pass; CI green.

## Review Focus

- **Matrix output exceeds 128K output tokens** → for a full ~130×130 cohort the matrix is ~170K output tokens, over Claude's 128K cap. Batching must be sized by **output** tokens (≈ `mentees × batch_mentors × ~15`), not the old TPM input heuristic, so no single call can exceed the cap; and the `stop_reason` guard must fire if one still does. (Task 2.)
- **Truncated / cut-off response** → `stop_reason == "max_tokens"` raises and the batch is marked failed (the `update_batch_progress(..., 'error')` path), instead of `parse_matrix_response` silently filling the missing cells with 30. A test must pin this. (Task 1 + Task 2.)
- **Prompt-cache miss** → if the mentees CSV or system prefix varies between batches (ordering, a timestamp), caching silently won't hit. The cached prefix must be byte-stable; a test or a `cache_read_input_tokens` assertion guards it. (Task 2.)
- **Missing `ANTHROPIC_API_KEY`** → a clear, early error (not a stack trace mid-match), mirroring `llm.py`'s constructor check. (Task 1 + Task 5.)
- **Normalization cache correctness** → `AICache` keys on an md5 of the input text (model-agnostic), so it carries over unchanged; the migrated normalization calls must keep producing the same *shape* of result (a short classification / keyword string) so cached and fresh entries stay compatible. (Task 4.)

---

## Task 1: Claude client module (`llm_client.py`) + tests

**Files:**
- Create: `llm_client.py`, `tests/test_llm_client.py`

**Interfaces:**
- Produces:
  - `class LLMError(Exception)`.
  - `class ClaudeClient` — `__init__(self, model: str, api_key: str)` (raises `LLMError` on empty key, message naming `ANTHROPIC_API_KEY`).
  - `complete_json(self, system: str, user: str, *, max_tokens: int, cache_system: bool = False) -> str` — streams `client.messages.stream(model, max_tokens, system=..., messages=[{"role":"user","content":user}])`, returns the concatenated text of the response's text blocks. **No temperature.** Appends "Respond with a single valid JSON object and nothing else." to the system prompt (as `llm.py` does). Raises `LLMError` on `stop_reason == "max_tokens"` with a clear "response was cut off" message, and wraps any SDK exception in `LLMError`. When `cache_system=True`, sends `system` as a block list with `cache_control: {"type": "ephemeral"}` on the stable prefix.
  - `parse_json(text: str) -> dict` — best-effort JSON extraction (strip fences, grab outermost `{...}`), raising `LLMError` on failure. (Port Peer Mentoring's `parse_json_response`.)

- [ ] **Step 1: Write failing tests** `tests/test_llm_client.py` (mock `anthropic.Anthropic` like Peer Mentoring's `test_llm.py` — a fake stream context manager whose `get_final_message()` returns an object with `.stop_reason` and `.content` = list of text blocks):
  - `complete_json` returns the joined text, sends `max_tokens`, sends NO `temperature`/`top_p`, and the system string ends with the JSON instruction.
  - `stop_reason == "max_tokens"` raises `LLMError` matching "cut off".
  - empty api_key raises `LLMError` naming `ANTHROPIC_API_KEY`.
  - an SDK exception is wrapped in `LLMError`.
  - `cache_system=True` puts `cache_control` on the system block.
  - `parse_json` handles clean JSON, ```` ```json ```` fences, and surrounding prose; raises on garbage.
- [ ] **Step 2: Run** `pytest tests/test_llm_client.py -v` → FAIL.
- [ ] **Step 3: Implement** `llm_client.py` (model the stream/stop-guard on Peer Mentoring's `matching/llm.py`; add the `cache_system` option and the `max_tokens` parameter).
- [ ] **Step 4: Run** → PASS; `ruff check .` clean.
- [ ] **Step 5: Commit** (`feat: Claude LLM client (streaming + stop-guard, no temperature) + tests`).

## Task 2: Migrate matrix scoring to Claude

**Files:**
- Modify: `app.py` (`call_openai_api_with_chat_completions` → a Claude implementation; `generate_matrix_in_batches`; delete the Responses/Chat fallbacks + the hand-rolled retry loop + the `extract_openai_answer` matrix path), `utils.py` (`calculate_optimal_batch_size` → size by output tokens), `tests/test_matrix_scoring.py` (create)

**Interfaces:**
- Consumes: `ClaudeClient` (Task 1), `matching_core.parse_matrix_response` (Phase 1, unchanged), `prompts.get_prompt_for_api`, `validation.prepare_data_for_assistant`.
- Produces: a matrix-scoring function that, per mentor batch, calls Claude via `ClaudeClient.complete_json` with an explicit `max_tokens` budget and `cache_system=True`, returns the raw text, and lets `parse_matrix_response` build the matrix. The batcher sizes each batch so its estimated **output** tokens stay under a safe cap (e.g. 100K, headroom under 128K).

- [ ] **Step 1:** Rename/replace `call_openai_api_with_chat_completions` with a Claude version (keep the name its callers use, or update the `call_openai_api_with_assistants` alias + `generate_matrix_in_batches` call site). Build `system` = the rubric prefix + the mentees CSV (the cacheable, batch-invariant part); `user` = the mentor batch CSV + "Generate matches for ALL mentees…". Call `ClaudeClient(model, api_key).complete_json(system, user, max_tokens=<budget>, cache_system=True)`. Set `max_tokens` from `estimate_output_tokens(mentees, batch_mentors)` × a headroom factor, capped at ~120000. Return the text; drop the `(content, response_id)` stateful return (prompt caching needs no id threading).
- [ ] **Step 2:** `calculate_optimal_batch_size` → size by output tokens (mentees × batch × ~15) under the output cap, not the TPM input heuristic. Remove the `use_stateful` plumbing (caching is transparent).
- [ ] **Step 3:** Delete the dead OpenAI paths this touches: the Responses/Chat `responses_api`/legacy branches, the `rate_limit_exceeded`/`429` retry loop (the SDK auto-retries), `store`/`previous_response_id`, `temperature`/`top_p`. Add the stop-reason guard (it lives in `ClaudeClient`, so a cut-off batch raises → mark the batch `error` via `update_batch_progress`, not fill-30).
- [ ] **Step 4: Tests** `tests/test_matrix_scoring.py` (mock `ClaudeClient`): a well-formed matrix response → `parse_matrix_response` builds the expected matrix; a `stop_reason`-cutoff (ClaudeClient raises) → the batch is reported as an error, NOT silently 30-filled; the batcher never produces a batch whose estimated output exceeds the cap.
- [ ] **Step 5: Verify** `python -c "import app"` clean, `ruff check .` clean, `pytest -q` green.
- [ ] **Step 6: Commit** (`feat: Claude matrix scoring (streaming + token budget + caching); drop OpenAI matrix path`).

## Task 3: Migrate the reasoning call to Claude

**Files:**
- Modify: `app.py` (`get_reasoning_for_assignments` → Claude), `tests/test_reasoning.py` (create)

**Interfaces:**
- Consumes: `ClaudeClient`.
- Produces: `get_reasoning_for_assignments(...)` keeps its signature + return (dict keyed `(mentee_id, mentor_id) -> reasoning`), but calls Claude via `ClaudeClient.complete_json` with `max_tokens` ≈ the old 8000 budget (streamed), then parses the `reasonings` list as today. Drop the Responses/Chat fallback, the reasoning-model branch, temperature, and the retry loop.

- [ ] **Step 1: Write failing tests** (mock `ClaudeClient`): a valid `{"reasonings":[{mentee_id,mentor_id,reasoning}]}` → the expected keyed dict; a malformed/empty response → an empty dict (graceful, as today), no crash.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the Claude rewrite.
- [ ] **Step 4: Run** → PASS; import + ruff clean.
- [ ] **Step 5: Commit** (`feat: Claude reasoning call; drop OpenAI reasoning path`).

## Task 4: Migrate the normalization pipeline to Claude

**Files:**
- Modify: `Normalization/merge_mentee_skills_enhanced.py`, `Normalization/merge_mentor_skills_enhanced.py`, `Normalization/normalization.py` (the 7 `chat.completions.create` sites), `Normalization/shared_utils.py` (a shared Claude call helper if useful), `tests/test_normalization_llm.py` (create)

**Interfaces:**
- Consumes: `ClaudeClient` (or a thin normalization-local Claude helper on `claude-haiku-5-5`).
- Produces: each of the 7 classification/merge calls (education level, program split, broad field, skill merge ×2, mentor education+field, field classification) returns the same *shape* of short text it did before, now from Claude Haiku. `AICache` (md5-keyed on input) is kept unchanged — it is model-agnostic.

- [ ] **Step 1:** Add a small helper (e.g. `shared_utils.claude_classify(system, user, max_tokens)`), using `claude-haiku-5-5`, streamed, no temperature. Replace the module-level `openai.chat.completions.create` calls + `setup_openai()` with it.
- [ ] **Step 2:** Keep each call's prompt and the `AICache` wrapping; only the transport changes. Preserve the `MAX_RETRIES` graceful-fallback behavior (SDK retries transport errors; keep the non-AI fallback for a hard failure).
- [ ] **Step 3: Tests** `tests/test_normalization_llm.py` (mock the Claude helper): each classifier returns the expected shape from a canned Claude response; the AICache returns the cached value on a repeat input without a second call.
- [ ] **Step 4: Verify** import + ruff + pytest green.
- [ ] **Step 5: Commit** (`feat: Claude normalization pipeline (haiku); drop gpt-4o-mini calls`).

## Task 5: Config, model selection, and secrets → Claude

**Files:**
- Modify: `info.py` (`MODEL_CONFIGS`, `MODEL_TOKEN_LIMITS`, `API_PRICING`), `ui_config.py` (allow-list, model selector, drop the stateful toggle, key status), `utils.py` (`fetch_available_models` → Claude models or a static list; `get_model_config`), `app.py` (key read: `ANTHROPIC_API_KEY`), `tests/test_config.py` (create or extend)

**Interfaces:**
- Produces: Claude model configs (opus-5-5, sonnet-5-5, haiku-5-5) with the fields the code still needs (context ~1M, output cap 128K, pricing from the Claude table); the UI model selector lists Claude models (default opus-5-5); the key gate reads `ANTHROPIC_API_KEY` (env + `st.secrets`). The `supports_structured_output`/`supports_stateful`/`supports_temperature`/`is_reasoning_model` branches collapse (Claude: no temperature, caching not stateful) — simplify `MODEL_CONFIGS` to what remains.

- [ ] **Step 1:** Replace the OpenAI allow-list (`ui_config.py:52-59`) + `fetch_available_models` fallback with Claude models; default `model_choice = "claude-opus-5-5"`; remove the stateful toggle and its `supports_stateful` gating.
- [ ] **Step 2:** Update `info.py` dicts to Claude: `MODEL_TOKEN_LIMITS` (opus/sonnet/haiku context + 128K output), `API_PRICING` (Claude per-MTok rates), `MODEL_CONFIGS` trimmed to live fields (`tpm_limit`/output-cap for batching, `family`). Point `estimate_api_cost` at the Claude pricing.
- [ ] **Step 3:** `app.py` key read → `ANTHROPIC_API_KEY` from env and `st.secrets`; `ui_config.py` status text updated.
- [ ] **Step 4: Tests:** `get_model_config("claude-opus-5-5")` returns sane fields; the default model is a Claude id; cost estimate uses Claude pricing.
- [ ] **Step 5: Verify + Commit** (`feat: Claude model config, selector, and ANTHROPIC_API_KEY secrets`).

## Task 6: Dependency swap + dead-code removal + CI

**Files:**
- Modify: `requirements.txt` (add `anthropic`, remove `openai`, remove `tiktoken` if confirmed unused), `app.py`/`utils.py` (remove now-dead OpenAI helpers: `extract_openai_answer` OpenAI-specifics, the chat-completions legacy functions, `fetch_available_models`'s OpenAI client call), any lingering `import openai`
- Verify: `.github/workflows/ci.yml` still green

**Interfaces:**
- Consumes: Tasks 1–5 (all call sites migrated, so nothing imports `openai` anymore).

- [ ] **Step 1:** `grep -rn "openai\|OpenAI\|tiktoken" --include=*.py .` — confirm every remaining reference is dead; remove the dead functions/branches. Confirm `tiktoken` has no import before dropping it.
- [ ] **Step 2:** `requirements.txt`: add `anthropic==<pinned>`, remove `openai`/`tiktoken`. `pip install -r …` into the venv.
- [ ] **Step 3: Verify** `python -c "import app"` clean, `ruff check .` exit 0, `pytest -q` all green, and no `import openai` remains.
- [ ] **Step 4: Commit** (`chore: add anthropic dep, remove openai/tiktoken + dead OpenAI code`).

---

## After Phase 2

- **Phase 3 — UX:** stepped navigation (reachability-gated rail), async background-thread generation with locked view + time-based progress + cancel (the matching run is still minutes-long), a data-review step, the organizer-facing **avoid-same-country toggle** (the configurable rule from Phase 1, now wired to the UI), IMFAHE theme + impeccable pass.
- **Phase 4 — Infra/ops finish:** `DEBUG_MODE` off by default, re-enable XSRF for a real deploy, `.streamlit/` config + Streamlit Cloud deployment + `APP_PASSWORD` gate, security-review Action.
- **First real run:** with a real `ANTHROPIC_API_KEY` + the `samples/` data, run one full match end-to-end and sanity-check the circles/assignments (the quality baseline we deferred from Phase 1).

## Self-review notes

- Every task ends with an independently testable deliverable; the app stays runnable (openai stays installed until Task 6 removes it after all call sites are migrated).
- `matching_core.py` is untouched — the new client feeds its existing `parse_matrix_response`; the Review Focus items (output-cap, truncation→error not fill-30, cache hits, missing key, normalization cache) each map to a task's tests.
- Types/interfaces stay consistent: `ClaudeClient.complete_json` is the single LLM entry point; IDs remain strings; the matrix/reasoning JSON shapes are preserved so Phase 1's parser and the downstream dicts keep working.
- Not a no-behavior-change phase (scores differ by model) — but the pipeline shape, schema, and deterministic assignment are preserved, and the truncation fix is a correctness improvement, not a behavior regression.
