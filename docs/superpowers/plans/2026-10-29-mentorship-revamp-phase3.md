# Mentorship Matching Revamp — Phase 3: Stepped-Wizard UX

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the app's 6-tab layout with a guided, numbered **stepped wizard** (one screen at a time, with a clickable progress rail that only unlocks steps the data supports), turn match generation into a **dedicated, rail-locked Generate screen**, and expose the country-matching rule as an **organizer-facing toggle** instead of a hardcoded Spain constant — so a non-technical IMFAHE organizer is walked through the flow the way Peer Mentoring already walks them.

**Architecture:** The six existing `render_*_tab()` modules (`ui_config`, `ui_upload`, `ui_prompt`, `ui_overview`, `ui_matches`, `ui_processing`) are **kept and reused unchanged as step bodies** — this phase rewrites only the *shell* in `app.py`: the `st.tabs(...)` block becomes a step rail + a `_resolve_step` dispatcher, modeled on Peer Mentoring's single-file `render_rail` / `_resolve_step` / `_reachable` / `_nav_to` pattern (real `st.button`s that rerun in place and preserve `session_state`, never HTML anchor links which full-reload and reset it). The match pipeline (`generate_matrix_in_batches` → `hungarian_assignment` → `get_reasoning_for_assignments`) is **not refactored**; it stays synchronous and keeps emitting its own `progress_ui` batch cards. "Async" here means a dedicated locked step around that existing streaming progress, not background threading (see the ruling in Task 2).

**Tech Stack:** Python 3.11, Streamlit, pandas, numpy, scipy, anthropic, pytest, ruff. No new dependencies.

**Spec:** `docs/revamp-plan.md` (lessons-transfer from Peer Mentoring). Prior phases: `docs/superpowers/plans/2026-10-28-mentorship-revamp-phase1.md`, `docs/superpowers/plans/2026-10-29-mentorship-revamp-phase2.md`. **Reference implementation:** the sibling Peer Mentoring repo's `app.py` — `_STEPS`, `render_rail(current, ...)`, `_resolve_step` / `_reachable` / `_nav_to` / `_goto`, and the locked-generation view (`render_generating_view`, `render_rail(2, clickable=False)`).

## Global Constraints

- **Reuse, don't rewrite.** The six `render_*_tab()` functions keep their names, signatures, and bodies. A step body is a call to the existing function. If a tab module reads/writes a `session_state` key today, it keeps doing so; the shell must not rename or reinterpret those keys.
- **Navigation preserves state.** The rail is built from `st.button`s that rerun in place (Peer Mentoring's approach), never `<a href>` anchors. Clicking a step must not clear uploads, prompt edits, or results.
- **No behavior change to the match pipeline.** `generate_matrix_in_batches`, `hungarian_assignment`, `get_reasoning_for_assignments`, `matching_core.py`, and the normalization layer are untouched. Generation stays synchronous.
- **Default behavior is preserved.** With the organizer touching no new control, the country rule must behave exactly as today: the Spain↔Spain restriction applied (`countries=["spain"]`).
- **Password gate stays first.** The authentication block at the top of `main()` runs before any rail or step renders, unchanged.
- **Keep it runnable at every task boundary.** `ruff check .` exits 0; all existing tests still pass; the app launches (`streamlit run app.py`) and every step renders.
- **Brand is already IMFAHE.** `ui.py` `CSS_STYLES` already defines the red `#E6122C` / orange `#F39220` / text `#3A3A3A` palette, Open Sans, and `.wizard-step` / `.wizard-step-number` classes. Do **not** import Peer Mentoring's Spectral serif or a second palette — style the new rail with the brand and classes that already exist here.

## Review Focus

- **A step is reached before its data exists** → navigating (or landing) on Overview/Generate with no uploaded files, or on Results with no assignments, must show a gentle "do step N first" state, never a `KeyError`/`AttributeError` on a `None` DataFrame. `_reachable` gates this and is unit-tested. (Task 1.)
- **A new upload silently strips stale results** → if the organizer re-uploads after generating, the old `matrix_df`/`assignments`/`matches` must not keep the rail showing a stale Results step as reachable. The stored step is clamped to what's currently reachable on every run. (Task 1.)
- **A stray click during a multi-minute match** → while generation runs, the rail and other controls must be withheld so a mid-run click can't reset the flow or trigger a second match. (Task 2.)
- **Country toggle off, then generate** → turning the restriction off must pass `countries=[]` (or equivalent) so `apply_country_restrictions` filters nothing; turning it on restores `["spain"]`. A unit test pins both directions through the real `apply_country_restrictions`. (Task 3.)
- **Debug mode and existing messaging survive the reshuffle** → `is_debug_mode()` gating, the parse-report warnings, and the country-restriction summary still render in their step, not lost between the tab→step move. (Task 2.)

---

## Task 1: Step rail + navigation shell (replace `st.tabs`)

Replace the tab layout in `app.py` `main()` with a numbered step rail and a step dispatcher. The rail is the themed, clickable progress indicator; each step renders one existing `render_*_tab()`.

**Files:**
- Modify: `app.py` (the `tab_names` / `st.tabs(...)` block in `main()`, ~lines 728–765, and add nav helpers near the other module-level helpers)
- Create: `tests/test_navigation.py`

**Interfaces:**
- Produces (module-level in `app.py`, for the dispatcher and tests):
  - `_STEPS: list[str]` — the six ordered labels, e.g. `["Configure", "Upload", "Customize Prompt", "Data Overview", "Generate", "Results"]`.
  - `_reachable(has_data: bool, has_results: bool) -> set[int]` — **pure**, no Streamlit. Returns the set of reachable step indices. Steps 0 (Configure), 1 (Upload), 2 (Prompt) are always reachable; 3 (Overview) and 4 (Generate) require `has_data`; 5 (Results) requires `has_results`.
  - `_resolve_step(has_data: bool, has_results: bool) -> int` — reads `st.session_state["step"]` (default 0), clamps it into `_reachable(...)`, writes the clamp back, returns it.
  - `_nav_to(step: int) -> None` — `st.session_state["step"] = step` (on_click callback; no rerun — the button triggers its own).
  - `render_rail(current: int, has_data: bool, has_results: bool, clickable: bool = True) -> None` — one `st.columns(len(_STEPS))` row of `st.button`s; the current step is `type="primary"`, done steps show a `✓`, unreachable steps are `disabled`. `clickable=False` disables all (locked mode for Task 2).
- `has_data` is computed in `main()` as `st.session_state.get("mentees_df") is not None and st.session_state.get("mentors_df") is not None`. `has_results` as `st.session_state.get("matches") is not None` (confirm the exact "results ready" key against the generation step — adjust to whatever `render_matches_tab`/`render_processing_tab` sets when a match completes; `assignments`/`matrix_df` are the fallbacks).

- [ ] **Step 1: Write the failing test** — `tests/test_navigation.py`, pure-function tests of `_reachable` (import `app`):

```python
def test_reachable_no_data():
    assert app._reachable(has_data=False, has_results=False) == {0, 1, 2}

def test_reachable_with_data():
    assert app._reachable(has_data=True, has_results=False) == {0, 1, 2, 3, 4}

def test_reachable_with_results():
    assert app._reachable(has_data=True, has_results=True) == {0, 1, 2, 3, 4, 5}

def test_results_unreachable_without_data():
    # results can't exist without data, but guard the invariant anyway
    assert 5 not in app._reachable(has_data=False, has_results=False)
```

- [ ] **Step 2: Run it to confirm it fails** — `pytest tests/test_navigation.py -v` → FAIL (`_reachable` undefined). `import app` already works under pytest today (the existing `tests/test_matrix_scoring.py` / `tests/test_reasoning.py` import it despite `app.py`'s top-level `st.set_page_config`/`st.markdown`), so no module restructure is needed — import `app` the same way they do.
- [ ] **Step 3: Implement the nav helpers** (`_STEPS`, `_reachable`, `_resolve_step`, `_nav_to`, `render_rail`) at module level in `app.py`, mirroring Peer Mentoring's versions but with this app's 6 steps and `has_data`/`has_results` signature. Style the rail buttons with the existing brand (reuse `.wizard-step*` intent); the current step is visually distinct.
- [ ] **Step 4: Replace the `st.tabs` block** in `main()` with: compute `has_data`/`has_results` → `step = _resolve_step(...)` → `render_rail(step, has_data, has_results)` → dispatch `if step == 0: render_config_tab()` … `elif step == 5: render_processing_tab()`. Keep the column-wrap workaround only if still needed; the tab-render bug it worked around no longer applies without tabs, so drop it.
- [ ] **Step 5: Run the test + launch** — `pytest tests/test_navigation.py -v` PASS; `ruff check .` exit 0; `streamlit run app.py` and confirm every step renders and the rail gates correctly (no data → steps 4/5 disabled).
- [ ] **Step 6: Commit** — `git add app.py tests/test_navigation.py && git commit`

## Task 2: Wire the two-step Generate→Results flow with a rail lock

The run is **already split across two of the existing tabs**, and that split maps cleanly onto two steps: step 4 (`render_matches_tab`) shows the pre-flight summary and the "🚀 Generate Matches" button, which today only sets `st.session_state.start_processing = True` and tells the user to open Tab 6; step 5 (`render_processing_tab`) runs the whole pipeline **synchronously** when `start_processing` is set, writes `matrix_df`→`assignments`→`matches`, then sets `start_processing = False` and renders results. This task makes that flow work without manual tab-hopping and locks the rail while the match runs.

**Ruling (threading):** Peer Mentoring runs generation in a background thread because its worker is pure. This app's pipeline (`generate_matrix_in_batches`, ~300 lines) calls `st.*` on every batch and streams `progress_ui` batch cards, so it cannot move into a non-script thread without a large, risky refactor — and the per-batch cards already defeat the "looks frozen" problem threading solved there. So generation stays **synchronous**; `start_processing` is the lock signal. True background threading is out of scope for Phase 3 (a later phase if the within-batch wait must disappear). Cost if wrong: the UI blocks within a single batch's Opus call; acceptable — cards show progress between batches.

**Files:**
- Modify: `app.py` (`main()` — add the pre-rail `start_processing` lock check)
- Modify: `ui_matches.py` (the "🚀 Generate Matches" button at ~l.153 — also advance the step and rerun, instead of instructing the user to open Tab 6; and fix any stale "Tab N" copy in this file)
- Modify: `ui_upload.py` (when a new upload replaces `mentees_df`/`mentors_df`, clear stale `matrix_df`/`assignments`/`matches` so the rail can't show a stale Results step — Task-1 review carried this; it is plan Review-Focus #2)

**Interfaces:**
- Consumes: Task 1's `_STEPS`, `render_rail(..., clickable=False)`, and the `step` session key.
- The lock lives in `main()`, checked **before** the normal `_resolve_step`/rail/dispatch, mirroring Peer Mentoring's `if st.session_state.get("gen"): render_rail(2, clickable=False); render_generating_view(); return`:
  ```python
  if st.session_state.get("start_processing"):
      render_rail(5, has_data, has_results, clickable=False)  # locked: a stray click can't derail the run
      render_processing_tab()  # runs the pipeline this rerun, then clears start_processing + stores matches
      return
  ```

- [ ] **Step 1: Advance from the Generate button** — in `ui_matches.py`, change the "🚀 Generate Matches" button handler so that in addition to `start_processing = True` it sets `st.session_state["step"] = 5` and calls `st.rerun()`. Remove the now-wrong "navigate to Tab 6 / open Tab 6" `st.success`/`st.info` copy (there are no tabs anymore).
- [ ] **Step 1b: Clear stale results on a new upload** — in `ui_upload.py`, at the point where a processed dataframe is stored into `mentees_df`/`mentors_df` (via `FILE_TYPE_CONFIG[...]['df_key']`), also reset `matrix_df`, `assignments`, and `matches` to `None`. This keeps `has_results` false after a re-upload so the rail doesn't offer a stale Results step. Keep it minimal — reset only those three result keys, don't touch upload/progress state.
- [ ] **Step 2: Lock the rail during the run** — in `main()`, add the pre-rail `if st.session_state.get("start_processing"):` block above (compute `has_data`/`has_results` first). `render_processing_tab()` already clears `start_processing` and stores `matches` on completion, so the next rerun renders normally with Results reachable.
- [ ] **Step 3: Land on Results after completion** — confirm that once `matches` is set the organizer sees step 5 as current/reachable. If `render_processing_tab`'s own early `st.stop()` guards (no data / duplicate IDs) fire, the lock block still shows the locked rail + that message; that's acceptable (an invalid run can't proceed). On a pipeline failure (`matrix_df is None`), `start_processing` is cleared by the existing code and the organizer can retry from step 4 — verify this leaves the rail usable again.
- [ ] **Step 4: Manual verification** — `streamlit run app.py`, upload the sample files (`samples/`), click Generate on step 4, and confirm: it jumps to step 5, the rail is disabled during the run, the batch cards stream, debug messages and the "Applying Country Restrictions" summary still appear, and on completion the rail re-enables with Results current.
- [ ] **Step 5: Run checks** — `ruff check .` exit 0; full `pytest` green.
- [ ] **Step 6: Commit** — `git add -A && git commit`

## Task 3: Organizer-facing country-restriction toggle

Replace the hardcoded `countries=["spain"]` with a Configure-step control, defaulting to today's behavior.

**Files:**
- Modify: `app.py` (`generate_matrix_in_batches` — read the restriction from `session_state` instead of the literal; ~line 349)
- Modify: `ui_config.py` (add the toggle control in the Configure step)
- Create/extend: `tests/test_country_toggle.py`

**Interfaces:**
- Produces: `st.session_state["country_restrictions"]` — a `list[str]` of lowercased country names to restrict same-country matches for. Initialized to `["spain"]` in `main()`'s session-state block (preserving default behavior).
- `generate_matrix_in_batches` passes `countries=st.session_state.get("country_restrictions", ["spain"])` to `apply_country_restrictions` instead of the literal `["spain"]`.
- `ui_config.py` renders a checkbox ("Apply the Spain↔Spain mentoring restriction", default on) that sets `country_restrictions` to `["spain"]` when checked and `[]` when unchecked. Keep it minimal — a single checkbox matching the one rule in use today, not a general country editor (YAGNI).

- [ ] **Step 1: Write the failing test** — `tests/test_country_toggle.py`: build a tiny matrix + mentee/mentor frames with two Spanish rows; assert that `apply_country_restrictions(matrix, mentees, mentors, countries=["spain"])` zeroes the Spain↔Spain cell and `countries=[]` leaves it unchanged. (Pins both directions through the real function; no Streamlit.)
- [ ] **Step 2: Run it to confirm it fails/passes shape** — `pytest tests/test_country_toggle.py -v`. (This exercises the existing `apply_country_restrictions` contract; if it already behaves this way, the test documents it and the remaining steps wire the toggle.)
- [ ] **Step 3: Wire the session key** — add `country_restrictions` default `["spain"]` to `main()`; change the `generate_matrix_in_batches` call site to read it.
- [ ] **Step 4: Add the Configure checkbox** in `ui_config.py`.
- [ ] **Step 5: Verify** — `pytest` green, `ruff check .` exit 0; in the running app, toggle off → the "Applying Country Restrictions" section reports 0 filtered.
- [ ] **Step 6: Commit** — `git add -A && git commit`

## Notes for the executor

- Tasks 1 → 2 are ordered (2 depends on the rail). Task 3 is independent of both and can be done any time after Task 1 (it also touches `app.py`, so sequence it after Task 1 to avoid overlapping edits to `main()`'s session-state block).
- The "IMFAHE theme" Phase-3 goal is satisfied inside Task 1 (the themed rail using the existing brand/classes) — there is no separate theme task, because the palette, fonts, and `.wizard-step` CSS already exist in `ui.py`.
- Keep each commit runnable; the app must launch and all steps render at every task boundary.
