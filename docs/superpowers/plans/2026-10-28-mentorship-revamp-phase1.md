# Mentorship Matching Revamp — Phase 1: Foundation (Decouple, De-risk, Test)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the matching logic into a tested, headless (no-Streamlit) core, remove dead code, and stand up a test/CI safety net — without changing matching behavior or the provider — so the later Claude migration and UX work are safe to do.

**Architecture:** Keep the hybrid matcher (LLM scores a matrix → scipy Hungarian assigns). Extract the pure logic out of `matching.py` so it returns plain data + a report object instead of calling `st.*`, and let the UI render the report. Delete the orphaned modules. Add pytest around the deterministic core. Still runs entirely on OpenAI after this phase.

**Tech Stack:** Python 3.11, Streamlit, pandas, numpy, scipy, openai (unchanged this phase), pytest, ruff.

**Spec:** `docs/revamp-plan.md` (the lessons-transfer plan and decisions this argues from).

## Global Constraints

- **No behavior change to matching outputs this phase** — this is a pure refactor + bug-fix + test phase. A before/after run on the sample data must produce the same assignments.
- **Provider stays OpenAI this phase.** The Claude migration is Phase 2.
- **Pure-logic modules must not `import streamlit`.** After this phase, `matching.py` (and any new `matching_core.py`) import no Streamlit. UI rendering of results/warnings moves to the caller.
- **Preserve the column-name contracts** in `info.py` (`MENTEE_COLUMNS`, `MENTOR_COLUMNS`) and the sequential-integer IDs produced by normalization. Matching code treats mentee/mentor IDs as **strings** internally (as `parse_matrix_response` already does).
- **Keep the app runnable end-to-end at every task boundary** (`streamlit run app.py` still works).
- Tag the pre-revamp state `v1.0` before any change (rollback point), mirroring Peer Mentoring.

## Review Focus

- **Truncated / incomplete LLM matrix** → missing mentee cells are currently filled with a hardcoded `30` *silently* (`matching.py:196-203`, `:235-238`). The refactor must make this a **reported, flagged** outcome (surfaced in the returned report, rendered by the UI), not a silent mutation. The real fix (output-token budget) lands in Phase 2 — Task 3 pins the flagged behavior with a test.
- **Integer vs string mentee-id columns** → the missing-mentor retry rebuild indexes with raw ids (`app.py:908 complete_matrix_df[sorted(all_mentee_ids)]`) while the matrix columns are strings → `KeyError`. Task 4's core must standardize on string columns; Task 6 fixes the caller.
- **Stateful-mode latent crash** → `prepare_data_for_assistant(None, mentors_to_use, [])` (`app.py:110`) passes 3 args to a 2-arg function. Fix in Task 6.
- **More mentees than mentor capacity** → Hungarian assigns only `min(n_mentees, n_mentors×cap)`; extra mentees go unmatched. The core must return unmatched mentees explicitly rather than dropping them unreported (Task 4).
- **Missing country column** → the country rule must skip gracefully (it does today via `st.warning` + early return; preserve that as a report flag, not a crash) (Task 5).

---

## Task 1: Dev tooling, pinned deps, and the v1.0 tag

**Files:**
- Create: `requirements-dev.txt`, `pyproject.toml` (ruff + pytest config), `tests/__init__.py`, `tests/conftest.py`
- Modify: `requirements.txt` (pin to exact versions currently resolved)

**Interfaces:**
- Produces: a `tests/` package importable by pytest; `pytest` and `ruff check .` both runnable from repo root.

- [ ] **Step 1: Tag the rollback point.** `git tag -a v1.0 -m "pre-revamp baseline" && git push origin v1.0` (confirm current `main` is the intended baseline first).
- [ ] **Step 2:** Pin `requirements.txt` to exact `==` versions (resolve from a clean install of the current `>=` set). Add `requirements-dev.txt` with `pytest`, `ruff`.
- [ ] **Step 3:** Add `pyproject.toml` with `[tool.ruff]` (line length, target py311) and `[tool.pytest.ini_options]` (`testpaths = ["tests"]`).
- [ ] **Step 4:** Create `tests/__init__.py` and `tests/conftest.py` with shared fixtures: a small in-memory `matrix_df` (mentors×mentees), tiny `mentees_df`/`mentors_df` with the `info.py` columns, all fully synthetic (no PII).
- [ ] **Step 5: Verify** `pytest -q` runs (collects 0 tests, exits 0) and `ruff check .` runs.
- [ ] **Step 6: Commit** (`chore: dev tooling, pinned deps, v1.0 tag`).

## Task 2: Remove dead code

**Files:**
- Delete: `ui_training.py`, `ui_processing_modal.py`, `openai_files.py`
- Modify: `app.py` (remove `training_file_ids`/`training_file_names`/`training_files` session keys and any imports of the deleted modules), any module importing the above.

**Interfaces:**
- Consumes: the explorer finding that these three are never wired into `app.py` and training files are never used by the matcher.
- Produces: a tree with no orphaned modules.

- [ ] **Step 1:** `grep -rn "ui_training\|ui_processing_modal\|openai_files\|training_file" --include=*.py .` — record every reference.
- [ ] **Step 2:** Delete the three files; remove the referencing imports and the dead session-state keys.
- [ ] **Step 3: Verify** `python -c "import app"` imports with no error, and the grep from Step 1 now returns only this plan/docs.
- [ ] **Step 4: Verify** `streamlit run app.py` still launches the 6-tab UI (manual smoke check, or `streamlit run --server.headless true` boots without exception).
- [ ] **Step 5: Commit** (`refactor: delete orphaned training/modal/files modules`).

## Task 3: Headless `parse_matrix_response` + tests

**Files:**
- Create: `matching_core.py` (new home for pure parsing/matrix logic), `tests/test_parse_matrix.py`
- Modify: `matching.py` (re-export or thin-wrap for callers during transition)

**Interfaces:**
- Produces:
  - `@dataclass MatrixParseReport` with fields: `df: pd.DataFrame | None`, `wrong_mentors: list[str]`, `wrong_mentees: list[str]`, `missing_mentors: list[str]`, `missing_mentees_filled: list[str]`, `completion_rate: float`, `error: str | None`.
  - `parse_matrix_response(response_text: str, expected_mentors: list, expected_mentees: list) -> MatrixParseReport` — same logic as today (strip fences, slice `{`..`}`, `json.loads`, validate IDs against the batch, fill missing mentee columns with `30`) but **returns the report instead of calling `st.*`**. The `30`-fill is recorded in `missing_mentees_filled`, not announced via `st.warning`.
  - Columns of `report.df` are **strings**, sorted by `expected_mentees` (as today, `matching.py:205`).

- [ ] **Step 1: Write failing tests** `tests/test_parse_matrix.py`:
```python
def test_parses_clean_matrix():
    txt = '{"matrix":[{"mentor_id":"1","scores":[{"mentee_id":"1","percentage":80},{"mentee_id":"2","percentage":40}]}]}'
    r = parse_matrix_response(txt, ["1"], ["1", "2"])
    assert r.error is None
    assert list(r.df.columns) == ["1", "2"]
    assert r.df.loc["1", "1"] == 80

def test_strips_markdown_fences():
    r = parse_matrix_response('```json\n{"matrix":[{"mentor_id":"1","scores":[{"mentee_id":"1","percentage":90}]}]}\n```', ["1"], ["1"])
    assert r.df.loc["1", "1"] == 90

def test_missing_mentee_filled_with_30_and_reported():
    txt = '{"matrix":[{"mentor_id":"1","scores":[{"mentee_id":"1","percentage":70}]}]}'
    r = parse_matrix_response(txt, ["1"], ["1", "2"])
    assert r.df.loc["1", "2"] == 30
    assert "2" in r.missing_mentees_filled

def test_wrong_mentor_id_rejected():
    txt = '{"matrix":[{"mentor_id":"99","scores":[{"mentee_id":"1","percentage":70}]}]}'
    r = parse_matrix_response(txt, ["1"], ["1"])
    assert "99" in r.wrong_mentors and r.df is None

def test_truncated_json_returns_error():
    r = parse_matrix_response('{"matrix":[{"mentor_id":"1","scores":[{"mentee_id":"1","percent', ["1"], ["1"])
    assert r.df is None and r.error is not None
```
- [ ] **Step 2: Run** `pytest tests/test_parse_matrix.py -v` → FAIL (import/undefined).
- [ ] **Step 3: Implement** `parse_matrix_response` + `MatrixParseReport` in `matching_core.py`, porting the logic from `matching.py:16-214` with all `st.*` calls removed and their information captured in the report. Keep the reasoning-model `ResponseTextConfig` guard (set `report.error`).
- [ ] **Step 4: Run** `pytest tests/test_parse_matrix.py -v` → PASS.
- [ ] **Step 5: Commit** (`refactor: headless parse_matrix_response with report object + tests`).

## Task 4: Headless `hungarian_assignment` + `check_mentor_conflicts` + tests

**Files:**
- Modify: `matching_core.py` (add the assignment functions), `tests/test_assignment.py` (create)

**Interfaces:**
- Consumes: `report.df` from Task 3 (string columns).
- Produces:
  - `@dataclass AssignmentResult` with `assignments: list[tuple[str, str, int]]` (mentee, mentor, score), `unmatched_mentees: list[str]`, `cleaning: dict` (counts of NaN→30, inf→50, clipped).
  - `hungarian_assignment(matrix_df, max_mentees_per_mentor=2) -> AssignmentResult` — same scipy `linear_sum_assignment` over `-score` with mentor column-cloning (`matching.py:262-304`), **no `st.*`**; NaN/inf/clip cleaning recorded in `cleaning`; mentees with no assigned mentor reported in `unmatched_mentees`.
  - `check_mentor_conflicts(matches, max_mentees_per_mentor=2) -> list[dict]` — moved unchanged (already pure, `matching.py:307-342`).

- [ ] **Step 1: Write failing tests** `tests/test_assignment.py`:
```python
def test_optimal_assignment_prefers_high_scores():
    df = pd.DataFrame({"1": [90, 10], "2": [20, 80]}, index=["m1", "m2"])  # mentors x mentees
    res = hungarian_assignment(df, max_mentees_per_mentor=1)
    pairs = {mentee: mentor for mentee, mentor, _ in res.assignments}
    assert pairs == {"1": "m1", "2": "m2"}

def test_cap_limits_mentees_per_mentor():
    df = pd.DataFrame({"1": [90, 1], "2": [85, 1], "3": [80, 1]}, index=["m1", "m2"])
    res = hungarian_assignment(df, max_mentees_per_mentor=2)
    counts = {}
    for _, mentor, _ in res.assignments:
        counts[mentor] = counts.get(mentor, 0) + 1
    assert all(c <= 2 for c in counts.values())

def test_more_mentees_than_capacity_reports_unmatched():
    df = pd.DataFrame({"1": [90], "2": [80], "3": [70]}, index=["m1"])  # 1 mentor, cap 2, 3 mentees
    res = hungarian_assignment(df, max_mentees_per_mentor=2)
    assert len(res.assignments) == 2 and len(res.unmatched_mentees) == 1

def test_nan_filled_and_recorded():
    df = pd.DataFrame({"1": [float("nan")], "2": [80.0]}, index=["m1"])
    res = hungarian_assignment(df, max_mentees_per_mentor=2)
    assert res.cleaning["nan_filled"] == 1
```
- [ ] **Step 2: Run** `pytest tests/test_assignment.py -v` → FAIL.
- [ ] **Step 3: Implement** in `matching_core.py` (port `matching.py:217-342`, strip `st.*`, add `unmatched_mentees` + `cleaning`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** (`refactor: headless hungarian_assignment + conflicts with tests`).

## Task 5: Configurable country rule + tests

**Files:**
- Modify: `matching_core.py` (replace hardcoded Spain logic), `info.py` (if a default config constant is added), `tests/test_country_rule.py` (create)

**Interfaces:**
- Produces:
  - `apply_country_restrictions(matrix_df, mentees_df, mentors_df, avoid_same_country: bool = False, countries: list[str] | None = None) -> tuple[pd.DataFrame, list[dict]]`.
    - `avoid_same_country=True`: zero any mentor↔mentee cell where both normalized countries are equal and non-empty.
    - `countries=["spain"]`: restrict only within those countries (reproduces today's Spain-only behavior).
    - Missing country column → return unchanged matrix + a `[{"skipped": "mentor country column missing"}]` marker (no `st.*`).
  - Default is **`avoid_same_country=False`** (no restriction) until the UI toggle is wired; the caller passes the organizer's choice.

- [ ] **Step 1: Write failing tests** `tests/test_country_rule.py`: same-country zeroed when `avoid_same_country=True`; Spain-only mode zeroes Spain-Spain but leaves Portugal-Portugal; missing column returns unchanged + skip marker; filtered matches reported with `original_score`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — generalize `matching.py:345-423` to the signature above; normalize countries with `.strip().lower()` as today.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** (`feat: configurable avoid-same-country rule (was hardcoded Spain) + tests`).

## Task 6: Wire the headless core into the UI + fix the two latent crashes

**Files:**
- Modify: `ui_processing.py` (render the Task 3–5 reports), `app.py` (call the new core; fix the two bugs), delete the now-duplicated logic from `matching.py` (leave thin re-exports or update imports).

**Interfaces:**
- Consumes: `MatrixParseReport`, `AssignmentResult`, the new country-rule signature.

- [ ] **Step 1:** Update callers (`ui_processing.py`, `app.py`) to call `matching_core` and render the report fields with `st.*` **at the call site** (warnings for `wrong_mentors`, `missing_mentees_filled`, `unmatched_mentees`, cleaning counts). Match the information users saw before.
- [ ] **Step 2: Fix the stateful-mode crash** — `app.py:110`: change `prepare_data_for_assistant(None, mentors_to_use, [])` to the 2-arg call, and confirm `prepare_data_for_assistant` tolerates `mentees_df=None` (add a guard in `validation.py:234` if not).
- [ ] **Step 3: Fix the retry-path KeyError** — `app.py:908`: index with string columns, `complete_matrix_df[[str(m) for m in sorted(all_mentee_ids)]]`.
- [ ] **Step 4: Verify** `python -c "import app"` clean; `ruff check .` clean; `pytest -q` all green.
- [ ] **Step 5: Verify behavior unchanged** — run `streamlit run app.py` against the `samples/` files (after normalization) and confirm the assignment output matches the pre-refactor result for the same inputs/model (spot-check a few pairs; the deterministic Hungarian step must be identical given the same matrix).
- [ ] **Step 6: Commit** (`refactor: UI renders matching reports; fix stateful + retry-path crashes`).

## Task 7: CI (ruff + pytest) + Dependabot

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/dependabot.yml`

**Interfaces:**
- Consumes: the test suite from Tasks 3–5 and `requirements-dev.txt`.

- [ ] **Step 1:** Add `ci.yml` — on push/PR to `main`: set up Python 3.11, `pip install -r requirements.txt -r requirements-dev.txt`, run `ruff check .` then `pytest -q`. Mirror Peer Mentoring's workflow (least-privilege `permissions`, concurrency cancel-in-progress, pinned action SHAs/tags).
- [ ] **Step 2:** Add `dependabot.yml` — monthly updates for `pip` and `github-actions`.
- [ ] **Step 3: Verify** the workflow is valid YAML and the commands it runs pass locally.
- [ ] **Step 4: Commit** (`ci: ruff + pytest on push/PR; dependabot`).

---

## After Phase 1

Separate plans (written when we reach them, each argued from `docs/revamp-plan.md`):
- **Phase 2 — Claude migration:** replace the OpenAI Responses-API scoring + reasoning (and normalization's `gpt-4o-mini`) with Claude; streaming + real output-token budget + stop-reason guard (kills the silent-truncation→fill-30 issue at the source); prompt caching in place of "stateful" mode; drop the dead model-compatibility branches.
- **Phase 3 — UX:** stepped navigation (reachability-gated `st.button` rail), async background-thread generation with locked view + time-based progress + cancel, a data-review step, lock-and-regenerate, IMFAHE theme + impeccable pass.
- **Phase 4 — Infra/ops finish:** `DEBUG_MODE` off by default, re-enable XSRF for any real deploy, `.streamlit/` config + Streamlit Cloud deployment + `APP_PASSWORD` gate, security-review Action.

## Self-review notes

- Every Phase-1 task ends with an independently testable deliverable and the app still runs.
- The refactor is behavior-preserving by constraint; Task 6 Step 5 is the explicit before/after check.
- The two latent bugs and the silent-fill-30 are each pinned to a task (6, 6, 3) and listed in Review Focus.
- Types are consistent: mentee/mentor IDs are strings throughout the core; `report.df` columns are strings; the caller converts only for display.
