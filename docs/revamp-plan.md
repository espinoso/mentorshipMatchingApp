# Mentorship Matching App — Revamp Plan

*Lessons-transfer analysis from the Peer Mentoring Circles revamp. Written 2026-10-28. Discovery output for review — no code changed yet.*

---

## 1. What this app is

IMFAHE's **International Mentor Program (IMP)** matcher: upload a mentees file and a mentors file, an LLM scores every mentor×mentee pair for compatibility, and an optimal assignment pairs each mentee to one mentor (with a cap on mentees-per-mentor). It's a sibling of Peer Mentoring Circles — same org, same Streamlit + LLM + Excel-upload shape — but a **different matching problem**: 1 mentor → N mentees (assignment), not circles of 3–6 (grouping).

## 2. How it works today

**Flow:** a **6-tab** layout behind a shared-password gate — Configure → Upload → Customize Prompt → Data Overview → Generate → Processing & Results. Navigation is manual `st.tabs`; "Generate" can't advance the tab, so it sets a flag and *tells the user to click the next tab by hand*. Tab numbers in the code/messages are stale from a 7-tab era and send stuck users to the wrong tab.

**Matching (this is the good part — hybrid):**
1. **LLM scores** a compatibility matrix (0–100 per pair), in mentor-batches — OpenAI **Responses API**, default model `gpt-5.1`, strict JSON-schema output, temp 0.1, auto-batch-sized to the model's token limit, 5× rate-limit retries.
2. **Deterministic optimal assignment** via `scipy` **Hungarian algorithm** (`linear_sum_assignment`); the mentees-per-mentor cap (1–3, default 2) is enforced by cloning each mentor into N columns. *This, not the LLM, decides the final pairing.*
3. A **second LLM call** generates human-readable reasoning per match (doesn't affect scores).
4. Post-hoc **conflict check** + a hardcoded **Spain-mentor↔Spain-mentee** block.

**Separate normalization pipeline** (`Normalization/`): raw Excel → cleaned "POST" schema, using regex/lookup tables plus `gpt-4o-mini` for field classification and skill-keyword merging, cached by content hash.

**Infra:** none. No tests, no CI, no `.streamlit/` config, dependencies pinned only with `>=`. `DEBUG_MODE` defaults **ON** (exposes token/cost internals). The dev container runs Streamlit with **CORS and XSRF protection disabled**. Deployment is Codespaces-only (no Streamlit Cloud config).

## 3. What's genuinely good — keep it

- **The hybrid design.** LLM for fuzzy pair scoring + Hungarian for provably-optimal assignment is the right architecture. Peer Mentoring's deterministic balancer is the analogue; here it already exists and is better-grounded. **Keep the Hungarian core.**
- **Structured JSON-schema output, batching, and rate-limit handling** — solid bones.
- **The normalization idea** (messy form data → clean schema) — valuable; needs hardening, not replacing.
- **Conflict reporting + per-match reasoning + the heatmap** — good organizer-facing touches.

## 4. Problems to fix (ranked by impact)

1. **Blocking generation.** Matching runs synchronously on the Streamlit thread with `time.sleep` delays and up to 65s rate-limit waits — UI frozen for minutes, "don't close the browser," and any rerun mid-run is dangerous. *(Peer Mentoring had this exact problem.)*
2. **Silent output truncation → corrupted matches.** The matrix call sets **no output-token cap**; large batches get cut off and missing pairs are **filled with a hardcoded 30%**, which can distort the Hungarian optimum. *(This is the same class of bug as Peer Mentoring's JSON truncation — a real correctness issue.)*
3. **Awkward, mislabeled navigation.** Manual tab handoff + stale tab numbers that misdirect users.
4. **No tests, no CI, no safety net.** Zero coverage on a non-trivial matching pipeline.
5. **Logic coupled to the UI.** `matching.py` / `validation.py` call `st.*` directly — can't be tested or run headless.
6. **Latent crashes.** Stateful-mode arg mismatch (`app.py:110`), retry-path int/string column `KeyError` — both blow up on specific paths.
7. **Half the selectable models can't do the core job** (reasoning models reject structured output); pricing/token-limit tables are incomplete and silently wrong.
8. **~440 lines of dead code** (`ui_training.py`, `ui_processing_modal.py`) + an orphaned `openai_files.py`; **training files are uploaded to OpenAI but never used in matching** — a misleading, cost-incurring dead feature.
9. **Hardcoded Spain rule** — invisible to organizers, fragile string-equality (bypassed by "España"/"ESP"), surprising for an *International* program.
10. **Security/ops:** `DEBUG_MODE` on by default; CORS/XSRF disabled; `>=` deps with no lockfile.
11. **`app.py` is 1,471 lines** with 400-line functions and a UI→app back-import (circular-import risk); a `# will be moved to api_client.py` TODO never done.

## 5. Peer Mentoring lessons → how they map here

| Lesson from Peer Mentoring | Applies? | What to do here |
|---|---|---|
| **Claude-only** (dropped OpenAI, `claude-opus-5-5`) | ⚠️ Decision | Bigger lift here (Responses API + structured output + batching + stateful + normalization all on OpenAI). See Decision A. |
| **Stream + high token budget** to avoid truncation | ✅ Direct | Fixes problem #2. Claude streaming with an ample cap; stop-reason guard. |
| **Async background-thread generation** + locked view + time-based progress bar + Cancel | ✅ Direct | Fixes #1 — the single biggest UX win. |
| **Stepped navigation** (reachability-gated, `st.button` not anchor links to preserve session) | ✅ Direct | Replaces the 6-tab manual handoff; fixes #3. |
| **Review-first data cleanup** (flag issues, exclude anyone) | ✅ Adapt | A review step over the normalized mentees/mentors before matching. |
| **Lock-and-regenerate** (keep good, regen rest) | ✅ Adapt | "Lock" good mentee→mentor pairs, re-run Hungarian on the rest. |
| **Misfit surfacing** | ✅ Already partly there | Lean on reasoning + conflict + low-score flags; surface weak matches. |
| **Expandable result visualization** | ✅ Adapt | Keep the heatmap; add clean per-match cards (already present). |
| **Brand theme + impeccable UI pass + dark tooltips** | ✅ Direct | `.streamlit/config.toml` IMFAHE theme, same as Peer Mentoring. |
| **CI (ruff + pytest) + pytest suite + Dependabot + security-review** | ✅ Direct | Build from zero; decouple logic from `st.*` first so it's testable. |
| **Pin dependencies** | ✅ Direct | Pin exact versions; add a lockfile. |
| **Secrets via Streamlit Cloud + gitignored local `secrets.toml`** | ✅ Direct | Same model; add `APP_PASSWORD` gate properly. |
| **`v1.0` tag as rollback point** | ✅ Direct | Tag current `main` before the revamp. |
| **Guidance/tooltips for non-technical organizers** | ✅ Direct | Especially around the hidden Spain rule + the cap. |

## 6. The big decisions (your call — this is the agenda)

**A. Provider: migrate to Claude, or stay on OpenAI?**
Peer Mentoring went Claude-only. Here it's a bigger lift: the matrix scoring, reasoning, *and* the normalization pipeline all use OpenAI, with Responses-API specifics (structured output, stateful `previous_response_id` chaining, TPM-based batch sizing). Claude can do all of it (tool/JSON structured output, streaming, large context, prompt caching to replace "stateful" mode), and consolidating to one provider matches Peer Mentoring + the job-search tool (one key, one bill). But it's real porting work. *Options: full migrate to Claude · keep OpenAI and just modernize · migrate matching to Claude but leave normalization on gpt-4o-mini for now.*

**B. Scope: targeted restructure, or ground-up rewrite?**
Peer Mentoring was rebuilt clean. Here the Hungarian core + normalization are worth preserving, but `app.py` is a 1,471-line tangle. *Recommendation: targeted restructure* — extract a headless `api_client` / `matching` core (no `st.*`), delete dead code, then layer the UX + infra upgrades — rather than a from-scratch rewrite.

**C. Keep the hybrid Hungarian matching?** *Recommendation: yes.* It's the strongest part. The revamp improves everything around it (speed, truncation, UX, tests), not the core algorithm.

**D. Who uses this and when?** Peer Mentoring had a clear calendar (responses Jan 12, groups Feb 25). I don't know IMP's timeline or whether a revamp is wanted before a specific cohort — this sets urgency and scope.

**E. The Spain rule.** Keep it, make it configurable (e.g. a general "avoid same-country" toggle like Peer Mentoring's same-university rule), or drop it? It's currently hidden and fragile.

## 7. Proposed phased approach (if you approve the direction)

0. **Tag `v1.0`** on current `main` (rollback point) + set up the IMFAHE folder worktree.
1. **Decouple + de-risk:** extract headless `matching`/`api_client` cores from `app.py`, delete dead code (`ui_training`, `ui_processing_modal`, unused training-file path), add a first pytest suite around the Hungarian assignment + parsing. *Fixes the latent crashes and truncation fill.*
2. **Provider** (per Decision A): migrate scoring + reasoning (+ maybe normalization) with streaming, a real token-budget, and a stop-reason guard.
3. **UX:** stepped navigation + async generation (locked view, progress, cancel) + review step + lock-and-regenerate + IMFAHE theme.
4. **Infra:** CI (ruff + pytest), Dependabot, security-review Action, pinned deps, `DEBUG_MODE` off by default, re-enable XSRF for any real deploy, `.streamlit/` config + Streamlit Cloud deployment.
5. **Impeccable pass** + organizer guidance, then ship.

## 8. Open items / data

- The sample files in `samples/` are **PRE-format** (`Code Mentee`, etc.) and need the normalization pass before matching — good, that exercises the full pipeline.
- Header-row handling disagrees between the normalization stage (`header=1`) and the matching stage (a separate sniffer) — needs reconciling.
- No OpenAI key is set locally yet, so I haven't *run* the app end-to-end — behavior above is from code, not a live run. We'll want one live baseline run (on the current code) before changing anything, to compare against.
