# 🤝 Mentorship Matching System

An AI-powered Streamlit app that matches mentees to mentors for IMFAHE's International Mentor Program (IMP), based on field, specialization, skills, guidance needs, and career goals. It pairs **Jev** (TypeSafe System One — calibrated compatibility scoring) with a **Hungarian assignment** solver (a provably optimal set of pairings), and uses **Claude** to normalize the data and explain each pairing, so matching is both nuanced and optimal.

## How it works

1. **Normalize** — messy Excel form exports are cleaned into a consistent schema (a Claude Haiku pass classifies fields and merges skills).
2. **Score** — Jev (TypeSafe System One) scores every mentee × mentor pair 0–100 into a compatibility matrix, running the pairs concurrently.
3. **Assign** — SciPy's Hungarian algorithm (`linear_sum_assignment`) computes the optimal assignment, honoring a configurable mentees-per-mentor cap and the same-country restriction (default-on Spain ↔ Spain).
4. **Explain** — Claude (Opus) writes a short reasoning for a pairing on demand (when you open a match); results show scores, reasoning, conflicts, and a heatmap, and export to Excel.

## Using the app

A step rail across the top unlocks each step once its data exists; shared settings live in the sidebar (max mentees per mentor, Spain ↔ Spain restriction). The steps are:

1. **Mentees** — upload the mentee Excel file (checked and normalized).
2. **Mentors** — upload the mentor Excel file.
3. **Generate** — run the match (the rail locks while it runs; a "← Back to Generate" button is always available if a run can't proceed).
4. **Results** — the at-a-glance overview: summary, score distribution, capacity banner, and the heatmap; download the Excel report and the full matrix.
5. **Review** — the per-match list you act on: search/sort, open a pairing for its reasoning, see pairs flagged for review, and swap a mentor (guarded by the matrix scores).

## Data requirements

The mentee/mentor files are the program's standard Excel exports; the normalization pass tolerates blank leading rows and column-name drift. Core columns:

**Mentees:** `Code Mentee`, `Mentee Field of expertise`, `Mentee Specialization`, `Mentee Specific hard skills that you have`, `Mentee Areas where guidance is needed`, `Mentee Career goals for the next 2 years`.

**Mentors:** `Code mentor`, `Field of expertise`, `Mentor Specialization`, `Mentor Specific Hard Skills and Professional Competencies has Mastered`, `Mentor Years of Professional Experience in his her Field`, `Mentor Type of Institution`.

Files are `.xlsx`. Real sample files live outside the repo (gitignored `samples/`), since they contain applicant data.

## Running locally

Requires **Python 3.12+**.

```bash
git clone git@github.com:espinoso/mentorshipMatchingApp.git
cd mentorshipMatchingApp
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Provide secrets via `.streamlit/secrets.toml` (gitignored) **or** environment variables:

```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-..."                   # normalization (Haiku) + per-match reasoning (Opus)
TYPESAFE_API_KEY = "..."                           # Jev compatibility scoring (JEV_API_KEY also accepted)
APP_ACCESS_PASSWORD = "choose-a-shared-password"   # optional; omit to skip the gate in local dev
```

Then:

```bash
streamlit run app.py
```

## Deploying (Streamlit Community Cloud)

The app is built to deploy on [Streamlit Community Cloud](https://streamlit.io/cloud), same as the sibling Peer Mentoring tool:

1. Point a new app at this repo / `main` / `app.py`.
2. In the app's **Settings → Secrets**, add `ANTHROPIC_API_KEY`, `TYPESAFE_API_KEY`, and `APP_ACCESS_PASSWORD` (same keys as `secrets.toml`).
3. Deploy. The committed `.streamlit/config.toml` applies the IMFAHE theme and keeps XSRF protection on.

The `.devcontainer/` setup is for local/Codespaces development only (it disables XSRF for the Codespaces preview proxy); it is not used by the Cloud deployment.

## Configuration

| Setting | Where | Default | Notes |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | secrets / env | — | Required. Normalization (Haiku) + per-match reasoning (Opus). |
| `TYPESAFE_API_KEY` | secrets / env | — | Required. Jev compatibility scoring (`JEV_API_KEY` also accepted). |
| `APP_ACCESS_PASSWORD` | secrets / env | unset | Shared password gate; if unset and `ALLOW_NO_AUTH` is set, the app is open (dev). |
| `DEBUG_MODE` | env | `FALSE` | Set `TRUE` to expose token/cost internals and extra diagnostics. **Off in production.** |
| Model | fixed in code | `claude-opus-5-5` | Reasoning + normalization; no user-facing selector. |
| Max mentees / mentor | sidebar | 2 | 1–3. |
| Spain ↔ Spain restriction | sidebar | On | Spanish mentors aren't matched to Spanish mentees (enforced before assignment). |

## Development

```bash
pip install -r requirements-dev.txt
ruff check .          # lint
python -m pytest -q   # tests
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on every push/PR; Dependabot keeps pip + GitHub Actions deps current. The matching logic is decoupled from Streamlit in `matching_core.py` so it is unit-tested headlessly.

## Security & privacy

- Secrets come from Streamlit secrets / environment variables and are never committed (`.streamlit/secrets.toml` is gitignored).
- Applicant files stay in the running environment; nothing is persisted to external storage beyond the Jev (scoring) and Anthropic (normalization + reasoning) API calls needed to match.
- A shared-password gate (`APP_ACCESS_PASSWORD`) protects the deployed app.

## Troubleshooting

- **"Prerequisites not met" on Generate** — set the API keys (secrets / env) and upload both files first; use "← Back to Generate" to return.
- **Missing required columns** — the file is missing a core column (see Data requirements); the normalization pass reports which.
- **"N pairs couldn't be scored"** — a transient Jev issue left some pairs unscored (treated as 0); the app warns and you can re-run for a clean matrix. If Jev scores nothing at all, it fails with an honest error rather than matching on zeros.

---

**Built for IMFAHE's International Mentor Program.**
