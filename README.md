# 🤝 Mentorship Matching System

An AI-powered Streamlit app that matches mentees to mentors for IMFAHE's International Mentor Program (IMP), based on field, specialization, skills, guidance needs, and career goals. It pairs **Claude** (for fuzzy compatibility scoring and per-match reasoning) with a **Hungarian assignment** solver (for a provably optimal set of pairings), so matching is both nuanced and optimal.

## How it works

1. **Normalize** — messy Excel form exports are cleaned into a consistent schema (a Claude Haiku pass classifies fields and merges skills).
2. **Score** — Claude (Opus) scores every mentee × mentor pair 0–100 into a compatibility matrix, batched and streamed with an output-token budget so large cohorts don't truncate.
3. **Assign** — SciPy's Hungarian algorithm (`linear_sum_assignment`) computes the optimal assignment, honoring a configurable mentees-per-mentor cap and an optional same-country restriction.
4. **Explain** — a second Claude call writes a short reasoning for each pairing; results show scores, reasoning, conflicts, and a heatmap, and export to Excel.

## Using the app (guided wizard)

The app walks an organizer through numbered steps; each unlocks once its data exists:

1. **Configure** — confirm the API key, pick the Claude model, set max mentees per mentor, and toggle the Spain ↔ Spain restriction.
2. **Upload** — upload the mentee and mentor Excel files.
3. **Customize Prompt** — optionally edit the matching rubric.
4. **Data Overview** — preview the normalized data.
5. **Generate** — run the match (the step rail locks while it runs; a "← Back to Generate" button is always available if a run can't proceed).
6. **Results** — browse pairings, reasoning, conflicts, and the heatmap; download the Excel report.

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
ANTHROPIC_API_KEY = "sk-ant-..."
APP_ACCESS_PASSWORD = "choose-a-shared-password"   # optional; omit to skip the gate in local dev
```

Then:

```bash
streamlit run app.py
```

## Deploying (Streamlit Community Cloud)

The app is built to deploy on [Streamlit Community Cloud](https://streamlit.io/cloud), same as the sibling Peer Mentoring tool:

1. Point a new app at this repo / `main` / `app.py`.
2. In the app's **Settings → Secrets**, add `ANTHROPIC_API_KEY` and `APP_ACCESS_PASSWORD` (same keys as `secrets.toml`).
3. Deploy. The committed `.streamlit/config.toml` applies the IMFAHE theme and keeps XSRF protection on.

The `.devcontainer/` setup is for local/Codespaces development only (it disables XSRF for the Codespaces preview proxy); it is not used by the Cloud deployment.

## Configuration

| Setting | Where | Default | Notes |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | secrets / env | — | Required. The only LLM credential. |
| `APP_ACCESS_PASSWORD` | secrets / env | unset | Shared password gate; if unset, the app is open (dev). |
| `DEBUG_MODE` | env | `FALSE` | Set `TRUE` to expose token/cost internals and extra diagnostics. **Off in production.** |
| Model | Configure step | `claude-opus-5-5` | Matrix scoring + reasoning. |
| Max mentees / mentor | Configure step | 2 | 1–3. |
| Spain ↔ Spain restriction | Configure step | On | Spanish mentors aren't matched to Spanish mentees. |

## Development

```bash
pip install -r requirements-dev.txt
ruff check .          # lint
python -m pytest -q   # tests
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on every push/PR; Dependabot keeps pip + GitHub Actions deps current. The matching logic is decoupled from Streamlit in `matching_core.py` so it is unit-tested headlessly.

## Security & privacy

- Secrets come from Streamlit secrets / environment variables and are never committed (`.streamlit/secrets.toml` is gitignored).
- Applicant files stay in the running environment; nothing is persisted to external storage beyond the Anthropic API calls needed to score matches.
- A shared-password gate (`APP_ACCESS_PASSWORD`) protects the deployed app.

## Troubleshooting

- **"Prerequisites not met" on Generate** — set the API key (Configure step) and upload both files first; use "← Back to Generate" to return.
- **Missing required columns** — the file is missing a core column (see Data requirements); the normalization pass reports which.
- **A batch failed / incomplete matrix** — usually a transient API issue; re-run. Truncation is surfaced as an honest error, never silently filled.

---

**Built for IMFAHE's International Mentor Program.**
