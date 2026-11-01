# Matching benchmark

A repeatable regression gate: grade our matching pipeline against the **original tool** on the
same cohort, so a change (a prompt tweak, a normalization change, a different scorer like Jev or
embeddings) can be shown to be **at least as good as the original** before it ships.

## Run

From the repo root, with `ANTHROPIC_API_KEY` in the environment or `.streamlit/secrets.toml`:

```bash
.venv/bin/python benchmark/benchmark_matching.py
```

First run normalizes the cohort and scores our matrix (LLM calls, a few minutes); both are cached,
so reruns are instant. Pass `BENCH_FRESH=1` to recompute.

## What it reports

- **Head-to-head** — each tool's assignments scored by *both* compatibility matrices. Each tool
  prefers its own picks; read the spread, not the raw win.
- **Agreement** — how often both tools pick the same mentor.
- **Field coherence** — exact same-field pairings, ours vs the original.
- **Calibration** — how many matrix cells sit on a hard cap (the original's old rubric pinned many
  pairs at exactly `55` or `0`; ours should be far smoother).
- **Same-field misses** — the regression signal: mentees the original placed with a same-field
  mentor that we did not. Investigate these before claiming parity.

## Swapping the scorer (Claude / Jev / …)

`score_matrix(ee, mo)` is the only expensive, swappable part — it dispatches on `BENCH_SCORER`:

```bash
BENCH_SCORER=claude .venv/bin/python benchmark/benchmark_matching.py   # default: the app engine
BENCH_SCORER=jev    .venv/bin/python benchmark/benchmark_matching.py   # TypeSafe Jev (needs typesafe-sdk + a key)
```

Each scorer returns a mentors×mentees DataFrame of 0–100 scores and is cached separately, so the
grading is identical and the comparison is apples-to-apples. Add a branch to `score_matrix` (and a
`_score_<name>` function) to benchmark another.

**Jev** (`_score_jev`) scores each pair with the Score primitive against `JEV_CRITERIA` (a 0–N rubric,
rescaled to 0–100). It needs `pip install typesafe-sdk` (benchmark-only; not an app dependency) and a
key in `TYPESAFE_API_KEY` env or `JEV_API_KEY`/`TYPESAFE_API_KEY` in `.streamlit/secrets.toml`. Full
86×109 is ~9,400 calls, a few minutes, roughly $0.20. Note: this sends the (normalized, de-identified)
cohort profiles to typesafe.ai.

## Data

The cohort and the original tool's outputs live in the **gitignored** `samples/` dir (real applicant
data — never committed). Override any input with env vars: `MENTEE_SURVEY`, `MENTOR_SURVEY`,
`ORIG_MATRIX`, `ORIG_MATCHES`, `BENCH_MODEL`, `BENCH_CACHE`. The cache defaults to a temp dir outside
the repo; if you point `BENCH_CACHE` inside the repo, note that normalized CSVs contain PII.
