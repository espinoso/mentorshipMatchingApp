"""Regression gate: grade our matching pipeline against the ORIGINAL tool on the same cohort.

Our scoring is the swappable, expensive part (`score_matrix` below — the drop-in point for a Jev
or embedding experiment). Everything else is a neutral referee: it grades BOTH tools' assignments
by BOTH compatibility matrices, then reports calibration, field coherence, and the same-field
matches the original caught that we missed (the "at least as good as the original" signal).

Data stays LOCAL. The real cohort lives in the gitignored samples/ dir; nothing here is committed
but this script and its README. Normalized CSVs and our matrix are cached so reruns are cheap.

Run (from the repo root, with ANTHROPIC_API_KEY in env or .streamlit/secrets.toml):

    .venv/bin/python benchmark/benchmark_matching.py

Override any input via env:
    MENTEE_SURVEY / MENTOR_SURVEY  raw coded surveys (default: the 2025 files in samples/)
    ORIG_MATRIX / ORIG_MATCHES     the original tool's outputs for the SAME cohort (samples/)
    BENCH_MODEL                    matrix model (default: info.DEFAULT_MODEL)
    BENCH_CACHE                    cache dir for normalized CSVs + our matrix (default: a temp dir)
    BENCH_FRESH=1                  ignore the cache and recompute
"""
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "Normalization"))

# API key: env first, then the local Streamlit secrets (never committed).
if not os.environ.get("ANTHROPIC_API_KEY"):
    secrets = os.path.join(REPO, ".streamlit", "secrets.toml")
    if os.path.exists(secrets):
        with open(secrets) as f:
            for line in f:
                if line.strip().startswith("ANTHROPIC_API_KEY"):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

import app  # noqa: E402
import matching_core  # noqa: E402
from info import DEFAULT_MODEL, MATRIX_BATCH_SIZE, MATRIX_MAX_WORKERS, MENTEE_COLUMNS, MENTOR_COLUMNS  # noqa: E402

SAMP = os.path.join(REPO, "samples")
MODEL = os.environ.get("BENCH_MODEL", DEFAULT_MODEL)
CACHE = os.environ.get("BENCH_CACHE") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "mm_benchmark")
FRESH = os.environ.get("BENCH_FRESH") == "1"
os.makedirs(CACHE, exist_ok=True)

MENTEE_SURVEY = os.environ.get("MENTEE_SURVEY", os.path.join(SAMP, "Nov20_86_Mentee_Survey_2025-2_CODED.xlsx"))
MENTOR_SURVEY = os.environ.get("MENTOR_SURVEY", os.path.join(SAMP, "112025_109_Mentor_Survey_2025-3_CODED.xlsx"))
ORIG_MATRIX = os.environ.get("ORIG_MATRIX", os.path.join(SAMP, "compatibility_matrix_20251121_144758.csv"))
ORIG_MATCHES = os.environ.get("ORIG_MATCHES", os.path.join(SAMP, "matches_20251121_150249.xlsx"))

EF, MF = MENTEE_COLUMNS['field'], MENTOR_COLUMNS['field']
EID, MID = MENTEE_COLUMNS['id'], MENTOR_COLUMNS['id']


def normalized_cohort():
    """Normalize the surveys (or load the cache). Returns (mentees_df, mentors_df)."""
    nee, nmo = os.path.join(CACHE, "norm_ee.csv"), os.path.join(CACHE, "norm_mo.csv")
    if not FRESH and os.path.exists(nee) and os.path.exists(nmo):
        print("reloading cached normalized cohort", flush=True)
        return pd.read_csv(nee), pd.read_csv(nmo)
    import merge_mentee_skills_enhanced as mme
    import merge_mentor_skills_enhanced as mmo
    print("normalizing cohort (this calls the LLM; cached afterwards)...", flush=True)
    r1 = mme.process_mentee_file(MENTEE_SURVEY, output_file=os.path.join(CACHE, "ee.xlsx"))
    r2 = mmo.process_mentor_file(MENTOR_SURVEY, output_file=os.path.join(CACHE, "mo.xlsx"))
    ee = r1[0] if isinstance(r1, tuple) else r1
    mo = r2[0] if isinstance(r2, tuple) else r2
    ee.to_csv(nee, index=False)
    mo.to_csv(nmo, index=False)
    return ee, mo


def score_matrix(ee, mo):
    """OUR compatibility matrix. THE SWAP POINT: replace this body to benchmark a different
    scorer (Jev, embeddings, ...). Here it mirrors the app engine: small mentor batches scored
    concurrently with Claude, then assembled."""
    cached = os.path.join(CACHE, f"our_matrix_{MODEL}.csv")
    if not FRESH and os.path.exists(cached):
        print("reloading cached our-matrix", flush=True)
        m = pd.read_csv(cached, index_col=0)
        m.columns = [str(c) for c in m.columns]
        m.index = [str(i) for i in m.index]
        return m

    app._model_choice = lambda default=MODEL: MODEL  # pin the model for this run
    mentor_ids = [str(x) for x in mo[MID].tolist()]
    mentee_ids = [str(x) for x in ee[EID].tolist()]
    batches = [mentor_ids[i:i + MATRIX_BATCH_SIZE] for i in range(0, len(mentor_ids), MATRIX_BATCH_SIZE)]

    def run_batch(bids):
        bdf = mo[mo[MID].astype(str).isin(bids)]
        for _ in range(3):
            try:
                rep = matching_core.parse_matrix_response(app._score_mentor_batch(ee, bdf), bids, mentee_ids)
                if rep.df is not None:
                    return rep.df
            except Exception:
                time.sleep(1)
        return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MATRIX_MAX_WORKERS) as ex:
        frames = [df for df in ex.map(run_batch, batches) if df is not None]
    print(f"  scored {len(frames)}/{len(batches)} batches "
          f"({MATRIX_BATCH_SIZE} mentors each, {MATRIX_MAX_WORKERS} workers) in {time.time()-t0:.0f}s", flush=True)
    if not frames:
        sys.exit("SCORING FAILED: no batch succeeded")
    full = pd.concat(frames)
    cols = [c for c in sorted(mentee_ids, key=lambda x: int(x) if x.isdigit() else 999) if c in full.columns]
    full = full[cols]
    full.to_csv(cached)
    full.columns = [str(c) for c in full.columns]
    full.index = [str(i) for i in full.index]
    return full


def main():
    print(f"cohort from {os.path.basename(MENTEE_SURVEY)} + {os.path.basename(MENTOR_SURVEY)} | model={MODEL}", flush=True)
    ee, mo = normalized_cohort()
    print(f"mentees={len(ee)} mentors={len(mo)}", flush=True)

    our = score_matrix(ee, mo)
    res = matching_core.hungarian_assignment(our, max_mentees_per_mentor=2)
    our_assign = {str(m): str(mt) for m, mt, _ in res.assignments}

    om = pd.read_csv(ORIG_MATRIX, index_col=0)
    om.columns = [str(c) for c in om.columns]
    om.index = [str(i) for i in om.index]
    omatch = pd.read_excel(ORIG_MATCHES)
    orig_assign = {str(r['Mentee_ID']): str(r['Mentor_ID']) for _, r in omatch.iterrows()}

    ee_f = {str(r[EID]): str(r[EF]).strip() for _, r in ee.iterrows()}
    mo_f = {str(r[MID]): str(r[MF]).strip() for _, r in mo.iterrows()}

    def og(mentor, mentee):
        try: return float(om.loc[str(mentor), str(mentee)])
        except Exception: return float('nan')
    def ug(mentor, mentee):
        try: return float(our.loc[str(mentor), str(mentee)])
        except Exception: return float('nan')

    mentees = [m for m in orig_assign if m in our_assign]
    print(f"\ncomparable mentees: {len(mentees)}", flush=True)

    so = [v for v in (og(orig_assign[m], m) for m in mentees) if v == v]
    sn = [v for v in (og(our_assign[m], m) for m in mentees) if v == v]
    to = [v for v in (ug(orig_assign[m], m) for m in mentees) if v == v]
    tn = [v for v in (ug(our_assign[m], m) for m in mentees) if v == v]
    print("\n##### HEAD-TO-HEAD (each tool scores its own picks higher; look at the spread) #####", flush=True)
    print(f"By the ORIGINAL's matrix:  original's picks mean={statistics.mean(so):.1f}  |  OUR picks mean={statistics.mean(sn):.1f}", flush=True)
    print(f"By OUR matrix:             original's picks mean={statistics.mean(to):.1f}  |  OUR picks mean={statistics.mean(tn):.1f}", flush=True)

    agree = sum(1 for m in mentees if orig_assign[m] == our_assign[m])
    print(f"\nsame mentor chosen: {agree}/{len(mentees)} ({100*agree/len(mentees):.0f}%)", flush=True)

    orig_sf = sum(1 for m in mentees if ee_f.get(m) and ee_f[m] == mo_f.get(orig_assign[m]))
    our_sf = sum(1 for m in mentees if ee_f.get(m) and ee_f[m] == mo_f.get(our_assign[m]))
    print(f"exact same-field pairings:  original {orig_sf}/{len(mentees)}  |  ours {our_sf}/{len(mentees)}", flush=True)

    ov = om.to_numpy(dtype=float).flatten()
    uv = our.to_numpy(dtype=float).flatten()
    print(f"\nCALIBRATION (fewer hard caps = finer discrimination):", flush=True)
    print(f"  original matrix:  ==55:{int((ov==55).sum())}  ==0:{int((ov==0).sum())}  of {ov.size}", flush=True)
    print(f"  our matrix:       ==55:{int((uv==55).sum())}  ==0:{int((uv==0).sum())}  of {uv.size}", flush=True)

    # The regression signal: same-field matches the ORIGINAL caught but we did NOT.
    misses = [m for m in mentees
              if ee_f.get(m) and ee_f[m] == mo_f.get(orig_assign[m]) and ee_f[m] != mo_f.get(our_assign[m])]
    print(f"\n##### SAME-FIELD MISSES: original kept field, we didn't ({len(misses)}) #####", flush=True)
    for m in sorted(misses, key=lambda x: int(x) if x.isdigit() else 999):
        o_mt, u_mt = orig_assign[m], our_assign[m]
        print(f"  mentee {m} ({ee_f.get(m,'?')[:18]}): ORIG->m{o_mt}({mo_f.get(o_mt,'?')[:14]}) "
              f"[our={ug(o_mt,m):.0f}] | OURS->m{u_mt}({mo_f.get(u_mt,'?')[:14]}) [our={ug(u_mt,m):.0f}]", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
