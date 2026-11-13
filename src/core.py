# ===== NB CELL 2 =====
import os, re, sys, json, math, itertools, subprocess, textwrap, getpass, random

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from collections import Counter, defaultdict

import numpy as np

import pandas as pd

try:
    from scipy.stats import wilcoxon, wasserstein_distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

LOGS = {
    "bpic12": dict(
        path="data/raw/b12.csv",
        rename={},
        # "Event" is the original event index -> gives away the true order
        leak=["Event"],
        ts_format="%Y-%m-%d %H:%M:%S.%f",
        note="BPIC 2012 loan application. Real log, millisecond precision."),

    "bpic15": dict(
        path="data/raw/BPIC_cleaned.csv",
        rename={},
        leak=[],
        ts_format="%Y-%m-%d %H:%M:%S",
        note="BPIC 2015 building permit. 46% of timestamps are date-only."),

    "credit": dict(
        path="data/raw/credit-card-new.csv",
        rename={"Case ID": "Case", "Complete Timestamp": "Timestamp"},
        # Start Timestamp of event i equals Complete Timestamp of event i-1 in
        # this simulator, so leaving it in hands over the entire true timeline.
        leak=["Start Timestamp", "Variant", "Variant index", "elementId",
              "processId", "resourceId", "resourceCost"],
        ts_format="%Y-%m-%d %H:%M:%S.%f",
        note="Simulated credit-card application process."),

    "pub": dict(
        path="data/raw/pub-new.csv",
        rename={"Case ID": "Case", "Complete Timestamp": "Timestamp"},
        leak=["Start Timestamp", "Variant", "Variant index", "elementId",
              "processId", "resourceId", "resourceCost"],
        ts_format="%Y-%m-%d %H:%M:%S.%f",
        note="Simulated restaurant service process."),

    "hospital": dict(
        path="data/raw/hospital_billing.csv",
        rename={},
        leak=["Event"],
        ts_format="%Y-%m-%d %H:%M:%S",
        note="Hospital billing. A REAL ERP log, not a simulation -- do not "
             "group it with credit/pub when describing the setup."),
}

ACTIVE = "bpic12"

RAW_CSV = PREFIX = TS_FORMAT = LEAK_COLS = RENAME = None

def use_log(name):
    """Point the whole notebook at one log. Cells 4-23 read these globals."""
    global ACTIVE, RAW_CSV, PREFIX, TS_FORMAT, LEAK_COLS, RENAME
    spec = LOGS[name]
    ACTIVE, PREFIX = name, name
    RAW_CSV   = spec["path"]
    TS_FORMAT = spec["ts_format"]
    LEAK_COLS = list(spec["leak"]) + ["_orig_row"]
    RENAME    = dict(spec["rename"])
    CFG["TS_FORMAT"] = TS_FORMAT
    ROOT_FOR = ROOT / name
    for d in ("datasets", "runs", "eval"):
        (ROOT_FOR / d).mkdir(parents=True, exist_ok=True)
    globals().update(DATA_DIR=ROOT_FOR / "datasets",
                     RUN_DIR=ROOT_FOR / "runs",
                     EVAL_DIR=ROOT_FOR / "eval")
    _SITE_CACHE.clear()
    for _c in ("_PROGRAM_CACHE", "_CTX_CACHE", "_PM4PY_CACHE",
               "_SYNTH_BUDGET"):
        if _c in globals():
            globals()[_c].clear()
    print("active log: %s  (%s)" % (name, spec["note"]))
    return spec

ROOT      = Path("/content/ts_repair")

DATA_DIR  = ROOT / "datasets"

RUN_DIR   = ROOT / "runs"

EVAL_DIR  = ROOT / "eval"

for d in (DATA_DIR, RUN_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

CASE_COL, ACT_COL, TIME_COL, RES_COL = "Case", "Activity", "Timestamp", "Resource"

GT_COL       = "Timestamp_original"

POLLUTED_COL = "_is_polluted"

ERR_COL      = "error_category"

LABEL_COL    = "label"

ROW_COL      = "row_id"

CLEAN = "CLEAN"

C1 = "C1_ORDERING_AMBIGUITY"

C2 = "C2_TEMPORAL_INCONSISTENCY"

C3 = "C3_MISSING_TEMPORAL_INFORMATION"

CATEGORIES = [C1, C2, C3]

CFG = dict(
    # --- Part 1: stratification -------------------------------------------
    COARSE_RULE="hour_floor",      # minute==0 and second==0 -> date-only field
    ACT_COARSE_THRESHOLD=0.80,
    MIN_CLEAN_RUN=2,
    MIN_SITE_SPAN_SEC=60,          # a site must span real time to be recoverable

    # --- Part 2: injection --------------------------------------------------
    C1_BLOCK_LENS=(2, 3, 4),
    C1_COPY_MODE="first",          # collapse everything onto the first stamp
    C1_SHUFFLE_ROWS=True,          # otherwise CSV row order leaks the true order
    # "swap" exchanges two adjacent timestamps and is EXACTLY invertible by
    # sorting: on hospital, M1, B1 and B3 all scored a median error of 0 s on
    # it, which measures nothing. "shift" pushes the later event back by a
    # random offset and cannot be undone by sorting.
    C2_MODE="shift",
    C2_ALSO_SWAP=True,             # also emit C2SWAP_* datasets for continuity
                                   # with the predecessor study's numbers
    C2_BLOCK_LENS=(2,),
    C3_MODES=("missing_timestamp", "missing_granularity"),
    MAX_ERRORS_PER_CASE=1,
    SEED=42,

    # --- Part 4: methods ----------------------------------------------------
    LLM_MODEL="gpt-4o",  # default only; run_llm.py --model overrides it
    #              and the model that ran is recorded per row in
    #              results/llm_per_synthesis.csv
    LLM_MAX_ATTEMPTS=2,   # retries inside one candidate; 3 was
                      # multiplicative on top of N candidates
    DIRECT_CHUNK_SIZE=100,  # safe at this size because a chunk that
                        # fails to parse is split and retried
    FALLBACK_DURATION=60.0,
    MIN_DURATION=1.0,
    EXHAUSTIVE_MAX=5,
    BEAM_WIDTH=5,

    # --- Part 6: evaluation --------------------------------------------------
    C1_DURATION_STAT="median",   # "median" -> lowest MAE; "p10" -> lowest regression rate
    TWO_STAGE=True,              # model plans in JSON first, then writes the code
    REFINE_ROUNDS=1,             # one plausibility-feedback revision round
    TOLERANCES=[1, 10, 60, 600, 3600, 86400],
    N_BOOTSTRAP=300,
    NOOP_TOLERANCE=0.99,     # >= this share of polluted rows unchanged = a no-op
    CODEGEN_NOOP_LIMIT=0.50, # codegen regenerates if it changed less than half
    # --- LLM strategy (v5) --------------------------------------------------
    CODEGEN_N_CANDIDATES=3,  # generate N programs, keep the most process-consistent
    C1_ANCHOR_ON_OBSERVED=True,   # the collapsed value is the block's FIRST event
    C1_ANCHOR_MIN_SHARE=0.80,     # share of blocks that must honour it
    C1_DISTINCT_MIN_SHARE=0.95,   # share of blocks that must end up separated
    DIRECT_MAX_ATTEMPTS=2,
    # --- cost and wall-clock (v6) -------------------------------------------
    CODEGEN_CACHE_SCOPE="log",    # "log": synthesize once per log and reuse it
                                  # across that log's datasets, which is what
                                  # the predecessor setup describes. "dataset":
                                  # regenerate per dataset (v5 behaviour, ~9x
                                  # the calls). "off": no reuse at all.
    CODEGEN_SEED_DATASET="MIX_30",# generate on the dataset that contains all
                                  # three categories, so every branch of the
                                  # program is exercised before it is reused
    CODEGEN_PARALLEL=True,        # candidates are network-bound; run them at once
    CODEGEN_SELECT_CASES=300,     # candidates are scored on a sample of this
                                  # many polluted cases; only the winner runs
                                  # on the full log
    DIRECT_MAX_WORKERS=4,         # parallel M4 chunks
    DIRECT_MIN_CHUNK=5,           # a failing chunk halves down to this size
    SEG_STATS_MAX=6,
    # A failed synthesis used to cost full price on every dataset of the log,
    # because the cache stored successes only. Now failures are cached and
    # synthesis is budgeted per (log, method).
    CODEGEN_SYNTH_BUDGET=2,
    # G4: share of C1 blocks a program may leave in file order. A correct
    # inference coincides with the row order about a quarter of the time for
    # blocks of 2-4 (1/k! averaged over k) and M1 measures 0.32, so 0.80 fires
    # only on a program that is not ordering at all.
    C1_ROWORDER_MAX_SHARE=0.80,
    C1_ROWORDER_MIN_BLOCKS=20,
    TS_FORMAT=None,          # set by use_log()
)

RNG_SEED = CFG["SEED"]

RUN_LLM = False

RUN_BASELINES = True

RUN_HEAVY_BASELINES = False

_SITE_CACHE = {}

import pathlib as _pathlib

import os as _os

import sys as _sys

import subprocess as _subprocess

if not getattr(_pathlib.Path, "_utf8_patched", False):
    _ORIG_WRITE_TEXT = _pathlib.Path.write_text
    _ORIG_READ_TEXT = _pathlib.Path.read_text

    def _utf8_write_text(self, data, encoding=None, errors=None, **kw):
        return _ORIG_WRITE_TEXT(self, data,
                                encoding=encoding or "utf-8",
                                errors=errors or "replace", **kw)

    def _utf8_read_text(self, encoding=None, errors=None, **kw):
        return _ORIG_READ_TEXT(self, encoding=encoding or "utf-8",
                               errors=errors or "replace", **kw)

    _pathlib.Path.write_text = _utf8_write_text
    _pathlib.Path.read_text = _utf8_read_text
    _pathlib.Path._utf8_patched = True

CHILD_ENV = dict(_os.environ)

CHILD_ENV["PYTHONUTF8"] = "1"

CHILD_ENV["PYTHONIOENCODING"] = "utf-8"

# ===== NB CELL 4 =====
def load_raw(path, cfg):
    df = pd.read_csv(path)
    if RENAME:
        df = df.rename(columns=RENAME)
    missing = [c for c in (CASE_COL, ACT_COL, TIME_COL) if c not in df.columns]
    if missing:
        raise ValueError("columns missing after rename: %s (have %s)"
                         % (missing, list(df.columns)))
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df["_orig_row"] = np.arange(len(df))
    df = (df.sort_values([CASE_COL, TIME_COL, "_orig_row"], kind="mergesort")
            .reset_index(drop=True))

    # some logs have no Resource column; keep the schema uniform so the same
    # prompts and the same baseline run unchanged across datasets
    if RES_COL not in df.columns:
        df[RES_COL] = "NA"
        print("note: no %s column in this log -- filled with 'NA'" % RES_COL)
    else:
        nulls = df[RES_COL].isna().mean()
        if nulls:
            print("note: %.1f%% of %s values are empty -- filled with 'NA'"
                  % (100 * nulls, RES_COL))
            df[RES_COL] = df[RES_COL].fillna("NA")

    dropped = [c for c in LEAK_COLS if c in df.columns]
    if dropped:
        print("dropped leak columns:", dropped)
    return df.drop(columns=dropped)

def stratify(df, cfg):
    """Assign each row to CLEAN / NAT_C1 / COARSE_C4 / MISSING_C3 and mark
    which rows may host an injected error."""
    ts = df[TIME_COL]
    df["is_missing_ts"] = ts.isna()

    if cfg["COARSE_RULE"] == "midnight":
        coarse = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
    else:
        coarse = (ts.dt.minute == 0) & (ts.dt.second == 0)
    df["is_coarse_ts"] = coarse.fillna(False)

    df["gap_prev_sec"] = df.groupby(CASE_COL)[TIME_COL].diff().dt.total_seconds()
    df["is_time_travel"] = df["gap_prev_sec"] < 0

    new_block = (df[CASE_COL] != df[CASE_COL].shift()) | (df[TIME_COL] != df[TIME_COL].shift())
    df["block_id"] = new_block.cumsum()
    bsize = df.groupby("block_id")[TIME_COL].transform("size")
    df["block_size"] = bsize
    df["in_dup_block"] = (bsize >= 2) & (~df["is_missing_ts"])

    df["stratum"] = np.where(df["is_missing_ts"], "MISSING_C3",
                     np.where(df["is_coarse_ts"], "COARSE_C4",
                      np.where(df["in_dup_block"], "NAT_C1", "CLEAN")))

    is_clean = df["stratum"].eq("CLEAN")
    run_break = ((~is_clean) | (df[CASE_COL] != df[CASE_COL].shift())
                 | (~is_clean.shift(fill_value=False)))
    df["clean_run_id"] = np.where(is_clean, run_break.cumsum(), -1)
    df["clean_run_len"] = -1
    df.loc[is_clean, "clean_run_len"] = (
        df.loc[is_clean].groupby("clean_run_id")[TIME_COL].transform("size"))

    df["is_trace_first"] = df.index.isin(df.groupby(CASE_COL).head(1).index)
    df["is_trace_last"]  = df.index.isin(df.groupby(CASE_COL).tail(1).index)

    # An injection site needs a neighbour on both sides inside the same case.
    # Trim the first and last event of each trace rather than discarding a whole
    # clean run that happens to touch a trace boundary -- in short-trace logs
    # (hospital: 5 events per case, 99.6% clean) the run IS the trace, so
    # discarding it would leave no sites at all.
    df["is_injection_site"] = (is_clean
                               & ~df["is_trace_first"]
                               & ~df["is_trace_last"]
                               & (df["clean_run_len"] >= cfg["MIN_CLEAN_RUN"]))
    return df

def block_table(df):
    rows = []
    for bid, sub in df[df["block_size"] >= 2].groupby("block_id"):
        rows.append(dict(block_id=bid, case=sub[CASE_COL].iloc[0],
                         timestamp=sub[TIME_COL].iloc[0], size=len(sub),
                         all_distinct=sub[ACT_COL].nunique() == len(sub),
                         single_resource=sub[RES_COL].nunique() == 1,
                         kind="COARSE_C4" if sub["is_coarse_ts"].iloc[0] else "NAT_C1"))
    return pd.DataFrame(rows)

def activity_profile(df, cfg):
    g = df.groupby(ACT_COL).agg(
        n=("is_coarse_ts", "size"),
        coarse_rate=("is_coarse_ts", "mean"),
        dup_rate=("in_dup_block", "mean"),
        clean_rate=("stratum", lambda s: (s == "CLEAN").mean()))
    g["is_date_only_field"] = g["coarse_rate"] >= cfg["ACT_COARSE_THRESHOLD"]
    return g.sort_values("coarse_rate", ascending=False)

# ===== NB CELL 6 =====
def _q(s, qs=(0, .1, .25, .5, .75, .9, .95, .99, 1)):
    s = pd.Series(s).dropna()
    return {"p%d" % int(x * 100): float(s.quantile(x)) for x in qs}

def build_report(df, blocks, actprof, cfg, src=""):
    n = len(df); L = []; add = L.append
    add("# Temporal Diagnostic Report\n")
    add("- source: `%s`" % src)
    add("- rows: **{:,}** | cases: **{:,}** | activities: **{:,}** | resources: **{:,}**"
        .format(n, df[CASE_COL].nunique(), df[ACT_COL].nunique(), df[RES_COL].nunique()))
    add("- span: %s -> %s\n" % (df[TIME_COL].min(), df[TIME_COL].max()))

    role = {"CLEAN": "injection site + statistics population (MAE valid here)",
            "NAT_C1": "naturally collapsed, no ground truth -> plausibility track",
            "COARSE_C4": "date-only granularity -> interval track, excluded from MAE",
            "MISSING_C3": "absent timestamp -> imputation track"}
    add("## 1. Strata\n"); add("| stratum | rows | share | role |"); add("|---|---:|---:|---|")
    st = df["stratum"].value_counts().reindex(list(role)).fillna(0).astype(int)
    for k, v in st.items():
        add("| %s | %s | %.1f%% | %s |" % (k, format(v, ","), 100 * v / n, role[k]))
    ns = int(df["is_injection_site"].sum())
    add("\nInjection-eligible rows: **{:,}** ({:.1%})\n".format(ns, ns / n))

    add("## 2. Granularity\n")
    add("- minute==0 and second==0: **%.1f%%**" % (100 * df["is_coarse_ts"].mean()))
    for k, v in df[TIME_COL].dt.strftime("%H:%M:%S").value_counts().head(4).items():
        add("  - `%s` : %s (%.1f%%)" % (k, format(v, ","), 100 * v / n))
    add("- date-only activities (>= %.0f%% coarse): **%d** of %d\n"
        % (100 * cfg["ACT_COARSE_THRESHOLD"],
           int(actprof["is_date_only_field"].sum()), len(actprof)))

    add("## 3. Inter-event gaps (s)\n")
    g = df["gap_prev_sec"].dropna(); qq = _q(g)
    add("| " + " | ".join(qq) + " |"); add("|" + "---:|" * len(qq))
    add("| " + " | ".join(format(int(v), ",") for v in qq.values()) + " |")
    add("\n- gap == 0 : **%.1f%%**" % (100 * (g == 0).mean()))
    add("- gap < 0 (natural C2): **%d**\n" % int((g < 0).sum()))

    add("## 4. Duplicate-timestamp blocks\n")
    add("| kind | blocks | rows | mean size | max |"); add("|---|---:|---:|---:|---:|")
    # a log with no duplicate-timestamp block at all yields an empty frame
    for k, sub in (blocks.groupby("kind") if len(blocks) else []):
        add("| %s | %s | %s | %.1f | %d |" % (k, format(len(sub), ","),
            format(int(sub["size"].sum()), ","), sub["size"].mean(), int(sub["size"].max())))
    add("")

    add("## 5. Design warnings\n")
    w = []
    if df["stratum"].eq("CLEAN").mean() < .5:
        w.append("Only %.1f%% of rows are clean at second granularity -- estimate all "
                 "statistics from the CLEAN stratum, never the full log."
                 % (100 * df["stratum"].eq("CLEAN").mean()))
    if df["is_coarse_ts"].mean() > .2:
        w.append("%.1f%% of timestamps are date-only. Second-level MAE against them is "
                 "meaningless -- use the interval track." % (100 * df["is_coarse_ts"].mean()))
    if (df["gap_prev_sec"] < 0).sum() == 0:
        w.append("No natural C2 in this log; C2 results are purely synthetic.")
    if df["in_dup_block"].mean() > .3:
        w.append("%.1f%% of rows are already collapsed before injection -- this log is "
                 "not a clean reference." % (100 * df["in_dup_block"].mean()))
    for x in (w or ["none"]):
        add("- " + x)
    return "\n".join(L)

# ===== NB CELL 10 =====
def build_base(raw):
    keep = [CASE_COL, ACT_COL, TIME_COL, RES_COL, "stratum", "is_injection_site"]
    base = raw[keep].copy().reset_index(drop=True)
    base[ROW_COL] = np.arange(len(base))
    base[GT_COL] = base[TIME_COL]
    base = base.rename(columns={"is_injection_site": "eligible"})
    return base

# ===== NB CELL 12 =====
def candidate_sites(base, cfg, block_len, min_span=None, rng=None):
    """All eligible windows of `block_len`, one per case. Cached: the sites
    depend only on the base log, and generate_all reuses them seven times."""
    rng = rng or np.random.default_rng(cfg["SEED"])
    min_span = cfg["MIN_SITE_SPAN_SEC"] if min_span is None else min_span
    key = (block_len, min_span, len(base))
    if key in _SITE_CACHE:
        return list(_SITE_CACHE[key])
    out = []
    elig = base[base["eligible"]]
    for case, sub in elig.groupby(CASE_COL, sort=False):
        idx = sub.index.to_numpy()
        for chunk in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(chunk) < block_len:
                continue
            cands = []
            for s in range(len(chunk) - block_len + 1):
                w = chunk[s:s + block_len]
                li, ri = w[0] - 1, w[-1] + 1
                # both anchors must exist, be in the same case, and be clean
                if li not in base.index or ri not in base.index:
                    continue
                if (base.loc[li, CASE_COL] != case) or (base.loc[ri, CASE_COL] != case):
                    continue
                if (base.loc[li, "stratum"] != "CLEAN") or (base.loc[ri, "stratum"] != "CLEAN"):
                    continue
                span = (base.loc[w[-1], GT_COL] - base.loc[w[0], GT_COL]).total_seconds()
                if span >= min_span:
                    cands.append(w)
            if cands:
                out.append((case, cands[int(rng.integers(len(cands)))]))
    _SITE_CACHE[key] = list(out)
    return out

# ===== NB CELL 14 =====
def _fmt(ts):
    return "NaT" if pd.isna(ts) else pd.to_datetime(ts).strftime(CFG["TS_FORMAT"])

def _label(cat, orig):
    return "%s(orig:%s)" % (cat, _fmt(orig))

def inject_c1(df, w, rng, cfg):
    """Collapse a block onto one timestamp; optionally shuffle its content."""
    w = np.asarray(w)
    if cfg["C1_COPY_MODE"] == "last":
        target = df.loc[w, TIME_COL].max()
    elif cfg["C1_COPY_MODE"] == "random":
        target = df.loc[w[int(rng.integers(len(w)))], TIME_COL]
    else:
        target = df.loc[w, TIME_COL].min()

    if cfg["C1_SHUFFLE_ROWS"] and len(w) > 1:
        perm = rng.permutation(len(w))
        # ground truth follows the event content, not the row position
        move = [ACT_COL, RES_COL, GT_COL]
        df.loc[w, move] = df.loc[w, move].to_numpy()[perm]

    df.loc[w, TIME_COL] = target
    return w

def inject_c2(df, w, rng, cfg):
    """swap  : exchange two adjacent timestamps -> local time travel
    shift : push the later event back before its predecessor by a random amount

    NOTE: `swap` is exactly invertible by sorting, so any method that simply
    re-sorts a C2 segment recovers the truth perfectly. Use `shift` when you
    need C2 to be a non-trivial estimation problem."""
    w = np.asarray(w)
    i = int(rng.integers(len(w) - 1))
    a, b = w[i], w[i + 1]
    if cfg["C2_MODE"] == "shift":
        # round to whole seconds: a float-seconds Timedelta carries nanosecond
        # precision and assigning it into a microsecond-dtype column raises
        # "Cannot losslessly convert units". This path had never run before v4
        # because swap was the default, so the bug shipped unnoticed.
        delta = pd.Timedelta(seconds=round(float(abs(rng.normal(3600, 1800)))))
        df.loc[b, TIME_COL] = (pd.to_datetime(df.loc[a, TIME_COL]) - delta).round("ms")
        # only `b` was corrupted. Labelling `a` as polluted too -- which is
        # correct for swap and wrong for shift -- handed every method a free
        # row whose observed error is already zero, and pinned the regression
        # rate at exactly 0.5 for anything that touched the pair.
        return np.array([b])
    else:
        ta, tb = df.loc[a, TIME_COL], df.loc[b, TIME_COL]
        df.loc[a, TIME_COL], df.loc[b, TIME_COL] = tb, ta
    return np.array([a, b])

def inject_c3(df, w, rng, cfg):
    """Blank the timestamp or drop its time-of-day component."""
    w = np.asarray(w)
    i = w[int(rng.integers(len(w)))]
    mode = cfg["C3_MODES"][int(rng.integers(len(cfg["C3_MODES"])))]
    if mode == "missing_timestamp":
        df.loc[i, TIME_COL] = pd.NaT
    else:                                   # missing_granularity
        df.loc[i, TIME_COL] = pd.to_datetime(df.loc[i, TIME_COL]).normalize()
    return np.array([i]), mode

INJECTORS = {C1: inject_c1, C2: inject_c2, C3: inject_c3}

BLOCK_LENS = {C1: "C1_BLOCK_LENS", C2: "C2_BLOCK_LENS", C3: (1,)}

# ===== NB CELL 16 =====
def inject_dataset(base, cfg, spec, seed=None, overrides=None):
    """spec: {C1: ratio, C2: ratio, C3: ratio}; ratios are per eligible case.

    Site selection is NESTED. v3 shuffled the candidate pool with a run-level
    RNG whose state depended on whether `candidate_sites` had already filled its
    cache, so the first dataset generated consumed randomness that later ones
    did not. The 10% and 30% datasets therefore injected into different sites,
    and their `MedAE_obs` differed by a factor of three on hospital — the two
    rates were never comparable, and any sentence of the form "the error falls
    as the injection rate rises" was an artefact of that.

    Now: one RNG per error category, seeded from SEED and the category alone.
    The chosen set is a prefix of a fixed permutation, so the 10% cases are a
    strict subset of the 30% cases and the difference between the two columns is
    the injection rate and nothing else.
    """
    if overrides:
        cfg = dict(cfg, **overrides)
    rng = np.random.default_rng(cfg["SEED"] if seed is None else seed)
    df = base.copy()
    df[ERR_COL] = CLEAN
    df[LABEL_COL] = ""
    df[POLLUTED_COL] = False
    df["_c3_mode"] = ""

    taken_cases = set()
    meta = {}

    for cat in CATEGORIES:
        ratio = spec.get(cat, 0.0)
        if ratio <= 0:
            meta[cat] = dict(ratio=0.0, n_cases=0, n_rows=0)
            continue

        lens = (cfg[BLOCK_LENS[cat]] if isinstance(BLOCK_LENS[cat], str)
                else BLOCK_LENS[cat])
        span = 0 if cat == C3 else cfg["MIN_SITE_SPAN_SEC"]

        # site enumeration gets its own RNG per block length, so a warm cache
        # and a cold cache produce the same sites
        pool = []
        for L in lens:
            srng_L = np.random.default_rng([cfg["SEED"], 977, int(L)])
            for case, w in candidate_sites(df, cfg, L, min_span=span, rng=srng_L):
                if case not in taken_cases:
                    pool.append((case, w))

        # one deterministic permutation per category, independent of everything
        # else in the run -> nested prefixes across ratios
        crng = np.random.default_rng([cfg["SEED"], 613, CATEGORIES.index(cat)])
        pool = [pool[k] for k in crng.permutation(len(pool))]

        seen, uniq = set(), []
        for case, w in pool:
            if case in seen:
                continue
            seen.add(case); uniq.append((case, w))

        n = int(round(len(uniq) * ratio))
        chosen = uniq[:n]

        rows_hit = 0
        for case, w in chosen:
            if cat == C3:
                touched, mode = inject_c3(df, w, rng, cfg)
                df.loc[touched, "_c3_mode"] = mode
            else:
                touched = INJECTORS[cat](df, w, rng, cfg)
            df.loc[touched, ERR_COL] = cat
            df.loc[touched, POLLUTED_COL] = True
            df.loc[touched, LABEL_COL] = [
                _label(cat, df.loc[r, GT_COL]) for r in touched]
            taken_cases.add(case)
            rows_hit += len(touched)

        meta[cat] = dict(ratio=ratio, n_cases=len(chosen), n_rows=rows_hit,
                         mode=cfg["C2_MODE"] if cat == C2 else
                              (cfg["C1_COPY_MODE"] if cat == C1 else "mixed"))

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df, meta

def check_nesting(base, cfg, cat=C1):
    """Verifies the property F1 depends on: the 10% polluted cases must be a
    subset of the 30% ones. Run it once per log; it is cheap and it is the only
    thing standing between the paper and a rate comparison that is not one."""
    a, _ = inject_dataset(base, cfg, {cat: 0.10})
    b, _ = inject_dataset(base, cfg, {cat: 0.30})
    ca = set(a.loc[a[ERR_COL].eq(cat), CASE_COL])
    cb = set(b.loc[b[ERR_COL].eq(cat), CASE_COL])
    ok = ca <= cb
    print("%s: 10%% cases=%d, 30%% cases=%d, nested=%s"
          % (cat, len(ca), len(cb), ok))
    if not ok:
        print("  !! %d cases in the 10%% set are absent from the 30%% set"
              % len(ca - cb))
    return ok

def generate_all(base, cfg, out_dir=DATA_DIR, prefix=PREFIX):
    """Seven datasets, plus two more when C2_ALSO_SWAP is on.

    C2SWAP_* exists so the predecessor study's C2 numbers stay reproducible while the
    headline C2 column measures something a sort cannot solve. Report them as
    two separate rows, never merged.
    """
    specs = {}
    for cat, short in [(C1, "C1"), (C2, "C2"), (C3, "C3")]:
        for r in (0.10, 0.30):
            specs["%s_%s_%d" % (prefix, short, int(r * 100))] = ({cat: r}, None)
    specs["%s_MIX_30" % prefix] = ({C1: 0.30, C2: 0.30, C3: 0.30}, None)
    if cfg.get("C2_ALSO_SWAP") and cfg["C2_MODE"] != "swap":
        for r in (0.10, 0.30):
            specs["%s_C2SWAP_%d" % (prefix, int(r * 100))] = (
                {C2: r}, {"C2_MODE": "swap"})

    index = {}
    for name, (spec, over) in specs.items():
        df, meta = inject_dataset(base, cfg, spec, overrides=over)
        path = Path(out_dir) / (name + ".csv")
        cols = [CASE_COL, ACT_COL, TIME_COL, RES_COL, ROW_COL,
                GT_COL, POLLUTED_COL, ERR_COL, LABEL_COL, "_c3_mode",
                "stratum", "eligible"]
        df[cols].to_csv(path, index=False)
        index[name] = dict(path=str(path), spec={k: v for k, v in spec.items()},
                           overrides=over or {}, meta=meta, n_rows=len(df),
                           counts=df[ERR_COL].value_counts().to_dict())
        print("saved %-18s  %s" % (name, {k.split("_")[0]: v["n_rows"]
                                          for k, v in meta.items() if v["n_rows"]}))
    (Path(out_dir) / "index.json").write_text(json.dumps(index, indent=2, default=str))
    return index

# ===== NB CELL 18 =====
def verify_injection(path):
    df = pd.read_csv(path)
    obs = pd.to_datetime(df[TIME_COL], errors="coerce")
    gt  = pd.to_datetime(df[GT_COL],  errors="coerce")
    err = (obs - gt).dt.total_seconds().abs()
    pol = df[ERR_COL].ne(CLEAN)
    row = dict(dataset=Path(path).stem, n_rows=len(df), n_polluted=int(pol.sum()),
               polluted_rate=round(pol.mean(), 4),
               n_unobservable=int((pol & obs.isna()).sum()))
    for cat in CATEGORIES:
        m = df[ERR_COL].eq(cat) & err.notna()
        row["%s_n" % cat[:2]] = int(df[ERR_COL].eq(cat).sum())
        row["%s_MedAE_obs" % cat[:2]] = round(err[m].median(), 1) if m.any() else None
        row["%s_MAE_obs" % cat[:2]] = round(err[m].mean(), 1) if m.any() else None
    # sanity: clean rows must be untouched
    cl = ~pol
    row["clean_intact"] = bool((obs[cl].fillna(pd.Timestamp(0))
                                == gt[cl].fillna(pd.Timestamp(0))).all())
    return row

# ===== NB CELL 20 =====
def build_context(df, cfg):
    """Adds three things the old context lacked:

    short-mode durations   BPIC12 gaps are bimodal (p25 1.2 s, p50 48 s,
                           p90 79,043 s). A per-pair median mixes an automated
                           mode with a waiting mode; p10/p25 isolate the fast one.
    block-span profiles    how long a run of k consecutive events actually lasts.
                           Pairwise durations never told a method how wide a
                           3-event burst should be.
    burst share            for each pair, the fraction of observations under a
                           minute. A method can use this to decide whether a
                           transition belongs to one working session.
    """
    d = df[df[ERR_COL].eq(CLEAN)].copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d[d[TIME_COL].notna()].sort_values([CASE_COL, TIME_COL])
    d["prev_act"] = d.groupby(CASE_COL)[ACT_COL].shift()
    d["gap"] = d.groupby(CASE_COL)[TIME_COL].diff().dt.total_seconds()
    p = d.dropna(subset=["prev_act", "gap"])
    p = p[p["gap"] >= 0]

    dur = p.groupby(["prev_act", ACT_COL])["gap"]
    prof = dur.agg(n="count", median="median",
                   p10=lambda s: s.quantile(.10),
                   p25=lambda s: s.quantile(.25),
                   p75=lambda s: s.quantile(.75),
                   p90=lambda s: s.quantile(.90),
                   burst_share=lambda s: float((s <= 60).mean()))
    cnt = p.groupby(["prev_act", ACT_COL]).size()
    prob = cnt / cnt.groupby(level=0).sum()

    # empirical span of k consecutive events, from clean rows only
    t = d.groupby(CASE_COL)[TIME_COL]
    span = {}
    for k in range(2, 6):
        sp = (t.shift(-(k - 1)) - t.shift(0)).dt.total_seconds().dropna()
        sp = sp[sp >= 0]
        if len(sp):
            span[k] = dict(n=int(len(sp)), p10=float(sp.quantile(.10)),
                           p25=float(sp.quantile(.25)), median=float(sp.median()),
                           p75=float(sp.quantile(.75)), p90=float(sp.quantile(.90)))

    return dict(trans_prob=prob.to_dict(),
                dur_median=dur.median().to_dict(),
                dur_p10=dur.quantile(.10).to_dict(),
                dur_profile=prof,
                span_profile=span,
                global_median=float(p["gap"].median()),
                clean_gaps=p["gap"].to_numpy(),
                activity_freq=d[ACT_COL].value_counts().to_dict())

def context_to_json(ctx, max_edges=60, max_dur=60):
    edges = sorted(ctx["trans_prob"].items(), key=lambda kv: -kv[1])[:max_edges]
    prof = ctx["dur_profile"].sort_values("n", ascending=False).head(max_dur)
    return dict(
        transition_probabilities_top=[
            dict(source=a, target=b, probability=round(pr, 4)) for (a, b), pr in edges],
        duration_profiles_top=[
            dict(source=a, target=b, n=int(r.n),
                 p10_seconds=float(r.p10), p25_seconds=float(r.p25),
                 median_seconds=float(r["median"]), p75_seconds=float(r.p75),
                 p90_seconds=float(r.p90),
                 share_under_60s=round(float(r.burst_share), 3))
            for (a, b), r in prof.iterrows()],
        block_span_profile_seconds=ctx["span_profile"],
        global_median_duration_seconds=ctx["global_median"],
        note="Estimated from CLEAN rows only. duration_profiles give the "
             "distribution, not a point estimate: these gaps are bimodal, so the "
             "median mixes a fast automated mode with a slow waiting mode. "
             "share_under_60s tells you how often a transition is the fast kind. "
             "block_span_profile gives how long a run of k consecutive events "
             "actually lasts -- use it to size a reconstructed block.")

# AUDIT-OFF CTX_PROBE = build_context(_probe, CFG)

# ===== NB CELL 24 =====
def _score(seq, left, right, prob, eps=1e-9):
    chain = ([left] if left else []) + list(seq) + ([right] if right else [])
    return sum(math.log(prob.get((a, b), eps)) for a, b in zip(chain, chain[1:]))

def _best_order(acts, left, right, prob, cfg):
    if len(acts) <= cfg["EXHAUSTIVE_MAX"]:
        best, bs = list(range(len(acts))), -np.inf
        for p in itertools.permutations(range(len(acts))):
            s = _score([acts[i] for i in p], left, right, prob)
            if s > bs:
                bs, best = s, list(p)
        return best
    beam = [([], list(range(len(acts))))]
    for _ in range(len(acts)):
        cand = []
        for chosen, rest in beam:
            for r in rest:
                nc = chosen + [r]
                cand.append((_score([acts[i] for i in nc], left, None, prob),
                             nc, [x for x in rest if x != r]))
        cand.sort(key=lambda t: -t[0])
        beam = [(c, r) for _, c, r in cand[:cfg["BEAM_WIDTH"]]]
    return beam[0][0]

def _segments(g, cat):
    """Contiguous runs of rows with error_category == cat inside one case."""
    out, i = [], 0
    v = g[ERR_COL].to_numpy()
    while i < len(g):
        if v[i] != cat:
            i += 1; continue
        j = i
        while j + 1 < len(g) and v[j + 1] == cat:
            j += 1
        out.append((i, j)); i = j + 1
    return out

def _anchors(g, s, e):
    left = right = None
    for j in range(s - 1, -1, -1):
        if pd.notna(g.loc[j, "_ts"]) and g.loc[j, ERR_COL] == CLEAN:
            left = j; break
    for j in range(e + 1, len(g)):
        if pd.notna(g.loc[j, "_ts"]) and g.loc[j, ERR_COL] == CLEAN:
            right = j; break
    return left, right

def baseline_repair(visible, ctx, cfg, workdir=None):
    """M1+.

    C1 changes vs the old baseline:
      1. The block starts at the OBSERVED collapsed timestamp, not at the left
         anchor. Under form-based capture the shared value is one of the block's
         own timestamps, so one event in the block is already correct and the
         reconstruction should extend outward from it.
      2. The block span is the sum of pair durations clipped to the room that
         actually exists between the observed value and the right anchor.
      3. `C1_DURATION_STAT` selects which percentile drives the spacing.
         'median' minimises MAE, 'p10' minimises the regression rate --
         report both, the trade-off is a result in itself.

    C2 and C3 are deliberately unchanged: C2 is already exact, and every C3
    variant tried was worse than the original.
    """
    stat = cfg.get("C1_DURATION_STAT", "median")
    out = visible.copy()
    out["_ts"] = pd.to_datetime(out[TIME_COL], errors="coerce")
    prob, med, gmed = ctx["trans_prob"], ctx["dur_median"], ctx["global_median"]
    prof = ctx["dur_profile"]
    if gmed != gmed:
        gmed = cfg["FALLBACK_DURATION"]
    new = out["_ts"].astype("datetime64[ns]").copy()

    def _dur(prev, a):
        if stat != "median" and (prev, a) in prof.index:
            v = prof.loc[(prev, a), stat]
        else:
            v = med.get((prev, a), gmed)
        v = float(v) if v == v else gmed
        return max(cfg["MIN_DURATION"], v)

    hit = out.loc[out[ERR_COL].ne(CLEAN), CASE_COL].unique()
    for _, g0 in out[out[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
        g = g0.reset_index(drop=False).rename(columns={"index": "_ix"})

        # ---------------- C1 ----------------
        for s, e in _segments(g, C1):
            li, ri = _anchors(g, s, e)
            acts = g.loc[s:e, ACT_COL].tolist()
            k = len(acts)
            la = g.loc[li, ACT_COL] if li is not None else None
            ra = g.loc[ri, ACT_COL] if ri is not None else None
            lt = g.loc[li, "_ts"] if li is not None else None
            rt = g.loc[ri, "_ts"] if ri is not None else None
            obs = g.loc[s, "_ts"]                      # the collapsed value

            order = _best_order(acts, la, ra, prob, cfg)
            ordered = [acts[i] for i in order]

            # internal gaps only: the first event sits ON the observed value
            gaps, prev = [], ordered[0]
            for a in ordered[1:]:
                gaps.append(_dur(prev, a)); prev = a
            if not gaps:
                gaps = [cfg["MIN_DURATION"]]

            S = sum(gaps)
            if rt is not None and pd.notna(rt):
                room = (rt - obs).total_seconds()
                if room > 0:
                    S = min(S, room * .95)
            S = max(S, cfg["MIN_DURATION"] * max(1, k - 1))

            tot = sum(gaps) or 1.0
            offs = np.concatenate([[0.0], np.cumsum([S * x / tot for x in gaps])])[:k]

            for rank, pos in enumerate(order):
                v = obs + pd.Timedelta(seconds=float(offs[rank]))
                if lt is not None and pd.notna(lt) and v <= lt:
                    v = lt + pd.Timedelta(seconds=cfg["MIN_DURATION"])
                if rt is not None and pd.notna(rt) and v >= rt:
                    v = rt - pd.Timedelta(seconds=cfg["MIN_DURATION"])
                new.loc[g.loc[s + pos, "_ix"]] = v

        # ---------------- C2 ----------------
        for s, e in _segments(g, C2):
            if e == s:
                # a one-row C2 segment is a DISPLACED value, not an exchanged
                # pair: its true timestamp was destroyed and sorting cannot
                # bring it back. Estimate it from the left anchor the same way
                # C3 does, which is the only defensible thing a statistical
                # baseline can do here.
                li, ri = _anchors(g, s, e)
                lt = g.loc[li, "_ts"] if li is not None else None
                rt = g.loc[ri, "_ts"] if ri is not None else None
                prev = g.loc[li, ACT_COL] if li is not None else None
                dd = _dur(prev, g.loc[s, ACT_COL])
                if lt is not None and pd.notna(lt):
                    cand = lt + pd.Timedelta(seconds=dd)
                elif rt is not None and pd.notna(rt):
                    cand = rt - pd.Timedelta(seconds=dd)
                else:
                    cand = g.loc[s, "_ts"]
                if rt is not None and pd.notna(rt) and pd.notna(cand) and cand >= rt:
                    cand = lt + (rt - lt) / 2 if (lt is not None and pd.notna(lt))                         else rt - pd.Timedelta(seconds=cfg["MIN_DURATION"])
                new.loc[g.loc[s, "_ix"]] = cand.round("ms") if pd.notna(cand) else cand
                continue
            vals = sorted(g.loc[s:e, "_ts"].tolist())
            li, ri = _anchors(g, s, e)
            lt = g.loc[li, "_ts"] if li is not None else None
            rt = g.loc[ri, "_ts"] if ri is not None else None
            for kk, v in enumerate(vals):
                if lt is not None and pd.notna(lt) and v <= lt:
                    v = lt + pd.Timedelta(seconds=cfg["MIN_DURATION"] * (kk + 1))
                if rt is not None and pd.notna(rt) and v >= rt:
                    v = rt - pd.Timedelta(seconds=cfg["MIN_DURATION"] * (len(vals) - kk))
                new.loc[g.loc[s + kk, "_ix"]] = v

        # ---------------- C3 ----------------
        for s, e in _segments(g, C3):
            li, ri = _anchors(g, s, e)
            lt = g.loc[li, "_ts"] if li is not None else None
            rt = g.loc[ri, "_ts"] if ri is not None else None
            prev = g.loc[li, ACT_COL] if li is not None else None
            acc = lt
            for kk in range(s, e + 1):
                obs = g.loc[kk, "_ts"]
                dd = _dur(prev, g.loc[kk, ACT_COL])
                if pd.notna(obs) and obs.normalize() == obs:
                    cand = obs + pd.Timedelta(seconds=dd)      # date is trustworthy
                elif acc is not None and pd.notna(acc):
                    cand = acc + pd.Timedelta(seconds=dd)
                else:
                    cand = rt - pd.Timedelta(seconds=dd) if rt is not None else obs
                if rt is not None and pd.notna(rt) and pd.notna(cand) and cand >= rt:
                    # measured: pinning just inside the right anchor beats
                    # splitting the remaining room (MedAE 3,054 s vs 6,523 s on
                    # bpic12_C3_30), so this stays as it was
                    cand = rt - pd.Timedelta(seconds=cfg["MIN_DURATION"])
                cand = cand.round("ms") if pd.notna(cand) else cand
                new.loc[g.loc[kk, "_ix"]] = cand
                acc, prev = cand, g.loc[kk, ACT_COL]

    res = visible.copy()
    res[TIME_COL] = pd.to_datetime(new).dt.strftime(cfg["TS_FORMAT"])
    return res

# ===== NB CELL 26 =====
import itertools as _it

def _prep(visible, cfg):
    out = visible.copy()
    out["_ts"] = pd.to_datetime(out[TIME_COL], errors="coerce")
    return out

def _finish(visible, new, cfg):
    res = visible.copy()
    res[TIME_COL] = pd.to_datetime(new).dt.strftime(cfg["TS_FORMAT"])
    return res

def null_repair(visible, ctx, cfg, workdir=None):
    """Do nothing. The reference every other method has to beat.

    This is not padding. HANDOFF F6 already observed that M4 is close to a no-op
    and still wins MIX_30; without M0 in the table a reader cannot tell whether
    the winner repaired well or simply refrained. On C1 in the real logs this
    row is expected to be competitive, and that IS the paper's finding."""
    return visible.copy()

def conforti_repair(visible, ctx, cfg, workdir=None):
    """Conforti, La Rosa, ter Hofstede & Augusto, BPM 2020,
    'Automatic Repair of Same-Timestamp Errors in Business Process Event Logs'.

    Their method orders a same-timestamp block using DEPENDENCY relations
    mined from the parts of the log that are not corrupted, then redistributes
    the timestamps across the surrounding interval.

    Faithful here:
      - the dependency measure D(a,b) = (|a>b| - |b>a|) / (|a>b| + |b>a| + 1),
        the heuristics-miner dependency used to decide precedence
      - ordering by maximising summed dependency along the chain, including the
        two anchor activities
      - uniform redistribution across the anchor interval
    Adapted:
      - their concurrency handling via the log's alpha relations is reduced to
        leaving near-zero-dependency pairs in their observed order
      - they repair only same-timestamp errors, so C2 and C3 fall back to M1

    The contrast with M1 that matters: M1 spaces events by MEDIAN DURATION and
    starts at the observed value; Conforti spaces them UNIFORMLY across the
    whole anchor gap. Uniform spacing is exactly what the predecessor prompt forbade
    the LLM from doing, so having it in the table shows what that prohibition
    is worth."""
    out = _prep(visible, cfg)
    new = out["_ts"].astype("datetime64[ns]").copy()

    # dependency measure from CLEAN rows only
    d = out[out[ERR_COL].eq(CLEAN)].sort_values([CASE_COL], kind="mergesort")
    d = d[d["_ts"].notna()]
    df_counts = Counter()
    for _, g in d.groupby(CASE_COL, sort=False):
        acts = g.sort_values("_ts", kind="mergesort")[ACT_COL].tolist()
        for a, b in zip(acts, acts[1:]):
            df_counts[(a, b)] += 1

    def dep(a, b):
        ab, ba = df_counts.get((a, b), 0), df_counts.get((b, a), 0)
        return (ab - ba) / (ab + ba + 1)

    def chain_score(seq, left, right):
        ch = ([left] if left else []) + list(seq) + ([right] if right else [])
        return sum(dep(x, y) for x, y in zip(ch, ch[1:]))

    hit = out.loc[out[ERR_COL].ne(CLEAN), CASE_COL].unique()
    for _, g0 in out[out[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
        g = g0.reset_index(drop=False).rename(columns={"index": "_ix"})
        for s, e in _segments(g, C1):
            li, ri = _anchors(g, s, e)
            acts = g.loc[s:e, ACT_COL].tolist()
            k = len(acts)
            la = g.loc[li, ACT_COL] if li is not None else None
            ra = g.loc[ri, ACT_COL] if ri is not None else None
            lt = g.loc[li, "_ts"] if li is not None else None
            rt = g.loc[ri, "_ts"] if ri is not None else None
            obs = g.loc[s, "_ts"]

            if k <= cfg["EXHAUSTIVE_MAX"]:
                order = max(_it.permutations(range(k)),
                            key=lambda p: chain_score([acts[i] for i in p], la, ra))
                order = list(order)
            else:
                order, rest = [], list(range(k))
                cur = la
                while rest:
                    nxt = max(rest, key=lambda i: dep(cur, acts[i]))
                    order.append(nxt); rest.remove(nxt); cur = acts[nxt]

            # uniform redistribution across the anchor interval
            lo = lt if (lt is not None and pd.notna(lt)) else obs
            hi = rt if (rt is not None and pd.notna(rt)) else \
                lo + pd.Timedelta(seconds=cfg["FALLBACK_DURATION"] * (k + 1))
            total = (hi - lo).total_seconds()
            if total <= 0:
                total = cfg["MIN_DURATION"] * (k + 1)
            step = total / (k + 1)
            for rank, pos in enumerate(order):
                new.loc[g.loc[s + pos, "_ix"]] = lo + pd.Timedelta(
                    seconds=step * (rank + 1))

    res = _finish(visible, new, cfg)
    # C2 / C3 are outside the paper's scope -> delegate to M1 so the row is
    # not empty, and say so in the caption
    fb = baseline_repair(visible, ctx, cfg)
    other = visible[ERR_COL].isin([C2, C3]).to_numpy()
    res.loc[other, TIME_COL] = fb.loc[other, TIME_COL].to_numpy()
    return res

def make_lstm_repair(epochs=8, hidden=64, prefix_len=5, seed=42):
    """Tax, Verenich, La Rosa & Dumas, CAiSE 2017, and Evermann, Rehse &
    Fettke, DSS 2017 -- the predictive-process-monitoring LSTM used as the
    learning baseline in the predecessor study paper.

    Faithful here:
      - an activity-embedding LSTM over the trace prefix
      - target log(1 + delta_t), inverted at inference, as in the predecessor setup
      - autoregressive application: reorder first, then predict each duration
        with the previously predicted event appended to the prefix
      - scaling into the anchor window when the predicted total overruns it
    Adapted:
      - trained on CLEAN rows of the corrupted log itself, since there is no
        separate clean training log in this setting
      - falls back to gradient boosting on the same features if torch is absent,
        which must be stated in the paper if the fallback is what ran

    Expected behaviour: the predecessor result was that the LSTM matched the median
    baseline on ordering and lost on spacing. Reproducing that on the real logs
    is the point -- it is the continuity link to the previous paper."""

    def method(visible, ctx, cfg, workdir=None):
        out = _prep(visible, cfg)
        new = out["_ts"].astype("datetime64[ns]").copy()

        d = out[out[ERR_COL].eq(CLEAN)].copy()
        d = d[d["_ts"].notna()].sort_values([CASE_COL, "_ts"], kind="mergesort")
        d["gap"] = d.groupby(CASE_COL)["_ts"].diff().dt.total_seconds()
        d = d[d["gap"].notna() & (d["gap"] >= 0)]
        if len(d) < 200:
            return baseline_repair(visible, ctx, cfg)

        acts = sorted(out[ACT_COL].astype(str).unique())
        a2i = {a: i + 1 for i, a in enumerate(acts)}      # 0 = padding

        X, Y = [], []
        for _, g in d.groupby(CASE_COL, sort=False):
            seq = [a2i.get(str(a), 0) for a in g[ACT_COL]]
            gaps = g["gap"].tolist()
            for i in range(len(seq)):
                pre = seq[max(0, i - prefix_len + 1): i + 1]
                X.append([0] * (prefix_len - len(pre)) + pre)
                Y.append(math.log1p(max(gaps[i], 0.0)))
        X = np.asarray(X, dtype=np.int64); Y = np.asarray(Y, dtype=np.float32)

        predict = None
        try:
            import torch
            import torch.nn as nn
            torch.manual_seed(seed)

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.emb = nn.Embedding(len(acts) + 1, 32, padding_idx=0)
                    self.lstm = nn.LSTM(32, hidden, batch_first=True)
                    self.head = nn.Linear(hidden, 1)

                def forward(self, x):
                    h, _ = self.lstm(self.emb(x))
                    return self.head(h[:, -1, :]).squeeze(-1)

            net = Net()
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            lossf = nn.L1Loss()
            xt = torch.as_tensor(X); yt = torch.as_tensor(Y)
            n = len(xt); bs = 256
            for _ in range(epochs):
                perm = torch.randperm(n)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    opt.zero_grad()
                    loss = lossf(net(xt[idx]), yt[idx])
                    loss.backward(); opt.step()
            net.eval()

            def predict(seq):
                with torch.no_grad():
                    t = torch.as_tensor(np.asarray([seq], dtype=np.int64))
                    return float(net(t).item())
        except Exception as ex:
            print("  [B2] torch unavailable (%s); using GBM fallback" % ex)
            from sklearn.ensemble import HistGradientBoostingRegressor
            gbm = HistGradientBoostingRegressor(max_iter=200,
                                                random_state=seed).fit(X, Y)

            def predict(seq):
                return float(gbm.predict(np.asarray([seq], dtype=np.int64))[0])

        prob = ctx["trans_prob"]
        hit = out.loc[out[ERR_COL].ne(CLEAN), CASE_COL].unique()
        for _, g0 in out[out[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
            g = g0.reset_index(drop=False).rename(columns={"index": "_ix"})
            hist = [a2i.get(str(a), 0) for a in g[ACT_COL]]
            for s, e in _segments(g, C1):
                li, ri = _anchors(g, s, e)
                blk = g.loc[s:e, ACT_COL].tolist(); k = len(blk)
                la = g.loc[li, ACT_COL] if li is not None else None
                ra = g.loc[ri, ACT_COL] if ri is not None else None
                lt = g.loc[li, "_ts"] if li is not None else None
                rt = g.loc[ri, "_ts"] if ri is not None else None
                order = _best_order(blk, la, ra, prob, cfg)
                ordered = [blk[i] for i in order]

                pre = hist[max(0, s - prefix_len): s]
                start = lt if (lt is not None and pd.notna(lt)) else g.loc[s, "_ts"]
                cur, ds = start, []
                for a in ordered:
                    seq = (pre + [a2i.get(str(a), 0)])[-prefix_len:]
                    seq = [0] * (prefix_len - len(seq)) + seq
                    dt = max(cfg["MIN_DURATION"], math.expm1(predict(seq)))
                    ds.append(dt); pre = seq
                if rt is not None and pd.notna(rt) and pd.notna(start):
                    room = (rt - start).total_seconds()
                    if room > 0 and sum(ds) > room:
                        ds = [x * room * .95 / sum(ds) for x in ds]
                for rank, pos in enumerate(order):
                    cur = cur + pd.Timedelta(seconds=float(ds[rank]))
                    new.loc[g.loc[s + pos, "_ix"]] = cur

        res = _finish(visible, new, cfg)
        fb = baseline_repair(visible, ctx, cfg)
        other = visible[ERR_COL].isin([C2, C3]).to_numpy()
        res.loc[other, TIME_COL] = fb.loc[other, TIME_COL].to_numpy()
        return res
    return method

def bridge_repair(visible, ctx, cfg, workdir=None):
    """Rogge-Solti, Mans, van der Aalst & Weske, 'Repairing event logs using
    timed process models' (OTM 2013 / and the stochastic Petri net line of
    work, BPM 2013), plus the Bayesian-network variant surveyed by Wu et al.

    Their idea: model each transition's duration as a random variable with a
    fitted distribution, then condition on the observed surrounding events and
    take the most likely assignment. That is a bridge problem, not an
    interpolation.

    Faithful here:
      - lognormal duration per activity pair, parameters (mu, sigma) fitted on
        CLEAN rows in log space
      - the repaired times are the MAP assignment given the anchor interval:
        minimise sum_i ((log d_i - mu_i)^2 / 2 sigma_i^2) subject to sum d_i = W
      - a transition with high variance absorbs more of the discrepancy than a
        tight one, which uniform or proportional scaling cannot express
    Adapted:
      - no Petri net is discovered; the ordering comes from the same transition
        probabilities M1 uses, so B3 differs from M1 only in HOW it spaces
      - independence across consecutive durations, as in the original

    This is the most informative comparison in the table: B1 spaces uniformly,
    M1 spaces proportionally to medians, B3 spaces by likelihood. If all three
    land in the same place, spacing strategy is not where the problem is."""
    out = _prep(visible, cfg)
    new = out["_ts"].astype("datetime64[ns]").copy()
    prob = ctx["trans_prob"]

    d = out[out[ERR_COL].eq(CLEAN)].copy()
    d = d[d["_ts"].notna()].sort_values([CASE_COL, "_ts"], kind="mergesort")
    d["prev"] = d.groupby(CASE_COL)[ACT_COL].shift()
    d["gap"] = d.groupby(CASE_COL)["_ts"].diff().dt.total_seconds()
    p = d.dropna(subset=["prev", "gap"])
    p = p[p["gap"] > 0].copy()
    p["lg"] = np.log(p["gap"])
    par = p.groupby(["prev", ACT_COL])["lg"].agg(["mean", "std", "count"])
    gmu = float(p["lg"].mean()) if len(p) else math.log(60.0)
    gsd = float(p["lg"].std()) if len(p) else 1.0

    def musig(a, b):
        if (a, b) in par.index:
            r = par.loc[(a, b)]
            if r["count"] >= 5 and r["std"] == r["std"] and r["std"] > 1e-6:
                return float(r["mean"]), float(r["std"])
        return gmu, max(gsd, .5)

    def map_bridge(pairs, W):
        """argmin sum ((log d_i - mu_i)^2 / 2 s_i^2) s.t. sum d_i = W, d_i > 0."""
        mus = np.array([m for m, _ in pairs], dtype=float)
        sds = np.array([max(s, 1e-3) for _, s in pairs], dtype=float)
        base = np.exp(mus)
        if W <= 0 or not np.isfinite(W):
            return np.maximum(base, cfg["MIN_DURATION"])
        try:
            from scipy.optimize import minimize
            # parametrise d_i = W * softmax(z)_i so the constraint is exact
            z0 = np.log(np.maximum(base, 1e-9))
            z0 = z0 - z0.mean()

            def obj(z):
                w = np.exp(z - z.max()); w = w / w.sum()
                dd = np.maximum(W * w, 1e-9)
                return float(np.sum((np.log(dd) - mus) ** 2 / (2 * sds ** 2)))
            r = minimize(obj, z0, method="Nelder-Mead",
                         options=dict(maxiter=400, xatol=1e-3, fatol=1e-6))
            w = np.exp(r.x - r.x.max()); w = w / w.sum()
            return np.maximum(W * w, cfg["MIN_DURATION"])
        except Exception:
            w = base / base.sum()
            return np.maximum(W * w, cfg["MIN_DURATION"])

    hit = out.loc[out[ERR_COL].ne(CLEAN), CASE_COL].unique()
    for _, g0 in out[out[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
        g = g0.reset_index(drop=False).rename(columns={"index": "_ix"})
        for s, e in _segments(g, C1):
            li, ri = _anchors(g, s, e)
            blk = g.loc[s:e, ACT_COL].tolist(); k = len(blk)
            la = g.loc[li, ACT_COL] if li is not None else None
            ra = g.loc[ri, ACT_COL] if ri is not None else None
            lt = g.loc[li, "_ts"] if li is not None else None
            rt = g.loc[ri, "_ts"] if ri is not None else None
            if lt is None or rt is None or pd.isna(lt) or pd.isna(rt):
                continue
            order = _best_order(blk, la, ra, prob, cfg)
            ordered = [blk[i] for i in order]
            chain = ([la] if la else [ordered[0]]) + ordered + ([ra] if ra else [])
            pairs = [musig(a, b) for a, b in zip(chain, chain[1:])]
            W = (rt - lt).total_seconds()
            ds = map_bridge(pairs, W)
            cur = lt
            for rank, pos in enumerate(order):
                cur = cur + pd.Timedelta(seconds=float(ds[rank]))
                if cur >= rt:
                    cur = rt - pd.Timedelta(seconds=cfg["MIN_DURATION"])
                new.loc[g.loc[s + pos, "_ix"]] = cur

        # C3: the same bridge, one unknown event between two anchors
        for s, e in _segments(g, C3):
            li, ri = _anchors(g, s, e)
            lt = g.loc[li, "_ts"] if li is not None else None
            rt = g.loc[ri, "_ts"] if ri is not None else None
            if lt is None or rt is None or pd.isna(lt) or pd.isna(rt):
                continue
            la = g.loc[li, ACT_COL] if li is not None else None
            ra = g.loc[ri, ACT_COL] if ri is not None else None
            blk = g.loc[s:e, ACT_COL].tolist()
            chain = [la] + blk + [ra]
            pairs = [musig(a, b) for a, b in zip(chain, chain[1:])]
            ds = map_bridge(pairs, (rt - lt).total_seconds())
            cur = lt
            for kk in range(s, e + 1):
                cur = cur + pd.Timedelta(seconds=float(ds[kk - s]))
                obs = g.loc[kk, "_ts"]
                if pd.notna(obs) and obs.normalize() == obs:
                    # date-only: keep the day, take only the time of day
                    cur = obs + pd.Timedelta(
                        seconds=(cur - cur.normalize()).total_seconds())
                new.loc[g.loc[kk, "_ix"]] = cur

    res = _finish(visible, new, cfg)
    fb = baseline_repair(visible, ctx, cfg)
    c2 = visible[ERR_COL].eq(C2).to_numpy()
    res.loc[c2, TIME_COL] = fb.loc[c2, TIME_COL].to_numpy()
    return res

def make_sampling_repair(seed=42):
    """Distributional surrogate for Schmid, Moder, Hofmann & Roeglinger,
    Information Systems 2023, 'Everything at the proper time: repairing
    identical timestamp errors in event logs with generative adversarial
    networks'.

    Their GAN learns the conditional duration DISTRIBUTION and samples from it,
    rather than emitting a point estimate. The consequence that matters for the
    evaluation is that the repaired log keeps the shape of the real duration
    distribution even where individual rows are wrong.

    THIS IS A SURROGATE, NOT A REIMPLEMENTATION. It reproduces the sampling
    behaviour by drawing from the empirical per-pair duration distribution of
    the CLEAN rows, not from a trained generator. Label it as such in the
    paper -- calling it 'the GAN baseline' would be wrong. If a full
    comparison is needed, check whether the authors released code; otherwise
    report it as a distributional-sampling baseline and cite Schmid as the
    motivation for including one.

    Why include it at all: it is the only baseline that should do well on the
    NAT_C1 plausibility track (wasserstein_log_gap) while doing badly on MedAE.
    That dissociation is worth a paragraph, because it shows the two tracks
    measure different things."""

    def method(visible, ctx, cfg, workdir=None):
        rng = np.random.default_rng(seed)
        out = _prep(visible, cfg)
        new = out["_ts"].astype("datetime64[ns]").copy()
        prob = ctx["trans_prob"]

        d = out[out[ERR_COL].eq(CLEAN)].copy()
        d = d[d["_ts"].notna()].sort_values([CASE_COL, "_ts"], kind="mergesort")
        d["prev"] = d.groupby(CASE_COL)[ACT_COL].shift()
        d["gap"] = d.groupby(CASE_COL)["_ts"].diff().dt.total_seconds()
        p = d.dropna(subset=["prev", "gap"])
        p = p[p["gap"] >= 0]
        pool = {k: v["gap"].to_numpy()
                for k, v in p.groupby(["prev", ACT_COL])}
        gpool = p["gap"].to_numpy()
        if not len(gpool):
            gpool = np.array([cfg["FALLBACK_DURATION"]])

        def draw(a, b):
            arr = pool.get((a, b))
            if arr is None or len(arr) < 5:
                arr = gpool
            return max(cfg["MIN_DURATION"], float(rng.choice(arr)))

        hit = out.loc[out[ERR_COL].ne(CLEAN), CASE_COL].unique()
        for _, g0 in out[out[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
            g = g0.reset_index(drop=False).rename(columns={"index": "_ix"})
            for s, e in _segments(g, C1):
                li, ri = _anchors(g, s, e)
                blk = g.loc[s:e, ACT_COL].tolist(); k = len(blk)
                la = g.loc[li, ACT_COL] if li is not None else None
                ra = g.loc[ri, ACT_COL] if ri is not None else None
                rt = g.loc[ri, "_ts"] if ri is not None else None
                obs = g.loc[s, "_ts"]
                if pd.isna(obs):
                    continue
                order = _best_order(blk, la, ra, prob, cfg)
                ordered = [blk[i] for i in order]
                gaps, prev = [], ordered[0]
                for a in ordered[1:]:
                    gaps.append(draw(prev, a)); prev = a
                if not gaps:
                    gaps = [cfg["MIN_DURATION"]]
                S = sum(gaps)
                if rt is not None and pd.notna(rt):
                    room = (rt - obs).total_seconds()
                    if room > 0:
                        S = min(S, room * .95)
                tot = sum(gaps) or 1.0
                offs = np.concatenate(
                    [[0.0], np.cumsum([S * x / tot for x in gaps])])[:k]
                for rank, pos in enumerate(order):
                    new.loc[g.loc[s + pos, "_ix"]] = obs + pd.Timedelta(
                        seconds=float(offs[rank]))

        res = _finish(visible, new, cfg)
        fb = baseline_repair(visible, ctx, cfg)
        other = visible[ERR_COL].isin([C2, C3]).to_numpy()
        res.loc[other, TIME_COL] = fb.loc[other, TIME_COL].to_numpy()
        return res
    return method

lstm_repair = make_lstm_repair()

sampling_repair = make_sampling_repair()

BASELINE_METHODS = {
    "M0_null": null_repair,
    "M1_baseline": baseline_repair,
    "B1_conforti": conforti_repair,
    "B2_lstm": lstm_repair,
    "B3_bridge": bridge_repair,
    "B4_sampling": sampling_repair,
}

# ===== NB CELL 36 =====
def enforce(original, candidate, cfg):
    """Return (final_df, report).

    The monotonicity step must not re-sort a segment into row order: that
    would overwrite whatever ordering the method inferred and make every
    method's ordering accuracy identical. It repairs the *values* while
    preserving which row holds which rank.
    """
    o = original.reset_index(drop=True).copy()
    c = candidate.reset_index(drop=True).copy()
    rep = Counter()

    obs = pd.to_datetime(o[TIME_COL], errors="coerce").astype("datetime64[ns]")
    new = pd.to_datetime(c[TIME_COL], errors="coerce").astype("datetime64[ns]")
    pol = o[ERR_COL].ne(CLEAN)

    # 1. unparseable -> observed
    bad = new.isna() & pol
    rep["unparseable_fixed"] = int(bad.sum())
    new[bad] = obs[bad]
    rep["remaining_unparseable"] = int(new.isna().sum())

    # 2. clean rows must be identical
    changed = (~pol) & (new != obs)
    rep["clean_restored"] = int(changed.sum())
    new[changed] = obs[changed]

    o["_new"] = new
    o["_obs"] = obs

    hit = o.loc[pol, CASE_COL].unique()
    for case, g0 in o[o[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
        g = g0.reset_index(drop=False).rename(columns={"index": "_ix"})
        g["_ts"] = g["_obs"]
        for cat in CATEGORIES:
            for s, e in _segments(g, cat):
                li, ri = _anchors(g, s, e)
                lt = g.loc[li, "_obs"] if li is not None else None
                rt = g.loc[ri, "_obs"] if ri is not None else None
                ix = [g.loc[k, "_ix"] for k in range(s, e + 1)]
                vals = [new.loc[i] for i in ix]

                # 3. anchor clamping
                for k, v in enumerate(vals):
                    if pd.isna(v):
                        continue
                    if lt is not None and pd.notna(lt) and v <= lt:
                        vals[k] = lt + pd.Timedelta(seconds=cfg["MIN_DURATION"])
                        rep["anchor_clamped"] += 1
                    if rt is not None and pd.notna(rt) and vals[k] >= rt:
                        vals[k] = rt - pd.Timedelta(seconds=cfg["MIN_DURATION"])
                        rep["anchor_clamped"] += 1

                # 4. monotonicity, RANK-PRESERVING
                #    `order` is the sequence of segment positions in increasing
                #    repaired time -- i.e. the ordering the method chose. We fix
                #    the values along that sequence and write each fixed value
                #    back to the row that owned that rank, so the method's
                #    ordering survives.
                big = np.iinfo(np.int64).max
                order = np.argsort([v.value if pd.notna(v) else big for v in vals],
                                   kind="stable")
                prev = lt
                for rank, pos in enumerate(order):
                    v = vals[pos]
                    if (prev is not None and pd.notna(prev)
                            and pd.notna(v) and v <= prev):
                        v = prev + pd.Timedelta(seconds=cfg["MIN_DURATION"])
                        rep["segment_monotonic_fixed"] += 1
                    new.loc[ix[pos]] = v
                    prev = v

    final = c.copy()
    final[TIME_COL] = pd.to_datetime(new).dt.strftime(cfg["TS_FORMAT"])
    return final, dict(rep)

# ===== NB CELL 38 =====
def _abs_err(a, b):
    return (pd.to_datetime(a, errors="coerce")
            - pd.to_datetime(b, errors="coerce")).dt.total_seconds().abs()

def core_metrics(orig, final, cfg, mask, tag, error_sink=None):
    """Replaces Cell 18's core_metrics.

    New keys: unchanged_rate, MedAE_ratio_gap, MedAE_ratio_case,
    MedAE_rep_lo / MedAE_rep_hi (bootstrap 95% CI), net_benefit.
    """
    obs = pd.to_datetime(orig[TIME_COL], errors="coerce")
    gt = pd.to_datetime(orig[GT_COL], errors="coerce")
    rep = pd.to_datetime(final[TIME_COL], errors="coerce")

    m = mask & gt.notna() & rep.notna()
    if not m.any():
        return dict(scope=tag, n=0)

    e_rep = (rep[m] - gt[m]).dt.total_seconds().abs()
    obs_ok = m & obs.notna()
    e_obs = (obs[obs_ok] - gt[obs_ok]).dt.total_seconds().abs()

    r = dict(scope=tag, n=int(m.sum()),
             n_unobservable=int((mask & obs.isna()).sum()),
             coverage=float(rep[mask].notna().mean()))

    r["MAE_rep"] = float(e_rep.mean())
    r["MedAE_rep"] = float(e_rep.median())
    r["RMSE_rep"] = float(np.sqrt((e_rep ** 2).mean()))
    r["P90_rep"] = float(e_rep.quantile(.90))
    r["logMAE_rep"] = float(np.log1p(e_rep).mean())

    # bootstrap CI on the median -- CFG["N_BOOTSTRAP"] was defined but unused
    nb = int(cfg.get("N_BOOTSTRAP", 0) or 0)
    if nb and len(e_rep) > 5:
        rs = np.random.default_rng(cfg["SEED"])
        arr = e_rep.to_numpy()
        boots = [np.median(rs.choice(arr, size=len(arr), replace=True))
                 for _ in range(nb)]
        r["MedAE_rep_lo"] = float(np.percentile(boots, 2.5))
        r["MedAE_rep_hi"] = float(np.percentile(boots, 97.5))

    if len(e_obs):
        common = e_obs.index.intersection(e_rep.index)
        a, b = e_obs.loc[common], e_rep.loc[common]
        r["MAE_obs"] = float(e_obs.mean())
        r["MedAE_obs"] = float(e_obs.median())
        r["MAE_reduction"] = float(e_obs.mean() - b.mean())
        r["MedAE_reduction"] = float(e_obs.median() - b.median())
        r["regression_rate"] = float((b > a).mean())
        r["improved_rate"] = float((b < a).mean())
        r["unchanged_rate"] = float((b == a).mean())          # B1
        # net benefit: seconds of error removed minus seconds added, per row.
        # Positive means the repair is worth applying even after its damage.
        r["net_benefit_s"] = float((a - b).sum() / max(len(common), 1))
        if len(common) > 5:
            try:
                from scipy.stats import wilcoxon as _wx
                d = (a - b).to_numpy()
                d = d[d != 0]
                if len(d) > 5:
                    r["wilcoxon_p"] = float(_wx(d).pvalue)
            except Exception:
                pass
    else:
        for k in ("MAE_obs", "MedAE_obs", "MAE_reduction", "MedAE_reduction",
                  "regression_rate", "improved_rate", "unchanged_rate",
                  "net_benefit_s"):
            r[k] = np.nan

    for tol in cfg["TOLERANCES"]:
        r["acc@%ds" % tol] = float((e_rep <= tol).mean())

    if error_sink is not None:
        error_sink.append(pd.DataFrame({
            "scope": tag,
            ROW_COL: orig.loc[m.values, ROW_COL].to_numpy()
            if ROW_COL in orig.columns else np.arange(int(m.sum())),
            "e_obs": e_obs.reindex(e_rep.index).to_numpy(),
            "e_rep": e_rep.to_numpy()}))
    return r

def ordering_metrics(orig, final):
    o = orig.reset_index(drop=True).copy()
    o["_rep"] = pd.to_datetime(final.reset_index(drop=True)[TIME_COL], errors="coerce")
    o["_gt"] = pd.to_datetime(o[GT_COL], errors="coerce")
    em, pa, po = [], [], []
    hit = o.loc[o[ERR_COL].eq(C1), CASE_COL].unique()
    for case, g0 in o[o[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
        g = g0.reset_index(drop=True)
        for s, e in _segments(g, C1):
            ids = list(range(s, e + 1))
            if len(ids) < 2:
                continue
            true_o = sorted(ids, key=lambda k: g.loc[k, "_gt"])
            pred_o = sorted(ids, key=lambda k: (g.loc[k, "_rep"], k))
            em.append(true_o == pred_o)
            po.append(np.mean([a == b for a, b in zip(true_o, pred_o)]))
            pos = {k: i for i, k in enumerate(true_o)}
            prs = [(i, j) for i in range(len(pred_o)) for j in range(i + 1, len(pred_o))]
            pa.append(sum(1 for i, j in prs if pos[pred_o[i]] < pos[pred_o[j]]) / len(prs))
    return dict(exact_match=float(np.mean(em)) if em else np.nan,
                pairwise_acc=float(np.mean(pa)) if pa else np.nan,
                position_acc=float(np.mean(po)) if po else np.nan,
                n_c1_blocks=len(em))

def validity_metrics(orig, final, ctx, cfg=None):
    """Monotonicity is evaluated in EVENT order, which is the file's row order.
    pandas' default sort is quicksort and therefore unstable: sorting by Case
    alone silently permutes rows inside a case and manufactures negative gaps
    that no method produced. kind='mergesort' keeps the row order.

    RENAMED IN v4. What this counts is how often the repaired times run
    backwards *relative to the row order of the file*. Under C1 the rows were
    deliberately shuffled, so a method that correctly recovers the true order
    necessarily disagrees with the row order and scores high, while a method
    that changes nothing leaves equal timestamps, gets zero negative gaps, and
    scores zero. On hospital C1_30 that produced M1 = 1,949 and M0 = 0, which
    reads as the exact opposite of the truth. It is a disagreement measure, not
    a damage measure, and the name now says so.

    `noop_rate` is the v4 addition that F4 needed: the share of polluted rows a
    method left exactly as it found them. M2 on hospital_C1_10 scored 1.00 here
    and was otherwise tabulated as a repair.
    """
    cfg = cfg or CFG
    o = orig.reset_index(drop=True).copy()
    rep = pd.to_datetime(final.reset_index(drop=True)[TIME_COL], errors="coerce")
    o["_rep"] = rep
    g = (o.sort_values([CASE_COL], kind="mergesort")
           .groupby(CASE_COL)["_rep"].diff().dt.total_seconds())
    obs = pd.to_datetime(o[TIME_COL], errors="coerce")
    cl = o[ERR_COL].eq(CLEAN)
    pol = ~cl

    same = ((rep[pol] == obs[pol]) | (rep[pol].isna() & obs[pol].isna()))
    noop = float(same.mean()) if pol.any() else np.nan

    out = dict(row_order_disagreement=int((g < 0).sum()),
               row_order_disagreement_rate=float((g < 0).mean()),
               clean_changed_rate=float((rep[cl] != obs[cl]).mean()),
               unparseable=int(rep.isna().sum()),
               unfilled_polluted=int(rep[pol].isna().sum()),
               noop_rate=noop,
               is_noop=bool(noop == noop and noop >= cfg.get("NOOP_TOLERANCE", .99)))
    if HAS_SCIPY:
        rg = g.dropna(); rg = rg[rg >= 0]
        if len(rg):
            out["wasserstein_log_gap"] = float(wasserstein_distance(
                np.log1p(rg), np.log1p(ctx["clean_gaps"])))
    return out

# ===== NB CELL 40 =====
def log_scale_reference(orig, ctx):
    """The numbers every error should be read against.

    median_clean_gap    the typical distance between two consecutive events in
                        this log, estimated on CLEAN rows only. An error below
                        this is inside the log's own noise floor.
    p90_clean_gap       where the heavy tail starts.
    median_case_span    how long a whole case takes. An error near this size
                        means the event was placed effectively at random within
                        its case.
    """
    d = orig[orig[ERR_COL].eq(CLEAN)].copy()
    d["_t"] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d[d["_t"].notna()]
    gaps = np.asarray(ctx.get("clean_gaps", []), dtype=float)
    gaps = gaps[np.isfinite(gaps)]
    span = d.groupby(CASE_COL)["_t"].agg(lambda s: (s.max() - s.min()).total_seconds())
    span = span[span > 0]
    return dict(
        median_clean_gap=float(np.median(gaps)) if len(gaps) else np.nan,
        p90_clean_gap=float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
        median_case_span=float(span.median()) if len(span) else np.nan,
        n_clean_gaps=int(len(gaps)))

def _dur_fn(ctx, cfg, stat="median"):
    med, gmed, prof = ctx["dur_median"], ctx["global_median"], ctx["dur_profile"]
    if gmed != gmed:
        gmed = cfg["FALLBACK_DURATION"]

    def f(prev, a):
        if stat != "median" and (prev, a) in prof.index:
            v = prof.loc[(prev, a), stat]
        else:
            v = med.get((prev, a), gmed)
        v = float(v) if v == v else gmed
        return max(cfg["MIN_DURATION"], v)
    return f

def reference_repair(orig, ctx, cfg, order_source="true", seed=0):
    """Guide rails for the charts. NOT methods -- `true` reads the ground truth.

    order_source="true"    the ORACLE-ORDERING reference. The ordering is given
                           for free and only the spacing is reconstructed, using
                           exactly M1's spacing rule. This is the best score any
                           method could reach by being perfect at ordering, so
                           the distance between a method's bar and this line is
                           the part of its error that is NOT an ordering problem.
    order_source="random"  a random permutation with the same spacing rule. This
                           is what "no behavioural knowledge at all" scores.

    Together they bracket the interval a result must be read inside. Reporting
    a method's position in [random, oracle] is the normalised recovery index.
    """
    rng = np.random.default_rng(seed)
    visible = orig[[c for c in [CASE_COL, ACT_COL, TIME_COL, RES_COL,
                                ERR_COL, ROW_COL] if c in orig.columns]].copy()
    out = baseline_repair(visible, ctx, cfg)          # C2 and C3 handled here
    dur = _dur_fn(ctx, cfg, cfg.get("C1_DURATION_STAT", "median"))

    work = orig.reset_index(drop=True).copy()
    work["_ts"] = pd.to_datetime(work[TIME_COL], errors="coerce")
    work["_gt"] = pd.to_datetime(work[GT_COL], errors="coerce")
    # cast to ns and round to ms before assigning: a us-dtype column rejects a
    # ns-precision Timestamp outright (the same TypeError the C3 branch of
    # baseline_repair guards against)
    new = pd.to_datetime(out.reset_index(drop=True)[TIME_COL],
                         errors="coerce").astype("datetime64[ns]")

    hit = work.loc[work[ERR_COL].eq(C1), CASE_COL].unique()
    for _, g0 in work[work[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
        g = g0.reset_index(drop=False).rename(columns={"index": "_ix"})
        for s, e in _segments(g, C1):
            li, ri = _anchors(g, s, e)
            acts = g.loc[s:e, ACT_COL].tolist()
            k = len(acts)
            lt = g.loc[li, "_ts"] if li is not None else None
            rt = g.loc[ri, "_ts"] if ri is not None else None
            obs = g.loc[s, "_ts"]
            if pd.isna(obs):
                continue

            if order_source == "true":
                gts = g.loc[s:e, "_gt"].tolist()
                order = list(np.argsort([pd.Timestamp(x).value for x in gts]))
            else:
                order = list(rng.permutation(k))

            ordered = [acts[i] for i in order]
            gaps, prev = [], ordered[0]
            for a in ordered[1:]:
                gaps.append(dur(prev, a)); prev = a
            if not gaps:
                gaps = [cfg["MIN_DURATION"]]
            S = sum(gaps)
            if rt is not None and pd.notna(rt):
                room = (rt - obs).total_seconds()
                if room > 0:
                    S = min(S, room * .95)
            S = max(S, cfg["MIN_DURATION"] * max(1, k - 1))
            tot = sum(gaps) or 1.0
            offs = np.concatenate([[0.0], np.cumsum([S * x / tot for x in gaps])])[:k]
            for rank, pos in enumerate(order):
                v = obs + pd.Timedelta(seconds=float(offs[rank]))
                if lt is not None and pd.notna(lt) and v <= lt:
                    v = lt + pd.Timedelta(seconds=cfg["MIN_DURATION"])
                if rt is not None and pd.notna(rt) and v >= rt:
                    v = rt - pd.Timedelta(seconds=cfg["MIN_DURATION"])
                new.loc[g.loc[s + int(pos), "_ix"]] = v.round("ms")

    res = out.copy()
    res[TIME_COL] = pd.to_datetime(new).dt.strftime(cfg["TS_FORMAT"])
    return res

def build_guides(dataset_name, dataset_path, cfg):
    """One row of guide rails per dataset. Cache it and hand it to the plots."""
    orig = pd.read_csv(dataset_path)
    orig[ERR_COL] = orig[ERR_COL].fillna(CLEAN)
    ctx = build_context(orig, cfg)
    pol = orig[ERR_COL].ne(CLEAN)

    g = dict(dataset=dataset_name)
    g.update(log_scale_reference(orig, ctx))

    for src, key in (("true", "oracle"), ("random", "random")):
        cand = reference_repair(orig, ctx, cfg, order_source=src)
        final, _ = enforce(orig, cand, cfg)
        for scope, mask in [("ALL_POLLUTED", pol)] + \
                           [(c, orig[ERR_COL].eq(c)) for c in CATEGORIES]:
            if not mask.any():
                continue
            m = core_metrics(orig, final, cfg, mask, scope)
            g["%s_MedAE__%s" % (key, scope)] = m.get("MedAE_rep", np.nan)
            g["%s_MAE__%s" % (key, scope)] = m.get("MAE_rep", np.nan)
            g["obs_MedAE__%s" % scope] = m.get("MedAE_obs", np.nan)
    return g

def attach_guides(results, guides):
    """Adds the interpretive columns the raw metric table lacks.

    err_in_gaps    repaired error expressed in multiples of the log's own
                   median clean gap. 0.5 means half a typical inter-event
                   distance -- i.e. inside the noise.
    err_in_case    repaired error as a fraction of a median case duration.
                   Approaching 1 means the event is placed at random in its case.
    NRI            normalised recovery index,
                     (E_random - E_method) / (E_random - E_oracle)
                   1.0 = as good as knowing the true order, 0.0 = no better than
                   a random order, negative = worse than random. This is the
                   number to put on the chart: it is comparable across logs,
                   across error types and across injection rates, which raw
                   seconds are not.
    recovered      (E_obs - E_method) / (E_obs - E_oracle), the share of the
                   recoverable error actually removed. Use alongside NRI: NRI
                   answers "did behaviour help?", recovered answers "was the
                   repair worth doing at all?"
    """
    r = results.merge(guides, on="dataset", how="left")

    def pick(row, key):
        return row.get("%s__%s" % (key, row["scope"]), np.nan)

    orc = r.apply(lambda x: pick(x, "oracle_MedAE"), axis=1)
    rnd = r.apply(lambda x: pick(x, "random_MedAE"), axis=1)
    obs = r["MedAE_obs"]
    rep = r["MedAE_rep"]

    r["oracle_MedAE"] = orc
    r["random_MedAE"] = rnd
    r["err_in_gaps"] = rep / r["median_clean_gap"]
    r["err_in_case"] = rep / r["median_case_span"]
    r["NRI"] = (rnd - rep) / (rnd - orc).replace(0, np.nan)
    r["recovered"] = (obs - rep) / (obs - orc).replace(0, np.nan)
    r["inside_noise"] = r["err_in_gaps"] < 1.0
    return r

# ===== NB CELL 42 =====
def coarse_track(orig, final, cfg):
    o = orig.reset_index(drop=True)
    rep = pd.to_datetime(final.reset_index(drop=True)[TIME_COL], errors="coerce")
    obs = pd.to_datetime(o[TIME_COL], errors="coerce")
    m = o["stratum"].eq("COARSE_C4") & obs.notna() & rep.notna()
    if not m.any():
        return dict(scope="COARSE_C4", n=0)
    same_day = (rep[m].dt.normalize() == obs[m].dt.normalize())
    invented = (rep[m] != obs[m]) & o.loc[m, ERR_COL].eq(CLEAN)
    return dict(scope="COARSE_C4", n=int(m.sum()),
                same_day_rate=float(same_day.mean()),
                day_shift_median=float((rep[m] - obs[m]).dt.days.abs().median()),
                precision_invented_rate=float(invented.mean()))

# ===== NB CELL 44 =====
def natural_track(orig, final, ctx, cfg):
    o = orig.reset_index(drop=True).copy()
    o["_rep"] = pd.to_datetime(final.reset_index(drop=True)[TIME_COL], errors="coerce")
    m = o["stratum"].eq("NAT_C1")
    if not m.any():
        return dict(scope="NAT_C1", n=0)

    # pandas' default sort is quicksort and therefore UNSTABLE: sorting by Case
    # alone silently permutes rows inside a case and manufactures negative gaps
    # that no method produced. mergesort preserves the file's event order.
    g = (o.sort_values([CASE_COL], kind="mergesort")
           .groupby(CASE_COL)["_rep"].diff().dt.total_seconds())
    gaps = g[m.reindex(g.index, fill_value=False)].dropna()
    clean = ctx["clean_gaps"]
    lo, hi = np.percentile(clean, 1), np.percentile(clean, 99)

    out = dict(scope="NAT_C1", n=int(m.sum()),
               still_collapsed_rate=float((gaps == 0).mean()) if len(gaps) else np.nan,
               out_of_clean_range_rate=float(((gaps < lo) | (gaps > hi)).mean())
                   if len(gaps) else np.nan,
               monotonicity_violations=int((gaps < 0).sum()))
    if HAS_SCIPY and len(gaps):
        pos = gaps[gaps >= 0]
        out["wasserstein_log_gap"] = float(wasserstein_distance(
            np.log1p(pos), np.log1p(clean)))
    return out

# ===== write_visible (from cell 28) =====
def write_visible(df, path):
    """Exactly what a method is allowed to read."""
    cols = [CASE_COL, ACT_COL, TIME_COL, RES_COL, ERR_COL, ROW_COL]
    v = df[cols].copy()
    v[ERR_COL] = v[ERR_COL].fillna(CLEAN)
    v.to_csv(path, index=False)
    return v
