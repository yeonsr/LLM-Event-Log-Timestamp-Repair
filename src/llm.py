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

# ===== NB CELL 22 =====
def _perf_value(x):
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, dict):
        for k in ("median", "mean", "avg", "sum"):
            if k in x:
                try:
                    return float(x[k])
                except Exception:
                    pass
        for v in x.values():
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v)
    return None

def build_pm4py_context(df, max_edges=60, max_variants=20):
    try:
        import pm4py
    except ImportError:
        return {"pm4py_available": False}

    d = df[df[ERR_COL].eq(CLEAN)].copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d[d[TIME_COL].notna()]
    if d.empty:
        return {"pm4py_available": False}

    pm = d[[CASE_COL, ACT_COL, TIME_COL]].copy()
    pm[CASE_COL] = pm[CASE_COL].astype(str)
    pm = pm4py.format_dataframe(pm, case_id=CASE_COL,
                                activity_key=ACT_COL, timestamp_key=TIME_COL)
    log = pm4py.convert_to_event_log(pm)

    dfg, start, end = pm4py.discover_dfg(log)
    perf, _, _ = pm4py.discover_performance_dfg(log)
    variants = pm4py.get_variants(log)

    top = sorted(dfg.items(), key=lambda kv: -kv[1])[:max_edges]
    var = sorted(variants.items(),
                 key=lambda kv: -(len(kv[1]) if hasattr(kv[1], "__len__") else kv[1])
                 )[:max_variants]

    return dict(
        pm4py_available=True,
        dfg_edges=[dict(source=a, target=b, frequency=int(f),
                        median_duration_seconds=_perf_value(perf.get((a, b))))
                   for (a, b), f in top],
        start_activities={str(k): int(v) for k, v in start.items()},
        end_activities={str(k): int(v) for k, v in end.items()},
        frequent_variants=[dict(variant=list(v) if isinstance(v, tuple) else str(v),
                                count=int(len(c) if hasattr(c, "__len__") else c))
                           for v, c in var],
        note="Discovered from CLEAN rows only.")

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

# ===== NB CELL 28 =====
def get_client():
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI_API_KEY: ")
    return OpenAI()

def extract_code(text):
    m = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def extract_json(text):
    m = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    raw = m.group(1) if m else text
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return json.loads(m.group(0)) if m else {}

def write_visible(df, path):
    """Exactly what a method is allowed to read."""
    cols = [CASE_COL, ACT_COL, TIME_COL, RES_COL, ERR_COL, ROW_COL]
    v = df[cols].copy()
    v[ERR_COL] = v[ERR_COL].fillna(CLEAN)
    v.to_csv(path, index=False)
    return v

def run_script(script_path, in_csv, out_csv, timeout=900):
    """UTF-8 on both sides: `-X utf8` and PYTHONUTF8 for the child, explicit
    encoding for the pipe we read its output back through."""
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(script_path),
         "--input", str(in_csv), "--output", str(out_csv)],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env=CHILD_ENV, timeout=timeout)
    return dict(returncode=proc.returncode,
                stdout=proc.stdout[-4000:], stderr=proc.stderr[-4000:])

def structural_check(visible_path, candidate_path, cfg=None):
    """Structure, then SEMANTICS.

    v4 checked only that the shape of the file survived. A program that returned
    its input unchanged passed every check and was tabulated as a repair: on
    hospital C1_10 that is exactly what M2 did. The three gates below are
    properties the task itself guarantees, so failing one is a generation
    failure, not a modelling choice, and the message is written to be useful as
    feedback on the next attempt.

    G1  a C1 block must end with k DISTINCT timestamps. It arrived with k
        identical ones; leaving them identical is not a repair.
    G2  under form-based capture the shared value is the block's own earliest
        timestamp, so exactly one row in the block must still carry it. A block
        that has drifted away from it entirely has thrown away the one value
        known to be correct. Relax C1_ANCHOR_ON_OBSERVED if the injector's
        C1_COPY_MODE is not "first".
    G3  a method may not leave almost every polluted row untouched.
    G4  the repaired order inside a C1 block must not simply be the row order.
        The rows were shuffled when the block was injected, so file order is
        random with respect to the truth. On credit_C1_30, M1 disagreed with the
        row order in 1,342 of 1,498 blocks because it had inferred an order;
        M2 disagreed twice and M3 three times, and their ordering accuracy
        equalled M0's to three decimals. They separated the blocks without
        ordering them, which is worse than leaving them collapsed -- M2 landed
        at 413 s where doing nothing lands at 214 s.
    """
    cfg = cfg or CFG
    a = pd.read_csv(visible_path)
    b = pd.read_csv(candidate_path)
    if len(a) != len(b):
        raise ValueError("row count changed: %d -> %d" % (len(a), len(b)))
    for c in [CASE_COL, ACT_COL, RES_COL, ERR_COL, ROW_COL]:
        if c not in b.columns:
            raise ValueError("missing column: %s" % c)
        if not a[c].astype(str).equals(b[c].astype(str)):
            raise ValueError("column modified: %s" % c)
    ts = pd.to_datetime(b[TIME_COL], errors="coerce")
    obs = pd.to_datetime(a[TIME_COL], errors="coerce")
    pol = a[ERR_COL].ne(CLEAN)
    if ts[~pol].isna().any():
        raise ValueError("clean timestamps became unparseable")

    # ---- G1 / G2: C1 blocks ----------------------------------------------
    v = a.copy()
    v["_rep"], v["_obs"] = ts, obs
    n_blocks = tied = adrift = in_row_order = 0
    example = None
    for _, g0 in v[v[CASE_COL].isin(v.loc[pol, CASE_COL].unique())].groupby(
            CASE_COL, sort=False):
        g = g0.reset_index(drop=True)
        for s0, e0 in _segments(g, C1):
            k = e0 - s0 + 1
            if k < 2:
                continue
            n_blocks += 1
            rep = g.loc[s0:e0, "_rep"]
            if rep.nunique(dropna=False) < k:
                tied += 1
                if example is None:
                    example = str(g.loc[s0, CASE_COL])
            shared = g.loc[s0, "_obs"]
            if pd.notna(shared) and not (rep == shared).any():
                adrift += 1
            vals = rep.tolist()
            if not any(pd.isna(x) for x in vals) and \
                    all(vals[t] <= vals[t + 1] for t in range(k - 1)):
                in_row_order += 1

    if n_blocks:
        keep = 1.0 - tied / n_blocks
        if keep < cfg.get("C1_DISTINCT_MIN_SHARE", .95):
            raise ValueError(
                "COLLAPSED BLOCKS REMAIN: %d of %d C1 blocks still contain "
                "duplicate timestamps after your repair (e.g. case %s). A C1 "
                "block arrives with k identical timestamps and must leave with "
                "k distinct ones, ordered by your inferred sequence."
                % (tied, n_blocks, example))
        if cfg.get("C1_ANCHOR_ON_OBSERVED", True):
            hold = 1.0 - adrift / n_blocks
            if hold < cfg.get("C1_ANCHOR_MIN_SHARE", .80):
                raise ValueError(
                    "OBSERVED VALUE DISCARDED: in %d of %d C1 blocks no row "
                    "kept the observed shared timestamp. That value is the "
                    "block's own earliest timestamp, so exactly one event in "
                    "the block is already correct. Place the first event of "
                    "your inferred order ON it and extend forward."
                    % (adrift, n_blocks))
        # ---- G4: the order must not be the row order --------------------
        if n_blocks >= cfg.get("C1_ROWORDER_MIN_BLOCKS", 20):
            share = in_row_order / n_blocks
            if share >= cfg.get("C1_ROWORDER_MAX_SHARE", .80):
                raise ValueError(
                    "ORDER NOT INFERRED: in %d of %d C1 blocks (%.0f%%) your "
                    "timestamps increase straight down the rows, so you "
                    "separated the blocks without ordering them. The row order "
                    "inside a corrupted block is RANDOM -- the events were "
                    "shuffled when the block was recorded -- so increasing "
                    "timestamps in row order assign a random sequence. Rank the "
                    "block's events by process behaviour first (transition "
                    "probabilities between the left anchor, the block's "
                    "activities and the right anchor), then place the "
                    "timestamps along THAT ranking, writing each event's value "
                    "back onto its own row."
                    % (in_row_order, n_blocks, 100 * share))
    # ---- G3: no-op -------------------------------------------------------
    same = ((ts[pol] == obs[pol]) | (ts[pol].isna() & obs[pol].isna()))
    noop = float(same.mean()) if pol.any() else 0.0
    if noop >= cfg.get("CODEGEN_NOOP_LIMIT", .5):
        raise ValueError(
            "NO-OP: %.0f%% of the %d corrupted rows came back with exactly the "
            "timestamp they went in with. Every row whose error_category is not "
            "CLEAN must be assigned a new value. If a validation branch in your "
            "code rejects a candidate repair, it must emit its next-best "
            "estimate; it must NOT fall back to the observed value, because the "
            "observed value is the corruption."
            % (100 * noop, int(pol.sum())))

    return b

def execute_program(script, visible_path, workdir, cfg, tag="candidate"):
    """Run an already-written program and validate it. Returns the repaired
    frame, or raises with a message suitable for feeding back to the model."""
    cand = Path(workdir) / (tag + ".csv")
    rep = run_script(Path(script), visible_path, cand)
    if rep["returncode"] != 0:
        raise RuntimeError("script exited %d\n%s" % (rep["returncode"], rep["stderr"]))
    structural_check(visible_path, cand, cfg)
    return pd.read_csv(cand)

def selection_sample(visible, cfg, path):
    """A sample used only to CHOOSE between candidate programs.

    Every case carrying a polluted row is a candidate for inclusion, capped at
    CODEGEN_SELECT_CASES; clean cases are dropped entirely. The selected
    program is then executed on the full log, so nothing in the reported
    results is computed on a sample. This exists because executing five
    candidate programs over 262k rows to pick one is the single slowest step in
    the pipeline, and the choice between them is just as visible on 300 cases.
    """
    n = int(cfg.get("CODEGEN_SELECT_CASES", 300) or 0)
    hit = visible.loc[visible[ERR_COL].ne(CLEAN), CASE_COL].unique()
    if not n or len(hit) <= n:
        visible.to_csv(path, index=False)
        return visible, False
    rng = np.random.default_rng(cfg["SEED"])
    keep = set(rng.choice(hit, size=n, replace=False).tolist())
    sub = visible[visible[CASE_COL].isin(keep)].copy()
    sub.to_csv(path, index=False)
    return sub, True

def generate_with_retry(prompt, visible_path, workdir, cfg, model=None):
    """Generate -> execute -> structurally validate; feed failures back in."""
    model = model or cfg["LLM_MODEL"]
    client = get_client()
    workdir = Path(workdir)
    script = workdir / "generated_repair.py"
    raw_p  = workdir / "llm_raw.txt"
    sample = pd.read_csv(visible_path).head(40).to_csv(index=False)

    feedback, log = "", []
    for attempt in range(1, cfg["LLM_MAX_ATTEMPTS"] + 1):
        full = (prompt
                + "\n\nHere is a sample of the input file:\n\n" + sample
                + ("\n\nPREVIOUS ATTEMPT FAILED:\n" + feedback if feedback else ""))
        resp = client.responses.create(model=model, input=full)
        text = resp.output_text
        raw_p.write_text(text)
        script.write_text(extract_code(text))
        try:
            out = execute_program(script, visible_path, workdir, cfg)
            log.append(dict(attempt=attempt, ok=True))
            return out, log
        except Exception as ex:
            feedback = "%s: %s" % (type(ex).__name__, str(ex)[:1500])
            log.append(dict(attempt=attempt, ok=False, error=feedback))
    raise RuntimeError("all %d attempts failed; last: %s" % (cfg["LLM_MAX_ATTEMPTS"], feedback))

# ===== NB CELL 30 =====
PROMPT_BASE = r"""
You are an expert in process mining, temporal event-log repair, and executable
Python program synthesis.

Input CSV columns: Case, Activity, Timestamp, Resource, error_category, row_id
Detection is already done. Repair ONLY rows whose error_category is not CLEAN.

=====================================================================
WHAT THE CORRUPTED VALUES ACTUALLY ARE
=====================================================================

Read this before designing anything. Methods fail on this task by treating the
corrupted timestamp as noise. It is not noise; it is a damaged but informative
observation, and the repair must start from it.

C1_ORDERING_AMBIGUITY
  Several consecutive events in one case carry one identical timestamp.
  That shared value is not invented: it is ONE OF THE BLOCK'S OWN TIMESTAMPS,
  kept when the others were overwritten by the batch write. So exactly one event
  in the block is ALREADY CORRECT and the rest are displaced from it.
  Therefore:
    - Do NOT rebuild the block from the left anchor by adding durations. That
      discards the one value you know to be right and drifts the whole block.
    - Decide the ordering first. Then place the block so that it STARTS at the
      observed shared timestamp and extends forward, unless the surrounding
      evidence clearly says the shared value is the block's end.
    - The first event in your inferred order keeps the observed value exactly.
    - Size the block from the room that exists between the observed value and
      the right anchor. Do NOT size it from block_span_profile: that profile is
      measured on uncorrupted consecutive events, most of which are separated by
      waiting time, whereas a form-based block by definition happened inside one
      working session. Using it to size a block, or to validate one, will make
      every block look wrong and push you into doing nothing.
    - Leaving the block collapsed is the one outcome that is always wrong. k
      identical timestamps must become k distinct ones.

C2_TEMPORAL_INCONSISTENCY
  A later event carries an earlier timestamp than its predecessor.
  Two different faults produce this, and they need different repairs. Decide
  which one you are looking at before acting:
    - EXCHANGED. The two values were swapped, so both rows are wrong and the
      pair is exactly recoverable: put each value back on the other row. The
      giveaway is that the two timestamps together still fit the surrounding
      gaps -- sorting them restores a plausible spacing.
    - DISPLACED. One row was pushed backwards by an arbitrary offset. Its
      partner is CORRECT and must not be touched. Sorting cannot recover this,
      because the true value was destroyed rather than moved elsewhere; you have
      to estimate it from the anchors and the duration statistics, exactly as
      you would for a missing value.
  Do not assume the exchanged case. Check whether swapping the pair produces
  durations consistent with the duration profiles for those transitions, and
  fall back to estimation when it does not.

C3_MISSING_TEMPORAL_INFORMATION
  Blank, or date-only with the time of day gone.
  If a date is present, KEEP THAT DATE and infer only the time of day. Never move
  the event to another day. When both anchors exist and the imputed value would
  run past the right anchor, place it between the anchors rather than pinning it
  one second before the right anchor.

=====================================================================
HARD CONSTRAINTS
=====================================================================
1. Output row count MUST equal input row count.
2. Case, Activity, Resource, error_category, row_id MUST be byte-identical.
3. Row order MUST NOT change. Express ordering through timestamp VALUES only.
4. error_category == CLEAN rows MUST keep their exact input timestamp.
5. Repaired timestamps MUST parse with pandas.to_datetime and keep the input's
   precision (milliseconds when the input has them).
6. Uniform or evenly spaced interpolation inside a block is FORBIDDEN.
7. FORBIDDEN fallbacks: 1970-01-01, the dataset minimum, a timestamp from
   another case, an elapsed-seconds value used as an absolute time.
8. Durations are RELATIVE intervals: add or subtract them from a same-case anchor.
9. For every repaired row: case_start <= t <= case_end, and when both clean
   anchors exist, left_anchor <= t <= right_anchor.

=====================================================================
REPAIR BUDGET
=====================================================================
Among repairs that are all valid, prefer the one that moves rows least. A
repair landing within seconds of the truth for most rows beats one that is
spectacular on a few and wrong by hours on the rest.

This is a tie-breaker between valid repairs. It is NOT a licence to do nothing.
Returning the observed value for a corrupted row is not the minimal repair, it
is the absence of one: the observed value IS the corruption. Every row whose
error_category is not CLEAN must leave your program with a different value from
the one it arrived with, with one exception -- the single row in each C1 block
that legitimately keeps the shared timestamp because it is the block's first
event.

=====================================================================
USING THE DURATION STATISTICS
=====================================================================
These gaps are bimodal: a fast mode (same working session, seconds) and a slow
mode (waiting for an external party, hours or days). A per-pair MEDIAN mixes the
two and is usually far too large for events inside one form-based block, which
by construction happened in one session. Use share_under_60s to tell which mode
a transition belongs to, and prefer p10/p25 for intra-block spacing.

=====================================================================
PROCESS CONTEXT (from CLEAN rows only)
=====================================================================

<<CONTEXT>>

=====================================================================
SELF-CHECK BEFORE WRITING THE OUTPUT
=====================================================================
Your program must verify its own output. When a check fails, REPAIR THE
CANDIDATE -- clamp it, rescale it, re-space it -- and never fall back to the
observed value.

  - inside every repaired C1 block the timestamps are strictly increasing and
    all distinct
  - exactly one row of each C1 block carries the observed shared timestamp, and
    it is the earliest of the block
  - the block's total span is at most the distance from the observed value to
    the right anchor, and at least (k-1) seconds
  - no repaired row sits outside its anchors
  - every corrupted row differs from its input value, except the one C1 row
    described above

Print one line to stdout with the count of blocks repaired, blocks that needed
clamping, and rows still equal to their input. If that last number is not close
to the number of C1 blocks, your program is not doing its job.

=====================================================================
THE ROW ORDER INSIDE A CORRUPTED BLOCK IS MEANINGLESS
=====================================================================
When a form-based block was recorded, the events were written in whatever order
the system happened to persist them. The rows you receive are NOT in execution
order, and treating them as if they were is the most common way to fail this
task.

If your program walks a block from its first row to its last and assigns
increasing timestamps as it goes, it has separated the block without ordering
it, and the result is a confidently wrong sequence. That is worse than leaving
the block collapsed, because the collapsed value at least sits at the true time
of the block's first event.

  1. score the candidate orderings of the block's activities against the
     transition statistics, including the left and right anchor activities
  2. take the best-scoring ordering
  3. lay the timestamps out along THAT ordering
  4. write each event's timestamp back onto ITS OWN ROW, wherever that row sits
     in the file

Step 4 keeps the row structure intact while the ordering lives in the values.
If, after your repair, every block's timestamps increase from its first row to
its last, you skipped steps 1 to 3.

=====================================================================
WORKED EXAMPLE
=====================================================================
Anchors: E4 at 09:00:00 (clean), E8 at 11:00:00 (clean).
Corrupted block, three rows, all reading 09:12:00:
    row 5  activity B
    row 6  activity D
    row 7  activity A
Transition statistics say A->B is frequent with p25 = 90 s, B->D is frequent
with p25 = 240 s, and D->A is rare.

Correct output: infer the order A, B, D. Row 7 (A) is the block's first event,
so it KEEPS 09:12:00. Row 5 (B) becomes 09:13:30 and row 6 (D) becomes
09:17:30. The span, 330 s, is well inside the 108-minute room to the right
anchor, all three values are distinct and increasing, and two of the three rows
changed.

Wrong outputs, all of which have been observed in practice: leaving all three at
09:12:00; rebuilding from 09:00:00 so that no row keeps the observed value;
spacing them evenly at 09:12, 09:47, 10:22 because a span profile said a
three-event run "should" last an hour.

=====================================================================
OUTPUT FORMAT
=====================================================================
Return ONE complete Python script in a single ```python code block, taking
--input and --output arguments. No explanation outside the code block.
"""

PM4PY_ADDENDUM = """

The process_abstraction block contains a directly-follows graph with edge
frequencies and median edge durations, start and end activities, and frequent
variants. Use it as follows, not as decoration:

  - Score a candidate ordering by whether each consecutive pair is a DFG edge,
    and by that edge's frequency. Orderings that use non-existent edges should
    be rejected before any timestamp is assigned.
  - Check the candidate order against frequent_variants: if the block's
    activities appear as a contiguous subsequence of a known variant, that
    subsequence order is strong evidence.
  - Activities that appear in both orders across variants with similar frequency
    are CONCURRENT. Their relative order is not recoverable, so do not spend
    effort on it -- assign them near-identical times instead of forcing a wide
    separation.
  - Use edge durations as spacing priors only for edges that are actually in
    the DFG.
"""

PLAN_PROMPT = r"""
You are planning a timestamp-repair program. Do NOT write code yet.

Read the task description and the process context, then return a JSON plan:

{
  "c1": {"ordering_method": "...", "span_estimator": "...",
         "placement": "start_at_observed | end_at_observed | other",
         "spacing_statistic": "p10 | p25 | median",
         "why": "one or two sentences"},
  "c2": {"strategy": "...", "why": "..."},
  "c3": {"strategy": "...", "why": "..."},
  "fallbacks": ["..."],
  "self_checks": ["..."]
}

Return only the JSON, no prose, no markdown fences.
"""

def _consistency_score(diag):
    """Rank candidate programs without ground truth. Lower is better.

    Every term is measured against the CLEAN part of the same log, so this is
    model selection on process consistency, not on the test score. The
    distinction matters: picking the best of N runs by MedAE is choosing on the
    answer key, and `result.xlsx` shows what that does to a results table.
    """
    return (round(diag.get("noop_rate", 0.0), 2),
            len(diag.get("issues", [])),
            diag.get("blocks_still_tied", 0),
            diag.get("blocks_with_implausible_span", 0),
            diag.get("monotonicity_violations", 0),
            diag.get("median_displacement_seconds") or 0.0)

_PROGRAM_CACHE = {}

def _prog_key(use_pm4py, dataset, cfg):
    scope = cfg.get("CODEGEN_CACHE_SCOPE", "log")
    if scope == "off":
        return None
    variant = "pm4py" if use_pm4py else "plain"
    if scope == "global":
        # One program for EVERY dataset of EVERY log. This is M2's claim: the
        # program is a general repair procedure, so it is synthesized once and
        # reused everywhere, and regenerated only where it cannot execute. The
        # trial number is part of the key so the N trials stay independent
        # universal programs rather than one program with N scores.
        return ("GLOBAL", "t%s" % cfg.get("CODEGEN_CACHE_TRIAL", 0), variant)
    return (ACTIVE, variant) if scope == "log" else (dataset, variant)


def _cache_path(key, cfg):
    """Where a cached program lives between processes.

    A global-scope run covers 15 datasets and takes hours; without this the
    program would be re-synthesized whenever the run is split or resumed, and
    "one program for every dataset" would stop being true.
    """
    d = cfg.get("CODEGEN_CACHE_DIR")
    if not d or key is None:
        return None
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p / ("prog_" + "_".join(str(x) for x in key) + ".json")


def _load_program_cache(key, cfg):
    p = _cache_path(key, cfg)
    if p is None or not p.exists():
        return False
    try:
        entry = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not entry.get("source"):
        return False
    _PROGRAM_CACHE[key] = entry
    print("  loaded the program synthesized on %s from disk"
          % entry.get("dataset", "?"))
    return True


def _save_program_cache(key, cfg):
    p = _cache_path(key, cfg)
    if p is None:
        return
    try:
        p.write_text(json.dumps(_PROGRAM_CACHE[key], indent=1),
                     encoding="utf-8")
    except Exception as ex:
        print("  could not persist the program cache: %s" % str(ex)[:120])

def _make_codegen_core(use_pm4py, two_stage=True, refine_rounds=1,
                       n_candidates=None):
    """Program synthesis.

    v6 synthesizes once per log and reuses the program across that log's
    datasets. The program is a general repair procedure that reads the log it
    is given; it was never specific to a pollution rate, and regenerating it
    nine times per log was paying nine times for the same artifact. The paper
    should state this plainly: one program per log per method, applied to every
    dataset of that log, which is also what the previous study did.

    Set CODEGEN_CACHE_SCOPE="dataset" to restore the v5 behaviour, e.g. for an
    ablation on whether per-dataset synthesis helps.
    """
    n_candidates = n_candidates or CFG.get("CODEGEN_N_CANDIDATES", 1)

    def method(visible, ctx, cfg, workdir):
        t0 = time.time()
        workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
        dataset = workdir.parent.name
        vpath = workdir / "visible_input.csv"
        visible.to_csv(vpath, index=False)
        key = _prog_key(use_pm4py, dataset, cfg)
        if key is not None and key not in _PROGRAM_CACHE:
            _load_program_cache(key, cfg)

        # ---- reuse ----------------------------------------------------
        if key is not None and key in _PROGRAM_CACHE:
            entry = _PROGRAM_CACHE[key]
            script = workdir / "generated_repair.py"
            script.write_text(entry["source"])
            try:
                out = execute_program(script, vpath, workdir, cfg)
                (workdir / "reused_from.json").write_text(json.dumps(
                    dict(generated_on=entry["dataset"], key=list(key),
                         scope=cfg.get("CODEGEN_CACHE_SCOPE"),
                         regenerated=False,
                         exec_seconds=round(time.time() - t0, 1)), indent=2))
                print("  reused the program synthesized on %s (%.0fs)"
                      % (entry["dataset"], time.time() - t0))
                return out
            except Exception as ex:
                # The ONLY sanctioned reason to synthesize a second program
                # under global scope: the universal one will not execute here.
                # Recorded so the paper can report how often it happened.
                (workdir / "reused_from.json").write_text(json.dumps(
                    dict(generated_on=entry["dataset"], key=list(key),
                         scope=cfg.get("CODEGEN_CACHE_SCOPE"),
                         regenerated=True,
                         regenerated_because=str(ex)[:400]), indent=2))
                print("  cached program failed on %s (%s); regenerating"
                      % (dataset, str(ex)[:120]))

        context = context_to_json(ctx)
        if use_pm4py:
            context = dict(context, process_abstraction=ctx.get("pm4py", {}))
        prompt = PROMPT_BASE.replace("<<CONTEXT>>",
                                     json.dumps(context, indent=2, default=str))
        if use_pm4py:
            prompt += PM4PY_ADDENDUM

        # ---- stage 1: plan ----
        if two_stage:
            client = get_client()
            plan_in = PLAN_PROMPT + "\n\nTASK AND CONTEXT:\n" + prompt
            plan_raw = client.responses.create(model=cfg["LLM_MODEL"],
                                               input=plan_in).output_text
            try:
                plan = extract_json(plan_raw)
            except Exception:
                plan = {"raw": plan_raw[:2000]}
            (workdir / "plan.json").write_text(json.dumps(plan, indent=2))
            prompt += ("\n\n=====================================================\n"
                       "YOUR OWN PLAN - implement exactly this\n"
                       "=====================================================\n"
                       + json.dumps(plan, indent=2))
        (workdir / "prompt.txt").write_text(prompt)

        # ---- stage 2: N candidates, scored on a sample -----------------
        spath = workdir / "selection_sample.csv"
        sample_vis, sampled = selection_sample(visible, cfg, spath)
        if sampled:
            print("  selecting on %d of %d cases"
                  % (sample_vis[CASE_COL].nunique(), visible[CASE_COL].nunique()))

        def _one(c):
            cdir = workdir / ("cand_%d" % c)
            cdir.mkdir(exist_ok=True)
            out_c, log_c = generate_with_retry(prompt, spath, cdir, cfg)
            diag = plausibility_report(sample_vis, out_c, ctx, cfg)
            (cdir / "diagnostics.json").write_text(
                json.dumps(diag, indent=2, default=str))
            return c, cdir, diag, log_c

        t_gen = time.time()
        results, log = [], []
        idx = list(range(int(n_candidates)))
        if cfg.get("CODEGEN_PARALLEL", True) and len(idx) > 1:
            with ThreadPoolExecutor(max_workers=len(idx)) as ex:
                futs = {ex.submit(_one, c): c for c in idx}
                for f in as_completed(futs):
                    try:
                        results.append(f.result())
                    except Exception as e:
                        log.append(dict(candidate=futs[f], ok=False,
                                        error=str(e)[:400]))
        else:
            for c in idx:
                try:
                    results.append(_one(c))
                except Exception as e:
                    log.append(dict(candidate=c, ok=False, error=str(e)[:400]))
        t_gen = time.time() - t_gen
        if not results:
            raise RuntimeError("all %d candidate programs failed" % n_candidates)

        for c, cdir, diag, log_c in results:
            log.append(dict(candidate=c, ok=True, attempts=log_c,
                            score=list(_consistency_score(diag)),
                            issues=diag["issues"]))
        results.sort(key=lambda t: _consistency_score(t[2]))
        best_c, best_dir, diag, _ = results[0]
        (workdir / "selection.json").write_text(json.dumps(
            dict(chosen=best_c, n_candidates=int(n_candidates),
                 selected_on_sample=bool(sampled),
                 generate_seconds=round(t_gen, 1),
                 scores={str(c): list(_consistency_score(d))
                         for c, _, d, _ in results}), indent=2, default=str))

        script = workdir / "generated_repair.py"
        script.write_text((best_dir / "generated_repair.py").read_text())
        for f in ("llm_raw.txt",):
            p = best_dir / f
            if p.exists():
                (workdir / f).write_text(p.read_text())

        # ---- stage 3: refinement, still on the sample ------------------
        for rnd in range(refine_rounds):
            (workdir / ("diagnostics_%d.json" % rnd)).write_text(
                json.dumps(diag, indent=2, default=str))
            if diag["verdict"] == "ok":
                break
            fb = ("\n\n=====================================================\n"
                  "DIAGNOSTICS ON YOUR PREVIOUS OUTPUT - fix these\n"
                  "=====================================================\n"
                  + json.dumps(diag, indent=2, default=str)
                  + "\nRewrite the complete program addressing every issue above. "
                    "Do not respond by repairing fewer rows.")
            try:
                # parentheses: `/` binds tighter than `%`, so
                # `workdir / "refine_%d" % rnd` is (Path) % int
                rdir = workdir / ("refine_%d" % rnd)
                rdir.mkdir(exist_ok=True)
                out2, _ = generate_with_retry(prompt + fb, spath, rdir, cfg)
                diag2 = plausibility_report(sample_vis, out2, ctx, cfg)
                if _consistency_score(diag2) <= _consistency_score(diag):
                    diag = diag2
                    script.write_text((rdir / "generated_repair.py").read_text())
                else:
                    log.append(dict(refine=rnd, kept="previous"))
            except Exception as ex:
                print("  refinement round %d failed, keeping previous: %s" % (rnd, ex))
                break

        # ---- the selected program on the FULL log ----------------------
        t_exec = time.time()
        out = execute_program(script, vpath, workdir, cfg)
        t_exec = time.time() - t_exec

        if key is not None:
            _PROGRAM_CACHE[key] = dict(source=script.read_text(), dataset=dataset)
            _save_program_cache(key, cfg)
        log.append(dict(generate_seconds=round(t_gen, 1),
                        full_exec_seconds=round(t_exec, 1),
                        total_seconds=round(time.time() - t0, 1)))
        (workdir / "generation_log.json").write_text(json.dumps(log, indent=2,
                                                                default=str))
        print("  synthesized in %.0fs, full-log execution %.0fs"
              % (t_gen, t_exec))
        return out
    return method

def plausibility_report(visible, repaired, ctx, cfg):
    """No ground truth is used. Everything here compares the repaired log with
    the CLEAN part of the same log, so it is legitimate feedback to the model."""
    v = visible.reset_index(drop=True).copy()
    r = repaired.reset_index(drop=True).copy()
    obs = pd.to_datetime(v[TIME_COL], errors="coerce")
    rep = pd.to_datetime(r[TIME_COL], errors="coerce")
    pol = v[ERR_COL].ne(CLEAN)
    span = ctx["span_profile"]

    issues, block_bad, moves = [], 0, []
    v["_rep"], v["_obs"] = rep, obs
    hit = v.loc[pol, CASE_COL].unique()
    n_blocks = 0
    for _, g0 in v[v[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
        g = g0.reset_index(drop=True)
        for s, e in _segments(g, C1):
            n_blocks += 1
            k = e - s + 1
            vals = g.loc[s:e, "_rep"].dropna()
            if len(vals) < 2:
                continue
            sp = (vals.max() - vals.min()).total_seconds()
            ref = span.get(k, span.get(2, {}))
            if ref and not (ref["p10"] <= sp <= ref["p90"]):
                block_bad += 1
            moves.append((vals - g.loc[s:e, "_obs"]).dt.total_seconds().abs().max())

    # F4: on hospital_C1_10 the generated program returned its input unchanged
    # and every downstream number treated that as a repair. A program that
    # declines to act is a failed generation, and the refinement round is where
    # it has to be caught.
    same = ((rep[pol] == obs[pol]) | (rep[pol].isna() & obs[pol].isna()))
    noop_rate = float(same.mean()) if pol.any() else 0.0

    d = (rep[pol] - obs[pol]).dt.total_seconds().abs().dropna()
    still_tied = 0
    for _, g0 in v[v[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
        g = g0.reset_index(drop=True)
        for s, e in _segments(g, C1):
            if g.loc[s:e, "_rep"].nunique() < (e - s + 1):
                still_tied += 1

    gaps = (r.assign(_t=rep).sort_values([CASE_COL], kind="mergesort")
             .groupby(CASE_COL)["_t"].diff().dt.total_seconds())
    mono = int((gaps < 0).sum())

    if noop_rate >= cfg.get("CODEGEN_NOOP_LIMIT", .5):
        issues.append(
            "Your program left %.0f%% of the corrupted rows exactly as it found "
            "them. Re-read the task: every row whose error_category is not "
            "CLEAN must be assigned a new timestamp. If a branch of your code "
            "cannot decide what to do, it must still emit its best estimate "
            "rather than falling through and returning the input value."
            % (100 * noop_rate))
    if n_blocks and block_bad / n_blocks > .25:
        issues.append("%.0f%% of repaired C1 blocks have a span outside the "
                      "p10-p90 range of block_span_profile. The spacing "
                      "statistic is wrong for this log." % (100 * block_bad / n_blocks))
    if n_blocks and still_tied / n_blocks > .05:
        issues.append("%d C1 blocks still contain duplicate timestamps after "
                      "repair." % still_tied)
    if mono:
        issues.append("%d negative durations remain after repair." % mono)
    if len(d) and d.median() > 3600:
        issues.append("The median repaired row moved %.0f seconds from its "
                      "observed value. Under form-based capture the observed "
                      "value is one of the block's own timestamps, so most rows "
                      "should move much less." % d.median())

    return dict(verdict="ok" if not issues else "revise",
                issues=issues,
                noop_rate=noop_rate,
                n_c1_blocks=n_blocks,
                blocks_with_implausible_span=block_bad,
                blocks_still_tied=still_tied,
                monotonicity_violations=mono,
                median_displacement_seconds=float(d.median()) if len(d) else None,
                p90_displacement_seconds=float(d.quantile(.9)) if len(d) else None)

_SYNTH_BUDGET = {}

def make_codegen_method(use_pm4py, two_stage=True, refine_rounds=1,
                        n_candidates=None):
    def method(visible, ctx, cfg, workdir):
        # built per call so CFG changes made after this cell take effect
        inner = _make_codegen_core(
            use_pm4py, two_stage, refine_rounds,
            n_candidates or cfg.get("CODEGEN_N_CANDIDATES", 1))
        dataset = Path(workdir).parent.name
        key = _prog_key(use_pm4py, dataset, cfg)
        if key is None:
            return inner(visible, ctx, cfg, workdir)

        cached = _PROGRAM_CACHE.get(key)
        if isinstance(cached, dict) and cached.get("failed"):
            raise RuntimeError(
                "synthesis for %s already failed %d times on this log (first "
                "failure on %s); not spending another generation. Last error: %s"
                % (key[1], cached["attempts"], cached["dataset"],
                   cached["error"][:300]))

        try:
            return inner(visible, ctx, cfg, workdir)
        except Exception as ex:
            used = _SYNTH_BUDGET.get(key, 0) + 1
            _SYNTH_BUDGET[key] = used
            budget = cfg.get("CODEGEN_SYNTH_BUDGET", 2)
            if used >= budget and key not in _PROGRAM_CACHE:
                _PROGRAM_CACHE[key] = dict(failed=True, attempts=used,
                                           dataset=dataset, error=str(ex))
                print("  synthesis budget for %s exhausted (%d/%d); the "
                      "remaining datasets of this log will fail immediately"
                      % (key[1], used, budget))
            raise
    return method

m2_codegen = make_codegen_method(use_pm4py=False)

m3_codegen_pm4py = make_codegen_method(use_pm4py=True)

# ===== NB CELL 34 =====
DIRECT_PROMPT = r"""
You are an expert in process mining and temporal event-log timestamp repair.

You repair corrupted event-log segments directly. You are NOT writing code.

=====================================================================
WHY YOU ANSWER IN OFFSETS
=====================================================================
You never write an absolute timestamp. For each corrupted row you return an
integer POSITION in the segment's inferred order and an OFFSET IN SECONDS from
that segment's reference_timestamp, which is given to you. The caller builds the
timestamp. This removes every failure mode that comes from writing dates by
hand: wrong day, wrong year, lost milliseconds, a value from another case, an
elapsed-seconds number mistaken for an absolute time.

Return ONLY valid JSON, no markdown, no prose:

{"repairs": [{"segment_id": 3, "row_id": 123, "position": 1,
              "offset_seconds": 0, "reason": "short reason"}]}

- exactly one object for every row_id in corrupted_rows of every segment
- position is 1..k inside its segment, each value used once
- offset_seconds is a non-negative integer, measured from reference_timestamp
- offsets must be strictly increasing with position, and all distinct
- offset_seconds must not exceed max_offset_seconds for that segment

=====================================================================
WHAT THE CORRUPTED VALUES ARE
=====================================================================
C1_ORDERING_AMBIGUITY
  k consecutive events collapsed onto ONE timestamp. That timestamp is not
  invented: it is the block's own earliest value, so the event you rank first is
  already correct. reference_timestamp is that value, so the row you give
  position 1 must get offset_seconds = 0, and the others get positive offsets.
  The rows are NOT in execution order -- they were shuffled when the block
  was recorded. Infer the order from the transition statistics supplied
  with the segment, and do not assume position 1 is the first row listed.
  If your positions come out as 1,2,3,... in the order the rows are
  given, you have not ordered anything.
  Never leave the block collapsed: every offset in a block must differ.
  Never space a block evenly.

C2_TEMPORAL_INCONSISTENCY
  Either two adjacent values were EXCHANGED -- both rows wrong, and swapping
  them back is exact -- or one row was DISPLACED backwards while its partner
  stayed correct. Check whether swapping produces durations consistent with the
  segment's transition statistics; if it does not, estimate the displaced value
  from the anchors instead. reference_timestamp is the left anchor.

C3_MISSING_TEMPORAL_INFORMATION
  Blank, or date-only. When the date survives, reference_timestamp is midnight
  of that date and max_offset_seconds is 86399: keep the day and infer only the
  time of day. When the value is blank, reference_timestamp is the left anchor.

=====================================================================
HOW TO CHOOSE AN OFFSET
=====================================================================
Each segment carries transition_stats for the activity pairs that actually occur
in it: n, p10, p25, median, p75, p90 and share_under_60s. These gaps are
bimodal. A high share_under_60s means the transition normally happens inside one
working session, so use p10 or p25. A low one means it normally waits, so the
median is the right scale. A per-pair median applied inside a C1 block is
usually far too large, because the block collapsed precisely because those
events happened together.

Sanity: the sum of the gaps you assign inside a segment must fit in
max_offset_seconds. If your first choice does not fit, scale the gaps down
proportionally rather than truncating the last one.
"""

def _seg_stats(ctx, acts, max_pairs=None):
    """Duration statistics for the transitions THIS segment can use.

    v4 sent a global top-40 list, which on a log with 289 activities contains
    almost nothing relevant to any given segment. The model then had no basis
    for a spacing decision and fell back on whatever it had seen in the
    examples.
    """
    prof = ctx["dur_profile"]
    max_pairs = max_pairs or CFG.get("SEG_STATS_MAX", 6)
    out = []
    for a in acts:
        for b in acts:
            if a == b or (a, b) not in prof.index:
                continue
            r = prof.loc[(a, b)]
            # whole seconds and short keys: at chunk=100 the payload is the
            # binding constraint, and nobody needs 9 decimal places on a
            # duration quantile
            out.append(dict(src=a, tgt=b, n=int(r.n),
                            p10=int(r.p10), p25=int(r.p25),
                            med=int(r["median"]), p90=int(r.p90),
                            burst=round(float(r.burst_share), 2),
                            prob=round(float(
                                ctx["trans_prob"].get((a, b), 0.0)), 3)))
    out.sort(key=lambda d: -d["n"])
    return out[:max_pairs]

def build_segments(visible, ctx, cfg):
    """One segment per contiguous run of corrupted rows, with both anchors, the
    reference the offsets are measured from, the room available, and the
    transition statistics for its own activities."""
    v = visible.copy()
    v["_ts"] = pd.to_datetime(v[TIME_COL], errors="coerce")
    segs = []
    hit = v.loc[v[ERR_COL].ne(CLEAN), CASE_COL].unique()
    sid = 0
    for case, g0 in v[v[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
        g = g0.reset_index(drop=True)
        cs, ce = g["_ts"].min(), g["_ts"].max()
        for cat in CATEGORIES:
            for s, e in _segments(g, cat):
                li, ri = _anchors(g, s, e)
                lt = g.loc[li, "_ts"] if li is not None else None
                rt = g.loc[ri, "_ts"] if ri is not None else None
                obs0 = g.loc[s, "_ts"]

                if cat == C1 and pd.notna(obs0):
                    ref, cap = obs0, rt
                elif cat == C3 and pd.notna(obs0) and obs0.normalize() == obs0:
                    ref, cap = obs0, obs0 + pd.Timedelta(seconds=86399)
                elif lt is not None and pd.notna(lt):
                    ref, cap = lt, rt
                else:
                    ref, cap = (obs0 if pd.notna(obs0) else cs), rt
                if cap is not None and pd.notna(cap) and pd.notna(ref):
                    room = max(1, int((cap - ref).total_seconds()))
                else:
                    room = int(cfg["FALLBACK_DURATION"] * 60)

                # note the parentheses: `x if c else [] + y` parses as
                # `x if c else ([] + y)`, which silently reduced this list to
                # the left anchor alone and left transition_stats empty
                acts = list(dict.fromkeys(
                    ([str(g.loc[li, ACT_COL])] if li is not None else [])
                    + [str(g.loc[k, ACT_COL]) for k in range(s, e + 1)]
                    + ([str(g.loc[ri, ACT_COL])] if ri is not None else [])))
                anc = lambda j: (None if j is None else
                                 dict(activity=str(g.loc[j, ACT_COL]),
                                      timestamp=_fmt(g.loc[j, "_ts"])))
                segs.append(dict(
                    segment_id=sid, case_id=str(case), error_category=cat,
                    case_start=_fmt(cs), case_end=_fmt(ce),
                    left_clean_anchor=anc(li), right_clean_anchor=anc(ri),
                    reference_timestamp=_fmt(ref),
                    max_offset_seconds=room,
                    transition_stats=_seg_stats(ctx, acts),
                    corrupted_rows=[dict(row_id=int(g.loc[k, ROW_COL]),
                                         activity=str(g.loc[k, ACT_COL]),
                                         resource=str(g.loc[k, RES_COL]),
                                         observed_timestamp=_fmt(g.loc[k, "_ts"]))
                                    for k in range(s, e + 1)]))
                sid += 1
    return segs

def _direct_call(batch, cfg, client):
    """One API call for one list of segments. Returns a list of repairs or
    raises. The payload is compact JSON: at chunk=100 an indented payload with
    full-precision statistics is several hundred kilobytes."""
    payload = json.dumps(dict(segments=batch), separators=(",", ":"),
                         ensure_ascii=False)
    resp = client.responses.create(
        model=cfg["LLM_MODEL"],
        input=DIRECT_PROMPT + "\n\nINPUT JSON:\n" + payload)
    got = extract_json(resp.output_text).get("repairs", [])
    if not got:
        raise ValueError("no repairs in response")
    return got, resp.output_text

def _direct_chunk(batch, cfg, client, depth=0):
    """Call, and on failure split the chunk and retry the halves.

    This is what makes a large DIRECT_CHUNK_SIZE safe. A big chunk is cheaper
    per segment and faster in wall-clock, but one malformed or truncated
    response used to lose every segment in it silently; now the failure costs
    two smaller calls instead of 100 unrepaired rows.
    """
    for attempt in range(cfg.get("DIRECT_MAX_ATTEMPTS", 2)):
        try:
            return _direct_call(batch, cfg, client)
        except Exception as ex:
            err = str(ex)[:120]
    if len(batch) > cfg.get("DIRECT_MIN_CHUNK", 5) and depth < 4:
        mid = len(batch) // 2
        print("    chunk of %d failed (%s); splitting" % (len(batch), err))
        a = _direct_chunk(batch[:mid], cfg, client, depth + 1)
        b = _direct_chunk(batch[mid:], cfg, client, depth + 1)
        return (a[0] + b[0], a[1] + "\n" + b[1])
    print("    giving up on %d segments: %s" % (len(batch), err))
    return [], ""

def m4_direct(visible, ctx, cfg, workdir):
    """Offsets in, timestamps built here. Chunks run in parallel: they are
    independent API calls that only ever waited on each other."""
    t0 = time.time()
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    client = get_client()
    segs = build_segments(visible, ctx, cfg)
    by_id = {s["segment_id"]: s for s in segs}
    n = cfg["DIRECT_CHUNK_SIZE"]
    chunks = [segs[i:i + n] for i in range(0, len(segs), n)]
    if chunks:
        print("  %d segments -> %d chunks of <=%d, %d in parallel"
              % (len(segs), len(chunks), n, cfg.get("DIRECT_MAX_WORKERS", 4)))

    repairs, raws = {}, []
    workers = max(1, int(cfg.get("DIRECT_MAX_WORKERS", 4)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_direct_chunk, ch, cfg, client) for ch in chunks]
        for f in as_completed(futs):
            got, raw = f.result()
            raws.append(raw)
            for r in got:
                try:
                    seg = by_id[int(r["segment_id"])]
                    off = min(max(0, int(float(r["offset_seconds"]))),
                              seg["max_offset_seconds"])
                    ref = pd.to_datetime(seg["reference_timestamp"],
                                         errors="coerce")
                    if pd.isna(ref):
                        continue
                    repairs[int(r["row_id"])] = (
                        ref + pd.Timedelta(seconds=off)).round("ms")
                except Exception:
                    continue

    (workdir / "raw_outputs.jsonl").write_text(
        "\n".join(json.dumps({"raw": r}) for r in raws))
    (workdir / "segments.json").write_text(json.dumps(segs, indent=2, default=str))

    out = visible.copy()
    ts = pd.to_datetime(out[TIME_COL], errors="coerce").astype("datetime64[ns]")
    mask = out[ERR_COL].ne(CLEAN)
    for k, row_id in enumerate(out[ROW_COL]):
        if mask.iloc[k] and int(row_id) in repairs:
            ts.iloc[k] = repairs[int(row_id)]
    out[TIME_COL] = ts.dt.strftime(cfg["TS_FORMAT"])
    cov = len(repairs) / max(int(mask.sum()), 1)
    print("  direct: %d/%d rows repaired (%.1f%%) in %.0fs"
          % (len(repairs), int(mask.sum()), 100 * cov, time.time() - t0))
    if cov < .5:
        print("  WARNING: coverage below 50%; this run is closer to a no-op "
              "than to a repair and should not be tabulated as one")
    return out

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