# -*- coding: utf-8 -*-
"""Per-category LLM experiment: M2 / M3 / M4 with one prompt per error category.

Differences from the run that produced results_all_logs.csv, and nothing else:

  1. PROMPT  one prompt per category instead of one prompt covering all three.
             Built by splitting the original (see prompts_percat.py).
  2. CACHE   CODEGEN_CACHE_SCOPE = "off". The previous run synthesized ONE
             program per log, seeded on MIX_30, and reused it on every dataset
             of that log -- so "M2 on credit_C1_30" was in fact the MIX_30
             program applied to C1. Verified from reused_from.json: 14 of 18
             (log, category, method) cells reused a MIX_30 program.
  3. TRIALS  3 independent syntheses per (dataset, method) for M2 and M3, each
             executed and scored separately, so M2-vs-M3 is no longer one draw
             against one draw. CODEGEN_N_CANDIDATES = 1 inside each trial, so a
             trial is one program, not a best-of-N pick.
  4. DATA    single-category datasets only: *_C1_30, *_C2_30, *_C3_30.

Everything else -- context construction, the symbolic enforcement layer, the
structural gates G1-G4, the metrics -- is the notebook's own code, imported
from llm.py, so M1 and B1-B4 remain directly comparable.

Usage:
    python run_llm_percat.py --list-models
    python run_llm_percat.py --dry-run
    python run_llm_percat.py --model <name>
    python run_llm_percat.py --model <name> --logs credit --methods M2
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import llm as N                                   # noqa: E402
import prompts_percat as P                           # noqa: E402
import prompts_v2 as P2                              # noqa: E402
import improve                                       # noqa: E402

improve.install()
LESSONS = improve.LessonBook()

# every gate rejection and retry failure, drained once per trial
_GATE_LOG = []
_BASE_GENERATE = N.generate_with_retry


def generate_with_retry_logged(prompt, visible_path, workdir, cfg, model=None):
    try:
        out, log = _BASE_GENERATE(prompt, visible_path, workdir, cfg, model)
    except Exception as ex:
        _GATE_LOG.append(str(ex))
        raise
    for a in log:
        if not a.get("ok") and a.get("error"):
            _GATE_LOG.append(a["error"])
    return out, log


N.generate_with_retry = generate_with_retry_logged

ROOT = Path(os.environ.get("TSREPAIR_ROOT", "."))
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def load_key():
    """Make OPENAI_API_KEY visible to this process.

    `setx` writes the User environment scope, but a process that was already
    running when it was set keeps the old environment and passes that to its
    children -- so the key is stored and still invisible here. Read it straight
    out of the User scope instead of asking for another app restart. The value
    is never printed.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return True
    if os.name != "nt":
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            val, _ = winreg.QueryValueEx(k, "OPENAI_API_KEY")
        if val:
            os.environ["OPENAI_API_KEY"] = val
            return True
    except OSError:
        pass
    return False

LOGS = ["credit", "pub", "hospital", "bpic12", "bpic15"]
CATS = ["C1", "C2", "C3"]
METHODS = ["M2", "M3", "M4"]
TRIALS = {"M2": 5, "M3": 5, "M4": 1}

# M2 and M3 differ in THREE ways, by design, and the paper has to say so:
#
#   M2  one universal program. Synthesized once per trial, on the first
#       dataset of the run, and executed unchanged on every other dataset of
#       every log. It never sees a second dataset's statistics, so its prompt
#       must describe all three error categories -- a C1-only prompt cannot
#       yield a program that also handles C2 and C3. It is regenerated only
#       where it fails to execute, and every such event is recorded in that
#       cell's reused_from.json.
#
#   M3  one program per dataset, with the PM4Py process abstraction in the
#       context and the category-specific prompt, as before.
#
# So M2 tests whether a single synthesized procedure generalizes across logs,
# and M3 tests what per-dataset synthesis buys on top of that.
M2_SCOPE, M3_SCOPE = "global", "off"

_CTX, _PM4PY = {}, {}

# the notebook's originals, captured before any per-category patching, so the
# control arm is byte-identical to what the previous run sent
_UNIFIED_CODEGEN = N.PROMPT_BASE
_UNIFIED_DIRECT = N.DIRECT_PROMPT


# ---------------------------------------------------------------- gate G5
_BASE_STRUCTURAL_CHECK = N.structural_check


def structural_check_g5(visible_path, candidate_path, cfg=None):
    """G1-G4 as the notebook defines them, plus G5: COVERAGE.

    G3 rejects a program that leaves most corrupted rows untouched, at
    CODEGEN_NOOP_LIMIT = 0.50. That threshold was tuned on MIX datasets. On a
    C3-only dataset, leaving every blanked timestamp blank is a no-op rate of
    738/1500 = 0.492 -- just under the gate, so it passes. The row set then
    shrinks to the 762 date-only rows and the cell's MedAE is no longer
    comparable to anything.

    The C3 prompt already states "every corrupted row has a parseable
    timestamp; none is left blank". G5 enforces that stated requirement, the
    same way G1-G4 enforce theirs, and its message is written to be useful as
    feedback on the next attempt.
    """
    b = _BASE_STRUCTURAL_CHECK(visible_path, candidate_path, cfg)
    cfg = cfg or N.CFG
    a = pd.read_csv(visible_path)
    pol = a[N.ERR_COL].ne(N.CLEAN)
    if pol.any():
        ts = pd.to_datetime(b[N.TIME_COL], errors="coerce")
        blank = int(ts[pol].isna().sum())
        if blank:
            raise ValueError(
                "ROWS LEFT UNFILLED: %d of %d corrupted rows came back with no "
                "parseable timestamp. A row you decline to fill is not scored "
                "at all, so declining is not a neutral choice -- it removes the "
                "row from the evaluation. Every row whose error_category is not "
                "CLEAN must leave your program with a parseable timestamp, "
                "including rows that arrived blank. Estimate from the left "
                "anchor plus the transition statistic and clamp to the right "
                "anchor." % (blank, int(pol.sum())))
    return b


N.structural_check = structural_check_g5


# ---------------------------------------------------------------- context
def context_for(log, dataset):
    if dataset in _CTX:
        return _CTX[dataset]
    orig = pd.read_csv(ROOT / log / "datasets" / (dataset + ".csv"))
    orig[N.ERR_COL] = orig[N.ERR_COL].fillna(N.CLEAN)
    N.CFG["TS_FORMAT"] = N.LOGS[log]["ts_format"]
    _CTX[dataset] = (orig, N.build_context(orig, N.CFG))
    return _CTX[dataset]


def pm4py_for(dataset, orig):
    if dataset not in _PM4PY:
        _PM4PY[dataset] = N.build_pm4py_context(orig)
    return _PM4PY[dataset]


# ---------------------------------------------------------------- scoring
def score(orig, final, extra):
    pol = orig[N.ERR_COL].ne(N.CLEAN)
    rows = []
    base = dict(extra)
    base.update(N.ordering_metrics(orig, final))
    for tag, mask in ([("ALL_POLLUTED", pol)]
                      + [(c, orig[N.ERR_COL].eq(c)) for c in N.CATEGORIES]):
        if not mask.any():
            continue
        r = dict(base)
        r.update(N.core_metrics(orig, final, N.CFG, mask, tag))
        # A row the method declined to fill is not scored at all, so a cell
        # with coverage < 1 was scored on a different row set from M1's and
        # the two MedAE values are not comparable. run_one() carries this
        # flag; the pilot showed why it matters -- three of four C3 cells
        # came back at coverage 0.508, scored on the 762 date-only rows while
        # the 738 blanks were left blank.
        r["comparable"] = bool(r.get("coverage", 1.0) >= .999)
        rows.append(r)
    return rows


# ---------------------------------------------------------------- one cell
args_ns = None


def run_cell(log, cat, method, trial, model, dry=False, arm="per_category",
             lessons=True):
    dataset = "%s_%s_30" % (log, cat)
    orig, ctx = context_for(log, dataset)
    N.CFG["TS_FORMAT"] = N.LOGS[log]["ts_format"]
    N.CFG["LLM_MODEL"] = model
    N.CFG["CODEGEN_N_CANDIDATES"] = 1
    if method == "M2":
        N.CFG["CODEGEN_CACHE_SCOPE"] = M2_SCOPE
        N.CFG["CODEGEN_CACHE_TRIAL"] = trial
        N.CFG["CODEGEN_CACHE_DIR"] = str(OUT / "program_cache" / arm)
    else:
        N.CFG["CODEGEN_CACHE_SCOPE"] = M3_SCOPE
        N.CFG["CODEGEN_CACHE_DIR"] = None

    mname = {"M2": "M2_codegen", "M3": "M3_codegen_pm4py",
             "M4": "M4_direct"}[method]
    wd = OUT / "runs" / arm / dataset / ("%s_t%d" % (mname, trial))
    wd.mkdir(parents=True, exist_ok=True)
    visible = N.write_visible(orig, wd / "visible_input.csv")

    if method == "M4":
        prompt = (P2.direct_prompt(cat) if arm == "authored"
                  else P.direct_prompt(cat) if arm == "per_category"
                  else _UNIFIED_DIRECT)
        (wd / "prompt.txt").write_text(prompt)
        if dry:
            segs = N.build_segments(visible, ctx, N.CFG)
            n = N.CFG["DIRECT_CHUNK_SIZE"]
            payload = json.dumps(dict(segments=segs[:n]), separators=(",", ":"))
            return dict(dataset=dataset, method=mname, trial=trial,
                        segments=len(segs), chunks=-(-len(segs) // n),
                        prompt_chars=len(prompt),
                        first_chunk_chars=len(payload))
        N.DIRECT_PROMPT = prompt
        cand = N.m4_direct(visible, ctx, N.CFG, wd)
    else:
        use_pm4py = (method == "M3")
        if method == "M2":
            # one program for every dataset -> it must be told about every
            # error category, so M2 always takes the unified prompt
            prompt = _UNIFIED_CODEGEN
        else:
            prompt = (P2.codegen_prompt(cat) if arm == "authored"
                      else P.codegen_prompt(cat) if arm == "per_category"
                      else _UNIFIED_CODEGEN)
        if getattr(args_ns, "addendum", False):
            prompt = prompt + P.PRECISION_ADDENDUM
        (wd / "prompt.txt").write_text(prompt)
        if dry:
            c = N.context_to_json(ctx)
            if use_pm4py:
                c = dict(c, process_abstraction=pm4py_for(dataset, orig))
            filled = prompt.replace("<<CONTEXT>>",
                                    json.dumps(c, indent=2, default=str))
            return dict(dataset=dataset, method=mname, trial=trial,
                        prompt_chars=len(prompt),
                        filled_prompt_chars=len(filled))
        if lessons:
            prompt = prompt + LESSONS.block_for(
                LESSONS.key(log, cat, method, arm))
        (wd / "prompt_with_lessons.txt").write_text(prompt)
        N.PROMPT_BASE = prompt
        c = dict(ctx)
        if use_pm4py:
            c["pm4py"] = pm4py_for(dataset, orig)
        fn = N._make_codegen_core(use_pm4py, two_stage=N.CFG["TWO_STAGE"],
                                  refine_rounds=N.CFG["REFINE_ROUNDS"],
                                  n_candidates=1)
        cand = fn(visible, c, N.CFG, wd)

    diag = {}
    if method != "M4":
        try:
            diag = N.plausibility_report(visible, cand, ctx, N.CFG)
            (wd / "diagnostics_final.json").write_text(
                json.dumps(diag, indent=2, default=str))
        except Exception:
            diag = {}
        LESSONS.record(LESSONS.key(log, cat, method, arm), trial,
                       list(_GATE_LOG), diag.get("issues", []), diag)
    _GATE_LOG.clear()

    # Flat archive of every synthesized program, for the code analysis that
    # comes next. The per-cell workdir keeps the full trail (plan.json,
    # llm_raw.txt, the refinement rounds); this is just one file per cell
    # under a single predictable name so the programs can be diffed and
    # clustered without walking the run tree. For M2 the file is the SAME
    # universal program in every cell of a trial unless reused_from.json in
    # that cell says it had to be regenerated.
    if method != "M4":
        src = wd / "generated_repair.py"
        if src.exists():
            arch = OUT / "code" / arm
            arch.mkdir(parents=True, exist_ok=True)
            (arch / ("%s__%s_t%d.py" % (dataset, mname, trial))).write_text(
                src.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8")

    final, enf = N.enforce(orig, cand, N.CFG)
    final.to_csv(wd / "final.csv", index=False)
    extra = dict(log=log, category=cat, dataset=dataset, method=mname,
                 trial=trial, model=model, arm=arm)
    extra.update({("enf_" + k): v for k, v in enf.items()})
    for k in ("order_at_best_share", "median_order_logprob_gap",
              "noop_rate_diag", "n_ordered_blocks"):
        if k in diag:
            extra[k] = diag[k]
    return score(orig, final, extra)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--logs", nargs="*", default=LOGS)
    ap.add_argument("--cats", nargs="*", default=CATS)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--trials", type=int, default=None,
                    help="override the per-method trial count")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--tag", default="percat")
    ap.add_argument("--addendum", action="store_true",
                    help="append the sub-second formatting / no-abort addendum")
    ap.add_argument("--no-lessons", action="store_true",
                    help="independent trials instead of a learning chain")
    ap.add_argument("--arm", default="per_category",
                    choices=["per_category", "unified", "authored"],
                    help="per_category = one prompt per error category; "
                         "unified = the notebook's original single prompt. "
                         "Both run with the cache off, so the two arms differ "
                         "only in the prompt.")
    ap.add_argument("--chunk", type=int, default=None,
                    help="M4 segments per API call (default 100). A chunk of "
                         "100 is a ~40k-token payload; lower this if the model "
                         "has a small context window.")
    ap.add_argument("--select-cases", type=int, default=None,
                    help="cases in the M2/M3 generation sample (default 300)")
    a = ap.parse_args()
    global args_ns
    args_ns = a

    if a.chunk:
        N.CFG["DIRECT_CHUNK_SIZE"] = a.chunk
    if a.select_cases:
        N.CFG["CODEGEN_SELECT_CASES"] = a.select_cases

    if a.list_models:
        if not load_key():
            sys.exit("OPENAI_API_KEY not found in the process or User scope.")
        from openai import OpenAI
        for m in sorted(x.id for x in OpenAI().models.list()):
            print(m)
        return

    if not a.dry_run:
        if not a.model:
            sys.exit("--model is required (see --list-models)")
        if not load_key():
            sys.exit("OPENAI_API_KEY not found in the process or User scope.")

    rows, dry_rows, failures = [], [], []
    jobs = [(lg, c, m, t)
            for lg in a.logs for c in a.cats for m in a.methods
            for t in range(1, (a.trials or TRIALS[m]) + 1)]
    print("%d cells to run\n" % len(jobs), flush=True)

    for lg, c, m, t in jobs:
        label = "%s_%s_30 %s trial %d" % (lg, c, m, t)
        t0 = time.time()
        try:
            r = run_cell(lg, c, m, t, a.model, dry=a.dry_run, arm=a.arm,
                         lessons=not a.no_lessons)
            if a.dry_run:
                dry_rows.append(r)
                print("%-34s %s" % (label, r), flush=True)
            else:
                rows += r
                al = [x for x in r if x["scope"] == "ALL_POLLUTED"][0]
                print("%-34s MedAE=%12.1f  reg=%.3f  noop=%.3f  %.0fs"
                      % (label, al["MedAE_rep"], al.get("regression_rate", float("nan")),
                         al.get("noop_rate", float("nan")), time.time() - t0),
                      flush=True)
        except Exception as ex:
            if m != "M4":
                LESSONS.record(LESSONS.key(lg, c, m, a.arm), t,
                               list(_GATE_LOG) + [repr(ex)[:400]], [], {})
            _GATE_LOG.clear()
            failures.append(dict(cell=label, error=repr(ex)[:400]))
            print("FAIL %-30s %s" % (label, repr(ex)[:200]), flush=True)
            traceback.print_exc(limit=2)
        if rows:
            pd.DataFrame(rows).to_csv(OUT / ("results_%s.csv" % a.tag), index=False)

    if a.dry_run:
        pd.DataFrame(dry_rows).to_csv(OUT / "dryrun.csv", index=False)
        print("\nwrote", OUT / "dryrun.csv")
        return
    pd.DataFrame(rows).to_csv(OUT / ("results_%s.csv" % a.tag), index=False)
    if failures:
        (OUT / ("failures_%s.json" % a.tag)).write_text(
            json.dumps(failures, indent=2))
    print("\nwrote %s  (%d rows, %d failures)"
          % (OUT / ("results_%s.csv" % a.tag), len(rows), len(failures)))


if __name__ == "__main__":
    main()
