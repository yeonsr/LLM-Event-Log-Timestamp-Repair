import os
# -*- coding: utf-8 -*-
"""M1 = STAT, the naive statistical baseline, scored on all 15 single-category datasets.

Why the baseline is being replaced
----------------------------------
The v4 M1 lays each C1 block out from the OBSERVED collapsed timestamp. Under
the injection procedure the collapse is onto the block's own FIRST timestamp,
so that rule starts every block from a value that is already exactly right and
only has to space the remaining rows. It scores well because it encodes a
property of our error generator, not because it reconstructs anything -- which
is why it beats every method that estimates the block honestly, including
three published baselines.

STAT rebuilds the block from the LEFT ANCHOR by adding a
duration for every event, including the first. It never reads the collapsed
value. That is the baseline this study compares against from here on.

C2 and C3 are identical between the two versions, so only C1 moves; the C2/C3
columns are carried over from baseline_repair() unchanged (see m1_stat.py).

Also scores M0_null, so the "is the baseline better than doing nothing?"
column is computed on exactly the same row sets in the same run.

Usage:
    python run_m1_stat.py                  # all 5 logs x C1/C2/C3
    python run_m1_stat.py credit pub       # a subset of logs
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import core as N                                              # noqa: E402
from core import (CFG, ERR_COL, CLEAN, CATEGORIES, build_context,  # noqa: E402
                     enforce, core_metrics, ordering_metrics, write_visible)
from m1_stat import make_m1_unanchored                          # noqa: E402

ROOT = Path(os.environ.get("TSREPAIR_ROOT", "."))
OUT = Path(__file__).parent / "audit_out"
OUT.mkdir(exist_ok=True)

LOGS = ["credit", "pub", "hospital", "bpic12", "bpic15"]
CATS = ["C1", "C2", "C3"]

METHODS = {
    # the new baseline
    "M1": make_m1_unanchored(),
    # kept in the same run so every comparison uses one row set
    "M0_null": N.null_repair,
    # the deprecated one, for the "cost of the change" row in the paper
    "M1_anchored_deprecated": N.baseline_repair,
}


def score(log, dataset, method_name, fn, ctx_cache):
    orig = pd.read_csv(ROOT / log / "datasets" / (dataset + ".csv"))
    orig[ERR_COL] = orig[ERR_COL].fillna(CLEAN)
    CFG["TS_FORMAT"] = N.LOGS[log]["ts_format"]
    if dataset not in ctx_cache:
        ctx_cache[dataset] = build_context(orig, CFG)
    ctx = ctx_cache[dataset]
    wd = OUT / "m1_stat" / dataset / method_name
    wd.mkdir(parents=True, exist_ok=True)
    vis = write_visible(orig, wd / "visible_input.csv")
    t0 = time.time()
    cand = fn(vis, ctx, CFG, wd)
    final, enf = enforce(orig, cand, CFG)
    final.to_csv(wd / "final.csv", index=False)
    pol = orig[ERR_COL].ne(CLEAN)
    base = dict(log=log, dataset=dataset,
                category=dataset.split("_")[-2], method=method_name,
                secs=round(time.time() - t0, 1))
    base.update(ordering_metrics(orig, final))
    base.update({("enf_" + k): v for k, v in enf.items()})
    recs = []
    for tag, mask in ([("ALL_POLLUTED", pol)]
                      + [(c, orig[ERR_COL].eq(c)) for c in CATEGORIES]):
        if not mask.any():
            continue
        r = dict(base)
        r.update(core_metrics(orig, final, CFG, mask, tag))
        r["comparable"] = bool(r.get("coverage", 1.0) >= .999)
        recs.append(r)
    return recs


def main(argv):
    logs = [x for x in argv if x in LOGS] or LOGS
    ctx_cache = {}
    rows, failures = [], []
    jobs = [(lg, "%s_%s_30" % (lg, c), mn, fn)
            for lg in logs for c in CATS for mn, fn in METHODS.items()]
    print("%d cells to run\n" % len(jobs), flush=True)
    for lg, ds, mn, fn in jobs:
        try:
            r = score(lg, ds, mn, fn, ctx_cache)
            rows += r
            a = [x for x in r if x["scope"] == "ALL_POLLUTED"][0]
            print("%-16s %-24s MedAE=%14.1f  exact=%.3f  reg=%.3f  cov=%.3f  %ss"
                  % (ds, mn, a["MedAE_rep"], a.get("exact_match", float("nan")),
                     a.get("regression_rate", float("nan")),
                     a.get("coverage", float("nan")), a["secs"]), flush=True)
        except Exception as ex:
            failures.append((ds, mn, repr(ex)[:300]))
            print("FAIL %-16s %-24s %r" % (ds, mn, ex), flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(OUT / "b0_and_stat.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "b0_and_stat.csv", index=False)
    print("\nwrote %s  (%d rows, %d failures)"
          % (OUT / "b0_and_stat.csv", len(rows), len(failures)))
    for f in failures:
        print("  FAILED:", f)


if __name__ == "__main__":
    main(sys.argv[1:])
