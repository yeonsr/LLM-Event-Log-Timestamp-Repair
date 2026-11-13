import os
"""Scores the notebook's baselines and the paper-faithful versions on the
same injected datasets, through the same enforce() + core_metrics() path
the notebook's run_one() uses."""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import core as N
from core import (CFG, ERR_COL, CLEAN, CATEGORIES, build_context, enforce,
                     core_metrics, ordering_metrics, write_visible)
import baselines as F

OUT = Path(__file__).parent / "audit_out"
OUT.mkdir(exist_ok=True)
ROOT = Path(os.environ.get("TSREPAIR_ROOT", "."))
_CTX = {}


def score(log, dataset, method_name, fn):
    ds = ROOT / log / "datasets" / (dataset + ".csv")
    orig = pd.read_csv(ds)
    orig[ERR_COL] = orig[ERR_COL].fillna(CLEAN)
    CFG["TS_FORMAT"] = N.LOGS[log]["ts_format"]
    if dataset not in _CTX:
        _CTX[dataset] = build_context(orig, CFG)
    ctx = _CTX[dataset]
    wd = OUT / dataset / method_name
    wd.mkdir(parents=True, exist_ok=True)
    vis = write_visible(orig, wd / "visible_input.csv")
    t0 = time.time()
    cand = fn(vis, ctx, CFG, wd)
    final, enf = enforce(orig, cand, CFG)
    pol = orig[ERR_COL].ne(CLEAN)
    recs = []
    base = dict(log=log, dataset=dataset, method=method_name,
                secs=round(time.time() - t0, 1))
    base.update(ordering_metrics(orig, final))
    for tag, mask in ([("ALL_POLLUTED", pol)]
                      + [(c, orig[ERR_COL].eq(c)) for c in CATEGORIES]):
        if not mask.any():
            continue
        r = dict(base); r.update(core_metrics(orig, final, CFG, mask, tag))
        recs.append(r)
    return recs


def main(argv):
    which = argv[0] if argv else "b1b3"
    jobs = []
    if which in ("b1b3", "all"):
        M = {
            "M1_baseline":        N.baseline_repair,
            "M0_null":            N.null_repair,
            "B1_nb(uniform)":     N.conforti_repair,
            "B1_fix(kde-raw)":    F.make_conforti_repair(log_space_kde=False),
            "B1_fix(kde-log)":    F.make_conforti_repair(log_space_kde=True),
            "B3_nb(lognormal)":   N.bridge_repair,
            "B3_fix(gaussian)":   F.make_bridge_repair("likely"),
            "B4_nb(rescaled)":    N.sampling_repair,
            "B4_fix(unscaled)":   F.make_sampling_repair_unscaled(),
        }
        for log, dss in (("credit", ["credit_C1_30", "credit_C3_30"]),
                         ("pub", ["pub_C1_30", "pub_C3_30"]),
                         ("hospital", ["hospital_C1_30", "hospital_C3_30"])):
            for ds in dss:
                for mn, fn in M.items():
                    jobs.append((log, ds, mn, fn))
    if which in ("b2", "all"):
        M2 = {"B2_nb(emb,log1p,pre5)": N.lstm_repair,
              "B2_fix(tax-features)": F.make_lstm_repair_tax()}
        for log, ds in (("credit", "credit_C1_30"), ("pub", "pub_C1_30")):
            for mn, fn in M2.items():
                jobs.append((log, ds, mn, fn))

    rows = []
    for log, ds, mn, fn in jobs:
        try:
            r = score(log, ds, mn, fn)
            rows += r
            a = [x for x in r if x["scope"] == "ALL_POLLUTED"][0]
            print("%-16s %-14s MedAE=%12.1f  exact=%.3f  reg=%.3f  %ss"
                  % (ds, mn, a["MedAE_rep"], a.get("exact_match", float("nan")),
                     a.get("regression_rate", float("nan")), a["secs"]), flush=True)
        except Exception as ex:
            print("FAIL %s %s: %r" % (ds, mn, ex), flush=True)
    df = pd.DataFrame(rows)
    p = OUT / ("audit_%s.csv" % which)
    df.to_csv(p, index=False)
    print("wrote", p, df.shape)


if __name__ == "__main__":
    main(sys.argv[1:])
