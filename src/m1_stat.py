# -*- coding: utf-8 -*-
"""The pre-v4 (predecessor study-era) M1, for the one question that matters here:

Why did the LLM methods beat M1 on C1 in the predecessor study paper and lose to it now?

The hypothesis, from Proposal_26 section 8: the predecessor study baseline rebuilt each C1
block FROM THE LEFT ANCHOR by adding durations. The current baseline starts at
the OBSERVED collapsed value. On bpic12 C1 that one change is reported as
24,164 s -> 71.8 s.

If the hypothesis holds, the LLM methods did not get worse. The baseline got
336x better, and the comparison flipped under them.

This reproduces the old rule exactly: same ordering search, same per-pair
median durations, same clamping -- only the starting point differs.
"""
import math

import numpy as np
import pandas as pd

from core import (CASE_COL, ACT_COL, TIME_COL, ERR_COL, CLEAN, C1, C2, C3,
                     _segments, _anchors, _best_order, baseline_repair)


def make_m1_unanchored():
    """M1 as it was before v4: the block is laid out from the LEFT ANCHOR."""

    def method(visible, ctx, cfg, workdir=None):
        out = visible.copy()
        out["_ts"] = pd.to_datetime(out[TIME_COL], errors="coerce")
        prob, med, gmed = ctx["trans_prob"], ctx["dur_median"], ctx["global_median"]
        if gmed != gmed:
            gmed = cfg["FALLBACK_DURATION"]
        new = out["_ts"].astype("datetime64[ns]").copy()

        def _dur(prev, a):
            v = med.get((prev, a), gmed)
            v = float(v) if v == v else gmed
            return max(cfg["MIN_DURATION"], v)

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

                order = _best_order(acts, la, ra, prob, cfg)
                ordered = [acts[i] for i in order]

                # ---- the only difference from current M1 -----------------
                # start from the left anchor and add a duration for EVERY
                # event, including the first, instead of placing the first
                # event on the observed collapsed value.
                start = lt if (lt is not None and pd.notna(lt)) else obs
                prev = la if la is not None else ordered[0]
                gaps = []
                for a in ordered:
                    gaps.append(_dur(prev, a))
                    prev = a
                # ----------------------------------------------------------

                S = sum(gaps)
                if rt is not None and pd.notna(rt) and pd.notna(start):
                    room = (rt - start).total_seconds()
                    if room > 0 and S > room:
                        gaps = [x * room * .95 / S for x in gaps]
                cur = start
                for rank, pos in enumerate(order):
                    cur = cur + pd.Timedelta(seconds=float(gaps[rank]))
                    if rt is not None and pd.notna(rt) and cur >= rt:
                        cur = rt - pd.Timedelta(seconds=cfg["MIN_DURATION"])
                    new.loc[g.loc[s + pos, "_ix"]] = cur

        res = visible.copy()
        res[TIME_COL] = pd.to_datetime(new).dt.strftime(cfg["TS_FORMAT"])
        # C2 and C3 are unchanged between the two baseline versions
        fb = baseline_repair(visible, ctx, cfg)
        other = visible[ERR_COL].isin([C2, C3]).to_numpy()
        res.loc[other, TIME_COL] = fb.loc[other, TIME_COL].to_numpy()
        return res
    return method
