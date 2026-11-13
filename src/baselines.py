"""Paper-faithful re-implementations of the literature baselines.

Each function is annotated with the exact equation / section of the source
paper it implements, and with the deviation it repairs relative to the
version in ts_repair_v7_1.ipynb Cell 12b.

Sources verified against the published text (Sept 2026):

B1  Conforti, La Rosa, ter Hofstede, Augusto.
    "Automatic Repair of Same-Timestamp Errors in Business Process Event
    Logs." BPM 2020, LNCS 12168, pp. 327-345.
    Sect. 3.2 (Eq. 1, 4), Sect. 3.3 (Eq. 5-9), Sect. 3.4 (Eq. 10, 11).

B3  Rogge-Solti, Mans, van der Aalst, Weske.
    "Repairing Event Logs Using Stochastic Process Models."
    HPI Technische Berichte 78 (2013)  [= OTM 2013 extended version].
    Sect. 3.1 (elicitation), Sect. 3.3 / 5 (linear Gaussian BN inference).

B2  Tax, Verenich, La Rosa, Dumas.
    "Predictive Business Process Monitoring with LSTM Neural Networks."
    CAiSE 2017, Sect. 4.1 (feature vector, targets, optimiser).
"""
import math
import itertools as _it
from collections import Counter

import numpy as np
import pandas as pd

from core import (CASE_COL, ACT_COL, TIME_COL, ERR_COL, CLEAN, C1, C2, C3,
                     _segments, _anchors, _prep, _finish, baseline_repair)


# =====================================================================
# B1 - Conforti et al., BPM 2020
# =====================================================================
#
# What the notebook version got wrong
# -----------------------------------
# ORDERING  it scored a chain with the heuristics-miner dependency
#           D(a,b) = (|a>b| - |b>a|) / (|a>b| + |b>a| + 1).
#           The paper uses Eq. 1, the NORMALISED OUTGOING DIRECTLY-FOLLOWS
#           FREQUENCY  Phi(x,y) = #(x,y) / sum_z #(x,z),  and Eq. 4, the
#           ARITHMETIC MEAN of Phi over consecutive pairs.  The paper
#           explicitly REJECTS the product-of-probabilities form (Eq. 3)
#           because a single unseen pair zeroes the whole sequence.
#           Anchors: the notebook was right - the ILP runs over
#           omega^{fs} = omega U {s, f}, s and f being the clean events
#           immediately before and after the block (Sect. 3.3).
#           The DFG is the RF-automaton: infrequent arcs filtered out AND
#           unreliable directly-follows relations (UDFRs, arcs derived from
#           a timestamp-equivalent pair) removed.
#
# SPACING   it spread the block UNIFORMLY across the anchor interval.
#           The paper has no anchor interval and no uniform rule.  Eq. 10-11:
#             D(e)      = T(e) - T(e_prev)   over pairs with DIFFERENT
#                         timestamps, collected PER ACTIVITY (not per pair)
#             f_a       = KDE (Silverman) of that per-activity multiset,
#                         explicitly to capture multimodality
#             T+(e)     = T(e_prev) + f_a_bar    applied CUMULATIVELY along
#                         the repaired order, starting from the left anchor
#           i.e. a forward duration walk, not an interval redistribution.
#           This is the deviation responsible for B1's blow-up on hospital.

def _rf_automaton(out, min_arc_share=0.0):
    """Phi(x,y) of Eq. 1, computed on the RF-automaton.

    UDFR removal (Sect. 3.2): a directly-follows pair is unreliable when it
    is derived from timestamp-equivalent events.  Restricting the counts to
    CLEAN rows removes every injected block, which is a superset of the
    UDFRs, so this is at least as strict as the paper.

    `min_arc_share` is the infrequency filter the paper recommends applying
    to the L-automaton before use (Sect. 3.2, "we rely on the method in
    [5]").  0.0 disables it; the paper does not publish a threshold, so it
    is exposed as an explicit experimental parameter rather than hidden.
    """
    d = out[out[ERR_COL].eq(CLEAN)]
    d = d[d["_ts"].notna()]
    cnt = Counter()
    for _, g in d.groupby(CASE_COL, sort=False):
        acts = g.sort_values("_ts", kind="mergesort")[ACT_COL].tolist()
        for a, b in zip(acts, acts[1:]):
            cnt[(a, b)] += 1
    outdeg = Counter()
    for (a, _b), n in cnt.items():
        outdeg[a] += n
    phi = {}
    for (a, b), n in cnt.items():
        share = n / outdeg[a]
        if share >= min_arc_share:            # F-automaton filter
            phi[(a, b)] = share
    return phi


def _activity_duration_pdf(out, cfg, log_space=False, max_pts=4000,
                           min_pts=5, rng=None):
    """Eq. 10 + the KDE of Sect. 3.4, one PDF per ACTIVITY.

    Returns {activity: point estimate}, the point estimate being the MODE of
    the Silverman-bandwidth Gaussian KDE ("extracts a likely duration from
    the PDF", Eq. 11).  Falls back to the median when an activity has too
    few observations for a KDE.

    D(e) is defined only when T(e) != T(e_prev) (Eq. 10), so identical-
    timestamp pairs never enter the sample; CLEAN-only is again stricter.
    """
    from scipy.stats import gaussian_kde
    rng = rng or np.random.default_rng(0)
    d = out[out[ERR_COL].eq(CLEAN)].copy()
    d = d[d["_ts"].notna()].sort_values([CASE_COL, "_ts"], kind="mergesort")
    d["gap"] = d.groupby(CASE_COL)["_ts"].diff().dt.total_seconds()
    d = d[d["gap"].notna() & (d["gap"] > 0)]          # T(e) != T(e_p)
    est, glob = {}, d["gap"].to_numpy()
    for a, g in d.groupby(ACT_COL, sort=False):
        x = g["gap"].to_numpy()
        if len(x) > max_pts:
            x = rng.choice(x, max_pts, replace=False)
        if len(x) < min_pts or np.allclose(x, x[0]):
            est[a] = float(np.median(x)) if len(x) else cfg["FALLBACK_DURATION"]
            continue
        z = np.log(x) if log_space else x
        try:
            kde = gaussian_kde(z, bw_method="silverman")
            lo, hi = np.quantile(z, [0.001, 0.999])
            grid = np.linspace(lo, hi, 512)
            m = float(grid[int(np.argmax(kde(grid)))])
            est[a] = float(np.exp(m)) if log_space else m
        except Exception:
            est[a] = float(np.median(x))
        est[a] = max(cfg["MIN_DURATION"], est[a])
    gfb = float(np.median(glob)) if len(glob) else cfg["FALLBACK_DURATION"]
    return est, max(cfg["MIN_DURATION"], gfb)


def make_conforti_repair(min_arc_share=0.0, log_space_kde=False):
    """B1, faithful to BPM 2020 Sect. 3.2-3.4."""

    def method(visible, ctx, cfg, workdir=None):
        out = _prep(visible, cfg)
        new = out["_ts"].astype("datetime64[ns]").copy()
        phi = _rf_automaton(out, min_arc_share)
        dur, gdur = _activity_duration_pdf(out, cfg, log_space=log_space_kde)

        def conf(seq_acts):
            # Eq. 4: arithmetic mean of Phi over consecutive pairs.  The
            # 1/(|w|-1) factor is constant across permutations of one block,
            # so the argmax is the plain sum; kept explicit for clarity.
            if len(seq_acts) < 2:
                return 0.0
            return sum(phi.get((x, y), 0.0)
                       for x, y in zip(seq_acts, seq_acts[1:])) / (len(seq_acts) - 1)

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
                obs = g.loc[s, "_ts"]

                # --- Sect. 3.3: max over omega^{fs} = omega U {s, f} ------
                if k <= cfg["EXHAUSTIVE_MAX"]:
                    # for k <= 5 the exhaustive optimum IS the ILP optimum
                    order = list(max(
                        _it.permutations(range(k)),
                        key=lambda p: conf(([la] if la else [])
                                           + [acts[i] for i in p]
                                           + ([ra] if ra else []))))
                else:
                    order, rest, cur = [], list(range(k)), la
                    while rest:
                        nxt = max(rest, key=lambda i: phi.get((cur, acts[i]), 0.0))
                        order.append(nxt); rest.remove(nxt); cur = acts[nxt]

                # --- Sect. 3.4 Eq. 11: cumulative per-activity duration ---
                # T+(e) = T(e_prev) + f_bar_{A(e)}, e_prev being the already
                # repaired predecessor; the first predecessor is the left
                # anchor.  No right anchor, no rescaling: the paper has none.
                cur = lt if (lt is not None and pd.notna(lt)) else obs
                for rank, pos in enumerate(order):
                    a = acts[pos]
                    cur = cur + pd.Timedelta(
                        seconds=float(max(cfg["MIN_DURATION"], dur.get(a, gdur))))
                    new.loc[g.loc[s + pos, "_ix"]] = cur

        res = _finish(visible, new, cfg)
        # Out of scope for the paper (same-timestamp errors only).
        fb = baseline_repair(visible, ctx, cfg)
        other = visible[ERR_COL].isin([C2, C3]).to_numpy()
        res.loc[other, TIME_COL] = fb.loc[other, TIME_COL].to_numpy()
        return res
    return method


# =====================================================================
# B3 - Rogge-Solti et al., HPI TR 78 / OTM 2013
# =====================================================================
#
# What the notebook version got wrong
# -----------------------------------
# DISTRIBUTION  it fitted LOGNORMAL durations ("parameters (mu, sigma)
#               fitted on CLEAN rows in log space").  The report states
#               plainly: "in this work, we assume normally distributed
#               durations", and the whole inference step depends on it -
#               the repair is exact only because the model is a LINEAR
#               GAUSSIAN Bayesian network, the one continuous BN family
#               with tractable inference (Sect. 3.3).
#
# INFERENCE     it ran Nelder-Mead over a softmax parametrisation to
#               minimise sum (log d_i - mu_i)^2 / 2 sigma_i^2 s.t. sum d = W.
#               Under the paper's Gaussian model the conditional mode has a
#               CLOSED FORM and needs no optimiser:
#                   d_i* = mu_i + sigma_i^2 / sum_j sigma_j^2 * (W - sum_j mu_j)
#               This is also the only form that actually delivers the
#               property the notebook docstring claims - "a transition with
#               high variance absorbs more of the discrepancy".  The
#               log-space softmax version does NOT have that property.
#
# REPAIR MODE   the paper offers two: "most likely" (the posterior mode
#               above) and "random" (a draw from the posterior).  Most
#               likely is the accuracy-oriented one and matches their MAE
#               evaluation, so it is the default here.

def _gauss_params(out, cfg):
    d = out[out[ERR_COL].eq(CLEAN)].copy()
    d = d[d["_ts"].notna()].sort_values([CASE_COL, "_ts"], kind="mergesort")
    d["prev"] = d.groupby(CASE_COL)[ACT_COL].shift()
    d["gap"] = d.groupby(CASE_COL)["_ts"].diff().dt.total_seconds()
    p = d.dropna(subset=["prev", "gap"])
    p = p[p["gap"] > 0]
    par = p.groupby(["prev", ACT_COL])["gap"].agg(["mean", "std", "count"])
    gmu = float(p["gap"].mean()) if len(p) else cfg["FALLBACK_DURATION"]
    gsd = float(p["gap"].std()) if len(p) else gmu

    def musig(a, b):
        if (a, b) in par.index:
            r = par.loc[(a, b)]
            if r["count"] >= 5 and r["std"] == r["std"] and r["std"] > 1e-6:
                return float(r["mean"]), float(r["std"])
        return gmu, max(gsd, 1.0)
    return musig


def _gauss_bridge(pairs, W, cfg, rng=None, mode="likely"):
    """Linear-Gaussian posterior of d_1..d_m given sum d_i = W.

    d_i ~ N(mu_i, s_i^2) independent  =>  conditioned on their sum,
        E[d_i | sum = W] = mu_i + s_i^2 / sum_j s_j^2 * (W - sum_j mu_j)
    which is also the joint mode.  Negative draws are corrected upward, as
    the report allows ("errors introduced by correction of negative
    durations are negligible").
    """
    mus = np.array([m for m, _ in pairs], float)
    sds = np.array([max(s, 1e-6) for _, s in pairs], float)
    if not np.isfinite(W):
        return np.maximum(mus, cfg["MIN_DURATION"])
    var = sds ** 2
    d = mus + var / var.sum() * (W - mus.sum())
    if mode == "random" and rng is not None:
        # posterior covariance of a Gaussian conditioned on its sum
        d = d + (rng.normal(size=len(d)) * sds
                 - (var / var.sum()) * float(rng.normal() * math.sqrt(var.sum())))
    d = np.maximum(d, cfg["MIN_DURATION"])
    if W > 0 and d.sum() > W:                 # keep the anchor constraint
        d = np.maximum(d * W / d.sum(), cfg["MIN_DURATION"] * 0.0 + 1e-9)
    return d


def make_bridge_repair(mode="likely", seed=42):
    """B3, faithful to HPI TR 78: Gaussian durations, closed-form posterior."""
    from core import _best_order

    def method(visible, ctx, cfg, workdir=None):
        rng = np.random.default_rng(seed)
        out = _prep(visible, cfg)
        new = out["_ts"].astype("datetime64[ns]").copy()
        prob = ctx["trans_prob"]
        musig = _gauss_params(out, cfg)

        hit = out.loc[out[ERR_COL].ne(CLEAN), CASE_COL].unique()
        for _, g0 in out[out[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
            g = g0.reset_index(drop=False).rename(columns={"index": "_ix"})

            for s, e in _segments(g, C1):
                li, ri = _anchors(g, s, e)
                lt = g.loc[li, "_ts"] if li is not None else None
                rt = g.loc[ri, "_ts"] if ri is not None else None
                if lt is None or rt is None or pd.isna(lt) or pd.isna(rt):
                    continue
                blk = g.loc[s:e, ACT_COL].tolist()
                la = g.loc[li, ACT_COL] if li is not None else None
                ra = g.loc[ri, ACT_COL] if ri is not None else None
                order = _best_order(blk, la, ra, prob, cfg)
                ordered = [blk[i] for i in order]
                chain = ([la] if la else [ordered[0]]) + ordered + ([ra] if ra else [])
                pairs = [musig(a, b) for a, b in zip(chain, chain[1:])]
                ds = _gauss_bridge(pairs, (rt - lt).total_seconds(), cfg,
                                   rng, mode)
                cur = lt
                for rank, pos in enumerate(order):
                    cur = cur + pd.Timedelta(seconds=float(ds[rank]))
                    if cur >= rt:
                        cur = rt - pd.Timedelta(seconds=cfg["MIN_DURATION"])
                    new.loc[g.loc[s + pos, "_ix"]] = cur

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
                ds = _gauss_bridge(pairs, (rt - lt).total_seconds(), cfg,
                                   rng, mode)
                cur = lt
                for kk in range(s, e + 1):
                    cur = cur + pd.Timedelta(seconds=float(ds[kk - s]))
                    obs = g.loc[kk, "_ts"]
                    if pd.notna(obs) and obs.normalize() == obs:
                        cur = obs + pd.Timedelta(
                            seconds=(cur - cur.normalize()).total_seconds())
                    new.loc[g.loc[kk, "_ix"]] = cur

        res = _finish(visible, new, cfg)
        fb = baseline_repair(visible, ctx, cfg)
        c2 = visible[ERR_COL].eq(C2).to_numpy()
        res.loc[c2, TIME_COL] = fb.loc[c2, TIME_COL].to_numpy()
        return res
    return method


# =====================================================================
# B2 - Tax et al., CAiSE 2017
# =====================================================================
#
# What the notebook version got wrong
# -----------------------------------
# FEATURES  activity-embedding only.  Tax Sect. 4.1 uses ONE-HOT activity
#           PLUS three time features: fvt1 = time since the previous event,
#           fvt2 = time since midnight, fvt3 = time since midnight on
#           Sunday.  Dropping fvt1-fvt3 removes exactly the signal the time
#           head is supposed to regress on, so the notebook's B2 was an
#           activity-only predictor wearing Tax's name.
# TARGET    log1p(delta_t).  Tax sets the time target to fvt1 of the next
#           step - the RAW time difference - and the reference
#           implementation scales all time features by the mean of fvt1.
# PREFIX    truncated at 5 events.  Tax feeds the whole prefix, zero-padded
#           to the maximum trace length.
# NETWORK   1 LSTM layer of 64 units, no dropout, 8 epochs, Adam 1e-3.
#           Tax: 2 LSTM layers of 100 units, dropout 0.2, Adam, MAE loss,
#           trained to convergence with early stopping.
# SPLIT     trained on the CLEAN rows of the log it then repairs (in-sample).
#           Tax hold out the last 1/3 of traces chronologically.  Keeping
#           the in-sample fit is defensible here - there is no clean
#           training log - but it makes B2 optimistic, not pessimistic, and
#           has to be stated.

def make_lstm_repair_tax(epochs=100, hidden=100, layers=2, dropout=0.2,
                         max_len=None, seed=42, patience=10, verbose=True):
    from core import _best_order

    def method(visible, ctx, cfg, workdir=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)

        out = _prep(visible, cfg)
        new = out["_ts"].astype("datetime64[ns]").copy()

        acts = sorted(out[ACT_COL].astype(str).unique())
        a2i = {a: i for i, a in enumerate(acts)}
        A = len(acts)

        d = out[out[ERR_COL].eq(CLEAN)].copy()
        d = d[d["_ts"].notna()].sort_values([CASE_COL, "_ts"], kind="mergesort")
        d["fvt1"] = d.groupby(CASE_COL)["_ts"].diff().dt.total_seconds().fillna(0.0)
        d = d[d["fvt1"] >= 0]
        if len(d) < 200:
            return baseline_repair(visible, ctx, cfg)
        divisor = float(d["fvt1"].mean()) or 1.0          # Tax reference impl.

        def feats(act, fvt1, ts):
            v = np.zeros(A + 3, dtype=np.float32)
            i = a2i.get(str(act))
            if i is not None:
                v[i] = 1.0
            v[A] = fvt1 / divisor
            v[A + 1] = (ts - ts.normalize()).total_seconds() / 86400.0
            v[A + 2] = ((ts.dayofweek * 86400.0
                         + (ts - ts.normalize()).total_seconds()) / 604800.0)
            return v

        traces = []
        for _, g in d.groupby(CASE_COL, sort=False):
            rows = list(zip(g[ACT_COL], g["fvt1"], g["_ts"]))
            traces.append([feats(a, f, t) for a, f, t in rows])
        L = max_len or min(max(len(t) for t in traces), 40)

        X, Y = [], []
        for tr, (_, g) in zip(traces, d.groupby(CASE_COL, sort=False)):
            nxt = g["fvt1"].to_numpy()
            for i in range(len(tr) - 1):
                pre = tr[max(0, i + 1 - L): i + 1]
                pad = [np.zeros(A + 3, dtype=np.float32)] * (L - len(pre))
                X.append(np.stack(pad + pre))
                Y.append(nxt[i + 1] / divisor)      # target = next fvt1, RAW
        X = np.stack(X).astype(np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(A + 3, hidden, num_layers=layers,
                                    batch_first=True, dropout=dropout)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                h, _ = self.lstm(x)
                return self.head(h[:, -1, :]).squeeze(-1)

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.L1Loss()                                   # Tax: MAE
        xt, yt = torch.as_tensor(X), torch.as_tensor(Y)
        n = len(xt)
        cut = max(1, int(n * 0.8))
        perm0 = torch.randperm(n)
        tr_i, va_i = perm0[:cut], perm0[cut:]
        best, bad, bstate = float("inf"), 0, None
        for ep in range(epochs):
            net.train()
            perm = tr_i[torch.randperm(len(tr_i))]
            for i in range(0, len(perm), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                lossf(net(xt[idx]), yt[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(xt[va_i]), yt[va_i])) if len(va_i) else 0.0
            if vl < best - 1e-5:
                best, bad = vl, 0
                bstate = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if bstate:
            net.load_state_dict(bstate)
        net.eval()
        if verbose:
            print("    [B2-tax] %d epochs, val MAE %.4f (scaled)" % (ep + 1, best))

        def predict(seq):
            with torch.no_grad():
                return float(net(torch.as_tensor(np.stack([seq]))).item()) * divisor

        prob = ctx["trans_prob"]
        hit = out.loc[out[ERR_COL].ne(CLEAN), CASE_COL].unique()
        for _, g0 in out[out[CASE_COL].isin(hit)].groupby(CASE_COL, sort=False):
            g = g0.reset_index(drop=False).rename(columns={"index": "_ix"})
            ts_all = g["_ts"].tolist()
            act_all = g[ACT_COL].tolist()
            for s, e in _segments(g, C1):
                li, ri = _anchors(g, s, e)
                blk = act_all[s:e + 1]
                k = len(blk)
                la = g.loc[li, ACT_COL] if li is not None else None
                ra = g.loc[ri, ACT_COL] if ri is not None else None
                lt = g.loc[li, "_ts"] if li is not None else None
                rt = g.loc[ri, "_ts"] if ri is not None else None
                order = _best_order(blk, la, ra, prob, cfg)
                ordered = [blk[i] for i in order]

                start = lt if (lt is not None and pd.notna(lt)) else g.loc[s, "_ts"]
                if pd.isna(start):
                    continue
                # autoregressive: each predicted event is appended to the
                # prefix before the next duration is predicted
                pre = []
                for j in range(max(0, s - L), s):
                    if pd.notna(ts_all[j]):
                        prev = ts_all[j - 1] if j > 0 and pd.notna(ts_all[j - 1]) else ts_all[j]
                        pre.append(feats(act_all[j], (ts_all[j] - prev).total_seconds(),
                                         ts_all[j]))
                cur, ds = start, []
                for a in ordered:
                    prev_gap = (cur - start).total_seconds() if ds else 0.0
                    seq = (pre + [feats(a, prev_gap, cur)])[-L:]
                    seq = [np.zeros(A + 3, dtype=np.float32)] * (L - len(seq)) + seq
                    dt = max(cfg["MIN_DURATION"], predict(np.stack(seq)))
                    ds.append(dt)
                    cur = cur + pd.Timedelta(seconds=float(dt))
                    pre = seq
                if rt is not None and pd.notna(rt):
                    room = (rt - start).total_seconds()
                    if room > 0 and sum(ds) > room:
                        ds = [x * room * .95 / sum(ds) for x in ds]
                cur = start
                for rank, pos in enumerate(order):
                    cur = cur + pd.Timedelta(seconds=float(ds[rank]))
                    new.loc[g.loc[s + pos, "_ix"]] = cur

        res = _finish(visible, new, cfg)
        fb = baseline_repair(visible, ctx, cfg)
        other = visible[ERR_COL].isin([C2, C3]).to_numpy()
        res.loc[other, TIME_COL] = fb.loc[other, TIME_COL].to_numpy()
        return res
    return method


# =====================================================================
# B4 - Schmid et al., Information Systems 118 (2023) 102246
# =====================================================================
#
# Only the abstract could be verified (paywalled, no OA copy, ScienceDirect
# served a CAPTCHA).  Two things it does state:
#   - the method "combines EXISTING method components for error detection
#     and REORDERING of erroneous events with a novel GAN-based repair",
#     and the reordering component it reuses is Conforti's.  So a faithful
#     surrogate should order with Phi, not with M1's transition
#     probabilities.
#   - the GAN samples the conditional duration DISTRIBUTION.
#
# The notebook surrogate does sample, but then rescales the sampled gaps to
# fit the anchor window and anchors the block on the OBSERVED collapsed
# value.  Both are M1 design decisions imported into the baseline; the
# rescale in particular destroys the one property the surrogate exists to
# demonstrate (keeping the shape of the duration distribution).  This
# variant keeps the draws unrescaled and walks forward from the left
# anchor, which is what a generative duration model does.

def make_sampling_repair_unscaled(seed=42, use_phi_order=True):
    def method(visible, ctx, cfg, workdir=None):
        from core import _best_order
        rng = np.random.default_rng(seed)
        out = _prep(visible, cfg)
        new = out["_ts"].astype("datetime64[ns]").copy()
        prob = ctx["trans_prob"]
        phi = _rf_automaton(out) if use_phi_order else None

        d = out[out[ERR_COL].eq(CLEAN)].copy()
        d = d[d["_ts"].notna()].sort_values([CASE_COL, "_ts"], kind="mergesort")
        d["prev"] = d.groupby(CASE_COL)[ACT_COL].shift()
        d["gap"] = d.groupby(CASE_COL)["_ts"].diff().dt.total_seconds()
        p = d.dropna(subset=["prev", "gap"])
        p = p[p["gap"] >= 0]
        pool = {k: v["gap"].to_numpy() for k, v in p.groupby(["prev", ACT_COL])}
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
                blk = g.loc[s:e, ACT_COL].tolist()
                k = len(blk)
                la = g.loc[li, ACT_COL] if li is not None else None
                ra = g.loc[ri, ACT_COL] if ri is not None else None
                lt = g.loc[li, "_ts"] if li is not None else None
                obs = g.loc[s, "_ts"]
                start = lt if (lt is not None and pd.notna(lt)) else obs
                if pd.isna(start):
                    continue
                if use_phi_order and k <= cfg["EXHAUSTIVE_MAX"]:
                    order = list(max(_it.permutations(range(k)),
                                     key=lambda pm: sum(
                                         phi.get((x, y), 0.0) for x, y in zip(
                                             ([la] if la else []) + [blk[i] for i in pm] + ([ra] if ra else []),
                                             (([la] if la else []) + [blk[i] for i in pm] + ([ra] if ra else []))[1:]))))
                else:
                    order = _best_order(blk, la, ra, prob, cfg)
                ordered = [blk[i] for i in order]
                cur, prev = start, la or ordered[0]
                for rank, pos in enumerate(order):
                    cur = cur + pd.Timedelta(seconds=draw(prev, ordered[rank]))
                    prev = ordered[rank]
                    new.loc[g.loc[s + pos, "_ix"]] = cur

        res = _finish(visible, new, cfg)
        fb = baseline_repair(visible, ctx, cfg)
        other = visible[ERR_COL].isin([C2, C3]).to_numpy()
        res.loc[other, TIME_COL] = fb.loc[other, TIME_COL].to_numpy()
        return res
    return method
