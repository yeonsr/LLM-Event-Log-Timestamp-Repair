import argparse
import math
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Embedded process knowledge from prompt
# ---------------------------------------------------------------------

DURATION_PROFILES = {
    ("FIN", "RELEASE"): {"p10": 2115.6000000000004, "p25": 6885.25, "median": 16217.0, "share_under_60s": 0.048},
    ("RELEASE", "CODE OK"): {"p10": 3665.1, "p25": 177283.0, "median": 343757.5, "share_under_60s": 0.043},
    ("CODE OK", "BILLED"): {"p10": 384101.4, "p25": 1317610.5, "median": 3783677.0, "share_under_60s": 0.0},
    ("NEW", "CHANGE DIAGN"): {"p10": 12.0, "p25": 16.0, "median": 25.0, "share_under_60s": 0.636},
    ("CHANGE DIAGN", "FIN"): {"p10": 1524200.0000000023, "p25": 4012910.0, "median": 7708691.0, "share_under_60s": 0.006},
    ("NEW", "FIN"): {"p10": 431237.0, "p25": 5458135.0, "median": 24051621.5, "share_under_60s": 0.033},
    ("NEW", "NEW"): {"p10": 1.0, "p25": 1.0, "median": 1.0, "share_under_60s": 1.0},
    ("NEW", "DELETE"): {"p10": 2443.7000000000007, "p25": 156903.5, "median": 1814597.5, "share_under_60s": 0.061},
    ("FIN", "CODE OK"): {"p10": 78171.0000000001, "p25": 251406.0, "median": 371464.0, "share_under_60s": 0.027},
    ("RELEASE", "BILLED"): {"p10": 736717.8, "p25": 1459903.25, "median": 3963059.0, "share_under_60s": 0.0},
    ("REOPEN", "FIN"): {"p10": 6.0, "p25": 10.0, "median": 53.0, "share_under_60s": 0.512},
    ("CODE OK", "REOPEN"): {"p10": 126745.6, "p25": 459994.0, "median": 1916316.0, "share_under_60s": 0.013},
    ("BILLED", "STORNO"): {"p10": 258883.10000000012, "p25": 434157.75, "median": 1077097.0, "share_under_60s": 0.002},
    ("CHANGE DIAGN", "RELEASE"): {"p10": 2299488.4, "p25": 4247154.0, "median": 7732648.0, "share_under_60s": 0.003},
    ("CHANGE DIAGN", "CHANGE DIAGN"): {"p10": 206.20000000000002, "p25": 36627.0, "median": 766567.0, "share_under_60s": 0.041},
    ("NEW", "RELEASE"): {"p10": 448298.60000000003, "p25": 7864724.0, "median": 31216839.5, "share_under_60s": 0.014},
    ("STORNO", "BILLED"): {"p10": 92763.2, "p25": 92835.0, "median": 169855.0, "share_under_60s": 0.0},
    ("STORNO", "REJECT"): {"p10": 5.700000000000003, "p25": 15.75, "median": 1242.0, "share_under_60s": 0.428},
    ("REOPEN", "DELETE"): {"p10": 2.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.974},
    ("REJECT", "BILLED"): {"p10": 1710.8000000000006, "p25": 160511.5, "median": 325498.5, "share_under_60s": 0.062},
    ("RELEASE", "CODE NOK"): {"p10": 331371.0, "p25": 599392.0, "median": 1473039.0, "share_under_60s": 0.008},
    ("CHANGE DIAGN", "DELETE"): {"p10": 203.4, "p25": 69873.0, "median": 961787.0, "share_under_60s": 0.064},
    ("CODE OK", "RELEASE"): {"p10": 1990550.0, "p25": 2961133.0, "median": 4592168.5, "share_under_60s": 0.01},
    ("REJECT", "REOPEN"): {"p10": 7.7, "p25": 10.75, "median": 38.5, "share_under_60s": 0.647},
    ("RELEASE", "REOPEN"): {"p10": 19776.0, "p25": 48101.0, "median": 292036.0, "share_under_60s": 0.033},
    ("JOIN-PAT", "JOIN-PAT"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("CODE NOK", "CODE OK"): {"p10": 3809.400000000001, "p25": 18932.0, "median": 49700.0, "share_under_60s": 0.023},
    ("CODE NOK", "SET STATUS"): {"p10": 9180.200000000003, "p25": 17044.5, "median": 25733.5, "share_under_60s": 0.0},
    ("CODE NOK", "CODE NOK"): {"p10": 345704.0, "p25": 590782.0, "median": 2352460.0, "share_under_60s": 0.0},
    ("FIN", "REOPEN"): {"p10": 5.800000000000001, "p25": 49.5, "median": 13837.0, "share_under_60s": 0.275},
    ("FIN", "FIN"): {"p10": 0.0, "p25": 0.0, "median": 31430606.0, "share_under_60s": 0.351},
    ("CODE OK", "MANUAL"): {"p10": 487.0, "p25": 469502.0, "median": 1652942.0, "share_under_60s": 0.061},
    ("FIN", "DELETE"): {"p10": 66.0, "p25": 57817.25, "median": 1337180.0, "share_under_60s": 0.094},
    ("BILLED", "BILLED"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.517},
    ("CODE NOK", "REOPEN"): {"p10": 12467.6, "p25": 22825.5, "median": 31343.5, "share_under_60s": 0.036},
    ("MANUAL", "RELEASE"): {"p10": 289.0, "p25": 27209.5, "median": 36319.0, "share_under_60s": 0.038},
    ("REOPEN", "CHANGE DIAGN"): {"p10": 8.200000000000001, "p25": 20.0, "median": 42.0, "share_under_60s": 0.52},
    ("RELEASE", "RELEASE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.55},
    ("BILLED", "CODE OK"): {"p10": 2311645.3, "p25": 7165299.25, "median": 11056150.0, "share_under_60s": 0.0},
    ("CODE OK", "CODE OK"): {"p10": 940142.6, "p25": 2474552.0, "median": 5718020.0, "share_under_60s": 0.0},
    ("CODE NOK", "DELETE"): {"p10": 12976.6, "p25": 21498.5, "median": 27554.0, "share_under_60s": 0.0},
    ("CODE OK", "DELETE"): {"p10": 806324.1000000001, "p25": 2109509.5, "median": 2712903.0, "share_under_60s": 0.0},
    ("CODE OK", "FIN"): {"p10": 183503.0, "p25": 938728.75, "median": 1682164.5, "share_under_60s": 0.062},
    ("CODE OK", "STORNO"): {"p10": 1925302.0000000005, "p25": 3485959.5, "median": 5570562.0, "share_under_60s": 0.0},
    ("REOPEN", "RELEASE"): {"p10": 11.9, "p25": 7614.0, "median": 215921.5, "share_under_60s": 0.214},
    ("FIN", "CODE NOK"): {"p10": 653233.0, "p25": 989636.0, "median": 1645922.0, "share_under_60s": 0.0},
    ("RELEASE", "DELETE"): {"p10": 33312.8, "p25": 123510.0, "median": 137503.0, "share_under_60s": 0.077},
    ("RELEASE", "SET STATUS"): {"p10": 145901.2, "p25": 197064.0, "median": 428082.0, "share_under_60s": 0.0},
    ("MANUAL", "MANUAL"): {"p10": 6.8, "p25": 41.0, "median": 171.0, "share_under_60s": 0.333},
    ("REJECT", "RELEASE"): {"p10": 7.7, "p25": 9.5, "median": 35.0, "share_under_60s": 0.75},
    ("JOIN-PAT", "FIN"): {"p10": 3050054.3000000003, "p25": 4138408.25, "median": 5127431.5, "share_under_60s": 0.0},
    ("CHANGE DIAGN", "JOIN-PAT"): {"p10": 5154.200000000001, "p25": 14279.75, "median": 596185.5, "share_under_60s": 0.0},
    ("MANUAL", "CODE OK"): {"p10": 31.8, "p25": 54.0, "median": 90.0, "share_under_60s": 0.25},
    ("STORNO", "SET STATUS"): {"p10": 12.5, "p25": 16.25, "median": 48.5, "share_under_60s": 0.625},
    ("RELEASE", "FIN"): {"p10": 1529460.8000000003, "p25": 2625254.5, "median": 6350998.0, "share_under_60s": 0.0},
    ("STORNO", "STORNO"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.857},
    ("BILLED", "REJECT"): {"p10": 136092.0, "p25": 1764092.0, "median": 7959808.0, "share_under_60s": 0.0},
    ("RELEASE", "MANUAL"): {"p10": 9625657.0, "p25": 17681111.5, "median": 20815914.0, "share_under_60s": 0.0},
    ("NEW", "JOIN-PAT"): {"p10": 1159796.5, "p25": 1457865.0, "median": 9127796.5, "share_under_60s": 0.0},
    ("FIN", "BILLED"): {"p10": 10045555.5, "p25": 17956271.75, "median": 20730728.0, "share_under_60s": 0.0},
}

BLOCK_SPAN_PROFILE = {
    2: {"p10": 18.0, "p25": 12215.0, "median": 405039.0},
    3: {"p10": 95866.1, "p25": 460038.5, "median": 3831559.5},
    4: {"p10": 384922.4, "p25": 2342423.0, "median": 6968368.0},
    5: {"p10": 440341.0, "p25": 4326198.0, "median": 8294963.0},
}

GLOBAL_MEDIAN = 405039.0
ALL_DFG_EDGES = set(DURATION_PROFILES.keys())


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def detect_precision_and_kind(s):
    if pd.isna(s):
        return {"kind": "blank", "precision": "seconds", "date": None}
    txt = str(s)
    if txt.strip() == "":
        return {"kind": "blank", "precision": "seconds", "date": None}
    t = txt.strip()

    # Explicit date-only
    if len(t) == 10 and t[4] == "-" and t[7] == "-":
        return {"kind": "date_only", "precision": "seconds", "date": t}

    # Date survived, time gone represented as midnight
    try:
        dt = pd.to_datetime(t)
        if ":" not in t:
            return {"kind": "date_only", "precision": "seconds", "date": dt.strftime("%Y-%m-%d")}
        precision = "milliseconds" if "." in t else "seconds"
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
            return {"kind": "date_only", "precision": precision, "date": dt.strftime("%Y-%m-%d")}
        return {"kind": "full", "precision": precision, "date": dt.strftime("%Y-%m-%d")}
    except Exception:
        return {"kind": "blank", "precision": "seconds", "date": None}


def round_half_up_millis(ts):
    ns = ts.value
    ms_ns = 1_000_000
    rounded = int(Decimal(ns / ms_ns).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) * ms_ns
    return pd.Timestamp(rounded)


def format_ts(ts, precision):
    if precision == "milliseconds":
        ts = round_half_up_millis(ts)
        return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def choose_edge_gap_seconds(src, tgt):
    prof = DURATION_PROFILES.get((src, tgt))
    if prof is None:
        return None
    # Use p25 as required by plan.
    return float(prof["p25"])


def choose_small_jitter_seconds(src, tgt, ordinal):
    prof = DURATION_PROFILES.get((src, tgt))
    if prof is not None and prof["share_under_60s"] >= 0.45:
        base = max(0.001, min(2.0, prof["p25"] if prof["p25"] > 0 else 0.2))
        return float(base + 0.137 * (ordinal + 1))
    return float(1.0 + 0.731 * (ordinal + 1))


def block_span_prior_seconds(k):
    prof = BLOCK_SPAN_PROFILE.get(k)
    if prof is not None:
        return float(prof["p25"])
    return float(GLOBAL_MEDIAN)


def supported_local_order(activities):
    if len(activities) <= 1:
        return True
    for a, b in zip(activities[:-1], activities[1:]):
        if (a, b) not in ALL_DFG_EDGES:
            return False
    return True


def infer_block_times(case_df, block_idx_positions):
    idxs = block_idx_positions
    n = len(idxs)
    activities = case_df.loc[idxs, "Activity"].tolist()

    left_pos = idxs[0] - 1 if idxs[0] - 1 in case_df.index else None
    right_pos = idxs[-1] + 1 if idxs[-1] + 1 in case_df.index else None

    left_clean = left_pos is not None and case_df.loc[left_pos, "error_category"] == "CLEAN"
    right_clean = right_pos is not None and case_df.loc[right_pos, "error_category"] == "CLEAN"

    left_anchor = pd.to_datetime(case_df.loc[left_pos, "_timestamp_out"]) if left_clean else None
    right_anchor = pd.to_datetime(case_df.loc[right_pos, "_timestamp_out"]) if right_clean else None

    left_act = case_df.loc[left_pos, "Activity"] if left_pos is not None else None
    right_act = case_df.loc[right_pos, "Activity"] if right_pos is not None else None

    # Validate local observed order via DFG; we do not reorder regardless.
    _ = supported_local_order(([left_act] if left_act is not None else []) + activities + ([right_act] if right_act is not None else []))

    # Forward cumulative gaps from left anchor to each corrupted row
    fwd = [None] * n
    if left_clean:
        cumul = 0.0
        prev = left_act
        for j, act in enumerate(activities):
            gap = choose_edge_gap_seconds(prev, act)
            if gap is None:
                gap = choose_small_jitter_seconds(prev, act, j)
            cumul += gap
            fwd[j] = left_anchor + pd.to_timedelta(cumul, unit="s")
            prev = act

    # Backward cumulative gaps from right anchor to each corrupted row
    bwd = [None] * n
    if right_clean:
        cumul = 0.0
        nxt = right_act
        for j in range(n - 1, -1, -1):
            act = activities[j]
            gap = choose_edge_gap_seconds(act, nxt)
            if gap is None:
                gap = choose_small_jitter_seconds(act, nxt, n - 1 - j)
            cumul += gap
            bwd[j] = right_anchor - pd.to_timedelta(cumul, unit="s")
            nxt = act

    candidates = []
    if left_clean and right_clean:
        window = (right_anchor - left_anchor).total_seconds()
        step_gaps = []
        seq = ([left_act] + activities + [right_act])
        for a, b in zip(seq[:-1], seq[1:]):
            gap = choose_edge_gap_seconds(a, b)
            if gap is None:
                gap = choose_small_jitter_seconds(a, b, len(step_gaps))
            step_gaps.append(float(max(gap, 0.0)))
        implied_span = sum(step_gaps)

        if implied_span <= window and implied_span > 0:
            prev = left_anchor
            local = []
            for j, gap in enumerate(step_gaps[:-1]):
                prev = prev + pd.to_timedelta(gap, unit="s")
                local.append(prev)
            candidates = local
        else:
            # Reconcile inside anchor window, no uniform interpolation.
            # Weighted average of forward/backward candidates, then monotone clamp.
            local = []
            for j in range(n):
                if fwd[j] is not None and bwd[j] is not None:
                    # position-aware weighting, non-uniform
                    wl = max(1.0, n - j)
                    wr = max(1.0, j + 1)
                    x = left_anchor.value * wl + right_anchor.value * wr
                    # blend with process-informed forward/backward estimates
                    blend = (fwd[j].value * wr + bwd[j].value * wl + x) / (2.0 * (wl + wr))
                    local.append(pd.Timestamp(int(blend)))
                elif fwd[j] is not None:
                    local.append(fwd[j])
                elif bwd[j] is not None:
                    local.append(bwd[j])
                else:
                    local.append(left_anchor + (right_anchor - left_anchor) / 2)

            # Monotone enforce with tiny non-uniform jitters and anchor containment
            repaired = []
            prev = left_anchor
            for j, t in enumerate(local):
                min_gap = choose_small_jitter_seconds(
                    left_act if j == 0 else activities[j - 1],
                    activities[j],
                    j
                )
                lower = prev if j == 0 else repaired[-1] + pd.to_timedelta(min_gap, unit="s")
                upper = right_anchor
                if t < lower:
                    t = lower
                if t > upper:
                    remaining = n - j
                    frac = (1.0 / (remaining + 1.3))  # not even spacing
                    t = lower + (upper - lower) * frac
                repaired.append(t)
                prev = t
            candidates = repaired

    elif left_clean and not right_clean:
        local = []
        prev = left_anchor
        seq_prev = left_act
        for j, act in enumerate(activities):
            gap = choose_edge_gap_seconds(seq_prev, act)
            if gap is None:
                gap = block_span_prior_seconds(n) / max(1, n) + choose_small_jitter_seconds(seq_prev, act, j)
            prev = prev + pd.to_timedelta(gap, unit="s")
            local.append(prev)
            seq_prev = act
        candidates = local

    elif right_clean and not left_clean:
        local = [None] * n
        nxt = right_act
        prev_t = right_anchor
        for j in range(n - 1, -1, -1):
            act = activities[j]
            gap = choose_edge_gap_seconds(act, nxt)
            if gap is None:
                gap = block_span_prior_seconds(n) / max(1, n) + choose_small_jitter_seconds(act, nxt, n - 1 - j)
            prev_t = prev_t - pd.to_timedelta(gap, unit="s")
            local[j] = prev_t
            nxt = act
        candidates = local

    else:
        # No clean anchors in case: construct relative chain from arbitrary internal origin,
        # later clamped by date rules if present.
        local = []
        base = pd.Timestamp("2010-01-01 12:00:00")
        prev = base
        prev_act = None
        for j, act in enumerate(activities):
            if prev_act is None:
                gap = choose_small_jitter_seconds(act, act, j)
            else:
                gap = choose_edge_gap_seconds(prev_act, act)
                if gap is None:
                    gap = block_span_prior_seconds(n) / max(1, n) + choose_small_jitter_seconds(prev_act, act, j)
            prev = prev + pd.to_timedelta(gap, unit="s")
            local.append(prev)
            prev_act = act
        candidates = local

    return candidates, left_anchor, right_anchor


def clamp_to_same_date(ts, date_str, precision):
    d0 = pd.Timestamp(date_str + " 00:00:00")
    d1 = pd.Timestamp(date_str + " 23:59:59.999") if precision == "milliseconds" else pd.Timestamp(date_str + " 23:59:59")
    if ts < d0:
        return d0
    if ts > d1:
        return d1
    return ts


def repair_case(case_df):
    case_df = case_df.copy()
    case_df["_timestamp_out"] = case_df["Timestamp"].astype(str)
    case_df["_parsed_clean"] = pd.NaT

    for i in case_df.index:
        if case_df.loc[i, "error_category"] == "CLEAN":
            case_df.loc[i, "_parsed_clean"] = pd.to_datetime(case_df.loc[i, "Timestamp"])

    blank_filled = 0
    dateonly_filled = 0

    # contiguous corrupted blocks in observed order
    corrupted = case_df["error_category"] != "CLEAN"
    idxs = list(case_df.index)
    blocks = []
    start = None
    for i in idxs:
        if corrupted.loc[i] and start is None:
            start = i
        elif (not corrupted.loc[i]) and start is not None:
            blocks.append(list(range(start, i)))
            start = None
    if start is not None:
        blocks.append(list(range(start, idxs[-1] + 1)))

    for block in blocks:
        inferred, left_anchor, right_anchor = infer_block_times(case_df, block)

        for pos_in_block, idx in enumerate(block):
            obs = case_df.loc[idx, "Timestamp"]
            meta = detect_precision_and_kind(obs)
            precision = meta["precision"]
            t = inferred[pos_in_block]

            # For date-only, keep exact observed date
            if meta["kind"] == "date_only" and meta["date"] is not None:
                t = clamp_to_same_date(t, meta["date"], precision)

                # If both anchors exist and are on same date and clamp pushed out, place between anchors on same date
                if left_anchor is not None and right_anchor is not None:
                    lower = max(left_anchor, pd.Timestamp(meta["date"] + " 00:00:00"))
                    upper = min(
                        right_anchor,
                        pd.Timestamp(meta["date"] + (" 23:59:59.999" if precision == "milliseconds" else " 23:59:59"))
                    )
                    if lower <= upper:
                        if t < lower:
                            t = lower
                        if t > upper:
                            t = lower + (upper - lower) * 0.618

            # Self-check containment in bracket
            if left_anchor is not None and t < left_anchor:
                t = left_anchor
            if right_anchor is not None and t > right_anchor:
                if left_anchor is not None and left_anchor <= right_anchor:
                    t = left_anchor + (right_anchor - left_anchor) * 0.618
                else:
                    t = right_anchor

            # Ensure corrupted rows differ from input value
            formatted = format_ts(t, precision)
            if str(obs) == formatted:
                bump = pd.to_timedelta(1, unit="ms" if precision == "milliseconds" else "s")
                if right_anchor is not None and t + bump <= right_anchor:
                    t = t + bump
                elif left_anchor is not None and t - bump >= left_anchor:
                    t = t - bump
                else:
                    t = t + bump
                if meta["kind"] == "date_only" and meta["date"] is not None:
                    t = clamp_to_same_date(t, meta["date"], precision)
                formatted = format_ts(t, precision)

            case_df.loc[idx, "_timestamp_out"] = formatted

            if meta["kind"] == "blank":
                blank_filled += 1
            elif meta["kind"] == "date_only":
                dateonly_filled += 1

    return case_df, blank_filled, dateonly_filled


def global_self_check_and_fix(df):
    still_equal = 0

    # Parseability and exact unchanged clean rows
    for i in df.index:
        if df.loc[i, "error_category"] == "CLEAN":
            if str(df.loc[i, "_timestamp_out"]) != str(df.loc[i, "Timestamp"]):
                df.loc[i, "_timestamp_out"] = df.loc[i, "Timestamp"]
        else:
            # must be parseable and non-blank
            out = str(df.loc[i, "_timestamp_out"])
            if out.strip() == "":
                # final emergency relative fix using nearest same-case clean/repaired timestamps
                case = df.loc[i, "Case"]
                case_rows = df[df["Case"] == case]
                local_idx = list(case_rows.index).index(i)
                prev_ts = None
                next_ts = None
                if local_idx > 0:
                    prev_i = case_rows.index[local_idx - 1]
                    prev_ts = pd.to_datetime(df.loc[prev_i, "_timestamp_out"])
                if local_idx < len(case_rows.index) - 1:
                    next_i = case_rows.index[local_idx + 1]
                    next_ts = pd.to_datetime(df.loc[next_i, "_timestamp_out"])
                if prev_ts is not None and next_ts is not None:
                    cand = prev_ts + (next_ts - prev_ts) * 0.618
                elif prev_ts is not None:
                    cand = prev_ts + pd.to_timedelta(GLOBAL_MEDIAN, unit="s")
                elif next_ts is not None:
                    cand = next_ts - pd.to_timedelta(GLOBAL_MEDIAN, unit="s")
                else:
                    cand = pd.Timestamp("2010-01-01 12:00:00")
                meta = detect_precision_and_kind(df.loc[i, "Timestamp"])
                if meta["kind"] == "date_only" and meta["date"] is not None:
                    cand = clamp_to_same_date(cand, meta["date"], meta["precision"])
                df.loc[i, "_timestamp_out"] = format_ts(cand, meta["precision"])

            # parseability
            try:
                parsed = pd.to_datetime(df.loc[i, "_timestamp_out"])
            except Exception:
                meta = detect_precision_and_kind(df.loc[i, "Timestamp"])
                fallback = pd.Timestamp("2010-01-01 12:00:01")
                if meta["kind"] == "date_only" and meta["date"] is not None:
                    fallback = pd.Timestamp(meta["date"] + " 12:00:00")
                df.loc[i, "_timestamp_out"] = format_ts(fallback, meta["precision"])

            # date-only same-date guarantee
            meta = detect_precision_and_kind(df.loc[i, "Timestamp"])
            if meta["kind"] == "date_only" and meta["date"] is not None:
                parsed = pd.to_datetime(df.loc[i, "_timestamp_out"])
                if parsed.strftime("%Y-%m-%d") != meta["date"]:
                    parsed = pd.Timestamp(meta["date"] + " 12:00:00")
                    df.loc[i, "_timestamp_out"] = format_ts(parsed, meta["precision"])

            if str(df.loc[i, "_timestamp_out"]) == str(df.loc[i, "Timestamp"]):
                still_equal += 1

    return df, still_equal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    required_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    if list(df.columns) != required_cols:
        # Preserve all required columns but enforce expected set/order
        df = df[required_cols]

    original = df.copy()

    # Process case by case, preserving row order globally
    repaired_parts = []
    total_blank = 0
    total_dateonly = 0

    for case_value, case_df in df.groupby("Case", sort=False):
        case_df = case_df.copy()
        repaired_case, bcnt, dcnt = repair_case(case_df)
        repaired_parts.append(repaired_case)
        total_blank += bcnt
        total_dateonly += dcnt

    out = pd.concat(repaired_parts).sort_index()
    out, still_equal = global_self_check_and_fix(out)

    # Final hard checks
    assert len(out) == len(original)
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        assert out[col].tolist() == original[col].tolist()
    assert out.index.tolist() == original.index.tolist()
    clean_mask = out["error_category"] == "CLEAN"
    assert out.loc[clean_mask, "_timestamp_out"].tolist() == original.loc[clean_mask, "Timestamp"].tolist()

    # parseability of repaired rows
    for i in out.index[out["error_category"] != "CLEAN"]:
        pd.to_datetime(out.loc[i, "_timestamp_out"])
        assert str(out.loc[i, "_timestamp_out"]).strip() != ""
        assert str(out.loc[i, "_timestamp_out"]) != str(original.loc[i, "Timestamp"])
        meta = detect_precision_and_kind(original.loc[i, "Timestamp"])
        if meta["kind"] == "date_only" and meta["date"] is not None:
            assert pd.to_datetime(out.loc[i, "_timestamp_out"]).strftime("%Y-%m-%d") == meta["date"]

    result = original.copy()
    result["Timestamp"] = out["_timestamp_out"]

    result.to_csv(args.output, index=False)

    print(f"blank_rows_filled={total_blank}, date_only_rows_given_time={total_dateonly}, rows_still_equal_to_input={still_equal}")


if __name__ == "__main__":
    main()