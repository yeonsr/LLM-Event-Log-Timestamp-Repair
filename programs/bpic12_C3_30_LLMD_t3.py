import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import timedelta

import pandas as pd


# ---------------------------------------------------------------------
# Embedded process knowledge from CLEAN rows only
# ---------------------------------------------------------------------

DURATION_PROFILES = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): {"p10": 50.5901, "p25": 14888.91575, "median": 167356.1795, "share_under_60s": 0.104},
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): {"p10": 10.8058, "p25": 41.519, "median": 99.459, "share_under_60s": 0.336},
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): {"p10": 18.568, "p25": 56.637, "median": 115.945, "share_under_60s": 0.265},
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 61.7376, "p25": 1598.129, "median": 14639.161, "share_under_60s": 0.099},
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): {"p10": 0.126, "p25": 0.175, "median": 0.275, "share_under_60s": 1.0},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 76.925, "p25": 1284.3235, "median": 10784.402, "share_under_60s": 0.093},
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 10.2459, "p25": 33.62275, "median": 140.2915, "share_under_60s": 0.33},
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): {"p10": 0.046, "p25": 0.13325, "median": 0.378, "share_under_60s": 1.0},
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): {"p10": 0.025, "p25": 0.028, "median": 0.035, "share_under_60s": 1.0},
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): {"p10": 0.083, "p25": 0.093, "median": 0.119, "share_under_60s": 1.0},
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9155, "p25": 1.08425, "median": 1.291, "share_under_60s": 1.0},
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): {"p10": 125.3631, "p25": 247.7525, "median": 373.9065, "share_under_60s": 0.048},
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): {"p10": 12.5822, "p25": 20.106, "median": 310.933, "share_under_60s": 0.421},
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.077, "p25": 1.18, "median": 1.3905, "share_under_60s": 1.0},
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): {"p10": 30.8955, "p25": 742.71875, "median": 6597.786, "share_under_60s": 0.144},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): {"p10": 32.8288, "p25": 35.5895, "median": 38.9175, "share_under_60s": 0.971},
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): {"p10": 83.1826, "p25": 597.00775, "median": 2551.737, "share_under_60s": 0.085},
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): {"p10": 8.6702, "p25": 11.885, "median": 22.736, "share_under_60s": 0.949},
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 63.8074, "p25": 118.415, "median": 183.87, "share_under_60s": 0.093},
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): {"p10": 0.141, "p25": 0.153, "median": 0.179, "share_under_60s": 0.999},
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): {"p10": 10.1838, "p25": 12.41, "median": 21.162, "share_under_60s": 0.827},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.115, "p25": 1.228, "median": 1.447, "share_under_60s": 1.0},
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 7064.724, "p25": 89340.6535, "median": 189136.3675, "share_under_60s": 0.049},
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): {"p10": 10.7182, "p25": 44.835, "median": 307.187, "share_under_60s": 0.284},
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 24.2998, "p25": 244.81225, "median": 2005.5505, "share_under_60s": 0.144},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.724, "p25": 34.09525, "median": 37.348, "share_under_60s": 0.977},
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 18.2621, "p25": 908.79625, "median": 5975.98, "share_under_60s": 0.148},
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): {"p10": 35.6547, "p25": 66.4165, "median": 107.805, "share_under_60s": 0.21},
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): {"p10": 0.898, "p25": 0.987, "median": 1.148, "share_under_60s": 1.0},
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 26.1244, "p25": 332.025, "median": 2855.616, "share_under_60s": 0.155},
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): {"p10": 1.446, "p25": 1.605, "median": 1.843, "share_under_60s": 1.0},
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): {"p10": 2.843, "p25": 3.0815, "median": 3.41, "share_under_60s": 0.998},
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): {"p10": 32.1372, "p25": 69.84475, "median": 113.2505, "share_under_60s": 0.203},
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): {"p10": 254.4338, "p25": 623.07, "median": 1174.31, "share_under_60s": 0.029},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 1.486, "p25": 1.649, "median": 1.895, "share_under_60s": 1.0},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 696.887, "p25": 2100.1995, "median": 5033.333, "share_under_60s": 0.031},
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): {"p10": 31.965, "p25": 48.504, "median": 106.471, "share_under_60s": 0.316},
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.2212, "p25": 1.3115, "median": 1.618, "share_under_60s": 0.999},
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): {"p10": 24.7997, "p25": 36.5235, "median": 65.3805, "share_under_60s": 0.466},
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share_under_60s": 1.0},
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): {"p10": 8.0472, "p25": 46.988, "median": 135.472, "share_under_60s": 0.292},
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 2.9624, "p25": 3.184, "median": 3.533, "share_under_60s": 0.999},
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): {"p10": 3.8414, "p25": 17.07475, "median": 362.5105, "share_under_60s": 0.327},
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): {"p10": 179.8682, "p25": 315.4015, "median": 474.467, "share_under_60s": 0.022},
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.4536, "p25": 7.027, "median": 7.763, "share_under_60s": 0.999},
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.837, "p25": 2.093, "median": 2.396, "share_under_60s": 0.998},
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): {"p10": 72.662, "p25": 111.11025, "median": 156.877, "share_under_60s": 0.081},
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.3086, "p25": 6.952, "median": 7.6685, "share_under_60s": 1.0},
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): {"p10": 141.3626, "p25": 317.4815, "median": 770.573, "share_under_60s": 0.019},
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("A_SUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.7182, "p25": 34.433, "median": 38.218, "share_under_60s": 0.987},
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.3078, "p25": 1.4635, "median": 1.661, "share_under_60s": 0.999},
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9084, "p25": 1.053, "median": 1.251, "share_under_60s": 1.0},
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share_under_60s": 1.0},
}

DFG_EDGES = set(DURATION_PROFILES.keys())

BLOCK_SPAN_P25 = {
    2: 1.305,
    3: 36.364,
    4: 217.364,
    5: 866.1295,
}
GLOBAL_MEDIAN_DURATION = 49.725


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_to_datetime(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def is_blank_ts(s):
    return s is None or str(s).strip() == ""


def parse_observed_date(ts_str):
    if ts_str is None:
        return None
    s = str(ts_str).strip()
    if not s:
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if not m:
        return None
    try:
        return pd.Timestamp(m.group(1)).date()
    except Exception:
        return None


def detect_precision(rows):
    max_digits = 0
    for r in rows:
        s = str(r["Timestamp"])
        m = re.search(r"\.(\d+)$", s)
        if m:
            max_digits = max(max_digits, len(m.group(1)))
    return 3 if max_digits >= 3 else max_digits


def observed_has_date_only_or_midnight(ts_str):
    if ts_str is None:
        return False
    s = str(ts_str).strip()
    if not s:
        return False
    dt = safe_to_datetime(s)
    if pd.isna(dt):
        return False
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return True
    if re.match(r"^\d{4}-\d{2}-\d{2}\s+00:00:00(?:\.0+)?$", s):
        return True
    return False


def format_ts(ts, precision):
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    frac = f"{ts.microsecond:06d}"[:precision]
    return f"{base}.{frac}"


def timedel(seconds):
    return pd.Timedelta(seconds=float(seconds))


def choose_edge_gap(a, b):
    prof = DURATION_PROFILES.get((a, b))
    if prof:
        if prof["share_under_60s"] >= 0.7:
            return max(prof["p25"], 0.001)
        if prof["share_under_60s"] >= 0.25:
            return max((prof["p10"] * 0.6 + prof["p25"] * 0.4), 0.001)
        return max(prof["p10"], 0.001)
    return None


def default_small_gap(pos):
    seq = [0.731, 1.217, 2.113, 3.317, 5.123, 8.219]
    return seq[(pos - 1) % len(seq)]


def run_cap_seconds(k):
    if k in BLOCK_SPAN_P25:
        return max(BLOCK_SPAN_P25[k], 1.0)
    return max(BLOCK_SPAN_P25[5] * (1 + 0.5 * (k - 5)), GLOBAL_MEDIAN_DURATION)


def date_floor_ceil(d, precision):
    start = pd.Timestamp(d)
    if precision >= 3:
        end = start + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    elif precision == 0:
        end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return start, end


def unique_monotone_within(bounds_lo, bounds_hi, n, precision):
    eps = pd.Timedelta(milliseconds=1 if precision >= 3 else 1)
    if precision == 0:
        eps = pd.Timedelta(seconds=1)
    if n == 1:
        mid = bounds_lo + (bounds_hi - bounds_lo) / 2
        return [min(max(mid, bounds_lo), bounds_hi)]
    total = bounds_hi - bounds_lo
    if total <= eps * (n - 1):
        out = []
        for i in range(n):
            out.append(bounds_lo + min(total, eps * i))
        return out
    ratios = [1.0 + (0.17 if i % 2 == 0 else 0.41) + 0.09 * i for i in range(n - 1)]
    s = sum(ratios)
    remaining = total - eps * (n - 1)
    gaps = [eps + remaining * (r / s) for r in ratios]
    out = [bounds_lo]
    for g in gaps:
        out.append(out[-1] + g)
    out[-1] = min(out[-1], bounds_hi)
    return out


def candidate_from_left(left_ts, left_act, target_act, row_pos):
    g = choose_edge_gap(left_act, target_act)
    if g is None:
        g = default_small_gap(row_pos)
    return left_ts + timedel(g)


def candidate_from_right(right_ts, source_act, right_act, row_pos):
    g = choose_edge_gap(source_act, right_act)
    if g is None:
        g = default_small_gap(row_pos)
    return right_ts - timedel(g)


def weighted_between(left_ts, right_ts, acts, left_act, right_act, precision):
    n = len(acts)
    if n == 0:
        return []
    slack = right_ts - left_ts
    if slack < pd.Timedelta(0):
        slack = pd.Timedelta(0)

    if n == 1:
        w1 = choose_edge_gap(left_act, acts[0])
        w2 = choose_edge_gap(acts[0], right_act)
        if w1 is None:
            w1 = 1.0
        if w2 is None:
            w2 = 1.7
        frac = w1 / (w1 + w2) if (w1 + w2) > 0 else 0.5
        t = left_ts + slack * frac
        return [t]

    edge_weights = []
    first = choose_edge_gap(left_act, acts[0])
    edge_weights.append(first if first is not None else default_small_gap(1))
    for i in range(n - 1):
        g = choose_edge_gap(acts[i], acts[i + 1])
        edge_weights.append(g if g is not None else default_small_gap(i + 2))
    last = choose_edge_gap(acts[-1], right_act)
    edge_weights.append(last if last is not None else default_small_gap(n + 1))

    cum = []
    total = 0.0
    for w in edge_weights:
        total += max(w, 0.001)
        cum.append(total)

    out = []
    for i in range(n):
        frac = cum[i] / total if total > 0 else (i + 1) / (n + 1)
        out.append(left_ts + slack * frac)

    out = enforce_nonuniform_and_order(out, left_ts, right_ts, precision)
    return out


def enforce_nonuniform_and_order(times, left_bound, right_bound, precision):
    if not times:
        return times
    eps = pd.Timedelta(milliseconds=1 if precision >= 3 else 1)
    if precision == 0:
        eps = pd.Timedelta(seconds=1)

    out = times[:]
    out[0] = max(out[0], left_bound)
    for i in range(1, len(out)):
        min_allowed = max(out[i - 1], left_bound)
        if out[i] < min_allowed:
            out[i] = min_allowed

    for i in range(len(out) - 2, -1, -1):
        if out[i] > right_bound:
            out[i] = right_bound
        if out[i] > out[i + 1]:
            out[i] = out[i + 1]

    out = [min(max(t, left_bound), right_bound) for t in out]

    if len(out) >= 2:
        diffs = []
        for i in range(1, len(out)):
            diffs.append((out[i] - out[i - 1]).total_seconds())
        if len(set(round(x, 6) for x in diffs)) == 1:
            window = right_bound - left_bound
            if window >= eps * (len(out) + 1):
                out = unique_monotone_within(left_bound, right_bound, len(out), precision)
    return out


def repair_run(case_rows, start_idx, end_idx, precision):
    acts = [case_rows[i]["Activity"] for i in range(start_idx, end_idx + 1)]
    run_len = len(acts)
    left_idx = start_idx - 1
    right_idx = end_idx + 1

    left_clean = left_idx >= 0 and case_rows[left_idx]["error_category"] == "CLEAN"
    right_clean = right_idx < len(case_rows) and case_rows[right_idx]["error_category"] == "CLEAN"

    left_ts = safe_to_datetime(case_rows[left_idx]["Timestamp"]) if left_clean else pd.NaT
    right_ts = safe_to_datetime(case_rows[right_idx]["Timestamp"]) if right_clean else pd.NaT
    left_act = case_rows[left_idx]["Activity"] if left_clean else None
    right_act = case_rows[right_idx]["Activity"] if right_clean else None

    case_clean_times = [safe_to_datetime(r["Timestamp"]) for r in case_rows if r["error_category"] == "CLEAN" and not pd.isna(safe_to_datetime(r["Timestamp"]))]
    if case_clean_times:
        case_start = min(case_clean_times)
        case_end = max(case_clean_times)
    else:
        now = pd.Timestamp("2011-01-01 12:00:00")
        case_start = now
        case_end = now

    repaired = [None] * run_len

    observed_dates = [parse_observed_date(case_rows[start_idx + j]["Timestamp"]) for j in range(run_len)]
    has_survived_date = [d is not None and observed_has_date_only_or_midnight(case_rows[start_idx + j]["Timestamp"]) for j, d in enumerate(observed_dates)]

    if left_clean and right_clean and not pd.isna(left_ts) and not pd.isna(right_ts) and left_ts <= right_ts:
        anchors_left = left_ts
        anchors_right = right_ts

        weighted = weighted_between(anchors_left, anchors_right, acts, left_act, right_act, precision)

        for j in range(run_len):
            row = case_rows[start_idx + j]
            cand = weighted[j]

            if has_survived_date[j]:
                d = observed_dates[j]
                d0, d1 = date_floor_ceil(d, precision)
                lo = max(anchors_left, d0)
                hi = min(anchors_right, d1)
                if lo <= hi:
                    left_proj = candidate_from_left(left_ts, left_act, acts[j], j + 1)
                    right_proj = candidate_from_right(right_ts, acts[j], right_act, run_len - j)
                    picks = [cand]
                    if left_proj is not None:
                        picks.append(left_proj)
                    if right_proj is not None:
                        picks.append(right_proj)
                    picks = [min(max(p, lo), hi) for p in picks]
                    obs_dt = safe_to_datetime(row["Timestamp"])
                    if not pd.isna(obs_dt):
                        midnight = pd.Timestamp(d)
                        obs_seconds = (obs_dt - midnight).total_seconds()
                        best = min(
                            picks,
                            key=lambda p: abs((p - midnight).total_seconds() - obs_seconds)
                        )
                    else:
                        best = min(
                            picks,
                            key=lambda p: abs((p - cand).total_seconds())
                        )
                    cand = best
                else:
                    cand = min(max(cand, anchors_left), anchors_right)

            repaired[j] = cand

        repaired = enforce_nonuniform_and_order(repaired, anchors_left, anchors_right, precision)

    elif left_clean and not pd.isna(left_ts):
        current = left_ts
        for j in range(run_len):
            row = case_rows[start_idx + j]
            act = acts[j]
            cand = candidate_from_left(current, left_act if j == 0 else acts[j - 1], act, j + 1)

            if has_survived_date[j]:
                d = observed_dates[j]
                d0, d1 = date_floor_ceil(d, precision)
                lo = max(case_start, d0)
                hi = min(case_end, d1)
                cand = min(max(cand, lo), hi)
            else:
                cap = left_ts + timedel(run_cap_seconds(run_len))
                hi = min(case_end, cap)
                cand = min(max(cand, case_start), hi)

            repaired[j] = cand
            current = cand

    elif right_clean and not pd.isna(right_ts):
        current = right_ts
        for j in range(run_len - 1, -1, -1):
            row = case_rows[start_idx + j]
            act = acts[j]
            next_act = right_act if j == run_len - 1 else acts[j + 1]
            cand = candidate_from_right(current, act, next_act, run_len - j)

            if has_survived_date[j]:
                d = observed_dates[j]
                d0, d1 = date_floor_ceil(d, precision)
                lo = max(case_start, d0)
                hi = min(case_end, d1)
                cand = min(max(cand, lo), hi)
            else:
                cap = right_ts - timedel(run_cap_seconds(run_len))
                lo = max(case_start, cap)
                cand = min(max(cand, lo), case_end)

            repaired[j] = cand
            current = cand

        repaired = enforce_nonuniform_and_order(repaired, case_start, right_ts, precision)

    else:
        base_dates = [d for d in observed_dates if d is not None]
        if base_dates:
            d = base_dates[0]
            d0, d1 = date_floor_ceil(d, precision)
            repaired = unique_monotone_within(d0 + timedel(9.0), min(d1, d0 + timedel(9.0 + run_cap_seconds(run_len))), run_len, precision)
        else:
            base = case_start
            repaired = unique_monotone_within(base, min(case_end, base + timedel(run_cap_seconds(run_len))), run_len, precision)

    # final rowwise clamping and "same date" preservation
    for j in range(run_len):
        row = case_rows[start_idx + j]
        t = repaired[j]

        if pd.isna(t):
            t = case_start

        t = min(max(t, case_start), case_end)

        if left_clean and right_clean and not pd.isna(left_ts) and not pd.isna(right_ts):
            t = min(max(t, left_ts), right_ts)

        if has_survived_date[j]:
            d = observed_dates[j]
            d0, d1 = date_floor_ceil(d, precision)
            lo = max(case_start, d0)
            hi = min(case_end, d1)
            if left_clean and right_clean and not pd.isna(left_ts) and not pd.isna(right_ts):
                lo = max(lo, left_ts)
                hi = min(hi, right_ts)
            elif left_clean and not pd.isna(left_ts):
                lo = max(lo, left_ts)
            elif right_clean and not pd.isna(right_ts):
                hi = min(hi, right_ts)
            if lo <= hi:
                t = min(max(t, lo), hi)
            else:
                if left_clean and right_clean and left_ts <= right_ts:
                    t = left_ts + (right_ts - left_ts) / 2
                elif left_clean:
                    t = left_ts
                elif right_clean:
                    t = right_ts
                t = min(max(t, case_start), case_end)

        repaired[j] = t

    repaired = enforce_nonuniform_and_order(
        repaired,
        left_ts if left_clean and not pd.isna(left_ts) else min(repaired),
        right_ts if right_clean and not pd.isna(right_ts) else max(repaired),
        precision
    )

    return repaired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        with open(args.input, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames
    except Exception:
        rows = []
        fieldnames = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]

    if not rows:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print("blank_rows_filled=0 date_only_rows_given_time=0 rows_still_equal_to_input=0")
        return

    precision = detect_precision(rows)

    for r in rows:
        r["_orig_ts"] = r["Timestamp"]
        r["_new_ts"] = r["Timestamp"]

    case_to_indices = defaultdict(list)
    for i, r in enumerate(rows):
        case_to_indices[r["Case"]].append(i)

    blank_filled = 0
    date_only_filled = 0

    for case, idxs in case_to_indices.items():
        case_rows = [rows[i] for i in idxs]
        n = len(case_rows)
        pos = 0
        while pos < n:
            if case_rows[pos]["error_category"] == "CLEAN":
                pos += 1
                continue
            start = pos
            while pos + 1 < n and case_rows[pos + 1]["error_category"] != "CLEAN":
                pos += 1
            end = pos

            repaired_times = repair_run(case_rows, start, end, precision)

            for local_j, t in enumerate(repaired_times):
                global_idx = idxs[start + local_j]
                orig = rows[global_idx]["Timestamp"]
                rows[global_idx]["_new_ts"] = format_ts(t, precision)

                if is_blank_ts(orig):
                    blank_filled += 1
                elif observed_has_date_only_or_midnight(orig):
                    date_only_filled += 1
            pos += 1

    # Self-check and repair in-place
    still_equal = 0
    for case, idxs in case_to_indices.items():
        case_rows = [rows[i] for i in idxs]
        clean_times = []
        for r in case_rows:
            if r["error_category"] == "CLEAN":
                dt = safe_to_datetime(r["Timestamp"])
                if not pd.isna(dt):
                    clean_times.append(dt)
        if clean_times:
            case_start = min(clean_times)
            case_end = max(clean_times)
        else:
            case_start = pd.Timestamp("2011-01-01 00:00:00")
            case_end = case_start + pd.Timedelta(seconds=1)

        for local_idx, global_idx in enumerate(idxs):
            r = rows[global_idx]
            if r["error_category"] == "CLEAN":
                r["_new_ts"] = r["_orig_ts"]
                continue

            dt = safe_to_datetime(r["_new_ts"])
            if pd.isna(dt):
                dt = case_start + timedel(default_small_gap(local_idx + 1))
                r["_new_ts"] = format_ts(dt, precision)

            # anchor checks
            left_anchor = None
            right_anchor = None
            for j in range(local_idx - 1, -1, -1):
                rr = case_rows[j]
                if rr["error_category"] == "CLEAN":
                    left_anchor = safe_to_datetime(rr["Timestamp"])
                    break
            for j in range(local_idx + 1, len(case_rows)):
                rr = case_rows[j]
                if rr["error_category"] == "CLEAN":
                    right_anchor = safe_to_datetime(rr["Timestamp"])
                    break

            dt = safe_to_datetime(r["_new_ts"])
            dt = min(max(dt, case_start), case_end)

            if left_anchor is not None and not pd.isna(left_anchor):
                dt = max(dt, left_anchor)
            if right_anchor is not None and not pd.isna(right_anchor):
                dt = min(dt, right_anchor)

            if observed_has_date_only_or_midnight(r["_orig_ts"]):
                d = parse_observed_date(r["_orig_ts"])
                if d is not None:
                    d0, d1 = date_floor_ceil(d, precision)
                    lo = max(case_start, d0)
                    hi = min(case_end, d1)
                    if left_anchor is not None and not pd.isna(left_anchor):
                        lo = max(lo, left_anchor)
                    if right_anchor is not None and not pd.isna(right_anchor):
                        hi = min(hi, right_anchor)
                    if lo <= hi:
                        dt = min(max(dt, lo), hi)

            candidate = format_ts(dt, precision)
            if candidate == r["_orig_ts"]:
                obs_dt = safe_to_datetime(r["_orig_ts"])
                if not pd.isna(obs_dt):
                    d = parse_observed_date(r["_orig_ts"])
                    delta = pd.Timedelta(milliseconds=1 if precision >= 3 else 1)
                    if precision == 0:
                        delta = pd.Timedelta(seconds=1)

                    cand1 = dt + delta
                    cand2 = dt - delta
                    options = [cand1, cand2]
                    picked = None
                    for c in options:
                        ok = True
                        if c < case_start or c > case_end:
                            ok = False
                        if left_anchor is not None and not pd.isna(left_anchor) and c < left_anchor:
                            ok = False
                        if right_anchor is not None and not pd.isna(right_anchor) and c > right_anchor:
                            ok = False
                        if d is not None and c.date() != d:
                            ok = False
                        if ok and format_ts(c, precision) != r["_orig_ts"]:
                            picked = c
                            break
                    if picked is None:
                        lo = dt
                        hi = dt
                        if left_anchor is not None and not pd.isna(left_anchor):
                            lo = max(lo, left_anchor)
                        if right_anchor is not None and not pd.isna(right_anchor):
                            hi = min(hi, right_anchor)
                        picked = dt
                    candidate = format_ts(picked, precision)
                else:
                    base = case_start
                    candidate = format_ts(base, precision)

            r["_new_ts"] = candidate
            if r["_new_ts"] == r["_orig_ts"]:
                still_equal += 1

    # write output preserving all untouched columns exactly
    out_rows = []
    for r in rows:
        out = {
            "Case": r["Case"],
            "Activity": r["Activity"],
            "Timestamp": r["_new_ts"] if r["error_category"] != "CLEAN" else r["Timestamp"],
            "Resource": r["Resource"],
            "error_category": r["error_category"],
            "row_id": r["row_id"],
        }
        out_rows.append(out)

    try:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
    except Exception:
        try:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])
                writer.writeheader()
                writer.writerows(out_rows)
        except Exception:
            pass

    print(f"blank_rows_filled={blank_filled} date_only_rows_given_time={date_only_filled} rows_still_equal_to_input={still_equal}")


if __name__ == "__main__":
    main()