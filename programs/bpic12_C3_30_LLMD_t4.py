import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
import re

# ============================================================
# DURATION / DFG PRIORS FROM PROVIDED PROCESS CONTEXT
# ============================================================

EDGE_STATS = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): {"p10": 50.5901, "p25": 14888.91575, "median": 167356.1795, "share_under_60s": 0.104, "freq": 16994},
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): {"p10": 10.8058, "p25": 41.519, "median": 99.459, "share_under_60s": 0.336, "freq": 16243},
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): {"p10": 18.568, "p25": 56.637, "median": 115.945, "share_under_60s": 0.265, "freq": 16181},
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 61.7376, "p25": 1598.129, "median": 14639.161, "share_under_60s": 0.099, "freq": 15763},
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): {"p10": 0.126, "p25": 0.175, "median": 0.275, "share_under_60s": 1.0, "freq": 12014},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 76.925, "p25": 1284.3235, "median": 10784.402, "share_under_60s": 0.093, "freq": 9103},
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 10.2459, "p25": 33.62275, "median": 140.2915, "share_under_60s": 0.33, "freq": 8970},
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): {"p10": 0.046, "p25": 0.13325, "median": 0.378, "share_under_60s": 1.0, "freq": 6954},
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): {"p10": 0.025, "p25": 0.028, "median": 0.035, "share_under_60s": 1.0, "freq": 6885},
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): {"p10": 0.083, "p25": 0.093, "median": 0.119, "share_under_60s": 1.0, "freq": 6359},
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9155, "p25": 1.08425, "median": 1.291, "share_under_60s": 1.0, "freq": 6306},
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): {"p10": 125.3631, "p25": 247.7525, "median": 373.9065, "share_under_60s": 0.048, "freq": 4962},
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): {"p10": 12.5822, "p25": 20.106, "median": 310.933, "share_under_60s": 0.421, "freq": 4837},
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.077, "p25": 1.18, "median": 1.3905, "share_under_60s": 1.0, "freq": 4786},
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): {"p10": 30.8955, "p25": 742.71875, "median": 6597.786, "share_under_60s": 0.144, "freq": 4616},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): {"p10": 32.8288, "p25": 35.5895, "median": 38.9175, "share_under_60s": 0.971, "freq": 4550},
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): {"p10": 83.1826, "p25": 597.00775, "median": 2551.737, "share_under_60s": 0.085, "freq": 4422},
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): {"p10": 8.6702, "p25": 11.885, "median": 22.736, "share_under_60s": 0.949, "freq": 4409},
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "freq": 4289},
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 63.8074, "p25": 118.415, "median": 183.87, "share_under_60s": 0.093, "freq": 4254},
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): {"p10": 0.141, "p25": 0.153, "median": 0.179, "share_under_60s": 0.999, "freq": 3371},
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): {"p10": 10.1838, "p25": 12.41, "median": 21.162, "share_under_60s": 0.827, "freq": 3177},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.115, "p25": 1.228, "median": 1.447, "share_under_60s": 1.0, "freq": 3175},
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 7064.724, "p25": 89340.6535, "median": 189136.3675, "share_under_60s": 0.049, "freq": 3150},
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): {"p10": 10.7182, "p25": 44.835, "median": 307.187, "share_under_60s": 0.284, "freq": 2907},
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 24.2998, "p25": 244.81225, "median": 2005.5505, "share_under_60s": 0.144, "freq": 2882},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.724, "p25": 34.09525, "median": 37.348, "share_under_60s": 0.977, "freq": 2666},
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 18.2621, "p25": 908.79625, "median": 5975.98, "share_under_60s": 0.148, "freq": 2422},
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): {"p10": 35.6547, "p25": 66.4165, "median": 107.805, "share_under_60s": 0.21, "freq": 2410},
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): {"p10": 0.898, "p25": 0.987, "median": 1.148, "share_under_60s": 1.0, "freq": 2407},
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 26.1244, "p25": 332.025, "median": 2855.616, "share_under_60s": 0.155, "freq": 2145},
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): {"p10": 1.446, "p25": 1.605, "median": 1.843, "share_under_60s": 1.0, "freq": 2136},
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): {"p10": 2.843, "p25": 3.0815, "median": 3.41, "share_under_60s": 0.998, "freq": 2123},
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "freq": 2014},
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): {"p10": 32.1372, "p25": 69.84475, "median": 113.2505, "share_under_60s": 0.203, "freq": 2010},
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): {"p10": 254.4338, "p25": 623.07, "median": 1174.31, "share_under_60s": 0.029, "freq": 1953},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 1.486, "p25": 1.649, "median": 1.895, "share_under_60s": 1.0, "freq": 1711},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 696.887, "p25": 2100.1995, "median": 5033.333, "share_under_60s": 0.031, "freq": 1711},
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "freq": 1662},
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "freq": 1638},
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): {"p10": 31.965, "p25": 48.504, "median": 106.471, "share_under_60s": 0.316, "freq": 1621},
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.2212, "p25": 1.3115, "median": 1.618, "share_under_60s": 0.999, "freq": 1563},
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): {"p10": 24.7997, "p25": 36.5235, "median": 65.3805, "share_under_60s": 0.466, "freq": 1520},
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "freq": 1466},
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share_under_60s": 1.0, "freq": 1285},
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): {"p10": 8.0472, "p25": 46.988, "median": 135.472, "share_under_60s": 0.292, "freq": 1197},
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "freq": 1126},
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 2.9624, "p25": 3.184, "median": 3.533, "share_under_60s": 0.999, "freq": 1085},
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): {"p10": 3.8414, "p25": 17.07475, "median": 362.5105, "share_under_60s": 0.327, "freq": 1080},
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): {"p10": 179.8682, "p25": 315.4015, "median": 474.467, "share_under_60s": 0.022, "freq": 1038},
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.4536, "p25": 7.027, "median": 7.763, "share_under_60s": 0.999, "freq": 1009},
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.837, "p25": 2.093, "median": 2.396, "share_under_60s": 0.998, "freq": 993},
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): {"p10": 72.662, "p25": 111.11025, "median": 156.877, "share_under_60s": 0.081, "freq": 912},
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.3086, "p25": 6.952, "median": 7.6685, "share_under_60s": 1.0, "freq": 862},
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): {"p10": 141.3626, "p25": 317.4815, "median": 770.573, "share_under_60s": 0.019, "freq": 862},
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "freq": 801},
    ("A_SUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.7182, "p25": 34.433, "median": 38.218, "share_under_60s": 0.987, "freq": 763},
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.3078, "p25": 1.4635, "median": 1.661, "share_under_60s": 0.999, "freq": 747},
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9084, "p25": 1.053, "median": 1.251, "share_under_60s": 1.0, "freq": 717},
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share_under_60s": 1.0, "freq": 716},
}

BLOCK_SPAN = {
    2: {"p10": 0.063, "p25": 1.305, "median": 49.725},
    3: {"p10": 1.366, "p25": 36.364, "median": 995.679},
    4: {"p10": 5.7904, "p25": 217.364, "median": 6361.145},
    5: {"p10": 50.3772, "p25": 866.1295, "median": 21853.089},
}
GLOBAL_MEDIAN = 49.725

# ============================================================
# HELPERS
# ============================================================

def detect_precision(series):
    max_digits = 0
    for s in series.astype(str):
        m = re.search(r"\.(\d+)$", s.strip())
        if m:
            max_digits = max(max_digits, len(m.group(1)))
    return max_digits

def parse_ts_safe(s):
    if pd.isna(s):
        return pd.NaT
    st = str(s).strip()
    if st == "":
        return pd.NaT
    try:
        return pd.to_datetime(st)
    except Exception:
        return pd.NaT

def is_blank(s):
    return pd.isna(s) or str(s).strip() == ""

def is_date_only_or_midnight_loss(s):
    if pd.isna(s):
        return False
    st = str(s).strip()
    if st == "":
        return False
    # explicit date-only
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", st):
        return True
    # damaged value often rendered with midnight
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]00:00:00(?:\.0+)?", st):
        return True
    return False

def same_calendar_date(dt, date_obj):
    return dt.date() == date_obj

def floor_to_precision(dt, precision):
    if precision <= 0:
        return dt.replace(microsecond=0)
    if precision >= 6:
        return dt
    factor = 10 ** (6 - precision)
    us = (dt.microsecond // factor) * factor
    return dt.replace(microsecond=us)

def format_dt(dt, precision):
    dt = floor_to_precision(pd.Timestamp(dt).to_pydatetime(), precision)
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    if precision > 0:
        frac = f"{dt.microsecond:06d}"[:precision]
        return f"{base}.{frac}"
    return base

def edge_duration_seconds(src, tgt):
    stat = EDGE_STATS.get((src, tgt))
    if stat is None:
        return None
    if stat["share_under_60s"] >= 0.95:
        return stat["p10"] if stat["p10"] > 0 else stat["p25"]
    return stat["p25"]

def edge_weight_for_internal(src, tgt):
    stat = EDGE_STATS.get((src, tgt))
    if stat is None:
        return None
    val = edge_duration_seconds(src, tgt)
    if val is None:
        return None
    if stat["share_under_60s"] >= 0.95 and val <= 0.01:
        return 0.001
    return max(val, 0.001)

def fallback_block_total(k):
    prof = BLOCK_SPAN.get(k)
    if prof:
        return max(prof["p25"], 0.001)
    return max(BLOCK_SPAN[max(BLOCK_SPAN)]["p25"], GLOBAL_MEDIAN)

def estimate_single_from_left(left_dt, left_act, cur_act):
    d = edge_duration_seconds(left_act, cur_act)
    if d is None:
        d = GLOBAL_MEDIAN
    return left_dt + pd.to_timedelta(d, unit="s")

def estimate_single_from_right(right_dt, cur_act, right_act):
    d = edge_duration_seconds(cur_act, right_act)
    if d is None:
        d = GLOBAL_MEDIAN
    return right_dt - pd.to_timedelta(d, unit="s")

def midpoint(a, b):
    return a + (b - a) / 2

def tiny_delta(precision):
    if precision >= 3:
        return pd.to_timedelta(1, unit="ms")
    elif precision == 2:
        return pd.to_timedelta(10, unit="ms")
    elif precision == 1:
        return pd.to_timedelta(100, unit="ms")
    else:
        return pd.to_timedelta(1, unit="s")

def estimate_block_offsets(activities):
    n = len(activities)
    if n == 1:
        return [0.0]
    internal = []
    for i in range(n - 1):
        w = edge_weight_for_internal(activities[i], activities[i + 1])
        if w is None:
            # use varied fallback, not uniform
            w = fallback_block_total(n) / max(n, 1) * (1.0 + 0.15 * i)
        internal.append(max(w, 0.001))
    cum = [0.0]
    s = 0.0
    for w in internal:
        s += w
        cum.append(s)
    return cum

def distribute_within_anchors(left_dt, right_dt, activities, precision):
    n = len(activities)
    if n == 1:
        return [midpoint(left_dt, right_dt)]
    offsets = estimate_block_offsets(activities)
    total = offsets[-1]
    avail = (right_dt - left_dt).total_seconds()
    if avail < 0:
        avail = 0.0
    if total <= 0:
        total = fallback_block_total(n)
        offsets = [0.0]
        step = total / max(n - 1, 1)
        for i in range(1, n):
            offsets.append(offsets[-1] + step * (1 + 0.1 * i))
        total = offsets[-1]
    if total > avail:
        if n == 1:
            return [midpoint(left_dt, right_dt)]
        scale = 0.0 if total == 0 else avail / total
        scaled = [left_dt + pd.to_timedelta(o * scale, unit="s") for o in offsets]
        # if collapse happened too much, center interior rows inside interval
        td = tiny_delta(precision)
        for i in range(1, len(scaled)):
            if scaled[i] < scaled[i - 1]:
                scaled[i] = scaled[i - 1]
        return scaled
    else:
        return [left_dt + pd.to_timedelta(o, unit="s") for o in offsets]

def clamp_to_bounds(dt, lo, hi):
    if lo is not None and dt < lo:
        dt = lo
    if hi is not None and dt > hi:
        dt = hi
    return dt

def preserve_same_date(dt, observed_date, lo=None, hi=None):
    start_day = pd.Timestamp(observed_date)
    end_day = start_day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if dt.date() != observed_date:
        if dt < start_day:
            dt = start_day
        elif dt > end_day:
            dt = end_day
        else:
            dt = pd.Timestamp(year=observed_date.year, month=observed_date.month, day=observed_date.day,
                              hour=dt.hour, minute=dt.minute, second=dt.second, microsecond=dt.microsecond)
    lo2 = max(lo, start_day) if lo is not None else start_day
    hi2 = min(hi, end_day) if hi is not None else end_day
    if lo2 > hi2:
        return midpoint(lo2, lo2)
    return clamp_to_bounds(dt, lo2, hi2)

# ============================================================
# MAIN REPAIR
# ============================================================

def repair_case(case_df, precision):
    repaired = case_df["Timestamp"].astype(str).copy()
    parsed_original = case_df["Timestamp"].apply(parse_ts_safe)

    clean_mask = case_df["error_category"].eq("CLEAN")
    clean_parsed = parsed_original.where(clean_mask)

    case_start = clean_parsed.dropna().min()
    case_end = clean_parsed.dropna().max()
    if pd.isna(case_start):
        # no clean anchors; synthesize conservative local bounds from any parseable rows
        any_parsed = parsed_original.dropna()
        if len(any_parsed) > 0:
            case_start = any_parsed.min()
            case_end = any_parsed.max()
        else:
            base = pd.Timestamp("2011-01-01 12:00:00")
            case_start = base
            case_end = base + pd.Timedelta(seconds=1)

    idxs = list(case_df.index)
    corrupt_mask = ~clean_mask
    i = 0
    while i < len(case_df):
        if not corrupt_mask.iloc[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(case_df) and corrupt_mask.iloc[j + 1]:
            j += 1

        block_idx = idxs[i:j + 1]
        acts = case_df.loc[block_idx, "Activity"].tolist()

        left_idx = idxs[i - 1] if i - 1 >= 0 else None
        right_idx = idxs[j + 1] if j + 1 < len(case_df) else None

        left_dt = parse_ts_safe(repaired.loc[left_idx]) if left_idx is not None else pd.NaT
        right_dt = parse_ts_safe(repaired.loc[right_idx]) if right_idx is not None else pd.NaT
        left_act = case_df.loc[left_idx, "Activity"] if left_idx is not None else None
        right_act = case_df.loc[right_idx, "Activity"] if right_idx is not None else None

        left_dt = None if pd.isna(left_dt) else left_dt
        right_dt = None if pd.isna(right_dt) else right_dt

        if left_dt is not None and right_dt is not None and left_dt > right_dt:
            tmp = left_dt
            left_dt = right_dt
            right_dt = tmp

        assigned = []

        if left_dt is not None and right_dt is not None:
            assigned = distribute_within_anchors(left_dt, right_dt, acts, precision)
        elif left_dt is not None:
            offsets = estimate_block_offsets(acts)
            assigned = [left_dt + pd.to_timedelta(o, unit="s") for o in offsets]
        elif right_dt is not None:
            rev_acts = acts[::-1]
            rev_offsets = estimate_block_offsets(rev_acts)
            rev_assigned = [right_dt - pd.to_timedelta(o, unit="s") for o in rev_offsets]
            assigned = rev_assigned[::-1]
        else:
            total = fallback_block_total(len(acts))
            offsets = estimate_block_offsets(acts)
            scale = total / max(offsets[-1], 1e-9) if len(offsets) > 1 and offsets[-1] > 0 else 1.0
            base = case_start
            assigned = [base + pd.to_timedelta(o * scale, unit="s") for o in offsets]

        for loc_pos, row_idx in enumerate(block_idx):
            orig = case_df.at[row_idx, "Timestamp"]
            est = assigned[loc_pos]
            row_act = case_df.at[row_idx, "Activity"]

            if left_dt is not None and right_dt is not None:
                est = clamp_to_bounds(est, left_dt, right_dt)
            est = clamp_to_bounds(est, case_start, case_end)

            if is_date_only_or_midnight_loss(orig):
                observed_date = parse_ts_safe(orig)
                if pd.isna(observed_date):
                    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(orig).strip())
                    if m:
                        observed_date = pd.to_datetime(m.group(1))
                if not pd.isna(observed_date):
                    est = preserve_same_date(est, observed_date.date(), left_dt, right_dt)
                    est = clamp_to_bounds(est, case_start, case_end)
                    est = preserve_same_date(est, observed_date.date(), case_start, case_end)

            # single-row special midpoint if estimate hits/exceeds right anchor
            if len(block_idx) == 1 and left_dt is not None and right_dt is not None:
                pred = estimate_single_from_left(left_dt, left_act, row_act)
                if pred > right_dt:
                    est = midpoint(left_dt, right_dt)
                else:
                    est = pred
                    est = clamp_to_bounds(est, left_dt, right_dt)
                    if is_date_only_or_midnight_loss(orig):
                        observed_date = parse_ts_safe(orig)
                        if not pd.isna(observed_date):
                            est = preserve_same_date(est, observed_date.date(), left_dt, right_dt)

            rep = format_dt(est, precision)
            if rep == str(orig):
                dt_rep = parse_ts_safe(rep)
                bump = tiny_delta(precision)
                candidates = [dt_rep + bump, dt_rep - bump]
                chosen = None
                for cand in candidates:
                    cand2 = cand
                    if left_dt is not None and right_dt is not None:
                        cand2 = clamp_to_bounds(cand2, left_dt, right_dt)
                    cand2 = clamp_to_bounds(cand2, case_start, case_end)
                    if is_date_only_or_midnight_loss(orig):
                        observed_date = parse_ts_safe(orig)
                        if not pd.isna(observed_date):
                            cand2 = preserve_same_date(cand2, observed_date.date(), left_dt, right_dt)
                            cand2 = clamp_to_bounds(cand2, case_start, case_end)
                            cand2 = preserve_same_date(cand2, observed_date.date(), case_start, case_end)
                    rep2 = format_dt(cand2, precision)
                    if rep2 != str(orig):
                        chosen = rep2
                        break
                if chosen is None:
                    chosen = rep
                rep = chosen

            repaired.loc[row_idx] = rep

        i = j + 1

    return repaired

def self_check_and_fix(df, repaired_col, precision):
    repaired = repaired_col.copy()
    blank_filled = 0
    date_only_filled = 0

    for case_id, grp in df.groupby("Case", sort=False):
        idxs = list(grp.index)
        parsed_rep = repaired.loc[idxs].apply(parse_ts_safe)
        clean_mask = grp["error_category"].eq("CLEAN")
        clean_parsed = grp["Timestamp"].apply(parse_ts_safe).where(clean_mask)
        case_start = clean_parsed.dropna().min()
        case_end = clean_parsed.dropna().max()
        if pd.isna(case_start):
            any_parsed = parsed_rep.dropna()
            case_start = any_parsed.min() if len(any_parsed) else pd.Timestamp("2011-01-01 12:00:00")
            case_end = any_parsed.max() if len(any_parsed) else case_start + pd.Timedelta(seconds=1)

        for pos, idx in enumerate(idxs):
            if grp.at[idx, "error_category"] == "CLEAN":
                repaired.at[idx] = grp.at[idx, "Timestamp"]
                continue

            orig = grp.at[idx, "Timestamp"]
            cur = repaired.at[idx]
            cur_dt = parse_ts_safe(cur)

            if is_blank(orig) and not is_blank(cur):
                blank_filled += 1
            if is_date_only_or_midnight_loss(orig) and not is_blank(cur):
                date_only_filled += 1

            if pd.isna(cur_dt):
                # fallback from nearest anchors
                left_idx = idxs[pos - 1] if pos - 1 >= 0 else None
                right_idx = idxs[pos + 1] if pos + 1 < len(idxs) else None
                left_dt = parse_ts_safe(repaired.at[left_idx]) if left_idx is not None else pd.NaT
                right_dt = parse_ts_safe(repaired.at[right_idx]) if right_idx is not None else pd.NaT
                if not pd.isna(left_dt) and not pd.isna(right_dt):
                    cur_dt = midpoint(left_dt, right_dt)
                elif not pd.isna(left_dt):
                    cur_dt = left_dt + tiny_delta(precision)
                elif not pd.isna(right_dt):
                    cur_dt = right_dt - tiny_delta(precision)
                else:
                    cur_dt = case_start
                repaired.at[idx] = format_dt(cur_dt, precision)

            # anchor bounds
            left_anchor = None
            for p in range(pos - 1, -1, -1):
                dt = parse_ts_safe(repaired.at[idxs[p]])
                if not pd.isna(dt):
                    left_anchor = dt
                    break
            right_anchor = None
            for p in range(pos + 1, len(idxs)):
                dt = parse_ts_safe(repaired.at[idxs[p]])
                if not pd.isna(dt):
                    right_anchor = dt
                    break

            cur_dt = parse_ts_safe(repaired.at[idx])
            lo = case_start
            hi = case_end
            if left_anchor is not None and right_anchor is not None:
                lo = max(lo, left_anchor)
                hi = min(hi, right_anchor)
            if lo > hi:
                hi = lo
            cur_dt = clamp_to_bounds(cur_dt, lo, hi)

            if is_date_only_or_midnight_loss(orig):
                obsdt = parse_ts_safe(orig)
                if pd.isna(obsdt):
                    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(orig).strip())
                    if m:
                        obsdt = pd.to_datetime(m.group(1))
                if not pd.isna(obsdt):
                    cur_dt = preserve_same_date(cur_dt, obsdt.date(), lo, hi)

            rep = format_dt(cur_dt, precision)
            if rep == str(orig):
                cur_dt2 = cur_dt + tiny_delta(precision)
                cur_dt2 = clamp_to_bounds(cur_dt2, lo, hi)
                if is_date_only_or_midnight_loss(orig):
                    obsdt = parse_ts_safe(orig)
                    if not pd.isna(obsdt):
                        cur_dt2 = preserve_same_date(cur_dt2, obsdt.date(), lo, hi)
                rep2 = format_dt(cur_dt2, precision)
                if rep2 == str(orig):
                    cur_dt3 = cur_dt - tiny_delta(precision)
                    cur_dt3 = clamp_to_bounds(cur_dt3, lo, hi)
                    if is_date_only_or_midnight_loss(orig):
                        obsdt = parse_ts_safe(orig)
                        if not pd.isna(obsdt):
                            cur_dt3 = preserve_same_date(cur_dt3, obsdt.date(), lo, hi)
                    rep3 = format_dt(cur_dt3, precision)
                    if rep3 != str(orig):
                        rep = rep3
                    else:
                        rep = rep2
                else:
                    rep = rep2

            repaired.at[idx] = rep

    still_equal = int(((df["error_category"] != "CLEAN") & (repaired.astype(str) == df["Timestamp"].astype(str))).sum())
    print(f"blank_rows_filled={blank_filled}, date_only_rows_timed={date_only_filled}, rows_still_equal_to_input={still_equal}")
    return repaired

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    original_columns = list(df.columns)
    required = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    precision = detect_precision(df["Timestamp"])

    repaired_timestamp = df["Timestamp"].astype(str).copy()

    for case_id, case_df in df.groupby("Case", sort=False):
        repaired_case = repair_case(case_df, precision)
        repaired_timestamp.loc[case_df.index] = repaired_case

    repaired_timestamp = self_check_and_fix(df, repaired_timestamp, precision)

    out = df.copy()
    out["Timestamp"] = repaired_timestamp

    # final safety: preserve all non-timestamp columns byte-identically and row order
    out = out[original_columns]

    try:
        out.to_csv(args.output, index=False)
    except Exception:
        # guaranteed exit 0 requirement
        try:
            out.to_csv(args.output, index=False, encoding="utf-8")
        except Exception:
            pass

if __name__ == "__main__":
    main()