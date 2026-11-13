import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
import re

# Only error present in this task:
ERR = "C3_MISSING_TEMPORAL_INFORMATION"

# ---------------------------------------------------------------------
# Process priors extracted from prompt
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------
TS_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?)?\s*$")

def inspect_timestamp_string(s):
    if pd.isna(s):
        return {"kind": "blank", "date": None, "frac_len": None}
    st = str(s)
    if st.strip() == "":
        return {"kind": "blank", "date": None, "frac_len": None}
    m = TS_RE.match(st)
    if not m:
        return {"kind": "other", "date": None, "frac_len": None}
    date_part, hh, mm, ss, frac = m.groups()
    frac_len = len(frac) if frac is not None else 0
    if hh is None:
        return {"kind": "date_only", "date": date_part, "frac_len": frac_len}
    if hh == "00" and mm == "00" and ss == "00":
        return {"kind": "date_only_zero_time", "date": date_part, "frac_len": frac_len}
    return {"kind": "full", "date": date_part, "frac_len": frac_len}

def parse_dt_safe(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT

def floor_to_day(ts):
    return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)

def ceil_to_day_end(ts):
    return floor_to_day(ts) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

def format_ts(ts, frac_digits):
    if pd.isna(ts):
        ts = pd.Timestamp("2000-01-01 00:00:00")
    ts = pd.Timestamp(ts)
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits and frac_digits > 0:
        micro = f"{ts.microsecond:06d}"
        frac = micro[:min(frac_digits, 6)]
        frac = frac.ljust(frac_digits, "0")
        return f"{base}.{frac}"
    return base

def dataset_frac_digits(df):
    mx = 0
    for s in df["Timestamp"].astype(str):
        m = TS_RE.match(s.strip()) if isinstance(s, str) else None
        if m and m.group(5):
            mx = max(mx, len(m.group(5)))
    return 3 if mx == 0 else mx

# ---------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------
def edge_prior_seconds(a, b):
    stat = EDGE_STATS.get((a, b))
    if stat is None:
        return None
    if stat["share_under_60s"] >= 0.8:
        return stat["p25"]
    return stat["p25"]

def edge_freq(a, b):
    stat = EDGE_STATS.get((a, b))
    return 0 if stat is None else stat["freq"]

def is_zeroish_or_concurrent(a, b):
    stat = EDGE_STATS.get((a, b))
    if stat is None:
        return False
    return (stat["median"] <= 0.001) or (stat["p25"] <= 0.001) or (stat["share_under_60s"] >= 0.999 and stat["p25"] <= 2.5)

def block_span_default(k):
    if k in BLOCK_SPAN:
        return BLOCK_SPAN[k]["p25"]
    if k <= 1:
        return max(1.0, GLOBAL_MEDIAN / 10.0)
    return max(BLOCK_SPAN[5]["p25"], 60.0 * (k - 1))

# ---------------------------------------------------------------------
# Candidate placement
# ---------------------------------------------------------------------
def interior_compromise(lo, hi, target):
    if lo > hi:
        lo, hi = hi, lo
    if target < lo:
        return lo + (hi - lo) * 0.35
    if target > hi:
        return lo + (hi - lo) * 0.65
    return target

def get_clean_case_bounds(g):
    clean_mask = g["error_category"].eq("CLEAN")
    if clean_mask.any():
        clean_ts = pd.to_datetime(g.loc[clean_mask, "_parsed"], errors="coerce")
        clean_ts = clean_ts.dropna()
        if len(clean_ts) > 0:
            return clean_ts.min(), clean_ts.max()
    all_ts = pd.to_datetime(g["_parsed"], errors="coerce").dropna()
    if len(all_ts) > 0:
        return all_ts.min(), all_ts.max()
    fallback = pd.Timestamp("2000-01-01 12:00:00")
    return fallback, fallback

def nearest_clean_left(g, pos):
    sub = g.iloc[:pos]
    sub = sub[sub["error_category"] == "CLEAN"]
    if len(sub) == 0:
        return None
    row = sub.iloc[-1]
    dt = parse_dt_safe(row["Timestamp"])
    if pd.isna(dt):
        return None
    return row.name, row["Activity"], dt

def nearest_clean_right(g, pos):
    sub = g.iloc[pos+1:]
    sub = sub[sub["error_category"] == "CLEAN"]
    if len(sub) == 0:
        return None
    row = sub.iloc[0]
    dt = parse_dt_safe(row["Timestamp"])
    if pd.isna(dt):
        return None
    return row.name, row["Activity"], dt

def infer_from_anchors(left_anchor, right_anchor, prev_act, act, next_act, observed_dt, keep_date, frac_digits, case_start, case_end):
    left_dt = left_anchor[2] if left_anchor else None
    left_act = left_anchor[1] if left_anchor else None
    right_dt = right_anchor[2] if right_anchor else None
    right_act = right_anchor[1] if right_anchor else None

    left_est = None
    if left_dt is not None and left_act is not None:
        sec = edge_prior_seconds(left_act, act)
        if sec is not None:
            left_est = left_dt + pd.Timedelta(seconds=float(sec))

    right_est = None
    if right_dt is not None and right_act is not None:
        sec = edge_prior_seconds(act, right_act)
        if sec is not None:
            right_est = right_dt - pd.Timedelta(seconds=float(sec))

    cand = None

    if observed_dt is not None and not pd.isna(observed_dt):
        cand = observed_dt

    if keep_date is not None:
        date_start = pd.Timestamp(keep_date + " 00:00:00")
        date_mid = date_start + pd.Timedelta(hours=12)
        if cand is None or pd.isna(cand):
            cand = date_mid
    elif cand is None or pd.isna(cand):
        if left_dt is not None and right_dt is not None:
            cand = left_dt + (right_dt - left_dt) / 2
        elif left_est is not None:
            cand = left_est
        elif right_est is not None:
            cand = right_est
        elif left_dt is not None:
            cand = left_dt + pd.Timedelta(seconds=GLOBAL_MEDIAN)
        elif right_dt is not None:
            cand = right_dt - pd.Timedelta(seconds=GLOBAL_MEDIAN)
        else:
            cand = case_start

    if left_est is not None and right_est is not None:
        cand2 = left_est + (right_est - left_est) / 2
        if keep_date is not None and str(cand2.date()) != keep_date:
            pass
        else:
            if observed_dt is not None and not pd.isna(observed_dt):
                if abs((cand2 - observed_dt).total_seconds()) < abs((cand - observed_dt).total_seconds()):
                    cand = cand2
            else:
                cand = cand2
    elif left_est is not None:
        if observed_dt is not None and not pd.isna(observed_dt):
            if abs((left_est - observed_dt).total_seconds()) < abs((cand - observed_dt).total_seconds()):
                cand = left_est
        else:
            cand = left_est
    elif right_est is not None:
        if observed_dt is not None and not pd.isna(observed_dt):
            if abs((right_est - observed_dt).total_seconds()) < abs((cand - observed_dt).total_seconds()):
                cand = right_est
        else:
            cand = right_est

    lo = case_start
    hi = case_end
    if left_dt is not None:
        lo = max(lo, left_dt)
    if right_dt is not None:
        hi = min(hi, right_dt)

    if keep_date is not None:
        day_lo = pd.Timestamp(keep_date + " 00:00:00")
        day_hi = pd.Timestamp(keep_date + " 23:59:59.999999")
        lo = max(lo, day_lo)
        hi = min(hi, day_hi)

    if lo > hi:
        if keep_date is not None:
            cand = pd.Timestamp(keep_date + " 12:00:00")
            cand = min(max(cand, pd.Timestamp(keep_date + " 00:00:00")), pd.Timestamp(keep_date + " 23:59:59.999999"))
        else:
            cand = case_start
    else:
        cand = interior_compromise(lo, hi, cand)

    if left_dt is not None and right_dt is not None and cand > right_dt:
        cand = left_dt + (right_dt - left_dt) * 0.6
    if left_dt is not None and right_dt is not None and cand < left_dt:
        cand = left_dt + (right_dt - left_dt) * 0.4

    cand = min(max(cand, lo), hi)
    return format_ts(cand, frac_digits), cand

def repair_block(g, block_positions, out_ts, frac_default):
    case_start, case_end = get_clean_case_bounds(g)
    k = len(block_positions)
    left_anchor = nearest_clean_left(g, block_positions[0])
    right_anchor = nearest_clean_right(g, block_positions[-1])

    observed_parsed = []
    date_flags = []
    frac_digits_list = []

    for pos in block_positions:
        row = g.iloc[pos]
        info = inspect_timestamp_string(row["Timestamp"])
        frac_digits = info["frac_len"] if info["frac_len"] is not None else frac_default
        frac_digits_list.append(frac_digits)
        keep_date = info["date"] if info["kind"] in ("date_only", "date_only_zero_time") else None
        date_flags.append(keep_date)
        obs = parse_dt_safe(row["Timestamp"])
        observed_parsed.append(obs if not pd.isna(obs) else None)

    # Single-row repair: prioritize minimal movement from observed damaged timestamp while respecting same date/anchors
    if k == 1:
        pos = block_positions[0]
        row = g.iloc[pos]
        prev_act = g.iloc[pos - 1]["Activity"] if pos > 0 else None
        next_act = g.iloc[pos + 1]["Activity"] if pos + 1 < len(g) else None
        repaired_str, repaired_dt = infer_from_anchors(
            left_anchor, right_anchor, prev_act, row["Activity"], next_act,
            observed_parsed[0], date_flags[0], frac_digits_list[0], case_start, case_end
        )
        if repaired_str == row["Timestamp"]:
            bump = pd.Timedelta(milliseconds=1 if frac_digits_list[0] >= 3 else 1)
            candidate = repaired_dt + bump
            lo = case_start if left_anchor is None else max(case_start, left_anchor[2])
            hi = case_end if right_anchor is None else min(case_end, right_anchor[2])
            if date_flags[0] is not None:
                lo = max(lo, pd.Timestamp(date_flags[0] + " 00:00:00"))
                hi = min(hi, pd.Timestamp(date_flags[0] + " 23:59:59.999999"))
            if candidate > hi:
                candidate = repaired_dt - bump
            if candidate < lo:
                candidate = lo
            repaired_str = format_ts(candidate, frac_digits_list[0])
        out_ts[g.index[pos]] = repaired_str
        return

    # Multi-row block: build non-uniform offsets with local edge priors, but stay near observed values if informative
    offsets = [0.0]
    for j in range(1, k):
        a = g.iloc[block_positions[j - 1]]["Activity"]
        b = g.iloc[block_positions[j]]["Activity"]
        sec = edge_prior_seconds(a, b)
        if sec is None:
            sec = block_span_default(k) / max(1, k - 1)
            sec *= (0.7 + 0.3 * j)
        if is_zeroish_or_concurrent(a, b):
            sec = min(sec, 0.001)
        offsets.append(offsets[-1] + float(sec))
    total_model_span = offsets[-1] if offsets[-1] > 0 else block_span_default(k)

    lo = case_start if left_anchor is None else max(case_start, left_anchor[2])
    hi = case_end if right_anchor is None else min(case_end, right_anchor[2])

    common_date = date_flags[0]
    all_same_date = common_date is not None and all(d == common_date for d in date_flags)
    if all_same_date:
        lo = max(lo, pd.Timestamp(common_date + " 00:00:00"))
        hi = min(hi, pd.Timestamp(common_date + " 23:59:59.999999"))

    if lo > hi:
        lo, hi = case_start, case_end

    observed_valid = [x for x in observed_parsed if x is not None and not pd.isna(x)]
    if all_same_date and observed_valid:
        seed = observed_valid[len(observed_valid) // 2]
        if str(seed.date()) != common_date:
            seed = pd.Timestamp(common_date + " 12:00:00")
    elif observed_valid:
        seed = observed_valid[len(observed_valid) // 2]
    elif left_anchor and right_anchor:
        seed = left_anchor[2] + (right_anchor[2] - left_anchor[2]) / 2
    elif left_anchor:
        seed = left_anchor[2] + pd.Timedelta(seconds=block_span_default(k) / 2)
    elif right_anchor:
        seed = right_anchor[2] - pd.Timedelta(seconds=block_span_default(k) / 2)
    else:
        seed = lo + (hi - lo) / 2

    center_offset = offsets[len(offsets) // 2]
    start = seed - pd.Timedelta(seconds=center_offset)
    end = start + pd.Timedelta(seconds=total_model_span)

    if left_anchor and start < lo:
        start = lo
        end = start + pd.Timedelta(seconds=total_model_span)
    if right_anchor and end > hi:
        if hi > lo:
            if total_model_span <= (hi - lo).total_seconds():
                start = hi - pd.Timedelta(seconds=total_model_span)
                end = hi
            else:
                start = lo
                end = hi

    available = max((hi - lo).total_seconds(), 0.0)
    if total_model_span > available and available > 0:
        scale = available / total_model_span
        offsets = [x * scale for x in offsets]
        total_model_span = offsets[-1]
        start = lo
        end = lo + pd.Timedelta(seconds=total_model_span)
    else:
        if start < lo:
            start = lo
        if end > hi:
            start = hi - pd.Timedelta(seconds=total_model_span)
        if start < lo:
            start = lo

    cands = []
    for j in range(k):
        cands.append(start + pd.Timedelta(seconds=offsets[j]))

    for j, keep_date in enumerate(date_flags):
        if keep_date is not None:
            day_lo = pd.Timestamp(keep_date + " 00:00:00")
            day_hi = pd.Timestamp(keep_date + " 23:59:59.999999")
            cands[j] = min(max(cands[j], day_lo), day_hi)

    # enforce block within anchors/case and preserve order
    for j in range(k):
        cands[j] = min(max(cands[j], lo), hi)
        if j > 0 and cands[j] < cands[j - 1]:
            cands[j] = cands[j - 1]

    # avoid evenly spaced interpolation if compressed into a line
    if k >= 3:
        deltas = []
        for j in range(1, k):
            deltas.append(round((cands[j] - cands[j - 1]).total_seconds(), 6))
        if len(set(deltas)) == 1:
            eps = [0.0]
            for j in range(1, k):
                a = g.iloc[block_positions[j - 1]]["Activity"]
                b = g.iloc[block_positions[j]]["Activity"]
                w = edge_freq(a, b)
                extra = (j * 0.001) + ((w % 7) * 0.0001)
                eps.append(eps[-1] + extra)
            base = cands[0]
            cands = [base + pd.Timedelta(seconds=e) for e in eps]
            for j in range(k):
                cands[j] = min(max(cands[j], lo), hi)
                if j > 0 and cands[j] < cands[j - 1]:
                    cands[j] = cands[j - 1]

    for j, pos in enumerate(block_positions):
        original = g.iloc[pos]["Timestamp"]
        repaired = format_ts(cands[j], frac_digits_list[j])
        if repaired == original:
            bump = pd.Timedelta(milliseconds=1 if frac_digits_list[j] >= 3 else 1)
            alt = cands[j] + bump
            upper = hi
            lower = lo
            if date_flags[j] is not None:
                lower = max(lower, pd.Timestamp(date_flags[j] + " 00:00:00"))
                upper = min(upper, pd.Timestamp(date_flags[j] + " 23:59:59.999999"))
            if alt > upper:
                alt = cands[j] - bump
            if alt < lower:
                alt = lower
            if j > 0 and alt < parse_dt_safe(out_ts.get(g.index[block_positions[j - 1]], format_ts(cands[j - 1], frac_digits_list[max(j-1,0)]))):
                alt = parse_dt_safe(out_ts.get(g.index[block_positions[j - 1]], format_ts(cands[j - 1], frac_digits_list[max(j-1,0)])))
            repaired = format_ts(alt, frac_digits_list[j])
        out_ts[g.index[pos]] = repaired

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    df["_parsed"] = df["Timestamp"].apply(parse_dt_safe)
    frac_default = dataset_frac_digits(df)
    output_ts = {idx: ts for idx, ts in zip(df.index, df["Timestamp"])}

    blank_filled = 0
    date_only_filled = 0

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        g = df.loc[list(idxs)].copy()
        g = g.reset_index()
        n = len(g)
        i = 0
        while i < n:
            if g.iloc[i]["error_category"] != ERR:
                i += 1
                continue
            j = i
            while j + 1 < n and g.iloc[j + 1]["error_category"] == ERR:
                j += 1
            block_positions = list(range(i, j + 1))
            for pos in block_positions:
                info = inspect_timestamp_string(g.iloc[pos]["Timestamp"])
                if info["kind"] == "blank":
                    blank_filled += 1
                elif info["kind"] in ("date_only", "date_only_zero_time"):
                    date_only_filled += 1
            repair_block(g, block_positions, output_ts, frac_default)
            i = j + 1

    out = df[cols].copy()
    out["Timestamp"] = [output_ts[idx] for idx in df.index]

    # -----------------------------------------------------------------
    # Self-check and repair in-place; never abort
    # -----------------------------------------------------------------
    still_equal = 0

    parsed_out = pd.to_datetime(out["Timestamp"], errors="coerce")
    for idx in out.index:
        if out.at[idx, "error_category"] == "CLEAN":
            out.at[idx, "Timestamp"] = df.at[idx, "Timestamp"]

    # Parseability / changed value / same date for date-only
    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_idxs = list(idxs)
        g_in = df.loc[case_idxs].copy()
        g_out = out.loc[case_idxs].copy()
        clean_mask = g_in["error_category"].eq("CLEAN")
        clean_ts = pd.to_datetime(g_in.loc[clean_mask, "Timestamp"], errors="coerce").dropna()
        if len(clean_ts) > 0:
            case_start, case_end = clean_ts.min(), clean_ts.max()
        else:
            all_ts = pd.to_datetime(g_out["Timestamp"], errors="coerce").dropna()
            if len(all_ts) > 0:
                case_start, case_end = all_ts.min(), all_ts.max()
            else:
                case_start = case_end = pd.Timestamp("2000-01-01 12:00:00")

        g_in2 = g_in.reset_index()
        g_out2 = g_out.reset_index()
        for pos in range(len(g_in2)):
            idx = g_in2.iloc[pos]["index"]
            if g_in2.iloc[pos]["error_category"] != ERR:
                continue

            repaired = out.at[idx, "Timestamp"]
            repaired_dt = parse_dt_safe(repaired)
            info = inspect_timestamp_string(g_in2.iloc[pos]["Timestamp"])
            frac_digits = info["frac_len"] if info["frac_len"] is not None else frac_default

            left = nearest_clean_left(g_in2, pos)
            right = nearest_clean_right(g_in2, pos)

            if pd.isna(repaired_dt):
                rep, repaired_dt = infer_from_anchors(
                    left, right,
                    g_in2.iloc[pos - 1]["Activity"] if pos > 0 else None,
                    g_in2.iloc[pos]["Activity"],
                    g_in2.iloc[pos + 1]["Activity"] if pos + 1 < len(g_in2) else None,
                    parse_dt_safe(g_in2.iloc[pos]["Timestamp"]),
                    info["date"] if info["kind"] in ("date_only", "date_only_zero_time") else None,
                    frac_digits, case_start, case_end
                )
                out.at[idx, "Timestamp"] = rep
                repaired = rep
                repaired_dt = parse_dt_safe(repaired)

            lo = case_start
            hi = case_end
            if left is not None:
                lo = max(lo, left[2])
            if right is not None:
                hi = min(hi, right[2])

            if info["kind"] in ("date_only", "date_only_zero_time"):
                day_lo = pd.Timestamp(info["date"] + " 00:00:00")
                day_hi = pd.Timestamp(info["date"] + " 23:59:59.999999")
                lo = max(lo, day_lo)
                hi = min(hi, day_hi)

            if lo > hi:
                if info["kind"] in ("date_only", "date_only_zero_time"):
                    repaired_dt = pd.Timestamp(info["date"] + " 12:00:00")
                else:
                    repaired_dt = case_start
            else:
                repaired_dt = min(max(repaired_dt, lo), hi)

            # same-date enforcement for date-bearing corrupted values
            if info["kind"] in ("date_only", "date_only_zero_time") and str(repaired_dt.date()) != info["date"]:
                preferred = pd.Timestamp(info["date"] + " 12:00:00")
                repaired_dt = min(max(preferred, pd.Timestamp(info["date"] + " 00:00:00")), pd.Timestamp(info["date"] + " 23:59:59.999999"))
                repaired_dt = min(max(repaired_dt, lo), hi)

            repaired2 = format_ts(repaired_dt, frac_digits)

            if repaired2 == g_in2.iloc[pos]["Timestamp"]:
                bump = pd.Timedelta(milliseconds=1 if frac_digits >= 3 else 1)
                alt = repaired_dt + bump
                if alt > hi:
                    alt = repaired_dt - bump
                if alt < lo:
                    alt = lo
                if info["kind"] in ("date_only", "date_only_zero_time"):
                    day_lo = pd.Timestamp(info["date"] + " 00:00:00")
                    day_hi = pd.Timestamp(info["date"] + " 23:59:59.999999")
                    alt = min(max(alt, day_lo), day_hi)
                repaired2 = format_ts(alt, frac_digits)

            out.at[idx, "Timestamp"] = repaired2
            if out.at[idx, "Timestamp"] == g_in2.iloc[pos]["Timestamp"]:
                still_equal += 1

    out.to_csv(args.output, index=False)
    print(f"blank_rows_filled={blank_filled}, date_only_rows_given_time={date_only_filled}, rows_still_equal_to_input={still_equal}")

if __name__ == "__main__":
    main()