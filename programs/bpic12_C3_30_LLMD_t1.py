import argparse
import pandas as pd
import numpy as np
import re
from datetime import timedelta

# ----------------------------
# Process priors from prompt
# ----------------------------

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

BLOCK_SPAN = {
    2: {"p10": 0.063, "p25": 1.305, "median": 49.725},
    3: {"p10": 1.366, "p25": 36.364, "median": 995.679},
    4: {"p10": 5.7904, "p25": 217.364, "median": 6361.145},
    5: {"p10": 50.3772, "p25": 866.1295, "median": 21853.089},
}
GLOBAL_MEDIAN = 49.725

FREQ_VARIANTS = [
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_DECLINED-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_DECLINED-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
]

# ----------------------------
# Helpers
# ----------------------------

def parse_any(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.NaT

def is_blank(ts):
    return pd.isna(ts) or str(ts).strip() == ""

def detect_date_only(ts):
    if is_blank(ts):
        return False
    s = str(ts).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}\s+00:00:00(?:\.0+)?", s):
        return True
    return False

def frac_width(ts):
    s = "" if pd.isna(ts) else str(ts)
    m = re.search(r"\.(\d+)$", s)
    return len(m.group(1)) if m else 0

def infer_global_frac_width(series):
    widths = [frac_width(v) for v in series if not is_blank(v)]
    return max(widths) if widths else 3

def format_ts(dt, width):
    if pd.isna(dt):
        return ""
    dt = pd.Timestamp(dt)
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    if width > 0:
        frac = f"{dt.microsecond:06d}"[:width]
        return f"{base}.{frac}"
    return base

def get_edge_profile(a, b):
    return DURATION_PROFILES.get((a, b))

def edge_gap_seconds(a, b):
    prof = get_edge_profile(a, b)
    if prof:
        if prof["share_under_60s"] >= 0.8:
            return max(prof["p10"], 0.001)
        return max(prof["p25"], 0.001)
    return None

def edge_weight_seconds(a, b):
    prof = get_edge_profile(a, b)
    if prof:
        x = prof["p25"]
        if x <= 0:
            x = 0.001
        return x
    return None

def sequence_in_variants(seq):
    for var in FREQ_VARIANTS:
        n, m = len(var), len(seq)
        for i in range(n - m + 1):
            if var[i:i+m] == seq:
                return True
    return False

def candidate_order_supported(activities):
    ok_pairs = 0
    total_pairs = max(len(activities) - 1, 0)
    for i in range(total_pairs):
        if get_edge_profile(activities[i], activities[i+1]) is not None:
            ok_pairs += 1
    return (total_pairs == 0) or (ok_pairs >= max(1, total_pairs - 1)) or sequence_in_variants(activities)

def smallest_plausible_gap(width):
    return pd.Timedelta(milliseconds=1 if width >= 3 else 1, unit="ms") if width >= 3 else pd.Timedelta(seconds=1)

def same_day_bounds(day):
    start = pd.Timestamp(day).normalize()
    end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return start, end

def clamp(ts, lo, hi):
    if lo is not None and ts < lo:
        ts = lo
    if hi is not None and ts > hi:
        ts = hi
    return ts

def build_internal_weights(acts, left_act=None, right_act=None):
    k = len(acts)
    weights = []
    labels = []
    if left_act is not None and k > 0:
        w = edge_weight_seconds(left_act, acts[0])
        if w is not None:
            weights.append(max(w, 0.001))
            labels.append(("L", 0))
    for i in range(k - 1):
        w = edge_weight_seconds(acts[i], acts[i+1])
        if w is not None:
            weights.append(max(w, 0.001))
            labels.append((i, i+1))
    if right_act is not None and k > 0:
        w = edge_weight_seconds(acts[-1], right_act)
        if w is not None:
            weights.append(max(w, 0.001))
            labels.append((k-1, "R"))
    return weights, labels

def estimate_block_span(k, left_act=None, right_act=None, acts=None):
    if acts is None:
        acts = []
    fast = False
    candidates = []
    if left_act and acts:
        p = get_edge_profile(left_act, acts[0])
        if p:
            candidates.append(p)
    for i in range(len(acts)-1):
        p = get_edge_profile(acts[i], acts[i+1])
        if p:
            candidates.append(p)
    if right_act and acts:
        p = get_edge_profile(acts[-1], right_act)
        if p:
            candidates.append(p)
    if any(p["share_under_60s"] >= 0.8 for p in candidates):
        fast = True
    if k in BLOCK_SPAN:
        return BLOCK_SPAN[k]["p25" if fast else "p10"]
    return 36.364 if fast else 1.366

def proportional_inside(left_ts, right_ts, acts, left_act, right_act, widths):
    k = len(acts)
    total_window = max((right_ts - left_ts).total_seconds(), 0.001)
    weights, _ = build_internal_weights(acts, left_act, right_act)
    if not weights:
        gaps = [0.001] * (k + 1)
    else:
        gaps = list(weights)
        while len(gaps) < k + 1:
            gaps.append(min(gaps[-1], 0.001))
        if len(gaps) > k + 1:
            gaps = gaps[:k+1]
    s = sum(gaps)
    scaled = [g / s * total_window for g in gaps]
    times = []
    cur = left_ts
    for i in range(k):
        cur = cur + pd.Timedelta(seconds=scaled[i])
        times.append(cur)
    if times and times[-1] > right_ts:
        times[-1] = left_ts + (right_ts - left_ts) / 2
    eps = smallest_plausible_gap(max(widths) if widths else 3)
    fixed = []
    prev = left_ts
    for i, t in enumerate(times):
        if t < prev:
            t = prev
        if t > right_ts:
            t = right_ts
        fixed.append(t)
        prev = t
    for i in range(len(fixed)-2, -1, -1):
        if fixed[i] > fixed[i+1]:
            fixed[i] = fixed[i+1]
    return fixed

# ----------------------------
# Main repair logic
# ----------------------------

def repair_case(group, global_frac):
    group = group.copy()
    idxs = list(group.index)
    n = len(group)

    parsed = [parse_any(group.at[i, "Timestamp"]) for i in idxs]
    clean_flags = [group.at[i, "error_category"] == "CLEAN" for i in idxs]
    acts = [group.at[i, "Activity"] for i in idxs]
    orig_ts = [group.at[i, "Timestamp"] for i in idxs]

    clean_times = [parsed[j] for j in range(n) if clean_flags[j] and not pd.isna(parsed[j])]
    case_start = min(clean_times) if clean_times else pd.Timestamp("2000-01-01 00:00:00")
    case_end = max(clean_times) if clean_times else case_start + pd.Timedelta(days=1)

    repaired = {idxs[j]: orig_ts[j] for j in range(n)}
    repaired_dt = {idxs[j]: parsed[j] for j in range(n)}

    j = 0
    while j < n:
        if clean_flags[j]:
            j += 1
            continue
        start = j
        while j < n and not clean_flags[j]:
            j += 1
        end = j - 1

        block_pos = list(range(start, end + 1))
        block_idxs = [idxs[p] for p in block_pos]
        block_acts = [acts[p] for p in block_pos]
        block_widths = [max(frac_width(orig_ts[p]), global_frac) for p in block_pos]

        left_pos = start - 1 if start - 1 >= 0 and clean_flags[start - 1] else None
        right_pos = end + 1 if end + 1 < n and clean_flags[end + 1] else None

        left_ts = parsed[left_pos] if left_pos is not None else None
        right_ts = parsed[right_pos] if right_pos is not None else None
        left_act = acts[left_pos] if left_pos is not None else None
        right_act = acts[right_pos] if right_pos is not None else None

        _ = candidate_order_supported(([left_act] if left_act else []) + block_acts + ([right_act] if right_act else []))

        # Both anchors exist
        if left_ts is not None and right_ts is not None:
            proposed = proportional_inside(left_ts, right_ts, block_acts, left_act, right_act, block_widths)
            # date-only constraint per row
            for local_i, p in enumerate(block_pos):
                idx = idxs[p]
                obs = orig_ts[p]
                t = proposed[local_i]
                if detect_date_only(obs):
                    obs_day = parse_any(obs).normalize()
                    day_lo, day_hi = same_day_bounds(obs_day)
                    lo = max(left_ts, day_lo)
                    hi = min(right_ts, day_hi)
                    if lo > hi:
                        # keep same date, stay in case bounds, choose midday then clamp
                        cand = obs_day + pd.Timedelta(hours=12)
                        cand = clamp(cand, day_lo, day_hi)
                        cand = clamp(cand, min(left_ts, right_ts), max(left_ts, right_ts))
                        t = cand
                    else:
                        if t < lo or t > hi:
                            mid = lo + (hi - lo) / 2
                            # weighted from left if possible
                            eg = edge_gap_seconds(left_act, acts[p]) if left_act else None
                            if eg is not None:
                                cand = left_ts + pd.Timedelta(seconds=eg)
                                if cand > hi:
                                    cand = mid
                                t = clamp(cand, lo, hi)
                            else:
                                t = clamp(mid, lo, hi)
                repaired_dt[idx] = t
        elif left_ts is not None:
            cur = left_ts
            for local_i, p in enumerate(block_pos):
                idx = idxs[p]
                prev_act = acts[p - 1] if p - 1 >= 0 else None
                gap = edge_gap_seconds(prev_act, acts[p]) if prev_act else None
                if gap is None:
                    span = estimate_block_span(len(block_pos), left_act=left_act, acts=block_acts)
                    gap = max(span / max(len(block_pos), 1), 0.001)
                cur = cur + pd.Timedelta(seconds=gap)
                if detect_date_only(orig_ts[p]):
                    obs_day = parse_any(orig_ts[p]).normalize()
                    day_lo, day_hi = same_day_bounds(obs_day)
                    cur = clamp(cur, day_lo, day_hi)
                    if cur.normalize() != obs_day:
                        cur = clamp(obs_day + pd.Timedelta(hours=12), day_lo, day_hi)
                cur = clamp(cur, case_start, case_end)
                repaired_dt[idx] = cur
        elif right_ts is not None:
            cur = right_ts
            tmp = []
            for local_i, p in reversed(list(enumerate(block_pos))):
                idx = idxs[p]
                next_act = acts[p + 1] if p + 1 < n else None
                gap = edge_gap_seconds(acts[p], next_act) if next_act else None
                if gap is None:
                    span = estimate_block_span(len(block_pos), right_act=right_act, acts=block_acts)
                    gap = max(span / max(len(block_pos), 1), 0.001)
                cur = cur - pd.Timedelta(seconds=gap)
                t = cur
                if detect_date_only(orig_ts[p]):
                    obs_day = parse_any(orig_ts[p]).normalize()
                    day_lo, day_hi = same_day_bounds(obs_day)
                    t = clamp(t, day_lo, day_hi)
                    if t.normalize() != obs_day:
                        t = clamp(obs_day + pd.Timedelta(hours=12), day_lo, day_hi)
                t = clamp(t, case_start, case_end)
                tmp.append((idx, t))
            for idx, t in reversed(tmp):
                repaired_dt[idx] = t
        else:
            # no clean anchors in case; still repair, anchored to observed date or synthetic case bounds
            base = case_start
            for local_i, p in enumerate(block_pos):
                idx = idxs[p]
                obs = orig_ts[p]
                if detect_date_only(obs):
                    obs_day = parse_any(obs).normalize()
                    t = obs_day + pd.Timedelta(hours=12, minutes=local_i)
                else:
                    t = base + pd.Timedelta(seconds=local_i + 1)
                repaired_dt[idx] = t

        # render repaired block
        for local_i, p in enumerate(block_pos):
            idx = idxs[p]
            width = max(frac_width(orig_ts[p]), global_frac)
            repaired[idx] = format_ts(repaired_dt[idx], width)

    return repaired, repaired_dt, case_start, case_end

def self_check_and_fix(df, out_ts, out_dt, case_bounds, global_frac):
    blank_filled = 0
    date_only_filled = 0
    unchanged = 0

    for i in df.index:
        orig = df.at[i, "Timestamp"]
        err = df.at[i, "error_category"]
        case = df.at[i, "Case"]

        if err == "CLEAN":
            out_ts[i] = orig
            out_dt[i] = parse_any(orig)
            continue

        if is_blank(orig):
            blank_filled += 1
        elif detect_date_only(orig):
            date_only_filled += 1

        dt = parse_any(out_ts[i])
        if pd.isna(dt):
            lo, hi = case_bounds.get(case, (pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-01 00:00:01")))
            dt = lo + (hi - lo) / 2
            out_dt[i] = dt
            out_ts[i] = format_ts(dt, max(frac_width(orig), global_frac))

        # case bounds
        lo, hi = case_bounds.get(case, (dt, dt))
        dt = clamp(pd.Timestamp(out_dt[i]), lo, hi)

        # same date for date-only
        if detect_date_only(orig):
            obs_day = parse_any(orig).normalize()
            day_lo, day_hi = same_day_bounds(obs_day)
            dt = clamp(dt, day_lo, day_hi)
            if dt.normalize() != obs_day:
                dt = obs_day + pd.Timedelta(hours=12)

        # anchor bounds inside case using nearest clean anchors
        case_mask = df["Case"] == case
        case_idx = list(df[case_mask].index)
        pos = case_idx.index(i)
        left_anchor = None
        right_anchor = None
        for p in range(pos - 1, -1, -1):
            ii = case_idx[p]
            if df.at[ii, "error_category"] == "CLEAN":
                left_anchor = parse_any(df.at[ii, "Timestamp"])
                break
        for p in range(pos + 1, len(case_idx)):
            ii = case_idx[p]
            if df.at[ii, "error_category"] == "CLEAN":
                right_anchor = parse_any(df.at[ii, "Timestamp"])
                break
        if left_anchor is not None and right_anchor is not None:
            if dt < left_anchor or dt > right_anchor:
                dt = left_anchor + (right_anchor - left_anchor) / 2

        # must differ from input
        candidate = format_ts(dt, max(frac_width(orig), global_frac))
        if candidate == ("" if pd.isna(orig) else str(orig)):
            if detect_date_only(orig):
                dt = dt + pd.Timedelta(seconds=1)
                obs_day = parse_any(orig).normalize()
                day_lo, day_hi = same_day_bounds(obs_day)
                dt = clamp(dt, day_lo, day_hi)
            else:
                dt = dt + pd.Timedelta(milliseconds=1 if global_frac >= 3 else 1)
                dt = clamp(dt, lo, hi)
                if left_anchor is not None and right_anchor is not None:
                    dt = clamp(dt, left_anchor, right_anchor)
            candidate = format_ts(dt, max(frac_width(orig), global_frac))

        out_dt[i] = dt
        out_ts[i] = candidate

        if out_ts[i] == ("" if pd.isna(orig) else str(orig)):
            unchanged += 1

    return blank_filled, date_only_filled, unchanged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_columns = list(df.columns)
    global_frac = infer_global_frac_width(df["Timestamp"])

    out_ts = {i: df.at[i, "Timestamp"] for i in df.index}
    out_dt = {i: parse_any(df.at[i, "Timestamp"]) for i in df.index}
    case_bounds = {}

    try:
        for case, group in df.groupby("Case", sort=False):
            repaired, repaired_dt, case_start, case_end = repair_case(group, global_frac)
            case_bounds[case] = (case_start, case_end)
            for i, v in repaired.items():
                out_ts[i] = v
            for i, v in repaired_dt.items():
                out_dt[i] = v
    except Exception:
        # continue; self-check will still try to fix
        pass

    blank_filled, date_only_filled, unchanged = self_check_and_fix(df, out_ts, out_dt, case_bounds, global_frac)

    result = df.copy()
    result["Timestamp"] = [out_ts[i] for i in result.index]

    # preserve exact untouched fields and row order/row count
    result = result[original_columns]
    result.to_csv(args.output, index=False)

    print(f"blank_rows_filled={blank_filled}, date_only_rows_timed={date_only_filled}, rows_still_equal_to_input={unchanged}")

if __name__ == "__main__":
    main()