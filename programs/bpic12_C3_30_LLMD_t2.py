import argparse
import math
import re
from datetime import timedelta
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Embedded process knowledge from prompt
# ------------------------------------------------------------

EDGE_STATS = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): {"p10": 50.590100000000014, "p25": 14888.91575, "median": 167356.1795, "share": 0.104},
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): {"p10": 10.8058, "p25": 41.519, "median": 99.459, "share": 0.336},
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): {"p10": 18.568, "p25": 56.637, "median": 115.945, "share": 0.265},
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 61.73760000000002, "p25": 1598.129, "median": 14639.161, "share": 0.099},
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): {"p10": 0.126, "p25": 0.175, "median": 0.275, "share": 1.0},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 76.92500000000001, "p25": 1284.3235, "median": 10784.402, "share": 0.093},
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 10.2459, "p25": 33.622749999999996, "median": 140.29149999999998, "share": 0.33},
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): {"p10": 0.046, "p25": 0.13325, "median": 0.378, "share": 1.0},
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): {"p10": 0.025, "p25": 0.028, "median": 0.035, "share": 1.0},
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): {"p10": 0.083, "p25": 0.093, "median": 0.119, "share": 1.0},
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9155, "p25": 1.08425, "median": 1.291, "share": 1.0},
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): {"p10": 125.3631, "p25": 247.7525, "median": 373.9065, "share": 0.048},
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): {"p10": 12.5822, "p25": 20.106, "median": 310.933, "share": 0.421},
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.077, "p25": 1.18, "median": 1.3904999999999998, "share": 1.0},
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): {"p10": 30.8955, "p25": 742.71875, "median": 6597.786, "share": 0.144},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): {"p10": 32.8288, "p25": 35.5895, "median": 38.917500000000004, "share": 0.971},
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): {"p10": 83.1826, "p25": 597.00775, "median": 2551.737, "share": 0.085},
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): {"p10": 8.6702, "p25": 11.885, "median": 22.736, "share": 0.949},
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 63.8074, "p25": 118.41499999999999, "median": 183.87, "share": 0.093},
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): {"p10": 0.141, "p25": 0.153, "median": 0.179, "share": 0.999},
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): {"p10": 10.1838, "p25": 12.41, "median": 21.162, "share": 0.827},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.115, "p25": 1.228, "median": 1.447, "share": 1.0},
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 7064.724000000002, "p25": 89340.6535, "median": 189136.3675, "share": 0.049},
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): {"p10": 10.718200000000001, "p25": 44.835, "median": 307.187, "share": 0.284},
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 24.299800000000005, "p25": 244.81225, "median": 2005.5505, "share": 0.144},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.724, "p25": 34.09525, "median": 37.348, "share": 0.977},
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 18.262100000000004, "p25": 908.79625, "median": 5975.98, "share": 0.148},
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): {"p10": 35.6547, "p25": 66.4165, "median": 107.805, "share": 0.21},
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): {"p10": 0.898, "p25": 0.987, "median": 1.148, "share": 1.0},
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 26.1244, "p25": 332.025, "median": 2855.616, "share": 0.155},
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): {"p10": 1.446, "p25": 1.605, "median": 1.843, "share": 1.0},
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): {"p10": 2.843, "p25": 3.0815, "median": 3.41, "share": 0.998},
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): {"p10": 32.1372, "p25": 69.84474999999999, "median": 113.25049999999999, "share": 0.203},
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): {"p10": 254.43380000000002, "p25": 623.07, "median": 1174.31, "share": 0.029},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 1.486, "p25": 1.649, "median": 1.895, "share": 1.0},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 696.887, "p25": 2100.1994999999997, "median": 5033.333, "share": 0.031},
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): {"p10": 31.965, "p25": 48.504, "median": 106.471, "share": 0.316},
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.2212, "p25": 1.3115, "median": 1.618, "share": 0.999},
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): {"p10": 24.7997, "p25": 36.5235, "median": 65.38050000000001, "share": 0.466},
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share": 1.0},
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): {"p10": 8.047200000000004, "p25": 46.988, "median": 135.472, "share": 0.292},
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 2.9624, "p25": 3.184, "median": 3.533, "share": 0.999},
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): {"p10": 3.8414, "p25": 17.074749999999998, "median": 362.5105, "share": 0.327},
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): {"p10": 179.8682, "p25": 315.4015, "median": 474.467, "share": 0.022},
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.4536, "p25": 7.027, "median": 7.763, "share": 0.999},
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.837, "p25": 2.093, "median": 2.396, "share": 0.998},
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): {"p10": 72.66199999999999, "p25": 111.11025, "median": 156.877, "share": 0.081},
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.3086, "p25": 6.952, "median": 7.6685, "share": 1.0},
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): {"p10": 141.36260000000001, "p25": 317.4815, "median": 770.573, "share": 0.019},
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_SUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.7182, "p25": 34.433, "median": 38.218, "share": 0.987},
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.3078, "p25": 1.4635, "median": 1.661, "share": 0.999},
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9084000000000001, "p25": 1.053, "median": 1.251, "share": 1.0},
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share": 1.0},
}

BLOCK_SPAN = {
    2: {"p10": 0.063, "p25": 1.305, "median": 49.725},
    3: {"p10": 1.366, "p25": 36.364, "median": 995.679},
    4: {"p10": 5.7904000000000195, "p25": 217.364, "median": 6361.145},
    5: {"p10": 50.37720000000001, "p25": 866.1295, "median": 21853.089},
}
GLOBAL_MEDIAN = 49.725

FREQ_VARIANTS = [
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def parse_ts(s):
    if pd.isna(s):
        return pd.NaT
    txt = str(s).strip()
    if txt == "":
        return pd.NaT
    try:
        return pd.to_datetime(txt)
    except Exception:
        return pd.NaT

def detect_precision_and_sep(series):
    max_digits = 0
    sep = "."
    for v in series.astype(str):
        m = re.search(r"(\d{2}:\d{2}:\d{2})([.,](\d+))?$", v.strip())
        if m:
            frac = m.group(3)
            if frac is not None:
                max_digits = max(max_digits, len(frac))
                if m.group(2):
                    sep = m.group(2)[0]
    if max_digits == 0:
        max_digits = 3
        sep = "."
    return max_digits, sep

def format_ts(ts, digits, sep="."):
    if pd.isna(ts):
        return ""
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if digits <= 0:
        return base
    ns = ts.value % 1_000_000_000
    frac = f"{ns:09d}"[:digits]
    return base + sep + frac

def is_blank_observation(s):
    return pd.isna(s) or str(s).strip() == ""

def observed_date_survived(s):
    if pd.isna(s):
        return False
    txt = str(s).strip()
    if txt == "":
        return False
    dt = parse_ts(txt)
    if pd.isna(dt):
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", txt):
        return True
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s+00:00:00(?:[.,]0+)?", txt)
    return m is not None

def get_same_observed_date(s):
    dt = parse_ts(s)
    if pd.isna(dt):
        return None
    return dt.date()

def edge_gap_seconds(a, b):
    st = EDGE_STATS.get((a, b))
    if st is None:
        return None
    return st["p10"] if st["share"] >= 0.8 else st["p25"]

def block_span_prior_seconds(k):
    if k in BLOCK_SPAN:
        return BLOCK_SPAN[k]["p25"]
    if k < 2:
        return max(0.001, GLOBAL_MEDIAN / 10.0)
    return BLOCK_SPAN[max(BLOCK_SPAN.keys())]["p25"] * (k / max(BLOCK_SPAN.keys()))

def tiny_gap_seconds(idx, digits):
    unit = 10 ** (-digits) if digits > 0 else 1.0
    return unit * (1 + (idx % 7))

def candidate_gap_seconds(left_act, right_act, digits, pos):
    g = edge_gap_seconds(left_act, right_act)
    if g is not None:
        return max(g, tiny_gap_seconds(pos, digits))
    return max(block_span_prior_seconds(2), tiny_gap_seconds(pos, digits))

def sequence_plausible(activities):
    if len(activities) <= 1:
        return True
    edge_hits = sum((activities[i], activities[i+1]) in EDGE_STATS for i in range(len(activities)-1))
    if edge_hits >= max(1, len(activities)-2):
        return True
    for var in FREQ_VARIANTS:
        for i in range(0, len(var)-len(activities)+1):
            if var[i:i+len(activities)] == activities:
                return True
    return edge_hits > 0

def non_uniformize(times, digits):
    if len(times) < 3:
        return times
    vals = [t.value for t in times]
    gaps = np.diff(vals)
    if len(set(gaps.tolist())) <= 1:
        unit_ns = int((10 ** (9 - digits)) if digits > 0 else 1_000_000_000)
        adjusted = [times[0]]
        for i in range(1, len(times)):
            bump = unit_ns * i
            adjusted.append(pd.Timestamp(adjusted[-1].value + max(unit_ns, bump)))
        return adjusted
    return times

def clamp(ts, lo, hi):
    if lo is not None and ts < lo:
        ts = lo
    if hi is not None and ts > hi:
        ts = hi
    return ts

def same_day_bounds(day, lo, hi):
    start = pd.Timestamp(f"{day} 00:00:00")
    end = pd.Timestamp(f"{day} 23:59:59.999999999")
    if lo is not None:
        start = max(start, lo)
    if hi is not None:
        end = min(end, hi)
    return start, end

def choose_inside_window(left_anchor, right_anchor, cumulative_seconds, block_len, digits):
    total_window = max((right_anchor - left_anchor).total_seconds(), 0.0)
    if block_len == 1:
        sec = min(cumulative_seconds[0], total_window / 2.0 if total_window > 0 else 0.0)
        if sec > total_window:
            sec = total_window / 2.0
        return [left_anchor + pd.to_timedelta(sec, unit="s")]
    target = cumulative_seconds[-1]
    if target <= total_window and target > 0:
        secs = cumulative_seconds
    else:
        if target <= 0:
            target = block_span_prior_seconds(block_len)
        if total_window <= 0:
            secs = [0.0] * block_len
        else:
            ratio = total_window / target
            secs = [s * ratio for s in cumulative_seconds]
    out = [left_anchor + pd.to_timedelta(s, unit="s") for s in secs]
    out = non_uniformize(out, digits)
    out = [clamp(t, left_anchor, right_anchor) for t in out]
    return out

def infer_block(case_df, idxs, digits, sep):
    acts = case_df["Activity"].tolist()
    orig_ts = case_df["Timestamp"].tolist()
    parsed = case_df["_parsed"].tolist()
    errs = case_df["error_category"].tolist()

    s = idxs[0]
    e = idxs[-1]
    n = len(idxs)

    clean_positions = [i for i, ec in enumerate(errs) if ec == "CLEAN" and not pd.isna(parsed[i])]
    case_start = parsed[min(clean_positions)] if clean_positions else None
    case_end = parsed[max(clean_positions)] if clean_positions else None

    left_idx = None
    for i in range(s - 1, -1, -1):
        if errs[i] == "CLEAN" and not pd.isna(parsed[i]):
            left_idx = i
            break

    right_idx = None
    for i in range(e + 1, len(case_df)):
        if errs[i] == "CLEAN" and not pd.isna(parsed[i]):
            right_idx = i
            break

    left_anchor = parsed[left_idx] if left_idx is not None else case_start
    right_anchor = parsed[right_idx] if right_idx is not None else case_end

    block_acts = acts[max(0, s-1):min(len(acts), e+2)]
    _ = sequence_plausible(block_acts)

    cumulative = []
    acc = 0.0
    for j, idx in enumerate(idxs):
        prev_act = acts[left_idx] if j == 0 and left_idx is not None else (acts[idxs[j-1]] if j > 0 else None)
        this_act = acts[idx]
        if prev_act is not None:
            g = candidate_gap_seconds(prev_act, this_act, digits, j+1)
        else:
            g = tiny_gap_seconds(j+1, digits)
        acc += max(g, tiny_gap_seconds(j+1, digits))
        cumulative.append(acc)

    if n >= 2 and cumulative[-1] < block_span_prior_seconds(n):
        cumulative[-1] = block_span_prior_seconds(n)
        for j in range(n - 2, -1, -1):
            cumulative[j] = min(cumulative[j], cumulative[-1] * (0.15 + 0.7 * (j + 1) / n))

    repaired = {}

    same_day_rows = {idx: get_same_observed_date(orig_ts[idx]) for idx in idxs if observed_date_survived(orig_ts[idx])}

    if left_anchor is not None and right_anchor is not None and left_anchor <= right_anchor:
        placed = choose_inside_window(left_anchor, right_anchor, cumulative, n, digits)
        for j, idx in enumerate(idxs):
            t = placed[j]
            if idx in same_day_rows:
                d = same_day_rows[idx]
                lo, hi = same_day_bounds(d, left_anchor, right_anchor)
                if lo <= hi:
                    if t.date() != d or t < lo or t > hi:
                        if j == 0 and j + 1 < n:
                            t = max(lo, min(hi, lo + (hi - lo) / 3))
                        elif j == n - 1 and j > 0:
                            t = max(lo, min(hi, lo + 2 * (hi - lo) / 3))
                        else:
                            t = max(lo, min(hi, lo + (hi - lo) / 2))
                else:
                    t = clamp(t, left_anchor, right_anchor)
            repaired[idx] = t
    elif left_anchor is not None:
        for j, idx in enumerate(idxs):
            t = left_anchor + pd.to_timedelta(cumulative[j], unit="s")
            if idx in same_day_rows:
                d = same_day_rows[idx]
                lo, hi = same_day_bounds(d, case_start, case_end)
                if lo <= hi:
                    if t.date() != d:
                        tod = (t - pd.Timestamp(t.date())).total_seconds()
                        candidate = pd.Timestamp(d) + pd.to_timedelta(min(max(tod, 0), 86399.999999), unit="s")
                        t = clamp(candidate, lo, hi)
                    else:
                        t = clamp(t, lo, hi)
                else:
                    t = clamp(t, case_start, case_end)
            else:
                t = clamp(t, case_start, case_end)
            repaired[idx] = t
    elif right_anchor is not None:
        rev_cum = []
        acc2 = 0.0
        for j in range(n - 1, -1, -1):
            idx = idxs[j]
            next_act = acts[right_idx] if j == n - 1 and right_idx is not None else (acts[idxs[j+1]] if j < n - 1 else None)
            this_act = acts[idx]
            if next_act is not None:
                g = candidate_gap_seconds(this_act, next_act, digits, j+1)
            else:
                g = tiny_gap_seconds(j+1, digits)
            acc2 += max(g, tiny_gap_seconds(j+1, digits))
            rev_cum.append(acc2)
        rev_cum = list(reversed(rev_cum))
        for j, idx in enumerate(idxs):
            t = right_anchor - pd.to_timedelta(rev_cum[j], unit="s")
            if idx in same_day_rows:
                d = same_day_rows[idx]
                lo, hi = same_day_bounds(d, case_start, case_end)
                if lo <= hi:
                    if t.date() != d:
                        tod = (t - pd.Timestamp(t.date())).total_seconds()
                        candidate = pd.Timestamp(d) + pd.to_timedelta(min(max(tod, 0), 86399.999999), unit="s")
                        t = clamp(candidate, lo, hi)
                    else:
                        t = clamp(t, lo, hi)
                else:
                    t = clamp(t, case_start, case_end)
            else:
                t = clamp(t, case_start, case_end)
            repaired[idx] = t
    else:
        base = pd.Timestamp("2011-01-01 12:00:00")
        for j, idx in enumerate(idxs):
            if idx in same_day_rows:
                d = same_day_rows[idx]
                t = pd.Timestamp(d) + pd.to_timedelta(12 * 3600 + j * max(1, tiny_gap_seconds(j+1, digits)), unit="s")
            else:
                t = base + pd.to_timedelta(j * max(1, tiny_gap_seconds(j+1, digits)), unit="s")
            repaired[idx] = t

    times = [repaired[idx] for idx in idxs]
    times = non_uniformize(times, digits)

    if left_anchor is not None and right_anchor is not None:
        times = [clamp(t, left_anchor, right_anchor) for t in times]
    else:
        times = [clamp(t, case_start, case_end) if (case_start is not None and case_end is not None) else t for t in times]

    # enforce monotone nondecreasing in row order
    unit_ns = int(10 ** (9 - digits)) if digits > 0 else 1_000_000_000
    for i in range(1, len(times)):
        if times[i].value < times[i-1].value:
            times[i] = pd.Timestamp(times[i-1].value + unit_ns)
    if right_anchor is not None:
        for i in range(len(times)-2, -1, -1):
            if times[i].value > times[i+1].value:
                times[i] = pd.Timestamp(max(times[i].value, times[i+1].value - unit_ns))
        times = [clamp(t, left_anchor if left_anchor is not None else None, right_anchor) for t in times]

    # preserve observed date for date-only rows
    for i, idx in enumerate(idxs):
        if idx in same_day_rows:
            d = same_day_rows[idx]
            lo = left_anchor if left_anchor is not None else case_start
            hi = right_anchor if right_anchor is not None else case_end
            sday_lo, sday_hi = same_day_bounds(d, lo, hi)
            t = times[i]
            if sday_lo <= sday_hi:
                if t.date() != d or t < sday_lo or t > sday_hi:
                    frac = (i + 1) / (len(times) + 1)
                    candidate = sday_lo + (sday_hi - sday_lo) * frac
                    times[i] = clamp(candidate, sday_lo, sday_hi)
            else:
                midpoint = pd.Timestamp(d) + pd.to_timedelta(12, unit="h")
                times[i] = midpoint

    # final anti-equality perturbation
    out = {}
    for i, idx in enumerate(idxs):
        ts = times[i]
        new_s = format_ts(ts, digits, sep)
        old_s = "" if pd.isna(orig_ts[idx]) else str(orig_ts[idx])
        if new_s == old_s:
            ts = ts + pd.to_timedelta(tiny_gap_seconds(i+3, digits), unit="s")
            if left_anchor is not None and right_anchor is not None:
                ts = clamp(ts, left_anchor, right_anchor)
            elif case_start is not None and case_end is not None:
                ts = clamp(ts, case_start, case_end)
            if idx in same_day_rows:
                d = same_day_rows[idx]
                lo = left_anchor if left_anchor is not None else case_start
                hi = right_anchor if right_anchor is not None else case_end
                sday_lo, sday_hi = same_day_bounds(d, lo, hi)
                if sday_lo <= sday_hi:
                    ts = clamp(ts, sday_lo, sday_hi)
            new_s = format_ts(ts, digits, sep)
            if new_s == old_s:
                ts = ts - pd.to_timedelta(tiny_gap_seconds(i+5, digits), unit="s")
        out[idx] = ts
    return out

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original = df.copy(deep=True)
    digits, sep = detect_precision_and_sep(df["Timestamp"] if "Timestamp" in df.columns else pd.Series(dtype=str))
    df["_parsed"] = df["Timestamp"].apply(parse_ts)

    repaired_ts_strings = df["Timestamp"].astype(str).tolist()
    blank_filled = 0
    date_only_filled = 0

    for case_id, idxs_case in df.groupby("Case", sort=False).groups.items():
        case_idx_list = list(idxs_case)
        case_df = df.loc[case_idx_list].reset_index(drop=False)
        corrupt_mask = case_df["error_category"] != "CLEAN"
        if not corrupt_mask.any():
            continue

        pos = 0
        while pos < len(case_df):
            if case_df.loc[pos, "error_category"] == "CLEAN":
                pos += 1
                continue
            start = pos
            while pos < len(case_df) and case_df.loc[pos, "error_category"] != "CLEAN":
                pos += 1
            end = pos - 1
            block_positions = list(range(start, end + 1))
            global_idxs = case_df.loc[block_positions, "index"].tolist()

            block_repair = infer_block(case_df, block_positions, digits, sep)

            for local_pos, gidx in zip(block_positions, global_idxs):
                old = df.at[gidx, "Timestamp"]
                if is_blank_observation(old):
                    blank_filled += 1
                elif observed_date_survived(old):
                    date_only_filled += 1
                ts = block_repair[local_pos]
                repaired_ts_strings[gidx] = format_ts(ts, digits, sep)
                df.at[gidx, "_parsed"] = ts

    # self-check and repair
    still_equal = 0
    for case_id, idxs_case in df.groupby("Case", sort=False).groups.items():
        case_idx_list = list(idxs_case)
        clean_mask = (df.loc[case_idx_list, "error_category"] == "CLEAN") & df.loc[case_idx_list, "_parsed"].notna()
        clean_idxs = [idx for idx in case_idx_list if clean_mask.loc[idx]]
        case_start = df.at[clean_idxs[0], "_parsed"] if clean_idxs else None
        case_end = df.at[clean_idxs[-1], "_parsed"] if clean_idxs else None

        for j, gidx in enumerate(case_idx_list):
            if df.at[gidx, "error_category"] == "CLEAN":
                repaired_ts_strings[gidx] = original.at[gidx, "Timestamp"]
                continue

            s = repaired_ts_strings[gidx]
            ts = parse_ts(s)
            if pd.isna(ts):
                left = None
                right = None
                for k in range(j - 1, -1, -1):
                    gi = case_idx_list[k]
                    t = parse_ts(repaired_ts_strings[gi])
                    if not pd.isna(t):
                        left = t
                        break
                for k in range(j + 1, len(case_idx_list)):
                    gi = case_idx_list[k]
                    t = parse_ts(repaired_ts_strings[gi])
                    if not pd.isna(t):
                        right = t
                        break
                if left is not None and right is not None:
                    ts = left + (right - left) / 2
                elif left is not None:
                    ts = left + pd.to_timedelta(max(1, tiny_gap_seconds(j+1, digits)), unit="s")
                elif right is not None:
                    ts = right - pd.to_timedelta(max(1, tiny_gap_seconds(j+1, digits)), unit="s")
                else:
                    ts = pd.Timestamp("2011-01-01 12:00:00")

            left_anchor = None
            right_anchor = None
            for k in range(j - 1, -1, -1):
                gi = case_idx_list[k]
                t = parse_ts(repaired_ts_strings[gi])
                if not pd.isna(t):
                    left_anchor = t
                    break
            for k in range(j + 1, len(case_idx_list)):
                gi = case_idx_list[k]
                t = parse_ts(repaired_ts_strings[gi])
                if not pd.isna(t):
                    right_anchor = t
                    break

            if case_start is not None and ts < case_start:
                ts = case_start
            if case_end is not None and ts > case_end:
                ts = case_end
            if left_anchor is not None and ts < left_anchor:
                ts = left_anchor
            if right_anchor is not None and ts > right_anchor:
                if left_anchor is not None and left_anchor <= right_anchor:
                    ts = left_anchor + (right_anchor - left_anchor) / 2
                else:
                    ts = right_anchor

            old = original.at[gidx, "Timestamp"]
            if observed_date_survived(old):
                d = get_same_observed_date(old)
                lo = case_start
                hi = case_end
                if left_anchor is not None:
                    lo = left_anchor if lo is None else max(lo, left_anchor)
                if right_anchor is not None:
                    hi = right_anchor if hi is None else min(hi, right_anchor)
                sday_lo, sday_hi = same_day_bounds(d, lo, hi)
                if sday_lo <= sday_hi:
                    if ts.date() != d or ts < sday_lo or ts > sday_hi:
                        ts = sday_lo + (sday_hi - sday_lo) / 2
                else:
                    ts = pd.Timestamp(d) + pd.to_timedelta(12, unit="h")

            new_s = format_ts(ts, digits, sep)
            if new_s == old:
                ts = ts + pd.to_timedelta(tiny_gap_seconds(j+9, digits), unit="s")
                if case_start is not None and ts < case_start:
                    ts = case_start
                if case_end is not None and ts > case_end:
                    ts = case_end
                if left_anchor is not None and ts < left_anchor:
                    ts = left_anchor
                if right_anchor is not None and ts > right_anchor:
                    if left_anchor is not None and left_anchor <= right_anchor:
                        ts = left_anchor + (right_anchor - left_anchor) / 2
                    else:
                        ts = right_anchor
                new_s = format_ts(ts, digits, sep)
                if new_s == old:
                    ts = ts - pd.to_timedelta(tiny_gap_seconds(j+11, digits), unit="s")
                    new_s = format_ts(ts, digits, sep)

            repaired_ts_strings[gidx] = new_s
            if new_s == old:
                still_equal += 1

    out = original.copy(deep=True)
    out["Timestamp"] = repaired_ts_strings

    # enforce exact untouched columns/order/count
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        out[col] = original[col]

    out.to_csv(args.output, index=False)

    print(f"blank_rows_filled={blank_filled} date_only_rows_given_time={date_only_filled} rows_still_equal_to_input={still_equal}")

if __name__ == "__main__":
    main()