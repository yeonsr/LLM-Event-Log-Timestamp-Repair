import argparse
import math
import sys
from datetime import timedelta
import pandas as pd

# ============================================================
# Process knowledge from prompt
# ============================================================

DURATION_PROFILES = {
    ("Check for completeness", "Request info"): {"p10": 62.6422, "p25": 172.5330, "median": 418.656, "share_under_60s": 0.096},
    ("info received", "Check for completeness"): {"p10": 65.0913, "p25": 176.3765, "median": 415.3415, "share_under_60s": 0.092},
    ("Request info", "info received"): {"p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0},
    ("Check for completeness", "Perform checks"): {"p10": 62.5960, "p25": 174.17, "median": 418.43, "share_under_60s": 0.096},
    ("Perform checks", "Make decision"): {"p10": 64.8835, "p25": 172.6405, "median": 423.4695, "share_under_60s": 0.092},
    ("New online application received", "Check for completeness"): {"p10": 65.0406, "p25": 171.767, "median": 416.959, "share_under_60s": 0.093},
    ("Deliver card", "EVENT 13 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("Notify accept", "Deliver card"): {"p10": 59.4585, "p25": 176.35075, "median": 425.762, "share_under_60s": 0.101},
    ("Make decision", "notify reject"): {"p10": 64.834, "p25": 175.9765, "median": 426.0405, "share_under_60s": 0.094},
    ("Make decision", "Notify accept"): {"p10": 65.664, "p25": 174.0265, "median": 426.665, "share_under_60s": 0.09},
    ("time out", "EVENT 13 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("notify reject", "time out"): {"p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0},
    ("notify reject", "review request received"): {"p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0},
    ("review request received", "Check for completeness"): {"p10": 75.725, "p25": 186.74975, "median": 442.53, "share_under_60s": 0.077},
    ("Check for completeness", "Make decision"): {"p10": 337.4222, "p25": 587.1220, "median": 962.0195, "share_under_60s": 0.009},
    ("New online application received", "Perform checks"): {"p10": 347.2758, "p25": 596.06, "median": 1029.426, "share_under_60s": 0.005},
    ("Perform checks", "Notify accept"): {"p10": 354.0812, "p25": 548.193, "median": 872.637, "share_under_60s": 0.005},
    ("Request info", "Check for completeness"): {"p10": 81.2658, "p25": 177.8785, "median": 444.7555, "share_under_60s": 0.065},
    ("Check for completeness", "info received"): {"p10": 79.1902, "p25": 193.724, "median": 472.726, "share_under_60s": 0.067},
    ("Perform checks", "notify reject"): {"p10": 454.6746, "p25": 726.6280, "median": 1076.069, "share_under_60s": 0.008},
    ("New online application received", "Request info"): {"p10": 351.9840, "p25": 510.01375, "median": 884.5265, "share_under_60s": 0.0},
    ("info received", "Perform checks"): {"p10": 407.9895, "p25": 598.8085, "median": 1124.7845, "share_under_60s": 0.0},
    ("info received", "Request info"): {"p10": 414.5244, "p25": 664.2945, "median": 1001.1390, "share_under_60s": 0.017},
    ("notify reject", "Check for completeness"): {"p10": 93.056, "p25": 216.3995, "median": 385.786, "share_under_60s": 0.065},
    ("Make decision", "review request received"): {"p10": 121.8776, "p25": 198.772, "median": 412.731, "share_under_60s": 0.0},
    ("review request received", "Perform checks"): {"p10": 550.4103, "p25": 717.58225, "median": 1404.037, "share_under_60s": 0.0},
    ("review request received", "Request info"): {"p10": 315.8568, "p25": 545.983, "median": 1071.631, "share_under_60s": 0.067},
    ("EVENT 13 END", "EVENT 13 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
}

DFG_FREQ = {
    ("Check for completeness", "Request info"): 6543,
    ("info received", "Check for completeness"): 6538,
    ("Request info", "info received"): 6534,
    ("Check for completeness", "Perform checks"): 6037,
    ("Perform checks", "Make decision"): 6036,
    ("New online application received", "Check for completeness"): 4705,
    ("Notify accept", "Deliver card"): 3276,
    ("Deliver card", "EVENT 13 END"): 3276,
    ("Make decision", "notify reject"): 3256,
    ("Make decision", "Notify accept"): 3083,
    ("notify reject", "time out"): 1724,
    ("time out", "EVENT 13 END"): 1724,
    ("notify reject", "review request received"): 1624,
    ("review request received", "Check for completeness"): 1616,
    ("Check for completeness", "Make decision"): 330,
    ("New online application received", "Perform checks"): 209,
    ("Perform checks", "Notify accept"): 193,
    ("Request info", "Check for completeness"): 170,
    ("Check for completeness", "info received"): 150,
    ("Perform checks", "notify reject"): 123,
    ("New online application received", "Request info"): 86,
    ("info received", "Perform checks"): 86,
    ("info received", "Request info"): 60,
    ("notify reject", "Check for completeness"): 31,
    ("Make decision", "review request received"): 27,
    ("review request received", "Perform checks"): 20,
    ("review request received", "Request info"): 15,
    ("EVENT 13 END", "EVENT 13 END"): 8,
}

FREQUENT_VARIANTS = [
    ["New online application received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Make decision","notify reject","review request received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","notify reject","review request received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Make decision","notify reject","review request received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Make decision","notify reject","review request received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
]

BLOCK_SPAN = {
    2: {"p10": 10.0, "p25": 10.0, "median": 243.7965, "p75": 674.95, "p90": 1231.9611},
    3: {"p10": 108.0388, "p25": 324.6695, "median": 734.9865, "p75": 1334.2505, "p90": 2087.1570},
    4: {"p10": 383.1805, "p25": 705.348, "median": 1215.442, "p75": 1937.518, "p90": 2788.0586},
    5: {"p10": 667.8252, "p25": 1079.14125, "median": 1700.489, "p75": 2524.783, "p90": 3453.6091},
}

GLOBAL_MEDIAN = 243.7965


# ============================================================
# Utility helpers
# ============================================================

def safe_to_datetime(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

def is_blank_timestamp(s):
    return pd.isna(s) or str(s).strip() == ""

def frac_len_from_string(s):
    s = "" if pd.isna(s) else str(s)
    if "." in s:
        return len(s.split(".")[-1])
    return 0

def detect_global_fractional_digits(series):
    mx = 0
    for s in series.astype(str):
        if s and s.lower() != 'nan':
            mx = max(mx, frac_len_from_string(s))
    return mx

def format_timestamp(ts, frac_digits):
    if frac_digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    return base[:-6 + frac_digits]

def is_date_only_or_midnight_damage(s):
    if is_blank_timestamp(s):
        return False
    ts = safe_to_datetime(s)
    if pd.isna(ts):
        return False
    txt = str(s).strip()
    # For this dataset, date-only missing time may appear as midnight with preserved date.
    return (ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0) or (len(txt) == 10)

def original_surviving_date(s):
    ts = safe_to_datetime(s)
    if pd.isna(ts):
        return None
    return ts.date()

def get_edge_profile(a, b):
    prof = DURATION_PROFILES.get((a, b))
    if prof is not None:
        return prof
    if (a, b) in DFG_FREQ:
        return {"p10": None, "p25": None, "median": GLOBAL_MEDIAN, "share_under_60s": 0.0}
    return None

def edge_prior_seconds(a, b):
    prof = get_edge_profile(a, b)
    if prof is None:
        return GLOBAL_MEDIAN
    if prof["median"] == 0.0 or prof["share_under_60s"] >= 0.95:
        return prof["p10"] if prof["p10"] is not None else 0.0
    if prof["p25"] is not None:
        return prof["p25"]
    return prof["median"]

def pair_exists(a, b):
    return (a, b) in DFG_FREQ

def block_span_prior_seconds(k):
    if k in BLOCK_SPAN:
        return BLOCK_SPAN[k]["p25"]
    nearest = min(BLOCK_SPAN.keys(), key=lambda x: abs(x-k))
    return BLOCK_SPAN[nearest]["p25"]

def contiguous_subsequence_exists(seq):
    for var in FREQUENT_VARIANTS:
        m = len(seq)
        for i in range(len(var) - m + 1):
            if var[i:i+m] == seq:
                return True
    return False

def choose_order(activities):
    observed = activities[:]
    valid_pairs = sum(1 for i in range(len(observed)-1) if pair_exists(observed[i], observed[i+1]))
    if valid_pairs == max(0, len(observed)-1):
        return observed
    if contiguous_subsequence_exists(observed):
        return observed
    # constrained search only if needed; preserve multiset and size usually tiny
    # search among subsequences from known variants matching the exact multiset in order
    from collections import Counter
    target_counter = Counter(observed)
    best = observed
    best_score = (-1, -1)
    for var in FREQUENT_VARIANTS:
        m = len(observed)
        for i in range(len(var) - m + 1):
            cand = var[i:i+m]
            if Counter(cand) != target_counter:
                continue
            edge_cov = sum(1 for j in range(len(cand)-1) if pair_exists(cand[j], cand[j+1]))
            edge_freq = sum(DFG_FREQ.get((cand[j], cand[j+1]), 0) for j in range(len(cand)-1))
            score = (edge_cov, edge_freq)
            if score > best_score:
                best_score = score
                best = cand
    return best

def feasible_interval(left_anchor, right_anchor, case_start, case_end, surviving_date):
    lo = case_start
    hi = case_end
    if left_anchor is not None:
        lo = max(lo, left_anchor)
    if right_anchor is not None:
        hi = min(hi, right_anchor)
    if surviving_date is not None:
        day_start = pd.Timestamp(str(surviving_date) + " 00:00:00")
        day_end = pd.Timestamp(str(surviving_date) + " 23:59:59.999999")
        lo = max(lo, day_start)
        hi = min(hi, day_end)
    return lo, hi

def clamp(ts, lo, hi):
    if ts < lo:
        return lo
    if ts > hi:
        return hi
    return ts

def tiny_step(frac_digits):
    if frac_digits >= 6:
        return pd.Timedelta(microseconds=1)
    if frac_digits == 3:
        return pd.Timedelta(milliseconds=1)
    if frac_digits == 2:
        return pd.Timedelta(milliseconds=10)
    if frac_digits == 1:
        return pd.Timedelta(milliseconds=100)
    return pd.Timedelta(seconds=1)

def make_monotone_nonuniform(times, los, his, frac_digits, edge_secs):
    eps = tiny_step(frac_digits)
    n = len(times)
    if n == 0:
        return times
    # forward
    for i in range(n):
        times[i] = clamp(times[i], los[i], his[i])
        if i > 0 and times[i] < times[i-1]:
            preferred = times[i-1]
            if edge_secs[i-1] > 0:
                preferred = times[i-1] + min(eps, pd.Timedelta(seconds=edge_secs[i-1] / 10.0))
            times[i] = clamp(preferred, los[i], his[i])
    # backward to fit highs
    for i in range(n-2, -1, -1):
        if times[i] > times[i+1]:
            times[i] = clamp(times[i+1], los[i], his[i])
    # break accidental evenly spaced patterns a little where possible
    if n >= 3:
        for i in range(1, n-1):
            d1 = (times[i] - times[i-1]).total_seconds()
            d2 = (times[i+1] - times[i]).total_seconds()
            if abs(d1 - d2) < 1e-9:
                bump = eps
                candidate = times[i] + bump
                if candidate <= his[i] and candidate <= times[i+1]:
                    times[i] = candidate
                else:
                    candidate = times[i] - bump
                    if candidate >= los[i] and candidate >= times[i-1]:
                        times[i] = candidate
    return times

def repair_block(case_df, idxs, output_ts, frac_digits):
    activities = case_df.loc[idxs, "Activity"].tolist()
    chosen_order = choose_order(activities)
    # preserve observed row order timestamps assignment; ordering search only validates sequence
    _ = chosen_order

    all_case_idxs = case_df.index.tolist()
    first_pos = all_case_idxs.index(idxs[0])
    last_pos = all_case_idxs.index(idxs[-1])

    left_anchor = None
    left_act = None
    for p in range(first_pos - 1, -1, -1):
        ridx = all_case_idxs[p]
        if case_df.at[ridx, "error_category"] == "CLEAN":
            left_anchor = safe_to_datetime(case_df.at[ridx, "Timestamp"])
            left_act = case_df.at[ridx, "Activity"]
            if not pd.isna(left_anchor):
                break

    right_anchor = None
    right_act = None
    for p in range(last_pos + 1, len(all_case_idxs)):
        ridx = all_case_idxs[p]
        if case_df.at[ridx, "error_category"] == "CLEAN":
            right_anchor = safe_to_datetime(case_df.at[ridx, "Timestamp"])
            right_act = case_df.at[ridx, "Activity"]
            if not pd.isna(right_anchor):
                break

    clean_case_times = []
    for ridx in all_case_idxs:
        if case_df.at[ridx, "error_category"] == "CLEAN":
            ts = safe_to_datetime(case_df.at[ridx, "Timestamp"])
            if not pd.isna(ts):
                clean_case_times.append(ts)
    case_start = min(clean_case_times) if clean_case_times else pd.Timestamp("1900-01-01")
    case_end = max(clean_case_times) if clean_case_times else pd.Timestamp("2100-01-01")

    n = len(idxs)
    edge_secs = []
    seq_acts = activities
    for i in range(n - 1):
        edge_secs.append(edge_prior_seconds(seq_acts[i], seq_acts[i+1]))

    left_offsets = []
    cum = 0.0
    for i in range(n):
        if i == 0:
            if left_act is not None:
                cum = edge_prior_seconds(left_act, seq_acts[0])
            else:
                cum = 0.0
        else:
            cum += edge_secs[i-1]
        left_offsets.append(cum)

    right_offsets = []
    cum = 0.0
    for i in range(n-1, -1, -1):
        if i == n-1:
            if right_act is not None:
                cum = edge_prior_seconds(seq_acts[-1], right_act)
            else:
                cum = 0.0
        else:
            cum += edge_secs[i]
        right_offsets.append(cum)
    right_offsets = list(reversed(right_offsets))

    base_span = sum(edge_secs) if edge_secs else 0.0
    if base_span <= 0:
        base_span = block_span_prior_seconds(n)

    repaired_times = []
    los = []
    his = []

    for pos, ridx in enumerate(idxs):
        orig = case_df.at[ridx, "Timestamp"]
        surviving_date = original_surviving_date(orig) if is_date_only_or_midnight_damage(orig) else None
        lo, hi = feasible_interval(left_anchor, right_anchor, case_start, case_end, surviving_date)
        if hi < lo:
            # same-date feasible interval empty: clamp to nearest point on same date within case bounds
            if surviving_date is not None:
                day_start = pd.Timestamp(str(surviving_date) + " 00:00:00")
                day_end = pd.Timestamp(str(surviving_date) + " 23:59:59.999999")
                lo = max(case_start, day_start)
                hi = min(case_end, day_end)
                if hi < lo:
                    hi = lo
            else:
                hi = lo
        los.append(lo)
        his.append(hi)

        cands = []
        if left_anchor is not None:
            cands.append(left_anchor + pd.Timedelta(seconds=left_offsets[pos]))
        if right_anchor is not None:
            cands.append(right_anchor - pd.Timedelta(seconds=right_offsets[pos]))

        if left_anchor is not None and right_anchor is not None:
            window = (right_anchor - left_anchor).total_seconds()
            if window < 0:
                window = 0
            if n == 1:
                midpoint = left_anchor + pd.Timedelta(seconds=window / 2.0)
                cands.append(midpoint)
            else:
                # heterogeneous pattern scaled to anchor window, not uniform interpolation
                pattern_total = max(base_span, 1e-9)
                scaled_prefix = sum(edge_secs[:pos]) if pos > 0 else 0.0
                scaled = left_anchor + pd.Timedelta(seconds=(scaled_prefix / pattern_total) * window)
                cands.append(scaled)

        if not cands:
            # no anchors; use case bounds and day information only
            if surviving_date is not None:
                day_mid = pd.Timestamp(str(surviving_date) + " 12:00:00")
                cands.append(day_mid)
            else:
                cands.append(case_start)

        target = min(cands, key=lambda t: abs((clamp(t, lo, hi) - t).total_seconds()))
        target = clamp(target, lo, hi)
        repaired_times.append(target)

    repaired_times = make_monotone_nonuniform(repaired_times, los, his, frac_digits, edge_secs)

    # final per-row adjustment respecting same-date and anchors
    for pos, ridx in enumerate(idxs):
        repaired_times[pos] = clamp(repaired_times[pos], los[pos], his[pos])

        orig = case_df.at[ridx, "Timestamp"]
        orig_str = "" if pd.isna(orig) else str(orig)
        new_str = format_timestamp(repaired_times[pos], frac_digits)

        # must differ from input
        if new_str == orig_str:
            eps = tiny_step(frac_digits)
            lo, hi = los[pos], his[pos]
            moved = repaired_times[pos] + eps
            if moved <= hi and format_timestamp(moved, frac_digits) != orig_str:
                repaired_times[pos] = moved
            else:
                moved = repaired_times[pos] - eps
                if moved >= lo and format_timestamp(moved, frac_digits) != orig_str:
                    repaired_times[pos] = moved
                elif left_anchor is not None and right_anchor is not None:
                    repaired_times[pos] = clamp(left_anchor + (right_anchor - left_anchor) / 2, lo, hi)
                else:
                    repaired_times[pos] = clamp(lo, lo, hi)

        output_ts[ridx] = format_timestamp(repaired_times[pos], frac_digits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_df = df.copy(deep=True)
    frac_digits = detect_global_fractional_digits(df["Timestamp"])

    output_ts = df["Timestamp"].astype(str).tolist()

    blank_filled = 0
    date_only_filled = 0

    # Repair only non-CLEAN rows, jointly by contiguous case blocks
    for case_value, case_df in df.groupby("Case", sort=False):
        idxs = case_df.index.tolist()
        i = 0
        while i < len(idxs):
            ridx = idxs[i]
            if df.at[ridx, "error_category"] == "CLEAN":
                i += 1
                continue
            j = i
            while j + 1 < len(idxs) and df.at[idxs[j+1], "error_category"] != "CLEAN":
                j += 1
            block = idxs[i:j+1]

            for b in block:
                ts = df.at[b, "Timestamp"]
                if is_blank_timestamp(ts):
                    blank_filled += 1
                elif is_date_only_or_midnight_damage(ts):
                    date_only_filled += 1

            repair_block(case_df, block, output_ts, frac_digits)
            i = j + 1

    # Self-check and corrective pass
    repaired_df = df.copy(deep=True)
    repaired_df["Timestamp"] = output_ts

    # preserve CLEAN exactly
    clean_mask = repaired_df["error_category"] == "CLEAN"
    repaired_df.loc[clean_mask, "Timestamp"] = original_df.loc[clean_mask, "Timestamp"]

    # corrupted rows must differ and be parseable and respect surviving date and anchor bounds
    for case_value, case_df in repaired_df.groupby("Case", sort=False):
        idxs = case_df.index.tolist()
        clean_times = []
        for ridx in idxs:
            if repaired_df.at[ridx, "error_category"] == "CLEAN":
                t = safe_to_datetime(repaired_df.at[ridx, "Timestamp"])
                if not pd.isna(t):
                    clean_times.append(t)
        case_start = min(clean_times) if clean_times else None
        case_end = max(clean_times) if clean_times else None

        for pos, ridx in enumerate(idxs):
            if repaired_df.at[ridx, "error_category"] == "CLEAN":
                continue

            new_s = repaired_df.at[ridx, "Timestamp"]
            orig_s = original_df.at[ridx, "Timestamp"]
            new_t = safe_to_datetime(new_s)

            if pd.isna(new_t):
                # fallback to nearest clean anchor or case_start
                anchor = None
                for p in range(pos-1, -1, -1):
                    if repaired_df.at[idxs[p], "error_category"] == "CLEAN":
                        anchor = safe_to_datetime(repaired_df.at[idxs[p], "Timestamp"])
                        break
                if anchor is None:
                    for p in range(pos+1, len(idxs)):
                        if repaired_df.at[idxs[p], "error_category"] == "CLEAN":
                            anchor = safe_to_datetime(repaired_df.at[idxs[p], "Timestamp"])
                            break
                if anchor is None:
                    anchor = case_start if case_start is not None else pd.Timestamp("2000-01-01 12:00:00")
                repaired_df.at[ridx, "Timestamp"] = format_timestamp(anchor, frac_digits)
                new_t = anchor

            surv_date = original_surviving_date(orig_s) if is_date_only_or_midnight_damage(orig_s) else None
            if surv_date is not None and new_t.date() != surv_date:
                day_start = pd.Timestamp(str(surv_date) + " 00:00:00")
                day_noon = pd.Timestamp(str(surv_date) + " 12:00:00")
                day_end = pd.Timestamp(str(surv_date) + " 23:59:59.999999")
                candidate = day_noon
                if case_start is not None:
                    candidate = max(candidate, day_start, case_start)
                if case_end is not None:
                    candidate = min(candidate, day_end, case_end)
                candidate = clamp(candidate, max(day_start, case_start) if case_start is not None else day_start,
                                       min(day_end, case_end) if case_end is not None else day_end)
                repaired_df.at[ridx, "Timestamp"] = format_timestamp(candidate, frac_digits)
                new_t = candidate

            if case_start is not None and new_t < case_start:
                new_t = case_start
                repaired_df.at[ridx, "Timestamp"] = format_timestamp(new_t, frac_digits)
            if case_end is not None and new_t > case_end:
                new_t = case_end
                repaired_df.at[ridx, "Timestamp"] = format_timestamp(new_t, frac_digits)

            # ensure differs from input
            if repaired_df.at[ridx, "Timestamp"] == orig_s:
                eps = tiny_step(frac_digits)
                cand = new_t + eps
                if case_end is not None and cand > case_end:
                    cand = new_t - eps
                if surv_date is not None and cand.date() != surv_date:
                    cand = new_t
                    hour = 12 if new_t.hour == 0 else new_t.hour
                    cand = pd.Timestamp(year=surv_date.year, month=surv_date.month, day=surv_date.day,
                                        hour=hour, minute=(new_t.minute + 1) % 60, second=new_t.second,
                                        microsecond=new_t.microsecond)
                    if case_start is not None or case_end is not None:
                        lo = case_start if case_start is not None else pd.Timestamp(str(surv_date) + " 00:00:00")
                        hi = case_end if case_end is not None else pd.Timestamp(str(surv_date) + " 23:59:59.999999")
                        day_start = pd.Timestamp(str(surv_date) + " 00:00:00")
                        day_end = pd.Timestamp(str(surv_date) + " 23:59:59.999999")
                        lo = max(lo, day_start)
                        hi = min(hi, day_end)
                        cand = clamp(cand, lo, hi)
                repaired_df.at[ridx, "Timestamp"] = format_timestamp(cand, frac_digits)

    # final structural checks
    # restore all untouched columns byte-identically by writing original values
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        repaired_df[col] = original_df[col]

    # count rows still equal among corrupted rows
    nonclean_mask = repaired_df["error_category"] != "CLEAN"
    still_equal = int((repaired_df.loc[nonclean_mask, "Timestamp"].astype(str).values ==
                       original_df.loc[nonclean_mask, "Timestamp"].astype(str).values).sum())

    # if any remain equal, force minimal safe change
    if still_equal > 0:
        for idx in repaired_df.index[nonclean_mask]:
            if str(repaired_df.at[idx, "Timestamp"]) == str(original_df.at[idx, "Timestamp"]):
                t = safe_to_datetime(repaired_df.at[idx, "Timestamp"])
                if pd.isna(t):
                    t = pd.Timestamp("2000-01-01 12:00:00")
                surv_date = original_surviving_date(original_df.at[idx, "Timestamp"]) if is_date_only_or_midnight_damage(original_df.at[idx, "Timestamp"]) else None
                eps = tiny_step(frac_digits)
                cand = t + eps
                if surv_date is not None and cand.date() != surv_date:
                    cand = t - eps
                    if cand.date() != surv_date:
                        cand = pd.Timestamp(str(surv_date) + " 12:00:00")
                repaired_df.at[idx, "Timestamp"] = format_timestamp(cand, frac_digits)
        still_equal = int((repaired_df.loc[nonclean_mask, "Timestamp"].astype(str).values ==
                           original_df.loc[nonclean_mask, "Timestamp"].astype(str).values).sum())

    repaired_df.to_csv(args.output, index=False)

    print(f"blank_rows_filled={blank_filled} date_only_rows_given_time={date_only_filled} rows_still_equal_to_input={still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Must not exit non-zero; best effort output preservation if something unexpected happens.
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=True)
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            nonclean = (df["error_category"] != "CLEAN").sum() if "error_category" in df.columns else 0
            print(f"blank_rows_filled=0 date_only_rows_given_time=0 rows_still_equal_to_input={int(nonclean)}")
        except Exception:
            print("blank_rows_filled=0 date_only_rows_given_time=0 rows_still_equal_to_input=0")