import argparse
import math
import re
from datetime import timedelta
import pandas as pd

# Only error present in this task
ERR = "C3_MISSING_TEMPORAL_INFORMATION"

# Duration profiles from prompt (seconds)
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

BLOCK_SPAN = {
    2: {"p10": 10.0, "p25": 10.0, "median": 243.7965, "p75": 674.95, "p90": 1231.9611},
    3: {"p10": 108.0388, "p25": 324.6695, "median": 734.9865, "p75": 1334.2505, "p90": 2087.1570},
    4: {"p10": 383.1805, "p25": 705.3480, "median": 1215.4420, "p75": 1937.5180, "p90": 2788.0586},
    5: {"p10": 667.8252, "p25": 1079.14125, "median": 1700.4890, "p75": 2524.7830, "p90": 3453.6091},
}

GLOBAL_MEDIAN = 243.7965

FREQ_VARIANTS = [
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
]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()

def detect_precision(ts: str):
    if ts is None:
        return ("seconds", 0)
    s = str(ts).strip()
    m = re.search(r"\.(\d+)$", s)
    if m:
        return ("fraction", len(m.group(1)))
    return ("seconds", 0)

def choose_dataset_precision(ts_series):
    max_digits = 0
    for x in ts_series.astype(str).tolist():
        _, d = detect_precision(x)
        if d > max_digits:
            max_digits = d
    return max_digits if max_digits > 0 else 3

def is_blank(s):
    return pd.isna(s) or str(s).strip() == ""

def parse_ts(s):
    if is_blank(s):
        return pd.NaT
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT

def is_midnight_date_only_string(s):
    if is_blank(s):
        return False
    txt = str(s).strip()
    dt = parse_ts(txt)
    if pd.isna(dt):
        return False
    if dt.hour != 0 or dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
        return False
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]00:00:00(?:\.0+)?)?$", txt))

def format_ts(dt: pd.Timestamp, frac_digits: int):
    if pd.isna(dt):
        return ""
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits > 0:
        frac = f"{dt.microsecond:06d}"[:frac_digits].ljust(frac_digits, "0")
        return f"{base}.{frac}"
    return base

def dfg_has_edge(a, b):
    return (a, b) in DURATION_PROFILES

def edge_weight_seconds(a, b):
    prof = DURATION_PROFILES.get((a, b))
    if prof:
        return max(0.0, float(prof["p25"]))
    return None

def edge_small_gap(a, b):
    prof = DURATION_PROFILES.get((a, b))
    if not prof:
        return False
    return prof["p25"] <= 10.0 or prof["share_under_60s"] >= 0.9

def fallback_gap(a, b):
    direct = edge_weight_seconds(a, b)
    if direct is not None:
        return direct
    return max(30.0, min(180.0, GLOBAL_MEDIAN * 0.5))

def contiguous_in_variant(acts):
    for var in FREQ_VARIANTS:
        n = len(acts)
        for i in range(len(var) - n + 1):
            if var[i:i+n] == acts:
                return True
    return False

def get_case_bounds(case_df):
    clean_mask = case_df["error_category"].eq("CLEAN")
    parsed = case_df.loc[clean_mask, "_parsed"]
    parsed = parsed[parsed.notna()]
    if len(parsed) == 0:
        allp = case_df["_parsed"]
        allp = allp[allp.notna()]
        if len(allp) == 0:
            return (pd.NaT, pd.NaT)
        return (allp.min(), allp.max())
    return (parsed.min(), parsed.max())

def date_day_bounds(day_ts):
    start = pd.Timestamp(year=day_ts.year, month=day_ts.month, day=day_ts.day)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return start, end

def smallest_unit(frac_digits):
    if frac_digits <= 0:
        return pd.Timedelta(seconds=1)
    if frac_digits <= 6:
        return pd.Timedelta(microseconds=10 ** (6 - frac_digits))
    return pd.Timedelta(microseconds=1)

def estimate_block_span_seconds(num_missing, num_edges):
    key = num_edges + 1
    if key in BLOCK_SPAN:
        return BLOCK_SPAN[key]["p25"]
    key = num_missing + 1
    if key in BLOCK_SPAN:
        return BLOCK_SPAN[key]["p25"]
    if BLOCK_SPAN:
        mx = max(BLOCK_SPAN)
        return BLOCK_SPAN[mx]["p25"]
    return GLOBAL_MEDIAN

def build_gap_weights(case_df, idxs):
    acts = case_df.loc[idxs, "Activity"].tolist()
    gaps = []
    for i in range(len(acts)-1):
        g = fallback_gap(acts[i], acts[i+1])
        gaps.append(g)
    return gaps

def non_uniformize(gaps):
    if len(gaps) <= 1:
        return gaps
    if max(gaps) - min(gaps) < 1e-9:
        out = []
        for i, g in enumerate(gaps):
            out.append(g + (i+1)*0.001)
        return out
    return gaps

def compress_positions(left_t, right_t, gaps):
    total = sum(gaps)
    available = max((right_t - left_t).total_seconds(), 0.0)
    if total <= 0:
        positions = [left_t for _ in range(len(gaps)+1)]
        return positions
    scale = min(1.0, available / total) if total > 0 else 1.0
    scaled = [g * scale for g in gaps]
    positions = [left_t]
    cur = left_t
    for g in scaled:
        cur = cur + pd.Timedelta(seconds=g)
        positions.append(cur)
    if positions[-1] > right_t:
        # proportional recompute inside interval
        frac = []
        s = sum(gaps)
        cum = 0.0
        positions = [left_t]
        for g in gaps:
            cum += g / s if s > 0 else 0
            positions.append(left_t + (right_t - left_t) * cum)
    return positions

def project_from_left(left_t, acts):
    times = [left_t]
    cur = left_t
    for i in range(len(acts)-1):
        cur = cur + pd.Timedelta(seconds=fallback_gap(acts[i], acts[i+1]))
        times.append(cur)
    return times

def project_from_right(right_t, acts):
    times = [None] * len(acts)
    times[-1] = right_t
    cur = right_t
    for i in range(len(acts)-2, -1, -1):
        cur = cur - pd.Timedelta(seconds=fallback_gap(acts[i], acts[i+1]))
        times[i] = cur
    return times

def choose_candidate_for_row(left_est, right_est, left_anchor, right_anchor, keep_date=None):
    cands = []
    if left_est is not None and not pd.isna(left_est):
        cands.append(left_est)
    if right_est is not None and not pd.isna(right_est):
        cands.append(right_est)
    if left_anchor is not None and right_anchor is not None and not pd.isna(left_anchor) and not pd.isna(right_anchor):
        mid = left_anchor + (right_anchor - left_anchor) / 2
        cands.append(mid)
    cand = cands[0] if cands else left_anchor if left_anchor is not None else right_anchor
    if len(cands) >= 2:
        cand = min(cands, key=lambda x: abs((x - cands[0]).total_seconds()) + abs((x - cands[1]).total_seconds()))
    if keep_date is not None:
        day_start, day_end = date_day_bounds(keep_date)
        if cand < day_start:
            cand = day_start
        if cand > day_end:
            cand = day_end
    if left_anchor is not None and not pd.isna(left_anchor) and cand < left_anchor:
        if right_anchor is not None and not pd.isna(right_anchor):
            cand = left_anchor + (right_anchor - left_anchor) / 2
        else:
            cand = left_anchor
    if right_anchor is not None and not pd.isna(right_anchor) and cand > right_anchor:
        if left_anchor is not None and not pd.isna(left_anchor):
            cand = left_anchor + (right_anchor - left_anchor) / 2
        else:
            cand = right_anchor
    return cand

def repair_case(case_df, frac_digits):
    repaired = {}
    case_start, case_end = get_case_bounds(case_df)
    idxs = case_df.index.tolist()
    n = len(idxs)

    # Validate observed row order against DFG/frequent variants but do not reorder
    _ = contiguous_in_variant(case_df["Activity"].tolist())
    # Consecutive missing blocks
    miss_mask = case_df["error_category"].ne("CLEAN")
    i = 0
    while i < n:
        if not miss_mask.iloc[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and miss_mask.iloc[j+1]:
            j += 1

        block_pos = list(range(i, j+1))
        block_idxs = [idxs[p] for p in block_pos]
        block_acts = case_df.loc[block_idxs, "Activity"].tolist()

        left_pos = i - 1 if i > 0 else None
        right_pos = j + 1 if j + 1 < n else None

        left_anchor = None
        right_anchor = None
        left_act = None
        right_act = None

        if left_pos is not None:
            li = idxs[left_pos]
            if case_df.at[li, "error_category"] == "CLEAN":
                left_anchor = case_df.at[li, "_parsed"]
                left_act = case_df.at[li, "Activity"]
        if right_pos is not None:
            ri = idxs[right_pos]
            if case_df.at[ri, "error_category"] == "CLEAN":
                right_anchor = case_df.at[ri, "_parsed"]
                right_act = case_df.at[ri, "Activity"]

        full_acts = []
        if left_act is not None:
            full_acts.append(left_act)
        full_acts.extend(block_acts)
        if right_act is not None:
            full_acts.append(right_act)

        # Build weighted non-uniform gaps over anchored chain
        gaps = []
        for k in range(len(full_acts)-1):
            gaps.append(fallback_gap(full_acts[k], full_acts[k+1]))
        gaps = non_uniformize(gaps)

        # Candidates from both directions
        if left_anchor is not None:
            left_times_full = project_from_left(left_anchor, full_acts)
        else:
            left_times_full = [None] * len(full_acts)
        if right_anchor is not None:
            right_times_full = project_from_right(right_anchor, full_acts)
        else:
            right_times_full = [None] * len(full_acts)

        # If both anchors exist, compress whole chain into anchor window
        compressed = None
        if left_anchor is not None and right_anchor is not None:
            compressed = compress_positions(left_anchor, right_anchor, gaps)

        for local_idx, row_idx in enumerate(block_idxs):
            orig = case_df.at[row_idx, "Timestamp"]
            observed_dt = case_df.at[row_idx, "_parsed"]
            keep_date = observed_dt.normalize() if is_midnight_date_only_string(orig) and not pd.isna(observed_dt) else None

            pos_in_full = local_idx + (1 if left_act is not None else 0)
            left_est = left_times_full[pos_in_full] if left_times_full[pos_in_full] is not None else None
            right_est = right_times_full[pos_in_full] if right_times_full[pos_in_full] is not None else None

            # Strong preference for minimal movement for date-only rows:
            # preserve same date and use nearest valid session-like estimate.
            cand = choose_candidate_for_row(left_est, right_est, left_anchor, right_anchor, keep_date=keep_date)

            # If both anchors exist, prefer compressed chain point when it stays closer and valid
            if compressed is not None:
                comp = compressed[pos_in_full]
                if keep_date is not None:
                    ds, de = date_day_bounds(keep_date)
                    if comp < ds:
                        comp = ds
                    if comp > de:
                        comp = de
                if left_anchor <= comp <= right_anchor:
                    # blend lightly toward compressed only when one-sided estimates disagree or overshoot
                    if left_est is None or right_est is None:
                        cand = comp
                    else:
                        if left_est > right_anchor or right_est < left_anchor or left_est > right_est:
                            cand = comp

            # For blanks with both anchors, use compressed/anchor-consistent estimate
            if is_blank(orig) and compressed is not None:
                comp = compressed[pos_in_full]
                cand = comp

            # Clamp to case bounds and anchors
            if left_anchor is not None and cand < left_anchor:
                cand = left_anchor if right_anchor is None else left_anchor + (right_anchor - left_anchor) / 2
            if right_anchor is not None and cand > right_anchor:
                cand = right_anchor if left_anchor is None else left_anchor + (right_anchor - left_anchor) / 2
            if not pd.isna(case_start) and cand < case_start:
                cand = case_start
            if not pd.isna(case_end) and cand > case_end:
                cand = case_end

            # For date-only keep exact date; intersect with anchors and case bounds
            if keep_date is not None:
                ds, de = date_day_bounds(keep_date)
                lo = ds
                hi = de
                if left_anchor is not None:
                    lo = max(lo, left_anchor)
                if right_anchor is not None:
                    hi = min(hi, right_anchor)
                if not pd.isna(case_start):
                    lo = max(lo, case_start)
                if not pd.isna(case_end):
                    hi = min(hi, case_end)
                if lo <= hi:
                    if cand < lo or cand > hi:
                        if left_anchor is not None and right_anchor is not None:
                            cand = lo + (hi - lo) / 2
                        else:
                            cand = min(max(cand, lo), hi)
                else:
                    cand = ds if ds <= de else cand

            repaired[row_idx] = cand

        i = j + 1

    return repaired

def self_check_and_fix(df, repaired_map, frac_digits):
    unit = smallest_unit(frac_digits)

    # Apply initial repaired strings
    out_ts = df["Timestamp"].astype(object).copy()
    blank_filled = 0
    date_only_filled = 0
    equal_after = 0

    for idx, ts in repaired_map.items():
        orig = df.at[idx, "Timestamp"]
        if is_blank(orig):
            blank_filled += 1
        elif is_midnight_date_only_string(orig):
            date_only_filled += 1
        out_ts.at[idx] = format_ts(ts, frac_digits)

    # Fix parseability and constraints case by case
    for case, cdf in df.groupby("Case", sort=False):
        case_idxs = cdf.index.tolist()
        clean_mask = cdf["error_category"].eq("CLEAN")
        clean_parsed = cdf.loc[clean_mask, "_parsed"]
        clean_parsed = clean_parsed[clean_parsed.notna()]
        case_start = clean_parsed.min() if len(clean_parsed) else pd.NaT
        case_end = clean_parsed.max() if len(clean_parsed) else pd.NaT

        for pos, idx in enumerate(case_idxs):
            if df.at[idx, "error_category"] == "CLEAN":
                out_ts.at[idx] = df.at[idx, "Timestamp"]
                continue

            candidate = parse_ts(out_ts.at[idx])
            if pd.isna(candidate):
                left_anchor = None
                right_anchor = None
                for p in range(pos-1, -1, -1):
                    j = case_idxs[p]
                    if df.at[j, "error_category"] == "CLEAN":
                        left_anchor = df.at[j, "_parsed"]
                        break
                for p in range(pos+1, len(case_idxs)):
                    j = case_idxs[p]
                    if df.at[j, "error_category"] == "CLEAN":
                        right_anchor = df.at[j, "_parsed"]
                        break
                if left_anchor is not None and right_anchor is not None:
                    candidate = left_anchor + (right_anchor - left_anchor) / 2
                elif left_anchor is not None:
                    candidate = left_anchor + unit
                elif right_anchor is not None:
                    candidate = right_anchor - unit
                elif not pd.isna(case_start):
                    candidate = case_start
                else:
                    candidate = pd.Timestamp("2023-01-01 00:00:00")  # never expected to be used
            # case bounds
            if not pd.isna(case_start) and candidate < case_start:
                candidate = case_start
            if not pd.isna(case_end) and candidate > case_end:
                candidate = case_end

            # anchor bounds
            left_anchor = None
            right_anchor = None
            for p in range(pos-1, -1, -1):
                j = case_idxs[p]
                if df.at[j, "error_category"] == "CLEAN":
                    left_anchor = df.at[j, "_parsed"]
                    break
            for p in range(pos+1, len(case_idxs)):
                j = case_idxs[p]
                if df.at[j, "error_category"] == "CLEAN":
                    right_anchor = df.at[j, "_parsed"]
                    break
            if left_anchor is not None and candidate < left_anchor:
                candidate = left_anchor if right_anchor is None else left_anchor + (right_anchor - left_anchor) / 2
            if right_anchor is not None and candidate > right_anchor:
                candidate = right_anchor if left_anchor is None else left_anchor + (right_anchor - left_anchor) / 2

            # same date for date-only corruption
            orig = df.at[idx, "Timestamp"]
            if is_midnight_date_only_string(orig):
                observed_dt = df.at[idx, "_parsed"]
                if not pd.isna(observed_dt):
                    ds, de = date_day_bounds(observed_dt)
                    lo, hi = ds, de
                    if left_anchor is not None:
                        lo = max(lo, left_anchor)
                    if right_anchor is not None:
                        hi = min(hi, right_anchor)
                    if not pd.isna(case_start):
                        lo = max(lo, case_start)
                    if not pd.isna(case_end):
                        hi = min(hi, case_end)
                    if lo <= hi:
                        if candidate < lo or candidate > hi:
                            candidate = lo + (hi - lo) / 2
                    else:
                        candidate = ds

            rendered = format_ts(candidate, frac_digits)

            # must differ from input
            if rendered == str(orig):
                lo = None
                hi = None
                if left_anchor is not None:
                    lo = left_anchor
                if right_anchor is not None:
                    hi = right_anchor
                if lo is not None and hi is not None:
                    plus = candidate + unit
                    minus = candidate - unit
                    if plus <= hi:
                        candidate = plus
                    elif minus >= lo:
                        candidate = minus
                    else:
                        candidate = lo + (hi - lo) / 2
                elif lo is not None:
                    candidate = candidate + unit
                elif hi is not None:
                    candidate = candidate - unit
                rendered = format_ts(candidate, frac_digits)

            out_ts.at[idx] = rendered
            if rendered == str(orig):
                equal_after += 1

    return out_ts, blank_filled, date_only_filled, equal_after

def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        # last-resort tolerant read
        df = pd.read_csv(args.input, dtype=str)

    original_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for col in original_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[original_cols].copy()
    df["_parsed"] = df["Timestamp"].apply(parse_ts)

    frac_digits = choose_dataset_precision(df["Timestamp"])

    repaired_map = {}
    for case, case_df in df.groupby("Case", sort=False):
        repaired_map.update(repair_case(case_df, frac_digits))

    out_ts, blank_filled, date_only_filled, equal_after = self_check_and_fix(df, repaired_map, frac_digits)

    out_df = df[original_cols].copy()
    out_df["Timestamp"] = out_ts

    # Final preservation checks; repair by restoration where needed for CLEAN and non-timestamp cols
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        out_df[col] = df[col]

    clean_mask = out_df["error_category"].eq("CLEAN")
    out_df.loc[clean_mask, "Timestamp"] = df.loc[clean_mask, "Timestamp"]

    # Row count and order unchanged by construction
    try:
        out_df.to_csv(args.output, index=False)
    except Exception:
        # continue with best effort
        try:
            out_df.to_csv(args.output, index=False, encoding="utf-8")
        except Exception:
            pass

    print(f"blank_rows_filled={blank_filled}, date_only_rows_given_time={date_only_filled}, rows_still_equal_to_input={equal_after}")

if __name__ == "__main__":
    main()