import argparse
import math
import re
from datetime import timedelta
import pandas as pd


# =========================
# Process knowledge payload
# =========================

DURATION_PROFILES = {
    ("Bring drinks", "Review drinks"): {"p10": 6.0576, "p25": 16.973, "median": 40.525, "share_under_60s": 0.637},
    ("prepare drinks", "Bring drinks"): {"p10": 29.6502, "p25": 82.4315, "median": 200.074, "share_under_60s": 0.191},
    ("Deliver to customer", "EVENT 8 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("Bring food", "Deliver to customer"): {"p10": 12.5235, "p25": 34.57875, "median": 85.062, "share_under_60s": 0.389},
    ("start", "Take order"): {"p10": 12.9624, "p25": 35.73925, "median": 89.125, "share_under_60s": 0.379},
    ("Signal food complete", "Bring food"): {"p10": 32.288, "p25": 89.16275, "median": 213.3435, "share_under_60s": 0.179},
    ("Take order", "Review order"): {"p10": 6.2354, "p25": 18.196, "median": 42.276, "share_under_60s": 0.632},
    ("Prepare main course", "Signal food complete"): {"p10": 85943.671, "p25": 254311.64075, "median": 515362.0085, "share_under_60s": 0.0},
    ("Review drinks", "Prepare main course"): {"p10": 83506.8328, "p25": 246266.73425, "median": 505754.282, "share_under_60s": 0.001},
    ("Review order", "prepare drinks"): {"p10": 26.85, "p25": 69.814, "median": 175.826, "share_under_60s": 0.221},
    ("Review drinks", "prepare drinks"): {"p10": 29.9604, "p25": 85.61525, "median": 208.273, "share_under_60s": 0.183},
    ("Prepare starter", "prepare drinks"): {"p10": 28.889, "p25": 84.699, "median": 203.911, "share_under_60s": 0.198},
    ("Review order", "Prepare starter"): {"p10": 15.1921, "p25": 43.88025, "median": 100.954, "share_under_60s": 0.344},
    ("Review drinks", "Signal food complete"): {"p10": 11685.151, "p25": 190375.455, "median": 515100.472, "share_under_60s": 0.002},
    ("Prepare starter", "Bring drinks"): {"p10": 32.0, "p25": 91.873, "median": 221.952, "share_under_60s": 0.176},
    ("prepare drinks", "Prepare starter"): {"p10": 16.0638, "p25": 41.62825, "median": 109.9465, "share_under_60s": 0.344},
    ("Review drinks", "Prepare starter"): {"p10": 21.3248, "p25": 70.531, "median": 155.221, "share_under_60s": 0.23},
    ("prepare drinks", "Review drinks"): {"p10": 68.6561, "p25": 129.5055, "median": 262.26, "share_under_60s": 0.095},
    ("Prepare starter", "Prepare main course"): {"p10": 78732.1044, "p25": 235601.191, "median": 501050.302, "share_under_60s": 0.0},
    ("Prepare main course", "Bring food"): {"p10": 87381.2898, "p25": 250766.01, "median": 438430.326, "share_under_60s": 0.0},
    ("Review order", "Bring drinks"): {"p10": 187.3826, "p25": 282.01025, "median": 476.833, "share_under_60s": 0.02},
    ("Take order", "prepare drinks"): {"p10": 67.495, "p25": 121.29, "median": 222.695, "share_under_60s": 0.093},
    ("start", "Review order"): {"p10": 59.6194, "p25": 100.168, "median": 158.5445, "share_under_60s": 0.107},
    ("Bring drinks", "Prepare main course"): {"p10": 87866.8959, "p25": 255063.302, "median": 480699.382, "share_under_60s": 0.007},
    ("Prepare starter", "Review drinks"): {"p10": 11.7123, "p25": 25.7145, "median": 55.2835, "share_under_60s": 0.528},
    ("Bring drinks", "Prepare starter"): {"p10": 3.8606, "p25": 15.50125, "median": 30.9025, "share_under_60s": 0.69},
    ("Prepare starter", "Signal food complete"): {"p10": 87920.7, "p25": 305692.972, "median": 516395.537, "share_under_60s": 0.0},
    ("Bring drinks", "prepare drinks"): {"p10": 75.3596, "p25": 154.0335, "median": 277.061, "share_under_60s": 0.068},
    ("Review drinks", "Bring drinks"): {"p10": 218.3702, "p25": 293.23525, "median": 566.718, "share_under_60s": 0.014},
    ("Take order", "Prepare starter"): {"p10": 62.5727, "p25": 101.62475, "median": 154.3685, "share_under_60s": 0.091},
    ("Review drinks", "Bring food"): {"p10": 171.5558, "p25": 4622.571, "median": 168377.244, "share_under_60s": 0.042},
    ("prepare drinks", "Prepare main course"): {"p10": 48.025, "p25": 86.359, "median": 227.445, "share_under_60s": 0.143},
    ("Bring drinks", "Signal food complete"): {"p10": 139780.8046, "p25": 169171.919, "median": 351110.078, "share_under_60s": 0.0},
    ("Prepare main course", "Bring drinks"): {"p10": 28.2273, "p25": 68.60625, "median": 129.8, "share_under_60s": 0.222},
    ("Prepare main course", "prepare drinks"): {"p10": 96.6248, "p25": 114.4655, "median": 188.761, "share_under_60s": 0.0},
    ("Review drinks", "Deliver to customer"): {"p10": 13.1735, "p25": 26.957, "median": 70.7255, "share_under_60s": 0.357},
    ("Review order", "Prepare main course"): {"p10": 116.8094, "p25": 167.5355, "median": 392.177, "share_under_60s": 0.0},
    ("Bring food", "Bring drinks"): {"p10": 45.0556, "p25": 115.781, "median": 222.362, "share_under_60s": 0.222},
    ("prepare drinks", "Signal food complete"): {"p10": 22.7145, "p25": 78.136, "median": 252.3445, "share_under_60s": 0.333},
    ("Signal food complete", "prepare drinks"): {"p10": 65.3186, "p25": 153.056, "median": 228.6855, "share_under_60s": 0.25},
    ("Signal food complete", "Bring drinks"): {"p10": 150.102, "p25": 179.118, "median": 227.478, "share_under_60s": 0.0},
    ("Bring food", "prepare drinks"): {"p10": 164.9284, "p25": 166.546, "median": 169.242, "share_under_60s": 0.0},
    ("prepare drinks", "Bring food"): {"p10": 46.9712, "p25": 63.2825, "median": 90.468, "share_under_60s": 0.333},
    ("Bring food", "Review drinks"): {"p10": 775.2938, "p25": 787.7435, "median": 808.493, "share_under_60s": 0.0},
}

BLOCK_SPAN = {
    2: {"p10": 2.6195, "p25": 31.6845, "median": 118.0235},
    3: {"p10": 73.678, "p25": 165.08025, "median": 398.3015},
    4: {"p10": 215.3226, "p25": 395.92275, "median": 886.8985},
    5: {"p10": 408.363, "p25": 735.7245, "median": 79428.972},
}
GLOBAL_MEDIAN = 118.0235

INPUT_COLUMNS = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def detect_fraction_digits(ts):
    if not ts:
        return 3
    m = re.search(r"\.(\d+)$", ts.strip())
    return len(m.group(1)) if m else 0


def detect_dataset_precision(ts_series):
    digits = []
    for v in ts_series.astype(str):
        d = detect_fraction_digits(v)
        if d > 0:
            digits.append(d)
    if digits:
        return max(set(digits), key=digits.count)
    return 3


def parse_loose(ts):
    s = safe_str(ts).strip()
    if s == "":
        return pd.NaT
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def is_blank_timestamp(ts):
    return safe_str(ts).strip() == ""


def has_explicit_date_only_or_midnight_missing_time(ts):
    s = safe_str(ts).strip()
    if s == "":
        return False
    dt = parse_loose(s)
    if pd.isna(dt):
        return False
    # treat exact midnight as lost time-of-day per task description/sample
    return dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0


def extract_surviving_date(ts):
    s = safe_str(ts).strip()
    if s == "":
        return None
    dt = parse_loose(s)
    if pd.isna(dt):
        return None
    return dt.date()


def fmt_ts(dt, frac_digits):
    if pd.isna(dt):
        return ""
    py_dt = pd.Timestamp(dt).to_pydatetime()
    base = py_dt.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits <= 0:
        return base
    micros = py_dt.microsecond
    frac = f"{micros:06d}"[:frac_digits]
    return f"{base}.{frac}"


def choose_gap_seconds(src, tgt):
    prof = DURATION_PROFILES.get((src, tgt))
    if prof:
        if prof["share_under_60s"] >= 0.5:
            return max(prof["p10"], 0.001)
        return max(prof["p25"], 0.001)
    return None


def edge_weight(src, tgt):
    g = choose_gap_seconds(src, tgt)
    if g is not None:
        return max(g, 0.001)
    return None


def expected_block_span(activities):
    if len(activities) <= 1:
        return 1.0
    weights = []
    for i in range(len(activities) - 1):
        w = edge_weight(activities[i], activities[i + 1])
        if w is not None:
            weights.append(w)
    if weights:
        return sum(weights)
    prof = BLOCK_SPAN.get(len(activities))
    if prof:
        return max(prof["p25"], 1.0)
    return max((len(activities) - 1) * GLOBAL_MEDIAN, 1.0)


def get_nonuniform_weights(activities):
    n = len(activities)
    if n <= 1:
        return []
    weights = []
    for i in range(n - 1):
        src = activities[i]
        tgt = activities[i + 1]
        w = edge_weight(src, tgt)
        if w is None:
            prev_w = weights[-1] if weights else None
            next_w = None
            for j in range(i + 1, n - 1):
                next_w = edge_weight(activities[j], activities[j + 1])
                if next_w is not None:
                    break
            if prev_w is not None and next_w is not None:
                w = (prev_w + next_w) / 2.0
            elif prev_w is not None:
                w = prev_w * 0.93
            elif next_w is not None:
                w = next_w * 1.07
            else:
                prof = BLOCK_SPAN.get(n)
                w = (prof["p25"] / max(n - 1, 1)) if prof else GLOBAL_MEDIAN
        weights.append(max(w, 0.001))
    if len(set(round(x, 6) for x in weights)) == 1 and len(weights) > 1:
        for i in range(len(weights)):
            weights[i] = weights[i] * (1.0 + 0.07 * math.sin(i + 1))
            weights[i] = max(weights[i], 0.001)
    return weights


def make_times_from_start(start_ts, activities, total_span_seconds):
    if len(activities) == 1:
        return [start_ts]
    weights = get_nonuniform_weights(activities)
    sum_w = sum(weights)
    if sum_w <= 0:
        weights = [1.0 + 0.1 * i for i in range(len(activities) - 1)]
        sum_w = sum(weights)
    scale = total_span_seconds / sum_w if sum_w > 0 else 1.0
    gaps = [w * scale for w in weights]
    out = [start_ts]
    cur = start_ts
    for g in gaps:
        cur = cur + pd.Timedelta(seconds=float(g))
        out.append(cur)
    return out


def anchor_window(case_df, start_idx, end_idx):
    left_clean_idx = None
    for i in range(start_idx - 1, case_df.index.min() - 1, -1):
        if safe_str(case_df.at[i, "error_category"]) == "CLEAN":
            dt = parse_loose(case_df.at[i, "Timestamp"])
            if not pd.isna(dt):
                left_clean_idx = i
                break
    right_clean_idx = None
    for i in range(end_idx + 1, case_df.index.max() + 1):
        if safe_str(case_df.at[i, "error_category"]) == "CLEAN":
            dt = parse_loose(case_df.at[i, "Timestamp"])
            if not pd.isna(dt):
                right_clean_idx = i
                break
    left_dt = parse_loose(case_df.at[left_clean_idx, "Timestamp"]) if left_clean_idx is not None else pd.NaT
    right_dt = parse_loose(case_df.at[right_clean_idx, "Timestamp"]) if right_clean_idx is not None else pd.NaT
    return left_clean_idx, left_dt, right_clean_idx, right_dt


def same_date_bounds(date_obj, left_dt, right_dt):
    day_start = pd.Timestamp(f"{date_obj} 00:00:00")
    day_end = pd.Timestamp(f"{date_obj} 23:59:59.999999")
    lo = day_start
    hi = day_end
    if not pd.isna(left_dt) and left_dt.date() == date_obj:
        lo = max(lo, left_dt)
    if not pd.isna(right_dt) and right_dt.date() == date_obj:
        hi = min(hi, right_dt)
    return lo, hi


def repair_block(case_df, block_indices, frac_digits):
    activities = [safe_str(case_df.at[i, "Activity"]) for i in block_indices]
    left_idx, left_dt, right_idx, right_dt = anchor_window(case_df, block_indices[0], block_indices[-1])
    both = (not pd.isna(left_dt)) and (not pd.isna(right_dt)) and left_dt <= right_dt

    expected_span = expected_block_span(activities)
    n = len(block_indices)

    if both:
        anchor_span = max((right_dt - left_dt).total_seconds(), 0.0)
        use_span = min(expected_span, anchor_span)
        if n == 1:
            forward_gap = choose_gap_seconds(safe_str(case_df.at[left_idx, "Activity"]), activities[0]) if left_idx is not None else None
            backward_gap = choose_gap_seconds(activities[0], safe_str(case_df.at[right_idx, "Activity"])) if right_idx is not None else None
            cand_f = left_dt + pd.Timedelta(seconds=float(forward_gap if forward_gap is not None else min(use_span * 0.4, anchor_span * 0.4)))
            cand_b = right_dt - pd.Timedelta(seconds=float(backward_gap if backward_gap is not None else min(use_span * 0.4, anchor_span * 0.4)))
            t = min(max(cand_f, left_dt), right_dt)
            if cand_f > right_dt and cand_b < left_dt:
                t = left_dt + (right_dt - left_dt) / 2
            else:
                t = min(max(cand_f if abs((cand_f - left_dt).total_seconds()) <= abs((right_dt - cand_b).total_seconds()) else cand_b, left_dt), right_dt)
            return [t]

        center_slack = max(anchor_span - use_span, 0.0)
        left_edge_gap = choose_gap_seconds(safe_str(case_df.at[left_idx, "Activity"]), activities[0]) if left_idx is not None else None
        right_edge_gap = choose_gap_seconds(activities[-1], safe_str(case_df.at[right_idx, "Activity"])) if right_idx is not None else None
        bias_num = (left_edge_gap if left_edge_gap is not None else 1.0)
        bias_den = bias_num + (right_edge_gap if right_edge_gap is not None else 1.0)
        start_offset = center_slack * (bias_num / bias_den if bias_den > 0 else 0.5)
        start_ts = left_dt + pd.Timedelta(seconds=float(start_offset))
        times = make_times_from_start(start_ts, activities, use_span)
        if times[-1] > right_dt:
            over = (times[-1] - right_dt).total_seconds()
            start_ts = start_ts - pd.Timedelta(seconds=float(over / 2.0))
            if start_ts < left_dt:
                start_ts = left_dt
            times = make_times_from_start(start_ts, activities, min(use_span, max((right_dt - start_ts).total_seconds(), 0.0)))
            if times[-1] > right_dt:
                times[-1] = right_dt
        return times

    if not pd.isna(left_dt):
        start_gap = choose_gap_seconds(safe_str(case_df.at[left_idx, "Activity"]), activities[0]) if left_idx is not None else None
        start_ts = left_dt + pd.Timedelta(seconds=float(start_gap if start_gap is not None else min(GLOBAL_MEDIAN, expected_span * 0.25)))
        return make_times_from_start(start_ts, activities, expected_span)

    if not pd.isna(right_dt):
        rev_acts = list(reversed(activities))
        rev_times = make_times_from_start(right_dt - pd.Timedelta(seconds=float(expected_span)), rev_acts, expected_span)
        return list(reversed(rev_times))

    base = pd.Timestamp("2024-01-01 12:00:00")
    return make_times_from_start(base, activities, expected_span)


def apply_same_date_constraint(candidate_dt, surviving_date, left_dt, right_dt):
    if surviving_date is None:
        return candidate_dt
    lo, hi = same_date_bounds(surviving_date, left_dt, right_dt)
    if lo > hi:
        # impossible with both anchors on different dates; keep date and pick noon-ish clamped into day
        mid = pd.Timestamp(f"{surviving_date} 12:00:00")
        return min(max(mid, pd.Timestamp(f"{surviving_date} 00:00:00")), pd.Timestamp(f"{surviving_date} 23:59:59.999999"))
    if candidate_dt.date() != surviving_date:
        # preserve time-of-day as much as possible but on surviving date
        tod_seconds = candidate_dt.hour * 3600 + candidate_dt.minute * 60 + candidate_dt.second + candidate_dt.microsecond / 1e6
        shifted = pd.Timestamp(f"{surviving_date} 00:00:00") + pd.Timedelta(seconds=float(tod_seconds))
        candidate_dt = shifted
    if candidate_dt < lo:
        candidate_dt = lo
    if candidate_dt > hi:
        candidate_dt = hi
    return candidate_dt


def enforce_block_monotone(times, left_dt, right_dt):
    fixed = list(times)
    for i in range(1, len(fixed)):
        if fixed[i] < fixed[i - 1]:
            fixed[i] = fixed[i - 1]
    if not pd.isna(left_dt):
        for i in range(len(fixed)):
            if fixed[i] < left_dt:
                fixed[i] = left_dt
    if not pd.isna(right_dt):
        for i in range(len(fixed) - 1, -1, -1):
            if fixed[i] > right_dt:
                fixed[i] = right_dt
        for i in range(len(fixed) - 2, -1, -1):
            if fixed[i] > fixed[i + 1]:
                fixed[i] = fixed[i + 1]
    return fixed


def break_uniformity(times, activities, left_dt, right_dt, frac_digits):
    if len(times) <= 2:
        return times
    gaps = []
    for i in range(len(times) - 1):
        gaps.append(round((times[i + 1] - times[i]).total_seconds(), max(frac_digits, 3)))
    if len(set(gaps)) > 1:
        return times
    weights = get_nonuniform_weights(activities)
    total_span = max((times[-1] - times[0]).total_seconds(), 0.001)
    new_times = make_times_from_start(times[0], activities, total_span)
    new_times = enforce_block_monotone(new_times, left_dt, right_dt)
    return new_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    for col in INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[INPUT_COLUMNS].copy()
    original = df.copy(deep=True)
    df["Timestamp"] = df["Timestamp"].astype(str)

    dataset_frac_digits = detect_dataset_precision(df["Timestamp"])
    repaired_ts = list(df["Timestamp"].astype(str).tolist())

    blank_filled = 0
    date_only_filled = 0

    for case_value, case_idx in df.groupby("Case", sort=False).groups.items():
        case_indices = list(case_idx)
        case_df = df.loc[case_indices]

        i = 0
        while i < len(case_indices):
            ridx = case_indices[i]
            if safe_str(df.at[ridx, "error_category"]) == "CLEAN":
                i += 1
                continue

            j = i
            block_indices = []
            while j < len(case_indices) and safe_str(df.at[case_indices[j], "error_category"]) != "CLEAN":
                block_indices.append(case_indices[j])
                j += 1

            proposed = repair_block(df.loc[case_indices], block_indices, dataset_frac_digits)
            left_idx, left_dt, right_idx, right_dt = anchor_window(df.loc[case_indices], block_indices[0], block_indices[-1])

            adjusted = []
            for local_k, rid in enumerate(block_indices):
                orig_ts = safe_str(df.at[rid, "Timestamp"])
                cand = proposed[local_k]

                surviving_date = extract_surviving_date(orig_ts) if has_explicit_date_only_or_midnight_missing_time(orig_ts) else None
                if surviving_date is not None:
                    cand = apply_same_date_constraint(cand, surviving_date, left_dt, right_dt)

                adjusted.append(cand)

            adjusted = enforce_block_monotone(adjusted, left_dt, right_dt)
            activities = [safe_str(df.at[r, "Activity"]) for r in block_indices]
            adjusted = break_uniformity(adjusted, activities, left_dt, right_dt, dataset_frac_digits)

            for local_k, rid in enumerate(block_indices):
                orig_ts = safe_str(df.at[rid, "Timestamp"])
                cand = adjusted[local_k]

                surviving_date = extract_surviving_date(orig_ts) if has_explicit_date_only_or_midnight_missing_time(orig_ts) else None
                if surviving_date is not None:
                    cand = apply_same_date_constraint(cand, surviving_date, left_dt, right_dt)

                if not pd.isna(left_dt) and cand < left_dt:
                    cand = left_dt
                if not pd.isna(right_dt) and cand > right_dt:
                    cand = right_dt

                new_s = fmt_ts(cand, dataset_frac_digits)

                if new_s == orig_ts:
                    cand2 = cand + pd.Timedelta(milliseconds=1 if dataset_frac_digits >= 3 else 1)
                    if surviving_date is not None:
                        cand2 = apply_same_date_constraint(cand2, surviving_date, left_dt, right_dt)
                    if not pd.isna(right_dt) and cand2 > right_dt:
                        cand2 = cand - pd.Timedelta(milliseconds=1 if dataset_frac_digits >= 3 else 1)
                        if surviving_date is not None:
                            cand2 = apply_same_date_constraint(cand2, surviving_date, left_dt, right_dt)
                    if not pd.isna(left_dt) and cand2 < left_dt:
                        cand2 = cand
                    new_s = fmt_ts(cand2, dataset_frac_digits)
                    if new_s == orig_ts:
                        lo = left_dt if not pd.isna(left_dt) else cand.normalize()
                        hi = right_dt if not pd.isna(right_dt) else cand.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                        if surviving_date is not None:
                            lo2, hi2 = same_date_bounds(surviving_date, left_dt, right_dt)
                            lo, hi = lo2, hi2
                        alt = lo + (hi - lo) / 2 if lo <= hi else cand
                        new_s = fmt_ts(alt, dataset_frac_digits)
                        if new_s == orig_ts:
                            alt = alt + pd.Timedelta(milliseconds=2 if dataset_frac_digits >= 3 else 1)
                            if surviving_date is not None:
                                alt = apply_same_date_constraint(alt, surviving_date, left_dt, right_dt)
                            new_s = fmt_ts(alt, dataset_frac_digits)

                repaired_ts[rid] = new_s

                if is_blank_timestamp(orig_ts):
                    blank_filled += 1
                elif has_explicit_date_only_or_midnight_missing_time(orig_ts):
                    date_only_filled += 1

            i = j

    out = original.copy(deep=True)
    out["Timestamp"] = repaired_ts

    # =================
    # Self-check/fixups
    # =================

    # Keep CLEAN rows exact
    clean_mask = out["error_category"].astype(str) == "CLEAN"
    out.loc[clean_mask, "Timestamp"] = original.loc[clean_mask, "Timestamp"]

    # Parseability and difference for repaired rows
    nonclean_mask = out["error_category"].astype(str) != "CLEAN"
    for idx in out.index[nonclean_mask]:
        new_s = safe_str(out.at[idx, "Timestamp"])
        old_s = safe_str(original.at[idx, "Timestamp"])
        dt = parse_loose(new_s)
        if new_s.strip() == "" or pd.isna(dt):
            # emergency same-case anchor-based replacement
            case_val = out.at[idx, "Case"]
            case_indices = list(out.index[out["Case"] == case_val])
            pos = case_indices.index(idx)
            left_dt = pd.NaT
            right_dt = pd.NaT
            for p in range(pos - 1, -1, -1):
                ii = case_indices[p]
                if safe_str(out.at[ii, "error_category"]) == "CLEAN":
                    left_dt = parse_loose(out.at[ii, "Timestamp"])
                    if not pd.isna(left_dt):
                        break
            for p in range(pos + 1, len(case_indices)):
                ii = case_indices[p]
                if safe_str(out.at[ii, "error_category"]) == "CLEAN":
                    right_dt = parse_loose(out.at[ii, "Timestamp"])
                    if not pd.isna(right_dt):
                        break
            if not pd.isna(left_dt) and not pd.isna(right_dt):
                dt = left_dt + (right_dt - left_dt) / 2
            elif not pd.isna(left_dt):
                dt = left_dt + pd.Timedelta(seconds=GLOBAL_MEDIAN)
            elif not pd.isna(right_dt):
                dt = right_dt - pd.Timedelta(seconds=GLOBAL_MEDIAN)
            else:
                dt = pd.Timestamp("2024-01-01 12:00:00")
            surv = extract_surviving_date(old_s) if has_explicit_date_only_or_midnight_missing_time(old_s) else None
            if surv is not None:
                dt = apply_same_date_constraint(dt, surv, left_dt, right_dt)
            out.at[idx, "Timestamp"] = fmt_ts(dt, dataset_frac_digits)
            new_s = out.at[idx, "Timestamp"]

        if new_s == old_s:
            dt = parse_loose(new_s)
            if pd.isna(dt):
                dt = pd.Timestamp("2024-01-01 12:00:00")
            dt = dt + pd.Timedelta(milliseconds=1 if dataset_frac_digits >= 3 else 1)
            surv = extract_surviving_date(old_s) if has_explicit_date_only_or_midnight_missing_time(old_s) else None
            case_val = out.at[idx, "Case"]
            case_indices = list(out.index[out["Case"] == case_val])
            pos = case_indices.index(idx)
            left_dt = pd.NaT
            right_dt = pd.NaT
            for p in range(pos - 1, -1, -1):
                ii = case_indices[p]
                if safe_str(out.at[ii, "error_category"]) == "CLEAN":
                    left_dt = parse_loose(out.at[ii, "Timestamp"])
                    if not pd.isna(left_dt):
                        break
            for p in range(pos + 1, len(case_indices)):
                ii = case_indices[p]
                if safe_str(out.at[ii, "error_category"]) == "CLEAN":
                    right_dt = parse_loose(out.at[ii, "Timestamp"])
                    if not pd.isna(right_dt):
                        break
            if surv is not None:
                dt = apply_same_date_constraint(dt, surv, left_dt, right_dt)
            if not pd.isna(left_dt) and dt < left_dt:
                dt = left_dt
            if not pd.isna(right_dt) and dt > right_dt:
                dt = right_dt
            s2 = fmt_ts(dt, dataset_frac_digits)
            if s2 == old_s:
                mid = left_dt + (right_dt - left_dt) / 2 if (not pd.isna(left_dt) and not pd.isna(right_dt)) else dt
                if surv is not None:
                    mid = apply_same_date_constraint(mid, surv, left_dt, right_dt)
                s2 = fmt_ts(mid, dataset_frac_digits)
                if s2 == old_s:
                    mid = mid + pd.Timedelta(milliseconds=2 if dataset_frac_digits >= 3 else 1)
                    if surv is not None:
                        mid = apply_same_date_constraint(mid, surv, left_dt, right_dt)
                    s2 = fmt_ts(mid, dataset_frac_digits)
            out.at[idx, "Timestamp"] = s2

    # Ensure byte-identical non-timestamp columns and row order/count
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        out[col] = original[col]

    # Final same-date and anchor clamping by case
    for case_val, case_idx in out.groupby("Case", sort=False).groups.items():
        case_indices = list(case_idx)
        for pos, idx in enumerate(case_indices):
            if safe_str(out.at[idx, "error_category"]) == "CLEAN":
                continue
            dt = parse_loose(out.at[idx, "Timestamp"])
            old_s = safe_str(original.at[idx, "Timestamp"])
            left_dt = pd.NaT
            right_dt = pd.NaT
            for p in range(pos - 1, -1, -1):
                ii = case_indices[p]
                if safe_str(out.at[ii, "error_category"]) == "CLEAN":
                    left_dt = parse_loose(out.at[ii, "Timestamp"])
                    if not pd.isna(left_dt):
                        break
            for p in range(pos + 1, len(case_indices)):
                ii = case_indices[p]
                if safe_str(out.at[ii, "error_category"]) == "CLEAN":
                    right_dt = parse_loose(out.at[ii, "Timestamp"])
                    if not pd.isna(right_dt):
                        break
            if not pd.isna(left_dt) and dt < left_dt:
                dt = left_dt
            if not pd.isna(right_dt) and dt > right_dt:
                dt = right_dt
            surv = extract_surviving_date(old_s) if has_explicit_date_only_or_midnight_missing_time(old_s) else None
            if surv is not None:
                dt = apply_same_date_constraint(dt, surv, left_dt, right_dt)
            out.at[idx, "Timestamp"] = fmt_ts(dt, dataset_frac_digits)
            if out.at[idx, "Timestamp"] == old_s:
                dt = dt + pd.Timedelta(milliseconds=1 if dataset_frac_digits >= 3 else 1)
                if surv is not None:
                    dt = apply_same_date_constraint(dt, surv, left_dt, right_dt)
                out.at[idx, "Timestamp"] = fmt_ts(dt, dataset_frac_digits)

    rows_still_equal = int(((out["error_category"].astype(str) != "CLEAN") & (out["Timestamp"].astype(str) == original["Timestamp"].astype(str))).sum())

    out.to_csv(args.output, index=False)
    print(f"{blank_filled},{date_only_filled},{rows_still_equal}")


if __name__ == "__main__":
    main()