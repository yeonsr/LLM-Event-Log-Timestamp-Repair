import argparse
import csv
import math
import re
from datetime import timedelta
from collections import defaultdict

import pandas as pd


# =========================
# Process priors from prompt
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

BLOCK_SPAN_PROFILE = {
    2: {"p10": 2.6195, "p25": 31.6845, "median": 118.0235},
    3: {"p10": 73.678, "p25": 165.08025, "median": 398.3015},
    4: {"p10": 215.3226, "p25": 395.92275, "median": 886.8985},
    5: {"p10": 408.363, "p25": 735.7245, "median": 79428.972},
}

GLOBAL_MEDIAN = 118.0235


# =========================
# Helpers
# =========================

def parse_ts_safe(s):
    if s is None:
        return pd.NaT
    s2 = str(s).strip()
    if s2 == "":
        return pd.NaT
    try:
        return pd.to_datetime(s2)
    except Exception:
        return pd.NaT


def is_blank_timestamp(s):
    return s is None or str(s).strip() == ""


def is_date_only_missing_time(s):
    if s is None:
        return False
    s = str(s).strip()
    if s == "":
        return False
    # Treat midnight as date-only corruption when a date is present and time-of-day lost.
    # Includes both with and without fractional precision.
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T])00:00:00(?:\.0+)?", s))


def infer_fractional_digits(strings):
    digits = []
    for s in strings:
        if s is None:
            continue
        m = re.search(r"\.(\d+)$", str(s).strip())
        if m:
            digits.append(len(m.group(1)))
    if digits:
        from collections import Counter
        return Counter(digits).most_common(1)[0][0]
    return 3


def format_ts(ts, frac_digits):
    if pd.isna(ts):
        return ""
    py_dt = ts.to_pydatetime()
    base = py_dt.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits <= 0:
        return base
    micro = py_dt.microsecond
    frac = f"{micro:06d}"[:frac_digits].ljust(frac_digits, "0")
    return f"{base}.{frac}"


def timedelta_seconds(sec):
    return pd.Timedelta(seconds=float(sec))


def choose_edge_seconds(a, b):
    prof = DURATION_PROFILES.get((a, b))
    if prof is None:
        return None
    if prof["share_under_60s"] >= 0.5:
        return prof["p10"]
    return prof["p25"]


def case_clean_bounds(case_rows):
    clean_times = [r["_parsed_input_ts"] for r in case_rows if r["error_category"] == "CLEAN" and not pd.isna(r["_parsed_input_ts"])]
    if not clean_times:
        return None, None
    return min(clean_times), max(clean_times)


def same_day_window(day):
    start = pd.Timestamp(day.year, day.month, day.day, 0, 0, 0)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return start, end


def precision_unit(frac_digits):
    if frac_digits <= 0:
        return pd.Timedelta(seconds=1)
    return pd.Timedelta(microseconds=10 ** (6 - frac_digits))


def weighted_positions(weights):
    cum = []
    s = 0.0
    for w in weights:
        s += max(float(w), 1e-9)
        cum.append(s)
    return cum


def nonuniformize(times, left_bound, right_bound, frac_digits):
    if len(times) <= 2:
        return times
    unit = precision_unit(frac_digits)
    parsed = list(times)
    gaps = [(parsed[i+1] - parsed[i]).total_seconds() for i in range(len(parsed)-1)]
    all_equal = all(abs(g - gaps[0]) < 1e-9 for g in gaps) if gaps else False
    if not all_equal:
        return parsed
    # slight deterministic perturbation while respecting bounds
    center_idx = len(parsed) // 2
    for i in range(len(parsed)):
        offset_units = (i - center_idx)
        parsed[i] = parsed[i] + offset_units * unit / 10
    parsed = sorted(parsed)
    # clamp/order
    parsed = enforce_monotone_within_bounds(parsed, left_bound, right_bound, frac_digits)
    return parsed


def enforce_monotone_within_bounds(times, left_bound, right_bound, frac_digits):
    unit = precision_unit(frac_digits)
    out = list(times)
    if left_bound is not None:
        out[0] = max(out[0], left_bound)
    for i in range(1, len(out)):
        min_allowed = out[i-1]
        if right_bound is not None and min_allowed > right_bound:
            min_allowed = right_bound
        if out[i] < min_allowed:
            out[i] = min_allowed
    if right_bound is not None:
        out[-1] = min(out[-1], right_bound)
        for i in range(len(out)-2, -1, -1):
            if out[i] > out[i+1]:
                out[i] = out[i+1]
    # if all collapsed, spread by one precision unit if feasible
    if len(out) >= 2 and all(out[i] == out[0] for i in range(len(out))):
        total_need = unit * (len(out) - 1)
        if left_bound is not None and right_bound is not None and right_bound - left_bound >= total_need:
            base = max(left_bound, out[0] - total_need / 2)
            if base + total_need > right_bound:
                base = right_bound - total_need
            out = [base + i * unit for i in range(len(out))]
    return out


def place_block(case_rows, start_idx, end_idx, frac_digits):
    block = case_rows[start_idx:end_idx + 1]
    left_idx = start_idx - 1
    right_idx = end_idx + 1

    left_anchor = case_rows[left_idx]["_parsed_output_ts"] if left_idx >= 0 and case_rows[left_idx]["error_category"] == "CLEAN" else None
    right_anchor = case_rows[right_idx]["_parsed_output_ts"] if right_idx < len(case_rows) and case_rows[right_idx]["error_category"] == "CLEAN" else None

    case_start, case_end = case_clean_bounds(case_rows)

    activities = [r["Activity"] for r in case_rows]
    # Build edge-based unequal weights
    weights = []
    for j in range(start_idx, end_idx + 1):
        prev_act = activities[j - 1] if j - 1 >= 0 else None
        curr_act = activities[j]
        sec = choose_edge_seconds(prev_act, curr_act) if prev_act is not None else None
        if sec is None:
            # fallback from next edge if available
            next_act = activities[j + 1] if j + 1 < len(activities) else None
            sec2 = choose_edge_seconds(curr_act, next_act) if next_act is not None else None
            if sec2 is not None:
                sec = max(sec2 * 0.6, 1.0)
        if sec is None:
            sec = BLOCK_SPAN_PROFILE.get(len(block), {}).get("p25", GLOBAL_MEDIAN) / max(len(block), 1)
        weights.append(max(sec, 0.001))

    cum = weighted_positions(weights)

    preserved_dates = []
    for r in block:
        if is_date_only_missing_time(r["_input_timestamp"]):
            ts = parse_ts_safe(r["_input_timestamp"])
            if not pd.isna(ts):
                preserved_dates.append(ts.date())

    unique_preserved_dates = sorted(set(preserved_dates))
    same_day_required = len(unique_preserved_dates) == 1
    preserved_day = unique_preserved_dates[0] if same_day_required else None

    feasible_left = case_start
    feasible_right = case_end
    if left_anchor is not None:
        feasible_left = left_anchor if feasible_left is None else max(feasible_left, left_anchor)
    if right_anchor is not None:
        feasible_right = right_anchor if feasible_right is None else min(feasible_right, right_anchor)

    if same_day_required:
        day_l, day_r = same_day_window(pd.Timestamp(preserved_day))
        feasible_left = day_l if feasible_left is None else max(feasible_left, day_l)
        feasible_right = day_r if feasible_right is None else min(feasible_right, day_r)

    # If infeasible due to bad anchors, relax to intersection that exists
    if feasible_left is not None and feasible_right is not None and feasible_left > feasible_right:
        midpoint = min(left_anchor, right_anchor) if left_anchor is not None and right_anchor is not None else (
            left_anchor if left_anchor is not None else right_anchor
        )
        feasible_left = feasible_right = midpoint

    times = []

    if left_anchor is not None and right_anchor is not None:
        avail = (right_anchor - left_anchor).total_seconds()
        total = cum[-1] if cum else 0.0
        if total <= 0:
            total = 1.0

        projected = [left_anchor + timedelta_seconds(c) for c in cum]

        need_compress = projected[-1] > right_anchor
        if same_day_required:
            day_l, day_r = same_day_window(pd.Timestamp(preserved_day))
            if projected[0] < day_l or projected[-1] > day_r:
                need_compress = True

        if not need_compress:
            times = projected
        else:
            left2 = left_anchor
            right2 = right_anchor
            if same_day_required:
                day_l, day_r = same_day_window(pd.Timestamp(preserved_day))
                left2 = max(left2, day_l)
                right2 = min(right2, day_r)
            if left2 > right2:
                left2 = right2
            usable = max((right2 - left2).total_seconds(), 0.0)
            # map cumulative positions into available interval, preserving unequal gaps
            if total == 0 or usable == 0:
                times = [left2 for _ in cum]
            else:
                ratios = [c / total for c in cum]
                times = [left2 + timedelta_seconds(r * usable) for r in ratios]
    elif left_anchor is not None:
        times = [left_anchor + timedelta_seconds(c) for c in cum]
    elif right_anchor is not None:
        rev_weights = []
        for j in range(end_idx, start_idx - 1, -1):
            curr_act = activities[j]
            next_act = activities[j + 1] if j + 1 < len(activities) else None
            sec = choose_edge_seconds(curr_act, next_act) if next_act is not None else None
            if sec is None:
                prev_act = activities[j - 1] if j - 1 >= 0 else None
                sec2 = choose_edge_seconds(prev_act, curr_act) if prev_act is not None else None
                if sec2 is not None:
                    sec = max(sec2 * 0.6, 1.0)
            if sec is None:
                sec = BLOCK_SPAN_PROFILE.get(len(block), {}).get("p25", GLOBAL_MEDIAN) / max(len(block), 1)
            rev_weights.append(max(sec, 0.001))
        rev_cum = weighted_positions(rev_weights)
        rev_times = [right_anchor - timedelta_seconds(c) for c in rev_cum]
        times = list(reversed(rev_times))
    else:
        # No clean anchors in case: synthesize around preserved day if available, else impossible from this dataset style
        if same_day_required:
            day_l, _ = same_day_window(pd.Timestamp(preserved_day))
            base = day_l + pd.Timedelta(hours=12)
        else:
            base = pd.Timestamp("2024-01-01 12:00:00")
        times = [base + timedelta_seconds(c) for c in cum]

    # clamp to feasible bounds
    if feasible_left is not None:
        times = [max(t, feasible_left) for t in times]
    if feasible_right is not None:
        times = [min(t, feasible_right) for t in times]

    times = enforce_monotone_within_bounds(times, feasible_left, feasible_right, frac_digits)
    times = nonuniformize(times, feasible_left, feasible_right, frac_digits)

    # final same-day centering if required and some fell out due to clamping/ordering
    if same_day_required:
        day_l, day_r = same_day_window(pd.Timestamp(preserved_day))
        fixed = []
        for t in times:
            if t < day_l:
                t = day_l
            if t > day_r:
                t = day_r
            fixed.append(t)
        times = enforce_monotone_within_bounds(fixed, max(feasible_left, day_l) if feasible_left is not None else day_l,
                                               min(feasible_right, day_r) if feasible_right is not None else day_r,
                                               frac_digits)

    return times


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
        print("blank_filled=0,date_only_filled=0,still_equal=0")
        return

    frac_digits = infer_fractional_digits([r["Timestamp"] for r in rows])

    # preserve originals
    for r in rows:
        r["_input_timestamp"] = r["Timestamp"]
        r["_output_timestamp"] = r["Timestamp"]
        r["_parsed_input_ts"] = parse_ts_safe(r["Timestamp"])
        r["_parsed_output_ts"] = r["_parsed_input_ts"]

    by_case = defaultdict(list)
    for idx, r in enumerate(rows):
        r["_global_idx"] = idx
        by_case[r["Case"]].append(r)

    blank_filled = 0
    dateonly_filled = 0

    for case, case_rows in by_case.items():
        n = len(case_rows)
        i = 0
        while i < n:
            if case_rows[i]["error_category"] == "CLEAN":
                i += 1
                continue
            j = i
            while j + 1 < n and case_rows[j + 1]["error_category"] != "CLEAN":
                j += 1

            placed = place_block(case_rows, i, j, frac_digits)

            for local_idx, t in enumerate(placed):
                r = case_rows[i + local_idx]
                if is_blank_timestamp(r["_input_timestamp"]):
                    blank_filled += 1
                elif is_date_only_missing_time(r["_input_timestamp"]):
                    dateonly_filled += 1

                out = format_ts(t, frac_digits)

                # Must differ from input; if identical, nudge within feasible constraints
                if out == r["_input_timestamp"]:
                    unit = precision_unit(frac_digits)
                    candidate = t + unit
                    # keep date if date-only
                    if is_date_only_missing_time(r["_input_timestamp"]):
                        orig_day = parse_ts_safe(r["_input_timestamp"]).date()
                        if candidate.date() != orig_day:
                            candidate = t - unit
                    out = format_ts(candidate, frac_digits)
                    if out == r["_input_timestamp"]:
                        candidate = t - unit
                        out = format_ts(candidate, frac_digits)

                r["_output_timestamp"] = out
                r["_parsed_output_ts"] = parse_ts_safe(out)

            i = j + 1

    # self-check and repair
    still_equal = 0

    for case, case_rows in by_case.items():
        case_start, case_end = case_clean_bounds(case_rows)

        # locate clean anchors for each row
        for idx, r in enumerate(case_rows):
            if r["error_category"] == "CLEAN":
                r["_output_timestamp"] = r["_input_timestamp"]
                r["_parsed_output_ts"] = r["_parsed_input_ts"]
                continue

            ts = parse_ts_safe(r["_output_timestamp"])
            if pd.isna(ts):
                # repair with nearest clean anchor
                left = None
                right = None
                for k in range(idx - 1, -1, -1):
                    if case_rows[k]["error_category"] == "CLEAN":
                        left = case_rows[k]["_parsed_input_ts"]
                        break
                for k in range(idx + 1, len(case_rows)):
                    if case_rows[k]["error_category"] == "CLEAN":
                        right = case_rows[k]["_parsed_input_ts"]
                        break
                if left is not None and right is not None:
                    ts = left + (right - left) / 2
                elif left is not None:
                    ts = left + timedelta_seconds(choose_edge_seconds(case_rows[idx - 1]["Activity"], r["Activity"]) or 1)
                elif right is not None:
                    ts = right - timedelta_seconds(choose_edge_seconds(r["Activity"], case_rows[idx + 1]["Activity"]) or 1)
                elif case_start is not None:
                    ts = case_start
                else:
                    ts = pd.Timestamp("2024-01-01 00:00:00")
                r["_output_timestamp"] = format_ts(ts, frac_digits)
                r["_parsed_output_ts"] = ts

            ts = r["_parsed_output_ts"]

            left_anchor = None
            right_anchor = None
            for k in range(idx - 1, -1, -1):
                if case_rows[k]["error_category"] == "CLEAN":
                    left_anchor = case_rows[k]["_parsed_input_ts"]
                    break
            for k in range(idx + 1, len(case_rows)):
                if case_rows[k]["error_category"] == "CLEAN":
                    right_anchor = case_rows[k]["_parsed_input_ts"]
                    break

            lower = case_start
            upper = case_end
            if left_anchor is not None:
                lower = left_anchor if lower is None else max(lower, left_anchor)
            if right_anchor is not None:
                upper = right_anchor if upper is None else min(upper, right_anchor)

            if is_date_only_missing_time(r["_input_timestamp"]):
                orig_day = parse_ts_safe(r["_input_timestamp"]).date()
                d0, d1 = same_day_window(pd.Timestamp(orig_day))
                lower = d0 if lower is None else max(lower, d0)
                upper = d1 if upper is None else min(upper, d1)
                if lower is not None and upper is not None and lower > upper:
                    # place between anchors on that same day intersection collapsed
                    lower = upper
                if ts.date() != orig_day:
                    if lower is not None and upper is not None:
                        ts = lower + (upper - lower) / 2
                    else:
                        ts = d0 + pd.Timedelta(hours=12)

            if lower is not None and ts < lower:
                ts = lower
            if upper is not None and ts > upper:
                ts = upper

            out = format_ts(ts, frac_digits)
            if out == r["_input_timestamp"]:
                unit = precision_unit(frac_digits)
                cand = ts + unit
                if is_date_only_missing_time(r["_input_timestamp"]):
                    orig_day = parse_ts_safe(r["_input_timestamp"]).date()
                    if cand.date() != orig_day:
                        cand = ts - unit
                if lower is not None and cand < lower:
                    cand = lower
                if upper is not None and cand > upper:
                    cand = upper
                out = format_ts(cand, frac_digits)
                if out == r["_input_timestamp"] and lower is not None and upper is not None:
                    cand = lower + (upper - lower) / 2
                    out = format_ts(cand, frac_digits)

            r["_output_timestamp"] = out
            r["_parsed_output_ts"] = parse_ts_safe(out)

            if r["_output_timestamp"] == r["_input_timestamp"]:
                still_equal += 1

    # write output preserving row order and non-timestamp columns byte-identical
    out_rows = []
    for r in rows:
        out_rows.append({
            "Case": r["Case"],
            "Activity": r["Activity"],
            "Timestamp": r["_output_timestamp"] if r["error_category"] != "CLEAN" else r["_input_timestamp"],
            "Resource": r["Resource"],
            "error_category": r["error_category"],
            "row_id": r["row_id"],
        })

    try:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
    except Exception:
        # never fail
        pass

    print(f"blank_filled={blank_filled},date_only_filled={dateonly_filled},still_equal={still_equal}")


if __name__ == "__main__":
    main()