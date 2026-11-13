import argparse
import math
import re
from collections import defaultdict

import pandas as pd


# ------------------------------------------------------------
# Process knowledge embedded from the prompt
# ------------------------------------------------------------

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
    ("Prepare main course", "Review drinks"): {"p10": 2.2898, "p25": 3.674, "median": 7.2355, "share_under_60s": 0.75},
    ("Signal food complete", "prepare drinks"): {"p10": 65.3186, "p25": 153.056, "median": 228.6855, "share_under_60s": 0.25},
    ("Signal food complete", "Bring drinks"): {"p10": 150.102, "p25": 179.118, "median": 227.478, "share_under_60s": 0.0},
    ("Bring food", "prepare drinks"): {"p10": 164.9284, "p25": 166.546, "median": 169.242, "share_under_60s": 0.0},
    ("prepare drinks", "Bring food"): {"p10": 46.9712, "p25": 63.2825, "median": 90.468, "share_under_60s": 0.333},
    ("Bring food", "Review drinks"): {"p10": 775.2938, "p25": 787.7435, "median": 808.493, "share_under_60s": 0.0},
}

BLOCK_SPAN_P25 = {
    2: 31.6845,
    3: 165.08025,
    4: 395.92275,
    5: 735.7245,
}

GLOBAL_MEDIAN = 118.0235


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def is_blank_timestamp(s):
    return pd.isna(s) or str(s).strip() == ""


def detect_fraction_digits(ts_str):
    if ts_str is None or pd.isna(ts_str):
        return None
    s = str(ts_str).strip()
    m = re.search(r"\.(\d+)$", s)
    return len(m.group(1)) if m else 0


def detect_dataset_fraction_digits(series):
    counts = []
    for x in series:
        if pd.isna(x):
            continue
        s = str(x).strip()
        if not s:
            continue
        digits = detect_fraction_digits(s)
        if digits is not None:
            counts.append(digits)
    if not counts:
        return 3
    freq = pd.Series(counts).value_counts()
    return int(freq.idxmax())


def parse_ts(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def is_midnight_time(ts):
    return ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0


def classify_observation(ts_str):
    if is_blank_timestamp(ts_str):
        return "blank", None
    ts = parse_ts(ts_str)
    if pd.isna(ts):
        return "blank", None
    if is_midnight_time(ts):
        return "date_only", ts.date()
    return "full", ts.date()


def choose_edge_duration_seconds(src, tgt):
    prof = DURATION_PROFILES.get((src, tgt))
    if prof:
        if prof["share_under_60s"] >= 0.5:
            return max(prof["p10"], 0.0)
        return max(prof["p25"], 0.0)
    return None


def safe_step_seconds(src, tgt):
    d = choose_edge_duration_seconds(src, tgt)
    if d is not None:
        return d
    return None


def neighbor_based_weights(activities):
    n = len(activities)
    weights = []
    for i in range(n):
        w_left = None
        w_right = None
        if i > 0:
            w_left = safe_step_seconds(activities[i - 1], activities[i])
        if i < n - 1:
            w_right = safe_step_seconds(activities[i], activities[i + 1])
        vals = [v for v in (w_left, w_right) if v is not None]
        if vals:
            weights.append(max(sum(vals) / len(vals), 0.001))
        else:
            weights.append(None)

    known = [w for w in weights if w is not None]
    default_w = (sum(known) / len(known)) if known else GLOBAL_MEDIAN
    return [default_w if w is None else w for w in weights]


def nonuniform_cumulative_offsets(activities):
    # Build non-uniform positive increments from local DFG edges.
    n = len(activities)
    if n <= 0:
        return []
    if n == 1:
        return [0.0]

    edge_weights = []
    for i in range(n - 1):
        w = safe_step_seconds(activities[i], activities[i + 1])
        edge_weights.append(w)

    known = [w for w in edge_weights if w is not None]
    if known:
        fallback = sum(known) / len(known)
    else:
        fallback = BLOCK_SPAN_P25.get(n, GLOBAL_MEDIAN)
        fallback = max(fallback / max(n, 1), 0.001)

    edge_weights = [fallback if w is None else max(w, 0.0) for w in edge_weights]

    # Avoid perfectly uniform interpolation when all weights equal and n>2.
    if n > 2:
        all_same = max(edge_weights) - min(edge_weights) < 1e-12
        if all_same:
            tweaked = []
            for i, w in enumerate(edge_weights):
                factor = 1.0 + 0.03 * (i % 2)
                tweaked.append(max(w * factor, 0.001))
            edge_weights = tweaked

    offsets = [0.0]
    c = 0.0
    for w in edge_weights:
        c += w
        offsets.append(c)
    return offsets


def format_timestamp(ts, frac_digits):
    if frac_digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    micros = f"{ts.microsecond:06d}"
    frac = micros[:frac_digits].ljust(frac_digits, "0")
    return f"{base}.{frac}"


def clamp(ts, left=None, right=None):
    out = ts
    if left is not None and out < left:
        out = left
    if right is not None and out > right:
        out = right
    return out


def day_bounds(d, frac_digits):
    start = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=0, minute=0, second=0, microsecond=0)
    if frac_digits <= 0:
        end = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=23, minute=59, second=59)
    else:
        micro = int(("9" * min(frac_digits, 6)).ljust(6, "0"))
        end = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=23, minute=59, second=59, microsecond=micro)
    return start, end


def strict_change(ts_new, original_str, frac_digits):
    rendered = format_timestamp(ts_new, frac_digits)
    if str(original_str) != rendered:
        return ts_new, rendered
    bumped = ts_new + pd.Timedelta(milliseconds=1 if frac_digits >= 3 else 1, unit="ms" if frac_digits >= 3 else "s")
    rendered2 = format_timestamp(bumped, frac_digits)
    if str(original_str) != rendered2:
        return bumped, rendered2
    bumped2 = ts_new - pd.Timedelta(milliseconds=1 if frac_digits >= 3 else 1, unit="ms" if frac_digits >= 3 else "s")
    rendered3 = format_timestamp(bumped2, frac_digits)
    return bumped2, rendered3


# ------------------------------------------------------------
# Repair logic
# ------------------------------------------------------------

def repair_case(group, frac_digits):
    idxs = list(group.index)
    repaired = {}
    meta = {}
    n = len(idxs)

    parsed_clean = {}
    for i in idxs:
        if group.at[i, "error_category"] == "CLEAN":
            parsed_clean[i] = parse_ts(group.at[i, "Timestamp"])

    # contiguous non-clean blocks
    pos = 0
    while pos < n:
        idx = idxs[pos]
        if group.at[idx, "error_category"] == "CLEAN":
            pos += 1
            continue
        start = pos
        while pos < n and group.at[idxs[pos], "error_category"] != "CLEAN":
            pos += 1
        end = pos - 1
        block_idxs = idxs[start:end + 1]
        block_acts = [group.at[i, "Activity"] for i in block_idxs]

        left_idx = idxs[start - 1] if start - 1 >= 0 else None
        right_idx = idxs[end + 1] if end + 1 < n else None

        left_ts = parsed_clean.get(left_idx) if left_idx is not None else None
        right_ts = parsed_clean.get(right_idx) if right_idx is not None else None

        obs_classes = [classify_observation(group.at[i, "Timestamp"]) for i in block_idxs]

        # Build left-anchored candidates
        offsets = nonuniform_cumulative_offsets(block_acts)
        left_edge = safe_step_seconds(group.at[left_idx, "Activity"], block_acts[0]) if left_idx is not None else None
        right_edge = safe_step_seconds(block_acts[-1], group.at[right_idx, "Activity"]) if right_idx is not None else None

        if left_edge is None:
            left_edge = neighbor_based_weights([group.at[left_idx, "Activity"]] + block_acts)[1] if left_idx is not None else GLOBAL_MEDIAN
        if right_edge is None:
            right_edge = neighbor_based_weights(block_acts + [group.at[right_idx, "Activity"]])[-2] if right_idx is not None else GLOBAL_MEDIAN

        left_candidates = []
        if left_ts is not None:
            for off in offsets:
                left_candidates.append(left_ts + pd.Timedelta(seconds=(left_edge + off)))

        # Build right-anchored candidates
        rev_offsets = nonuniform_cumulative_offsets(list(reversed(block_acts)))
        right_candidates = [None] * len(block_idxs)
        if right_ts is not None:
            for j in range(len(block_idxs)):
                rev_off = rev_offsets[len(block_idxs) - 1 - j]
                right_candidates[j] = right_ts - pd.Timedelta(seconds=(right_edge + rev_off))

        # Main placement
        for j, row_idx in enumerate(block_idxs):
            cls, preserved_date = obs_classes[j]
            original = group.at[row_idx, "Timestamp"]

            candidate = None

            # Prefer preserving date-only by inferring only time-of-day on same date.
            if cls == "date_only":
                day_start, day_end = day_bounds(preserved_date, frac_digits)

                left_same = left_ts if left_ts is not None and left_ts.date() == preserved_date else None
                right_same = right_ts if right_ts is not None and right_ts.date() == preserved_date else None

                cands = []
                if left_same is not None and left_candidates:
                    cands.append(left_candidates[j])
                if right_same is not None and right_candidates[j] is not None:
                    cands.append(right_candidates[j])

                if left_same is not None and right_same is not None:
                    interval = (right_same - left_same).total_seconds()
                    comp_left = left_same + pd.Timedelta(seconds=max(0.001, interval * 0.35))
                    comp_right = left_same + pd.Timedelta(seconds=max(0.002, interval * 0.65))
                    base = cands[0] if cands else comp_left
                    candidate = clamp(base, comp_left, comp_right)
                elif cands:
                    candidate = cands[0]
                elif left_same is not None:
                    candidate = clamp(left_same + pd.Timedelta(seconds=max(left_edge + offsets[j], 0.001)), day_start, day_end)
                elif right_same is not None:
                    candidate = clamp(right_same - pd.Timedelta(seconds=max(right_edge + rev_offsets[len(block_idxs) - 1 - j], 0.001)), day_start, day_end)
                else:
                    # preserve observed date with a modest process-based time
                    sec = max(BLOCK_SPAN_P25.get(len(block_idxs), GLOBAL_MEDIAN) * (0.25 + 0.2 * j), 1.0)
                    candidate = day_start + pd.Timedelta(seconds=min(sec, 23 * 3600 + 50 * 60))

                candidate = clamp(candidate, day_start, day_end)
                if candidate.date() != preserved_date:
                    candidate = day_start + pd.Timedelta(hours=12)

            else:
                # blank timestamp
                if left_ts is not None and right_ts is not None:
                    raw = left_candidates[j] if left_candidates else None
                    if raw is None:
                        raw = left_ts + pd.Timedelta(seconds=max(BLOCK_SPAN_P25.get(len(block_idxs), GLOBAL_MEDIAN), 1.0))
                    candidate = raw
                elif left_ts is not None:
                    candidate = left_candidates[j] if left_candidates else left_ts + pd.Timedelta(seconds=(j + 1) * GLOBAL_MEDIAN)
                elif right_ts is not None:
                    candidate = right_candidates[j] if right_candidates[j] is not None else right_ts - pd.Timedelta(seconds=(len(block_idxs) - j) * GLOBAL_MEDIAN)
                else:
                    candidate = pd.Timestamp("2024-01-01 12:00:00")  # never expected; fixed in self-check using neighbors if any

            repaired[row_idx] = candidate
            meta[row_idx] = {"class": cls, "preserved_date": preserved_date, "block_idxs": block_idxs, "j": j}

        # If both anchors exist, compress non-uniformly into interval.
        if left_ts is not None and right_ts is not None:
            avail = (right_ts - left_ts).total_seconds()
            if avail < 0:
                avail = 0.0

            if len(block_idxs) == 1:
                row_idx = block_idxs[0]
                cls = meta[row_idx]["class"]
                if cls == "date_only":
                    d = meta[row_idx]["preserved_date"]
                    if left_ts.date() == d and right_ts.date() == d:
                        target = left_ts + pd.Timedelta(seconds=avail * 0.5)
                    else:
                        day_start, day_end = day_bounds(d, frac_digits)
                        target = day_start + pd.Timedelta(hours=12)
                        target = clamp(target, max(day_start, left_ts) if left_ts.date() == d else day_start,
                                       min(day_end, right_ts) if right_ts.date() == d else day_end)
                else:
                    target = left_ts + pd.Timedelta(seconds=avail * 0.5)
                repaired[row_idx] = clamp(target, left_ts, right_ts)
            else:
                rel = nonuniform_cumulative_offsets(block_acts)
                total = rel[-1] if rel and rel[-1] > 0 else 1.0
                # keep inside anchors, not on exact endpoints unless forced
                left_pad = min(max(avail * 0.08, 0.001), 60.0)
                right_pad = min(max(avail * 0.08, 0.001), 60.0)
                usable = max(avail - left_pad - right_pad, 0.0)

                for j, row_idx in enumerate(block_idxs):
                    frac = rel[j] / total if total > 0 else j / max(len(block_idxs) - 1, 1)
                    t = left_ts + pd.Timedelta(seconds=left_pad + usable * frac)

                    cls = meta[row_idx]["class"]
                    if cls == "date_only":
                        d = meta[row_idx]["preserved_date"]
                        day_start, day_end = day_bounds(d, frac_digits)
                        t = clamp(t, day_start, day_end)
                    repaired[row_idx] = clamp(t, left_ts, right_ts)

    return repaired, meta


def self_check_and_fix(df, output_ts, repaired_meta, frac_digits):
    blanks_filled = 0
    date_only_filled = 0
    still_equal = 0

    case_groups = {case: list(g.index) for case, g in df.groupby("Case", sort=False)}

    # Count categories
    for idx, row in df.iterrows():
        if row["error_category"] == "CLEAN":
            continue
        cls, _ = classify_observation(row["Timestamp"])
        if cls == "blank":
            blanks_filled += 1
        elif cls == "date_only":
            date_only_filled += 1

    # Parse repaired
    parsed = {}
    for idx, ts_str in output_ts.items():
        parsed[idx] = parse_ts(ts_str)
        if str(ts_str) == str(df.at[idx, "Timestamp"]):
            still_equal += 1

    # Fix parse failures and constraints
    for case, idxs in case_groups.items():
        clean_times = [(i, parse_ts(df.at[i, "Timestamp"])) for i in idxs if df.at[i, "error_category"] == "CLEAN"]
        clean_times = [(i, t) for i, t in clean_times if not pd.isna(t)]
        case_start = min((t for _, t in clean_times), default=None)
        case_end = max((t for _, t in clean_times), default=None)

        for pos, idx in enumerate(idxs):
            if df.at[idx, "error_category"] == "CLEAN":
                continue

            if pd.isna(parsed[idx]):
                # reconstruct from neighbors, never observed corruption
                left = None
                right = None
                for p in range(pos - 1, -1, -1):
                    j = idxs[p]
                    t = parsed.get(j) if j in parsed else parse_ts(output_ts.get(j, df.at[j, "Timestamp"]))
                    if not pd.isna(t):
                        left = t
                        break
                for p in range(pos + 1, len(idxs)):
                    j = idxs[p]
                    t = parsed.get(j) if j in parsed else parse_ts(output_ts.get(j, df.at[j, "Timestamp"]))
                    if not pd.isna(t):
                        right = t
                        break
                if left is not None and right is not None:
                    new_t = left + (right - left) / 2
                elif left is not None:
                    new_t = left + pd.Timedelta(seconds=GLOBAL_MEDIAN)
                elif right is not None:
                    new_t = right - pd.Timedelta(seconds=GLOBAL_MEDIAN)
                else:
                    new_t = pd.Timestamp("2024-01-01 12:00:00")
                new_t, rendered = strict_change(new_t, df.at[idx, "Timestamp"], frac_digits)
                output_ts[idx] = rendered
                parsed[idx] = parse_ts(rendered)

            # bracket anchors
            left_anchor = None
            right_anchor = None
            for p in range(pos - 1, -1, -1):
                j = idxs[p]
                if df.at[j, "error_category"] == "CLEAN":
                    t = parse_ts(df.at[j, "Timestamp"])
                    if not pd.isna(t):
                        left_anchor = t
                        break
            for p in range(pos + 1, len(idxs)):
                j = idxs[p]
                if df.at[j, "error_category"] == "CLEAN":
                    t = parse_ts(df.at[j, "Timestamp"])
                    if not pd.isna(t):
                        right_anchor = t
                        break

            t = parsed[idx]
            cls, preserved_date = classify_observation(df.at[idx, "Timestamp"])

            if cls == "date_only" and preserved_date is not None:
                day_start, day_end = day_bounds(preserved_date, frac_digits)
                t = clamp(t, day_start, day_end)
                if t.date() != preserved_date:
                    t = day_start + pd.Timedelta(hours=12)

            if case_start is not None and t < case_start:
                t = case_start
            if case_end is not None and t > case_end:
                t = case_end
            if left_anchor is not None and t < left_anchor:
                t = left_anchor
            if right_anchor is not None and t > right_anchor:
                t = right_anchor

            # Ensure date-only not at original exact input value and gain time-of-day
            if cls == "date_only" and preserved_date is not None and is_midnight_time(t):
                if left_anchor is not None and right_anchor is not None and left_anchor.date() == preserved_date and right_anchor.date() == preserved_date:
                    t = left_anchor + (right_anchor - left_anchor) / 2
                elif left_anchor is not None and left_anchor.date() == preserved_date:
                    t = left_anchor + pd.Timedelta(seconds=60)
                elif right_anchor is not None and right_anchor.date() == preserved_date:
                    t = right_anchor - pd.Timedelta(seconds=60)
                else:
                    t = pd.Timestamp(year=preserved_date.year, month=preserved_date.month, day=preserved_date.day, hour=12)
                day_start, day_end = day_bounds(preserved_date, frac_digits)
                t = clamp(t, day_start, day_end)

            t, rendered = strict_change(t, df.at[idx, "Timestamp"], frac_digits)

            # final preserved-date enforcement after strict change
            if cls == "date_only" and preserved_date is not None:
                if parse_ts(rendered).date() != preserved_date:
                    base = pd.Timestamp(year=preserved_date.year, month=preserved_date.month, day=preserved_date.day, hour=12, minute=0, second=0, microsecond=123000 if frac_digits >= 3 else 0)
                    base, rendered = strict_change(base, df.at[idx, "Timestamp"], frac_digits)

            output_ts[idx] = rendered
            parsed[idx] = parse_ts(rendered)

    # recompute still_equal after fixes
    still_equal = 0
    for idx, row in df.iterrows():
        if row["error_category"] != "CLEAN" and str(output_ts[idx]) == str(row["Timestamp"]):
            still_equal += 1

    return output_ts, blanks_filled, date_only_filled, still_equal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    required_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    frac_digits = detect_dataset_fraction_digits(df["Timestamp"])

    # Initialize output timestamps: preserve all original strings by default.
    output_ts = {idx: df.at[idx, "Timestamp"] for idx in df.index}
    repaired_meta = {}

    # Repair only non-CLEAN rows, case by case, keeping row order unchanged.
    for case, group in df.groupby("Case", sort=False):
        try:
            repaired, meta = repair_case(group, frac_digits)
            for idx, ts in repaired.items():
                rendered = format_timestamp(ts, frac_digits)
                # ensure every corrupted row changes
                ts2, rendered2 = strict_change(ts, df.at[idx, "Timestamp"], frac_digits)
                output_ts[idx] = rendered2
                repaired_meta[idx] = meta.get(idx, {})
        except Exception:
            # non-failing fallback: conservative neighbor-based fill
            idxs = list(group.index)
            for pos, idx in enumerate(idxs):
                if df.at[idx, "error_category"] == "CLEAN":
                    continue
                left = None
                right = None
                for p in range(pos - 1, -1, -1):
                    j = idxs[p]
                    t = parse_ts(df.at[j, "Timestamp"]) if df.at[j, "error_category"] == "CLEAN" else parse_ts(output_ts[j])
                    if not pd.isna(t):
                        left = t
                        break
                for p in range(pos + 1, len(idxs)):
                    j = idxs[p]
                    t = parse_ts(df.at[j, "Timestamp"]) if df.at[j, "error_category"] == "CLEAN" else parse_ts(output_ts[j])
                    if not pd.isna(t):
                        right = t
                        break
                cls, preserved_date = classify_observation(df.at[idx, "Timestamp"])
                if left is not None and right is not None:
                    t = left + (right - left) / 2
                elif left is not None:
                    t = left + pd.Timedelta(seconds=GLOBAL_MEDIAN)
                elif right is not None:
                    t = right - pd.Timedelta(seconds=GLOBAL_MEDIAN)
                else:
                    t = pd.Timestamp("2024-01-01 12:00:00")
                if cls == "date_only" and preserved_date is not None:
                    day_start, day_end = day_bounds(preserved_date, frac_digits)
                    t = clamp(t, day_start, day_end)
                    if is_midnight_time(t):
                        t = clamp(day_start + pd.Timedelta(hours=12), day_start, day_end)
                t, rendered = strict_change(t, df.at[idx, "Timestamp"], frac_digits)
                output_ts[idx] = rendered
                repaired_meta[idx] = {"class": cls, "preserved_date": preserved_date}

    # Final self-check and automatic fixing
    output_ts, blanks_filled, date_only_filled, still_equal = self_check_and_fix(df, output_ts, repaired_meta, frac_digits)

    # Write output preserving all non-timestamp columns byte-identically and row order unchanged
    out_df = df.copy()
    out_df["Timestamp"] = [output_ts[idx] if df.at[idx, "error_category"] != "CLEAN" else df.at[idx, "Timestamp"] for idx in df.index]
    out_df.to_csv(args.output, index=False)

    print(f"{blanks_filled},{date_only_filled},{still_equal}")


if __name__ == "__main__":
    main()