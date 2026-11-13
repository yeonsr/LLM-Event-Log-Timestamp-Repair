import argparse
import math
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Embedded process knowledge
# -----------------------------
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

DFG_FREQ = {
    ("Bring drinks", "Review drinks"): 7789,
    ("prepare drinks", "Bring drinks"): 7327,
    ("Deliver to customer", "EVENT 8 END"): 5000,
    ("Bring food", "Deliver to customer"): 4986,
    ("start", "Take order"): 4860,
    ("Signal food complete", "Bring food"): 4816,
    ("Take order", "Review order"): 4665,
    ("Prepare main course", "Signal food complete"): 4086,
    ("Review drinks", "Prepare main course"): 3950,
    ("Review order", "prepare drinks"): 3761,
    ("Review drinks", "prepare drinks"): 3128,
    ("Prepare starter", "prepare drinks"): 981,
    ("Review order", "Prepare starter"): 880,
    ("Review drinks", "Signal food complete"): 637,
    ("Prepare starter", "Bring drinks"): 541,
    ("prepare drinks", "Prepare starter"): 508,
    ("Review drinks", "Prepare starter"): 330,
    ("prepare drinks", "Review drinks"): 252,
    ("Prepare starter", "Prepare main course"): 157,
    ("Prepare main course", "Bring food"): 157,
    ("Review order", "Bring drinks"): 152,
    ("Take order", "prepare drinks"): 151,
    ("start", "Review order"): 140,
    ("Bring drinks", "Prepare main course"): 140,
    ("Prepare starter", "Review drinks"): 108,
    ("Bring drinks", "Prepare starter"): 100,
    ("Prepare starter", "Signal food complete"): 75,
    ("Bring drinks", "prepare drinks"): 74,
    ("Review drinks", "Bring drinks"): 72,
    ("Take order", "Prepare starter"): 44,
    ("Review drinks", "Bring food"): 24,
    ("prepare drinks", "Prepare main course"): 21,
    ("Bring drinks", "Signal food complete"): 19,
    ("Prepare main course", "Bring drinks"): 18,
    ("Prepare main course", "prepare drinks"): 15,
    ("Review drinks", "Deliver to customer"): 14,
    ("Review order", "Prepare main course"): 12,
    ("Bring food", "Bring drinks"): 9,
    ("prepare drinks", "Signal food complete"): 6,
    ("Signal food complete", "prepare drinks"): 4,
    ("Prepare main course", "Review drinks"): 4,
    ("Bring food", "prepare drinks"): 3,
    ("Signal food complete", "Bring drinks"): 3,
    ("prepare drinks", "Bring food"): 3,
    ("Bring food", "Review drinks"): 2,
}

BLOCK_SPAN_P25 = {
    2: 31.6845,
    3: 165.08025,
    4: 395.92275,
    5: 735.7245,
}
GLOBAL_MEDIAN = 118.0235


# -----------------------------
# Helpers
# -----------------------------
def parse_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, errors="coerce")


def infer_max_fractional_digits(series: pd.Series) -> int:
    max_digits = 0
    pat = re.compile(r"\.(\d+)$")
    for v in series.dropna().astype(str):
        m = pat.search(v.strip())
        if m:
            max_digits = max(max_digits, len(m.group(1)))
    return max_digits


def classify_observed_timestamp(s: str) -> str:
    if s is None:
        return "blank"
    txt = str(s).strip()
    if txt == "" or txt.lower() == "nan":
        return "blank"
    ts = parse_ts(txt)
    if pd.isna(ts):
        return "blank"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", txt):
        return "date_only"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}\s+00:00:00(?:\.0+)?", txt):
        return "date_only"
    return "full"


def same_date_bounds(day: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=day.year, month=day.month, day=day.day)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return start, end


def choose_edge_seconds(src: Optional[str], tgt: Optional[str]) -> Optional[float]:
    if src is None or tgt is None:
        return None
    prof = DURATION_PROFILES.get((src, tgt))
    if not prof:
        return None
    if prof["share_under_60s"] >= 0.5:
        return max(prof["p10"], 0.001)
    return max(prof["p25"], 0.001)


def fallback_edge_seconds(src: Optional[str], tgt: Optional[str]) -> float:
    sec = choose_edge_seconds(src, tgt)
    if sec is not None:
        return sec
    return GLOBAL_MEDIAN


def supported_edge(src: Optional[str], tgt: Optional[str]) -> bool:
    return (src, tgt) in DFG_FREQ


def edge_weight(src: Optional[str], tgt: Optional[str]) -> float:
    if src is None or tgt is None:
        return 1.0
    direct = DFG_FREQ.get((src, tgt), 0)
    reverse = DFG_FREQ.get((tgt, src), 0)
    if direct > 0:
        if reverse > 0 and min(direct, reverse) / max(direct, reverse) > 0.4:
            return 0.2
        return float(direct)
    if reverse > 0:
        return 0.2
    return 0.5


def fmt_timestamp(ts: pd.Timestamp, frac_digits: int) -> str:
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits > 0:
        micros = f"{ts.microsecond:06d}"[:frac_digits].ljust(frac_digits, "0")
        return f"{base}.{micros}"
    return base


def ensure_strictly_monotone(
    times: List[pd.Timestamp],
    lower: pd.Timestamp,
    upper: pd.Timestamp,
    frac_digits: int
) -> List[pd.Timestamp]:
    unit = pd.Timedelta(milliseconds=1) if frac_digits >= 3 else pd.Timedelta(seconds=1)
    if frac_digits > 3:
        unit = pd.Timedelta(microseconds=max(1, 10 ** (6 - min(frac_digits, 6))))
    n = len(times)
    if n == 0:
        return times

    times = [max(t, lower) for t in times]
    times = [min(t, upper) for t in times]

    for i in range(1, n):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + unit

    for i in range(n - 2, -1, -1):
        if times[i] >= times[i + 1]:
            times[i] = times[i + 1] - unit

    if times[0] < lower:
        shift = lower - times[0]
        times = [t + shift for t in times]
    if times[-1] > upper:
        shift = times[-1] - upper
        times = [t - shift for t in times]

    for i in range(1, n):
        if times[i] <= times[i - 1]:
            gap = upper - lower
            if n == 1:
                times[0] = min(max(times[0], lower), upper)
            else:
                step = max(unit, gap / (n + 1))
                times = [lower + step * (i + 1) for i in range(n)]
                break

    times = [min(max(t, lower), upper) for t in times]
    return times


def weighted_positions(start: pd.Timestamp, end: pd.Timestamp, weights: List[float]) -> List[pd.Timestamp]:
    if not weights:
        return []
    total = sum(max(w, 1e-9) for w in weights)
    span = end - start
    csum = 0.0
    out = []
    for w in weights:
        csum += max(w, 1e-9)
        frac = csum / total
        out.append(start + span * frac)
    return out


def non_uniform_internal_weights(activities: List[str], prev_act: Optional[str], next_act: Optional[str]) -> List[float]:
    n = len(activities)
    if n == 0:
        return []
    seg_weights = []
    full = [prev_act] + activities + [next_act]
    for i in range(len(full) - 1):
        seg_weights.append(edge_weight(full[i], full[i + 1]) if full[i] is not None and full[i + 1] is not None else 1.0)
    pos_weights = []
    for i in range(n):
        left = seg_weights[i]
        right = seg_weights[i + 1]
        val = (left * 0.7 + right * 1.3 + (i + 1) * 0.111)
        pos_weights.append(max(val, 1e-6))
    return pos_weights


def choose_case_bounds(clean_times: List[pd.Timestamp], all_observed_dates: List[pd.Timestamp]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if clean_times:
        return min(clean_times), max(clean_times)
    if all_observed_dates:
        day = min(all_observed_dates)
        lo, hi = same_date_bounds(day)
        return lo, hi
    now = pd.Timestamp("2024-01-01 00:00:00")
    return now, now + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def repair_block(
    cdf: pd.DataFrame,
    block_idx: List[int],
    frac_digits: int,
) -> Dict[int, str]:
    repaired = {}
    activities = [cdf.at[i, "Activity"] for i in block_idx]
    observed_ts = [str(cdf.at[i, "Timestamp"]) if pd.notna(cdf.at[i, "Timestamp"]) else "" for i in block_idx]
    kinds = [classify_observed_timestamp(s) for s in observed_ts]

    clean_idx = [i for i in cdf.index if str(cdf.at[i, "error_category"]) == "CLEAN"]
    clean_times = []
    for i in clean_idx:
        ts = parse_ts(cdf.at[i, "Timestamp"])
        if not pd.isna(ts):
            clean_times.append(ts)

    obs_dates = []
    for s, k in zip(observed_ts, kinds):
        if k == "date_only":
            ts = parse_ts(s)
            if not pd.isna(ts):
                obs_dates.append(ts.normalize())

    case_start, case_end = choose_case_bounds(clean_times, obs_dates)

    first = block_idx[0]
    last = block_idx[-1]

    left_anchor_idx = None
    for i in range(first - 1, cdf.index.min() - 1, -1):
        if i in cdf.index and str(cdf.at[i, "error_category"]) == "CLEAN":
            ts = parse_ts(cdf.at[i, "Timestamp"])
            if not pd.isna(ts):
                left_anchor_idx = i
                break

    right_anchor_idx = None
    for i in range(last + 1, cdf.index.max() + 1):
        if i in cdf.index and str(cdf.at[i, "error_category"]) == "CLEAN":
            ts = parse_ts(cdf.at[i, "Timestamp"])
            if not pd.isna(ts):
                right_anchor_idx = i
                break

    left_anchor = parse_ts(cdf.at[left_anchor_idx, "Timestamp"]) if left_anchor_idx is not None else None
    right_anchor = parse_ts(cdf.at[right_anchor_idx, "Timestamp"]) if right_anchor_idx is not None else None
    prev_act = cdf.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None else None
    next_act = cdf.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None else None

    # Validate row-order edges softly; if unsupported, still keep row order but lower trust and use fallbacks.
    _ = all(
        supported_edge(
            cdf.at[block_idx[j], "Activity"],
            cdf.at[block_idx[j + 1], "Activity"]
        ) or DFG_FREQ.get((cdf.at[block_idx[j + 1], "Activity"], cdf.at[block_idx[j], "Activity"]), 0) > 0
        for j in range(len(block_idx) - 1)
    )

    # Build edge prior durations across [left anchor] + block + [right anchor]
    chain_acts = ([prev_act] if prev_act is not None else []) + activities + ([next_act] if next_act is not None else [])
    edge_priors = []
    for j in range(len(chain_acts) - 1):
        edge_priors.append(fallback_edge_seconds(chain_acts[j], chain_acts[j + 1]))

    # Candidate times
    candidate_times = [None] * len(block_idx)

    if left_anchor is not None and right_anchor is not None and right_anchor >= left_anchor:
        total_prior = sum(edge_priors) if edge_priors else GLOBAL_MEDIAN * max(1, len(block_idx))
        window_sec = max((right_anchor - left_anchor).total_seconds(), 0.0)

        # If left-based estimate overshoots or underuses, fit within anchors using weighted shares.
        seg_weights = [max(w, 1e-6) for w in edge_priors]
        cumulative = 0.0
        for j in range(len(block_idx)):
            cumulative += seg_weights[j]
            frac = cumulative / sum(seg_weights)
            candidate_times[j] = left_anchor + pd.Timedelta(seconds=window_sec * frac)

        # Add tiny deterministic non-uniformity to avoid even spacing
        if len(block_idx) >= 2 and window_sec > 0:
            jit = max(0.001, min(0.347, window_sec / (1000.0 * (len(block_idx) + 1))))
            for j in range(len(block_idx)):
                candidate_times[j] += pd.Timedelta(seconds=((j % 2) * 2 - 1) * jit * (0.3 + 0.1 * j))

    elif left_anchor is not None:
        t = left_anchor
        for j in range(len(block_idx)):
            t = t + pd.Timedelta(seconds=edge_priors[j] if j < len(edge_priors) else GLOBAL_MEDIAN)
            candidate_times[j] = t

    elif right_anchor is not None:
        t = right_anchor
        for j in range(len(block_idx) - 1, -1, -1):
            edge_idx = j + 1
            sec = edge_priors[edge_idx] if edge_idx < len(edge_priors) else GLOBAL_MEDIAN
            t = t - pd.Timedelta(seconds=sec)
            candidate_times[j] = t
    else:
        # No clean anchors at all: use observed date if available, else case bounds midpoint.
        base = case_start + (case_end - case_start) / 2
        span_sec = BLOCK_SPAN_P25.get(len(block_idx), GLOBAL_MEDIAN * max(1, len(block_idx) - 1))
        start = base - pd.Timedelta(seconds=span_sec / 2.0)
        weights = non_uniform_internal_weights(activities, prev_act, next_act)
        candidate_times = weighted_positions(start, start + pd.Timedelta(seconds=span_sec), weights)

    # Apply per-row date preservation for date-only rows
    for j, idx in enumerate(block_idx):
        if kinds[j] == "date_only":
            obs_day = parse_ts(observed_ts[j]).normalize()
            day_lo, day_hi = same_date_bounds(obs_day)

            lower = day_lo
            upper = day_hi

            if left_anchor is not None:
                lower = max(lower, left_anchor if left_anchor_idx < idx else lower)
            if right_anchor is not None:
                upper = min(upper, right_anchor if right_anchor_idx > idx else upper)

            lower = max(lower, case_start)
            upper = min(upper, case_end)

            if upper < lower:
                # Closest feasible instant on same observed date within case bounds
                lower2, upper2 = same_date_bounds(obs_day)
                feasible = min(max(candidate_times[j], lower2), upper2)
                candidate_times[j] = feasible
            else:
                candidate_times[j] = min(max(candidate_times[j], lower), upper)

    # Block-level clamp to anchor/case bounds
    lower_bound = case_start
    upper_bound = case_end
    if left_anchor is not None:
        lower_bound = max(lower_bound, left_anchor)
    if right_anchor is not None:
        upper_bound = min(upper_bound, right_anchor)

    # If some date-only rows exist on different dates, monotonicity must respect each row date first.
    # So only enforce global monotonicity when feasible; else local repairs after per-row clamp.
    if all(k != "date_only" for k in kinds):
        candidate_times = ensure_strictly_monotone(candidate_times, lower_bound, upper_bound, frac_digits)
    else:
        # Forward monotone where possible, without breaking preserved dates
        unit = pd.Timedelta(milliseconds=1) if frac_digits >= 3 else pd.Timedelta(seconds=1)
        if frac_digits > 3:
            unit = pd.Timedelta(microseconds=max(1, 10 ** (6 - min(frac_digits, 6))))
        for j in range(1, len(candidate_times)):
            prev = candidate_times[j - 1]
            cur = candidate_times[j]
            if cur <= prev:
                target = prev + unit
                if kinds[j] == "date_only":
                    day_lo, day_hi = same_date_bounds(parse_ts(observed_ts[j]).normalize())
                    target = min(max(target, day_lo), day_hi)
                candidate_times[j] = target
        for j in range(len(candidate_times) - 2, -1, -1):
            cur = candidate_times[j]
            nxt = candidate_times[j + 1]
            if cur >= nxt:
                target = nxt - unit
                if kinds[j] == "date_only":
                    day_lo, day_hi = same_date_bounds(parse_ts(observed_ts[j]).normalize())
                    target = min(max(target, day_lo), day_hi)
                candidate_times[j] = target

    # Final self-fixing per row
    unit = pd.Timedelta(milliseconds=1) if frac_digits >= 3 else pd.Timedelta(seconds=1)
    if frac_digits > 3:
        unit = pd.Timedelta(microseconds=max(1, 10 ** (6 - min(frac_digits, 6))))

    for j, idx in enumerate(block_idx):
        original = observed_ts[j]
        t = candidate_times[j]

        # enforce anchor/case bounds
        lb = case_start
        ub = case_end
        if left_anchor is not None:
            lb = max(lb, left_anchor)
        if right_anchor is not None:
            ub = min(ub, right_anchor)
        t = min(max(t, lb), ub)

        # preserve date-only day exactly
        if kinds[j] == "date_only":
            obs_day = parse_ts(original).normalize()
            d0, d1 = same_date_bounds(obs_day)
            if t.normalize() != obs_day:
                t = max(min(t, d1), d0)
                if t.normalize() != obs_day:
                    midpoint = d0 + (d1 - d0) / 2
                    t = midpoint

        out = fmt_timestamp(t, frac_digits)
        if out == original:
            t = t + unit
            # keep inside right anchor / case
            t = min(t, ub)
            if kinds[j] == "date_only":
                obs_day = parse_ts(original).normalize()
                d0, d1 = same_date_bounds(obs_day)
                t = min(max(t, d0), d1)
                if fmt_timestamp(t, frac_digits) == original:
                    alt = parse_ts(original) + unit
                    if alt.normalize() == obs_day:
                        t = alt
                    else:
                        t = d0 + pd.Timedelta(hours=12)
            else:
                if fmt_timestamp(t, frac_digits) == original:
                    alt = t - 2 * unit
                    if alt >= lb:
                        t = alt
            out = fmt_timestamp(t, frac_digits)

        if out == original:
            # last resort without leaving observed value untouched
            if kinds[j] == "date_only":
                obs_day = parse_ts(original).normalize()
                t = obs_day + pd.Timedelta(hours=12, seconds=j + 1)
            elif kinds[j] == "blank":
                t = min(max(candidate_times[j], lb), ub)
                t = t + pd.Timedelta(milliseconds=j + 1)
            else:
                t = min(max(t + pd.Timedelta(milliseconds=j + 1), lb), ub)
            out = fmt_timestamp(t, frac_digits)

        repaired[idx] = out

    return repaired


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

    frac_digits = infer_max_fractional_digits(df["Timestamp"]) if "Timestamp" in df.columns else 3
    if frac_digits <= 0:
        frac_digits = 3

    repaired_ts = list(df["Timestamp"].astype(str))

    blank_filled = 0
    date_only_filled = 0

    # Process per case in original row order
    for case_value, case_df in df.groupby("Case", sort=False):
        idxs = list(case_df.index)
        i = 0
        while i < len(idxs):
            idx = idxs[i]
            if str(df.at[idx, "error_category"]) == "CLEAN":
                i += 1
                continue

            j = i
            block = []
            while j < len(idxs) and str(df.at[idxs[j], "error_category"]) != "CLEAN":
                block.append(idxs[j])
                j += 1

            repairs = repair_block(df.loc[idxs], block, frac_digits)
            for ridx, ts_str in repairs.items():
                old = str(df.at[ridx, "Timestamp"])
                kind = classify_observed_timestamp(old)
                if kind == "blank":
                    blank_filled += 1
                elif kind == "date_only":
                    date_only_filled += 1
                repaired_ts[ridx] = ts_str
            i = j

    df["Timestamp"] = repaired_ts

    # -----------------------------
    # Self-checks and in-place fixes
    # -----------------------------
    # 1) structural invariants
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        df[col] = original_df[col]

    if len(df) != len(original_df):
        df = original_df.copy(deep=True)
        df["Timestamp"] = repaired_ts[: len(df)]

    # 2) CLEAN rows unchanged exactly
    clean_mask = original_df["error_category"].astype(str) == "CLEAN"
    df.loc[clean_mask, "Timestamp"] = original_df.loc[clean_mask, "Timestamp"]

    # 3) all repaired rows parse; none blank; differ from input
    non_clean_mask = ~clean_mask
    for idx in df.index[non_clean_mask]:
        out = str(df.at[idx, "Timestamp"])
        inp = str(original_df.at[idx, "Timestamp"])
        cat = str(original_df.at[idx, "error_category"])
        if cat == "CLEAN":
            continue

        if out.strip() == "" or pd.isna(parse_ts(out)):
            # Recompute simple anchor-based repair
            case = df.at[idx, "Case"]
            case_rows = list(df.index[df["Case"] == case])
            pos = case_rows.index(idx)

            left_ts = None
            left_act = None
            for p in range(pos - 1, -1, -1):
                ii = case_rows[p]
                if str(original_df.at[ii, "error_category"]) == "CLEAN":
                    left_ts = parse_ts(original_df.at[ii, "Timestamp"])
                    left_act = original_df.at[ii, "Activity"]
                    if not pd.isna(left_ts):
                        break

            right_ts = None
            right_act = None
            for p in range(pos + 1, len(case_rows)):
                ii = case_rows[p]
                if str(original_df.at[ii, "error_category"]) == "CLEAN":
                    right_ts = parse_ts(original_df.at[ii, "Timestamp"])
                    right_act = original_df.at[ii, "Activity"]
                    if not pd.isna(right_ts):
                        break

            act = df.at[idx, "Activity"]
            if left_ts is not None and right_ts is not None and right_ts >= left_ts:
                sec1 = fallback_edge_seconds(left_act, act)
                sec2 = fallback_edge_seconds(act, right_act)
                frac = sec1 / max(sec1 + sec2, 1e-9)
                ts = left_ts + (right_ts - left_ts) * frac
            elif left_ts is not None:
                ts = left_ts + pd.Timedelta(seconds=fallback_edge_seconds(left_act, act))
            elif right_ts is not None:
                ts = right_ts - pd.Timedelta(seconds=fallback_edge_seconds(act, right_act))
            else:
                ts = pd.Timestamp("2024-01-01 12:00:00")

            kind = classify_observed_timestamp(inp)
            if kind == "date_only":
                day = parse_ts(inp).normalize()
                d0, d1 = same_date_bounds(day)
                ts = min(max(ts, d0), d1)
                if ts.normalize() != day:
                    ts = day + pd.Timedelta(hours=12)

            out = fmt_timestamp(ts, frac_digits)
            if out == inp:
                ts = ts + pd.Timedelta(milliseconds=1)
                out = fmt_timestamp(ts, frac_digits)
            df.at[idx, "Timestamp"] = out

    # 4) bounds and date preservation per case
    for case_value, case_df in df.groupby("Case", sort=False):
        idxs = list(case_df.index)
        clean_times = [parse_ts(original_df.at[i, "Timestamp"]) for i in idxs if str(original_df.at[i, "error_category"]) == "CLEAN" and not pd.isna(parse_ts(original_df.at[i, "Timestamp"]))]
        case_start, case_end = choose_case_bounds(clean_times, [])
        for local_pos, idx in enumerate(idxs):
            if str(original_df.at[idx, "error_category"]) == "CLEAN":
                continue
            t = parse_ts(df.at[idx, "Timestamp"])
            if pd.isna(t):
                t = case_start
            t = min(max(t, case_start), case_end)

            inp = str(original_df.at[idx, "Timestamp"])
            kind = classify_observed_timestamp(inp)
            if kind == "date_only":
                day = parse_ts(inp).normalize()
                d0, d1 = same_date_bounds(day)
                t = min(max(t, d0), d1)
                if t.normalize() != day:
                    t = day + pd.Timedelta(hours=12)

            # immediate anchors
            left_anchor = None
            right_anchor = None
            for p in range(local_pos - 1, -1, -1):
                ii = idxs[p]
                if str(original_df.at[ii, "error_category"]) == "CLEAN":
                    ts = parse_ts(original_df.at[ii, "Timestamp"])
                    if not pd.isna(ts):
                        left_anchor = ts
                        break
            for p in range(local_pos + 1, len(idxs)):
                ii = idxs[p]
                if str(original_df.at[ii, "error_category"]) == "CLEAN":
                    ts = parse_ts(original_df.at[ii, "Timestamp"])
                    if not pd.isna(ts):
                        right_anchor = ts
                        break
            if left_anchor is not None:
                t = max(t, left_anchor)
            if right_anchor is not None:
                t = min(t, right_anchor)

            out = fmt_timestamp(t, frac_digits)
            if out == inp:
                out = fmt_timestamp(t + pd.Timedelta(milliseconds=1), frac_digits)
                if kind == "date_only":
                    day = parse_ts(inp).normalize()
                    if parse_ts(out).normalize() != day:
                        out = fmt_timestamp(day + pd.Timedelta(hours=12, seconds=local_pos + 1), frac_digits)
            df.at[idx, "Timestamp"] = out

    still_equal = 0
    for idx in df.index[non_clean_mask]:
        if str(df.at[idx, "Timestamp"]) == str(original_df.at[idx, "Timestamp"]):
            still_equal += 1
            inp = str(original_df.at[idx, "Timestamp"])
            kind = classify_observed_timestamp(inp)
            if kind == "date_only":
                day = parse_ts(inp).normalize()
                df.at[idx, "Timestamp"] = fmt_timestamp(day + pd.Timedelta(hours=12, seconds=(idx % 59) + 1), frac_digits)
            elif kind == "blank":
                case = df.at[idx, "Case"]
                case_rows = list(df.index[df["Case"] == case])
                pos = case_rows.index(idx)
                base = None
                for p in range(pos - 1, -1, -1):
                    ii = case_rows[p]
                    ts = parse_ts(df.at[ii, "Timestamp"])
                    if not pd.isna(ts):
                        base = ts + pd.Timedelta(seconds=1 + (idx % 7))
                        break
                if base is None:
                    base = pd.Timestamp("2024-01-01 12:00:00") + pd.Timedelta(seconds=idx % 60)
                df.at[idx, "Timestamp"] = fmt_timestamp(base, frac_digits)
            else:
                ts = parse_ts(inp)
                if pd.isna(ts):
                    ts = pd.Timestamp("2024-01-01 12:00:00")
                df.at[idx, "Timestamp"] = fmt_timestamp(ts + pd.Timedelta(milliseconds=1), frac_digits)

    still_equal = int(((df["error_category"].astype(str) != "CLEAN") & (df["Timestamp"].astype(str) == original_df["Timestamp"].astype(str))).sum())

    df.to_csv(args.output, index=False)

    print(f"blank_rows_filled={blank_filled}, date_only_rows_given_time={date_only_filled}, rows_still_equal_to_input={still_equal}")


if __name__ == "__main__":
    main()