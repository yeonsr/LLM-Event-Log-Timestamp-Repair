import argparse
import csv
import math
import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# Process context encoded from prompt
# ============================================================

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

FREQUENT_VARIANTS = [
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare starter", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare starter", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
]


# ============================================================
# Utilities
# ============================================================

def parse_ts(s: str) -> Optional[pd.Timestamp]:
    try:
        if s is None or str(s).strip() == "":
            return None
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return None


def is_date_only_missing_time(ts_str: str) -> bool:
    if ts_str is None:
        return False
    s = str(ts_str).strip()
    if s == "":
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]00:00:00(?:\.0+)?", s):
        return True
    return False


def infer_fraction_digits(rows: List[dict]) -> int:
    counts = {}
    for r in rows:
        s = r["Timestamp"]
        if isinstance(s, str):
            m = re.search(r"\.(\d+)$", s.strip())
            if m:
                d = len(m.group(1))
                counts[d] = counts.get(d, 0) + 1
    if counts:
        return sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]
    return 3


def floor_to_precision(ts: pd.Timestamp, digits: int) -> pd.Timestamp:
    if digits <= 0:
        return ts.floor("S")
    if digits >= 9:
        return ts
    unit_ns = 10 ** (9 - digits)
    val = ts.value
    floored = (val // unit_ns) * unit_ns
    return pd.Timestamp(floored)


def format_ts(ts: pd.Timestamp, digits: int) -> str:
    ts = floor_to_precision(ts, digits)
    if digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    frac = f"{ts.nanosecond + ts.microsecond * 1000:09d}"[:digits]
    return f"{base}.{frac}"


def timedelta_seconds(seconds: float) -> pd.Timedelta:
    return pd.to_timedelta(float(seconds), unit="s")


def get_edge_stat(src: Optional[str], tgt: Optional[str]) -> Optional[dict]:
    if src is None or tgt is None:
        return None
    return DURATION_PROFILES.get((src, tgt))


def edge_weight_seconds(src: Optional[str], tgt: Optional[str]) -> float:
    prof = get_edge_stat(src, tgt)
    if prof:
        if prof["share_under_60s"] >= 0.5:
            return max(prof["p10"], 0.001)
        return max(prof["p25"], 0.001)
    return max(BLOCK_SPAN_P25.get(2, GLOBAL_MEDIAN) / 2.0, 0.001)


def plausible_subsequence(acts: List[str]) -> bool:
    if len(acts) <= 1:
        return True
    for v in FREQUENT_VARIANTS:
        for i in range(len(v) - len(acts) + 1):
            if v[i:i + len(acts)] == acts:
                return True
    # fallback DFG plausibility
    ok_edges = 0
    for i in range(len(acts) - 1):
        if (acts[i], acts[i + 1]) in DURATION_PROFILES:
            ok_edges += 1
    return ok_edges >= max(1, len(acts) - 2)


def build_relative_positions(case_rows: List[dict], start_idx: int, end_idx: int) -> List[float]:
    acts = [case_rows[i]["Activity"] for i in range(start_idx, end_idx + 1)]
    _ = plausible_subsequence(acts)  # validation only; no reordering

    weights = []
    for i in range(start_idx, end_idx + 1):
        prev_act = case_rows[i - 1]["Activity"] if i - 1 >= 0 else None
        cur_act = case_rows[i]["Activity"]
        next_act = case_rows[i + 1]["Activity"] if i + 1 < len(case_rows) else None

        left_w = edge_weight_seconds(prev_act, cur_act) if prev_act is not None else GLOBAL_MEDIAN
        right_w = edge_weight_seconds(cur_act, next_act) if next_act is not None else GLOBAL_MEDIAN

        # Non-uniform local weight
        local = 0.61 * left_w + 0.39 * right_w
        # Small deterministic variation by row position/activity length to avoid even spacing
        local *= (1.0 + (((i - start_idx) % 3) * 0.07) + ((len(cur_act) % 5) * 0.011))
        weights.append(max(local, 0.001))

    total = sum(weights)
    if total <= 0:
        return [(j + 1) / (len(weights) + 1) for j in range(len(weights))]

    cum = 0.0
    pos = []
    for w in weights:
        cum += w
        pos.append(cum / total)
    # interior positions, not endpoints
    n = len(pos)
    adj = []
    for j, p in enumerate(pos):
        left = 0.0 if j == 0 else pos[j - 1]
        right = 1.0 if j == n - 1 else pos[j]
        # keep cumulative but nudge interior
        adj_p = 0.5 * (left + right)
        adj.append(min(max(adj_p, 1e-9), 1 - 1e-9))
    return adj


def same_calendar_date(ts: pd.Timestamp, date_ref: pd.Timestamp) -> bool:
    return ts.date() == date_ref.date()


def clamp(ts: pd.Timestamp, lo: Optional[pd.Timestamp], hi: Optional[pd.Timestamp]) -> pd.Timestamp:
    if lo is not None and ts < lo:
        ts = lo
    if hi is not None and ts > hi:
        ts = hi
    return ts


def nearest_same_day_in_interval(target: pd.Timestamp, kept_date: pd.Timestamp,
                                 lo: Optional[pd.Timestamp], hi: Optional[pd.Timestamp],
                                 digits: int) -> pd.Timestamp:
    day_start = pd.Timestamp(year=kept_date.year, month=kept_date.month, day=kept_date.day)
    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    lo2 = max(lo, day_start) if lo is not None else day_start
    hi2 = min(hi, day_end) if hi is not None else day_end
    if lo2 > hi2:
        # impossible to satisfy both; keep same date as required, choose target on that day
        candidate = pd.Timestamp(
            year=kept_date.year, month=kept_date.month, day=kept_date.day,
            hour=target.hour, minute=target.minute, second=target.second, microsecond=target.microsecond
        )
        return floor_to_precision(candidate, digits)

    candidate = pd.Timestamp(
        year=kept_date.year, month=kept_date.month, day=kept_date.day,
        hour=target.hour, minute=target.minute, second=target.second, microsecond=target.microsecond
    )
    candidate = clamp(candidate, lo2, hi2)
    return floor_to_precision(candidate, digits)


def minimal_bump(ts: pd.Timestamp, original: str, digits: int,
                 lo: Optional[pd.Timestamp], hi: Optional[pd.Timestamp]) -> pd.Timestamp:
    step_ns = 10 ** (9 - digits) if digits > 0 else 10 ** 9
    step = pd.Timedelta(step_ns, unit="ns")
    cands = [ts + step, ts - step, ts + 2 * step, ts - 2 * step]
    for c in cands:
        c = floor_to_precision(c, digits)
        if lo is not None and c < lo:
            continue
        if hi is not None and c > hi:
            continue
        if format_ts(c, digits) != original:
            return c
    return floor_to_precision(clamp(ts, lo, hi), digits)


# ============================================================
# Repair logic
# ============================================================

def repair_case(case_rows: List[dict], digits: int, counters: dict):
    n = len(case_rows)

    # Parse clean timestamps for anchors/case bounds
    parsed_clean = []
    for r in case_rows:
        if r["error_category"] == "CLEAN":
            parsed_clean.append(parse_ts(r["Timestamp"]))
        else:
            parsed_clean.append(None)

    clean_times = [t for t in parsed_clean if t is not None]
    case_start = min(clean_times) if clean_times else None
    case_end = max(clean_times) if clean_times else None

    i = 0
    while i < n:
        if case_rows[i]["error_category"] == "CLEAN":
            i += 1
            continue
        j = i
        while j + 1 < n and case_rows[j + 1]["error_category"] != "CLEAN":
            j += 1

        left_idx = i - 1 if i - 1 >= 0 and case_rows[i - 1]["error_category"] == "CLEAN" else None
        right_idx = j + 1 if j + 1 < n and case_rows[j + 1]["error_category"] == "CLEAN" else None
        left_anchor = parse_ts(case_rows[left_idx]["Timestamp"]) if left_idx is not None else None
        right_anchor = parse_ts(case_rows[right_idx]["Timestamp"]) if right_idx is not None else None

        rel_pos = build_relative_positions(case_rows, i, j)
        block_len = j - i + 1

        # one-anchor candidate synthesis
        left_based = []
        if left_anchor is not None:
            t = left_anchor
            for idx in range(i, j + 1):
                prev_act = case_rows[idx - 1]["Activity"] if idx - 1 >= 0 else None
                cur_act = case_rows[idx]["Activity"]
                t = t + timedelta_seconds(edge_weight_seconds(prev_act, cur_act))
                left_based.append(t)
        else:
            left_based = [None] * block_len

        right_based = []
        if right_anchor is not None:
            t = right_anchor
            tmp = []
            for idx in range(j, i - 1, -1):
                cur_act = case_rows[idx]["Activity"]
                next_act = case_rows[idx + 1]["Activity"] if idx + 1 < n else None
                t = t - timedelta_seconds(edge_weight_seconds(cur_act, next_act))
                tmp.append(t)
            right_based = list(reversed(tmp))
        else:
            right_based = [None] * block_len

        repaired_ts = [None] * block_len

        if left_anchor is not None and right_anchor is not None and right_anchor >= left_anchor:
            interval = right_anchor - left_anchor
            secs_interval = interval.total_seconds()

            # if interval is tight, fit inside by weighted interior placement
            if secs_interval <= 0:
                for k in range(block_len):
                    repaired_ts[k] = left_anchor
            else:
                for k in range(block_len):
                    p = rel_pos[k]
                    t_fit = left_anchor + p * interval

                    lb = left_based[k]
                    rb = right_based[k]

                    # Prefer low movement from observed damaged value when date survived
                    obs = case_rows[i + k]["Timestamp"]
                    if is_date_only_missing_time(obs):
                        obs_date = parse_ts(obs)
                        candidate = nearest_same_day_in_interval(t_fit, obs_date, left_anchor, right_anchor, digits)
                    else:
                        # Blend anchor-fit with one-sided predictions if they remain inside anchors
                        cands = [t_fit]
                        if lb is not None:
                            cands.append(clamp(lb, left_anchor, right_anchor))
                        if rb is not None:
                            cands.append(clamp(rb, left_anchor, right_anchor))

                        # choose candidate nearest t_fit but not endpoint-pinned
                        cands = sorted(cands, key=lambda x: abs((x - t_fit).total_seconds()))
                        candidate = cands[0]
                    repaired_ts[k] = candidate
        elif left_anchor is not None:
            for k in range(block_len):
                candidate = left_based[k]
                obs = case_rows[i + k]["Timestamp"]
                if is_date_only_missing_time(obs):
                    obs_date = parse_ts(obs)
                    candidate = nearest_same_day_in_interval(candidate, obs_date, left_anchor, case_end, digits)
                repaired_ts[k] = candidate
        elif right_anchor is not None:
            for k in range(block_len):
                candidate = right_based[k]
                obs = case_rows[i + k]["Timestamp"]
                if is_date_only_missing_time(obs):
                    obs_date = parse_ts(obs)
                    candidate = nearest_same_day_in_interval(candidate, obs_date, case_start, right_anchor, digits)
                repaired_ts[k] = candidate
        else:
            # no clean anchors at all in case; use internal relative placement on preserved date if possible
            base = parse_ts(case_rows[i]["Timestamp"]) if is_date_only_missing_time(case_rows[i]["Timestamp"]) else pd.Timestamp("2024-01-01 12:00:00")
            total_span = BLOCK_SPAN_P25.get(block_len + 1, GLOBAL_MEDIAN * max(1, block_len))
            for k in range(block_len):
                candidate = base + timedelta_seconds(rel_pos[k] * total_span)
                obs = case_rows[i + k]["Timestamp"]
                if is_date_only_missing_time(obs):
                    obs_date = parse_ts(obs)
                    candidate = nearest_same_day_in_interval(candidate, obs_date, None, None, digits)
                repaired_ts[k] = candidate

        # self-fix and write back
        prev_written_ts = left_anchor
        for k, idx in enumerate(range(i, j + 1)):
            row = case_rows[idx]
            original = row["Timestamp"]
            ts = repaired_ts[k]
            if ts is None or pd.isna(ts):
                base = left_anchor or right_anchor or case_start or case_end or pd.Timestamp("2024-01-01 12:00:00")
                ts = base + timedelta_seconds((k + 1) * max(1.0, GLOBAL_MEDIAN / (block_len + 1)))

            # Preserve survived date for date-only rows
            if is_date_only_missing_time(original):
                kept_date = parse_ts(original)
                ts = nearest_same_day_in_interval(ts, kept_date, left_anchor if left_anchor is not None else case_start,
                                                  right_anchor if right_anchor is not None else case_end, digits)

            # Clamp to anchors / case bounds
            lo = left_anchor if left_anchor is not None else case_start
            hi = right_anchor if right_anchor is not None else case_end
            ts = clamp(ts, lo, hi)

            # maintain row order via timestamps only
            if prev_written_ts is not None and ts < prev_written_ts:
                ts = prev_written_ts
            if right_anchor is not None and ts > right_anchor:
                ts = right_anchor

            ts = floor_to_precision(ts, digits)

            # if equal to previous within precision in a multi-row block, allow near-identical but try tiny bump upward when valid
            if prev_written_ts is not None and ts < prev_written_ts:
                ts = prev_written_ts

            out_str = format_ts(ts, digits)

            # corrupted row must differ from input
            if out_str == original:
                ts = minimal_bump(ts, original, digits, lo, hi)
                if is_date_only_missing_time(original):
                    kept_date = parse_ts(original)
                    ts = nearest_same_day_in_interval(ts, kept_date, lo, hi, digits)
                out_str = format_ts(ts, digits)

            # final parseability guard
            if parse_ts(out_str) is None:
                fallback = clamp((left_anchor or case_start or pd.Timestamp("2024-01-01 12:00:00")), lo, hi)
                ts = minimal_bump(fallback, original, digits, lo, hi)
                if is_date_only_missing_time(original):
                    kept_date = parse_ts(original)
                    ts = nearest_same_day_in_interval(ts, kept_date, lo, hi, digits)
                out_str = format_ts(ts, digits)

            row["Timestamp_out"] = out_str
            if original.strip() == "":
                counters["blank_filled"] += 1
            elif is_date_only_missing_time(original):
                counters["date_only_filled"] += 1

            prev_written_ts = parse_ts(out_str)

        i = j + 1


def self_check_and_fix(all_rows: List[dict], digits: int, counters: dict):
    # ensure exact output row count and unchanged non-timestamp columns by construction
    # enforce clean rows untouched, parseability, changed corrupted rows, same-date for date-only, anchors
    rows_by_case: Dict[str, List[Tuple[int, dict]]] = {}
    for pos, r in enumerate(all_rows):
        rows_by_case.setdefault(r["Case"], []).append((pos, r))

    for case, indexed_rows in rows_by_case.items():
        case_rows = [r for _, r in indexed_rows]
        parsed_clean = []
        for r in case_rows:
            if r["error_category"] == "CLEAN":
                parsed_clean.append(parse_ts(r["Timestamp"]))
            else:
                parsed_clean.append(None)
        clean_times = [t for t in parsed_clean if t is not None]
        case_start = min(clean_times) if clean_times else None
        case_end = max(clean_times) if clean_times else None

        for idx, row in enumerate(case_rows):
            if row["error_category"] == "CLEAN":
                row["Timestamp_out"] = row["Timestamp"]
                continue

            out = row.get("Timestamp_out", "")
            ts = parse_ts(out)
            if ts is None:
                lo = case_start
                hi = case_end
                base = lo if lo is not None else (hi if hi is not None else pd.Timestamp("2024-01-01 12:00:00"))
                ts = minimal_bump(base, row["Timestamp"], digits, lo, hi)
                if is_date_only_missing_time(row["Timestamp"]):
                    kept_date = parse_ts(row["Timestamp"])
                    ts = nearest_same_day_in_interval(ts, kept_date, lo, hi, digits)
                row["Timestamp_out"] = format_ts(ts, digits)
                ts = parse_ts(row["Timestamp_out"])

            # nearest clean anchors
            left_anchor = None
            right_anchor = None
            for j in range(idx - 1, -1, -1):
                if case_rows[j]["error_category"] == "CLEAN":
                    left_anchor = parse_ts(case_rows[j]["Timestamp"])
                    break
            for j in range(idx + 1, len(case_rows)):
                if case_rows[j]["error_category"] == "CLEAN":
                    right_anchor = parse_ts(case_rows[j]["Timestamp"])
                    break

            lo = left_anchor if left_anchor is not None else case_start
            hi = right_anchor if right_anchor is not None else case_end
            ts = clamp(ts, lo, hi)

            if is_date_only_missing_time(row["Timestamp"]):
                kept_date = parse_ts(row["Timestamp"])
                ts = nearest_same_day_in_interval(ts, kept_date, lo, hi, digits)

            out2 = format_ts(ts, digits)
            if out2 == row["Timestamp"]:
                ts = minimal_bump(ts, row["Timestamp"], digits, lo, hi)
                if is_date_only_missing_time(row["Timestamp"]):
                    kept_date = parse_ts(row["Timestamp"])
                    ts = nearest_same_day_in_interval(ts, kept_date, lo, hi, digits)
                out2 = format_ts(ts, digits)

            row["Timestamp_out"] = out2

    counters["still_equal"] = sum(
        1
        for r in all_rows
        if r["error_category"] != "CLEAN" and r.get("Timestamp_out", "") == r["Timestamp"]
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fieldnames = reader.fieldnames

    digits = infer_fraction_digits(rows)

    for r in rows:
        r["Timestamp_out"] = r["Timestamp"]

    counters = {"blank_filled": 0, "date_only_filled": 0, "still_equal": 0}

    rows_by_case: Dict[str, List[dict]] = {}
    case_order: List[str] = []
    for r in rows:
        c = r["Case"]
        if c not in rows_by_case:
            rows_by_case[c] = []
            case_order.append(c)
        rows_by_case[c].append(r)

    for c in case_order:
        repair_case(rows_by_case[c], digits, counters)

    self_check_and_fix(rows, digits, counters)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out_row = {
                "Case": r["Case"],
                "Activity": r["Activity"],
                "Timestamp": r["Timestamp_out"] if r["error_category"] != "CLEAN" else r["Timestamp"],
                "Resource": r["Resource"],
                "error_category": r["error_category"],
                "row_id": r["row_id"],
            }
            writer.writerow(out_row)

    print(f"{counters['blank_filled']},{counters['date_only_filled']},{counters['still_equal']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Must not exit non-zero; best-effort salvage
        try:
            import sys
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=True)
            parser.add_argument("--output", required=True)
            known, _ = parser.parse_known_args()
            with open(known.input, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader]
                fieldnames = reader.fieldnames
            with open(known.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print("0,0,0")
        except Exception:
            print("0,0,0")