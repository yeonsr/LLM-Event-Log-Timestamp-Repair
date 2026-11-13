import argparse
import csv
import math
import sys
from collections import defaultdict, Counter
from datetime import timedelta
from itertools import permutations

import pandas as pd


TRANSITION_PROB = {
    ("Deliver card", "EVENT 13 END"): 1.0,
    ("EVENT 13 END", "EVENT 13 END"): 1.0,
    ("Notify accept", "Deliver card"): 1.0,
    ("time out", "EVENT 13 END"): 1.0,
    ("Perform checks", "Make decision"): 0.9919,
    ("review request received", "Check for completeness"): 0.9708,
    ("info received", "Check for completeness"): 0.9662,
    ("Request info", "info received"): 0.962,
    ("New online application received", "Check for completeness"): 0.91,
    ("notify reject", "time out"): 0.5226,
    ("Check for completeness", "Request info"): 0.5124,
    ("Make decision", "notify reject"): 0.5079,
    ("Make decision", "Notify accept"): 0.4883,
    ("notify reject", "review request received"): 0.465,
    ("Check for completeness", "Perform checks"): 0.4649,
    ("New online application received", "Notify accept"): 0.0276,
    ("New online application received", "Make decision"): 0.0234,
    ("New online application received", "notify reject"): 0.0164,
    ("info received", "Make decision"): 0.0106,
    ("New online application received", "info received"): 0.0106,
    ("Check for completeness", "Notify accept"): 0.0085,
    ("Request info", "Perform checks"): 0.0084,
    ("info received", "info received"): 0.0084,
    ("Request info", "Make decision"): 0.0079,
    ("review request received", "Make decision"): 0.0076,
    ("Request info", "Request info"): 0.0068,
    ("New online application received", "Perform checks"): 0.0066,
    ("Check for completeness", "Make decision"): 0.0059,
    ("review request received", "Notify accept"): 0.0057,
    ("review request received", "Perform checks"): 0.0057,
    ("Request info", "Check for completeness"): 0.0054,
    ("Check for completeness", "notify reject"): 0.0049,
    ("Request info", "Notify accept"): 0.0047,
    ("Request info", "notify reject"): 0.0047,
    ("info received", "notify reject"): 0.0047,
    ("info received", "Perform checks"): 0.0043,
    ("info received", "Notify accept"): 0.0039,
    ("notify reject", "Make decision"): 0.0036,
    ("New online application received", "Request info"): 0.0034,
    ("review request received", "Request info"): 0.0032,
    ("review request received", "notify reject"): 0.0032,
    ("notify reject", "Perform checks"): 0.0027,
    ("Perform checks", "Check for completeness"): 0.0026,
    ("Perform checks", "review request received"): 0.0026,
    ("review request received", "info received"): 0.0025,
    ("New online application received", "review request received"): 0.002,
    ("Perform checks", "Request info"): 0.0019,
    ("notify reject", "Check for completeness"): 0.0018,
    ("Check for completeness", "info received"): 0.0016,
    ("notify reject", "notify reject"): 0.0015,
    ("info received", "Request info"): 0.0014,
    ("review request received", "review request received"): 0.0013,
    ("Make decision", "info received"): 0.0012,
    ("Check for completeness", "Check for completeness"): 0.0011,
    ("Perform checks", "Perform checks"): 0.001,
    ("Make decision", "Make decision"): 0.001,
    ("notify reject", "Notify accept"): 0.0009,
    ("notify reject", "Request info"): 0.0009,
    ("notify reject", "info received"): 0.0009,
    ("Make decision", "Perform checks"): 0.0008,
}

DURATION = {
    ("Check for completeness", "Request info"): {"n": 6227, "p10": 61.9314, "p25": 170.1115, "median": 418.728, "share60": 0.097},
    ("Request info", "info received"): {"n": 6080, "p10": 10.0, "p25": 10.0, "median": 10.0, "share60": 0.996},
    ("info received", "Check for completeness"): {"n": 6008, "p10": 63.739, "p25": 176.05, "median": 420.682, "share60": 0.095},
    ("Perform checks", "Make decision"): {"n": 5745, "p10": 62.8388, "p25": 171.483, "median": 422.924, "share60": 0.095},
    ("Check for completeness", "Perform checks"): {"n": 5650, "p10": 61.0229, "p25": 171.5965, "median": 420.3715, "share60": 0.098},
    ("New online application received", "Check for completeness"): {"n": 4550, "p10": 65.3893, "p25": 171.7955, "median": 426.0135, "share60": 0.092},
    ("Deliver card", "EVENT 13 END"): {"n": 3276, "p10": 0.0, "p25": 0.0, "median": 0.0, "share60": 1.0},
    ("Notify accept", "Deliver card"): {"n": 3276, "p10": 59.4585, "p25": 176.35075, "median": 425.762, "share60": 0.101},
    ("Make decision", "notify reject"): {"n": 3088, "p10": 63.7974, "p25": 174.02075, "median": 426.0405, "share60": 0.095},
    ("Make decision", "Notify accept"): {"n": 2969, "p10": 65.5868, "p25": 174.826, "median": 428.03, "share60": 0.089},
    ("time out", "EVENT 13 END"): {"n": 1724, "p10": 0.0, "p25": 0.0, "median": 0.0, "share60": 1.0},
    ("notify reject", "time out"): {"n": 1724, "p10": 10.0, "p25": 10.0, "median": 10.0, "share60": 1.0},
    ("notify reject", "review request received"): {"n": 1534, "p10": 10.0, "p25": 10.0, "median": 10.0, "share60": 1.0},
    ("review request received", "Check for completeness"): {"n": 1527, "p10": 78.0242, "p25": 188.8875, "median": 441.348, "share60": 0.075},
    ("New online application received", "Notify accept"): {"n": 138, "p10": 944.0129, "p25": 1469.49875, "median": 2020.112, "share60": 0.0},
    ("New online application received", "Make decision"): {"n": 117, "p10": 634.1, "p25": 963.121, "median": 1661.293, "share60": 0.0},
    ("Check for completeness", "Notify accept"): {"n": 103, "p10": 711.8126, "p25": 1094.916, "median": 1780.137, "share60": 0.0},
    ("New online application received", "notify reject"): {"n": 82, "p10": 1132.4149, "p25": 1538.03275, "median": 2159.5775, "share60": 0.0},
    ("Check for completeness", "Make decision"): {"n": 72, "p10": 1072.0155, "p25": 1536.0985, "median": 2023.867, "share60": 0.0},
    ("info received", "Make decision"): {"n": 66, "p10": 721.489, "p25": 1122.10325, "median": 1868.191, "share60": 0.0},
    ("Check for completeness", "notify reject"): {"n": 60, "p10": 945.7493, "p25": 1361.11625, "median": 1757.751, "share60": 0.0},
    ("Request info", "Perform checks"): {"n": 53, "p10": 403.0504, "p25": 643.92, "median": 928.08, "share60": 0.0},
    ("New online application received", "info received"): {"n": 53, "p10": 452.3138, "p25": 639.021, "median": 857.047, "share60": 0.0},
    ("info received", "info received"): {"n": 52, "p10": 613.9782, "p25": 895.2945, "median": 1302.5635, "share60": 0.0},
    ("Request info", "Make decision"): {"n": 50, "p10": 669.1557, "p25": 855.869, "median": 1523.8755, "share60": 0.0},
    ("Request info", "Request info"): {"n": 43, "p10": 569.524, "p25": 796.317, "median": 1164.851, "share60": 0.0},
    ("Request info", "Check for completeness"): {"n": 34, "p10": 803.7977, "p25": 952.73675, "median": 1889.409, "share60": 0.0},
    ("New online application received", "Perform checks"): {"n": 33, "p10": 1091.8104, "p25": 1559.19, "median": 2363.648, "share60": 0.0},
    ("Request info", "Notify accept"): {"n": 30, "p10": 1429.9083, "p25": 1610.589, "median": 2396.886, "share60": 0.0},
    ("Request info", "notify reject"): {"n": 30, "p10": 1539.1919, "p25": 1979.3755, "median": 2454.903, "share60": 0.0},
    ("info received", "notify reject"): {"n": 29, "p10": 1059.5184, "p25": 1682.395, "median": 2806.157, "share60": 0.0},
    ("info received", "Perform checks"): {"n": 27, "p10": 809.8986, "p25": 1087.906, "median": 1874.355, "share60": 0.0},
    ("info received", "Notify accept"): {"n": 24, "p10": 1006.2042, "p25": 1510.50375, "median": 2230.734, "share60": 0.0},
    ("Check for completeness", "info received"): {"n": 19, "p10": 817.942, "p25": 1530.727, "median": 1910.593, "share60": 0.0},
    ("New online application received", "Request info"): {"n": 17, "p10": 809.1352, "p25": 1687.418, "median": 2254.706, "share60": 0.0},
    ("Perform checks", "Check for completeness"): {"n": 15, "p10": 1023.0088, "p25": 1420.4865, "median": 1995.413, "share60": 0.0},
    ("Perform checks", "review request received"): {"n": 15, "p10": 378.412, "p25": 555.4955, "median": 705.0, "share60": 0.0},
    ("Check for completeness", "Check for completeness"): {"n": 13, "p10": 1452.7526, "p25": 1939.982, "median": 2577.014, "share60": 0.0},
    ("notify reject", "Make decision"): {"n": 12, "p10": 785.8718, "p25": 1341.0605, "median": 1518.5795, "share60": 0.0},
    ("review request received", "Make decision"): {"n": 12, "p10": 727.506, "p25": 964.28475, "median": 2227.617, "share60": 0.0},
    ("Perform checks", "Request info"): {"n": 11, "p10": 1288.529, "p25": 1794.608, "median": 2625.632, "share60": 0.0},
    ("New online application received", "review request received"): {"n": 10, "p10": 783.2576, "p25": 1569.0395, "median": 1845.21, "share60": 0.0},
    ("notify reject", "Perform checks"): {"n": 9, "p10": 318.1018, "p25": 699.898, "median": 874.95, "share60": 0.0},
    ("review request received", "Notify accept"): {"n": 9, "p10": 2066.8628, "p25": 2191.667, "median": 2296.639, "share60": 0.0},
    ("Check for completeness", "review request received"): {"n": 9, "p10": 786.9502, "p25": 1083.927, "median": 1363.124, "share60": 0.0},
    ("review request received", "Perform checks"): {"n": 9, "p10": 1338.8826, "p25": 1593.822, "median": 2249.376, "share60": 0.0},
    ("info received", "Request info"): {"n": 9, "p10": 824.7008, "p25": 1305.603, "median": 1993.345, "share60": 0.0},
    ("EVENT 13 END", "EVENT 13 END"): {"n": 8, "p10": 0.0, "p25": 0.0, "median": 0.0, "share60": 1.0},
    ("Make decision", "info received"): {"n": 7, "p10": 615.6284, "p25": 944.3135, "median": 1863.824, "share60": 0.0},
    ("notify reject", "Check for completeness"): {"n": 6, "p10": 1769.177, "p25": 2725.29575, "median": 3075.0545, "share60": 0.0},
    ("Make decision", "Make decision"): {"n": 6, "p10": 1349.042, "p25": 1693.34325, "median": 2015.0765, "share60": 0.0},
    ("Perform checks", "Perform checks"): {"n": 6, "p10": 1054.435, "p25": 1415.281, "median": 2294.3645, "share60": 0.0},
    ("review request received", "notify reject"): {"n": 5, "p10": 1334.3154, "p25": 1360.086, "median": 1469.711, "share60": 0.0},
    ("review request received", "Request info"): {"n": 5, "p10": 1863.9608, "p25": 2128.058, "median": 2501.573, "share60": 0.0},
    ("Make decision", "Request info"): {"n": 5, "p10": 1027.2532, "p25": 1078.57, "median": 1325.858, "share60": 0.0},
    ("notify reject", "notify reject"): {"n": 5, "p10": 1475.265, "p25": 2749.563, "median": 2967.996, "share60": 0.0},
    ("Make decision", "Perform checks"): {"n": 5, "p10": 783.9354, "p25": 796.233, "median": 823.397, "share60": 0.0},
    ("review request received", "info received"): {"n": 4, "p10": 1391.3396, "p25": 1447.0055, "median": 1983.33, "share60": 0.0},
    ("notify reject", "info received"): {"n": 3, "p10": 652.6522, "p25": 721.4095, "median": 836.005, "share60": 0.0},
    ("notify reject", "Request info"): {"n": 3, "p10": 389.1752, "p25": 613.13, "median": 986.388, "share60": 0.0},
}

GLOBAL_FAST_DEFAULT = 10.0
GLOBAL_MEDIAN = 243.721


def safe_parse(ts):
    if ts is None:
        return pd.NaT
    s = str(ts).strip()
    if s == "":
        return pd.NaT
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def frac_digits(ts_str):
    if ts_str is None:
        return 3
    s = str(ts_str)
    if "." in s:
        tail = s.split(".")[-1]
        return len(tail)
    return 0


def format_dt(dt, digits):
    if digits <= 0:
        return pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")
    base = pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S.%f")
    return base[:-max(0, 6 - digits)] if digits < 6 else base


def transition_prob(a, b):
    return TRANSITION_PROB.get((a, b), 1e-6)


def duration_profile(a, b):
    return DURATION.get((a, b))


def preferred_gap_seconds(a, b):
    prof = duration_profile(a, b)
    if prof:
        if prof["share60"] >= 0.5:
            return max(0.0, prof["p10"])
        return max(0.0, prof["p25"])
    return GLOBAL_FAST_DEFAULT


def gap_support_score(a, b):
    prof = duration_profile(a, b)
    if not prof:
        return -5.0
    return math.log(max(prof["n"], 1))


def enumerate_or_beam(activities, left_act, right_act, beam_width=50):
    k = len(activities)
    idxs = list(range(k))
    if k <= 7:
        best = None
        best_score = -1e100
        for perm in permutations(idxs):
            seq = [activities[i] for i in perm]
            score = score_order(seq, left_act, right_act)
            if score > best_score:
                best_score = score
                best = perm
        return list(best)
    partials = [([], set(), 0.0)]
    for _ in range(k):
        new_partials = []
        for seq_idx, used, sc in partials:
            for i in idxs:
                if i in used:
                    continue
                act = activities[i]
                add = 0.0
                if not seq_idx:
                    if left_act is not None:
                        add += math.log(max(transition_prob(left_act, act), 1e-12))
                        add += 0.02 * gap_support_score(left_act, act)
                else:
                    prev = activities[seq_idx[-1]]
                    add += math.log(max(transition_prob(prev, act), 1e-12))
                    add += 0.02 * gap_support_score(prev, act)
                new_partials.append((seq_idx + [i], used | {i}, sc + add))
        new_partials.sort(key=lambda x: x[2], reverse=True)
        partials = new_partials[:beam_width]
    best = None
    best_score = -1e100
    for seq_idx, used, sc in partials:
        seq = [activities[i] for i in seq_idx]
        s2 = sc
        if right_act is not None:
            s2 += math.log(max(transition_prob(seq[-1], right_act), 1e-12))
            s2 += 0.02 * gap_support_score(seq[-1], right_act)
        if s2 > best_score:
            best_score = s2
            best = seq_idx
    return best


def score_order(seq, left_act, right_act):
    score = 0.0
    if left_act is not None and seq:
        score += math.log(max(transition_prob(left_act, seq[0]), 1e-12))
        score += 0.02 * gap_support_score(left_act, seq[0])
    for a, b in zip(seq, seq[1:]):
        score += math.log(max(transition_prob(a, b), 1e-12))
        score += 0.02 * gap_support_score(a, b)
    if right_act is not None and seq:
        score += math.log(max(transition_prob(seq[-1], right_act), 1e-12))
        score += 0.02 * gap_support_score(seq[-1], right_act)
    return score


def get_case_bounds(parsed_list):
    vals = [t for t in parsed_list if pd.notna(t)]
    if not vals:
        return None, None
    return min(vals), max(vals)


def nearest_clean_left(case_rows, idx):
    for j in range(idx - 1, -1, -1):
        if case_rows[j]["error_category"] == "CLEAN":
            t = case_rows[j]["parsed"]
            if pd.notna(t):
                return j, case_rows[j]
    return None, None


def nearest_clean_right(case_rows, idx):
    for j in range(idx + 1, len(case_rows)):
        if case_rows[j]["error_category"] == "CLEAN":
            t = case_rows[j]["parsed"]
            if pd.notna(t):
                return j, case_rows[j]
    return None, None


def clamp_ts(ts, lo, hi):
    if lo is not None and pd.notna(lo) and ts < lo:
        ts = lo
    if hi is not None and pd.notna(hi) and ts > hi:
        ts = hi
    return ts


def build_block_times(start_ts, ordered_acts, right_anchor_ts=None):
    times = [start_ts]
    gaps = []
    for a, b in zip(ordered_acts, ordered_acts[1:]):
        g = preferred_gap_seconds(a, b)
        if g <= 0:
            g = 1.0
        gaps.append(g)
    if not gaps:
        return times, False
    raw_span = sum(gaps)
    min_span = len(ordered_acts) - 1
    clamped = False
    if right_anchor_ts is not None and pd.notna(right_anchor_ts):
        room = max((right_anchor_ts - start_ts).total_seconds(), 0.0)
        max_span = max(min_span, room)
        if raw_span > max_span:
            clamped = True
            if raw_span > 0:
                scale = max_span / raw_span
                gaps = [max(1.0, g * scale) for g in gaps]
            span2 = sum(gaps)
            if span2 > max_span:
                extra = span2 - max_span
                reducible = sum(max(0.0, g - 1.0) for g in gaps)
                if reducible > 0:
                    new_gaps = []
                    for g in gaps:
                        rg = max(0.0, g - 1.0)
                        cut = extra * (rg / reducible)
                        new_gaps.append(max(1.0, g - cut))
                    gaps = new_gaps
    cur = start_ts
    for g in gaps:
        sec = max(1.0, float(g))
        cur = cur + pd.to_timedelta(sec, unit="s")
        times.append(cur)
    if right_anchor_ts is not None and pd.notna(right_anchor_ts):
        for i in range(1, len(times)):
            if times[i] > right_anchor_ts:
                clamped = True
                times[i] = min(right_anchor_ts, times[i])
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                clamped = True
                times[i] = times[i - 1] + pd.to_timedelta(1, unit="s")
        if times[-1] > right_anchor_ts:
            clamped = True
            times[-1] = right_anchor_ts
            for i in range(len(times) - 2, -1, -1):
                latest = times[i + 1] - pd.to_timedelta(1, unit="s")
                if times[i] > latest:
                    times[i] = latest
    return times, clamped


def plausibility_score(prev_act, cur_act, next_act, prev_ts, cur_ts, next_ts):
    score = 0.0
    if prev_act is not None and prev_ts is not None and pd.notna(prev_ts) and cur_ts is not None and pd.notna(cur_ts):
        dt = (cur_ts - prev_ts).total_seconds()
        prof = duration_profile(prev_act, cur_act)
        score += interval_fit_score(dt, prof)
    if next_act is not None and next_ts is not None and pd.notna(next_ts) and cur_ts is not None and pd.notna(cur_ts):
        dt = (next_ts - cur_ts).total_seconds()
        prof = duration_profile(cur_act, next_act)
        score += interval_fit_score(dt, prof)
    return score


def interval_fit_score(dt, prof):
    if dt < 0:
        return -1e6
    if prof is None:
        target = GLOBAL_FAST_DEFAULT if dt <= 120 else GLOBAL_MEDIAN
        return -abs(dt - target) / max(target, 1.0)
    p10 = prof["p10"]
    p25 = prof["p25"]
    med = prof["median"]
    if prof["share60"] >= 0.5:
        target = p10 if p10 > 0 else 1.0
    else:
        target = p25 if p25 > 0 else med if med > 0 else GLOBAL_MEDIAN
    return -abs(dt - target) / max(target, 1.0) + 0.001 * prof["n"]


def estimate_between(prev_row, cur_row, next_row, keep_date=False):
    prev_ts = prev_row["parsed"] if prev_row else None
    next_ts = next_row["parsed"] if next_row else None
    prev_act = prev_row["Activity"] if prev_row else None
    next_act = next_row["Activity"] if next_row else None
    cur_act = cur_row["Activity"]

    candidates = []
    if prev_row and prev_ts is not None and pd.notna(prev_ts):
        g = preferred_gap_seconds(prev_act, cur_act)
        if g <= 0:
            g = 1.0
        candidates.append(prev_ts + pd.to_timedelta(g, unit="s"))
    if next_row and next_ts is not None and pd.notna(next_ts):
        g = preferred_gap_seconds(cur_act, next_act)
        if g <= 0:
            g = 1.0
        candidates.append(next_ts - pd.to_timedelta(g, unit="s"))
    if prev_row and next_row and pd.notna(prev_ts) and pd.notna(next_ts):
        mid = prev_ts + (next_ts - prev_ts) / 2
        candidates.append(mid)
    if not candidates:
        return cur_row["parsed"] if pd.notna(cur_row["parsed"]) else pd.Timestamp("2023-01-01 00:00:10")
    if prev_row and next_row and pd.notna(prev_ts) and pd.notna(next_ts):
        lo, hi = prev_ts, next_ts
        bounded = []
        for c in candidates:
            bounded.append(clamp_ts(c, lo, hi))
        best = None
        best_sc = -1e100
        for c in bounded:
            sc = plausibility_score(prev_act, cur_act, next_act, prev_ts, c, next_ts)
            if best is None or sc > best_sc:
                best = c
                best_sc = sc
        est = best
        if est >= hi:
            est = lo + (hi - lo) / 2
        if est <= lo:
            est = lo + pd.to_timedelta(1, unit="s")
        if est >= hi:
            est = hi - pd.to_timedelta(1, unit="s")
        return est
    return candidates[0]


def repair():
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

    for r in rows:
        r["orig_timestamp"] = r["Timestamp"]
        r["parsed"] = safe_parse(r["Timestamp"])
        r["repaired_timestamp"] = r["Timestamp"]
        r["_changed"] = False

    by_case = defaultdict(list)
    for i, r in enumerate(rows):
        by_case[r["Case"]].append((i, r))

    c1_blocks_repaired = 0
    clamped_count = 0

    for case, items in by_case.items():
        case_rows = [r for _, r in items]
        case_indices = [i for i, _ in items]
        parsed_list = [r["parsed"] for r in case_rows]
        case_start, case_end = get_case_bounds(parsed_list)

        j = 0
        while j < len(case_rows):
            r = case_rows[j]
            if r["error_category"] == "C1_ORDERING_AMBIGUITY" and pd.notna(r["parsed"]):
                block = [j]
                shared = r["Timestamp"]
                shared_parsed = r["parsed"]
                j2 = j + 1
                while j2 < len(case_rows):
                    rr = case_rows[j2]
                    if rr["error_category"] == "C1_ORDERING_AMBIGUITY" and rr["Timestamp"] == shared:
                        block.append(j2)
                        j2 += 1
                    else:
                        break
                if len(block) >= 2:
                    left_idx = block[0] - 1
                    left_row = case_rows[left_idx] if left_idx >= 0 and case_rows[left_idx]["error_category"] == "CLEAN" and pd.notna(case_rows[left_idx]["parsed"]) else None
                    right_idx = block[-1] + 1
                    right_row = case_rows[right_idx] if right_idx < len(case_rows) and case_rows[right_idx]["error_category"] == "CLEAN" and pd.notna(case_rows[right_idx]["parsed"]) else None
                    acts = [case_rows[b]["Activity"] for b in block]
                    perm_idx = enumerate_or_beam(acts, left_row["Activity"] if left_row else None, right_row["Activity"] if right_row else None)
                    ordered_positions = [block[p] for p in perm_idx]
                    ordered_acts = [case_rows[p]["Activity"] for p in ordered_positions]
                    right_anchor_ts = right_row["parsed"] if right_row else case_end
                    times, clamped = build_block_times(shared_parsed, ordered_acts, right_anchor_ts)
                    if clamped:
                        clamped_count += 1
                    earliest_pos = ordered_positions[0]
                    assign = {}
                    for pos, ts in zip(ordered_positions, times):
                        assign[pos] = ts
                    for pos in block:
                        global_i = case_indices[pos]
                        ts_new = assign[pos]
                        if pos == earliest_pos:
                            rows[global_i]["repaired_timestamp"] = rows[global_i]["orig_timestamp"]
                            rows[global_i]["_changed"] = False
                        else:
                            lo = left_row["parsed"] if left_row else case_start
                            hi = right_row["parsed"] if right_row else case_end
                            ts_new = clamp_ts(ts_new, lo, hi)
                            if ts_new <= shared_parsed:
                                delta_sec = (times.index(assign[pos]) if assign[pos] in times else 1)
                                ts_new = shared_parsed + pd.to_timedelta(max(1, delta_sec), unit="s")
                                ts_new = clamp_ts(ts_new, lo, hi)
                            digits = frac_digits(rows[global_i]["orig_timestamp"])
                            rows[global_i]["repaired_timestamp"] = format_dt(ts_new, digits)
                            rows[global_i]["_changed"] = rows[global_i]["repaired_timestamp"] != rows[global_i]["orig_timestamp"]
                    c1_blocks_repaired += 1
                j = j2
            else:
                j += 1

        for local_idx, row in enumerate(case_rows):
            global_i = case_indices[local_idx]
            cat = row["error_category"]
            if cat == "CLEAN":
                continue

            if cat == "C2_TEMPORAL_INCONSISTENCY":
                cur_ts = safe_parse(rows[global_i]["repaired_timestamp"])
                prev_idx, prev_row = nearest_clean_left(case_rows, local_idx)
                next_idx, next_row = nearest_clean_right(case_rows, local_idx)

                if local_idx > 0:
                    prev_any = case_rows[local_idx - 1]
                    prev_any_global = case_indices[local_idx - 1]
                    prev_any_ts = safe_parse(rows[prev_any_global]["repaired_timestamp"])
                else:
                    prev_any = None
                    prev_any_ts = None

                if prev_any is not None and pd.notna(prev_any_ts) and pd.notna(cur_ts) and cur_ts < prev_any_ts:
                    swap_candidate_ok = False
                    if prev_any["error_category"] == "C2_TEMPORAL_INCONSISTENCY":
                        prev_old = safe_parse(prev_any["orig_timestamp"])
                        cur_old = safe_parse(row["orig_timestamp"])
                        if pd.notna(prev_old) and pd.notna(cur_old):
                            sc_swap = plausibility_score(
                                prev_row["Activity"] if prev_row else None,
                                prev_any["Activity"],
                                row["Activity"],
                                prev_row["parsed"] if prev_row else None,
                                cur_old,
                                prev_old
                            ) + plausibility_score(
                                prev_any["Activity"],
                                row["Activity"],
                                next_row["Activity"] if next_row else None,
                                cur_old,
                                prev_old,
                                next_row["parsed"] if next_row else None
                            )
                            sc_keep = plausibility_score(
                                prev_row["Activity"] if prev_row else None,
                                prev_any["Activity"],
                                row["Activity"],
                                prev_row["parsed"] if prev_row else None,
                                prev_old,
                                cur_old
                            ) + plausibility_score(
                                prev_any["Activity"],
                                row["Activity"],
                                next_row["Activity"] if next_row else None,
                                prev_old,
                                cur_old,
                                next_row["parsed"] if next_row else None
                            )
                            if sc_swap > sc_keep:
                                swap_candidate_ok = True
                                digits_prev = frac_digits(rows[prev_any_global]["orig_timestamp"])
                                digits_cur = frac_digits(rows[global_i]["orig_timestamp"])
                                rows[prev_any_global]["repaired_timestamp"] = format_dt(cur_old, digits_prev)
                                rows[global_i]["repaired_timestamp"] = format_dt(prev_old, digits_cur)
                                rows[prev_any_global]["_changed"] = rows[prev_any_global]["repaired_timestamp"] != rows[prev_any_global]["orig_timestamp"]
                                rows[global_i]["_changed"] = rows[global_i]["repaired_timestamp"] != rows[global_i]["orig_timestamp"]

                    if not swap_candidate_ok:
                        est = estimate_between(prev_row, row, next_row)
                        lo = prev_row["parsed"] if prev_row else case_start
                        hi = next_row["parsed"] if next_row else case_end
                        est = clamp_ts(est, lo, hi)
                        if prev_any_ts is not None and pd.notna(prev_any_ts) and est <= prev_any_ts:
                            est = prev_any_ts + pd.to_timedelta(1, unit="s")
                        if hi is not None and pd.notna(hi) and est >= hi:
                            est = hi - pd.to_timedelta(1, unit="s")
                        digits = frac_digits(rows[global_i]["orig_timestamp"])
                        rows[global_i]["repaired_timestamp"] = format_dt(est, digits)
                        rows[global_i]["_changed"] = rows[global_i]["repaired_timestamp"] != rows[global_i]["orig_timestamp"]

            elif cat == "C3_MISSING_TEMPORAL_INFORMATION":
                prev_idx, prev_row = nearest_clean_left(case_rows, local_idx)
                next_idx, next_row = nearest_clean_right(case_rows, local_idx)
                orig = row["orig_timestamp"]
                keep_date = False
                parsed_orig = safe_parse(orig)
                if orig and str(orig).strip() and len(str(orig).strip()) <= 10 and str(orig).count("-") == 2:
                    keep_date = True
                est = estimate_between(prev_row, row, next_row, keep_date=keep_date)
                if keep_date:
                    try:
                        date_part = str(orig).strip()
                        day_start = pd.Timestamp(date_part + " 00:00:00")
                        day_end = pd.Timestamp(date_part + " 23:59:59.999999")
                        lo = day_start
                        hi = day_end
                        if prev_row and pd.notna(prev_row["parsed"]):
                            lo = max(lo, prev_row["parsed"])
                        if next_row and pd.notna(next_row["parsed"]):
                            hi = min(hi, next_row["parsed"])
                        if lo > hi:
                            target = lo
                        else:
                            target = clamp_ts(est, lo, hi)
                            if target == hi and lo < hi:
                                target = lo + (hi - lo) / 2
                        est = target
                    except Exception:
                        pass
                else:
                    lo = prev_row["parsed"] if prev_row else case_start
                    hi = next_row["parsed"] if next_row else case_end
                    est = clamp_ts(est, lo, hi)
                    if prev_row and next_row and pd.notna(lo) and pd.notna(hi):
                        if est <= lo:
                            est = lo + pd.to_timedelta(1, unit="s")
                        if est >= hi:
                            est = lo + (hi - lo) / 2
                            if est <= lo:
                                est = lo + pd.to_timedelta(1, unit="s")
                            if est >= hi:
                                est = hi - pd.to_timedelta(1, unit="s")
                digits = frac_digits(orig)
                if digits == 0:
                    counts = Counter(frac_digits(rr["Timestamp"]) for rr in rows if str(rr["Timestamp"]).strip())
                    digits = counts.most_common(1)[0][0] if counts else 3
                rows[global_i]["repaired_timestamp"] = format_dt(est, digits)
                rows[global_i]["_changed"] = rows[global_i]["repaired_timestamp"] != rows[global_i]["orig_timestamp"]

    rows_equal_input = 0

    for case, items in by_case.items():
        case_rows = [rows[i] for i, _ in items]
        parsed_now = [safe_parse(r["repaired_timestamp"]) for r in case_rows]
        case_start, case_end = get_case_bounds(parsed_now)

        i = 0
        while i < len(case_rows):
            if case_rows[i]["error_category"] == "C1_ORDERING_AMBIGUITY":
                shared = case_rows[i]["orig_timestamp"]
                block = [i]
                j = i + 1
                while j < len(case_rows) and case_rows[j]["error_category"] == "C1_ORDERING_AMBIGUITY" and case_rows[j]["orig_timestamp"] == shared:
                    block.append(j)
                    j += 1
                if len(block) >= 2:
                    times = [safe_parse(case_rows[b]["repaired_timestamp"]) for b in block]
                    changed_flags = [case_rows[b]["repaired_timestamp"] != case_rows[b]["orig_timestamp"] for b in block]
                    eq_count = sum(1 for b in block if case_rows[b]["repaired_timestamp"] == case_rows[b]["orig_timestamp"])
                    min_idx = min(range(len(times)), key=lambda z: times[z] if pd.notna(times[z]) else pd.Timestamp.max)
                    need_fix = False
                    if eq_count != 1:
                        need_fix = True
                    uniq = len(set(str(t) for t in times))
                    if uniq != len(times):
                        need_fix = True
                    if any(pd.notna(times[k]) and pd.notna(times[k - 1]) and times[k] <= times[k - 1] for k in range(1, len(times))):
                        pass
                    if need_fix:
                        obs = safe_parse(shared)
                        right_anchor = None
                        if j < len(case_rows) and case_rows[j]["error_category"] == "CLEAN":
                            right_anchor = safe_parse(case_rows[j]["repaired_timestamp"])
                        seq_sorted = sorted(range(len(block)), key=lambda z: (times[z] if pd.notna(times[z]) else obs))
                        fixed = [obs]
                        room = max((right_anchor - obs).total_seconds(), len(block) - 1) if right_anchor is not None and pd.notna(right_anchor) else max(len(block) - 1, GLOBAL_FAST_DEFAULT * (len(block) - 1))
                        step = max(1.0, min(GLOBAL_FAST_DEFAULT, room / max(1, len(block) - 1)))
                        cur = obs
                        for _ in range(1, len(block)):
                            cur = cur + pd.to_timedelta(step, unit="s")
                            fixed.append(cur)
                        if right_anchor is not None and fixed[-1] > right_anchor:
                            cur = right_anchor
                            for z in range(len(fixed) - 1, -1, -1):
                                fixed[z] = cur
                                cur = cur - pd.to_timedelta(1, unit="s")
                        for rank, rel_idx in enumerate(seq_sorted):
                            b = block[rel_idx]
                            if rank == 0:
                                case_rows[b]["repaired_timestamp"] = case_rows[b]["orig_timestamp"]
                                case_rows[b]["_changed"] = False
                            else:
                                digits = frac_digits(case_rows[b]["orig_timestamp"])
                                case_rows[b]["repaired_timestamp"] = format_dt(fixed[rank], digits)
                                case_rows[b]["_changed"] = case_rows[b]["repaired_timestamp"] != case_rows[b]["orig_timestamp"]
                        clamped_count += 1
                i = j
            else:
                i += 1

        for idx in range(len(case_rows)):
            r = case_rows[idx]
            if r["error_category"] == "CLEAN":
                r["repaired_timestamp"] = r["orig_timestamp"]
            else:
                t = safe_parse(r["repaired_timestamp"])
                if pd.isna(t):
                    prev_idx, prev_row = nearest_clean_left(case_rows, idx)
                    next_idx, next_row = nearest_clean_right(case_rows, idx)
                    est = estimate_between(prev_row, r, next_row)
                    digits = frac_digits(r["orig_timestamp"])
                    r["repaired_timestamp"] = format_dt(est, digits)
                    clamped_count += 1
                    t = safe_parse(r["repaired_timestamp"])
                lo = case_start
                hi = case_end
                if idx > 0:
                    left_clean = None
                    for jj in range(idx - 1, -1, -1):
                        if case_rows[jj]["error_category"] == "CLEAN":
                            left_clean = safe_parse(case_rows[jj]["orig_timestamp"])
                            break
                    if left_clean is not None and pd.notna(left_clean):
                        lo = max(lo, left_clean) if lo is not None else left_clean
                if idx < len(case_rows) - 1:
                    right_clean = None
                    for jj in range(idx + 1, len(case_rows)):
                        if case_rows[jj]["error_category"] == "CLEAN":
                            right_clean = safe_parse(case_rows[jj]["orig_timestamp"])
                            break
                    if right_clean is not None and pd.notna(right_clean):
                        hi = min(hi, right_clean) if hi is not None else right_clean
                clamped = clamp_ts(t, lo, hi)
                if pd.notna(t) and clamped != t:
                    digits = frac_digits(r["orig_timestamp"])
                    r["repaired_timestamp"] = format_dt(clamped, digits)
                    clamped_count += 1

    for r in rows:
        if r["error_category"] != "CLEAN" and r["repaired_timestamp"] == r["orig_timestamp"]:
            rows_equal_input += 1

    try:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                out = {k: r[k] for k in fieldnames}
                out["Timestamp"] = r["repaired_timestamp"] if r["error_category"] != "CLEAN" else r["orig_timestamp"]
                writer.writerow(out)
    except Exception:
        try:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                for r in rows:
                    vals = []
                    for k in fieldnames:
                        if k == "Timestamp":
                            vals.append(r["repaired_timestamp"] if r["error_category"] != "CLEAN" else r["orig_timestamp"])
                        else:
                            vals.append(r.get(k, ""))
                    writer.writerow(vals)
        except Exception:
            pass

    print(f"blocks_repaired={c1_blocks_repaired} blocks_clamped={clamped_count} rows_equal_input={rows_equal_input}")


if __name__ == "__main__":
    try:
        repair()
    except Exception as e:
        try:
            print("blocks_repaired=0 blocks_clamped=0 rows_equal_input=0")
        except Exception:
            pass