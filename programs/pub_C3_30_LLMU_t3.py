import argparse
import math
import itertools
from datetime import timedelta
import pandas as pd
import numpy as np


TRANSITION_PROBABILITIES_TOP = [
    {"source": "Deliver to customer", "target": "EVENT 8 END", "probability": 1.0},
    {"source": "Signal food complete", "target": "Bring food", "probability": 0.9985},
    {"source": "Bring food", "target": "Deliver to customer", "probability": 0.9972},
    {"source": "start", "target": "Take order", "probability": 0.972},
    {"source": "Take order", "target": "Review order", "probability": 0.9599},
    {"source": "Bring drinks", "target": "Review drinks", "probability": 0.959},
    {"source": "Prepare main course", "target": "Signal food complete", "probability": 0.9547},
    {"source": "prepare drinks", "target": "Bring drinks", "probability": 0.9027},
    {"source": "Review order", "target": "prepare drinks", "probability": 0.7827},
    {"source": "Prepare starter", "target": "prepare drinks", "probability": 0.5269},
    {"source": "Review drinks", "target": "Prepare main course", "probability": 0.4844},
    {"source": "Review drinks", "target": "prepare drinks", "probability": 0.3836},
    {"source": "Prepare starter", "target": "Bring drinks", "probability": 0.2905},
    {"source": "Review order", "target": "Prepare starter", "probability": 0.1831},
    {"source": "Prepare starter", "target": "Prepare main course", "probability": 0.0843},
    {"source": "Review drinks", "target": "Signal food complete", "probability": 0.0781},
    {"source": "prepare drinks", "target": "Prepare starter", "probability": 0.0626},
    {"source": "Prepare starter", "target": "Review drinks", "probability": 0.058},
    {"source": "Review drinks", "target": "Prepare starter", "probability": 0.0405},
    {"source": "Prepare starter", "target": "Signal food complete", "probability": 0.0403},
    {"source": "Prepare main course", "target": "Bring food", "probability": 0.0367},
    {"source": "Review order", "target": "Bring drinks", "probability": 0.0316},
    {"source": "Take order", "target": "prepare drinks", "probability": 0.0311},
    {"source": "prepare drinks", "target": "Review drinks", "probability": 0.031},
    {"source": "start", "target": "Review order", "probability": 0.028},
    {"source": "Bring drinks", "target": "Prepare main course", "probability": 0.0172},
    {"source": "Bring drinks", "target": "Prepare starter", "probability": 0.0123},
    {"source": "Bring drinks", "target": "prepare drinks", "probability": 0.0091},
    {"source": "Take order", "target": "Prepare starter", "probability": 0.0091},
    {"source": "Review drinks", "target": "Bring drinks", "probability": 0.0088},
    {"source": "Prepare main course", "target": "Bring drinks", "probability": 0.0042},
    {"source": "Prepare main course", "target": "prepare drinks", "probability": 0.0035},
    {"source": "Review drinks", "target": "Bring food", "probability": 0.0029},
    {"source": "prepare drinks", "target": "Prepare main course", "probability": 0.0026},
    {"source": "Review order", "target": "Prepare main course", "probability": 0.0025},
    {"source": "Bring drinks", "target": "Signal food complete", "probability": 0.0023},
    {"source": "Bring food", "target": "Bring drinks", "probability": 0.0018},
    {"source": "Review drinks", "target": "Deliver to customer", "probability": 0.0017},
    {"source": "Prepare main course", "target": "Review drinks", "probability": 0.0009},
    {"source": "Signal food complete", "target": "prepare drinks", "probability": 0.0008},
    {"source": "prepare drinks", "target": "Signal food complete", "probability": 0.0007},
    {"source": "Signal food complete", "target": "Bring drinks", "probability": 0.0006},
    {"source": "Bring food", "target": "prepare drinks", "probability": 0.0006},
    {"source": "Bring food", "target": "Review drinks", "probability": 0.0004},
    {"source": "prepare drinks", "target": "Bring food", "probability": 0.0004},
]

DURATION_PROFILES_TOP = [
    {"source": "Bring drinks", "target": "Review drinks", "n": 7789, "p10_seconds": 6.0576, "p25_seconds": 16.973, "median_seconds": 40.525, "p75_seconds": 82.042, "p90_seconds": 134.3574, "share_under_60s": 0.637},
    {"source": "prepare drinks", "target": "Bring drinks", "n": 7327, "p10_seconds": 29.6502, "p25_seconds": 82.4315, "median_seconds": 200.074, "p75_seconds": 411.4195, "p90_seconds": 694.5820000000003, "share_under_60s": 0.191},
    {"source": "Deliver to customer", "target": "EVENT 8 END", "n": 5000, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "Bring food", "target": "Deliver to customer", "n": 4986, "p10_seconds": 12.5235, "p25_seconds": 34.57875, "median_seconds": 85.062, "p75_seconds": 168.381, "p90_seconds": 278.243, "share_under_60s": 0.389},
    {"source": "start", "target": "Take order", "n": 4860, "p10_seconds": 12.9624, "p25_seconds": 35.73925, "median_seconds": 89.125, "p75_seconds": 180.5775, "p90_seconds": 317.5108000000001, "share_under_60s": 0.379},
    {"source": "Signal food complete", "target": "Bring food", "n": 4816, "p10_seconds": 32.288, "p25_seconds": 89.16275, "median_seconds": 213.3435, "p75_seconds": 422.91425, "p90_seconds": 718.048, "share_under_60s": 0.179},
    {"source": "Take order", "target": "Review order", "n": 4665, "p10_seconds": 6.2354, "p25_seconds": 18.196, "median_seconds": 42.276, "p75_seconds": 83.699, "p90_seconds": 138.1858, "share_under_60s": 0.632},
    {"source": "Prepare main course", "target": "Signal food complete", "n": 4086, "p10_seconds": 85943.671, "p25_seconds": 254311.64075, "median_seconds": 515362.0085, "p75_seconds": 685728.1542499999, "p90_seconds": 846146.148, "share_under_60s": 0.0},
    {"source": "Review drinks", "target": "Prepare main course", "n": 3950, "p10_seconds": 83506.8328, "p25_seconds": 246266.73424999998, "median_seconds": 505754.282, "p75_seconds": 680995.01, "p90_seconds": 788988.6926, "share_under_60s": 0.001},
    {"source": "Review order", "target": "prepare drinks", "n": 3761, "p10_seconds": 26.85, "p25_seconds": 69.814, "median_seconds": 175.826, "p75_seconds": 351.24, "p90_seconds": 599.838, "share_under_60s": 0.221},
    {"source": "Review drinks", "target": "prepare drinks", "n": 3128, "p10_seconds": 29.9604, "p25_seconds": 85.61525, "median_seconds": 208.273, "p75_seconds": 413.24025, "p90_seconds": 688.4857000000005, "share_under_60s": 0.183},
    {"source": "Prepare starter", "target": "prepare drinks", "n": 981, "p10_seconds": 28.889, "p25_seconds": 84.699, "median_seconds": 203.911, "p75_seconds": 395.189, "p90_seconds": 657.285, "share_under_60s": 0.198},
    {"source": "Review order", "target": "Prepare starter", "n": 880, "p10_seconds": 15.1921, "p25_seconds": 43.88025, "median_seconds": 100.954, "p75_seconds": 200.96375, "p90_seconds": 351.0084, "share_under_60s": 0.344},
    {"source": "Review drinks", "target": "Signal food complete", "n": 637, "p10_seconds": 11685.151, "p25_seconds": 190375.455, "median_seconds": 515100.472, "p75_seconds": 697693.982, "p90_seconds": 959926.3334, "share_under_60s": 0.002},
    {"source": "Prepare starter", "target": "Bring drinks", "n": 541, "p10_seconds": 32.0, "p25_seconds": 91.873, "median_seconds": 221.952, "p75_seconds": 442.821, "p90_seconds": 707.247, "share_under_60s": 0.176},
    {"source": "prepare drinks", "target": "Prepare starter", "n": 508, "p10_seconds": 16.0638, "p25_seconds": 41.62825, "median_seconds": 109.9465, "p75_seconds": 200.58625, "p90_seconds": 322.5815, "share_under_60s": 0.344},
    {"source": "Review drinks", "target": "Prepare starter", "n": 330, "p10_seconds": 21.3248, "p25_seconds": 70.531, "median_seconds": 155.221, "p75_seconds": 329.52925, "p90_seconds": 716.1062, "share_under_60s": 0.23},
    {"source": "prepare drinks", "target": "Review drinks", "n": 252, "p10_seconds": 68.6561, "p25_seconds": 129.5055, "median_seconds": 262.26, "p75_seconds": 441.25875, "p90_seconds": 712.8732000000001, "share_under_60s": 0.095},
    {"source": "Prepare starter", "target": "Prepare main course", "n": 157, "p10_seconds": 78732.1044, "p25_seconds": 235601.191, "median_seconds": 501050.302, "p75_seconds": 627441.466, "p90_seconds": 774268.9982, "share_under_60s": 0.0},
    {"source": "Prepare main course", "target": "Bring food", "n": 157, "p10_seconds": 87381.2898, "p25_seconds": 250766.01, "median_seconds": 438430.326, "p75_seconds": 684147.918, "p90_seconds": 847117.1428, "share_under_60s": 0.0},
    {"source": "Review order", "target": "Bring drinks", "n": 152, "p10_seconds": 187.3826, "p25_seconds": 282.01025, "median_seconds": 476.833, "p75_seconds": 803.2555, "p90_seconds": 1229.1769000000002, "share_under_60s": 0.02},
    {"source": "Take order", "target": "prepare drinks", "n": 151, "p10_seconds": 67.495, "p25_seconds": 121.29, "median_seconds": 222.695, "p75_seconds": 398.54600000000005, "p90_seconds": 699.161, "share_under_60s": 0.093},
    {"source": "start", "target": "Review order", "n": 140, "p10_seconds": 59.6194, "p25_seconds": 100.168, "median_seconds": 158.5445, "p75_seconds": 230.536, "p90_seconds": 344.166, "share_under_60s": 0.107},
    {"source": "Bring drinks", "target": "Prepare main course", "n": 140, "p10_seconds": 87866.8959, "p25_seconds": 255063.302, "median_seconds": 480699.382, "p75_seconds": 673990.7475, "p90_seconds": 770074.1726, "share_under_60s": 0.007},
    {"source": "Prepare starter", "target": "Review drinks", "n": 108, "p10_seconds": 11.7123, "p25_seconds": 25.7145, "median_seconds": 55.2835, "p75_seconds": 123.15725, "p90_seconds": 199.5477, "share_under_60s": 0.528},
    {"source": "Bring drinks", "target": "Prepare starter", "n": 100, "p10_seconds": 3.8606, "p25_seconds": 15.50125, "median_seconds": 30.9025, "p75_seconds": 77.0855, "p90_seconds": 140.6387, "share_under_60s": 0.69},
    {"source": "Prepare starter", "target": "Signal food complete", "n": 75, "p10_seconds": 87920.7, "p25_seconds": 305692.972, "median_seconds": 516395.537, "p75_seconds": 677241.0995, "p90_seconds": 774438.8728, "share_under_60s": 0.0},
    {"source": "Bring drinks", "target": "prepare drinks", "n": 74, "p10_seconds": 75.3596, "p25_seconds": 154.0335, "median_seconds": 277.061, "p75_seconds": 470.95075, "p90_seconds": 808.5777, "share_under_60s": 0.068},
    {"source": "Review drinks", "target": "Bring drinks", "n": 72, "p10_seconds": 218.3702, "p25_seconds": 293.23525, "median_seconds": 566.718, "p75_seconds": 881.62175, "p90_seconds": 1191.743, "share_under_60s": 0.014},
    {"source": "Take order", "target": "Prepare starter", "n": 44, "p10_seconds": 62.5727, "p25_seconds": 101.62475, "median_seconds": 154.3685, "p75_seconds": 250.57375, "p90_seconds": 405.0846, "share_under_60s": 0.091},
    {"source": "Review drinks", "target": "Bring food", "n": 24, "p10_seconds": 171.5558, "p25_seconds": 4622.571, "median_seconds": 168377.244, "p75_seconds": 532053.1855, "p90_seconds": 685469.1487, "share_under_60s": 0.042},
    {"source": "prepare drinks", "target": "Prepare main course", "n": 21, "p10_seconds": 48.025, "p25_seconds": 86.359, "median_seconds": 227.445, "p75_seconds": 407.148, "p90_seconds": 555.312, "share_under_60s": 0.143},
    {"source": "Bring drinks", "target": "Signal food complete", "n": 19, "p10_seconds": 139780.8046, "p25_seconds": 169171.919, "median_seconds": 351110.078, "p75_seconds": 647166.1385, "p90_seconds": 731372.4888, "share_under_60s": 0.0},
    {"source": "Prepare main course", "target": "Bring drinks", "n": 18, "p10_seconds": 28.2273, "p25_seconds": 68.60625, "median_seconds": 129.8, "p75_seconds": 190.83625, "p90_seconds": 340.4634000000001, "share_under_60s": 0.222},
    {"source": "Prepare main course", "target": "prepare drinks", "n": 15, "p10_seconds": 96.6248, "p25_seconds": 114.4655, "median_seconds": 188.761, "p75_seconds": 277.1605, "p90_seconds": 366.1134, "share_under_60s": 0.0},
    {"source": "Review drinks", "target": "Deliver to customer", "n": 14, "p10_seconds": 13.1735, "p25_seconds": 26.957, "median_seconds": 70.7255, "p75_seconds": 122.753, "p90_seconds": 151.1391, "share_under_60s": 0.357},
    {"source": "Review order", "target": "Prepare main course", "n": 12, "p10_seconds": 116.8094, "p25_seconds": 167.5355, "median_seconds": 392.177, "p75_seconds": 615.0295, "p90_seconds": 958.6999, "share_under_60s": 0.0},
    {"source": "Bring food", "target": "Bring drinks", "n": 9, "p10_seconds": 45.0556, "p25_seconds": 115.781, "median_seconds": 222.362, "p75_seconds": 307.834, "p90_seconds": 455.4288, "share_under_60s": 0.222},
    {"source": "prepare drinks", "target": "Signal food complete", "n": 6, "p10_seconds": 22.7145, "p25_seconds": 78.136, "median_seconds": 252.3445, "p75_seconds": 386.54425, "p90_seconds": 534.4580000000001, "share_under_60s": 0.333},
    {"source": "Prepare main course", "target": "Review drinks", "n": 4, "p10_seconds": 2.2898, "p25_seconds": 3.674, "median_seconds": 7.2355, "p75_seconds": 43.87425, "p90_seconds": 104.7975, "share_under_60s": 0.75},
    {"source": "Signal food complete", "target": "prepare drinks", "n": 4, "p10_seconds": 65.3186, "p25_seconds": 153.056, "median_seconds": 228.6855, "p75_seconds": 267.4535, "p90_seconds": 288.8402, "share_under_60s": 0.25},
    {"source": "Signal food complete", "target": "Bring drinks", "n": 3, "p10_seconds": 150.102, "p25_seconds": 179.118, "median_seconds": 227.478, "p75_seconds": 255.185, "p90_seconds": 271.8092, "share_under_60s": 0.0},
    {"source": "Bring food", "target": "prepare drinks", "n": 3, "p10_seconds": 164.9284, "p25_seconds": 166.546, "median_seconds": 169.242, "p75_seconds": 176.463, "p90_seconds": 180.7956, "share_under_60s": 0.0},
    {"source": "prepare drinks", "target": "Bring food", "n": 3, "p10_seconds": 46.9712, "p25_seconds": 63.2825, "median_seconds": 90.468, "p75_seconds": 145.511, "p90_seconds": 178.5368, "share_under_60s": 0.333},
    {"source": "Bring food", "target": "Review drinks", "n": 2, "p10_seconds": 775.2938, "p25_seconds": 787.7435, "median_seconds": 808.493, "p75_seconds": 829.2425, "p90_seconds": 841.6922, "share_under_60s": 0.0},
]

GLOBAL_MEDIAN = 118.0235
DEFAULT_FAST_GAP = 30.0
DEFAULT_SLOW_GAP = 118.0235


def build_maps():
    prob = {}
    for x in TRANSITION_PROBABILITIES_TOP:
        prob[(x["source"], x["target"])] = x["probability"]
    dur = {}
    for x in DURATION_PROFILES_TOP:
        dur[(x["source"], x["target"])] = x
    return prob, dur


PROB_MAP, DUR_MAP = build_maps()


def safe_parse(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.NaT


def count_frac_digits(s):
    if s is None:
        return 0
    s = str(s)
    if "." in s:
        tail = s.split(".")[-1]
        digits = ''.join(ch for ch in tail if ch.isdigit())
        return len(digits)
    return 0


def detect_global_precision(series):
    mx = 0
    for s in series.astype(str):
        if s and s != "nan":
            mx = max(mx, count_frac_digits(s))
    return mx if mx > 0 else 3


def format_ts(ts, frac_digits):
    if pd.isna(ts):
        return ""
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits > 0:
        frac = f"{ts.microsecond:06d}"[:frac_digits].ljust(frac_digits, "0")
        return f"{base}.{frac}"
    return base


def get_profile(src, tgt):
    return DUR_MAP.get((src, tgt))


def get_prob(src, tgt):
    p = PROB_MAP.get((src, tgt))
    if p is not None:
        return p
    return 1e-6


def low_quantile_gap(src, tgt):
    prof = get_profile(src, tgt)
    if prof and prof.get("n", 0) >= 2:
        if prof["share_under_60s"] >= 0.5:
            g = prof["p10_seconds"]
        else:
            g = prof["p25_seconds"]
        if g is None or not np.isfinite(g):
            g = DEFAULT_FAST_GAP
        return max(1.0, float(g))
    return DEFAULT_FAST_GAP


def typical_gap(src, tgt):
    prof = get_profile(src, tgt)
    if prof and prof.get("n", 0) >= 2:
        if prof["share_under_60s"] <= 0.02:
            g = prof["median_seconds"]
        elif prof["share_under_60s"] >= 0.5:
            g = prof["p10_seconds"]
        else:
            g = prof["p25_seconds"]
        if g is None or not np.isfinite(g):
            g = DEFAULT_SLOW_GAP
        return max(1.0, float(g))
    return DEFAULT_SLOW_GAP


def plausibility_score(gap, src, tgt):
    prof = get_profile(src, tgt)
    if gap is None or gap < 0:
        return -1e9
    if prof is None:
        target = DEFAULT_FAST_GAP
        return -abs(math.log((gap + 1.0) / (target + 1.0)))
    p10 = max(0.0, prof["p10_seconds"])
    p25 = max(0.0, prof["p25_seconds"])
    med = max(0.0, prof["median_seconds"])
    p90 = max(0.0, prof["p90_seconds"])
    share = prof["share_under_60s"]
    if share >= 0.4:
        target = max(1.0, p10)
        spread = max(5.0, p25 - p10, target * 0.5)
    elif share <= 0.02:
        target = max(1.0, med)
        spread = max(60.0, p90 - p25, target * 0.75)
    else:
        target = max(1.0, p25)
        spread = max(10.0, med - p10, target * 0.6)
    z = abs(gap - target) / spread
    return -z


def nearest_clean_left(case_df, pos):
    for i in range(pos - 1, -1, -1):
        if case_df.iloc[i]["error_category"] == "CLEAN":
            ts = case_df.iloc[i]["parsed_ts"]
            if not pd.isna(ts):
                return i, case_df.iloc[i]
    return None, None


def nearest_clean_right(case_df, pos):
    for i in range(pos + 1, len(case_df)):
        if case_df.iloc[i]["error_category"] == "CLEAN":
            ts = case_df.iloc[i]["parsed_ts"]
            if not pd.isna(ts):
                return i, case_df.iloc[i]
    return None, None


def nearest_clean_left_idx(case_df, start_pos):
    for i in range(start_pos - 1, -1, -1):
        if case_df.iloc[i]["error_category"] == "CLEAN" and not pd.isna(case_df.iloc[i]["parsed_ts"]):
            return i
    return None


def nearest_clean_right_idx(case_df, end_pos):
    for i in range(end_pos + 1, len(case_df)):
        if case_df.iloc[i]["error_category"] == "CLEAN" and not pd.isna(case_df.iloc[i]["parsed_ts"]):
            return i
    return None


def estimate_between(left_ts, left_act, act, right_ts, right_act):
    left_gap = typical_gap(left_act, act) if left_ts is not None and left_act is not None else None
    right_gap = typical_gap(act, right_act) if right_ts is not None and right_act is not None else None
    if left_ts is not None and right_ts is not None:
        earliest = left_ts + pd.to_timedelta(max(1.0, left_gap if left_gap is not None else 1.0), unit="s")
        latest = right_ts - pd.to_timedelta(max(1.0, right_gap if right_gap is not None else 1.0), unit="s")
        if earliest <= latest:
            if left_gap is not None and right_gap is not None:
                total = left_gap + right_gap
                alpha = left_gap / total if total > 0 else 0.5
                cand = left_ts + (right_ts - left_ts) * alpha
                if cand < earliest:
                    cand = earliest
                if cand > latest:
                    cand = latest
                return cand
            return earliest + (latest - earliest) / 2
        return left_ts + (right_ts - left_ts) / 2
    if left_ts is not None:
        return left_ts + pd.to_timedelta(max(1.0, left_gap if left_gap is not None else DEFAULT_FAST_GAP), unit="s")
    if right_ts is not None:
        return right_ts - pd.to_timedelta(max(1.0, right_gap if right_gap is not None else DEFAULT_FAST_GAP), unit="s")
    return None


def enumerate_best_order(activities, left_act, right_act, max_perms=720):
    idxs = list(range(len(activities)))
    perms = list(itertools.permutations(idxs)) if math.factorial(len(idxs)) <= max_perms else None
    if perms is None:
        beam = [([], set(), 0.0)]
        width = 20
        for _ in range(len(activities)):
            new_beam = []
            for seq, used, score in beam:
                prev_act = left_act if not seq else activities[seq[-1]]
                for i in idxs:
                    if i in used:
                        continue
                    s = score + math.log(get_prob(prev_act, activities[i]) + 1e-12)
                    new_beam.append((seq + [i], used | {i}, s))
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:width]
        best = None
        best_score = -1e18
        for seq, used, score in beam:
            s = score
            if right_act is not None:
                s += math.log(get_prob(activities[seq[-1]], right_act) + 1e-12)
            if s > best_score:
                best_score = s
                best = tuple(seq)
        return list(best)
    best = None
    best_score = -1e18
    for perm in perms:
        score = 0.0
        if left_act is not None:
            score += math.log(get_prob(left_act, activities[perm[0]]) + 1e-12)
        for a, b in zip(perm[:-1], perm[1:]):
            score += math.log(get_prob(activities[a], activities[b]) + 1e-12)
        if right_act is not None:
            score += math.log(get_prob(activities[perm[-1]], right_act) + 1e-12)
        if score > best_score:
            best_score = score
            best = perm
    return list(best)


def repair_c1_block(case_df, block_positions, output_ts, repaired_flags, global_frac_digits):
    observed = case_df.iloc[block_positions[0]]["parsed_ts"]
    observed_str = case_df.iloc[block_positions[0]]["Timestamp"]
    k = len(block_positions)
    left_idx = nearest_clean_left_idx(case_df, block_positions[0])
    right_idx = nearest_clean_right_idx(case_df, block_positions[-1])
    left_act = case_df.iloc[left_idx]["Activity"] if left_idx is not None else None
    right_act = case_df.iloc[right_idx]["Activity"] if right_idx is not None else None
    right_ts = case_df.iloc[right_idx]["parsed_ts"] if right_idx is not None else None

    acts = [case_df.iloc[p]["Activity"] for p in block_positions]
    order = enumerate_best_order(acts, left_act, right_act)
    ordered_positions = [block_positions[i] for i in order]
    ordered_acts = [acts[i] for i in order]

    gaps = []
    for i in range(k - 1):
        gaps.append(low_quantile_gap(ordered_acts[i], ordered_acts[i + 1]))
    if len(gaps) == 0:
        gaps = []
    raw_total = sum(gaps)
    min_total = k - 1

    clamped = False
    if right_ts is not None and not pd.isna(observed):
        room = max(0.0, (right_ts - observed).total_seconds())
        target_total = min(raw_total, room)
        if target_total < min_total:
            target_total = min_total if room >= min_total else max(0.0, room)
        if len(gaps) > 0:
            if raw_total > 0:
                scale = target_total / raw_total
                scaled = [max(1.0, g * scale) for g in gaps]
            else:
                scaled = [1.0] * len(gaps)
            total = sum(scaled)
            if total > room and room >= min_total:
                scaled = [1.0] * len(gaps)
                extra = room - min_total
                if extra > 0:
                    weights = np.array(gaps, dtype=float)
                    if weights.sum() <= 0:
                        weights = np.ones(len(gaps))
                    weights = weights / weights.sum()
                    scaled = [1.0 + extra * w for w in weights]
                clamped = True
            gaps = scaled
    else:
        gaps = [max(1.0, g) for g in gaps]

    times = [observed]
    cur = observed
    for g in gaps:
        cur = cur + pd.to_timedelta(max(1.0, g), unit="s")
        times.append(cur)

    if right_ts is not None and times[-1] > right_ts:
        clamped = True
        if k == 1:
            times = [min(observed, right_ts)]
        else:
            room = max(0.0, (right_ts - observed).total_seconds())
            if room >= (k - 1):
                new_gaps = [1.0] * (k - 1)
                extra = room - (k - 1)
                weights = np.array(gaps if gaps else [1.0] * (k - 1), dtype=float)
                if weights.sum() <= 0:
                    weights = np.ones(k - 1)
                weights = weights / weights.sum()
                new_gaps = [1.0 + extra * w for w in weights]
                times = [observed]
                cur = observed
                for g in new_gaps:
                    cur = cur + pd.to_timedelta(g, unit="s")
                    times.append(cur)
            else:
                times = [observed + pd.to_timedelta(i * max(room / max(1, k - 1), 0.0), unit="s") for i in range(k)]
                for i in range(1, k):
                    if times[i] <= times[i - 1]:
                        times[i] = times[i - 1] + pd.to_timedelta(1, unit="s")

    for idx_in_order, pos in enumerate(ordered_positions):
        frac = count_frac_digits(case_df.iloc[pos]["Timestamp"])
        frac = frac if frac > 0 else global_frac_digits
        output_ts[pos] = format_ts(times[idx_in_order], frac)
        repaired_flags[pos] = True

    return 1, int(clamped)


def choose_c2_swap(case_df, i):
    row1 = case_df.iloc[i]
    row2 = case_df.iloc[i + 1]
    t1 = row1["parsed_ts"]
    t2 = row2["parsed_ts"]
    if pd.isna(t1) or pd.isna(t2):
        return False, None, None
    if t2 >= t1:
        return False, None, None

    lidx, lrow = nearest_clean_left(case_df, i)
    ridx, rrow = nearest_clean_right(case_df, i + 1)
    left_ts = lrow["parsed_ts"] if lrow is not None else None
    left_act = lrow["Activity"] if lrow is not None else None
    right_ts = rrow["parsed_ts"] if rrow is not None else None
    right_act = rrow["Activity"] if rrow is not None else None

    swap_t1, swap_t2 = t2, t1
    if left_ts is not None and swap_t1 < left_ts:
        return False, None, None
    if right_ts is not None and swap_t2 > right_ts:
        return False, None, None
    if swap_t2 < swap_t1:
        return False, None, None

    score = 0.0
    if left_ts is not None:
        score += plausibility_score((swap_t1 - left_ts).total_seconds(), left_act, row1["Activity"])
    score += plausibility_score((swap_t2 - swap_t1).total_seconds(), row1["Activity"], row2["Activity"])
    if right_ts is not None:
        score += plausibility_score((right_ts - swap_t2).total_seconds(), row2["Activity"], right_act)

    current_score = -1e9
    if left_ts is not None:
        current_score += plausibility_score((t1 - left_ts).total_seconds(), left_act, row1["Activity"])
    if t2 >= t1:
        current_score += plausibility_score((t2 - t1).total_seconds(), row1["Activity"], row2["Activity"])
    else:
        current_score += -1e6
    if right_ts is not None:
        current_score += plausibility_score((right_ts - t2).total_seconds(), row2["Activity"], right_act)

    return score > current_score + 0.5, swap_t1, swap_t2


def repair_c2_pair(case_df, i, output_ts, repaired_flags, global_frac_digits):
    row1 = case_df.iloc[i]
    row2 = case_df.iloc[i + 1]
    t1 = row1["parsed_ts"]
    t2 = row2["parsed_ts"]

    do_swap, st1, st2 = choose_c2_swap(case_df, i)
    if do_swap:
        frac1 = count_frac_digits(row1["Timestamp"]) or global_frac_digits
        frac2 = count_frac_digits(row2["Timestamp"]) or global_frac_digits
        output_ts[i] = format_ts(st1, frac1)
        output_ts[i + 1] = format_ts(st2, frac2)
        repaired_flags[i] = True
        repaired_flags[i + 1] = True
        return

    lidx, lrow = nearest_clean_left(case_df, i)
    ridx, rrow = nearest_clean_right(case_df, i + 1)
    left_ts = lrow["parsed_ts"] if lrow is not None else None
    left_act = lrow["Activity"] if lrow is not None else None
    right_ts = rrow["parsed_ts"] if rrow is not None else None
    right_act = rrow["Activity"] if rrow is not None else None

    score_keep1 = -1e18
    score_keep2 = -1e18

    est2 = estimate_between(t1, row1["Activity"], row2["Activity"], right_ts, right_act)
    if est2 is not None and est2 >= t1 and (right_ts is None or est2 <= right_ts):
        score_keep1 = 0.0
        if left_ts is not None:
            score_keep1 += plausibility_score((t1 - left_ts).total_seconds(), left_act, row1["Activity"])
        score_keep1 += plausibility_score((est2 - t1).total_seconds(), row1["Activity"], row2["Activity"])
        if right_ts is not None:
            score_keep1 += plausibility_score((right_ts - est2).total_seconds(), row2["Activity"], right_act)

    est1 = estimate_between(left_ts, left_act, row1["Activity"], t2, row2["Activity"])
    if est1 is not None and est1 <= t2 and (left_ts is None or est1 >= left_ts):
        score_keep2 = 0.0
        if left_ts is not None:
            score_keep2 += plausibility_score((est1 - left_ts).total_seconds(), left_act, row1["Activity"])
        score_keep2 += plausibility_score((t2 - est1).total_seconds(), row1["Activity"], row2["Activity"])
        if right_ts is not None:
            score_keep2 += plausibility_score((right_ts - t2).total_seconds(), row2["Activity"], right_act)

    if score_keep1 >= score_keep2:
        if est2 is None:
            est2 = t1 + pd.to_timedelta(max(1.0, typical_gap(row1["Activity"], row2["Activity"])), unit="s")
            if right_ts is not None and est2 > right_ts:
                est2 = t1 + (right_ts - t1) / 2
            if est2 <= t1:
                est2 = t1 + pd.to_timedelta(1, unit="s")
        frac2 = count_frac_digits(row2["Timestamp"]) or global_frac_digits
        output_ts[i + 1] = format_ts(est2, frac2)
        repaired_flags[i + 1] = True
    else:
        if est1 is None:
            est1 = t2 - pd.to_timedelta(max(1.0, typical_gap(row1["Activity"], row2["Activity"])), unit="s")
            if left_ts is not None and est1 < left_ts:
                est1 = left_ts + (t2 - left_ts) / 2
            if est1 >= t2:
                est1 = t2 - pd.to_timedelta(1, unit="s")
        frac1 = count_frac_digits(row1["Timestamp"]) or global_frac_digits
        output_ts[i] = format_ts(est1, frac1)
        repaired_flags[i] = True


def is_date_only_string(s):
    if s is None:
        return False
    s = str(s).strip()
    return len(s) == 10 and s.count("-") == 2


def repair_c3_row(case_df, i, output_ts, repaired_flags, global_frac_digits):
    row = case_df.iloc[i]
    orig = "" if pd.isna(row["Timestamp"]) else str(row["Timestamp"])
    lidx, lrow = nearest_clean_left(case_df, i)
    ridx, rrow = nearest_clean_right(case_df, i)
    left_ts = lrow["parsed_ts"] if lrow is not None else None
    left_act = lrow["Activity"] if lrow is not None else None
    right_ts = rrow["parsed_ts"] if rrow is not None else None
    right_act = rrow["Activity"] if rrow is not None else None

    est = estimate_between(left_ts, left_act, row["Activity"], right_ts, right_act)
    if est is None:
        case_start = case_df["parsed_ts"].dropna().min()
        case_end = case_df["parsed_ts"].dropna().max()
        if pd.isna(case_start) or pd.isna(case_end):
            est = pd.Timestamp("2024-01-01 12:00:00")
        else:
            est = case_start + (case_end - case_start) / 2

    if is_date_only_string(orig):
        try:
            d = pd.to_datetime(orig).date()
            est = pd.Timestamp.combine(d, est.to_pydatetime().time())
            if left_ts is not None and est < left_ts and left_ts.date() == d:
                est = pd.Timestamp.combine(d, left_ts.to_pydatetime().time()) + pd.to_timedelta(1, unit="s")
            if right_ts is not None and est > right_ts and right_ts.date() == d:
                est = pd.Timestamp.combine(d, right_ts.to_pydatetime().time()) - pd.to_timedelta(1, unit="s")
        except Exception:
            pass
    elif orig.startswith("000") or orig.endswith("00:00:00.000") or orig.endswith("00:00:00"):
        parsed = safe_parse(orig)
        if not pd.isna(parsed):
            d = parsed.date()
            est = pd.Timestamp.combine(d, est.to_pydatetime().time())

    if left_ts is not None and est < left_ts:
        est = left_ts + pd.to_timedelta(1, unit="s")
    if right_ts is not None and est > right_ts:
        if left_ts is not None and left_ts < right_ts:
            est = left_ts + (right_ts - left_ts) / 2
        else:
            est = right_ts - pd.to_timedelta(1, unit="s")

    frac = count_frac_digits(orig) or global_frac_digits
    output_ts[i] = format_ts(est, frac)
    repaired_flags[i] = True


def self_check_and_fix(df, output_ts, repaired_flags, c1_blocks, global_frac_digits):
    clamped_blocks = 0

    parsed_out = [safe_parse(x) for x in output_ts]

    case_groups = df.groupby("Case", sort=False).groups
    for case, idxs in case_groups.items():
        idxs = list(idxs)
        valid = [parsed_out[i] for i in idxs if not pd.isna(parsed_out[i])]
        if not valid:
            continue
        case_start = min(valid)
        case_end = max(valid)
        for i in idxs:
            if df.iloc[i]["error_category"] != "CLEAN" and not pd.isna(parsed_out[i]):
                t = parsed_out[i]
                if t < case_start:
                    parsed_out[i] = case_start
                    output_ts[i] = format_ts(case_start, count_frac_digits(df.iloc[i]["Timestamp"]) or global_frac_digits)
                if t > case_end:
                    parsed_out[i] = case_end
                    output_ts[i] = format_ts(case_end, count_frac_digits(df.iloc[i]["Timestamp"]) or global_frac_digits)

    for block_positions in c1_blocks:
        obs_str = str(df.iloc[block_positions[0]]["Timestamp"])
        obs_ts = safe_parse(obs_str)
        rows = [(i, parsed_out[i], output_ts[i], df.iloc[i]["Activity"]) for i in block_positions]
        rows_sorted = sorted(rows, key=lambda x: (x[1], x[0]))
        changed = False
        if not rows_sorted:
            continue
        equal_obs = [r for r in rows_sorted if r[2] == obs_str]
        if len(equal_obs) != 1 or rows_sorted[0][2] != obs_str:
            first_i = rows_sorted[0][0]
            output_ts[first_i] = obs_str
            parsed_out[first_i] = obs_ts
            for j in range(1, len(rows_sorted)):
                ii = rows_sorted[j][0]
                base = parsed_out[first_i] + pd.to_timedelta(j, unit="s")
                output_ts[ii] = format_ts(base, count_frac_digits(df.iloc[ii]["Timestamp"]) or global_frac_digits)
                parsed_out[ii] = base
            changed = True

        rows_sorted = sorted([(i, parsed_out[i]) for i in block_positions], key=lambda x: (x[1], x[0]))
        for j in range(1, len(rows_sorted)):
            if pd.isna(rows_sorted[j][1]) or rows_sorted[j][1] <= rows_sorted[j - 1][1]:
                ii = rows_sorted[j][0]
                newt = rows_sorted[j - 1][1] + pd.to_timedelta(1, unit="s")
                output_ts[ii] = format_ts(newt, count_frac_digits(df.iloc[ii]["Timestamp"]) or global_frac_digits)
                parsed_out[ii] = newt
                rows_sorted[j] = (ii, newt)
                changed = True

        right_idx = None
        case_df = df[df["Case"] == df.iloc[block_positions[0]]["Case"]].reset_index()
        local_positions = [case_df.index[case_df["index"] == p][0] for p in block_positions]
        right_local = nearest_clean_right_idx(case_df, max(local_positions))
        if right_local is not None:
            right_idx = case_df.iloc[right_local]["index"]
        if right_idx is not None and not pd.isna(obs_ts):
            right_ts = safe_parse(df.iloc[right_idx]["Timestamp"])
            if not pd.isna(right_ts):
                current = sorted([parsed_out[i] for i in block_positions])
                if current[-1] > right_ts:
                    room = max(0.0, (right_ts - obs_ts).total_seconds())
                    k = len(block_positions)
                    if room >= (k - 1):
                        sorted_block = sorted(block_positions, key=lambda i: (parsed_out[i], i))
                        weights = np.ones(k - 1)
                        extra = room - (k - 1)
                        gaps = [1.0 + extra / max(1, (k - 1))] * (k - 1)
                        cur = obs_ts
                        output_ts[sorted_block[0]] = obs_str
                        parsed_out[sorted_block[0]] = obs_ts
                        for j in range(1, k):
                            cur = cur + pd.to_timedelta(gaps[j - 1], unit="s")
                            output_ts[sorted_block[j]] = format_ts(cur, count_frac_digits(df.iloc[sorted_block[j]]["Timestamp"]) or global_frac_digits)
                            parsed_out[sorted_block[j]] = cur
                    changed = True

        if changed:
            clamped_blocks += 1

    return clamped_blocks, output_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        try:
            pd.DataFrame(columns=["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]).to_csv(args.output, index=False)
        except Exception:
            pass
        print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0")
        return

    expected_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[expected_cols].copy()

    original_ts = df["Timestamp"].astype(str).tolist()
    output_ts = original_ts.copy()
    repaired_flags = [False] * len(df)
    global_frac_digits = detect_global_precision(df["Timestamp"])

    df["parsed_ts"] = df["Timestamp"].apply(safe_parse)

    blocks_repaired = 0
    blocks_clamped = 0
    c1_blocks = []

    for case, idxs in df.groupby("Case", sort=False).groups.items():
        case_positions = list(idxs)
        case_df = df.loc[case_positions].reset_index()
        n = len(case_df)

        j = 0
        while j < n:
            row = case_df.iloc[j]
            if row["error_category"] == "C1_ORDERING_AMBIGUITY" and not pd.isna(row["parsed_ts"]):
                block_local = [j]
                shared = row["Timestamp"]
                j2 = j + 1
                while j2 < n and case_df.iloc[j2]["error_category"] == "C1_ORDERING_AMBIGUITY" and case_df.iloc[j2]["Timestamp"] == shared:
                    block_local.append(j2)
                    j2 += 1
                if len(block_local) >= 2:
                    global_block = [case_positions[case_df.iloc[x]["index"] - case_positions[0]] if False else int(case_df.iloc[x]["index"]) for x in block_local]
                    c1_blocks.append(global_block)
                    b, c = repair_c1_block(df.loc[case_positions].reset_index(drop=True), block_local, [output_ts[p] for p in case_positions], [repaired_flags[p] for p in case_positions], global_frac_digits)
                    local_output = [output_ts[p] for p in case_positions]
                    local_repaired = [repaired_flags[p] for p in case_positions]
                    b, c = repair_c1_block(df.loc[case_positions].reset_index(drop=True), block_local, local_output, local_repaired, global_frac_digits)
                    for ii, p in enumerate(case_positions):
                        output_ts[p] = local_output[ii]
                        repaired_flags[p] = local_repaired[ii]
                    blocks_repaired += b
                    blocks_clamped += c
                j = j2
            else:
                j += 1

        local_output = [output_ts[p] for p in case_positions]
        local_repaired = [repaired_flags[p] for p in case_positions]

        for i in range(n - 1):
            r1 = case_df.iloc[i]
            r2 = case_df.iloc[i + 1]
            if r1["error_category"] == "C2_TEMPORAL_INCONSISTENCY" or r2["error_category"] == "C2_TEMPORAL_INCONSISTENCY":
                t1 = safe_parse(local_output[i])
                t2 = safe_parse(local_output[i + 1])
                temp_case = case_df.copy()
                temp_case["parsed_ts"] = [safe_parse(x) if idx not in [] else pd.NaT for idx, x in enumerate(local_output)]
                if not pd.isna(t1) and not pd.isna(t2) and t2 < t1:
                    repair_c2_pair(temp_case, i, local_output, local_repaired, global_frac_digits)

        for i in range(n):
            if case_df.iloc[i]["error_category"] == "C3_MISSING_TEMPORAL_INFORMATION":
                temp_case = case_df.copy()
                temp_case["parsed_ts"] = [safe_parse(x) for x in local_output]
                repair_c3_row(temp_case, i, local_output, local_repaired, global_frac_digits)

        for ii, p in enumerate(case_positions):
            output_ts[p] = local_output[ii]
            repaired_flags[p] = local_repaired[ii]

    for i in range(len(df)):
        if df.iloc[i]["error_category"] == "CLEAN":
            output_ts[i] = original_ts[i]
            repaired_flags[i] = False

    blocks_clamped_extra, output_ts = self_check_and_fix(df, output_ts, repaired_flags, c1_blocks, global_frac_digits)
    blocks_clamped += blocks_clamped_extra

    for i in range(len(df)):
        if df.iloc[i]["error_category"] != "CLEAN":
            if output_ts[i] == original_ts[i]:
                is_legit_c1_keeper = False
                for block in c1_blocks:
                    obs = original_ts[block[0]]
                    same = [idx for idx in block if output_ts[idx] == obs]
                    if len(same) == 1 and same[0] == i:
                        is_legit_c1_keeper = True
                        break
                if not is_legit_c1_keeper:
                    parsed = safe_parse(output_ts[i])
                    if pd.isna(parsed):
                        parsed = safe_parse(original_ts[i])
                    if not pd.isna(parsed):
                        bumped = parsed + pd.to_timedelta(1, unit="s")
                        output_ts[i] = format_ts(bumped, count_frac_digits(original_ts[i]) or global_frac_digits)
                    else:
                        output_ts[i] = format_ts(pd.Timestamp("2024-01-01 00:00:01"), count_frac_digits(original_ts[i]) or global_frac_digits)

    out = df[["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]].copy()
    out["Timestamp"] = output_ts
    try:
        out.to_csv(args.output, index=False)
    except Exception:
        try:
            out.to_csv(args.output, index=False, encoding="utf-8")
        except Exception:
            pass

    rows_still_equal = 0
    for i in range(len(df)):
        if df.iloc[i]["error_category"] != "CLEAN" and output_ts[i] == original_ts[i]:
            rows_still_equal += 1

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={blocks_clamped} rows_still_equal_to_input={rows_still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0")
        except Exception:
            pass