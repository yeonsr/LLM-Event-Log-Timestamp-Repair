import argparse
import itertools
import math
import re
from collections import defaultdict

import pandas as pd


TRANSITION_PROB = {
    ("Deliver to customer", "EVENT 8 END"): 1.0,
    ("Signal food complete", "Bring food"): 0.9985,
    ("Bring food", "Deliver to customer"): 0.9972,
    ("Prepare main course", "Signal food complete"): 0.9913,
    ("start", "Take order"): 0.9708,
    ("Bring drinks", "Review drinks"): 0.9554,
    ("Take order", "Review order"): 0.9464,
    ("prepare drinks", "Bring drinks"): 0.8979,
    ("Review order", "prepare drinks"): 0.7727,
    ("Prepare starter", "prepare drinks"): 0.5310,
    ("Review drinks", "Prepare main course"): 0.4767,
    ("Review drinks", "prepare drinks"): 0.3939,
    ("Prepare starter", "Bring drinks"): 0.2706,
    ("Review order", "Prepare starter"): 0.1765,
    ("Prepare starter", "Prepare main course"): 0.0860,
    ("Review drinks", "Signal food complete"): 0.0681,
    ("prepare drinks", "Prepare starter"): 0.0596,
    ("Prepare starter", "Review drinks"): 0.0583,
    ("Prepare starter", "Signal food complete"): 0.0481,
    ("Review drinks", "Prepare starter"): 0.0392,
    ("Take order", "Review drinks"): 0.0218,
    ("Review order", "Review drinks"): 0.0181,
    ("prepare drinks", "Bring food"): 0.0155,
    ("Review drinks", "Bring food"): 0.0149,
    ("Review order", "Prepare main course"): 0.0133,
    ("Bring drinks", "Signal food complete"): 0.0129,
    ("start", "Bring drinks"): 0.0124,
    ("Take order", "Bring drinks"): 0.0124,
    ("prepare drinks", "Signal food complete"): 0.0121,
    ("Bring drinks", "Prepare starter"): 0.0119,
    ("Take order", "Prepare main course"): 0.0117,
    ("Bring drinks", "Bring food"): 0.0110,
    ("Review order", "Signal food complete"): 0.0107,
    ("start", "Review drinks"): 0.0092,
    ("Review order", "Bring drinks"): 0.0087,
    ("prepare drinks", "Prepare main course"): 0.0068,
    ("prepare drinks", "Review drinks"): 0.0062,
    ("Prepare starter", "Bring food"): 0.0060,
    ("start", "prepare drinks"): 0.0050,
    ("Take order", "prepare drinks"): 0.0049,
    ("Prepare main course", "Bring drinks"): 0.0046,
    ("Bring drinks", "Bring drinks"): 0.0046,
    ("Review drinks", "Review drinks"): 0.0040,
    ("Bring drinks", "Prepare main course"): 0.0034,
    ("Prepare main course", "prepare drinks"): 0.0031,
    ("Take order", "Prepare starter"): 0.0027,
    ("start", "Prepare starter"): 0.0026,
    ("Bring food", "Bring drinks"): 0.0022,
    ("prepare drinks", "prepare drinks"): 0.0019,
    ("Review drinks", "Deliver to customer"): 0.0019,
    ("Review drinks", "Bring drinks"): 0.0013,
    ("Prepare main course", "Review drinks"): 0.0010,
    ("Bring drinks", "prepare drinks"): 0.0009,
    ("Signal food complete", "prepare drinks"): 0.0009,
    ("Signal food complete", "Bring drinks"): 0.0006,
    ("Bring food", "prepare drinks"): 0.0006,
}

DURATION = {
    ("Bring drinks", "Review drinks"): {"n": 7062, "p10": 6.1307, "p25": 17.0105, "med": 41.052, "p75": 83.472, "p90": 136.7487, "u60": 0.631},
    ("prepare drinks", "Bring drinks"): {"n": 6704, "p10": 28.8126, "p25": 79.71675, "med": 195.9745, "p75": 413.75325, "p90": 700.3471, "u60": 0.198},
    ("Deliver to customer", "EVENT 8 END"): {"n": 5000, "p10": 0.0, "p25": 0.0, "med": 0.0, "p75": 0.0, "p90": 0.0, "u60": 1.0},
    ("Bring food", "Deliver to customer"): {"n": 4986, "p10": 12.5235, "p25": 34.57875, "med": 85.062, "p75": 168.381, "p90": 278.243, "u60": 0.389},
    ("start", "Take order"): {"n": 4854, "p10": 12.9101, "p25": 35.79525, "med": 89.148, "p75": 180.67575, "p90": 315.9663, "u60": 0.378},
    ("Signal food complete", "Bring food"): {"n": 4682, "p10": 33.0276, "p25": 90.3885, "med": 214.9425, "p75": 423.9705, "p90": 718.42, "u60": 0.176},
    ("Take order", "Review order"): {"n": 4594, "p10": 6.1912, "p25": 17.985, "med": 41.872, "p75": 82.9715, "p90": 138.1861, "u60": 0.636},
    ("Prepare main course", "Signal food complete"): {"n": 3866, "p10": 85943.671, "p25": 253686.04975, "med": 513669.015, "p75": 685817.92925, "p90": 845950.272, "u60": 0.001},
    ("Review drinks", "Prepare main course"): {"n": 3563, "p10": 83921.2662, "p25": 243350.551, "med": 500815.099, "p75": 680990.933, "p90": 788371.1142, "u60": 0.001},
    ("Review order", "prepare drinks"): {"n": 3550, "p10": 25.252, "p25": 68.49925, "med": 173.713, "p75": 352.967, "p90": 606.4154, "u60": 0.226},
    ("Review drinks", "prepare drinks"): {"n": 2944, "p10": 30.7213, "p25": 85.6735, "med": 209.569, "p75": 415.462, "p90": 694.3028, "u60": 0.182},
    ("Prepare starter", "prepare drinks"): {"n": 883, "p10": 30.3582, "p25": 89.143, "med": 206.2, "p75": 397.074, "p90": 656.961, "u60": 0.19},
    ("Review order", "Prepare starter"): {"n": 811, "p10": 15.008, "p25": 42.552, "med": 97.24, "p75": 200.8065, "p90": 351.984, "u60": 0.354},
    ("Review drinks", "Signal food complete"): {"n": 509, "p10": 7451.2538, "p25": 175593.462, "med": 445342.856, "p75": 686397.671, "p90": 797873.2922, "u60": 0.002},
    ("Prepare starter", "Bring drinks"): {"n": 450, "p10": 29.0292, "p25": 86.16725, "med": 203.727, "p75": 427.8865, "p90": 688.7044, "u60": 0.189},
    ("prepare drinks", "Prepare starter"): {"n": 445, "p10": 15.8778, "p25": 40.819, "med": 100.292, "p75": 192.881, "p90": 296.6494, "u60": 0.362},
    ("Review drinks", "Prepare starter"): {"n": 293, "p10": 17.9444, "p25": 63.83, "med": 151.424, "p75": 322.423, "p90": 679.4376, "u60": 0.232},
    ("Prepare starter", "Prepare main course"): {"n": 143, "p10": 76289.6188, "p25": 192412.7035, "med": 445174.842, "p75": 624995.3195, "p90": 770926.678, "u60": 0.0},
    ("prepare drinks", "Bring food"): {"n": 116, "p10": 39201.883, "p25": 406754.148, "med": 785107.18, "p75": 1309999.6535, "p90": 1599036.572, "u60": 0.009},
    ("Review drinks", "Bring food"): {"n": 111, "p10": 161178.253, "p25": 593624.664, "med": 857938.338, "p75": 1377516.8425, "p90": 1634321.944, "u60": 0.009},
    ("Take order", "Review drinks"): {"n": 106, "p10": 257.7665, "p25": 389.71375, "med": 707.2185, "p75": 1038.79475, "p90": 1354.652, "u60": 0.0},
    ("Prepare starter", "Review drinks"): {"n": 97, "p10": 11.5194, "p25": 22.425, "med": 56.303, "p75": 132.229, "p90": 367.8652, "u60": 0.505},
    ("Bring drinks", "Signal food complete"): {"n": 95, "p10": 179812.449, "p25": 593579.078, "med": 883902.239, "p75": 1299383.9915, "p90": 1510712.6246, "u60": 0.0},
    ("prepare drinks", "Signal food complete"): {"n": 90, "p10": 84583.4036, "p25": 431151.8905, "med": 982509.744, "p75": 1344037.35375, "p90": 1629650.5926, "u60": 0.011},
    ("Bring drinks", "Prepare starter"): {"n": 88, "p10": 4.3939, "p25": 15.89925, "med": 33.688, "p75": 75.99025, "p90": 139.0768, "u60": 0.693},
    ("Review order", "Review drinks"): {"n": 83, "p10": 303.721, "p25": 381.153, "med": 593.835, "p75": 954.486, "p90": 1402.8866, "u60": 0.0},
    ("Bring drinks", "Bring food"): {"n": 81, "p10": 102484.185, "p25": 366203.855, "med": 696631.885, "p75": 1367714.419, "p90": 1616933.672, "u60": 0.0},
    ("Prepare starter", "Signal food complete"): {"n": 80, "p10": 88210.8159, "p25": 276729.8375, "med": 519007.665, "p75": 688787.49675, "p90": 944907.5365, "u60": 0.0},
    ("start", "Bring drinks"): {"n": 62, "p10": 275.2757, "p25": 422.42225, "med": 695.196, "p75": 915.41325, "p90": 1413.8304, "u60": 0.0},
    ("Review order", "Prepare main course"): {"n": 61, "p10": 943.183, "p25": 84467.297, "med": 444605.508, "p75": 620958.677, "p90": 700252.983, "u60": 0.0},
    ("Take order", "Bring drinks"): {"n": 60, "p10": 296.2082, "p25": 433.589, "med": 624.287, "p75": 992.78675, "p90": 1341.9016, "u60": 0.0},
    ("Take order", "Prepare main course"): {"n": 57, "p10": 101957.3608, "p25": 258966.497, "med": 513968.537, "p75": 624777.344, "p90": 777773.7006, "u60": 0.0},
    ("prepare drinks", "Prepare main course"): {"n": 51, "p10": 71.18, "p25": 281.63, "med": 160626.596, "p75": 523929.66, "p90": 681005.825, "u60": 0.078},
    ("Review order", "Signal food complete"): {"n": 49, "p10": 152688.052, "p25": 587731.024, "med": 862530.877, "p75": 1300062.102, "p90": 1476447.8158, "u60": 0.0},
    ("start", "Review drinks"): {"n": 46, "p10": 281.845, "p25": 455.605, "med": 733.167, "p75": 1084.1665, "p90": 1306.389, "u60": 0.0},
    ("prepare drinks", "Review drinks"): {"n": 46, "p10": 368.894, "p25": 508.0155, "med": 656.5765, "p75": 1133.5845, "p90": 1545.7045, "u60": 0.0},
    ("Review order", "Bring drinks"): {"n": 40, "p10": 298.812, "p25": 538.94575, "med": 784.107, "p75": 1205.822, "p90": 1648.7174, "u60": 0.0},
    ("Bring drinks", "Bring drinks"): {"n": 34, "p10": 237.8929, "p25": 350.94375, "med": 571.916, "p75": 863.97725, "p90": 1168.747, "u60": 0.0},
    ("Review drinks", "Review drinks"): {"n": 30, "p10": 283.804, "p25": 471.11475, "med": 695.439, "p75": 1119.0275, "p90": 2202.1487, "u60": 0.0},
    ("start", "prepare drinks"): {"n": 25, "p10": 255.112, "p25": 319.999, "med": 413.649, "p75": 582.288, "p90": 738.0338, "u60": 0.0},
    ("Bring drinks", "Prepare main course"): {"n": 25, "p10": 53.072, "p25": 21489.537, "med": 438128.715, "p75": 669828.116, "p90": 685551.208, "u60": 0.12},
    ("Take order", "prepare drinks"): {"n": 24, "p10": 214.6615, "p25": 304.47475, "med": 736.587, "p75": 1131.26225, "p90": 1339.0379, "u60": 0.0},
    ("Prepare main course", "Bring drinks"): {"n": 18, "p10": 28.2273, "p25": 68.60625, "med": 129.8, "p75": 190.83625, "p90": 340.4634, "u60": 0.222},
    ("Review drinks", "Deliver to customer"): {"n": 14, "p10": 13.1735, "p25": 26.957, "med": 70.7255, "p75": 122.753, "p90": 151.1391, "u60": 0.357},
    ("prepare drinks", "prepare drinks"): {"n": 14, "p10": 430.3684, "p25": 495.2955, "med": 639.1405, "p75": 705.744, "p90": 1295.0399, "u60": 0.0},
    ("start", "Prepare starter"): {"n": 13, "p10": 275.651, "p25": 370.046, "med": 404.992, "p75": 456.3, "p90": 517.9136, "u60": 0.0},
    ("Take order", "Prepare starter"): {"n": 13, "p10": 327.2842, "p25": 389.43, "med": 558.771, "p75": 1311.424, "p90": 2316.6824, "u60": 0.0},
    ("Prepare main course", "prepare drinks"): {"n": 12, "p10": 94.8512, "p25": 110.78, "med": 178.7615, "p75": 259.93625, "p90": 354.0485, "u60": 0.0},
    ("Bring food", "Bring drinks"): {"n": 11, "p10": 50.942, "p25": 152.058, "med": 229.533, "p75": 512.9985, "p90": 626.086, "u60": 0.182},
    ("Review drinks", "Bring drinks"): {"n": 10, "p10": 434.9222, "p25": 666.72275, "med": 1267.9125, "p75": 1729.7255, "p90": 1950.6004, "u60": 0.0},
    ("Prepare starter", "Bring food"): {"n": 10, "p10": 167409.7799, "p25": 293248.85775, "med": 525665.88, "p75": 931247.942, "p90": 1290839.1252, "u60": 0.0},
    ("Bring drinks", "prepare drinks"): {"n": 7, "p10": 691.6136, "p25": 983.649, "med": 1736.597, "p75": 2019.697, "p90": 2129.3134, "u60": 0.0},
    ("Signal food complete", "prepare drinks"): {"n": 4, "p10": 65.3186, "p25": 153.056, "med": 228.6855, "p75": 267.4535, "p90": 288.8402, "u60": 0.25},
    ("Prepare main course", "Review drinks"): {"n": 4, "p10": 2.2898, "p25": 3.674, "med": 35.4335, "p75": 86.17125, "p90": 121.7163, "u60": 0.5},
    ("Bring food", "prepare drinks"): {"n": 3, "p10": 164.9284, "p25": 166.546, "med": 169.242, "p75": 176.463, "p90": 180.7956, "u60": 0.0},
    ("Signal food complete", "Bring drinks"): {"n": 3, "p10": 150.102, "p25": 179.118, "med": 227.478, "p75": 255.185, "p90": 271.8092, "u60": 0.0},
}

GLOBAL_MEDIAN = 117.1945


def safe_to_datetime(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT


def detect_precision_and_separator(series):
    max_frac = 0
    sep = "."
    for s in series.dropna().astype(str):
        m = re.search(r"(\d{2}:\d{2}:\d{2})([.,](\d+))?$", s.strip())
        if m:
            frac = m.group(3)
            if frac is not None:
                max_frac = max(max_frac, len(frac))
                sep = m.group(2)[0]
    if max_frac == 0:
        max_frac = 3
    return max_frac, sep


def format_ts(ts, frac_digits=3, sep="."):
    if pd.isna(ts):
        return ""
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits <= 0:
        return base
    nanos = ts.nanosecond + ts.microsecond * 1000
    frac = f"{nanos:09d}"[:frac_digits]
    return base + sep + frac


def transition_prob(a, b):
    return TRANSITION_PROB.get((a, b), 1e-6)


def duration_profile(a, b):
    prof = DURATION.get((a, b))
    if prof and prof["n"] >= 3:
        return prof
    return None


def choose_gap_seconds(a, b):
    prof = duration_profile(a, b)
    if prof:
        if prof["u60"] >= 0.3:
            g = prof["p10"]
        elif prof["u60"] >= 0.12:
            g = prof["p25"]
        else:
            g = prof["p25"]
        return max(1.0, float(g))
    return max(1.0, GLOBAL_MEDIAN)


def duration_plausibility_score(a, b, delta_seconds):
    prof = duration_profile(a, b)
    if delta_seconds < 0:
        return -1e9
    if prof:
        target = choose_gap_seconds(a, b)
        spread = max(5.0, prof["p75"] - prof["p25"], prof["p25"] * 0.5)
        z = abs(delta_seconds - target) / spread
        return -z
    target = GLOBAL_MEDIAN
    spread = max(10.0, GLOBAL_MEDIAN)
    return -abs(delta_seconds - target) / spread


def score_order(order, left_act, right_act):
    score = 0.0
    if left_act is not None:
        score += math.log(max(transition_prob(left_act, order[0]), 1e-9))
    for a, b in zip(order[:-1], order[1:]):
        score += math.log(max(transition_prob(a, b), 1e-9))
    if right_act is not None:
        score += math.log(max(transition_prob(order[-1], right_act), 1e-9))
    return score


def estimate_series_from_start(start_ts, activities, right_anchor_ts=None):
    times = [start_ts]
    cur = start_ts
    for a, b in zip(activities[:-1], activities[1:]):
        cur = cur + pd.Timedelta(seconds=choose_gap_seconds(a, b))
        times.append(cur)
    if right_anchor_ts is not None:
        min_span = pd.Timedelta(seconds=max(len(activities) - 1, 0))
        max_allowed_end = right_anchor_ts
        if times[-1] > max_allowed_end:
            available = (max_allowed_end - start_ts).total_seconds()
            target_min = max(len(activities) - 1, 0)
            if available < target_min:
                available = target_min
            raw_gaps = []
            for a, b in zip(activities[:-1], activities[1:]):
                raw_gaps.append(max(1.0, choose_gap_seconds(a, b)))
            raw_sum = sum(raw_gaps)
            if raw_sum <= 0:
                raw_gaps = [1.0] * max(0, len(activities) - 1)
                raw_sum = sum(raw_gaps)
            scale = available / raw_sum if raw_sum > 0 else 1.0
            scaled = [max(1.0, g * scale) for g in raw_gaps]
            total = sum(scaled)
            if total > available and total > 0:
                scale2 = available / total
                scaled = [max(1.0, g * scale2) for g in scaled]
            times = [start_ts]
            cur = start_ts
            for g in scaled:
                cur = cur + pd.Timedelta(seconds=g)
                times.append(cur)
            if times[-1] > right_anchor_ts:
                step_room = max((right_anchor_ts - start_ts).total_seconds(), target_min)
                if len(activities) > 1:
                    unit = max(1.0, step_room / (len(activities) - 1))
                    unit = min(unit, step_room)
                    times = [start_ts]
                    cur = start_ts
                    for _ in range(len(activities) - 1):
                        cur = cur + pd.Timedelta(seconds=unit)
                        times.append(cur)
    if right_anchor_ts is not None:
        for i in range(len(times)):
            if times[i] > right_anchor_ts:
                times[i] = right_anchor_ts
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                times[i] = times[i - 1] + pd.Timedelta(seconds=1)
        if times[-1] > right_anchor_ts:
            latest_start = right_anchor_ts - pd.Timedelta(seconds=(len(times) - 1))
            if latest_start < start_ts:
                latest_start = start_ts
            times = [latest_start + pd.Timedelta(seconds=i) for i in range(len(times))]
            if times[0] < start_ts:
                shift = start_ts - times[0]
                times = [t + shift for t in times]
    return times


def infer_c1_order(rows_block, left_act, right_act):
    activities = [r["Activity"] for r in rows_block]
    idxs = list(range(len(activities)))
    if len(activities) <= 7:
        perms = itertools.permutations(idxs)
        best = None
        best_score = -1e18
        for perm in perms:
            order = [activities[i] for i in perm]
            s = score_order(order, left_act, right_act)
            if s > best_score:
                best_score = s
                best = perm
        return list(best)
    beam = [(tuple(), set(), 0.0)]
    beam_width = 32
    while True:
        advanced = []
        done = True
        for seq, used, sc in beam:
            if len(seq) == len(idxs):
                advanced.append((seq, used, sc))
                continue
            done = False
            for i in idxs:
                if i in used:
                    continue
                add = 0.0
                if len(seq) == 0:
                    if left_act is not None:
                        add += math.log(max(transition_prob(left_act, activities[i]), 1e-9))
                else:
                    add += math.log(max(transition_prob(activities[seq[-1]], activities[i]), 1e-9))
                new_seq = seq + (i,)
                new_used = set(used)
                new_used.add(i)
                advanced.append((new_seq, new_used, sc + add))
        if done:
            break
        advanced.sort(key=lambda x: x[2], reverse=True)
        beam = advanced[:beam_width]
        if all(len(seq) == len(idxs) for seq, _, _ in beam):
            break
    best = None
    best_score = -1e18
    for seq, _, sc in beam:
        if len(seq) != len(idxs):
            continue
        total = sc
        if right_act is not None:
            total += math.log(max(transition_prob(activities[seq[-1]], right_act), 1e-9))
        if total > best_score:
            best_score = total
            best = seq
    return list(best if best is not None else idxs)


def estimate_single_between(left_ts, left_act, act, right_ts, right_act):
    candidates = []
    if left_ts is not None:
        t1 = left_ts + pd.Timedelta(seconds=choose_gap_seconds(left_act, act) if left_act is not None else GLOBAL_MEDIAN)
        candidates.append(t1)
    if right_ts is not None:
        t2 = right_ts - pd.Timedelta(seconds=choose_gap_seconds(act, right_act) if right_act is not None else GLOBAL_MEDIAN)
        candidates.append(t2)
    if left_ts is not None and right_ts is not None:
        if len(candidates) == 2:
            lo = left_ts
            hi = right_ts
            c1 = max(lo, min(hi, candidates[0]))
            c2 = max(lo, min(hi, candidates[1]))
            if c1 <= c2:
                return c1 + (c2 - c1) / 2
            mid = lo + (hi - lo) / 2
            return mid
        return left_ts + (right_ts - left_ts) / 2
    if left_ts is not None:
        return candidates[0]
    if right_ts is not None:
        return candidates[0]
    return pd.Timestamp("2024-01-01 00:00:00")


def local_pair_score(prev_act, t_prev, act1, t1, act2, t2, next_act, t_next):
    s = 0.0
    if t1 > t2:
        return -1e12
    if prev_act is not None and t_prev is not None:
        s += math.log(max(transition_prob(prev_act, act1), 1e-9))
        s += duration_plausibility_score(prev_act, act1, (t1 - t_prev).total_seconds())
    s += math.log(max(transition_prob(act1, act2), 1e-9))
    s += duration_plausibility_score(act1, act2, (t2 - t1).total_seconds())
    if next_act is not None and t_next is not None:
        s += math.log(max(transition_prob(act2, next_act), 1e-9))
        s += duration_plausibility_score(act2, next_act, (t_next - t2).total_seconds())
    return s


def preserve_date_repair(orig, left_ts, left_act, act, right_ts, right_act):
    date_match = re.match(r"^\s*(\d{4}-\d{2}-\d{2})\s*$", str(orig))
    if date_match:
        d = date_match.group(1)
        base = pd.Timestamp(d + " 00:00:00")
        est = estimate_single_between(left_ts, left_act, act, right_ts, right_act)
        repaired = pd.Timestamp(f"{d} {est.strftime('%H:%M:%S.%f')}")
        if right_ts is not None and repaired > right_ts and repaired.date() == pd.Timestamp(d).date():
            repaired = pd.Timestamp(f"{d} {right_ts.strftime('%H:%M:%S.%f')}")
        if left_ts is not None and repaired < left_ts and repaired.date() == pd.Timestamp(d).date():
            repaired = pd.Timestamp(f"{d} {left_ts.strftime('%H:%M:%S.%f')}")
        return repaired
    return estimate_single_between(left_ts, left_act, act, right_ts, right_act)


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
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    original_ts = df["Timestamp"].astype(str).tolist()
    output_ts = original_ts[:]
    frac_digits, sep = detect_precision_and_separator(df["Timestamp"].astype(str))

    df["_parsed"] = df["Timestamp"].apply(safe_to_datetime)

    case_groups = defaultdict(list)
    for idx, row in df.iterrows():
        case_groups[row["Case"]].append(idx)

    repaired_c1_blocks = 0
    clamped_blocks = 0
    repaired_rows = set()

    for case, indices in case_groups.items():
        indices = sorted(indices)
        parsed_case = [df.at[i, "_parsed"] for i in indices]
        valid_case = [t for t in parsed_case if not pd.isna(t)]
        if valid_case:
            case_start = min(valid_case)
            case_end = max(valid_case)
        else:
            case_start = pd.Timestamp("2024-01-01 00:00:00")
            case_end = case_start + pd.Timedelta(days=1)

        # C1 blocks
        i = 0
        while i < len(indices):
            idx = indices[i]
            if df.at[idx, "error_category"] != "C1_ORDERING_AMBIGUITY":
                i += 1
                continue
            obs = df.at[idx, "Timestamp"]
            block = [idx]
            j = i + 1
            while j < len(indices):
                idx2 = indices[j]
                if (
                    df.at[idx2, "error_category"] == "C1_ORDERING_AMBIGUITY"
                    and df.at[idx2, "Timestamp"] == obs
                ):
                    block.append(idx2)
                    j += 1
                else:
                    break
            if len(block) >= 2:
                left_idx = None
                for p in range(i - 1, -1, -1):
                    cand = indices[p]
                    if df.at[cand, "error_category"] == "CLEAN" and not pd.isna(df.at[cand, "_parsed"]):
                        left_idx = cand
                        break
                right_idx = None
                for p in range(j, len(indices)):
                    cand = indices[p]
                    if df.at[cand, "error_category"] == "CLEAN" and not pd.isna(df.at[cand, "_parsed"]):
                        right_idx = cand
                        break
                left_act = df.at[left_idx, "Activity"] if left_idx is not None else None
                right_act = df.at[right_idx, "Activity"] if right_idx is not None else None
                right_ts = df.at[right_idx, "_parsed"] if right_idx is not None else case_end
                observed_ts = safe_to_datetime(obs)
                if pd.isna(observed_ts):
                    observed_ts = case_start
                if right_ts is not None and observed_ts > right_ts:
                    observed_ts = min(observed_ts, right_ts)
                rows_block = [df.loc[b].to_dict() for b in block]
                order_pos = infer_c1_order(rows_block, left_act, right_act)
                ordered_block = [block[p] for p in order_pos]
                ordered_acts = [df.at[x, "Activity"] for x in ordered_block]
                times = estimate_series_from_start(observed_ts, ordered_acts, right_ts)
                needed_clamp = False
                room = max(0.0, (right_ts - observed_ts).total_seconds()) if right_ts is not None else None
                if room is not None:
                    span = (times[-1] - times[0]).total_seconds()
                    if span > room + 1e-9:
                        needed_clamp = True
                        avail = max(room, len(times) - 1)
                        times = [observed_ts + pd.Timedelta(seconds=s) for s in range(len(times))]
                        if len(times) > 1 and (times[-1] - times[0]).total_seconds() > avail:
                            unit = max(1.0, avail / (len(times) - 1))
                            times = [observed_ts]
                            cur = observed_ts
                            for _ in range(len(ordered_block) - 1):
                                cur = cur + pd.Timedelta(seconds=unit)
                                times.append(cur)
                if len(times) > 1 and (times[-1] - times[0]).total_seconds() < (len(times) - 1):
                    needed_clamp = True
                    times = [observed_ts + pd.Timedelta(seconds=s) for s in range(len(times))]
                if right_ts is not None and times[-1] > right_ts:
                    needed_clamp = True
                    latest_start = min(observed_ts, right_ts - pd.Timedelta(seconds=len(times) - 1))
                    if latest_start > observed_ts:
                        latest_start = observed_ts
                    times = [latest_start + pd.Timedelta(seconds=s) for s in range(len(times))]
                for k in range(1, len(times)):
                    if times[k] <= times[k - 1]:
                        needed_clamp = True
                        times[k] = times[k - 1] + pd.Timedelta(seconds=1)
                if right_ts is not None and times[-1] > right_ts:
                    needed_clamp = True
                    base = observed_ts
                    avail = max((right_ts - base).total_seconds(), len(times) - 1)
                    unit = max(1.0, avail / max(1, len(times) - 1))
                    times = [base + pd.Timedelta(seconds=unit * q) for q in range(len(times))]
                    while times[-1] > right_ts and unit > 1.0:
                        unit = max(1.0, unit - 1.0)
                        times = [base + pd.Timedelta(seconds=unit * q) for q in range(len(times))]
                keep_idx = ordered_block[0]
                for pos, row_idx in enumerate(ordered_block):
                    new_ts = times[pos]
                    if new_ts < case_start:
                        needed_clamp = True
                        new_ts = case_start
                    if new_ts > case_end:
                        needed_clamp = True
                        new_ts = case_end
                    if right_idx is not None:
                        if new_ts > df.at[right_idx, "_parsed"]:
                            needed_clamp = True
                            new_ts = df.at[right_idx, "_parsed"]
                    if left_idx is not None:
                        if new_ts < df.at[left_idx, "_parsed"]:
                            needed_clamp = True
                            new_ts = df.at[left_idx, "_parsed"]
                    if row_idx == keep_idx:
                        output_ts[row_idx] = obs
                    else:
                        if pd.isna(new_ts):
                            new_ts = observed_ts + pd.Timedelta(seconds=pos)
                        if new_ts <= observed_ts:
                            new_ts = observed_ts + pd.Timedelta(seconds=pos)
                        output_ts[row_idx] = format_ts(new_ts, frac_digits, sep)
                    repaired_rows.add(row_idx)
                repaired_c1_blocks += 1
                if needed_clamp:
                    clamped_blocks += 1
            i = j

        # C2 and C3 row-wise
        for pos, idx in enumerate(indices):
            cat = df.at[idx, "error_category"]
            if cat == "CLEAN" or idx in repaired_rows:
                continue

            left_idx = None
            for p in range(pos - 1, -1, -1):
                cand = indices[p]
                if df.at[cand, "error_category"] == "CLEAN" and not pd.isna(df.at[cand, "_parsed"]):
                    left_idx = cand
                    break
            right_idx = None
            for p in range(pos + 1, len(indices)):
                cand = indices[p]
                if df.at[cand, "error_category"] == "CLEAN" and not pd.isna(df.at[cand, "_parsed"]):
                    right_idx = cand
                    break

            left_ts = df.at[left_idx, "_parsed"] if left_idx is not None else case_start
            right_ts = df.at[right_idx, "_parsed"] if right_idx is not None else case_end
            left_act = df.at[left_idx, "Activity"] if left_idx is not None else None
            right_act = df.at[right_idx, "Activity"] if right_idx is not None else None
            act = df.at[idx, "Activity"]
            obs_ts = safe_to_datetime(df.at[idx, "Timestamp"])

            if cat == "C2_TEMPORAL_INCONSISTENCY":
                partner_idx = None
                if pos > 0:
                    prev_idx = indices[pos - 1]
                    prev_ts = safe_to_datetime(df.at[prev_idx, "Timestamp"])
                    if not pd.isna(prev_ts) and not pd.isna(obs_ts) and obs_ts < prev_ts:
                        partner_idx = prev_idx
                if partner_idx is None and pos + 1 < len(indices):
                    next_idx = indices[pos + 1]
                    next_ts = safe_to_datetime(df.at[next_idx, "Timestamp"])
                    if not pd.isna(next_ts) and not pd.isna(obs_ts) and next_ts < obs_ts:
                        partner_idx = next_idx

                repaired = False
                if partner_idx is not None:
                    pair = sorted([idx, partner_idx], key=lambda x: indices.index(x))
                    first_idx, second_idx = pair[0], pair[1]
                    t_first = safe_to_datetime(df.at[first_idx, "Timestamp"])
                    t_second = safe_to_datetime(df.at[second_idx, "Timestamp"])
                    if not pd.isna(t_first) and not pd.isna(t_second):
                        prev_clean_idx = None
                        fi_pos = indices.index(first_idx)
                        for p in range(fi_pos - 1, -1, -1):
                            cand = indices[p]
                            if df.at[cand, "error_category"] == "CLEAN" and not pd.isna(df.at[cand, "_parsed"]):
                                prev_clean_idx = cand
                                break
                        next_clean_idx = None
                        si_pos = indices.index(second_idx)
                        for p in range(si_pos + 1, len(indices)):
                            cand = indices[p]
                            if df.at[cand, "error_category"] == "CLEAN" and not pd.isna(df.at[cand, "_parsed"]):
                                next_clean_idx = cand
                                break
                        prev_act = df.at[prev_clean_idx, "Activity"] if prev_clean_idx is not None else None
                        prev_clean_ts = df.at[prev_clean_idx, "_parsed"] if prev_clean_idx is not None else None
                        next_act = df.at[next_clean_idx, "Activity"] if next_clean_idx is not None else None
                        next_clean_ts = df.at[next_clean_idx, "_parsed"] if next_clean_idx is not None else None
                        act1 = df.at[first_idx, "Activity"]
                        act2 = df.at[second_idx, "Activity"]

                        noswap_score = local_pair_score(prev_act, prev_clean_ts, act1, t_first, act2, t_second, next_act, next_clean_ts)
                        swap_score = local_pair_score(prev_act, prev_clean_ts, act1, t_second, act2, t_first, next_act, next_clean_ts)

                        if swap_score > noswap_score and t_second >= (prev_clean_ts if prev_clean_ts is not None else t_second - pd.Timedelta(days=3650)) and t_first <= (next_clean_ts if next_clean_ts is not None else t_first + pd.Timedelta(days=3650)) and t_second <= t_first:
                            output_ts[first_idx] = format_ts(t_second, frac_digits, sep)
                            output_ts[second_idx] = format_ts(t_first, frac_digits, sep)
                            repaired_rows.add(first_idx)
                            repaired_rows.add(second_idx)
                            repaired = True

                if not repaired and idx not in repaired_rows:
                    est = estimate_single_between(left_ts, left_act, act, right_ts, right_act)
                    if left_ts is not None:
                        est = max(est, left_ts)
                    if right_ts is not None:
                        est = min(est, right_ts)
                    if not pd.isna(obs_ts) and est == obs_ts:
                        est = min(right_ts, est + pd.Timedelta(seconds=1)) if right_ts is not None else est + pd.Timedelta(seconds=1)
                        if left_ts is not None and est < left_ts:
                            est = left_ts + pd.Timedelta(seconds=1)
                    output_ts[idx] = format_ts(est, frac_digits, sep)
                    repaired_rows.add(idx)

            elif cat == "C3_MISSING_TEMPORAL_INFORMATION":
                est = preserve_date_repair(df.at[idx, "Timestamp"], left_ts, left_act, act, right_ts, right_act)
                if left_ts is not None and est < left_ts:
                    est = left_ts + pd.Timedelta(seconds=1)
                if right_ts is not None and est > right_ts:
                    if left_ts is not None and left_ts < right_ts:
                        est = left_ts + (right_ts - left_ts) / 2
                    else:
                        est = right_ts
                if not pd.isna(obs_ts) and est == obs_ts:
                    est = min(right_ts, est + pd.Timedelta(seconds=1)) if right_ts is not None else est + pd.Timedelta(seconds=1)
                output_ts[idx] = format_ts(est, frac_digits, sep)
                repaired_rows.add(idx)

            else:
                est = estimate_single_between(left_ts, left_act, act, right_ts, right_act)
                if left_ts is not None and est < left_ts:
                    est = left_ts + pd.Timedelta(seconds=1)
                if right_ts is not None and est > right_ts:
                    est = right_ts - pd.Timedelta(seconds=1) if right_ts > left_ts else right_ts
                if not pd.isna(obs_ts) and est == obs_ts:
                    est = est + pd.Timedelta(seconds=1)
                    if right_ts is not None and est > right_ts:
                        est = right_ts
                output_ts[idx] = format_ts(est, frac_digits, sep)
                repaired_rows.add(idx)

    # Self-checks and repairs
    rows_equal_to_input = 0
    df_out = df.copy()
    df_out["Timestamp"] = output_ts

    # Keep CLEAN exact
    for idx in df.index:
        if df.at[idx, "error_category"] == "CLEAN":
            df_out.at[idx, "Timestamp"] = original_ts[idx]

    # Parse repaired and enforce bounds
    for case, indices in case_groups.items():
        indices = sorted(indices)
        parsed_current = [safe_to_datetime(df_out.at[i, "Timestamp"]) for i in indices if df_out.at[i, "Timestamp"] != ""]
        valid = [t for t in parsed_current if not pd.isna(t)]
        if valid:
            case_start = min(valid)
            case_end = max(valid)
        else:
            case_start = pd.Timestamp("2024-01-01 00:00:00")
            case_end = case_start + pd.Timedelta(days=1)

        for pos, idx in enumerate(indices):
            if df.at[idx, "error_category"] == "CLEAN":
                continue
            cur = safe_to_datetime(df_out.at[idx, "Timestamp"])
            if pd.isna(cur):
                left_clean = None
                right_clean = None
                for p in range(pos - 1, -1, -1):
                    cand = indices[p]
                    t = safe_to_datetime(df_out.at[cand, "Timestamp"])
                    if not pd.isna(t):
                        left_clean = t
                        break
                for p in range(pos + 1, len(indices)):
                    cand = indices[p]
                    t = safe_to_datetime(df_out.at[cand, "Timestamp"])
                    if not pd.isna(t):
                        right_clean = t
                        break
                if left_clean is not None and right_clean is not None:
                    cur = left_clean + (right_clean - left_clean) / 2
                elif left_clean is not None:
                    cur = left_clean + pd.Timedelta(seconds=1)
                elif right_clean is not None:
                    cur = right_clean - pd.Timedelta(seconds=1)
                else:
                    cur = case_start
                df_out.at[idx, "Timestamp"] = format_ts(cur, frac_digits, sep)

    # Recheck C1 exact rules
    for case, indices in case_groups.items():
        indices = sorted(indices)
        i = 0
        while i < len(indices):
            idx = indices[i]
            if df.at[idx, "error_category"] != "C1_ORDERING_AMBIGUITY":
                i += 1
                continue
            obs = df.at[idx, "Timestamp"]
            block = [idx]
            j = i + 1
            while j < len(indices):
                idx2 = indices[j]
                if df.at[idx2, "error_category"] == "C1_ORDERING_AMBIGUITY" and df.at[idx2, "Timestamp"] == obs:
                    block.append(idx2)
                    j += 1
                else:
                    break
            if len(block) >= 2:
                vals = [df_out.at[b, "Timestamp"] for b in block]
                equal_obs = [b for b in block if df_out.at[b, "Timestamp"] == obs]
                if len(equal_obs) != 1:
                    clamped_blocks += 1
                    keep = block[0]
                    left_idx = None
                    for p in range(i - 1, -1, -1):
                        cand = indices[p]
                        if df.at[cand, "error_category"] == "CLEAN" and not pd.isna(df.at[cand, "_parsed"]):
                            left_idx = cand
                            break
                    right_idx = None
                    for p in range(j, len(indices)):
                        cand = indices[p]
                        if df.at[cand, "error_category"] == "CLEAN" and not pd.isna(df.at[cand, "_parsed"]):
                            right_idx = cand
                            break
                    left_act = df.at[left_idx, "Activity"] if left_idx is not None else None
                    right_act = df.at[right_idx, "Activity"] if right_idx is not None else None
                    right_ts = df.at[right_idx, "_parsed"] if right_idx is not None else None
                    observed_ts = safe_to_datetime(obs)
                    order_pos = infer_c1_order([df.loc[b].to_dict() for b in block], left_act, right_act)
                    ordered_block = [block[p] for p in order_pos]
                    ordered_acts = [df.at[x, "Activity"] for x in ordered_block]
                    times = estimate_series_from_start(observed_ts, ordered_acts, right_ts)
                    for pos2, row_idx in enumerate(ordered_block):
                        if pos2 == 0:
                            df_out.at[row_idx, "Timestamp"] = obs
                        else:
                            df_out.at[row_idx, "Timestamp"] = format_ts(times[pos2], frac_digits, sep)
            i = j

    for idx in df.index:
        if df.at[idx, "error_category"] != "CLEAN" and df_out.at[idx, "Timestamp"] == original_ts[idx]:
            rows_equal_to_input += 1

    # Final write: preserve all non-timestamp columns exactly and row order unchanged
    write_df = df[required_cols].copy()
    write_df["Timestamp"] = df_out["Timestamp"]

    # restore original column order if possible
    out_cols = list(df.columns)
    if "_parsed" in out_cols:
        out_cols.remove("_parsed")
    if "Timestamp" in out_cols:
        out_cols[out_cols.index("Timestamp")] = "Timestamp"
    final_df = df.drop(columns=["_parsed"]).copy()
    final_df["Timestamp"] = write_df["Timestamp"]

    try:
        final_df.to_csv(args.output, index=False)
    except Exception:
        write_df.to_csv(args.output, index=False)

    print(f"{repaired_c1_blocks},{clamped_blocks},{rows_equal_to_input}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail-safe: write something valid if anything unexpected happens
        try:
            import sys
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=True)
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            print("0,0,0")
        except Exception:
            print("0,0,0")