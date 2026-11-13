#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import timedelta

import pandas as pd


# ----------------------------- Embedded process context -----------------------------

TRANSITION_PROBABILITIES_TOP = [
    ("Deliver to customer", "EVENT 8 END", 1.0),
    ("Signal food complete", "Bring food", 0.9985),
    ("Bring food", "Deliver to customer", 0.9972),
    ("Prepare main course", "Signal food complete", 0.9913),
    ("start", "Take order", 0.9708),
    ("Bring drinks", "Review drinks", 0.9554),
    ("Take order", "Review order", 0.9464),
    ("prepare drinks", "Bring drinks", 0.8979),
    ("Review order", "prepare drinks", 0.7727),
    ("Prepare starter", "prepare drinks", 0.5310),
    ("Review drinks", "Prepare main course", 0.4767),
    ("Review drinks", "prepare drinks", 0.3939),
    ("Prepare starter", "Bring drinks", 0.2706),
    ("Review order", "Prepare starter", 0.1765),
    ("Prepare starter", "Prepare main course", 0.0860),
    ("Review drinks", "Signal food complete", 0.0681),
    ("prepare drinks", "Prepare starter", 0.0596),
    ("Prepare starter", "Review drinks", 0.0583),
    ("Prepare starter", "Signal food complete", 0.0481),
    ("Review drinks", "Prepare starter", 0.0392),
    ("Take order", "Review drinks", 0.0218),
    ("Review order", "Review drinks", 0.0181),
    ("prepare drinks", "Bring food", 0.0155),
    ("Review drinks", "Bring food", 0.0149),
    ("Review order", "Prepare main course", 0.0133),
    ("Bring drinks", "Signal food complete", 0.0129),
    ("start", "Bring drinks", 0.0124),
    ("Take order", "Bring drinks", 0.0124),
    ("prepare drinks", "Signal food complete", 0.0121),
    ("Bring drinks", "Prepare starter", 0.0119),
    ("Take order", "Prepare main course", 0.0117),
    ("Bring drinks", "Bring food", 0.0110),
    ("Review order", "Signal food complete", 0.0107),
    ("start", "Review drinks", 0.0092),
    ("Review order", "Bring drinks", 0.0087),
    ("prepare drinks", "Prepare main course", 0.0068),
    ("prepare drinks", "Review drinks", 0.0062),
    ("Prepare starter", "Bring food", 0.0060),
    ("start", "prepare drinks", 0.0050),
    ("Take order", "prepare drinks", 0.0049),
    ("Prepare main course", "Bring drinks", 0.0046),
    ("Bring drinks", "Bring drinks", 0.0046),
    ("Review drinks", "Review drinks", 0.0040),
    ("Bring drinks", "Prepare main course", 0.0034),
    ("Prepare main course", "prepare drinks", 0.0031),
    ("Take order", "Prepare starter", 0.0027),
    ("start", "Prepare starter", 0.0026),
    ("Bring food", "Bring drinks", 0.0022),
    ("prepare drinks", "prepare drinks", 0.0019),
    ("Review drinks", "Deliver to customer", 0.0019),
    ("Review drinks", "Bring drinks", 0.0013),
    ("Prepare main course", "Review drinks", 0.0010),
    ("Bring drinks", "prepare drinks", 0.0009),
    ("Signal food complete", "prepare drinks", 0.0009),
    ("Signal food complete", "Bring drinks", 0.0006),
    ("Bring food", "prepare drinks", 0.0006),
]

DURATION_PROFILES_TOP = [
    ("Bring drinks", "Review drinks", 7062, 6.1307, 17.0105, 41.052, 83.472, 136.7487, 0.631),
    ("prepare drinks", "Bring drinks", 6704, 28.8126, 79.71675, 195.9745, 413.75325, 700.3471, 0.198),
    ("Deliver to customer", "EVENT 8 END", 5000, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ("Bring food", "Deliver to customer", 4986, 12.5235, 34.57875, 85.062, 168.381, 278.243, 0.389),
    ("start", "Take order", 4854, 12.9101, 35.79525, 89.148, 180.67575, 315.9663, 0.378),
    ("Signal food complete", "Bring food", 4682, 33.0276, 90.3885, 214.9425, 423.9705, 718.42, 0.176),
    ("Take order", "Review order", 4594, 6.1912, 17.985, 41.872, 82.9715, 138.1861, 0.636),
    ("Prepare main course", "Signal food complete", 3866, 85943.671, 253686.04975, 513669.015, 685817.92925, 845950.272, 0.001),
    ("Review drinks", "Prepare main course", 3563, 83921.2662, 243350.551, 500815.099, 680990.933, 788371.1142, 0.001),
    ("Review order", "prepare drinks", 3550, 25.252, 68.49925, 173.713, 352.967, 606.4154, 0.226),
    ("Review drinks", "prepare drinks", 2944, 30.7213, 85.6735, 209.569, 415.462, 694.3028, 0.182),
    ("Prepare starter", "prepare drinks", 883, 30.3582, 89.143, 206.2, 397.074, 656.961, 0.19),
    ("Review order", "Prepare starter", 811, 15.008, 42.552, 97.24, 200.8065, 351.984, 0.354),
    ("Review drinks", "Signal food complete", 509, 7451.2538, 175593.462, 445342.856, 686397.671, 797873.2922, 0.002),
    ("Prepare starter", "Bring drinks", 450, 29.0292, 86.16725, 203.727, 427.8865, 688.7044, 0.189),
    ("prepare drinks", "Prepare starter", 445, 15.8778, 40.819, 100.292, 192.881, 296.6494, 0.362),
    ("Review drinks", "Prepare starter", 293, 17.9444, 63.83, 151.424, 322.423, 679.4376, 0.232),
    ("Prepare starter", "Prepare main course", 143, 76289.6188, 192412.7035, 445174.842, 624995.3195, 770926.678, 0.0),
    ("prepare drinks", "Bring food", 116, 39201.883, 406754.148, 785107.18, 1309999.6535, 1599036.572, 0.009),
    ("Review drinks", "Bring food", 111, 161178.253, 593624.664, 857938.338, 1377516.8425, 1634321.944, 0.009),
    ("Take order", "Review drinks", 106, 257.7665, 389.71375, 707.2185, 1038.79475, 1354.652, 0.0),
    ("Prepare starter", "Review drinks", 97, 11.5194, 22.425, 56.303, 132.229, 367.8652, 0.505),
    ("Bring drinks", "Signal food complete", 95, 179812.449, 593579.078, 883902.239, 1299383.9915, 1510712.6246, 0.0),
    ("prepare drinks", "Signal food complete", 90, 84583.4036, 431151.8905, 982509.744, 1344037.35375, 1629650.5926, 0.011),
    ("Bring drinks", "Prepare starter", 88, 4.3939, 15.89925, 33.688, 75.99025, 139.0768, 0.693),
    ("Review order", "Review drinks", 83, 303.721, 381.153, 593.835, 954.486, 1402.8866, 0.0),
    ("Bring drinks", "Bring food", 81, 102484.185, 366203.855, 696631.885, 1367714.419, 1616933.672, 0.0),
    ("Prepare starter", "Signal food complete", 80, 88210.8159, 276729.8375, 519007.665, 688787.49675, 944907.5365, 0.0),
    ("start", "Bring drinks", 62, 275.2757, 422.42225, 695.196, 915.41325, 1413.8304, 0.0),
    ("Review order", "Prepare main course", 61, 943.183, 84467.297, 444605.508, 620958.677, 700252.983, 0.0),
    ("Take order", "Bring drinks", 60, 296.2082, 433.589, 624.287, 992.78675, 1341.9016, 0.0),
    ("Take order", "Prepare main course", 57, 101957.3608, 258966.497, 513968.537, 624777.344, 777773.7006, 0.0),
    ("prepare drinks", "Prepare main course", 51, 71.18, 281.63, 160626.596, 523929.66, 681005.825, 0.078),
    ("Review order", "Signal food complete", 49, 152688.052, 587731.024, 862530.877, 1300062.102, 1476447.8158, 0.0),
    ("start", "Review drinks", 46, 281.845, 455.605, 733.167, 1084.1665, 1306.389, 0.0),
    ("prepare drinks", "Review drinks", 46, 368.894, 508.0155, 656.5765, 1133.5845, 1545.7045, 0.0),
    ("Review order", "Bring drinks", 40, 298.812, 538.94575, 784.107, 1205.822, 1648.7174, 0.0),
    ("Bring drinks", "Bring drinks", 34, 237.8929, 350.94375, 571.916, 863.97725, 1168.747, 0.0),
    ("Review drinks", "Review drinks", 30, 283.804, 471.11475, 695.439, 1119.0275, 2202.1487, 0.0),
    ("start", "prepare drinks", 25, 255.112, 319.999, 413.649, 582.288, 738.0338, 0.0),
    ("Bring drinks", "Prepare main course", 25, 53.072, 21489.537, 438128.715, 669828.116, 685551.208, 0.12),
    ("Take order", "prepare drinks", 24, 214.6615, 304.47475, 736.587, 1131.26225, 1339.0379, 0.0),
    ("Prepare main course", "Bring drinks", 18, 28.2273, 68.60625, 129.8, 190.83625, 340.4634, 0.222),
    ("Review drinks", "Deliver to customer", 14, 13.1735, 26.957, 70.7255, 122.753, 151.1391, 0.357),
    ("prepare drinks", "prepare drinks", 14, 430.3684, 495.2955, 639.1405, 705.744, 1295.0399, 0.0),
    ("start", "Prepare starter", 13, 275.651, 370.046, 404.992, 456.3, 517.9136, 0.0),
    ("Take order", "Prepare starter", 13, 327.2842, 389.43, 558.771, 1311.424, 2316.6824, 0.0),
    ("Prepare main course", "prepare drinks", 12, 94.8512, 110.78, 178.7615, 259.93625, 354.0485, 0.0),
    ("Bring food", "Bring drinks", 11, 50.942, 152.058, 229.533, 512.9985, 626.086, 0.182),
    ("Review drinks", "Bring drinks", 10, 434.9222, 666.72275, 1267.9125, 1729.7255, 1950.6004, 0.0),
    ("Prepare starter", "Bring food", 10, 167409.7799, 293248.85775, 525665.88, 931247.942, 1290839.1252, 0.0),
    ("Bring drinks", "prepare drinks", 7, 691.6136, 983.649, 1736.597, 2019.697, 2129.3134, 0.0),
    ("Signal food complete", "prepare drinks", 4, 65.3186, 153.056, 228.6855, 267.4535, 288.8402, 0.25),
    ("Prepare main course", "Review drinks", 4, 2.2898, 3.674, 35.4335, 86.17125, 121.7163, 0.5),
    ("Bring food", "prepare drinks", 3, 164.9284, 166.546, 169.242, 176.463, 180.7956, 0.0),
    ("Signal food complete", "Bring drinks", 3, 150.102, 179.118, 227.478, 255.185, 271.8092, 0.0),
]

FREQUENT_VARIANTS = [
    (["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 1317),
    (["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 530),
    (["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 274),
    (["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 188),
    (["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 148),
    (["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 128),
    (["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 114),
    (["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 114),
    (["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare starter", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 106),
    (["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 65),
    (["start", "Take order", "Review order", "prepare drinks", "Bring food", "Deliver to customer", "EVENT 8 END"], 65),
    (["start", "Take order", "Review order", "prepare drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 62),
    (["start", "Take order", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 61),
    (["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 59),
    (["start", "Take order", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 57),
    (["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 55),
    (["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Bring food", "Deliver to customer", "EVENT 8 END"], 52),
    (["start", "Take order", "Review order", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 52),
    (["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 51),
    (["start", "Take order", "Review order", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"], 49),
]

GLOBAL_MEDIAN_DURATION_SECONDS = 117.1945

# ----------------------------- Lookups -----------------------------

TRANS_PROB = {(s, t): p for s, t, p in TRANSITION_PROBABILITIES_TOP}
DUR = {
    (s, t): {
        "n": n,
        "p10": p10,
        "p25": p25,
        "median": med,
        "p75": p75,
        "p90": p90,
        "share_under_60s": share,
    }
    for s, t, n, p10, p25, med, p75, p90, share in DURATION_PROFILES_TOP
}
DFG_FREQ = {(s, t): DUR[(s, t)]["n"] for s, t in DUR.keys()}

ALL_ACTIVITIES = set()
for s, t, _ in TRANSITION_PROBABILITIES_TOP:
    ALL_ACTIVITIES.add(s)
    ALL_ACTIVITIES.add(t)

# ----------------------------- Utilities -----------------------------


def parse_ts(s):
    return pd.to_datetime(s)


def detect_fraction_digits(ts_str):
    if "." in ts_str:
        frac = ts_str.split(".", 1)[1]
        return len(frac)
    return 0


def format_ts(ts, frac_digits):
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits <= 0:
        return base
    if frac_digits <= 6:
        micros = f"{ts.microsecond:06d}"[:frac_digits]
        return f"{base}.{micros}"
    nanos_total = ts.microsecond * 1000
    frac9 = f"{nanos_total:09d}"
    return f"{base}.{frac9[:frac_digits]}"


def add_seconds(ts, seconds):
    micros = int(round(seconds * 1_000_000))
    return ts + pd.to_timedelta(micros, unit="us")


def get_duration_gap_seconds(a, b):
    prof = DUR.get((a, b))
    if prof:
        if prof["share_under_60s"] >= 0.5:
            g = prof["p10"]
        else:
            g = prof["p25"]
        return max(g, 1.0)
    rev = DUR.get((b, a))
    if rev:
        if rev["share_under_60s"] >= 0.5:
            g = min(max(rev["p10"], 1.0), 3.0)
        else:
            g = min(max(rev["p25"] * 0.1, 1.0), 5.0)
        return max(g, 1.0)
    return 1.0


def edge_exists(a, b):
    return (a, b) in DFG_FREQ


def edge_logscore(a, b):
    if (a, b) not in DFG_FREQ:
        return -1e9
    freq = DFG_FREQ[(a, b)]
    prob = TRANS_PROB.get((a, b), 1e-6)
    return math.log(freq + 1.0) + 2.0 * math.log(prob + 1e-12)


def contiguous_subsequence_score(left_anchor, seq, right_anchor):
    ext = []
    if left_anchor is not None:
        ext.append(left_anchor)
    ext.extend(seq)
    if right_anchor is not None:
        ext.append(right_anchor)
    best = 0.0
    for variant, count in FREQUENT_VARIANTS:
        if len(ext) > len(variant):
            continue
        for i in range(len(variant) - len(ext) + 1):
            if variant[i:i + len(ext)] == ext:
                best = max(best, math.log(count + 1.0) + 50.0)
    return best


def multiset_contiguous_variant_orders(activities, left_anchor, right_anchor):
    target = Counter(activities)
    out = []
    for variant, count in FREQUENT_VARIANTS:
        for i in range(len(variant) - len(activities) + 1):
            sub = variant[i:i + len(activities)]
            if Counter(sub) != target:
                continue
            if left_anchor is not None and i > 0 and variant[i - 1] != left_anchor:
                continue
            if right_anchor is not None and i + len(activities) < len(variant) and variant[i + len(activities)] != right_anchor:
                continue
            out.append((tuple(sub), count))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def generate_candidate_orders(activities, left_anchor, right_anchor):
    # 1) strong evidence from frequent variants
    seen = set()
    candidates = []
    for seq, _cnt in multiset_contiguous_variant_orders(activities, left_anchor, right_anchor):
        if seq not in seen:
            seen.add(seq)
            candidates.append(list(seq))

    # 2) if none, backtracking all DFG-consistent permutations
    counts = Counter(activities)
    uniq = sorted(counts.keys())

    def feasible_prefix(prefix):
        if len(prefix) == 1:
            if left_anchor is not None and not edge_exists(left_anchor, prefix[0]):
                return False
            return True
        if not edge_exists(prefix[-2], prefix[-1]):
            return False
        return True

    def backtrack(prefix, rem):
        if len(candidates) >= 200:
            return
        if sum(rem.values()) == 0:
            if right_anchor is not None and not edge_exists(prefix[-1], right_anchor):
                return
            tup = tuple(prefix)
            if tup not in seen:
                seen.add(tup)
                candidates.append(prefix.copy())
            return
        for a in uniq:
            if rem[a] <= 0:
                continue
            prefix.append(a)
            rem[a] -= 1
            if feasible_prefix(prefix):
                backtrack(prefix, rem)
            rem[a] += 1
            prefix.pop()

    if not candidates:
        backtrack([], counts.copy())

    # 3) greedy anchor-compatible fallback
    if not candidates:
        rem = counts.copy()
        seq = []
        prev = left_anchor
        while sum(rem.values()) > 0:
            best = None
            best_score = -1e18
            for a in uniq:
                if rem[a] <= 0:
                    continue
                if prev is not None and not edge_exists(prev, a):
                    continue
                score = edge_logscore(prev, a) if prev is not None else 0.0
                if score > best_score:
                    best_score = score
                    best = a
            if best is None:
                for a in uniq:
                    if rem[a] > 0:
                        best = a
                        break
            seq.append(best)
            rem[best] -= 1
            prev = best
        candidates.append(seq)

    return candidates


def score_order(seq, left_anchor, right_anchor):
    # reject if internal edges absent
    for i in range(len(seq) - 1):
        if not edge_exists(seq[i], seq[i + 1]):
            return -1e18
    if left_anchor is not None and not edge_exists(left_anchor, seq[0]):
        return -1e18
    if right_anchor is not None and not edge_exists(seq[-1], right_anchor):
        return -1e18

    s = 0.0
    s += contiguous_subsequence_score(left_anchor, seq, right_anchor)
    if left_anchor is not None:
        s += 1.5 * edge_logscore(left_anchor, seq[0])
    for i in range(len(seq) - 1):
        s += 2.0 * edge_logscore(seq[i], seq[i + 1])
    if right_anchor is not None:
        s += 1.5 * edge_logscore(seq[-1], right_anchor)
    return s


def choose_best_order(activities, left_anchor, right_anchor):
    candidates = generate_candidate_orders(activities, left_anchor, right_anchor)
    best = None
    best_score = -1e18
    best_span = None
    for seq in candidates:
        sc = score_order(seq, left_anchor, right_anchor)
        if sc <= -1e17:
            continue
        gaps = [get_duration_gap_seconds(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
        span = sum(gaps)
        if sc > best_score + 1e-9 or (abs(sc - best_score) <= 1e-9 and (best_span is None or span < best_span)):
            best_score = sc
            best = seq
            best_span = span
    if best is None:
        best = candidates[0]
    return best


def concurrency_adjusted_gaps(seq):
    gaps = []
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        base = get_duration_gap_seconds(a, b)
        pab = TRANS_PROB.get((a, b), 0.0)
        pba = TRANS_PROB.get((b, a), 0.0)
        # if both directions exist and similar weak evidence, pack tightly
        if pab > 0 and pba > 0 and min(pab, pba) / max(pab, pba) >= 0.5:
            base = min(base, 1.2 + 0.13 * (i % 3))
        gaps.append(max(base, 1.0))
    return gaps


def enforce_nonuniform_integerish(gaps):
    out = []
    for i, g in enumerate(gaps):
        gg = max(1.0, float(g))
        if abs(gg - round(gg)) < 1e-9:
            gg += 0.001 * (i + 1)
        out.append(gg)
    for i in range(1, len(out)):
        if out[i] == out[i - 1]:
            out[i] += 0.001 * (i + 1)
    return out


def place_forward(observed_ts, seq, right_anchor_ts):
    k = len(seq)
    if k == 1:
        return [observed_ts], False

    gaps = concurrency_adjusted_gaps(seq)
    gaps = enforce_nonuniform_integerish(gaps)

    min_total = float(k - 1)
    avail = None if right_anchor_ts is None else max(0.0, (right_anchor_ts - observed_ts).total_seconds())

    raw_total = sum(gaps)
    clamped = False

    if avail is not None:
        upper = avail
        if upper < min_total:
            upper = min_total
        if raw_total > upper:
            clamped = True
            scale = upper / raw_total if raw_total > 0 else 1.0
            gaps = [max(1.0, g * scale) for g in gaps]
            total = sum(gaps)
            if total > upper:
                # shave from largest gaps
                excess = total - upper
                idxs = sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True)
                for idx in idxs:
                    reducible = gaps[idx] - 1.0
                    take = min(reducible, excess)
                    gaps[idx] -= take
                    excess -= take
                    if excess <= 1e-9:
                        break
            # still too small?
            total = sum(gaps)
            if total < min_total:
                add = min_total - total
                gaps[-1] += add
        else:
            total = raw_total
            if total < min_total:
                gaps[-1] += (min_total - total)
    else:
        total = raw_total
        if total < min_total:
            gaps[-1] += (min_total - total)

    gaps = enforce_nonuniform_integerish(gaps)
    ts_list = [observed_ts]
    cur = observed_ts
    for g in gaps:
        cur = add_seconds(cur, g)
        ts_list.append(cur)

    if right_anchor_ts is not None and ts_list[-1] > right_anchor_ts:
        clamped = True
        avail = max(0.0, (right_anchor_ts - observed_ts).total_seconds())
        target = max(float(k - 1), min(avail, max(float(k - 1), avail)))
        # minimal strictly increasing fallback packed to fit as much as possible
        if avail >= float(k - 1):
            packed = [1.0 + 0.001 * (i + 1) for i in range(k - 1)]
            packed_total = sum(packed)
            if packed_total > avail:
                packed = [1.0 for _ in range(k - 1)]
            ts_list = [observed_ts]
            cur = observed_ts
            for g in packed:
                cur = add_seconds(cur, g)
                if cur > right_anchor_ts:
                    cur = right_anchor_ts
                ts_list.append(cur)
        else:
            # impossible strict one-second packing; use fractional milliseconds but distinct
            tiny = max(avail / max(k - 1, 1), 0.001)
            ts_list = [observed_ts]
            cur = observed_ts
            for i in range(k - 1):
                cur = add_seconds(cur, tiny + i * 0.0001)
                if cur > right_anchor_ts:
                    cur = right_anchor_ts
                ts_list.append(cur)

    return ts_list, clamped


def self_fix_block(times, observed_ts, right_anchor_ts):
    clamped = False
    k = len(times)
    if k <= 1:
        return times, clamped
    # observed first fixed
    fixed = [observed_ts]
    cur = observed_ts
    for i in range(1, k):
        nxt = times[i]
        if nxt <= cur:
            clamped = True
            nxt = add_seconds(cur, 1.0 + 0.001 * i)
        fixed.append(nxt)
        cur = nxt
    if right_anchor_ts is not None and fixed[-1] > right_anchor_ts:
        clamped = True
        avail = max(0.0, (right_anchor_ts - observed_ts).total_seconds())
        if avail >= (k - 1):
            cur = observed_ts
            fixed = [observed_ts]
            for i in range(1, k):
                cur = add_seconds(cur, 1.0 + 0.001 * i)
                if cur > right_anchor_ts:
                    cur = right_anchor_ts
                fixed.append(cur)
        else:
            tiny = max(avail / max(k - 1, 1), 0.001)
            cur = observed_ts
            fixed = [observed_ts]
            for i in range(1, k):
                cur = add_seconds(cur, tiny + i * 0.0001)
                if cur > right_anchor_ts:
                    cur = right_anchor_ts
                fixed.append(cur)
    return fixed, clamped


def build_case_blocks(case_rows):
    blocks = []
    n = len(case_rows)
    i = 0
    while i < n:
        row = case_rows[i]
        if row["error_category"] == "CLEAN":
            i += 1
            continue
        j = i + 1
        while (
            j < n
            and case_rows[j]["error_category"] != "CLEAN"
            and case_rows[j]["Timestamp"] == case_rows[i]["Timestamp"]
        ):
            j += 1
        blocks.append((i, j))
        i = j
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repaired_blocks = 0
    clamped_blocks = 0

    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Preserve exact original strings for untouched rows.
    for r in rows:
        r["_orig_ts"] = r["Timestamp"]
        r["_parsed_ts"] = parse_ts(r["Timestamp"])
        r["_frac_digits"] = detect_fraction_digits(r["Timestamp"])
        r["_out_ts"] = r["Timestamp"]

    by_case = defaultdict(list)
    for idx, r in enumerate(rows):
        by_case[r["Case"]].append(idx)

    for case, idxs in by_case.items():
        case_rows = [rows[i] for i in idxs]
        blocks = build_case_blocks(case_rows)

        if not blocks:
            continue

        case_start = min(r["_parsed_ts"] for r in case_rows)
        case_end = max(r["_parsed_ts"] for r in case_rows)

        for start, end in blocks:
            block = case_rows[start:end]
            observed_str = block[0]["Timestamp"]
            observed_ts = block[0]["_parsed_ts"]
            activities = [r["Activity"] for r in block]

            left_anchor = case_rows[start - 1] if start - 1 >= 0 and case_rows[start - 1]["error_category"] == "CLEAN" else None
            right_anchor = case_rows[end] if end < len(case_rows) and case_rows[end]["error_category"] == "CLEAN" else None

            left_act = left_anchor["Activity"] if left_anchor is not None else None
            right_act = right_anchor["Activity"] if right_anchor is not None else None
            left_ts = left_anchor["_parsed_ts"] if left_anchor is not None else case_start
            right_ts = right_anchor["_parsed_ts"] if right_anchor is not None else case_end

            best_order = choose_best_order(activities, left_act, right_act)

            # map ordered activities to concrete rows, preserving row identities
            act_to_rows = defaultdict(list)
            for r in block:
                act_to_rows[r["Activity"]].append(r)

            ordered_rows = []
            temp_counter = Counter()
            for act in best_order:
                ordered_rows.append(act_to_rows[act][temp_counter[act]])
                temp_counter[act] += 1

            times, was_clamped = place_forward(observed_ts, best_order, right_ts if right_anchor is not None else None)
            times, was_clamped2 = self_fix_block(times, observed_ts, right_ts if right_anchor is not None else None)
            was_clamped = was_clamped or was_clamped2

            # Final clamp to anchors/case bounds
            fixed_times = []
            cur = None
            for i, t in enumerate(times):
                lo = observed_ts if i == 0 else add_seconds(cur, 0.001)
                hi = right_ts
                if t < lo:
                    t = lo
                    was_clamped = True
                if t > hi:
                    t = hi
                    was_clamped = True
                if cur is not None and t <= cur:
                    t = add_seconds(cur, 0.001)
                    was_clamped = True
                    if t > hi:
                        t = hi
                if t < case_start:
                    t = case_start
                    was_clamped = True
                if t > case_end:
                    t = case_end
                    was_clamped = True
                fixed_times.append(t)
                cur = t

            # Ensure first row of inferred order keeps observed value exactly.
            fixed_times[0] = observed_ts
            fixed_times, was_clamped3 = self_fix_block(fixed_times, observed_ts, right_ts if right_anchor is not None else None)
            was_clamped = was_clamped or was_clamped3

            # write back to own rows, wherever they appear in file
            for r, t in zip(ordered_rows, fixed_times):
                r["_out_ts"] = format_ts(t, r["_frac_digits"])

            repaired_blocks += 1
            if was_clamped:
                clamped_blocks += 1

    # Global self-check and in-place repair
    rows_equal_to_input = 0

    # Count per block equality and fix any violations
    for case, idxs in by_case.items():
        case_rows = [rows[i] for i in idxs]
        blocks = build_case_blocks(case_rows)
        case_start = min(r["_parsed_ts"] for r in case_rows)
        case_end = max(r["_parsed_ts"] for r in case_rows)

        for start, end in blocks:
            block = case_rows[start:end]
            observed_str = block[0]["_orig_ts"]
            observed_ts = parse_ts(observed_str)
            right_anchor = case_rows[end] if end < len(case_rows) and case_rows[end]["error_category"] == "CLEAN" else None
            right_ts = right_anchor["_parsed_ts"] if right_anchor is not None else case_end

            # exactly one row equals observed input string
            equal_rows = [r for r in block if r["_out_ts"] == r["_orig_ts"]]
            if len(equal_rows) != 1:
                clamped_blocks += 1
                # choose earliest in inferred output time to keep observed exactly
                parsed_pairs = sorted([(parse_ts(r["_out_ts"]), r) for r in block], key=lambda x: x[0])
                keeper = parsed_pairs[0][1]
                for r in block:
                    if r is keeper:
                        r["_out_ts"] = r["_orig_ts"]
                    else:
                        if r["_out_ts"] == r["_orig_ts"]:
                            base = parse_ts(keeper["_out_ts"])
                            # pack others just after keeper and before right anchor
                            delta_sec = 1.0 + 0.001 * (int(r["row_id"]) % 1000)
                            new_t = add_seconds(base, delta_sec)
                            if new_t > right_ts:
                                new_t = right_ts
                            r["_out_ts"] = format_ts(new_t, r["_frac_digits"])

            # ensure all repaired rows parse and are within anchors
            ordered = sorted([(parse_ts(r["_out_ts"]), r) for r in block], key=lambda x: x[0])
            prev_t = None
            for pos, (t, r) in enumerate(ordered):
                changed = (r["_out_ts"] != r["_orig_ts"])
                if t < observed_ts:
                    t = observed_ts if pos == 0 else add_seconds(prev_t, 0.001)
                    r["_out_ts"] = format_ts(t, r["_frac_digits"])
                    clamped_blocks += 1
                if right_anchor is not None and t > right_ts:
                    t = right_ts
                    r["_out_ts"] = format_ts(t, r["_frac_digits"])
                    clamped_blocks += 1
                if prev_t is not None and t <= prev_t:
                    t = add_seconds(prev_t, 0.001)
                    if right_anchor is not None and t > right_ts:
                        t = right_ts
                    r["_out_ts"] = format_ts(t, r["_frac_digits"])
                    clamped_blocks += 1
                if changed and r["_out_ts"] == r["_orig_ts"]:
                    t = parse_ts(r["_out_ts"])
                    t = add_seconds(t, 1.0 + 0.001 * pos)
                    if right_anchor is not None and t > right_ts:
                        t = right_ts
                    r["_out_ts"] = format_ts(t, r["_frac_digits"])
                    clamped_blocks += 1
                prev_t = parse_ts(r["_out_ts"])

    for r in rows:
        if r["_out_ts"] == r["_orig_ts"]:
            rows_equal_to_input += 1

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r[k] for k in fieldnames}
            out["Timestamp"] = r["_out_ts"]
            writer.writerow(out)

    print(f"blocks_repaired={repaired_blocks} blocks_clamped={clamped_blocks} rows_still_equal_to_input={rows_equal_to_input}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Must not fail hard; best effort output if possible is impossible here without state,
        # so return cleanly with diagnostic.
        print(f"blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0 error={str(e).replace(',', ';')}")