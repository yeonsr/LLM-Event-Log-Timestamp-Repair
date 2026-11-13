import argparse
import csv
import itertools
import math
from collections import Counter, defaultdict

import pandas as pd


# -------------------------
# Embedded process context
# -------------------------

TRANSITION_PROBS_TOP = [
    ("Deliver card", "EVENT 13 END", 1.0),
    ("EVENT 13 END", "EVENT 13 END", 1.0),
    ("Notify accept", "Deliver card", 1.0),
    ("time out", "EVENT 13 END", 1.0),
    ("Perform checks", "Make decision", 0.9919),
    ("review request received", "Check for completeness", 0.9708),
    ("info received", "Check for completeness", 0.9662),
    ("Request info", "info received", 0.962),
    ("New online application received", "Check for completeness", 0.91),
    ("notify reject", "time out", 0.5226),
    ("Check for completeness", "Request info", 0.5124),
    ("Make decision", "notify reject", 0.5079),
    ("Make decision", "Notify accept", 0.4883),
    ("notify reject", "review request received", 0.465),
    ("Check for completeness", "Perform checks", 0.4649),
    ("New online application received", "Notify accept", 0.0276),
    ("New online application received", "Make decision", 0.0234),
    ("New online application received", "notify reject", 0.0164),
    ("info received", "Make decision", 0.0106),
    ("New online application received", "info received", 0.0106),
    ("Check for completeness", "Notify accept", 0.0085),
    ("Request info", "Perform checks", 0.0084),
    ("info received", "info received", 0.0084),
    ("Request info", "Make decision", 0.0079),
    ("review request received", "Make decision", 0.0076),
    ("Request info", "Request info", 0.0068),
    ("New online application received", "Perform checks", 0.0066),
    ("Check for completeness", "Make decision", 0.0059),
    ("review request received", "Notify accept", 0.0057),
    ("review request received", "Perform checks", 0.0057),
    ("Request info", "Check for completeness", 0.0054),
    ("Check for completeness", "notify reject", 0.0049),
    ("Request info", "Notify accept", 0.0047),
    ("Request info", "notify reject", 0.0047),
    ("info received", "notify reject", 0.0047),
    ("info received", "Perform checks", 0.0043),
    ("info received", "Notify accept", 0.0039),
    ("notify reject", "Make decision", 0.0036),
    ("New online application received", "Request info", 0.0034),
    ("review request received", "Request info", 0.0032),
    ("review request received", "notify reject", 0.0032),
    ("notify reject", "Perform checks", 0.0027),
    ("Perform checks", "Check for completeness", 0.0026),
    ("Perform checks", "review request received", 0.0026),
    ("review request received", "info received", 0.0025),
    ("New online application received", "review request received", 0.002),
    ("Perform checks", "Request info", 0.0019),
    ("notify reject", "Check for completeness", 0.0018),
    ("Check for completeness", "info received", 0.0016),
    ("notify reject", "notify reject", 0.0015),
    ("info received", "Request info", 0.0014),
    ("review request received", "review request received", 0.0013),
    ("Make decision", "info received", 0.0012),
    ("Check for completeness", "Check for completeness", 0.0011),
    ("Perform checks", "Perform checks", 0.001),
    ("Make decision", "Make decision", 0.001),
    ("notify reject", "Notify accept", 0.0009),
    ("notify reject", "Request info", 0.0009),
    ("notify reject", "info received", 0.0009),
    ("Make decision", "Perform checks", 0.0008),
]

DURATION_PROFILES_TOP = [
    ("Check for completeness", "Request info", 6227, 61.9314, 170.1115, 418.728, 830.2675, 1373.2918, 0.097),
    ("Request info", "info received", 6080, 10.0, 10.0, 10.0, 10.0, 10.0, 0.996),
    ("info received", "Check for completeness", 6008, 63.7390, 176.05, 420.682, 839.701, 1404.5438, 0.095),
    ("Perform checks", "Make decision", 5745, 62.8388, 171.483, 422.924, 835.934, 1392.8122, 0.095),
    ("Check for completeness", "Perform checks", 5650, 61.0229, 171.5965, 420.3715, 832.8305, 1416.3387, 0.098),
    ("New online application received", "Check for completeness", 4550, 65.3893, 171.7955, 426.0135, 834.95025, 1385.6882, 0.092),
    ("Deliver card", "EVENT 13 END", 3276, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ("Notify accept", "Deliver card", 3276, 59.4585, 176.35075, 425.762, 851.3035, 1370.5795, 0.101),
    ("Make decision", "notify reject", 3088, 63.7974, 174.02075, 426.0405, 834.3485, 1427.2486, 0.095),
    ("Make decision", "Notify accept", 2969, 65.5868, 174.826, 428.03, 867.335, 1436.6726, 0.089),
    ("time out", "EVENT 13 END", 1724, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ("notify reject", "time out", 1724, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0),
    ("notify reject", "review request received", 1534, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0),
    ("review request received", "Check for completeness", 1527, 78.0242, 188.8875, 441.348, 841.8565, 1413.9960, 0.075),
    ("New online application received", "Notify accept", 138, 944.0129, 1469.49875, 2020.112, 2906.02375, 3878.7890, 0.0),
    ("New online application received", "Make decision", 117, 634.1, 963.121, 1661.293, 2178.854, 3090.3764, 0.0),
    ("Check for completeness", "Notify accept", 103, 711.8126, 1094.916, 1780.137, 2325.416, 3107.5866, 0.0),
    ("New online application received", "notify reject", 82, 1132.4149, 1538.03275, 2159.5775, 2824.4955, 3498.6097, 0.0),
    ("Check for completeness", "Make decision", 72, 1072.0155, 1536.0985, 2023.867, 2917.63875, 3699.0501, 0.0),
    ("info received", "Make decision", 66, 721.489, 1122.10325, 1868.191, 2613.01725, 3118.5855, 0.0),
    ("Check for completeness", "notify reject", 60, 945.7493, 1361.11625, 1757.751, 2556.73125, 3239.0085, 0.0),
    ("Request info", "Perform checks", 53, 403.0504, 643.92, 928.08, 1395.19, 2110.2466, 0.0),
    ("New online application received", "info received", 53, 452.3138, 639.021, 857.047, 1303.053, 1811.6944, 0.0),
    ("info received", "info received", 52, 613.9782, 895.2945, 1302.5635, 1835.09975, 2340.9239, 0.0),
    ("Request info", "Make decision", 50, 669.1557, 855.869, 1523.8755, 2372.5625, 3437.5409, 0.0),
    ("Request info", "Request info", 43, 569.524, 796.317, 1164.851, 1761.032, 2732.683, 0.0),
    ("Request info", "Check for completeness", 34, 803.7977, 952.73675, 1889.409, 2738.841, 4159.5683, 0.0),
    ("New online application received", "Perform checks", 33, 1091.8104, 1559.19, 2363.648, 3302.949, 3864.007, 0.0),
    ("Request info", "Notify accept", 30, 1429.9083, 1610.589, 2396.886, 3179.76125, 3601.6131, 0.0),
    ("Request info", "notify reject", 30, 1539.1919, 1979.3755, 2454.903, 2997.263, 3804.9144, 0.0),
    ("info received", "notify reject", 29, 1059.5184, 1682.395, 2806.157, 3442.502, 3679.6308, 0.0),
    ("info received", "Perform checks", 27, 809.8986, 1087.906, 1874.355, 2956.759, 4336.3136, 0.0),
    ("info received", "Notify accept", 24, 1006.2042, 1510.50375, 2230.734, 3852.069, 4100.492, 0.0),
    ("Check for completeness", "info received", 19, 817.942, 1530.727, 1910.593, 2581.938, 2788.1602, 0.0),
    ("New online application received", "Request info", 17, 809.1352, 1687.418, 2254.706, 2700.696, 4232.1842, 0.0),
    ("Perform checks", "Check for completeness", 15, 1023.0088, 1420.4865, 1995.413, 2965.38, 3838.7202, 0.0),
    ("Perform checks", "review request received", 15, 378.412, 555.4955, 705.0, 849.476, 1327.28, 0.0),
    ("Check for completeness", "Check for completeness", 13, 1452.7526, 1939.982, 2577.014, 3296.051, 3589.3102, 0.0),
    ("notify reject", "Make decision", 12, 785.8718, 1341.0605, 1518.5795, 1958.461, 2274.8716, 0.0),
    ("review request received", "Make decision", 12, 727.506, 964.28475, 2227.617, 2260.87725, 3134.5752, 0.0),
    ("Perform checks", "Request info", 11, 1288.529, 1794.608, 2625.632, 3426.251, 4208.01, 0.0),
    ("New online application received", "review request received", 10, 783.2576, 1569.0395, 1845.21, 2201.5355, 2416.4854, 0.0),
    ("Check for completeness", "review request received", 9, 786.9502, 1083.927, 1363.124, 2554.657, 2954.3854, 0.0),
    ("review request received", "Notify accept", 9, 2066.8628, 2191.667, 2296.639, 4048.847, 4780.2734, 0.0),
    ("info received", "Request info", 9, 824.7008, 1305.603, 1993.345, 3112.501, 4079.4476, 0.0),
    ("notify reject", "Perform checks", 9, 318.1018, 699.898, 874.95, 1271.62, 1995.78, 0.0),
    ("review request received", "Perform checks", 9, 1338.8826, 1593.822, 2249.376, 3393.616, 4267.802, 0.0),
    ("EVENT 13 END", "EVENT 13 END", 8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ("Make decision", "info received", 7, 615.6284, 944.3135, 1863.824, 1941.4715, 2142.2018, 0.0),
    ("notify reject", "Check for completeness", 6, 1769.177, 2725.29575, 3075.0545, 3917.45225, 4349.0205, 0.0),
    ("Make decision", "Make decision", 6, 1349.042, 1693.34325, 2015.0765, 2417.1235, 2768.227, 0.0),
    ("Perform checks", "Perform checks", 6, 1054.435, 1415.281, 2294.3645, 2990.205, 5170.2335, 0.0),
    ("notify reject", "notify reject", 5, 1475.265, 2749.563, 2967.996, 3278.084, 3452.2106, 0.0),
    ("Make decision", "Perform checks", 5, 783.9354, 796.233, 823.397, 1650.51, 1824.5034, 0.0),
    ("review request received", "Request info", 5, 1863.9608, 2128.058, 2501.573, 3114.888, 4176.5796, 0.0),
    ("review request received", "notify reject", 5, 1334.3154, 1360.086, 1469.711, 2384.126, 3042.9542, 0.0),
    ("Make decision", "Request info", 5, 1027.2532, 1078.57, 1325.858, 1553.253, 2857.497, 0.0),
    ("review request received", "info received", 4, 1391.3396, 1447.0055, 1983.33, 2496.487, 2510.4514, 0.0),
    ("notify reject", "info received", 3, 652.6522, 721.4095, 836.005, 1576.566, 2020.9026, 0.0),
    ("notify reject", "Request info", 3, 389.1752, 613.13, 986.388, 1967.2505, 2555.768, 0.0),
]

FREQUENT_VARIANTS = [
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
]


# -------------------------
# Precomputed maps
# -------------------------

EDGE_FREQ = {}
EDGE_PROB = {}
for s, t, p in TRANSITION_PROBS_TOP:
    EDGE_PROB[(s, t)] = p

DUR = {}
for s, t, n, p10, p25, med, p75, p90, share in DURATION_PROFILES_TOP:
    EDGE_FREQ[(s, t)] = n
    DUR[(s, t)] = {
        "n": n,
        "p10": p10,
        "p25": p25,
        "median": med,
        "p75": p75,
        "p90": p90,
        "share_under_60s": share,
    }

KNOWN_FAST_DEFAULT = 10.0
KNOWN_MINUTE_DEFAULT = 90.0


# -------------------------
# Helpers
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def parse_ts(s):
    return pd.to_datetime(s)


def frac_digits_from_string(ts_str):
    if "." not in ts_str:
        return 0
    return len(ts_str.split(".")[-1])


def format_ts(ts, frac_digits):
    if frac_digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    ns = ts.nanosecond + ts.microsecond * 1000
    frac = f"{ns:09d}"[:frac_digits]
    return f"{base}.{frac}"


def add_seconds(ts, seconds):
    return ts + pd.to_timedelta(seconds, unit="s")


def candidate_gap_seconds(a, b):
    d = DUR.get((a, b))
    if d is not None:
        if d["share_under_60s"] >= 0.8:
            g = d["p10"]
        else:
            g = d["p25"]
        if g <= 0:
            g = 1.0
        return float(g)
    back = DUR.get((b, a))
    if back is not None and back["share_under_60s"] >= 0.8:
        return 10.0
    if (a, b) in EDGE_PROB:
        p = EDGE_PROB[(a, b)]
        if p > 0.9:
            return 10.0
    if a in ("notify reject", "Request info") and b in ("time out", "info received", "review request received"):
        return 10.0
    return KNOWN_MINUTE_DEFAULT


def jittered_positive_gaps(base_gaps):
    out = []
    for i, g in enumerate(base_gaps):
        g = max(float(g), 1.0)
        jig = ((i % 3) - 1) * 0.173
        out.append(max(1.0, g + jig))
    return out


def contiguous_subsequence_support(seq, left_anchor, right_anchor):
    score = 0.0
    n = len(seq)
    for var in FREQUENT_VARIANTS:
        for i in range(len(var) - n + 1):
            if var[i:i + n] == list(seq):
                bonus = 8.0
                if left_anchor is not None and i > 0 and var[i - 1] == left_anchor:
                    bonus += 6.0
                if right_anchor is not None and i + n < len(var) and var[i + n] == right_anchor:
                    bonus += 6.0
                score = max(score, bonus)
    return score


def order_score(order, left_anchor_act, right_anchor_act):
    score = 0.0
    impossible_internal = 0
    impossible_anchors = 0

    edges = []
    if left_anchor_act is not None:
        edges.append((left_anchor_act, order[0], "left"))
    for i in range(len(order) - 1):
        edges.append((order[i], order[i + 1], "internal"))
    if right_anchor_act is not None:
        edges.append((order[-1], right_anchor_act, "right"))

    for s, t, kind in edges:
        if (s, t) in EDGE_FREQ or (s, t) in EDGE_PROB:
            freq = EDGE_FREQ.get((s, t), 1)
            prob = EDGE_PROB.get((s, t), 1e-6)
            score += 4.0 + math.log1p(freq) + math.log(max(prob, 1e-12))
        else:
            if kind == "internal":
                impossible_internal += 1
                score -= 100.0
            else:
                impossible_anchors += 1
                score -= 12.0

    score += contiguous_subsequence_support(order, left_anchor_act, right_anchor_act)
    score -= 0.01 * sum(order[i] > order[i + 1] for i in range(len(order) - 1))
    return score, impossible_internal, impossible_anchors


def infer_order(activities, left_anchor_act, right_anchor_act):
    idxs = list(range(len(activities)))
    perms = []

    if len(idxs) <= 7:
        iterable = itertools.permutations(idxs)
    else:
        # deterministic fallback for rare larger blocks
        counts = Counter(activities)
        grouped = defaultdict(list)
        for i, a in enumerate(activities):
            grouped[a].append(i)
        act_order = sorted(counts.keys())
        base = []
        for a in act_order:
            base.extend(grouped[a])
        iterable = [tuple(base)]

    best = None
    best_score = -10**18

    viable = []
    all_scored = []
    for perm in iterable:
        order = [activities[i] for i in perm]
        s, imp_i, imp_a = order_score(order, left_anchor_act, right_anchor_act)
        rec = (s, imp_i, imp_a, perm, order)
        all_scored.append(rec)
        if imp_i == 0:
            viable.append(rec)

    pool = viable if viable else all_scored
    for rec in pool:
        s, imp_i, imp_a, perm, order = rec
        if s > best_score:
            best_score = s
            best = rec

    return best[3], best[4]


def compress_gaps_to_room(base_gaps, room_seconds):
    n = len(base_gaps) + 1
    min_total = max(n - 1, 0)
    if room_seconds is None:
        return jittered_positive_gaps(base_gaps), False
    room_seconds = max(float(room_seconds), float(min_total))
    gaps = jittered_positive_gaps(base_gaps)
    total = sum(gaps)
    clamped = False
    if total <= room_seconds:
        return gaps, clamped

    clamped = True
    if total <= 0:
        return [1.0] * (n - 1), clamped

    scale = room_seconds / total
    gaps = [max(1.0, g * scale) for g in gaps]
    total = sum(gaps)

    if total > room_seconds:
        excess = total - room_seconds
        reducible = [g - 1.0 for g in gaps]
        reducible_sum = sum(max(r, 0.0) for r in reducible)
        if reducible_sum > 0:
            new_gaps = []
            for g, r in zip(gaps, reducible):
                cut = excess * (max(r, 0.0) / reducible_sum)
                new_gaps.append(max(1.0, g - cut))
            gaps = new_gaps

    total = sum(gaps)
    if total > room_seconds + 1e-9:
        # final deterministic trimming from the end
        for i in range(len(gaps) - 1, -1, -1):
            if total <= room_seconds + 1e-9:
                break
            can_cut = max(0.0, gaps[i] - 1.0)
            cut = min(can_cut, total - room_seconds)
            gaps[i] -= cut
            total -= cut
    return gaps, clamped


def build_forward_times(observed_ts, order_acts, right_anchor_ts, case_end_ts):
    base_gaps = [candidate_gap_seconds(order_acts[i], order_acts[i + 1]) for i in range(len(order_acts) - 1)]
    upper = None
    if right_anchor_ts is not None:
        upper = (right_anchor_ts - observed_ts).total_seconds()
    elif case_end_ts is not None:
        upper = (case_end_ts - observed_ts).total_seconds()
    gaps, clamped = compress_gaps_to_room(base_gaps, upper)
    times = [observed_ts]
    for g in gaps:
        times.append(add_seconds(times[-1], g))
    return times, clamped


def build_backward_times(observed_ts, order_acts, left_anchor_ts, case_start_ts):
    # observed is block end; return times in execution order
    base_gaps = [candidate_gap_seconds(order_acts[i], order_acts[i + 1]) for i in range(len(order_acts) - 1)]
    upper = None
    if left_anchor_ts is not None:
        upper = (observed_ts - left_anchor_ts).total_seconds()
    elif case_start_ts is not None:
        upper = (observed_ts - case_start_ts).total_seconds()
    gaps, clamped = compress_gaps_to_room(base_gaps, upper)
    times = [observed_ts]
    for g in reversed(gaps):
        times.append(add_seconds(times[-1], -g))
    times = list(reversed(times))
    return times, clamped


def choose_orientation(observed_ts, order_acts, left_anchor_act, right_anchor_act, left_anchor_ts, right_anchor_ts, case_start_ts, case_end_ts):
    # default: start_at_observed
    f_times, f_clamped = build_forward_times(observed_ts, order_acts, right_anchor_ts, case_end_ts)
    f_ok = True
    if right_anchor_ts is not None and f_times[-1] > right_anchor_ts:
        f_ok = False
    if left_anchor_ts is not None and f_times[0] < left_anchor_ts:
        f_ok = False
    if case_end_ts is not None and f_times[-1] > case_end_ts:
        f_ok = False
    if case_start_ts is not None and f_times[0] < case_start_ts:
        f_ok = False

    b_times, b_clamped = build_backward_times(observed_ts, order_acts, left_anchor_ts, case_start_ts)
    b_ok = True
    if left_anchor_ts is not None and b_times[0] < left_anchor_ts:
        b_ok = False
    if right_anchor_ts is not None and b_times[-1] > right_anchor_ts:
        b_ok = False
    if case_end_ts is not None and b_times[-1] > case_end_ts:
        b_ok = False
    if case_start_ts is not None and b_times[0] < case_start_ts:
        b_ok = False

    if f_ok:
        return "forward", f_times, f_clamped
    if b_ok:
        return "backward", b_times, True or b_clamped
    return "forward", f_times, True


def make_times_strict(times, left_bound=None, right_bound=None):
    times = list(times)
    n = len(times)
    if n <= 1:
        return times, False
    changed = False

    # forward monotonic pass
    for i in range(1, n):
        min_allowed = add_seconds(times[i - 1], 1.0)
        if times[i] <= times[i - 1]:
            times[i] = min_allowed
            changed = True

    # clamp to right bound if needed while keeping strict increase
    if right_bound is not None and times[-1] > right_bound:
        changed = True
        latest = right_bound
        times[-1] = latest
        for i in range(n - 2, -1, -1):
            latest = add_seconds(times[i + 1], -1.0)
            if times[i] >= times[i + 1]:
                times[i] = latest
            elif times[i] > latest:
                times[i] = latest

    # clamp to left bound if needed while keeping strict increase
    if left_bound is not None and times[0] < left_bound:
        changed = True
        times[0] = left_bound
        for i in range(1, n):
            min_allowed = add_seconds(times[i - 1], 1.0)
            if times[i] <= times[i - 1]:
                times[i] = min_allowed

    # second right check
    if right_bound is not None and times[-1] > right_bound:
        changed = True
        room = (right_bound - left_bound).total_seconds() if left_bound is not None else None
        if room is not None and room >= n - 1:
            times = [left_bound]
            step = room / (n - 1)
            # non-uniform deterministic
            current = left_bound
            for i in range(1, n):
                inc = step + (0.217 if i % 2 else -0.113)
                inc = max(1.0, inc)
                current = add_seconds(current, inc)
                times.append(current)
            if times[-1] > right_bound:
                delta = (times[-1] - right_bound).total_seconds()
                times = [add_seconds(t, -delta) for t in times]
        else:
            times[-1] = right_bound
            for i in range(n - 2, -1, -1):
                times[i] = add_seconds(times[i + 1], -1.0)

    return times, changed


def map_times_back(rows_idx, perm, ordered_times):
    row_time = {}
    for pos, row_local_idx in enumerate(perm):
        row_time[rows_idx[row_local_idx]] = ordered_times[pos]
    return row_time


def find_blocks(df):
    blocks = []
    for case, g in df.groupby("Case", sort=False):
        idxs = list(g.index)
        i = 0
        while i < len(idxs):
            gi = idxs[i]
            if df.at[gi, "error_category"] != "C1_ORDERING_AMBIGUITY":
                i += 1
                continue
            ts = df.at[gi, "Timestamp"]
            j = i + 1
            while j < len(idxs):
                gj = idxs[j]
                if df.at[gj, "error_category"] == "C1_ORDERING_AMBIGUITY" and df.at[gj, "Timestamp"] == ts:
                    j += 1
                else:
                    break
            if j - i >= 2:
                blocks.append((case, idxs[i:j]))
            else:
                blocks.append((case, idxs[i:j]))
            i = j
    return blocks


def nearest_clean_anchor(df, case_indices, block_indices):
    pos_map = {idx: p for p, idx in enumerate(case_indices)}
    start_p = pos_map[block_indices[0]]
    end_p = pos_map[block_indices[-1]]

    left_idx = None
    for p in range(start_p - 1, -1, -1):
        idx = case_indices[p]
        if df.at[idx, "error_category"] == "CLEAN":
            left_idx = idx
            break

    right_idx = None
    for p in range(end_p + 1, len(case_indices)):
        idx = case_indices[p]
        if df.at[idx, "error_category"] == "CLEAN":
            right_idx = idx
            break

    return left_idx, right_idx


def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    df = df[original_cols].copy()

    original_ts = df["Timestamp"].tolist()
    out_ts = df["Timestamp"].tolist()

    parsed_ts = [parse_ts(x) for x in df["Timestamp"]]
    frac_digits = [frac_digits_from_string(x) for x in df["Timestamp"]]

    case_indices_map = {case: list(g.index) for case, g in df.groupby("Case", sort=False)}
    case_bounds = {}
    for case, idxs in case_indices_map.items():
        tss = [parsed_ts[i] for i in idxs]
        case_bounds[case] = (min(tss), max(tss))

    blocks = find_blocks(df)

    repaired_blocks = 0
    clamped_blocks = 0

    for case, block_idxs in blocks:
        # Only repair non-clean rows; all these are C1 by construction.
        activities = [df.at[idx, "Activity"] for idx in block_idxs]
        observed_str = df.at[block_idxs[0], "Timestamp"]
        observed_ts = parse_ts(observed_str)
        case_start_ts, case_end_ts = case_bounds[case]
        case_idxs = case_indices_map[case]
        left_idx, right_idx = nearest_clean_anchor(df, case_idxs, block_idxs)

        left_anchor_act = df.at[left_idx, "Activity"] if left_idx is not None else None
        right_anchor_act = df.at[right_idx, "Activity"] if right_idx is not None else None
        left_anchor_ts = parsed_ts[left_idx] if left_idx is not None else None
        right_anchor_ts = parsed_ts[right_idx] if right_idx is not None else None

        perm, ordered_acts = infer_order(activities, left_anchor_act, right_anchor_act)

        orientation, ordered_times, was_clamped = choose_orientation(
            observed_ts, ordered_acts,
            left_anchor_act, right_anchor_act,
            left_anchor_ts, right_anchor_ts,
            case_start_ts, case_end_ts
        )

        if orientation == "forward":
            left_bound = max(x for x in [observed_ts, left_anchor_ts, case_start_ts] if x is not None)
            right_bound = min(x for x in [right_anchor_ts, case_end_ts] if x is not None) if (right_anchor_ts is not None or case_end_ts is not None) else None
            # earliest must keep observed exactly
            ordered_times[0] = observed_ts if observed_ts >= left_bound else left_bound
            ordered_times, changed = make_times_strict(ordered_times, left_bound=ordered_times[0], right_bound=right_bound)
            was_clamped = was_clamped or changed
            if ordered_times[0] != observed_ts:
                # preserve the requirement whenever possible by shifting forward from observed
                ordered_times[0] = observed_ts
                ordered_times, changed2 = make_times_strict(ordered_times, left_bound=observed_ts, right_bound=right_bound)
                was_clamped = was_clamped or changed2
        else:
            left_bound = max(x for x in [left_anchor_ts, case_start_ts] if x is not None) if (left_anchor_ts is not None or case_start_ts is not None) else None
            right_bound = min(x for x in [observed_ts, right_anchor_ts, case_end_ts] if x is not None)
            ordered_times[-1] = observed_ts
            ordered_times, changed = make_times_strict(ordered_times, left_bound=left_bound, right_bound=ordered_times[-1])
            was_clamped = True or was_clamped or changed
            # If backward would violate "first event keeps observed" preference, keep it only when forward impossible.
            # This branch is already only chosen when forward infeasible.

        # Final guaranteed feasibility and positivity
        if right_anchor_ts is not None and ordered_times[-1] > right_anchor_ts:
            room = max((right_anchor_ts - observed_ts).total_seconds(), len(block_idxs) - 1)
            tiny = [1.0 + (0.137 if i % 2 else 0.0) for i in range(len(block_idxs) - 1)]
            tiny, _ = compress_gaps_to_room(tiny, room)
            ordered_times = [observed_ts]
            for g in tiny:
                ordered_times.append(add_seconds(ordered_times[-1], g))
            was_clamped = True

        # Ensure exact one observed timestamp in inferred order:
        # choose earliest event in ordered execution to keep observed;
        # if duplicate emerges by formatting/rounding, spread later rows.
        for i in range(1, len(ordered_times)):
            if ordered_times[i] <= ordered_times[i - 1]:
                ordered_times[i] = add_seconds(ordered_times[i - 1], 1.0)
                was_clamped = True

        row_time_map = map_times_back(block_idxs, perm, ordered_times)

        # write back preserving each repaired row precision
        observed_holder_global_idx = block_idxs[perm[0]]
        for idx in block_idxs:
            if idx == observed_holder_global_idx:
                out_ts[idx] = original_ts[idx]
            else:
                out_ts[idx] = format_ts(row_time_map[idx], frac_digits[idx])

        repaired_blocks += 1
        if was_clamped:
            clamped_blocks += 1

    # -------------------------
    # Self-check and corrective pass
    # -------------------------

    rows_equal_to_input = 0

    for case, block_idxs in blocks:
        case_idxs = case_indices_map[case]
        left_idx, right_idx = nearest_clean_anchor(df, case_idxs, block_idxs)
        left_anchor_ts = parse_ts(df.at[left_idx, "Timestamp"]) if left_idx is not None else None
        right_anchor_ts = parse_ts(df.at[right_idx, "Timestamp"]) if right_idx is not None else None
        case_start_ts, case_end_ts = case_bounds[case]
        observed_ts = parse_ts(df.at[block_idxs[0], "Timestamp"])

        # recover order again
        activities = [df.at[idx, "Activity"] for idx in block_idxs]
        left_anchor_act = df.at[left_idx, "Activity"] if left_idx is not None else None
        right_anchor_act = df.at[right_idx, "Activity"] if right_idx is not None else None
        perm, ordered_acts = infer_order(activities, left_anchor_act, right_anchor_act)
        ordered_global_idxs = [block_idxs[i] for i in perm]

        current_times = [parse_ts(out_ts[idx]) for idx in ordered_global_idxs]

        # earliest keeps observed exactly; others must differ
        if out_ts[ordered_global_idxs[0]] != original_ts[ordered_global_idxs[0]]:
            out_ts[ordered_global_idxs[0]] = original_ts[ordered_global_idxs[0]]
            current_times[0] = observed_ts
            clamped_blocks += 1

        # strict increase
        max_right = min(x for x in [right_anchor_ts, case_end_ts] if x is not None) if (right_anchor_ts is not None or case_end_ts is not None) else None
        current_times, changed = make_times_strict(current_times, left_bound=observed_ts, right_bound=max_right)
        if changed:
            clamped_blocks += 1

        # enforce span <= room to right anchor and >= k-1 sec
        k = len(block_idxs)
        min_span = k - 1
        span = (current_times[-1] - current_times[0]).total_seconds()
        room = None
        if right_anchor_ts is not None:
            room = (right_anchor_ts - observed_ts).total_seconds()
        elif case_end_ts is not None:
            room = (case_end_ts - observed_ts).total_seconds()

        if room is not None:
            room = max(room, float(min_span))
            if span > room + 1e-9 or span < min_span - 1e-9:
                base = [candidate_gap_seconds(ordered_acts[i], ordered_acts[i + 1]) for i in range(k - 1)]
                gaps, _ = compress_gaps_to_room(base, room)
                if sum(gaps) < min_span:
                    gaps = [1.0 + (0.071 if i % 2 else 0.0) for i in range(k - 1)]
                current_times = [observed_ts]
                for g in gaps:
                    current_times.append(add_seconds(current_times[-1], g))
                current_times, _ = make_times_strict(current_times, left_bound=observed_ts, right_bound=max_right)
                clamped_blocks += 1

        # bounds
        for i, t in enumerate(current_times):
            if t < case_start_ts:
                current_times[i] = case_start_ts if i == 0 else add_seconds(current_times[i - 1], 1.0)
                clamped_blocks += 1
            if t > case_end_ts:
                # shift backward deterministically
                delta = (t - case_end_ts).total_seconds()
                current_times = [add_seconds(x, -delta) for x in current_times]
                clamped_blocks += 1
                break
        if left_anchor_ts is not None and current_times[0] < left_anchor_ts:
            shift = (left_anchor_ts - current_times[0]).total_seconds()
            current_times = [add_seconds(x, shift) for x in current_times]
            clamped_blocks += 1
        if right_anchor_ts is not None and current_times[-1] > right_anchor_ts:
            delta = (current_times[-1] - right_anchor_ts).total_seconds()
            current_times = [add_seconds(x, -delta) for x in current_times]
            current_times, _ = make_times_strict(current_times, left_bound=observed_ts, right_bound=right_anchor_ts)
            clamped_blocks += 1

        # write back again
        out_ts[ordered_global_idxs[0]] = original_ts[ordered_global_idxs[0]]
        for pos, idx in enumerate(ordered_global_idxs[1:], start=1):
            if format_ts(current_times[pos], frac_digits[idx]) == original_ts[idx]:
                bumped = add_seconds(current_times[pos], 1.0 + (pos * 0.001))
                if right_anchor_ts is not None and bumped > right_anchor_ts:
                    bumped = add_seconds(current_times[pos - 1], 1.0)
                current_times[pos] = bumped
            out_ts[idx] = format_ts(current_times[pos], frac_digits[idx])

    # Clean rows must remain exact
    for i in range(len(df)):
        if df.at[i, "error_category"] == "CLEAN":
            out_ts[i] = original_ts[i]

    # Count rows still equal to input
    for i in range(len(df)):
        if out_ts[i] == original_ts[i]:
            rows_equal_to_input += 1

    # Final dataframe
    out_df = df.copy()
    out_df["Timestamp"] = out_ts

    # Preserve row count and non-timestamp columns byte-identical by writing as-is
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(original_cols)
        for _, row in out_df.iterrows():
            writer.writerow([row[c] for c in original_cols])

    print(f"blocks_repaired={repaired_blocks} blocks_clamped={clamped_blocks} rows_still_equal_to_input={rows_equal_to_input}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # As required: do not exit non-zero. Try to emit a pass-through copy if possible.
        try:
            import sys
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=True)
            parser.add_argument("--output", required=True)
            ns, _ = parser.parse_known_args()
            df = pd.read_csv(ns.input, dtype=str, keep_default_na=False)
            df.to_csv(ns.output, index=False)
            print(f"blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input={len(df)}")
        except Exception:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0")