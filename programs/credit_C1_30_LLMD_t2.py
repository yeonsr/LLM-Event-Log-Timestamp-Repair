import argparse
import math
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations

import pandas as pd


TRANSITION_PROBABILITIES_TOP = [
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
    ("review request received", "Check for completeness", 1527, 78.0242, 188.8875, 441.348, 841.8565, 1413.996, 0.075),
    ("New online application received", "Notify accept", 138, 944.0129, 1469.49875, 2020.112, 2906.02375, 3878.789, 0.0),
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
    ("notify reject", "Perform checks", 9, 318.1018, 699.898, 874.95, 1271.62, 1995.78, 0.0),
    ("review request received", "Notify accept", 9, 2066.8628, 2191.667, 2296.639, 4048.847, 4780.2734, 0.0),
    ("Check for completeness", "review request received", 9, 786.9502, 1083.927, 1363.124, 2554.657, 2954.3854, 0.0),
    ("review request received", "Perform checks", 9, 1338.8826, 1593.822, 2249.376, 3393.616, 4267.802, 0.0),
    ("info received", "Request info", 9, 824.7008, 1305.603, 1993.345, 3112.501, 4079.4476, 0.0),
    ("EVENT 13 END", "EVENT 13 END", 8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ("Make decision", "info received", 7, 615.6284, 944.3135, 1863.824, 1941.4715, 2142.2018, 0.0),
    ("notify reject", "Check for completeness", 6, 1769.177, 2725.29575, 3075.0545, 3917.45225, 4349.0205, 0.0),
    ("Make decision", "Make decision", 6, 1349.042, 1693.34325, 2015.0765, 2417.1235, 2768.227, 0.0),
    ("Perform checks", "Perform checks", 6, 1054.435, 1415.281, 2294.3645, 2990.205, 5170.2335, 0.0),
    ("review request received", "notify reject", 5, 1334.3154, 1360.086, 1469.711, 2384.126, 3042.9542, 0.0),
    ("review request received", "Request info", 5, 1863.9608, 2128.058, 2501.573, 3114.888, 4176.5796, 0.0),
    ("Make decision", "Request info", 5, 1027.2532, 1078.57, 1325.858, 1553.253, 2857.497, 0.0),
    ("notify reject", "notify reject", 5, 1475.265, 2749.563, 2967.996, 3278.084, 3452.2106, 0.0),
    ("Make decision", "Perform checks", 5, 783.9354, 796.233, 823.397, 1650.51, 1824.5034, 0.0),
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

GLOBAL_MEDIAN_DURATION_SECONDS = 243.721


def build_maps():
    prob = {}
    for s, t, p in TRANSITION_PROBABILITIES_TOP:
        prob[(s, t)] = p

    dur = {}
    freq = {}
    for s, t, n, p10, p25, median, p75, p90, share in DURATION_PROFILES_TOP:
        dur[(s, t)] = {
            "n": n,
            "p10": p10,
            "p25": p25,
            "median": median,
            "p75": p75,
            "p90": p90,
            "share_under_60s": share,
        }
        freq[(s, t)] = n
    return prob, dur, freq


TRANS_PROB, DUR_PROF, EDGE_FREQ = build_maps()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def frac_digits(ts):
    if "." in ts:
        return len(ts.split(".")[-1])
    return 0


def quantum_seconds(max_digits):
    if max_digits <= 0:
        return Decimal("1")
    return Decimal(1) / (Decimal(10) ** max_digits)


def format_dt(ts, digits):
    if digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    nanos = ts.value % 1_000_000_000
    frac = f"{nanos:09d}"[:digits]
    return f"{base}.{frac}"


def to_timestamp(s):
    return pd.to_datetime(s)


def to_dec_seconds(td):
    return Decimal(str(td.total_seconds()))


def pdtd_from_decimal_seconds(sec_dec):
    ns = int((sec_dec * Decimal("1000000000")).to_integral_value(rounding=ROUND_HALF_UP))
    return pd.to_timedelta(ns, unit="ns")


def edge_exists(a, b):
    return (a, b) in EDGE_FREQ or (a, b) in TRANS_PROB


def edge_score(a, b):
    prob = TRANS_PROB.get((a, b), 0.0)
    freq = EDGE_FREQ.get((a, b), 0)
    if prob == 0.0 and freq == 0:
        return None
    return math.log(max(prob, 1e-12)) + 0.5 * math.log(freq + 1.0)


def variant_subsequence_score(seq):
    best = 0
    for var in FREQUENT_VARIANTS:
        if len(seq) > len(var):
            continue
        for i in range(len(var) - len(seq) + 1):
            if var[i:i + len(seq)] == list(seq):
                best = max(best, 1)
                return best
    return best


def duration_gap_seconds(a, b):
    prof = DUR_PROF.get((a, b))
    if prof:
        if prof["share_under_60s"] >= 0.8:
            g = prof["p10"]
        else:
            g = prof["p25"]
        if g <= 0:
            g = 0.001
        return g
    if edge_exists(a, b):
        return 10.0
    return 1.0


def multiset_permutations(items):
    ctr = Counter(items)
    elems = sorted(ctr.keys())
    n = len(items)
    out = []

    def rec(path):
        if len(path) == n:
            out.append(tuple(path))
            return
        for e in elems:
            if ctr[e] > 0:
                ctr[e] -= 1
                path.append(e)
                rec(path)
                path.pop()
                ctr[e] += 1

    rec([])
    return out


def infer_order(activities, left_act, right_act):
    candidates = multiset_permutations(activities)
    scored = []
    feasible = []
    for perm in candidates:
        pairs = []
        if left_act is not None:
            pairs.append((left_act, perm[0]))
        for i in range(len(perm) - 1):
            pairs.append((perm[i], perm[i + 1]))
        if right_act is not None:
            pairs.append((perm[-1], right_act))

        unsupported = sum(0 if edge_exists(a, b) else 1 for a, b in pairs)
        total = 0.0
        ok_edges = 0
        for a, b in pairs:
            es = edge_score(a, b)
            if es is not None:
                total += es
                ok_edges += 1

        seq_anchor = tuple(([left_act] if left_act is not None else []) + list(perm) + ([right_act] if right_act is not None else []))
        seq_block = tuple(perm)
        variant_bonus = 0
        if variant_subsequence_score(seq_anchor):
            variant_bonus += 50.0
        if variant_subsequence_score(seq_block):
            variant_bonus += 20.0

        internal_supported = sum(1 for i in range(len(perm) - 1) if edge_exists(perm[i], perm[i + 1]))
        score = total + variant_bonus + internal_supported * 2.0 - unsupported * 100.0
        item = (unsupported, -variant_bonus, -internal_supported, -ok_edges, -score, perm)
        scored.append((score, unsupported, variant_bonus, internal_supported, ok_edges, perm, item))
        if unsupported == 0:
            feasible.append((score, variant_bonus, internal_supported, ok_edges, perm))

    if feasible:
        feasible.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        return list(feasible[0][4])

    scored.sort(key=lambda x: (x[1], -x[2], -x[3], -x[4], x[0]))
    return list(scored[0][5])


def assign_rows_for_order(block_rows, ordered_activities):
    by_act = defaultdict(list)
    for idx, act in block_rows:
        by_act[act].append(idx)
    for act in by_act:
        by_act[act] = sorted(by_act[act])

    chosen = []
    used = set()
    for act in ordered_activities:
        avail = [i for i in by_act[act] if i not in used]
        idx = avail[0]
        used.add(idx)
        chosen.append(idx)
    return chosen


def create_times(start_ts, ordered_activities, left_ts, right_ts, max_frac_digits):
    k = len(ordered_activities)
    q = quantum_seconds(max_frac_digits)
    min_step = max(q, Decimal("0.001"))
    if max_frac_digits == 0:
        min_step = Decimal("1")
    min_total = min_step * Decimal(k - 1)

    preferred = []
    for i in range(k - 1):
        g = Decimal(str(duration_gap_seconds(ordered_activities[i], ordered_activities[i + 1])))
        if g < min_step:
            g = min_step
        preferred.append(g)

    start = start_ts
    room_right = None
    if right_ts is not None:
        room_right = to_dec_seconds(right_ts - start)
        if room_right < Decimal("0"):
            room_right = Decimal("0")

    gaps = preferred[:]
    clamped = False

    if room_right is not None:
        if room_right < min_total:
            gaps = [min_step] * (k - 1)
            clamped = True
        else:
            pref_total = sum(gaps, Decimal("0"))
            if pref_total > room_right:
                scale = room_right / pref_total if pref_total > 0 else Decimal("1")
                scaled = []
                for g in gaps:
                    ng = g * scale
                    if ng < min_step:
                        ng = min_step
                    scaled.append(ng)
                total_scaled = sum(scaled, Decimal("0"))
                if total_scaled > room_right:
                    excess = total_scaled - room_right
                    shrinkable = [max(Decimal("0"), g - min_step) for g in scaled]
                    shrink_total = sum(shrinkable, Decimal("0"))
                    if shrink_total > 0:
                        adjusted = []
                        for g, s in zip(scaled, shrinkable):
                            red = excess * (s / shrink_total)
                            ng = g - red
                            if ng < min_step:
                                ng = min_step
                            adjusted.append(ng)
                        scaled = adjusted
                if sum(scaled, Decimal("0")) > room_right:
                    scaled = [min_step] * (k - 1)
                gaps = scaled
                clamped = True

    times = [start]
    cur = start
    for g in gaps:
        cur = cur + pdtd_from_decimal_seconds(g)
        times.append(cur)

    if right_ts is not None and times[-1] > right_ts:
        clamped = True
        end_allowed = right_ts
        if k == 1:
            times = [start]
        else:
            gaps = [min_step] * (k - 1)
            cur = start
            times = [start]
            for g in gaps:
                cur = cur + pdtd_from_decimal_seconds(g)
                times.append(cur)
            if times[-1] > end_allowed:
                back = times[-1] - end_allowed
                times = [t - back for t in times]
                if left_ts is not None and times[0] < left_ts:
                    shift = left_ts - times[0]
                    times = [t + shift for t in times]

    if left_ts is not None:
        for i in range(len(times)):
            if times[i] < left_ts:
                times[i] = left_ts
                clamped = True

    if right_ts is not None:
        for i in range(len(times)):
            if times[i] > right_ts:
                times[i] = right_ts
                clamped = True

    repaired = enforce_strict(times, left_ts, right_ts, max_frac_digits)
    if repaired != times:
        clamped = True
    return repaired, clamped


def enforce_strict(times, left_ts, right_ts, max_frac_digits):
    q = quantum_seconds(max_frac_digits)
    min_step = max(q, Decimal("0.001"))
    if max_frac_digits == 0:
        min_step = Decimal("1")

    out = list(times)
    for i in range(1, len(out)):
        need = out[i - 1] + pdtd_from_decimal_seconds(min_step)
        if out[i] <= out[i - 1]:
            out[i] = need

    if right_ts is not None and len(out) >= 2 and out[-1] > right_ts:
        total_need = pdtd_from_decimal_seconds(min_step * Decimal(len(out) - 1))
        start = right_ts - total_need
        if left_ts is not None and start < left_ts:
            start = left_ts
        out = [start]
        cur = start
        for _ in range(1, len(times)):
            cur = cur + pdtd_from_decimal_seconds(min_step)
            out.append(cur)
        if right_ts is not None and out[-1] > right_ts:
            end = right_ts
            out[-1] = end
            for i in range(len(out) - 2, -1, -1):
                prev = out[i + 1] - pdtd_from_decimal_seconds(min_step)
                out[i] = prev
            if left_ts is not None and out[0] < left_ts:
                base = left_ts
                out = [base]
                cur = base
                for _ in range(1, len(times)):
                    cur = cur + pdtd_from_decimal_seconds(min_step)
                    out.append(cur)
    return out


def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    orig_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    df = df[orig_cols].copy()
    df["_orig_ts"] = df["Timestamp"]
    df["_parsed_ts"] = pd.to_datetime(df["Timestamp"])
    df["_digits"] = df["Timestamp"].map(frac_digits)
    df["_out_ts"] = df["_orig_ts"]

    repaired_blocks = 0
    clamped_blocks = 0

    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        case_df = df.loc[idxs]
        case_start = case_df["_parsed_ts"].min()
        case_end = case_df["_parsed_ts"].max()

        i = 0
        while i < len(idxs):
            idx = idxs[i]
            if df.at[idx, "error_category"] == "CLEAN":
                i += 1
                continue

            j = i + 1
            shared = df.at[idx, "Timestamp"]
            while (
                j < len(idxs)
                and df.at[idxs[j], "error_category"] != "CLEAN"
                and df.at[idxs[j], "Timestamp"] == shared
            ):
                j += 1

            block = idxs[i:j]
            if len(block) >= 1:
                activities = [df.at[b, "Activity"] for b in block]
                observed_ts = df.at[block[0], "_parsed_ts"]
                max_digits = max(df.at[b, "_digits"] for b in block)

                left_idx = idxs[i - 1] if i - 1 >= 0 else None
                right_idx = idxs[j] if j < len(idxs) else None

                left_clean = left_idx is not None and df.at[left_idx, "error_category"] == "CLEAN"
                right_clean = right_idx is not None and df.at[right_idx, "error_category"] == "CLEAN"

                left_act = df.at[left_idx, "Activity"] if left_clean else None
                right_act = df.at[right_idx, "Activity"] if right_clean else None
                left_ts = df.at[left_idx, "_parsed_ts"] if left_clean else case_start
                right_ts = df.at[right_idx, "_parsed_ts"] if right_clean else case_end

                order = infer_order(activities, left_act, right_act)
                ordered_row_idxs = assign_rows_for_order([(b, df.at[b, "Activity"]) for b in block], order)

                times, was_clamped = create_times(
                    start_ts=observed_ts,
                    ordered_activities=order,
                    left_ts=left_ts if left_clean else case_start,
                    right_ts=right_ts if right_clean else case_end,
                    max_frac_digits=max_digits,
                )

                if len(times) == 1:
                    if right_ts > observed_ts:
                        times = [observed_ts]
                    else:
                        times = [observed_ts]

                mapping = {}
                for ridx, t in zip(ordered_row_idxs, times):
                    mapping[ridx] = t

                # self-check and corrective pass
                sorted_exec = [(r, mapping[r]) for r in ordered_row_idxs]
                exec_times = [t for _, t in sorted_exec]

                need_fix = False
                if any(exec_times[x] >= exec_times[x + 1] for x in range(len(exec_times) - 1)):
                    need_fix = True
                if exec_times[0] != observed_ts:
                    need_fix = True
                if right_clean:
                    total_span = exec_times[-1] - exec_times[0]
                    room = right_ts - observed_ts
                    if total_span > room:
                        need_fix = True
                if left_clean:
                    if any(t < left_ts for t in exec_times):
                        need_fix = True
                if right_clean:
                    if any(t > right_ts for t in exec_times):
                        need_fix = True
                if any(t < case_start or t > case_end for t in exec_times):
                    need_fix = True

                if need_fix:
                    times = enforce_strict(
                        [observed_ts] + exec_times[1:],
                        left_ts if left_clean else case_start,
                        right_ts if right_clean else case_end,
                        max_digits,
                    )
                    if right_clean and len(times) >= 2 and times[-1] > right_ts:
                        q = quantum_seconds(max_digits)
                        min_step = max(q, Decimal("0.001"))
                        if max_digits == 0:
                            min_step = Decimal("1")
                        start = observed_ts
                        room = to_dec_seconds(right_ts - start)
                        if room < Decimal("0"):
                            room = Decimal("0")
                        if len(times) > 1:
                            if room >= min_step * Decimal(len(times) - 1):
                                cur = start
                                times = [start]
                                small_gaps = [min_step] * (len(times) - 1)
                                for _ in range(1, len(block)):
                                    cur = cur + pdtd_from_decimal_seconds(min_step)
                                    times.append(cur)
                            else:
                                # keep within anchors even if extremely tight
                                times = [start]
                                cur = start
                                for _ in range(1, len(block)):
                                    cur = cur + pdtd_from_decimal_seconds(min_step)
                                    if cur > right_ts:
                                        cur = right_ts
                                    times.append(cur)
                                times = enforce_strict(
                                    times,
                                    left_ts if left_clean else case_start,
                                    right_ts if right_clean else case_end,
                                    max_digits,
                                )
                    was_clamped = True
                    mapping = {}
                    for ridx, t in zip(ordered_row_idxs, times):
                        mapping[ridx] = t

                for b in block:
                    t = mapping[b]
                    if b == ordered_row_idxs[0]:
                        out_s = df.at[b, "_orig_ts"]
                    else:
                        out_s = format_dt(t, df.at[b, "_digits"])
                        if out_s == df.at[b, "_orig_ts"]:
                            q = quantum_seconds(df.at[b, "_digits"])
                            step = max(q, Decimal("0.001"))
                            if df.at[b, "_digits"] == 0:
                                step = Decimal("1")
                            t2 = t + pdtd_from_decimal_seconds(step)
                            if right_clean and t2 > right_ts:
                                t2 = t - pdtd_from_decimal_seconds(step)
                            out_s = format_dt(t2, df.at[b, "_digits"])
                    df.at[b, "_out_ts"] = out_s

                repaired_blocks += 1
                if was_clamped:
                    clamped_blocks += 1

            i = j

    # Final self-check and fixes
    rows_still_equal = 0
    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        i = 0
        while i < len(idxs):
            idx = idxs[i]
            if df.at[idx, "error_category"] == "CLEAN":
                if df.at[idx, "_out_ts"] != df.at[idx, "_orig_ts"]:
                    df.at[idx, "_out_ts"] = df.at[idx, "_orig_ts"]
                i += 1
                continue

            j = i + 1
            shared = df.at[idx, "_orig_ts"]
            while (
                j < len(idxs)
                and df.at[idxs[j], "error_category"] != "CLEAN"
                and df.at[idxs[j], "_orig_ts"] == shared
            ):
                j += 1

            block = idxs[i:j]
            parsed = [to_timestamp(df.at[b, "_out_ts"]) for b in block]
            equal_mask = [df.at[b, "_out_ts"] == df.at[b, "_orig_ts"] for b in block]

            if sum(equal_mask) != 1:
                leftmost_keep = None
                for b in block:
                    if df.at[b, "_out_ts"] == df.at[b, "_orig_ts"]:
                        leftmost_keep = b
                        break
                if leftmost_keep is None:
                    leftmost_keep = block[0]
                    df.at[leftmost_keep, "_out_ts"] = df.at[leftmost_keep, "_orig_ts"]
                for b in block:
                    if b != leftmost_keep and df.at[b, "_out_ts"] == df.at[b, "_orig_ts"]:
                        d = df.at[b, "_digits"]
                        q = quantum_seconds(d)
                        step = max(q, Decimal("0.001"))
                        if d == 0:
                            step = Decimal("1")
                        t = to_timestamp(df.at[b, "_out_ts"]) + pdtd_from_decimal_seconds(step)
                        df.at[b, "_out_ts"] = format_dt(t, d)
                clamped_blocks += 1

            rows_still_equal += sum(1 for b in block if df.at[b, "_out_ts"] == df.at[b, "_orig_ts"])
            i = j

    out_df = df[orig_cols].copy()
    out_df["Timestamp"] = df["_out_ts"]

    try:
        out_df.to_csv(args.output, index=False)
    except Exception:
        # Last-resort non-failing write
        out_df.to_csv(args.output, index=False)

    print(f"blocks_repaired={repaired_blocks} blocks_clamped={clamped_blocks} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    main()