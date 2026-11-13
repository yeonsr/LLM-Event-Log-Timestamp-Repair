import argparse
import itertools
import math
from collections import Counter, defaultdict

import pandas as pd


# =========================
# Embedded process context
# =========================

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
    (["New online application received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 947),
    (["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], 531),
    (["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 437),
    (["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], 215),
    (["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 202),
    (["New online application received", "Notify accept", "Deliver card", "EVENT 13 END"], 138),
    (["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], 120),
    (["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 106),
    (["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 89),
    (["New online application received", "notify reject", "time out", "EVENT 13 END"], 75),
    (["New online application received", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 75),
    (["New online application received", "Check for completeness", "Notify accept", "Deliver card", "EVENT 13 END"], 72),
    (["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 61),
    (["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], 54),
    (["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 49),
    (["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], 49),
    (["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 47),
    (["New online application received", "Make decision", "notify reject", "time out", "EVENT 13 END"], 36),
    (["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], 36),
    (["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], 36),
]

DFG_EDGES = [
    ("Check for completeness", "Request info", 6227, 418.728),
    ("Request info", "info received", 6080, 10.0),
    ("info received", "Check for completeness", 6008, 420.682),
    ("Perform checks", "Make decision", 5745, 422.924),
    ("Check for completeness", "Perform checks", 5650, 420.3715),
    ("New online application received", "Check for completeness", 4550, 426.0135),
    ("Notify accept", "Deliver card", 3276, 425.762),
    ("Deliver card", "EVENT 13 END", 3276, 0.0),
    ("Make decision", "notify reject", 3088, 426.0405),
    ("Make decision", "Notify accept", 2969, 428.03),
    ("notify reject", "time out", 1724, 10.0),
    ("time out", "EVENT 13 END", 1724, 0.0),
    ("notify reject", "review request received", 1534, 10.0),
    ("review request received", "Check for completeness", 1527, 441.348),
    ("New online application received", "Notify accept", 138, 2020.112),
    ("New online application received", "Make decision", 117, 1661.293),
    ("Check for completeness", "Notify accept", 103, 1780.137),
    ("New online application received", "notify reject", 82, 2159.5775),
    ("Check for completeness", "Make decision", 72, 2023.867),
    ("info received", "Make decision", 66, 1868.191),
    ("Check for completeness", "notify reject", 60, 1757.751),
    ("New online application received", "info received", 53, 857.047),
    ("Request info", "Perform checks", 53, 928.08),
    ("info received", "info received", 52, 1302.5635),
    ("Request info", "Make decision", 50, 1523.8755),
    ("Request info", "Request info", 43, 1164.851),
    ("Request info", "Check for completeness", 34, 1889.409),
    ("New online application received", "Perform checks", 33, 2363.648),
    ("Request info", "notify reject", 30, 2454.903),
    ("Request info", "Notify accept", 30, 2396.886),
    ("info received", "notify reject", 29, 2806.157),
    ("info received", "Perform checks", 27, 1874.355),
    ("info received", "Notify accept", 24, 2230.734),
    ("Check for completeness", "info received", 19, 1910.593),
    ("New online application received", "Request info", 17, 2254.706),
    ("Perform checks", "Check for completeness", 15, 1995.413),
    ("Perform checks", "review request received", 15, 705.0),
    ("Check for completeness", "Check for completeness", 13, 2577.014),
    ("notify reject", "Make decision", 12, 1518.5795),
    ("review request received", "Make decision", 12, 2227.617),
    ("Perform checks", "Request info", 11, 2625.632),
    ("New online application received", "review request received", 10, 1845.21),
    ("Check for completeness", "review request received", 9, 1363.124),
    ("review request received", "Notify accept", 9, 2296.639),
    ("info received", "Request info", 9, 1993.345),
    ("notify reject", "Perform checks", 9, 874.95),
    ("review request received", "Perform checks", 9, 2249.376),
    ("EVENT 13 END", "EVENT 13 END", 8, 0.0),
    ("Make decision", "info received", 7, 1863.824),
    ("notify reject", "Check for completeness", 6, 3075.0545),
    ("Make decision", "Make decision", 6, 2015.0765),
    ("Perform checks", "Perform checks", 6, 2294.3645),
    ("notify reject", "notify reject", 5, 2967.996),
    ("Make decision", "Perform checks", 5, 823.397),
    ("review request received", "Request info", 5, 2501.573),
    ("review request received", "notify reject", 5, 1469.711),
    ("Make decision", "Request info", 5, 1325.858),
    ("review request received", "info received", 4, 1983.33),
    ("notify reject", "info received", 3, 836.005),
    ("notify reject", "Request info", 3, 986.388),
]


# =========================
# Precompute maps
# =========================

TRANS_PROB = {(s, t): p for s, t, p in TRANSITION_PROBABILITIES_TOP}
DURATION_MAP = {}
for s, t, n, p10, p25, median, p75, p90, share in DURATION_PROFILES_TOP:
    DURATION_MAP[(s, t)] = {
        "n": n,
        "p10": p10,
        "p25": p25,
        "median": median,
        "p75": p75,
        "p90": p90,
        "share_under_60s": share,
    }
DFG_FREQ = {(s, t): freq for s, t, freq, med in DFG_EDGES}
ALL_EDGES = set(DFG_FREQ.keys())


# =========================
# Utility
# =========================

def detect_precision(ts_str: str) -> int:
    if "." in ts_str:
        frac = ts_str.split(".")[-1]
        return len(frac)
    return 0


def parse_ts(ts_str: str) -> pd.Timestamp:
    return pd.to_datetime(ts_str)


def fmt_ts(ts: pd.Timestamp, decimals: int) -> str:
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if decimals <= 0:
        return base
    frac = f"{ts.microsecond:06d}"[:decimals]
    return f"{base}.{frac}"


def min_increment_ns(decimals: int) -> int:
    if decimals <= 0:
        return 1_000_000_000
    if decimals <= 6:
        return 10 ** (9 - decimals)
    return 1


def choose_gap_seconds(a, b):
    prof = DURATION_MAP.get((a, b))
    if prof:
        if prof["share_under_60s"] >= 0.8:
            g = prof["p10"]
        else:
            g = prof["p25"]
        if g <= 0:
            return 1.0
        return max(g, 1.0)
    if (a, b) in ALL_EDGES:
        return 1.0
    return 1.0


def multiset_equal(seq1, seq2):
    return Counter(seq1) == Counter(seq2)


def contiguous_subsequence_support(order, left_act, right_act):
    support = 0
    seq = list(order)
    for var, cnt in FREQUENT_VARIANTS:
        ext = var
        if left_act is not None and right_act is not None:
            target = [left_act] + seq + [right_act]
            w = len(target)
            for i in range(len(ext) - w + 1):
                if ext[i:i + w] == target:
                    support += cnt * 4
        if left_act is not None:
            target = [left_act] + seq
            w = len(target)
            for i in range(len(ext) - w + 1):
                if ext[i:i + w] == target:
                    support += cnt * 2
        if right_act is not None:
            target = seq + [right_act]
            w = len(target)
            for i in range(len(ext) - w + 1):
                if ext[i:i + w] == target:
                    support += cnt * 2
        target = seq
        w = len(target)
        for i in range(len(ext) - w + 1):
            if ext[i:i + w] == target:
                support += cnt
    return support


def variant_subsequence_orders(block_acts, left_act, right_act):
    results = defaultdict(int)
    target_counter = Counter(block_acts)
    k = len(block_acts)
    for var, cnt in FREQUENT_VARIANTS:
        n = len(var)
        for i in range(n - k + 1):
            sub = var[i:i + k]
            if Counter(sub) != target_counter:
                continue
            if left_act is not None and i > 0 and var[i - 1] != left_act:
                continue
            if right_act is not None and i + k < n and var[i + k] != right_act:
                continue
            results[tuple(sub)] += cnt
    return results


def score_order(order, left_act, right_act):
    support = contiguous_subsequence_support(order, left_act, right_act)
    edge_score = 0.0
    seq = list(order)
    ext = []
    if left_act is not None:
        ext.append(left_act)
    ext.extend(seq)
    if right_act is not None:
        ext.append(right_act)

    valid = True
    for a, b in zip(ext[:-1], ext[1:]):
        if (a, b) not in ALL_EDGES:
            valid = False
            break
        freq = DFG_FREQ.get((a, b), 1)
        prob = TRANS_PROB.get((a, b), 1e-6)
        edge_score += math.log(freq + 1.0) + 3.0 * math.log(prob + 1e-12)
    return valid, support, edge_score


def infer_order(activities, left_act, right_act):
    k = len(activities)
    candidates = []

    var_orders = variant_subsequence_orders(activities, left_act, right_act)
    for ord_tup, cnt in var_orders.items():
        valid, support, edge_score = score_order(ord_tup, left_act, right_act)
        if valid:
            candidates.append((ord_tup, support + 1000 * cnt, edge_score))

    uniq_perms = set(itertools.permutations(activities, k))
    for perm in uniq_perms:
        valid, support, edge_score = score_order(perm, left_act, right_act)
        if valid:
            candidates.append((perm, support, edge_score))

    if not candidates:
        relaxed = []
        for perm in uniq_perms:
            seq = list(perm)
            ext = []
            if left_act is not None:
                ext.append(left_act)
            ext.extend(seq)
            if right_act is not None:
                ext.append(right_act)
            misses = 0
            edge_score = 0.0
            for a, b in zip(ext[:-1], ext[1:]):
                if (a, b) not in ALL_EDGES:
                    misses += 1
                    edge_score -= 100.0
                else:
                    edge_score += math.log(DFG_FREQ.get((a, b), 1) + 1.0) + 3.0 * math.log(TRANS_PROB.get((a, b), 1e-6) + 1e-12)
            support = contiguous_subsequence_support(seq, left_act, right_act)
            relaxed.append((perm, -misses, support, edge_score))
        relaxed.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        best = relaxed[0][0]
        return list(best)

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return list(candidates[0][0])


def detect_concurrency_pairs(order):
    conc = []
    for a, b in zip(order[:-1], order[1:]):
        pab = TRANS_PROB.get((a, b), 0.0)
        pba = TRANS_PROB.get((b, a), 0.0)
        fab = DFG_FREQ.get((a, b), 0)
        fba = DFG_FREQ.get((b, a), 0)
        if (a, b) in ALL_EDGES and (b, a) in ALL_EDGES:
            if pab > 0 and pba > 0:
                ratio_p = min(pab, pba) / max(pab, pba)
            else:
                ratio_p = 0.0
            if fab > 0 and fba > 0:
                ratio_f = min(fab, fba) / max(fab, fba)
            else:
                ratio_f = 0.0
            if ratio_p >= 0.8 and ratio_f >= 0.8:
                conc.append(True)
                continue
        conc.append(False)
    return conc


def build_forward_times(order, observed_ts, right_anchor_ts, case_end_ts, decimals):
    min_ns = min_increment_ns(decimals)
    min_td = pd.to_timedelta(min_ns, unit="ns")
    cap_ts = right_anchor_ts if right_anchor_ts is not None else case_end_ts
    if cap_ts is None or cap_ts < observed_ts:
        cap_ts = observed_ts
    room_ns = max(0, int((cap_ts - observed_ts).value))
    k = len(order)
    min_span_ns = (k - 1) * min_ns

    gaps_ns = []
    concurrency = detect_concurrency_pairs(order)
    for idx, (a, b) in enumerate(zip(order[:-1], order[1:])):
        if concurrency[idx]:
            gaps_ns.append(min_ns)
        else:
            sec = choose_gap_seconds(a, b)
            ns = max(int(round(sec * 1e9)), min_ns)
            gaps_ns.append(ns)

    total_ns = sum(gaps_ns)

    clamped = False
    if room_ns < min_span_ns:
        gaps_ns = [min_ns] * (k - 1)
        clamped = True
    elif total_ns > room_ns:
        scalable = [g - min_ns for g in gaps_ns]
        scalable_total = sum(scalable)
        target_extra = room_ns - min_span_ns
        if scalable_total <= 0:
            gaps_ns = [min_ns] * (k - 1)
        else:
            new_gaps = []
            raw = []
            for g in gaps_ns:
                extra = g - min_ns
                raw_extra = extra * target_extra / scalable_total
                raw.append(raw_extra)
            floors = [int(x // min_ns) * min_ns for x in raw]
            rem = target_extra - sum(floors)
            frac_order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
            extras = floors[:]
            j = 0
            while rem >= min_ns and frac_order:
                extras[frac_order[j % len(frac_order)]] += min_ns
                rem -= min_ns
                j += 1
            for i, g in enumerate(gaps_ns):
                new_gaps.append(min_ns + extras[i])
            gaps_ns = new_gaps
        clamped = True

    times = [observed_ts]
    cur = observed_ts
    for g in gaps_ns:
        cur = cur + pd.to_timedelta(g, unit="ns")
        if cur > cap_ts:
            cur = cap_ts
            clamped = True
        times.append(cur)

    # Enforce strict increase in execution order without exceeding cap if possible
    repaired = [times[0]]
    for i in range(1, len(times)):
        t = times[i]
        min_allowed = repaired[i - 1] + min_td
        if t < min_allowed:
            t = min_allowed
            clamped = True
        repaired.append(t)

    if repaired[-1] > cap_ts:
        repaired = [observed_ts + i * min_td for i in range(k)]
        clamped = True
        if repaired[-1] > cap_ts:
            # impossible to fit; pin backward from cap then restore start if needed
            repaired = [cap_ts - (k - 1 - i) * min_td for i in range(k)]
            if repaired[0] < observed_ts:
                repaired = [observed_ts + i * min_td for i in range(k)]
            clamped = True

    return repaired, clamped


def build_backward_times(order, observed_ts, left_anchor_ts, case_start_ts, decimals):
    min_ns = min_increment_ns(decimals)
    min_td = pd.to_timedelta(min_ns, unit="ns")
    floor_ts = left_anchor_ts if left_anchor_ts is not None else case_start_ts
    if floor_ts is None or floor_ts > observed_ts:
        floor_ts = observed_ts
    room_ns = max(0, int((observed_ts - floor_ts).value))
    k = len(order)
    min_span_ns = (k - 1) * min_ns

    gaps_ns = []
    rev = list(reversed(order))
    concurrency = detect_concurrency_pairs(rev)
    for idx, (a, b) in enumerate(zip(rev[:-1], rev[1:])):
        if concurrency[idx]:
            gaps_ns.append(min_ns)
        else:
            sec = choose_gap_seconds(a, b)
            ns = max(int(round(sec * 1e9)), min_ns)
            gaps_ns.append(ns)

    total_ns = sum(gaps_ns)
    clamped = False
    if room_ns < min_span_ns:
        gaps_ns = [min_ns] * (k - 1)
        clamped = True
    elif total_ns > room_ns:
        scalable = [g - min_ns for g in gaps_ns]
        scalable_total = sum(scalable)
        target_extra = room_ns - min_span_ns
        if scalable_total <= 0:
            gaps_ns = [min_ns] * (k - 1)
        else:
            raw = [extra * target_extra / scalable_total for extra in scalable]
            floors = [int(x // min_ns) * min_ns for x in raw]
            rem = target_extra - sum(floors)
            frac_order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
            extras = floors[:]
            j = 0
            while rem >= min_ns and frac_order:
                extras[frac_order[j % len(frac_order)]] += min_ns
                rem -= min_ns
                j += 1
            gaps_ns = [min_ns + extras[i] for i in range(len(gaps_ns))]
        clamped = True

    rev_times = [observed_ts]
    cur = observed_ts
    for g in gaps_ns:
        cur = cur - pd.to_timedelta(g, unit="ns")
        if cur < floor_ts:
            cur = floor_ts
            clamped = True
        rev_times.append(cur)
    times = list(reversed(rev_times))

    repaired = [times[0]]
    for i in range(1, len(times)):
        t = times[i]
        min_allowed = repaired[i - 1] + min_td
        if t < min_allowed:
            t = min_allowed
            clamped = True
        repaired.append(t)

    if repaired[-1] > observed_ts:
        repaired = [observed_ts - (k - 1 - i) * min_td for i in range(k)]
        if repaired[0] < floor_ts:
            repaired = [floor_ts + i * min_td for i in range(k)]
        clamped = True

    return repaired, clamped


def find_blocks(df):
    blocks = []
    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        idxs = list(idxs)
        i = 0
        while i < len(idxs):
            idx = idxs[i]
            if df.at[idx, "error_category"] == "C1_ORDERING_AMBIGUITY":
                ts = df.at[idx, "Timestamp"]
                j = i + 1
                members = [idx]
                while j < len(idxs):
                    idx2 = idxs[j]
                    if df.at[idx2, "error_category"] == "C1_ORDERING_AMBIGUITY" and df.at[idx2, "Timestamp"] == ts:
                        members.append(idx2)
                        j += 1
                    else:
                        break
                blocks.append((case_id, members))
                i = j
            else:
                i += 1
    return blocks


def self_check_and_fix(df, original_ts, repaired_idxs, block_meta):
    clamp_blocks = 0
    rows_equal_input = 0

    for meta in block_meta:
        idxs = meta["idxs"]
        order_rows = meta["ordered_rows"]
        placement = meta["placement"]
        observed_str = meta["observed_str"]
        observed_ts = parse_ts(observed_str)
        decimals = meta["decimals"]
        min_ns = min_increment_ns(decimals)
        min_td = pd.to_timedelta(min_ns, unit="ns")
        left_anchor_ts = meta["left_anchor_ts"]
        right_anchor_ts = meta["right_anchor_ts"]
        case_start_ts = meta["case_start_ts"]
        case_end_ts = meta["case_end_ts"]

        ts_map = {idx: parse_ts(df.at[idx, "Timestamp"]) for idx in idxs}
        exec_times = [ts_map[idx] for idx in order_rows]

        bad = False
        for i in range(1, len(exec_times)):
            if not (exec_times[i] > exec_times[i - 1]):
                bad = True
                break

        if placement == "start":
            observed_count = sum(df.at[idx, "Timestamp"] == observed_str for idx in idxs)
            if observed_count != 1 or df.at[order_rows[0], "Timestamp"] != observed_str:
                bad = True
        else:
            observed_count = sum(df.at[idx, "Timestamp"] == observed_str for idx in idxs)
            if observed_count != 1 or df.at[order_rows[-1], "Timestamp"] != observed_str:
                bad = True

        for idx in idxs:
            t = parse_ts(df.at[idx, "Timestamp"])
            if t < case_start_ts or t > case_end_ts:
                bad = True
                break
            if left_anchor_ts is not None and right_anchor_ts is not None:
                if t < left_anchor_ts or t > right_anchor_ts:
                    bad = True
                    break

        span_ok = True
        if placement == "start":
            cap_ts = right_anchor_ts if right_anchor_ts is not None else case_end_ts
            span = exec_times[-1] - observed_ts
            if span.value > max(0, (cap_ts - observed_ts).value) or span.value < (len(order_rows) - 1) * min_ns:
                span_ok = False
        else:
            floor_ts = left_anchor_ts if left_anchor_ts is not None else case_start_ts
            span = observed_ts - exec_times[0]
            if span.value > max(0, (observed_ts - floor_ts).value) or span.value < (len(order_rows) - 1) * min_ns:
                span_ok = False

        if bad or not span_ok:
            clamp_blocks += 1
            if placement == "start":
                rebuilt, _ = build_forward_times(
                    meta["ordered_activities"], observed_ts, right_anchor_ts, case_end_ts, decimals
                )
            else:
                rebuilt, _ = build_backward_times(
                    meta["ordered_activities"], observed_ts, left_anchor_ts, case_start_ts, decimals
                )
            for row_idx, t in zip(order_rows, rebuilt):
                if placement == "start" and row_idx == order_rows[0]:
                    df.at[row_idx, "Timestamp"] = observed_str
                elif placement == "end" and row_idx == order_rows[-1]:
                    df.at[row_idx, "Timestamp"] = observed_str
                else:
                    df.at[row_idx, "Timestamp"] = fmt_ts(t, decimals)

            # final anti-negative-duration safeguard by row order within case:
            case_idxs = list(df.index[df["Case"] == meta["case_id"]])
            pos = {idx: p for p, idx in enumerate(case_idxs)}
            block_positions = sorted([pos[idx] for idx in idxs])
            for p in block_positions:
                idx = case_idxs[p]
                cur = parse_ts(df.at[idx, "Timestamp"])
                prev_clean_or_repaired = parse_ts(df.at[case_idxs[p - 1], "Timestamp"]) if p > 0 else None
                next_clean_or_repaired = parse_ts(df.at[case_idxs[p + 1], "Timestamp"]) if p + 1 < len(case_idxs) else None
                low = case_start_ts if prev_clean_or_repaired is None else max(case_start_ts, prev_clean_or_repaired)
                high = case_end_ts if next_clean_or_repaired is None else min(case_end_ts, next_clean_or_repaired)
                if cur < low:
                    cur = low
                if cur > high:
                    cur = high
                if p > 0 and cur < prev_clean_or_repaired:
                    cur = prev_clean_or_repaired
                if p + 1 < len(case_idxs) and cur > next_clean_or_repaired:
                    cur = next_clean_or_repaired
                df.at[idx, "Timestamp"] = fmt_ts(cur, decimals)

    for idx in repaired_idxs:
        if df.at[idx, "Timestamp"] == original_ts[idx]:
            rows_equal_input += 1

    return clamp_blocks, rows_equal_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        original_ts = df["Timestamp"].copy().to_dict()
        output_ts = df["Timestamp"].copy()

        blocks = find_blocks(df)

        repaired_block_count = 0
        repaired_idxs = set()
        block_meta = []

        for case_id, idxs in blocks:
            if len(idxs) == 0:
                continue

            case_df = df[df["Case"] == case_id]
            case_idxs = list(case_df.index)
            pos_map = {idx: i for i, idx in enumerate(case_idxs)}

            observed_str = df.at[idxs[0], "Timestamp"]
            observed_ts = parse_ts(observed_str)
            decimals = detect_precision(observed_str)

            case_start_ts = min(parse_ts(ts) for ts in case_df["Timestamp"])
            case_end_ts = max(parse_ts(ts) for ts in case_df["Timestamp"])

            left_anchor_idx = None
            p0 = pos_map[idxs[0]]
            for p in range(p0 - 1, -1, -1):
                idx = case_idxs[p]
                if df.at[idx, "error_category"] == "CLEAN":
                    left_anchor_idx = idx
                    break

            right_anchor_idx = None
            p1 = pos_map[idxs[-1]]
            for p in range(p1 + 1, len(case_idxs)):
                idx = case_idxs[p]
                if df.at[idx, "error_category"] == "CLEAN":
                    right_anchor_idx = idx
                    break

            left_anchor_act = df.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None else None
            right_anchor_act = df.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None else None
            left_anchor_ts = parse_ts(df.at[left_anchor_idx, "Timestamp"]) if left_anchor_idx is not None else None
            right_anchor_ts = parse_ts(df.at[right_anchor_idx, "Timestamp"]) if right_anchor_idx is not None else None

            acts = [df.at[idx, "Activity"] for idx in idxs]
            order = infer_order(acts, left_anchor_act, right_anchor_act)

            # map ordered activities back to concrete rows, handling duplicates stably by original row order
            act_to_rows = defaultdict(list)
            for idx in idxs:
                act_to_rows[df.at[idx, "Activity"]].append(idx)
            ordered_rows = []
            used = set()
            for act in order:
                for idx in act_to_rows[act]:
                    if idx not in used:
                        ordered_rows.append(idx)
                        used.add(idx)
                        break

            # default placement: start at observed
            times_fwd, clamp_fwd = build_forward_times(order, observed_ts, right_anchor_ts, case_end_ts, decimals)
            valid_fwd = True
            if left_anchor_ts is not None:
                for t in times_fwd:
                    if t < left_anchor_ts:
                        valid_fwd = False
                        break
            if right_anchor_ts is not None:
                for t in times_fwd:
                    if t > right_anchor_ts:
                        valid_fwd = False
                        break

            placement = "start"
            final_times = times_fwd
            clamped = clamp_fwd

            # rare end-at-observed fallback
            if not valid_fwd:
                rev_order = list(reversed(order))
                times_bwd, clamp_bwd = build_backward_times(order, observed_ts, left_anchor_ts, case_start_ts, decimals)
                valid_bwd = True
                if left_anchor_ts is not None:
                    for t in times_bwd:
                        if t < left_anchor_ts:
                            valid_bwd = False
                            break
                if right_anchor_ts is not None:
                    for t in times_bwd:
                        if t > right_anchor_ts:
                            valid_bwd = False
                            break
                if valid_bwd:
                    placement = "end"
                    final_times = times_bwd
                    clamped = clamp_bwd

            # write back to original rows, preserving row order
            if placement == "start":
                for i, row_idx in enumerate(ordered_rows):
                    if i == 0:
                        output_ts.at[row_idx] = observed_str
                    else:
                        output_ts.at[row_idx] = fmt_ts(final_times[i], decimals)
            else:
                for i, row_idx in enumerate(ordered_rows):
                    if i == len(ordered_rows) - 1:
                        output_ts.at[row_idx] = observed_str
                    else:
                        output_ts.at[row_idx] = fmt_ts(final_times[i], decimals)

            repaired_block_count += 1
            for row_idx in idxs:
                repaired_idxs.add(row_idx)

            block_meta.append({
                "case_id": case_id,
                "idxs": idxs,
                "ordered_rows": ordered_rows,
                "ordered_activities": order,
                "placement": placement,
                "observed_str": observed_str,
                "decimals": decimals,
                "left_anchor_ts": left_anchor_ts,
                "right_anchor_ts": right_anchor_ts,
                "case_start_ts": case_start_ts,
                "case_end_ts": case_end_ts,
                "clamped": clamped,
            })

        # Apply repaired timestamps only to corrupted rows; CLEAN remain byte-identical
        for idx in repaired_idxs:
            df.at[idx, "Timestamp"] = output_ts.at[idx]

        # hard preserve all CLEAN rows
        clean_mask = df["error_category"] == "CLEAN"
        df.loc[clean_mask, "Timestamp"] = pd.Series(original_ts)

        clamp_blocks, rows_equal_input = self_check_and_fix(df, original_ts, repaired_idxs, block_meta)

        # final preserve non-timestamp columns byte-identical by not touching them
        df.to_csv(args.output, index=False)

        print(f"blocks_repaired={repaired_block_count} blocks_clamped={clamp_blocks} rows_still_equal_input={rows_equal_input}")
    except Exception:
        # fail-safe: write original input unchanged if anything unexpected happens
        try:
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_input=0")
        except Exception:
            # last-resort no-raise behavior
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    with open(args.input, "r", encoding="utf-8") as src:
                        f.write(src.read())
                print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_input=0")
            except Exception:
                print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_input=0")


if __name__ == "__main__":
    main()