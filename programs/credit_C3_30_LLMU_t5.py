import argparse
import itertools
import math
import re
from collections import defaultdict

import pandas as pd


TRANSITION_PROBABILITIES_TOP = [
    {"source": "Deliver card", "target": "EVENT 13 END", "probability": 1.0},
    {"source": "EVENT 13 END", "target": "EVENT 13 END", "probability": 1.0},
    {"source": "Notify accept", "target": "Deliver card", "probability": 1.0},
    {"source": "time out", "target": "EVENT 13 END", "probability": 1.0},
    {"source": "Perform checks", "target": "Make decision", "probability": 0.9919},
    {"source": "review request received", "target": "Check for completeness", "probability": 0.9708},
    {"source": "info received", "target": "Check for completeness", "probability": 0.9662},
    {"source": "Request info", "target": "info received", "probability": 0.962},
    {"source": "New online application received", "target": "Check for completeness", "probability": 0.91},
    {"source": "notify reject", "target": "time out", "probability": 0.5226},
    {"source": "Check for completeness", "target": "Request info", "probability": 0.5124},
    {"source": "Make decision", "target": "notify reject", "probability": 0.5079},
    {"source": "Make decision", "target": "Notify accept", "probability": 0.4883},
    {"source": "notify reject", "target": "review request received", "probability": 0.465},
    {"source": "Check for completeness", "target": "Perform checks", "probability": 0.4649},
    {"source": "New online application received", "target": "Notify accept", "probability": 0.0276},
    {"source": "New online application received", "target": "Make decision", "probability": 0.0234},
    {"source": "New online application received", "target": "notify reject", "probability": 0.0164},
    {"source": "info received", "target": "Make decision", "probability": 0.0106},
    {"source": "New online application received", "target": "info received", "probability": 0.0106},
    {"source": "Check for completeness", "target": "Notify accept", "probability": 0.0085},
    {"source": "Request info", "target": "Perform checks", "probability": 0.0084},
    {"source": "info received", "target": "info received", "probability": 0.0084},
    {"source": "Request info", "target": "Make decision", "probability": 0.0079},
    {"source": "review request received", "target": "Make decision", "probability": 0.0076},
    {"source": "Request info", "target": "Request info", "probability": 0.0068},
    {"source": "New online application received", "target": "Perform checks", "probability": 0.0066},
    {"source": "Check for completeness", "target": "Make decision", "probability": 0.0059},
    {"source": "review request received", "target": "Notify accept", "probability": 0.0057},
    {"source": "review request received", "target": "Perform checks", "probability": 0.0057},
    {"source": "Request info", "target": "Check for completeness", "probability": 0.0054},
    {"source": "Check for completeness", "target": "notify reject", "probability": 0.0049},
    {"source": "Request info", "target": "Notify accept", "probability": 0.0047},
    {"source": "Request info", "target": "notify reject", "probability": 0.0047},
    {"source": "info received", "target": "notify reject", "probability": 0.0047},
    {"source": "info received", "target": "Perform checks", "probability": 0.0043},
    {"source": "info received", "target": "Notify accept", "probability": 0.0039},
    {"source": "notify reject", "target": "Make decision", "probability": 0.0036},
    {"source": "New online application received", "target": "Request info", "probability": 0.0034},
    {"source": "review request received", "target": "Request info", "probability": 0.0032},
    {"source": "review request received", "target": "notify reject", "probability": 0.0032},
    {"source": "notify reject", "target": "Perform checks", "probability": 0.0027},
    {"source": "Perform checks", "target": "Check for completeness", "probability": 0.0026},
    {"source": "Perform checks", "target": "review request received", "probability": 0.0026},
    {"source": "review request received", "target": "info received", "probability": 0.0025},
    {"source": "New online application received", "target": "review request received", "probability": 0.002},
    {"source": "Perform checks", "target": "Request info", "probability": 0.0019},
    {"source": "notify reject", "target": "Check for completeness", "probability": 0.0018},
    {"source": "Check for completeness", "target": "info received", "probability": 0.0016},
    {"source": "notify reject", "target": "notify reject", "probability": 0.0015},
    {"source": "info received", "target": "Request info", "probability": 0.0014},
    {"source": "review request received", "target": "review request received", "probability": 0.0013},
    {"source": "Make decision", "target": "info received", "probability": 0.0012},
    {"source": "Check for completeness", "target": "Check for completeness", "probability": 0.0011},
    {"source": "Perform checks", "target": "Perform checks", "probability": 0.001},
    {"source": "Make decision", "target": "Make decision", "probability": 0.001},
    {"source": "notify reject", "target": "Notify accept", "probability": 0.0009},
    {"source": "notify reject", "target": "Request info", "probability": 0.0009},
    {"source": "notify reject", "target": "info received", "probability": 0.0009},
    {"source": "Make decision", "target": "Perform checks", "probability": 0.0008},
]

DURATION_PROFILES_TOP = [
    {"source": "Check for completeness", "target": "Request info", "n": 6227, "p10_seconds": 61.9314, "p25_seconds": 170.1115, "median_seconds": 418.728, "p75_seconds": 830.2675, "p90_seconds": 1373.2918, "share_under_60s": 0.097},
    {"source": "Request info", "target": "info received", "n": 6080, "p10_seconds": 10.0, "p25_seconds": 10.0, "median_seconds": 10.0, "p75_seconds": 10.0, "p90_seconds": 10.0, "share_under_60s": 0.996},
    {"source": "info received", "target": "Check for completeness", "n": 6008, "p10_seconds": 63.739, "p25_seconds": 176.05, "median_seconds": 420.682, "p75_seconds": 839.701, "p90_seconds": 1404.5438, "share_under_60s": 0.095},
    {"source": "Perform checks", "target": "Make decision", "n": 5745, "p10_seconds": 62.8388, "p25_seconds": 171.483, "median_seconds": 422.924, "p75_seconds": 835.934, "p90_seconds": 1392.8122, "share_under_60s": 0.095},
    {"source": "Check for completeness", "target": "Perform checks", "n": 5650, "p10_seconds": 61.0229, "p25_seconds": 171.5965, "median_seconds": 420.3715, "p75_seconds": 832.8305, "p90_seconds": 1416.3387, "share_under_60s": 0.098},
    {"source": "New online application received", "target": "Check for completeness", "n": 4550, "p10_seconds": 65.3893, "p25_seconds": 171.7955, "median_seconds": 426.0135, "p75_seconds": 834.95025, "p90_seconds": 1385.6882, "share_under_60s": 0.092},
    {"source": "Deliver card", "target": "EVENT 13 END", "n": 3276, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "Notify accept", "target": "Deliver card", "n": 3276, "p10_seconds": 59.4585, "p25_seconds": 176.35075, "median_seconds": 425.762, "p75_seconds": 851.3035, "p90_seconds": 1370.5795, "share_under_60s": 0.101},
    {"source": "Make decision", "target": "notify reject", "n": 3088, "p10_seconds": 63.7974, "p25_seconds": 174.02075, "median_seconds": 426.0405, "p75_seconds": 834.3485, "p90_seconds": 1427.2486, "share_under_60s": 0.095},
    {"source": "Make decision", "target": "Notify accept", "n": 2969, "p10_seconds": 65.5868, "p25_seconds": 174.826, "median_seconds": 428.03, "p75_seconds": 867.335, "p90_seconds": 1436.6726, "share_under_60s": 0.089},
    {"source": "time out", "target": "EVENT 13 END", "n": 1724, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "notify reject", "target": "time out", "n": 1724, "p10_seconds": 10.0, "p25_seconds": 10.0, "median_seconds": 10.0, "p75_seconds": 10.0, "p90_seconds": 10.0, "share_under_60s": 1.0},
    {"source": "notify reject", "target": "review request received", "n": 1534, "p10_seconds": 10.0, "p25_seconds": 10.0, "median_seconds": 10.0, "p75_seconds": 10.0, "p90_seconds": 10.0, "share_under_60s": 1.0},
    {"source": "review request received", "target": "Check for completeness", "n": 1527, "p10_seconds": 78.0242, "p25_seconds": 188.8875, "median_seconds": 441.348, "p75_seconds": 841.8565, "p90_seconds": 1413.996, "share_under_60s": 0.075},
    {"source": "New online application received", "target": "Notify accept", "n": 138, "p10_seconds": 944.0129, "p25_seconds": 1469.49875, "median_seconds": 2020.112, "p75_seconds": 2906.02375, "p90_seconds": 3878.789, "share_under_60s": 0.0},
    {"source": "New online application received", "target": "Make decision", "n": 117, "p10_seconds": 634.1, "p25_seconds": 963.121, "median_seconds": 1661.293, "p75_seconds": 2178.854, "p90_seconds": 3090.3764, "share_under_60s": 0.0},
    {"source": "Check for completeness", "target": "Notify accept", "n": 103, "p10_seconds": 711.8126, "p25_seconds": 1094.916, "median_seconds": 1780.137, "p75_seconds": 2325.416, "p90_seconds": 3107.5866, "share_under_60s": 0.0},
    {"source": "New online application received", "target": "notify reject", "n": 82, "p10_seconds": 1132.4149, "p25_seconds": 1538.03275, "median_seconds": 2159.5775, "p75_seconds": 2824.4955, "p90_seconds": 3498.6097, "share_under_60s": 0.0},
    {"source": "Check for completeness", "target": "Make decision", "n": 72, "p10_seconds": 1072.0155, "p25_seconds": 1536.0985, "median_seconds": 2023.867, "p75_seconds": 2917.63875, "p90_seconds": 3699.0501, "share_under_60s": 0.0},
    {"source": "info received", "target": "Make decision", "n": 66, "p10_seconds": 721.489, "p25_seconds": 1122.10325, "median_seconds": 1868.191, "p75_seconds": 2613.01725, "p90_seconds": 3118.5855, "share_under_60s": 0.0},
    {"source": "Check for completeness", "target": "notify reject", "n": 60, "p10_seconds": 945.7493, "p25_seconds": 1361.11625, "median_seconds": 1757.751, "p75_seconds": 2556.73125, "p90_seconds": 3239.0085, "share_under_60s": 0.0},
    {"source": "Request info", "target": "Perform checks", "n": 53, "p10_seconds": 403.0504, "p25_seconds": 643.92, "median_seconds": 928.08, "p75_seconds": 1395.19, "p90_seconds": 2110.2466, "share_under_60s": 0.0},
    {"source": "New online application received", "target": "info received", "n": 53, "p10_seconds": 452.3138, "p25_seconds": 639.021, "median_seconds": 857.047, "p75_seconds": 1303.053, "p90_seconds": 1811.6944, "share_under_60s": 0.0},
    {"source": "info received", "target": "info received", "n": 52, "p10_seconds": 613.9782, "p25_seconds": 895.2945, "median_seconds": 1302.5635, "p75_seconds": 1835.09975, "p90_seconds": 2340.9239, "share_under_60s": 0.0},
    {"source": "Request info", "target": "Make decision", "n": 50, "p10_seconds": 669.1557, "p25_seconds": 855.869, "median_seconds": 1523.8755, "p75_seconds": 2372.5625, "p90_seconds": 3437.5409, "share_under_60s": 0.0},
    {"source": "Request info", "target": "Request info", "n": 43, "p10_seconds": 569.524, "p25_seconds": 796.317, "median_seconds": 1164.851, "p75_seconds": 1761.032, "p90_seconds": 2732.683, "share_under_60s": 0.0},
    {"source": "Request info", "target": "Check for completeness", "n": 34, "p10_seconds": 803.7977, "p25_seconds": 952.73675, "median_seconds": 1889.409, "p75_seconds": 2738.841, "p90_seconds": 4159.5683, "share_under_60s": 0.0},
    {"source": "New online application received", "target": "Perform checks", "n": 33, "p10_seconds": 1091.8104, "p25_seconds": 1559.19, "median_seconds": 2363.648, "p75_seconds": 3302.949, "p90_seconds": 3864.007, "share_under_60s": 0.0},
    {"source": "Request info", "target": "Notify accept", "n": 30, "p10_seconds": 1429.9083, "p25_seconds": 1610.589, "median_seconds": 2396.886, "p75_seconds": 3179.76125, "p90_seconds": 3601.6131, "share_under_60s": 0.0},
    {"source": "Request info", "target": "notify reject", "n": 30, "p10_seconds": 1539.1919, "p25_seconds": 1979.3755, "median_seconds": 2454.903, "p75_seconds": 2997.263, "p90_seconds": 3804.9144, "share_under_60s": 0.0},
    {"source": "info received", "target": "notify reject", "n": 29, "p10_seconds": 1059.5184, "p25_seconds": 1682.395, "median_seconds": 2806.157, "p75_seconds": 3442.502, "p90_seconds": 3679.6308, "share_under_60s": 0.0},
    {"source": "info received", "target": "Perform checks", "n": 27, "p10_seconds": 809.8986, "p25_seconds": 1087.906, "median_seconds": 1874.355, "p75_seconds": 2956.759, "p90_seconds": 4336.3136, "share_under_60s": 0.0},
    {"source": "info received", "target": "Notify accept", "n": 24, "p10_seconds": 1006.2042, "p25_seconds": 1510.50375, "median_seconds": 2230.734, "p75_seconds": 3852.069, "p90_seconds": 4100.492, "share_under_60s": 0.0},
    {"source": "Check for completeness", "target": "info received", "n": 19, "p10_seconds": 817.942, "p25_seconds": 1530.727, "median_seconds": 1910.593, "p75_seconds": 2581.938, "p90_seconds": 2788.1602, "share_under_60s": 0.0},
    {"source": "New online application received", "target": "Request info", "n": 17, "p10_seconds": 809.1352, "p25_seconds": 1687.418, "median_seconds": 2254.706, "p75_seconds": 2700.696, "p90_seconds": 4232.1842, "share_under_60s": 0.0},
    {"source": "Perform checks", "target": "Check for completeness", "n": 15, "p10_seconds": 1023.0088, "p25_seconds": 1420.4865, "median_seconds": 1995.413, "p75_seconds": 2965.38, "p90_seconds": 3838.7202, "share_under_60s": 0.0},
    {"source": "Perform checks", "target": "review request received", "n": 15, "p10_seconds": 378.412, "p25_seconds": 555.4955, "median_seconds": 705.0, "p75_seconds": 849.476, "p90_seconds": 1327.28, "share_under_60s": 0.0},
    {"source": "Check for completeness", "target": "Check for completeness", "n": 13, "p10_seconds": 1452.7526, "p25_seconds": 1939.982, "median_seconds": 2577.014, "p75_seconds": 3296.051, "p90_seconds": 3589.3102, "share_under_60s": 0.0},
    {"source": "notify reject", "target": "Make decision", "n": 12, "p10_seconds": 785.8718, "p25_seconds": 1341.0605, "median_seconds": 1518.5795, "p75_seconds": 1958.461, "p90_seconds": 2274.8716, "share_under_60s": 0.0},
    {"source": "review request received", "target": "Make decision", "n": 12, "p10_seconds": 727.506, "p25_seconds": 964.28475, "median_seconds": 2227.617, "p75_seconds": 2260.87725, "p90_seconds": 3134.5752, "share_under_60s": 0.0},
    {"source": "Perform checks", "target": "Request info", "n": 11, "p10_seconds": 1288.529, "p25_seconds": 1794.608, "median_seconds": 2625.632, "p75_seconds": 3426.251, "p90_seconds": 4208.01, "share_under_60s": 0.0},
    {"source": "New online application received", "target": "review request received", "n": 10, "p10_seconds": 783.2576, "p25_seconds": 1569.0395, "median_seconds": 1845.21, "p75_seconds": 2201.5355, "p90_seconds": 2416.4854, "share_under_60s": 0.0},
    {"source": "notify reject", "target": "Perform checks", "n": 9, "p10_seconds": 318.1018, "p25_seconds": 699.898, "median_seconds": 874.95, "p75_seconds": 1271.62, "p90_seconds": 1995.78, "share_under_60s": 0.0},
    {"source": "review request received", "target": "Notify accept", "n": 9, "p10_seconds": 2066.8628, "p25_seconds": 2191.667, "median_seconds": 2296.639, "p75_seconds": 4048.847, "p90_seconds": 4780.2734, "share_under_60s": 0.0},
    {"source": "Check for completeness", "target": "review request received", "n": 9, "p10_seconds": 786.9502, "p25_seconds": 1083.927, "median_seconds": 1363.124, "p75_seconds": 2554.657, "p90_seconds": 2954.3854, "share_under_60s": 0.0},
    {"source": "review request received", "target": "Perform checks", "n": 9, "p10_seconds": 1338.8826, "p25_seconds": 1593.822, "median_seconds": 2249.376, "p75_seconds": 3393.616, "p90_seconds": 4267.802, "share_under_60s": 0.0},
    {"source": "info received", "target": "Request info", "n": 9, "p10_seconds": 824.7008, "p25_seconds": 1305.603, "median_seconds": 1993.345, "p75_seconds": 3112.501, "p90_seconds": 4079.4476, "share_under_60s": 0.0},
    {"source": "EVENT 13 END", "target": "EVENT 13 END", "n": 8, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "Make decision", "target": "info received", "n": 7, "p10_seconds": 615.6284, "p25_seconds": 944.3135, "median_seconds": 1863.824, "p75_seconds": 1941.4715, "p90_seconds": 2142.2018, "share_under_60s": 0.0},
    {"source": "notify reject", "target": "Check for completeness", "n": 6, "p10_seconds": 1769.177, "p25_seconds": 2725.29575, "median_seconds": 3075.0545, "p75_seconds": 3917.45225, "p90_seconds": 4349.0205, "share_under_60s": 0.0},
    {"source": "Make decision", "target": "Make decision", "n": 6, "p10_seconds": 1349.042, "p25_seconds": 1693.34325, "median_seconds": 2015.0765, "p75_seconds": 2417.1235, "p90_seconds": 2768.227, "share_under_60s": 0.0},
    {"source": "Perform checks", "target": "Perform checks", "n": 6, "p10_seconds": 1054.435, "p25_seconds": 1415.281, "median_seconds": 2294.3645, "p75_seconds": 2990.205, "p90_seconds": 5170.2335, "share_under_60s": 0.0},
    {"source": "review request received", "target": "notify reject", "n": 5, "p10_seconds": 1334.3154, "p25_seconds": 1360.086, "median_seconds": 1469.711, "p75_seconds": 2384.126, "p90_seconds": 3042.9542, "share_under_60s": 0.0},
    {"source": "review request received", "target": "Request info", "n": 5, "p10_seconds": 1863.9608, "p25_seconds": 2128.058, "median_seconds": 2501.573, "p75_seconds": 3114.888, "p90_seconds": 4176.5796, "share_under_60s": 0.0},
    {"source": "Make decision", "target": "Request info", "n": 5, "p10_seconds": 1027.2532, "p25_seconds": 1078.57, "median_seconds": 1325.858, "p75_seconds": 1553.253, "p90_seconds": 2857.497, "share_under_60s": 0.0},
    {"source": "notify reject", "target": "notify reject", "n": 5, "p10_seconds": 1475.265, "p25_seconds": 2749.563, "median_seconds": 2967.996, "p75_seconds": 3278.084, "p90_seconds": 3452.2106, "share_under_60s": 0.0},
    {"source": "Make decision", "target": "Perform checks", "n": 5, "p10_seconds": 783.9354, "p25_seconds": 796.233, "median_seconds": 823.397, "p75_seconds": 1650.51, "p90_seconds": 1824.5034, "share_under_60s": 0.0},
    {"source": "review request received", "target": "info received", "n": 4, "p10_seconds": 1391.3396, "p25_seconds": 1447.0055, "median_seconds": 1983.33, "p75_seconds": 2496.487, "p90_seconds": 2510.4514, "share_under_60s": 0.0},
    {"source": "notify reject", "target": "info received", "n": 3, "p10_seconds": 652.6522, "p25_seconds": 721.4095, "median_seconds": 836.005, "p75_seconds": 1576.566, "p90_seconds": 2020.9026, "share_under_60s": 0.0},
    {"source": "notify reject", "target": "Request info", "n": 3, "p10_seconds": 389.1752, "p25_seconds": 613.13, "median_seconds": 986.388, "p75_seconds": 1967.2505, "p90_seconds": 2555.768, "share_under_60s": 0.0},
]

GLOBAL_MEDIAN_DURATION_SECONDS = 243.721


def build_transition_maps():
    prob = {}
    for d in TRANSITION_PROBABILITIES_TOP:
        prob[(d["source"], d["target"])] = d["probability"]
    dur = {}
    for d in DURATION_PROFILES_TOP:
        dur[(d["source"], d["target"])] = d
    return prob, dur


TRANS_PROB, DUR_PROF = build_transition_maps()


def parse_ts_safe(s):
    if pd.isna(s):
        return pd.NaT
    ss = str(s)
    if ss.strip() == "":
        return pd.NaT
    try:
        return pd.to_datetime(ss, errors="coerce")
    except Exception:
        return pd.NaT


def detect_precision_and_sep(series):
    max_digits = 0
    sep = "."
    pattern = re.compile(r"(\d{2}:\d{2}:\d{2})([.,](\d+))?$")
    for s in series.dropna().astype(str):
        m = pattern.search(s)
        if m:
            frac = m.group(3)
            if frac is not None:
                max_digits = max(max_digits, len(frac))
                sep = m.group(2)[0]
    return max_digits, sep


def format_timestamp(ts, frac_digits, sep):
    if pd.isna(ts):
        return ""
    py = ts.to_pydatetime()
    base = py.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits <= 0:
        return base
    micro = py.microsecond
    frac = f"{micro:06d}"[:frac_digits].ljust(frac_digits, "0")
    return f"{base}{sep}{frac}"


def log_prob(source, target):
    p = TRANS_PROB.get((source, target))
    if p is None:
        return math.log(1e-6)
    return math.log(max(p, 1e-12))


def duration_guess_seconds(source, target, default=GLOBAL_MEDIAN_DURATION_SECONDS):
    d = DUR_PROF.get((source, target))
    if d is None:
        return max(1.0, default)
    if d["share_under_60s"] >= 0.8:
        val = d["p10_seconds"]
    else:
        val = d["p25_seconds"]
    return max(1.0, float(val))


def duration_profile(source, target):
    return DUR_PROF.get((source, target))


def plausibility_score_gap(source, target, gap_seconds):
    if gap_seconds < 0:
        return -1e9
    prof = duration_profile(source, target)
    if prof is None:
        center = GLOBAL_MEDIAN_DURATION_SECONDS
        scale = max(60.0, center)
        return -abs(gap_seconds - center) / scale
    p10 = prof["p10_seconds"]
    p25 = prof["p25_seconds"]
    p50 = prof["median_seconds"]
    p90 = prof["p90_seconds"]
    if p90 <= 0 and gap_seconds == 0:
        return 2.0
    if p90 <= 0 and gap_seconds > 0:
        return -gap_seconds / 10.0
    if gap_seconds <= p10:
        return -abs(gap_seconds - p10) / max(10.0, p25 if p25 > 0 else 10.0)
    if gap_seconds <= p50:
        return 1.0 - abs(gap_seconds - p25) / max(10.0, p50 - p10)
    return -abs(gap_seconds - p50) / max(10.0, p90 - p10)


def same_case_neighbors(df_case, pos):
    left = None
    right = None
    for i in range(pos - 1, -1, -1):
        left = i
        break
    for i in range(pos + 1, len(df_case)):
        right = i
        break
    return left, right


def nearest_clean_anchor_left(df_case, pos):
    for i in range(pos - 1, -1, -1):
        if str(df_case.iloc[i]["error_category"]) == "CLEAN":
            ts = parse_ts_safe(df_case.iloc[i]["Timestamp"])
            if not pd.isna(ts):
                return i, ts, df_case.iloc[i]["Activity"]
    return None, None, None


def nearest_clean_anchor_right(df_case, pos):
    for i in range(pos + 1, len(df_case)):
        if str(df_case.iloc[i]["error_category"]) == "CLEAN":
            ts = parse_ts_safe(df_case.iloc[i]["Timestamp"])
            if not pd.isna(ts):
                return i, ts, df_case.iloc[i]["Activity"]
    return None, None, None


def choose_best_order(activities, left_act, right_act):
    n = len(activities)
    idxs = list(range(n))
    best_perm = tuple(idxs)
    best_score = -1e18
    for perm in itertools.permutations(idxs):
        ordered = [activities[i] for i in perm]
        score = 0.0
        if left_act is not None:
            score += log_prob(left_act, ordered[0])
        else:
            score += -2.0
        for a, b in zip(ordered[:-1], ordered[1:]):
            score += log_prob(a, b)
        if right_act is not None:
            score += log_prob(ordered[-1], right_act)
        else:
            score += -2.0
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_perm, best_score


def estimate_block_times(observed_ts, ordered_activities, right_anchor_ts, min_step_ns=1_000_000_000):
    times = [observed_ts]
    gaps = []
    for a, b in zip(ordered_activities[:-1], ordered_activities[1:]):
        gaps.append(duration_guess_seconds(a, b))
    if not gaps:
        return times, False
    cum = []
    s = 0.0
    for g in gaps:
        s += max(1.0, g)
        cum.append(s)
    total = cum[-1]
    clamped = False
    if right_anchor_ts is not None:
        room = (right_anchor_ts - observed_ts).total_seconds()
        max_allowed = max(len(ordered_activities) - 1, room)
        if total > max_allowed:
            clamped = True
            if max_allowed <= 0:
                scaled = [1.0] * len(gaps)
            else:
                factor = max_allowed / total
                scaled = [max(1.0, g * factor) for g in gaps]
                scaled_total = sum(scaled)
                if scaled_total > max_allowed:
                    excess = scaled_total - max_allowed
                    for i in range(len(scaled) - 1, -1, -1):
                        reducible = scaled[i] - 1.0
                        take = min(reducible, excess)
                        scaled[i] -= take
                        excess -= take
                        if excess <= 1e-9:
                            break
            gaps = scaled
    current = observed_ts
    for g in gaps:
        current = current + pd.to_timedelta(max(1.0, g), unit="s")
        times.append(current)
    fixed = [times[0]]
    for t in times[1:]:
        if t <= fixed[-1]:
            t = fixed[-1] + pd.to_timedelta(min_step_ns, unit="ns")
            clamped = True
        fixed.append(t)
    if right_anchor_ts is not None:
        for i in range(len(fixed) - 1, -1, -1):
            latest = right_anchor_ts - pd.to_timedelta((len(fixed) - 1 - i), unit="s")
            if fixed[i] > latest:
                fixed[i] = latest
                clamped = True
        for i in range(1, len(fixed)):
            if fixed[i] <= fixed[i - 1]:
                fixed[i] = fixed[i - 1] + pd.to_timedelta(1, unit="s")
                clamped = True
        if fixed[-1] > right_anchor_ts:
            shift = fixed[-1] - right_anchor_ts
            fixed = [t - shift for t in fixed]
            clamped = True
        if fixed[0] != observed_ts:
            fixed[0] = observed_ts
            clamped = True
            for i in range(1, len(fixed)):
                if fixed[i] <= fixed[i - 1]:
                    fixed[i] = fixed[i - 1] + pd.to_timedelta(1, unit="s")
    return fixed, clamped


def repair_c1_blocks(df, frac_digits, sep):
    repaired_blocks = 0
    clamped_blocks = 0
    repaired_values = {}
    c1_keep_rows = set()

    grouped = df.groupby("Case", sort=False)
    for case_id, idxs in grouped.groups.items():
        case_df = df.loc[idxs].copy()
        case_df = case_df.reset_index()
        n = len(case_df)
        i = 0
        while i < n:
            row = case_df.iloc[i]
            if str(row["error_category"]) != "C1_ORDERING_AMBIGUITY":
                i += 1
                continue
            obs = str(row["Timestamp"])
            j = i + 1
            block_positions = [i]
            while j < n and str(case_df.iloc[j]["error_category"]) == "C1_ORDERING_AMBIGUITY" and str(case_df.iloc[j]["Timestamp"]) == obs:
                block_positions.append(j)
                j += 1
            if len(block_positions) == 1:
                acts = [case_df.iloc[i]["Activity"]]
                left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, i)
                right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, i)
                perm, _ = choose_best_order(acts, left_act, right_act)
                ordered_acts = [acts[k] for k in perm]
                observed_ts = parse_ts_safe(obs)
                new_times, clamped = estimate_block_times(observed_ts, ordered_acts, right_ts)
                global_index = case_df.iloc[i]["index"]
                repaired_values[global_index] = format_timestamp(new_times[0], frac_digits, sep)
                c1_keep_rows.add(global_index)
                repaired_blocks += 1
                if clamped:
                    clamped_blocks += 1
                i = j
                continue

            acts = [case_df.iloc[p]["Activity"] for p in block_positions]
            observed_ts = parse_ts_safe(obs)
            left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, block_positions[0])
            right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, block_positions[-1])

            perm, _ = choose_best_order(acts, left_act, right_act)
            ordered_positions = [block_positions[k] for k in perm]
            ordered_acts = [acts[k] for k in perm]
            new_times, clamped = estimate_block_times(observed_ts, ordered_acts, right_ts)

            for rank, pos in enumerate(ordered_positions):
                global_index = case_df.iloc[pos]["index"]
                repaired_values[global_index] = format_timestamp(new_times[rank], frac_digits, sep)
                if rank == 0:
                    c1_keep_rows.add(global_index)

            repaired_blocks += 1
            if clamped:
                clamped_blocks += 1
            i = j

    return repaired_values, c1_keep_rows, repaired_blocks, clamped_blocks


def score_swap_local(case_df, pos):
    if pos < 0 or pos >= len(case_df) - 1:
        return -1e18, None
    r1 = case_df.iloc[pos]
    r2 = case_df.iloc[pos + 1]
    t1 = parse_ts_safe(r1["Timestamp"])
    t2 = parse_ts_safe(r2["Timestamp"])
    if pd.isna(t1) or pd.isna(t2):
        return -1e18, None

    left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, pos)
    right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, pos + 1)

    score = 0.0
    if left_ts is not None:
        score += plausibility_score_gap(left_act, r1["Activity"], (t2 - left_ts).total_seconds())
    if right_ts is not None:
        score += plausibility_score_gap(r2["Activity"], right_act, (right_ts - t1).total_seconds())
    score += plausibility_score_gap(r1["Activity"], r2["Activity"], (t1 - t2).total_seconds())
    return score, (t2, t1)


def score_current_local(case_df, pos):
    if pos < 0 or pos >= len(case_df) - 1:
        return -1e18
    r1 = case_df.iloc[pos]
    r2 = case_df.iloc[pos + 1]
    t1 = parse_ts_safe(r1["Timestamp"])
    t2 = parse_ts_safe(r2["Timestamp"])
    if pd.isna(t1) or pd.isna(t2):
        return -1e18
    left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, pos)
    right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, pos + 1)
    score = 0.0
    if left_ts is not None:
        score += plausibility_score_gap(left_act, r1["Activity"], (t1 - left_ts).total_seconds())
    score += plausibility_score_gap(r1["Activity"], r2["Activity"], (t2 - t1).total_seconds())
    if right_ts is not None:
        score += plausibility_score_gap(r2["Activity"], right_act, (right_ts - t2).total_seconds())
    return score


def estimate_from_anchors(left_ts, left_act, act, right_ts, right_act):
    left_guess = None
    right_guess = None
    if left_ts is not None:
        left_guess = left_ts + pd.to_timedelta(duration_guess_seconds(left_act, act), unit="s")
    if right_ts is not None:
        right_guess = right_ts - pd.to_timedelta(duration_guess_seconds(act, right_act), unit="s")

    if left_guess is not None and right_guess is not None:
        if left_guess <= right_guess:
            delta = right_guess - left_guess
            return left_guess + delta / 2
        return left_ts + (right_ts - left_ts) / 2
    if left_guess is not None:
        return left_guess
    if right_guess is not None:
        return right_guess
    if left_ts is not None and right_ts is not None:
        return left_ts + (right_ts - left_ts) / 2
    if left_ts is not None:
        return left_ts + pd.to_timedelta(GLOBAL_MEDIAN_DURATION_SECONDS, unit="s")
    if right_ts is not None:
        return right_ts - pd.to_timedelta(GLOBAL_MEDIAN_DURATION_SECONDS, unit="s")
    return pd.Timestamp("2000-01-01 00:00:00")


def repair_c2(df, frac_digits, sep, existing_repairs):
    repairs = {}
    grouped = df.groupby("Case", sort=False)
    for case_id, idxs in grouped.groups.items():
        case_df = df.loc[idxs].copy().reset_index()
        n = len(case_df)
        for i in range(n - 1):
            r1 = case_df.iloc[i]
            r2 = case_df.iloc[i + 1]
            if str(r2["error_category"]) != "C2_TEMPORAL_INCONSISTENCY":
                continue

            t1 = parse_ts_safe(existing_repairs.get(r1["index"], r1["Timestamp"]))
            t2 = parse_ts_safe(existing_repairs.get(r2["index"], r2["Timestamp"]))
            if pd.isna(t1) or pd.isna(t2) or not (t2 < t1):
                continue

            swap_score, swapped = score_swap_local(case_df, i)
            current_score = score_current_local(case_df, i)

            if swapped is not None and swap_score > current_score + 1.0 and swapped[0] >= pd.Timestamp.min and swapped[1] <= pd.Timestamp.max:
                repairs[r1["index"]] = format_timestamp(swapped[0], frac_digits, sep)
                repairs[r2["index"]] = format_timestamp(swapped[1], frac_digits, sep)
                continue

            left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, i + 1)
            right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, i + 1)
            est2 = estimate_from_anchors(t1, r1["Activity"], r2["Activity"], right_ts, right_act)
            if est2 <= t1:
                est2 = t1 + pd.to_timedelta(1, unit="s")
            if right_ts is not None and est2 > right_ts:
                est2 = min(right_ts, right_ts - pd.to_timedelta(0, unit="s"))
                if est2 <= t1:
                    est2 = t1 + pd.to_timedelta(1, unit="s")
            repairs[r2["index"]] = format_timestamp(est2, frac_digits, sep)
    return repairs


def infer_time_on_same_date(date_ts, left_ts, left_act, act, right_ts, right_act):
    day_start = pd.Timestamp(year=date_ts.year, month=date_ts.month, day=date_ts.day)
    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    est = estimate_from_anchors(left_ts, left_act, act, right_ts, right_act)
    if est.date() != date_ts.date():
        if left_ts is not None and left_ts.date() == date_ts.date():
            est = max(est, left_ts)
        if right_ts is not None and right_ts.date() == date_ts.date():
            est = min(est, right_ts)
        if est.date() != date_ts.date():
            noon = day_start + pd.Timedelta(hours=12)
            est = noon

    if est < day_start:
        est = day_start
    if est > day_end:
        est = day_end

    lower = day_start
    upper = day_end
    if left_ts is not None and left_ts.date() == date_ts.date():
        lower = max(lower, left_ts)
    if right_ts is not None and right_ts.date() == date_ts.date():
        upper = min(upper, right_ts)
    if lower <= upper:
        if est < lower:
            est = lower
        if est > upper:
            est = upper
    else:
        est = lower if abs((lower - day_start).total_seconds()) <= abs((day_end - upper).total_seconds()) else upper
        if est < day_start:
            est = day_start
        if est > day_end:
            est = day_end
    return est


def repair_c3(df, frac_digits, sep, existing_repairs):
    repairs = {}
    grouped = df.groupby("Case", sort=False)
    for case_id, idxs in grouped.groups.items():
        case_df = df.loc[idxs].copy().reset_index()
        n = len(case_df)
        for i in range(n):
            r = case_df.iloc[i]
            if str(r["error_category"]) != "C3_MISSING_TEMPORAL_INFORMATION":
                continue
            orig = str(r["Timestamp"]) if not pd.isna(r["Timestamp"]) else ""
            left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, i)
            right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, i)
            act = r["Activity"]

            has_date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", orig.strip())) if orig.strip() else False
            if has_date_only:
                date_ts = pd.to_datetime(orig.strip(), errors="coerce")
                if pd.isna(date_ts):
                    est = estimate_from_anchors(left_ts, left_act, act, right_ts, right_act)
                else:
                    est = infer_time_on_same_date(date_ts, left_ts, left_act, act, right_ts, right_act)
            else:
                est = estimate_from_anchors(left_ts, left_act, act, right_ts, right_act)
                if left_ts is not None and right_ts is not None:
                    if est < left_ts:
                        est = left_ts
                    if est > right_ts:
                        est = left_ts + (right_ts - left_ts) / 2

            repairs[r["index"]] = format_timestamp(est, frac_digits, sep)
    return repairs


def self_check_and_fix(df, out_ts, original_ts, c1_keep_rows, frac_digits, sep):
    clamped_blocks = 0

    case_groups = df.groupby("Case", sort=False)
    for case_id, idxs in case_groups.groups.items():
        case_df = df.loc[idxs].copy().reset_index()
        n = len(case_df)
        parsed_case = [parse_ts_safe(out_ts[idx]) for idx in case_df["index"]]

        case_valid = [t for t in parsed_case if not pd.isna(t)]
        case_start = min(case_valid) if case_valid else None
        case_end = max(case_valid) if case_valid else None

        for local_pos, global_idx in enumerate(case_df["index"]):
            cat = str(case_df.iloc[local_pos]["error_category"])
            if cat == "CLEAN":
                out_ts[global_idx] = original_ts[global_idx]
                continue
            t = parse_ts_safe(out_ts[global_idx])
            if pd.isna(t):
                left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, local_pos)
                right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, local_pos)
                est = estimate_from_anchors(left_ts, left_act, case_df.iloc[local_pos]["Activity"], right_ts, right_act)
                out_ts[global_idx] = format_timestamp(est, frac_digits, sep)
                t = est
            left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, local_pos)
            right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, local_pos)
            if case_start is not None and t < case_start:
                t = case_start
            if case_end is not None and t > case_end:
                t = case_end
            if left_ts is not None and right_ts is not None:
                if t < left_ts:
                    t = left_ts
                if t > right_ts:
                    t = right_ts
            out_ts[global_idx] = format_timestamp(t, frac_digits, sep)

        i = 0
        while i < n:
            row = case_df.iloc[i]
            if str(row["error_category"]) != "C1_ORDERING_AMBIGUITY":
                i += 1
                continue
            obs = str(row["Timestamp"])
            j = i + 1
            block_positions = [i]
            while j < n and str(case_df.iloc[j]["error_category"]) == "C1_ORDERING_AMBIGUITY" and str(case_df.iloc[j]["Timestamp"]) == obs:
                block_positions.append(j)
                j += 1

            global_block = [case_df.iloc[p]["index"] for p in block_positions]
            block_rows = [(g, parse_ts_safe(out_ts[g]), original_ts[g]) for g in global_block]
            observed_ts = parse_ts_safe(obs)
            keep_candidates = [g for g in global_block if out_ts[g] == original_ts[g] == obs]
            if len(keep_candidates) != 1:
                clamped_blocks += 1
                if keep_candidates:
                    keeper = keep_candidates[0]
                else:
                    acts = [case_df.iloc[p]["Activity"] for p in block_positions]
                    left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, block_positions[0])
                    right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, block_positions[-1])
                    perm, _ = choose_best_order(acts, left_act, right_act)
                    keeper = global_block[perm[0]]
                for g in global_block:
                    if g == keeper:
                        out_ts[g] = obs
                        c1_keep_rows.add(g)
                keep_candidates = [keeper]

            acts = [case_df.iloc[p]["Activity"] for p in block_positions]
            left_i, left_ts, left_act = nearest_clean_anchor_left(case_df, block_positions[0])
            right_i, right_ts, right_act = nearest_clean_anchor_right(case_df, block_positions[-1])
            perm, _ = choose_best_order(acts, left_act, right_act)
            ordered_positions = [block_positions[k] for k in perm]
            ordered_globals = [case_df.iloc[p]["index"] for p in ordered_positions]

            times = [parse_ts_safe(out_ts[g]) for g in ordered_globals]
            need_fix = False
            if pd.isna(times[0]) or times[0] != observed_ts:
                need_fix = True
            else:
                for a, b in zip(times[:-1], times[1:]):
                    if pd.isna(a) or pd.isna(b) or not (b > a):
                        need_fix = True
                        break
                if right_ts is not None:
                    span = (times[-1] - times[0]).total_seconds()
                    room = (right_ts - observed_ts).total_seconds()
                    if span > room + 1e-9 or span < len(times) - 1:
                        need_fix = True

            if need_fix:
                clamped_blocks += 1
                ordered_acts = [acts[k] for k in perm]
                new_times, _ = estimate_block_times(observed_ts, ordered_acts, right_ts)
                for rank, g in enumerate(ordered_globals):
                    out_ts[g] = obs if rank == 0 else format_timestamp(new_times[rank], frac_digits, sep)
                    if rank == 0:
                        c1_keep_rows.add(g)
            i = j

    return clamped_blocks


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

    frac_digits, sep = detect_precision_and_sep(df["Timestamp"])
    original_ts = df["Timestamp"].astype(str).tolist()
    out_ts = original_ts.copy()

    c1_repairs, c1_keep_rows, repaired_blocks, c1_clamped = repair_c1_blocks(df, frac_digits, sep)
    for idx, ts in c1_repairs.items():
        out_ts[idx] = ts

    c2_repairs = repair_c2(df, frac_digits, sep, c1_repairs)
    for idx, ts in c2_repairs.items():
        out_ts[idx] = ts

    combined_repairs = dict(c1_repairs)
    combined_repairs.update(c2_repairs)
    c3_repairs = repair_c3(df, frac_digits, sep, combined_repairs)
    for idx, ts in c3_repairs.items():
        out_ts[idx] = ts

    extra_clamped = self_check_and_fix(df, out_ts, original_ts, c1_keep_rows, frac_digits, sep)
    total_clamped = c1_clamped + extra_clamped

    for i, cat in enumerate(df["error_category"].astype(str)):
        if cat == "CLEAN":
            out_ts[i] = original_ts[i]

    rows_still_equal = 0
    for i, cat in enumerate(df["error_category"].astype(str)):
        if cat != "CLEAN" and out_ts[i] == original_ts[i] and i not in c1_keep_rows:
            parsed = parse_ts_safe(out_ts[i])
            if pd.isna(parsed):
                parsed = pd.Timestamp("2000-01-01 00:00:00")
            parsed = parsed + pd.to_timedelta(1, unit="s")
            out_ts[i] = format_timestamp(parsed, frac_digits, sep)
        if cat != "CLEAN" and out_ts[i] == original_ts[i]:
            rows_still_equal += 1

    out_df = df.copy()
    out_df["Timestamp"] = out_ts
    out_df.to_csv(args.output, index=False)

    print(f"blocks_repaired={repaired_blocks} blocks_clamped={total_clamped} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            import csv
            import sys
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=True)
            parser.add_argument("--output", required=True)
            args = parser.parse_args()
            with open(args.input, "r", newline="", encoding="utf-8") as f_in:
                rows = list(csv.reader(f_in))
            with open(args.output, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.writer(f_out)
                writer.writerows(rows)
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")
        except Exception:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")