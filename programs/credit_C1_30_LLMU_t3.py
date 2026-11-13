import argparse
import math
import itertools
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


# -----------------------------
# Process context embedded
# -----------------------------
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

TPROBS = {(d["source"], d["target"]): d["probability"] for d in TRANSITION_PROBABILITIES_TOP}
DPROF = {(d["source"], d["target"]): d for d in DURATION_PROFILES_TOP}


# -----------------------------
# Utility
# -----------------------------
def parse_ts(s):
    if pd.isna(s):
        return pd.NaT
    s2 = str(s)
    if s2.strip() == "":
        return pd.NaT
    return pd.to_datetime(s2, errors="coerce")


def frac_digits(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0
    s = str(s)
    if "." in s:
        return len(s.split(".")[-1])
    return 0


def date_only_string(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return False
    s = str(s).strip()
    return len(s) == 10 and s.count("-") == 2


def fmt_ts(ts, digits):
    if pd.isna(ts):
        return ""
    ts = pd.Timestamp(ts)
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if digits <= 0:
        return base
    frac = f"{ts.microsecond:06d}"[:digits].ljust(digits, "0")
    return base + "." + frac


def trans_prob(a, b):
    return TPROBS.get((a, b), 1e-6)


def duration_stat(a, b):
    d = DPROF.get((a, b))
    if d:
        if d["share_under_60s"] >= 0.9:
            return max(1.0, d["p10_seconds"])
        return max(1.0, d["p25_seconds"])
    p = trans_prob(a, b)
    if p > 0.4:
        return 10.0
    if p > 0.05:
        return 120.0
    return max(60.0, min(180.0, GLOBAL_MEDIAN_DURATION_SECONDS))


def duration_bounds(a, b):
    d = DPROF.get((a, b))
    if d:
        lo = max(0.0, d["p10_seconds"] * 0.25 if d["p10_seconds"] > 0 else 0.0)
        hi = max(d["p90_seconds"] * 2.5, d["p25_seconds"] + 1.0)
        if d["share_under_60s"] >= 0.9:
            hi = max(120.0, d["p90_seconds"] * 5 + 1)
        return lo, hi
    est = duration_stat(a, b)
    return 0.0, max(est * 5, 300.0)


def score_order(order, left_act, right_act):
    acts = list(order)
    score = 0.0
    if left_act is not None:
        score += math.log(trans_prob(left_act, acts[0]))
    for i in range(len(acts) - 1):
        score += math.log(trans_prob(acts[i], acts[i + 1]))
    if right_act is not None:
        score += math.log(trans_prob(acts[-1], right_act))
    return score


def best_order(rows, left_act, right_act):
    acts = [r["Activity"] for r in rows]
    idxs = list(range(len(rows)))

    if len(rows) <= 7:
        best = None
        best_s = -1e100
        for perm in itertools.permutations(idxs):
            order = [acts[i] for i in perm]
            s = score_order(order, left_act, right_act)
            if s > best_s:
                best_s = s
                best = perm
        return list(best)

    beam = [([], set(), 0.0)]
    beam_width = 40
    for _ in range(len(rows)):
        new_beam = []
        for seq, used, _sc in beam:
            for i in idxs:
                if i in used:
                    continue
                seq2 = seq + [i]
                used2 = set(used)
                used2.add(i)
                order = [acts[j] for j in seq2]
                sc = 0.0
                if left_act is not None:
                    sc += math.log(trans_prob(left_act, order[0]))
                for k in range(len(order) - 1):
                    sc += math.log(trans_prob(order[k], order[k + 1]))
                remaining = len(rows) - len(seq2)
                if remaining == 0 and right_act is not None:
                    sc += math.log(trans_prob(order[-1], right_act))
                new_beam.append((seq2, used2, sc))
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]
    return beam[0][0]


def enforce_within(ts_list, left_bound, right_bound):
    out = list(ts_list)
    n = len(out)
    if n == 0:
        return out, False
    changed = False
    if left_bound is not None and out[0] < left_bound:
        delta = left_bound - out[0]
        out = [t + delta for t in out]
        changed = True
    if right_bound is not None and out[-1] > right_bound:
        room = (right_bound - out[0]).total_seconds()
        min_needed = n - 1
        if room < min_needed:
            start = right_bound - pd.Timedelta(seconds=min_needed)
            out = [start + pd.Timedelta(seconds=i) for i in range(n)]
            changed = True
        else:
            gaps = [(out[i] - out[i - 1]).total_seconds() for i in range(1, n)]
            total = sum(gaps)
            target = room
            if total <= 0:
                out = [out[0] + pd.Timedelta(seconds=i) for i in range(n)]
            else:
                surplus = target - (n - 1)
                if surplus < 0:
                    surplus = 0
                weights = [max(0.0, g - 1.0) for g in gaps]
                wsum = sum(weights)
                new_gaps = []
                if wsum <= 0:
                    extra_each = surplus / len(gaps) if gaps else 0
                    new_gaps = [1.0 + extra_each for _ in gaps]
                else:
                    new_gaps = [1.0 + surplus * w / wsum for w in weights]
                rebuilt = [out[0]]
                cur = out[0]
                for g in new_gaps:
                    cur = cur + pd.Timedelta(seconds=float(g))
                    rebuilt.append(cur)
                out = rebuilt
            changed = True
    for i in range(1, n):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + pd.Timedelta(seconds=1)
            changed = True
    if right_bound is not None and out[-1] > right_bound:
        start = right_bound - pd.Timedelta(seconds=n - 1)
        if left_bound is not None and start < left_bound:
            start = left_bound
        out = [start + pd.Timedelta(seconds=i) for i in range(n)]
        changed = True
    return out, changed


def local_negative_violations(case_rows, ts_map):
    cnt = 0
    seq = sorted(case_rows, key=lambda x: x["__pos"])
    prev = None
    for r in seq:
        t = ts_map[r["__pos"]]
        if prev is not None and t < prev:
            cnt += 1
        prev = t
    return cnt


def estimate_inside(left_t, right_t, left_act, act, right_act, keep_date=None):
    cand = None
    if left_t is not None and right_t is not None and right_t >= left_t:
        dl = duration_stat(left_act, act) if left_act is not None else None
        dr = duration_stat(act, right_act) if right_act is not None else None
        cands = []
        if dl is not None:
            cands.append(left_t + pd.Timedelta(seconds=dl))
        if dr is not None:
            cands.append(right_t - pd.Timedelta(seconds=dr))
        if cands:
            cand = cands[0] if len(cands) == 1 else cands[0] + (cands[1] - cands[0]) / 2
        else:
            cand = left_t + (right_t - left_t) / 2
        if cand < left_t:
            cand = left_t
        if cand > right_t:
            cand = left_t + (right_t - left_t) / 2
    elif left_t is not None:
        dl = duration_stat(left_act, act) if left_act is not None else GLOBAL_MEDIAN_DURATION_SECONDS
        cand = left_t + pd.Timedelta(seconds=dl)
    elif right_t is not None:
        dr = duration_stat(act, right_act) if right_act is not None else GLOBAL_MEDIAN_DURATION_SECONDS
        cand = right_t - pd.Timedelta(seconds=dr)
    else:
        cand = pd.Timestamp.now().normalize() + pd.Timedelta(hours=12)

    if keep_date is not None:
        d = pd.Timestamp(keep_date).date()
        cand = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                            hour=cand.hour, minute=cand.minute, second=cand.second,
                            microsecond=cand.microsecond)
    return cand


# -----------------------------
# Main repair routines
# -----------------------------
def repair_c1_block(df, case_idx_positions, start_ptr, end_ptr, out_ts, parsed_ts, stats):
    block_positions = case_idx_positions[start_ptr:end_ptr + 1]
    rows = []
    for pos in block_positions:
        rows.append({
            "__pos": pos,
            "Activity": df.at[pos, "Activity"],
            "obs_ts": parsed_ts[pos],
            "obs_str": df.at[pos, "Timestamp"],
            "digits": frac_digits(df.at[pos, "Timestamp"])
        })
    k = len(rows)
    shared_ts = rows[0]["obs_ts"]

    left_anchor_pos = case_idx_positions[start_ptr - 1] if start_ptr - 1 >= 0 else None
    right_anchor_pos = case_idx_positions[end_ptr + 1] if end_ptr + 1 < len(case_idx_positions) else None

    left_act = df.at[left_anchor_pos, "Activity"] if left_anchor_pos is not None else None
    right_act = df.at[right_anchor_pos, "Activity"] if right_anchor_pos is not None else None
    left_t = parsed_ts[left_anchor_pos] if left_anchor_pos is not None else None
    right_t = parsed_ts[right_anchor_pos] if right_anchor_pos is not None else None

    order_idx = best_order(rows, left_act, right_act)
    ordered_rows = [rows[i] for i in order_idx]

    gaps = []
    for i in range(k - 1):
        a = ordered_rows[i]["Activity"]
        b = ordered_rows[i + 1]["Activity"]
        gaps.append(max(1.0, duration_stat(a, b)))

    placed = [shared_ts]
    cur = shared_ts
    for g in gaps:
        cur = cur + pd.Timedelta(seconds=float(g))
        placed.append(cur)

    room_right = None
    if right_t is not None and not pd.isna(right_t):
        room_right = right_t
    placed, clamped = enforce_within(placed, shared_ts, room_right)

    if len(placed) != k:
        placed = [shared_ts + pd.Timedelta(seconds=i) for i in range(k)]
        if right_t is not None:
            placed, _ = enforce_within(placed, shared_ts, right_t)
        clamped = True

    for i in range(1, k):
        if placed[i] <= placed[i - 1]:
            placed[i] = placed[i - 1] + pd.Timedelta(seconds=1)
            clamped = True

    if right_t is not None and placed[-1] > right_t:
        start = min(shared_ts, right_t - pd.Timedelta(seconds=k - 1))
        if start < shared_ts:
            start = shared_ts
        placed = [start + pd.Timedelta(seconds=i) for i in range(k)]
        if placed[-1] > right_t:
            placed = [right_t - pd.Timedelta(seconds=(k - 1 - i)) for i in range(k)]
        clamped = True

    for i, r in enumerate(ordered_rows):
        pos = r["__pos"]
        out_ts[pos] = placed[i]

    stats["c1_blocks"] += 1
    if clamped:
        stats["clamped_blocks"] += 1


def swap_quality(prev_t, t1, t2, next_t, prev_act, act1, act2, next_act):
    penalty = 0.0
    if prev_t is not None:
        d = (t1 - prev_t).total_seconds()
        if d < 0:
            return 1e18
        lo, hi = duration_bounds(prev_act, act1)
        if d < lo:
            penalty += (lo - d) * 5
        elif d > hi:
            penalty += (d - hi)
    d12 = (t2 - t1).total_seconds()
    if d12 < 0:
        return 1e18
    lo, hi = duration_bounds(act1, act2)
    if d12 < lo:
        penalty += (lo - d12) * 5
    elif d12 > hi:
        penalty += (d12 - hi)
    if next_t is not None:
        d = (next_t - t2).total_seconds()
        if d < 0:
            return 1e18
        lo, hi = duration_bounds(act2, next_act)
        if d < lo:
            penalty += (lo - d) * 5
        elif d > hi:
            penalty += (d - hi)
    penalty -= 20.0 * math.log(trans_prob(act1, act2))
    return penalty


def repair_c2_pair(df, case_positions, idx_in_case, out_ts, parsed_ts, stats):
    p1 = case_positions[idx_in_case]
    p2 = case_positions[idx_in_case + 1]

    prev_p = case_positions[idx_in_case - 1] if idx_in_case - 1 >= 0 else None
    next_p = case_positions[idx_in_case + 2] if idx_in_case + 2 < len(case_positions) else None

    t1 = out_ts[p1]
    t2 = out_ts[p2]
    prev_t = out_ts[prev_p] if prev_p is not None else None
    next_t = out_ts[next_p] if next_p is not None else None

    a_prev = df.at[prev_p, "Activity"] if prev_p is not None else None
    a1 = df.at[p1, "Activity"]
    a2 = df.at[p2, "Activity"]
    a_next = df.at[next_p, "Activity"] if next_p is not None else None

    obs1 = parsed_ts[p1]
    obs2 = parsed_ts[p2]

    swap_pen = swap_quality(prev_t, obs2, obs1, next_t, a_prev, a1, a2, a_next)
    keep_sorted_pen = swap_quality(prev_t, min(obs1, obs2), max(obs1, obs2), next_t, a_prev, a1, a2, a_next)

    accepted_swap = False
    if obs2 <= obs1 and swap_pen <= keep_sorted_pen + 30:
        if (prev_t is None or obs2 >= prev_t) and (next_t is None or obs1 <= next_t) and obs2 <= obs1:
            out_ts[p1] = obs2
            out_ts[p2] = obs1
            accepted_swap = True

    if accepted_swap:
        stats["c2_swaps"] += 1
        return

    cand1 = estimate_inside(prev_t, t2 if next_p is None or p2 is not None else next_t, a_prev, a1, a2)
    cand2 = estimate_inside(t1 if prev_p is not None else prev_t, next_t, a1, a2, a_next)

    c1 = cand1
    if prev_t is not None and c1 < prev_t:
        c1 = prev_t
    if c1 > t2:
        c1 = t2 - pd.Timedelta(seconds=1)

    c2 = cand2
    if c2 < t1:
        c2 = t1 + pd.Timedelta(seconds=1)
    if next_t is not None and c2 > next_t:
        c2 = next_t

    move1 = abs((c1 - obs1).total_seconds()) if not pd.isna(obs1) else 1e12
    move2 = abs((c2 - obs2).total_seconds()) if not pd.isna(obs2) else 1e12

    base_neg = 1 if t2 < t1 else 0
    option1 = dict(p1=c1, p2=t2, cost=move1, moved="p1")
    option2 = dict(p1=t1, p2=c2, cost=move2, moved="p2")

    best = None
    for opt in [option1, option2]:
        tt1 = opt["p1"]
        tt2 = opt["p2"]
        if tt2 < tt1:
            continue
        pen = swap_quality(prev_t, tt1, tt2, next_t, a_prev, a1, a2, a_next)
        total = opt["cost"] + pen
        if best is None or total < best[0]:
            best = (total, opt)

    if best is None:
        low = prev_t if prev_t is not None else min(obs1, obs2)
        high = next_t if next_t is not None else max(obs1, obs2) + pd.Timedelta(seconds=2)
        if high <= low:
            high = low + pd.Timedelta(seconds=2)
        new1 = low
        new2 = min(high, low + pd.Timedelta(seconds=1))
        if new2 < new1:
            new2 = new1 + pd.Timedelta(seconds=1)
        out_ts[p1] = new1
        out_ts[p2] = new2
    else:
        opt = best[1]
        out_ts[p1] = opt["p1"]
        out_ts[p2] = opt["p2"]
        stats["c2_displaced"] += 1


def repair_c3_row(df, case_positions, idx_in_case, out_ts, parsed_ts):
    p = case_positions[idx_in_case]
    prev_p = case_positions[idx_in_case - 1] if idx_in_case - 1 >= 0 else None
    next_p = case_positions[idx_in_case + 1] if idx_in_case + 1 < len(case_positions) else None

    left_t = out_ts[prev_p] if prev_p is not None else None
    right_t = out_ts[next_p] if next_p is not None else None
    left_act = df.at[prev_p, "Activity"] if prev_p is not None else None
    right_act = df.at[next_p, "Activity"] if next_p is not None else None
    act = df.at[p, "Activity"]

    orig = df.at[p, "Timestamp"]
    keep_date = str(orig).strip() if date_only_string(orig) else None
    cand = estimate_inside(left_t, right_t, left_act, act, right_act, keep_date=keep_date)

    if left_t is not None and cand < left_t:
        cand = left_t
    if right_t is not None and cand > right_t:
        cand = left_t + (right_t - left_t) / 2 if left_t is not None else right_t
    out_ts[p] = cand


def final_case_monotonic_pass(df, case_positions, out_ts, original_ts, nonclean_mask):
    n = len(case_positions)
    if n == 0:
        return
    # Forward enforce nondecreasing
    prev = None
    for j, p in enumerate(case_positions):
        t = out_ts[p]
        if prev is not None and t < prev:
            if nonclean_mask[p]:
                t = prev
                out_ts[p] = t
            else:
                # push prior corrupt rows back if possible
                k = j - 1
                anchor = t
                while k >= 0 and nonclean_mask[case_positions[k]] and out_ts[case_positions[k]] > anchor:
                    out_ts[case_positions[k]] = anchor
                    anchor = anchor - pd.Timedelta(seconds=1)
                    k -= 1
        prev = out_ts[p]

    # Backward strict fixes for equalities among repaired C1-derived or any repaired rows if needed
    for j in range(1, n):
        p_prev = case_positions[j - 1]
        p = case_positions[j]
        if out_ts[p] < out_ts[p_prev]:
            if nonclean_mask[p]:
                out_ts[p] = out_ts[p_prev]
            elif nonclean_mask[p_prev]:
                out_ts[p_prev] = out_ts[p]
    # second pass for any negatives remaining
    for j in range(1, n):
        p_prev = case_positions[j - 1]
        p = case_positions[j]
        if out_ts[p] < out_ts[p_prev]:
            if nonclean_mask[p]:
                out_ts[p] = out_ts[p_prev]
            else:
                out_ts[p_prev] = out_ts[p]


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
            if c == "Resource":
                df[c] = ""
            else:
                df[c] = ""

    df["__pos"] = np.arange(len(df))
    original_timestamp_strings = df["Timestamp"].astype(str).tolist()
    parsed_ts = [parse_ts(x) for x in original_timestamp_strings]
    out_ts = list(parsed_ts)

    nonclean_mask = [str(ec) != "CLEAN" for ec in df["error_category"].tolist()]
    digits_mode = 3
    all_digits = [frac_digits(s) for s in original_timestamp_strings if str(s).strip() != ""]
    if all_digits:
        digits_mode = Counter(all_digits).most_common(1)[0][0]

    stats = {"c1_blocks": 0, "clamped_blocks": 0, "c2_swaps": 0, "c2_displaced": 0}

    case_to_positions = defaultdict(list)
    for pos, case in enumerate(df["Case"].tolist()):
        case_to_positions[case].append(pos)

    # First pass: C1/C3 and local C2
    for case, positions in case_to_positions.items():
        i = 0
        while i < len(positions):
            p = positions[i]
            ec = df.at[p, "error_category"]
            if ec == "C1_ORDERING_AMBIGUITY":
                j = i
                ts0 = df.at[positions[i], "Timestamp"]
                while j + 1 < len(positions):
                    pj = positions[j + 1]
                    if df.at[pj, "error_category"] == "C1_ORDERING_AMBIGUITY" and df.at[pj, "Timestamp"] == ts0:
                        j += 1
                    else:
                        break
                repair_c1_block(df, positions, i, j, out_ts, parsed_ts, stats)
                i = j + 1
            elif ec == "C3_MISSING_TEMPORAL_INFORMATION":
                repair_c3_row(df, positions, i, out_ts, parsed_ts)
                i += 1
            else:
                i += 1

        i = 0
        while i < len(positions) - 1:
            p1 = positions[i]
            p2 = positions[i + 1]
            if df.at[p1, "error_category"] == "C2_TEMPORAL_INCONSISTENCY" or df.at[p2, "error_category"] == "C2_TEMPORAL_INCONSISTENCY":
                if pd.notna(out_ts[p1]) and pd.notna(out_ts[p2]) and out_ts[p2] < out_ts[p1]:
                    repair_c2_pair(df, positions, i, out_ts, parsed_ts, stats)
                    i += 2
                    continue
            i += 1

    # Global monotonic repair per case to remove remaining negative durations
    for case, positions in case_to_positions.items():
        final_case_monotonic_pass(df, positions, out_ts, parsed_ts, nonclean_mask)
        # strict repeat until stable
        for _ in range(3):
            changed = False
            for j in range(1, len(positions)):
                p_prev = positions[j - 1]
                p = positions[j]
                if out_ts[p] < out_ts[p_prev]:
                    changed = True
                    if nonclean_mask[p]:
                        out_ts[p] = out_ts[p_prev]
                    elif nonclean_mask[p_prev]:
                        out_ts[p_prev] = out_ts[p]
                    else:
                        out_ts[p] = out_ts[p_prev]
            if not changed:
                break

    # Build output timestamp strings while preserving untouched rows exactly
    out_strings = original_timestamp_strings[:]
    repaired_positions = set()
    for pos in range(len(df)):
        if not nonclean_mask[pos]:
            continue
        ec = df.at[pos, "error_category"]
        if ec == "C1_ORDERING_AMBIGUITY":
            # keep exact observed string only if repaired timestamp equals observed parsed timestamp
            if pd.notna(parsed_ts[pos]) and pd.notna(out_ts[pos]) and out_ts[pos] == parsed_ts[pos]:
                continue
        if str(original_timestamp_strings[pos]) == "" or pd.isna(parsed_ts[pos]) or pd.isna(out_ts[pos]) or out_ts[pos] != parsed_ts[pos]:
            repaired_positions.add(pos)
        elif ec in {"C2_TEMPORAL_INCONSISTENCY", "C3_MISSING_TEMPORAL_INFORMATION"}:
            repaired_positions.add(pos)

    for pos in repaired_positions:
        digits = frac_digits(original_timestamp_strings[pos])
        if digits == 0 and not date_only_string(original_timestamp_strings[pos]):
            digits = digits_mode
        if date_only_string(original_timestamp_strings[pos]):
            digits = digits_mode
        out_strings[pos] = fmt_ts(out_ts[pos], digits)

    # Self-check and corrective pass
    blocks_checked = 0
    rows_still_equal = 0

    for case, positions in case_to_positions.items():
        # per-case negatives final hard fix
        for _ in range(5):
            bad = False
            for j in range(1, len(positions)):
                p_prev = positions[j - 1]
                p = positions[j]
                t_prev = parse_ts(out_strings[p_prev]) if p_prev in repaired_positions else parsed_ts[p_prev]
                t = parse_ts(out_strings[p]) if p in repaired_positions else parsed_ts[p]
                if pd.notna(t_prev) and pd.notna(t) and t < t_prev:
                    bad = True
                    if nonclean_mask[p]:
                        t = t_prev
                        out_ts[p] = t
                        digits = frac_digits(original_timestamp_strings[p]) or digits_mode
                        out_strings[p] = fmt_ts(t, digits)
                        repaired_positions.add(p)
                    elif nonclean_mask[p_prev]:
                        t_prev = t
                        out_ts[p_prev] = t_prev
                        digits = frac_digits(original_timestamp_strings[p_prev]) or digits_mode
                        out_strings[p_prev] = fmt_ts(t_prev, digits)
                        repaired_positions.add(p_prev)
            if not bad:
                break

        i = 0
        while i < len(positions):
            p = positions[i]
            if df.at[p, "error_category"] == "C1_ORDERING_AMBIGUITY":
                ts0 = df.at[p, "Timestamp"]
                j = i
                while j + 1 < len(positions):
                    pj = positions[j + 1]
                    if df.at[pj, "error_category"] == "C1_ORDERING_AMBIGUITY" and df.at[pj, "Timestamp"] == ts0:
                        j += 1
                    else:
                        break
                block_pos = positions[i:j + 1]
                vals = [parse_ts(out_strings[q]) if q in repaired_positions else parsed_ts[q] for q in block_pos]
                unique_obs_kept = sum(1 for q, v in zip(block_pos, vals) if pd.notna(parsed_ts[q]) and v == parsed_ts[q])
                sorted_vals = sorted(vals)
                need_fix = False
                if len(set(vals)) != len(vals):
                    need_fix = True
                for a, b in zip(sorted_vals, sorted_vals[1:]):
                    if b <= a:
                        need_fix = True
                        break
                if unique_obs_kept != 1:
                    need_fix = True
                if need_fix:
                    blocks_checked += 1
                    order_idx = best_order(
                        [{"__pos": q, "Activity": df.at[q, "Activity"]} for q in block_pos],
                        df.at[positions[i - 1], "Activity"] if i - 1 >= 0 else None,
                        df.at[positions[j + 1], "Activity"] if j + 1 < len(positions) else None
                    )
                    ordered_pos = [block_pos[k] for k in order_idx]
                    base = parsed_ts[ordered_pos[0]]
                    if pd.isna(base):
                        base = parse_ts(df.at[ordered_pos[0], "Timestamp"])
                    right = parsed_ts[positions[j + 1]] if j + 1 < len(positions) else None
                    times = [base]
                    cur = base
                    for m in range(1, len(ordered_pos)):
                        g = max(1.0, duration_stat(df.at[ordered_pos[m - 1], "Activity"], df.at[ordered_pos[m], "Activity"]))
                        cur = cur + pd.Timedelta(seconds=g)
                        times.append(cur)
                    times, _ = enforce_within(times, base, right)
                    for q, t in zip(ordered_pos, times):
                        out_ts[q] = t
                        if not (q == ordered_pos[0] and t == parsed_ts[q]):
                            repaired_positions.add(q)
                            digits = frac_digits(original_timestamp_strings[q]) or digits_mode
                            out_strings[q] = fmt_ts(t, digits)
                        else:
                            out_strings[q] = original_timestamp_strings[q]
                i = j + 1
            else:
                i += 1

    # count equality on corrupted rows
    for pos in range(len(df)):
        if nonclean_mask[pos]:
            if out_strings[pos] == original_timestamp_strings[pos]:
                rows_still_equal += 1

    # restore byte-identical untouched columns
    out_df = df[required_cols].copy()
    out_df["Timestamp"] = out_strings

    # Safety: CLEAN rows exact original timestamp
    clean_mask = out_df["error_category"] == "CLEAN"
    out_df.loc[clean_mask, "Timestamp"] = [original_timestamp_strings[i] for i in out_df.index[clean_mask]]

    try:
        out_df.to_csv(args.output, index=False)
    except Exception:
        # guaranteed output attempt
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(required_cols) + "\n")
            for _, r in out_df.iterrows():
                vals = [("" if pd.isna(r[c]) else str(r[c])) for c in required_cols]
                escaped = []
                for v in vals:
                    if any(ch in v for ch in [",", '"', "\n"]):
                        v = '"' + v.replace('"', '""') + '"'
                    escaped.append(v)
                f.write(",".join(escaped) + "\n")

    print(f"blocks_repaired={stats['c1_blocks']} blocks_clamped={stats['clamped_blocks']} rows_equal_input={rows_still_equal}")


if __name__ == "__main__":
    main()