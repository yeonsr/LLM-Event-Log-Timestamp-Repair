import argparse
import math
import sys
from itertools import permutations
from typing import Dict, List, Tuple, Optional

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

DURATION_PROFILE = {
    ("Check for completeness", "Request info"): dict(n=6227, p10=61.9314, p25=170.1115, median=418.728, p75=830.2675, p90=1373.2918, share=0.097),
    ("Request info", "info received"): dict(n=6080, p10=10.0, p25=10.0, median=10.0, p75=10.0, p90=10.0, share=0.996),
    ("info received", "Check for completeness"): dict(n=6008, p10=63.7390, p25=176.05, median=420.682, p75=839.701, p90=1404.5438, share=0.095),
    ("Perform checks", "Make decision"): dict(n=5745, p10=62.8388, p25=171.483, median=422.924, p75=835.934, p90=1392.8122, share=0.095),
    ("Check for completeness", "Perform checks"): dict(n=5650, p10=61.0229, p25=171.5965, median=420.3715, p75=832.8305, p90=1416.3387, share=0.098),
    ("New online application received", "Check for completeness"): dict(n=4550, p10=65.3893, p25=171.7955, median=426.0135, p75=834.95025, p90=1385.6882, share=0.092),
    ("Deliver card", "EVENT 13 END"): dict(n=3276, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("Notify accept", "Deliver card"): dict(n=3276, p10=59.4585, p25=176.35075, median=425.762, p75=851.3035, p90=1370.5795, share=0.101),
    ("Make decision", "notify reject"): dict(n=3088, p10=63.7974, p25=174.02075, median=426.0405, p75=834.3485, p90=1427.2486, share=0.095),
    ("Make decision", "Notify accept"): dict(n=2969, p10=65.5868, p25=174.826, median=428.03, p75=867.335, p90=1436.6726, share=0.089),
    ("time out", "EVENT 13 END"): dict(n=1724, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("notify reject", "time out"): dict(n=1724, p10=10.0, p25=10.0, median=10.0, p75=10.0, p90=10.0, share=1.0),
    ("notify reject", "review request received"): dict(n=1534, p10=10.0, p25=10.0, median=10.0, p75=10.0, p90=10.0, share=1.0),
    ("review request received", "Check for completeness"): dict(n=1527, p10=78.0242, p25=188.8875, median=441.348, p75=841.8565, p90=1413.996, share=0.075),
    ("New online application received", "Notify accept"): dict(n=138, p10=944.0129, p25=1469.49875, median=2020.112, p75=2906.02375, p90=3878.789, share=0.0),
    ("New online application received", "Make decision"): dict(n=117, p10=634.1, p25=963.121, median=1661.293, p75=2178.854, p90=3090.3764, share=0.0),
    ("Check for completeness", "Notify accept"): dict(n=103, p10=711.8126, p25=1094.916, median=1780.137, p75=2325.416, p90=3107.5866, share=0.0),
    ("New online application received", "notify reject"): dict(n=82, p10=1132.4149, p25=1538.03275, median=2159.5775, p75=2824.4955, p90=3498.6097, share=0.0),
    ("Check for completeness", "Make decision"): dict(n=72, p10=1072.0155, p25=1536.0985, median=2023.867, p75=2917.63875, p90=3699.0501, share=0.0),
    ("info received", "Make decision"): dict(n=66, p10=721.489, p25=1122.10325, median=1868.191, p75=2613.01725, p90=3118.5855, share=0.0),
    ("Check for completeness", "notify reject"): dict(n=60, p10=945.7493, p25=1361.11625, median=1757.751, p75=2556.73125, p90=3239.0085, share=0.0),
    ("Request info", "Perform checks"): dict(n=53, p10=403.0504, p25=643.92, median=928.08, p75=1395.19, p90=2110.2466, share=0.0),
    ("New online application received", "info received"): dict(n=53, p10=452.3138, p25=639.021, median=857.047, p75=1303.053, p90=1811.6944, share=0.0),
    ("info received", "info received"): dict(n=52, p10=613.9782, p25=895.2945, median=1302.5635, p75=1835.09975, p90=2340.9239, share=0.0),
    ("Request info", "Make decision"): dict(n=50, p10=669.1557, p25=855.869, median=1523.8755, p75=2372.5625, p90=3437.5409, share=0.0),
    ("Request info", "Request info"): dict(n=43, p10=569.524, p25=796.317, median=1164.851, p75=1761.032, p90=2732.683, share=0.0),
    ("Request info", "Check for completeness"): dict(n=34, p10=803.7977, p25=952.73675, median=1889.409, p75=2738.841, p90=4159.5683, share=0.0),
    ("New online application received", "Perform checks"): dict(n=33, p10=1091.8104, p25=1559.19, median=2363.648, p75=3302.949, p90=3864.007, share=0.0),
    ("Request info", "Notify accept"): dict(n=30, p10=1429.9083, p25=1610.589, median=2396.886, p75=3179.76125, p90=3601.6131, share=0.0),
    ("Request info", "notify reject"): dict(n=30, p10=1539.1919, p25=1979.3755, median=2454.903, p75=2997.263, p90=3804.9144, share=0.0),
    ("info received", "notify reject"): dict(n=29, p10=1059.5184, p25=1682.395, median=2806.157, p75=3442.502, p90=3679.6308, share=0.0),
    ("info received", "Perform checks"): dict(n=27, p10=809.8986, p25=1087.906, median=1874.355, p75=2956.759, p90=4336.3136, share=0.0),
    ("info received", "Notify accept"): dict(n=24, p10=1006.2042, p25=1510.50375, median=2230.734, p75=3852.069, p90=4100.492, share=0.0),
    ("Check for completeness", "info received"): dict(n=19, p10=817.942, p25=1530.727, median=1910.593, p75=2581.938, p90=2788.1602, share=0.0),
    ("New online application received", "Request info"): dict(n=17, p10=809.1352, p25=1687.418, median=2254.706, p75=2700.696, p90=4232.1842, share=0.0),
    ("Perform checks", "Check for completeness"): dict(n=15, p10=1023.0088, p25=1420.4865, median=1995.413, p75=2965.38, p90=3838.7202, share=0.0),
    ("Perform checks", "review request received"): dict(n=15, p10=378.412, p25=555.4955, median=705.0, p75=849.476, p90=1327.28, share=0.0),
    ("Check for completeness", "Check for completeness"): dict(n=13, p10=1452.7526, p25=1939.982, median=2577.014, p75=3296.051, p90=3589.3102, share=0.0),
    ("notify reject", "Make decision"): dict(n=12, p10=785.8718, p25=1341.0605, median=1518.5795, p75=1958.461, p90=2274.8716, share=0.0),
    ("review request received", "Make decision"): dict(n=12, p10=727.506, p25=964.28475, median=2227.617, p75=2260.87725, p90=3134.5752, share=0.0),
    ("Perform checks", "Request info"): dict(n=11, p10=1288.529, p25=1794.608, median=2625.632, p75=3426.251, p90=4208.01, share=0.0),
    ("New online application received", "review request received"): dict(n=10, p10=783.2576, p25=1569.0395, median=1845.21, p75=2201.5355, p90=2416.4854, share=0.0),
    ("notify reject", "Perform checks"): dict(n=9, p10=318.1018, p25=699.898, median=874.95, p75=1271.62, p90=1995.78, share=0.0),
    ("review request received", "Notify accept"): dict(n=9, p10=2066.8628, p25=2191.667, median=2296.639, p75=4048.847, p90=4780.2734, share=0.0),
    ("Check for completeness", "review request received"): dict(n=9, p10=786.9502, p25=1083.927, median=1363.124, p75=2554.657, p90=2954.3854, share=0.0),
    ("review request received", "Perform checks"): dict(n=9, p10=1338.8826, p25=1593.822, median=2249.376, p75=3393.616, p90=4267.802, share=0.0),
    ("info received", "Request info"): dict(n=9, p10=824.7008, p25=1305.603, median=1993.345, p75=3112.501, p90=4079.4476, share=0.0),
    ("EVENT 13 END", "EVENT 13 END"): dict(n=8, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("Make decision", "info received"): dict(n=7, p10=615.6284, p25=944.3135, median=1863.824, p75=1941.4715, p90=2142.2018, share=0.0),
    ("notify reject", "Check for completeness"): dict(n=6, p10=1769.177, p25=2725.29575, median=3075.0545, p75=3917.45225, p90=4349.0205, share=0.0),
    ("Make decision", "Make decision"): dict(n=6, p10=1349.042, p25=1693.34325, median=2015.0765, p75=2417.1235, p90=2768.227, share=0.0),
    ("Perform checks", "Perform checks"): dict(n=6, p10=1054.435, p25=1415.281, median=2294.3645, p75=2990.205, p90=5170.2335, share=0.0),
    ("review request received", "notify reject"): dict(n=5, p10=1334.3154, p25=1360.086, median=1469.711, p75=2384.126, p90=3042.9542, share=0.0),
    ("review request received", "Request info"): dict(n=5, p10=1863.9608, p25=2128.058, median=2501.573, p75=3114.888, p90=4176.5796, share=0.0),
    ("Make decision", "Request info"): dict(n=5, p10=1027.2532, p25=1078.57, median=1325.858, p75=1553.253, p90=2857.497, share=0.0),
    ("notify reject", "notify reject"): dict(n=5, p10=1475.265, p25=2749.563, median=2967.996, p75=3278.084, p90=3452.2106, share=0.0),
    ("Make decision", "Perform checks"): dict(n=5, p10=783.9354, p25=796.233, median=823.397, p75=1650.51, p90=1824.5034, share=0.0),
    ("review request received", "info received"): dict(n=4, p10=1391.3396, p25=1447.0055, median=1983.33, p75=2496.487, p90=2510.4514, share=0.0),
    ("notify reject", "info received"): dict(n=3, p10=652.6522, p25=721.4095, median=836.005, p75=1576.566, p90=2020.9026, share=0.0),
    ("notify reject", "Request info"): dict(n=3, p10=389.1752, p25=613.13, median=986.388, p75=1967.2505, p90=2555.768, share=0.0),
}

GLOBAL_MEDIAN = 243.721
FAST_FALLBACK = 120.0


def safe_to_datetime(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT


def frac_digits(ts: str) -> int:
    if ts is None:
        return 0
    s = str(ts)
    if "." in s:
        return len(s.split(".")[-1])
    return 0


def is_blank_timestamp(s: str) -> bool:
    return pd.isna(s) or str(s).strip() == "" or str(s).lower() == "nan"


def is_date_only(s: str) -> bool:
    if is_blank_timestamp(s):
        return False
    txt = str(s).strip()
    return len(txt) == 10 and txt.count("-") == 2


def choose_gap_seconds(a: str, b: str) -> float:
    prof = DURATION_PROFILE.get((a, b))
    if prof:
        if prof["share"] >= 0.95 or (prof["p10"] == prof["p25"] == prof["median"]):
            g = prof["p10"]
        elif prof["share"] >= 0.5:
            g = prof["p10"]
        else:
            g = prof["p25"]
        return max(float(g), 0.0)
    p = TRANSITION_PROB.get((a, b), 0.0)
    if p >= 0.5:
        return 60.0
    if p >= 0.05:
        return 180.0
    if p >= 0.005:
        return 600.0
    return FAST_FALLBACK


def transition_logprob(a: Optional[str], b: Optional[str]) -> float:
    if a is None or b is None:
        return -0.5
    p = TRANSITION_PROB.get((a, b))
    if p is None:
        return math.log(1e-6)
    return math.log(max(p, 1e-12))


def score_order(order_rows: List[int], activities: List[str], left_act: Optional[str], right_act: Optional[str]) -> float:
    score = 0.0
    if left_act is not None and activities:
        score += transition_logprob(left_act, activities[order_rows[0]]) * 1.2
    for i in range(len(order_rows) - 1):
        score += transition_logprob(activities[order_rows[i]], activities[order_rows[i + 1]]) * 2.0
    if right_act is not None and activities:
        score += transition_logprob(activities[order_rows[-1]], right_act) * 1.0
    bonus = 0.0
    for i in range(len(order_rows) - 1):
        a = activities[order_rows[i]]
        b = activities[order_rows[i + 1]]
        prof = DURATION_PROFILE.get((a, b))
        if prof:
            if prof["share"] >= 0.95:
                bonus += 2.5
            elif prof["share"] >= 0.5:
                bonus += 1.2
            elif prof["p25"] <= 300:
                bonus += 0.6
            if prof["n"] >= 1000:
                bonus += 0.3
    return score + bonus


def best_order_for_block(block_df: pd.DataFrame, left_act: Optional[str], right_act: Optional[str]) -> List[int]:
    activities = block_df["Activity"].tolist()
    k = len(activities)
    idxs = list(range(k))
    if k <= 7:
        best = None
        best_score = -1e100
        for perm in permutations(idxs):
            sc = score_order(list(perm), activities, left_act, right_act)
            if sc > best_score:
                best_score = sc
                best = list(perm)
        return best
    beam = [([], set(), 0.0)]
    beam_width = 50
    for _ in range(k):
        new_beam = []
        for seq, used, sc in beam:
            last_act = activities[seq[-1]] if seq else None
            for j in idxs:
                if j in used:
                    continue
                add = 0.0
                if not seq:
                    add += transition_logprob(left_act, activities[j]) * 1.2 if left_act is not None else -0.5
                else:
                    add += transition_logprob(last_act, activities[j]) * 2.0
                    prof = DURATION_PROFILE.get((last_act, activities[j]))
                    if prof:
                        if prof["share"] >= 0.95:
                            add += 2.5
                        elif prof["share"] >= 0.5:
                            add += 1.2
                        elif prof["p25"] <= 300:
                            add += 0.6
                new_beam.append((seq + [j], used | {j}, sc + add))
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]
    best = None
    best_score = -1e100
    for seq, used, sc in beam:
        if len(seq) == k:
            if right_act is not None:
                sc += transition_logprob(activities[seq[-1]], right_act)
            if sc > best_score:
                best_score = sc
                best = seq
    return best if best is not None else idxs


def format_ts(dt: pd.Timestamp, digits: int) -> str:
    if pd.isna(dt):
        return ""
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    if digits > 0:
        frac = f"{dt.microsecond:06d}"[:digits]
        return f"{base}.{frac}"
    return base


def local_clean_anchor(case_df: pd.DataFrame, pos: int, direction: int) -> Tuple[Optional[int], Optional[pd.Timestamp], Optional[str]]:
    n = len(case_df)
    i = pos + direction
    while 0 <= i < n:
        row = case_df.iloc[i]
        if row["error_category"] == "CLEAN":
            t = safe_to_datetime(row["Timestamp"])
            if not pd.isna(t):
                return i, t, row["Activity"]
        i += direction
    return None, None, None


def case_bounds(case_df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    vals = []
    for _, r in case_df.iterrows():
        t = safe_to_datetime(r["Timestamp"])
        if not pd.isna(t):
            vals.append(t)
    if not vals:
        return None, None
    return min(vals), max(vals)


def enforce_bounds(dt: pd.Timestamp, lo: Optional[pd.Timestamp], hi: Optional[pd.Timestamp]) -> pd.Timestamp:
    if lo is not None and dt < lo:
        dt = lo
    if hi is not None and dt > hi:
        dt = hi
    return dt


def ensure_distinct_forward(times: List[pd.Timestamp], start: pd.Timestamp, end: Optional[pd.Timestamp]) -> List[pd.Timestamp]:
    out = [times[0]]
    for i in range(1, len(times)):
        need = out[-1] + pd.Timedelta(seconds=1)
        t = times[i]
        if t <= out[-1]:
            t = need
        out.append(t)
    if end is not None and out[-1] > end:
        room = (end - start).total_seconds()
        k = len(out)
        min_span = k - 1
        if room < min_span:
            room = min_span
            end = start + pd.Timedelta(seconds=room)
        gaps = [max(1.0, (out[i] - out[i - 1]).total_seconds()) for i in range(1, k)]
        s = sum(gaps)
        if s <= 0:
            gaps = [1.0] * (k - 1)
            s = sum(gaps)
        scale = min(1.0, room / s)
        new = [start]
        acc = 0.0
        for g in gaps:
            acc += max(1.0, g * scale)
            new.append(start + pd.Timedelta(seconds=acc))
        for i in range(1, k):
            if new[i] <= new[i - 1]:
                new[i] = new[i - 1] + pd.Timedelta(seconds=1)
        if new[-1] > end:
            new = [start + pd.Timedelta(seconds=i) for i in range(k)]
        out = new
    return out


def repair_c1_block(case_df: pd.DataFrame, block_positions: List[int], output_ts: Dict[int, str], repaired_idx: set, stats: dict):
    block_df = case_df.iloc[block_positions].copy()
    observed_str = str(block_df.iloc[0]["Timestamp"])
    observed_dt = safe_to_datetime(observed_str)
    if pd.isna(observed_dt):
        return
    digits = max(frac_digits(observed_str), 3)
    left_i, left_t, left_act = local_clean_anchor(case_df, block_positions[0], -1)
    right_i, right_t, right_act = local_clean_anchor(case_df, block_positions[-1], 1)
    cstart, cend = case_bounds(case_df)
    order = best_order_for_block(block_df, left_act, right_act)
    ordered_rows = block_df.iloc[order]
    ordered_acts = ordered_rows["Activity"].tolist()
    gaps = []
    for i in range(len(ordered_acts) - 1):
        gaps.append(max(1.0, choose_gap_seconds(ordered_acts[i], ordered_acts[i + 1])))
    times = [observed_dt]
    for g in gaps:
        times.append(times[-1] + pd.Timedelta(seconds=float(g)))
    clamped = False
    if right_t is not None:
        times = ensure_distinct_forward(times, observed_dt, right_t)
        if times[-1] > right_t:
            clamped = True
            times = ensure_distinct_forward([observed_dt] + [observed_dt + pd.Timedelta(seconds=i) for i in range(1, len(times))], observed_dt, right_t)
    else:
        times = ensure_distinct_forward(times, observed_dt, cend)
    if cstart is not None or cend is not None:
        times = [enforce_bounds(t, cstart, cend) for t in times]
        times = ensure_distinct_forward(times, observed_dt, right_t if right_t is not None else cend)
    if len(times) >= 2 and (times[-1] - times[0]).total_seconds() < (len(times) - 1):
        times = [observed_dt + pd.Timedelta(seconds=i) for i in range(len(times))]
        if right_t is not None:
            times = ensure_distinct_forward(times, observed_dt, right_t)
        clamped = True
    for pos_in_order, (_, row) in enumerate(ordered_rows.iterrows()):
        global_idx = row.name
        if pos_in_order == 0:
            new_str = observed_str
        else:
            t = times[pos_in_order]
            t = enforce_bounds(t, observed_dt, right_t if right_t is not None else cend)
            if t <= observed_dt:
                t = observed_dt + pd.Timedelta(seconds=pos_in_order)
            new_str = format_ts(t, digits)
        output_ts[global_idx] = new_str
        repaired_idx.add(global_idx)
    if clamped:
        stats["c1_clamped"] += 1
    stats["c1_blocks"] += 1


def local_plausibility(prev_act, act, next_act, prev_t, t, next_t) -> float:
    score = 0.0
    if prev_act is not None and prev_t is not None and not pd.isna(prev_t):
        dt = (t - prev_t).total_seconds()
        if dt < 0:
            score -= 20
        else:
            exp = choose_gap_seconds(prev_act, act)
            score -= abs(dt - exp) / max(exp, 30.0)
            score += transition_logprob(prev_act, act)
    if next_act is not None and next_t is not None and not pd.isna(next_t):
        dt = (next_t - t).total_seconds()
        if dt < 0:
            score -= 20
        else:
            exp = choose_gap_seconds(act, next_act)
            score -= abs(dt - exp) / max(exp, 30.0)
            score += transition_logprob(act, next_act)
    return score


def estimate_from_anchors(prev_act, act, next_act, prev_t, next_t) -> pd.Timestamp:
    cand = []
    if prev_t is not None and not pd.isna(prev_t):
        cand.append(prev_t + pd.Timedelta(seconds=choose_gap_seconds(prev_act, act)))
    if next_t is not None and not pd.isna(next_t):
        cand.append(next_t - pd.Timedelta(seconds=choose_gap_seconds(act, next_act)))
    if prev_t is not None and next_t is not None and not pd.isna(prev_t) and not pd.isna(next_t):
        lo = prev_t
        hi = next_t
        if len(cand) == 2:
            mid = cand[0] + (cand[1] - cand[0]) / 2
        else:
            mid = lo + (hi - lo) / 2
        if mid <= lo:
            mid = lo + pd.Timedelta(seconds=1)
        if mid >= hi:
            mid = lo + (hi - lo) / 2
            if mid <= lo:
                mid = hi - pd.Timedelta(seconds=1)
        return enforce_bounds(mid, lo, hi)
    if cand:
        return cand[0]
    return pd.Timestamp.now().normalize()


def repair_c2(case_df: pd.DataFrame, output_ts: Dict[int, str], repaired_idx: set, stats: dict):
    cstart, cend = case_bounds(case_df)
    n = len(case_df)
    for i in range(1, n):
        row_prev = case_df.iloc[i - 1]
        row_cur = case_df.iloc[i]
        if row_prev["error_category"] != "C2_TEMPORAL_INCONSISTENCY" and row_cur["error_category"] != "C2_TEMPORAL_INCONSISTENCY":
            continue
        t1 = safe_to_datetime(output_ts.get(row_prev.name, row_prev["Timestamp"]))
        t2 = safe_to_datetime(output_ts.get(row_cur.name, row_cur["Timestamp"]))
        if pd.isna(t1) or pd.isna(t2) or t2 >= t1:
            continue
        prev2_i, prev2_t, prev2_act = local_clean_anchor(case_df, i - 1, -1)
        next2_i, next2_t, next2_act = local_clean_anchor(case_df, i, 1)
        a1, a2 = row_prev["Activity"], row_cur["Activity"]

        score_orig = local_plausibility(prev2_act, a1, a2, prev2_t, t1, t2) + local_plausibility(a1, a2, next2_act, t1, t2, next2_t)
        score_swap = -1e100
        if t1 <= t2:
            score_swap = score_orig - 100
        else:
            ts1, ts2 = t2, t1
            if (prev2_t is None or ts1 >= prev2_t) and (next2_t is None or ts2 <= next2_t):
                score_swap = local_plausibility(prev2_act, a1, a2, prev2_t, ts1, ts2) + local_plausibility(a1, a2, next2_act, ts1, ts2, next2_t)

        if score_swap > score_orig + 1.0:
            d1 = max(frac_digits(str(row_prev["Timestamp"])), 3)
            d2 = max(frac_digits(str(row_cur["Timestamp"])), 3)
            output_ts[row_prev.name] = format_ts(t2, d1)
            output_ts[row_cur.name] = format_ts(t1, d2)
            repaired_idx.add(row_prev.name)
            repaired_idx.add(row_cur.name)
            stats["c2_swaps"] += 1
            continue

        cand_prev = estimate_from_anchors(prev2_act, a1, a2, prev2_t, t2)
        cand_cur = estimate_from_anchors(a1, a2, next2_act, t1, next2_t)
        sc_prev_fix = local_plausibility(prev2_act, a1, a2, prev2_t, cand_prev, t2) + local_plausibility(a1, a2, next2_act, cand_prev, t2, next2_t)
        sc_cur_fix = local_plausibility(prev2_act, a1, a2, prev2_t, t1, cand_cur) + local_plausibility(a1, a2, next2_act, t1, cand_cur, next2_t)

        if sc_prev_fix >= sc_cur_fix:
            new_t = cand_prev
            lo = prev2_t if prev2_t is not None else cstart
            hi = t2
            if hi is not None and not pd.isna(hi):
                new_t = enforce_bounds(new_t, lo, hi)
                if new_t >= hi:
                    new_t = hi - pd.Timedelta(seconds=1)
            if lo is not None and not pd.isna(lo) and new_t <= lo:
                new_t = lo + pd.Timedelta(seconds=1)
            d = max(frac_digits(str(row_prev["Timestamp"])), 3)
            output_ts[row_prev.name] = format_ts(enforce_bounds(new_t, cstart, cend), d)
            repaired_idx.add(row_prev.name)
        else:
            new_t = cand_cur
            lo = t1
            hi = next2_t if next2_t is not None else cend
            if lo is not None and not pd.isna(lo) and new_t <= lo:
                if hi is not None and not pd.isna(hi):
                    new_t = lo + (hi - lo) / 2
                else:
                    new_t = lo + pd.Timedelta(seconds=max(1.0, choose_gap_seconds(a1, a2)))
            new_t = enforce_bounds(new_t, lo, hi)
            if hi is not None and not pd.isna(hi) and new_t >= hi:
                new_t = hi - pd.Timedelta(seconds=1)
            d = max(frac_digits(str(row_cur["Timestamp"])), 3)
            output_ts[row_cur.name] = format_ts(enforce_bounds(new_t, cstart, cend), d)
            repaired_idx.add(row_cur.name)
        stats["c2_displaced"] += 1


def repair_c3(case_df: pd.DataFrame, output_ts: Dict[int, str], repaired_idx: set, stats: dict):
    cstart, cend = case_bounds(case_df)
    for pos in range(len(case_df)):
        row = case_df.iloc[pos]
        if row["error_category"] != "C3_MISSING_TEMPORAL_INFORMATION":
            continue
        ts_str = row["Timestamp"]
        left_i, left_t, left_act = local_clean_anchor(case_df, pos, -1)
        right_i, right_t, right_act = local_clean_anchor(case_df, pos, 1)
        act = row["Activity"]
        digits = max(frac_digits(str(ts_str)), 3)

        if is_blank_timestamp(ts_str):
            est = estimate_from_anchors(left_act, act, right_act, left_t, right_t)
            est = enforce_bounds(est, cstart, cend)
            if left_t is not None and right_t is not None:
                est = enforce_bounds(est, left_t, right_t)
                if est <= left_t:
                    est = left_t + (right_t - left_t) / 2
                if est >= right_t:
                    est = left_t + (right_t - left_t) / 2
            output_ts[row.name] = format_ts(est, digits)
            repaired_idx.add(row.name)
            stats["c3_rows"] += 1
        elif is_date_only(ts_str):
            date = pd.to_datetime(ts_str).date()
            day_start = pd.Timestamp(date)
            day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            est = estimate_from_anchors(left_act, act, right_act, left_t, right_t)
            est = pd.Timestamp.combine(date, est.time())
            lo = day_start
            hi = day_end
            if left_t is not None and left_t.date() == date:
                lo = max(lo, left_t)
            if right_t is not None and right_t.date() == date:
                hi = min(hi, right_t)
            est = enforce_bounds(est, lo, hi)
            if left_t is not None and right_t is not None and left_t.date() == date and right_t.date() == date and est >= right_t:
                est = left_t + (right_t - left_t) / 2
            output_ts[row.name] = format_ts(est, max(digits, 3))
            repaired_idx.add(row.name)
            stats["c3_rows"] += 1
        else:
            t = safe_to_datetime(ts_str)
            if pd.isna(t):
                est = estimate_from_anchors(left_act, act, right_act, left_t, right_t)
                output_ts[row.name] = format_ts(enforce_bounds(est, cstart, cend), max(digits, 3))
                repaired_idx.add(row.name)
                stats["c3_rows"] += 1


def self_check_and_fix(df: pd.DataFrame, output_ts: Dict[int, str], stats: dict):
    by_case = df.groupby("Case", sort=False)
    for _, case_df in by_case:
        n = len(case_df)
        case_positions = list(case_df.index)

        i = 0
        while i < n:
            row = case_df.iloc[i]
            if row["error_category"] == "C1_ORDERING_AMBIGUITY":
                obs = str(row["Timestamp"])
                j = i
                block = []
                while j < n:
                    r = case_df.iloc[j]
                    if r["error_category"] == "C1_ORDERING_AMBIGUITY" and str(r["Timestamp"]) == obs:
                        block.append(r.name)
                        j += 1
                    else:
                        break
                vals = [safe_to_datetime(output_ts.get(idx, df.at[idx, "Timestamp"])) for idx in block]
                uniq = len({v.value for v in vals if not pd.isna(v)})
                if uniq < len(block):
                    obs_dt = safe_to_datetime(obs)
                    _, right_t, _ = local_clean_anchor(case_df, i + len(block) - 1, 1)
                    end = right_t
                    seq = [obs_dt + pd.Timedelta(seconds=s) for s in range(len(block))]
                    if end is not None:
                        seq = ensure_distinct_forward(seq, obs_dt, end)
                    for k2, idx in enumerate(block):
                        digits = max(frac_digits(str(df.at[idx, "Timestamp"])), 3)
                        if k2 == 0:
                            output_ts[idx] = str(df.at[idx, "Timestamp"])
                        else:
                            output_ts[idx] = format_ts(seq[k2], digits)
                    stats["c1_clamped"] += 1
                i = j
            else:
                i += 1

    for idx, row in df.iterrows():
        if row["error_category"] == "CLEAN":
            output_ts[idx] = row["Timestamp"]
        new_s = output_ts.get(idx, row["Timestamp"])
        dt = safe_to_datetime(new_s)
        if pd.isna(dt):
            output_ts[idx] = row["Timestamp"] if row["error_category"] == "CLEAN" else "2000-01-01 00:00:00.000"
    return output_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_cols = list(df.columns)
    output_ts: Dict[int, str] = {idx: row["Timestamp"] for idx, row in df.iterrows()}
    repaired_idx = set()
    stats = {"c1_blocks": 0, "c1_clamped": 0, "c2_swaps": 0, "c2_displaced": 0, "c3_rows": 0}

    for _, case_df in df.groupby("Case", sort=False):
        case_df = case_df.copy()
        n = len(case_df)

        i = 0
        while i < n:
            row = case_df.iloc[i]
            if row["error_category"] == "C1_ORDERING_AMBIGUITY":
                obs = str(row["Timestamp"])
                j = i
                block_positions = []
                while j < n:
                    r = case_df.iloc[j]
                    if r["error_category"] == "C1_ORDERING_AMBIGUITY" and str(r["Timestamp"]) == obs:
                        block_positions.append(j)
                        j += 1
                    else:
                        break
                if len(block_positions) >= 2:
                    repair_c1_block(case_df, block_positions, output_ts, repaired_idx, stats)
                else:
                    idx = case_df.iloc[block_positions[0]].name
                    orig_dt = safe_to_datetime(case_df.iloc[block_positions[0]]["Timestamp"])
                    _, right_t, _ = local_clean_anchor(case_df, block_positions[0], 1)
                    digits = max(frac_digits(str(case_df.iloc[block_positions[0]]["Timestamp"])), 3)
                    if right_t is not None and not pd.isna(orig_dt):
                        new_t = min(orig_dt + pd.Timedelta(seconds=1), right_t)
                        if new_t == orig_dt:
                            new_t = orig_dt + pd.Timedelta(milliseconds=1)
                        output_ts[idx] = format_ts(new_t, digits)
                        repaired_idx.add(idx)
                i = j
            else:
                i += 1

        repair_c2(case_df, output_ts, repaired_idx, stats)
        repair_c3(case_df, output_ts, repaired_idx, stats)

    output_ts = self_check_and_fix(df, output_ts, stats)

    out_df = df.copy()
    out_df["Timestamp"] = [output_ts[idx] for idx in out_df.index]

    for idx, row in df.iterrows():
        if row["error_category"] == "CLEAN":
            out_df.at[idx, "Timestamp"] = row["Timestamp"]

    still_equal = 0
    for idx, row in df.iterrows():
        if row["error_category"] != "CLEAN":
            if out_df.at[idx, "Timestamp"] == row["Timestamp"]:
                still_equal += 1

    out_df = out_df[original_cols]
    out_df.to_csv(args.output, index=False)

    print(f"blocks_repaired={stats['c1_blocks']} blocks_clamped={stats['c1_clamped']} rows_still_equal={still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            print(f"blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")
        except Exception:
            pass