import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP

# -----------------------------
# Embedded process knowledge
# -----------------------------

DURATION_PROFILES = {
    ("FIN", "RELEASE"): {"n": 6486, "p10": 2048.0, "p25": 6744.5, "median": 16358.0, "p75": 25456.75, "p90": 36587.0, "share_under_60s": 0.053},
    ("CODE OK", "BILLED"): {"n": 6214, "p10": 398373.10000000003, "p25": 1350225.25, "median": 3790599.0, "p75": 7240106.5, "p90": 11149642.3, "share_under_60s": 0.0},
    ("RELEASE", "CODE OK"): {"n": 6021, "p10": 3515.0, "p25": 176004.0, "median": 343375.0, "p75": 440228.0, "p90": 534115.0, "share_under_60s": 0.049},
    ("NEW", "CHANGE DIAGN"): {"n": 4748, "p10": 12.0, "p25": 16.0, "median": 25.0, "p75": 9184.25, "p90": 1734835.3000000007, "share_under_60s": 0.639},
    ("CHANGE DIAGN", "FIN"): {"n": 4199, "p10": 1595249.4000000001, "p25": 4010767.0, "median": 7708017.0, "p75": 7904282.0, "p90": 8064003.0, "share_under_60s": 0.007},
    ("NEW", "FIN"): {"n": 2813, "p10": 430931.4, "p25": 5613832.0, "median": 30069111.0, "p75": 31622139.0, "p90": 31634310.2, "share_under_60s": 0.036},
    ("NEW", "NEW"): {"n": 1106, "p10": 1.0, "p25": 1.0, "median": 1.0, "p75": 1.0, "p90": 1.0, "share_under_60s": 1.0},
    ("RELEASE", "BILLED"): {"n": 775, "p10": 673459.0, "p25": 1305167.0, "median": 3654832.0, "p75": 7884415.5, "p90": 11870267.6, "share_under_60s": 0.0},
    ("FIN", "CODE OK"): {"n": 769, "p10": 174331.20000000004, "p25": 269794.0, "median": 379548.0, "p75": 522371.0, "p90": 570947.6, "share_under_60s": 0.001},
    ("NEW", "DELETE"): {"n": 590, "p10": 2360.100000000002, "p25": 167520.75, "median": 1834565.0, "p75": 7132370.25, "p90": 23068173.700000007, "share_under_60s": 0.063},
    ("REOPEN", "FIN"): {"n": 470, "p10": 5.900000000000006, "p25": 10.0, "median": 56.0, "p75": 198754.75, "p90": 1119424.900000001, "share_under_60s": 0.506},
    ("BILLED", "STORNO"): {"n": 439, "p10": 195330.6, "p25": 434152.5, "median": 1077095.0, "p75": 3227309.5, "p90": 11913231.79999999, "share_under_60s": 0.002},
    ("CODE OK", "REOPEN"): {"n": 437, "p10": 123231.6, "p25": 460198.0, "median": 1914409.0, "p75": 4429460.0, "p90": 7604387.000000012, "share_under_60s": 0.014},
    ("CHANGE DIAGN", "RELEASE"): {"n": 417, "p10": 2303730.2, "p25": 4374232.0, "median": 7729262.0, "p75": 7974821.0, "p90": 8085701.0, "share_under_60s": 0.0},
    ("CHANGE DIAGN", "CHANGE DIAGN"): {"n": 300, "p10": 240.70000000000005, "p25": 52849.0, "median": 861391.0, "p75": 2428618.25, "p90": 4874708.100000005, "share_under_60s": 0.04},
    ("STORNO", "BILLED"): {"n": 246, "p10": 92765.5, "p25": 92835.25, "median": 169858.5, "p75": 170045.5, "p90": 504917.0, "share_under_60s": 0.0},
    ("STORNO", "REJECT"): {"n": 203, "p10": 5.200000000000003, "p25": 14.0, "median": 1237.0, "p75": 1916190.0, "p90": 9494887.80000001, "share_under_60s": 0.448},
    ("REOPEN", "DELETE"): {"n": 151, "p10": 2.0, "p25": 2.0, "median": 3.0, "p75": 4.0, "p90": 9.0, "share_under_60s": 0.967},
    ("CHANGE DIAGN", "DELETE"): {"n": 139, "p10": 203.4, "p25": 66091.0, "median": 946414.0, "p75": 4221145.0, "p90": 7573763.0, "share_under_60s": 0.065},
    ("REJECT", "BILLED"): {"n": 126, "p10": 1882.5, "p25": 163280.0, "median": 328872.5, "p75": 411299.5, "p90": 4350796.5, "share_under_60s": 0.056},
    ("RELEASE", "CODE NOK"): {"n": 120, "p10": 346460.4, "p25": 609418.75, "median": 1439807.5, "p75": 3817948.25, "p90": 6746112.0000000065, "share_under_60s": 0.008},
    ("CODE OK", "RELEASE"): {"n": 95, "p10": 2059007.0, "p25": 2987528.5, "median": 4572613.0, "p75": 5923328.5, "p90": 6849198.2, "share_under_60s": 0.011},
    ("RELEASE", "REOPEN"): {"n": 75, "p10": 28564.8, "p25": 129282.5, "median": 399253.0, "p75": 2230172.0, "p90": 4985157.400000005, "share_under_60s": 0.027},
    ("REJECT", "REOPEN"): {"n": 72, "p10": 8.0, "p25": 10.0, "median": 30.5, "p75": 95.0, "p90": 464.9, "share_under_60s": 0.667},
    ("JOIN-PAT", "JOIN-PAT"): {"n": 47, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 1.0, "p90": 1.0, "share_under_60s": 1.0},
    ("FIN", "REOPEN"): {"n": 45, "p10": 6.4, "p25": 30.0, "median": 18130.0, "p75": 360631.0, "p90": 3614904.0000000014, "share_under_60s": 0.289},
    ("CODE NOK", "CODE NOK"): {"n": 44, "p10": 343362.5, "p25": 579864.25, "median": 1682059.0, "p75": 6435773.25, "p90": 7345313.0, "share_under_60s": 0.0},
    ("CODE NOK", "SET STATUS"): {"n": 43, "p10": 8537.8, "p25": 15714.5, "median": 25242.0, "p75": 32981.5, "p90": 40848.40000000001, "share_under_60s": 0.023},
    ("CODE NOK", "CODE OK"): {"n": 39, "p10": 2626.0, "p25": 19872.5, "median": 49685.0, "p75": 52020.0, "p90": 88020.2, "share_under_60s": 0.0},
    ("FIN", "DELETE"): {"n": 37, "p10": 69.8, "p25": 40850.0, "median": 1180180.0, "p75": 2238393.0, "p90": 4879596.799999998, "share_under_60s": 0.081},
    ("FIN", "FIN"): {"n": 35, "p10": 0.0, "p25": 0.0, "median": 31430606.0, "p75": 35271797.5, "p90": 38507410.2, "share_under_60s": 0.314},
    ("CODE OK", "MANUAL"): {"n": 34, "p10": 494.5, "p25": 328332.5, "median": 1539666.5, "p75": 2052885.5, "p90": 2670681.8, "share_under_60s": 0.059},
    ("MANUAL", "RELEASE"): {"n": 27, "p10": 194.60000000000002, "p25": 24535.0, "median": 36278.0, "p75": 41857.0, "p90": 42132.2, "share_under_60s": 0.074},
    ("REOPEN", "CHANGE DIAGN"): {"n": 27, "p10": 8.8, "p25": 17.0, "median": 42.0, "p75": 217.0, "p90": 867713.400000002, "share_under_60s": 0.519},
    ("CODE NOK", "REOPEN"): {"n": 26, "p10": 11420.0, "p25": 21800.5, "median": 31239.5, "p75": 48681.75, "p90": 585812.0, "share_under_60s": 0.038},
    ("BILLED", "BILLED"): {"n": 26, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 1028659.5, "p90": 1170203.0, "share_under_60s": 0.577},
    ("CODE OK", "FIN"): {"n": 23, "p10": 769.6, "p25": 882370.5, "median": 2073019.0, "p75": 6294111.0, "p90": 7931658.600000001, "share_under_60s": 0.043},
    ("CODE OK", "DELETE"): {"n": 22, "p10": 382465.9, "p25": 1677310.75, "median": 2805210.0, "p75": 5376157.75, "p90": 7365047.900000002, "share_under_60s": 0.0},
    ("CODE OK", "CODE OK"): {"n": 22, "p10": 1087843.1, "p25": 3184653.0, "median": 10164512.0, "p75": 12655999.75, "p90": 15934178.300000003, "share_under_60s": 0.0},
    ("RELEASE", "RELEASE"): {"n": 21, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 2701508.0, "p90": 6877633.0, "share_under_60s": 0.524},
    ("CODE NOK", "DELETE"): {"n": 19, "p10": 12976.6, "p25": 24252.0, "median": 29918.0, "p75": 41426.5, "p90": 49063.6, "share_under_60s": 0.0},
    ("BILLED", "CODE OK"): {"n": 18, "p10": 2281385.9, "p25": 7145231.75, "median": 8266342.5, "p75": 24065387.0, "p90": 36390457.90000002, "share_under_60s": 0.0},
    ("RELEASE", "DELETE"): {"n": 16, "p10": 36141.5, "p25": 131670.75, "median": 302460.0, "p75": 1770106.75, "p90": 14471341.5, "share_under_60s": 0.062},
    ("RELEASE", "SET STATUS"): {"n": 15, "p10": 142643.8, "p25": 221989.0, "median": 602525.0, "p75": 1068407.0, "p90": 1852543.2, "share_under_60s": 0.0},
    ("BILLED", "REJECT"): {"n": 13, "p10": 1489822.8, "p25": 3271504.0, "median": 5749219.0, "p75": 14042919.0, "p90": 17629354.8, "share_under_60s": 0.0},
    ("STORNO", "STORNO"): {"n": 12, "p10": 0.0, "p25": 0.0, "median": 263483.5, "p75": 526973.75, "p90": 526990.7, "share_under_60s": 0.5},
    ("FIN", "CODE NOK"): {"n": 12, "p10": 371461.6, "p25": 443938.75, "median": 1355136.0, "p75": 3591795.75, "p90": 6373977.900000001, "share_under_60s": 0.0},
    ("CODE OK", "STORNO"): {"n": 10, "p10": 3089782.7, "p25": 4453998.0, "median": 8504658.5, "p75": 15467726.5, "p90": 31049188.499999996, "share_under_60s": 0.0},
    ("REOPEN", "RELEASE"): {"n": 9, "p10": 49212.0, "p25": 63815.0, "median": 300349.0, "p75": 377370.0, "p90": 1019041.2000000004, "share_under_60s": 0.0},
    ("MANUAL", "MANUAL"): {"n": 8, "p10": 6.7, "p25": 32.5, "median": 162.0, "p75": 581409.5, "p90": 3156177.7999999993, "share_under_60s": 0.375},
    ("CHANGE DIAGN", "JOIN-PAT"): {"n": 8, "p10": 5154.200000000001, "p25": 14279.75, "median": 596185.5, "p75": 1898689.0, "p90": 4362698.8, "share_under_60s": 0.0},
    ("JOIN-PAT", "FIN"): {"n": 8, "p10": 3050054.3000000003, "p25": 4138408.25, "median": 5127431.5, "p75": 6350856.5, "p90": 7059534.8, "share_under_60s": 0.0},
    ("RELEASE", "FIN"): {"n": 8, "p10": 2291.0, "p25": 87809.75, "median": 410920.0, "p75": 7410449.5, "p90": 12657664.899999999, "share_under_60s": 0.0},
    ("REJECT", "RELEASE"): {"n": 8, "p10": 7.7, "p25": 9.5, "median": 35.0, "p75": 235.0, "p90": 3314850.099999998, "share_under_60s": 0.75},
    ("MANUAL", "CODE OK"): {"n": 8, "p10": 31.8, "p25": 65.25, "median": 160.5, "p75": 970.5, "p90": 14081.999999999993, "share_under_60s": 0.25},
    ("NEW", "JOIN-PAT"): {"n": 6, "p10": 1159796.5, "p25": 1457865.0, "median": 9127796.5, "p75": 24589207.25, "p90": 28188016.0, "share_under_60s": 0.0},
    ("REJECT", "SET STATUS"): {"n": 6, "p10": 43.5, "p25": 67.0, "median": 28418.0, "p75": 9107637.0, "p90": 20266604.0, "share_under_60s": 0.333},
    ("RELEASE", "MANUAL"): {"n": 5, "p10": 6706390.4, "p25": 13381733.0, "median": 16995152.0, "p75": 21892838.0, "p90": 21944594.0, "share_under_60s": 0.0},
    ("BILLED", "MANUAL"): {"n": 5, "p10": 4856873.0, "p25": 7738406.0, "median": 8161208.0, "p75": 8170059.0, "p90": 9479404.8, "share_under_60s": 0.0},
    ("CHANGE DIAGN", "SET STATUS"): {"n": 5, "p10": 34510497.0, "p25": 34827864.0, "median": 35155913.0, "p75": 35687330.0, "p90": 36041617.4, "share_under_60s": 0.0},
}

FREQUENT_VARIANTS = [
    ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "BILLED"],
    ["NEW"],
    ["NEW", "FIN", "RELEASE", "CODE OK", "BILLED"],
    ["NEW", "DELETE"],
    ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "BILLED"],
    ["NEW", "FIN", "CODE OK", "BILLED"],
    ["NEW", "CHANGE DIAGN", "RELEASE", "CODE OK", "BILLED"],
    ["NEW", "FIN", "RELEASE", "BILLED"],
    ["NEW", "CHANGE DIAGN", "FIN", "CODE OK", "BILLED"],
    ["NEW", "FIN"],
    ["NEW", "CHANGE DIAGN", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "BILLED"],
    ["NEW", "CHANGE DIAGN", "DELETE"],
    ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "REOPEN", "FIN", "RELEASE", "CODE OK", "BILLED"],
    ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "BILLED", "STORNO", "BILLED", "STORNO", "BILLED"],
    ["NEW", "FIN", "RELEASE", "CODE OK", "RELEASE", "CODE OK", "BILLED"],
    ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "REOPEN", "DELETE"],
    ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "BILLED", "STORNO", "REJECT", "BILLED"],
    ["NEW", "FIN", "RELEASE", "CODE NOK", "SET STATUS"],
    ["NEW", "FIN", "RELEASE", "CODE OK", "REOPEN", "FIN", "RELEASE", "CODE OK", "BILLED"],
    ["NEW", "FIN", "DELETE"],
]

GLOBAL_MEDIAN = 402242.0

# -----------------------------
# Helpers
# -----------------------------

def detect_precision(ts_str: str):
    if "." in ts_str:
        frac = ts_str.split(".")[-1]
        return len(frac)
    return 0

def parse_ts(s):
    return pd.to_datetime(s, errors="coerce")

def td_seconds(sec):
    return pd.Timedelta(seconds=float(sec))

def round_frac(nanos, digits):
    if digits <= 0:
        return None
    base = 10 ** (9 - digits)
    return int((Decimal(nanos) / Decimal(base)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def format_ts_like(ts, digits):
    ts = pd.Timestamp(ts)
    if digits == 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    frac_val = round_frac(ts.nanosecond + ts.microsecond * 1000, digits)
    frac = f"{frac_val:0{digits}d}"
    return ts.strftime("%Y-%m-%d %H:%M:%S") + "." + frac

def min_step(digits):
    if digits <= 0:
        return pd.Timedelta(seconds=1)
    return pd.Timedelta(seconds=10 ** (-digits))

def get_profile(a, b):
    return DURATION_PROFILES.get((a, b))

def dfg_edge_exists(a, b):
    return (a, b) in DURATION_PROFILES

def contiguous_supported(seq):
    for v in FREQUENT_VARIANTS:
        m = len(seq)
        for i in range(len(v) - m + 1):
            if v[i:i+m] == seq:
                return True
    return False

def edge_frequency(a, b):
    p = get_profile(a, b)
    return 0 if p is None else p["n"]

def choose_spacing(a, b):
    p = get_profile(a, b)
    if p is None:
        return GLOBAL_MEDIAN
    return p["p25"]

def plausibility_cost(delta_sec, a, b):
    p = get_profile(a, b)
    if p is None:
        return 5.0 + abs(delta_sec - GLOBAL_MEDIAN) / max(GLOBAL_MEDIAN, 1.0)
    p10, p25, med, p75, p90 = p["p10"], p["p25"], p["median"], p["p75"], p["p90"]
    if delta_sec < 0:
        return 1e9
    if p10 <= delta_sec <= p90:
        center = p25
        scale = max((p90 - p10), 1.0)
        return abs(delta_sec - center) / scale
    dist = min(abs(delta_sec - p10), abs(delta_sec - p90))
    scale = max((p90 - p10), 1.0)
    return 2.0 + dist / scale

def safe_case_bounds(times):
    valid = [t for t in times if pd.notna(t)]
    if not valid:
        return None, None
    return min(valid), max(valid)

def nearest_dist_seconds(t1, t2):
    return abs((t1 - t2).total_seconds())

def clamp_timestamp(ts, lo, hi):
    if lo is not None and ts < lo:
        ts = lo
    if hi is not None and ts > hi:
        ts = hi
    return ts

def make_different(ts, original_str, digits, lo=None, hi=None, prefer_after=True):
    out = format_ts_like(ts, digits)
    if out != original_str:
        return ts
    step = min_step(digits)
    candidates = []
    if prefer_after:
        candidates = [ts + step, ts - step]
    else:
        candidates = [ts - step, ts + step]
    for cand in candidates:
        cand = clamp_timestamp(cand, lo, hi)
        if format_ts_like(cand, digits) != original_str:
            return cand
    return ts

def estimate_from_anchors(prev_act, prev_t, curr_obs, curr_act, next_act, next_t, digits):
    left_est = None
    right_est = None
    lo = prev_t
    hi = next_t
    if prev_t is not None and prev_act is not None:
        left_est = prev_t + td_seconds(choose_spacing(prev_act, curr_act))
    if next_t is not None and next_act is not None:
        right_est = next_t - td_seconds(choose_spacing(curr_act, next_act))

    if left_est is not None and right_est is not None:
        low_bound = prev_t if prev_t is not None else min(left_est, right_est)
        high_bound = next_t if next_t is not None else max(left_est, right_est)
        if left_est <= right_est:
            target = curr_obs
            target = clamp_timestamp(target, left_est, right_est)
            target = clamp_timestamp(target, low_bound, high_bound)
            return target
        cand = curr_obs
        cand = clamp_timestamp(cand, low_bound, high_bound)
        if prev_t is not None and cand < prev_t:
            cand = prev_t
        if next_t is not None and cand > next_t:
            cand = next_t
        if prev_t is not None and next_t is not None:
            midpoint = prev_t + (next_t - prev_t) / 2
            if cand < prev_t or cand > next_t:
                cand = midpoint
        return cand

    if left_est is not None:
        cand = left_est
        cand = clamp_timestamp(cand, lo, hi)
        return cand
    if right_est is not None:
        cand = right_est
        cand = clamp_timestamp(cand, lo, hi)
        return cand

    return curr_obs

def local_order_score(acts):
    score = 0.0
    for a, b in zip(acts[:-1], acts[1:]):
        if not dfg_edge_exists(a, b):
            return -1e9
        score += np.log1p(edge_frequency(a, b))
    if contiguous_supported(acts):
        score += 10.0
    return score

# -----------------------------
# Core repair
# -----------------------------

def repair_case(case_df, repaired_strings, classifications, fix_msgs):
    idxs = case_df.index.tolist()
    acts = case_df["Activity"].tolist()
    cats = case_df["error_category"].tolist()
    orig_ts_strs = case_df["Timestamp"].tolist()
    parsed = [parse_ts(x) for x in orig_ts_strs]
    precisions = [detect_precision(x) for x in orig_ts_strs]

    case_start, case_end = safe_case_bounds(parsed)

    n = len(case_df)
    i = 1
    while i < n:
        if cats[i] != "CLEAN" and parsed[i] < parsed[i - 1]:
            j_prev = i - 1
            j = i

            prev_act = acts[j_prev - 1] if j_prev - 1 >= 0 else None
            prev_t = parsed[j_prev - 1] if j_prev - 1 >= 0 else None
            a1, a2 = acts[j_prev], acts[j]
            t1_obs, t2_obs = parsed[j_prev], parsed[j]
            next_act = acts[j + 1] if j + 1 < n else None
            next_t = parsed[j + 1] if j + 1 < n else None

            # Candidate 1: exchanged
            exchanged_valid = True
            ex_t1, ex_t2 = t2_obs, t1_obs

            if ex_t1 > ex_t2:
                exchanged_valid = False
            if prev_t is not None and ex_t1 < prev_t:
                exchanged_valid = False
            if next_t is not None and ex_t2 > next_t:
                exchanged_valid = False

            seq = acts[max(0, j_prev - 1): min(n, j + 2)]
            seq_score = local_order_score(seq)
            if seq_score <= -1e8:
                exchanged_valid = False

            ex_cost = 1e18
            if exchanged_valid:
                ex_cost = 0.0
                if prev_t is not None and prev_act is not None:
                    ex_cost += plausibility_cost((ex_t1 - prev_t).total_seconds(), prev_act, a1)
                ex_cost += plausibility_cost((ex_t2 - ex_t1).total_seconds(), a1, a2)
                if next_t is not None and next_act is not None:
                    ex_cost += plausibility_cost((next_t - ex_t2).total_seconds(), a2, next_act)
                ex_cost += 0.000001 * (
                    nearest_dist_seconds(ex_t1, t1_obs) + nearest_dist_seconds(ex_t2, t2_obs)
                )

            # Candidate 2A: displaced current row only
            disp2_t2 = estimate_from_anchors(a1, t1_obs, t2_obs, a2, next_act, next_t, precisions[j])
            disp2_t2 = clamp_timestamp(disp2_t2, t1_obs, next_t if next_t is not None else case_end)
            disp2_t2 = make_different(disp2_t2, orig_ts_strs[j], precisions[j], t1_obs, next_t if next_t is not None else case_end, prefer_after=True)
            disp2_valid = disp2_t2 >= t1_obs and (next_t is None or disp2_t2 <= next_t)
            disp2_cost = 1e18
            if disp2_valid:
                disp2_cost = 0.0
                ex_seq = acts[max(0, j_prev - 1): min(n, j + 2)]
                if local_order_score(ex_seq) <= -1e8:
                    disp2_valid = False
                else:
                    if prev_t is not None and prev_act is not None:
                        disp2_cost += plausibility_cost((t1_obs - prev_t).total_seconds(), prev_act, a1)
                    disp2_cost += plausibility_cost((disp2_t2 - t1_obs).total_seconds(), a1, a2)
                    if next_t is not None and next_act is not None:
                        disp2_cost += plausibility_cost((next_t - disp2_t2).total_seconds(), a2, next_act)
                    disp2_cost += 0.000001 * nearest_dist_seconds(disp2_t2, t2_obs)

            # Candidate 2B: displaced previous row only
            disp1_t1 = estimate_from_anchors(prev_act, prev_t, t1_obs, a1, a2, t2_obs, precisions[j_prev])
            disp1_t1 = clamp_timestamp(disp1_t1, prev_t if prev_t is not None else case_start, t2_obs)
            disp1_t1 = make_different(disp1_t1, orig_ts_strs[j_prev], precisions[j_prev], prev_t if prev_t is not None else case_start, t2_obs, prefer_after=False)
            disp1_valid = disp1_t1 <= t2_obs and (prev_t is None or disp1_t1 >= prev_t)
            disp1_cost = 1e18
            if disp1_valid:
                ex_seq = acts[max(0, j_prev - 1): min(n, j + 2)]
                if local_order_score(ex_seq) <= -1e8:
                    disp1_valid = False
                else:
                    disp1_cost = 0.0
                    if prev_t is not None and prev_act is not None:
                        disp1_cost += plausibility_cost((disp1_t1 - prev_t).total_seconds(), prev_act, a1)
                    disp1_cost += plausibility_cost((t2_obs - disp1_t1).total_seconds(), a1, a2)
                    if next_t is not None and next_act is not None:
                        disp1_cost += plausibility_cost((next_t - t2_obs).total_seconds(), a2, next_act)
                    disp1_cost += 0.000001 * nearest_dist_seconds(disp1_t1, t1_obs)

            chosen = None
            if exchanged_valid and ex_cost <= min(disp1_cost, disp2_cost):
                chosen = ("EXCHANGED", ex_t1, ex_t2)
            else:
                if disp1_valid and disp2_valid:
                    chosen = ("DISPLACED_PREV", disp1_t1, t2_obs) if disp1_cost <= disp2_cost else ("DISPLACED_CURR", t1_obs, disp2_t2)
                elif disp1_valid:
                    chosen = ("DISPLACED_PREV", disp1_t1, t2_obs)
                elif disp2_valid:
                    chosen = ("DISPLACED_CURR", t1_obs, disp2_t2)
                elif exchanged_valid:
                    chosen = ("EXCHANGED", ex_t1, ex_t2)
                else:
                    # last-resort bounded fixes, never observed unchanged for corrupted row
                    fallback_t2 = t1_obs + min_step(precisions[j])
                    fallback_t2 = clamp_timestamp(fallback_t2, t1_obs, next_t if next_t is not None else case_end)
                    fallback_t2 = make_different(fallback_t2, orig_ts_strs[j], precisions[j], t1_obs, next_t if next_t is not None else case_end, prefer_after=True)
                    chosen = ("DISPLACED_CURR", t1_obs, fallback_t2)
                    fix_msgs.append(f"fallback_reestimate case={case_df['Case'].iloc[0]} row_id={case_df['row_id'].iloc[j]}")

            if chosen[0] == "EXCHANGED":
                s1 = orig_ts_strs[j]
                s2 = orig_ts_strs[j_prev]
                repaired_strings[idxs[j_prev]] = s1
                repaired_strings[idxs[j]] = s2
                classifications.append(("EXCHANGED", idxs[j_prev], idxs[j]))
                parsed[j_prev], parsed[j] = parse_ts(s1), parse_ts(s2)
            elif chosen[0] == "DISPLACED_PREV":
                new_t1 = chosen[1]
                new_t1 = clamp_timestamp(new_t1, prev_t if prev_t is not None else case_start, t2_obs)
                new_t1 = make_different(new_t1, orig_ts_strs[j_prev], precisions[j_prev], prev_t if prev_t is not None else case_start, t2_obs, prefer_after=False)
                repaired_strings[idxs[j_prev]] = format_ts_like(new_t1, precisions[j_prev])
                classifications.append(("DISPLACED", idxs[j_prev], idxs[j]))
                parsed[j_prev] = new_t1
            else:
                new_t2 = chosen[2]
                new_t2 = clamp_timestamp(new_t2, t1_obs, next_t if next_t is not None else case_end)
                new_t2 = make_different(new_t2, orig_ts_strs[j], precisions[j], t1_obs, next_t if next_t is not None else case_end, prefer_after=True)
                repaired_strings[idxs[j]] = format_ts_like(new_t2, precisions[j])
                classifications.append(("DISPLACED", idxs[j_prev], idxs[j]))
                parsed[j] = new_t2
        i += 1

def self_check_and_fix(df, repaired_strings, classifications, fix_msgs):
    parsed_out = [parse_ts(s) for s in repaired_strings]
    parsed_in = [parse_ts(s) for s in df["Timestamp"].tolist()]
    precisions = [detect_precision(x) for x in df["Timestamp"].tolist()]

    # CLEAN rows unchanged exactly
    for idx, row in df.iterrows():
        if row["error_category"] == "CLEAN" and repaired_strings[idx] != row["Timestamp"]:
            repaired_strings[idx] = row["Timestamp"]
            parsed_out[idx] = parse_ts(repaired_strings[idx])
            fix_msgs.append(f"restore_clean row_id={row['row_id']}")

    # Case-wise monotonicity and anchor bounds
    for case, cdf in df.groupby("Case", sort=False):
        idxs = cdf.index.tolist()
        case_times = [parsed_out[i] for i in idxs if pd.notna(parsed_out[i])]
        if not case_times:
            continue
        case_start = min(case_times)
        case_end = max(case_times)

        for pos, idx in enumerate(idxs):
            row = df.loc[idx]
            if row["error_category"] == "CLEAN":
                continue

            lo = case_start
            hi = case_end
            if pos - 1 >= 0:
                lo = max(lo, parsed_out[idxs[pos - 1]])
            if pos + 1 < len(idxs):
                hi = min(hi, parsed_out[idxs[pos + 1]])

            t = parsed_out[idx]
            t = clamp_timestamp(t, lo, hi)
            t = make_different(t, df.loc[idx, "Timestamp"], precisions[idx], lo, hi, prefer_after=True)
            repaired_strings[idx] = format_ts_like(t, precisions[idx])
            parsed_out[idx] = parse_ts(repaired_strings[idx])

        # enforce non-decreasing final pass
        for pos in range(1, len(idxs)):
            cur_idx = idxs[pos]
            prev_idx = idxs[pos - 1]
            if parsed_out[cur_idx] < parsed_out[prev_idx]:
                if df.loc[cur_idx, "error_category"] != "CLEAN":
                    t = parsed_out[prev_idx]
                    next_bound = parsed_out[idxs[pos + 1]] if pos + 1 < len(idxs) else case_end
                    t = clamp_timestamp(t, parsed_out[prev_idx], next_bound)
                    t = make_different(t, df.loc[cur_idx, "Timestamp"], precisions[cur_idx], parsed_out[prev_idx], next_bound, prefer_after=True)
                    repaired_strings[cur_idx] = format_ts_like(t, precisions[cur_idx])
                    parsed_out[cur_idx] = parse_ts(repaired_strings[cur_idx])
                    fix_msgs.append(f"monotonic_fix row_id={df.loc[cur_idx, 'row_id']}")
                elif df.loc[prev_idx, "error_category"] != "CLEAN":
                    prev_prev = parsed_out[idxs[pos - 2]] if pos - 2 >= 0 else case_start
                    t = parsed_out[cur_idx]
                    t = clamp_timestamp(t, prev_prev, parsed_out[cur_idx])
                    t = make_different(t, df.loc[prev_idx, "Timestamp"], precisions[prev_idx], prev_prev, parsed_out[cur_idx], prefer_after=False)
                    repaired_strings[prev_idx] = format_ts_like(t, precisions[prev_idx])
                    parsed_out[prev_idx] = parse_ts(repaired_strings[prev_idx])
                    fix_msgs.append(f"monotonic_fix row_id={df.loc[prev_idx, 'row_id']}")

    # Every non-clean row differs from input
    for idx, row in df.iterrows():
        if row["error_category"] != "CLEAN" and repaired_strings[idx] == row["Timestamp"]:
            t = parsed_out[idx]
            case = row["Case"]
            cdf = df[df["Case"] == case]
            idxs = cdf.index.tolist()
            pos = idxs.index(idx)
            lo = parsed_out[idxs[pos - 1]] if pos - 1 >= 0 else t
            hi = parsed_out[idxs[pos + 1]] if pos + 1 < len(idxs) else t
            if hi < lo:
                hi = lo
            t2 = make_different(t, row["Timestamp"], precisions[idx], lo, hi, prefer_after=True)
            repaired_strings[idx] = format_ts_like(t2, precisions[idx])
            parsed_out[idx] = parse_ts(repaired_strings[idx])
            fix_msgs.append(f"force_change row_id={row['row_id']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    exchanged_count = 0
    displaced_count = 0
    fix_msgs = []

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        original_columns = list(df.columns)

        repaired_strings = df["Timestamp"].tolist()

        classifications = []
        for _, case_df in df.groupby("Case", sort=False):
            repair_case(case_df, repaired_strings, classifications, fix_msgs)

        self_check_and_fix(df, repaired_strings, classifications, fix_msgs)

        # Count classifications
        for kind, _, _ in classifications:
            if kind == "EXCHANGED":
                exchanged_count += 1
            elif kind == "DISPLACED":
                displaced_count += 1

        rows_equal = int(sum(1 for i, s in enumerate(repaired_strings) if s == df.iloc[i]["Timestamp"]))

        out_df = df.copy()
        out_df["Timestamp"] = repaired_strings

        # Preserve all untouched byte-identical columns and row order
        out_df = out_df[original_columns]
        out_df.to_csv(args.output, index=False)

        print(f"exchanged={exchanged_count} displaced={displaced_count} rows_equal_input={rows_equal}")
        for msg in fix_msgs:
            print(msg)
    except Exception as e:
        # Must not exit non-zero; write a safe output preserving structure
        try:
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            rows_equal = len(df)
            print(f"exchanged=0 displaced=0 rows_equal_input={rows_equal}")
            print(f"fatal_recovered={str(e)}")
        except Exception as e2:
            # Last-ditch: produce empty-compatible file if impossible to read
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write("Case,Activity,Timestamp,Resource,error_category,row_id\n")
                print("exchanged=0 displaced=0 rows_equal_input=0")
                print(f"fatal_recovered={str(e)}")
                print(f"output_stub_written={str(e2)}")
            except Exception:
                print("exchanged=0 displaced=0 rows_equal_input=0")
                print(f"fatal_recovered={str(e)}")

if __name__ == "__main__":
    main()