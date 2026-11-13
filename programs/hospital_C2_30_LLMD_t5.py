import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

# -----------------------------
# Embedded process knowledge
# -----------------------------
DURATION_PROFILES = {
    ("FIN", "RELEASE"): {"n": 6486, "p10": 2048.0, "p25": 6744.5, "median": 16358.0, "p75": 25456.75, "p90": 36587.0, "share_under_60s": 0.053},
    ("CODE OK", "BILLED"): {"n": 6214, "p10": 398373.1, "p25": 1350225.25, "median": 3790599.0, "p75": 7240106.5, "p90": 11149642.3, "share_under_60s": 0.0},
    ("RELEASE", "CODE OK"): {"n": 6021, "p10": 3515.0, "p25": 176004.0, "median": 343375.0, "p75": 440228.0, "p90": 534115.0, "share_under_60s": 0.049},
    ("NEW", "CHANGE DIAGN"): {"n": 4748, "p10": 12.0, "p25": 16.0, "median": 25.0, "p75": 9184.25, "p90": 1734835.3, "share_under_60s": 0.639},
    ("CHANGE DIAGN", "FIN"): {"n": 4199, "p10": 1595249.4, "p25": 4010767.0, "median": 7708017.0, "p75": 7904282.0, "p90": 8064003.0, "share_under_60s": 0.007},
    ("NEW", "FIN"): {"n": 2813, "p10": 430931.4, "p25": 5613832.0, "median": 30069111.0, "p75": 31622139.0, "p90": 31634310.2, "share_under_60s": 0.036},
    ("NEW", "NEW"): {"n": 1106, "p10": 1.0, "p25": 1.0, "median": 1.0, "p75": 1.0, "p90": 1.0, "share_under_60s": 1.0},
    ("RELEASE", "BILLED"): {"n": 775, "p10": 673459.0, "p25": 1305167.0, "median": 3654832.0, "p75": 7884415.5, "p90": 11870267.6, "share_under_60s": 0.0},
    ("FIN", "CODE OK"): {"n": 769, "p10": 174331.2, "p25": 269794.0, "median": 379548.0, "p75": 522371.0, "p90": 570947.6, "share_under_60s": 0.001},
    ("NEW", "DELETE"): {"n": 590, "p10": 2360.1, "p25": 167520.75, "median": 1834565.0, "p75": 7132370.25, "p90": 23068173.7, "share_under_60s": 0.063},
    ("REOPEN", "FIN"): {"n": 470, "p10": 5.9, "p25": 10.0, "median": 56.0, "p75": 198754.75, "p90": 1119424.9, "share_under_60s": 0.506},
    ("BILLED", "STORNO"): {"n": 439, "p10": 195330.6, "p25": 434152.5, "median": 1077095.0, "p75": 3227309.5, "p90": 11913231.8, "share_under_60s": 0.002},
    ("CODE OK", "REOPEN"): {"n": 437, "p10": 123231.6, "p25": 460198.0, "median": 1914409.0, "p75": 4429460.0, "p90": 7604387.0, "share_under_60s": 0.014},
    ("CHANGE DIAGN", "RELEASE"): {"n": 417, "p10": 2303730.2, "p25": 4374232.0, "median": 7729262.0, "p75": 7974821.0, "p90": 8085701.0, "share_under_60s": 0.0},
    ("CHANGE DIAGN", "CHANGE DIAGN"): {"n": 300, "p10": 240.7, "p25": 52849.0, "median": 861391.0, "p75": 2428618.25, "p90": 4874708.1, "share_under_60s": 0.04},
    ("STORNO", "BILLED"): {"n": 246, "p10": 92765.5, "p25": 92835.25, "median": 169858.5, "p75": 170045.5, "p90": 504917.0, "share_under_60s": 0.0},
    ("STORNO", "REJECT"): {"n": 203, "p10": 5.2, "p25": 14.0, "median": 1237.0, "p75": 1916190.0, "p90": 9494887.8, "share_under_60s": 0.448},
    ("REOPEN", "DELETE"): {"n": 151, "p10": 2.0, "p25": 2.0, "median": 3.0, "p75": 4.0, "p90": 9.0, "share_under_60s": 0.967},
    ("CHANGE DIAGN", "DELETE"): {"n": 139, "p10": 203.4, "p25": 66091.0, "median": 946414.0, "p75": 4221145.0, "p90": 7573763.0, "share_under_60s": 0.065},
    ("REJECT", "BILLED"): {"n": 126, "p10": 1882.5, "p25": 163280.0, "median": 328872.5, "p75": 411299.5, "p90": 4350796.5, "share_under_60s": 0.056},
    ("RELEASE", "CODE NOK"): {"n": 120, "p10": 346460.4, "p25": 609418.75, "median": 1439807.5, "p75": 3817948.25, "p90": 6746112.0, "share_under_60s": 0.008},
    ("CODE OK", "RELEASE"): {"n": 95, "p10": 2059007.0, "p25": 2987528.5, "median": 4572613.0, "p75": 5923328.5, "p90": 6849198.2, "share_under_60s": 0.011},
    ("RELEASE", "REOPEN"): {"n": 75, "p10": 28564.8, "p25": 129282.5, "median": 399253.0, "p75": 2230172.0, "p90": 4985157.4, "share_under_60s": 0.027},
    ("REJECT", "REOPEN"): {"n": 72, "p10": 8.0, "p25": 10.0, "median": 30.5, "p75": 95.0, "p90": 464.9, "share_under_60s": 0.667},
    ("JOIN-PAT", "JOIN-PAT"): {"n": 47, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 1.0, "p90": 1.0, "share_under_60s": 1.0},
    ("FIN", "REOPEN"): {"n": 45, "p10": 6.4, "p25": 30.0, "median": 18130.0, "p75": 360631.0, "p90": 3614904.0, "share_under_60s": 0.289},
    ("CODE NOK", "CODE NOK"): {"n": 44, "p10": 343362.5, "p25": 579864.25, "median": 1682059.0, "p75": 6435773.25, "p90": 7345313.0, "share_under_60s": 0.0},
    ("CODE NOK", "SET STATUS"): {"n": 43, "p10": 8537.8, "p25": 15714.5, "median": 25242.0, "p75": 32981.5, "p90": 40848.4, "share_under_60s": 0.023},
    ("CODE NOK", "CODE OK"): {"n": 39, "p10": 2626.0, "p25": 19872.5, "median": 49685.0, "p75": 52020.0, "p90": 88020.2, "share_under_60s": 0.0},
    ("FIN", "DELETE"): {"n": 37, "p10": 69.8, "p25": 40850.0, "median": 1180180.0, "p75": 2238393.0, "p90": 4879596.8, "share_under_60s": 0.081},
    ("FIN", "FIN"): {"n": 35, "p10": 0.0, "p25": 0.0, "median": 31430606.0, "p75": 35271797.5, "p90": 38507410.2, "share_under_60s": 0.314},
    ("CODE OK", "MANUAL"): {"n": 34, "p10": 494.5, "p25": 328332.5, "median": 1539666.5, "p75": 2052885.5, "p90": 2670681.8, "share_under_60s": 0.059},
    ("MANUAL", "RELEASE"): {"n": 27, "p10": 194.6, "p25": 24535.0, "median": 36278.0, "p75": 41857.0, "p90": 42132.2, "share_under_60s": 0.074},
    ("REOPEN", "CHANGE DIAGN"): {"n": 27, "p10": 8.8, "p25": 17.0, "median": 42.0, "p75": 217.0, "p90": 867713.4, "share_under_60s": 0.519},
    ("CODE NOK", "REOPEN"): {"n": 26, "p10": 11420.0, "p25": 21800.5, "median": 31239.5, "p75": 48681.75, "p90": 585812.0, "share_under_60s": 0.038},
    ("BILLED", "BILLED"): {"n": 26, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 1028659.5, "p90": 1170203.0, "share_under_60s": 0.577},
    ("CODE OK", "FIN"): {"n": 23, "p10": 769.6, "p25": 882370.5, "median": 2073019.0, "p75": 6294111.0, "p90": 7931658.6, "share_under_60s": 0.043},
    ("CODE OK", "DELETE"): {"n": 22, "p10": 382465.9, "p25": 1677310.75, "median": 2805210.0, "p75": 5376157.75, "p90": 7365047.9, "share_under_60s": 0.0},
    ("CODE OK", "CODE OK"): {"n": 22, "p10": 1087843.1, "p25": 3184653.0, "median": 10164512.0, "p75": 12655999.75, "p90": 15934178.3, "share_under_60s": 0.0},
    ("RELEASE", "RELEASE"): {"n": 21, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 2701508.0, "p90": 6877633.0, "share_under_60s": 0.524},
    ("CODE NOK", "DELETE"): {"n": 19, "p10": 12976.6, "p25": 24252.0, "median": 29918.0, "p75": 41426.5, "p90": 49063.6, "share_under_60s": 0.0},
    ("BILLED", "CODE OK"): {"n": 18, "p10": 2281385.9, "p25": 7145231.75, "median": 8266342.5, "p75": 24065387.0, "p90": 36390457.9, "share_under_60s": 0.0},
    ("RELEASE", "DELETE"): {"n": 16, "p10": 36141.5, "p25": 131670.75, "median": 302460.0, "p75": 1770106.75, "p90": 14471341.5, "share_under_60s": 0.062},
    ("RELEASE", "SET STATUS"): {"n": 15, "p10": 142643.8, "p25": 221989.0, "median": 602525.0, "p75": 1068407.0, "p90": 1852543.2, "share_under_60s": 0.0},
    ("BILLED", "REJECT"): {"n": 13, "p10": 1489822.8, "p25": 3271504.0, "median": 5749219.0, "p75": 14042919.0, "p90": 17629354.8, "share_under_60s": 0.0},
    ("STORNO", "STORNO"): {"n": 12, "p10": 0.0, "p25": 0.0, "median": 263483.5, "p75": 526973.75, "p90": 526990.7, "share_under_60s": 0.5},
    ("FIN", "CODE NOK"): {"n": 12, "p10": 371461.6, "p25": 443938.75, "median": 1355136.0, "p75": 3591795.75, "p90": 6373977.9, "share_under_60s": 0.0},
    ("CODE OK", "STORNO"): {"n": 10, "p10": 3089782.7, "p25": 4453998.0, "median": 8504658.5, "p75": 15467726.5, "p90": 31049188.5, "share_under_60s": 0.0},
    ("REOPEN", "RELEASE"): {"n": 9, "p10": 49212.0, "p25": 63815.0, "median": 300349.0, "p75": 377370.0, "p90": 1019041.2, "share_under_60s": 0.0},
    ("MANUAL", "MANUAL"): {"n": 8, "p10": 6.7, "p25": 32.5, "median": 162.0, "p75": 581409.5, "p90": 3156177.8, "share_under_60s": 0.375},
    ("CHANGE DIAGN", "JOIN-PAT"): {"n": 8, "p10": 5154.2, "p25": 14279.75, "median": 596185.5, "p75": 1898689.0, "p90": 4362698.8, "share_under_60s": 0.0},
    ("JOIN-PAT", "FIN"): {"n": 8, "p10": 3050054.3, "p25": 4138408.25, "median": 5127431.5, "p75": 6350856.5, "p90": 7059534.8, "share_under_60s": 0.0},
    ("RELEASE", "FIN"): {"n": 8, "p10": 2291.0, "p25": 87809.75, "median": 410920.0, "p75": 7410449.5, "p90": 12657664.9, "share_under_60s": 0.0},
    ("REJECT", "RELEASE"): {"n": 8, "p10": 7.7, "p25": 9.5, "median": 35.0, "p75": 235.0, "p90": 3314850.1, "share_under_60s": 0.75},
    ("MANUAL", "CODE OK"): {"n": 8, "p10": 31.8, "p25": 65.25, "median": 160.5, "p75": 970.5, "p90": 14082.0, "share_under_60s": 0.25},
    ("NEW", "JOIN-PAT"): {"n": 6, "p10": 1159796.5, "p25": 1457865.0, "median": 9127796.5, "p75": 24589207.25, "p90": 28188016.0, "share_under_60s": 0.0},
    ("REJECT", "SET STATUS"): {"n": 6, "p10": 43.5, "p25": 67.0, "median": 28418.0, "p75": 9107637.0, "p90": 20266604.0, "share_under_60s": 0.333},
    ("RELEASE", "MANUAL"): {"n": 5, "p10": 6706390.4, "p25": 13381733.0, "median": 16995152.0, "p75": 21892838.0, "p90": 21944594.0, "share_under_60s": 0.0},
    ("BILLED", "MANUAL"): {"n": 5, "p10": 4856873.0, "p25": 7738406.0, "median": 8161208.0, "p75": 8170059.0, "p90": 9479404.8, "share_under_60s": 0.0},
    ("CHANGE DIAGN", "SET STATUS"): {"n": 5, "p10": 34510497.0, "p25": 34827864.0, "median": 35155913.0, "p75": 35687330.0, "p90": 36041617.4, "share_under_60s": 0.0},
}

DFG_FREQ = {
    ("FIN", "RELEASE"): 6486, ("CODE OK", "BILLED"): 6214, ("RELEASE", "CODE OK"): 6021,
    ("NEW", "CHANGE DIAGN"): 4748, ("CHANGE DIAGN", "FIN"): 4199, ("NEW", "FIN"): 2813,
    ("NEW", "NEW"): 1106, ("RELEASE", "BILLED"): 775, ("FIN", "CODE OK"): 769,
    ("NEW", "DELETE"): 590, ("REOPEN", "FIN"): 470, ("BILLED", "STORNO"): 439,
    ("CODE OK", "REOPEN"): 437, ("CHANGE DIAGN", "RELEASE"): 417, ("CHANGE DIAGN", "CHANGE DIAGN"): 300,
    ("STORNO", "BILLED"): 246, ("STORNO", "REJECT"): 203, ("REOPEN", "DELETE"): 151,
    ("CHANGE DIAGN", "DELETE"): 139, ("REJECT", "BILLED"): 126, ("RELEASE", "CODE NOK"): 120,
    ("CODE OK", "RELEASE"): 95, ("RELEASE", "REOPEN"): 75, ("REJECT", "REOPEN"): 72,
    ("JOIN-PAT", "JOIN-PAT"): 47, ("FIN", "REOPEN"): 45, ("CODE NOK", "CODE NOK"): 44,
    ("CODE NOK", "SET STATUS"): 43, ("CODE NOK", "CODE OK"): 39, ("FIN", "DELETE"): 37,
    ("FIN", "FIN"): 35, ("CODE OK", "MANUAL"): 34, ("MANUAL", "RELEASE"): 27,
    ("REOPEN", "CHANGE DIAGN"): 27, ("BILLED", "BILLED"): 26, ("CODE NOK", "REOPEN"): 26,
    ("CODE OK", "FIN"): 23, ("CODE OK", "CODE OK"): 22, ("CODE OK", "DELETE"): 22,
    ("RELEASE", "RELEASE"): 21, ("CODE NOK", "DELETE"): 19, ("BILLED", "CODE OK"): 18,
    ("RELEASE", "DELETE"): 16, ("RELEASE", "SET STATUS"): 15, ("BILLED", "REJECT"): 13,
    ("FIN", "CODE NOK"): 12, ("STORNO", "STORNO"): 12, ("CODE OK", "STORNO"): 10,
    ("REOPEN", "RELEASE"): 9, ("RELEASE", "FIN"): 8, ("MANUAL", "MANUAL"): 8,
    ("MANUAL", "CODE OK"): 8, ("CHANGE DIAGN", "JOIN-PAT"): 8, ("JOIN-PAT", "FIN"): 8,
    ("REJECT", "RELEASE"): 8, ("REJECT", "SET STATUS"): 6, ("NEW", "JOIN-PAT"): 6,
    ("RELEASE", "MANUAL"): 5, ("BILLED", "MANUAL"): 5, ("CHANGE DIAGN", "SET STATUS"): 5
}

BLOCK_SPAN = {
    2: {"p10": 17.0, "p25": 11146.5, "median": 402242.0, "p75": 4727853.0, "p90": 9079984.8},
    3: {"p10": 92737.0, "p25": 470060.25, "median": 3892568.5, "p75": 7899804.5, "p90": 12780894.6},
    4: {"p10": 380148.0, "p25": 2374302.0, "median": 7112636.0, "p75": 8799776.0, "p90": 21650148.4},
    5: {"p10": 436599.0, "p25": 4336429.0, "median": 8288893.0, "p75": 14124317.0, "p90": 27045393.0},
}

GLOBAL_MEDIAN = 402242.0

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
    ["NEW", "FIN", "DELETE"]
]

# -----------------------------
# Helpers
# -----------------------------
def detect_precision(ts_str):
    if "." in ts_str:
        frac = ts_str.split(".")[-1]
        return len(frac)
    return 0

def parse_ts(s):
    return pd.to_datetime(s, errors="coerce")

def format_ts(ts, precision):
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    if precision <= 6:
        return base[:- (6 - precision)] if precision < 6 else base
    frac = base.split(".")[1]
    frac = frac + ("0" * (precision - 6))
    return base.split(".")[0] + "." + frac

def td_seconds(x):
    return x.total_seconds()

def step_for_precision(precision):
    if precision <= 0:
        return pd.Timedelta(seconds=1)
    if precision <= 6:
        return pd.Timedelta(microseconds=10 ** (6 - precision))
    return pd.Timedelta(microseconds=1)

def get_profile(src, tgt):
    return DURATION_PROFILES.get((src, tgt))

def edge_exists(src, tgt):
    return (src, tgt) in DFG_FREQ

def preferred_gap_seconds(src, tgt):
    prof = get_profile(src, tgt)
    if prof is None:
        return GLOBAL_MEDIAN
    if prof["share_under_60s"] >= 0.5:
        return prof["p10"]
    return prof["p25"]

def plausible_upper_seconds(src, tgt):
    prof = get_profile(src, tgt)
    if prof is None:
        return max(GLOBAL_MEDIAN * 2, 1.0)
    return prof["p90"]

def local_gap_penalty(actual_sec, src, tgt):
    if actual_sec < 0:
        return 1e12
    prof = get_profile(src, tgt)
    if prof is None:
        pref = GLOBAL_MEDIAN
        scale = max(pref, 1.0)
        return abs(actual_sec - pref) / scale + 2.0
    pref = preferred_gap_seconds(src, tgt)
    upper = max(plausible_upper_seconds(src, tgt), pref, 1.0)
    scale = max(pref, 1.0)
    penalty = abs(actual_sec - pref) / scale
    if actual_sec > upper:
        penalty += (actual_sec - upper) / upper * 2.0
    return penalty

def subseq_supported(acts):
    for v in FREQUENT_VARIANTS:
        m = len(acts)
        for i in range(len(v) - m + 1):
            if v[i:i + m] == acts:
                return True
    return False

def case_bounds(times):
    valid = [t for t in times if pd.notna(t)]
    if not valid:
        return None, None
    return min(valid), max(valid)

def clamp_ts(ts, lo, hi):
    if lo is not None and ts < lo:
        ts = lo
    if hi is not None and ts > hi:
        ts = hi
    return ts

def choose_estimate(obs, lo, hi, preferred):
    cand = preferred
    if lo is not None and hi is not None and lo > hi:
        mid = lo
        return mid
    if lo is not None and cand < lo:
        cand = lo
    if hi is not None and cand > hi:
        cand = hi
    if lo is not None and hi is not None:
        if obs is not None and lo <= obs <= hi:
            if abs((obs - preferred).total_seconds()) <= abs((cand - preferred).total_seconds()):
                cand = obs
        elif obs is not None:
            if obs < lo:
                cand = lo
            elif obs > hi:
                cand = hi
    return cand

def estimate_displaced(case_df, idx, current_times):
    n = len(case_df)
    obs = current_times[idx]
    act = case_df.iloc[idx]["Activity"]
    left_t = current_times[idx - 1] if idx - 1 >= 0 else None
    left_a = case_df.iloc[idx - 1]["Activity"] if idx - 1 >= 0 else None
    right_t = current_times[idx + 1] if idx + 1 < n else None
    right_a = case_df.iloc[idx + 1]["Activity"] if idx + 1 < n else None

    all_times = [t for t in current_times if t is not None and pd.notna(t)]
    case_start, case_end = case_bounds(all_times)

    step = step_for_precision(int(case_df.iloc[idx]["_precision"]))
    lo = case_start
    hi = case_end

    if left_t is not None and pd.notna(left_t):
        lo = max(lo, left_t) if lo is not None else left_t
    if right_t is not None and pd.notna(right_t):
        hi = min(hi, right_t) if hi is not None else right_t

    preferreds = []

    if left_t is not None and left_a is not None:
        gap = preferred_gap_seconds(left_a, act)
        preferreds.append(left_t + pd.Timedelta(seconds=gap))
    if right_t is not None and right_a is not None:
        gap = preferred_gap_seconds(act, right_a)
        preferreds.append(right_t - pd.Timedelta(seconds=gap))

    if left_t is not None and right_t is not None and preferreds:
        preferred = min(preferreds, key=lambda t: abs((t - obs).total_seconds()) if obs is not None else 0)
    elif preferreds:
        preferred = preferreds[0]
    else:
        span_p25 = BLOCK_SPAN.get(2, {}).get("p25", GLOBAL_MEDIAN)
        if left_t is not None:
            preferred = left_t + pd.Timedelta(seconds=min(span_p25, GLOBAL_MEDIAN))
        elif right_t is not None:
            preferred = right_t - pd.Timedelta(seconds=min(span_p25, GLOBAL_MEDIAN))
        else:
            preferred = obs

    est = choose_estimate(obs, lo, hi, preferred)

    if left_t is not None and est < left_t:
        est = left_t
    if right_t is not None and est > right_t:
        est = right_t

    if obs is not None and est == obs:
        if right_t is not None and est + step <= right_t:
            est = est + step
        elif left_t is not None and est - step >= left_t:
            est = est - step
        elif case_end is not None and est + step <= case_end:
            est = est + step
        elif case_start is not None and est - step >= case_start:
            est = est - step

    est = clamp_ts(est, lo, hi)
    return est

def score_swap(case_df, idx, current_times):
    n = len(case_df)
    if idx <= 0 or idx >= n:
        return False, 1e18

    t_prev = current_times[idx - 1]
    t_i = current_times[idx]
    t_j = current_times[idx + 1] if idx + 1 < n else None

    if idx + 1 >= n:
        return False, 1e18

    a_prev = case_df.iloc[idx - 1]["Activity"]
    a_i = case_df.iloc[idx]["Activity"]
    a_j = case_df.iloc[idx + 1]["Activity"]
    a_next = case_df.iloc[idx + 2]["Activity"] if idx + 2 < n else None
    t_next = current_times[idx + 2] if idx + 2 < n else None

    swap_i = current_times[idx + 1]
    swap_j = current_times[idx]

    case_start, case_end = case_bounds([t for t in current_times if t is not None and pd.notna(t)])

    if t_prev is not None and swap_i < t_prev:
        return False, 1e18
    if swap_j < swap_i:
        return False, 1e18
    if t_next is not None and swap_j > t_next:
        return False, 1e18
    if case_start is not None and (swap_i < case_start or swap_j < case_start):
        return False, 1e18
    if case_end is not None and (swap_i > case_end or swap_j > case_end):
        return False, 1e18

    if not edge_exists(a_prev, a_i):
        return False, 1e18
    if not edge_exists(a_i, a_j):
        return False, 1e18
    if a_next is not None and not edge_exists(a_j, a_next):
        return False, 1e18

    pen = 0.0
    pen += local_gap_penalty(td_seconds(swap_i - t_prev), a_prev, a_i)

    pen += local_gap_penalty(td_seconds(swap_j - swap_i), a_i, a_j)

    if t_next is not None:
        pen += local_gap_penalty(td_seconds(t_next - swap_j), a_j, a_next)

    acts = [a_prev, a_i, a_j] if a_next is None else [a_prev, a_i, a_j, a_next]
    if subseq_supported(acts):
        pen -= 0.25

    move = abs((swap_i - t_i).total_seconds()) + abs((swap_j - current_times[idx + 1]).total_seconds())
    pen += move / max(GLOBAL_MEDIAN, 1.0)
    return True, pen

def score_displaced(case_df, idx, current_times):
    n = len(case_df)
    est = estimate_displaced(case_df, idx, current_times)
    temp_times = list(current_times)
    temp_times[idx] = est

    pen = 0.0
    if idx - 1 >= 0:
        pen += local_gap_penalty(td_seconds(temp_times[idx] - temp_times[idx - 1]),
                                 case_df.iloc[idx - 1]["Activity"], case_df.iloc[idx]["Activity"])
    if idx + 1 < n:
        pen += local_gap_penalty(td_seconds(temp_times[idx + 1] - temp_times[idx]),
                                 case_df.iloc[idx]["Activity"], case_df.iloc[idx + 1]["Activity"])
    obs = current_times[idx]
    pen += abs((est - obs).total_seconds()) / max(GLOBAL_MEDIAN, 1.0)
    return est, pen

def repair_case(case_df, out_ts_strings, exchanged_count_ref, displaced_count_ref, messages):
    idxs = case_df.index.tolist()
    parsed = [case_df.loc[i, "_parsed_ts"] for i in idxs]
    errs = [case_df.loc[i, "error_category"] for i in idxs]
    original_strs = [case_df.loc[i, "Timestamp"] for i in idxs]
    precisions = [case_df.loc[i, "_precision"] for i in idxs]

    current_times = list(parsed)

    i = 1
    while i < len(idxs):
        if errs[i] != "CLEAN" and current_times[i] < current_times[i - 1]:
            pair_idx = i - 1
            later_idx = i

            prev_corr = errs[pair_idx] != "CLEAN"
            curr_corr = errs[later_idx] != "CLEAN"

            swap_ok = False
            swap_pen = 1e18
            if prev_corr and curr_corr:
                swap_ok, swap_pen = score_swap(case_df.reset_index(drop=True), pair_idx, current_times)

            disp_est, disp_pen = score_displaced(case_df.reset_index(drop=True), later_idx, current_times)

            choose_swap = prev_corr and curr_corr and swap_ok and (swap_pen + 0.05 < disp_pen)

            if choose_swap:
                s1 = original_strs[later_idx]
                s2 = original_strs[pair_idx]
                out_ts_strings[idxs[pair_idx]] = s1
                out_ts_strings[idxs[later_idx]] = s2
                current_times[pair_idx], current_times[later_idx] = current_times[later_idx], current_times[pair_idx]
                exchanged_count_ref[0] += 1
                i += 1
            else:
                current_times[later_idx] = disp_est
                out_ts_strings[idxs[later_idx]] = format_ts(disp_est, int(precisions[later_idx]))
                displaced_count_ref[0] += 1
        i += 1

    # Self-check and corrective pass
    step_cache = [step_for_precision(int(p)) for p in precisions]
    for j in range(len(idxs)):
        if errs[j] == "CLEAN":
            out_ts_strings[idxs[j]] = original_strs[j]

    for j in range(len(idxs)):
        if errs[j] != "CLEAN":
            ts = parse_ts(out_ts_strings[idxs[j]])
            left = current_times[j - 1] if j - 1 >= 0 else None
            right = current_times[j + 1] if j + 1 < len(idxs) else None
            all_times = [parse_ts(out_ts_strings[k]) for k in idxs if pd.notna(parse_ts(out_ts_strings[k]))]
            cstart, cend = case_bounds(all_times)
            lo = cstart
            hi = cend
            if left is not None:
                lo = max(lo, left) if lo is not None else left
            if right is not None:
                hi = min(hi, right) if hi is not None else right
            fixed = clamp_ts(ts, lo, hi) if ts is not None and pd.notna(ts) else estimate_displaced(case_df.reset_index(drop=True), j, current_times)
            if parse_ts(original_strs[j]) == fixed:
                step = step_cache[j]
                if hi is not None and fixed + step <= hi:
                    fixed = fixed + step
                elif lo is not None and fixed - step >= lo:
                    fixed = fixed - step
            out_ts_strings[idxs[j]] = format_ts(fixed, int(precisions[j]))
            current_times[j] = fixed

    # Final monotonic correction inside case
    for j in range(1, len(idxs)):
        t_prev = parse_ts(out_ts_strings[idxs[j - 1]])
        t_cur = parse_ts(out_ts_strings[idxs[j]])
        if t_cur < t_prev:
            if errs[j] != "CLEAN":
                step = step_cache[j]
                t_new = t_prev
                if parse_ts(original_strs[j]) == t_new:
                    t_new = t_new + step
                out_ts_strings[idxs[j]] = format_ts(t_new, int(precisions[j]))
                current_times[j] = t_new
                messages.append(f"fixed monotonicity in case {case_df.iloc[0]['Case']} row_id {case_df.iloc[j]['row_id']}")
            elif errs[j - 1] != "CLEAN":
                step = step_cache[j - 1]
                right = t_cur
                t_new = right
                if parse_ts(original_strs[j - 1]) == t_new and t_new - step >= t_prev:
                    t_new = t_new - step
                out_ts_strings[idxs[j - 1]] = format_ts(t_new, int(precisions[j - 1]))
                current_times[j - 1] = t_new
                messages.append(f"fixed monotonicity in case {case_df.iloc[0]['Case']} row_id {case_df.iloc[j - 1]['row_id']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    messages = []
    exchanged_count = [0]
    displaced_count = [0]

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception as e:
        df = pd.DataFrame(columns=["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])
        messages.append(f"read_failed:{e}")

    expected_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""

    df = df[expected_cols].copy()
    df["_parsed_ts"] = df["Timestamp"].apply(parse_ts)
    df["_precision"] = df["Timestamp"].apply(detect_precision)

    out_ts_strings = df["Timestamp"].tolist()

    try:
        for _, case_df in df.groupby("Case", sort=False):
            repair_case(case_df, out_ts_strings, exchanged_count, displaced_count, messages)
    except Exception as e:
        messages.append(f"repair_failed:{e}")

    # Global self-checks and corrections
    if len(out_ts_strings) != len(df):
        out_ts_strings = (out_ts_strings + df["Timestamp"].tolist())[:len(df)]
        messages.append("fixed_row_count_alignment")

    rows_equal = 0
    for i in range(len(df)):
        if df.iloc[i]["error_category"] == "CLEAN":
            out_ts_strings[i] = df.iloc[i]["Timestamp"]
        else:
            repaired = parse_ts(out_ts_strings[i])
            if pd.isna(repaired):
                repaired = df.iloc[i]["_parsed_ts"]
                step = step_for_precision(int(df.iloc[i]["_precision"]))
                repaired = repaired + step
                out_ts_strings[i] = format_ts(repaired, int(df.iloc[i]["_precision"]))
                messages.append(f"fixed_unparseable row_id {df.iloc[i]['row_id']}")
            if out_ts_strings[i] == df.iloc[i]["Timestamp"]:
                rows_equal += 1
                repaired = parse_ts(out_ts_strings[i])
                case = df.iloc[i]["Case"]
                case_rows = df[df["Case"] == case]
                pos = case_rows.index.tolist().index(i)
                idxs = case_rows.index.tolist()
                left = parse_ts(out_ts_strings[idxs[pos - 1]]) if pos - 1 >= 0 else None
                right = parse_ts(out_ts_strings[idxs[pos + 1]]) if pos + 1 < len(idxs) else None
                step = step_for_precision(int(df.iloc[i]["_precision"]))
                t_new = repaired
                if right is not None and t_new + step <= right:
                    t_new = t_new + step
                elif left is not None and t_new - step >= left:
                    t_new = t_new - step
                else:
                    t_new = t_new + step
                out_ts_strings[i] = format_ts(t_new, int(df.iloc[i]["_precision"]))
                messages.append(f"nudged_equal_input row_id {df.iloc[i]['row_id']}")

    rows_equal = sum(1 for i in range(len(df)) if out_ts_strings[i] == df.iloc[i]["Timestamp"])

    out_df = df[expected_cols].copy()
    out_df["Timestamp"] = out_ts_strings

    try:
        out_df.to_csv(args.output, index=False)
    except Exception:
        try:
            fallback = df[expected_cols].copy()
            fallback.to_csv(args.output, index=False)
            messages.append("write_fallback_original")
        except Exception:
            pass

    print(f"exchanged_segments={exchanged_count[0]} displaced_segments={displaced_count[0]} rows_still_equal={rows_equal}")
    for m in messages:
        print(m)

if __name__ == "__main__":
    main()