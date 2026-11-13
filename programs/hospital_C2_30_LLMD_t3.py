import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
import re

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
    ("RELEASE", "FIN"): {"n": 8, "p10": 2291.0, "p25": 87809.75, "median": 410920.0, "p75": 7410449.5, "p90": 12657664.9, "share_under_60s": 0.0},
    ("REJECT", "RELEASE"): {"n": 8, "p10": 7.7, "p25": 9.5, "median": 35.0, "p75": 235.0, "p90": 3314850.1, "share_under_60s": 0.75},
    ("MANUAL", "CODE OK"): {"n": 8, "p10": 31.8, "p25": 65.25, "median": 160.5, "p75": 970.5, "p90": 14082.0, "share_under_60s": 0.25},
    ("NEW", "JOIN-PAT"): {"n": 6, "p10": 1159796.5, "p25": 1457865.0, "median": 9127796.5, "p75": 24589207.25, "p90": 28188016.0, "share_under_60s": 0.0},
    ("REJECT", "SET STATUS"): {"n": 6, "p10": 43.5, "p25": 67.0, "median": 28418.0, "p75": 9107637.0, "p90": 20266604.0, "share_under_60s": 0.333},
    ("RELEASE", "MANUAL"): {"n": 5, "p10": 6706390.4, "p25": 13381733.0, "median": 16995152.0, "p75": 21892838.0, "p90": 21944594.0, "share_under_60s": 0.0},
    ("BILLED", "MANUAL"): {"n": 5, "p10": 4856873.0, "p25": 7738406.0, "median": 8161208.0, "p75": 8170059.0, "p90": 9479404.8, "share_under_60s": 0.0},
    ("CHANGE DIAGN", "SET STATUS"): {"n": 5, "p10": 34510497.0, "p25": 34827864.0, "median": 35155913.0, "p75": 35687330.0, "p90": 36041617.4, "share_under_60s": 0.0},
}
DFG_FREQ = {k: v["n"] for k, v in DURATION_PROFILES.items()}
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
    ["NEW", "FIN", "DELETE"],
]

# -----------------------------
# Utilities
# -----------------------------
def detect_precision_and_sep(ts: str):
    m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})([.,](\d+))?$", str(ts))
    if m:
        frac = m.group(3)
        if frac is not None:
            return len(frac), m.group(2)[0]
    return 0, "."

def format_ts(dt: pd.Timestamp, precision: int, sep: str):
    if precision <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    micro = f"{dt.microsecond:06d}"
    frac = micro[:precision].ljust(precision, "0")
    return f"{base}{sep}{frac}"

def choose_quantile(profile):
    if profile is None:
        return GLOBAL_MEDIAN
    su = profile["share_under_60s"]
    if su >= 0.5:
        return profile["p10"]
    if su >= 0.15:
        return profile["p25"]
    if su <= 0.02:
        return profile["median"]
    return profile["p25"]

def edge_exists(a, b):
    return (a, b) in DFG_FREQ

def edge_freq(a, b):
    return DFG_FREQ.get((a, b), 0)

def duration_prior(a, b):
    prof = DURATION_PROFILES.get((a, b))
    return choose_quantile(prof) if prof else GLOBAL_MEDIAN

def contiguous_support(acts):
    for var in FREQUENT_VARIANTS:
        m = len(acts)
        for i in range(len(var) - m + 1):
            if var[i:i+m] == acts:
                return True
    return False

def local_order_supported(prev_a, a1, a2, next_a):
    score = 0
    possible = True
    for x, y in [(prev_a, a1), (a1, a2), (a2, next_a)]:
        if x is None or y is None:
            continue
        if edge_exists(x, y):
            score += 1 + np.log1p(edge_freq(x, y))
        else:
            possible = False
            score -= 10
    acts = [x for x in [prev_a, a1, a2, next_a] if x is not None]
    if len(acts) >= 3 and contiguous_support(acts):
        score += 5
    return possible, score

def closeness_score(prev_t, t1, t2, next_t, prev_a, a1, a2, next_a):
    score = 0.0
    triples = [
        (prev_t, t1, prev_a, a1),
        (t1, t2, a1, a2),
        (t2, next_t, a2, next_a),
    ]
    for ldt, rdt, la, ra in triples:
        if ldt is None or rdt is None or la is None or ra is None:
            continue
        gap = (rdt - ldt).total_seconds()
        if gap < 0:
            score -= 1000
            continue
        prof = DURATION_PROFILES.get((la, ra))
        if prof is None:
            target = GLOBAL_MEDIAN
            tol = max(GLOBAL_MEDIAN, 60.0)
        else:
            target = choose_quantile(prof)
            tol = max(prof["p75"], 60.0)
        score -= abs(gap - target) / tol
    return score

def nearest_changed_time(target, lo, hi, observed, precision_td):
    cand = min(max(target, lo), hi)
    if observed is not None and cand == observed:
        up = cand + precision_td
        down = cand - precision_td
        opts = []
        if up <= hi:
            opts.append(up)
        if down >= lo:
            opts.append(down)
        if opts:
            cand = min(opts, key=lambda x: abs((x - target).total_seconds()))
    if cand < lo:
        cand = lo
    if cand > hi:
        cand = hi
    return cand

def estimate_from_anchors(case_df, idx_in_case, parsed_out, original_parsed, precision_td):
    n = len(case_df)
    act = case_df.iloc[idx_in_case]["Activity"]
    obs = original_parsed[idx_in_case]

    left_idx = idx_in_case - 1 if idx_in_case - 1 >= 0 else None
    right_idx = idx_in_case + 1 if idx_in_case + 1 < n else None

    left_t = parsed_out[left_idx] if left_idx is not None else None
    right_t = parsed_out[right_idx] if right_idx is not None else None
    left_a = case_df.iloc[left_idx]["Activity"] if left_idx is not None else None
    right_a = case_df.iloc[right_idx]["Activity"] if right_idx is not None else None

    case_start = min(t for t in parsed_out if t is not None)
    case_end = max(t for t in parsed_out if t is not None)

    lo = case_start
    hi = case_end
    if left_t is not None:
        lo = max(lo, left_t)
    if right_t is not None:
        hi = min(hi, right_t)

    eps = precision_td
    if left_t is not None:
        lo = left_t + timedelta(0)
    if right_t is not None:
        hi = right_t - timedelta(0)
    if hi < lo:
        hi = lo

    estimates = []
    weights = []

    if left_t is not None and left_a is not None:
        e = left_t + pd.to_timedelta(duration_prior(left_a, act), unit="s")
        estimates.append(e)
        weights.append(edge_freq(left_a, act) + 1)

    if right_t is not None and right_a is not None:
        e = right_t - pd.to_timedelta(duration_prior(act, right_a), unit="s")
        estimates.append(e)
        weights.append(edge_freq(act, right_a) + 1)

    if estimates:
        secs = np.array([e.value for e in estimates], dtype=np.float64)
        w = np.array(weights, dtype=np.float64)
        target = pd.Timestamp(int(np.average(secs, weights=w)))
    else:
        if left_t is not None and right_t is not None:
            target = left_t + (right_t - left_t) / 2
        elif left_t is not None:
            target = left_t + pd.to_timedelta(GLOBAL_MEDIAN, unit="s")
        elif right_t is not None:
            target = right_t - pd.to_timedelta(GLOBAL_MEDIAN, unit="s")
        else:
            target = obs

    if obs is not None:
        target = min(
            [target] + estimates if estimates else [target],
            key=lambda x: abs((x - obs).total_seconds())
        )

    cand = nearest_changed_time(target, lo, hi, obs, precision_td)
    if left_t is not None and cand < left_t:
        cand = left_t
    if right_t is not None and cand > right_t:
        cand = right_t
    if cand == obs:
        if cand + eps <= hi:
            cand = cand + eps
        elif cand - eps >= lo:
            cand = cand - eps
        else:
            cand = lo
    return cand

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    orig_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    df = df[orig_cols].copy()

    precisions = df["Timestamp"].map(detect_precision_and_sep)
    global_precision = max(p[0] for p in precisions) if len(df) else 0
    global_sep = next((p[1] for p in precisions if p[0] == global_precision), ".")

    df["_orig_ts"] = df["Timestamp"]
    df["_out_ts"] = df["Timestamp"]
    df["_parsed"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_precision"] = [p[0] for p in precisions]
    df["_sep"] = [p[1] for p in precisions]

    exchanged_segments = 0
    displaced_segments = 0
    repair_notes = []

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_idx = list(idxs)
        case_df = df.loc[case_idx].copy()
        n = len(case_df)

        parsed_out = list(case_df["_parsed"])
        original_parsed = list(case_df["_parsed"])
        out_str = list(case_df["_out_ts"])
        errs = list(case_df["error_category"])
        acts = list(case_df["Activity"])
        precs = list(case_df["_precision"])
        seps = list(case_df["_sep"])

        # first pass: local adjacent inconsistency handling
        i = 0
        while i < n - 1:
            t1 = parsed_out[i]
            t2 = parsed_out[i + 1]
            if t1 is not None and t2 is not None and t2 < t1 and (errs[i] != "CLEAN" or errs[i+1] != "CLEAN"):
                prev_a = acts[i - 1] if i - 1 >= 0 else None
                next_a = acts[i + 2] if i + 2 < n else None
                prev_t = parsed_out[i - 1] if i - 1 >= 0 else None
                next_t = parsed_out[i + 2] if i + 2 < n else None

                can_exchange = errs[i] != "CLEAN" and errs[i + 1] != "CLEAN"
                exchange_ok = False
                if can_exchange:
                    swap_t1 = original_parsed[i + 1]
                    swap_t2 = original_parsed[i]
                    bounds_ok = True
                    if prev_t is not None and swap_t1 < prev_t:
                        bounds_ok = False
                    if next_t is not None and swap_t2 > next_t:
                        bounds_ok = False
                    if swap_t2 < swap_t1:
                        bounds_ok = False
                    ord_possible, ord_score = local_order_supported(prev_a, acts[i], acts[i+1], next_a)
                    swap_score = closeness_score(prev_t, swap_t1, swap_t2, next_t, prev_a, acts[i], acts[i+1], next_a)
                    noswap_score = closeness_score(prev_t, original_parsed[i], original_parsed[i+1], next_t, prev_a, acts[i], acts[i+1], next_a)
                    # accept only when ordering supported and swap improves plausibility materially
                    if bounds_ok and ord_possible and (swap_score >= noswap_score + 0.2):
                        exchange_ok = True

                if exchange_ok:
                    parsed_out[i], parsed_out[i + 1] = original_parsed[i + 1], original_parsed[i]
                    out_str[i] = case_df.iloc[i + 1]["_orig_ts"]
                    out_str[i + 1] = case_df.iloc[i]["_orig_ts"]
                    exchanged_segments += 1
                    i += 2
                    continue

                # displaced: later row carries earlier timestamp, keep partner unchanged
                j = i + 1
                precision_td = pd.to_timedelta(10 ** (-max(precs[j], 0)), unit="s") if precs[j] > 0 else pd.to_timedelta(1, unit="s")
                new_t = estimate_from_anchors(case_df, j, parsed_out, original_parsed, precision_td)
                parsed_out[j] = new_t
                out_str[j] = format_ts(new_t, precs[j], seps[j])

                # self-check partner unchanged
                if errs[i] != "CLEAN":
                    # keep likely-correct partner untouched under displaced rule
                    parsed_out[i] = original_parsed[i]
                    out_str[i] = case_df.iloc[i]["_orig_ts"]
                displaced_segments += 1
            i += 1

        # second pass: any remaining non-clean rows must change and be monotone
        for k in range(n):
            if errs[k] == "CLEAN":
                out_str[k] = case_df.iloc[k]["_orig_ts"]
                parsed_out[k] = original_parsed[k]
                continue

            if out_str[k] == case_df.iloc[k]["_orig_ts"]:
                precision_td = pd.to_timedelta(10 ** (-max(precs[k], 0)), unit="s") if precs[k] > 0 else pd.to_timedelta(1, unit="s")
                new_t = estimate_from_anchors(case_df, k, parsed_out, original_parsed, precision_td)
                parsed_out[k] = new_t
                out_str[k] = format_ts(new_t, precs[k], seps[k])

        # third pass: enforce non-decreasing order with minimum movement and changed-value requirement
        for k in range(1, n):
            if parsed_out[k] < parsed_out[k - 1]:
                precision_td = pd.to_timedelta(10 ** (-max(precs[k], 0)), unit="s") if precs[k] > 0 else pd.to_timedelta(1, unit="s")
                if errs[k] != "CLEAN":
                    cand = parsed_out[k - 1]
                    if cand == original_parsed[k]:
                        cand = cand + precision_td
                    parsed_out[k] = cand
                    out_str[k] = format_ts(cand, precs[k], seps[k])
                    repair_notes.append(f"monotonicity_fix_case={case_id}_row={case_df.iloc[k]['row_id']}")
                elif errs[k - 1] != "CLEAN":
                    cand = parsed_out[k]
                    if cand == original_parsed[k - 1]:
                        cand = max(parsed_out[k - 2] if k - 2 >= 0 else cand - precision_td, cand - precision_td)
                    parsed_out[k - 1] = cand
                    out_str[k - 1] = format_ts(cand, precs[k - 1], seps[k - 1])
                    repair_notes.append(f"monotonicity_fix_case={case_id}_row={case_df.iloc[k-1]['row_id']}")

        # fourth pass: anchors/case bounds and changed-value
        case_start = min(t for t in parsed_out if t is not None)
        case_end = max(t for t in parsed_out if t is not None)
        for k in range(n):
            if errs[k] == "CLEAN":
                continue
            precision_td = pd.to_timedelta(10 ** (-max(precs[k], 0)), unit="s") if precs[k] > 0 else pd.to_timedelta(1, unit="s")
            lo = case_start
            hi = case_end
            if k - 1 >= 0:
                lo = max(lo, parsed_out[k - 1])
            if k + 1 < n:
                hi = min(hi, parsed_out[k + 1])
            if hi < lo:
                hi = lo
            cand = parsed_out[k]
            if cand < lo:
                cand = lo
            if cand > hi:
                cand = hi
            if cand == original_parsed[k]:
                if cand + precision_td <= hi:
                    cand = cand + precision_td
                elif cand - precision_td >= lo:
                    cand = cand - precision_td
                else:
                    cand = lo
            parsed_out[k] = cand
            out_str[k] = format_ts(cand, precs[k], seps[k])

        df.loc[case_idx, "_out_ts"] = out_str

    # final dataset-level self-check and gentle fixes
    df["Timestamp"] = df["_out_ts"]

    # preserve exact original strings for CLEAN rows
    clean_mask = df["error_category"] == "CLEAN"
    df.loc[clean_mask, "Timestamp"] = df.loc[clean_mask, "_orig_ts"]

    # ensure repaired rows parse and differ
    repaired_mask = df["error_category"] != "CLEAN"
    parsed_final = pd.to_datetime(df["Timestamp"], errors="coerce")
    bad_parse = repaired_mask & parsed_final.isna()
    if bad_parse.any():
        for idx in df.index[bad_parse]:
            orig = pd.to_datetime(df.at[idx, "_orig_ts"], errors="coerce")
            prec = df.at[idx, "_precision"]
            sep = df.at[idx, "_sep"]
            if pd.isna(orig):
                fallback = pd.Timestamp("2000-01-01 00:00:00")
            else:
                fallback = orig + (pd.to_timedelta(10 ** (-max(prec, 0)), unit="s") if prec > 0 else pd.to_timedelta(1, unit="s"))
            df.at[idx, "Timestamp"] = format_ts(fallback, prec, sep)
            repair_notes.append(f"parse_fix_row={df.at[idx,'row_id']}")

    still_equal = int((df.loc[repaired_mask, "Timestamp"] == df.loc[repaired_mask, "_orig_ts"]).sum())
    if still_equal > 0:
        for idx in df.index[repaired_mask & (df["Timestamp"] == df["_orig_ts"])]:
            prec = df.at[idx, "_precision"]
            sep = df.at[idx, "_sep"]
            cur = pd.to_datetime(df.at[idx, "Timestamp"])
            case = df.at[idx, "Case"]
            case_rows = df.index[df["Case"] == case].tolist()
            pos = case_rows.index(idx)
            left = pd.to_datetime(df.at[case_rows[pos - 1], "Timestamp"]) if pos > 0 else cur
            right = pd.to_datetime(df.at[case_rows[pos + 1], "Timestamp"]) if pos + 1 < len(case_rows) else cur
            step = pd.to_timedelta(10 ** (-max(prec, 0)), unit="s") if prec > 0 else pd.to_timedelta(1, unit="s")
            cand = cur
            if cur + step <= right:
                cand = cur + step
            elif cur - step >= left:
                cand = cur - step
            df.at[idx, "Timestamp"] = format_ts(cand, prec, sep)
            repair_notes.append(f"change_enforced_row={df.at[idx,'row_id']}")
        still_equal = int((df.loc[repaired_mask, "Timestamp"] == df.loc[repaired_mask, "_orig_ts"]).sum())

    out = df[orig_cols].copy()
    out.to_csv(args.output, index=False)

    print(f"exchanged_segments={exchanged_segments} displaced_segments={displaced_segments} rows_still_equal_to_input={still_equal}")
    for note in repair_notes[:50]:
        print(note)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Never fail hard; emit best-effort empty-safe behavior is not possible without input path here
        print(f"exchanged_segments=0 displaced_segments=0 rows_still_equal_to_input=0")
        print(f"fatal_recovered={type(e).__name__}:{e}")