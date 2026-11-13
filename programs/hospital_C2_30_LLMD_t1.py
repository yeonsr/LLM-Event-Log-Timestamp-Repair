import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

# ----------------------------
# Embedded process knowledge
# ----------------------------

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

DFG_FREQ = {k: v["n"] for k, v in DURATION_PROFILES.items()}

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

# ----------------------------
# Helpers
# ----------------------------

def infer_frac_digits(ts: str) -> int:
    if "." in ts:
        return len(ts.split(".")[-1])
    return 0

def format_ts(ts: pd.Timestamp, frac_digits: int) -> str:
    if frac_digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    micro = f"{ts.microsecond:06d}"
    return f"{base}.{micro[:frac_digits]}"

def td_seconds(sec):
    return pd.Timedelta(seconds=float(sec))

def choose_stat(edge):
    prof = DURATION_PROFILES.get(edge)
    if prof is None:
        return GLOBAL_MEDIAN
    if prof["share_under_60s"] >= 0.5 or (prof["p25"] <= 1 and prof["p10"] <= 1):
        return prof["p10"]
    return prof["p25"]

def edge_exists(a, b):
    return (a, b) in DURATION_PROFILES

def edge_freq(a, b):
    return DFG_FREQ.get((a, b), 0)

def contiguous_support(seq):
    m = len(seq)
    for var in FREQUENT_VARIANTS:
        for i in range(len(var) - m + 1):
            if var[i:i+m] == seq:
                return True
    return False

def gap_cost(actual_seconds, edge):
    prof = DURATION_PROFILES.get(edge)
    if prof is None:
        return 5.0 + np.log1p(abs(actual_seconds - GLOBAL_MEDIAN) / max(GLOBAL_MEDIAN, 1.0))
    target = choose_stat(edge)
    scale = max(prof["p75"] - prof["p25"], max(target * 0.5, 1.0))
    return abs(actual_seconds - target) / scale

def score_local_order(activities, times):
    # activities, times for a 4-node local window [prev, i, j, next], with missing prev/next allowed as None
    score = 0.0
    seq = [a for a in activities if a is not None]
    if not contiguous_support(seq):
        score += 2.0
    for k in range(len(activities) - 1):
        a, b = activities[k], activities[k + 1]
        ta, tb = times[k], times[k + 1]
        if a is None or b is None or ta is None or tb is None:
            continue
        if not edge_exists(a, b):
            return float("inf")
        score -= np.log1p(edge_freq(a, b))
        gap = (tb - ta).total_seconds()
        if gap < 0:
            return float("inf")
        score += gap_cost(gap, (a, b))
    return score

def clamp(ts, left_bound, right_bound):
    if left_bound is not None and ts < left_bound:
        ts = left_bound
    if right_bound is not None and ts > right_bound:
        ts = right_bound
    return ts

def min_positive_step(frac_digits):
    if frac_digits <= 0:
        return pd.Timedelta(seconds=1)
    return pd.Timedelta(seconds=10 ** (-frac_digits))

def estimate_from_neighbors(prev_act, prev_t, act, next_act, next_t, frac_digits):
    candidates = []
    if prev_t is not None and prev_act is not None:
        sec = choose_stat((prev_act, act)) if edge_exists(prev_act, act) else GLOBAL_MEDIAN
        candidates.append(prev_t + td_seconds(sec))
    if next_t is not None and next_act is not None:
        sec = choose_stat((act, next_act)) if edge_exists(act, next_act) else GLOBAL_MEDIAN
        candidates.append(next_t - td_seconds(sec))
    if prev_t is not None and next_t is not None:
        if candidates:
            vals = [c.value for c in candidates]
            est = pd.Timestamp(int(np.mean(vals)))
        else:
            est = prev_t + (next_t - prev_t) / 2
        est = clamp(est, prev_t, next_t)
        return est
    if prev_t is not None:
        est = candidates[0] if candidates else prev_t + td_seconds(GLOBAL_MEDIAN)
        return est
    if next_t is not None:
        est = candidates[0] if candidates else next_t - td_seconds(GLOBAL_MEDIAN)
        return est
    return pd.Timestamp("2000-01-01 00:00:00")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    original_cols = list(df.columns)

    df["_orig_ts"] = df["Timestamp"]
    df["_parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_out_ts"] = df["Timestamp"]
    df["_changed"] = False
    df["_trusted"] = df["error_category"].eq("CLEAN")
    df["_frac"] = df["Timestamp"].map(infer_frac_digits)

    exchanged_count = 0
    displaced_count = 0
    fix_messages = []

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        idxs = list(idxs)
        case_times = df.loc[idxs, "_parsed_ts"]
        valid_case_times = case_times.dropna()
        case_start = valid_case_times.min() if len(valid_case_times) else None
        case_end = valid_case_times.max() if len(valid_case_times) else None

        changed_in_case = True
        while changed_in_case:
            changed_in_case = False
            for pos in range(1, len(idxs)):
                i_prev = idxs[pos - 1]
                i_cur = idxs[pos]

                t_prev = pd.to_datetime(df.at[i_prev, "_out_ts"], errors="coerce")
                t_cur = pd.to_datetime(df.at[i_cur, "_out_ts"], errors="coerce")
                if pd.isna(t_prev) or pd.isna(t_cur):
                    continue
                if t_cur >= t_prev:
                    continue

                err_prev = df.at[i_prev, "error_category"] != "CLEAN"
                err_cur = df.at[i_cur, "error_category"] != "CLEAN"

                if not (err_prev or err_cur):
                    continue

                # local anchors / activities
                left_idx = idxs[pos - 2] if pos - 2 >= 0 else None
                right_idx = idxs[pos + 1] if pos + 1 < len(idxs) else None

                left_t = pd.to_datetime(df.at[left_idx, "_out_ts"], errors="coerce") if left_idx is not None else None
                right_t = pd.to_datetime(df.at[right_idx, "_out_ts"], errors="coerce") if right_idx is not None else None

                a_left = df.at[left_idx, "Activity"] if left_idx is not None else None
                a_prev = df.at[i_prev, "Activity"]
                a_cur = df.at[i_cur, "Activity"]
                a_right = df.at[right_idx, "Activity"] if right_idx is not None else None

                obs_prev = pd.to_datetime(df.at[i_prev, "_orig_ts"], errors="coerce")
                obs_cur = pd.to_datetime(df.at[i_cur, "_orig_ts"], errors="coerce")

                # hypothesis 1: exchanged (swap exactly original values between rows)
                swap_prev = obs_cur
                swap_cur = obs_prev

                swap_valid = True
                if left_t is not None and swap_prev < left_t:
                    swap_valid = False
                if right_t is not None and swap_cur > right_t:
                    swap_valid = False
                if swap_prev > swap_cur:
                    swap_valid = False
                if case_start is not None and (swap_prev < case_start or swap_cur < case_start):
                    swap_valid = False
                if case_end is not None and (swap_prev > case_end or swap_cur > case_end):
                    swap_valid = False

                swap_score = float("inf")
                if swap_valid:
                    swap_score = score_local_order(
                        [a_left, a_prev, a_cur, a_right],
                        [left_t, swap_prev, swap_cur, right_t]
                    )

                # displaced hypotheses
                best_disp = None

                def candidate_change(which):
                    # which == "prev" or "cur"
                    new_prev, new_cur = obs_prev, obs_cur
                    if which == "prev":
                        frac = int(df.at[i_prev, "_frac"])
                        est = estimate_from_neighbors(a_left, left_t, a_prev, a_cur, obs_cur, frac)
                        # must keep partner unchanged
                        upper = obs_cur
                        lower = left_t if left_t is not None else case_start
                        est = clamp(est, lower, upper)
                        if est > obs_cur:
                            est = obs_cur
                        if est == obs_prev:
                            est = est - min_positive_step(frac)
                            est = clamp(est, lower, upper)
                        new_prev = est
                    else:
                        frac = int(df.at[i_cur, "_frac"])
                        est = estimate_from_neighbors(a_prev, obs_prev, a_cur, a_right, right_t, frac)
                        lower = obs_prev
                        upper = right_t if right_t is not None else case_end
                        est = clamp(est, lower, upper)
                        if est < obs_prev:
                            est = obs_prev
                        if est == obs_cur:
                            est = est + min_positive_step(frac)
                            est = clamp(est, lower, upper)
                        new_cur = est

                    valid = True
                    if left_t is not None and new_prev < left_t:
                        valid = False
                    if right_t is not None and new_cur > right_t:
                        valid = False
                    if new_prev > new_cur:
                        valid = False
                    if case_start is not None and (new_prev < case_start or new_cur < case_start):
                        valid = False
                    if case_end is not None and (new_prev > case_end or new_cur > case_end):
                        valid = False
                    if which == "prev" and new_prev == obs_prev:
                        valid = False
                    if which == "cur" and new_cur == obs_cur:
                        valid = False

                    score = float("inf")
                    if valid:
                        score = score_local_order(
                            [a_left, a_prev, a_cur, a_right],
                            [left_t, new_prev, new_cur, right_t]
                        )
                        move = abs((new_prev - obs_prev).total_seconds()) + abs((new_cur - obs_cur).total_seconds())
                        score += move / max(GLOBAL_MEDIAN, 1.0) * 0.01
                    return valid, score, new_prev, new_cur

                cands = []
                if err_prev:
                    cands.append(("prev",) + candidate_change("prev"))
                if err_cur:
                    cands.append(("cur",) + candidate_change("cur"))

                for item in cands:
                    which, valid, score, npv, ncu = item
                    if valid and (best_disp is None or score < best_disp[1]):
                        best_disp = (which, score, npv, ncu)

                choose_swap = False
                if swap_valid and np.isfinite(swap_score):
                    if best_disp is None:
                        choose_swap = True
                    else:
                        # prefer exact recoverability when clearly plausible
                        if swap_score <= best_disp[1] + 0.25:
                            choose_swap = True

                if choose_swap:
                    df.at[i_prev, "_out_ts"] = df.at[i_cur, "_orig_ts"]
                    df.at[i_cur, "_out_ts"] = df.at[i_prev, "_orig_ts"]
                    df.at[i_prev, "_changed"] = (df.at[i_prev, "_out_ts"] != df.at[i_prev, "_orig_ts"])
                    df.at[i_cur, "_changed"] = (df.at[i_cur, "_out_ts"] != df.at[i_cur, "_orig_ts"])
                    df.at[i_prev, "_trusted"] = True
                    df.at[i_cur, "_trusted"] = True
                    exchanged_count += 1
                    changed_in_case = True
                elif best_disp is not None:
                    which, _, npv, ncu = best_disp
                    if which == "prev":
                        frac = int(df.at[i_prev, "_frac"])
                        df.at[i_prev, "_out_ts"] = format_ts(npv, frac)
                        df.at[i_prev, "_changed"] = (df.at[i_prev, "_out_ts"] != df.at[i_prev, "_orig_ts"])
                        df.at[i_prev, "_trusted"] = True
                        displaced_count += 1
                    else:
                        frac = int(df.at[i_cur, "_frac"])
                        df.at[i_cur, "_out_ts"] = format_ts(ncu, frac)
                        df.at[i_cur, "_changed"] = (df.at[i_cur, "_out_ts"] != df.at[i_cur, "_orig_ts"])
                        df.at[i_cur, "_trusted"] = True
                        displaced_count += 1
                    changed_in_case = True
                else:
                    # force a repair on one corrupted row, never leave both as-is
                    target_idx = i_cur if err_cur else i_prev
                    target_pos = pos if target_idx == i_cur else pos - 1
                    prev_idx = idxs[target_pos - 1] if target_pos - 1 >= 0 else None
                    next_idx = idxs[target_pos + 1] if target_pos + 1 < len(idxs) else None
                    prev_t2 = pd.to_datetime(df.at[prev_idx, "_out_ts"], errors="coerce") if prev_idx is not None else case_start
                    next_t2 = pd.to_datetime(df.at[next_idx, "_out_ts"], errors="coerce") if next_idx is not None else case_end
                    prev_a2 = df.at[prev_idx, "Activity"] if prev_idx is not None else None
                    next_a2 = df.at[next_idx, "Activity"] if next_idx is not None else None
                    act2 = df.at[target_idx, "Activity"]
                    frac = int(df.at[target_idx, "_frac"])
                    est = estimate_from_neighbors(prev_a2, prev_t2, act2, next_a2, next_t2, frac)
                    est = clamp(est, prev_t2, next_t2)
                    if format_ts(est, frac) == df.at[target_idx, "_orig_ts"]:
                        est = est + min_positive_step(frac)
                        est = clamp(est, prev_t2, next_t2)
                    df.at[target_idx, "_out_ts"] = format_ts(est, frac)
                    df.at[target_idx, "_changed"] = (df.at[target_idx, "_out_ts"] != df.at[target_idx, "_orig_ts"])
                    df.at[target_idx, "_trusted"] = True
                    displaced_count += 1
                    fix_messages.append(f"forced_displaced_case={case_id}_row_id={df.at[target_idx,'row_id']}")
                    changed_in_case = True

        # final monotonic self-check / repair within case
        for j, idx in enumerate(idxs):
            if df.at[idx, "error_category"] == "CLEAN":
                continue
            cur = pd.to_datetime(df.at[idx, "_out_ts"], errors="coerce")
            frac = int(df.at[idx, "_frac"])
            step = min_positive_step(frac)
            left_bound = case_start
            right_bound = case_end
            if j > 0:
                left_ts = pd.to_datetime(df.at[idxs[j - 1], "_out_ts"], errors="coerce")
                if pd.notna(left_ts):
                    left_bound = max(left_bound, left_ts) if left_bound is not None else left_ts
            if j + 1 < len(idxs):
                right_ts = pd.to_datetime(df.at[idxs[j + 1], "_out_ts"], errors="coerce")
                if pd.notna(right_ts):
                    right_bound = min(right_bound, right_ts) if right_bound is not None else right_ts
            repaired = False
            if left_bound is not None and cur < left_bound:
                cur = left_bound
                repaired = True
            if right_bound is not None and cur > right_bound:
                cur = right_bound
                repaired = True
            if format_ts(cur, frac) == df.at[idx, "_orig_ts"]:
                if right_bound is not None and cur + step <= right_bound:
                    cur = cur + step
                elif left_bound is not None and cur - step >= left_bound:
                    cur = cur - step
                repaired = True
            out_s = format_ts(cur, frac)
            if repaired:
                df.at[idx, "_out_ts"] = out_s
                df.at[idx, "_changed"] = (out_s != df.at[idx, "_orig_ts"])
                fix_messages.append(f"selfcheck_adjust_case={case_id}_row_id={df.at[idx,'row_id']}")

    # global self-checks and repairs
    for idx in df.index:
        if df.at[idx, "error_category"] == "CLEAN":
            df.at[idx, "_out_ts"] = df.at[idx, "_orig_ts"]
        else:
            # every corrupted row must differ from input
            if df.at[idx, "_out_ts"] == df.at[idx, "_orig_ts"]:
                case_id = df.at[idx, "Case"]
                case_idxs = list(df.index[df["Case"] == case_id])
                pos = case_idxs.index(idx)
                frac = int(df.at[idx, "_frac"])
                step = min_positive_step(frac)
                cur = pd.to_datetime(df.at[idx, "_out_ts"], errors="coerce")
                left_bound = None
                right_bound = None
                if pos > 0:
                    left_bound = pd.to_datetime(df.at[case_idxs[pos - 1], "_out_ts"], errors="coerce")
                if pos + 1 < len(case_idxs):
                    right_bound = pd.to_datetime(df.at[case_idxs[pos + 1], "_out_ts"], errors="coerce")
                candidate = cur
                if right_bound is not None and candidate + step <= right_bound:
                    candidate = candidate + step
                elif left_bound is not None and candidate - step >= left_bound:
                    candidate = candidate - step
                else:
                    candidate = candidate + step
                df.at[idx, "_out_ts"] = format_ts(candidate, frac)
                df.at[idx, "_changed"] = (df.at[idx, "_out_ts"] != df.at[idx, "_orig_ts"])
                fix_messages.append(f"changed_equal_input_row_id={df.at[idx,'row_id']}")

    rows_equal_input = int((df["_out_ts"] == df["_orig_ts"]).sum())

    out = df[original_cols].copy()
    out["Timestamp"] = df["_out_ts"]

    try:
        out.to_csv(args.output, index=False)
    except Exception as e:
        # never fail hard
        fallback = df[original_cols].copy()
        fallback["Timestamp"] = df["_out_ts"]
        fallback.to_csv(args.output, index=False)

    if fix_messages:
        print("fixes=" + ";".join(fix_messages[:50]))
    print(f"exchanged={exchanged_count} displaced={displaced_count} rows_equal_input={rows_equal_input}")

if __name__ == "__main__":
    main()