#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

ERROR_CAT = "C2_TEMPORAL_INCONSISTENCY"

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
GLOBAL_MEDIAN = 402242.0
BLOCK2_P25 = 11146.5
FREQS = {k: v["n"] for k, v in DURATION_PROFILES.items()}


def detect_precision(series):
    max_digits = 0
    for s in series.dropna().astype(str):
        if "." in s:
            frac = s.split(".")[-1]
            frac = "".join(ch for ch in frac if ch.isdigit())
            max_digits = max(max_digits, len(frac))
    return max_digits


def parse_ts(s):
    return pd.to_datetime(s, errors="coerce")


def fmt_ts(ts, precision):
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    if precision < 6:
        keep = 19 + 1 + precision
        return base[:keep]
    return base


def td(seconds):
    return pd.Timedelta(seconds=float(seconds))


def profile_for(a, b):
    return DURATION_PROFILES.get((a, b))


def preferred_gap_seconds(a, b):
    prof = profile_for(a, b)
    if prof is None:
        return GLOBAL_MEDIAN
    if prof["share_under_60s"] >= 0.5:
        return prof["p10"]
    return prof["p25"]


def edge_exists(a, b):
    return (a, b) in DURATION_PROFILES


def gap_penalty(obs_seconds, a, b):
    prof = profile_for(a, b)
    if prof is None:
        target = GLOBAL_MEDIAN
        spread = max(target, 60.0)
        extra = 4.0
    else:
        target = preferred_gap_seconds(a, b)
        spread = max(abs(prof["p75"] - prof["p25"]) if prof["p75"] != prof["p25"] else max(target, 60.0), 1.0)
        extra = 0.0 if prof["n"] >= 20 else 1.0
    return abs(obs_seconds - target) / spread + extra


def candidate_score(prev_act, act1, act2, next_act, tprev, t1, t2, tnext):
    score = 0.0
    if t1 > t2:
        return 1e18
    if tprev is not None and t1 < tprev:
        return 1e18
    if tnext is not None and t2 > tnext:
        return 1e18
    if prev_act is not None:
        if not edge_exists(prev_act, act1):
            score += 50.0
        else:
            score += gap_penalty((t1 - tprev).total_seconds(), prev_act, act1) if tprev is not None else 0.0
            score -= min(np.log1p(FREQS.get((prev_act, act1), 1)) * 0.05, 1.0)
    if not edge_exists(act1, act2):
        score += 50.0
    else:
        score += gap_penalty((t2 - t1).total_seconds(), act1, act2)
        score -= min(np.log1p(FREQS.get((act1, act2), 1)) * 0.05, 1.0)
    if next_act is not None:
        if not edge_exists(act2, next_act):
            score += 50.0
        else:
            score += gap_penalty((tnext - t2).total_seconds(), act2, next_act) if tnext is not None else 0.0
            score -= min(np.log1p(FREQS.get((act2, next_act), 1)) * 0.05, 1.0)
    return score


def nearest_clean_index(case_df, pos, direction):
    if direction < 0:
        rng = range(pos - 1, -1, -1)
    else:
        rng = range(pos + 1, len(case_df))
    for j in rng:
        if case_df.iloc[j]["error_category"] == "CLEAN":
            return j
    return None


def inward_nudge(ts, lo, hi, unit):
    if ts <= lo:
        cand = lo + unit
        return min(cand, hi)
    if ts >= hi:
        cand = hi - unit
        return max(cand, lo)
    return ts


def estimate_single(prev_t, prev_act, act, next_t, next_act, observed_t, precision_td):
    lo = prev_t if prev_t is not None else pd.Timestamp.min.tz_localize(None) + pd.Timedelta(days=1)
    hi = next_t if next_t is not None else pd.Timestamp.max.tz_localize(None) - pd.Timedelta(days=1)

    cands = []

    if prev_t is not None:
        cands.append(prev_t + td(preferred_gap_seconds(prev_act, act)))
    if next_t is not None:
        cands.append(next_t - td(preferred_gap_seconds(act, next_act)))
    if prev_t is not None and next_t is not None:
        left_gap = preferred_gap_seconds(prev_act, act)
        right_gap = preferred_gap_seconds(act, next_act)
        total = left_gap + right_gap
        if total <= 0:
            total = 1.0
        cands.append(prev_t + (next_t - prev_t) * (left_gap / total))
        span_hint = td(BLOCK2_P25)
        cands.append(min(max(prev_t + span_hint, lo), hi))
        cands.append(max(min(next_t - span_hint, hi), lo))

    best = None
    best_score = 1e18
    for c in cands:
        c = min(max(c, lo), hi)
        if prev_t is not None and c < prev_t:
            c = prev_t
        if next_t is not None and c > next_t:
            c = next_t
        c = inward_nudge(c, lo, hi, precision_td)
        score = 0.0
        if prev_t is not None:
            score += gap_penalty((c - prev_t).total_seconds(), prev_act, act)
        if next_t is not None:
            score += gap_penalty((next_t - c).total_seconds(), act, next_act)
        score += abs((c - observed_t).total_seconds()) / max(BLOCK2_P25, 1.0)
        if score < best_score:
            best_score = score
            best = c

    if best is None:
        if prev_t is not None and next_t is not None:
            best = prev_t + (next_t - prev_t) / 2
        elif prev_t is not None:
            best = prev_t + td(GLOBAL_MEDIAN)
        elif next_t is not None:
            best = next_t - td(GLOBAL_MEDIAN)
        else:
            best = observed_t

    best = min(max(best, lo), hi)
    best = inward_nudge(best, lo, hi, precision_td)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    orig_cols = list(df.columns)
    original_ts = df["Timestamp"].astype(str).copy()
    parsed = pd.to_datetime(df["Timestamp"], errors="coerce")
    precision = detect_precision(df["Timestamp"])
    unit = pd.Timedelta(milliseconds=1) if precision == 3 else (pd.Timedelta(seconds=1) if precision == 0 else pd.Timedelta(microseconds=max(1, 10 ** (6 - min(precision, 6)))))
    repaired_ts_str = original_ts.copy()
    repaired_dt = parsed.copy()

    exchanged_segments = 0
    displaced_segments = 0
    fixes_report = []

    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        case_df = df.loc[idxs].reset_index()
        case_times = repaired_dt.loc[idxs].reset_index(drop=True)

        i = 0
        while i < len(case_df):
            row = case_df.iloc[i]
            if row["error_category"] != ERROR_CAT:
                i += 1
                continue

            if i == 0:
                prev_i = None
            else:
                prev_i = i - 1
            if i == len(case_df) - 1:
                next_i = None
            else:
                next_i = i + 1

            segment_done = False

            if prev_i is not None and case_df.iloc[prev_i]["error_category"] == ERROR_CAT:
                i += 1
                continue

            if next_i is not None and case_df.iloc[next_i]["error_category"] == ERROR_CAT:
                gidx1 = case_df.iloc[i]["index"]
                gidx2 = case_df.iloc[next_i]["index"]

                act1 = case_df.iloc[i]["Activity"]
                act2 = case_df.iloc[next_i]["Activity"]
                t1_obs = repaired_dt.loc[gidx1]
                t2_obs = repaired_dt.loc[gidx2]

                left_clean_pos = nearest_clean_index(case_df, i, -1)
                right_clean_pos = nearest_clean_index(case_df, next_i, +1)

                left_t = repaired_dt.loc[case_df.iloc[left_clean_pos]["index"]] if left_clean_pos is not None else None
                right_t = repaired_dt.loc[case_df.iloc[right_clean_pos]["index"]] if right_clean_pos is not None else None
                left_act = case_df.iloc[left_clean_pos]["Activity"] if left_clean_pos is not None else None
                right_act = case_df.iloc[right_clean_pos]["Activity"] if right_clean_pos is not None else None

                swap_t1, swap_t2 = t2_obs, t1_obs
                swap_score = candidate_score(left_act, act1, act2, right_act, left_t, swap_t1, swap_t2, right_t)
                swap_move = abs((swap_t1 - t1_obs).total_seconds()) + abs((swap_t2 - t2_obs).total_seconds())

                if swap_score < 1e17:
                    swap_score += swap_move / max(BLOCK2_P25, 1.0)

                best_disp = None
                best_disp_score = 1e18

                est1 = estimate_single(left_t, left_act, act1, t2_obs, act2, t1_obs, unit)
                if (left_t is None or est1 >= left_t) and est1 <= t2_obs and (right_t is None or t2_obs <= right_t):
                    sc1 = candidate_score(left_act, act1, act2, right_act, left_t, est1, t2_obs, right_t)
                    if t2_obs == t2_obs:
                        sc1 += abs((est1 - t1_obs).total_seconds()) / max(BLOCK2_P25, 1.0)
                    best_disp = ("first", est1, t2_obs)
                    best_disp_score = sc1

                est2 = estimate_single(t1_obs, act1, act2, right_t, right_act, t2_obs, unit)
                if (left_t is None or t1_obs >= left_t) and est2 >= t1_obs and (right_t is None or est2 <= right_t):
                    sc2 = candidate_score(left_act, act1, act2, right_act, left_t, t1_obs, est2, right_t)
                    sc2 += abs((est2 - t2_obs).total_seconds()) / max(BLOCK2_P25, 1.0)
                    if sc2 < best_disp_score:
                        best_disp = ("second", t1_obs, est2)
                        best_disp_score = sc2

                if swap_score + 0.25 < best_disp_score:
                    repaired_dt.loc[gidx1] = swap_t1
                    repaired_dt.loc[gidx2] = swap_t2
                    repaired_ts_str.loc[gidx1] = original_ts.loc[gidx2]
                    repaired_ts_str.loc[gidx2] = original_ts.loc[gidx1]
                    exchanged_segments += 1
                else:
                    if best_disp is None:
                        est2 = estimate_single(t1_obs, act1, act2, right_t, right_act, t2_obs, unit)
                        best_disp = ("second", t1_obs, est2)
                    who, nt1, nt2 = best_disp
                    nt1 = nt1 if who == "first" else t1_obs
                    nt2 = nt2 if who == "second" else t2_obs

                    if left_t is not None and nt1 < left_t:
                        nt1 = inward_nudge(left_t, left_t, t2_obs if who == "first" else nt2, unit)
                    if nt2 < nt1:
                        if who == "first":
                            nt1 = min(nt2, max(nt1, left_t if left_t is not None else nt1))
                        else:
                            nt2 = max(nt1, min(nt2, right_t if right_t is not None else nt2))
                    if right_t is not None and nt2 > right_t:
                        nt2 = inward_nudge(right_t, nt1 if who == "second" else t1_obs, right_t, unit)
                    if nt2 < nt1:
                        nt2 = nt1

                    if who == "first":
                        if nt1 == t1_obs:
                            nt1 = inward_nudge(nt1, left_t if left_t is not None else nt1 - unit, t2_obs, unit)
                            if nt1 == t1_obs:
                                nt1 = min(t2_obs, nt1 + unit)
                        repaired_dt.loc[gidx1] = nt1
                        repaired_ts_str.loc[gidx1] = fmt_ts(nt1, precision)
                    else:
                        if nt2 == t2_obs:
                            nt2 = inward_nudge(nt2, t1_obs, right_t if right_t is not None else nt2 + unit, unit)
                            if nt2 == t2_obs:
                                nt2 = max(t1_obs, nt2 - unit)
                                if nt2 == t2_obs:
                                    nt2 = nt2 + unit
                        repaired_dt.loc[gidx2] = nt2
                        repaired_ts_str.loc[gidx2] = fmt_ts(nt2, precision)
                    displaced_segments += 1

                segment_done = True
                i += 2

            if segment_done:
                continue

            gidx = case_df.iloc[i]["index"]
            t_obs = repaired_dt.loc[gidx]

            left_clean_pos = nearest_clean_index(case_df, i, -1)
            right_clean_pos = nearest_clean_index(case_df, i, +1)
            left_t = repaired_dt.loc[case_df.iloc[left_clean_pos]["index"]] if left_clean_pos is not None else None
            right_t = repaired_dt.loc[case_df.iloc[right_clean_pos]["index"]] if right_clean_pos is not None else None
            left_act = case_df.iloc[left_clean_pos]["Activity"] if left_clean_pos is not None else None
            right_act = case_df.iloc[right_clean_pos]["Activity"] if right_clean_pos is not None else None
            act = case_df.iloc[i]["Activity"]

            est = estimate_single(left_t, left_act, act, right_t, right_act, t_obs, unit)
            if est == t_obs:
                if left_t is not None and est <= right_t if right_t is not None else True:
                    est = est + unit
                elif right_t is not None:
                    est = est - unit
                else:
                    est = est + unit
            if left_t is not None and est < left_t:
                est = left_t
            if right_t is not None and est > right_t:
                est = right_t
            if left_t is not None and right_t is not None and est == t_obs:
                room1 = (est - left_t).total_seconds()
                room2 = (right_t - est).total_seconds()
                est = est - unit if room1 >= room2 and room1 >= unit.total_seconds() else est + unit
                est = min(max(est, left_t), right_t)

            repaired_dt.loc[gidx] = est
            repaired_ts_str.loc[gidx] = fmt_ts(est, precision)
            displaced_segments += 1
            i += 1

    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        case_rows = df.loc[idxs]
        case_dt = repaired_dt.loc[idxs].copy()
        for j in range(1, len(idxs)):
            prev_idx = idxs[j - 1]
            cur_idx = idxs[j]
            if case_dt.loc[cur_idx] < case_dt.loc[prev_idx]:
                if df.loc[cur_idx, "error_category"] != "CLEAN":
                    lo = case_dt.loc[prev_idx]
                    hi = None
                    for k in range(j + 1, len(idxs)):
                        nxt_idx = idxs[k]
                        if df.loc[nxt_idx, "error_category"] == "CLEAN":
                            hi = case_dt.loc[nxt_idx]
                            break
                    newt = lo
                    if hi is not None:
                        newt = inward_nudge(lo, lo, hi, unit)
                    else:
                        newt = lo
                    if newt == pd.to_datetime(original_ts.loc[cur_idx], errors="coerce"):
                        newt = newt + unit
                    repaired_dt.loc[cur_idx] = newt
                    repaired_ts_str.loc[cur_idx] = fmt_ts(newt, precision)
                    case_dt.loc[cur_idx] = newt
                    fixes_report.append(f"monotonic_fix:{case_id}:{df.loc[cur_idx,'row_id']}")
                elif df.loc[prev_idx, "error_category"] != "CLEAN":
                    hi = case_dt.loc[cur_idx]
                    lo = None
                    for k in range(j - 2, -1, -1):
                        prv_idx2 = idxs[k]
                        if df.loc[prv_idx2, "error_category"] == "CLEAN":
                            lo = case_dt.loc[prv_idx2]
                            break
                    newt = hi
                    if lo is not None:
                        newt = inward_nudge(hi, lo, hi, unit)
                    else:
                        newt = hi
                    if newt == pd.to_datetime(original_ts.loc[prev_idx], errors="coerce"):
                        newt = newt - unit
                    repaired_dt.loc[prev_idx] = newt
                    repaired_ts_str.loc[prev_idx] = fmt_ts(newt, precision)
                    case_dt.loc[prev_idx] = newt
                    fixes_report.append(f"monotonic_fix:{case_id}:{df.loc[prev_idx,'row_id']}")

    for idx in df.index:
        if df.loc[idx, "error_category"] == "CLEAN":
            repaired_ts_str.loc[idx] = original_ts.loc[idx]
            repaired_dt.loc[idx] = pd.to_datetime(original_ts.loc[idx], errors="coerce")
        else:
            if repaired_ts_str.loc[idx] == original_ts.loc[idx]:
                t = repaired_dt.loc[idx]
                case = df.loc[idx, "Case"]
                case_idxs = list(df.index[df["Case"] == case])
                pos = case_idxs.index(idx)
                lo = repaired_dt.loc[case_idxs[pos - 1]] if pos > 0 else t - td(GLOBAL_MEDIAN)
                hi = repaired_dt.loc[case_idxs[pos + 1]] if pos < len(case_idxs) - 1 else t + td(GLOBAL_MEDIAN)
                if pd.notna(lo) and pd.notna(hi):
                    if t + unit <= hi:
                        t2 = t + unit
                    elif t - unit >= lo:
                        t2 = t - unit
                    else:
                        t2 = inward_nudge(t, lo, hi, unit)
                else:
                    t2 = t + unit
                if pd.notna(t2):
                    repaired_dt.loc[idx] = t2
                    repaired_ts_str.loc[idx] = fmt_ts(t2, precision)
                    fixes_report.append(f"diff_fix:{case}:{df.loc[idx,'row_id']}")

    out = df.copy()
    out["Timestamp"] = repaired_ts_str

    try:
        parsed_out = pd.to_datetime(out["Timestamp"], errors="coerce")
        bad_parse = parsed_out.isna()
        if bad_parse.any():
            for idx in out.index[bad_parse]:
                if df.loc[idx, "error_category"] != "CLEAN":
                    repaired_ts_str.loc[idx] = fmt_ts(repaired_dt.loc[idx], precision)
                    out.loc[idx, "Timestamp"] = repaired_ts_str.loc[idx]
                    fixes_report.append(f"parse_fix:{df.loc[idx,'Case']}:{df.loc[idx,'row_id']}")
    except Exception:
        pass

    out = out[orig_cols]
    out.to_csv(args.output, index=False)

    still_equal = int(((df["error_category"] != "CLEAN") & (out["Timestamp"] == original_ts)).sum())
    print(f"exchanged_segments={exchanged_segments} displaced_segments={displaced_segments} rows_still_equal={still_equal}" + ("" if not fixes_report else " fixes=" + str(len(fixes_report))))

if __name__ == "__main__":
    main()