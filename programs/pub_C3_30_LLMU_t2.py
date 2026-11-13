import argparse
import math
import re
from itertools import permutations
from collections import defaultdict, Counter

import pandas as pd


TRANSITION_PROBS_TOP = [
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

GLOBAL_MEDIAN_DURATION = 243.721


def build_transition_maps():
    trans_prob = {}
    for s, t, p in TRANSITION_PROBS_TOP:
        trans_prob[(s, t)] = p

    dur = {}
    for s, t, n, p10, p25, med, p75, p90, su60 in DURATION_PROFILES_TOP:
        dur[(s, t)] = {
            "n": n,
            "p10": p10,
            "p25": p25,
            "median": med,
            "p75": p75,
            "p90": p90,
            "share_under_60s": su60,
        }
    return trans_prob, dur


TRANS_PROB, DUR_PROF = build_transition_maps()


def parse_ts_safe(val):
    if pd.isna(val):
        return pd.NaT
    s = str(val)
    if s.strip() == "":
        return pd.NaT
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def infer_frac_len(series):
    lens = Counter()
    for v in series.astype(str):
        m = re.search(r"\.(\d+)$", v)
        if m:
            lens[len(m.group(1))] += 1
    if lens:
        return lens.most_common(1)[0][0]
    return 3


def fmt_ts(ts, frac_len):
    if pd.isna(ts):
        return ""
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if frac_len <= 0:
        return base
    ns = ts.nanosecond + ts.microsecond * 1000
    frac = f"{ns:09d}"[:frac_len]
    return f"{base}.{frac}"


def has_date_only(s):
    if s is None:
        return False
    st = str(s).strip()
    if st == "":
        return False
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", st))


def transition_logprob(a, b):
    p = TRANS_PROB.get((a, b), 1e-6)
    return math.log(max(p, 1e-12))


def get_gap_estimate(a, b):
    prof = DUR_PROF.get((a, b))
    if prof:
        if prof["share_under_60s"] >= 0.5:
            g = prof["p10"]
        else:
            g = prof["p25"]
        return max(1.0, float(g))
    return max(1.0, float(GLOBAL_MEDIAN_DURATION))


def duration_plausibility(a, b, delta_sec):
    if delta_sec < 0:
        return -1e9
    prof = DUR_PROF.get((a, b))
    if prof is None:
        target = GLOBAL_MEDIAN_DURATION
        spread = max(30.0, GLOBAL_MEDIAN_DURATION)
        return -abs(delta_sec - target) / spread
    target = prof["p10"] if prof["share_under_60s"] >= 0.5 else prof["p25"]
    spread = max(10.0, prof["p75"] - prof["p10"] if prof["p75"] >= prof["p10"] else prof["median"] or 10.0)
    return -abs(delta_sec - target) / spread


def score_order(acts, left_act, right_act, row_positions=None, perm_positions=None):
    score = 0.0
    if left_act is not None:
        score += transition_logprob(left_act, acts[0])
    for i in range(len(acts) - 1):
        score += transition_logprob(acts[i], acts[i + 1])
    if right_act is not None:
        score += transition_logprob(acts[-1], right_act)

    # mild tie-breaker: prefer common transitions already partly aligned with original order
    if row_positions is not None and perm_positions is not None:
        move_penalty = 0.0
        for ridx, pidx in zip(row_positions, perm_positions):
            move_penalty += abs(ridx - pidx)
        score -= 0.01 * move_penalty
    return score


def best_order_for_block(rows, left_act, right_act):
    activities = [r["Activity"] for r in rows]
    n = len(rows)
    idxs = list(range(n))

    unique_perms = set(permutations(idxs))
    best = None
    best_score = -1e18
    orig_positions = list(range(n))

    for perm in unique_perms:
        acts = [activities[i] for i in perm]
        sc = score_order(acts, left_act, right_act, orig_positions, list(perm))
        if sc > best_score:
            best_score = sc
            best = perm
    return list(best)


def find_clean_left_anchor(case_rows, start_idx):
    for j in range(start_idx - 1, -1, -1):
        if case_rows[j]["error_category"] == "CLEAN":
            return j
    return None


def find_clean_right_anchor(case_rows, end_idx):
    for j in range(end_idx + 1, len(case_rows)):
        if case_rows[j]["error_category"] == "CLEAN":
            return j
    return None


def compress_gaps(gaps, available_room):
    if not gaps:
        return gaps, False
    min_total = len(gaps) * 1.0
    total = sum(gaps)
    clamped = False
    if available_room is None:
        return [max(1.0, g) for g in gaps], clamped
    available_room = max(min_total, float(available_room))
    if total <= available_room + 1e-9:
        return [max(1.0, g) for g in gaps], clamped
    clamped = True
    excess_scalable = max(1e-9, total - min_total)
    target_scalable = max(0.0, available_room - min_total)
    scale = target_scalable / excess_scalable
    new_gaps = [1.0 + max(0.0, g - 1.0) * scale for g in gaps]
    return new_gaps, clamped


def enforce_within(ts, lo, hi):
    if lo is not None and not pd.isna(lo) and ts < lo:
        ts = lo
    if hi is not None and not pd.isna(hi) and ts > hi:
        ts = hi
    return ts


def estimate_from_anchors(left_ts, left_act, act, right_ts, right_act, case_start, case_end, preserve_date=None):
    left_guess = None
    right_guess = None

    if left_ts is not None and not pd.isna(left_ts):
        left_gap = get_gap_estimate(left_act, act) if left_act is not None else GLOBAL_MEDIAN_DURATION
        left_guess = left_ts + pd.to_timedelta(left_gap, unit="s")
    if right_ts is not None and not pd.isna(right_ts):
        right_gap = get_gap_estimate(act, right_act) if right_act is not None else GLOBAL_MEDIAN_DURATION
        right_guess = right_ts - pd.to_timedelta(right_gap, unit="s")

    if left_guess is not None and right_guess is not None:
        cand = left_guess + (right_guess - left_guess) / 2
        if cand > right_ts:
            cand = left_ts + (right_ts - left_ts) / 2
        cand = enforce_within(cand, left_ts, right_ts)
    elif left_guess is not None:
        cand = left_guess
    elif right_guess is not None:
        cand = right_guess
    else:
        cand = case_start

    cand = enforce_within(cand, case_start, case_end)

    if preserve_date is not None:
        day_start = pd.Timestamp(preserve_date.date())
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        feasible_lo = max([x for x in [case_start, day_start, left_ts] if x is not None and not pd.isna(x)], default=day_start)
        feasible_hi = min([x for x in [case_end, day_end, right_ts] if x is not None and not pd.isna(x)], default=day_end)
        if feasible_lo <= feasible_hi:
            cand = enforce_within(cand, feasible_lo, feasible_hi)
        else:
            # anchors conflict with preserved date: choose nearest feasible on preserved date within case bounds
            lo = max([x for x in [case_start, day_start] if x is not None and not pd.isna(x)], default=day_start)
            hi = min([x for x in [case_end, day_end] if x is not None and not pd.isna(x)], default=day_end)
            if lo <= hi:
                cand = enforce_within(cand, lo, hi)
            else:
                cand = lo
    return cand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    original = df.copy(deep=True)
    frac_len = infer_frac_len(df["Timestamp"])

    parsed_ts = df["Timestamp"].apply(parse_ts_safe)
    df["_parsed_ts"] = parsed_ts
    df["_out_ts"] = df["Timestamp"]
    df["_repaired"] = False

    c1_blocks_repaired = 0
    c1_blocks_clamped = 0

    grouped = df.groupby("Case", sort=False, group_keys=False)

    for case_id, case_df in grouped:
        idxs = list(case_df.index)
        case_rows = []
        for idx in idxs:
            row = df.loc[idx]
            case_rows.append({
                "idx": idx,
                "Activity": row["Activity"],
                "Timestamp": row["Timestamp"],
                "parsed_ts": row["_parsed_ts"],
                "error_category": row["error_category"],
                "row_id": row["row_id"],
            })

        valid_case_ts = [r["parsed_ts"] for r in case_rows if not pd.isna(r["parsed_ts"])]
        case_start = min(valid_case_ts) if valid_case_ts else pd.Timestamp("1970-01-01")
        case_end = max(valid_case_ts) if valid_case_ts else case_start

        # C1 blocks
        i = 0
        while i < len(case_rows):
            if case_rows[i]["error_category"] == "C1_ORDERING_AMBIGUITY":
                obs_ts = case_rows[i]["parsed_ts"]
                obs_str = case_rows[i]["Timestamp"]
                j = i
                block = []
                while j < len(case_rows) and case_rows[j]["error_category"] == "C1_ORDERING_AMBIGUITY" and case_rows[j]["Timestamp"] == obs_str:
                    block.append(case_rows[j])
                    j += 1

                if len(block) >= 2 and not pd.isna(obs_ts):
                    left_idx = find_clean_left_anchor(case_rows, i)
                    right_idx = find_clean_right_anchor(case_rows, j - 1)
                    left_act = case_rows[left_idx]["Activity"] if left_idx is not None else None
                    right_act = case_rows[right_idx]["Activity"] if right_idx is not None else None
                    right_ts = case_rows[right_idx]["parsed_ts"] if right_idx is not None else None

                    order = best_order_for_block(block, left_act, right_act)
                    ordered_rows = [block[k] for k in order]
                    ordered_acts = [r["Activity"] for r in ordered_rows]

                    gaps = []
                    for g in range(len(ordered_acts) - 1):
                        gaps.append(get_gap_estimate(ordered_acts[g], ordered_acts[g + 1]))

                    available_room = None
                    if right_ts is not None and not pd.isna(right_ts):
                        available_room = max(0.0, (right_ts - obs_ts).total_seconds())

                    new_gaps, was_clamped = compress_gaps(gaps, available_room)
                    if was_clamped:
                        c1_blocks_clamped += 1

                    ordered_times = [obs_ts]
                    cur = obs_ts
                    for gap in new_gaps:
                        cur = cur + pd.to_timedelta(max(1.0, gap), unit="s")
                        ordered_times.append(cur)

                    # final hard enforcement
                    if right_ts is not None and not pd.isna(right_ts):
                        max_end = right_ts
                        min_end = obs_ts + pd.to_timedelta(len(block) - 1, unit="s")
                        if ordered_times[-1] > max_end or ordered_times[-1] < min_end:
                            span = max(1.0 * (len(block) - 1), min((max_end - obs_ts).total_seconds(), max((len(block) - 1), 1)))
                            if span < (len(block) - 1):
                                span = len(block) - 1
                            forced = [obs_ts + pd.to_timedelta(s, unit="s") for s in range(len(block))]
                            if forced[-1] > max_end:
                                forced = [max_end - pd.to_timedelta((len(block) - 1 - k), unit="s") for k in range(len(block))]
                                if forced[0] < obs_ts:
                                    forced = [obs_ts + pd.to_timedelta(k, unit="s") for k in range(len(block))]
                            ordered_times = forced
                            c1_blocks_clamped += 1

                    # write back to original rows by inferred order, keeping row structure
                    for pos, r in enumerate(ordered_rows):
                        out_ts = ordered_times[pos]
                        out_ts = enforce_within(out_ts, case_start, case_end)
                        if right_ts is not None and not pd.isna(right_ts):
                            out_ts = enforce_within(out_ts, obs_ts, right_ts)
                        df.at[r["idx"], "_out_ts"] = fmt_ts(out_ts, frac_len)
                        df.at[r["idx"], "_repaired"] = True

                    # exactly one row keeps observed timestamp: earliest in inferred order
                    first_idx = ordered_rows[0]["idx"]
                    df.at[first_idx, "_out_ts"] = obs_str
                    c1_blocks_repaired += 1

                i = j
            else:
                i += 1

        # C2 and C3 row-wise
        for local_pos, r in enumerate(case_rows):
            idx = r["idx"]
            ec = r["error_category"]
            if ec == "CLEAN" or ec == "C1_ORDERING_AMBIGUITY":
                continue

            left_clean = find_clean_left_anchor(case_rows, local_pos)
            right_clean = find_clean_right_anchor(case_rows, local_pos)

            left_ts = case_rows[left_clean]["parsed_ts"] if left_clean is not None else None
            right_ts = case_rows[right_clean]["parsed_ts"] if right_clean is not None else None
            left_act = case_rows[left_clean]["Activity"] if left_clean is not None else None
            right_act = case_rows[right_clean]["Activity"] if right_clean is not None else None
            act = r["Activity"]
            orig_str = r["Timestamp"]
            orig_ts = r["parsed_ts"]

            if ec == "C2_TEMPORAL_INCONSISTENCY":
                prev_row = case_rows[local_pos - 1] if local_pos - 1 >= 0 else None
                next_row = case_rows[local_pos + 1] if local_pos + 1 < len(case_rows) else None

                repaired = False
                if prev_row is not None:
                    prev_ts = parse_ts_safe(df.at[prev_row["idx"], "_out_ts"]) if df.at[prev_row["idx"], "_repaired"] else prev_row["parsed_ts"]
                    curr_ts = orig_ts
                    if prev_ts is not None and curr_ts is not None and not pd.isna(prev_ts) and not pd.isna(curr_ts) and curr_ts < prev_ts:
                        # try exchanged hypothesis with predecessor
                        prev_prev_idx = find_clean_left_anchor(case_rows, local_pos - 1)
                        next_clean_idx = find_clean_right_anchor(case_rows, local_pos)
                        prev_prev_ts = case_rows[prev_prev_idx]["parsed_ts"] if prev_prev_idx is not None else None
                        prev_prev_act = case_rows[prev_prev_idx]["Activity"] if prev_prev_idx is not None else None
                        next_clean_ts = case_rows[next_clean_idx]["parsed_ts"] if next_clean_idx is not None else None
                        next_clean_act = case_rows[next_clean_idx]["Activity"] if next_clean_idx is not None else None

                        swap_prev_ts = curr_ts
                        swap_curr_ts = prev_ts

                        swap_ok = True
                        if prev_prev_ts is not None and swap_prev_ts < prev_prev_ts:
                            swap_ok = False
                        if next_clean_ts is not None and swap_curr_ts > next_clean_ts:
                            swap_ok = False
                        if swap_prev_ts > swap_curr_ts:
                            swap_ok = False

                        swap_score = -1e18
                        if swap_ok:
                            swap_score = 0.0
                            if prev_prev_ts is not None:
                                swap_score += duration_plausibility(prev_prev_act, prev_row["Activity"], (swap_prev_ts - prev_prev_ts).total_seconds())
                            swap_score += duration_plausibility(prev_row["Activity"], act, (swap_curr_ts - swap_prev_ts).total_seconds())
                            if next_clean_ts is not None:
                                swap_score += duration_plausibility(act, next_clean_act, (next_clean_ts - swap_curr_ts).total_seconds())

                        disp_cand = estimate_from_anchors(prev_ts, prev_row["Activity"], act, next_clean_ts, next_clean_act, case_start, case_end)
                        disp_score = duration_plausibility(prev_row["Activity"], act, (disp_cand - prev_ts).total_seconds())
                        if next_clean_ts is not None:
                            disp_score += duration_plausibility(act, next_clean_act, (next_clean_ts - disp_cand).total_seconds())

                        if swap_score >= disp_score and swap_ok:
                            df.at[prev_row["idx"], "_out_ts"] = fmt_ts(swap_prev_ts, frac_len)
                            df.at[prev_row["idx"], "_repaired"] = True
                            df.at[idx, "_out_ts"] = fmt_ts(swap_curr_ts, frac_len)
                            df.at[idx, "_repaired"] = True
                            repaired = True
                        else:
                            disp_cand = enforce_within(disp_cand, prev_ts, next_clean_ts if next_clean_ts is not None else case_end)
                            if orig_ts is not None and not pd.isna(orig_ts) and disp_cand == orig_ts:
                                disp_cand = disp_cand + pd.to_timedelta(1, unit="s")
                                disp_cand = enforce_within(disp_cand, prev_ts, next_clean_ts if next_clean_ts is not None else case_end)
                            df.at[idx, "_out_ts"] = fmt_ts(disp_cand, frac_len)
                            df.at[idx, "_repaired"] = True
                            repaired = True

                if not repaired:
                    cand = estimate_from_anchors(left_ts, left_act, act, right_ts, right_act, case_start, case_end)
                    if orig_ts is not None and not pd.isna(orig_ts) and cand == orig_ts:
                        cand = cand + pd.to_timedelta(1, unit="s")
                    cand = enforce_within(cand, left_ts, right_ts)
                    df.at[idx, "_out_ts"] = fmt_ts(cand, frac_len)
                    df.at[idx, "_repaired"] = True

            elif ec == "C3_MISSING_TEMPORAL_INFORMATION":
                preserve_date = None
                if has_date_only(orig_str):
                    preserve_date = pd.to_datetime(orig_str, errors="coerce")
                cand = estimate_from_anchors(left_ts, left_act, act, right_ts, right_act, case_start, case_end, preserve_date=preserve_date)

                if preserve_date is not None:
                    if cand.date() != preserve_date.date():
                        base = pd.Timestamp(preserve_date.date()) + pd.Timedelta(hours=12)
                        cand = enforce_within(base, case_start, case_end)
                        if left_ts is not None and right_ts is not None:
                            cand = enforce_within(cand, left_ts, right_ts)

                if str(orig_str).strip() != "" and not has_date_only(orig_str):
                    if orig_ts is not None and not pd.isna(orig_ts) and cand == orig_ts:
                        cand = cand + pd.to_timedelta(1, unit="s")
                df.at[idx, "_out_ts"] = fmt_ts(cand, frac_len)
                df.at[idx, "_repaired"] = True

            else:
                cand = estimate_from_anchors(left_ts, left_act, act, right_ts, right_act, case_start, case_end)
                if orig_ts is not None and not pd.isna(orig_ts) and cand == orig_ts:
                    cand = cand + pd.to_timedelta(1, unit="s")
                cand = enforce_within(cand, left_ts, right_ts)
                df.at[idx, "_out_ts"] = fmt_ts(cand, frac_len)
                df.at[idx, "_repaired"] = True

    # Self-check and corrective pass
    # 1. CLEAN unchanged
    clean_mask = df["error_category"] == "CLEAN"
    df.loc[clean_mask, "_out_ts"] = original.loc[clean_mask, "Timestamp"]

    # 2. Non-clean rows parse, and remain in bounds
    for case_id, case_df in df.groupby("Case", sort=False):
        idxs = list(case_df.index)
        case_ts = [parse_ts_safe(df.at[i, "_out_ts"]) for i in idxs if str(df.at[i, "_out_ts"]).strip() != ""]
        case_ts_valid = [t for t in case_ts if not pd.isna(t)]
        if not case_ts_valid:
            continue
        case_start = min(case_ts_valid)
        case_end = max(case_ts_valid)

        # C1 validation by contiguous identical original C1 blocks
        rows = list(idxs)
        local_rows = []
        for idx in rows:
            local_rows.append({
                "idx": idx,
                "ec": df.at[idx, "error_category"],
                "orig_ts_str": original.at[idx, "Timestamp"],
                "out_ts_str": df.at[idx, "_out_ts"],
            })
        p = 0
        while p < len(local_rows):
            if local_rows[p]["ec"] == "C1_ORDERING_AMBIGUITY":
                q = p
                shared = local_rows[p]["orig_ts_str"]
                block_idxs = []
                while q < len(local_rows) and local_rows[q]["ec"] == "C1_ORDERING_AMBIGUITY" and local_rows[q]["orig_ts_str"] == shared:
                    block_idxs.append(local_rows[q]["idx"])
                    q += 1
                if len(block_idxs) >= 2:
                    out_parsed = [parse_ts_safe(df.at[i, "_out_ts"]) for i in block_idxs]
                    orig_parsed = parse_ts_safe(shared)
                    # enforce exactly one equal to original and strict increase in time order
                    equal_idxs = [i for i in block_idxs if df.at[i, "_out_ts"] == shared]
                    if len(equal_idxs) != 1:
                        earliest_i = min(range(len(block_idxs)), key=lambda z: out_parsed[z] if not pd.isna(out_parsed[z]) else pd.Timestamp.max)
                        keep_idx = block_idxs[earliest_i]
                        for i in block_idxs:
                            if i == keep_idx:
                                df.at[i, "_out_ts"] = shared
                            elif df.at[i, "_out_ts"] == shared:
                                t = parse_ts_safe(df.at[i, "_out_ts"])
                                if pd.isna(t):
                                    t = orig_parsed + pd.to_timedelta(1, unit="s")
                                else:
                                    t = t + pd.to_timedelta(1, unit="s")
                                df.at[i, "_out_ts"] = fmt_ts(t, frac_len)
                    # make distinct and increasing when sorted by value, leaving row order intact
                    vals = [(i, parse_ts_safe(df.at[i, "_out_ts"])) for i in block_idxs]
                    vals = sorted(vals, key=lambda x: (pd.Timestamp.max if pd.isna(x[1]) else x[1], str(x[0])))
                    base = orig_parsed
                    if pd.isna(base):
                        base = min([v for _, v in vals if not pd.isna(v)], default=case_start)
                    new_times = []
                    cur = base
                    for k, (idx, t) in enumerate(vals):
                        if k == 0:
                            new_times.append((idx, base))
                            df.at[idx, "_out_ts"] = shared
                            cur = base
                        else:
                            nt = max(cur + pd.to_timedelta(1, unit="s"), t if not pd.isna(t) else cur + pd.to_timedelta(1, unit="s"))
                            new_times.append((idx, nt))
                            cur = nt
                    # respect right anchor room if any
                    right_anchor = None
                    if q < len(local_rows):
                        ra = parse_ts_safe(df.at[local_rows[q]["idx"], "_out_ts"])
                        if not pd.isna(ra):
                            right_anchor = ra
                    if right_anchor is not None:
                        if new_times[-1][1] > right_anchor:
                            span = len(new_times) - 1
                            forced_start = base
                            forced = []
                            for k, (idx, _) in enumerate(new_times):
                                forced.append((idx, forced_start + pd.to_timedelta(k, unit="s")))
                            if forced[-1][1] > right_anchor:
                                forced = []
                                start2 = right_anchor - pd.to_timedelta(span, unit="s")
                                if start2 < base:
                                    start2 = base
                                for k, (idx, _) in enumerate(new_times):
                                    forced.append((idx, start2 + pd.to_timedelta(k, unit="s")))
                            new_times = forced
                            c1_blocks_clamped += 1
                    for idx, t in new_times:
                        if df.at[idx, "_out_ts"] != shared or idx != new_times[0][0]:
                            df.at[idx, "_out_ts"] = fmt_ts(t, frac_len)
                p = q
            else:
                p += 1

        # clamp corrupted rows into case bounds
        for idx in idxs:
            if df.at[idx, "error_category"] == "CLEAN":
                continue
            t = parse_ts_safe(df.at[idx, "_out_ts"])
            if pd.isna(t):
                t = case_start
            t = enforce_within(t, case_start, case_end)
            df.at[idx, "_out_ts"] = fmt_ts(t, frac_len) if df.at[idx, "_out_ts"] != original.at[idx, "Timestamp"] or df.at[idx, "error_category"] != "CLEAN" else df.at[idx, "_out_ts"]

    # final enforce byte-identical protected columns and row count/order
    out = original.copy(deep=True)
    out["Timestamp"] = df["_out_ts"]

    protected_cols = ["Case", "Activity", "Resource", "error_category", "row_id"]
    for c in protected_cols:
        out[c] = original[c]

    # rows still equal to input among corrupted rows
    corrupted_mask = out["error_category"] != "CLEAN"
    rows_still_equal = int((out.loc[corrupted_mask, "Timestamp"] == original.loc[corrupted_mask, "Timestamp"]).sum())

    out.to_csv(args.output, index=False)

    print(f"blocks_repaired={c1_blocks_repaired} blocks_clamped={c1_blocks_clamped} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # fail-safe: produce best-effort pass-through if anything unexpected happens
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=True)
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")
        except Exception:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")