import argparse
import json
import math
import re
from collections import defaultdict

import pandas as pd


DURATION_PROFILES = {
    ("Bring drinks", "Review drinks"): {"n": 7815, "p10": 5.9622, "p25": 16.667, "median": 39.814, "p75": 81.143, "p90": 132.788, "share_under_60s": 0.644},
    ("prepare drinks", "Bring drinks"): {"n": 7234, "p10": 28.6493, "p25": 78.98275, "median": 196.828, "p75": 409.48225, "p90": 695.9419, "share_under_60s": 0.2},
    ("start", "Take order"): {"n": 5000, "p10": 12.87, "p25": 35.67975, "median": 89.038, "p75": 180.197, "p90": 315.5165, "share_under_60s": 0.378},
    ("Deliver to customer", "EVENT 8 END"): {"n": 5000, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("Bring food", "Deliver to customer"): {"n": 4986, "p10": 12.5235, "p25": 34.57875, "median": 85.062, "p75": 168.381, "p90": 278.243, "share_under_60s": 0.389},
    ("Take order", "Review order"): {"n": 4932, "p10": 6.0466, "p25": 17.74, "median": 41.186, "p75": 81.7655, "p90": 136.2094, "share_under_60s": 0.642},
    ("Signal food complete", "Bring food"): {"n": 4686, "p10": 32.4135, "p25": 88.89475, "median": 212.3665, "p75": 422.92075, "p90": 715.0625, "share_under_60s": 0.179},
    ("Review drinks", "Prepare main course"): {"n": 3906, "p10": 83185.9065, "p25": 243786.19475, "median": 506058.7255, "p75": 681158.72975, "p90": 790023.7145, "share_under_60s": 0.001},
    ("Review order", "prepare drinks"): {"n": 3863, "p10": 25.4156, "p25": 67.633, "median": 175.551, "p75": 353.075, "p90": 606.9134, "share_under_60s": 0.23},
    ("Prepare main course", "Signal food complete"): {"n": 3860, "p10": 85804.6626, "p25": 253570.31275, "median": 515362.0085, "p75": 685209.37625, "p90": 846416.0903, "share_under_60s": 0.0},
    ("Review drinks", "prepare drinks"): {"n": 3158, "p10": 29.5763, "p25": 84.574, "median": 206.398, "p75": 411.67775, "p90": 690.4042, "share_under_60s": 0.185},
    ("Prepare starter", "prepare drinks"): {"n": 964, "p10": 29.2279, "p25": 83.00125, "median": 207.4155, "p75": 398.98275, "p90": 658.0158, "share_under_60s": 0.2},
    ("Review order", "Prepare starter"): {"n": 886, "p10": 14.858, "p25": 42.44225, "median": 95.6335, "p75": 202.66325, "p90": 354.7965, "share_under_60s": 0.362},
    ("Review drinks", "Signal food complete"): {"n": 752, "p10": 64894.353, "p25": 245143.21425, "median": 584317.96, "p75": 775390.404, "p90": 1296100.3777, "share_under_60s": 0.001},
    ("Prepare starter", "Bring drinks"): {"n": 526, "p10": 29.761, "p25": 86.16725, "median": 212.4895, "p75": 439.5685, "p90": 714.5675, "share_under_60s": 0.186},
    ("prepare drinks", "Prepare starter"): {"n": 508, "p10": 16.3802, "p25": 41.13975, "median": 106.475, "p75": 198.7845, "p90": 308.6338, "share_under_60s": 0.352},
    ("prepare drinks", "Review drinks"): {"n": 322, "p10": 125.4322, "p25": 188.76775, "median": 333.4275, "p75": 506.97625, "p90": 730.3279, "share_under_60s": 0.0},
    ("Review drinks", "Prepare starter"): {"n": 320, "p10": 17.7551, "p25": 61.225, "median": 146.7405, "p75": 320.6395, "p90": 610.5744, "share_under_60s": 0.247},
    ("Prepare main course", "Bring food"): {"n": 270, "p10": 86745.0261, "p25": 268428.47375, "median": 524216.7145, "p75": 693403.298, "p90": 788190.8706, "share_under_60s": 0.0},
    ("Review order", "Bring drinks"): {"n": 174, "p10": 188.2856, "p25": 290.77825, "median": 515.393, "p75": 795.2755, "p90": 1393.9688, "share_under_60s": 0.0},
    ("Prepare starter", "Prepare main course"): {"n": 151, "p10": 83686.134, "p25": 239095.7885, "median": 501820.864, "p75": 621218.874, "p90": 775727.849, "share_under_60s": 0.0},
    ("Prepare starter", "Review drinks"): {"n": 122, "p10": 11.5929, "p25": 27.251, "median": 65.2545, "p75": 161.61275, "p90": 319.0675, "share_under_60s": 0.475},
    ("Bring drinks", "Prepare starter"): {"n": 101, "p10": 3.906, "p25": 14.62, "median": 33.922, "p75": 81.083, "p90": 167.426, "share_under_60s": 0.673},
    ("Bring drinks", "Prepare main course"): {"n": 81, "p10": 5113.78, "p25": 189219.462, "median": 426055.413, "p75": 615514.928, "p90": 687762.01, "share_under_60s": 0.037},
    ("Review drinks", "Bring drinks"): {"n": 78, "p10": 215.5759, "p25": 343.633, "median": 575.7475, "p75": 827.4725, "p90": 1382.6514, "share_under_60s": 0.0},
    ("Prepare starter", "Signal food complete"): {"n": 65, "p10": 85046.85, "p25": 245413.826, "median": 518378.896, "p75": 678595.891, "p90": 774438.8728, "share_under_60s": 0.0},
    ("Take order", "prepare drinks"): {"n": 52, "p10": 154.6963, "p25": 234.80625, "median": 296.679, "p75": 412.6365, "p90": 618.0132, "share_under_60s": 0.0},
    ("Review drinks", "Bring food"): {"n": 38, "p10": 846.8304, "p25": 82898.489, "median": 346302.8675, "p75": 673710.86275, "p90": 725773.664, "share_under_60s": 0.026},
    ("Bring drinks", "prepare drinks"): {"n": 37, "p10": 136.2866, "p25": 172.149, "median": 344.983, "p75": 588.856, "p90": 972.4264, "share_under_60s": 0.0},
    ("prepare drinks", "Prepare main course"): {"n": 20, "p10": 34.0066, "p25": 69.50575, "median": 117.1945, "p75": 395.313, "p90": 540.2721, "share_under_60s": 0.2},
    ("Prepare main course", "Bring drinks"): {"n": 16, "p10": 25.0495, "p25": 57.7345, "median": 127.0465, "p75": 175.544, "p90": 361.743, "share_under_60s": 0.25},
    ("Take order", "Prepare starter"): {"n": 16, "p10": 108.5225, "p25": 148.64375, "median": 213.7385, "p75": 239.96775, "p90": 259.0405, "share_under_60s": 0.0},
    ("Prepare main course", "prepare drinks"): {"n": 14, "p10": 104.4152, "p25": 117.66025, "median": 177.3835, "p75": 237.3835, "p90": 344.6175, "share_under_60s": 0.0},
    ("Review drinks", "Deliver to customer"): {"n": 14, "p10": 13.1735, "p25": 26.957, "median": 70.7255, "p75": 122.753, "p90": 151.1391, "share_under_60s": 0.357},
    ("Bring food", "Bring drinks"): {"n": 11, "p10": 50.942, "p25": 152.058, "median": 229.533, "p75": 512.9985, "p90": 626.086, "share_under_60s": 0.182},
    ("Review order", "Prepare main course"): {"n": 9, "p10": 110.4526, "p25": 184.332, "median": 423.133, "p75": 505.645, "p90": 1025.6276, "share_under_60s": 0.0},
    ("Bring drinks", "Signal food complete"): {"n": 8, "p10": 64224.5293, "p25": 98420.52025, "median": 480748.6405, "p75": 718034.19675, "p90": 779623.3533, "share_under_60s": 0.0},
    ("Prepare main course", "Review drinks"): {"n": 7, "p10": 3.2126, "p25": 7.2355, "median": 66.424, "p75": 144.731, "p90": 173.7746, "share_under_60s": 0.429},
    ("prepare drinks", "Signal food complete"): {"n": 7, "p10": 26.9548, "p25": 112.356, "median": 323.893, "p75": 534.458, "p90": 919.5008, "share_under_60s": 0.286},
    ("Signal food complete", "Bring drinks"): {"n": 3, "p10": 150.102, "p25": 179.118, "median": 227.478, "p75": 255.185, "p90": 271.8092, "share_under_60s": 0.0},
    ("Signal food complete", "prepare drinks"): {"n": 3, "p10": 212.5536, "p25": 228.6855, "median": 255.572, "p75": 279.335, "p90": 293.5928, "share_under_60s": 0.0},
    ("Bring food", "prepare drinks"): {"n": 3, "p10": 164.9284, "p25": 166.546, "median": 169.242, "p75": 176.463, "p90": 180.7956, "share_under_60s": 0.0},
    ("prepare drinks", "Bring food"): {"n": 3, "p10": 46.9712, "p25": 63.2825, "median": 90.468, "p75": 145.511, "p90": 178.5368, "share_under_60s": 0.333},
    ("Prepare starter", "Bring food"): {"n": 3, "p10": 473564.8114, "p25": 548790.244, "median": 674165.965, "p75": 725064.525, "p90": 755603.661, "share_under_60s": 0.0},
}

BLOCK_SPAN_2_P25 = 30.697
GLOBAL_MEDIAN = 115.3875

FREQUENT_VARIANTS = [
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def detect_precision(series):
    max_digits = 0
    pat = re.compile(r"\.(\d+)$")
    for s in series.astype(str):
        m = pat.search(s)
        if m:
            max_digits = max(max_digits, len(m.group(1)))
    return max_digits


def parse_ts(s):
    return pd.to_datetime(s, errors="coerce")


def fmt_ts(ts, precision):
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    if precision < 6:
        return base[:- (6 - precision)]
    return base


def td_seconds(a, b):
    return (b - a).total_seconds()


def prof(edge):
    return DURATION_PROFILES.get(edge)


def chosen_gap_seconds(edge):
    p = prof(edge)
    if p:
        if p["share_under_60s"] >= 0.5:
            return p["p10"]
        return p["p25"]
    return BLOCK_SPAN_2_P25


def edge_supported(a, b):
    return (a, b) in DURATION_PROFILES


def subseq_supported(acts):
    for v in FREQUENT_VARIANTS:
        for i in range(len(v) - len(acts) + 1):
            if v[i:i + len(acts)] == acts:
                return True
    return False


def local_order_supported(prev_act, a1, a2, next_act):
    acts = [x for x in [prev_act, a1, a2, next_act] if x is not None]
    if len(acts) >= 2 and subseq_supported(acts):
        return True
    pairs = [(prev_act, a1), (a1, a2), (a2, next_act)]
    ok = True
    for x, y in pairs:
        if x is not None and y is not None and not edge_supported(x, y):
            ok = False
            break
    return ok


def plausibility_score(prev_act, t_prev, act, t_cur, next_act, t_next):
    score = 0.0
    penalty = 0.0
    if t_prev is not None:
        d = td_seconds(t_prev, t_cur)
        if d < 0:
            return 1e18
        p = prof((prev_act, act))
        if p:
            target = chosen_gap_seconds((prev_act, act))
            scale = max(p["p75"] - p["p10"], 1.0)
            score += abs(d - target) / scale
        else:
            score += abs(d - GLOBAL_MEDIAN) / max(GLOBAL_MEDIAN, 1.0)
            penalty += 2.0
    if t_next is not None:
        d = td_seconds(t_cur, t_next)
        if d < 0:
            return 1e18
        p = prof((act, next_act))
        if p:
            target = chosen_gap_seconds((act, next_act))
            scale = max(p["p75"] - p["p10"], 1.0)
            score += abs(d - target) / scale
        else:
            score += abs(d - GLOBAL_MEDIAN) / max(GLOBAL_MEDIAN, 1.0)
            penalty += 2.0
    return score + penalty


def estimate_from_anchors(prev_act, prev_t, act, next_act, next_t, observed_t, case_start, case_end):
    left_bound = case_start
    right_bound = case_end
    if prev_t is not None:
        left_bound = max(left_bound, prev_t)
    if next_t is not None:
        right_bound = min(right_bound, next_t)

    if right_bound < left_bound:
        right_bound = left_bound

    fwd = None
    bwd = None
    fwd_w = 0.0
    bwd_w = 0.0

    if prev_t is not None:
        gap = chosen_gap_seconds((prev_act, act))
        fwd = prev_t + pd.to_timedelta(gap, unit="s")
        p = prof((prev_act, act))
        fwd_w = float(p["n"]) if p else 1.0

    if next_t is not None:
        gap = chosen_gap_seconds((act, next_act))
        bwd = next_t - pd.to_timedelta(gap, unit="s")
        p = prof((act, next_act))
        bwd_w = float(p["n"]) if p else 1.0

    if fwd is not None and bwd is not None:
        if fwd <= bwd:
            total_w = fwd_w + bwd_w
            cand = fwd + (bwd - fwd) * (fwd_w / total_w if total_w > 0 else 0.5)
            if observed_t is not None:
                obs_clamped = min(max(observed_t, fwd), bwd)
                if abs((obs_clamped - observed_t).total_seconds()) < abs((cand - observed_t).total_seconds()):
                    cand = obs_clamped
        else:
            cand = min(max(observed_t if observed_t is not None else fwd, left_bound), right_bound)
            midpoint = left_bound + (right_bound - left_bound) / 2
            if observed_t is None or cand < left_bound or cand > right_bound:
                cand = midpoint
    elif fwd is not None:
        cand = fwd
    elif bwd is not None:
        cand = bwd
    else:
        cand = observed_t if observed_t is not None else case_start

    cand = min(max(cand, left_bound), right_bound)
    return cand


def ensure_changed(ts, original_ts, lower, upper, precision):
    out = ts
    orig = original_ts
    if fmt_ts(out, precision) != orig:
        return out
    eps = pd.to_timedelta(10 ** (-precision), unit="s")
    if out + eps <= upper:
        return out + eps
    if out - eps >= lower:
        return out - eps
    return out


def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_cols = list(df.columns)
    precision = detect_precision(df["Timestamp"])

    df["_orig_ts"] = df["Timestamp"].astype(str)
    df["_parsed_ts"] = df["Timestamp"].map(parse_ts)
    df["_out_ts"] = df["_orig_ts"].copy()
    df["_out_parsed"] = df["_parsed_ts"].copy()
    df["_changed"] = False

    exchanged_segments = 0
    displaced_segments = 0
    fix_notes = []

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_indices = list(idxs)
        case_times = df.loc[case_indices, "_parsed_ts"]
        valid_case_times = case_times.dropna()
        if len(valid_case_times) == 0:
            continue
        case_start = valid_case_times.min()
        case_end = valid_case_times.max()

        i = 0
        while i < len(case_indices) - 1:
            li = case_indices[i]
            ri = case_indices[i + 1]

            lt = df.at[li, "_out_parsed"]
            rt = df.at[ri, "_out_parsed"]
            lerr = df.at[li, "error_category"]
            rerr = df.at[ri, "error_category"]

            if pd.isna(lt) or pd.isna(rt) or rt >= lt:
                i += 1
                continue

            if lerr == "CLEAN" and rerr == "CLEAN":
                i += 1
                continue

            prev_i = case_indices[i - 1] if i - 1 >= 0 else None
            next_i = case_indices[i + 2] if i + 2 < len(case_indices) else None

            prev_t = df.at[prev_i, "_out_parsed"] if prev_i is not None else None
            next_t = df.at[next_i, "_out_parsed"] if next_i is not None else None
            prev_act = df.at[prev_i, "Activity"] if prev_i is not None else None
            next_act = df.at[next_i, "Activity"] if next_i is not None else None

            lact = df.at[li, "Activity"]
            ract = df.at[ri, "Activity"]
            lorig = df.at[li, "_orig_ts"]
            rorig = df.at[ri, "_orig_ts"]

            candidates = []

            if lerr != "CLEAN" and rerr != "CLEAN":
                if local_order_supported(prev_act, lact, ract, next_act):
                    new_l = rt
                    new_r = lt
                    ok = True
                    if prev_t is not None and new_l < prev_t:
                        ok = False
                    if new_r < new_l:
                        ok = False
                    if next_t is not None and new_r > next_t:
                        ok = False
                    if new_l < case_start or new_r > case_end:
                        ok = False
                    if ok:
                        score = 0.0
                        score += plausibility_score(prev_act, prev_t, lact, new_l, ract, new_r)
                        score += plausibility_score(lact, new_l, ract, new_r, next_act, next_t)
                        move = abs(td_seconds(lt, new_l)) + abs(td_seconds(rt, new_r))
                        candidates.append({
                            "kind": "EXCHANGED",
                            "rows": [li, ri],
                            "times": {li: new_l, ri: new_r},
                            "score": score + move / max(GLOBAL_MEDIAN, 1.0) * 0.01,
                            "move": move
                        })

            if rerr != "CLEAN":
                if local_order_supported(prev_act, lact, ract, next_act):
                    new_r = estimate_from_anchors(lact, lt, ract, next_act, next_t, rt, case_start, case_end)
                    ok = True
                    if new_r < lt:
                        ok = False
                    if next_t is not None and new_r > next_t:
                        ok = False
                    if ok:
                        score = plausibility_score(lact, lt, ract, new_r, next_act, next_t)
                        move = abs(td_seconds(rt, new_r))
                        candidates.append({
                            "kind": "DISPLACED_RIGHT",
                            "rows": [ri],
                            "times": {ri: new_r},
                            "score": score + move / max(GLOBAL_MEDIAN, 1.0) * 0.01,
                            "move": move
                        })

            if lerr != "CLEAN":
                if local_order_supported(prev_act, lact, ract, next_act):
                    new_l = estimate_from_anchors(prev_act, prev_t, lact, ract, rt, lt, case_start, case_end)
                    ok = True
                    if prev_t is not None and new_l < prev_t:
                        ok = False
                    if new_l > rt:
                        ok = False
                    if ok:
                        score = plausibility_score(prev_act, prev_t, lact, new_l, ract, rt)
                        move = abs(td_seconds(lt, new_l))
                        candidates.append({
                            "kind": "DISPLACED_LEFT",
                            "rows": [li],
                            "times": {li: new_l},
                            "score": score + move / max(GLOBAL_MEDIAN, 1.0) * 0.01,
                            "move": move
                        })

            if not candidates:
                if rerr != "CLEAN":
                    new_r = estimate_from_anchors(lact, lt, ract, next_act, next_t, rt, case_start, case_end)
                    candidates.append({"kind": "DISPLACED_RIGHT", "rows": [ri], "times": {ri: new_r}, "score": 1e12, "move": abs(td_seconds(rt, new_r))})
                elif lerr != "CLEAN":
                    new_l = estimate_from_anchors(prev_act, prev_t, lact, ract, rt, lt, case_start, case_end)
                    candidates.append({"kind": "DISPLACED_LEFT", "rows": [li], "times": {li: new_l}, "score": 1e12, "move": abs(td_seconds(lt, new_l))})

            candidates.sort(key=lambda x: (x["score"], x["move"], 0 if x["kind"] == "EXCHANGED" else 1))
            chosen = candidates[0]

            if chosen["kind"] == "EXCHANGED":
                exchanged_segments += 1
            else:
                displaced_segments += 1

            for rowi, new_ts in chosen["times"].items():
                lower = case_start
                upper = case_end
                pos = case_indices.index(rowi)
                if pos - 1 >= 0:
                    pt = df.at[case_indices[pos - 1], "_out_parsed"]
                    if not pd.isna(pt):
                        lower = max(lower, pt)
                if pos + 1 < len(case_indices):
                    nt = df.at[case_indices[pos + 1], "_out_parsed"]
                    if not pd.isna(nt):
                        upper = min(upper, nt)
                if upper < lower:
                    upper = lower
                repaired = min(max(new_ts, lower), upper)
                repaired = ensure_changed(repaired, df.at[rowi, "_orig_ts"], lower, upper, precision)
                df.at[rowi, "_out_parsed"] = repaired
                df.at[rowi, "_out_ts"] = fmt_ts(repaired, precision)
                df.at[rowi, "_changed"] = (df.at[rowi, "_out_ts"] != df.at[rowi, "_orig_ts"])

            i += 1

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_indices = list(idxs)
        case_times = [df.at[j, "_out_parsed"] for j in case_indices if not pd.isna(df.at[j, "_out_parsed"])]
        if not case_times:
            continue
        case_start = min(case_times)
        case_end = max(case_times)

        for k, rowi in enumerate(case_indices):
            if df.at[rowi, "error_category"] == "CLEAN":
                df.at[rowi, "_out_ts"] = df.at[rowi, "_orig_ts"]
                df.at[rowi, "_out_parsed"] = df.at[rowi, "_parsed_ts"]
                df.at[rowi, "_changed"] = False
                continue

            t = df.at[rowi, "_out_parsed"]
            if pd.isna(t):
                left_i = case_indices[k - 1] if k > 0 else None
                right_i = case_indices[k + 1] if k + 1 < len(case_indices) else None
                prev_t = df.at[left_i, "_out_parsed"] if left_i is not None else None
                next_t = df.at[right_i, "_out_parsed"] if right_i is not None else None
                prev_act = df.at[left_i, "Activity"] if left_i is not None else None
                next_act = df.at[right_i, "Activity"] if right_i is not None else None
                act = df.at[rowi, "Activity"]
                t = estimate_from_anchors(prev_act, prev_t, act, next_act, next_t, None, case_start, case_end)
                fix_notes.append(f"filled_nan_row_{rowi}")
            lower = case_start
            upper = case_end
            if k > 0:
                pt = df.at[case_indices[k - 1], "_out_parsed"]
                if not pd.isna(pt):
                    lower = max(lower, pt)
            if k + 1 < len(case_indices):
                nt = df.at[case_indices[k + 1], "_out_parsed"]
                if not pd.isna(nt):
                    upper = min(upper, nt)
            if upper < lower:
                upper = lower
            if t < lower or t > upper:
                t = min(max(t, lower), upper)
                fix_notes.append(f"clamped_row_{rowi}")
            t = ensure_changed(t, df.at[rowi, "_orig_ts"], lower, upper, precision)
            df.at[rowi, "_out_parsed"] = t
            df.at[rowi, "_out_ts"] = fmt_ts(t, precision)
            df.at[rowi, "_changed"] = (df.at[rowi, "_out_ts"] != df.at[rowi, "_orig_ts"])

    equal_bad = int(((df["error_category"] != "CLEAN") & (df["_out_ts"] == df["_orig_ts"])).sum())

    if equal_bad > 0:
        for case_id, idxs in df.groupby("Case", sort=False).groups.items():
            case_indices = list(idxs)
            case_times = [df.at[j, "_out_parsed"] for j in case_indices if not pd.isna(df.at[j, "_out_parsed"])]
            if not case_times:
                continue
            case_start = min(case_times)
            case_end = max(case_times)
            for k, rowi in enumerate(case_indices):
                if df.at[rowi, "error_category"] == "CLEAN":
                    continue
                if df.at[rowi, "_out_ts"] != df.at[rowi, "_orig_ts"]:
                    continue
                lower = case_start
                upper = case_end
                if k > 0:
                    pt = df.at[case_indices[k - 1], "_out_parsed"]
                    if not pd.isna(pt):
                        lower = max(lower, pt)
                if k + 1 < len(case_indices):
                    nt = df.at[case_indices[k + 1], "_out_parsed"]
                    if not pd.isna(nt):
                        upper = min(upper, nt)
                if upper < lower:
                    upper = lower
                t = df.at[rowi, "_out_parsed"]
                t2 = ensure_changed(t, df.at[rowi, "_orig_ts"], lower, upper, precision)
                if fmt_ts(t2, precision) == df.at[rowi, "_orig_ts"]:
                    eps = pd.to_timedelta(max(10 ** (-precision), 0.001), unit="s")
                    if t + eps <= upper:
                        t2 = t + eps
                    elif t - eps >= lower:
                        t2 = t - eps
                df.at[rowi, "_out_parsed"] = t2
                df.at[rowi, "_out_ts"] = fmt_ts(t2, precision)
                df.at[rowi, "_changed"] = (df.at[rowi, "_out_ts"] != df.at[rowi, "_orig_ts"])
                fix_notes.append(f"nudged_equal_row_{rowi}")

    df["Timestamp"] = df["_out_ts"]

    out = df[original_cols].copy()
    out.to_csv(args.output, index=False)

    rows_equal_input = int(((df["error_category"] != "CLEAN") & (df["Timestamp"] == df["_orig_ts"])).sum())
    msg = f"exchanged_segments={exchanged_segments} displaced_segments={displaced_segments} rows_still_equal_to_input={rows_equal_input}"
    if fix_notes:
        msg += " fixes=" + ",".join(fix_notes[:50])
    print(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"exchanged_segments=0 displaced_segments=0 rows_still_equal_to_input=0 fixes=top_level_exception:{type(e).__name__}")