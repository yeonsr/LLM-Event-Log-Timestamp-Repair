import argparse
import csv
import math
from collections import defaultdict, Counter
import pandas as pd

# Only error present in this task:
ERR = "C2_TEMPORAL_INCONSISTENCY"

# ---------------------------------------------------------------------
# Process knowledge from prompt
# ---------------------------------------------------------------------
DURATION_PROFILES = {
    ("Bring drinks", "Review drinks"): {"n": 7815, "p10": 5.9622, "p25": 16.667, "median": 39.814, "p75": 81.143, "p90": 132.788, "share": 0.644},
    ("prepare drinks", "Bring drinks"): {"n": 7234, "p10": 28.6493, "p25": 78.98275, "median": 196.828, "p75": 409.48225, "p90": 695.9419, "share": 0.2},
    ("start", "Take order"): {"n": 5000, "p10": 12.87, "p25": 35.67975, "median": 89.038, "p75": 180.197, "p90": 315.5165, "share": 0.378},
    ("Deliver to customer", "EVENT 8 END"): {"n": 5000, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("Bring food", "Deliver to customer"): {"n": 4986, "p10": 12.5235, "p25": 34.57875, "median": 85.062, "p75": 168.381, "p90": 278.243, "share": 0.389},
    ("Take order", "Review order"): {"n": 4932, "p10": 6.0466, "p25": 17.74, "median": 41.186, "p75": 81.7655, "p90": 136.2094, "share": 0.642},
    ("Signal food complete", "Bring food"): {"n": 4686, "p10": 32.4135, "p25": 88.89475, "median": 212.3665, "p75": 422.92075, "p90": 715.0625, "share": 0.179},
    ("Review drinks", "Prepare main course"): {"n": 3906, "p10": 83185.9065, "p25": 243786.19475, "median": 506058.7255, "p75": 681158.72975, "p90": 790023.7145, "share": 0.001},
    ("Review order", "prepare drinks"): {"n": 3863, "p10": 25.4156, "p25": 67.633, "median": 175.551, "p75": 353.075, "p90": 606.9134, "share": 0.23},
    ("Prepare main course", "Signal food complete"): {"n": 3860, "p10": 85804.6626, "p25": 253570.31275, "median": 515362.0085, "p75": 685209.37625, "p90": 846416.0903, "share": 0.0},
    ("Review drinks", "prepare drinks"): {"n": 3158, "p10": 29.5763, "p25": 84.574, "median": 206.398, "p75": 411.67775, "p90": 690.4042, "share": 0.185},
    ("Prepare starter", "prepare drinks"): {"n": 964, "p10": 29.2279, "p25": 83.00125, "median": 207.4155, "p75": 398.98275, "p90": 658.0158, "share": 0.2},
    ("Review order", "Prepare starter"): {"n": 886, "p10": 14.858, "p25": 42.44225, "median": 95.6335, "p75": 202.66325, "p90": 354.7965, "share": 0.362},
    ("Review drinks", "Signal food complete"): {"n": 752, "p10": 64894.353, "p25": 245143.21425, "median": 584317.96, "p75": 775390.404, "p90": 1296100.3777, "share": 0.001},
    ("Prepare starter", "Bring drinks"): {"n": 526, "p10": 29.761, "p25": 86.16725, "median": 212.4895, "p75": 439.5685, "p90": 714.5675, "share": 0.186},
    ("prepare drinks", "Prepare starter"): {"n": 508, "p10": 16.3802, "p25": 41.13975, "median": 106.475, "p75": 198.7845, "p90": 308.6338, "share": 0.352},
    ("prepare drinks", "Review drinks"): {"n": 322, "p10": 125.4322, "p25": 188.76775, "median": 333.4275, "p75": 506.97625, "p90": 730.3279, "share": 0.0},
    ("Review drinks", "Prepare starter"): {"n": 320, "p10": 17.7551, "p25": 61.225, "median": 146.7405, "p75": 320.6395, "p90": 610.5744, "share": 0.247},
    ("Prepare main course", "Bring food"): {"n": 270, "p10": 86745.0261, "p25": 268428.47375, "median": 524216.7145, "p75": 693403.298, "p90": 788190.8706, "share": 0.0},
    ("Review order", "Bring drinks"): {"n": 174, "p10": 188.2856, "p25": 290.77825, "median": 515.393, "p75": 795.2755, "p90": 1393.9688, "share": 0.0},
    ("Prepare starter", "Prepare main course"): {"n": 151, "p10": 83686.134, "p25": 239095.7885, "median": 501820.864, "p75": 621218.874, "p90": 775727.849, "share": 0.0},
    ("Prepare starter", "Review drinks"): {"n": 122, "p10": 11.5929, "p25": 27.251, "median": 65.2545, "p75": 161.61275, "p90": 319.0675, "share": 0.475},
    ("Bring drinks", "Prepare starter"): {"n": 101, "p10": 3.906, "p25": 14.62, "median": 33.922, "p75": 81.083, "p90": 167.426, "share": 0.673},
    ("Bring drinks", "Prepare main course"): {"n": 81, "p10": 5113.78, "p25": 189219.462, "median": 426055.413, "p75": 615514.928, "p90": 687762.01, "share": 0.037},
    ("Review drinks", "Bring drinks"): {"n": 78, "p10": 215.5759, "p25": 343.633, "median": 575.7475, "p75": 827.4725, "p90": 1382.6514, "share": 0.0},
    ("Prepare starter", "Signal food complete"): {"n": 65, "p10": 85046.85, "p25": 245413.826, "median": 518378.896, "p75": 678595.891, "p90": 774438.8728, "share": 0.0},
    ("Take order", "prepare drinks"): {"n": 52, "p10": 154.6963, "p25": 234.80625, "median": 296.679, "p75": 412.6365, "p90": 618.0132, "share": 0.0},
    ("Review drinks", "Bring food"): {"n": 38, "p10": 846.8304, "p25": 82898.489, "median": 346302.8675, "p75": 673710.86275, "p90": 725773.664, "share": 0.026},
    ("Bring drinks", "prepare drinks"): {"n": 37, "p10": 136.2866, "p25": 172.149, "median": 344.983, "p75": 588.856, "p90": 972.4264, "share": 0.0},
    ("prepare drinks", "Prepare main course"): {"n": 20, "p10": 34.0066, "p25": 69.50575, "median": 117.1945, "p75": 395.313, "p90": 540.2721, "share": 0.2},
    ("Prepare main course", "Bring drinks"): {"n": 16, "p10": 25.0495, "p25": 57.7345, "median": 127.0465, "p75": 175.544, "p90": 361.743, "share": 0.25},
    ("Take order", "Prepare starter"): {"n": 16, "p10": 108.5225, "p25": 148.64375, "median": 213.7385, "p75": 239.96775, "p90": 259.0405, "share": 0.0},
    ("Prepare main course", "prepare drinks"): {"n": 14, "p10": 104.4152, "p25": 117.66025, "median": 177.3835, "p75": 237.3835, "p90": 344.6175, "share": 0.0},
    ("Review drinks", "Deliver to customer"): {"n": 14, "p10": 13.1735, "p25": 26.957, "median": 70.7255, "p75": 122.753, "p90": 151.1391, "share": 0.357},
    ("Bring food", "Bring drinks"): {"n": 11, "p10": 50.942, "p25": 152.058, "median": 229.533, "p75": 512.9985, "p90": 626.086, "share": 0.182},
    ("Review order", "Prepare main course"): {"n": 9, "p10": 110.4526, "p25": 184.332, "median": 423.133, "p75": 505.645, "p90": 1025.6276, "share": 0.0},
    ("Bring drinks", "Signal food complete"): {"n": 8, "p10": 64224.5293, "p25": 98420.52025, "median": 480748.6405, "p75": 718034.19675, "p90": 779623.3533, "share": 0.0},
    ("Prepare main course", "Review drinks"): {"n": 7, "p10": 3.2126, "p25": 7.2355, "median": 66.424, "p75": 144.731, "p90": 173.7746, "share": 0.429},
    ("prepare drinks", "Signal food complete"): {"n": 7, "p10": 26.9548, "p25": 112.356, "median": 323.893, "p75": 534.458, "p90": 919.5008, "share": 0.286},
    ("Signal food complete", "Bring drinks"): {"n": 3, "p10": 150.102, "p25": 179.118, "median": 227.478, "p75": 255.185, "p90": 271.8092, "share": 0.0},
    ("Signal food complete", "prepare drinks"): {"n": 3, "p10": 212.5536, "p25": 228.6855, "median": 255.572, "p75": 279.335, "p90": 293.5928, "share": 0.0},
    ("Bring food", "prepare drinks"): {"n": 3, "p10": 164.9284, "p25": 166.546, "median": 169.242, "p75": 176.463, "p90": 180.7956, "share": 0.0},
    ("prepare drinks", "Bring food"): {"n": 3, "p10": 46.9712, "p25": 63.2825, "median": 90.468, "p75": 145.511, "p90": 178.5368, "share": 0.333},
    ("Prepare starter", "Bring food"): {"n": 3, "p10": 473564.8114, "p25": 548790.244, "median": 674165.965, "p75": 725064.525, "p90": 755603.661, "share": 0.0},
}

VARIANTS = [
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
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare starter", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare starter", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Bring food", "Deliver to customer", "EVENT 8 END"],
]

BLOCK_SPAN_2_P25 = 30.697
GLOBAL_MEDIAN = 115.3875

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def detect_precision(ts_str):
    if "." in ts_str:
        frac = ts_str.split(".")[-1]
        return len(frac)
    return 0

def parse_ts(ts_str):
    return pd.to_datetime(ts_str)

def fmt_ts(ts, precision):
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    if precision <= 6:
        return base[:- (6 - precision)] if precision < 6 else base
    return base

def td_seconds(a, b):
    return (b - a).total_seconds()

def profile_for(src, tgt):
    return DURATION_PROFILES.get((src, tgt))

def choose_spacing_seconds(src, tgt):
    prof = profile_for(src, tgt)
    if prof:
        if prof["share"] >= 0.5:
            return prof["p10"]
        if prof["share"] <= 0.02:
            return prof["median"]
        return prof["p25"]
    return BLOCK_SPAN_2_P25

def edge_exists(src, tgt):
    return (src, tgt) in DURATION_PROFILES or (src == "Deliver to customer" and tgt == "EVENT 8 END")

def variant_support(local_acts):
    score = 0
    rev = list(local_acts)
    rev[1], rev[2] = rev[2], rev[1]
    for v in VARIANTS:
        for s in (local_acts, rev):
            found = False
            for i in range(len(v) - len(s) + 1):
                if v[i:i + len(s)] == s:
                    score += 1 if s is local_acts else -1
                    found = True
                    break
            if found:
                break
    return score

def local_gap_penalty(src, tgt, gap):
    prof = profile_for(src, tgt)
    if gap < 0:
        return 1e9
    if prof is None:
        target = BLOCK_SPAN_2_P25
        return abs(gap - target) / max(target, 1.0) + 1.5
    pref = choose_spacing_seconds(src, tgt)
    scale = max(prof["p90"] - prof["p10"], 1.0)
    penalty = abs(gap - pref) / scale
    if gap > prof["p90"] * 3 + 1:
        penalty += 4.0
    return penalty

def swap_plausibility(case_rows, i):
    n = len(case_rows)
    r0 = case_rows[i]
    r1 = case_rows[i + 1]
    t0 = r0["_parsed"]
    t1 = r1["_parsed"]
    ts0_sw = t1
    ts1_sw = t0

    left_t = case_rows[i - 1]["_parsed"] if i - 1 >= 0 else None
    right_t = case_rows[i + 2]["_parsed"] if i + 2 < n else None

    if left_t is not None and ts0_sw < left_t:
        return False, 1e18
    if ts1_sw < ts0_sw:
        return False, 1e18
    if right_t is not None and ts1_sw > right_t:
        return False, 1e18

    acts = []
    if i - 1 >= 0:
        acts.append(case_rows[i - 1]["Activity"])
    acts.extend([r0["Activity"], r1["Activity"]])
    if i + 2 < n:
        acts.append(case_rows[i + 2]["Activity"])

    dfg_bad = 0
    edges = []
    if i - 1 >= 0:
        edges.append((case_rows[i - 1]["Activity"], r0["Activity"], left_t, ts0_sw))
    edges.append((r0["Activity"], r1["Activity"], ts0_sw, ts1_sw))
    if i + 2 < n:
        edges.append((r1["Activity"], case_rows[i + 2]["Activity"], ts1_sw, right_t))

    penalty = 0.0
    for a, b, ta, tb in edges:
        if not edge_exists(a, b):
            dfg_bad += 1
            penalty += 10.0
        penalty += local_gap_penalty(a, b, td_seconds(ta, tb))

    penalty -= 0.5 * max(variant_support(acts), 0)
    if dfg_bad > 0:
        return False, penalty + 100.0
    return True, penalty

def choose_displaced_repair(case_rows, i, precision):
    n = len(case_rows)
    r0 = case_rows[i]
    r1 = case_rows[i + 1]
    obs0 = r0["_parsed"]
    obs1 = r1["_parsed"]

    left_t = case_rows[i - 1]["_parsed"] if i - 1 >= 0 else None
    right_t = case_rows[i + 2]["_parsed"] if i + 2 < n else None

    eps = pd.Timedelta(milliseconds=max(1, 10 ** (3 - min(3, precision))) if precision >= 3 else 1)

    candidates = []

    # Option A: keep first row, move second forward
    lower = max(obs0, left_t) if left_t is not None else obs0
    est = lower + pd.to_timedelta(choose_spacing_seconds(r0["Activity"], r1["Activity"]), unit="s")
    if est < lower:
        est = lower
    if right_t is not None:
        est = min(est, right_t)
    if est < lower:
        est = lower
    if right_t is not None and est > right_t:
        est = right_t
    if est < obs0:
        est = obs0
    move = abs((est - obs1).total_seconds())
    pen = 0.0
    pen += local_gap_penalty(r0["Activity"], r1["Activity"], td_seconds(obs0, est))
    if right_t is not None and i + 2 < n:
        pen += local_gap_penalty(r1["Activity"], case_rows[i + 2]["Activity"], td_seconds(est, right_t))
    if est == obs1:
        pen += 1000.0
    candidates.append(("move_second", est, move, pen))

    # Option B: keep second row, move first backward/earlier
    upper = min(obs1, right_t) if right_t is not None else obs1
    back = choose_spacing_seconds(r0["Activity"], r1["Activity"])
    est2 = upper - pd.to_timedelta(back, unit="s")
    if left_t is not None:
        est2 = max(est2, left_t)
    if est2 > upper:
        est2 = upper
    if est2 > obs1:
        est2 = obs1
    move2 = abs((est2 - obs0).total_seconds())
    pen2 = 0.0
    if left_t is not None and i - 1 >= 0:
        pen2 += local_gap_penalty(case_rows[i - 1]["Activity"], r0["Activity"], td_seconds(left_t, est2))
    pen2 += local_gap_penalty(r0["Activity"], r1["Activity"], td_seconds(est2, obs1))
    if est2 == obs0:
        pen2 += 1000.0
    candidates.append(("move_first", est2, move2, pen2))

    candidates.sort(key=lambda x: (x[3], x[2]))

    kind, est_ts, _, _ = candidates[0]
    if kind == "move_second":
        return {
            "kind": kind,
            "new0": obs0,
            "new1": est_ts,
            "changed_rows": [1],
        }
    return {
        "kind": kind,
        "new0": est_ts,
        "new1": obs1,
        "changed_rows": [0],
    }

def clamp_and_fix_pair(case_rows, i, out0, out1, changed0, changed1, precision):
    n = len(case_rows)
    left_t = case_rows[i - 1]["_parsed"] if i - 1 >= 0 else None
    right_t = case_rows[i + 2]["_parsed"] if i + 2 < n else None
    case_start = case_rows[0]["_parsed"]
    case_end = case_rows[-1]["_parsed"]

    unit = pd.Timedelta(milliseconds=1 if precision >= 3 else 1)

    lo0 = case_start
    hi1 = case_end
    if left_t is not None:
        lo0 = max(lo0, left_t)
    if right_t is not None:
        hi1 = min(hi1, right_t)

    if out0 < lo0:
        out0 = lo0
    if out1 > hi1:
        out1 = hi1
    if out0 > out1:
        mid = out0
        out1 = mid
    if left_t is not None and out0 < left_t:
        out0 = left_t
    if right_t is not None and out1 > right_t:
        out1 = right_t
    if out1 < out0:
        out1 = out0

    if changed0 and out0 == case_rows[i]["_parsed"]:
        if out0 + unit <= out1:
            out0 = out0 + unit
        elif left_t is not None and out0 - unit >= left_t:
            out0 = out0 - unit
        else:
            out0 = out0 + unit
            if out0 > out1:
                out0 = out1
    if changed1 and out1 == case_rows[i + 1]["_parsed"]:
        if out1 - unit >= out0:
            out1 = out1 - unit
        elif right_t is not None and out1 + unit <= right_t:
            out1 = out1 + unit
        else:
            out1 = out0

    if out1 < out0:
        out1 = out0
    return out0, out1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        with open(args.input, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
    except Exception:
        rows = []

    if not rows:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])
        print("exchanged=0 displaced=0 rows_equal_input=0")
        return

    for r in rows:
        r["_orig_ts"] = r["Timestamp"]
        r["_parsed"] = parse_ts(r["Timestamp"])
        r["_precision"] = detect_precision(r["Timestamp"])
        r["_out_ts"] = r["Timestamp"]

    by_case = defaultdict(list)
    for idx, r in enumerate(rows):
        r["_idx"] = idx
        by_case[r["Case"]].append(r)

    exchanged_segments = 0
    displaced_segments = 0
    fixes = []

    for case_id, case_rows in by_case.items():
        case_rows.sort(key=lambda x: x["_idx"])
        i = 0
        while i < len(case_rows) - 1:
            r0 = case_rows[i]
            r1 = case_rows[i + 1]
            if r0["error_category"] != "CLEAN" and r1["error_category"] != "CLEAN" and r0["_parsed"] > r1["_parsed"]:
                ok_swap, swap_pen = swap_plausibility(case_rows, i)
                disp = choose_displaced_repair(case_rows, i, max(r0["_precision"], r1["_precision"]))
                disp_pen = 0.0
                if i - 1 >= 0:
                    disp_pen += local_gap_penalty(case_rows[i - 1]["Activity"], r0["Activity"], td_seconds(case_rows[i - 1]["_parsed"], disp["new0"]))
                disp_pen += local_gap_penalty(r0["Activity"], r1["Activity"], td_seconds(disp["new0"], disp["new1"]))
                if i + 2 < len(case_rows):
                    disp_pen += local_gap_penalty(r1["Activity"], case_rows[i + 2]["Activity"], td_seconds(disp["new1"], case_rows[i + 2]["_parsed"]))

                if ok_swap and swap_pen <= disp_pen:
                    out0 = r1["_parsed"]
                    out1 = r0["_parsed"]
                    out0, out1 = clamp_and_fix_pair(case_rows, i, out0, out1, True, True, max(r0["_precision"], r1["_precision"]))
                    r0["_out_ts"] = r1["_orig_ts"]
                    r1["_out_ts"] = r0["_orig_ts"]
                    # self-check exact swap requirement; if clamping altered datetime validity, reclassify below
                    if parse_ts(r0["_out_ts"]) == out0 and parse_ts(r1["_out_ts"]) == out1 and r0["_out_ts"] != r0["_orig_ts"] and r1["_out_ts"] != r1["_orig_ts"]:
                        r0["_parsed"] = out0
                        r1["_parsed"] = out1
                        exchanged_segments += 1
                        i += 2
                        continue
                    # fall through to displaced if exact swapped strings violate checks
                # displaced branch
                prec = max(r0["_precision"], r1["_precision"])
                out0, out1 = disp["new0"], disp["new1"]
                changed0 = disp["kind"] == "move_first"
                changed1 = disp["kind"] == "move_second"
                out0, out1 = clamp_and_fix_pair(case_rows, i, out0, out1, changed0, changed1, prec)

                if changed0:
                    r0["_out_ts"] = fmt_ts(out0, r0["_precision"])
                    r0["_parsed"] = out0
                else:
                    r0["_out_ts"] = r0["_orig_ts"]
                    r0["_parsed"] = parse_ts(r0["_out_ts"])

                if changed1:
                    r1["_out_ts"] = fmt_ts(out1, r1["_precision"])
                    r1["_parsed"] = out1
                else:
                    r1["_out_ts"] = r1["_orig_ts"]
                    r1["_parsed"] = parse_ts(r1["_out_ts"])

                # enforce partner unchanged in displaced
                if changed0 and r1["_out_ts"] != r1["_orig_ts"]:
                    r1["_out_ts"] = r1["_orig_ts"]
                    r1["_parsed"] = parse_ts(r1["_out_ts"])
                if changed1 and r0["_out_ts"] != r0["_orig_ts"]:
                    r0["_out_ts"] = r0["_orig_ts"]
                    r0["_parsed"] = parse_ts(r0["_out_ts"])

                displaced_segments += 1
                i += 2
                continue

            if r0["error_category"] != "CLEAN" and r0["_out_ts"] == r0["_orig_ts"]:
                # single corrupted row in an inconsistency neighborhood: repair against nearest anchors
                left_t = case_rows[i - 1]["_parsed"] if i - 1 >= 0 else case_rows[0]["_parsed"]
                right_t = case_rows[i + 1]["_parsed"] if i + 1 < len(case_rows) else case_rows[-1]["_parsed"]
                prec = r0["_precision"]
                est = left_t
                if i - 1 >= 0:
                    est = left_t + pd.to_timedelta(choose_spacing_seconds(case_rows[i - 1]["Activity"], r0["Activity"]), unit="s")
                if est > right_t:
                    est = right_t
                if est < left_t:
                    est = left_t
                est0, _ = clamp_and_fix_pair(case_rows, i if i < len(case_rows) - 1 else i - 1, est, right_t, True, False, prec) if len(case_rows) > 1 else (est, est)
                if est0 == r0["_parsed"]:
                    est0 = est0 + pd.Timedelta(milliseconds=1 if prec >= 3 else 1)
                    if i + 1 < len(case_rows) and est0 > right_t:
                        est0 = right_t
                r0["_out_ts"] = fmt_ts(est0, prec)
                r0["_parsed"] = parse_ts(r0["_out_ts"])
                displaced_segments += 1
            i += 1

    # Final case-wise self-check and repair
    for case_id, case_rows in by_case.items():
        case_rows.sort(key=lambda x: x["_idx"])
        case_start = min(parse_ts(r["_out_ts"]) for r in case_rows)
        case_end = max(parse_ts(r["_out_ts"]) for r in case_rows)
        for j, r in enumerate(case_rows):
            if r["error_category"] == "CLEAN":
                r["_out_ts"] = r["_orig_ts"]
                r["_parsed"] = parse_ts(r["_out_ts"])
            else:
                cur = parse_ts(r["_out_ts"])
                if j > 0:
                    prev = parse_ts(case_rows[j - 1]["_out_ts"])
                    if cur < prev:
                        cur = prev
                if j + 1 < len(case_rows):
                    nxt = parse_ts(case_rows[j + 1]["_out_ts"])
                    if cur > nxt:
                        cur = nxt
                if cur < case_start:
                    cur = case_start
                if cur > case_end:
                    cur = case_end
                if fmt_ts(cur, r["_precision"]) == r["_orig_ts"]:
                    cur = cur + pd.Timedelta(milliseconds=1 if r["_precision"] >= 3 else 1)
                    if j + 1 < len(case_rows):
                        nxt = parse_ts(case_rows[j + 1]["_out_ts"])
                        if cur > nxt:
                            cur = nxt
                r["_out_ts"] = fmt_ts(cur, r["_precision"])
                r["_parsed"] = parse_ts(r["_out_ts"])
                if r["_out_ts"] == r["_orig_ts"]:
                    fixes.append(f"nudge_failed_row_id={r['row_id']}")

    # Write output preserving row order and non-timestamp columns byte-identically
    fieldnames = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "Case": r["Case"],
                "Activity": r["Activity"],
                "Timestamp": r["_out_ts"],
                "Resource": r["Resource"],
                "error_category": r["error_category"],
                "row_id": r["row_id"],
            })

    rows_equal_input = sum(1 for r in rows if r["error_category"] != "CLEAN" and r["_out_ts"] == r["_orig_ts"])
    print(f"exchanged={exchanged_segments} displaced={displaced_segments} rows_equal_input={rows_equal_input}")

if __name__ == "__main__":
    main()