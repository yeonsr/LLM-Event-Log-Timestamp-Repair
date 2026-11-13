import argparse
import math
import pandas as pd
from datetime import timedelta

# ============================================================
# Embedded process knowledge from CLEAN rows only
# ============================================================

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

DFG_FREQ = {k: v["n"] for k, v in DURATION_PROFILES.items()}

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

BLOCK_SPAN_K2_P25 = 30.697
GLOBAL_MEDIAN = 115.3875

# ============================================================
# Helpers
# ============================================================

def detect_fractional_digits(ts_series):
    max_digits = 0
    for s in ts_series.dropna().astype(str):
        if "." in s:
            frac = s.split(".")[-1]
            digits = 0
            for ch in frac:
                if ch.isdigit():
                    digits += 1
                else:
                    break
            max_digits = max(max_digits, digits)
    return max_digits

def parse_ts(s):
    return pd.to_datetime(s, errors="coerce")

def td_seconds(a, b):
    return (b - a).total_seconds()

def get_profile(a, b):
    return DURATION_PROFILES.get((a, b))

def choose_gap_seconds(a, b):
    prof = get_profile(a, b)
    if prof is None:
        return None
    if prof["share_under_60s"] >= 0.5:
        return prof["p10"]
    if prof["share_under_60s"] >= 0.15:
        return prof["p25"]
    return prof["median"]

def format_ts(dt, frac_digits):
    if frac_digits <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    micros = f"{dt.microsecond:06d}"
    frac = micros[:frac_digits].ljust(frac_digits, "0")
    return f"{base}.{frac}"

def precision_timedelta(frac_digits):
    if frac_digits <= 0:
        return pd.Timedelta(seconds=1)
    if frac_digits <= 6:
        return pd.Timedelta(microseconds=10 ** (6 - frac_digits))
    return pd.Timedelta(microseconds=1)

def edge_exists(a, b):
    return (a, b) in DFG_FREQ

def edge_freq(a, b):
    return DFG_FREQ.get((a, b), 0)

def subseq_supported(acts):
    for var in FREQUENT_VARIANTS:
        for i in range(len(var) - len(acts) + 1):
            if var[i:i+len(acts)] == acts:
                return True
    return False

def local_swap_supported(prev_act, a1, a2, next_act):
    acts = []
    if prev_act is not None:
        acts.append(prev_act)
    acts.extend([a1, a2])
    if next_act is not None:
        acts.append(next_act)
    return subseq_supported(acts)

def gap_plausibility(obs_gap, prof):
    if prof is None:
        return -5.0
    if obs_gap < 0:
        return -1000.0
    p10, p25, med, p75, p90 = prof["p10"], prof["p25"], prof["median"], prof["p75"], prof["p90"]
    center = p25 if prof["share_under_60s"] >= 0.15 else med
    scale = max(p90 - p10, 1.0)
    dev = abs(obs_gap - center) / scale
    score = -dev
    if p10 <= obs_gap <= p90:
        score += 1.0
    if p25 <= obs_gap <= p75:
        score += 0.7
    if obs_gap <= 2 * max(center, 1.0):
        score += 0.2
    score += math.log1p(prof["n"]) / 10.0
    return score

def score_swap_candidate(prev_row, r1, r2, next_row, t1_new, t2_new):
    # hard monotonicity with neighbors if available
    if t1_new is pd.NaT or t2_new is pd.NaT:
        return -1e18
    if t1_new > t2_new:
        return -1e18
    if prev_row is not None and prev_row["parsed_ts"] > t1_new:
        return -1e18
    if next_row is not None and t2_new > next_row["parsed_ts"]:
        return -1e18

    score = 0.0
    if prev_row is not None:
        if not edge_exists(prev_row["Activity"], r1["Activity"]):
            return -1e12
        score += gap_plausibility(td_seconds(prev_row["parsed_ts"], t1_new), get_profile(prev_row["Activity"], r1["Activity"]))
        score += math.log1p(edge_freq(prev_row["Activity"], r1["Activity"])) / 5.0
    if not edge_exists(r1["Activity"], r2["Activity"]):
        return -1e12
    score += gap_plausibility(td_seconds(t1_new, t2_new), get_profile(r1["Activity"], r2["Activity"]))
    score += math.log1p(edge_freq(r1["Activity"], r2["Activity"])) / 5.0
    if next_row is not None:
        if not edge_exists(r2["Activity"], next_row["Activity"]):
            return -1e12
        score += gap_plausibility(td_seconds(t2_new, next_row["parsed_ts"]), get_profile(r2["Activity"], next_row["Activity"]))
        score += math.log1p(edge_freq(r2["Activity"], next_row["Activity"])) / 5.0

    prev_act = prev_row["Activity"] if prev_row is not None else None
    next_act = next_row["Activity"] if next_row is not None else None
    if local_swap_supported(prev_act, r1["Activity"], r2["Activity"], next_act):
        score += 3.0
    return score

def estimate_from_left(left_ts, left_act, cur_act):
    gap = choose_gap_seconds(left_act, cur_act)
    if gap is None:
        gap = BLOCK_SPAN_K2_P25
    return left_ts + pd.Timedelta(seconds=float(gap)), gap

def estimate_from_right(cur_act, right_act, right_ts):
    gap = choose_gap_seconds(cur_act, right_act)
    if gap is None:
        gap = BLOCK_SPAN_K2_P25
    return right_ts - pd.Timedelta(seconds=float(gap)), gap

def weighted_estimate(left_info, right_info, lower, upper, observed, freq_left=0, freq_right=0):
    candidates = []
    weights = []
    if left_info is not None:
        candidates.append(left_info[0])
        weights.append(max(freq_left, 1))
    if right_info is not None:
        candidates.append(right_info[0])
        weights.append(max(freq_right, 1))
    if not candidates:
        mid = lower + (upper - lower) / 2
        return min(max(mid, lower), upper)

    # weighted average on ns axis
    nums = [c.value for c in candidates]
    total_w = sum(weights)
    est_ns = sum(w * n for w, n in zip(weights, nums)) / total_w
    est = pd.Timestamp(int(est_ns))
    if observed is not None and lower <= observed <= upper:
        obs_dist = min(abs((est - observed).total_seconds()), 1e18)
        # if estimates conflict strongly, prefer observed only inside feasible interval
        if len(candidates) == 2:
            conflict = abs((candidates[0] - candidates[1]).total_seconds())
            if conflict > 120 and abs((observed - est).total_seconds()) < conflict:
                est = observed
    if est < lower:
        est = lower
    if est > upper:
        est = upper
    return est

def nudge_inside(value, lower, upper, step, prefer_up=True):
    v = value
    if v < lower:
        v = lower
    if v > upper:
        v = upper
    if lower == upper:
        return lower
    if v <= lower:
        v = lower + step if lower + step <= upper else lower
    elif v >= upper:
        v = upper - step if upper - step >= lower else upper
    else:
        if prefer_up and v + step <= upper:
            v = v + step
        elif (not prefer_up) and v - step >= lower:
            v = v - step
    if v < lower:
        v = lower
    if v > upper:
        v = upper
    return v

# ============================================================
# Core repair logic
# ============================================================

def repair_case(case_df, frac_digits, stats):
    step = precision_timedelta(frac_digits)
    out = case_df.copy()

    # current working timestamps: parsed originals, then patched as we go
    cur_ts = list(out["parsed_ts"])

    error_idxs = [i for i, ec in enumerate(out["error_category"]) if ec != "CLEAN"]

    for i in error_idxs:
        # treat inconsistency as adjacent inverted pair: predecessor i-1 and row i
        if i <= 0:
            # impossible in this process, but handle safely as displaced estimate
            left_idx = None
            cur_idx = i
        else:
            left_idx = i - 1
            cur_idx = i

        # classify using adjacent pair only when predecessor exists and inversion visible
        pair_done = False
        if left_idx is not None:
            r1 = out.iloc[left_idx]
            r2 = out.iloc[cur_idx]
            t1_obs = cur_ts[left_idx]
            t2_obs = cur_ts[cur_idx]
            if pd.notna(t1_obs) and pd.notna(t2_obs) and t2_obs < t1_obs:
                prev_row = out.iloc[left_idx - 1] if left_idx - 1 >= 0 else None
                next_row = out.iloc[cur_idx + 1] if cur_idx + 1 < len(out) else None

                swap_score = score_swap_candidate(prev_row, r1, r2, next_row, t2_obs, t1_obs)

                # displaced hypothesis candidates:
                # A: predecessor row is displaced, current row correct
                # B: current row is displaced, predecessor correct
                displaced_scores = []

                # A
                lowerA = prev_row["parsed_ts"] if prev_row is not None else out["parsed_ts"].min()
                upperA = t2_obs
                feasibleA = pd.notna(lowerA) and pd.notna(upperA) and lowerA <= upperA
                estA = None
                scoreA = -1e18
                if feasibleA:
                    left_est = None
                    right_est = (estimate_from_right(r1["Activity"], r2["Activity"], t2_obs), choose_gap_seconds(r1["Activity"], r2["Activity"]))
                    if prev_row is not None and edge_exists(prev_row["Activity"], r1["Activity"]):
                        left_est = (estimate_from_left(prev_row["parsed_ts"], prev_row["Activity"], r1["Activity"])[0], choose_gap_seconds(prev_row["Activity"], r1["Activity"]))
                    estA = weighted_estimate(
                        left_est,
                        right_est[0] if right_est is not None else None,
                        lowerA,
                        upperA,
                        t1_obs if lowerA <= t1_obs <= upperA else None,
                        edge_freq(prev_row["Activity"], r1["Activity"]) if prev_row is not None else 0,
                        edge_freq(r1["Activity"], r2["Activity"]),
                    )
                    if prev_row is not None and not edge_exists(prev_row["Activity"], r1["Activity"]):
                        scoreA = -1e12
                    else:
                        scoreA = 0.0
                        if prev_row is not None:
                            scoreA += gap_plausibility(td_seconds(prev_row["parsed_ts"], estA), get_profile(prev_row["Activity"], r1["Activity"]))
                        scoreA += gap_plausibility(td_seconds(estA, t2_obs), get_profile(r1["Activity"], r2["Activity"]))
                        if next_row is not None:
                            if edge_exists(r2["Activity"], next_row["Activity"]):
                                scoreA += gap_plausibility(td_seconds(t2_obs, next_row["parsed_ts"]), get_profile(r2["Activity"], next_row["Activity"]))
                            else:
                                scoreA -= 5.0
                        moveA = abs((estA - t1_obs).total_seconds())
                        scoreA -= moveA / 600.0
                displaced_scores.append(("left_displaced", scoreA, estA))

                # B
                lowerB = t1_obs
                upperB = next_row["parsed_ts"] if next_row is not None else out["parsed_ts"].max()
                feasibleB = pd.notna(lowerB) and pd.notna(upperB) and lowerB <= upperB
                estB = None
                scoreB = -1e18
                if feasibleB:
                    left_est = None
                    right_est = None
                    if edge_exists(r1["Activity"], r2["Activity"]):
                        left_est = (estimate_from_left(t1_obs, r1["Activity"], r2["Activity"])[0], choose_gap_seconds(r1["Activity"], r2["Activity"]))
                    if next_row is not None and edge_exists(r2["Activity"], next_row["Activity"]):
                        right_est = (estimate_from_right(r2["Activity"], next_row["Activity"], next_row["parsed_ts"])[0], choose_gap_seconds(r2["Activity"], next_row["Activity"]))
                    estB = weighted_estimate(
                        left_est,
                        right_est,
                        lowerB,
                        upperB,
                        t2_obs if lowerB <= t2_obs <= upperB else None,
                        edge_freq(r1["Activity"], r2["Activity"]),
                        edge_freq(r2["Activity"], next_row["Activity"]) if next_row is not None else 0,
                    )
                    scoreB = 0.0
                    if edge_exists(r1["Activity"], r2["Activity"]):
                        scoreB += gap_plausibility(td_seconds(t1_obs, estB), get_profile(r1["Activity"], r2["Activity"]))
                    else:
                        scoreB -= 20.0
                    if next_row is not None:
                        if edge_exists(r2["Activity"], next_row["Activity"]):
                            scoreB += gap_plausibility(td_seconds(estB, next_row["parsed_ts"]), get_profile(r2["Activity"], next_row["Activity"]))
                        else:
                            scoreB -= 5.0
                    moveB = abs((estB - t2_obs).total_seconds())
                    scoreB -= moveB / 600.0
                displaced_scores.append(("right_displaced", scoreB, estB))

                best_disp = max(displaced_scores, key=lambda x: x[1])

                # choose exchanged only when clearly supported
                if swap_score >= best_disp[1] + 0.75:
                    cur_ts[left_idx] = t2_obs
                    cur_ts[cur_idx] = t1_obs
                    out.at[out.index[left_idx], "repaired_ts"] = t2_obs
                    out.at[out.index[cur_idx], "repaired_ts"] = t1_obs
                    out.at[out.index[left_idx], "repair_kind"] = "EXCHANGED"
                    out.at[out.index[cur_idx], "repair_kind"] = "EXCHANGED"
                    stats["exchanged_segments"] += 1
                    pair_done = True
                else:
                    if best_disp[0] == "left_displaced" and best_disp[2] is not None:
                        new_t = best_disp[2]
                        # strict monotonicity bounds
                        low = prev_row["parsed_ts"] if prev_row is not None else out["parsed_ts"].min()
                        high = t2_obs
                        new_t = min(max(new_t, low), high)
                        if new_t == t1_obs:
                            new_t = nudge_inside(new_t, low, high, step, prefer_up=False)
                        cur_ts[left_idx] = new_t
                        out.at[out.index[left_idx], "repaired_ts"] = new_t
                        out.at[out.index[left_idx], "repair_kind"] = "DISPLACED"
                        stats["displaced_segments"] += 1
                        pair_done = True
                    elif best_disp[0] == "right_displaced" and best_disp[2] is not None:
                        new_t = best_disp[2]
                        low = t1_obs
                        high = next_row["parsed_ts"] if next_row is not None else out["parsed_ts"].max()
                        new_t = min(max(new_t, low), high)
                        if new_t == t2_obs:
                            new_t = nudge_inside(new_t, low, high, step, prefer_up=True)
                        cur_ts[cur_idx] = new_t
                        out.at[out.index[cur_idx], "repaired_ts"] = new_t
                        out.at[out.index[cur_idx], "repair_kind"] = "DISPLACED"
                        stats["displaced_segments"] += 1
                        pair_done = True

        if not pair_done:
            # Single-row displaced fallback for the corrupted row itself
            row = out.iloc[cur_idx]
            prev_row = out.iloc[cur_idx - 1] if cur_idx - 1 >= 0 else None
            next_row = out.iloc[cur_idx + 1] if cur_idx + 1 < len(out) else None
            low = cur_ts[cur_idx - 1] if cur_idx - 1 >= 0 else out["parsed_ts"].min()
            high = next_row["parsed_ts"] if next_row is not None else out["parsed_ts"].max()

            left_est = None
            right_est = None
            if prev_row is not None and edge_exists(prev_row["Activity"], row["Activity"]):
                left_est = (estimate_from_left(cur_ts[cur_idx - 1], prev_row["Activity"], row["Activity"])[0], choose_gap_seconds(prev_row["Activity"], row["Activity"]))
            if next_row is not None and edge_exists(row["Activity"], next_row["Activity"]):
                right_est = (estimate_from_right(row["Activity"], next_row["Activity"], next_row["parsed_ts"])[0], choose_gap_seconds(row["Activity"], next_row["Activity"]))
            est = weighted_estimate(
                left_est,
                right_est,
                low,
                high,
                cur_ts[cur_idx] if low <= cur_ts[cur_idx] <= high else None,
                edge_freq(prev_row["Activity"], row["Activity"]) if prev_row is not None else 0,
                edge_freq(row["Activity"], next_row["Activity"]) if next_row is not None else 0,
            )
            if est == cur_ts[cur_idx]:
                est = nudge_inside(est, low, high, step, prefer_up=True)
            cur_ts[cur_idx] = est
            out.at[out.index[cur_idx], "repaired_ts"] = est
            out.at[out.index[cur_idx], "repair_kind"] = "DISPLACED"
            stats["displaced_segments"] += 1

    # self-fix within case
    for i in range(len(out)):
        if out.iloc[i]["error_category"] == "CLEAN":
            continue
        t = cur_ts[i]
        prev_t = cur_ts[i - 1] if i - 1 >= 0 else cur_ts[0]
        next_t = cur_ts[i + 1] if i + 1 < len(out) else cur_ts[-1]
        low = prev_t if i > 0 else cur_ts[0]
        high = next_t if i + 1 < len(out) else cur_ts[-1]
        if t < low:
            t = low
        if t > high:
            t = high
        if i > 0 and t < prev_t:
            t = prev_t
        if i + 1 < len(out) and t > next_t:
            t = next_t
        # ensure changed from input
        if t == out.iloc[i]["parsed_ts"]:
            t = nudge_inside(t, low, high, step, prefer_up=True)
            if t == out.iloc[i]["parsed_ts"]:
                t = nudge_inside(t, low, high, step, prefer_up=False)
        cur_ts[i] = t
        out.at[out.index[i], "repaired_ts"] = t

    out["final_ts"] = cur_ts
    return out

# ============================================================
# Main
# ============================================================

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
    required = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in required:
        if c not in df.columns:
            df[c] = ""

    df["orig_timestamp"] = df["Timestamp"].astype(str)
    df["parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    frac_digits = detect_fractional_digits(df["Timestamp"])

    df["repaired_ts"] = pd.NaT
    df["repair_kind"] = ""

    stats = {
        "exchanged_segments": 0,
        "displaced_segments": 0,
        "fixed_messages": 0,
    }

    repaired_cases = []
    for case_id, g in df.groupby("Case", sort=False):
        repaired_cases.append(repair_case(g.copy(), frac_digits, stats))
    repaired = pd.concat(repaired_cases).sort_index()

    # final timestamp strings: preserve all untouched strings byte-identically
    final_ts_strings = repaired["orig_timestamp"].copy()

    for idx, row in repaired.iterrows():
        if row["error_category"] != "CLEAN":
            ts = row["repaired_ts"]
            if pd.isna(ts):
                ts = row["parsed_ts"]
            final_ts_strings.at[idx] = format_ts(ts, frac_digits)

    repaired["Timestamp"] = final_ts_strings

    # Global self-checks and fixes
    rows_equal_input = 0
    step = precision_timedelta(frac_digits)

    for case_id, idxs in repaired.groupby("Case", sort=False).groups.items():
        idxs = list(idxs)
        for pos, idx in enumerate(idxs):
            row = repaired.loc[idx]
            if row["error_category"] == "CLEAN":
                repaired.at[idx, "Timestamp"] = row["orig_timestamp"]
                continue

            # parse output
            t = parse_ts(repaired.at[idx, "Timestamp"])
            if pd.isna(t):
                t = row["parsed_ts"]

            low = parse_ts(repaired.at[idxs[pos - 1], "Timestamp"]) if pos > 0 else t
            high = parse_ts(repaired.at[idxs[pos + 1], "Timestamp"]) if pos + 1 < len(idxs) else t

            if pos > 0 and t < low:
                t = low
            if pos + 1 < len(idxs) and t > high:
                t = high

            if repaired.at[idx, "repair_kind"] == "EXCHANGED":
                # exact swap check
                prev_idx = idxs[pos - 1] if pos > 0 else None
                if prev_idx is not None:
                    prev_row = repaired.loc[prev_idx]
                    if prev_row["repair_kind"] == "EXCHANGED":
                        a = prev_row["orig_timestamp"]
                        b = row["orig_timestamp"]
                        if repaired.at[prev_idx, "Timestamp"] != b or repaired.at[idx, "Timestamp"] != a:
                            repaired.at[prev_idx, "Timestamp"] = b
                            repaired.at[idx, "Timestamp"] = a

            # every corrupted row must differ from input
            if repaired.at[idx, "Timestamp"] == row["orig_timestamp"]:
                if pos > 0 and pos + 1 < len(idxs):
                    t = parse_ts(repaired.at[idx, "Timestamp"])
                    t = nudge_inside(t, low, high, step, prefer_up=True)
                    if format_ts(t, frac_digits) == row["orig_timestamp"]:
                        t = nudge_inside(t, low, high, step, prefer_up=False)
                    repaired.at[idx, "Timestamp"] = format_ts(t, frac_digits)
                elif pos > 0:
                    t = parse_ts(repaired.at[idx, "Timestamp"]) + step
                    if t < low:
                        t = low
                    repaired.at[idx, "Timestamp"] = format_ts(t, frac_digits)
                elif pos + 1 < len(idxs):
                    t = parse_ts(repaired.at[idx, "Timestamp"]) - step
                    if t > high:
                        t = high
                    repaired.at[idx, "Timestamp"] = format_ts(t, frac_digits)

            if repaired.at[idx, "Timestamp"] == row["orig_timestamp"]:
                rows_equal_input += 1

    # restore exact byte-identical non-timestamp columns and row order/count
    out_df = repaired[original_cols].copy()
    out_df["Timestamp"] = repaired["Timestamp"]

    # write output safely
    try:
        out_df.to_csv(args.output, index=False)
    except Exception:
        # last-resort still write something valid
        safe_df = out_df.copy()
        safe_df.to_csv(args.output, index=False)

    print(f"exchanged_segments={stats['exchanged_segments']} displaced_segments={stats['displaced_segments']} rows_still_equal_input={rows_equal_input}")

if __name__ == "__main__":
    main()