import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

# Only C2_TEMPORAL_INCONSISTENCY exists in this task.

DURATION_PROFILES = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): {"p10": 46.211, "p25": 13993.699, "median": 165899.213, "share": 0.106, "n": 16831},
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): {"p10": 10.6268, "p25": 40.38, "median": 98.527, "share": 0.341, "n": 16077},
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): {"p10": 18.1906, "p25": 55.7485, "median": 114.822, "share": 0.27, "n": 16023},
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 61.138, "p25": 1594.0265, "median": 14749.843, "share": 0.099, "n": 15568},
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): {"p10": 0.1256, "p25": 0.174, "median": 0.273, "share": 1.0, "n": 13087},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 71.4828, "p25": 1248.542, "median": 10755.049, "share": 0.094, "n": 9065},
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 10.157, "p25": 32.9045, "median": 138.258, "share": 0.332, "n": 8927},
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): {"p10": 0.046, "p25": 0.14525, "median": 0.38, "share": 1.0, "n": 7266},
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): {"p10": 0.025, "p25": 0.028, "median": 0.035, "share": 1.0, "n": 7030},
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): {"p10": 0.083, "p25": 0.094, "median": 0.119, "share": 1.0, "n": 6633},
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9152, "p25": 1.0845, "median": 1.291, "share": 1.0, "n": 6303},
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.077, "p25": 1.179, "median": 1.386, "share": 1.0, "n": 5014},
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): {"p10": 12.5592, "p25": 19.679, "median": 217.278, "share": 0.43, "n": 4913},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): {"p10": 32.8284, "p25": 35.57925, "median": 38.9175, "share": 0.974, "n": 4838},
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): {"p10": 120.3837, "p25": 242.27225, "median": 368.7625, "share": 0.051, "n": 4772},
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): {"p10": 8.7146, "p25": 11.882, "median": 22.6655, "share": 0.953, "n": 4728},
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): {"p10": 28.1661, "p25": 633.0405, "median": 6399.7575, "share": 0.155, "n": 4508},
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): {"p10": 72.8682, "p25": 571.685, "median": 2538.547, "share": 0.092, "n": 4405},
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0, "n": 4289},
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 63.848, "p25": 118.72775, "median": 184.1315, "share": 0.092, "n": 4266},
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): {"p10": 0.141, "p25": 0.15275, "median": 0.18, "share": 0.999, "n": 3432},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.6694, "p25": 34.088, "median": 37.411, "share": 0.98, "n": 3429},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.115, "p25": 1.229, "median": 1.4505, "share": 1.0, "n": 3254},
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 6703.3079, "p25": 89236.39275, "median": 189494.1955, "share": 0.049, "n": 3164},
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): {"p10": 10.182, "p25": 12.3975, "median": 21.0825, "share": 0.832, "n": 3120},
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): {"p10": 10.801, "p25": 45.634, "median": 310.438, "share": 0.282, "n": 2881},
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 24.091, "p25": 238.427, "median": 2016.402, "share": 0.146, "n": 2851},
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): {"p10": 0.899, "p25": 0.988, "median": 1.148, "share": 1.0, "n": 2515},
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 17.3691, "p25": 858.7675, "median": 5769.2875, "share": 0.154, "n": 2438},
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): {"p10": 35.3822, "p25": 65.219, "median": 107.408, "share": 0.213, "n": 2349},
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 25.928, "p25": 331.6065, "median": 2857.8005, "share": 0.158, "n": 2166},
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): {"p10": 1.4473, "p25": 1.61025, "median": 1.847, "share": 1.0, "n": 2154},
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): {"p10": 2.838, "p25": 3.08, "median": 3.412, "share": 0.999, "n": 2026},
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0, "n": 2011},
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): {"p10": 256.972, "p25": 627.48, "median": 1177.7265, "share": 0.028, "n": 1924},
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): {"p10": 31.1872, "p25": 68.05325, "median": 111.291, "share": 0.214, "n": 1798},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 1.4856, "p25": 1.649, "median": 1.895, "share": 1.0, "n": 1727},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 699.4614, "p25": 2120.23, "median": 5024.022, "share": 0.031, "n": 1719},
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0, "n": 1662},
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0, "n": 1639},
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.22, "p25": 1.3115, "median": 1.618, "share": 0.999, "n": 1619},
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): {"p10": 31.9515, "p25": 48.48525, "median": 106.2115, "share": 0.317, "n": 1616},
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): {"p10": 24.7373, "p25": 36.484, "median": 64.6275, "share": 0.471, "n": 1514},
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0, "n": 1466},
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share": 1.0, "n": 1285},
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): {"p10": 8.8974, "p25": 56.97725, "median": 141.1065, "share": 0.259, "n": 1260},
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0, "n": 1126},
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): {"p10": 3.8134, "p25": 16.2075, "median": 363.885, "share": 0.335, "n": 1067},
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 2.9611, "p25": 3.1895, "median": 3.534, "share": 0.999, "n": 1052},
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.4536, "p25": 7.027, "median": 7.763, "share": 0.999, "n": 1009},
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.837, "p25": 2.093, "median": 2.396, "share": 0.998, "n": 993},
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): {"p10": 167.5932, "p25": 312.663, "median": 469.317, "share": 0.024, "n": 990},
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): {"p10": 72.685, "p25": 111.148, "median": 156.879, "share": 0.081, "n": 913},
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): {"p10": 141.3626, "p25": 317.4815, "median": 770.573, "share": 0.019, "n": 862},
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.3086, "p25": 6.952, "median": 7.6685, "share": 1.0, "n": 862},
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0, "n": 801},
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.3039, "p25": 1.463, "median": 1.6655, "share": 0.999, "n": 728},
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.908, "p25": 1.053, "median": 1.2515, "share": 1.0, "n": 726},
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"p10": 48200.9575, "p25": 98554.79725, "median": 306248.4655, "share": 0.0, "n": 712},
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share": 1.0, "n": 709},
}

GLOBAL_MEDIAN = 47.217
BLOCK2_P25 = 1.248
BLOCK2_P10 = 0.062

def infer_precision(ts_series):
    max_digits = 0
    for s in ts_series.dropna().astype(str):
        if "." in s:
            frac = s.split(".")[-1]
            if frac.isdigit():
                max_digits = max(max_digits, len(frac))
    return max_digits

def parse_ts(s):
    return pd.to_datetime(s, errors="coerce")

def format_ts(ts, precision):
    if pd.isna(ts):
        return ""
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if precision <= 0:
        return base
    frac = f"{ts.microsecond:06d}"[:precision]
    return f"{base}.{frac}"

def epsilon_td(precision):
    if precision <= 0:
        return pd.Timedelta(seconds=1)
    if precision <= 6:
        return pd.Timedelta(microseconds=10 ** (6 - precision))
    return pd.Timedelta(microseconds=1)

def get_profile(a, b):
    return DURATION_PROFILES.get((a, b))

def choose_duration_seconds(a, b):
    prof = get_profile(a, b)
    if prof is None:
        return min(GLOBAL_MEDIAN, max(BLOCK2_P25, BLOCK2_P10))
    share = prof["share"]
    if share >= 0.2:
        return prof["p25"]
    return prof["p10"]

def transition_cost(a, b, gap_seconds):
    if gap_seconds is None or gap_seconds < 0:
        return 1e12
    prof = get_profile(a, b)
    if prof is None:
        target = min(GLOBAL_MEDIAN, 60.0)
        scale = max(target, 1.0)
        return abs(gap_seconds - target) / scale + 3.0
    target = choose_duration_seconds(a, b)
    scale = max(target, 1.0)
    freq_bonus = np.log1p(prof["n"]) / 10.0
    return abs(gap_seconds - target) / scale - freq_bonus

def anchored_bounds(case_df):
    parsed = case_df["_parsed_ts"]
    non_na = parsed.dropna()
    if len(non_na) == 0:
        return pd.NaT, pd.NaT
    return non_na.min(), non_na.max()

def estimate_from_neighbors(case_df, pos, precision):
    eps = epsilon_td(precision)
    act = case_df.iloc[pos]["Activity"]
    obs = case_df.iloc[pos]["_parsed_ts"]

    case_start, case_end = anchored_bounds(case_df)

    left_idx = pos - 1 if pos - 1 >= 0 else None
    right_idx = pos + 1 if pos + 1 < len(case_df) else None

    left_t = case_df.iloc[left_idx]["_repaired_ts"] if left_idx is not None else pd.NaT
    right_t = case_df.iloc[right_idx]["_repaired_ts"] if right_idx is not None else pd.NaT
    left_a = case_df.iloc[left_idx]["Activity"] if left_idx is not None else None
    right_a = case_df.iloc[right_idx]["Activity"] if right_idx is not None else None

    lower = case_start
    upper = case_end
    if left_idx is not None and not pd.isna(left_t):
        lower = max(lower, left_t) if not pd.isna(lower) else left_t
    if right_idx is not None and not pd.isna(right_t):
        upper = min(upper, right_t) if not pd.isna(upper) else right_t

    candidates = []

    if left_idx is not None and not pd.isna(left_t):
        d = choose_duration_seconds(left_a, act)
        candidates.append(left_t + pd.Timedelta(seconds=d))
    if right_idx is not None and not pd.isna(right_t):
        d = choose_duration_seconds(act, right_a)
        candidates.append(right_t - pd.Timedelta(seconds=d))

    if candidates:
        cand = min(candidates, key=lambda t: abs((t - obs).total_seconds()) if not pd.isna(obs) else 0)
        if len(candidates) == 2:
            avg_ns = int((candidates[0].value + candidates[1].value) / 2)
            avg_cand = pd.Timestamp(avg_ns)
            if not pd.isna(obs):
                cands2 = [cand, avg_cand]
                cand = min(cands2, key=lambda t: abs((t - obs).total_seconds()))
            else:
                cand = avg_cand
    else:
        cand = obs if not pd.isna(obs) else case_start

    if pd.isna(cand):
        cand = case_start if not pd.isna(case_start) else pd.Timestamp("2000-01-01")

    if not pd.isna(lower) and cand < lower:
        cand = lower
    if not pd.isna(upper) and cand > upper:
        cand = upper

    if left_idx is not None and not pd.isna(left_t) and cand < left_t:
        cand = left_t
    if right_idx is not None and not pd.isna(right_t) and cand > right_t:
        cand = right_t

    orig_str = case_df.iloc[pos]["Timestamp"]
    if format_ts(cand, precision) == orig_str:
        if right_idx is not None and not pd.isna(upper) and cand + eps <= upper:
            cand = cand + eps
        elif left_idx is not None and not pd.isna(lower) and cand - eps >= lower:
            cand = cand - eps
        elif not pd.isna(upper) and cand + eps <= upper:
            cand = cand + eps
        elif not pd.isna(lower) and cand - eps >= lower:
            cand = cand - eps
        else:
            if right_idx is not None and not pd.isna(right_t):
                cand = min(right_t, cand + eps)
            elif left_idx is not None and not pd.isna(left_t):
                cand = max(left_t, cand - eps)

    if left_idx is not None and not pd.isna(left_t) and cand < left_t:
        cand = left_t
    if right_idx is not None and not pd.isna(right_t) and cand > right_t:
        cand = right_t

    if format_ts(cand, precision) == orig_str:
        if right_idx is not None and not pd.isna(right_t) and cand < right_t:
            nudged = min(right_t, cand + eps)
            if format_ts(nudged, precision) != orig_str:
                cand = nudged
        if format_ts(cand, precision) == orig_str and left_idx is not None and not pd.isna(left_t) and cand > left_t:
            nudged = max(left_t, cand - eps)
            if format_ts(nudged, precision) != orig_str:
                cand = nudged

    return cand

def score_swap(case_df, i, j):
    t_i = case_df.iloc[i]["_parsed_ts"]
    t_j = case_df.iloc[j]["_parsed_ts"]
    if pd.isna(t_i) or pd.isna(t_j):
        return None

    left_idx = i - 1 if i - 1 >= 0 else None
    right_idx = j + 1 if j + 1 < len(case_df) else None

    left_t = case_df.iloc[left_idx]["_repaired_ts"] if left_idx is not None else pd.NaT
    right_t = case_df.iloc[right_idx]["_repaired_ts"] if right_idx is not None else pd.NaT

    swapped_i = t_j
    swapped_j = t_i

    if swapped_i > swapped_j:
        return None

    case_start, case_end = anchored_bounds(case_df)
    for tt in [swapped_i, swapped_j]:
        if not pd.isna(case_start) and tt < case_start:
            return None
        if not pd.isna(case_end) and tt > case_end:
            return None

    if left_idx is not None and not pd.isna(left_t) and swapped_i < left_t:
        return None
    if right_idx is not None and not pd.isna(right_t) and swapped_j > right_t:
        return None

    cost = 0.0
    if left_idx is not None and not pd.isna(left_t):
        cost += transition_cost(case_df.iloc[left_idx]["Activity"], case_df.iloc[i]["Activity"], (swapped_i - left_t).total_seconds())
    cost += transition_cost(case_df.iloc[i]["Activity"], case_df.iloc[j]["Activity"], (swapped_j - swapped_i).total_seconds())
    if right_idx is not None and not pd.isna(right_t):
        cost += transition_cost(case_df.iloc[j]["Activity"], case_df.iloc[right_idx]["Activity"], (right_t - swapped_j).total_seconds())

    return {"i_ts": swapped_i, "j_ts": swapped_j, "cost": cost}

def displaced_candidates(case_df, i, j, precision):
    results = []

    cand_i = estimate_from_neighbors(case_df, i, precision)
    fixed_j = case_df.iloc[j]["_parsed_ts"]
    if not pd.isna(cand_i) and not pd.isna(fixed_j) and cand_i <= fixed_j:
        cost = 0.0
        left_idx = i - 1 if i - 1 >= 0 else None
        right_idx = j + 1 if j + 1 < len(case_df) else None
        if left_idx is not None:
            left_t = case_df.iloc[left_idx]["_repaired_ts"]
            if not pd.isna(left_t):
                cost += transition_cost(case_df.iloc[left_idx]["Activity"], case_df.iloc[i]["Activity"], (cand_i - left_t).total_seconds())
        cost += transition_cost(case_df.iloc[i]["Activity"], case_df.iloc[j]["Activity"], (fixed_j - cand_i).total_seconds())
        if right_idx is not None:
            right_t = case_df.iloc[right_idx]["_repaired_ts"]
            if not pd.isna(right_t):
                cost += transition_cost(case_df.iloc[j]["Activity"], case_df.iloc[right_idx]["Activity"], (right_t - fixed_j).total_seconds())
        obs = case_df.iloc[i]["_parsed_ts"]
        if not pd.isna(obs):
            cost += abs((cand_i - obs).total_seconds()) / max(choose_duration_seconds(case_df.iloc[i]["Activity"], case_df.iloc[j]["Activity"]), 1.0)
        results.append(("i_displaced", cand_i, fixed_j, cost))

    cand_j = estimate_from_neighbors(case_df, j, precision)
    fixed_i = case_df.iloc[i]["_parsed_ts"]
    if not pd.isna(cand_j) and not pd.isna(fixed_i) and fixed_i <= cand_j:
        cost = 0.0
        left_idx = i - 1 if i - 1 >= 0 else None
        right_idx = j + 1 if j + 1 < len(case_df) else None
        if left_idx is not None:
            left_t = case_df.iloc[left_idx]["_repaired_ts"]
            if not pd.isna(left_t):
                cost += transition_cost(case_df.iloc[left_idx]["Activity"], case_df.iloc[i]["Activity"], (fixed_i - left_t).total_seconds())
        cost += transition_cost(case_df.iloc[i]["Activity"], case_df.iloc[j]["Activity"], (cand_j - fixed_i).total_seconds())
        if right_idx is not None:
            right_t = case_df.iloc[right_idx]["_repaired_ts"]
            if not pd.isna(right_t):
                cost += transition_cost(case_df.iloc[j]["Activity"], case_df.iloc[right_idx]["Activity"], (right_t - cand_j).total_seconds())
        obs = case_df.iloc[j]["_parsed_ts"]
        if not pd.isna(obs):
            cost += abs((cand_j - obs).total_seconds()) / max(choose_duration_seconds(case_df.iloc[i]["Activity"], case_df.iloc[j]["Activity"]), 1.0)
        results.append(("j_displaced", fixed_i, cand_j, cost))

    return results

def repair_case(case_df, precision):
    case_df = case_df.copy()
    case_df["_repaired_str"] = case_df["Timestamp"].astype(str)
    case_df["_repaired_ts"] = case_df["_parsed_ts"]
    case_df["_classification"] = ""
    case_df["_changed"] = False

    exchanged_segments = 0
    displaced_segments = 0

    idxs = list(range(len(case_df)))
    bad = case_df["error_category"].ne("CLEAN")

    for i in idxs:
        if not bad.iloc[i]:
            continue

        paired = False
        if i + 1 < len(case_df):
            t_i = case_df.iloc[i]["_repaired_ts"]
            t_j = case_df.iloc[i + 1]["_repaired_ts"]
            if not pd.isna(t_i) and not pd.isna(t_j) and t_i > t_j:
                paired = True
                j = i + 1

                swap = score_swap(case_df, i, j)
                disp = displaced_candidates(case_df, i, j, precision)

                choose_swap = False
                if swap is not None:
                    if disp:
                        best_disp = min(disp, key=lambda x: x[3])
                        if swap["cost"] <= best_disp[3]:
                            choose_swap = True
                    else:
                        choose_swap = True

                if choose_swap:
                    out_i = format_ts(swap["i_ts"], precision)
                    out_j = format_ts(swap["j_ts"], precision)
                    if out_i != case_df.iloc[i]["Timestamp"] and out_j != case_df.iloc[j]["Timestamp"]:
                        case_df.iloc[i, case_df.columns.get_loc("_repaired_str")] = case_df.iloc[j]["Timestamp"]
                        case_df.iloc[j, case_df.columns.get_loc("_repaired_str")] = case_df.iloc[i]["Timestamp"]
                        case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = swap["i_ts"]
                        case_df.iloc[j, case_df.columns.get_loc("_repaired_ts")] = swap["j_ts"]
                        case_df.iloc[i, case_df.columns.get_loc("_classification")] = "EXCHANGED"
                        case_df.iloc[j, case_df.columns.get_loc("_classification")] = "EXCHANGED"
                        case_df.iloc[i, case_df.columns.get_loc("_changed")] = True
                        case_df.iloc[j, case_df.columns.get_loc("_changed")] = True
                        exchanged_segments += 1
                        continue

                if disp:
                    best = min(disp, key=lambda x: x[3])
                    label, ts_i, ts_j, _ = best
                    if label == "i_displaced":
                        case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = ts_i
                        case_df.iloc[i, case_df.columns.get_loc("_repaired_str")] = format_ts(ts_i, precision)
                        case_df.iloc[i, case_df.columns.get_loc("_classification")] = "DISPLACED"
                        case_df.iloc[i, case_df.columns.get_loc("_changed")] = case_df.iloc[i]["Timestamp"] != case_df.iloc[i]["_repaired_str"]
                        displaced_segments += 1
                    else:
                        case_df.iloc[j, case_df.columns.get_loc("_repaired_ts")] = ts_j
                        case_df.iloc[j, case_df.columns.get_loc("_repaired_str")] = format_ts(ts_j, precision)
                        case_df.iloc[j, case_df.columns.get_loc("_classification")] = "DISPLACED"
                        case_df.iloc[j, case_df.columns.get_loc("_changed")] = case_df.iloc[j]["Timestamp"] != case_df.iloc[j]["_repaired_str"]
                        displaced_segments += 1
                    continue

        if not paired:
            new_ts = estimate_from_neighbors(case_df, i, precision)
            case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = new_ts
            case_df.iloc[i, case_df.columns.get_loc("_repaired_str")] = format_ts(new_ts, precision)
            case_df.iloc[i, case_df.columns.get_loc("_classification")] = "DISPLACED"
            case_df.iloc[i, case_df.columns.get_loc("_changed")] = case_df.iloc[i]["Timestamp"] != case_df.iloc[i]["_repaired_str"]
            displaced_segments += 1

    return case_df, exchanged_segments, displaced_segments

def self_check_and_fix(case_df, precision):
    eps = epsilon_td(precision)
    fixes = 0

    case_start, case_end = anchored_bounds(case_df)

    for i in range(len(case_df)):
        err = case_df.iloc[i]["error_category"]
        if err == "CLEAN":
            if case_df.iloc[i]["_repaired_str"] != case_df.iloc[i]["Timestamp"]:
                case_df.iloc[i, case_df.columns.get_loc("_repaired_str")] = case_df.iloc[i]["Timestamp"]
                case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = case_df.iloc[i]["_parsed_ts"]
                fixes += 1

    for i in range(len(case_df)):
        if case_df.iloc[i]["error_category"] == "CLEAN":
            continue

        t = parse_ts(case_df.iloc[i]["_repaired_str"])
        if pd.isna(t):
            t = estimate_from_neighbors(case_df, i, precision)
            case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = t
            case_df.iloc[i, case_df.columns.get_loc("_repaired_str")] = format_ts(t, precision)
            fixes += 1
        else:
            case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = t

        if not pd.isna(case_start) and t < case_start:
            t = case_start
            case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = t
            case_df.iloc[i, case_df.columns.get_loc("_repaired_str")] = format_ts(t, precision)
            fixes += 1
        if not pd.isna(case_end) and t > case_end:
            t = case_end
            case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = t
            case_df.iloc[i, case_df.columns.get_loc("_repaired_str")] = format_ts(t, precision)
            fixes += 1

    for i in range(1, len(case_df)):
        prev_t = case_df.iloc[i - 1]["_repaired_ts"]
        cur_t = case_df.iloc[i]["_repaired_ts"]
        if pd.isna(prev_t) or pd.isna(cur_t):
            continue
        if cur_t < prev_t:
            if case_df.iloc[i]["error_category"] != "CLEAN":
                new_t = prev_t
                right_bound = case_end
                if i + 1 < len(case_df) and not pd.isna(case_df.iloc[i + 1]["_repaired_ts"]):
                    right_bound = min(right_bound, case_df.iloc[i + 1]["_repaired_ts"]) if not pd.isna(right_bound) else case_df.iloc[i + 1]["_repaired_ts"]
                if not pd.isna(right_bound) and new_t > right_bound:
                    new_t = right_bound
                if format_ts(new_t, precision) == case_df.iloc[i]["Timestamp"] and not pd.isna(right_bound) and new_t + eps <= right_bound:
                    new_t = new_t + eps
                case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = new_t
                case_df.iloc[i, case_df.columns.get_loc("_repaired_str")] = format_ts(new_t, precision)
                fixes += 1

    for i in range(len(case_df)):
        if case_df.iloc[i]["error_category"] != "CLEAN":
            if case_df.iloc[i]["_repaired_str"] == case_df.iloc[i]["Timestamp"]:
                new_t = estimate_from_neighbors(case_df, i, precision)
                if i > 0 and not pd.isna(case_df.iloc[i - 1]["_repaired_ts"]) and new_t < case_df.iloc[i - 1]["_repaired_ts"]:
                    new_t = case_df.iloc[i - 1]["_repaired_ts"]
                if i + 1 < len(case_df) and not pd.isna(case_df.iloc[i + 1]["_repaired_ts"]) and new_t > case_df.iloc[i + 1]["_repaired_ts"]:
                    new_t = case_df.iloc[i + 1]["_repaired_ts"]
                if format_ts(new_t, precision) == case_df.iloc[i]["Timestamp"]:
                    if i + 1 < len(case_df) and not pd.isna(case_df.iloc[i + 1]["_repaired_ts"]) and new_t + eps <= case_df.iloc[i + 1]["_repaired_ts"]:
                        new_t = new_t + eps
                    elif i > 0 and not pd.isna(case_df.iloc[i - 1]["_repaired_ts"]) and new_t - eps >= case_df.iloc[i - 1]["_repaired_ts"]:
                        new_t = new_t - eps
                case_df.iloc[i, case_df.columns.get_loc("_repaired_ts")] = new_t
                case_df.iloc[i, case_df.columns.get_loc("_repaired_str")] = format_ts(new_t, precision)
                fixes += 1

    return case_df, fixes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        original_columns = list(df.columns)

        precision = infer_precision(df["Timestamp"])
        df["_parsed_ts"] = df["Timestamp"].apply(parse_ts)
        df["_out_ts"] = df["Timestamp"].astype(str)

        exchanged_total = 0
        displaced_total = 0
        fixed_total = 0

        out_parts = []
        for case_id, case_df in df.groupby("Case", sort=False, group_keys=False):
            repaired_case, ex_cnt, dis_cnt = repair_case(case_df, precision)
            repaired_case, fix_cnt = self_check_and_fix(repaired_case, precision)

            exchanged_total += ex_cnt
            displaced_total += dis_cnt
            fixed_total += fix_cnt

            repaired_case["_out_ts"] = repaired_case["_repaired_str"]
            out_parts.append(repaired_case)

        out_df = pd.concat(out_parts, axis=0).sort_index()

        rows_equal_input = int(((out_df["error_category"] != "CLEAN") & (out_df["_out_ts"] == out_df["Timestamp"])).sum())

        result = out_df[original_columns].copy()
        result["Timestamp"] = out_df["_out_ts"]

        result.to_csv(args.output, index=False)

        print(f"exchanged_segments={exchanged_total} displaced_segments={displaced_total} rows_still_equal_to_input={rows_equal_input} fixes_applied={fixed_total}")
    except Exception as e:
        try:
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            print(f"exchanged_segments=0 displaced_segments=0 rows_still_equal_to_input={(df['error_category'] != 'CLEAN').sum()} fixes_applied=0 error_recovered={str(e).replace(',', ';')}")
        except Exception:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("Case,Activity,Timestamp,Resource,error_category,row_id\n")
            print(f"exchanged_segments=0 displaced_segments=0 rows_still_equal_to_input=0 fixes_applied=0 fatal_recovered={str(e).replace(',', ';')}")

if __name__ == "__main__":
    main()