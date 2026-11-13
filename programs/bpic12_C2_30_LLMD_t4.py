import argparse
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_FLOOR
from collections import defaultdict

# Only error present in this task:
ERR = "C2_TEMPORAL_INCONSISTENCY"

# ---------------------------------------------------------------------
# Process knowledge embedded from prompt
# ---------------------------------------------------------------------
DURATION_PROFILES = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): {"n": 16831, "p10": 46.211, "p25": 13993.699, "median": 165899.213, "share": 0.106},
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): {"n": 16077, "p10": 10.6268, "p25": 40.38, "median": 98.527, "share": 0.341},
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): {"n": 16023, "p10": 18.1906, "p25": 55.7485, "median": 114.822, "share": 0.27},
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): {"n": 15568, "p10": 61.138, "p25": 1594.0265, "median": 14749.843, "share": 0.099},
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): {"n": 13087, "p10": 0.1256, "p25": 0.174, "median": 0.273, "share": 1.0},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"n": 9065, "p10": 71.4828, "p25": 1248.542, "median": 10755.049, "share": 0.094},
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): {"n": 8927, "p10": 10.157, "p25": 32.9045, "median": 138.258, "share": 0.332},
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): {"n": 7266, "p10": 0.046, "p25": 0.14525, "median": 0.38, "share": 1.0},
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): {"n": 7030, "p10": 0.025, "p25": 0.028, "median": 0.035, "share": 1.0},
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): {"n": 6633, "p10": 0.083, "p25": 0.094, "median": 0.119, "share": 1.0},
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): {"n": 6303, "p10": 0.9152, "p25": 1.0845, "median": 1.291, "share": 1.0},
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): {"n": 5014, "p10": 1.077, "p25": 1.179, "median": 1.386, "share": 1.0},
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): {"n": 4913, "p10": 12.5592, "p25": 19.679, "median": 217.278, "share": 0.43},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): {"n": 4838, "p10": 32.8284, "p25": 35.57925, "median": 38.9175, "share": 0.974},
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): {"n": 4772, "p10": 120.3837, "p25": 242.27225, "median": 368.7625, "share": 0.051},
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): {"n": 4728, "p10": 8.7146, "p25": 11.882, "median": 22.6655, "share": 0.953},
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): {"n": 4508, "p10": 28.1661, "p25": 633.0405, "median": 6399.7575, "share": 0.155},
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): {"n": 4405, "p10": 72.8682, "p25": 571.685, "median": 2538.547, "share": 0.092},
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): {"n": 4289, "p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"n": 4266, "p10": 63.848, "p25": 118.72775, "median": 184.1315, "share": 0.092},
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): {"n": 3432, "p10": 0.141, "p25": 0.15275, "median": 0.18, "share": 0.999},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"n": 3429, "p10": 31.6694, "p25": 34.088, "median": 37.411, "share": 0.98},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"n": 3254, "p10": 1.115, "p25": 1.229, "median": 1.4505, "share": 1.0},
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): {"n": 3164, "p10": 6703.3079, "p25": 89236.39275, "median": 189494.1955, "share": 0.049},
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): {"n": 3120, "p10": 10.182, "p25": 12.3975, "median": 21.0825, "share": 0.832},
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): {"n": 2881, "p10": 10.801, "p25": 45.634, "median": 310.438, "share": 0.282},
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): {"n": 2851, "p10": 24.091, "p25": 238.427, "median": 2016.402, "share": 0.146},
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): {"n": 2515, "p10": 0.899, "p25": 0.988, "median": 1.148, "share": 1.0},
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): {"n": 2438, "p10": 17.3691, "p25": 858.7675, "median": 5769.2875, "share": 0.154},
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): {"n": 2349, "p10": 35.3822, "p25": 65.219, "median": 107.408, "share": 0.213},
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"n": 2166, "p10": 25.928, "p25": 331.6065, "median": 2857.8005, "share": 0.158},
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): {"n": 2154, "p10": 1.4473, "p25": 1.61025, "median": 1.847, "share": 1.0},
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): {"n": 2026, "p10": 2.838, "p25": 3.08, "median": 3.412, "share": 0.999},
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): {"n": 2011, "p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): {"n": 1924, "p10": 256.972, "p25": 627.48, "median": 1177.7265, "share": 0.028},
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): {"n": 1798, "p10": 31.1872, "p25": 68.05325, "median": 111.291, "share": 0.214},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): {"n": 1727, "p10": 1.4856, "p25": 1.649, "median": 1.895, "share": 1.0},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): {"n": 1719, "p10": 699.4614, "p25": 2120.23, "median": 5024.022, "share": 0.031},
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): {"n": 1662, "p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): {"n": 1639, "p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"n": 1619, "p10": 1.22, "p25": 1.3115, "median": 1.618, "share": 0.999},
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): {"n": 1616, "p10": 31.9515, "p25": 48.48525, "median": 106.2115, "share": 0.317},
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): {"n": 1514, "p10": 24.7373, "p25": 36.484, "median": 64.6275, "share": 0.471},
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): {"n": 1466, "p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): {"n": 1285, "p10": 0.001, "p25": 0.001, "median": 0.001, "share": 1.0},
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): {"n": 1260, "p10": 8.8974, "p25": 56.97725, "median": 141.1065, "share": 0.259},
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): {"n": 1126, "p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): {"n": 1067, "p10": 3.8134, "p25": 16.2075, "median": 363.885, "share": 0.335},
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"n": 1052, "p10": 2.9611, "p25": 3.1895, "median": 3.534, "share": 0.999},
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"n": 1009, "p10": 6.4536, "p25": 7.027, "median": 7.763, "share": 0.999},
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"n": 993, "p10": 1.837, "p25": 2.093, "median": 2.396, "share": 0.998},
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): {"n": 990, "p10": 167.5932, "p25": 312.663, "median": 469.317, "share": 0.024},
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): {"n": 913, "p10": 72.685, "p25": 111.148, "median": 156.879, "share": 0.081},
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): {"n": 862, "p10": 141.3626, "p25": 317.4815, "median": 770.573, "share": 0.019},
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"n": 862, "p10": 6.3086, "p25": 6.952, "median": 7.6685, "share": 1.0},
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): {"n": 801, "p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"n": 728, "p10": 1.3039, "p25": 1.463, "median": 1.6655, "share": 0.999},
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): {"n": 726, "p10": 0.908, "p25": 1.053, "median": 1.2515, "share": 1.0},
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"n": 712, "p10": 48200.9575, "p25": 98554.79725, "median": 306248.4655, "share": 0.0},
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"n": 709, "p10": 0.001, "p25": 0.001, "median": 0.001, "share": 1.0},
}

DFG_FREQ = {k: v["n"] for k, v in DURATION_PROFILES.items()}

FREQUENT_VARIANTS = [
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_DECLINED-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE"]  # weak helper, contiguous only if present
]
GLOBAL_MEDIAN = 47.217


def choose_stat(profile):
    if profile is None:
        return GLOBAL_MEDIAN
    if profile["share"] >= 0.8:
        return profile["p10"]
    if profile["share"] >= 0.25:
        return profile["p25"]
    return profile["median"]


def parse_ts(s):
    return pd.to_datetime(s)


def frac_digits(ts_str):
    if "." in ts_str:
        return len(ts_str.split(".")[-1])
    return 0


def fmt_ts(ts, ndigits):
    if ndigits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    ns = ts.value % 1_000_000_000
    frac = f"{ns:09d}"[:ndigits]
    return f"{base}.{frac}"


def quantize_ts(ts, ndigits):
    if ndigits >= 9:
        return ts
    unit_ns = 10 ** (9 - ndigits) if ndigits > 0 else 1_000_000_000
    val = ts.value
    q = (val // unit_ns) * unit_ns
    return pd.Timestamp(q)


def add_seconds(ts, sec):
    return ts + pd.to_timedelta(float(sec), unit="s")


def edge_exists(a, b):
    return (a, b) in DFG_FREQ


def edge_score(a, b):
    if (a, b) not in DFG_FREQ:
        return -1_000_000
    return DFG_FREQ[(a, b)]


def variant_support(local_acts):
    best = 0
    for v in FREQUENT_VARIANTS:
        m = len(local_acts)
        for i in range(len(v) - m + 1):
            if v[i:i+m] == local_acts:
                best = max(best, 1)
    return best


def local_order_score(acts4):
    score = 0
    for a, b in zip(acts4[:-1], acts4[1:]):
        score += edge_score(a, b)
    score += 100000 * variant_support(acts4)
    return score


def duration_penalty(a, b, gap_sec):
    prof = DURATION_PROFILES.get((a, b))
    if prof is None:
        return 50.0
    target = choose_stat(prof)
    scale = max(target, 1.0)
    return abs(gap_sec - target) / scale


def within(ts, lo, hi):
    if lo is not None and ts < lo:
        return False
    if hi is not None and ts > hi:
        return False
    return True


def clamp(ts, lo, hi):
    out = ts
    if lo is not None and out < lo:
        out = lo
    if hi is not None and out > hi:
        out = hi
    return out


def smallest_step(ndigits):
    return pd.to_timedelta(10 ** (-ndigits if ndigits > 0 else 0), unit="s") if ndigits > 0 else pd.to_timedelta(1, unit="s")


def make_non_decreasing(ts, prev_ts, next_ts, ndigits):
    out = ts
    if prev_ts is not None and out < prev_ts:
        out = prev_ts
    if next_ts is not None and out > next_ts:
        out = next_ts
    return quantize_ts(out, ndigits)


def estimate_single(case_rows, idx_move, idx_fixed, parsed_out, original_parsed):
    n = len(case_rows)
    moved_act = case_rows[idx_move]["Activity"]
    fixed_ts = parsed_out[idx_fixed]
    fixed_act = case_rows[idx_fixed]["Activity"]
    nd = case_rows[idx_move]["precision"]

    prev_idx = idx_move - 1 if idx_move - 1 >= 0 else None
    next_idx = idx_move + 1 if idx_move + 1 < n else None

    left_anchor = parsed_out[prev_idx] if prev_idx is not None else None
    right_anchor = parsed_out[next_idx] if next_idx is not None else None

    candidates = []
    weights = []

    if idx_move < idx_fixed:
        prof = DURATION_PROFILES.get((moved_act, fixed_act))
        if prof is not None:
            candidates.append(add_seconds(fixed_ts, -choose_stat(prof)))
            weights.append(max(prof["n"], 1))
        if prev_idx is not None:
            prof2 = DURATION_PROFILES.get((case_rows[prev_idx]["Activity"], moved_act))
            if prof2 is not None:
                candidates.append(add_seconds(left_anchor, choose_stat(prof2)))
                weights.append(max(prof2["n"], 1))
    else:
        prof = DURATION_PROFILES.get((fixed_act, moved_act))
        if prof is not None:
            candidates.append(add_seconds(fixed_ts, choose_stat(prof)))
            weights.append(max(prof["n"], 1))
        if next_idx is not None:
            prof2 = DURATION_PROFILES.get((moved_act, case_rows[next_idx]["Activity"]))
            if prof2 is not None:
                candidates.append(add_seconds(right_anchor, -choose_stat(prof2)))
                weights.append(max(prof2["n"], 1))

    observed = original_parsed[idx_move]
    candidates.append(observed)
    weights.append(1)

    if candidates:
        vals = np.array([c.value for c in candidates], dtype=np.float64)
        w = np.array(weights, dtype=np.float64)
        target_val = np.average(vals, weights=w)
        est = pd.Timestamp(int(target_val))
    else:
        est = fixed_ts

    lo = left_anchor
    hi = right_anchor

    if idx_move < idx_fixed:
        hi = fixed_ts
    else:
        lo = fixed_ts

    est = clamp(est, lo, hi)
    est = quantize_ts(est, nd)

    if est == original_parsed[idx_move]:
        step = smallest_step(nd)
        if idx_move < idx_fixed:
            est = clamp(est - step, lo, hi)
        else:
            est = clamp(est + step, lo, hi)
        est = quantize_ts(est, nd)

    est = make_non_decreasing(est, left_anchor, right_anchor, nd)
    return est


def candidate_score(case_rows, parsed_base, replacements):
    parsed = list(parsed_base)
    for k, v in replacements.items():
        parsed[k] = v
    n = len(case_rows)
    score = 0.0
    for i in range(n - 1):
        a = case_rows[i]["Activity"]
        b = case_rows[i + 1]["Activity"]
        gap = (parsed[i + 1] - parsed[i]).total_seconds()
        if gap < 0:
            score -= 1e9
        score += edge_score(a, b)
        if gap >= 0:
            score -= duration_penalty(a, b, gap)
    # reward smaller movement
    for i, ts in replacements.items():
        score -= abs((ts - parsed_base[i]).total_seconds()) * 0.01
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_cols = list(df.columns)
    df["__orig_ts"] = df["Timestamp"]
    df["__out_ts"] = df["Timestamp"]
    df["__parsed_orig"] = df["Timestamp"].map(parse_ts)
    df["__precision"] = df["Timestamp"].map(frac_digits)

    exchanged_segments = 0
    displaced_segments = 0

    grouped = df.groupby("Case", sort=False).indices

    for case, idxs in grouped.items():
        idxs = list(idxs)
        case_rows = []
        for gidx in idxs:
            row = df.loc[gidx]
            case_rows.append({
                "gidx": gidx,
                "Activity": row["Activity"],
                "error_category": row["error_category"],
                "precision": int(row["__precision"]),
            })

        parsed_out = [df.loc[r["gidx"], "__parsed_orig"] for r in case_rows]
        original_parsed = list(parsed_out)

        i = 0
        while i < len(case_rows) - 1:
            t1 = parsed_out[i]
            t2 = parsed_out[i + 1]
            if t2 < t1 and (
                case_rows[i]["error_category"] != "CLEAN" or case_rows[i + 1]["error_category"] != "CLEAN"
            ):
                acts = [r["Activity"] for r in case_rows]

                swap_ok = True
                swap_repl = {i: original_parsed[i + 1], i + 1: original_parsed[i]}
                # exact exchange requires both rows change
                if original_parsed[i] == original_parsed[i + 1]:
                    swap_ok = False

                # local order/anchor feasibility
                loc_indices = [j for j in [i - 1, i, i + 1, i + 2] if 0 <= j < len(case_rows)]
                local_acts = [acts[j] for j in loc_indices]
                if len(local_acts) >= 2 and local_order_score(local_acts) < -500000:
                    swap_ok = False

                tmp = list(parsed_out)
                tmp[i] = original_parsed[i + 1]
                tmp[i + 1] = original_parsed[i]

                if tmp[i] > tmp[i + 1]:
                    swap_ok = False
                if i - 1 >= 0 and tmp[i] < tmp[i - 1]:
                    swap_ok = False
                if i + 2 < len(tmp) and tmp[i + 1] > tmp[i + 2]:
                    swap_ok = False

                if swap_ok:
                    for k in [i - 1, i, i + 1]:
                        if 0 <= k < len(case_rows) - 1:
                            a, b = acts[k], acts[k + 1]
                            if not edge_exists(a, b):
                                swap_ok = False
                                break

                swap_score = candidate_score(case_rows, parsed_out, swap_repl) if swap_ok else -1e18

                # displaced candidates: move only one row, keep partner unchanged
                cand1_ts = estimate_single(case_rows, i, i + 1, parsed_out, original_parsed)
                repl1 = {i: cand1_ts}
                if cand1_ts == original_parsed[i]:
                    # force difference
                    step = smallest_step(case_rows[i]["precision"])
                    cand1_ts = clamp(cand1_ts - step, parsed_out[i - 1] if i - 1 >= 0 else None, parsed_out[i + 1])
                    cand1_ts = quantize_ts(cand1_ts, case_rows[i]["precision"])
                    repl1 = {i: cand1_ts}
                score1 = candidate_score(case_rows, parsed_out, repl1)

                cand2_ts = estimate_single(case_rows, i + 1, i, parsed_out, original_parsed)
                repl2 = {i + 1: cand2_ts}
                if cand2_ts == original_parsed[i + 1]:
                    step = smallest_step(case_rows[i + 1]["precision"])
                    cand2_ts = clamp(cand2_ts + step, parsed_out[i], parsed_out[i + 2] if i + 2 < len(case_rows) else None)
                    cand2_ts = quantize_ts(cand2_ts, case_rows[i + 1]["precision"])
                    repl2 = {i + 1: cand2_ts}
                score2 = candidate_score(case_rows, parsed_out, repl2)

                best_disp_repl = repl1 if score1 >= score2 else repl2
                best_disp_score = max(score1, score2)

                if swap_ok and swap_score > best_disp_score:
                    parsed_out[i] = original_parsed[i + 1]
                    parsed_out[i + 1] = original_parsed[i]
                    exchanged_segments += 1
                else:
                    for k, v in best_disp_repl.items():
                        parsed_out[k] = v
                    displaced_segments += 1
                i += 2
            else:
                i += 1

        # write repaired values back for non-clean rows only
        for local_i, row in enumerate(case_rows):
            gidx = row["gidx"]
            if df.loc[gidx, "error_category"] != "CLEAN":
                out_ts = parsed_out[local_i]
                prev_ts = parsed_out[local_i - 1] if local_i - 1 >= 0 else None
                next_ts = parsed_out[local_i + 1] if local_i + 1 < len(parsed_out) else None
                out_ts = make_non_decreasing(out_ts, prev_ts, next_ts, row["precision"])

                if out_ts == original_parsed[local_i]:
                    step = smallest_step(row["precision"])
                    if next_ts is not None and out_ts < next_ts:
                        out_ts = clamp(out_ts + step, prev_ts, next_ts)
                    elif prev_ts is not None and out_ts > prev_ts:
                        out_ts = clamp(out_ts - step, prev_ts, next_ts)
                    out_ts = quantize_ts(out_ts, row["precision"])

                df.loc[gidx, "__out_ts"] = fmt_ts(out_ts, row["precision"])

    # -----------------------------------------------------------------
    # Self-checks and auto-fixes
    # -----------------------------------------------------------------
    fixed_messages = []

    # only non-clean may change, clean must remain exact
    clean_mask = df["error_category"] == "CLEAN"
    df.loc[clean_mask, "__out_ts"] = df.loc[clean_mask, "__orig_ts"]

    # every corrupted row must differ from input
    bad_same = (df["error_category"] != "CLEAN") & (df["__out_ts"] == df["__orig_ts"])
    if bad_same.any():
        for idx in df.index[bad_same]:
            prec = int(df.loc[idx, "__precision"])
            cur = parse_ts(df.loc[idx, "__out_ts"])
            case = df.loc[idx, "Case"]
            case_idx = list(grouped[case])
            pos = case_idx.index(idx)
            prev_ts = parse_ts(df.loc[case_idx[pos - 1], "__out_ts"]) if pos > 0 else None
            next_ts = parse_ts(df.loc[case_idx[pos + 1], "__out_ts"]) if pos + 1 < len(case_idx) else None
            step = smallest_step(prec)
            new_ts = cur
            if next_ts is not None and cur < next_ts:
                new_ts = clamp(cur + step, prev_ts, next_ts)
            elif prev_ts is not None and cur > prev_ts:
                new_ts = clamp(cur - step, prev_ts, next_ts)
            new_ts = quantize_ts(new_ts, prec)
            if new_ts == cur:
                if prev_ts is not None:
                    new_ts = prev_ts
                elif next_ts is not None:
                    new_ts = next_ts
            if fmt_ts(new_ts, prec) == df.loc[idx, "__orig_ts"]:
                if next_ts is not None:
                    new_ts = clamp(cur + step, prev_ts, next_ts)
                elif prev_ts is not None:
                    new_ts = clamp(cur - step, prev_ts, next_ts)
            df.loc[idx, "__out_ts"] = fmt_ts(new_ts, prec)
        fixed_messages.append(f"fixed_same_as_input={int(bad_same.sum())}")

    # parseability and monotonicity within case
    for case, idxs in grouped.items():
        idxs = list(idxs)
        out_parsed = [parse_ts(df.loc[idx, "__out_ts"]) for idx in idxs]
        for j in range(1, len(idxs)):
            if out_parsed[j] < out_parsed[j - 1]:
                idx = idxs[j]
                prec = int(df.loc[idx, "__precision"])
                lo = out_parsed[j - 1]
                hi = out_parsed[j + 1] if j + 1 < len(idxs) else None
                candidate = clamp(lo, lo, hi)
                candidate = quantize_ts(candidate, prec)
                if df.loc[idx, "error_category"] != "CLEAN":
                    orig = parse_ts(df.loc[idx, "__orig_ts"])
                    if candidate == orig:
                        step = smallest_step(prec)
                        candidate = clamp(candidate + step, lo, hi)
                        candidate = quantize_ts(candidate, prec)
                    df.loc[idx, "__out_ts"] = fmt_ts(candidate, prec)
                    out_parsed[j] = candidate
        # clean rows still exact after monotonic fixes
        for idx in idxs:
            if df.loc[idx, "error_category"] == "CLEAN":
                df.loc[idx, "__out_ts"] = df.loc[idx, "__orig_ts"]

    still_equal = int(((df["error_category"] != "CLEAN") & (df["__out_ts"] == df["__orig_ts"])).sum())

    if fixed_messages:
        print("auto_fixed " + " ".join(fixed_messages))
    print(f"exchanged_segments={exchanged_segments} displaced_segments={displaced_segments} rows_still_equal_to_input={still_equal}")

    out = df[original_cols].copy()
    out["Timestamp"] = df["__out_ts"]

    # preserve all other columns byte-identical and row order unchanged
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()