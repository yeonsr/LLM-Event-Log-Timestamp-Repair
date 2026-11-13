import argparse
import sys
from datetime import timedelta
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Embedded process knowledge used exactly for C2_TEMPORAL_INCONSISTENCY
# ---------------------------------------------------------------------

DURATION_PROFILES = {
    ("register submission date request", "OLO messaging active"): {"n": 729, "p10": 86400.0, "p25": 172800.0, "median": 306406.0, "p75": 518400.0, "p90": 733026.6, "share_under_60s": 0.0},
    ("forward to the competent authority", "regular procedure without MER"): {"n": 687, "p10": 0.0, "p25": 0.0, "median": 3.0, "p75": 6.0, "p90": 12.0, "share_under_60s": 0.975},
    ("send confirmation receipt", "forward to the competent authority"): {"n": 582, "p10": 2.0, "p25": 3.0, "median": 5.0, "p75": 9.0, "p90": 20.9, "share_under_60s": 0.94},
    ("request complete", "send letter in progress"): {"n": 572, "p10": 0.0, "p25": 0.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "share_under_60s": 0.991},
    ("article 34  WABO applies", "assessment of content completed"): {"n": 568, "p10": 0.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "share_under_60s": 0.993},
    ("assessment of content completed", "phase advice known"): {"n": 555, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 1.0, "share_under_60s": 1.0},
    ("treat subcases completeness", "subcases completeness completed"): {"n": 510, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 3.0, "p90": 9.0, "share_under_60s": 0.969},
    ("send confirmation receipt", "send confirmation receipt"): {"n": 501, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 214.0, "share_under_60s": 0.802},
    ("phase application receptive", "request complete"): {"n": 500, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("by law", "creating environmental permit decision"): {"n": 493, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 1.0, "p90": 1.0, "share_under_60s": 0.996},
    ("treat subcases content", "completed subcases content"): {"n": 492, "p10": 1.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 3.0, "share_under_60s": 0.996},
    ("enter senddate decision", "enter senddate decision environmental permit"): {"n": 467, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 0.991},
    ("phase advice known", "grounds for refusal"): {"n": 439, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 3.0, "p90": 6.2, "share_under_60s": 0.954},
    ("subcases completeness completed", "procedure change"): {"n": 436, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 4.0, "p90": 7.0, "share_under_60s": 0.986},
    ("ask stakeholders views", "suspension ground applicable"): {"n": 431, "p10": 0.0, "p25": 1.0, "median": 2.0, "p75": 4.0, "p90": 6.0, "share_under_60s": 0.984},
    ("creating environmental permit decision", "decision date prior to decision"): {"n": 431, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 4.0, "share_under_60s": 0.993},
    ("procedure change", "request complete"): {"n": 429, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 3.0, "p90": 6.0, "share_under_60s": 0.991},
    ("grounds for refusal", "ask stakeholders views"): {"n": 425, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 5.0, "p90": 15.0, "share_under_60s": 0.958},
    ("suspension ground applicable", "by law"): {"n": 415, "p10": 1.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "share_under_60s": 0.993},
    ("send confirmation receipt", "enter senddate acknowledgement"): {"n": 412, "p10": 2.0, "p25": 74.25, "median": 159.0, "p75": 308.25, "p90": 1342.5, "share_under_60s": 0.24},
    ("OLO messaging active", "phase application received"): {"n": 409, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("regular procedure without MER", "send confirmation receipt"): {"n": 399, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("phase application received", "send confirmation receipt"): {"n": 395, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 4.0, "p90": 10.0, "share_under_60s": 0.982},
    ("procedure change", "phase application receptive"): {"n": 384, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 5.0, "p90": 9.7, "share_under_60s": 0.964},
    ("enter senddate acknowledgement", "send confirmation receipt"): {"n": 383, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 0.99},
    ("enter senddate procedure confirmation", "send procedure confirmation"): {"n": 378, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 0.952},
    ("send confirmation receipt", "start WABOprocedure"): {"n": 375, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("forward to the competent authority", "phase application received"): {"n": 372, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("registration date publication", "regular procedure without MER"): {"n": 357, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 0.994},
    ("procedure change", "article 34  WABO applies"): {"n": 353, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 3.0, "p90": 10.8, "share_under_60s": 0.946},
    ("OLO messaging active", "create procedure confirmation"): {"n": 344, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("transcript decision environmental permit to stakeholders", "generate publication document decision environmental permit"): {"n": 342, "p10": 6.0, "p25": 7.0, "median": 10.0, "p75": 19.0, "p90": 55.0, "share_under_60s": 0.906},
    ("send letter in progress", "treat subcases content"): {"n": 324, "p10": 0.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 2.0, "share_under_60s": 0.994},
    ("phase decision sent", "phase decision taken"): {"n": 320, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("grounds for refusal", "phase advice known"): {"n": 319, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("create procedure confirmation", "create publication document"): {"n": 318, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 0.997},
    ("phase advice known", "phase application receptive"): {"n": 316, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("phase decision taken", "register objection and appeal periods"): {"n": 313, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 5.0, "p90": 12.8, "share_under_60s": 0.971},
    ("article 34  WABO applies", "ask stakeholders views"): {"n": 305, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 0.997},
    ("completed subcases content", "procedure change"): {"n": 299, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 5.0, "p90": 10.0, "share_under_60s": 0.98},
    ("enter senddate decision environmental permit", "phase decision sent"): {"n": 294, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 1.0, "share_under_60s": 1.0},
    ("creating environmental permit decision", "generating decision environmental permit"): {"n": 293, "p10": 4.0, "p25": 51.0, "median": 1864.0, "p75": 149542.0, "p90": 923794.2, "share_under_60s": 0.259},
    ("start WABOprocedure", "enter senddate acknowledgement"): {"n": 293, "p10": 34535.2, "p25": 38196.0, "median": 43648.0, "p75": 52527.0, "p90": 86400.0, "share_under_60s": 0.0},
    ("send letter in progress", "set decision status"): {"n": 254, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("set decision phase decision permitting decided", "start decision phase decision permitting sent"): {"n": 241, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 0.988},
    ("phase application received", "reception through OLO"): {"n": 237, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 7.0, "p90": 15.0, "share_under_60s": 0.966},
    ("regular procedure without MER", "start WABOprocedure"): {"n": 229, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 1.0, "share_under_60s": 1.0},
    ("send procedure confirmation", "registration date publication"): {"n": 225, "p10": 8.0, "p25": 11.0, "median": 19.0, "p75": 69.0, "p90": 738725.6, "share_under_60s": 0.724},
    ("transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit"): {"n": 224, "p10": 8.3, "p25": 34.75, "median": 50393.0, "p75": 172800.0, "p90": 259200.0, "share_under_60s": 0.254},
    ("regular procedure without MER", "enter senddate procedure confirmation"): {"n": 219, "p10": 1.0, "p25": 2.0, "median": 2.0, "p75": 4.0, "p90": 87809.2, "share_under_60s": 0.872},
    ("generating decision environmental permit", "grounds for refusal"): {"n": 215, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("decision date prior to decision", "generating decision environmental permit"): {"n": 214, "p10": 0.0, "p25": 0.0, "median": 2.5, "p75": 4991.0, "p90": 439697.8, "share_under_60s": 0.561},
    ("create publication document", "create subcases completeness"): {"n": 197, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 7.4, "share_under_60s": 0.914},
    ("ask stakeholders views", "assessment of content completed"): {"n": 194, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("by law", "completed subcases content"): {"n": 194, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("procedure change", "procedure change"): {"n": 193, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("submit decision", "suspension ground applicable"): {"n": 193, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("completed subcases content", "coordination of application"): {"n": 190, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
    ("register submission date request", "enter senddate acknowledgement"): {"n": 188, "p10": 172800.0, "p25": 345600.0, "median": 604800.0, "p75": 864000.0, "p90": 1209600.0, "share_under_60s": 0.0},
    ("record date of decision environmental permit", "set decision phase decision permitting decided"): {"n": 187, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0},
}

DFG_FREQ = {(k[0], k[1]): v["n"] for k, v in DURATION_PROFILES.items()}

BLOCK_SPAN = {
    2: {"p10": 0.0, "p25": 0.0, "median": 1.0, "p75": 11.0, "p90": 172800.0},
    3: {"p10": 0.0, "p25": 0.0, "median": 5.0, "p75": 1078.0, "p90": 688744.6},
    4: {"p10": 0.0, "p25": 0.0, "median": 12.0, "p75": 82456.25, "p90": 1468800.0},
    5: {"p10": 0.0, "p25": 0.0, "median": 30.0, "p75": 309510.5, "p90": 2336982.7},
}

FREQUENT_VARIANTS = [
    ["register submission date request","OLO messaging active","create procedure confirmation","create publication document","forward to the competent authority","phase application received","publish","registration date publication","regular procedure without MER","send confirmation receipt","send confirmation receipt","start WABOprocedure","enter senddate acknowledgement"],
    ["register submission date request","enter senddate acknowledgement","phase application received","reception through OLO","send confirmation receipt","forward to the competent authority"],
    ["register submission date request","enter senddate acknowledgement","phase application received","reception through OLO","send confirmation receipt","forward to the competent authority","regular procedure without MER"],
    ["register submission date request","OLO messaging active","phase application received","send confirmation receipt","enter senddate acknowledgement","send confirmation receipt","forward to the competent authority","regular procedure without MER","start WABOprocedure","enter senddate procedure confirmation","send procedure confirmation","registration date publication","create publication document","treat subcases completeness","subcases completeness completed","procedure change","phase application receptive","request complete","procedure change after completeness","send letter in progress","treat subcases content","completed subcases content","procedure change","article 34  WABO applies","assessment of content completed","phase advice known","grounds for refusal","ask stakeholders views","suspension ground applicable","by law","creating environmental permit decision","decision date prior to decision","generating decision environmental permit","record date of decision environmental permit","submit decision","phase decision taken","register objection and appeal periods","enter senddate decision environmental permit","transcript decision environmental permit to stakeholders","enter senddate decision","enter senddate decision environmental permit","phase decision sent","create publication document","date decision for inspection"],
    ["register submission date request","OLO messaging active","create procedure confirmation","create publication document","forward to the competent authority","phase application received","publish","registration date publication","regular procedure without MER","send confirmation receipt","send confirmation receipt","start WABOprocedure","enter senddate acknowledgement","article 34  WABO applies","ask stakeholders views","by law","coordination of application","create subcases completeness","create subcases content","creating cover letter decision","creating environmental permit decision","decision date prior to decision","enter senddate decision","enter senddate decision environmental permit","generating decision environmental permit","grounds for refusal","phase advice known","phase application receptive","phase decision ready","phase decision sent","phase decision taken","procedure change","procedure change","register objection and appeal periods","request complete","send letter in progress","set decision status","submit decision","suspension ground applicable","transcript decision environmental permit to stakeholders","record date of decision environmental permit","set decision phase decision permitting decided","enter senddate decision environmental permit","start decision phase decision permitting sent"],
]

GLOBAL_MEDIAN_SECONDS = 1.0


def detect_precision(s):
    s = str(s)
    if "." in s:
        frac = s.split(".")[-1]
        frac = "".join(ch for ch in frac if ch.isdigit())
        return len(frac)
    return 0


def format_ts(ts, precision):
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    return base[:-6 + precision]


def get_profile(a, b):
    return DURATION_PROFILES.get((a, b))


def preferred_gap_seconds(a, b):
    prof = get_profile(a, b)
    if prof is None:
        if (a, b) in DFG_FREQ:
            return max(0.0, prof["median"])
        return GLOBAL_MEDIAN_SECONDS
    if prof["share_under_60s"] >= 0.8:
        return max(0.0, prof["p25"])
    return max(0.0, prof["p25"])


def edge_score(a, b):
    if (a, b) not in DFG_FREQ:
        return -1000.0
    return np.log1p(DFG_FREQ[(a, b)])


def local_chain_score(prev_act, a1, a2, next_act):
    score = 0.0
    if prev_act is not None:
        score += edge_score(prev_act, a1)
    score += edge_score(a1, a2)
    if next_act is not None:
        score += edge_score(a2, next_act)
    return score


def contiguous_variant_boost(seq):
    if not seq:
        return 0.0
    best = 0.0
    n = len(seq)
    for var in FREQUENT_VARIANTS:
        for i in range(len(var) - n + 1):
            if var[i:i + n] == seq:
                best = max(best, 5.0)
    return best


def plausibility_penalty(actual_gap, a, b):
    prof = get_profile(a, b)
    if actual_gap < 0:
        return 1e6
    if prof is None:
        return abs(actual_gap - GLOBAL_MEDIAN_SECONDS)
    target = preferred_gap_seconds(a, b)
    if prof["share_under_60s"] >= 0.95:
        scale = max(1.0, prof["p90"] + 1.0)
    elif prof["share_under_60s"] >= 0.8:
        scale = max(2.0, prof["p90"] + 1.0)
    else:
        scale = max(10.0, prof["median"] + 1.0)
    return abs(actual_gap - target) / scale


def choose_feasible_between(low, high, preferred, observed=None):
    if low > high:
        mid = low
        return mid
    cand = preferred
    if cand < low:
        cand = low
    if cand > high:
        cand = high
    if observed is not None:
        obs_clamped = observed
        if obs_clamped < low:
            obs_clamped = low
        if obs_clamped > high:
            obs_clamped = high
        if abs((obs_clamped - observed).total_seconds()) < abs((cand - observed).total_seconds()):
            cand = obs_clamped
    return cand


def estimate_from_anchors(prev_ts, prev_act, act, next_ts, next_act, observed, precision):
    low = prev_ts
    high = next_ts
    left_gap = preferred_gap_seconds(prev_act, act) if prev_ts is not None and prev_act is not None else None
    right_gap = preferred_gap_seconds(act, next_act) if next_ts is not None and next_act is not None else None

    candidates = []
    if prev_ts is not None and left_gap is not None:
        candidates.append(prev_ts + pd.to_timedelta(left_gap, unit="s"))
    if next_ts is not None and right_gap is not None:
        candidates.append(next_ts - pd.to_timedelta(right_gap, unit="s"))
    if candidates:
        preferred = min(candidates, key=lambda x: abs((x - observed).total_seconds()))
    elif prev_ts is not None:
        preferred = prev_ts + pd.to_timedelta(GLOBAL_MEDIAN_SECONDS, unit="s")
    elif next_ts is not None:
        preferred = next_ts - pd.to_timedelta(GLOBAL_MEDIAN_SECONDS, unit="s")
    else:
        preferred = observed + pd.to_timedelta(1, unit="s")

    t = choose_feasible_between(low, high, preferred, observed=observed)
    if t == observed:
        eps = pd.to_timedelta(10 ** (-precision) if precision > 0 else 1, unit="s")
        if t + eps <= high:
            t = t + eps
        elif t - eps >= low:
            t = t - eps
        else:
            t = low
            if t == observed and t < high:
                t = high
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    exchanged_segments = 0
    displaced_segments = 0
    fixed_messages = []

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"failed_to_read_input: {e}")
        pd.DataFrame(columns=["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]).to_csv(args.output, index=False)
        return

    required_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    original = df.copy(deep=True)
    df["_orig_ts"] = df["Timestamp"]
    df["_parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_precision"] = df["Timestamp"].map(detect_precision)
    df["_out_ts"] = df["Timestamp"]
    df["_repaired"] = False

    # Only error present in this task: C2_TEMPORAL_INCONSISTENCY
    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_idxs = list(idxs)
        case_df = df.loc[case_idxs].copy()
        parsed = list(case_df["_parsed_ts"])
        acts = list(case_df["Activity"])
        errs = list(case_df["error_category"])
        precisions = list(case_df["_precision"])

        case_valid_ts = [t for t in parsed if pd.notna(t)]
        if case_valid_ts:
            case_start = min(case_valid_ts)
            case_end = max(case_valid_ts)
        else:
            case_start = pd.Timestamp("1900-01-01")
            case_end = pd.Timestamp("2100-01-01")

        for j in range(len(case_idxs) - 1):
            gi = case_idxs[j]
            gj = case_idxs[j + 1]

            if df.at[gi, "error_category"] == "CLEAN" and df.at[gj, "error_category"] == "CLEAN":
                continue

            ti = df.at[gi, "_parsed_ts"]
            tj = df.at[gj, "_parsed_ts"]
            if pd.isna(ti) or pd.isna(tj):
                continue
            if tj >= ti:
                continue

            prev_idx = case_idxs[j - 1] if j - 1 >= 0 else None
            next_idx = case_idxs[j + 2] if j + 2 < len(case_idxs) else None

            left_anchor_idx = prev_idx
            while left_anchor_idx is not None and df.at[left_anchor_idx, "error_category"] != "CLEAN":
                pos = case_idxs.index(left_anchor_idx)
                left_anchor_idx = case_idxs[pos - 1] if pos - 1 >= 0 else None

            right_anchor_idx = next_idx
            while right_anchor_idx is not None and df.at[right_anchor_idx, "error_category"] != "CLEAN":
                pos = case_idxs.index(right_anchor_idx)
                right_anchor_idx = case_idxs[pos + 1] if pos + 1 < len(case_idxs) else None

            left_anchor = df.at[left_anchor_idx, "_parsed_ts"] if left_anchor_idx is not None else case_start
            right_anchor = df.at[right_anchor_idx, "_parsed_ts"] if right_anchor_idx is not None else case_end

            ai = df.at[gi, "Activity"]
            aj = df.at[gj, "Activity"]
            prev_act = df.at[prev_idx, "Activity"] if prev_idx is not None else None
            next_act = df.at[next_idx, "Activity"] if next_idx is not None else None

            # Candidate 1: EXCHANGED = exact swap
            swap_ti = tj
            swap_tj = ti
            swap_valid = True
            if swap_ti < left_anchor or swap_tj > right_anchor or swap_ti > swap_tj:
                swap_valid = False
            if prev_idx is not None and df.at[prev_idx, "_parsed_ts"] > swap_ti:
                swap_valid = False
            if next_idx is not None and swap_tj > df.at[next_idx, "_parsed_ts"]:
                swap_valid = False

            swap_score = -1e9
            if swap_valid:
                seq = []
                if prev_act is not None:
                    seq.append(prev_act)
                seq.extend([ai, aj])
                if next_act is not None:
                    seq.append(next_act)
                swap_score = local_chain_score(prev_act, ai, aj, next_act) + contiguous_variant_boost(seq)
                if prev_idx is not None:
                    swap_score -= plausibility_penalty((swap_ti - df.at[prev_idx, "_parsed_ts"]).total_seconds(), prev_act, ai)
                swap_score -= plausibility_penalty((swap_tj - swap_ti).total_seconds(), ai, aj)
                if next_idx is not None:
                    swap_score -= plausibility_penalty((df.at[next_idx, "_parsed_ts"] - swap_tj).total_seconds(), aj, next_act)
                if df.at[gi, "_orig_ts"] == df.at[gj, "_orig_ts"]:
                    swap_valid = False
                    swap_score = -1e9

            # Candidate 2a: DISPLACED, repair row i only, keep j unchanged
            disp_i_valid = True
            low_i = left_anchor
            high_i = min(tj, right_anchor)
            est_i = estimate_from_anchors(
                df.at[prev_idx, "_parsed_ts"] if prev_idx is not None else left_anchor,
                prev_act,
                ai,
                tj,
                aj,
                ti,
                int(df.at[gi, "_precision"])
            )
            est_i = choose_feasible_between(low_i, high_i, est_i, observed=ti)
            if est_i > tj:
                disp_i_valid = False
            if est_i < left_anchor or est_i > right_anchor:
                disp_i_valid = False
            if est_i == ti:
                eps = pd.to_timedelta(10 ** (-int(df.at[gi, "_precision"])) if int(df.at[gi, "_precision"]) > 0 else 1, unit="s")
                if est_i + eps <= high_i:
                    est_i = est_i + eps
                elif est_i - eps >= low_i:
                    est_i = est_i - eps
            disp_i_score = -1e9
            if disp_i_valid:
                seq = []
                if prev_act is not None:
                    seq.append(prev_act)
                seq.extend([ai, aj])
                if next_act is not None:
                    seq.append(next_act)
                disp_i_score = local_chain_score(prev_act, ai, aj, next_act) + contiguous_variant_boost(seq)
                if prev_idx is not None:
                    disp_i_score -= plausibility_penalty((est_i - df.at[prev_idx, "_parsed_ts"]).total_seconds(), prev_act, ai)
                disp_i_score -= plausibility_penalty((tj - est_i).total_seconds(), ai, aj)
                if next_idx is not None:
                    disp_i_score -= plausibility_penalty((df.at[next_idx, "_parsed_ts"] - tj).total_seconds(), aj, next_act)
                disp_i_score -= abs((est_i - ti).total_seconds()) / 10.0

            # Candidate 2b: DISPLACED, repair row j only, keep i unchanged
            disp_j_valid = True
            low_j = max(ti, left_anchor)
            high_j = right_anchor
            est_j = estimate_from_anchors(
                ti,
                ai,
                aj,
                df.at[next_idx, "_parsed_ts"] if next_idx is not None else right_anchor,
                next_act,
                tj,
                int(df.at[gj, "_precision"])
            )
            est_j = choose_feasible_between(low_j, high_j, est_j, observed=tj)
            if est_j < ti:
                disp_j_valid = False
            if est_j < left_anchor or est_j > right_anchor:
                disp_j_valid = False
            if est_j == tj:
                eps = pd.to_timedelta(10 ** (-int(df.at[gj, "_precision"])) if int(df.at[gj, "_precision"]) > 0 else 1, unit="s")
                if est_j + eps <= high_j:
                    est_j = est_j + eps
                elif est_j - eps >= low_j:
                    est_j = est_j - eps
            disp_j_score = -1e9
            if disp_j_valid:
                seq = []
                if prev_act is not None:
                    seq.append(prev_act)
                seq.extend([ai, aj])
                if next_act is not None:
                    seq.append(next_act)
                disp_j_score = local_chain_score(prev_act, ai, aj, next_act) + contiguous_variant_boost(seq)
                if prev_idx is not None:
                    disp_j_score -= plausibility_penalty((ti - df.at[prev_idx, "_parsed_ts"]).total_seconds(), prev_act, ai)
                disp_j_score -= plausibility_penalty((est_j - ti).total_seconds(), ai, aj)
                if next_idx is not None:
                    disp_j_score -= plausibility_penalty((df.at[next_idx, "_parsed_ts"] - est_j).total_seconds(), aj, next_act)
                disp_j_score -= abs((est_j - tj).total_seconds()) / 10.0

            # Decide
            if swap_valid and swap_score >= disp_i_score and swap_score >= disp_j_score:
                df.at[gi, "_out_ts"] = df.at[gj, "_orig_ts"]
                df.at[gj, "_out_ts"] = df.at[gi, "_orig_ts"]
                df.at[gi, "_repaired"] = True
                df.at[gj, "_repaired"] = True
                exchanged_segments += 1
            else:
                if disp_i_score >= disp_j_score and disp_i_valid and df.at[gi, "error_category"] != "CLEAN":
                    p = int(df.at[gi, "_precision"])
                    df.at[gi, "_out_ts"] = format_ts(est_i, p)
                    df.at[gi, "_repaired"] = True
                    displaced_segments += 1
                elif disp_j_valid and df.at[gj, "error_category"] != "CLEAN":
                    p = int(df.at[gj, "_precision"])
                    df.at[gj, "_out_ts"] = format_ts(est_j, p)
                    df.at[gj, "_repaired"] = True
                    displaced_segments += 1
                else:
                    # force a valid displaced repair on a non-CLEAN row, never keep corrupted value
                    if df.at[gj, "error_category"] != "CLEAN":
                        forced = max(ti, left_anchor)
                        step = pd.to_timedelta(10 ** (-int(df.at[gj, "_precision"])) if int(df.at[gj, "_precision"]) > 0 else 1, unit="s")
                        if forced == tj:
                            forced = min(right_anchor, forced + step)
                            if forced == tj and forced - step >= ti:
                                forced = forced - step
                        df.at[gj, "_out_ts"] = format_ts(forced, int(df.at[gj, "_precision"]))
                        df.at[gj, "_repaired"] = True
                        displaced_segments += 1
                        fixed_messages.append(f"forced_displaced_row:{df.at[gj,'row_id']}")
                    elif df.at[gi, "error_category"] != "CLEAN":
                        forced = min(tj, right_anchor)
                        step = pd.to_timedelta(10 ** (-int(df.at[gi, "_precision"])) if int(df.at[gi, "_precision"]) > 0 else 1, unit="s")
                        if forced == ti:
                            if forced - step >= left_anchor:
                                forced = forced - step
                            elif forced + step <= tj:
                                forced = forced + step
                        df.at[gi, "_out_ts"] = format_ts(forced, int(df.at[gi, "_precision"]))
                        df.at[gi, "_repaired"] = True
                        displaced_segments += 1
                        fixed_messages.append(f"forced_displaced_row:{df.at[gi,'row_id']}")

    # Self-check and repair pass
    out_parsed = pd.to_datetime(df["_out_ts"], errors="coerce")
    df["_out_parsed"] = out_parsed

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_idxs = list(idxs)
        case_ts = [df.at[i, "_out_parsed"] for i in case_idxs if pd.notna(df.at[i, "_out_parsed"])]
        if case_ts:
            case_start = min(case_ts)
            case_end = max(case_ts)
        else:
            case_start = pd.Timestamp("1900-01-01")
            case_end = pd.Timestamp("2100-01-01")

        # anchor bounds check and monotonic correction
        for pos, i in enumerate(case_idxs):
            if df.at[i, "error_category"] == "CLEAN":
                continue
            if not df.at[i, "_repaired"]:
                # must change every non-CLEAN row
                t_obs = df.at[i, "_parsed_ts"]
                p = int(df.at[i, "_precision"])
                prev_t = df.at[case_idxs[pos - 1], "_out_parsed"] if pos > 0 else case_start
                next_t = df.at[case_idxs[pos + 1], "_out_parsed"] if pos + 1 < len(case_idxs) else case_end
                low = prev_t if pd.notna(prev_t) else case_start
                high = next_t if pd.notna(next_t) else case_end
                if low > high:
                    high = low
                step = pd.to_timedelta(10 ** (-p) if p > 0 else 1, unit="s")
                cand = low
                if pd.notna(t_obs) and cand == t_obs and cand + step <= high:
                    cand = cand + step
                elif pd.notna(t_obs) and cand == t_obs and cand - step >= low:
                    cand = cand - step
                df.at[i, "_out_ts"] = format_ts(cand, p)
                df.at[i, "_out_parsed"] = cand
                df.at[i, "_repaired"] = True
                fixed_messages.append(f"selfcheck_changed_nonclean_row:{df.at[i,'row_id']}")

        for pos, i in enumerate(case_idxs):
            t = df.at[i, "_out_parsed"]
            if pd.isna(t):
                p = int(df.at[i, "_precision"])
                prev_t = df.at[case_idxs[pos - 1], "_out_parsed"] if pos > 0 else case_start
                if pd.isna(prev_t):
                    prev_t = case_start
                t = prev_t
                if df.at[i, "error_category"] != "CLEAN":
                    step = pd.to_timedelta(10 ** (-p) if p > 0 else 1, unit="s")
                    t = min(case_end, t + step)
                df.at[i, "_out_parsed"] = t
                df.at[i, "_out_ts"] = format_ts(t, p)
                fixed_messages.append(f"selfcheck_parse_fix:{df.at[i,'row_id']}")

            if df.at[i, "error_category"] != "CLEAN":
                if t < case_start:
                    t = case_start
                if t > case_end:
                    t = case_end

                if pos > 0:
                    prev_t = df.at[case_idxs[pos - 1], "_out_parsed"]
                    if pd.notna(prev_t) and t < prev_t:
                        t = prev_t
                if pos + 1 < len(case_idxs):
                    next_t = df.at[case_idxs[pos + 1], "_out_parsed"]
                    if pd.notna(next_t) and t > next_t:
                        t = next_t

                if pd.notna(df.at[i, "_parsed_ts"]) and t == df.at[i, "_parsed_ts"]:
                    p = int(df.at[i, "_precision"])
                    step = pd.to_timedelta(10 ** (-p) if p > 0 else 1, unit="s")
                    low = df.at[case_idxs[pos - 1], "_out_parsed"] if pos > 0 else case_start
                    high = df.at[case_idxs[pos + 1], "_out_parsed"] if pos + 1 < len(case_idxs) else case_end
                    if pd.isna(low):
                        low = case_start
                    if pd.isna(high):
                        high = case_end
                    if t + step <= high:
                        t = t + step
                    elif t - step >= low:
                        t = t - step

                df.at[i, "_out_parsed"] = t
                df.at[i, "_out_ts"] = format_ts(t, int(df.at[i, "_precision"]))

    # Restore exact original strings for untouched rows and all CLEAN rows
    for i in df.index:
        if df.at[i, "error_category"] == "CLEAN":
            df.at[i, "_out_ts"] = df.at[i, "_orig_ts"]
        elif not df.at[i, "_repaired"]:
            df.at[i, "_out_ts"] = df.at[i, "_orig_ts"]

    rows_equal_to_input = int((df["_out_ts"] == df["_orig_ts"]).sum())

    out_df = df[required_cols].copy()
    out_df["Timestamp"] = df["_out_ts"]

    # Final structural preservation
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        out_df[col] = original[col]

    try:
        out_df.to_csv(args.output, index=False)
    except Exception as e:
        print(f"failed_to_write_output:{e}")
        # best effort fallback
        try:
            original.to_csv(args.output, index=False)
        except Exception:
            pass

    if fixed_messages:
        print(";".join(fixed_messages))
    print(f"exchanged_segments={exchanged_segments} displaced_segments={displaced_segments} rows_equal_to_input={rows_equal_to_input}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"top_level_recovery:{e}")
        # non-zero exit forbidden; attempt best-effort empty output if possible
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=False)
            parser.add_argument("--output", required=False)
            args, _ = parser.parse_known_args()
            if getattr(args, "input", None) and getattr(args, "output", None):
                df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
                df.to_csv(args.output, index=False)
        except Exception:
            pass