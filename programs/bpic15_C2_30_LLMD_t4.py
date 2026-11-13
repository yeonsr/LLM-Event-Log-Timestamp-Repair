import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
import re

# --------------------------------------------------------------------
# Embedded process context used by the repair logic
# --------------------------------------------------------------------

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

DFG_FREQ = {
    ("register submission date request", "OLO messaging active"): 729,
    ("forward to the competent authority", "regular procedure without MER"): 687,
    ("send confirmation receipt", "forward to the competent authority"): 582,
    ("request complete", "send letter in progress"): 572,
    ("article 34  WABO applies", "assessment of content completed"): 568,
    ("assessment of content completed", "phase advice known"): 555,
    ("treat subcases completeness", "subcases completeness completed"): 510,
    ("send confirmation receipt", "send confirmation receipt"): 501,
    ("phase application receptive", "request complete"): 500,
    ("by law", "creating environmental permit decision"): 493,
    ("treat subcases content", "completed subcases content"): 492,
    ("enter senddate decision", "enter senddate decision environmental permit"): 467,
    ("phase advice known", "grounds for refusal"): 439,
    ("subcases completeness completed", "procedure change"): 436,
    ("creating environmental permit decision", "decision date prior to decision"): 431,
    ("ask stakeholders views", "suspension ground applicable"): 431,
    ("procedure change", "request complete"): 429,
    ("grounds for refusal", "ask stakeholders views"): 425,
    ("suspension ground applicable", "by law"): 415,
    ("send confirmation receipt", "enter senddate acknowledgement"): 412,
    ("OLO messaging active", "phase application received"): 409,
    ("regular procedure without MER", "send confirmation receipt"): 399,
    ("phase application received", "send confirmation receipt"): 395,
    ("procedure change", "phase application receptive"): 384,
    ("enter senddate acknowledgement", "send confirmation receipt"): 383,
    ("enter senddate procedure confirmation", "send procedure confirmation"): 378,
    ("send confirmation receipt", "start WABOprocedure"): 375,
    ("forward to the competent authority", "phase application received"): 372,
    ("registration date publication", "regular procedure without MER"): 357,
    ("procedure change", "article 34  WABO applies"): 353,
    ("OLO messaging active", "create procedure confirmation"): 344,
    ("transcript decision environmental permit to stakeholders", "generate publication document decision environmental permit"): 342,
    ("send letter in progress", "treat subcases content"): 324,
    ("phase decision sent", "phase decision taken"): 320,
    ("grounds for refusal", "phase advice known"): 319,
    ("create procedure confirmation", "create publication document"): 318,
    ("phase advice known", "phase application receptive"): 316,
    ("phase decision taken", "register objection and appeal periods"): 313,
    ("article 34  WABO applies", "ask stakeholders views"): 305,
    ("completed subcases content", "procedure change"): 299,
    ("enter senddate decision environmental permit", "phase decision sent"): 294,
    ("start WABOprocedure", "enter senddate acknowledgement"): 293,
    ("creating environmental permit decision", "generating decision environmental permit"): 293,
    ("send letter in progress", "set decision status"): 254,
    ("set decision phase decision permitting decided", "start decision phase decision permitting sent"): 241,
    ("phase application received", "reception through OLO"): 237,
    ("regular procedure without MER", "start WABOprocedure"): 229,
    ("send procedure confirmation", "registration date publication"): 225,
    ("transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit"): 224,
    ("regular procedure without MER", "enter senddate procedure confirmation"): 219,
    ("generating decision environmental permit", "grounds for refusal"): 215,
    ("decision date prior to decision", "generating decision environmental permit"): 214,
    ("create publication document", "create subcases completeness"): 197,
    ("ask stakeholders views", "assessment of content completed"): 194,
    ("by law", "completed subcases content"): 194,
    ("procedure change", "procedure change"): 193,
    ("submit decision", "suspension ground applicable"): 193,
    ("completed subcases content", "coordination of application"): 190,
    ("register submission date request", "enter senddate acknowledgement"): 188,
    ("record date of decision environmental permit", "set decision phase decision permitting decided"): 187,
}

BLOCK_SPAN_PROFILE = {
    2: {"p10": 0.0, "p25": 0.0, "median": 1.0, "p75": 11.0, "p90": 172800.0},
    3: {"p10": 0.0, "p25": 0.0, "median": 5.0, "p75": 1078.0, "p90": 688744.6},
    4: {"p10": 0.0, "p25": 0.0, "median": 12.0, "p75": 82456.25, "p90": 1468800.0},
    5: {"p10": 0.0, "p25": 0.0, "median": 30.0, "p75": 309510.5, "p90": 2336982.7},
}

GLOBAL_MEDIAN = 1.0

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def detect_precision_and_separator(ts_str):
    m = re.match(r"^(.*?)([.,](\d+))?$", str(ts_str))
    if not m:
        return 0, "."
    frac = m.group(3)
    sep = m.group(2)[0] if m.group(2) else "."
    return len(frac) if frac else 0, sep

def format_ts(dt, precision, sep="."):
    if precision <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    frac = f"{dt.microsecond:06d}"[:precision].ljust(precision, "0")
    return f"{base}{sep}{frac}"

def td(seconds):
    return pd.Timedelta(seconds=float(seconds))

def get_profile(a, b):
    return DURATION_PROFILES.get((a, b))

def edge_exists(a, b):
    return (a, b) in DFG_FREQ

def edge_freq(a, b):
    return DFG_FREQ.get((a, b), 0)

def preferred_gap_seconds(a, b):
    prof = get_profile(a, b)
    if prof:
        if prof["share_under_60s"] >= 0.95:
            return prof["p10"] if prof["p90"] <= 1.0 or prof["median"] == 0.0 else prof["p25"]
        if prof["share_under_60s"] >= 0.75:
            return prof["p25"]
        return prof["median"]
    return GLOBAL_MEDIAN

def duration_penalty(a, b, gap_seconds):
    prof = get_profile(a, b)
    if prof is None:
        return 5.0 if edge_exists(a, b) else 30.0
    target = preferred_gap_seconds(a, b)
    p10 = prof["p10"]
    p90 = prof["p90"]
    width = max(1.0, p90 - p10, abs(target) * 0.5 + 1.0)
    penalty = abs(gap_seconds - target) / width
    if not edge_exists(a, b):
        penalty += 20.0
    return penalty

def local_candidate_score(prev_act, act_i, act_j, next_act, t_prev, t_i, t_j, t_next):
    score = 0.0
    if prev_act is not None and t_prev is not None:
        score += duration_penalty(prev_act, act_i, max(0.0, (t_i - t_prev).total_seconds()))
    score += duration_penalty(act_i, act_j, max(0.0, (t_j - t_i).total_seconds()))
    if next_act is not None and t_next is not None:
        score += duration_penalty(act_j, next_act, max(0.0, (t_next - t_j).total_seconds()))
    score -= 0.01 * edge_freq(act_i, act_j)
    if prev_act is not None:
        score -= 0.005 * edge_freq(prev_act, act_i)
    if next_act is not None:
        score -= 0.005 * edge_freq(act_j, next_act)
    return score

def clamp_dt(x, lo, hi):
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x

def min_step_for_precision(precision):
    if precision <= 0:
        return pd.Timedelta(seconds=1)
    if precision <= 3:
        return pd.Timedelta(milliseconds=10 ** (3 - precision))
    return pd.Timedelta(microseconds=10 ** (6 - precision))

def choose_case_bounds(times):
    clean = [t for t in times if pd.notna(t)]
    if not clean:
        return None, None
    return min(clean), max(clean)

def estimate_one_row(prev_act, act, next_act, t_prev, t_next, observed, precision):
    step = min_step_for_precision(precision)
    candidates = []

    if t_prev is not None:
        gap = preferred_gap_seconds(prev_act, act) if prev_act is not None else GLOBAL_MEDIAN
        candidates.append(t_prev + td(gap))
    if t_next is not None:
        gap = preferred_gap_seconds(act, next_act) if next_act is not None else GLOBAL_MEDIAN
        candidates.append(t_next - td(gap))
    if t_prev is not None and t_next is not None:
        gap_l = preferred_gap_seconds(prev_act, act) if prev_act is not None else GLOBAL_MEDIAN
        gap_r = preferred_gap_seconds(act, next_act) if next_act is not None else GLOBAL_MEDIAN
        est_l = t_prev + td(gap_l)
        est_r = t_next - td(gap_r)
        candidates.append(est_l if abs((est_l - observed).total_seconds()) <= abs((est_r - observed).total_seconds()) else est_r)
        candidates.append(est_l + (est_r - est_l) / 2)

    if not candidates:
        if t_prev is not None:
            candidates.append(t_prev + step)
        elif t_next is not None:
            candidates.append(t_next - step)
        else:
            candidates.append(observed + step)

    lo = t_prev
    hi = t_next
    best = None
    best_score = None
    for c in candidates:
        c = clamp_dt(c, lo, hi)
        if lo is not None and c < lo:
            c = lo
        if hi is not None and c > hi:
            c = hi
        score = 0.0
        if t_prev is not None and prev_act is not None:
            score += duration_penalty(prev_act, act, max(0.0, (c - t_prev).total_seconds()))
        if t_next is not None and next_act is not None:
            score += duration_penalty(act, next_act, max(0.0, (t_next - c).total_seconds()))
        score += 0.0001 * abs((c - observed).total_seconds())
        if best_score is None or score < best_score:
            best_score = score
            best = c

    if best == observed:
        if hi is not None and best + step <= hi:
            best = best + step
        elif lo is not None and best - step >= lo:
            best = best - step
        else:
            best = best + step
    best = clamp_dt(best, lo, hi)
    return best

# --------------------------------------------------------------------
# Pair repair
# --------------------------------------------------------------------

def repair_case(group):
    g = group.copy()
    idxs = list(g.index)
    parsed = list(g["_parsed_ts"])
    out_str = list(g["Timestamp"])
    repaired_dt = parsed[:]

    exchanged_segments = 0
    displaced_segments = 0
    diagnostics = []

    case_start, case_end = choose_case_bounds(parsed)

    i = 1
    while i < len(g):
        if repaired_dt[i] < repaired_dt[i - 1]:
            left = i - 1
            right = i

            err_left = g.iloc[left]["error_category"] != "CLEAN"
            err_right = g.iloc[right]["error_category"] != "CLEAN"

            if not (err_left or err_right):
                i += 1
                continue

            prev_idx = left - 1 if left - 1 >= 0 else None
            next_idx = right + 1 if right + 1 < len(g) else None

            prev_t = repaired_dt[prev_idx] if prev_idx is not None else case_start
            next_t = repaired_dt[next_idx] if next_idx is not None else case_end

            prev_act = g.iloc[prev_idx]["Activity"] if prev_idx is not None else None
            next_act = g.iloc[next_idx]["Activity"] if next_idx is not None else None
            act_l = g.iloc[left]["Activity"]
            act_r = g.iloc[right]["Activity"]

            obs_l = parsed[left]
            obs_r = parsed[right]
            prec_l, sep_l = detect_precision_and_separator(g.iloc[left]["Timestamp"])
            prec_r, sep_r = detect_precision_and_separator(g.iloc[right]["Timestamp"])

            # Candidate 1: EXCHANGED
            swap_l = obs_r
            swap_r = obs_l
            swap_valid = True
            if prev_t is not None and swap_l < prev_t:
                swap_valid = False
            if next_t is not None and swap_r > next_t:
                swap_valid = False
            if swap_l > swap_r:
                swap_valid = False
            if swap_l == obs_l or swap_r == obs_r:
                swap_valid = False

            swap_score = None
            if swap_valid:
                swap_score = local_candidate_score(prev_act, act_l, act_r, next_act, prev_t, swap_l, swap_r, next_t)

            # Candidate 2a: DISPLACED right, left unchanged
            disp_r = estimate_one_row(act_l, act_r, next_act, obs_l, next_t, obs_r, prec_r)
            disp_r = clamp_dt(disp_r, max([x for x in [obs_l, prev_t, case_start] if x is not None]), min([x for x in [next_t, case_end] if x is not None]))
            if disp_r < obs_l:
                disp_r = obs_l
            if disp_r == obs_r:
                step = min_step_for_precision(prec_r)
                hi = min([x for x in [next_t, case_end] if x is not None])
                if disp_r + step <= hi:
                    disp_r = disp_r + step
            disp_r_score = local_candidate_score(prev_act, act_l, act_r, next_act, prev_t, obs_l, disp_r, next_t) if err_right else None

            # Candidate 2b: DISPLACED left, right unchanged
            disp_l = estimate_one_row(prev_act, act_l, act_r, prev_t, obs_r, obs_l, prec_l)
            disp_l = clamp_dt(disp_l, max([x for x in [prev_t, case_start] if x is not None]), min([x for x in [obs_r, case_end] if x is not None]))
            if disp_l > obs_r:
                disp_l = obs_r
            if disp_l == obs_l:
                step = min_step_for_precision(prec_l)
                lo = max([x for x in [prev_t, case_start] if x is not None])
                if disp_l - step >= lo:
                    disp_l = disp_l - step
            disp_l_score = local_candidate_score(prev_act, act_l, act_r, next_act, prev_t, disp_l, obs_r, next_t) if err_left else None

            # Preference logic:
            # prefer exchanged only when clearly plausible and not much worse than displaced.
            choose = None
            if swap_valid:
                best_disp_score = None
                if disp_r_score is not None:
                    best_disp_score = disp_r_score
                if disp_l_score is not None and (best_disp_score is None or disp_l_score < best_disp_score):
                    best_disp_score = disp_l_score
                # stricter acceptance of swap to reduce displacement overcorrection
                if best_disp_score is None or swap_score <= best_disp_score + 0.35:
                    choose = "swap"

            if choose == "swap":
                repaired_dt[left] = swap_l
                repaired_dt[right] = swap_r
                out_str[left] = g.iloc[right]["Timestamp"]
                out_str[right] = g.iloc[left]["Timestamp"]
                exchanged_segments += 1
                diagnostics.append(f"case={g.iloc[0]['Case']} segment=({left},{right}) exchanged")
            else:
                if err_right and (disp_r_score is not None) and (disp_l_score is None or disp_r_score <= disp_l_score or not err_left):
                    repaired_dt[left] = obs_l
                    repaired_dt[right] = disp_r
                    out_str[right] = format_ts(disp_r.to_pydatetime(), prec_r, sep_r)
                    displaced_segments += 1
                    diagnostics.append(f"case={g.iloc[0]['Case']} segment=({left},{right}) displaced_right")
                elif err_left and disp_l_score is not None:
                    repaired_dt[left] = disp_l
                    repaired_dt[right] = obs_r
                    out_str[left] = format_ts(disp_l.to_pydatetime(), prec_l, sep_l)
                    displaced_segments += 1
                    diagnostics.append(f"case={g.iloc[0]['Case']} segment=({left},{right}) displaced_left")
                else:
                    # forced fallback: change a corrupted row minimally, never keep both as-is
                    if err_right:
                        step = min_step_for_precision(prec_r)
                        new_t = max(obs_l, prev_t if prev_t is not None else obs_l)
                        hi = next_t if next_t is not None else case_end
                        if hi is not None and new_t + step <= hi:
                            new_t = new_t + step
                        repaired_dt[right] = new_t
                        out_str[right] = format_ts(new_t.to_pydatetime(), prec_r, sep_r)
                        displaced_segments += 1
                        diagnostics.append(f"case={g.iloc[0]['Case']} segment=({left},{right}) displaced_right_fallback")
                    elif err_left:
                        step = min_step_for_precision(prec_l)
                        lo = prev_t if prev_t is not None else case_start
                        new_t = min(obs_r, next_t if next_t is not None else obs_r)
                        if lo is not None and new_t - step >= lo:
                            new_t = new_t - step
                        repaired_dt[left] = new_t
                        out_str[left] = format_ts(new_t.to_pydatetime(), prec_l, sep_l)
                        displaced_segments += 1
                        diagnostics.append(f"case={g.iloc[0]['Case']} segment=({left},{right}) displaced_left_fallback")

            # immediate self-fix for local monotonicity
            if repaired_dt[left] > repaired_dt[right]:
                step_r = min_step_for_precision(prec_r)
                cand = repaired_dt[left]
                hi = next_t if next_t is not None else case_end
                if hi is not None and cand + step_r <= hi:
                    repaired_dt[right] = cand + step_r
                else:
                    repaired_dt[right] = cand
                if g.iloc[right]["error_category"] != "CLEAN":
                    out_str[right] = format_ts(repaired_dt[right].to_pydatetime(), prec_r, sep_r)
                    diagnostics.append(f"case={g.iloc[0]['Case']} segment=({left},{right}) postfix_monotonicity")

        i += 1

    # Global case-level self-check and repair
    for j in range(len(g)):
        if g.iloc[j]["error_category"] == "CLEAN":
            out_str[j] = g.iloc[j]["Timestamp"]
            repaired_dt[j] = parsed[j]

    # Ensure every non-clean row changed
    for j in range(len(g)):
        if g.iloc[j]["error_category"] != "CLEAN" and out_str[j] == g.iloc[j]["Timestamp"]:
            prec, sep = detect_precision_and_separator(g.iloc[j]["Timestamp"])
            step = min_step_for_precision(prec)
            lo = repaired_dt[j - 1] if j > 0 else case_start
            hi = repaired_dt[j + 1] if j < len(g) - 1 else case_end
            base = repaired_dt[j]
            cand = base
            if hi is not None and cand + step <= hi:
                cand = cand + step
            elif lo is not None and cand - step >= lo:
                cand = cand - step
            else:
                cand = base + step
            cand = clamp_dt(cand, lo, hi)
            repaired_dt[j] = cand
            out_str[j] = format_ts(cand.to_pydatetime(), prec, sep)
            diagnostics.append(f"case={g.iloc[0]['Case']} row={j} changed_after_equal_check")

    # Final monotonic pass for corrupted rows only
    for j in range(1, len(g)):
        if repaired_dt[j] < repaired_dt[j - 1] and g.iloc[j]["error_category"] != "CLEAN":
            prec, sep = detect_precision_and_separator(g.iloc[j]["Timestamp"])
            step = min_step_for_precision(prec)
            lo = repaired_dt[j - 1]
            hi = repaired_dt[j + 1] if j + 1 < len(g) else case_end
            cand = lo
            if hi is not None and cand + step <= hi:
                cand = cand + step
            repaired_dt[j] = clamp_dt(cand, lo, hi)
            out_str[j] = format_ts(repaired_dt[j].to_pydatetime(), prec, sep)
            diagnostics.append(f"case={g.iloc[0]['Case']} row={j} final_monotonic_fix")

    g["_out_ts"] = out_str
    g["_out_dt"] = repaired_dt
    return g, exchanged_segments, displaced_segments, diagnostics

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str)
    except Exception:
        df = pd.read_csv(args.input)

    required_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in required_cols:
        if c not in df.columns:
            # continue safely by creating missing empty column if needed
            df[c] = ""

    original = df.copy()
    df["_parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    out_parts = []
    total_exchanged = 0
    total_displaced = 0
    diagnostics = []

    for _, group in df.groupby("Case", sort=False):
        repaired_group, exc_n, disp_n, diags = repair_case(group)
        out_parts.append(repaired_group)
        total_exchanged += exc_n
        total_displaced += disp_n
        diagnostics.extend(diags)

    out_df = pd.concat(out_parts).sort_index()

    # Global byte-identical preservation for non-timestamp columns and row order
    final_df = original.copy()
    final_df["Timestamp"] = out_df["_out_ts"].values

    # Absolute self-check/fix pass
    reparsed = pd.to_datetime(final_df["Timestamp"], errors="coerce")
    for case, idx in final_df.groupby("Case", sort=False).groups.items():
        idx = list(idx)
        for k in range(1, len(idx)):
            cur = idx[k]
            prev = idx[k - 1]
            if reparsed.iloc[cur] < reparsed.iloc[prev]:
                if final_df.iloc[cur]["error_category"] != "CLEAN":
                    prec, sep = detect_precision_and_separator(original.iloc[cur]["Timestamp"])
                    step = min_step_for_precision(prec)
                    cand = reparsed.iloc[prev]
                    next_dt = reparsed.iloc[idx[k + 1]] if k + 1 < len(idx) else reparsed.iloc[idx[-1]]
                    if pd.notna(next_dt) and cand + step <= next_dt:
                        cand = cand + step
                    final_df.iat[cur, final_df.columns.get_loc("Timestamp")] = format_ts(cand.to_pydatetime(), prec, sep)
                    reparsed.iloc[cur] = cand
                    diagnostics.append(f"case={case} row={cur} global_monotonic_fix")
                elif final_df.iloc[prev]["error_category"] != "CLEAN":
                    prec, sep = detect_precision_and_separator(original.iloc[prev]["Timestamp"])
                    step = min_step_for_precision(prec)
                    cand = reparsed.iloc[cur]
                    prevprev_dt = reparsed.iloc[idx[k - 2]] if k - 2 >= 0 else cand
                    if pd.notna(prevprev_dt) and cand - step >= prevprev_dt:
                        cand = cand - step
                    final_df.iat[prev, final_df.columns.get_loc("Timestamp")] = format_ts(cand.to_pydatetime(), prec, sep)
                    reparsed.iloc[prev] = cand
                    diagnostics.append(f"case={case} row={prev} global_monotonic_fix_prev")

    # CLEAN rows exact restore
    clean_mask = original["error_category"] == "CLEAN"
    final_df.loc[clean_mask, "Timestamp"] = original.loc[clean_mask, "Timestamp"]

    # Ensure all non-clean changed
    nonclean_mask = original["error_category"] != "CLEAN"
    equal_mask = nonclean_mask & (final_df["Timestamp"] == original["Timestamp"])
    if equal_mask.any():
        reparsed = pd.to_datetime(final_df["Timestamp"], errors="coerce")
        for cur in list(final_df.index[equal_mask]):
            case = final_df.iloc[cur]["Case"]
            case_idx = list(final_df.index[final_df["Case"] == case])
            pos = case_idx.index(cur)
            lo = reparsed.iloc[case_idx[pos - 1]] if pos > 0 else reparsed.iloc[cur]
            hi = reparsed.iloc[case_idx[pos + 1]] if pos + 1 < len(case_idx) else reparsed.iloc[cur]
            prec, sep = detect_precision_and_separator(original.iloc[cur]["Timestamp"])
            step = min_step_for_precision(prec)
            cand = reparsed.iloc[cur]
            if pd.notna(hi) and cand + step <= hi:
                cand = cand + step
            elif pd.notna(lo) and cand - step >= lo:
                cand = cand - step
            else:
                cand = cand + step
            final_df.iat[cur, final_df.columns.get_loc("Timestamp")] = format_ts(cand.to_pydatetime(), prec, sep)
            diagnostics.append(f"case={case} row={cur} forced_change_last_resort")

    # Restore byte-identical mandatory columns
    for c in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        final_df[c] = original[c]

    # Preserve exact column order if possible
    final_df = final_df[original.columns]

    try:
        final_df.to_csv(args.output, index=False)
    except Exception:
        # Never fail hard
        original.to_csv(args.output, index=False)

    rows_equal = int((final_df["Timestamp"] == original["Timestamp"]).sum())
    print(f"exchanged_segments={total_exchanged} displaced_segments={total_displaced} rows_still_equal_to_input={rows_equal}")

if __name__ == "__main__":
    main()