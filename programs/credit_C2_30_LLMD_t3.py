import argparse
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

# ============================================================
# Fixed process knowledge from prompt
# ============================================================

DURATION_PROFILES = {
    ("Request info", "info received"): {
        "p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0, "n": 6634
    },
    ("info received", "Check for completeness"): {
        "p10": 62.6256, "p25": 174.0865, "median": 415.276, "share_under_60s": 0.096, "n": 6607
    },
    ("Check for completeness", "Request info"): {
        "p10": 61.3589, "p25": 170.3105, "median": 417.4255, "share_under_60s": 0.099, "n": 6530
    },
    ("Check for completeness", "Perform checks"): {
        "p10": 58.2756, "p25": 166.811, "median": 414.721, "share_under_60s": 0.103, "n": 6017
    },
    ("Perform checks", "Make decision"): {
        "p10": 63.6329, "p25": 171.37975, "median": 423.057, "share_under_60s": 0.094, "n": 5732
    },
    ("New online application received", "Check for completeness"): {
        "p10": 66.159, "p25": 172.4395, "median": 421.0585, "share_under_60s": 0.091, "n": 5000
    },
    ("Deliver card", "EVENT 13 END"): {
        "p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 3276
    },
    ("Notify accept", "Deliver card"): {
        "p10": 59.4585, "p25": 176.35075, "median": 425.762, "share_under_60s": 0.101, "n": 3276
    },
    ("Make decision", "notify reject"): {
        "p10": 64.22, "p25": 175.774, "median": 421.641, "share_under_60s": 0.095, "n": 3161
    },
    ("Make decision", "Notify accept"): {
        "p10": 65.4006, "p25": 172.952, "median": 426.665, "share_under_60s": 0.091, "n": 3013
    },
    ("time out", "EVENT 13 END"): {
        "p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 1724
    },
    ("notify reject", "time out"): {
        "p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0, "n": 1724
    },
    ("notify reject", "review request received"): {
        "p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0, "n": 1644
    },
    ("review request received", "Check for completeness"): {
        "p10": 74.1377, "p25": 186.74975, "median": 434.7365, "share_under_60s": 0.079, "n": 1640
    },
    ("Check for completeness", "Make decision"): {
        "p10": 336.7619, "p25": 585.2495, "median": 1035.1695, "share_under_60s": 0.0, "n": 480
    },
    ("Perform checks", "Notify accept"): {
        "p10": 365.2176, "p25": 560.366, "median": 925.236, "share_under_60s": 0.0, "n": 263
    },
    ("Check for completeness", "info received"): {
        "p10": 133.281, "p25": 244.0815, "median": 455.3365, "share_under_60s": 0.0, "n": 220
    },
    ("Perform checks", "notify reject"): {
        "p10": 456.3862, "p25": 714.8795, "median": 1130.353, "share_under_60s": 0.0, "n": 207
    },
    ("info received", "Perform checks"): {
        "p10": 378.526, "p25": 613.034, "median": 949.19, "share_under_60s": 0.0, "n": 161
    },
    ("info received", "Request info"): {
        "p10": 436.676, "p25": 612.23, "median": 1057.673, "share_under_60s": 0.0, "n": 86
    },
    ("Make decision", "review request received"): {
        "p10": 126.7211, "p25": 263.86525, "median": 523.334, "share_under_60s": 0.0, "n": 38
    },
    ("review request received", "Perform checks"): {
        "p10": 340.5695, "p25": 510.8235, "median": 881.5615, "share_under_60s": 0.0, "n": 24
    },
    ("review request received", "Request info"): {
        "p10": 464.9404, "p25": 603.8945, "median": 1077.302, "share_under_60s": 0.0, "n": 18
    },
    ("EVENT 13 END", "EVENT 13 END"): {
        "p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 8
    },
}

DFG_FREQ = {
    ("Request info", "info received"): 6634,
    ("info received", "Check for completeness"): 6607,
    ("Check for completeness", "Request info"): 6530,
    ("Check for completeness", "Perform checks"): 6017,
    ("Perform checks", "Make decision"): 5732,
    ("New online application received", "Check for completeness"): 5000,
    ("Notify accept", "Deliver card"): 3276,
    ("Deliver card", "EVENT 13 END"): 3276,
    ("Make decision", "notify reject"): 3161,
    ("Make decision", "Notify accept"): 3013,
    ("notify reject", "time out"): 1724,
    ("time out", "EVENT 13 END"): 1724,
    ("notify reject", "review request received"): 1644,
    ("review request received", "Check for completeness"): 1640,
    ("Check for completeness", "Make decision"): 480,
    ("Perform checks", "Notify accept"): 263,
    ("Check for completeness", "info received"): 220,
    ("Perform checks", "notify reject"): 207,
    ("info received", "Perform checks"): 161,
    ("info received", "Request info"): 86,
    ("Make decision", "review request received"): 38,
    ("review request received", "Perform checks"): 24,
    ("review request received", "Request info"): 18,
    ("EVENT 13 END", "EVENT 13 END"): 8,
}

FREQUENT_VARIANTS = [
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
]

GLOBAL_MEDIAN = 241.935

# ============================================================
# Helpers
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()

def detect_fractional_digits(ts: str) -> int:
    if "." in str(ts):
        return len(str(ts).split(".")[-1])
    return 0

def parse_ts(ts: str):
    return pd.to_datetime(ts, errors="coerce")

def get_case_bounds(parsed_series):
    valid = parsed_series.dropna()
    if len(valid) == 0:
        return pd.NaT, pd.NaT
    return valid.min(), valid.max()

def get_profile(a, b):
    return DURATION_PROFILES.get((a, b))

def get_edge_freq(a, b):
    return DFG_FREQ.get((a, b), 0)

def candidate_duration_seconds(profile):
    if profile is None:
        return GLOBAL_MEDIAN
    return profile["p25"]

def min_positive_step_from_digits(digits):
    if digits <= 0:
        return pd.Timedelta(seconds=1)
    return pd.Timedelta(seconds=10 ** (-digits))

def variant_subsequence_score(activities):
    best = 0
    for var in FREQUENT_VARIANTS:
        m = len(activities)
        for i in range(len(var) - m + 1):
            if var[i:i+m] == activities:
                best = max(best, 1)
    return best

def duration_fit_score(obs_gap, profile):
    if profile is None:
        target = GLOBAL_MEDIAN
        scale = max(60.0, GLOBAL_MEDIAN)
        return abs(obs_gap - target) / scale + 1.0
    target = profile["p25"]
    scale = max(10.0, profile["median"] if profile["median"] > 0 else 10.0)
    penalty = abs(obs_gap - target) / scale
    if profile["share_under_60s"] == 1.0 and obs_gap <= 60.0:
        penalty *= 0.5
    return penalty

def local_order_score(prev_act, a1, a2, next_act, prev_t, t1, t2, next_t):
    if pd.isna(t1) or pd.isna(t2):
        return -1e18
    if t1 > t2:
        return -1e18
    score = 0.0

    edges = []
    if prev_act is not None:
        edges.append((prev_act, a1))
    edges.append((a1, a2))
    if next_act is not None:
        edges.append((a2, next_act))

    for e in edges:
        freq = get_edge_freq(*e)
        if freq > 0:
            score += 1000.0 + np.log1p(freq)
        else:
            score -= 10000.0

    acts = []
    if prev_act is not None:
        acts.append(prev_act)
    acts.extend([a1, a2])
    if next_act is not None:
        acts.append(next_act)
    score += 500.0 * variant_subsequence_score(acts)

    if prev_t is not None and not pd.isna(prev_t):
        gap = (t1 - prev_t).total_seconds()
        if gap < 0:
            score -= 1e7
        else:
            score -= duration_fit_score(gap, get_profile(prev_act, a1))
    gap = (t2 - t1).total_seconds()
    if gap < 0:
        score -= 1e7
    else:
        score -= duration_fit_score(gap, get_profile(a1, a2))
    if next_t is not None and not pd.isna(next_t):
        gap = (next_t - t2).total_seconds()
        if gap < 0:
            score -= 1e7
        else:
            score -= duration_fit_score(gap, get_profile(a2, next_act))

    return score

def quantize_timestamp(ts, digits):
    if pd.isna(ts):
        return ts
    if digits <= 0:
        return ts.floor("S")
    unit_ns = 10 ** (9 - digits)
    value = ts.value
    q = int(Decimal(value / unit_ns).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return pd.Timestamp(q * unit_ns)

def format_timestamp(ts, digits):
    ts = quantize_timestamp(ts, digits)
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if digits <= 0:
        return base
    frac = f"{ts.nanosecond + ts.microsecond * 1000:09d}"[:digits]
    return f"{base}.{frac}"

def nearest_changed_time(candidate, original_parsed, lower, upper, digits):
    step = min_positive_step_from_digits(digits)
    candidate = max(candidate, lower)
    candidate = min(candidate, upper)
    candidate = quantize_timestamp(candidate, digits)

    if not pd.isna(original_parsed) and candidate == original_parsed:
        plus = quantize_timestamp(candidate + step, digits)
        minus = quantize_timestamp(candidate - step, digits)
        feasible = []
        if plus <= upper:
            feasible.append(plus)
        if minus >= lower:
            feasible.append(minus)
        if feasible:
            feasible.sort(key=lambda x: abs((x - candidate).total_seconds()))
            candidate = feasible[0]
        else:
            candidate = quantize_timestamp(lower if lower != candidate else upper, digits)
    return candidate

def estimate_displaced_time(prev_act, cur_act, next_act, prev_t, cur_obs, next_t, lower, upper, digits):
    candidates = []

    if prev_t is not None and not pd.isna(prev_t):
        prof = get_profile(prev_act, cur_act)
        sec = candidate_duration_seconds(prof)
        candidates.append(prev_t + pd.to_timedelta(sec, unit="s"))

    if next_t is not None and not pd.isna(next_t):
        prof = get_profile(cur_act, next_act)
        sec = candidate_duration_seconds(prof)
        candidates.append(next_t - pd.to_timedelta(sec, unit="s"))

    if prev_t is not None and not pd.isna(prev_t) and next_t is not None and not pd.isna(next_t):
        if get_profile(prev_act, cur_act) is not None and get_profile(cur_act, next_act) is not None:
            sec1 = candidate_duration_seconds(get_profile(prev_act, cur_act))
            sec2 = candidate_duration_seconds(get_profile(cur_act, next_act))
            total = sec1 + sec2
            if total > 0:
                alpha = sec1 / total
                candidates.append(prev_t + (next_t - prev_t) * alpha)

    if len(candidates) == 0:
        if prev_t is not None and not pd.isna(prev_t):
            candidates.append(prev_t + pd.to_timedelta(GLOBAL_MEDIAN, unit="s"))
        elif next_t is not None and not pd.isna(next_t):
            candidates.append(next_t - pd.to_timedelta(GLOBAL_MEDIAN, unit="s"))
        else:
            candidates.append(cur_obs if not pd.isna(cur_obs) else lower)

    obs = cur_obs if not pd.isna(cur_obs) else candidates[0]
    candidates = [max(lower, min(upper, c)) for c in candidates]
    candidates.sort(key=lambda c: abs((c - obs).total_seconds()))
    chosen = candidates[0]
    chosen = nearest_changed_time(chosen, cur_obs, lower, upper, digits)
    return chosen

# ============================================================
# Main repair logic
# ============================================================

def repair_case(g_case):
    g_case = g_case.copy()
    idxs = list(g_case.index)
    acts = g_case["Activity"].tolist()
    errs = g_case["error_category"].tolist()
    orig_strs = g_case["Timestamp"].tolist()
    parsed = [parse_ts(x) for x in orig_strs]
    digits = [detect_fractional_digits(x) for x in orig_strs]

    out_strs = orig_strs[:]
    out_parsed = parsed[:]

    case_start, case_end = get_case_bounds(pd.Series(parsed))

    exchanged_segments = 0
    displaced_segments = 0
    diagnostics = []

    i = 1
    while i < len(g_case):
        if out_parsed[i] < out_parsed[i - 1]:
            left_idx = i - 1
            right_idx = i

            prev_idx = left_idx - 1 if left_idx - 1 >= 0 else None
            next_idx = right_idx + 1 if right_idx + 1 < len(g_case) else None

            prev_act = acts[prev_idx] if prev_idx is not None else None
            left_act = acts[left_idx]
            right_act = acts[right_idx]
            next_act = acts[next_idx] if next_idx is not None else None

            prev_t = out_parsed[prev_idx] if prev_idx is not None else None
            left_t = out_parsed[left_idx]
            right_t = out_parsed[right_idx]
            next_t = out_parsed[next_idx] if next_idx is not None else None

            left_err = errs[left_idx] != "CLEAN"
            right_err = errs[right_idx] != "CLEAN"

            # Candidate 1: exact exchange only when both rows are corrupted
            swap_valid = False
            swap_score = -1e18
            if left_err and right_err:
                cand_left = right_t
                cand_right = left_t
                if ((prev_t is None or cand_left >= prev_t) and
                    cand_left <= cand_right and
                    (next_t is None or cand_right <= next_t)):
                    swap_valid = True
                    swap_score = local_order_score(
                        prev_act, left_act, right_act, next_act,
                        prev_t, cand_left, cand_right, next_t
                    )

            # Candidate 2: displaced later row backwards (most common inversion)
            disp_valid = right_err
            disp_score = -1e18
            est_right = None
            if disp_valid:
                lower = max(case_start, left_t) if not pd.isna(case_start) else left_t
                upper = min(case_end, next_t) if (next_t is not None and not pd.isna(next_t) and not pd.isna(case_end)) else (next_t if next_t is not None and not pd.isna(next_t) else case_end)
                if pd.isna(upper):
                    upper = case_end
                if pd.isna(lower):
                    lower = left_t
                if pd.isna(upper):
                    upper = max(lower, left_t)
                step = min_positive_step_from_digits(digits[right_idx])
                lower = max(lower, left_t)
                if upper < lower:
                    upper = lower
                if upper == lower and upper == left_t:
                    upper = upper + step
                    if not pd.isna(case_end):
                        upper = min(upper, case_end)
                est_right = estimate_displaced_time(
                    left_act, right_act, next_act,
                    left_t, right_t, next_t,
                    lower, upper, digits[right_idx]
                )
                if est_right < left_t:
                    est_right = nearest_changed_time(left_t, right_t, left_t, upper, digits[right_idx])
                if next_t is not None and est_right > next_t:
                    est_right = nearest_changed_time(next_t, right_t, left_t, next_t, digits[right_idx])
                disp_score = local_order_score(
                    prev_act, left_act, right_act, next_act,
                    prev_t, left_t, est_right, next_t
                )
                # minimal movement preference
                disp_score -= abs((est_right - right_t).total_seconds()) / 1000.0

            choose_swap = False
            if swap_valid:
                if not disp_valid:
                    choose_swap = True
                else:
                    move_swap = abs((right_t - left_t).total_seconds()) * 2.0
                    move_disp = abs((est_right - right_t).total_seconds()) if est_right is not None else 1e18
                    # prefer displaced unless swap has clearly better local evidence
                    if (swap_score > disp_score + 50.0) or (
                        abs(swap_score - disp_score) <= 50.0 and move_swap <= move_disp * 0.5
                    ):
                        choose_swap = True

            if choose_swap:
                out_parsed[left_idx] = right_t
                out_parsed[right_idx] = left_t
                out_strs[left_idx] = orig_strs[right_idx]
                out_strs[right_idx] = orig_strs[left_idx]
                exchanged_segments += 1
                diagnostics.append(f"swap_case={g_case['Case'].iloc[0]} rows=({g_case['row_id'].iloc[left_idx]},{g_case['row_id'].iloc[right_idx]})")
            else:
                if right_err and est_right is not None:
                    out_parsed[right_idx] = est_right
                    out_strs[right_idx] = format_timestamp(est_right, digits[right_idx])
                    displaced_segments += 1
                    diagnostics.append(f"displaced_case={g_case['Case'].iloc[0]} row={g_case['row_id'].iloc[right_idx]}")
                elif left_err:
                    # rare fallback: if right is clean and left corrupted, clamp left just before right
                    upper = min(case_end, right_t) if not pd.isna(case_end) else right_t
                    lower = max(case_start, prev_t) if prev_t is not None and not pd.isna(prev_t) and not pd.isna(case_start) else (prev_t if prev_t is not None and not pd.isna(prev_t) else case_start)
                    if pd.isna(lower):
                        lower = upper
                    if pd.isna(upper):
                        upper = lower
                    est_left = estimate_displaced_time(
                        prev_act, left_act, right_act,
                        prev_t, left_t, right_t,
                        lower, upper, digits[left_idx]
                    )
                    if est_left > right_t:
                        est_left = nearest_changed_time(right_t, left_t, lower, right_t, digits[left_idx])
                    out_parsed[left_idx] = est_left
                    out_strs[left_idx] = format_timestamp(est_left, digits[left_idx])
                    displaced_segments += 1
                    diagnostics.append(f"displaced_case={g_case['Case'].iloc[0]} row={g_case['row_id'].iloc[left_idx]}")
            i = max(i, 1)
        else:
            i += 1

    # Self-check and fix pass
    for j in range(len(g_case)):
        if errs[j] == "CLEAN":
            out_strs[j] = orig_strs[j]
            out_parsed[j] = parsed[j]

    # enforce every non-clean row changed
    for j in range(len(g_case)):
        if errs[j] != "CLEAN" and out_strs[j] == orig_strs[j]:
            prev_t = out_parsed[j - 1] if j - 1 >= 0 else case_start
            next_t = out_parsed[j + 1] if j + 1 < len(g_case) else case_end
            lower = case_start if pd.notna(case_start) else prev_t
            upper = case_end if pd.notna(case_end) else next_t
            if j - 1 >= 0 and pd.notna(prev_t):
                lower = max(lower, prev_t)
            if j + 1 < len(g_case) and pd.notna(next_t):
                upper = min(upper, next_t)
            if pd.isna(lower):
                lower = parsed[j]
            if pd.isna(upper):
                upper = parsed[j]
            if upper < lower:
                upper = lower
            repaired = estimate_displaced_time(
                acts[j - 1] if j - 1 >= 0 else None,
                acts[j],
                acts[j + 1] if j + 1 < len(g_case) else None,
                prev_t if j - 1 >= 0 else None,
                parsed[j],
                next_t if j + 1 < len(g_case) else None,
                lower, upper, digits[j]
            )
            repaired = nearest_changed_time(repaired, parsed[j], lower, upper, digits[j])
            out_parsed[j] = repaired
            out_strs[j] = format_timestamp(repaired, digits[j])
            diagnostics.append(f"forced_change_case={g_case['Case'].iloc[0]} row={g_case['row_id'].iloc[j]}")

    # monotonic clamp within case
    for j in range(1, len(g_case)):
        if out_parsed[j] < out_parsed[j - 1]:
            if errs[j] != "CLEAN":
                lower = out_parsed[j - 1]
                upper = out_parsed[j + 1] if j + 1 < len(g_case) else case_end
                if pd.isna(upper):
                    upper = lower
                fixed = nearest_changed_time(max(out_parsed[j], lower), parsed[j], lower, upper, digits[j])
                if fixed < lower:
                    fixed = lower
                out_parsed[j] = fixed
                out_strs[j] = format_timestamp(fixed, digits[j])
                diagnostics.append(f"monotonic_fix_case={g_case['Case'].iloc[0]} row={g_case['row_id'].iloc[j]}")
            elif errs[j - 1] != "CLEAN":
                upper = out_parsed[j]
                lower = out_parsed[j - 2] if j - 2 >= 0 else case_start
                if pd.isna(lower):
                    lower = upper
                fixed = nearest_changed_time(min(out_parsed[j - 1], upper), parsed[j - 1], lower, upper, digits[j - 1])
                if fixed > upper:
                    fixed = upper
                out_parsed[j - 1] = fixed
                out_strs[j - 1] = format_timestamp(fixed, digits[j - 1])
                diagnostics.append(f"monotonic_fix_case={g_case['Case'].iloc[0]} row={g_case['row_id'].iloc[j - 1]}")

    # final bounds clamp
    for j in range(len(g_case)):
        if errs[j] != "CLEAN":
            lower = case_start
            upper = case_end
            if j - 1 >= 0 and errs[j - 1] == "CLEAN":
                lower = max(lower, parsed[j - 1]) if pd.notna(lower) else parsed[j - 1]
            elif j - 1 >= 0:
                lower = max(lower, out_parsed[j - 1]) if pd.notna(lower) else out_parsed[j - 1]
            if j + 1 < len(g_case) and errs[j + 1] == "CLEAN":
                upper = min(upper, parsed[j + 1]) if pd.notna(upper) else parsed[j + 1]
            elif j + 1 < len(g_case):
                upper = min(upper, out_parsed[j + 1]) if pd.notna(upper) else out_parsed[j + 1]
            if pd.isna(lower):
                lower = out_parsed[j]
            if pd.isna(upper):
                upper = out_parsed[j]
            if upper < lower:
                upper = lower
            fixed = max(lower, min(upper, out_parsed[j]))
            fixed = nearest_changed_time(fixed, parsed[j], lower, upper, digits[j])
            out_parsed[j] = fixed
            out_strs[j] = format_timestamp(fixed, digits[j])

    g_case["Timestamp"] = out_strs
    return g_case, exchanged_segments, displaced_segments, diagnostics

def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        # fail soft
        df = pd.read_csv(args.input, dtype=str)

    original_df = df.copy(deep=True)

    repaired_parts = []
    exchanged_total = 0
    displaced_total = 0
    diagnostics = []

    for _, g in df.groupby("Case", sort=False):
        repaired_case, ex_cnt, dis_cnt, diags = repair_case(g)
        repaired_parts.append(repaired_case)
        exchanged_total += ex_cnt
        displaced_total += dis_cnt
        diagnostics.extend(diags)

    repaired = pd.concat(repaired_parts).sort_index()

    # Global self-checks and non-abort corrections
    if len(repaired) != len(original_df):
        repaired = repaired.reindex(original_df.index)
        for col in original_df.columns:
            if col != "Timestamp":
                repaired[col] = original_df[col]

    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        repaired[col] = original_df[col]

    repaired = repaired[original_df.columns]

    # CLEAN rows unchanged exactly
    clean_mask = original_df["error_category"] == "CLEAN"
    repaired.loc[clean_mask, "Timestamp"] = original_df.loc[clean_mask, "Timestamp"]

    # parseability rescue for repaired rows
    non_clean_mask = ~clean_mask
    for idx in repaired.index[non_clean_mask]:
        ts = repaired.at[idx, "Timestamp"]
        if pd.isna(parse_ts(ts)):
            # re-estimate using local same-case anchors, continue
            case = repaired.at[idx, "Case"]
            case_idx = repaired.index[repaired["Case"] == case].tolist()
            pos = case_idx.index(idx)
            case_slice = repaired.loc[case_idx]
            orig_case_slice = original_df.loc[case_idx]
            digits = detect_fractional_digits(orig_case_slice.iloc[pos]["Timestamp"])
            parsed_case = [parse_ts(x) for x in case_slice["Timestamp"].tolist()]
            case_start, case_end = get_case_bounds(pd.Series(parsed_case))
            prev_t = parsed_case[pos - 1] if pos - 1 >= 0 else case_start
            next_t = parsed_case[pos + 1] if pos + 1 < len(parsed_case) else case_end
            lower = prev_t if pd.notna(prev_t) else case_start
            upper = next_t if pd.notna(next_t) else case_end
            if pd.isna(lower):
                lower = parse_ts(orig_case_slice.iloc[pos]["Timestamp"])
            if pd.isna(upper):
                upper = lower
            if upper < lower:
                upper = lower
            new_t = nearest_changed_time(lower, parse_ts(orig_case_slice.iloc[pos]["Timestamp"]), lower, upper, digits)
            repaired.at[idx, "Timestamp"] = format_timestamp(new_t, digits)
            diagnostics.append(f"parse_fix_row={repaired.at[idx, 'row_id']}")

    rows_still_equal = int((repaired["Timestamp"].astype(str) == original_df["Timestamp"].astype(str)).sum())

    repaired.to_csv(args.output, index=False)

    print(f"exchanged_segments={exchanged_total} displaced_segments={displaced_total} rows_still_equal={rows_still_equal}")
    for d in diagnostics[:200]:
        print(d)

if __name__ == "__main__":
    main()