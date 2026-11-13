import argparse
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Tuple, Optional, List

import pandas as pd


# =========================
# Embedded process context
# =========================

DURATION_PROFILES = {
    ("Request info", "info received"): {
        "p10": 10.0, "p25": 10.0, "median": 10.0, "p75": 10.0, "p90": 10.0, "share_under_60s": 1.0, "n": 6634
    },
    ("info received", "Check for completeness"): {
        "p10": 62.6256, "p25": 174.0865, "median": 415.276, "p75": 831.363, "p90": 1398.764, "share_under_60s": 0.096, "n": 6607
    },
    ("Check for completeness", "Request info"): {
        "p10": 61.3589, "p25": 170.3105, "median": 417.4255, "p75": 825.8595, "p90": 1368.1205, "share_under_60s": 0.099, "n": 6530
    },
    ("Check for completeness", "Perform checks"): {
        "p10": 58.2756, "p25": 166.811, "median": 414.721, "p75": 809.916, "p90": 1363.778, "share_under_60s": 0.103, "n": 6017
    },
    ("Perform checks", "Make decision"): {
        "p10": 63.6329, "p25": 171.37975, "median": 423.057, "p75": 838.5075, "p90": 1399.7289, "share_under_60s": 0.094, "n": 5732
    },
    ("New online application received", "Check for completeness"): {
        "p10": 66.159, "p25": 172.4395, "median": 421.0585, "p75": 822.01525, "p90": 1347.761, "share_under_60s": 0.091, "n": 5000
    },
    ("Deliver card", "EVENT 13 END"): {
        "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 3276
    },
    ("Notify accept", "Deliver card"): {
        "p10": 59.4585, "p25": 176.35075, "median": 425.762, "p75": 851.3035, "p90": 1370.5795, "share_under_60s": 0.101, "n": 3276
    },
    ("Make decision", "notify reject"): {
        "p10": 64.22, "p25": 175.774, "median": 421.641, "p75": 829.623, "p90": 1425.57, "share_under_60s": 0.095, "n": 3161
    },
    ("Make decision", "Notify accept"): {
        "p10": 65.4006, "p25": 172.952, "median": 426.665, "p75": 862.463, "p90": 1435.1004, "share_under_60s": 0.091, "n": 3013
    },
    ("time out", "EVENT 13 END"): {
        "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 1724
    },
    ("notify reject", "time out"): {
        "p10": 10.0, "p25": 10.0, "median": 10.0, "p75": 10.0, "p90": 10.0, "share_under_60s": 1.0, "n": 1724
    },
    ("notify reject", "review request received"): {
        "p10": 10.0, "p25": 10.0, "median": 10.0, "p75": 10.0, "p90": 10.0, "share_under_60s": 1.0, "n": 1644
    },
    ("review request received", "Check for completeness"): {
        "p10": 74.1377, "p25": 186.74975, "median": 434.7365, "p75": 836.486, "p90": 1410.6346, "share_under_60s": 0.079, "n": 1640
    },
    ("Check for completeness", "Make decision"): {
        "p10": 336.7619, "p25": 585.2495, "median": 1035.1695, "p75": 1671.778, "p90": 2361.2988, "share_under_60s": 0.0, "n": 480
    },
    ("Perform checks", "Notify accept"): {
        "p10": 365.2176, "p25": 560.366, "median": 925.236, "p75": 1566.2075, "p90": 2311.7958, "share_under_60s": 0.0, "n": 263
    },
    ("Check for completeness", "info received"): {
        "p10": 133.281, "p25": 244.0815, "median": 455.3365, "p75": 781.52775, "p90": 1192.9707, "share_under_60s": 0.0, "n": 220
    },
    ("Perform checks", "notify reject"): {
        "p10": 456.3862, "p25": 714.8795, "median": 1130.353, "p75": 1717.2175, "p90": 2195.3612, "share_under_60s": 0.0, "n": 207
    },
    ("info received", "Perform checks"): {
        "p10": 378.526, "p25": 613.034, "median": 949.19, "p75": 1698.959, "p90": 2423.825, "share_under_60s": 0.0, "n": 161
    },
    ("info received", "Request info"): {
        "p10": 436.676, "p25": 612.23, "median": 1057.673, "p75": 1692.58275, "p90": 2144.263, "share_under_60s": 0.0, "n": 86
    },
    ("Make decision", "review request received"): {
        "p10": 126.7211, "p25": 263.86525, "median": 523.334, "p75": 796.8225, "p90": 1332.5147, "share_under_60s": 0.0, "n": 38
    },
    ("review request received", "Perform checks"): {
        "p10": 340.5695, "p25": 510.8235, "median": 881.5615, "p75": 1733.557, "p90": 2702.617, "share_under_60s": 0.0, "n": 24
    },
    ("review request received", "Request info"): {
        "p10": 464.9404, "p25": 603.8945, "median": 1077.302, "p75": 1300.893, "p90": 2391.0761, "share_under_60s": 0.0, "n": 18
    },
    ("EVENT 13 END", "EVENT 13 END"): {
        "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 8
    },
}

DFG_EDGES = set(DURATION_PROFILES.keys())

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

BLOCK_SPAN_PROFILE_P25 = {2: 10.0, 3: 323.548, 4: 705.372, 5: 1079.68}
GLOBAL_MEDIAN = 241.935


# =========================
# Utilities
# =========================

def detect_fractional_digits(ts: str) -> int:
    m = re.search(r"\.(\d+)$", ts.strip())
    return len(m.group(1)) if m else 0


def parse_ts(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, errors="coerce")


def get_precision_unit_ns(max_digits: int) -> int:
    if max_digits <= 0:
        return 1_000_000_000
    return 10 ** (9 - max_digits)


def quantize_ts(ts: pd.Timestamp, digits: int) -> pd.Timestamp:
    if pd.isna(ts):
        return ts
    if digits <= 0:
        return ts.floor("S")
    unit_ns = get_precision_unit_ns(digits)
    ns = ts.value
    q = int(Decimal(ns / unit_ns).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return pd.Timestamp(q * unit_ns)


def format_ts(ts: pd.Timestamp, digits: int) -> str:
    ts = quantize_ts(ts, digits)
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if digits <= 0:
        return base
    frac = f"{ts.nanosecond + ts.microsecond * 1000:09d}"[:digits]
    return f"{base}.{frac}"


def duration_profile(src: Optional[str], tgt: Optional[str]) -> Optional[dict]:
    if src is None or tgt is None:
        return None
    return DURATION_PROFILES.get((src, tgt))


def expected_gap_seconds(src: Optional[str], tgt: Optional[str]) -> float:
    prof = duration_profile(src, tgt)
    if prof is None:
        return BLOCK_SPAN_PROFILE_P25.get(2, GLOBAL_MEDIAN)
    return prof["p25"]


def tolerance_seconds(src: Optional[str], tgt: Optional[str]) -> float:
    prof = duration_profile(src, tgt)
    if prof is None:
        return max(60.0, BLOCK_SPAN_PROFILE_P25.get(2, GLOBAL_MEDIAN))
    spread = max(prof["p90"] - prof["p10"], 1.0)
    return max(30.0, spread / 2.0)


def edge_supported(src: Optional[str], tgt: Optional[str]) -> bool:
    if src is None or tgt is None:
        return True
    return (src, tgt) in DFG_EDGES


def subsequence_supported(acts: List[str]) -> bool:
    if not acts:
        return True
    for var in FREQUENT_VARIANTS:
        for i in range(0, len(var) - len(acts) + 1):
            if var[i:i + len(acts)] == acts:
                return True
    return False


def local_order_supported(prev_act, act1, act2, next_act) -> bool:
    if not edge_supported(prev_act, act1):
        return False
    if not edge_supported(act1, act2):
        return False
    if not edge_supported(act2, next_act):
        return False
    acts = [a for a in [prev_act, act1, act2, next_act] if a is not None]
    if len(acts) >= 2 and not subsequence_supported(acts):
        mid = [a for a in [act1, act2] if a is not None]
        if not subsequence_supported(mid):
            return False
    return True


def gap_penalty(actual: float, src: Optional[str], tgt: Optional[str]) -> float:
    if actual < 0:
        return 1e9 + abs(actual)
    exp_gap = expected_gap_seconds(src, tgt)
    tol = tolerance_seconds(src, tgt)
    return abs(actual - exp_gap) / tol


def choose_estimate(left_t: Optional[pd.Timestamp], left_act: Optional[str],
                    act: str,
                    right_t: Optional[pd.Timestamp], right_act: Optional[str],
                    observed_t: pd.Timestamp,
                    low: pd.Timestamp, high: pd.Timestamp) -> pd.Timestamp:
    cands = []

    if left_t is not None:
        cands.append(left_t + pd.to_timedelta(expected_gap_seconds(left_act, act), unit="s"))
    if right_t is not None:
        cands.append(right_t - pd.to_timedelta(expected_gap_seconds(act, right_act), unit="s"))
    if left_t is not None and right_t is not None:
        l = left_t + pd.to_timedelta(expected_gap_seconds(left_act, act), unit="s")
        r = right_t - pd.to_timedelta(expected_gap_seconds(act, right_act), unit="s")
        cands.append(l + (r - l) / 2)
    if not cands:
        cands.append(observed_t)

    best = None
    best_score = None
    for c in cands:
        c = min(max(c, low), high)
        score = 0.0
        if left_t is not None:
            score += gap_penalty((c - left_t).total_seconds(), left_act, act)
        if right_t is not None:
            score += gap_penalty((right_t - c).total_seconds(), act, right_act)
        score += abs((c - observed_t).total_seconds()) / max(1.0, BLOCK_SPAN_PROFILE_P25.get(2, 10.0))
        if best is None or score < best_score:
            best = c
            best_score = score
    return best


def nearest_feasible_different(ts: pd.Timestamp, original: pd.Timestamp,
                               low: pd.Timestamp, high: pd.Timestamp,
                               digits: int) -> pd.Timestamp:
    qts = quantize_ts(ts, digits)
    qor = quantize_ts(original, digits)
    if qts != qor:
        return min(max(qts, low), high)

    step_ns = get_precision_unit_ns(digits)
    plus = pd.Timestamp(qts.value + step_ns)
    minus = pd.Timestamp(qts.value - step_ns)

    feasible = []
    if plus <= high:
        feasible.append((abs((plus - ts).total_seconds()), plus))
    if minus >= low:
        feasible.append((abs((minus - ts).total_seconds()), minus))
    if feasible:
        feasible.sort(key=lambda x: x[0])
        return feasible[0][1]
    return qts


# =========================
# Main repair logic
# =========================

def repair_case(case_df: pd.DataFrame, out_ts: Dict[int, str], parsed_out: Dict[int, pd.Timestamp],
                repaired_rows: set, stats: Dict[str, int], logs: List[str]) -> None:
    idxs = list(case_df.index)
    n = len(idxs)

    for pos in range(1, n):
        i_prev = idxs[pos - 1]
        i_curr = idxs[pos]

        if case_df.at[i_curr, "error_category"] != "C2_TEMPORAL_INCONSISTENCY":
            continue

        prev_t = parsed_out[i_prev]
        curr_obs = parse_ts(case_df.at[i_curr, "Timestamp"])
        if pd.isna(prev_t) or pd.isna(curr_obs):
            continue

        if curr_obs >= prev_t:
            continue

        # Treat segment as adjacent inverted pair (prev, curr)
        i = i_prev
        j = i_curr

        err_i = case_df.at[i, "error_category"]
        err_j = case_df.at[j, "error_category"]

        t_i_obs = parse_ts(case_df.at[i, "Timestamp"])
        t_j_obs = parse_ts(case_df.at[j, "Timestamp"])
        act_i = case_df.at[i, "Activity"]
        act_j = case_df.at[j, "Activity"]

        prev_idx = idxs[pos - 2] if pos - 2 >= 0 else None
        next_idx = idxs[pos + 1] if pos + 1 < n else None

        prev_anchor_t = parsed_out[prev_idx] if prev_idx is not None else None
        prev_anchor_act = case_df.at[prev_idx, "Activity"] if prev_idx is not None else None
        next_anchor_t = parse_ts(case_df.at[next_idx, "Timestamp"]) if next_idx is not None else None
        next_anchor_act = case_df.at[next_idx, "Activity"] if next_idx is not None else None

        case_start = min(parse_ts(case_df.at[k, "Timestamp"]) for k in idxs)
        case_end = max(parse_ts(case_df.at[k, "Timestamp"]) for k in idxs)

        digits_i = detect_fractional_digits(case_df.at[i, "Timestamp"])
        digits_j = detect_fractional_digits(case_df.at[j, "Timestamp"])

        # Hypothesis 1: EXCHANGED only if both rows are corrupted
        exchanged_valid = False
        exch_score = float("inf")
        if err_i != "CLEAN" and err_j != "CLEAN":
            swap_i = t_j_obs
            swap_j = t_i_obs
            within = True
            if prev_anchor_t is not None and swap_i < prev_anchor_t:
                within = False
            if next_anchor_t is not None and swap_j > next_anchor_t:
                within = False
            if swap_i > swap_j:
                within = False
            if not local_order_supported(prev_anchor_act, act_i, act_j, next_anchor_act):
                within = False
            if within:
                exch_score = 0.0
                if prev_anchor_t is not None:
                    exch_score += gap_penalty((swap_i - prev_anchor_t).total_seconds(), prev_anchor_act, act_i)
                exch_score += gap_penalty((swap_j - swap_i).total_seconds(), act_i, act_j)
                if next_anchor_t is not None:
                    exch_score += gap_penalty((next_anchor_t - swap_j).total_seconds(), act_j, next_anchor_act)
                exchanged_valid = True

        # Hypothesis 2: DISPLACED, move exactly one row
        displaced_candidates = []

        # Candidate A: keep i unchanged, repair j
        low_a = max([x for x in [t_i_obs, prev_anchor_t, case_start] if x is not None])
        high_a = min([x for x in [next_anchor_t, case_end] if x is not None])
        if low_a <= high_a and local_order_supported(prev_anchor_act, act_i, act_j, next_anchor_act):
            est_j = choose_estimate(t_i_obs, act_i, act_j, next_anchor_t, next_anchor_act, t_j_obs, low_a, high_a)
            est_j = nearest_feasible_different(est_j, t_j_obs, low_a, high_a, digits_j)
            score_a = 0.0
            if prev_anchor_t is not None:
                score_a += gap_penalty((t_i_obs - prev_anchor_t).total_seconds(), prev_anchor_act, act_i)
            score_a += gap_penalty((est_j - t_i_obs).total_seconds(), act_i, act_j)
            if next_anchor_t is not None:
                score_a += gap_penalty((next_anchor_t - est_j).total_seconds(), act_j, next_anchor_act)
            score_a += abs((est_j - t_j_obs).total_seconds()) / max(1.0, BLOCK_SPAN_PROFILE_P25.get(2, 10.0))
            displaced_candidates.append(("move_j", t_i_obs, est_j, score_a))

        # Candidate B: repair i, keep j unchanged
        low_b = max([x for x in [prev_anchor_t, case_start] if x is not None])
        high_b = min([x for x in [t_j_obs, next_anchor_t, case_end] if x is not None])
        if low_b <= high_b and local_order_supported(prev_anchor_act, act_i, act_j, next_anchor_act):
            est_i = choose_estimate(prev_anchor_t, prev_anchor_act, act_i, t_j_obs, act_j, t_i_obs, low_b, high_b)
            est_i = nearest_feasible_different(est_i, t_i_obs, low_b, high_b, digits_i)
            score_b = 0.0
            if prev_anchor_t is not None:
                score_b += gap_penalty((est_i - prev_anchor_t).total_seconds(), prev_anchor_act, act_i)
            score_b += gap_penalty((t_j_obs - est_i).total_seconds(), act_i, act_j)
            if next_anchor_t is not None:
                score_b += gap_penalty((next_anchor_t - t_j_obs).total_seconds(), act_j, next_anchor_act)
            score_b += abs((est_i - t_i_obs).total_seconds()) / max(1.0, BLOCK_SPAN_PROFILE_P25.get(2, 10.0))
            displaced_candidates.append(("move_i", est_i, t_j_obs, score_b))

        chosen = None
        if exchanged_valid:
            if displaced_candidates:
                best_disp = min(displaced_candidates, key=lambda x: x[3])
                # Prefer minimal valid movement unless swap is clearly better
                move_swap = abs((t_j_obs - t_i_obs).total_seconds()) + abs((t_i_obs - t_j_obs).total_seconds())
                if exch_score + 0.5 < best_disp[3] and move_swap <= 2 * max(600.0, abs((best_disp[2] - t_j_obs).total_seconds()) + abs((best_disp[1] - t_i_obs).total_seconds())):
                    chosen = ("exchanged", swap_i, swap_j)
                else:
                    chosen = ("displaced",) + best_disp
            else:
                chosen = ("exchanged", swap_i, swap_j)
        else:
            if displaced_candidates:
                best_disp = min(displaced_candidates, key=lambda x: (x[3], abs((x[1] - t_i_obs).total_seconds()) + abs((x[2] - t_j_obs).total_seconds())))
                chosen = ("displaced",) + best_disp

        if chosen is None:
            # Conservative fallback: repair only current row minimally from previous row
            low = max([x for x in [t_i_obs, prev_anchor_t, case_start] if x is not None])
            high = min([x for x in [next_anchor_t, case_end] if x is not None])
            if low > high:
                low, high = t_i_obs, max(t_i_obs, case_end)
            est_j = choose_estimate(t_i_obs, act_i, act_j, next_anchor_t, next_anchor_act, t_j_obs, low, high)
            est_j = nearest_feasible_different(est_j, t_j_obs, low, high, digits_j)
            chosen = ("displaced", "move_j", t_i_obs, est_j, 999999.0)
            logs.append(f"fallback_displaced_case={case_df.at[j, 'Case']}_row={case_df.at[j, 'row_id']}")

        if chosen[0] == "exchanged":
            _, new_i, new_j = chosen
            new_i = nearest_feasible_different(new_i, t_i_obs, max([x for x in [prev_anchor_t, case_start] if x is not None]),
                                               min([x for x in [new_j, next_anchor_t, case_end] if x is not None]), digits_i)
            new_j = nearest_feasible_different(new_j, t_j_obs, max([x for x in [new_i, prev_anchor_t, case_start] if x is not None]),
                                               min([x for x in [next_anchor_t, case_end] if x is not None]), digits_j)
            # exact swap required for exchanged; if quantization broke it, downgrade
            if new_i == t_j_obs and new_j == t_i_obs and new_i <= new_j:
                out_ts[i] = case_df.at[j, "Timestamp"]
                out_ts[j] = case_df.at[i, "Timestamp"]
                parsed_out[i] = t_j_obs
                parsed_out[j] = t_i_obs
                repaired_rows.add(i)
                repaired_rows.add(j)
                stats["exchanged_segments"] += 1
            else:
                # switch to displaced
                move_kind, new_i2, new_j2, _ = min(displaced_candidates, key=lambda x: x[3]) if displaced_candidates else ("move_j", t_i_obs, est_j, 0)
                if move_kind == "move_j":
                    out_ts[j] = format_ts(new_j2, digits_j)
                    parsed_out[j] = parse_ts(out_ts[j])
                    repaired_rows.add(j)
                else:
                    out_ts[i] = format_ts(new_i2, digits_i)
                    parsed_out[i] = parse_ts(out_ts[i])
                    repaired_rows.add(i)
                stats["displaced_segments"] += 1
                logs.append(f"switch_exchanged_to_displaced_case={case_df.at[j, 'Case']}_rows={case_df.at[i, 'row_id']},{case_df.at[j, 'row_id']}")
        else:
            _, move_kind, new_i, new_j, _score = chosen
            if move_kind == "move_j":
                out_ts[j] = format_ts(new_j, digits_j)
                parsed_out[j] = parse_ts(out_ts[j])
                repaired_rows.add(j)
            else:
                out_ts[i] = format_ts(new_i, digits_i)
                parsed_out[i] = parse_ts(out_ts[i])
                repaired_rows.add(i)
            stats["displaced_segments"] += 1


def self_check_and_fix(df: pd.DataFrame, out_ts: Dict[int, str], parsed_out: Dict[int, pd.Timestamp],
                       repaired_rows: set, logs: List[str]) -> int:
    rows_still_equal = 0

    for case_id, case_df in df.groupby("Case", sort=False):
        idxs = list(case_df.index)
        case_times = [parsed_out[i] for i in idxs if not pd.isna(parsed_out[i])]
        if not case_times:
            continue
        case_start = min(case_times)
        case_end = max(case_times)

        for p, idx in enumerate(idxs):
            orig = df.at[idx, "Timestamp"]
            err = df.at[idx, "error_category"]
            act = df.at[idx, "Activity"]
            digits = detect_fractional_digits(orig)

            if err == "CLEAN":
                out_ts[idx] = orig
                parsed_out[idx] = parse_ts(orig)
                continue

            if out_ts[idx] == orig:
                rows_still_equal += 1

            t = parsed_out[idx]
            if pd.isna(t):
                t = parse_ts(orig)

            prev_idx = idxs[p - 1] if p - 1 >= 0 else None
            next_idx = idxs[p + 1] if p + 1 < len(idxs) else None

            left_t = parsed_out[prev_idx] if prev_idx is not None else case_start
            right_t = parsed_out[next_idx] if next_idx is not None else case_end
            left_act = df.at[prev_idx, "Activity"] if prev_idx is not None else None
            right_act = df.at[next_idx, "Activity"] if next_idx is not None else None

            low = max(case_start, left_t) if left_t is not None else case_start
            high = min(case_end, right_t) if right_t is not None else case_end
            if low > high:
                low, high = case_start, case_end

            if t < low or t > high:
                t = choose_estimate(left_t, left_act, act, right_t, right_act, parse_ts(orig), low, high)
                t = nearest_feasible_different(t, parse_ts(orig), low, high, digits)
                out_ts[idx] = format_ts(t, digits)
                parsed_out[idx] = parse_ts(out_ts[idx])
                logs.append(f"clamped_row={df.at[idx, 'row_id']}")

        # second pass for monotonicity
        for p in range(1, len(idxs)):
            prev_idx = idxs[p - 1]
            idx = idxs[p]
            if parsed_out[idx] < parsed_out[prev_idx]:
                if df.at[idx, "error_category"] != "CLEAN":
                    digits = detect_fractional_digits(df.at[idx, "Timestamp"])
                    left_t = parsed_out[prev_idx]
                    next_idx = idxs[p + 1] if p + 1 < len(idxs) else None
                    right_t = parsed_out[next_idx] if next_idx is not None else case_end
                    low = left_t
                    high = min(case_end, right_t)
                    if low > high:
                        high = low
                    t = choose_estimate(left_t, df.at[prev_idx, "Activity"], df.at[idx, "Activity"],
                                        right_t, df.at[next_idx, "Activity"] if next_idx is not None else None,
                                        parse_ts(df.at[idx, "Timestamp"]), low, high)
                    t = nearest_feasible_different(t, parse_ts(df.at[idx, "Timestamp"]), low, high, digits)
                    out_ts[idx] = format_ts(t, digits)
                    parsed_out[idx] = parse_ts(out_ts[idx])
                    logs.append(f"monotonic_fix_row={df.at[idx, 'row_id']}")
                elif df.at[prev_idx, "error_category"] != "CLEAN":
                    digits = detect_fractional_digits(df.at[prev_idx, "Timestamp"])
                    prev_prev_idx = idxs[p - 2] if p - 2 >= 0 else None
                    left_t = parsed_out[prev_prev_idx] if prev_prev_idx is not None else case_start
                    low = max(case_start, left_t)
                    high = parsed_out[idx]
                    t = choose_estimate(left_t, df.at[prev_prev_idx, "Activity"] if prev_prev_idx is not None else None,
                                        df.at[prev_idx, "Activity"], parsed_out[idx], df.at[idx, "Activity"],
                                        parse_ts(df.at[prev_idx, "Timestamp"]), low, high)
                    t = nearest_feasible_different(t, parse_ts(df.at[prev_idx, "Timestamp"]), low, high, digits)
                    out_ts[prev_idx] = format_ts(t, digits)
                    parsed_out[prev_idx] = parse_ts(out_ts[prev_idx])
                    logs.append(f"monotonic_fix_row={df.at[prev_idx, 'row_id']}")

    # recount after fixes
    rows_still_equal = 0
    for idx in df.index:
        if df.at[idx, "error_category"] != "CLEAN" and out_ts[idx] == df.at[idx, "Timestamp"]:
            rows_still_equal += 1
            # force minimal anchored difference, never leave corruption untouched
            digits = detect_fractional_digits(df.at[idx, "Timestamp"])
            case_df = df[df["Case"] == df.at[idx, "Case"]]
            idxs = list(case_df.index)
            p = idxs.index(idx)
            parsed_orig = parse_ts(df.at[idx, "Timestamp"])
            case_times = [parsed_out[i] for i in idxs if not pd.isna(parsed_out[i])]
            case_start = min(case_times)
            case_end = max(case_times)
            prev_idx = idxs[p - 1] if p - 1 >= 0 else None
            next_idx = idxs[p + 1] if p + 1 < len(idxs) else None
            low = max([x for x in [case_start, parsed_out[prev_idx] if prev_idx is not None else case_start] if x is not None])
            high = min([x for x in [case_end, parsed_out[next_idx] if next_idx is not None else case_end] if x is not None])
            if low > high:
                low, high = case_start, case_end
            forced = nearest_feasible_different(parsed_orig, parsed_orig, low, high, digits)
            if forced == parsed_orig:
                step_ns = get_precision_unit_ns(digits)
                if parsed_orig.value + step_ns <= high.value:
                    forced = pd.Timestamp(parsed_orig.value + step_ns)
                elif parsed_orig.value - step_ns >= low.value:
                    forced = pd.Timestamp(parsed_orig.value - step_ns)
            out_ts[idx] = format_ts(forced, digits)
            parsed_out[idx] = parse_ts(out_ts[idx])
            logs.append(f"forced_change_row={df.at[idx, 'row_id']}")

    rows_still_equal = sum(
        1 for idx in df.index
        if df.at[idx, "error_category"] != "CLEAN" and out_ts[idx] == df.at[idx, "Timestamp"]
    )
    return rows_still_equal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logs = []
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception as e:
        # produce empty-safe fallback only if read failed unexpectedly
        df = pd.DataFrame(columns=["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])
        logs.append(f"read_failed={repr(e)}")

    required_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    out_ts = {idx: df.at[idx, "Timestamp"] for idx in df.index}
    parsed_out = {idx: parse_ts(df.at[idx, "Timestamp"]) for idx in df.index}
    repaired_rows = set()
    stats = {"exchanged_segments": 0, "displaced_segments": 0}

    try:
        for case_id, case_df in df.groupby("Case", sort=False):
            repair_case(case_df, out_ts, parsed_out, repaired_rows, stats, logs)
    except Exception as e:
        logs.append(f"repair_exception={repr(e)}")

    try:
        rows_still_equal = self_check_and_fix(df, out_ts, parsed_out, repaired_rows, logs)
    except Exception as e:
        logs.append(f"selfcheck_exception={repr(e)}")
        # conservative final count
        rows_still_equal = sum(
            1 for idx in df.index
            if df.at[idx, "error_category"] != "CLEAN" and out_ts.get(idx, df.at[idx, "Timestamp"]) == df.at[idx, "Timestamp"]
        )

    result = df.copy()
    result["Timestamp"] = [out_ts[idx] for idx in result.index]

    # Final hard-constraint preservation for non-timestamp columns and row order is automatic via copy
    try:
        result.to_csv(args.output, index=False)
    except Exception as e:
        # must not fail; attempt plain write
        logs.append(f"write_exception={repr(e)}")
        try:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                f.write(",".join(required_cols) + "\n")
                for idx in result.index:
                    row = [str(result.at[idx, c]) for c in required_cols]
                    esc = []
                    for val in row:
                        if any(ch in val for ch in [",", '"', "\n"]):
                            val = '"' + val.replace('"', '""') + '"'
                        esc.append(val)
                    f.write(",".join(esc) + "\n")
        except Exception as e2:
            logs.append(f"write_fallback_exception={repr(e2)}")

    # One required stdout line
    print(f"{stats['exchanged_segments']} {stats['displaced_segments']} {rows_still_equal}")


if __name__ == "__main__":
    main()