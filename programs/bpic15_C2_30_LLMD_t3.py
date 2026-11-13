import argparse
import math
import sys
from datetime import timedelta
from typing import Dict, Tuple, Optional, List

import pandas as pd


DURATION_PROFILES = {
    ("register submission date request", "OLO messaging active"): {"n": 729, "p10": 86400.0, "p25": 172800.0, "median": 306406.0, "p75": 518400.0, "p90": 733026.6, "share": 0.0},
    ("forward to the competent authority", "regular procedure without MER"): {"n": 687, "p10": 0.0, "p25": 0.0, "median": 3.0, "p75": 6.0, "p90": 12.0, "share": 0.975},
    ("send confirmation receipt", "forward to the competent authority"): {"n": 582, "p10": 2.0, "p25": 3.0, "median": 5.0, "p75": 9.0, "p90": 20.9, "share": 0.94},
    ("request complete", "send letter in progress"): {"n": 572, "p10": 0.0, "p25": 0.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "share": 0.991},
    ("article 34  WABO applies", "assessment of content completed"): {"n": 568, "p10": 0.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "share": 0.993},
    ("assessment of content completed", "phase advice known"): {"n": 555, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 1.0, "share": 1.0},
    ("treat subcases completeness", "subcases completeness completed"): {"n": 510, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 3.0, "p90": 9.0, "share": 0.969},
    ("send confirmation receipt", "send confirmation receipt"): {"n": 501, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 214.0, "share": 0.802},
    ("phase application receptive", "request complete"): {"n": 500, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("by law", "creating environmental permit decision"): {"n": 493, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 1.0, "p90": 1.0, "share": 0.996},
    ("treat subcases content", "completed subcases content"): {"n": 492, "p10": 1.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 3.0, "share": 0.996},
    ("enter senddate decision", "enter senddate decision environmental permit"): {"n": 467, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.991},
    ("phase advice known", "grounds for refusal"): {"n": 439, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 3.0, "p90": 6.2, "share": 0.954},
    ("subcases completeness completed", "procedure change"): {"n": 436, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 4.0, "p90": 7.0, "share": 0.986},
    ("ask stakeholders views", "suspension ground applicable"): {"n": 431, "p10": 0.0, "p25": 1.0, "median": 2.0, "p75": 4.0, "p90": 6.0, "share": 0.984},
    ("creating environmental permit decision", "decision date prior to decision"): {"n": 431, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 4.0, "share": 0.993},
    ("procedure change", "request complete"): {"n": 429, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 3.0, "p90": 6.0, "share": 0.991},
    ("grounds for refusal", "ask stakeholders views"): {"n": 425, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 5.0, "p90": 15.0, "share": 0.958},
    ("suspension ground applicable", "by law"): {"n": 415, "p10": 1.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "share": 0.993},
    ("send confirmation receipt", "enter senddate acknowledgement"): {"n": 412, "p10": 2.0, "p25": 74.25, "median": 159.0, "p75": 308.25, "p90": 1342.5, "share": 0.24},
    ("OLO messaging active", "phase application received"): {"n": 409, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("regular procedure without MER", "send confirmation receipt"): {"n": 399, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("phase application received", "send confirmation receipt"): {"n": 395, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 4.0, "p90": 10.0, "share": 0.982},
    ("procedure change", "phase application receptive"): {"n": 384, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 5.0, "p90": 9.7, "share": 0.964},
    ("enter senddate acknowledgement", "send confirmation receipt"): {"n": 383, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.99},
    ("enter senddate procedure confirmation", "send procedure confirmation"): {"n": 378, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.952},
    ("send confirmation receipt", "start WABOprocedure"): {"n": 375, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("forward to the competent authority", "phase application received"): {"n": 372, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("registration date publication", "regular procedure without MER"): {"n": 357, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.994},
    ("procedure change", "article 34  WABO applies"): {"n": 353, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 3.0, "p90": 10.8, "share": 0.946},
    ("OLO messaging active", "create procedure confirmation"): {"n": 344, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("transcript decision environmental permit to stakeholders", "generate publication document decision environmental permit"): {"n": 342, "p10": 6.0, "p25": 7.0, "median": 10.0, "p75": 19.0, "p90": 55.0, "share": 0.906},
    ("send letter in progress", "treat subcases content"): {"n": 324, "p10": 0.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 2.0, "share": 0.994},
    ("phase decision sent", "phase decision taken"): {"n": 320, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("grounds for refusal", "phase advice known"): {"n": 319, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("create procedure confirmation", "create publication document"): {"n": 318, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.997},
    ("phase advice known", "phase application receptive"): {"n": 316, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("phase decision taken", "register objection and appeal periods"): {"n": 313, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 5.0, "p90": 12.8, "share": 0.971},
    ("article 34  WABO applies", "ask stakeholders views"): {"n": 305, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.997},
    ("completed subcases content", "procedure change"): {"n": 299, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 5.0, "p90": 10.0, "share": 0.98},
    ("enter senddate decision environmental permit", "phase decision sent"): {"n": 294, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 1.0, "share": 1.0},
    ("creating environmental permit decision", "generating decision environmental permit"): {"n": 293, "p10": 4.0, "p25": 51.0, "median": 1864.0, "p75": 149542.0, "p90": 923794.2, "share": 0.259},
    ("start WABOprocedure", "enter senddate acknowledgement"): {"n": 293, "p10": 34535.2, "p25": 38196.0, "median": 43648.0, "p75": 52527.0, "p90": 86400.0, "share": 0.0},
    ("send letter in progress", "set decision status"): {"n": 254, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("set decision phase decision permitting decided", "start decision phase decision permitting sent"): {"n": 241, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.988},
    ("phase application received", "reception through OLO"): {"n": 237, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 7.0, "p90": 15.0, "share": 0.966},
    ("regular procedure without MER", "start WABOprocedure"): {"n": 229, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 1.0, "share": 1.0},
    ("send procedure confirmation", "registration date publication"): {"n": 225, "p10": 8.0, "p25": 11.0, "median": 19.0, "p75": 69.0, "p90": 738725.6, "share": 0.724},
    ("transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit"): {"n": 224, "p10": 8.3, "p25": 34.75, "median": 50393.0, "p75": 172800.0, "p90": 259200.0, "share": 0.254},
    ("regular procedure without MER", "enter senddate procedure confirmation"): {"n": 219, "p10": 1.0, "p25": 2.0, "median": 2.0, "p75": 4.0, "p90": 87809.2, "share": 0.872},
    ("generating decision environmental permit", "grounds for refusal"): {"n": 215, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("decision date prior to decision", "generating decision environmental permit"): {"n": 214, "p10": 0.0, "p25": 0.0, "median": 2.5, "p75": 4991.0, "p90": 439697.8, "share": 0.561},
    ("create publication document", "create subcases completeness"): {"n": 197, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 7.4, "share": 0.914},
    ("ask stakeholders views", "assessment of content completed"): {"n": 194, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("by law", "completed subcases content"): {"n": 194, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("procedure change", "procedure change"): {"n": 193, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("submit decision", "suspension ground applicable"): {"n": 193, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("completed subcases content", "coordination of application"): {"n": 190, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("register submission date request", "enter senddate acknowledgement"): {"n": 188, "p10": 172800.0, "p25": 345600.0, "median": 604800.0, "p75": 864000.0, "p90": 1209600.0, "share": 0.0},
    ("record date of decision environmental permit", "set decision phase decision permitting decided"): {"n": 187, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
}

DFG_FREQ = {(k[0], k[1]): v["n"] for k, v in DURATION_PROFILES.items()}
GLOBAL_FAST_DEFAULT = 1.0


def detect_precision(series: pd.Series) -> int:
    mx = 0
    for s in series.dropna().astype(str):
        if "." in s:
            frac = s.split(".")[-1]
            mx = max(mx, len(frac))
    return mx


def parse_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, errors="coerce")


def format_ts(ts: pd.Timestamp, precision: int) -> str:
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    frac = base.split(".")[-1][:precision]
    return base.split(".")[0] + "." + frac


def td(seconds: float) -> pd.Timedelta:
    return pd.to_timedelta(float(seconds), unit="s")


def profile(src: Optional[str], tgt: Optional[str]) -> Optional[dict]:
    if src is None or tgt is None:
        return None
    return DURATION_PROFILES.get((src, tgt))


def preferred_gap(src: Optional[str], tgt: Optional[str]) -> float:
    p = profile(src, tgt)
    if p is None:
        return GLOBAL_FAST_DEFAULT
    if p["share"] >= 0.9:
        return p["p25"]
    if p["share"] >= 0.7:
        return p["p10"]
    return min(p["p10"], p["p25"], 60.0) if min(p["p10"], p["p25"]) > 0 else GLOBAL_FAST_DEFAULT


def edge_score(src: Optional[str], tgt: Optional[str]) -> float:
    if src is None or tgt is None:
        return 0.0
    n = DFG_FREQ.get((src, tgt), 0)
    if n <= 0:
        return -8.0
    return math.log1p(n)


def gap_penalty(src: Optional[str], tgt: Optional[str], gap_s: Optional[float]) -> float:
    if gap_s is None or gap_s < 0:
        return 1e6
    p = profile(src, tgt)
    if p is None:
        return abs(gap_s - GLOBAL_FAST_DEFAULT) * 0.02
    pref = preferred_gap(src, tgt)
    p90 = max(p["p90"], pref, 1.0)
    if p["share"] >= 0.9:
        scale = max(5.0, p90)
    elif p["share"] >= 0.7:
        scale = max(20.0, p90)
    else:
        scale = max(60.0, p90)
    return abs(gap_s - pref) / scale


def local_swap_score(prev_act, act_i, act_j, next_act, t_prev, t_i, t_j, t_next) -> float:
    score = 0.0
    score += edge_score(prev_act, act_i) if prev_act is not None else 0.0
    score += edge_score(act_i, act_j)
    score += edge_score(act_j, next_act) if next_act is not None else 0.0
    if t_prev is not None:
        score -= gap_penalty(prev_act, act_i, (t_i - t_prev).total_seconds())
    score -= gap_penalty(act_i, act_j, (t_j - t_i).total_seconds())
    if t_next is not None:
        score -= gap_penalty(act_j, next_act, (t_next - t_j).total_seconds())
    return score


def clamp_ts(ts: pd.Timestamp, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Timestamp:
    if ts < lo:
        return lo
    if ts > hi:
        return hi
    return ts


def smallest_unit(precision: int) -> pd.Timedelta:
    if precision <= 0:
        return pd.Timedelta(seconds=1)
    if precision <= 6:
        return pd.to_timedelta(10 ** (6 - precision), unit="us")
    return pd.Timedelta(microseconds=1)


def estimate_row(left_ts, right_ts, left_act, act, act_right, precision: int, original: pd.Timestamp) -> pd.Timestamp:
    eps = smallest_unit(precision)
    candidates = []
    if left_ts is not None:
        candidates.append(left_ts + td(preferred_gap(left_act, act)))
        candidates.append(left_ts + td(max(preferred_gap(left_act, act), 0.0)))
    if right_ts is not None:
        candidates.append(right_ts - td(preferred_gap(act, act_right)))
        candidates.append(right_ts - td(max(preferred_gap(act, act_right), 0.0)))
    if left_ts is not None and right_ts is not None:
        lg = preferred_gap(left_act, act)
        rg = preferred_gap(act, act_right)
        total = max(lg + rg, eps.total_seconds())
        span = right_ts - left_ts
        frac = lg / total
        candidates.append(left_ts + pd.to_timedelta(span.total_seconds() * frac, unit="s"))
        candidates.append(left_ts + eps)
        candidates.append(right_ts - eps)
    if not candidates:
        candidates.append(original + eps)

    lo = left_ts if left_ts is not None else original - pd.Timedelta(days=36500)
    hi = right_ts if right_ts is not None else original + pd.Timedelta(days=36500)
    fixed = []
    for c in candidates:
        c = clamp_ts(c, lo, hi)
        fixed.append(c)

    fixed = [c for c in fixed if lo <= c <= hi]
    if not fixed:
        fixed = [lo]

    def obj(c):
        move = abs((c - original).total_seconds())
        pen = 0.0
        if left_ts is not None:
            pen += gap_penalty(left_act, act, (c - left_ts).total_seconds())
        if right_ts is not None:
            pen += gap_penalty(act, act_right, (right_ts - c).total_seconds())
        return (pen, move)

    best = min(fixed, key=obj)
    if best == original:
        if best + eps <= hi:
            best = best + eps
        elif best - eps >= lo:
            best = best - eps
    return clamp_ts(best, lo, hi)


def repair_case(case_df: pd.DataFrame, precision: int):
    out = case_df["Timestamp"].astype(str).tolist()
    parsed = [parse_ts(x) for x in out]
    errs = case_df["error_category"].astype(str).tolist()
    acts = case_df["Activity"].astype(str).tolist()
    n = len(case_df)

    case_start = min([t for t in parsed if pd.notna(t)])
    case_end = max([t for t in parsed if pd.notna(t)])

    exchanged = 0
    displaced = 0

    i = 1
    while i < n:
        if errs[i] != "CLEAN" and errs[i - 1] != "CLEAN" and pd.notna(parsed[i]) and pd.notna(parsed[i - 1]) and parsed[i] < parsed[i - 1]:
            a = i - 1
            b = i

            left_anchor_idx = a - 1 if a - 1 >= 0 else None
            right_anchor_idx = b + 1 if b + 1 < n else None

            left_ts = parsed[left_anchor_idx] if left_anchor_idx is not None else case_start
            right_ts = parsed[right_anchor_idx] if right_anchor_idx is not None else case_end
            left_act = acts[left_anchor_idx] if left_anchor_idx is not None else None
            right_act = acts[right_anchor_idx] if right_anchor_idx is not None else None

            t_a_obs = parsed[a]
            t_b_obs = parsed[b]
            act_a = acts[a]
            act_b = acts[b]

            swap_a = t_b_obs
            swap_b = t_a_obs

            swap_valid = True
            if swap_a < left_ts or swap_b > right_ts or swap_a > swap_b:
                swap_valid = False
            if left_anchor_idx is not None and swap_a < parsed[left_anchor_idx]:
                swap_valid = False
            if right_anchor_idx is not None and swap_b > parsed[right_anchor_idx]:
                swap_valid = False

            swap_score = -1e18
            if swap_valid:
                swap_score = local_swap_score(left_act, act_a, act_b, right_act, left_ts if left_anchor_idx is not None else None, swap_a, swap_b, right_ts if right_anchor_idx is not None else None)

            est_b = estimate_row(t_a_obs, right_ts, act_a, act_b, right_act, precision, t_b_obs)
            disp1_valid = (t_a_obs >= left_ts) and (est_b >= t_a_obs) and (est_b <= right_ts)
            disp1_score = -1e18
            if disp1_valid:
                disp1_score = local_swap_score(left_act, act_a, act_b, right_act, left_ts if left_anchor_idx is not None else None, t_a_obs, est_b, right_ts if right_anchor_idx is not None else None) - abs((est_b - t_b_obs).total_seconds()) / 30.0

            est_a = estimate_row(left_ts, t_b_obs, left_act, act_a, act_b, precision, t_a_obs)
            disp2_valid = (est_a >= left_ts) and (est_a <= t_b_obs) and (t_b_obs <= right_ts)
            disp2_score = -1e18
            if disp2_valid:
                disp2_score = local_swap_score(left_act, act_a, act_b, right_act, left_ts if left_anchor_idx is not None else None, est_a, t_b_obs, right_ts if right_anchor_idx is not None else None) - abs((est_a - t_a_obs).total_seconds()) / 30.0

            choose = "displaced"
            if swap_valid and swap_score >= max(disp1_score, disp2_score) + 0.25:
                choose = "exchanged"

            if choose == "exchanged":
                out[a] = case_df.iloc[b]["Timestamp"]
                out[b] = case_df.iloc[a]["Timestamp"]
                parsed[a] = t_b_obs
                parsed[b] = t_a_obs
                exchanged += 1
            else:
                if disp1_score >= disp2_score:
                    repaired = est_b
                    repaired = clamp_ts(repaired, max(t_a_obs, left_ts), right_ts)
                    rep_s = format_ts(repaired, precision)
                    if rep_s == case_df.iloc[b]["Timestamp"]:
                        eps = smallest_unit(precision)
                        alt = repaired + eps if repaired + eps <= right_ts else repaired - eps
                        alt = clamp_ts(alt, max(t_a_obs, left_ts), right_ts)
                        rep_s = format_ts(alt, precision)
                        repaired = parse_ts(rep_s)
                    out[b] = rep_s
                    parsed[b] = repaired
                else:
                    repaired = est_a
                    repaired = clamp_ts(repaired, left_ts, min(t_b_obs, right_ts))
                    rep_s = format_ts(repaired, precision)
                    if rep_s == case_df.iloc[a]["Timestamp"]:
                        eps = smallest_unit(precision)
                        alt = repaired - eps if repaired - eps >= left_ts else repaired + eps
                        alt = clamp_ts(alt, left_ts, min(t_b_obs, right_ts))
                        rep_s = format_ts(alt, precision)
                        repaired = parse_ts(rep_s)
                    out[a] = rep_s
                    parsed[a] = repaired
                displaced += 1
            i += 2
        else:
            i += 1

    return out, exchanged, displaced


def self_check_and_fix(df: pd.DataFrame, repaired_ts: List[str], precision: int):
    msgs = []
    parsed_out = [parse_ts(x) for x in repaired_ts]
    parsed_in = [parse_ts(x) for x in df["Timestamp"].astype(str).tolist()]
    errs = df["error_category"].astype(str).tolist()
    cases = df["Case"].astype(str).tolist()

    for case, idxs in df.groupby("Case", sort=False).groups.items():
        idxs = list(idxs)
        case_times = [parsed_out[i] for i in idxs if pd.notna(parsed_out[i])]
        if not case_times:
            continue
        cstart = min(case_times)
        cend = max(case_times)
        for pos, i in enumerate(idxs):
            if errs[i] == "CLEAN":
                repaired_ts[i] = df.iloc[i]["Timestamp"]
                parsed_out[i] = parse_ts(repaired_ts[i])
            else:
                if repaired_ts[i] == df.iloc[i]["Timestamp"]:
                    lo = cstart
                    hi = cend
                    if pos > 0:
                        lo = max(lo, parsed_out[idxs[pos - 1]])
                    if pos < len(idxs) - 1:
                        hi = min(hi, parsed_out[idxs[pos + 1]])
                    eps = smallest_unit(precision)
                    cand = clamp_ts((lo if lo == hi else lo + eps), lo, hi)
                    s = format_ts(cand, precision)
                    if s == df.iloc[i]["Timestamp"]:
                        cand2 = clamp_ts(cand + eps, lo, hi)
                        s = format_ts(cand2, precision)
                    repaired_ts[i] = s
                    parsed_out[i] = parse_ts(s)
                    msgs.append(f"fixed_equal_row:{i}")

        for pos, i in enumerate(idxs):
            lo = cstart
            hi = cend
            if pos > 0:
                lo = max(lo, parsed_out[idxs[pos - 1]])
            if pos < len(idxs) - 1:
                hi = min(hi, parsed_out[idxs[pos + 1]])
            if parsed_out[i] < lo or parsed_out[i] > hi:
                clamped = clamp_ts(parsed_out[i], lo, hi)
                s = format_ts(clamped, precision)
                if errs[i] != "CLEAN" and s == df.iloc[i]["Timestamp"]:
                    eps = smallest_unit(precision)
                    if clamped + eps <= hi:
                        s = format_ts(clamped + eps, precision)
                    elif clamped - eps >= lo:
                        s = format_ts(clamped - eps, precision)
                repaired_ts[i] = s
                parsed_out[i] = parse_ts(s)
                msgs.append(f"clamped_row:{i}")

    if msgs:
        print(";".join(msgs))
    return repaired_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str)
        original_cols = list(df.columns)
        precision = detect_precision(df["Timestamp"])

        result_ts = df["Timestamp"].astype(str).tolist()
        total_exchanged = 0
        total_displaced = 0

        for case, case_idx in df.groupby("Case", sort=False).groups.items():
            case_df = df.loc[list(case_idx)].copy()
            repaired_case_ts, ex_cnt, dis_cnt = repair_case(case_df, precision)
            for local_pos, global_idx in enumerate(case_df.index.tolist()):
                if case_df.iloc[local_pos]["error_category"] != "CLEAN":
                    result_ts[global_idx] = repaired_case_ts[local_pos]
            total_exchanged += ex_cnt
            total_displaced += dis_cnt

        result_ts = self_check_and_fix(df, result_ts, precision)

        out_df = df.copy()
        out_df["Timestamp"] = result_ts

        # Final exact preservation for immutable columns and CLEAN timestamps.
        for c in ["Case", "Activity", "Resource", "error_category", "row_id"]:
            out_df[c] = df[c]
        clean_mask = out_df["error_category"] == "CLEAN"
        out_df.loc[clean_mask, "Timestamp"] = df.loc[clean_mask, "Timestamp"]

        rows_equal = int((out_df["Timestamp"].astype(str) == df["Timestamp"].astype(str)).sum())

        out_df = out_df[original_cols]
        out_df.to_csv(args.output, index=False)

        print(f"exchanged_segments={total_exchanged} displaced_segments={total_displaced} rows_still_equal={rows_equal}")
    except Exception as e:
        try:
            df = pd.read_csv(args.input, dtype=str)
            df.to_csv(args.output, index=False)
            print(f"recovery_mode error={type(e).__name__} exchanged_segments=0 displaced_segments=0 rows_still_equal={len(df)}")
        except Exception as e2:
            fallback = pd.DataFrame(columns=["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])
            fallback.to_csv(args.output, index=False)
            print(f"recovery_mode error={type(e).__name__} secondary={type(e2).__name__} exchanged_segments=0 displaced_segments=0 rows_still_equal=0")


if __name__ == "__main__":
    main()