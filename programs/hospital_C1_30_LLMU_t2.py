import argparse
import itertools
import math
import re
from collections import defaultdict, Counter

import pandas as pd


TRANSITION_PROBS = {
    ("RELEASE", "CODE OK"): 0.9536,
    ("FIN", "RELEASE"): 0.9243,
    ("CODE OK", "BILLED"): 0.9101,
    ("BILLED", "STORNO"): 0.9013,
    ("CHANGE DIAGN", "FIN"): 0.8112,
    ("JOIN-PAT", "JOIN-PAT"): 0.7966,
    ("REOPEN", "FIN"): 0.6865,
    ("CHANGE END", "DELETE"): 0.6,
    ("DELETE", "DELETE"): 0.6,
    ("REJECT", "BILLED"): 0.5991,
    ("STORNO", "BILLED"): 0.5158,
    ("CODE ERROR", "MANUAL"): 0.5,
    ("CODE ERROR", "RELEASE"): 0.5,
    ("MANUAL", "RELEASE"): 0.4889,
    ("STORNO", "REJECT"): 0.4572,
    ("NEW", "CHANGE DIAGN"): 0.4233,
    ("CHANGE END", "FIN"): 0.4,
    ("DELETE", "JOIN-PAT"): 0.4,
    ("SET STATUS", "DELETE"): 0.3333,
    ("REJECT", "REOPEN"): 0.3208,
    ("NEW", "FIN"): 0.2573,
    ("CODE NOK", "CODE NOK"): 0.2531,
    ("REOPEN", "DELETE"): 0.2508,
    ("CODE NOK", "CODE OK"): 0.2407,
    ("CODE NOK", "SET STATUS"): 0.2222,
    ("MANUAL", "MANUAL"): 0.2,
    ("SET STATUS", "CHANGE END"): 0.1667,
    ("SET STATUS", "FIN"): 0.1667,
    ("SET STATUS", "RELEASE"): 0.1667,
    ("SET STATUS", "SET STATUS"): 0.1667,
    ("MANUAL", "CODE OK"): 0.1556,
    ("CODE NOK", "REOPEN"): 0.1358,
    ("JOIN-PAT", "FIN"): 0.1356,
    ("NEW", "NEW"): 0.1185,
    ("CODE NOK", "DELETE"): 0.1173,
    ("NEW", "BILLED"): 0.0745,
    ("MANUAL", "BILLED"): 0.0667,
    ("CODE OK", "REOPEN"): 0.065,
    ("NEW", "DELETE"): 0.0649,
    ("CHANGE DIAGN", "CHANGE DIAGN"): 0.0594,
    ("CHANGE DIAGN", "BILLED"): 0.0505,
    ("FIN", "BILLED"): 0.0496,
    ("BILLED", "BILLED"): 0.0461,
    ("MANUAL", "DELETE"): 0.0444,
    ("NEW", "CODE OK"): 0.0408,
    ("REOPEN", "CHANGE DIAGN"): 0.0396,
    ("BILLED", "CODE OK"): 0.0373,
    ("CHANGE DIAGN", "CODE OK"): 0.0343,
    ("JOIN-PAT", "BILLED"): 0.0339,
    ("JOIN-PAT", "DELETE"): 0.0339,
    ("CHANGE DIAGN", "DELETE"): 0.0336,
    ("REJECT", "RELEASE"): 0.033,
    ("REJECT", "SET STATUS"): 0.0283,
    ("MANUAL", "REOPEN"): 0.0222,
    ("MANUAL", "SET STATUS"): 0.0222,
    ("RELEASE", "CODE NOK"): 0.0192,
    ("CODE NOK", "RELEASE"): 0.0185,
    ("NEW", "RELEASE"): 0.0173,
    ("REJECT", "REJECT"): 0.0142,
    ("STORNO", "SET STATUS"): 0.0135,
}

DURATION_PROFILES = {
    ("CODE OK", "BILLED"): {"p10": 370618.4, "p25": 1226151.0, "median": 3696259.5, "share": 0.0},
    ("RELEASE", "CODE OK"): {"p10": 3489.0, "p25": 176021.0, "median": 342906.0, "share": 0.05},
    ("FIN", "RELEASE"): {"p10": 1970.3, "p25": 6714.0, "median": 16196.5, "share": 0.058},
    ("NEW", "CHANGE DIAGN"): {"p10": 12.0, "p25": 16.0, "median": 24.0, "share": 0.637},
    ("CHANGE DIAGN", "FIN"): {"p10": 1327999.8, "p25": 4002043.0, "median": 7707138.0, "share": 0.009},
    ("NEW", "FIN"): {"p10": 429682.0, "p25": 5192415.0, "median": 29527664.0, "share": 0.04},
    ("NEW", "NEW"): {"p10": 1.0, "p25": 1.0, "median": 1.0, "share": 1.0},
    ("NEW", "BILLED"): {"p10": 10067520.7, "p25": 12572347.25, "median": 15589142.5, "share": 0.0},
    ("NEW", "DELETE"): {"p10": 2555.0, "p25": 171326.0, "median": 1865565.0, "share": 0.062},
    ("REOPEN", "FIN"): {"p10": 5.0, "p25": 10.0, "median": 42.0, "share": 0.526},
    ("BILLED", "STORNO"): {"p10": 195856.0, "p25": 434146.5, "median": 1077095.0, "share": 0.002},
    ("CODE OK", "REOPEN"): {"p10": 124052.7, "p25": 463729.25, "median": 1921091.5, "share": 0.015},
    ("NEW", "CODE OK"): {"p10": 4135423.0, "p25": 7154291.5, "median": 8264453.5, "share": 0.0},
    ("FIN", "BILLED"): {"p10": 775793.4, "p25": 1720658.5, "median": 3633863.0, "share": 0.0},
    ("CHANGE DIAGN", "CHANGE DIAGN"): {"p10": 201.6, "p25": 42275.0, "median": 698451.0, "share": 0.04},
    ("STORNO", "BILLED"): {"p10": 92767.8, "p25": 92840.0, "median": 169860.0, "share": 0.0},
    ("CHANGE DIAGN", "BILLED"): {"p10": 8103533.1, "p25": 10535503.25, "median": 12718535.5, "share": 0.0},
    ("STORNO", "REJECT"): {"p10": 5.0, "p25": 13.5, "median": 1240.0, "share": 0.438},
    ("NEW", "RELEASE"): {"p10": 3683105.6, "p25": 5029015.5, "median": 7740827.0, "share": 0.0},
    ("REOPEN", "DELETE"): {"p10": 2.0, "p25": 2.0, "median": 3.0, "share": 0.967},
    ("CHANGE DIAGN", "CODE OK"): {"p10": 1583545.3, "p25": 4364398.0, "median": 8186331.0, "share": 0.007},
    ("CHANGE DIAGN", "DELETE"): {"p10": 206.0, "p25": 69873.0, "median": 946414.0, "share": 0.064},
    ("REJECT", "BILLED"): {"p10": 1892.6, "p25": 163288.0, "median": 328893.0, "share": 0.055},
    ("RELEASE", "CODE NOK"): {"p10": 330802.6, "p25": 606467.75, "median": 1430526.5, "share": 0.009},
    ("CODE OK", "RELEASE"): {"p10": 1976833.6, "p25": 2891266.75, "median": 4348244.5, "share": 0.012},
    ("REJECT", "REOPEN"): {"p10": 7.7, "p25": 10.0, "median": 30.5, "share": 0.662},
    ("RELEASE", "BILLED"): {"p10": 1835036.0, "p25": 2669431.0, "median": 7875755.0, "share": 0.0},
    ("FIN", "REOPEN"): {"p10": 6.6, "p25": 65.0, "median": 63422.0, "share": 0.255},
    ("FIN", "FIN"): {"p10": 0.0, "p25": 204302.5, "median": 9034907.0, "share": 0.234},
    ("JOIN-PAT", "JOIN-PAT"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("RELEASE", "REOPEN"): {"p10": 6480.0, "p25": 35903.0, "median": 225448.0, "share": 0.023},
    ("CODE NOK", "CODE NOK"): {"p10": 342359.0, "p25": 518520.0, "median": 1897911.0, "share": 0.0},
    ("CODE NOK", "CODE OK"): {"p10": 2036.4, "p25": 12560.0, "median": 43358.0, "share": 0.026},
    ("CODE NOK", "SET STATUS"): {"p10": 13580.5, "p25": 18966.0, "median": 30387.0, "share": 0.0},
    ("FIN", "DELETE"): {"p10": 68.2, "p25": 22685.5, "median": 1255926.0, "share": 0.086},
    ("CODE OK", "MANUAL"): {"p10": 448.2, "p25": 111643.0, "median": 1652942.0, "share": 0.069},
    ("REOPEN", "CHANGE DIAGN"): {"p10": 10.9, "p25": 21.5, "median": 88.5, "share": 0.458},
    ("CODE NOK", "REOPEN"): {"p10": 9324.8, "p25": 21800.5, "median": 31220.0, "share": 0.045},
    ("MANUAL", "RELEASE"): {"p10": 453.8, "p25": 27384.25, "median": 36655.5, "share": 0.045},
    ("BILLED", "BILLED"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 0.714},
    ("CODE OK", "CODE OK"): {"p10": 828668.8, "p25": 2361117.5, "median": 12213631.0, "share": 0.0},
    ("CODE NOK", "DELETE"): {"p10": 12976.6, "p25": 23342.0, "median": 27554.0, "share": 0.0},
    ("RELEASE", "RELEASE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 0.611},
    ("BILLED", "CODE OK"): {"p10": 2266256.2, "p25": 7135198.0, "median": 8266846.0, "share": 0.0},
    ("RELEASE", "DELETE"): {"p10": 34255.7, "p25": 124880.75, "median": 249504.5, "share": 0.071},
    ("CHANGE DIAGN", "RELEASE"): {"p10": 4419902.6, "p25": 7074031.0, "median": 7794016.0, "share": 0.0},
    ("CHANGE DIAGN", "STORNO"): {"p10": 10634565.1, "p25": 11483076.25, "median": 16546335.0, "share": 0.0},
    ("NEW", "REOPEN"): {"p10": 4866878.0, "p25": 7547122.25, "median": 8322668.5, "share": 0.0},
    ("FIN", "CODE OK"): {"p10": 411574.0, "p25": 2954281.0, "median": 5896170.0, "share": 0.091},
    ("RELEASE", "SET STATUS"): {"p10": 140567.6, "p25": 150606.0, "median": 368578.5, "share": 0.0},
    ("CODE OK", "DELETE"): {"p10": 2419266.4, "p25": 2776767.75, "median": 4336982.5, "share": 0.0},
    ("RELEASE", "FIN"): {"p10": 491295.5, "p25": 848418.25, "median": 2666622.0, "share": 0.0},
    ("MANUAL", "MANUAL"): {"p10": 6.8, "p25": 41.0, "median": 171.0, "share": 0.333},
    ("JOIN-PAT", "FIN"): {"p10": 3050054.3, "p25": 4138408.25, "median": 5127431.5, "share": 0.0},
    ("FIN", "SET STATUS"): {"p10": 59.6, "p25": 346488.5, "median": 739344.5, "share": 0.125},
    ("CHANGE DIAGN", "JOIN-PAT"): {"p10": 5154.2, "p25": 14279.75, "median": 596185.5, "share": 0.0},
    ("REJECT", "RELEASE"): {"p10": 7.6, "p25": 9.0, "median": 34.0, "share": 0.714},
    ("RELEASE", "MANUAL"): {"p10": 8931504.6, "p25": 15188442.5, "median": 19738990.0, "share": 0.0},
    ("MANUAL", "CODE OK"): {"p10": 31.4, "p25": 54.5, "median": 104.0, "share": 0.286},
    ("RELEASE", "STORNO"): {"p10": 4772324.2, "p25": 5108938.0, "median": 5648782.0, "share": 0.0},
}

GLOBAL_MEDIAN = 431590.0
DEFAULT_SMALL_FAST = 15.0
DEFAULT_SMALL = 60.0
DEFAULT_MEDIUM = 600.0


def safe_to_datetime(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT


def infer_fraction_digits(ts):
    if ts is None:
        return 0
    m = re.search(r"\.(\d+)$", str(ts).strip())
    return len(m.group(1)) if m else 0


def format_ts(dt, frac_digits):
    if pd.isna(dt):
        return ""
    py_dt = pd.Timestamp(dt).to_pydatetime()
    base = py_dt.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits <= 0:
        return base
    micro = py_dt.microsecond
    frac = f"{micro:06d}"[:frac_digits].ljust(frac_digits, "0")
    return f"{base}.{frac}"


def precision_unit_ns(frac_digits):
    if frac_digits <= 0:
        return 1_000_000_000
    if frac_digits <= 6:
        return 10 ** (9 - frac_digits)
    return 1


def round_to_precision(ts, frac_digits):
    if pd.isna(ts):
        return ts
    unit = precision_unit_ns(frac_digits)
    val = ts.value
    rounded = int(round(val / unit) * unit)
    return pd.Timestamp(rounded)


def transition_prob(a, b):
    return TRANSITION_PROBS.get((a, b), 1e-6)


def duration_quantile(a, b):
    prof = DURATION_PROFILES.get((a, b))
    if prof:
        if prof["share"] >= 0.4:
            return max(prof["p10"], 1.0)
        return max(prof["p25"], 1.0)
    rev = DURATION_PROFILES.get((b, a))
    if rev:
        if rev["share"] >= 0.4:
            return max(rev["p10"], 1.0)
        return max(rev["p25"], 1.0)
    if a == b:
        return 1.0
    if transition_prob(a, b) > 0.2:
        return DEFAULT_SMALL
    if transition_prob(a, b) > 0.02:
        return DEFAULT_MEDIUM
    return DEFAULT_SMALL_FAST


def duration_penalty(a, b, obs_seconds):
    prof = DURATION_PROFILES.get((a, b))
    if not prof:
        rev = DURATION_PROFILES.get((b, a))
        if rev:
            prof = rev
    if obs_seconds < 0:
        return 1e6 + abs(obs_seconds)
    if prof:
        target = prof["p10"] if prof["share"] >= 0.4 else prof["p25"]
        scale = max(target, 1.0)
        return abs(obs_seconds - target) / scale
    target = duration_quantile(a, b)
    return abs(obs_seconds - target) / max(target, 1.0) + 1.0


def best_order(block_acts, left_act, right_act, beam_width=80):
    n = len(block_acts)
    if n <= 1:
        return tuple(range(n))
    idxs = list(range(n))
    if n <= 7 and len(set(block_acts)) == n:
        best = None
        best_score = -1e18
        for perm in itertools.permutations(idxs):
            acts = [block_acts[i] for i in perm]
            score = 0.0
            if left_act is not None:
                score += math.log(transition_prob(left_act, acts[0]))
            for i in range(len(acts) - 1):
                score += math.log(transition_prob(acts[i], acts[i + 1]))
            if right_act is not None:
                score += math.log(transition_prob(acts[-1], right_act))
            dur_sum = sum(duration_quantile(acts[i], acts[i + 1]) for i in range(len(acts) - 1))
            score -= 1e-7 * dur_sum
            if score > best_score:
                best_score = score
                best = perm
        return best

    states = [([], set(), 0.0)]
    for _ in range(n):
        new_states = []
        for seq, used, score in states:
            prev_act = left_act if not seq else block_acts[seq[-1]]
            for j in idxs:
                if j in used:
                    continue
                next_score = score + math.log(transition_prob(prev_act, block_acts[j])) if prev_act is not None else score
                if seq:
                    next_score -= 1e-7 * duration_quantile(block_acts[seq[-1]], block_acts[j])
                new_states.append((seq + [j], used | {j}, next_score))
        new_states.sort(key=lambda x: x[2], reverse=True)
        states = new_states[:beam_width]
    best = None
    best_score = -1e18
    for seq, used, score in states:
        final_score = score
        if right_act is not None:
            final_score += math.log(transition_prob(block_acts[seq[-1]], right_act))
        if final_score > best_score:
            best_score = final_score
            best = tuple(seq)
    return best if best is not None else tuple(idxs)


def enforce_strictly_increasing(ts_list, frac_digits, left_bound=None, right_bound=None):
    out = [round_to_precision(t, frac_digits) for t in ts_list]
    step_ns = precision_unit_ns(frac_digits)
    if step_ns < 1_000_000_000:
        min_delta = pd.Timedelta(step_ns, unit="ns")
    else:
        min_delta = pd.Timedelta(seconds=1)
    if left_bound is not None and not pd.isna(left_bound):
        out[0] = max(out[0], round_to_precision(left_bound, frac_digits))
    for i in range(1, len(out)):
        need = out[i - 1] + min_delta
        if out[i] < need:
            out[i] = need
    if right_bound is not None and not pd.isna(right_bound):
        rb = round_to_precision(right_bound, frac_digits)
        if out[-1] > rb:
            latest_start = rb - min_delta * (len(out) - 1)
            if left_bound is not None and not pd.isna(left_bound):
                latest_start = max(latest_start, round_to_precision(left_bound, frac_digits))
            out = [latest_start + min_delta * i for i in range(len(out))]
            if out[-1] > rb:
                out[-1] = rb
                for i in range(len(out) - 2, -1, -1):
                    prev_max = out[i + 1] - min_delta
                    if out[i] > prev_max:
                        out[i] = prev_max
                if left_bound is not None and not pd.isna(left_bound) and out[0] < left_bound:
                    out[0] = round_to_precision(left_bound, frac_digits)
                    for i in range(1, len(out)):
                        need = out[i - 1] + min_delta
                        if out[i] < need:
                            out[i] = need
    return out


def c1_repair_case_block(df, idxs, case_idxs, orig_ts, parsed_ts, out_ts, repaired_mask, stats):
    block_rows = idxs
    k = len(block_rows)
    observed_str = orig_ts[block_rows[0]]
    observed_dt = parsed_ts[block_rows[0]]
    frac_digits = infer_fraction_digits(observed_str)
    acts = [df.at[i, "Activity"] for i in block_rows]

    pos0 = case_idxs.index(block_rows[0])
    left_idx = case_idxs[pos0 - 1] if pos0 - 1 >= 0 else None
    right_idx = case_idxs[pos0 + k] if pos0 + k < len(case_idxs) else None

    left_act = df.at[left_idx, "Activity"] if left_idx is not None else None
    right_act = df.at[right_idx, "Activity"] if right_idx is not None else None
    left_ts = parsed_ts[left_idx] if left_idx is not None else pd.NaT
    right_ts = parsed_ts[right_idx] if right_idx is not None else pd.NaT

    order = best_order(acts, left_act, right_act)
    ordered_rows = [block_rows[j] for j in order]
    ordered_acts = [acts[j] for j in order]

    gaps = []
    for i in range(k - 1):
        g = duration_quantile(ordered_acts[i], ordered_acts[i + 1])
        gaps.append(max(g, 1.0))

    available = None
    if right_idx is not None and not pd.isna(right_ts) and not pd.isna(observed_dt):
        available = max((right_ts - observed_dt).total_seconds(), 0.0)

    min_gap = max(1.0, precision_unit_ns(frac_digits) / 1e9)
    min_span = (k - 1) * min_gap
    raw_span = sum(gaps)

    clamped = False
    if available is not None:
        max_span = max(available, min_span)
        if raw_span > max_span and raw_span > 0:
            scale = max_span / raw_span
            gaps = [max(min_gap, g * scale) for g in gaps]
            clamped = True
        span_now = sum(gaps)
        if span_now > max_span:
            if k > 1:
                gaps = [min_gap] * (k - 1)
                leftover = max_span - min_span
                if leftover > 0:
                    weights = [max(0.0, g - min_gap) for g in gaps]
                    s = sum(weights)
                    if s > 0:
                        gaps = [min_gap + leftover * w / s for w in weights]
            clamped = True

    times = [observed_dt]
    curr = observed_dt
    for g in gaps:
        curr = curr + pd.Timedelta(seconds=float(g))
        times.append(curr)

    right_bound = right_ts if right_idx is not None else None
    times = enforce_strictly_increasing(times, frac_digits, left_bound=observed_dt, right_bound=right_bound)

    if available is not None and times[-1] > right_ts:
        clamped = True
        step_ns = precision_unit_ns(frac_digits)
        delta = pd.Timedelta(step_ns, unit="ns") if step_ns < 1_000_000_000 else pd.Timedelta(seconds=1)
        latest_start = observed_dt
        latest_end = right_ts
        if latest_start + delta * (k - 1) > latest_end:
            latest_end = latest_start + delta * (k - 1)
        times = [latest_start + delta * i for i in range(k)]
        times = enforce_strictly_increasing(times, frac_digits, left_bound=observed_dt, right_bound=right_ts)

    for row_idx in block_rows:
        repaired_mask[row_idx] = True

    first_row = ordered_rows[0]
    out_ts[first_row] = observed_str
    repaired_mask[first_row] = False

    for rank, row_idx in enumerate(ordered_rows[1:], start=1):
        out_ts[row_idx] = format_ts(times[rank], frac_digits)
        repaired_mask[row_idx] = True

    stats["c1_blocks_repaired"] += 1
    if clamped:
        stats["blocks_clamped"] += 1

    return {
        "block_rows": block_rows,
        "ordered_rows": ordered_rows,
        "observed_str": observed_str,
        "observed_dt": observed_dt,
        "frac_digits": frac_digits,
    }


def classify_and_repair_c2(df, i, case_idxs, orig_ts, parsed_ts, out_ts, repaired_mask):
    pos = case_idxs.index(i)
    if pos == 0:
        return
    prev_i = case_idxs[pos - 1]
    if df.at[prev_i, "Case"] != df.at[i, "Case"]:
        return

    t_prev = safe_to_datetime(out_ts[prev_i]) if repaired_mask[prev_i] else parsed_ts[prev_i]
    t_curr = parsed_ts[i]
    if pd.isna(t_prev) or pd.isna(t_curr):
        return

    if t_curr >= t_prev:
        return

    left_idx = case_idxs[pos - 2] if pos - 2 >= 0 else None
    right_idx = case_idxs[pos + 1] if pos + 1 < len(case_idxs) else None

    left_act = df.at[left_idx, "Activity"] if left_idx is not None else None
    prev_act = df.at[prev_i, "Activity"]
    curr_act = df.at[i, "Activity"]
    right_act = df.at[right_idx, "Activity"] if right_idx is not None else None

    left_ts = safe_to_datetime(out_ts[left_idx]) if (left_idx is not None and repaired_mask[left_idx]) else (parsed_ts[left_idx] if left_idx is not None else pd.NaT)
    right_ts = parsed_ts[right_idx] if right_idx is not None else pd.NaT

    frac_digits = max(infer_fraction_digits(orig_ts[prev_i]), infer_fraction_digits(orig_ts[i]))

    swap_prev = t_curr
    swap_curr = t_prev
    swap_valid = swap_prev <= swap_curr
    if not pd.isna(left_ts):
        swap_valid = swap_valid and (swap_prev >= left_ts)
    if not pd.isna(right_ts):
        swap_valid = swap_valid and (swap_curr <= right_ts)

    swap_pen = 0.0
    if swap_valid:
        if left_act is not None and not pd.isna(left_ts):
            swap_pen += duration_penalty(left_act, prev_act, (swap_prev - left_ts).total_seconds())
        swap_pen += duration_penalty(prev_act, curr_act, (swap_curr - swap_prev).total_seconds())
        if right_act is not None and not pd.isna(right_ts):
            swap_pen += duration_penalty(curr_act, right_act, (right_ts - swap_curr).total_seconds())
    else:
        swap_pen = 1e12

    disp_pen_prev_fixed = 0.0
    est_curr = t_prev + pd.Timedelta(seconds=max(duration_quantile(prev_act, curr_act), 1.0))
    if not pd.isna(right_ts) and est_curr > right_ts:
        room = (right_ts - t_prev).total_seconds()
        if room > 0:
            target = t_prev + pd.Timedelta(seconds=room / 2.0)
            est_curr = target
        else:
            est_curr = t_prev
    if not pd.isna(right_ts):
        est_curr = min(est_curr, right_ts)
    est_curr = max(est_curr, t_prev)
    if left_act is not None and not pd.isna(left_ts):
        disp_pen_prev_fixed += duration_penalty(left_act, prev_act, (t_prev - left_ts).total_seconds())
    disp_pen_prev_fixed += duration_penalty(prev_act, curr_act, (est_curr - t_prev).total_seconds())
    if right_act is not None and not pd.isna(right_ts):
        disp_pen_prev_fixed += duration_penalty(curr_act, right_act, (right_ts - est_curr).total_seconds())

    est_prev = None
    disp_pen_curr_fixed = 1e12
    if (right_idx is None) or (not pd.isna(t_curr) and (pd.isna(right_ts) or t_curr <= right_ts)):
        est_prev_candidate = t_curr - pd.Timedelta(seconds=max(duration_quantile(prev_act, curr_act), 1.0))
        if not pd.isna(left_ts) and est_prev_candidate < left_ts:
            room = (t_curr - left_ts).total_seconds()
            if room > 0:
                est_prev_candidate = left_ts + pd.Timedelta(seconds=room / 2.0)
            else:
                est_prev_candidate = left_ts
        if not pd.isna(left_ts):
            est_prev_candidate = max(est_prev_candidate, left_ts)
        est_prev_candidate = min(est_prev_candidate, t_curr)
        disp_pen_curr_fixed = 0.0
        if left_act is not None and not pd.isna(left_ts):
            disp_pen_curr_fixed += duration_penalty(left_act, prev_act, (est_prev_candidate - left_ts).total_seconds())
        disp_pen_curr_fixed += duration_penalty(prev_act, curr_act, (t_curr - est_prev_candidate).total_seconds())
        if right_act is not None and not pd.isna(right_ts):
            disp_pen_curr_fixed += duration_penalty(curr_act, right_act, (right_ts - t_curr).total_seconds())
        est_prev = est_prev_candidate

    if swap_pen <= min(disp_pen_prev_fixed, disp_pen_curr_fixed) * 0.95:
        out_ts[prev_i] = format_ts(round_to_precision(swap_prev, frac_digits), infer_fraction_digits(orig_ts[prev_i]))
        out_ts[i] = format_ts(round_to_precision(swap_curr, frac_digits), infer_fraction_digits(orig_ts[i]))
        repaired_mask[prev_i] = True
        repaired_mask[i] = True
    else:
        if disp_pen_prev_fixed <= disp_pen_curr_fixed:
            final = round_to_precision(est_curr, infer_fraction_digits(orig_ts[i]))
            if final <= t_prev:
                unit = precision_unit_ns(infer_fraction_digits(orig_ts[i]))
                final = t_prev + (pd.Timedelta(unit, unit="ns") if unit < 1_000_000_000 else pd.Timedelta(seconds=1))
                if not pd.isna(right_ts) and final > right_ts:
                    final = right_ts
            out_ts[i] = format_ts(final, infer_fraction_digits(orig_ts[i]))
            repaired_mask[i] = True
        else:
            final = round_to_precision(est_prev, infer_fraction_digits(orig_ts[prev_i]))
            if final > t_curr:
                unit = precision_unit_ns(infer_fraction_digits(orig_ts[prev_i]))
                final = t_curr - (pd.Timedelta(unit, unit="ns") if unit < 1_000_000_000 else pd.Timedelta(seconds=1))
                if not pd.isna(left_ts) and final < left_ts:
                    final = left_ts
            out_ts[prev_i] = format_ts(final, infer_fraction_digits(orig_ts[prev_i]))
            repaired_mask[prev_i] = True


def repair_c3(df, i, case_idxs, orig_ts, parsed_ts, out_ts, repaired_mask):
    pos = case_idxs.index(i)
    left_idx = case_idxs[pos - 1] if pos - 1 >= 0 else None
    right_idx = case_idxs[pos + 1] if pos + 1 < len(case_idxs) else None

    left_ts = safe_to_datetime(out_ts[left_idx]) if (left_idx is not None and repaired_mask[left_idx]) else (parsed_ts[left_idx] if left_idx is not None else pd.NaT)
    right_ts = parsed_ts[right_idx] if right_idx is not None else pd.NaT

    left_act = df.at[left_idx, "Activity"] if left_idx is not None else None
    act = df.at[i, "Activity"]
    right_act = df.at[right_idx, "Activity"] if right_idx is not None else None

    raw = orig_ts[i]
    frac_digits = infer_fraction_digits(raw) if raw else 0

    date_only = False
    date_part = None
    if isinstance(raw, str) and raw.strip():
        s = raw.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            date_only = True
            date_part = s

    estimate = None
    if left_idx is not None and not pd.isna(left_ts):
        estimate = left_ts + pd.Timedelta(seconds=max(duration_quantile(left_act, act), 1.0))
    if right_idx is not None and not pd.isna(right_ts):
        back_est = right_ts - pd.Timedelta(seconds=max(duration_quantile(act, right_act), 1.0))
        if estimate is None or pd.isna(estimate):
            estimate = back_est
        else:
            lo = max(left_ts, back_est) if not pd.isna(left_ts) else back_est
            hi = min(right_ts, estimate) if not pd.isna(right_ts) else estimate
            if left_ts <= right_ts:
                estimate = left_ts + (right_ts - left_ts) / 2
                if estimate < left_ts:
                    estimate = left_ts
                if estimate > right_ts:
                    estimate = right_ts

    if estimate is None or pd.isna(estimate):
        estimate = pd.Timestamp("2000-01-01 00:00:00")
        if left_idx is not None and not pd.isna(left_ts):
            estimate = left_ts
        elif right_idx is not None and not pd.isna(right_ts):
            estimate = right_ts

    if date_only and date_part is not None:
        base_date = pd.Timestamp(date_part)
        tod_seconds = estimate.hour * 3600 + estimate.minute * 60 + estimate.second + estimate.microsecond / 1e6
        estimate = base_date + pd.Timedelta(seconds=tod_seconds)

        if left_idx is not None and not pd.isna(left_ts) and estimate < left_ts and left_ts.date() == base_date.date():
            estimate = left_ts
        if right_idx is not None and not pd.isna(right_ts) and estimate > right_ts and right_ts.date() == base_date.date():
            estimate = right_ts

        day_start = base_date
        day_end = base_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        if estimate < day_start:
            estimate = day_start
        if estimate > day_end:
            estimate = day_end

    else:
        if left_idx is not None and not pd.isna(left_ts) and estimate < left_ts:
            estimate = left_ts
        if right_idx is not None and not pd.isna(right_ts) and estimate > right_ts:
            if left_idx is not None and not pd.isna(left_ts) and left_ts <= right_ts:
                estimate = left_ts + (right_ts - left_ts) / 2
            else:
                estimate = right_ts

    estimate = round_to_precision(estimate, frac_digits)
    out_ts[i] = format_ts(estimate, frac_digits)
    repaired_mask[i] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_columns = list(df.columns)
    expected = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    n = len(df)
    orig_ts = df["Timestamp"].fillna("").astype(str).tolist()
    parsed_ts = [safe_to_datetime(x) if str(x).strip() != "" else pd.NaT for x in orig_ts]
    out_ts = orig_ts[:]
    repaired_mask = [False] * n

    case_to_idxs = defaultdict(list)
    for i, c in enumerate(df["Case"].tolist()):
        case_to_idxs[c].append(i)

    stats = {"c1_blocks_repaired": 0, "blocks_clamped": 0}
    c1_infos = []

    for case, idxs in case_to_idxs.items():
        j = 0
        while j < len(idxs):
            i = idxs[j]
            cat = df.at[i, "error_category"]
            if cat == "C1_ORDERING_AMBIGUITY":
                block = [i]
                obs = orig_ts[i]
                j2 = j + 1
                while j2 < len(idxs):
                    ii = idxs[j2]
                    if df.at[ii, "error_category"] == "C1_ORDERING_AMBIGUITY" and orig_ts[ii] == obs:
                        block.append(ii)
                        j2 += 1
                    else:
                        break
                if len(block) >= 2:
                    info = c1_repair_case_block(df, block, idxs, orig_ts, parsed_ts, out_ts, repaired_mask, stats)
                    c1_infos.append(info)
                else:
                    frac = infer_fraction_digits(orig_ts[i])
                    base = parsed_ts[i]
                    left = idxs[j - 1] if j > 0 else None
                    right = idxs[j + 1] if j + 1 < len(idxs) else None
                    if left is not None and not pd.isna(parsed_ts[left]):
                        base = parsed_ts[left] + pd.Timedelta(seconds=1)
                    if right is not None and not pd.isna(parsed_ts[right]) and (pd.isna(base) or base >= parsed_ts[right]):
                        base = parsed_ts[right] - pd.Timedelta(seconds=1)
                    base = round_to_precision(base, frac)
                    if format_ts(base, frac) == orig_ts[i]:
                        base = base + pd.Timedelta(seconds=1)
                    out_ts[i] = format_ts(base, frac)
                    repaired_mask[i] = True
                j = j2
            else:
                j += 1

    for case, idxs in case_to_idxs.items():
        for i in idxs:
            if df.at[i, "error_category"] == "C2_TEMPORAL_INCONSISTENCY":
                classify_and_repair_c2(df, i, idxs, orig_ts, parsed_ts, out_ts, repaired_mask)

    for case, idxs in case_to_idxs.items():
        for i in idxs:
            if df.at[i, "error_category"] == "C3_MISSING_TEMPORAL_INFORMATION":
                repair_c3(df, i, idxs, orig_ts, parsed_ts, out_ts, repaired_mask)

    for case, idxs in case_to_idxs.items():
        current = [safe_to_datetime(out_ts[i]) if out_ts[i] != "" else pd.NaT for i in idxs]
        case_start = min([t for t in current if not pd.isna(t)], default=pd.NaT)
        case_end = max([t for t in current if not pd.isna(t)], default=pd.NaT)
        for pos, i in enumerate(idxs):
            if df.at[i, "error_category"] == "CLEAN":
                continue
            t = safe_to_datetime(out_ts[i]) if out_ts[i] != "" else pd.NaT
            frac = infer_fraction_digits(orig_ts[i])
            left_t = current[pos - 1] if pos > 0 else pd.NaT
            right_t = current[pos + 1] if pos + 1 < len(idxs) else pd.NaT
            if pd.isna(t):
                if not pd.isna(left_t):
                    t = left_t
                elif not pd.isna(right_t):
                    t = right_t
                else:
                    t = case_start if not pd.isna(case_start) else pd.Timestamp("2000-01-01 00:00:00")
            if not pd.isna(case_start) and t < case_start:
                t = case_start
            if not pd.isna(case_end) and t > case_end:
                t = case_end
            if not pd.isna(left_t) and t < left_t:
                t = left_t
            if not pd.isna(right_t) and t > right_t:
                if not pd.isna(left_t) and left_t <= right_t:
                    t = left_t + (right_t - left_t) / 2
                else:
                    t = right_t
            out_ts[i] = format_ts(round_to_precision(t, frac), frac)
            current[pos] = safe_to_datetime(out_ts[i])

        changed = True
        while changed:
            changed = False
            current = [safe_to_datetime(out_ts[i]) if out_ts[i] != "" else pd.NaT for i in idxs]
            for pos in range(1, len(idxs)):
                i_prev = idxs[pos - 1]
                i_curr = idxs[pos]
                t_prev = current[pos - 1]
                t_curr = current[pos]
                if pd.isna(t_prev) or pd.isna(t_curr):
                    continue
                if t_curr < t_prev:
                    if df.at[i_curr, "error_category"] != "CLEAN":
                        frac = infer_fraction_digits(orig_ts[i_curr])
                        unit = precision_unit_ns(frac)
                        new_t = t_prev + (pd.Timedelta(unit, unit="ns") if unit < 1_000_000_000 else pd.Timedelta(seconds=1))
                        right_t = current[pos + 1] if pos + 1 < len(idxs) else pd.NaT
                        if not pd.isna(right_t) and new_t > right_t:
                            if df.at[i_prev, "error_category"] != "CLEAN":
                                fracp = infer_fraction_digits(orig_ts[i_prev])
                                unitp = precision_unit_ns(fracp)
                                prev_new = t_curr - (pd.Timedelta(unitp, unit="ns") if unitp < 1_000_000_000 else pd.Timedelta(seconds=1))
                                left_t = current[pos - 2] if pos - 2 >= 0 else pd.NaT
                                if not pd.isna(left_t) and prev_new < left_t:
                                    prev_new = left_t
                                out_ts[i_prev] = format_ts(round_to_precision(prev_new, fracp), fracp)
                                current[pos - 1] = safe_to_datetime(out_ts[i_prev])
                                changed = True
                            else:
                                new_t = right_t
                                if new_t < t_prev:
                                    new_t = t_prev
                        out_ts[i_curr] = format_ts(round_to_precision(new_t, frac), frac)
                        current[pos] = safe_to_datetime(out_ts[i_curr])
                        changed = True
                    elif df.at[i_prev, "error_category"] != "CLEAN":
                        frac = infer_fraction_digits(orig_ts[i_prev])
                        unit = precision_unit_ns(frac)
                        new_t = t_curr - (pd.Timedelta(unit, unit="ns") if unit < 1_000_000_000 else pd.Timedelta(seconds=1))
                        left_t = current[pos - 2] if pos - 2 >= 0 else pd.NaT
                        if not pd.isna(left_t) and new_t < left_t:
                            new_t = left_t
                        out_ts[i_prev] = format_ts(round_to_precision(new_t, frac), frac)
                        current[pos - 1] = safe_to_datetime(out_ts[i_prev])
                        changed = True

    for info in c1_infos:
        block_rows = info["block_rows"]
        ordered_rows = info["ordered_rows"]
        obs_str = info["observed_str"]
        obs_dt = info["observed_dt"]
        frac = info["frac_digits"]
        vals = [safe_to_datetime(out_ts[r]) for r in ordered_rows]
        vals[0] = obs_dt
        vals = enforce_strictly_increasing(vals, frac, left_bound=obs_dt, right_bound=max(vals[-1], obs_dt))
        out_ts[ordered_rows[0]] = obs_str
        for r, t in zip(ordered_rows[1:], vals[1:]):
            s = format_ts(t, frac)
            if s == orig_ts[r]:
                unit = precision_unit_ns(frac)
                t2 = t + (pd.Timedelta(unit, unit="ns") if unit < 1_000_000_000 else pd.Timedelta(seconds=1))
                s = format_ts(t2, frac)
            out_ts[r] = s

    rows_still_equal = 0
    for i in range(n):
        if df.at[i, "error_category"] == "CLEAN":
            out_ts[i] = orig_ts[i]
        if out_ts[i] == orig_ts[i] and df.at[i, "error_category"] != "CLEAN":
            rows_still_equal += 1

    df_out = df.copy()
    df_out["Timestamp"] = out_ts
    df_out = df_out[original_columns]
    df_out.to_csv(args.output, index=False)

    print(f"{stats['c1_blocks_repaired']},{stats['blocks_clamped']},{rows_still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        print("0,0,0")