import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

# -----------------------------
# Embedded process knowledge
# -----------------------------
DURATION_PROFILES = {
    ("Request info", "info received"): {"p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0},
    ("info received", "Check for completeness"): {"p10": 62.6256, "p25": 174.0865, "median": 415.276, "share_under_60s": 0.096},
    ("Check for completeness", "Request info"): {"p10": 61.3589, "p25": 170.3105, "median": 417.4255, "share_under_60s": 0.099},
    ("Check for completeness", "Perform checks"): {"p10": 58.2756, "p25": 166.811, "median": 414.721, "share_under_60s": 0.103},
    ("Perform checks", "Make decision"): {"p10": 63.6329, "p25": 171.37975, "median": 423.057, "share_under_60s": 0.094},
    ("New online application received", "Check for completeness"): {"p10": 66.1590, "p25": 172.4395, "median": 421.0585, "share_under_60s": 0.091},
    ("Deliver card", "EVENT 13 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("Notify accept", "Deliver card"): {"p10": 59.4585, "p25": 176.35075, "median": 425.762, "share_under_60s": 0.101},
    ("Make decision", "notify reject"): {"p10": 64.22, "p25": 175.774, "median": 421.641, "share_under_60s": 0.095},
    ("Make decision", "Notify accept"): {"p10": 65.4006, "p25": 172.952, "median": 426.665, "share_under_60s": 0.091},
    ("time out", "EVENT 13 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("notify reject", "time out"): {"p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0},
    ("notify reject", "review request received"): {"p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0},
    ("review request received", "Check for completeness"): {"p10": 74.1377, "p25": 186.74975, "median": 434.7365, "share_under_60s": 0.079},
    ("Check for completeness", "Make decision"): {"p10": 336.7619, "p25": 585.2495, "median": 1035.1695, "share_under_60s": 0.0},
    ("Perform checks", "Notify accept"): {"p10": 365.2176, "p25": 560.366, "median": 925.236, "share_under_60s": 0.0},
    ("Check for completeness", "info received"): {"p10": 133.281, "p25": 244.0815, "median": 455.3365, "share_under_60s": 0.0},
    ("Perform checks", "notify reject"): {"p10": 456.3862, "p25": 714.8795, "median": 1130.353, "share_under_60s": 0.0},
    ("info received", "Perform checks"): {"p10": 378.526, "p25": 613.034, "median": 949.19, "share_under_60s": 0.0},
    ("info received", "Request info"): {"p10": 436.676, "p25": 612.23, "median": 1057.673, "share_under_60s": 0.0},
    ("Make decision", "review request received"): {"p10": 126.7211, "p25": 263.86525, "median": 523.334, "share_under_60s": 0.0},
    ("review request received", "Perform checks"): {"p10": 340.5695, "p25": 510.8235, "median": 881.5615, "share_under_60s": 0.0},
    ("review request received", "Request info"): {"p10": 464.9404, "p25": 603.8945, "median": 1077.302, "share_under_60s": 0.0},
    ("EVENT 13 END", "EVENT 13 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
}

DFG_EDGES = set(DURATION_PROFILES.keys())

BLOCK_SPAN_PROFILE = {
    2: {"p10": 10.0, "p25": 10.0, "median": 241.935},
    3: {"p10": 105.8982, "p25": 323.548, "median": 736.099},
    4: {"p10": 382.8938, "p25": 705.372, "median": 1219.731},
    5: {"p10": 668.8492, "p25": 1079.68, "median": 1705.245},
}
GLOBAL_MEDIAN = 241.935

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

# -----------------------------
# Helpers
# -----------------------------
def detect_precision(ts_str: str) -> int:
    if "." not in ts_str:
        return 0
    frac = ts_str.split(".")[-1]
    return len(frac)

def parse_ts(ts_str: str):
    return pd.to_datetime(ts_str)

def fmt_ts(ts: pd.Timestamp, precision: int) -> str:
    ts = pd.Timestamp(ts)
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    frac = f"{ts.microsecond:06d}"[:precision]
    return f"{base}.{frac}"

def expected_gap_seconds(a, b):
    prof = DURATION_PROFILES.get((a, b))
    if prof is None:
        return GLOBAL_MEDIAN
    if prof["share_under_60s"] >= 0.9 or prof["p25"] in (0.0, 10.0):
        return prof["p10"]
    return prof["p25"]

def edge_exists(a, b):
    return (a, b) in DFG_EDGES

def local_variant_support(acts):
    for v in FREQUENT_VARIANTS:
        for i in range(len(v) - len(acts) + 1):
            if v[i:i + len(acts)] == acts:
                return True
    return False

def score_local_sequence(prev_act, act1, act2, next_act):
    score = 0.0
    if prev_act is not None:
        if edge_exists(prev_act, act1):
            score += 2.0
        else:
            score -= 10.0
    if edge_exists(act1, act2):
        score += 3.0
    else:
        score -= 12.0
    if next_act is not None:
        if edge_exists(act2, next_act):
            score += 2.0
        else:
            score -= 10.0
    seq = [x for x in [prev_act, act1, act2, next_act] if x is not None]
    if len(seq) >= 2 and local_variant_support(seq):
        score += 2.0
    return score

def duration_error(prev_act, act1, act2, next_act, prev_t, t1, t2, next_t):
    err = 0.0
    count = 0
    if prev_act is not None and prev_t is not None and edge_exists(prev_act, act1):
        gap = max(0.0, (t1 - prev_t).total_seconds())
        exp = expected_gap_seconds(prev_act, act1)
        err += abs(gap - exp)
        count += 1
    if edge_exists(act1, act2):
        gap = max(0.0, (t2 - t1).total_seconds())
        exp = expected_gap_seconds(act1, act2)
        err += abs(gap - exp)
        count += 1
    if next_act is not None and next_t is not None and edge_exists(act2, next_act):
        gap = max(0.0, (next_t - t2).total_seconds())
        exp = expected_gap_seconds(act2, next_act)
        err += abs(gap - exp)
        count += 1
    if count == 0:
        return 1e18
    return err / count

def minimal_epsilon(precision):
    if precision <= 0:
        return pd.Timedelta(seconds=1)
    if precision <= 6:
        return pd.Timedelta(microseconds=10 ** (6 - precision))
    return pd.Timedelta(microseconds=1)

def choose_inside_bounds(cands, lower, upper, original_ts, precision):
    eps = minimal_epsilon(precision)
    filtered = []
    for c in cands:
        c = pd.Timestamp(c)
        if lower is not None and c < lower:
            c = lower
        if upper is not None and c > upper:
            c = upper
        filtered.append(c)

    valid = []
    for c in filtered:
        s = fmt_ts(c, precision)
        if s != original_ts:
            valid.append(c)
    if valid:
        valid.sort(key=lambda x: abs((x - parse_ts(original_ts)).total_seconds()))
        return valid[0]

    base = parse_ts(original_ts)
    trial = []
    if lower is not None:
        trial.append(pd.Timestamp(lower))
        trial.append(pd.Timestamp(lower) + eps)
    if upper is not None:
        trial.append(pd.Timestamp(upper))
        trial.append(pd.Timestamp(upper) - eps)
    trial.extend(filtered)

    best = None
    best_dist = float("inf")
    for c in trial:
        cc = pd.Timestamp(c)
        if lower is not None and cc < lower:
            continue
        if upper is not None and cc > upper:
            continue
        if fmt_ts(cc, precision) == original_ts:
            continue
        d = abs((cc - base).total_seconds())
        if d < best_dist:
            best = cc
            best_dist = d
    if best is not None:
        return best

    # final deterministic fallback within bounds
    if lower is not None:
        cc = pd.Timestamp(lower)
        if fmt_ts(cc, precision) != original_ts:
            return cc
        return cc + eps
    if upper is not None:
        cc = pd.Timestamp(upper)
        if fmt_ts(cc, precision) != original_ts:
            return cc
        return cc - eps
    return base + eps

def estimate_one_row(prev_act, act, next_act, prev_t, next_t, observed_ts, precision, case_start, case_end):
    lower = case_start
    upper = case_end
    if prev_t is not None:
        lower = max(lower, prev_t)
    if next_t is not None:
        upper = min(upper, next_t)

    cands = []
    if prev_t is not None and edge_exists(prev_act, act):
        cands.append(prev_t + pd.Timedelta(seconds=expected_gap_seconds(prev_act, act)))
    if next_t is not None and edge_exists(act, next_act):
        cands.append(next_t - pd.Timedelta(seconds=expected_gap_seconds(act, next_act)))

    if prev_t is not None and next_t is not None and not cands:
        span = (next_t - prev_t).total_seconds()
        cands.append(prev_t + pd.Timedelta(seconds=max(0.0, min(span, BLOCK_SPAN_PROFILE.get(2, {}).get("p25", GLOBAL_MEDIAN)))))
    elif prev_t is not None and not cands:
        cands.append(prev_t + pd.Timedelta(seconds=GLOBAL_MEDIAN))
    elif next_t is not None and not cands:
        cands.append(next_t - pd.Timedelta(seconds=GLOBAL_MEDIAN))

    if len(cands) >= 2:
        avg_ns = int(sum(c.value for c in cands) / len(cands))
        cands.insert(0, pd.Timestamp(avg_ns))

    return choose_inside_bounds(cands, lower, upper, observed_ts, precision)

def repair_displaced(case_df, idx_bad, output_ts, orig_ts, parsed_ts, precision_map):
    n = len(case_df)
    local_idx = idx_bad
    row = case_df.iloc[local_idx]
    act = row["Activity"]
    obs_ts = orig_ts[row.name]
    prec = precision_map[row.name]

    prev_t = parsed_ts[case_df.iloc[local_idx - 1].name] if local_idx - 1 >= 0 else None
    prev_act = case_df.iloc[local_idx - 1]["Activity"] if local_idx - 1 >= 0 else None
    next_t = parsed_ts[case_df.iloc[local_idx + 1].name] if local_idx + 1 < n else None
    next_act = case_df.iloc[local_idx + 1]["Activity"] if local_idx + 1 < n else None

    case_start = min(parsed_ts[r.name] for _, r in case_df.iterrows())
    case_end = max(parsed_ts[r.name] for _, r in case_df.iterrows())

    est = estimate_one_row(prev_act, act, next_act, prev_t, next_t, obs_ts, prec, case_start, case_end)
    output_ts[row.name] = fmt_ts(est, prec)
    parsed_ts[row.name] = est

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
    orig_ts = df["Timestamp"].astype(str).to_dict()
    precision_map = {idx: detect_precision(ts) for idx, ts in orig_ts.items()}
    parsed_ts = {idx: parse_ts(ts) for idx, ts in orig_ts.items()}
    output_ts = dict(orig_ts)

    exchanged_segments = 0
    displaced_segments = 0

    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        case_df = df.loc[list(case_idx)]
        case_df = case_df.reset_index().rename(columns={"index": "_orig_index"})
        n = len(case_df)

        i = 1
        while i < n:
            prev_row = case_df.iloc[i - 1]
            curr_row = case_df.iloc[i]
            prev_idx = prev_row["_orig_index"]
            curr_idx = curr_row["_orig_index"]

            t_prev = parsed_ts[prev_idx]
            t_curr = parsed_ts[curr_idx]

            if t_curr < t_prev and (
                prev_row["error_category"] != "CLEAN" or curr_row["error_category"] != "CLEAN"
            ):
                prev2_act = case_df.iloc[i - 2]["Activity"] if i - 2 >= 0 else None
                next1_act = case_df.iloc[i + 1]["Activity"] if i + 1 < n else None
                prev_act = prev_row["Activity"]
                curr_act = curr_row["Activity"]

                prev2_t = parsed_ts[case_df.iloc[i - 2]["_orig_index"]] if i - 2 >= 0 else None
                next1_t = parsed_ts[case_df.iloc[i + 1]["_orig_index"]] if i + 1 < n else None

                # Test exchanged: exact swap of the two observed values.
                swapped_t_prev = parsed_ts[curr_idx]
                swapped_t_curr = parsed_ts[prev_idx]

                swap_valid_order = True
                if prev2_t is not None and swapped_t_prev < prev2_t:
                    swap_valid_order = False
                if swapped_t_curr < swapped_t_prev:
                    swap_valid_order = False
                if next1_t is not None and swapped_t_curr > next1_t:
                    swap_valid_order = False

                score_keep = score_local_sequence(prev2_act, prev_act, curr_act, next1_act)
                score_swap = score_keep  # same activity order; DFG score identical under this plan

                err_keep = duration_error(prev2_act, prev_act, curr_act, next1_act, prev2_t, t_prev, t_curr, next1_t)
                err_swap = duration_error(prev2_act, prev_act, curr_act, next1_act, prev2_t, swapped_t_prev, swapped_t_curr, next1_t)

                both_change_if_swap = (output_ts[prev_idx] != orig_ts[curr_idx] or prev_idx != curr_idx) and (output_ts[curr_idx] != orig_ts[prev_idx] or prev_idx != curr_idx)
                exact_swap_changes = (orig_ts[prev_idx] != orig_ts[curr_idx])

                accept_swap = (
                    swap_valid_order and
                    exact_swap_changes and
                    score_swap >= score_keep - 0.1 and
                    err_swap + 1e-9 < err_keep
                )

                if accept_swap:
                    output_ts[prev_idx] = orig_ts[curr_idx]
                    output_ts[curr_idx] = orig_ts[prev_idx]
                    parsed_ts[prev_idx] = swapped_t_prev
                    parsed_ts[curr_idx] = swapped_t_curr
                    exchanged_segments += 1
                else:
                    # displaced: try fixing exactly one row, leave partner unchanged
                    case_start = min(parsed_ts[r["_orig_index"]] for _, r in case_df.iterrows())
                    case_end = max(parsed_ts[r["_orig_index"]] for _, r in case_df.iterrows())

                    candidates = []

                    # fix prev row only
                    if prev_row["error_category"] != "CLEAN":
                        prec = precision_map[prev_idx]
                        est_prev = estimate_one_row(
                            prev2_act, prev_act, curr_act,
                            prev2_t, t_curr, orig_ts[prev_idx], prec,
                            case_start, case_end
                        )
                        if (prev2_t is None or est_prev >= prev2_t) and est_prev <= t_curr:
                            err = duration_error(prev2_act, prev_act, curr_act, next1_act, prev2_t, est_prev, t_curr, next1_t)
                            move = abs((est_prev - t_prev).total_seconds())
                            candidates.append(("prev", est_prev, err, move))

                    # fix curr row only
                    if curr_row["error_category"] != "CLEAN":
                        prec = precision_map[curr_idx]
                        est_curr = estimate_one_row(
                            prev_act, curr_act, next1_act,
                            t_prev, next1_t, orig_ts[curr_idx], prec,
                            case_start, case_end
                        )
                        if est_curr >= t_prev and (next1_t is None or est_curr <= next1_t):
                            err = duration_error(prev2_act, prev_act, curr_act, next1_act, prev2_t, t_prev, est_curr, next1_t)
                            move = abs((est_curr - t_curr).total_seconds())
                            candidates.append(("curr", est_curr, err, move))

                    if candidates:
                        candidates.sort(key=lambda x: (x[2], x[3]))
                        side, est, _, _ = candidates[0]
                        if side == "prev":
                            output_ts[prev_idx] = fmt_ts(est, precision_map[prev_idx])
                            parsed_ts[prev_idx] = est
                        else:
                            output_ts[curr_idx] = fmt_ts(est, precision_map[curr_idx])
                            parsed_ts[curr_idx] = est
                        displaced_segments += 1
                    else:
                        # robust fallback: repair whichever corrupted member exists with nearest valid anchor-bounded estimate
                        if curr_row["error_category"] != "CLEAN":
                            repair_displaced(case_df.set_index("_orig_index", drop=False), i, output_ts, orig_ts, parsed_ts, precision_map)
                        elif prev_row["error_category"] != "CLEAN":
                            repair_displaced(case_df.set_index("_orig_index", drop=False), i - 1, output_ts, orig_ts, parsed_ts, precision_map)
                        displaced_segments += 1
            i += 1

    # Self-check and corrective pass
    fixes = []
    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        times = [parse_ts(output_ts[idx]) for idx in idxs]
        case_start = min(times)
        case_end = max(times)

        for pos, idx in enumerate(idxs):
            if df.at[idx, "error_category"] == "CLEAN":
                output_ts[idx] = orig_ts[idx]
                parsed_ts[idx] = parse_ts(output_ts[idx])

        times = [parse_ts(output_ts[idx]) for idx in idxs]
        for pos, idx in enumerate(idxs):
            if df.at[idx, "error_category"] != "CLEAN":
                t = parse_ts(output_ts[idx])
                lower = case_start
                upper = case_end
                if pos - 1 >= 0:
                    lower = max(lower, parse_ts(output_ts[idxs[pos - 1]]))
                if pos + 1 < len(idxs):
                    upper = min(upper, parse_ts(output_ts[idxs[pos + 1]]))
                new_t = choose_inside_bounds([t], lower, upper, orig_ts[idx], precision_map[idx])
                new_s = fmt_ts(new_t, precision_map[idx])
                if new_s != output_ts[idx]:
                    fixes.append(f"adjusted_row_{idx}")
                    output_ts[idx] = new_s
                    parsed_ts[idx] = new_t

        # enforce non-decreasing within case by adjusting corrupted rows only
        for pos in range(1, len(idxs)):
            left = idxs[pos - 1]
            right = idxs[pos]
            tl = parse_ts(output_ts[left])
            tr = parse_ts(output_ts[right])
            if tr < tl:
                if df.at[right, "error_category"] != "CLEAN":
                    lower = tl
                    upper = parse_ts(output_ts[idxs[pos + 1]]) if pos + 1 < len(idxs) else case_end
                    new_t = choose_inside_bounds([lower], lower, upper, orig_ts[right], precision_map[right])
                    output_ts[right] = fmt_ts(new_t, precision_map[right])
                    parsed_ts[right] = new_t
                    fixes.append(f"monotone_fix_{right}")
                elif df.at[left, "error_category"] != "CLEAN":
                    lower = parse_ts(output_ts[idxs[pos - 2]]) if pos - 2 >= 0 else case_start
                    upper = tr
                    new_t = choose_inside_bounds([upper], lower, upper, orig_ts[left], precision_map[left])
                    output_ts[left] = fmt_ts(new_t, precision_map[left])
                    parsed_ts[left] = new_t
                    fixes.append(f"monotone_fix_{left}")

    # final guarantee: every corrupted row changed from input
    for idx in df.index:
        if df.at[idx, "error_category"] != "CLEAN" and output_ts[idx] == orig_ts[idx]:
            # estimate from same-case anchors
            case = df.at[idx, "Case"]
            case_rows = list(df.index[df["Case"] == case])
            pos = case_rows.index(idx)
            prev_idx = case_rows[pos - 1] if pos > 0 else None
            next_idx = case_rows[pos + 1] if pos + 1 < len(case_rows) else None
            prev_t = parse_ts(output_ts[prev_idx]) if prev_idx is not None else None
            next_t = parse_ts(output_ts[next_idx]) if next_idx is not None else None
            prev_act = df.at[prev_idx, "Activity"] if prev_idx is not None else None
            next_act = df.at[next_idx, "Activity"] if next_idx is not None else None
            act = df.at[idx, "Activity"]
            case_times = [parse_ts(output_ts[j]) for j in case_rows]
            case_start = min(case_times)
            case_end = max(case_times)
            est = estimate_one_row(prev_act, act, next_act, prev_t, next_t, orig_ts[idx], precision_map[idx], case_start, case_end)
            output_ts[idx] = fmt_ts(est, precision_map[idx])
            parsed_ts[idx] = est
            fixes.append(f"changed_equal_row_{idx}")

    out_df = df.copy()
    out_df["Timestamp"] = [output_ts[idx] for idx in df.index]

    # preserve exact non-timestamp columns and row count/order
    out_df = out_df[original_columns]
    out_df.to_csv(args.output, index=False)

    rows_still_equal = int(((df["error_category"] != "CLEAN") & (out_df["Timestamp"] == df["Timestamp"])).sum())
    print(f"exchanged={exchanged_segments} displaced={displaced_segments} rows_still_equal={rows_still_equal}")

if __name__ == "__main__":
    main()