import argparse
import itertools
import math
import sys
from datetime import timedelta

import pandas as pd


TRANSITION_PROBS = {
    ("Deliver card", "EVENT 13 END"): 1.0,
    ("EVENT 13 END", "EVENT 13 END"): 1.0,
    ("Notify accept", "Deliver card"): 1.0,
    ("time out", "EVENT 13 END"): 1.0,
    ("review request received", "Check for completeness"): 0.9788,
    ("info received", "Check for completeness"): 0.9782,
    ("Request info", "info received"): 0.9746,
    ("Perform checks", "Make decision"): 0.9503,
    ("New online application received", "Check for completeness"): 0.941,
    ("Make decision", "notify reject"): 0.5115,
    ("notify reject", "time out"): 0.5102,
    ("Check for completeness", "Request info"): 0.501,
    ("Make decision", "Notify accept"): 0.4843,
    ("notify reject", "review request received"): 0.4806,
    ("Check for completeness", "Perform checks"): 0.4623,
    ("New online application received", "Perform checks"): 0.0418,
    ("Perform checks", "Notify accept"): 0.0304,
    ("Request info", "Check for completeness"): 0.0254,
    ("Check for completeness", "Make decision"): 0.0253,
    ("Perform checks", "notify reject"): 0.0194,
    ("New online application received", "Request info"): 0.0172,
    ("info received", "Perform checks"): 0.0129,
    ("review request received", "Perform checks"): 0.0121,
    ("Check for completeness", "info received"): 0.0115,
    ("notify reject", "Check for completeness"): 0.0092,
    ("review request received", "Request info"): 0.0091,
    ("info received", "Request info"): 0.009,
    ("Make decision", "review request received"): 0.0042,
}

DURATION_PROFILES = {
    ("Check for completeness", "Request info"): {"p10": 62.6422, "p25": 172.533, "median": 418.656, "share_under_60s": 0.096},
    ("info received", "Check for completeness"): {"p10": 65.0913, "p25": 176.3765, "median": 415.3415, "share_under_60s": 0.092},
    ("Request info", "info received"): {"p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0},
    ("Check for completeness", "Perform checks"): {"p10": 62.596, "p25": 174.17, "median": 418.43, "share_under_60s": 0.096},
    ("Perform checks", "Make decision"): {"p10": 64.8835, "p25": 172.6405, "median": 423.4695, "share_under_60s": 0.092},
    ("New online application received", "Check for completeness"): {"p10": 65.0406, "p25": 171.767, "median": 416.959, "share_under_60s": 0.093},
    ("Deliver card", "EVENT 13 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("Notify accept", "Deliver card"): {"p10": 59.4585, "p25": 176.35075, "median": 425.762, "share_under_60s": 0.101},
    ("Make decision", "notify reject"): {"p10": 64.834, "p25": 175.9765, "median": 426.0405, "share_under_60s": 0.094},
    ("Make decision", "Notify accept"): {"p10": 65.664, "p25": 174.0265, "median": 426.665, "share_under_60s": 0.09},
    ("time out", "EVENT 13 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("notify reject", "time out"): {"p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0},
    ("notify reject", "review request received"): {"p10": 10.0, "p25": 10.0, "median": 10.0, "share_under_60s": 1.0},
    ("review request received", "Check for completeness"): {"p10": 75.725, "p25": 186.74975, "median": 442.53, "share_under_60s": 0.077},
    ("Check for completeness", "Make decision"): {"p10": 337.4222, "p25": 587.122, "median": 962.0195, "share_under_60s": 0.009},
    ("New online application received", "Perform checks"): {"p10": 347.2758, "p25": 596.06, "median": 1029.426, "share_under_60s": 0.005},
    ("Perform checks", "Notify accept"): {"p10": 354.0812, "p25": 548.193, "median": 872.637, "share_under_60s": 0.005},
    ("Request info", "Check for completeness"): {"p10": 81.2658, "p25": 177.8785, "median": 444.7555, "share_under_60s": 0.065},
    ("Check for completeness", "info received"): {"p10": 79.1902, "p25": 193.724, "median": 472.726, "share_under_60s": 0.067},
    ("Perform checks", "notify reject"): {"p10": 454.6746, "p25": 726.628, "median": 1076.069, "share_under_60s": 0.008},
    ("New online application received", "Request info"): {"p10": 351.984, "p25": 510.01375, "median": 884.5265, "share_under_60s": 0.0},
    ("info received", "Perform checks"): {"p10": 407.9895, "p25": 598.8085, "median": 1124.7845, "share_under_60s": 0.0},
    ("info received", "Request info"): {"p10": 414.5244, "p25": 664.2945, "median": 1001.139, "share_under_60s": 0.017},
    ("notify reject", "Check for completeness"): {"p10": 93.056, "p25": 216.3995, "median": 385.786, "share_under_60s": 0.065},
    ("Make decision", "review request received"): {"p10": 121.8776, "p25": 198.772, "median": 412.731, "share_under_60s": 0.0},
    ("review request received", "Perform checks"): {"p10": 550.4103, "p25": 717.58225, "median": 1404.037, "share_under_60s": 0.0},
    ("review request received", "Request info"): {"p10": 315.8568, "p25": 545.983, "median": 1071.631, "share_under_60s": 0.067},
    ("EVENT 13 END", "EVENT 13 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
}

GLOBAL_MEDIAN = 243.7965
SMALL = 1e-9


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def safe_to_datetime(val):
    try:
        if pd.isna(val) or str(val).strip() == "":
            return pd.NaT
        return pd.to_datetime(val, errors="coerce")
    except Exception:
        return pd.NaT


def frac_len_of_string(s):
    if s is None or pd.isna(s):
        return 3
    s = str(s)
    if "." in s:
        tail = s.split(".")[-1]
        tail = "".join(ch for ch in tail if ch.isdigit())
        return max(0, len(tail))
    return 0


def detect_global_precision(strings):
    mx = 0
    for s in strings:
        mx = max(mx, frac_len_of_string(s))
    return mx if mx > 0 else 3


def format_timestamp(ts, frac_len):
    if pd.isna(ts):
        return ""
    py = ts.to_pydatetime()
    base = py.strftime("%Y-%m-%d %H:%M:%S")
    if frac_len <= 0:
        return base
    micro = py.microsecond
    frac = f"{micro:06d}"[:min(frac_len, 6)]
    if frac_len > 6:
        frac = frac + ("0" * (frac_len - 6))
    return f"{base}.{frac}"


def profile_for(src, tgt):
    return DURATION_PROFILES.get((src, tgt))


def transition_prob(src, tgt):
    return TRANSITION_PROBS.get((src, tgt), 1e-6)


def chosen_gap_seconds(src, tgt):
    prof = profile_for(src, tgt)
    if prof:
        if prof["share_under_60s"] >= 0.5:
            return max(1.0, prof["p10"])
        return max(1.0, prof["p25"])
    return max(1.0, GLOBAL_MEDIAN)


def adjacent_activity_gap(activity, side="from"):
    vals = []
    for (s, t), prof in DURATION_PROFILES.items():
        if side == "from" and s == activity:
            vals.append(prof["p10"] if prof["share_under_60s"] >= 0.5 else prof["p25"])
        if side == "to" and t == activity:
            vals.append(prof["p10"] if prof["share_under_60s"] >= 0.5 else prof["p25"])
    if vals:
        vals = sorted(vals)
        return max(1.0, vals[len(vals) // 2])
    return max(1.0, GLOBAL_MEDIAN)


def estimate_from_left(left_ts, left_act, act):
    gap = chosen_gap_seconds(left_act, act) if left_act is not None else adjacent_activity_gap(act, side="to")
    return left_ts + pd.to_timedelta(gap, unit="s")


def estimate_from_right(act, right_act, right_ts):
    gap = chosen_gap_seconds(act, right_act) if right_act is not None else adjacent_activity_gap(act, side="from")
    return right_ts - pd.to_timedelta(gap, unit="s")


def score_gap(src, tgt, delta_seconds):
    prof = profile_for(src, tgt)
    if delta_seconds < -SMALL:
        return -1e9
    if prof is None:
        center = GLOBAL_MEDIAN
        spread = max(30.0, GLOBAL_MEDIAN)
    else:
        center = prof["p10"] if prof["share_under_60s"] >= 0.5 else prof["p25"]
        spread = max(10.0, abs(prof["median"] - center), prof["p25"] if "p25" in prof else 60.0)
    z = abs(delta_seconds - center) / spread
    return -z


def local_swap_score(prev_act, act1, act2, next_act, prev_ts, t1, t2, next_ts):
    score = 0.0
    if prev_ts is not None:
        score += score_gap(prev_act, act1, (t1 - prev_ts).total_seconds())
    score += score_gap(act1, act2, (t2 - t1).total_seconds())
    if next_ts is not None:
        score += score_gap(act2, next_act, (next_ts - t2).total_seconds())
    score += math.log(max(transition_prob(prev_act, act1), 1e-6)) if prev_act is not None else 0.0
    score += math.log(max(transition_prob(act1, act2), 1e-6))
    score += math.log(max(transition_prob(act2, next_act), 1e-6)) if next_act is not None else 0.0
    return score


def clamp_ts(ts, low, high):
    if low is not None and ts < low:
        ts = low
    if high is not None and ts > high:
        ts = high
    return ts


def case_bounds(case_df):
    parsed = case_df["_parsed_ts"]
    valid = parsed.dropna()
    if len(valid) == 0:
        return None, None
    return valid.min(), valid.max()


def nearest_clean_left(case_df, pos):
    for i in range(pos - 1, -1, -1):
        if case_df.iloc[i]["error_category"] == "CLEAN":
            ts = case_df.iloc[i]["_parsed_ts"]
            if pd.notna(ts):
                return i, ts, case_df.iloc[i]["Activity"]
    return None, None, None


def nearest_clean_right(case_df, pos):
    for i in range(pos + 1, len(case_df)):
        if case_df.iloc[i]["error_category"] == "CLEAN":
            ts = case_df.iloc[i]["_parsed_ts"]
            if pd.notna(ts):
                return i, ts, case_df.iloc[i]["Activity"]
    return None, None, None


def date_only_or_zero_time(s):
    if s is None or pd.isna(s):
        return False
    st = str(s).strip()
    if st == "":
        return False
    if len(st) == 10 and st.count("-") == 2:
        return True
    return st.endswith("00:00:00.000") or st.endswith("00:00:00")


def extract_observed_date(s):
    try:
        st = str(s).strip()
        if st == "":
            return None
        return pd.to_datetime(st).date()
    except Exception:
        return None


def choose_balanced_between(left_est, right_est, low, high):
    cands = []
    if left_est is not None:
        cands.append(left_est)
    if right_est is not None:
        cands.append(right_est)
    if left_est is not None and right_est is not None:
        cands.append(left_est + (right_est - left_est) / 2)
    if low is not None and high is not None:
        cands.append(low + (high - low) / 2)
    best = cands[0] if cands else low if low is not None else high
    best = clamp_ts(best, low, high)
    return best


def order_score(acts, left_act, right_act):
    score = 0.0
    if left_act is not None and acts:
        score += math.log(max(transition_prob(left_act, acts[0]), 1e-6))
    for a, b in zip(acts, acts[1:]):
        score += math.log(max(transition_prob(a, b), 1e-6))
    if right_act is not None and acts:
        score += math.log(max(transition_prob(acts[-1], right_act), 1e-6))
    canon_bonus = 0.0
    canonical = {
        "New online application received": 0,
        "review request received": 0,
        "Check for completeness": 1,
        "Request info": 2,
        "info received": 3,
        "Perform checks": 4,
        "Make decision": 5,
        "Notify accept": 6,
        "notify reject": 6,
        "Deliver card": 7,
        "time out": 7,
        "EVENT 13 END": 8,
    }
    idxs = [canonical.get(a, 99) for a in acts]
    for i in range(len(idxs) - 1):
        if idxs[i] <= idxs[i + 1]:
            canon_bonus += 0.02
    return score + canon_bonus


def enumerate_best_order(block_rows, left_act, right_act, observed_ts):
    activities = list(block_rows["Activity"])
    n = len(activities)
    best = None
    max_perm = math.factorial(n)
    perms = itertools.permutations(range(n)) if max_perm <= 720 else heuristic_permutations(activities, left_act, right_act)
    for perm in perms:
        ord_acts = [activities[i] for i in perm]
        s = order_score(ord_acts, left_act, right_act)
        if best is None or s > best[0]:
            best = (s, perm, ord_acts)
    return best[1], best[2]


def heuristic_permutations(activities, left_act, right_act):
    n = len(activities)
    idxs = list(range(n))
    scored = []
    for perm in itertools.permutations(idxs, min(n, 4)):
        used = set(perm)
        rest = [i for i in idxs if i not in used]
        order = list(perm) + rest
        acts = [activities[i] for i in order]
        scored.append((order_score(acts, left_act, right_act), tuple(order)))
    scored.sort(reverse=True)
    yielded = set()
    for _, p in scored[:60]:
        if p not in yielded:
            yielded.add(p)
            yield p
    canon = {
        "New online application received": 0,
        "review request received": 0,
        "Check for completeness": 1,
        "Request info": 2,
        "info received": 3,
        "Perform checks": 4,
        "Make decision": 5,
        "Notify accept": 6,
        "notify reject": 6,
        "Deliver card": 7,
        "time out": 7,
        "EVENT 13 END": 8,
    }
    fallback = tuple(sorted(idxs, key=lambda i: (canon.get(activities[i], 99), i)))
    if fallback not in yielded:
        yield fallback


def place_c1_block(case_df, idxs, global_frac_len):
    observed_ts = case_df.loc[idxs[0], "_parsed_ts"]
    shared_str = case_df.loc[idxs[0], "Timestamp"]
    _, left_ts, left_act = nearest_clean_left(case_df, idxs[0])
    _, right_ts, right_act = nearest_clean_right(case_df, idxs[-1])
    block_rows = case_df.loc[idxs]
    perm, ordered_acts = enumerate_best_order(block_rows, left_act, right_act, observed_ts)

    ordered_rows = [idxs[i] for i in perm]
    gaps = []
    for a, b in zip(ordered_acts, ordered_acts[1:]):
        gaps.append(max(1.0, chosen_gap_seconds(a, b)))
    total = sum(gaps)

    clamped = False
    if right_ts is not None:
        room = max(0.0, (right_ts - observed_ts).total_seconds())
        min_span = len(ordered_rows) - 1
        if room < min_span:
            room = min_span
        if total > room and total > 0:
            factor = room / total
            gaps = [max(1.0, g * factor) for g in gaps]
            total2 = sum(gaps)
            if total2 > room:
                excess = total2 - room
                for i in range(len(gaps) - 1, -1, -1):
                    reducible = max(0.0, gaps[i] - 1.0)
                    take = min(reducible, excess)
                    gaps[i] -= take
                    excess -= take
                    if excess <= 1e-9:
                        break
            clamped = True

    times = [observed_ts]
    cur = observed_ts
    for g in gaps:
        cur = cur + pd.to_timedelta(g, unit="s")
        times.append(cur)

    if right_ts is not None:
        times = [clamp_ts(t, observed_ts, right_ts) for t in times]
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                times[i] = times[i - 1] + pd.to_timedelta(1, unit="s")
        if times[-1] > right_ts:
            end = right_ts
            start = observed_ts
            span = max((end - start).total_seconds(), len(times) - 1)
            step = max(1.0, span / (len(times) - 1))
            times = [start + pd.to_timedelta(step * i, unit="s") for i in range(len(times))]
            if times[-1] > right_ts:
                times[-1] = right_ts
            clamped = True

    mapping = {}
    for row_idx, t in zip(ordered_rows, times):
        frac = frac_len_of_string(case_df.loc[row_idx, "Timestamp"])
        if frac == 0:
            frac = global_frac_len
        mapping[row_idx] = format_timestamp(t, frac)

    return mapping, clamped, ordered_rows[0]


def repair_c3_row(case_df, pos, global_frac_len):
    row = case_df.iloc[pos]
    left_i, left_ts, left_act = nearest_clean_left(case_df, pos)
    right_i, right_ts, right_act = nearest_clean_right(case_df, pos)
    act = row["Activity"]
    case_start, case_end = case_bounds(case_df)

    low = left_ts if left_ts is not None else case_start
    high = right_ts if right_ts is not None else case_end

    left_est = estimate_from_left(left_ts, left_act, act) if left_ts is not None else None
    right_est = estimate_from_right(act, right_act, right_ts) if right_ts is not None else None
    chosen = choose_balanced_between(left_est, right_est, low, high)

    orig = str(row["Timestamp"]) if not pd.isna(row["Timestamp"]) else ""
    if date_only_or_zero_time(orig):
        obs_date = extract_observed_date(orig)
        if obs_date is not None:
            chosen = pd.Timestamp.combine(obs_date, chosen.time())
            day_start = pd.Timestamp.combine(obs_date, pd.Timestamp("00:00:00").time())
            day_end = pd.Timestamp.combine(obs_date, pd.Timestamp("23:59:59.999999").time())
            low2 = max([x for x in [low, day_start] if x is not None], default=None)
            high2 = min([x for x in [high, day_end] if x is not None], default=None)
            chosen = clamp_ts(chosen, low2, high2)
            if low2 is not None and high2 is not None and chosen > high2:
                chosen = low2 + (high2 - low2) / 2
            if low2 is not None and high2 is not None and chosen < low2:
                chosen = low2 + (high2 - low2) / 2

    chosen = clamp_ts(chosen, low, high)
    frac = frac_len_of_string(orig)
    if frac == 0:
        frac = global_frac_len
    return format_timestamp(chosen, frac)


def repair_c2_pair(case_df, pos1, pos2, global_frac_len):
    r1 = case_df.iloc[pos1]
    r2 = case_df.iloc[pos2]
    t1 = r1["_parsed_ts"]
    t2 = r2["_parsed_ts"]

    pidx, prev_ts, prev_act = nearest_clean_left(case_df, pos1)
    nidx, next_ts, next_act = nearest_clean_right(case_df, pos2)

    swap_ok = True
    st1, st2 = t2, t1
    if prev_ts is not None and st1 < prev_ts:
        swap_ok = False
    if next_ts is not None and st2 > next_ts:
        swap_ok = False
    if st2 < st1:
        swap_ok = False

    swap_score = -1e18
    if swap_ok:
        swap_score = local_swap_score(prev_act, r1["Activity"], r2["Activity"], next_act, prev_ts, st1, st2, next_ts)

    low1 = prev_ts
    high1 = next_ts
    low2 = prev_ts
    high2 = next_ts

    cand1 = estimate_from_left(prev_ts, prev_act, r1["Activity"]) if prev_ts is not None else None
    cand1r = estimate_from_right(r1["Activity"], r2["Activity"], t2) if pd.notna(t2) else None
    new1 = choose_balanced_between(cand1, cand1r, low1, t2 if pd.notna(t2) else high1)
    if pd.notna(t2) and new1 > t2:
        new1 = min(new1, t2)
    disp_score1 = local_swap_score(prev_act, r1["Activity"], r2["Activity"], next_act, prev_ts, new1, t2, next_ts) if pd.notna(t2) else -1e18

    cand2l = estimate_from_left(t1, r1["Activity"], r2["Activity"]) if pd.notna(t1) else None
    cand2 = estimate_from_right(r2["Activity"], next_act, next_ts) if next_ts is not None else None
    new2 = choose_balanced_between(cand2l, cand2, t1 if pd.notna(t1) else low2, high2)
    if pd.notna(t1) and new2 < t1:
        new2 = max(new2, t1)
    disp_score2 = local_swap_score(prev_act, r1["Activity"], r2["Activity"], next_act, prev_ts, t1, new2, next_ts) if pd.notna(t1) else -1e18

    mapping = {}
    if swap_score >= max(disp_score1, disp_score2):
        f1 = frac_len_of_string(r1["Timestamp"]) or global_frac_len
        f2 = frac_len_of_string(r2["Timestamp"]) or global_frac_len
        mapping[case_df.index[pos1]] = format_timestamp(st1, f1)
        mapping[case_df.index[pos2]] = format_timestamp(st2, f2)
    else:
        if disp_score1 >= disp_score2:
            f1 = frac_len_of_string(r1["Timestamp"]) or global_frac_len
            mapping[case_df.index[pos1]] = format_timestamp(clamp_ts(new1, low1, t2 if pd.notna(t2) else high1), f1)
        else:
            f2 = frac_len_of_string(r2["Timestamp"]) or global_frac_len
            mapping[case_df.index[pos2]] = format_timestamp(clamp_ts(new2, t1 if pd.notna(t1) else low2, high2), f2)
    return mapping


def self_check_and_fix(df, repaired_mask, c1_blocks):
    blocks_repaired = len(c1_blocks)
    blocks_clamped = 0

    for block in c1_blocks:
        idxs = block["idxs"]
        observed = block["observed"]
        right_anchor = block["right_anchor"]
        vals = [safe_to_datetime(df.at[i, "Timestamp"]) for i in idxs]
        changed = [df.at[i, "Timestamp"] != observed for i in idxs]
        equal_obs = [df.at[i, "Timestamp"] == observed for i in idxs]

        if sum(equal_obs) != 1:
            first = idxs[0]
            df.at[first, "Timestamp"] = observed
            for j, i in enumerate(idxs[1:], start=1):
                base = safe_to_datetime(observed) + pd.to_timedelta(j, unit="s")
                if right_anchor is not None and base > right_anchor:
                    base = right_anchor - pd.to_timedelta(len(idxs) - j - 1, unit="s")
                frac = frac_len_of_string(observed)
                df.at[i, "Timestamp"] = format_timestamp(base, frac if frac > 0 else 3)
            blocks_clamped += 1

        vals = [safe_to_datetime(df.at[i, "Timestamp"]) for i in idxs]
        bad = False
        for a, b in zip(vals, vals[1:]):
            if pd.isna(a) or pd.isna(b) or not (b > a):
                bad = True
                break
        if bad:
            start = safe_to_datetime(observed)
            room = max((right_anchor - start).total_seconds(), len(idxs) - 1) if right_anchor is not None else len(idxs) * 10
            step = max(1.0, room / max(1, len(idxs) - 1))
            seq = [start + pd.to_timedelta(step * j, unit="s") for j in range(len(idxs))]
            if right_anchor is not None and seq[-1] > right_anchor:
                seq = [start + pd.to_timedelta(j, unit="s") for j in range(len(idxs))]
                while seq[-1] > right_anchor and len(idxs) > 1:
                    seq[-1] = right_anchor
                    for k in range(len(idxs) - 2, -1, -1):
                        if seq[k] >= seq[k + 1]:
                            seq[k] = seq[k + 1] - pd.to_timedelta(1, unit="s")
            for j, i in enumerate(idxs):
                frac = frac_len_of_string(df.at[i, "Timestamp"])
                if j == 0:
                    df.at[i, "Timestamp"] = observed
                else:
                    df.at[i, "Timestamp"] = format_timestamp(seq[j], frac if frac > 0 else 3)
            blocks_clamped += 1

    rows_equal = 0
    for i in df.index:
        if repaired_mask.get(i, False) and str(df.at[i, "Timestamp"]) == str(df.at[i, "_orig_timestamp"]):
            rows_equal += 1
    return blocks_repaired, blocks_clamped, rows_equal


def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_cols = list(df.columns)
    for col in ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]:
        if col not in df.columns:
            df[col] = ""

    df["_orig_timestamp"] = df["Timestamp"]
    df["_parsed_ts"] = df["Timestamp"].apply(safe_to_datetime)

    global_frac_len = detect_global_precision(df["Timestamp"].tolist())
    repaired_mask = {}
    c1_blocks = []

    try:
        for case_id, idxs in df.groupby("Case", sort=False).groups.items():
            case_idx = list(idxs)
            case_df = df.loc[case_idx].copy()

            # C1 blocks: consecutive rows in case with same timestamp and non-clean
            i = 0
            while i < len(case_df):
                row = case_df.iloc[i]
                if row["error_category"] == "C1_ORDERING_AMBIGUITY":
                    same_ts = row["Timestamp"]
                    j = i + 1
                    block_pos = [i]
                    while j < len(case_df):
                        rowj = case_df.iloc[j]
                        if rowj["error_category"] == "C1_ORDERING_AMBIGUITY" and rowj["Timestamp"] == same_ts:
                            block_pos.append(j)
                            j += 1
                        else:
                            break
                    if len(block_pos) >= 2:
                        real_idxs = [case_df.index[p] for p in block_pos]
                        mapping, clamped, keeper = place_c1_block(case_df, real_idxs, global_frac_len)
                        for ridx, ts_str in mapping.items():
                            df.at[ridx, "Timestamp"] = ts_str
                            repaired_mask[ridx] = True
                        right_anchor = None
                        _, rt, _ = nearest_clean_right(case_df, block_pos[-1])
                        right_anchor = rt
                        c1_blocks.append({
                            "idxs": real_idxs,
                            "observed": same_ts,
                            "right_anchor": right_anchor,
                        })
                        i = j
                    else:
                        i += 1
                else:
                    i += 1

            # refresh parsed within case if c1 changed some rows
            for ridx in case_idx:
                df.at[ridx, "_parsed_ts"] = safe_to_datetime(df.at[ridx, "Timestamp"])
            case_df = df.loc[case_idx].copy()

            # C2 repair
            for i in range(len(case_df) - 1):
                r1 = case_df.iloc[i]
                r2 = case_df.iloc[i + 1]
                if r1["error_category"] == "C2_TEMPORAL_INCONSISTENCY" or r2["error_category"] == "C2_TEMPORAL_INCONSISTENCY":
                    t1 = safe_to_datetime(df.at[case_df.index[i], "Timestamp"])
                    t2 = safe_to_datetime(df.at[case_df.index[i + 1], "Timestamp"])
                    if pd.notna(t1) and pd.notna(t2) and t2 < t1:
                        mapping = repair_c2_pair(case_df, i, i + 1, global_frac_len)
                        for ridx, ts_str in mapping.items():
                            df.at[ridx, "Timestamp"] = ts_str
                            repaired_mask[ridx] = True
                        for ridx in case_idx:
                            df.at[ridx, "_parsed_ts"] = safe_to_datetime(df.at[ridx, "Timestamp"])
                        case_df = df.loc[case_idx].copy()

            # C3 repair
            for i in range(len(case_df)):
                if case_df.iloc[i]["error_category"] == "C3_MISSING_TEMPORAL_INFORMATION":
                    ridx = case_df.index[i]
                    new_ts = repair_c3_row(case_df, i, global_frac_len)
                    if new_ts == str(df.at[ridx, "_orig_timestamp"]):
                        # ensure corrupted rows change
                        current = safe_to_datetime(new_ts)
                        _, rt, _ = nearest_clean_right(case_df, i)
                        _, lt, _ = nearest_clean_left(case_df, i)
                        low = lt
                        high = rt
                        shifted = current + pd.to_timedelta(1, unit="s") if pd.notna(current) else None
                        if shifted is not None:
                            shifted = clamp_ts(shifted, low, high)
                            if low is not None and shifted <= low:
                                shifted = low + pd.to_timedelta(1, unit="s")
                            if high is not None and shifted >= high and low is not None:
                                shifted = low + (high - low) / 2
                            frac = frac_len_of_string(df.at[ridx, "Timestamp"]) or global_frac_len
                            new_ts = format_timestamp(shifted, frac)
                    df.at[ridx, "Timestamp"] = new_ts
                    repaired_mask[ridx] = True
                    df.at[ridx, "_parsed_ts"] = safe_to_datetime(new_ts)
                    case_df = df.loc[case_idx].copy()

        # final anchor/case clamping for repaired rows
        for case_id, idxs in df.groupby("Case", sort=False).groups.items():
            case_idx = list(idxs)
            case_df = df.loc[case_idx].copy()
            cstart, cend = case_bounds(case_df)
            for p in range(len(case_df)):
                ridx = case_df.index[p]
                if not repaired_mask.get(ridx, False):
                    continue
                ts = safe_to_datetime(df.at[ridx, "Timestamp"])
                _, lt, _ = nearest_clean_left(case_df, p)
                _, rt, _ = nearest_clean_right(case_df, p)
                low = lt if lt is not None else cstart
                high = rt if rt is not None else cend
                if pd.notna(ts):
                    ts2 = clamp_ts(ts, low, high)
                    if case_df.iloc[p]["error_category"] != "C1_ORDERING_AMBIGUITY" and str(df.at[ridx, "Timestamp"]) == str(df.at[ridx, "_orig_timestamp"]):
                        ts2 = ts2 + pd.to_timedelta(1, unit="s")
                        ts2 = clamp_ts(ts2, low, high)
                    frac = frac_len_of_string(df.at[ridx, "_orig_timestamp"]) or global_frac_len
                    df.at[ridx, "Timestamp"] = format_timestamp(ts2, frac)

        blocks_repaired, blocks_clamped, rows_equal = self_check_and_fix(df, repaired_mask, c1_blocks)

    except Exception:
        # never fail hard; write best-effort output
        blocks_repaired, blocks_clamped, rows_equal = self_check_and_fix(df, repaired_mask, c1_blocks)

    # CLEAN rows byte-identical
    clean_mask = df["error_category"] == "CLEAN"
    df.loc[clean_mask, "Timestamp"] = df.loc[clean_mask, "_orig_timestamp"]

    # ensure non-timestamp protected columns byte-identical
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    out = df[original_cols].copy()
    try:
        out.to_csv(args.output, index=False)
    except Exception:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            out.to_csv(f, index=False)

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={blocks_clamped} rows_still_equal={rows_equal}")


if __name__ == "__main__":
    main()