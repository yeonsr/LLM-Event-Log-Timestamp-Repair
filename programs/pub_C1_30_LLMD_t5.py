import argparse
import itertools
import math
from collections import Counter, defaultdict

import pandas as pd


TRANSITION_FREQ = {
    ("Deliver to customer", "EVENT 8 END"): 5000,
    ("Signal food complete", "Bring food"): 4682,
    ("Bring food", "Deliver to customer"): 4986,
    ("Prepare main course", "Signal food complete"): 3866,
    ("start", "Take order"): 4854,
    ("Bring drinks", "Review drinks"): 7062,
    ("Take order", "Review order"): 4594,
    ("prepare drinks", "Bring drinks"): 6704,
    ("Review order", "prepare drinks"): 3550,
    ("Prepare starter", "prepare drinks"): 883,
    ("Review drinks", "Prepare main course"): 3563,
    ("Review drinks", "prepare drinks"): 2944,
    ("Prepare starter", "Bring drinks"): 450,
    ("Review order", "Prepare starter"): 811,
    ("Prepare starter", "Prepare main course"): 143,
    ("Review drinks", "Signal food complete"): 509,
    ("prepare drinks", "Prepare starter"): 445,
    ("Prepare starter", "Review drinks"): 97,
    ("Review drinks", "Prepare starter"): 293,
    ("Take order", "Review drinks"): 106,
    ("Review order", "Review drinks"): 83,
    ("prepare drinks", "Bring food"): 116,
    ("Bring drinks", "Signal food complete"): 95,
    ("start", "Bring drinks"): 62,
    ("Take order", "Bring drinks"): 60,
    ("prepare drinks", "Signal food complete"): 90,
    ("Bring drinks", "Prepare starter"): 88,
    ("Review order", "Prepare main course"): 61,
    ("prepare drinks", "Prepare main course"): 51,
    ("Bring drinks", "Bring food"): 81,
    ("Review order", "Signal food complete"): 49,
    ("start", "Review drinks"): 46,
    ("Review order", "Bring drinks"): 40,
    ("prepare drinks", "Review drinks"): 46,
    ("Prepare main course", "Bring drinks"): 18,
    ("Bring drinks", "Bring drinks"): 34,
    ("Review drinks", "Review drinks"): 30,
    ("start", "prepare drinks"): 25,
    ("Take order", "prepare drinks"): 24,
    ("Prepare main course", "Review drinks"): 4,
    ("Review drinks", "Deliver to customer"): 14,
    ("prepare drinks", "prepare drinks"): 14,
    ("start", "Prepare starter"): 13,
    ("Take order", "Prepare starter"): 13,
    ("Prepare main course", "prepare drinks"): 12,
    ("Bring food", "Bring drinks"): 11,
    ("Review drinks", "Bring drinks"): 10,
    ("Prepare starter", "Bring food"): 10,
    ("Bring drinks", "prepare drinks"): 7,
    ("Signal food complete", "prepare drinks"): 4,
    ("Signal food complete", "Bring drinks"): 3,
    ("Bring food", "prepare drinks"): 3,
}

DURATION_PROFILES = {
    ("Bring drinks", "Review drinks"): dict(p10=6.1307, p25=17.0105, median=41.052, share=0.631),
    ("prepare drinks", "Bring drinks"): dict(p10=28.8126, p25=79.71675, median=195.9745, share=0.198),
    ("Deliver to customer", "EVENT 8 END"): dict(p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("Bring food", "Deliver to customer"): dict(p10=12.5235, p25=34.57875, median=85.062, share=0.389),
    ("start", "Take order"): dict(p10=12.9101, p25=35.79525, median=89.148, share=0.378),
    ("Signal food complete", "Bring food"): dict(p10=33.0276, p25=90.3885, median=214.9425, share=0.176),
    ("Take order", "Review order"): dict(p10=6.1912, p25=17.985, median=41.872, share=0.636),
    ("Prepare main course", "Signal food complete"): dict(p10=85943.671, p25=253686.04975, median=513669.015, share=0.001),
    ("Review drinks", "Prepare main course"): dict(p10=83921.2662, p25=243350.551, median=500815.099, share=0.001),
    ("Review order", "prepare drinks"): dict(p10=25.252, p25=68.49925, median=173.713, share=0.226),
    ("Review drinks", "prepare drinks"): dict(p10=30.7213, p25=85.6735, median=209.569, share=0.182),
    ("Prepare starter", "prepare drinks"): dict(p10=30.3582, p25=89.143, median=206.2, share=0.19),
    ("Review order", "Prepare starter"): dict(p10=15.008, p25=42.552, median=97.24, share=0.354),
    ("Review drinks", "Signal food complete"): dict(p10=7451.2538, p25=175593.462, median=445342.856, share=0.002),
    ("Prepare starter", "Bring drinks"): dict(p10=29.0292, p25=86.16725, median=203.727, share=0.189),
    ("prepare drinks", "Prepare starter"): dict(p10=15.8778, p25=40.819, median=100.292, share=0.362),
    ("Review drinks", "Prepare starter"): dict(p10=17.9444, p25=63.83, median=151.424, share=0.232),
    ("Prepare starter", "Prepare main course"): dict(p10=76289.6188, p25=192412.7035, median=445174.842, share=0.0),
    ("prepare drinks", "Bring food"): dict(p10=39201.883, p25=406754.148, median=785107.18, share=0.009),
    ("Review drinks", "Bring food"): dict(p10=161178.253, p25=593624.664, median=857938.338, share=0.009),
    ("Take order", "Review drinks"): dict(p10=257.7665, p25=389.71375, median=707.2185, share=0.0),
    ("Prepare starter", "Review drinks"): dict(p10=11.5194, p25=22.425, median=56.303, share=0.505),
    ("Bring drinks", "Signal food complete"): dict(p10=179812.449, p25=593579.078, median=883902.239, share=0.0),
    ("prepare drinks", "Signal food complete"): dict(p10=84583.4036, p25=431151.8905, median=982509.744, share=0.011),
    ("Bring drinks", "Prepare starter"): dict(p10=4.3939, p25=15.89925, median=33.688, share=0.693),
    ("Review order", "Review drinks"): dict(p10=303.721, p25=381.153, median=593.835, share=0.0),
    ("Bring drinks", "Bring food"): dict(p10=102484.185, p25=366203.855, median=696631.885, share=0.0),
    ("Prepare starter", "Signal food complete"): dict(p10=88210.8159, p25=276729.8375, median=519007.665, share=0.0),
    ("start", "Bring drinks"): dict(p10=275.2757, p25=422.42225, median=695.196, share=0.0),
    ("Review order", "Prepare main course"): dict(p10=943.183, p25=84467.297, median=444605.508, share=0.0),
    ("Take order", "Bring drinks"): dict(p10=296.2082, p25=433.589, median=624.287, share=0.0),
    ("Take order", "Prepare main course"): dict(p10=101957.3608, p25=258966.497, median=513968.537, share=0.0),
    ("prepare drinks", "Prepare main course"): dict(p10=71.18, p25=281.63, median=160626.596, share=0.078),
    ("Review order", "Signal food complete"): dict(p10=152688.052, p25=587731.024, median=862530.877, share=0.0),
    ("start", "Review drinks"): dict(p10=281.845, p25=455.605, median=733.167, share=0.0),
    ("prepare drinks", "Review drinks"): dict(p10=368.894, p25=508.0155, median=656.5765, share=0.0),
    ("Review order", "Bring drinks"): dict(p10=298.812, p25=538.94575, median=784.107, share=0.0),
    ("Bring drinks", "Bring drinks"): dict(p10=237.8929, p25=350.94375, median=571.916, share=0.0),
    ("Review drinks", "Review drinks"): dict(p10=283.804, p25=471.11475, median=695.439, share=0.0),
    ("start", "prepare drinks"): dict(p10=255.112, p25=319.999, median=413.649, share=0.0),
    ("Take order", "prepare drinks"): dict(p10=214.6615, p25=304.47475, median=736.587, share=0.0),
    ("Prepare main course", "Bring drinks"): dict(p10=28.2273, p25=68.60625, median=129.8, share=0.222),
    ("Review drinks", "Deliver to customer"): dict(p10=13.1735, p25=26.957, median=70.7255, share=0.357),
    ("prepare drinks", "prepare drinks"): dict(p10=430.3684, p25=495.2955, median=639.1405, share=0.0),
    ("start", "Prepare starter"): dict(p10=275.651, p25=370.046, median=404.992, share=0.0),
    ("Take order", "Prepare starter"): dict(p10=327.2842, p25=389.43, median=558.771, share=0.0),
    ("Prepare main course", "prepare drinks"): dict(p10=94.8512, p25=110.78, median=178.7615, share=0.0),
    ("Bring food", "Bring drinks"): dict(p10=50.942, p25=152.058, median=229.533, share=0.182),
    ("Review drinks", "Bring drinks"): dict(p10=434.9222, p25=666.72275, median=1267.9125, share=0.0),
    ("Prepare starter", "Bring food"): dict(p10=167409.7799, p25=293248.85775, median=525665.88, share=0.0),
    ("Bring drinks", "prepare drinks"): dict(p10=691.6136, p25=983.649, median=1736.597, share=0.0),
    ("Signal food complete", "prepare drinks"): dict(p10=65.3186, p25=153.056, median=228.6855, share=0.25),
    ("Prepare main course", "Review drinks"): dict(p10=2.2898, p25=3.674, median=35.4335, share=0.5),
    ("Bring food", "prepare drinks"): dict(p10=164.9284, p25=166.546, median=169.242, share=0.0),
    ("Signal food complete", "Bring drinks"): dict(p10=150.102, p25=179.118, median=227.478, share=0.0),
}

FREQUENT_VARIANTS = [
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare starter", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
]


def detect_fractional_digits(ts: str) -> int:
    if "." in ts:
        return len(ts.split(".")[-1])
    return 0


def parse_ts(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts)


def format_ts(ts: pd.Timestamp, digits: int) -> str:
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if digits <= 0:
        return base
    frac = f"{ts.microsecond:06d}"[:digits].ljust(digits, "0")
    return f"{base}.{frac}"


def min_step_seconds(digits: int) -> float:
    if digits <= 0:
        return 1.0
    return 10 ** (-digits)


def is_contiguous_subsequence(seq, subseq):
    k = len(subseq)
    if k == 0:
        return True
    for i in range(len(seq) - k + 1):
        if seq[i:i + k] == subseq:
            return True
    return False


def variant_match_score(left_act, ordered, right_act):
    full = []
    if left_act is not None:
        full.append(left_act)
    full.extend(ordered)
    if right_act is not None:
        full.append(right_act)
    best = 0
    for v in FREQUENT_VARIANTS:
        if is_contiguous_subsequence(v, full):
            return 3
        if is_contiguous_subsequence(v, ordered):
            best = max(best, 2)
        elif left_act is not None and is_contiguous_subsequence(v, [left_act] + ordered):
            best = max(best, 1)
        elif right_act is not None and is_contiguous_subsequence(v, ordered + [right_act]):
            best = max(best, 1)
    return best


def multiset_permutations(items):
    counter = Counter(items)
    uniq = list(counter.keys())
    n = len(items)

    def backtrack(path):
        if len(path) == n:
            yield tuple(path)
            return
        for x in uniq:
            if counter[x] > 0:
                counter[x] -= 1
                path.append(x)
                yield from backtrack(path)
                path.pop()
                counter[x] += 1
    yield from backtrack([])


def edge_supported(a, b):
    return (a, b) in TRANSITION_FREQ


def edge_logfreq(a, b):
    f = TRANSITION_FREQ.get((a, b), 0)
    if f <= 0:
        return -1e9
    return math.log(f + 1.0)


def score_order(left_act, ordered, right_act):
    internal_pairs = list(zip(ordered[:-1], ordered[1:]))
    supported_internal = sum(1 for a, b in internal_pairs if edge_supported(a, b))
    all_internal_supported = supported_internal == len(internal_pairs)

    anchor_supported = 0
    if left_act is not None and edge_supported(left_act, ordered[0]):
        anchor_supported += 1
    if right_act is not None and edge_supported(ordered[-1], right_act):
        anchor_supported += 1

    logsum = 0.0
    if left_act is not None and edge_supported(left_act, ordered[0]):
        logsum += edge_logfreq(left_act, ordered[0])
    for a, b in internal_pairs:
        if edge_supported(a, b):
            logsum += edge_logfreq(a, b)
    if right_act is not None and edge_supported(ordered[-1], right_act):
        logsum += edge_logfreq(ordered[-1], right_act)

    vscore = variant_match_score(left_act, list(ordered), right_act)
    unsupported_internal = len(internal_pairs) - supported_internal
    return (
        1 if all_internal_supported else 0,
        vscore,
        supported_internal,
        anchor_supported,
        -unsupported_internal,
        logsum,
    )


def choose_order(activities, left_act, right_act):
    perms = list(multiset_permutations(activities))
    scored = []
    for p in perms:
        scored.append((score_order(left_act, p, right_act), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    best = [p for s, p in scored if s == best_score]
    if len(best) == 1:
        return list(best[0])

    def predicted_span(order):
        gaps = estimate_gaps(order, left_act, right_act, 1.0)
        return sum(gaps)

    best.sort(key=lambda p: (predicted_span(p),))
    return list(best[0])


def estimate_pair_gap(a, b, step_sec):
    prof = DURATION_PROFILES.get((a, b))
    if prof:
        if prof["share"] >= 0.5:
            g = prof["p10"]
        else:
            g = prof["p25"]
        g = max(g, step_sec)
        return g
    rev = DURATION_PROFILES.get((b, a))
    if rev and edge_supported(a, b) and rev["share"] >= 0.4:
        return max(step_sec, min(rev["p10"], rev["p25"]))
    return None


def estimate_gaps(order, left_act, right_act, step_sec):
    gaps = []
    for a, b in zip(order[:-1], order[1:]):
        g = estimate_pair_gap(a, b, step_sec)
        if g is None:
            if edge_supported(a, b):
                g = 2.3 * step_sec
            else:
                g = 1.0 * step_sec
        gaps.append(max(g, step_sec))
    if not gaps:
        return gaps

    adjusted = []
    for i, (a, b) in enumerate(zip(order[:-1], order[1:])):
        g = gaps[i]
        if edge_supported(a, b) and edge_supported(b, a):
            fa = TRANSITION_FREQ.get((a, b), 0)
            fb = TRANSITION_FREQ.get((b, a), 0)
            if min(fa, fb) > 0 and max(fa, fb) / min(fa, fb) <= 3.0:
                g = min(g, 2.0 * step_sec)
        adjusted.append(max(g, step_sec))
    return adjusted


def build_times_start(observed_ts, order, left_ts, right_ts, left_act, right_act, digits):
    step_sec = min_step_seconds(digits)
    k = len(order)
    if k == 1:
        return [observed_ts], False

    gaps = estimate_gaps(order, left_act, right_act, step_sec)
    min_total = (k - 1) * step_sec
    total = max(sum(gaps), min_total)

    clamp_needed = False
    if right_ts is not None:
        room = max((right_ts - observed_ts).total_seconds(), 0.0)
        max_total = room
        if max_total < min_total:
            max_total = min_total
        if total > max_total:
            clamp_needed = True
            scale = max_total / total if total > 0 else 1.0
            gaps = [max(g * scale, step_sec) for g in gaps]
            total = sum(gaps)

    times = [observed_ts]
    cur = observed_ts
    for g in gaps:
        cur = cur + pd.to_timedelta(g, unit="s")
        times.append(cur)

    if right_ts is not None and times[-1] > right_ts:
        clamp_needed = True
        end_limit = right_ts
        latest_start = end_limit - pd.to_timedelta((k - 1) * step_sec, unit="s")
        base = min(observed_ts, latest_start)
        times = [base + pd.to_timedelta(i * step_sec, unit="s") for i in range(k)]
        times[0] = observed_ts
        for i in range(1, k):
            min_allowed = times[i - 1] + pd.to_timedelta(step_sec, unit="s")
            if times[i] < min_allowed:
                times[i] = min_allowed
        if times[-1] > right_ts:
            overflow = (times[-1] - right_ts).total_seconds()
            shift = pd.to_timedelta(overflow, unit="s")
            times = [t - shift for t in times]
            times[0] = observed_ts
            for i in range(1, k):
                min_allowed = times[i - 1] + pd.to_timedelta(step_sec, unit="s")
                if times[i] < min_allowed:
                    times[i] = min_allowed

    if left_ts is not None:
        for i in range(k):
            if times[i] < left_ts:
                clamp_needed = True
                times[i] = left_ts
        times[0] = max(times[0], observed_ts)

    for i in range(1, k):
        min_allowed = times[i - 1] + pd.to_timedelta(step_sec, unit="s")
        if times[i] < min_allowed:
            clamp_needed = True
            times[i] = min_allowed

    if right_ts is not None and times[-1] > right_ts:
        clamp_needed = True
        max_total = max((right_ts - observed_ts).total_seconds(), min_total)
        span = max_total
        ladder = []
        if k == 2:
            ladder = [span]
        else:
            weights = [1.0 + 0.17 * i for i in range(k - 1)]
            sw = sum(weights)
            ladder = [max(step_sec, span * w / sw) for w in weights]
        times = [observed_ts]
        cur = observed_ts
        for g in ladder:
            cur = cur + pd.to_timedelta(g, unit="s")
            times.append(cur)
        for i in range(1, k):
            min_allowed = times[i - 1] + pd.to_timedelta(step_sec, unit="s")
            if times[i] < min_allowed:
                times[i] = min_allowed
        if times[-1] > right_ts:
            times[-1] = right_ts
            for i in range(k - 2, -1, -1):
                max_allowed = times[i + 1] - pd.to_timedelta(step_sec, unit="s")
                if times[i] > max_allowed:
                    times[i] = max_allowed
            times[0] = observed_ts
            for i in range(1, k):
                min_allowed = times[i - 1] + pd.to_timedelta(step_sec, unit="s")
                if times[i] < min_allowed:
                    times[i] = min_allowed

    return times, clamp_needed


def map_times_to_rows(block_df, ordered_activities, ordered_times):
    act_to_rows = defaultdict(list)
    for idx, row in block_df.iterrows():
        act_to_rows[row["Activity"]].append(idx)
    assignment = {}
    used_count = defaultdict(int)
    for act, ts in zip(ordered_activities, ordered_times):
        pos = used_count[act]
        row_idx = act_to_rows[act][pos]
        assignment[row_idx] = ts
        used_count[act] += 1
    return assignment


def repair_block(case_df, block_indices, out_ts, parsed_ts):
    block_df = case_df.loc[block_indices]
    observed_str = block_df.iloc[0]["Timestamp"]
    observed_ts = parse_ts(observed_str)
    digits = detect_fractional_digits(observed_str)
    case_indices = list(case_df.index)
    first_pos = case_indices.index(block_indices[0])
    last_pos = case_indices.index(block_indices[-1])

    left_anchor_idx = None
    for pos in range(first_pos - 1, -1, -1):
        idx = case_indices[pos]
        if case_df.loc[idx, "error_category"] == "CLEAN":
            left_anchor_idx = idx
            break

    right_anchor_idx = None
    for pos in range(last_pos + 1, len(case_indices)):
        idx = case_indices[pos]
        if case_df.loc[idx, "error_category"] == "CLEAN":
            right_anchor_idx = idx
            break

    left_act = case_df.loc[left_anchor_idx, "Activity"] if left_anchor_idx is not None else None
    right_act = case_df.loc[right_anchor_idx, "Activity"] if right_anchor_idx is not None else None
    left_ts = parse_ts(case_df.loc[left_anchor_idx, "Timestamp"]) if left_anchor_idx is not None else None
    right_ts = parse_ts(case_df.loc[right_anchor_idx, "Timestamp"]) if right_anchor_idx is not None else None

    acts = list(block_df["Activity"])
    order = choose_order(acts, left_act, right_act)

    times, clamped = build_times_start(observed_ts, order, left_ts, right_ts, left_act, right_act, digits)

    if right_ts is not None and len(times) > 1 and times[-1] > right_ts:
        step_sec = min_step_seconds(digits)
        backward_supported = 0
        if right_act is not None and edge_supported(order[-1], right_act):
            backward_supported += 1
        if left_act is not None and edge_supported(left_act, order[0]):
            backward_supported += 1
        forward_room = (right_ts - observed_ts).total_seconds()
        backward_room = (observed_ts - left_ts).total_seconds() if left_ts is not None else float("inf")
        if forward_room < (len(order) - 1) * step_sec and backward_room >= (len(order) - 1) * step_sec and backward_supported >= 1:
            clamped = True
            rev_order = order[:]
            rev_times = [observed_ts - pd.to_timedelta((len(order) - 1 - i) * step_sec, unit="s") for i in range(len(order))]
            if left_ts is not None and rev_times[0] < left_ts:
                shift = left_ts - rev_times[0]
                rev_times = [t + shift for t in rev_times]
            if right_ts is not None and rev_times[-1] > right_ts:
                shift = rev_times[-1] - right_ts
                rev_times = [t - shift for t in rev_times]
            order = rev_order
            times = rev_times

    assignment = map_times_to_rows(block_df, order, times)

    for idx, ts in assignment.items():
        out_ts[idx] = format_ts(ts, digits)
        parsed_ts[idx] = ts

    return {
        "indices": block_indices,
        "observed_str": observed_str,
        "observed_ts": observed_ts,
        "digits": digits,
        "order": order,
        "times": times,
        "assignment": assignment,
        "left_ts": left_ts,
        "right_ts": right_ts,
        "clamped": clamped,
    }


def self_check_and_fix(df, out_ts, block_infos):
    clamped_blocks = 0
    for info in block_infos:
        idxs = info["indices"]
        observed_str = info["observed_str"]
        observed_ts = info["observed_ts"]
        digits = info["digits"]
        step_sec = min_step_seconds(digits)
        left_ts = info["left_ts"]
        right_ts = info["right_ts"]

        assigned = {idx: parse_ts(out_ts[idx]) for idx in idxs}
        kept = [idx for idx in idxs if out_ts[idx] == observed_str]

        need_fix = False
        if len(kept) != 1:
            need_fix = True
        else:
            kept_idx = kept[0]
            kept_ts = assigned[kept_idx]
            if kept_ts != observed_ts:
                need_fix = True

        ordered_pairs = sorted(info["assignment"].items(), key=lambda kv: kv[1])
        ordered_times = [t for _, t in ordered_pairs]
        for i in range(1, len(ordered_times)):
            if not (ordered_times[i] > ordered_times[i - 1]):
                need_fix = True

        if right_ts is not None:
            span = (max(ordered_times) - observed_ts).total_seconds()
            room = (right_ts - observed_ts).total_seconds()
            if span > room + 1e-9:
                need_fix = True
        if len(idxs) > 1:
            span = (max(ordered_times) - min(ordered_times)).total_seconds()
            if span + 1e-9 < (len(idxs) - 1) * step_sec:
                need_fix = True

        for t in ordered_times:
            if left_ts is not None and t < left_ts:
                need_fix = True
            if right_ts is not None and t > right_ts:
                need_fix = True

        if need_fix:
            clamped_blocks += 1
            k = len(idxs)
            if right_ts is not None:
                room = max((right_ts - observed_ts).total_seconds(), (k - 1) * step_sec)
            else:
                room = max((k - 1) * step_sec, (k - 1) * step_sec * 1.3)
            if k == 1:
                new_times = [observed_ts]
            else:
                weights = [1.0 + 0.13 * i for i in range(k - 1)]
                sw = sum(weights)
                gaps = [max(step_sec, room * w / sw) for w in weights]
                total = sum(gaps)
                if right_ts is not None and total > room:
                    scale = room / total if total > 0 else 1.0
                    gaps = [max(step_sec, g * scale) for g in gaps]
                new_times = [observed_ts]
                cur = observed_ts
                for g in gaps:
                    cur = cur + pd.to_timedelta(g, unit="s")
                    new_times.append(cur)
                if right_ts is not None and new_times[-1] > right_ts:
                    new_times[-1] = right_ts
                    for i in range(k - 2, -1, -1):
                        max_allowed = new_times[i + 1] - pd.to_timedelta(step_sec, unit="s")
                        if new_times[i] > max_allowed:
                            new_times[i] = max_allowed
                    new_times[0] = observed_ts
                    for i in range(1, k):
                        min_allowed = new_times[i - 1] + pd.to_timedelta(step_sec, unit="s")
                        if new_times[i] < min_allowed:
                            new_times[i] = min_allowed
            ordered_acts = info["order"]
            assignment = map_times_to_rows(df.loc[idxs], ordered_acts, new_times)
            for idx, ts in assignment.items():
                out_ts[idx] = format_ts(ts, digits)
            still_kept = [idx for idx in idxs if out_ts[idx] == observed_str]
            if len(still_kept) != 1:
                earliest_idx = min(assignment.items(), key=lambda kv: kv[1])[0]
                for idx in idxs:
                    if idx == earliest_idx:
                        out_ts[idx] = observed_str
                    else:
                        base = assignment[idx]
                        if base == observed_ts:
                            base = base + pd.to_timedelta(step_sec, unit="s")
                        out_ts[idx] = format_ts(base, digits)
    return clamped_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        original_cols = list(df.columns)
        out_ts = df["Timestamp"].astype(str).tolist()
        parsed_ts = [parse_ts(x) for x in df["Timestamp"].astype(str)]

        block_infos = []
        blocks_repaired = 0

        for case_value, case_df in df.groupby("Case", sort=False):
            case_indices = list(case_df.index)
            i = 0
            while i < len(case_indices):
                idx = case_indices[i]
                if df.loc[idx, "error_category"] != "C1_ORDERING_AMBIGUITY":
                    i += 1
                    continue
                ts_str = df.loc[idx, "Timestamp"]
                block = [idx]
                j = i + 1
                while j < len(case_indices):
                    idx2 = case_indices[j]
                    if df.loc[idx2, "error_category"] == "C1_ORDERING_AMBIGUITY" and df.loc[idx2, "Timestamp"] == ts_str:
                        block.append(idx2)
                        j += 1
                    else:
                        break
                info = repair_block(case_df, block, out_ts, parsed_ts)
                block_infos.append(info)
                blocks_repaired += 1
                i = j

        clamped_blocks = self_check_and_fix(df, out_ts, block_infos)

        result = df.copy()
        result["Timestamp"] = out_ts

        # final guards
        if len(result) != len(df):
            result = df.copy()
            result["Timestamp"] = out_ts[:len(df)]

        for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
            result[col] = df[col]

        # Ensure CLEAN untouched
        clean_mask = df["error_category"] == "CLEAN"
        result.loc[clean_mask, "Timestamp"] = df.loc[clean_mask, "Timestamp"]

        # Ensure parseable
        repaired_mask = df["error_category"] != "CLEAN"
        for idx in result.index[repaired_mask]:
            try:
                parse_ts(result.at[idx, "Timestamp"])
            except Exception:
                result.at[idx, "Timestamp"] = df.at[idx, "Timestamp"]

        rows_still_equal = int(((result["error_category"] != "CLEAN") & (result["Timestamp"] == df["Timestamp"])).sum())

        result = result[original_cols]
        result.to_csv(args.output, index=False)

        print(f"blocks_repaired={blocks_repaired}, blocks_clamped={clamped_blocks}, rows_still_equal_to_input={rows_still_equal}")
    except Exception:
        try:
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            print("blocks_repaired=0, blocks_clamped=0, rows_still_equal_to_input=0")
        except Exception:
            print("blocks_repaired=0, blocks_clamped=0, rows_still_equal_to_input=0")


if __name__ == "__main__":
    main()