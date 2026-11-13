import argparse
import itertools
import math
from collections import Counter, defaultdict

import pandas as pd


# -----------------------------
# Process context embedded
# -----------------------------
TRANSITION_PROB = {
    ("Deliver card", "EVENT 13 END"): 1.0,
    ("EVENT 13 END", "EVENT 13 END"): 1.0,
    ("Notify accept", "Deliver card"): 1.0,
    ("time out", "EVENT 13 END"): 1.0,
    ("Perform checks", "Make decision"): 0.9919,
    ("review request received", "Check for completeness"): 0.9708,
    ("info received", "Check for completeness"): 0.9662,
    ("Request info", "info received"): 0.962,
    ("New online application received", "Check for completeness"): 0.91,
    ("notify reject", "time out"): 0.5226,
    ("Check for completeness", "Request info"): 0.5124,
    ("Make decision", "notify reject"): 0.5079,
    ("Make decision", "Notify accept"): 0.4883,
    ("notify reject", "review request received"): 0.465,
    ("Check for completeness", "Perform checks"): 0.4649,
    ("New online application received", "Notify accept"): 0.0276,
    ("New online application received", "Make decision"): 0.0234,
    ("New online application received", "notify reject"): 0.0164,
    ("info received", "Make decision"): 0.0106,
    ("New online application received", "info received"): 0.0106,
    ("Check for completeness", "Notify accept"): 0.0085,
    ("Request info", "Perform checks"): 0.0084,
    ("info received", "info received"): 0.0084,
    ("Request info", "Make decision"): 0.0079,
    ("review request received", "Make decision"): 0.0076,
    ("Request info", "Request info"): 0.0068,
    ("New online application received", "Perform checks"): 0.0066,
    ("Check for completeness", "Make decision"): 0.0059,
    ("review request received", "Notify accept"): 0.0057,
    ("review request received", "Perform checks"): 0.0057,
    ("Request info", "Check for completeness"): 0.0054,
    ("Check for completeness", "notify reject"): 0.0049,
    ("Request info", "Notify accept"): 0.0047,
    ("Request info", "notify reject"): 0.0047,
    ("info received", "notify reject"): 0.0047,
    ("info received", "Perform checks"): 0.0043,
    ("info received", "Notify accept"): 0.0039,
    ("notify reject", "Make decision"): 0.0036,
    ("New online application received", "Request info"): 0.0034,
    ("review request received", "Request info"): 0.0032,
    ("review request received", "notify reject"): 0.0032,
    ("notify reject", "Perform checks"): 0.0027,
    ("Perform checks", "Check for completeness"): 0.0026,
    ("Perform checks", "review request received"): 0.0026,
    ("review request received", "info received"): 0.0025,
    ("New online application received", "review request received"): 0.002,
    ("Perform checks", "Request info"): 0.0019,
    ("notify reject", "Check for completeness"): 0.0018,
    ("Check for completeness", "info received"): 0.0016,
    ("notify reject", "notify reject"): 0.0015,
    ("info received", "Request info"): 0.0014,
    ("review request received", "review request received"): 0.0013,
    ("Make decision", "info received"): 0.0012,
    ("Check for completeness", "Check for completeness"): 0.0011,
    ("Perform checks", "Perform checks"): 0.001,
    ("Make decision", "Make decision"): 0.001,
    ("notify reject", "Notify accept"): 0.0009,
    ("notify reject", "Request info"): 0.0009,
    ("notify reject", "info received"): 0.0009,
    ("Make decision", "Perform checks"): 0.0008,
}

DURATION_PROFILE = {
    ("Check for completeness", "Request info"): dict(p10=61.9314, p25=170.1115, median=418.728, share=0.097),
    ("Request info", "info received"): dict(p10=10.0, p25=10.0, median=10.0, share=0.996),
    ("info received", "Check for completeness"): dict(p10=63.739, p25=176.05, median=420.682, share=0.095),
    ("Perform checks", "Make decision"): dict(p10=62.8388, p25=171.483, median=422.924, share=0.095),
    ("Check for completeness", "Perform checks"): dict(p10=61.0229, p25=171.5965, median=420.3715, share=0.098),
    ("New online application received", "Check for completeness"): dict(p10=65.3893, p25=171.7955, median=426.0135, share=0.092),
    ("Deliver card", "EVENT 13 END"): dict(p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("Notify accept", "Deliver card"): dict(p10=59.4585, p25=176.35075, median=425.762, share=0.101),
    ("Make decision", "notify reject"): dict(p10=63.7974, p25=174.02075, median=426.0405, share=0.095),
    ("Make decision", "Notify accept"): dict(p10=65.5868, p25=174.826, median=428.03, share=0.089),
    ("time out", "EVENT 13 END"): dict(p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("notify reject", "time out"): dict(p10=10.0, p25=10.0, median=10.0, share=1.0),
    ("notify reject", "review request received"): dict(p10=10.0, p25=10.0, median=10.0, share=1.0),
    ("review request received", "Check for completeness"): dict(p10=78.0242, p25=188.8875, median=441.348, share=0.075),
    ("New online application received", "Notify accept"): dict(p10=944.0129, p25=1469.49875, median=2020.112, share=0.0),
    ("New online application received", "Make decision"): dict(p10=634.1, p25=963.121, median=1661.293, share=0.0),
    ("Check for completeness", "Notify accept"): dict(p10=711.8126, p25=1094.916, median=1780.137, share=0.0),
    ("New online application received", "notify reject"): dict(p10=1132.4149, p25=1538.03275, median=2159.5775, share=0.0),
    ("Check for completeness", "Make decision"): dict(p10=1072.0155, p25=1536.0985, median=2023.867, share=0.0),
    ("info received", "Make decision"): dict(p10=721.489, p25=1122.10325, median=1868.191, share=0.0),
    ("Check for completeness", "notify reject"): dict(p10=945.7493, p25=1361.11625, median=1757.751, share=0.0),
    ("Request info", "Perform checks"): dict(p10=403.0504, p25=643.92, median=928.08, share=0.0),
    ("New online application received", "info received"): dict(p10=452.3138, p25=639.021, median=857.047, share=0.0),
    ("info received", "info received"): dict(p10=613.9782, p25=895.2945, median=1302.5635, share=0.0),
    ("Request info", "Make decision"): dict(p10=669.1557, p25=855.869, median=1523.8755, share=0.0),
    ("Request info", "Request info"): dict(p10=569.524, p25=796.317, median=1164.851, share=0.0),
    ("Request info", "Check for completeness"): dict(p10=803.7977, p25=952.73675, median=1889.409, share=0.0),
    ("New online application received", "Perform checks"): dict(p10=1091.8104, p25=1559.19, median=2363.648, share=0.0),
    ("Request info", "Notify accept"): dict(p10=1429.9083, p25=1610.589, median=2396.886, share=0.0),
    ("Request info", "notify reject"): dict(p10=1539.1919, p25=1979.3755, median=2454.903, share=0.0),
    ("info received", "notify reject"): dict(p10=1059.5184, p25=1682.395, median=2806.157, share=0.0),
    ("info received", "Perform checks"): dict(p10=809.8986, p25=1087.906, median=1874.355, share=0.0),
    ("info received", "Notify accept"): dict(p10=1006.2042, p25=1510.50375, median=2230.734, share=0.0),
    ("Check for completeness", "info received"): dict(p10=817.942, p25=1530.727, median=1910.593, share=0.0),
    ("New online application received", "Request info"): dict(p10=809.1352, p25=1687.418, median=2254.706, share=0.0),
    ("Perform checks", "Check for completeness"): dict(p10=1023.0088, p25=1420.4865, median=1995.413, share=0.0),
    ("Perform checks", "review request received"): dict(p10=378.412, p25=555.4955, median=705.0, share=0.0),
    ("Check for completeness", "Check for completeness"): dict(p10=1452.7526, p25=1939.982, median=2577.014, share=0.0),
    ("notify reject", "Make decision"): dict(p10=785.8718, p25=1341.0605, median=1518.5795, share=0.0),
    ("review request received", "Make decision"): dict(p10=727.506, p25=964.28475, median=2227.617, share=0.0),
    ("Perform checks", "Request info"): dict(p10=1288.529, p25=1794.608, median=2625.632, share=0.0),
    ("New online application received", "review request received"): dict(p10=783.2576, p25=1569.0395, median=1845.21, share=0.0),
    ("notify reject", "Perform checks"): dict(p10=318.1018, p25=699.898, median=874.95, share=0.0),
    ("review request received", "Notify accept"): dict(p10=2066.8628, p25=2191.667, median=2296.639, share=0.0),
    ("Check for completeness", "review request received"): dict(p10=786.9502, p25=1083.927, median=1363.124, share=0.0),
    ("review request received", "Perform checks"): dict(p10=1338.8826, p25=1593.822, median=2249.376, share=0.0),
    ("info received", "Request info"): dict(p10=824.7008, p25=1305.603, median=1993.345, share=0.0),
    ("EVENT 13 END", "EVENT 13 END"): dict(p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("Make decision", "info received"): dict(p10=615.6284, p25=944.3135, median=1863.824, share=0.0),
    ("notify reject", "Check for completeness"): dict(p10=1769.177, p25=2725.29575, median=3075.0545, share=0.0),
    ("Make decision", "Make decision"): dict(p10=1349.042, p25=1693.34325, median=2015.0765, share=0.0),
    ("Perform checks", "Perform checks"): dict(p10=1054.435, p25=1415.281, median=2294.3645, share=0.0),
    ("review request received", "notify reject"): dict(p10=1334.3154, p25=1360.086, median=1469.711, share=0.0),
    ("review request received", "Request info"): dict(p10=1863.9608, p25=2128.058, median=2501.573, share=0.0),
    ("Make decision", "Request info"): dict(p10=1027.2532, p25=1078.57, median=1325.858, share=0.0),
    ("notify reject", "notify reject"): dict(p10=1475.265, p25=2749.563, median=2967.996, share=0.0),
    ("Make decision", "Perform checks"): dict(p10=783.9354, p25=796.233, median=823.397, share=0.0),
    ("review request received", "info received"): dict(p10=1391.3396, p25=1447.0055, median=1983.33, share=0.0),
    ("notify reject", "info received"): dict(p10=652.6522, p25=721.4095, median=836.005, share=0.0),
    ("notify reject", "Request info"): dict(p10=389.1752, p25=613.13, median=986.388, share=0.0),
}

FREQUENT_VARIANTS = [
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
    ["New online application received", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"],
    ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"],
]

GLOBAL_FAST_GAP_SECONDS = 10.0
MIN_GAP_SECONDS = 1.0


# -----------------------------
# Helpers
# -----------------------------
def detect_precision(timestamp_str):
    if "." in str(timestamp_str):
        frac = str(timestamp_str).split(".")[-1]
        return len(frac)
    return 0


def parse_ts(s):
    return pd.to_datetime(s)


def format_ts(dt, precision):
    if precision <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    if precision <= 6:
        whole, frac = base.split(".")
        return whole + "." + frac[:precision]
    return base


def tick_timedelta(precision):
    if precision <= 0:
        return pd.Timedelta(seconds=1)
    return pd.Timedelta(seconds=10 ** (-precision))


def gap_seconds_for_edge(a, b):
    prof = DURATION_PROFILE.get((a, b))
    if prof is None:
        return GLOBAL_FAST_GAP_SECONDS
    if prof["share"] >= 0.8:
        g = prof["p10"]
    else:
        g = prof["p25"]
    if g <= 0:
        g = MIN_GAP_SECONDS
    return max(g, MIN_GAP_SECONDS)


def contiguous_support(seq):
    best = 0
    seq = list(seq)
    for var in FREQUENT_VARIANTS:
        n = len(seq)
        if n > len(var):
            continue
        for i in range(len(var) - n + 1):
            if var[i:i + n] == seq:
                best = max(best, n)
    return best


def full_context_contiguous_support(left_act, perm, right_act):
    best = 0
    target = []
    if left_act is not None:
        target.append(left_act)
    target.extend(perm)
    if right_act is not None:
        target.append(right_act)
    n = len(target)
    for var in FREQUENT_VARIANTS:
        if n > len(var):
            continue
        for i in range(len(var) - n + 1):
            if var[i:i + n] == target:
                best = max(best, n)
    return best


def edge_exists(a, b):
    return (a, b) in TRANSITION_PROB


def score_permutation(left_act, perm, right_act):
    acts = list(perm)

    hard_fail = False
    edge_log = 0.0
    existing_edges = 0
    missing_internal = 0

    chain = []
    if left_act is not None:
        chain.append(left_act)
    chain.extend(acts)
    if right_act is not None:
        chain.append(right_act)

    for i in range(len(chain) - 1):
        x, y = chain[i], chain[i + 1]
        p = TRANSITION_PROB.get((x, y))
        if p is None:
            if i > 0 and i < len(chain) - 2:
                missing_internal += 1
            hard_fail = True
        else:
            existing_edges += 1
            edge_log += math.log(max(p, 1e-12))

    vfull = full_context_contiguous_support(left_act, acts, right_act)
    vblock = contiguous_support(acts)

    return {
        "perm": acts,
        "hard_fail": hard_fail,
        "missing_internal": missing_internal,
        "existing_edges": existing_edges,
        "edge_log": edge_log,
        "vfull": vfull,
        "vblock": vblock,
    }


def choose_order(left_act, block_acts, right_act):
    idx_acts = list(enumerate(block_acts))
    perms = []
    n = len(idx_acts)

    # Enumerate all permutations up to reasonable size; block sizes here are small.
    for perm_idx in itertools.permutations(range(n)):
        perm = [block_acts[i] for i in perm_idx]
        s = score_permutation(left_act, perm, right_act)
        s["perm_idx"] = perm_idx
        perms.append(s)

    valid = [p for p in perms if not p["hard_fail"]]
    if valid:
        ranked = sorted(
            valid,
            key=lambda x: (
                x["vfull"],
                x["vblock"],
                x["existing_edges"],
                x["edge_log"],
            ),
            reverse=True,
        )
        return ranked[0]

    # Relaxation: minimize impossible edges, then maximize variant support and edge evidence.
    ranked = sorted(
        perms,
        key=lambda x: (
            -x["missing_internal"],
            x["vfull"],
            x["vblock"],
            x["existing_edges"],
            x["edge_log"],
        ),
        reverse=True,
    )
    return ranked[0]


def build_times(observed_dt, right_anchor_dt, ordered_acts, precision):
    k = len(ordered_acts)
    tick = tick_timedelta(precision)
    min_gap = max(pd.Timedelta(seconds=MIN_GAP_SECONDS), tick)

    if k == 1:
        return [observed_dt]

    proposed = []
    for i in range(k - 1):
        gsec = gap_seconds_for_edge(ordered_acts[i], ordered_acts[i + 1])
        proposed.append(pd.Timedelta(seconds=float(gsec)))

    # Enforce positive strictly increasing and non-uniform by preserving proposals where possible.
    proposed = [max(g, min_gap) for g in proposed]

    room = None
    if right_anchor_dt is not None:
        room = right_anchor_dt - observed_dt
        if room < pd.Timedelta(0):
            room = pd.Timedelta(0)

    min_total = min_gap * (k - 1)
    total_prop = sum(proposed, pd.Timedelta(0))

    if room is not None:
        # span must be <= room and >= min_total where possible
        if room < min_total:
            # Pack strictly before/on right anchor by backfilling from anchor.
            times = [observed_dt]
            for _ in range(1, k):
                times.append(times[-1] + min_gap)
            # compress with tick if needed but still distinct
            if times[-1] > right_anchor_dt:
                tiny_gap = tick
                times = [observed_dt]
                for _ in range(1, k):
                    times.append(times[-1] + tiny_gap)
                if times[-1] > right_anchor_dt:
                    end = right_anchor_dt
                    start = end - tiny_gap * (k - 1)
                    if start < observed_dt:
                        start = observed_dt
                    times = [start + i * tiny_gap for i in range(k)]
                    if times[0] != observed_dt:
                        times[0] = observed_dt
                        for i in range(1, k):
                            times[i] = max(times[i], times[i - 1] + tiny_gap)
                        if times[-1] > right_anchor_dt:
                            shift = times[-1] - right_anchor_dt
                            times = [t - shift for t in times]
                            if times[0] < observed_dt:
                                times[0] = observed_dt
                                for i in range(1, k):
                                    times[i] = max(times[i], times[i - 1] + tiny_gap)
            return times

        if total_prop > room:
            factor = room / total_prop if total_prop > pd.Timedelta(0) else 1.0
            scaled = [max(pd.Timedelta(seconds=g.total_seconds() * factor), tick) for g in proposed]
            # ensure total <= room by iterative trim while preserving positivity
            if sum(scaled, pd.Timedelta(0)) > room:
                excess = sum(scaled, pd.Timedelta(0)) - room
                flex = sum((g - tick for g in scaled), pd.Timedelta(0))
                if flex > pd.Timedelta(0):
                    new_scaled = []
                    for g in scaled:
                        reducible = g - tick
                        cut = excess * (reducible / flex) if flex > pd.Timedelta(0) else pd.Timedelta(0)
                        ng = max(g - cut, tick)
                        new_scaled.append(ng)
                    scaled = new_scaled
            proposed = scaled

    times = [observed_dt]
    for g in proposed:
        times.append(times[-1] + g)

    return times


def enforce_block_constraints(observed_dt, right_anchor_dt, times, precision):
    tick = tick_timedelta(precision)
    min_gap = max(pd.Timedelta(seconds=MIN_GAP_SECONDS), tick)
    clamped = False

    times[0] = observed_dt

    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + min_gap
            clamped = True

    if right_anchor_dt is not None and len(times) > 1:
        max_span = right_anchor_dt - observed_dt
        if max_span < pd.Timedelta(0):
            max_span = pd.Timedelta(0)

        min_needed = min_gap * (len(times) - 1)
        if max_span >= min_needed:
            if times[-1] > right_anchor_dt:
                actual_gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
                total = sum(actual_gaps, pd.Timedelta(0))
                if total <= pd.Timedelta(0):
                    actual_gaps = [min_gap] * (len(times) - 1)
                    total = min_needed
                factor = max_span / total if total > pd.Timedelta(0) else 1.0
                scaled = [max(pd.Timedelta(seconds=g.total_seconds() * factor), min_gap) for g in actual_gaps]
                total2 = sum(scaled, pd.Timedelta(0))
                if total2 > max_span:
                    excess = total2 - max_span
                    flex = sum((g - min_gap for g in scaled), pd.Timedelta(0))
                    if flex > pd.Timedelta(0):
                        adjusted = []
                        for g in scaled:
                            reducible = g - min_gap
                            cut = excess * (reducible / flex)
                            adjusted.append(max(g - cut, min_gap))
                        scaled = adjusted
                new_times = [observed_dt]
                for g in scaled:
                    new_times.append(new_times[-1] + g)
                times = new_times
                clamped = True
        else:
            tiny_gap = tick
            new_times = [observed_dt]
            for _ in range(1, len(times)):
                new_times.append(new_times[-1] + tiny_gap)
            if new_times[-1] > right_anchor_dt:
                end = right_anchor_dt
                start = max(observed_dt, end - tiny_gap * (len(times) - 1))
                new_times = [start + i * tiny_gap for i in range(len(times))]
                new_times[0] = observed_dt
                for i in range(1, len(new_times)):
                    if new_times[i] <= new_times[i - 1]:
                        new_times[i] = new_times[i - 1] + tiny_gap
            times = new_times
            clamped = True

    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + tick
            clamped = True

    if right_anchor_dt is not None:
        for i in range(len(times)):
            if times[i] > right_anchor_dt:
                times[i] = right_anchor_dt
                clamped = True
        for i in range(len(times) - 2, -1, -1):
            if times[i] >= times[i + 1]:
                times[i] = times[i + 1] - tick
                clamped = True
        if times[0] < observed_dt:
            times[0] = observed_dt
            clamped = True
            for i in range(1, len(times)):
                if times[i] <= times[i - 1]:
                    times[i] = times[i - 1] + tick
                    clamped = True

    return times, clamped


def subseq_positions(order_acts):
    pos = defaultdict(list)
    for i, a in enumerate(order_acts):
        pos[a].append(i)
    return pos


# -----------------------------
# Main repair
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.DataFrame(columns=["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])

    orig_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in orig_cols:
        if c not in df.columns:
            df[c] = ""

    df = df[orig_cols].copy()
    df["_orig_ts"] = df["Timestamp"]
    df["_parsed_ts"] = df["Timestamp"].map(parse_ts)
    precision = 0
    for s in df["Timestamp"]:
        precision = max(precision, detect_precision(s))

    df["_out_ts"] = df["_orig_ts"]

    blocks_repaired = 0
    blocks_clamped = 0

    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        case_df = df.loc[idxs]
        case_start = case_df["_parsed_ts"].min()
        case_end = case_df["_parsed_ts"].max()

        i = 0
        while i < len(idxs):
            ridx = idxs[i]
            if df.at[ridx, "error_category"] != "C1_ORDERING_AMBIGUITY":
                i += 1
                continue

            obs_ts_str = df.at[ridx, "_orig_ts"]
            obs_ts = df.at[ridx, "_parsed_ts"]

            j = i
            block = []
            while j < len(idxs):
                rr = idxs[j]
                if (
                    df.at[rr, "error_category"] == "C1_ORDERING_AMBIGUITY"
                    and df.at[rr, "_orig_ts"] == obs_ts_str
                ):
                    block.append(rr)
                    j += 1
                else:
                    break

            k = len(block)
            left_anchor_idx = None
            for p in range(i - 1, -1, -1):
                rr = idxs[p]
                if df.at[rr, "error_category"] == "CLEAN":
                    left_anchor_idx = rr
                    break

            right_anchor_idx = None
            for p in range(j, len(idxs)):
                rr = idxs[p]
                if df.at[rr, "error_category"] == "CLEAN":
                    right_anchor_idx = rr
                    break

            left_act = df.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None else None
            right_act = df.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None else None
            right_anchor_dt = df.at[right_anchor_idx, "_parsed_ts"] if right_anchor_idx is not None else case_end

            # Clamp right anchor to case end and ensure not earlier than observed
            if right_anchor_dt is None or pd.isna(right_anchor_dt):
                right_anchor_dt = case_end
            if right_anchor_dt < obs_ts:
                right_anchor_dt = obs_ts

            block_acts = [df.at[r, "Activity"] for r in block]
            chosen = choose_order(left_act, block_acts, right_act)
            perm_idx = chosen["perm_idx"]
            ordered_rows = [block[p] for p in perm_idx]
            ordered_acts = [df.at[r, "Activity"] for r in ordered_rows]

            times = build_times(obs_ts, right_anchor_dt, ordered_acts, precision)
            times, clamped = enforce_block_constraints(obs_ts, right_anchor_dt, times, precision)

            # Final local self-healing against any negative durations or duplicates
            tick = tick_timedelta(precision)
            times[0] = obs_ts
            for t in range(1, len(times)):
                if times[t] <= times[t - 1]:
                    times[t] = times[t - 1] + max(pd.Timedelta(seconds=MIN_GAP_SECONDS), tick)
                    clamped = True
            if right_anchor_dt is not None and times[-1] > right_anchor_dt:
                # Rebuild with growing tiny increments from observed within room.
                room = right_anchor_dt - obs_ts
                if room < pd.Timedelta(0):
                    room = pd.Timedelta(0)
                base_tick = tick
                gaps = [base_tick * (m + 1) for m in range(len(times) - 1)]
                total = sum(gaps, pd.Timedelta(0))
                if total > room and room > pd.Timedelta(0):
                    gaps = [base_tick] * (len(times) - 1)
                    total = sum(gaps, pd.Timedelta(0))
                new_times = [obs_ts]
                for g in gaps:
                    new_times.append(new_times[-1] + g)
                if new_times[-1] > right_anchor_dt:
                    end = right_anchor_dt
                    start = max(obs_ts, end - base_tick * (len(times) - 1))
                    new_times = [start + q * base_tick for q in range(len(times))]
                    new_times[0] = obs_ts
                    for q in range(1, len(new_times)):
                        if new_times[q] <= new_times[q - 1]:
                            new_times[q] = new_times[q - 1] + base_tick
                times = new_times
                clamped = True

            # Write timestamps back to each original row by inferred order position.
            for row_idx, ts in zip(ordered_rows, times):
                df.at[row_idx, "_out_ts"] = format_ts(ts, precision)

            # Block-level self-check and corrective remap if some non-first row remained unchanged.
            unchanged_nonfirst = []
            for pos, row_idx in enumerate(ordered_rows):
                if pos == 0:
                    continue
                if df.at[row_idx, "_out_ts"] == df.at[row_idx, "_orig_ts"]:
                    unchanged_nonfirst.append((pos, row_idx))
            if unchanged_nonfirst:
                clamped = True
                for pos, row_idx in unchanged_nonfirst:
                    ts = parse_ts(df.at[row_idx, "_out_ts"]) + tick * (pos + 1)
                    if ts > right_anchor_dt:
                        ts = right_anchor_dt - tick * max(0, (len(ordered_rows) - 1 - pos))
                    if ts <= parse_ts(df.at[ordered_rows[pos - 1], "_out_ts"]):
                        ts = parse_ts(df.at[ordered_rows[pos - 1], "_out_ts"]) + tick
                    df.at[row_idx, "_out_ts"] = format_ts(ts, precision)

            if clamped:
                blocks_clamped += 1
            blocks_repaired += 1
            i = j

    # Global self-check / repair pass per block again
    rows_equal_input = 0
    for idx in df.index:
        if df.at[idx, "_out_ts"] == df.at[idx, "_orig_ts"]:
            rows_equal_input += 1

    # Ensure CLEAN rows unchanged and corrupted rows changed except one per block.
    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        i = 0
        while i < len(idxs):
            ridx = idxs[i]
            if df.at[ridx, "error_category"] != "C1_ORDERING_AMBIGUITY":
                if df.at[ridx, "_out_ts"] != df.at[ridx, "_orig_ts"]:
                    df.at[ridx, "_out_ts"] = df.at[ridx, "_orig_ts"]
                i += 1
                continue

            obs_ts_str = df.at[ridx, "_orig_ts"]
            j = i
            block = []
            while j < len(idxs):
                rr = idxs[j]
                if (
                    df.at[rr, "error_category"] == "C1_ORDERING_AMBIGUITY"
                    and df.at[rr, "_orig_ts"] == obs_ts_str
                ):
                    block.append(rr)
                    j += 1
                else:
                    break

            # Determine inferred first as unique row still carrying observed timestamp, otherwise force one.
            same_rows = [r for r in block if df.at[r, "_out_ts"] == obs_ts_str]
            if len(same_rows) != 1:
                # Reconstruct from current output order by sorted repaired timestamps
                parsed = [(r, parse_ts(df.at[r, "_out_ts"])) for r in block]
                parsed.sort(key=lambda x: x[1])
                first_row = parsed[0][0]
                obs_dt = parse_ts(obs_ts_str)
                tick = tick_timedelta(precision)
                current_times = [obs_dt + n * max(pd.Timedelta(seconds=MIN_GAP_SECONDS), tick) for n in range(len(block))]
                for n, (r, _) in enumerate(parsed):
                    df.at[r, "_out_ts"] = format_ts(current_times[n], precision)
                same_rows = [first_row]
                blocks_clamped += 1

            # Make sure all corrupted non-first rows differ from input.
            parsed = [(r, parse_ts(df.at[r, "_out_ts"])) for r in block]
            parsed.sort(key=lambda x: x[1])
            tick = tick_timedelta(precision)
            for pos, (r, ts) in enumerate(parsed):
                if pos == 0:
                    df.at[r, "_out_ts"] = obs_ts_str
                else:
                    if df.at[r, "_out_ts"] == df.at[r, "_orig_ts"]:
                        new_ts = ts + tick * pos
                        df.at[r, "_out_ts"] = format_ts(new_ts, precision)
                        blocks_clamped += 1
            i = j

    rows_equal_input = int((df["_out_ts"] == df["_orig_ts"]).sum())

    # Final output assembly
    out = df[orig_cols].copy()
    out["Timestamp"] = df["_out_ts"]

    try:
        out.to_csv(args.output, index=False)
    except Exception:
        # Last-resort best effort
        try:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                f.write(",".join(orig_cols) + "\n")
                for _, row in out.iterrows():
                    vals = [str(row[c]) for c in orig_cols]
                    f.write(",".join(vals) + "\n")
        except Exception:
            pass

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={blocks_clamped} rows_still_equal_input={rows_equal_input}")


if __name__ == "__main__":
    main()