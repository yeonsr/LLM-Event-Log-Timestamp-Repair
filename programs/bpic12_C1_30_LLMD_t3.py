import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from itertools import permutations

import pandas as pd


TRANSITION_FREQ = {
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): 6995,
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): 3345,
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): 12671,
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): 6642,
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): 4038,
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): 6533,
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): 2128,
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): 6298,
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): 4289,
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): 4249,
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): 8973,
    ("O_DECLINED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 0,
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): 16578,
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): 8830,
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): 1662,
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): 4829,
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): 15747,
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): 15330,
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): 15647,
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): 1638,
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): 2361,
    ("W_Beoordelen fraude-COMPLETE", "W_Beoordelen fraude-START"): 0,
    ("W_Wijzigen contractgegevens-SCHEDULE", "W_Wijzigen contractgegevens-SCHEDULE"): 0,
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): 1466,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 3171,
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): 4318,
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): 2007,
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): 1285,
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): 2830,
    ("W_Beoordelen fraude-START", "W_Beoordelen fraude-COMPLETE"): 0,
    ("W_Beoordelen fraude-SCHEDULE", "W_Beoordelen fraude-START"): 0,
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): 1923,
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): 1126,
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 1009,
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): 2298,
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): 2145,
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 862,
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): 2817,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): 4564,
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): 2328,
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): 1770,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): 1698,
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): 4175,
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): 992,
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 730,
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): 1031,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): 3533,
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 1054,
    ("A_ACTIVATED-COMPLETE", "A_REGISTERED-COMPLETE"): 0,
    ("W_Beoordelen fraude-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): 0,
    ("A_APPROVED-COMPLETE", "O_ACCEPTED-COMPLETE"): 0,
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 1632,
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): 1927,
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): 4800,
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): 4764,
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): 801,
    ("W_Beoordelen fraude-START", "A_DECLINED-COMPLETE"): 0,
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): 1027,
    ("W_Beoordelen fraude-SCHEDULE", "W_Afhandelen leads-COMPLETE"): 0,
    ("W_Wijzigen contractgegevens-SCHEDULE", "W_Nabellen offertes-START"): 0,
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): 713,
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): 697,
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): 3127,
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): 3123,
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): 1612,
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): 911,
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): 1502,
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): 1706,
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): 862,
    ("W_Valideren aanvraag-START", "A_APPROVED-COMPLETE"): 702,
    ("A_SUBMITTED-COMPLETE", "W_Afhandelen leads-COMPLETE"): 0,
}

DURATION_PROFILES = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): {"p10": 51.48200000000001, "p25": 14904.8565, "median": 167152.1195, "share": 0.104},
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): {"p10": 18.4252, "p25": 56.2145, "median": 116.085, "share": 0.267},
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): {"p10": 10.703, "p25": 40.826, "median": 98.688, "share": 0.339},
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 62.131800000000005, "p25": 1614.967, "median": 14829.922999999999, "share": 0.099},
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): {"p10": 0.125, "p25": 0.173, "median": 0.273, "share": 1.0},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 75.33920000000005, "p25": 1281.503, "median": 10850.015, "share": 0.093},
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 10.189, "p25": 33.673, "median": 140.75650000000002, "share": 0.329},
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): {"p10": 0.025, "p25": 0.028, "median": 0.035, "share": 1.0},
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): {"p10": 0.046, "p25": 0.131, "median": 0.378, "share": 1.0},
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): {"p10": 0.08220000000000005, "p25": 0.094, "median": 0.119, "share": 1.0},
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9147000000000001, "p25": 1.084, "median": 1.29, "share": 1.0},
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.077, "p25": 1.179, "median": 1.387, "share": 1.0},
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): {"p10": 12.5843, "p25": 19.862750000000002, "median": 305.6565, "share": 0.425},
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): {"p10": 123.8471, "p25": 245.35725, "median": 373.4235, "share": 0.049},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): {"p10": 32.8282, "p25": 35.614, "median": 38.9685, "share": 0.968},
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): {"p10": 28.597200000000004, "p25": 698.53475, "median": 6869.639, "share": 0.152},
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 63.803, "p25": 118.448, "median": 183.982, "share": 0.093},
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): {"p10": 8.712200000000001, "p25": 11.942, "median": 22.613, "share": 0.95},
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): {"p10": 78.05730000000003, "p25": 584.8657499999999, "median": 2541.5485, "share": 0.089},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.7478, "p25": 34.197, "median": 37.565, "share": 0.951},
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): {"p10": 0.141, "p25": 0.153, "median": 0.18, "share": 0.999},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.115, "p25": 1.229, "median": 1.45, "share": 1.0},
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 6744.789800000003, "p25": 89237.016, "median": 189089.061, "share": 0.048},
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): {"p10": 10.1908, "p25": 12.4835, "median": 21.627, "share": 0.818},
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 24.4568, "p25": 242.11625, "median": 2012.101, "share": 0.143},
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): {"p10": 10.74, "p25": 44.596, "median": 303.645, "share": 0.285},
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 17.397, "p25": 844.995, "median": 5791.562, "share": 0.153},
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): {"p10": 0.898, "p25": 0.987, "median": 1.148, "share": 1.0},
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): {"p10": 35.6911, "p25": 65.77924999999999, "median": 107.6855, "share": 0.213},
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 26.0428, "p25": 332.025, "median": 2923.088, "share": 0.157},
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): {"p10": 1.448, "p25": 1.61, "median": 1.8525, "share": 1.0},
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): {"p10": 249.9444, "p25": 623.466, "median": 1169.758, "share": 0.029},
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): {"p10": 2.8494, "p25": 3.0905, "median": 3.415, "share": 0.998},
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): {"p10": 32.098000000000006, "p25": 69.0895, "median": 112.2, "share": 0.204},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 697.8025, "p25": 2132.33425, "median": 5040.9805, "share": 0.03},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 1.4857, "p25": 1.65025, "median": 1.8965, "share": 0.999},
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.2241, "p25": 1.3207499999999999, "median": 1.694, "share": 0.96},
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): {"p10": 31.9407, "p25": 48.406000000000006, "median": 106.4295, "share": 0.318},
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): {"p10": 24.8903, "p25": 36.49575, "median": 65.1735, "share": 0.469},
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share": 1.0},
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 2.9613, "p25": 3.188, "median": 3.5300000000000002, "share": 0.999},
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): {"p10": 3.878, "p25": 16.207500000000003, "median": 339.963, "share": 0.335},
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): {"p10": 7.172200000000002, "p25": 42.710499999999996, "median": 141.496, "share": 0.299},
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.4536, "p25": 7.027, "median": 7.763, "share": 0.999},
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): {"p10": 176.1539, "p25": 313.63275, "median": 474.154, "share": 0.023},
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.837, "p25": 2.092, "median": 2.3964999999999996, "share": 0.998},
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): {"p10": 72.639, "p25": 111.07249999999999, "median": 156.875, "share": 0.081},
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.3086, "p25": 6.952, "median": 7.6685, "share": 1.0},
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): {"p10": 141.36260000000001, "p25": 317.4815, "median": 770.573, "share": 0.019},
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.3109, "p25": 1.466, "median": 1.668, "share": 0.999},
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9154000000000001, "p25": 1.056, "median": 1.252, "share": 1.0},
    ("W_Valideren aanvraag-START", "A_APPROVED-COMPLETE"): {"p10": 142.19940000000005, "p25": 329.30825, "median": 739.4095, "share": 0.02},
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "share": 1.0},
}

FREQUENT_VARIANTS = [
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def detect_precision(ts_str):
    if "." in ts_str:
        frac = ts_str.split(".")[-1]
        return len(frac)
    return 0


def parse_ts(ts_str):
    return pd.to_datetime(ts_str)


def format_ts(dt, precision):
    if precision <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    nanos = dt.nanosecond + dt.microsecond * 1000
    frac = f"{nanos:09d}"[:precision]
    return f"{base}.{frac}"


def smallest_unit_seconds(precision):
    if precision <= 0:
        return 1.0
    return 10 ** (-precision)


def subseq_score(seq, left_act, right_act):
    best = 0
    target = seq[:]
    for var in FREQUENT_VARIANTS:
        ext = var[:]
        if left_act is not None:
            ext = [left_act] + ext
        if right_act is not None:
            ext = ext + [right_act]
        for i in range(len(ext) - len(target) + 1):
            if ext[i:i + len(target)] == target:
                best = max(best, 1)
    return best


def edge_score(a, b):
    return TRANSITION_FREQ.get((a, b), 0)


def score_order(order, left_act, right_act):
    internal_supported = 0
    internal_missing = 0
    score = 0.0

    for i in range(len(order) - 1):
        f = edge_score(order[i], order[i + 1])
        if f > 0:
            internal_supported += 1
            score += math.log1p(f) * 8.0
        else:
            internal_missing += 1
            score -= 50.0

    if left_act is not None:
        lf = edge_score(left_act, order[0])
        if lf > 0:
            score += math.log1p(lf) * 3.0
        else:
            score -= 3.0

    if right_act is not None:
        rf = edge_score(order[-1], right_act)
        if rf > 0:
            score += math.log1p(rf) * 3.0
        else:
            score -= 3.0

    if subseq_score(order, left_act, right_act):
        score += 100.0

    score += internal_supported * 5.0
    score -= internal_missing * 10.0
    return score, internal_supported, internal_missing


def generate_candidate_orders(activities, left_act, right_act):
    n = len(activities)
    if n <= 7:
        return list(dict.fromkeys(permutations(activities, n)))

    candidates = []
    counts = Counter(activities)

    for var in FREQUENT_VARIANTS:
        for i in range(len(var) - n + 1):
            sub = var[i:i + n]
            if Counter(sub) == counts:
                candidates.append(tuple(sub))

    if not candidates:
        remaining = list(activities)
        built = []
        if left_act is not None:
            while remaining:
                best_idx = max(range(len(remaining)), key=lambda i: edge_score(left_act if not built else built[-1], remaining[i]))
                built.append(remaining.pop(best_idx))
        else:
            start_idx = max(range(len(remaining)), key=lambda i: edge_score(remaining[i], right_act) if right_act is not None else 0)
            built.append(remaining.pop(start_idx))
            while remaining:
                best_idx = max(range(len(remaining)), key=lambda i: edge_score(built[-1], remaining[i]))
                built.append(remaining.pop(best_idx))
        candidates.append(tuple(built))

    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:200]


def choose_order(activities, left_act, right_act):
    candidates = generate_candidate_orders(activities, left_act, right_act)
    scored = []
    for cand in candidates:
        s, supported, missing = score_order(list(cand), left_act, right_act)
        scored.append((s, supported, -missing, cand))
    scored.sort(reverse=True)

    if scored:
        best = scored[0][3]
        return list(best)

    return list(activities)


def edge_step_seconds(a, b, precision):
    prof = DURATION_PROFILES.get((a, b))
    min_step = max(1.0, smallest_unit_seconds(precision))
    if prof is None:
        return min_step
    val = prof["p10"] if prof["share"] >= 0.8 or prof["p25"] <= 2.0 else prof["p25"]
    return max(min_step, float(val))


def assign_times_for_order(order, observed_dt, right_anchor_dt, precision):
    min_step = max(1.0, smallest_unit_seconds(precision))
    steps = []
    for i in range(len(order) - 1):
        steps.append(edge_step_seconds(order[i], order[i + 1], precision))

    if not steps:
        return [observed_dt]

    cumulative = [0.0]
    for s in steps:
        cumulative.append(cumulative[-1] + s)

    required_min_span = max((len(order) - 1) * 1.0, cumulative[-1])
    span = cumulative[-1]

    if right_anchor_dt is not None:
        room = (right_anchor_dt - observed_dt).total_seconds()
        room = max(0.0, room)
        max_allowed = room
        target_max = max_allowed
        if target_max < (len(order) - 1) * min_step:
            target_max = (len(order) - 1) * min_step
        if span > max_allowed and span > 0:
            factor = max_allowed / span if max_allowed > 0 else 0.0
            scaled_steps = [max(min_step, s * factor) for s in steps]
            scaled_sum = sum(scaled_steps)
            if scaled_sum > max_allowed and scaled_sum > 0:
                overflow = scaled_sum - max_allowed
                reducible = [s - min_step for s in scaled_steps]
                total_reducible = sum(max(0.0, x) for x in reducible)
                if total_reducible > 0:
                    for i in range(len(scaled_steps)):
                        r = max(0.0, reducible[i])
                        scaled_steps[i] -= overflow * (r / total_reducible)
                        if scaled_steps[i] < min_step:
                            scaled_steps[i] = min_step
            steps = scaled_steps

    times = [observed_dt]
    cur = observed_dt
    for s in steps:
        cur = cur + pd.to_timedelta(s, unit="s")
        times.append(cur)
    return times


def enforce_block_constraints(block_rows, row_to_newdt, observed_dt, left_anchor_dt, right_anchor_dt, case_start, case_end, precision):
    clamped = False
    min_delta = pd.to_timedelta(max(1.0, smallest_unit_seconds(precision)), unit="s")
    ordered = sorted(block_rows, key=lambda r: row_to_newdt[r["__idx"]])

    row_to_newdt[ordered[0]["__idx"]] = observed_dt

    for i in range(1, len(ordered)):
        prev = row_to_newdt[ordered[i - 1]["__idx"]]
        cur = row_to_newdt[ordered[i]["__idx"]]
        if cur <= prev:
            row_to_newdt[ordered[i]["__idx"]] = prev + min_delta
            clamped = True

    lower = max([x for x in [case_start, left_anchor_dt] if x is not None], default=None)
    upper = min([x for x in [case_end, right_anchor_dt] if x is not None], default=None)

    if lower is not None and row_to_newdt[ordered[0]["__idx"]] < lower:
        shift = lower - row_to_newdt[ordered[0]["__idx"]]
        for r in ordered:
            row_to_newdt[r["__idx"]] = row_to_newdt[r["__idx"]] + shift
        row_to_newdt[ordered[0]["__idx"]] = observed_dt if observed_dt >= lower else lower
        clamped = True

    for i in range(1, len(ordered)):
        prev = row_to_newdt[ordered[i - 1]["__idx"]]
        if row_to_newdt[ordered[i]["__idx"]] <= prev:
            row_to_newdt[ordered[i]["__idx"]] = prev + min_delta
            clamped = True

    if upper is not None and row_to_newdt[ordered[-1]["__idx"]] > upper:
        latest_allowed = upper
        row_to_newdt[ordered[-1]["__idx"]] = latest_allowed
        for i in range(len(ordered) - 2, -1, -1):
            nxt = row_to_newdt[ordered[i + 1]["__idx"]]
            allowed = nxt - min_delta
            if i == 0:
                allowed = max(allowed, observed_dt)
            row_to_newdt[ordered[i]["__idx"]] = min(row_to_newdt[ordered[i]["__idx"]], allowed)
        row_to_newdt[ordered[0]["__idx"]] = observed_dt
        for i in range(1, len(ordered)):
            prev = row_to_newdt[ordered[i - 1]["__idx"]]
            if row_to_newdt[ordered[i]["__idx"]] <= prev:
                row_to_newdt[ordered[i]["__idx"]] = prev + min_delta
        clamped = True

    if right_anchor_dt is not None:
        max_span = (right_anchor_dt - observed_dt).total_seconds()
        cur_span = (row_to_newdt[ordered[-1]["__idx"]] - observed_dt).total_seconds()
        if cur_span > max_span:
            row_to_newdt[ordered[-1]["__idx"]] = right_anchor_dt
            for i in range(len(ordered) - 2, 0, -1):
                nxt = row_to_newdt[ordered[i + 1]["__idx"]]
                target = min(row_to_newdt[ordered[i]["__idx"]], nxt - min_delta)
                row_to_newdt[ordered[i]["__idx"]] = target
            row_to_newdt[ordered[0]["__idx"]] = observed_dt
            for i in range(1, len(ordered)):
                prev = row_to_newdt[ordered[i - 1]["__idx"]]
                if row_to_newdt[ordered[i]["__idx"]] <= prev:
                    row_to_newdt[ordered[i]["__idx"]] = prev + min_delta
            clamped = True

    min_required_end = observed_dt + pd.to_timedelta((len(ordered) - 1), unit="s")
    if row_to_newdt[ordered[-1]["__idx"]] < min_required_end:
        candidate_end = min_required_end
        if upper is not None and candidate_end > upper:
            candidate_end = upper
        row_to_newdt[ordered[-1]["__idx"]] = candidate_end
        for i in range(len(ordered) - 2, 0, -1):
            row_to_newdt[ordered[i]["__idx"]] = min(
                row_to_newdt[ordered[i]["__idx"]],
                row_to_newdt[ordered[i + 1]["__idx"]] - min_delta
            )
        row_to_newdt[ordered[0]["__idx"]] = observed_dt
        for i in range(1, len(ordered)):
            prev = row_to_newdt[ordered[i - 1]["__idx"]]
            if row_to_newdt[ordered[i]["__idx"]] <= prev:
                row_to_newdt[ordered[i]["__idx"]] = prev + min_delta
        clamped = True

    return clamped


def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    required = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in required:
        if c not in df.columns:
            pass

    df["__idx"] = range(len(df))
    df["__orig_ts"] = df["Timestamp"]
    precisions = [detect_precision(x) for x in df["Timestamp"].astype(str).tolist()]
    global_precision = max(precisions) if precisions else 3

    out_ts = df["Timestamp"].astype(str).tolist()
    parsed_ts = [parse_ts(x) for x in df["Timestamp"].astype(str).tolist()]

    blocks_repaired = 0
    blocks_clamped = 0

    by_case = defaultdict(list)
    for rec in df.to_dict("records"):
        by_case[rec["Case"]].append(rec)

    for case, rows in by_case.items():
        rows_sorted = sorted(rows, key=lambda r: r["__idx"])
        case_times = [parsed_ts[r["__idx"]] for r in rows_sorted]
        case_start = min(case_times) if case_times else None
        case_end = max(case_times) if case_times else None

        i = 0
        while i < len(rows_sorted):
            r = rows_sorted[i]
            if r["error_category"] != "C1_ORDERING_AMBIGUITY":
                i += 1
                continue

            observed_str = r["Timestamp"]
            block = [r]
            j = i + 1
            while j < len(rows_sorted):
                rr = rows_sorted[j]
                if rr["error_category"] == "C1_ORDERING_AMBIGUITY" and rr["Timestamp"] == observed_str:
                    block.append(rr)
                    j += 1
                else:
                    break

            observed_dt = parse_ts(observed_str)
            precision = max(detect_precision(x["Timestamp"]) for x in block) if block else global_precision

            left_anchor = rows_sorted[i - 1] if i - 1 >= 0 and rows_sorted[i - 1]["error_category"] == "CLEAN" else None
            k = j
            while k < len(rows_sorted) and rows_sorted[k]["error_category"] != "CLEAN":
                k += 1
            right_anchor = rows_sorted[k] if k < len(rows_sorted) and rows_sorted[k]["error_category"] == "CLEAN" else None

            left_anchor_dt = parse_ts(left_anchor["Timestamp"]) if left_anchor is not None else None
            right_anchor_dt = parse_ts(right_anchor["Timestamp"]) if right_anchor is not None else None
            left_act = left_anchor["Activity"] if left_anchor is not None else None
            right_act = right_anchor["Activity"] if right_anchor is not None else None

            activities = [x["Activity"] for x in block]
            order = choose_order(activities, left_act, right_act)

            act_to_rows = defaultdict(list)
            for br in block:
                act_to_rows[br["Activity"]].append(br)
            for act in act_to_rows:
                act_to_rows[act] = sorted(act_to_rows[act], key=lambda z: z["__idx"])

            ordered_rows = []
            used_count = defaultdict(int)
            for act in order:
                idx = used_count[act]
                ordered_rows.append(act_to_rows[act][idx])
                used_count[act] += 1

            times = assign_times_for_order(order, observed_dt, right_anchor_dt, precision)
            row_to_newdt = {}
            for br, t in zip(ordered_rows, times):
                row_to_newdt[br["__idx"]] = t

            clamped = enforce_block_constraints(
                block, row_to_newdt, observed_dt, left_anchor_dt, right_anchor_dt, case_start, case_end, precision
            )

            ordered_rows_sorted_by_time = sorted(block, key=lambda rr: row_to_newdt[rr["__idx"]])
            earliest_idx = ordered_rows_sorted_by_time[0]["__idx"]

            for br in block:
                idx = br["__idx"]
                if idx == earliest_idx:
                    out_ts[idx] = observed_str
                else:
                    dt = row_to_newdt[idx]
                    if dt == observed_dt:
                        dt = observed_dt + pd.to_timedelta(
                            (ordered_rows_sorted_by_time.index(br) + 1), unit="s"
                        )
                        row_to_newdt[idx] = dt
                        clamped = True
                    out_ts[idx] = format_ts(dt, precision)

            changed_nonfirst = 0
            for br in block:
                idx = br["__idx"]
                if idx != earliest_idx and out_ts[idx] == br["Timestamp"]:
                    dt = row_to_newdt[idx] + pd.to_timedelta(1, unit="s")
                    if right_anchor_dt is not None and dt > right_anchor_dt:
                        dt = row_to_newdt[idx] - pd.to_timedelta(1, unit="s")
                        if dt <= observed_dt:
                            dt = observed_dt + pd.to_timedelta(ordered_rows_sorted_by_time.index(br), unit="s")
                    row_to_newdt[idx] = dt
                    out_ts[idx] = format_ts(dt, precision)
                    clamped = True
                elif idx != earliest_idx:
                    changed_nonfirst += 1

            parsed_block = [(br["__idx"], parse_ts(out_ts[br["__idx"]])) for br in block]
            parsed_block.sort(key=lambda x: x[1])

            if parsed_block[0][1] != observed_dt:
                out_ts[parsed_block[0][0]] = observed_str
                clamped = True

            time_values = [parse_ts(out_ts[br["__idx"]]) for br in ordered_rows]
            for z in range(1, len(time_values)):
                if time_values[z] <= time_values[z - 1]:
                    fixed = time_values[z - 1] + pd.to_timedelta(1, unit="s")
                    if right_anchor_dt is not None and fixed > right_anchor_dt:
                        fixed = right_anchor_dt
                    out_ts[ordered_rows[z]["__idx"]] = format_ts(fixed, precision)
                    time_values[z] = fixed
                    clamped = True

            blocks_repaired += 1
            if clamped:
                blocks_clamped += 1

            i = j

    rows_still_equal = 0
    for idx in range(len(df)):
        if df.iloc[idx]["error_category"] == "CLEAN":
            out_ts[idx] = df.iloc[idx]["Timestamp"]
        if out_ts[idx] == df.iloc[idx]["Timestamp"]:
            rows_still_equal += 1

    out_df = df[["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]].copy()
    out_df["Timestamp"] = out_ts

    try:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])
            for _, row in out_df.iterrows():
                writer.writerow([row["Case"], row["Activity"], row["Timestamp"], row["Resource"], row["error_category"], row["row_id"]])
    except Exception:
        pass

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={blocks_clamped} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    main()