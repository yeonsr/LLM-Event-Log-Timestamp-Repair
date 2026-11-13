import argparse
import math
import itertools
import pandas as pd
from collections import Counter, defaultdict

# -----------------------------
# Embedded process abstraction
# -----------------------------
TRANSITION_PROB = {
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): 0.996,
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): 0.9944,
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): 0.9682,
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): 0.9653,
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): 0.9601,
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): 0.933,
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): 0.909,
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): 0.9002,
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): 0.8575,
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): 0.8397,
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): 0.8371,
    ("O_DECLINED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 0.834,
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): 0.7948,
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): 0.792,
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): 0.74,
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): 0.7391,
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): 0.7321,
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): 0.7267,
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): 0.7062,
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): 0.6917,
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): 0.6855,
    ("W_Beoordelen fraude-COMPLETE", "W_Beoordelen fraude-START"): 0.67,
    ("W_Wijzigen contractgegevens-SCHEDULE", "W_Wijzigen contractgegevens-SCHEDULE"): 0.6667,
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): 0.6527,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 0.6423,
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): 0.6405,
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): 0.6367,
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): 0.5729,
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): 0.5623,
    ("W_Beoordelen fraude-START", "W_Beoordelen fraude-COMPLETE"): 0.556,
    ("W_Beoordelen fraude-SCHEDULE", "W_Beoordelen fraude-START"): 0.5086,
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): 0.5017,
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): 0.5016,
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 0.4494,
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): 0.4478,
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): 0.4262,
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 0.3843,
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): 0.3648,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): 0.3602,
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): 0.3453,
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): 0.3449,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): 0.3439,
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): 0.3295,
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): 0.3147,
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 0.3083,
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): 0.2994,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): 0.2788,
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 0.275,
    ("A_ACTIVATED-COMPLETE", "A_REGISTERED-COMPLETE"): 0.26,
    ("W_Beoordelen fraude-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): 0.2586,
    ("A_APPROVED-COMPLETE", "O_ACCEPTED-COMPLETE"): 0.2547,
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 0.2498,
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): 0.2496,
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): 0.2276,
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): 0.215,
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): 0.209,
    ("W_Beoordelen fraude-START", "A_DECLINED-COMPLETE"): 0.208,
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): 0.2001,
    ("W_Beoordelen fraude-SCHEDULE", "W_Afhandelen leads-COMPLETE"): 0.181,
    ("W_Wijzigen contractgegevens-SCHEDULE", "W_Nabellen offertes-START"): 0.1667,
}

DUR = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): dict(n=16578, p10=51.48200000000001, p25=14904.8565, median=167152.1195, share=0.104),
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): dict(n=15747, p10=18.4252, p25=56.2145, median=116.085, share=0.267),
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): dict(n=15647, p10=10.703, p25=40.826, median=98.688, share=0.339),
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): dict(n=15330, p10=62.131800000000005, p25=1614.967, median=14829.922999999999, share=0.099),
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): dict(n=12671, p10=0.125, p25=0.173, median=0.273, share=1.0),
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): dict(n=8973, p10=75.33920000000005, p25=1281.503, median=10850.015, share=0.093),
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): dict(n=8830, p10=10.189, p25=33.673, median=140.75650000000002, share=0.329),
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): dict(n=6995, p10=0.025, p25=0.028, median=0.035, share=1.0),
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): dict(n=6642, p10=0.046, p25=0.131, median=0.378, share=1.0),
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): dict(n=6533, p10=0.0822, p25=0.094, median=0.119, share=1.0),
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): dict(n=6298, p10=0.9147, p25=1.084, median=1.29, share=1.0),
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): dict(n=4829, p10=1.077, p25=1.179, median=1.387, share=1.0),
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): dict(n=4800, p10=12.5843, p25=19.86275, median=305.6565, share=0.425),
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): dict(n=4764, p10=123.8471, p25=245.35725, median=373.4235, share=0.049),
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): dict(n=4564, p10=32.8282, p25=35.614, median=38.9685, share=0.968),
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): dict(n=4318, p10=28.5972, p25=698.53475, median=6869.639, share=0.152),
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): dict(n=4289, p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): dict(n=4249, p10=63.803, p25=118.448, median=183.982, share=0.093),
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): dict(n=4175, p10=8.7122, p25=11.942, median=22.613, share=0.95),
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): dict(n=4038, p10=78.0573, p25=584.86575, median=2541.5485, share=0.089),
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): dict(n=3533, p10=31.7478, p25=34.197, median=37.565, share=0.951),
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): dict(n=3345, p10=0.141, p25=0.153, median=0.18, share=0.999),
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): dict(n=3171, p10=1.115, p25=1.229, median=1.45, share=1.0),
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): dict(n=3127, p10=6744.7898, p25=89237.016, median=189089.061, share=0.048),
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): dict(n=3123, p10=10.1908, p25=12.4835, median=21.627, share=0.818),
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): dict(n=2830, p10=24.4568, p25=242.11625, median=2012.101, share=0.143),
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): dict(n=2817, p10=10.74, p25=44.596, median=303.645, share=0.285),
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): dict(n=2361, p10=17.397, p25=844.995, median=5791.562, share=0.153),
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): dict(n=2328, p10=0.898, p25=0.987, median=1.148, share=1.0),
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): dict(n=2298, p10=35.6911, p25=65.77925, median=107.6855, share=0.213),
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): dict(n=2145, p10=26.0428, p25=332.025, median=2923.088, share=0.157),
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): dict(n=2128, p10=1.448, p25=1.61, median=1.8525, share=1.0),
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): dict(n=2007, p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): dict(n=1927, p10=249.9444, p25=623.466, median=1169.758, share=0.029),
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): dict(n=1923, p10=2.8494, p25=3.0905, median=3.415, share=0.998),
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): dict(n=1770, p10=32.098, p25=69.0895, median=112.2, share=0.204),
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): dict(n=1706, p10=697.8025, p25=2132.33425, median=5040.9805, share=0.03),
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): dict(n=1698, p10=1.4857, p25=1.65025, median=1.8965, share=0.999),
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): dict(n=1662, p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): dict(n=1638, p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): dict(n=1632, p10=1.2241, p25=1.32075, median=1.694, share=0.96),
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): dict(n=1612, p10=31.9407, p25=48.406, median=106.4295, share=0.318),
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): dict(n=1502, p10=24.8903, p25=36.49575, median=65.1735, share=0.469),
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): dict(n=1466, p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): dict(n=1285, p10=0.001, p25=0.001, median=0.001, share=1.0),
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): dict(n=1126, p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): dict(n=1054, p10=2.9613, p25=3.188, median=3.53, share=0.999),
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): dict(n=1031, p10=3.878, p25=16.2075, median=339.963, share=0.335),
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): dict(n=1027, p10=7.1722, p25=42.7105, median=141.496, share=0.299),
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): dict(n=1009, p10=6.4536, p25=7.027, median=7.763, share=0.999),
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): dict(n=1004, p10=176.1539, p25=313.63275, median=474.154, share=0.023),
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): dict(n=992, p10=1.837, p25=2.092, median=2.3965, share=0.998),
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): dict(n=911, p10=72.639, p25=111.0725, median=156.875, share=0.081),
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): dict(n=862, p10=6.3086, p25=6.952, median=7.6685, share=1.0),
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): dict(n=862, p10=141.3626, p25=317.4815, median=770.573, share=0.019),
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): dict(n=801, p10=0.0, p25=0.0, median=0.0, share=1.0),
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): dict(n=730, p10=1.3109, p25=1.466, median=1.668, share=0.999),
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): dict(n=713, p10=0.9154, p25=1.056, median=1.252, share=1.0),
    ("W_Valideren aanvraag-START", "A_APPROVED-COMPLETE"): dict(n=702, p10=142.1994, p25=329.30825, median=739.4095, share=0.02),
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): dict(n=697, p10=0.001, p25=0.001, median=0.001, share=1.0),
}

FREQ_VARIANTS = [
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"],
]

# -----------------------------
# Utilities
# -----------------------------
def detect_precision(ts_str: str) -> int:
    if "." in ts_str:
        frac = ts_str.split(".")[-1]
        return len(frac)
    return 0

def fmt_ts(ts: pd.Timestamp, precision: int) -> str:
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    s = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    return s[:-6 + precision]

def contiguous_subsequence_bonus(seq):
    bonus = 0.0
    L = len(seq)
    for var in FREQ_VARIANTS:
        for i in range(len(var) - L + 1):
            if var[i:i+L] == list(seq):
                bonus += 50.0
    return bonus

def multiset_contiguous_orders(activities):
    counts = Counter(activities)
    uniq = list(counts.keys())
    results = set()
    for var in FREQ_VARIANTS:
        n = len(activities)
        for i in range(len(var) - n + 1):
            sub = var[i:i+n]
            if Counter(sub) == counts:
                results.add(tuple(sub))
    return list(results)

def boundary_supported(a, b):
    return a is not None and b is not None and (a, b) in TRANSITION_PROB

def internal_supported(order):
    return all((order[i], order[i+1]) in TRANSITION_PROB for i in range(len(order)-1))

def score_order(order, left_act, right_act):
    # Reject if any internal edge unsupported
    if len(order) >= 2 and not internal_supported(order):
        return -1e18
    s = 0.0
    s += contiguous_subsequence_bonus(order)
    for i in range(len(order) - 1):
        p = TRANSITION_PROB.get((order[i], order[i+1]), 1e-12)
        s += 20.0 + math.log(max(p, 1e-12))
    if left_act is not None and (left_act, order[0]) in TRANSITION_PROB:
        s += 5.0 + math.log(max(TRANSITION_PROB[(left_act, order[0])], 1e-12))
    elif left_act is not None:
        s -= 2.0
    if right_act is not None and (order[-1], right_act) in TRANSITION_PROB:
        s += 5.0 + math.log(max(TRANSITION_PROB[(order[-1], right_act)], 1e-12))
    elif right_act is not None:
        s -= 2.0
    return s

def candidate_orders(activities, left_act, right_act):
    n = len(activities)
    cands = multiset_contiguous_orders(activities)
    if not cands:
        if n <= 7:
            perms = set(itertools.permutations(activities))
            cands = [p for p in perms if internal_supported(p)]
        else:
            # greedy fallback by best local support
            items = list(activities)
            best = None
            best_score = -1e18
            for start_idx, start in enumerate(items):
                remaining = items[:start_idx] + items[start_idx+1:]
                order = [start]
                while remaining:
                    prev = order[-1]
                    supported = [(TRANSITION_PROB.get((prev, x), -1), j, x) for j, x in enumerate(remaining)]
                    supported = [t for t in supported if t[0] >= 0]
                    if not supported:
                        break
                    supported.sort(reverse=True)
                    _, j, x = supported[0]
                    order.append(x)
                    remaining.pop(j)
                if len(order) == n and internal_supported(order):
                    sc = score_order(tuple(order), left_act, right_act)
                    if sc > best_score:
                        best_score = sc
                        best = tuple(order)
            cands = [best] if best is not None else []
    if not cands:
        # allow at most one unsupported boundary edge but no unsupported internal edge
        if n <= 8:
            perms = set(itertools.permutations(activities))
            cands = [p for p in perms if internal_supported(p)]
    if not cands:
        # final fallback: keep supported adjacency fragments via variant if any, else original multiset sorted by support
        cands = [tuple(activities)]
    scored = sorted([(score_order(c, left_act, right_act), c) for c in cands], key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]

def choose_gap(pair):
    prof = DUR.get(pair)
    if prof:
        if prof["share"] >= 0.95 or prof["p25"] <= 2.0:
            return max(prof["p10"], 0.001)
        return max(prof["p25"], 0.001)
    return None

def make_nonuniform_gaps(order):
    gaps = []
    borrowed = []
    for i in range(len(order)-1):
        g = choose_gap((order[i], order[i+1]))
        gaps.append(g)
    known = [g for g in gaps if g is not None]
    base_small = 1.137
    if known:
        small = min(max(min(known), 0.001), 5.0)
    else:
        small = base_small
    out = []
    unk_idx = 0
    for g in gaps:
        if g is None:
            # conservative, compact, non-uniform fallback
            val = small * (1.0 + 0.17 * ((unk_idx % 3) + 1))
            unk_idx += 1
            borrowed.append(True)
            out.append(val)
        else:
            borrowed.append(False)
            out.append(g)
    return out

def assign_times(start_ts, right_anchor_ts, case_end_ts, order, precision):
    n = len(order)
    if n == 1:
        return [start_ts], False
    gaps = make_nonuniform_gaps(order)
    min_gap_sec = 1.0
    desired = [max(g, min_gap_sec) for g in gaps]
    total_desired = sum(desired)

    upper_bound = right_anchor_ts if right_anchor_ts is not None else case_end_ts
    room = (upper_bound - start_ts).total_seconds()
    clamped = False

    min_total = (n - 1) * min_gap_sec
    if room < min_total:
        room = min_total
        clamped = True

    if total_desired > room:
        extra = total_desired - min_total
        target_extra = max(room - min_total, 0.0)
        scale = (target_extra / extra) if extra > 0 else 0.0
        scaled = [min_gap_sec + (g - min_gap_sec) * scale for g in desired]
        desired = scaled
        clamped = True

    # cumulative times
    times = [start_ts]
    cur = start_ts
    for g in desired:
        cur = cur + pd.to_timedelta(g, unit="s")
        times.append(cur)

    # format-round-reparse self-fix to ensure distinctness at precision and within bound
    times = enforce_formatted_distinct_increasing(times, precision, start_ts, upper_bound)
    if (times[-1] > upper_bound) or any(times[i] >= times[i+1] for i in range(len(times)-1)):
        clamped = True
        times = force_within_bounds(times, precision, start_ts, upper_bound)
    return times, clamped

def enforce_formatted_distinct_increasing(times, precision, start_ts, upper_bound):
    vals = [pd.to_datetime(fmt_ts(t, precision)) for t in times]
    changed = False
    for i in range(1, len(vals)):
        if vals[i] <= vals[i-1]:
            vals[i] = vals[i-1] + pd.to_timedelta(1, unit="s")
            changed = True
    if vals[0] != pd.to_datetime(fmt_ts(start_ts, precision)):
        vals[0] = pd.to_datetime(fmt_ts(start_ts, precision))
        changed = True
        for i in range(1, len(vals)):
            if vals[i] <= vals[i-1]:
                vals[i] = vals[i-1] + pd.to_timedelta(1, unit="s")
    if vals[-1] > upper_bound:
        changed = True
    return vals if changed else vals

def force_within_bounds(times, precision, start_ts, upper_bound):
    n = len(times)
    start_fmt = pd.to_datetime(fmt_ts(start_ts, precision))
    end_max = pd.to_datetime(fmt_ts(upper_bound, precision))
    vals = [start_fmt]
    for i in range(1, n):
        vals.append(vals[-1] + pd.to_timedelta(1, unit="s"))
    if vals[-1] > end_max:
        vals[-1] = end_max
        for i in range(n-2, -1, -1):
            latest = vals[i+1] - pd.to_timedelta(1, unit="s")
            if i == 0:
                vals[i] = start_fmt
            else:
                vals[i] = min(latest, vals[i])
        if vals[0] < start_fmt:
            vals[0] = start_fmt
            for i in range(1, n):
                if vals[i] <= vals[i-1]:
                    vals[i] = vals[i-1] + pd.to_timedelta(1, unit="s")
        if vals[-1] > end_max:
            # last-resort tight packing from end
            vals = [end_max - pd.to_timedelta(n-1-i, unit="s") for i in range(n)]
            if vals[0] < start_fmt:
                vals = [start_fmt + pd.to_timedelta(i, unit="s") for i in range(n)]
    return vals

def find_c1_blocks(df):
    blocks = []
    i = 0
    n = len(df)
    while i < n:
        if df.at[i, "error_category"] == "C1_ORDERING_AMBIGUITY":
            case = df.at[i, "Case"]
            ts = df.at[i, "Timestamp"]
            j = i + 1
            while j < n and df.at[j, "Case"] == case and df.at[j, "error_category"] == "C1_ORDERING_AMBIGUITY" and df.at[j, "Timestamp"] == ts:
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    original = df.copy(deep=True)
    out_ts = df["Timestamp"].tolist()

    parsed_cache = {}
    def parse_ts(s):
        if s not in parsed_cache:
            parsed_cache[s] = pd.to_datetime(s)
        return parsed_cache[s]

    precisions = [detect_precision(x) for x in df["Timestamp"] if isinstance(x, str)]
    global_precision = max(precisions) if precisions else 0

    blocks = find_c1_blocks(df)
    blocks_repaired = 0
    blocks_clamped = 0

    # case boundaries from existing rows
    case_indices = defaultdict(list)
    for idx, case in enumerate(df["Case"].tolist()):
        case_indices[case].append(idx)

    for start, end in blocks:
        try:
            case = df.at[start, "Case"]
            obs_str = df.at[start, "Timestamp"]
            obs_ts = parse_ts(obs_str)
            block_idxs = list(range(start, end))
            acts = [df.at[i, "Activity"] for i in block_idxs]

            # anchors
            left_idx = None
            i = start - 1
            while i >= 0 and df.at[i, "Case"] == case:
                if df.at[i, "error_category"] == "CLEAN":
                    left_idx = i
                    break
                i -= 1

            right_idx = None
            i = end
            while i < len(df) and df.at[i, "Case"] == case:
                if df.at[i, "error_category"] == "CLEAN":
                    right_idx = i
                    break
                i += 1

            left_act = df.at[left_idx, "Activity"] if left_idx is not None else None
            right_act = df.at[right_idx, "Activity"] if right_idx is not None else None
            left_ts = parse_ts(df.at[left_idx, "Timestamp"]) if left_idx is not None else None
            right_ts = parse_ts(df.at[right_idx, "Timestamp"]) if right_idx is not None else None

            case_rows = case_indices[case]
            case_start = min(parse_ts(df.at[i, "Timestamp"]) for i in case_rows)
            case_end = max(parse_ts(df.at[i, "Timestamp"]) for i in case_rows)

            orders = candidate_orders(acts, left_act, right_act)
            chosen = None
            chosen_map = None
            chosen_times = None
            chosen_clamped = False

            for order in orders:
                # feasible internal support already preferred; now assign
                times, clamped = assign_times(obs_ts, right_ts, case_end, order, global_precision)

                # map order positions to rows of matching activities, preserving row identity but not row-order semantics
                pos_by_act = defaultdict(list)
                for pos, act in enumerate(order):
                    pos_by_act[act].append(pos)

                row_assignment = {}
                for idx in block_idxs:
                    act = df.at[idx, "Activity"]
                    if pos_by_act[act]:
                        row_assignment[idx] = pos_by_act[act].pop(0)

                # Exactly one row keeps observed shared timestamp and it must be earliest event
                first_pos = 0
                keeper_idx = None
                for idx, pos in row_assignment.items():
                    if pos == first_pos:
                        keeper_idx = idx
                        break
                if keeper_idx is None:
                    continue

                tmp_out = {}
                for idx, pos in row_assignment.items():
                    t = times[pos]
                    s = fmt_ts(t, global_precision)
                    tmp_out[idx] = s

                # self-fix inside block
                tmp_vals = {idx: pd.to_datetime(tmp_out[idx]) for idx in tmp_out}
                ordered_pairs = sorted([(pos, idx) for idx, pos in row_assignment.items()])
                ordered_ts = [tmp_vals[idx] for _, idx in ordered_pairs]
                if any(ordered_ts[i] >= ordered_ts[i+1] for i in range(len(ordered_ts)-1)):
                    fixed = force_within_bounds(ordered_ts, global_precision, obs_ts, right_ts if right_ts is not None else case_end)
                    for (pos, idx), t in zip(ordered_pairs, fixed):
                        tmp_out[idx] = fmt_ts(t, global_precision)
                    clamped = True

                # ensure first event keeps observed exactly
                tmp_out[keeper_idx] = obs_str

                # ensure others differ from input
                for pos, idx in ordered_pairs[1:]:
                    if tmp_out[idx] == df.at[idx, "Timestamp"]:
                        t = pd.to_datetime(tmp_out[idx]) + pd.to_timedelta(1, unit="s")
                        upper = right_ts if right_ts is not None else case_end
                        if t > upper:
                            t = pd.to_datetime(tmp_out[idx]) - pd.to_timedelta(1, unit="s")
                        tmp_out[idx] = fmt_ts(t, global_precision)
                        clamped = True

                # final ordered check after string formatting
                ordered_ts2 = [pd.to_datetime(tmp_out[idx] if idx in tmp_out else df.at[idx, "Timestamp"]) for _, idx in ordered_pairs]
                ok = all(ordered_ts2[i] < ordered_ts2[i+1] for i in range(len(ordered_ts2)-1))
                if not ok:
                    fixed = [obs_ts + pd.to_timedelta(i, unit="s") for i in range(len(ordered_pairs))]
                    upper = right_ts if right_ts is not None else case_end
                    if fixed[-1] > upper:
                        fixed = force_within_bounds(fixed, global_precision, obs_ts, upper)
                    for (pos, idx), t in zip(ordered_pairs, fixed):
                        tmp_out[idx] = fmt_ts(t, global_precision)
                    tmp_out[keeper_idx] = obs_str
                    clamped = True

                # anchor bounds
                bounded_ok = True
                for idx in block_idxs:
                    t = pd.to_datetime(tmp_out[idx])
                    if t < case_start or t > case_end:
                        bounded_ok = False
                    if left_ts is not None and right_ts is not None:
                        if t < left_ts or t > right_ts:
                            bounded_ok = False
                if not bounded_ok:
                    ordered_ts = [pd.to_datetime(tmp_out[idx]) for _, idx in ordered_pairs]
                    upper = right_ts if right_ts is not None else case_end
                    lower = obs_ts
                    fixed = force_within_bounds(ordered_ts, global_precision, lower, upper)
                    for (pos, idx), t in zip(ordered_pairs, fixed):
                        tmp_out[idx] = fmt_ts(t, global_precision)
                    tmp_out[keeper_idx] = obs_str
                    clamped = True

                chosen = order
                chosen_map = row_assignment
                chosen_times = tmp_out
                chosen_clamped = clamped
                break

            if chosen_times is None:
                # deterministic last-resort compact repair
                ordered_pairs = list(enumerate(block_idxs))
                times = [obs_ts + pd.to_timedelta(i, unit="s") for i in range(len(block_idxs))]
                upper = right_ts if right_ts is not None else case_end
                if times[-1] > upper:
                    times = force_within_bounds(times, global_precision, obs_ts, upper)
                chosen_times = {}
                for pos, idx in ordered_pairs:
                    chosen_times[idx] = fmt_ts(times[pos], global_precision)
                chosen_times[block_idxs[0]] = obs_str
                chosen_clamped = True

            for idx in block_idxs:
                out_ts[idx] = chosen_times[idx]

            blocks_repaired += 1
            if chosen_clamped:
                blocks_clamped += 1
        except Exception:
            # Never abort. Compact forward repair.
            case = df.at[start, "Case"]
            obs_str = df.at[start, "Timestamp"]
            obs_ts = parse_ts(obs_str)
            block_idxs = list(range(start, end))
            case_rows = case_indices[case]
            case_end = max(parse_ts(df.at[i, "Timestamp"]) for i in case_rows)
            right_ts = None
            i = end
            while i < len(df) and df.at[i, "Case"] == case:
                if df.at[i, "error_category"] == "CLEAN":
                    right_ts = parse_ts(df.at[i, "Timestamp"])
                    break
                i += 1
            upper = right_ts if right_ts is not None else case_end
            times = [obs_ts + pd.to_timedelta(i, unit="s") for i in range(len(block_idxs))]
            if times[-1] > upper:
                times = force_within_bounds(times, global_precision, obs_ts, upper)
            for k, idx in enumerate(block_idxs):
                out_ts[idx] = fmt_ts(times[k], global_precision)
            out_ts[block_idxs[0]] = obs_str
            blocks_repaired += 1
            blocks_clamped += 1

    df["Timestamp"] = out_ts

    # -----------------------------
    # Global self-check and repair
    # -----------------------------
    # Preserve exact non-timestamp columns
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        df[col] = original[col]

    # CLEAN rows exact timestamp preservation
    clean_mask = original["error_category"] == "CLEAN"
    df.loc[clean_mask, "Timestamp"] = original.loc[clean_mask, "Timestamp"]

    # Re-check each C1 block
    rows_still_equal = 0
    for start, end in blocks:
        block_idxs = list(range(start, end))
        obs_str = original.at[start, "Timestamp"]
        obs_ts = pd.to_datetime(obs_str)
        case = original.at[start, "Case"]

        right_ts = None
        i = end
        while i < len(original) and original.at[i, "Case"] == case:
            if original.at[i, "error_category"] == "CLEAN":
                right_ts = pd.to_datetime(original.at[i, "Timestamp"])
                break
            i += 1

        case_rows = [j for j in range(len(original)) if original.at[j, "Case"] == case]
        case_end = max(pd.to_datetime(original.at[j, "Timestamp"]) for j in case_rows)
        upper = right_ts if right_ts is not None else case_end

        block_vals = [pd.to_datetime(df.at[idx, "Timestamp"]) for idx in block_idxs]
        eq_count = sum(df.at[idx, "Timestamp"] == original.at[idx, "Timestamp"] for idx in block_idxs)
        rows_still_equal += eq_count

        # exactly one observed timestamp
        same_obs = [idx for idx in block_idxs if df.at[idx, "Timestamp"] == obs_str]
        needs_fix = False
        if len(set(df.loc[block_idxs, "Timestamp"])) != len(block_idxs):
            needs_fix = True
        if len(same_obs) != 1:
            needs_fix = True
        if any(t < obs_ts or t > upper for t in block_vals):
            needs_fix = True
        span = (max(block_vals) - min(block_vals)).total_seconds()
        if span < (len(block_idxs) - 1):
            needs_fix = True
        if right_ts is not None and span > (right_ts - obs_ts).total_seconds():
            needs_fix = True

        if needs_fix:
            times = [obs_ts + pd.to_timedelta(i, unit="s") for i in range(len(block_idxs))]
            if times[-1] > upper:
                times = force_within_bounds(times, global_precision, obs_ts, upper)
            for k, idx in enumerate(block_idxs):
                df.at[idx, "Timestamp"] = fmt_ts(times[k], global_precision)
            # ensure exactly one keeper
            df.at[block_idxs[0], "Timestamp"] = obs_str
            blocks_clamped += 1

    rows_still_equal = sum(df.loc[original["error_category"] != "CLEAN", "Timestamp"].values ==
                           original.loc[original["error_category"] != "CLEAN", "Timestamp"].values)

    # Parseability pass
    repaired_mask = original["error_category"] != "CLEAN"
    for idx in df.index[repaired_mask]:
        try:
            pd.to_datetime(df.at[idx, "Timestamp"])
        except Exception:
            # restore to safe compact time in same block/case
            df.at[idx, "Timestamp"] = original.at[idx, "Timestamp"]

    # Write output
    df.to_csv(args.output, index=False)

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={blocks_clamped} rows_still_equal={rows_still_equal}")

if __name__ == "__main__":
    main()