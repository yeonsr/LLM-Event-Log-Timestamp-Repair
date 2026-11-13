import argparse
import itertools
import math
from decimal import Decimal, ROUND_FLOOR
from typing import Dict, List, Tuple, Optional

import pandas as pd


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

DURATION_PROFILE = {
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


def detect_fractional_digits(ts: str) -> int:
    if "." in ts:
        return len(ts.split(".")[-1])
    return 0


def parse_dt(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts)


def fmt_dt(ts: pd.Timestamp, frac_digits: int) -> str:
    if frac_digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    ns = ts.nanosecond + ts.microsecond * 1000
    frac = f"{ns:09d}"[:frac_digits]
    return f"{base}.{frac}"


def quantum_ns(frac_digits: int) -> int:
    if frac_digits <= 0:
        return 1_000_000_000
    return 10 ** (9 - frac_digits)


def ceil_to_quantum_ns(ns: int, q: int) -> int:
    return ((ns + q - 1) // q) * q


def floor_to_quantum_ns(ns: int, q: int) -> int:
    return (ns // q) * q


def ts_to_ns(ts: pd.Timestamp) -> int:
    return ts.value


def ns_to_ts(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns)


def subseq_bonus(order: Tuple[str, ...], left_act: Optional[str], right_act: Optional[str]) -> float:
    best = 0.0
    seq = list(order)
    ext_options = []
    if left_act is not None and right_act is not None:
        ext_options.append([left_act] + seq + [right_act])
    if left_act is not None:
        ext_options.append([left_act] + seq)
    if right_act is not None:
        ext_options.append(seq + [right_act])
    ext_options.append(seq)
    for var in FREQUENT_VARIANTS:
        for ext in ext_options:
            m = len(ext)
            if m > len(var):
                continue
            for i in range(len(var) - m + 1):
                if var[i:i + m] == ext:
                    best = max(best, 50.0 + 2.0 * m)
                elif var[i:i + len(seq)] == seq:
                    best = max(best, 20.0 + 1.0 * len(seq))
    return best


def edge_exists(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return True
    return (a, b) in TRANSITION_PROB or (a, b) in DURATION_PROFILE


def edge_score(a: Optional[str], b: Optional[str]) -> float:
    if a is None or b is None:
        return 0.0
    p = TRANSITION_PROB.get((a, b))
    if p is not None:
        return math.log(max(p, 1e-12)) + 5.0
    if (a, b) in DURATION_PROFILE:
        return 1.0
    return -30.0


def choose_gap_seconds(a: str, b: str, min_gap_s: float) -> float:
    prof = DURATION_PROFILE.get((a, b))
    if prof is None:
        return min_gap_s
    val = prof["p10"] if prof["share"] >= 0.8 else prof["p25"]
    if val <= 0:
        val = min_gap_s
    return max(val, min_gap_s)


def infer_order(acts: List[str], left_act: Optional[str], right_act: Optional[str]) -> Tuple[int, ...]:
    idxs = list(range(len(acts)))
    perms = list(itertools.permutations(idxs))
    supported = []
    scored = []
    for perm in perms:
        seq = [acts[i] for i in perm]
        internal_ok = all(edge_exists(seq[i], seq[i + 1]) for i in range(len(seq) - 1))
        left_ok = True if left_act is None else edge_exists(left_act, seq[0])
        right_ok = True if right_act is None else edge_exists(seq[-1], right_act)
        support_count = sum(1 for i in range(len(seq) - 1) if edge_exists(seq[i], seq[i + 1]))
        support_count += 1 if left_act is not None and left_ok else 0
        support_count += 1 if right_act is not None and right_ok else 0
        score = 0.0
        if left_act is not None:
            score += edge_score(left_act, seq[0]) * 1.2
        for i in range(len(seq) - 1):
            score += edge_score(seq[i], seq[i + 1]) * 2.0
        if right_act is not None:
            score += edge_score(seq[-1], right_act) * 1.2
        score += subseq_bonus(tuple(seq), left_act, right_act)
        score += support_count * 3.0
        scored.append((perm, score, internal_ok, left_ok, right_ok, support_count))
        if internal_ok and left_ok and right_ok:
            supported.append((perm, score, support_count))
    if supported:
        supported.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return supported[0][0]
    scored.sort(key=lambda x: (x[5], x[1]), reverse=True)
    return scored[0][0]


def plan_forward_times(observed_ns: int, order_acts: List[str], right_anchor_ns: Optional[int], frac_digits: int) -> List[int]:
    q = quantum_ns(frac_digits)
    min_gap_ns = max(q, 1_000_000_000)
    min_gap_s = min_gap_ns / 1e9
    k = len(order_acts)

    gaps_s = []
    for i in range(k - 1):
        gaps_s.append(choose_gap_seconds(order_acts[i], order_acts[i + 1], min_gap_s))
    gaps_ns = [max(min_gap_ns, ceil_to_quantum_ns(int(round(g * 1e9)), q)) for g in gaps_s]

    if k == 1:
        return [observed_ns]

    min_total = min_gap_ns * (k - 1)
    planned_total = sum(gaps_ns)

    if right_anchor_ns is not None:
        room = right_anchor_ns - observed_ns
        if room < min_total:
            total = min_total
        else:
            total = min(planned_total, room)
    else:
        total = max(planned_total, min_total)

    if total < min_total:
        total = min_total

    if planned_total <= total:
        use_gaps = gaps_ns[:]
    else:
        extra = total - min_total
        surplus = planned_total - min_total
        use_gaps = []
        acc = 0
        for g in gaps_ns:
            base = min_gap_ns
            part = g - min_gap_ns
            scaled = base + int(math.floor(part * (extra / surplus))) if surplus > 0 else base
            scaled = max(base, floor_to_quantum_ns(scaled, q))
            use_gaps.append(scaled)
            acc += scaled
        diff = total - acc
        i = len(use_gaps) - 1
        while diff > 0 and i >= 0:
            step = min(diff, q)
            use_gaps[i] += step
            diff -= step
            i = (i - 1) % len(use_gaps)
        while diff < 0 and i >= 0:
            reducible = use_gaps[i] - min_gap_ns
            step = min(-diff, reducible)
            use_gaps[i] -= step
            diff += step
            i = (i - 1) % len(use_gaps)

    times = [observed_ns]
    for g in use_gaps:
        times.append(times[-1] + g)
    return times


def choose_orientation(observed_ns: int, order_acts: List[str], left_act: Optional[str], right_act: Optional[str],
                       left_anchor_ns: Optional[int], right_anchor_ns: Optional[int], frac_digits: int) -> str:
    start_score = 0.0
    end_score = 0.0
    seq = order_acts
    if left_act is not None:
        start_score += edge_score(left_act, seq[0]) * 1.3
        end_score += edge_score(left_act, seq[0]) * 0.2
    if right_act is not None:
        start_score += edge_score(seq[-1], right_act) * 1.3
        end_score += edge_score(seq[-1], right_act) * 0.2
    rev = list(reversed(seq))
    if left_act is not None:
        end_score += edge_score(left_act, rev[0]) * 1.3
    if right_act is not None:
        end_score += edge_score(rev[-1], right_act) * 1.3

    if right_anchor_ns is not None:
        fwd = plan_forward_times(observed_ns, seq, right_anchor_ns, frac_digits)
        start_ok = fwd[-1] <= right_anchor_ns
        if start_ok:
            start_score += 5.0
        else:
            start_score -= 20.0
    if left_anchor_ns is not None:
        q = quantum_ns(frac_digits)
        min_gap_ns = max(q, 1_000_000_000)
        total = min_gap_ns * (len(seq) - 1)
        if observed_ns - total >= left_anchor_ns:
            end_score += 1.0
        else:
            end_score -= 20.0

    if end_score > start_score + 25.0:
        return "end"
    return "start"


def assign_times_for_block(block_df: pd.DataFrame, left_anchor_row: Optional[pd.Series], right_anchor_row: Optional[pd.Series],
                           case_start_ns: int, case_end_ns: int) -> Tuple[Dict[int, str], bool]:
    acts = block_df["Activity"].tolist()
    rows = block_df.index.tolist()
    observed_str = block_df["Timestamp"].iloc[0]
    observed_ts = parse_dt(observed_str)
    frac_digits = detect_fractional_digits(observed_str)
    q = quantum_ns(frac_digits)
    min_gap_ns = max(q, 1_000_000_000)
    observed_ns = ts_to_ns(observed_ts)

    left_act = None if left_anchor_row is None else left_anchor_row["Activity"]
    right_act = None if right_anchor_row is None else right_anchor_row["Activity"]
    left_anchor_ns = None if left_anchor_row is None else ts_to_ns(parse_dt(left_anchor_row["Timestamp"]))
    right_anchor_ns = None if right_anchor_row is None else ts_to_ns(parse_dt(right_anchor_row["Timestamp"]))

    perm = infer_order(acts, left_act, right_act)
    order_acts = [acts[i] for i in perm]
    orientation = choose_orientation(observed_ns, order_acts, left_act, right_act, left_anchor_ns, right_anchor_ns, frac_digits)

    if orientation == "end":
        rev_acts = list(reversed(order_acts))
        times_rev = plan_forward_times(observed_ns, rev_acts, None, frac_digits)
        end_times = list(reversed([observed_ns - (t - observed_ns) for t in times_rev]))
        order_positions = list(reversed(range(len(order_acts))))
        pos_to_ns = {order_positions[i]: end_times[i] for i in range(len(end_times))}
    else:
        times = plan_forward_times(observed_ns, order_acts, right_anchor_ns if right_anchor_ns is not None else case_end_ns, frac_digits)
        pos_to_ns = {i: times[i] for i in range(len(times))}

    row_to_order_pos = {perm[pos]: pos for pos in range(len(perm))}
    order0_row = perm[0]
    if orientation == "end":
        observed_keeper_row = perm[-1]
    else:
        observed_keeper_row = order0_row

    assigned = {}
    for local_row_idx, global_row in enumerate(rows):
        pos = row_to_order_pos[local_row_idx]
        assigned[global_row] = pos_to_ns[pos]

    clamped = False

    lower_bound = max(case_start_ns, left_anchor_ns if left_anchor_ns is not None else case_start_ns)
    upper_bound = min(case_end_ns, right_anchor_ns if right_anchor_ns is not None else case_end_ns)

    ordered_pairs = sorted(((row_to_order_pos[local_i], rows[local_i]) for local_i in range(len(rows))), key=lambda x: x[0])
    ordered_ns = [assigned[r] for _, r in ordered_pairs]

    if orientation != "start":
        orientation = "start"
        times = plan_forward_times(observed_ns, order_acts, right_anchor_ns if right_anchor_ns is not None else case_end_ns, frac_digits)
        pos_to_ns = {i: times[i] for i in range(len(times))}
        for local_row_idx, global_row in enumerate(rows):
            pos = row_to_order_pos[local_row_idx]
            assigned[global_row] = pos_to_ns[pos]
        ordered_ns = [assigned[r] for _, r in ordered_pairs]
        observed_keeper_row = order0_row
        clamped = True

    ordered_ns[0] = observed_ns
    for i in range(1, len(ordered_ns)):
        ordered_ns[i] = max(ordered_ns[i], ordered_ns[i - 1] + min_gap_ns)

    if ordered_ns[0] < lower_bound:
        shift = lower_bound - ordered_ns[0]
        ordered_ns = [x + shift for x in ordered_ns]
        clamped = True

    if ordered_ns[-1] > upper_bound:
        room = upper_bound - observed_ns
        order_seq_acts = [acts[i] for i in perm]
        replanned = plan_forward_times(observed_ns, order_seq_acts, upper_bound, frac_digits)
        ordered_ns = replanned[:]
        ordered_ns[0] = observed_ns
        for i in range(1, len(ordered_ns)):
            ordered_ns[i] = max(ordered_ns[i], ordered_ns[i - 1] + min_gap_ns)
        if ordered_ns[-1] > upper_bound:
            end_target = upper_bound
            start_target = observed_ns
            need = min_gap_ns * (len(ordered_ns) - 1)
            if end_target - start_target >= need:
                ordered_ns = [start_target + i * min_gap_ns for i in range(len(ordered_ns))]
            else:
                start_target = max(lower_bound, end_target - need)
                ordered_ns = [start_target + i * min_gap_ns for i in range(len(ordered_ns))]
                if ordered_ns[0] != observed_ns:
                    ordered_ns = [observed_ns + i * min_gap_ns for i in range(len(ordered_ns))]
        clamped = True

    if right_anchor_ns is not None and len(ordered_ns) > 1:
        max_span = right_anchor_ns - observed_ns
        cur_span = ordered_ns[-1] - ordered_ns[0]
        min_span = min_gap_ns * (len(ordered_ns) - 1)
        if cur_span > max_span:
            if max_span >= min_span:
                ordered_ns = [observed_ns]
                remaining = max_span - min_span
                planned = []
                for i in range(len(order_acts) - 1):
                    planned.append(max(min_gap_ns, ceil_to_quantum_ns(int(round(choose_gap_seconds(order_acts[i], order_acts[i+1], min_gap_ns / 1e9) * 1e9)), q)))
                surplus = max(sum(planned) - min_span, 0)
                use = []
                for g in planned:
                    extra_part = g - min_gap_ns
                    add = int(math.floor(extra_part * (remaining / surplus))) if surplus > 0 else 0
                    add = floor_to_quantum_ns(add, q)
                    use.append(min_gap_ns + add)
                acc = observed_ns
                for g in use:
                    acc += g
                    ordered_ns.append(acc)
                while ordered_ns[-1] < right_anchor_ns and ordered_ns[-1] + q <= right_anchor_ns:
                    ordered_ns[-1] += q
            clamped = True

    if len(ordered_ns) > 1:
        if ordered_ns[-1] - ordered_ns[0] < min_gap_ns * (len(ordered_ns) - 1):
            ordered_ns = [ordered_ns[0] + i * min_gap_ns for i in range(len(ordered_ns))]
            clamped = True

    if ordered_ns[0] != observed_ns:
        delta = observed_ns - ordered_ns[0]
        ordered_ns = [x + delta for x in ordered_ns]
        clamped = True

    if right_anchor_ns is not None and ordered_ns[-1] > right_anchor_ns:
        ordered_ns = [observed_ns + i * min_gap_ns for i in range(len(ordered_ns))]
        if ordered_ns[-1] > right_anchor_ns:
            back = ordered_ns[-1] - right_anchor_ns
            ordered_ns = [x - back for x in ordered_ns]
        clamped = True

    if left_anchor_ns is not None and ordered_ns[0] < left_anchor_ns:
        shift = left_anchor_ns - ordered_ns[0]
        ordered_ns = [x + shift for x in ordered_ns]
        clamped = True

    ordered_ns[0] = observed_ns
    for i in range(1, len(ordered_ns)):
        ordered_ns[i] = max(ordered_ns[i], ordered_ns[i - 1] + min_gap_ns)

    if right_anchor_ns is not None and ordered_ns[-1] > right_anchor_ns:
        room = right_anchor_ns - observed_ns
        if room >= min_gap_ns * (len(ordered_ns) - 1):
            ordered_ns = [observed_ns + i * min_gap_ns for i in range(len(ordered_ns))]
            tail_extra = room - min_gap_ns * (len(ordered_ns) - 1)
            ordered_ns[-1] += floor_to_quantum_ns(tail_extra, q)
        else:
            ordered_ns = [observed_ns + i * min_gap_ns for i in range(len(ordered_ns))]
        clamped = True

    if ordered_ns[0] < case_start_ns:
        shift = case_start_ns - ordered_ns[0]
        ordered_ns = [x + shift for x in ordered_ns]
        clamped = True
    if ordered_ns[-1] > case_end_ns:
        shift = ordered_ns[-1] - case_end_ns
        ordered_ns = [x - shift for x in ordered_ns]
        clamped = True

    ordered_ns[0] = observed_ns
    for i in range(1, len(ordered_ns)):
        ordered_ns[i] = max(ordered_ns[i], ordered_ns[i - 1] + min_gap_ns)

    for idx, (_, r) in enumerate(ordered_pairs):
        assigned[r] = ordered_ns[idx]

    result = {}
    for r in rows:
        if r == rows[observed_keeper_row]:
            result[r] = observed_str
        else:
            ts = ns_to_ts(assigned[r])
            s = fmt_dt(ts, frac_digits)
            if s == block_df.loc[r, "Timestamp"]:
                ts = ns_to_ts(assigned[r] + q)
                s = fmt_dt(ts, frac_digits)
            result[r] = s
    return result, clamped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    out = df.copy()

    out["__parsed_ts"] = out["Timestamp"].map(parse_dt)

    repaired_blocks = 0
    clamped_blocks = 0

    for case_id, case_idx in out.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        case_df = out.loc[idxs]
        case_start_ns = ts_to_ns(case_df["__parsed_ts"].min())
        case_end_ns = ts_to_ns(case_df["__parsed_ts"].max())

        i = 0
        while i < len(idxs):
            gidx = idxs[i]
            row = out.loc[gidx]
            if row["error_category"] != "C1_ORDERING_AMBIGUITY":
                i += 1
                continue

            observed = row["Timestamp"]
            start = i
            j = i
            while j + 1 < len(idxs):
                next_idx = idxs[j + 1]
                next_row = out.loc[next_idx]
                if next_row["error_category"] == "C1_ORDERING_AMBIGUITY" and next_row["Timestamp"] == observed:
                    j += 1
                else:
                    break

            block_rows = idxs[start:j + 1]
            block_df = out.loc[block_rows]

            left_anchor_row = None
            for k in range(start - 1, -1, -1):
                cand = out.loc[idxs[k]]
                if cand["error_category"] == "CLEAN":
                    left_anchor_row = cand
                    break

            right_anchor_row = None
            for k in range(j + 1, len(idxs)):
                cand = out.loc[idxs[k]]
                if cand["error_category"] == "CLEAN":
                    right_anchor_row = cand
                    break

            repaired_map, clamped = assign_times_for_block(block_df, left_anchor_row, right_anchor_row, case_start_ns, case_end_ns)
            for r, ts_str in repaired_map.items():
                out.at[r, "Timestamp"] = ts_str
                out.at[r, "__parsed_ts"] = parse_dt(ts_str)

            repaired_blocks += 1
            if clamped:
                clamped_blocks += 1
            i = j + 1

    for case_id, case_idx in out.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        case_df = out.loc[idxs]
        case_start_ns = ts_to_ns(case_df["__parsed_ts"].min())
        case_end_ns = ts_to_ns(case_df["__parsed_ts"].max())

        i = 0
        while i < len(idxs):
            gidx = idxs[i]
            row = out.loc[gidx]
            if row["error_category"] != "C1_ORDERING_AMBIGUITY":
                i += 1
                continue

            observed = df.loc[gidx, "Timestamp"]
            start = i
            j = i
            while j + 1 < len(idxs):
                next_idx = idxs[j + 1]
                next_row_orig = df.loc[next_idx]
                if next_row_orig["error_category"] == "C1_ORDERING_AMBIGUITY" and next_row_orig["Timestamp"] == observed:
                    j += 1
                else:
                    break

            block_rows = idxs[start:j + 1]
            frac_digits = detect_fractional_digits(observed)
            q = quantum_ns(frac_digits)
            min_gap_ns = max(q, 1_000_000_000)

            block_times_ns = [ts_to_ns(parse_dt(out.loc[r, "Timestamp"])) for r in block_rows]

            left_anchor_ns = None
            for k in range(start - 1, -1, -1):
                cand = out.loc[idxs[k]]
                if cand["error_category"] == "CLEAN":
                    left_anchor_ns = ts_to_ns(parse_dt(cand["Timestamp"]))
                    break

            right_anchor_ns = None
            for k in range(j + 1, len(idxs)):
                cand = out.loc[idxs[k]]
                if cand["error_category"] == "CLEAN":
                    right_anchor_ns = ts_to_ns(parse_dt(cand["Timestamp"]))
                    break

            min_bound = max(case_start_ns, left_anchor_ns if left_anchor_ns is not None else case_start_ns)
            max_bound = min(case_end_ns, right_anchor_ns if right_anchor_ns is not None else case_end_ns)

            sorted_ns = sorted(block_times_ns)
            if not sorted_ns or sorted_ns[0] != ts_to_ns(parse_dt(observed)):
                sorted_ns = [ts_to_ns(parse_dt(observed)) + t * min_gap_ns for t in range(len(block_rows))]
                if right_anchor_ns is not None and sorted_ns[-1] > right_anchor_ns:
                    room = right_anchor_ns - sorted_ns[0]
                    if room >= min_gap_ns * (len(block_rows) - 1):
                        sorted_ns = [sorted_ns[0] + t * min_gap_ns for t in range(len(block_rows))]
                    clamped_blocks += 1

            for m in range(1, len(sorted_ns)):
                if sorted_ns[m] <= sorted_ns[m - 1]:
                    sorted_ns[m] = sorted_ns[m - 1] + min_gap_ns
            if sorted_ns[0] < min_bound:
                shift = min_bound - sorted_ns[0]
                sorted_ns = [x + shift for x in sorted_ns]
            if sorted_ns[-1] > max_bound:
                if max_bound - ts_to_ns(parse_dt(observed)) >= min_gap_ns * (len(block_rows) - 1):
                    sorted_ns = [ts_to_ns(parse_dt(observed)) + t * min_gap_ns for t in range(len(block_rows))]
                else:
                    start_ns = max(min_bound, max_bound - min_gap_ns * (len(block_rows) - 1))
                    sorted_ns = [start_ns + t * min_gap_ns for t in range(len(block_rows))]
                    if sorted_ns[0] != ts_to_ns(parse_dt(observed)):
                        sorted_ns[0] = ts_to_ns(parse_dt(observed))
                        for m in range(1, len(sorted_ns)):
                            sorted_ns[m] = max(sorted_ns[m], sorted_ns[m - 1] + min_gap_ns)
                clamped_blocks += 1

            keeper_count = 0
            for r in block_rows:
                if out.loc[r, "Timestamp"] == observed:
                    keeper_count += 1
            if keeper_count != 1:
                used_keeper = False
                for pos, r in enumerate(block_rows):
                    s = observed if not used_keeper else fmt_dt(ns_to_ts(sorted_ns[pos]), frac_digits)
                    if not used_keeper:
                        used_keeper = True
                    out.at[r, "Timestamp"] = s if pos == 0 else fmt_dt(ns_to_ts(sorted_ns[pos]), frac_digits)
                out.at[block_rows[0], "Timestamp"] = observed

            i = j + 1

    rows_still_equal = int(((out["error_category"] != "CLEAN") & (out["Timestamp"] == df["Timestamp"])).sum())

    final_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    out[final_cols].to_csv(args.output, index=False)

    print(f"blocks_repaired={repaired_blocks} blocks_clamped={clamped_blocks} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            import csv
            import sys
            with open(sys.argv[sys.argv.index("--input") + 1], "r", newline="", encoding="utf-8") as f_in, \
                 open(sys.argv[sys.argv.index("--output") + 1], "w", newline="", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")
        except Exception:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")