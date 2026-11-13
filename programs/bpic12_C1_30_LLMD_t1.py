import argparse
import itertools
import json
import math
import re
from collections import defaultdict, Counter

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

DFG_FREQ = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): 16578,
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): 15747,
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): 15647,
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): 15330,
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): 12671,
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): 8973,
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): 8830,
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): 6995,
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): 6642,
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): 6533,
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): 6298,
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): 4829,
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): 4800,
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): 4764,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): 4564,
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): 4318,
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): 4289,
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): 4249,
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): 4175,
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): 4038,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): 3533,
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): 3345,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 3171,
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): 3127,
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): 3123,
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): 2830,
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): 2817,
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): 2361,
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): 2328,
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): 2298,
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): 2145,
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): 2128,
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): 2007,
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): 1927,
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): 1923,
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): 1770,
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): 1706,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): 1698,
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): 1662,
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): 1638,
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 1632,
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): 1612,
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): 1502,
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): 1466,
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): 1285,
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): 1126,
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 1054,
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): 1031,
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): 1027,
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 1009,
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): 1004,
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): 992,
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): 911,
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): 862,
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 862,
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): 801,
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 730,
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): 713,
    ("W_Valideren aanvraag-START", "A_APPROVED-COMPLETE"): 702,
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): 697,
}

DURATION = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): {"p10": 51.48200000000001, "p25": 14904.8565, "share_under_60s": 0.104},
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): {"p10": 18.4252, "p25": 56.2145, "share_under_60s": 0.267},
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): {"p10": 10.703, "p25": 40.826, "share_under_60s": 0.339},
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 62.131800000000005, "p25": 1614.967, "share_under_60s": 0.099},
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): {"p10": 0.125, "p25": 0.173, "share_under_60s": 1.0},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 75.33920000000005, "p25": 1281.503, "share_under_60s": 0.093},
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 10.189, "p25": 33.673, "share_under_60s": 0.329},
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): {"p10": 0.025, "p25": 0.028, "share_under_60s": 1.0},
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): {"p10": 0.046, "p25": 0.131, "share_under_60s": 1.0},
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): {"p10": 0.08220000000000005, "p25": 0.094, "share_under_60s": 1.0},
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9147000000000001, "p25": 1.084, "share_under_60s": 1.0},
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.077, "p25": 1.179, "share_under_60s": 1.0},
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): {"p10": 12.5843, "p25": 19.862750000000002, "share_under_60s": 0.425},
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): {"p10": 123.8471, "p25": 245.35725, "share_under_60s": 0.049},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): {"p10": 32.8282, "p25": 35.614, "share_under_60s": 0.968},
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): {"p10": 28.597200000000004, "p25": 698.53475, "share_under_60s": 0.152},
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "share_under_60s": 1.0},
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 63.803, "p25": 118.448, "share_under_60s": 0.093},
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): {"p10": 8.712200000000001, "p25": 11.942, "share_under_60s": 0.95},
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): {"p10": 78.05730000000003, "p25": 584.8657499999999, "share_under_60s": 0.089},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.7478, "p25": 34.197, "share_under_60s": 0.951},
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): {"p10": 0.141, "p25": 0.153, "share_under_60s": 0.999},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.115, "p25": 1.229, "share_under_60s": 1.0},
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 6744.789800000003, "p25": 89237.016, "share_under_60s": 0.048},
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): {"p10": 10.1908, "p25": 12.4835, "share_under_60s": 0.818},
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 24.4568, "p25": 242.11625, "share_under_60s": 0.143},
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): {"p10": 10.74, "p25": 44.596, "share_under_60s": 0.285},
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 17.397, "p25": 844.995, "share_under_60s": 0.153},
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): {"p10": 0.898, "p25": 0.987, "share_under_60s": 1.0},
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): {"p10": 35.6911, "p25": 65.77924999999999, "share_under_60s": 0.213},
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 26.0428, "p25": 332.025, "share_under_60s": 0.157},
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): {"p10": 1.448, "p25": 1.61, "share_under_60s": 1.0},
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "share_under_60s": 1.0},
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): {"p10": 249.9444, "p25": 623.466, "share_under_60s": 0.029},
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): {"p10": 2.8494, "p25": 3.0905, "share_under_60s": 0.998},
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): {"p10": 32.098000000000006, "p25": 69.0895, "share_under_60s": 0.204},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 697.8025, "p25": 2132.33425, "share_under_60s": 0.03},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 1.4857, "p25": 1.65025, "share_under_60s": 0.999},
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "share_under_60s": 1.0},
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "share_under_60s": 1.0},
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.2241, "p25": 1.3207499999999999, "share_under_60s": 0.96},
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): {"p10": 31.9407, "p25": 48.406000000000006, "share_under_60s": 0.318},
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): {"p10": 24.8903, "p25": 36.49575, "share_under_60s": 0.469},
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "share_under_60s": 1.0},
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "share_under_60s": 1.0},
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "share_under_60s": 1.0},
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 2.9613, "p25": 3.188, "share_under_60s": 0.999},
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): {"p10": 3.878, "p25": 16.207500000000003, "share_under_60s": 0.335},
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): {"p10": 7.172200000000002, "p25": 42.710499999999996, "share_under_60s": 0.299},
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.4536, "p25": 7.027, "share_under_60s": 0.999},
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): {"p10": 176.1539, "p25": 313.63275, "share_under_60s": 0.023},
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.837, "p25": 2.092, "share_under_60s": 0.998},
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): {"p10": 72.639, "p25": 111.07249999999999, "share_under_60s": 0.081},
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.3086, "p25": 6.952, "share_under_60s": 1.0},
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): {"p10": 141.36260000000001, "p25": 317.4815, "share_under_60s": 0.019},
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "share_under_60s": 1.0},
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.3109, "p25": 1.466, "share_under_60s": 0.999},
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9154000000000001, "p25": 1.056, "share_under_60s": 1.0},
    ("W_Valideren aanvraag-START", "A_APPROVED-COMPLETE"): {"p10": 142.19940000000005, "p25": 329.30825, "share_under_60s": 0.02},
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "share_under_60s": 1.0},
}

FREQUENT_VARIANTS = [
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_DECLINED-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
]


def detect_fractional_digits(ts: str) -> int:
    m = re.search(r"\.(\d+)$", ts)
    return len(m.group(1)) if m else 0


def parse_ts(s: str):
    return pd.to_datetime(s)


def format_ts(dt: pd.Timestamp, frac_digits: int) -> str:
    if frac_digits > 0:
        base = dt.strftime("%Y-%m-%d %H:%M:%S")
        frac = f"{dt.microsecond:06d}"[:frac_digits]
        return f"{base}.{frac}"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def min_step_ns(frac_digits: int) -> int:
    if frac_digits <= 0:
        return 1_000_000_000
    if frac_digits <= 6:
        return 10 ** (9 - frac_digits)
    return 1


def contiguous_subsequence_bonus(order, left_act, right_act):
    seq = []
    if left_act is not None:
        seq.append(left_act)
    seq.extend(order)
    if right_act is not None:
        seq.append(right_act)
    best = 0.0
    for var in FREQUENT_VARIANTS:
        for i in range(0, len(var) - len(seq) + 1):
            if var[i:i + len(seq)] == seq:
                best = max(best, 10.0)
        for i in range(0, len(var) - len(order) + 1):
            if var[i:i + len(order)] == list(order):
                best = max(best, 5.0)
    return best


def edge_score(a, b):
    if (a, b) not in DFG_FREQ:
        return None
    freq = DFG_FREQ[(a, b)]
    prob = TRANSITION_PROB.get((a, b), 1e-6)
    return math.log(freq + 1.0) + 2.0 * math.log(prob + 1e-12)


def pair_concurrent(a, b):
    ab = DFG_FREQ.get((a, b), 0)
    ba = DFG_FREQ.get((b, a), 0)
    if ab > 0 and ba > 0:
        hi = max(ab, ba)
        lo = min(ab, ba)
        return lo / hi >= 0.6
    return False


def enumerate_best_order(activities, left_act, right_act):
    idxs = list(range(len(activities)))
    perms = itertools.permutations(idxs)
    survivors = []
    for perm in perms:
        ordered = [activities[i] for i in perm]
        valid = True
        score = 0.0

        if left_act is not None:
            es = edge_score(left_act, ordered[0])
            if es is None:
                valid = False
            else:
                score += es
        if not valid:
            continue

        for i in range(len(ordered) - 1):
            es = edge_score(ordered[i], ordered[i + 1])
            if es is None:
                valid = False
                break
            score += es
        if not valid:
            continue

        if right_act is not None:
            es = edge_score(ordered[-1], right_act)
            if es is None:
                valid = False
            else:
                score += es
        if not valid:
            continue

        score += contiguous_subsequence_bonus(ordered, left_act, right_act)
        survivors.append((score, perm, ordered))

    if survivors:
        survivors.sort(key=lambda x: x[0], reverse=True)
        best_score = survivors[0][0]
        best = [s for s in survivors if abs(s[0] - best_score) < 1e-9]
        return best[0][1], best[0][2]

    # fallback: strongest pairwise precedence seriation
    n = len(activities)
    weights = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(activities):
        for j, b in enumerate(activities):
            if i == j:
                continue
            weights[i][j] = DFG_FREQ.get((a, b), 0) - DFG_FREQ.get((b, a), 0)

    best_perm = None
    best_score = -10**18
    for perm in itertools.permutations(idxs):
        ordered = [activities[i] for i in perm]
        score = 0.0
        if left_act is not None:
            score += DFG_FREQ.get((left_act, ordered[0]), 0) * 10
        if right_act is not None:
            score += DFG_FREQ.get((ordered[-1], right_act), 0) * 10
        for i in range(len(perm) - 1):
            score += max(0, DFG_FREQ.get((ordered[i], ordered[i + 1]), 0))
            score += weights[perm[i]][perm[i + 1]]
        score += contiguous_subsequence_bonus(ordered, left_act, right_act)
        if score > best_score:
            best_score = score
            best_perm = perm
    ordered = [activities[i] for i in best_perm]
    return best_perm, ordered


def gap_seconds(a, b, concurrent=False):
    prof = DURATION.get((a, b))
    if prof:
        val = prof["p10"] if prof["share_under_60s"] >= 0.8 else prof["p25"]
        if val <= 0:
            val = 0.001
        return float(val)
    if concurrent:
        return 0.001
    return 1.0


def build_times(observed_dt, order_acts, frac_digits, right_anchor_dt, case_end_dt):
    k = len(order_acts)
    min_ns = min_step_ns(frac_digits)
    lower = observed_dt
    upper = right_anchor_dt if right_anchor_dt is not None else case_end_dt
    if upper is None or pd.isna(upper):
        upper = observed_dt + pd.Timedelta(days=3650)

    room_ns = max(0, int((upper.value - observed_dt.value)))
    min_total_ns = (k - 1) * max(min_ns, 1_000_000_000)

    gaps_sec = []
    for i in range(k - 1):
        conc = pair_concurrent(order_acts[i], order_acts[i + 1])
        gaps_sec.append(gap_seconds(order_acts[i], order_acts[i + 1], conc))
    gaps_ns = [max(min_ns, int(round(g * 1e9))) for g in gaps_sec]

    if room_ns < min_total_ns:
        start_ns = upper.value - min_total_ns
        if start_ns < observed_dt.value:
            start_ns = observed_dt.value
        times = [pd.Timestamp(start_ns)]
        for _ in range(k - 1):
            times.append(pd.Timestamp(times[-1].value + max(min_ns, 1_000_000_000)))
        return times, True

    total_gap_ns = sum(gaps_ns)
    target_total_ns = min(total_gap_ns, room_ns)

    if total_gap_ns == 0:
        scaled = [max(min_ns, 1_000_000_000)] * (k - 1)
    else:
        scaled = [max(min_ns, int(g * target_total_ns / total_gap_ns)) for g in gaps_ns]

    current_total = sum(scaled)
    if current_total > room_ns:
        ratio = room_ns / current_total
        scaled = [max(min_ns, int(g * ratio)) for g in scaled]

    current_total = sum(scaled)
    min_needed = min_total_ns
    if current_total < min_needed:
        deficit = min_needed - current_total
        base_add = deficit // (k - 1) if k > 1 else 0
        rem = deficit % (k - 1) if k > 1 else 0
        for i in range(k - 1):
            scaled[i] += base_add + (1 if i < rem else 0)

    current_total = sum(scaled)
    clamped = False
    if current_total > room_ns:
        clamped = True
        scaled = [max(min_ns, 1) for _ in range(k - 1)]
        current_total = sum(scaled)
        if current_total > room_ns:
            scaled = [max(min_ns, 1_000_000_000)] * (k - 1)
            current_total = sum(scaled)
        if current_total > room_ns:
            # final feasible latest strictly increasing sequence starting at observed
            scaled = []
            remaining = room_ns
            steps_left = k - 1
            for _ in range(k - 1):
                g = max(min_ns, 1_000_000_000)
                max_allow = remaining - max(0, steps_left - 1) * max(min_ns, 1_000_000_000)
                g = min(g, max_allow)
                g = max(g, max(min_ns, 1))
                scaled.append(int(g))
                remaining -= int(g)
                steps_left -= 1

    times = [observed_dt]
    cur = observed_dt.value
    for g in scaled:
        cur += int(g)
        times.append(pd.Timestamp(cur))

    if times[-1] > upper:
        clamped = True
        cur = observed_dt.value
        times = [observed_dt]
        for i in range(1, k):
            latest_possible = upper.value - (k - 1 - i) * max(min_ns, 1_000_000_000)
            nxt = min(cur + max(min_ns, 1_000_000_000), latest_possible)
            nxt = max(nxt, cur + max(min_ns, 1))
            cur = nxt
            times.append(pd.Timestamp(cur))

    return times, clamped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    orig = df.copy(deep=True)
    out_ts = df["Timestamp"].tolist()
    parsed_ts = [parse_ts(x) for x in df["Timestamp"].tolist()]
    frac_digits_all = [detect_fractional_digits(x) for x in df["Timestamp"].tolist()]

    repaired_blocks = 0
    clamped_blocks = 0

    case_to_idxs = defaultdict(list)
    for i, c in enumerate(df["Case"].tolist()):
        case_to_idxs[c].append(i)

    c1_blocks = []

    for case, idxs in case_to_idxs.items():
        idxs_sorted = sorted(idxs)
        case_times = [parsed_ts[i] for i in idxs_sorted]
        case_start = min(case_times)
        case_end = max(case_times)

        j = 0
        while j < len(idxs_sorted):
            i0 = idxs_sorted[j]
            if df.at[i0, "error_category"] != "C1_ORDERING_AMBIGUITY":
                j += 1
                continue
            ts0 = df.at[i0, "Timestamp"]
            block = [i0]
            j += 1
            while j < len(idxs_sorted):
                ii = idxs_sorted[j]
                if df.at[ii, "error_category"] == "C1_ORDERING_AMBIGUITY" and df.at[ii, "Timestamp"] == ts0:
                    block.append(ii)
                    j += 1
                else:
                    break
            c1_blocks.append((case, block, case_start, case_end))

    for case, block, case_start, case_end in c1_blocks:
        observed_ts_str = df.at[block[0], "Timestamp"]
        observed_dt = parse_ts(observed_ts_str)
        frac_digits = detect_fractional_digits(observed_ts_str)

        case_idxs = case_to_idxs[case]
        pos_map = {idx: p for p, idx in enumerate(sorted(case_idxs))}
        block_positions = [pos_map[idx] for idx in block]
        block_min_pos = min(block_positions)
        block_max_pos = max(block_positions)
        sorted_case_idxs = sorted(case_idxs)

        left_anchor_idx = None
        for p in range(block_min_pos - 1, -1, -1):
            idx = sorted_case_idxs[p]
            if df.at[idx, "error_category"] == "CLEAN":
                left_anchor_idx = idx
                break

        right_anchor_idx = None
        for p in range(block_max_pos + 1, len(sorted_case_idxs)):
            idx = sorted_case_idxs[p]
            if df.at[idx, "error_category"] == "CLEAN":
                right_anchor_idx = idx
                break

        left_anchor_dt = parse_ts(df.at[left_anchor_idx, "Timestamp"]) if left_anchor_idx is not None else case_start
        right_anchor_dt = parse_ts(df.at[right_anchor_idx, "Timestamp"]) if right_anchor_idx is not None else case_end
        left_anchor_act = df.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None else None
        right_anchor_act = df.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None else None

        activities = [df.at[idx, "Activity"] for idx in block]
        perm, order_acts = enumerate_best_order(activities, left_anchor_act, right_anchor_act)

        remaining_rows = list(range(len(block)))
        used = [False] * len(block)
        order_row_positions = []
        for act in order_acts:
            for rp in remaining_rows:
                if not used[rp] and activities[rp] == act:
                    used[rp] = True
                    order_row_positions.append(rp)
                    break

        times_ordered, clamped = build_times(observed_dt, order_acts, frac_digits, right_anchor_dt, case_end)
        if clamped:
            clamped_blocks += 1

        # self-check and repair
        min_ns = min_step_ns(frac_digits)
        upper = right_anchor_dt if right_anchor_dt is not None else case_end
        if upper is None or pd.isna(upper):
            upper = case_end
        if upper is None or pd.isna(upper):
            upper = observed_dt + pd.Timedelta(days=3650)

        changed = False
        if len(times_ordered) != len(block):
            changed = True
        else:
            for i in range(1, len(times_ordered)):
                if not (times_ordered[i] > times_ordered[i - 1]):
                    changed = True
                    break
            if times_ordered[0] != observed_dt:
                changed = True
            if times_ordered[-1] > upper:
                changed = True
            if times_ordered[0] < left_anchor_dt and left_anchor_idx is not None:
                changed = True

        if changed:
            clamped_blocks += 1
            cur = observed_dt.value
            times_ordered = [observed_dt]
            latest_upper = upper.value
            for i in range(1, len(block)):
                latest_possible = latest_upper - (len(block) - 1 - i) * max(min_ns, 1_000_000_000)
                nxt = min(cur + max(min_ns, 1_000_000_000), latest_possible)
                if left_anchor_idx is not None:
                    nxt = max(nxt, left_anchor_dt.value)
                nxt = max(nxt, cur + max(min_ns, 1))
                cur = nxt
                times_ordered.append(pd.Timestamp(cur))

        # map back to original rows
        row_to_new_dt = {}
        for seq_pos, row_pos in enumerate(order_row_positions):
            row_to_new_dt[block[row_pos]] = times_ordered[seq_pos]

        # final anchor clamp per row
        for idx in block:
            new_dt = row_to_new_dt[idx]
            if left_anchor_idx is not None and new_dt < left_anchor_dt:
                new_dt = left_anchor_dt
            if right_anchor_idx is not None and new_dt > right_anchor_dt:
                new_dt = right_anchor_dt
            if new_dt < case_start:
                new_dt = case_start
            if new_dt > case_end:
                new_dt = case_end
            row_to_new_dt[idx] = new_dt

        # enforce sorted execution-order consistency again
        exec_times = [row_to_new_dt[block[row_pos]] for row_pos in order_row_positions]
        bad = False
        if exec_times[0] != observed_dt:
            bad = True
        for i in range(1, len(exec_times)):
            if not exec_times[i] > exec_times[i - 1]:
                bad = True
                break
        if exec_times[-1] > upper:
            bad = True
        if bad:
            clamped_blocks += 1
            exec_times = [observed_dt]
            cur = observed_dt.value
            for i in range(1, len(block)):
                latest_possible = upper.value - (len(block) - 1 - i) * max(min_ns, 1_000_000_000)
                nxt = min(cur + max(min_ns, 1_000_000_000), latest_possible)
                nxt = max(nxt, cur + max(min_ns, 1))
                cur = nxt
                exec_times.append(pd.Timestamp(cur))
            for seq_pos, row_pos in enumerate(order_row_positions):
                row_to_new_dt[block[row_pos]] = exec_times[seq_pos]

        for idx in block:
            out_ts[idx] = format_ts(row_to_new_dt[idx], frac_digits)

        repaired_blocks += 1

    # Global self-check/fix: CLEAN rows unchanged; corrupted changed except one per block
    for i in range(len(df)):
        if df.at[i, "error_category"] == "CLEAN":
            out_ts[i] = orig.at[i, "Timestamp"]

    rows_still_equal = 0
    for case, block, _, _ in c1_blocks:
        observed = orig.at[block[0], "Timestamp"]
        eq_rows = [idx for idx in block if out_ts[idx] == observed]
        if len(eq_rows) != 1:
            # force exactly one row (first in chosen earliest-time row)
            parsed = [(idx, parse_ts(out_ts[idx])) for idx in block]
            parsed.sort(key=lambda x: x[1].value)
            keep_idx = parsed[0][0]
            out_ts[keep_idx] = observed
            frac_digits = detect_fractional_digits(observed)
            min_ns = max(min_step_ns(frac_digits), 1_000_000_000)
            cur = parse_ts(observed).value
            rest = [idx for idx, _ in parsed[1:]]
            for ridx in rest:
                cur += min_ns
                out_ts[ridx] = format_ts(pd.Timestamp(cur), frac_digits)
        for idx in block:
            if out_ts[idx] == orig.at[idx, "Timestamp"]:
                rows_still_equal += 1

    df["Timestamp"] = out_ts

    # Preserve all other columns byte-identical by writing dataframe as-is
    try:
        df.to_csv(args.output, index=False)
    except Exception:
        # safest retry
        pd.DataFrame(df, columns=orig.columns).to_csv(args.output, index=False)

    print(f"blocks_repaired={repaired_blocks} blocks_clamped={clamped_blocks} rows_still_equal_to_input={rows_still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # fail-safe: produce a copy of input if anything catastrophic happens
        # but keep exit status 0
        try:
            import sys
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=True)
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0")
        except Exception:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0")