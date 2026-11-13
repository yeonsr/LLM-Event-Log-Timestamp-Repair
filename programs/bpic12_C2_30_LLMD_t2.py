import argparse
import json
import math
import re
from collections import defaultdict

import pandas as pd


# Embedded process context used by the repair logic.
PROCESS_CONTEXT = {
    "duration_profiles_top": [
        {"source": "W_Nabellen offertes-COMPLETE", "target": "W_Nabellen offertes-START", "n": 16831, "p10_seconds": 46.211, "p25_seconds": 13993.699, "median_seconds": 165899.213, "p75_seconds": 442997.29150000005, "p90_seconds": 637001.579, "share_under_60s": 0.106},
        {"source": "W_Completeren aanvraag-START", "target": "W_Completeren aanvraag-COMPLETE", "n": 16077, "p10_seconds": 10.626800000000001, "p25_seconds": 40.38, "median_seconds": 98.527, "p75_seconds": 202.45, "p90_seconds": 475.39899999999983, "share_under_60s": 0.341},
        {"source": "W_Nabellen offertes-START", "target": "W_Nabellen offertes-COMPLETE", "n": 16023, "p10_seconds": 18.190600000000003, "p25_seconds": 55.7485, "median_seconds": 114.822, "p75_seconds": 206.486, "p90_seconds": 399.88480000000027, "share_under_60s": 0.27},
        {"source": "W_Completeren aanvraag-COMPLETE", "target": "W_Completeren aanvraag-START", "n": 15568, "p10_seconds": 61.138000000000005, "p25_seconds": 1594.0265, "median_seconds": 14749.843, "p75_seconds": 69796.52475, "p90_seconds": 164699.61340000006, "share_under_60s": 0.099},
        {"source": "A_SUBMITTED-COMPLETE", "target": "A_PARTLYSUBMITTED-COMPLETE", "n": 13087, "p10_seconds": 0.12560000000000013, "p25_seconds": 0.174, "median_seconds": 0.273, "p75_seconds": 0.499, "p90_seconds": 1.0807999999999993, "share_under_60s": 1.0},
        {"source": "W_Nabellen incomplete dossiers-COMPLETE", "target": "W_Nabellen incomplete dossiers-START", "n": 9065, "p10_seconds": 71.48280000000008, "p25_seconds": 1248.542, "median_seconds": 10755.049, "p75_seconds": 65504.338, "p90_seconds": 178981.7996, "share_under_60s": 0.094},
        {"source": "W_Nabellen incomplete dossiers-START", "target": "W_Nabellen incomplete dossiers-COMPLETE", "n": 8927, "p10_seconds": 10.157, "p25_seconds": 32.9045, "median_seconds": 138.258, "p75_seconds": 382.422, "p90_seconds": 890.1490000000011, "share_under_60s": 0.332},
        {"source": "A_PREACCEPTED-COMPLETE", "target": "W_Completeren aanvraag-SCHEDULE", "n": 7266, "p10_seconds": 0.046, "p25_seconds": 0.14525, "median_seconds": 0.38, "p75_seconds": 0.548, "p90_seconds": 0.892, "share_under_60s": 1.0},
        {"source": "O_CREATED-COMPLETE", "target": "O_SENT-COMPLETE", "n": 7030, "p10_seconds": 0.025, "p25_seconds": 0.028, "median_seconds": 0.035, "p75_seconds": 0.049, "p90_seconds": 0.079, "share_under_60s": 1.0},
        {"source": "O_SENT-COMPLETE", "target": "W_Nabellen offertes-SCHEDULE", "n": 6633, "p10_seconds": 0.083, "p25_seconds": 0.094, "median_seconds": 0.119, "p75_seconds": 0.227, "p90_seconds": 0.31, "share_under_60s": 1.0},
        {"source": "O_SELECTED-COMPLETE", "target": "O_CREATED-COMPLETE", "n": 6303, "p10_seconds": 0.9152000000000001, "p25_seconds": 1.0845, "median_seconds": 1.291, "p75_seconds": 1.5885, "p90_seconds": 2.5616000000000003, "share_under_60s": 1.0},
        {"source": "W_Nabellen offertes-SCHEDULE", "target": "W_Completeren aanvraag-COMPLETE", "n": 5014, "p10_seconds": 1.077, "p25_seconds": 1.179, "median_seconds": 1.386, "p75_seconds": 1.95075, "p90_seconds": 2.28, "share_under_60s": 1.0},
        {"source": "W_Completeren aanvraag-COMPLETE", "target": "W_Nabellen offertes-START", "n": 4913, "p10_seconds": 12.5592, "p25_seconds": 19.679, "median_seconds": 217.278, "p75_seconds": 601336.423, "p90_seconds": 673190.7256, "share_under_60s": 0.43},
        {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "A_PREACCEPTED-COMPLETE", "n": 4838, "p10_seconds": 32.8284, "p25_seconds": 35.57925, "median_seconds": 38.917500000000004, "p75_seconds": 43.059749999999994, "p90_seconds": 48.744400000000006, "share_under_60s": 0.974},
        {"source": "W_Completeren aanvraag-START", "target": "A_ACCEPTED-COMPLETE", "n": 4772, "p10_seconds": 120.3837, "p25_seconds": 242.27224999999999, "median_seconds": 368.76250000000005, "p75_seconds": 516.8530000000001, "p90_seconds": 722.8312, "share_under_60s": 0.051},
        {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "W_Afhandelen leads-SCHEDULE", "n": 4728, "p10_seconds": 8.7146, "p25_seconds": 11.882, "median_seconds": 22.6655, "p75_seconds": 39.5405, "p90_seconds": 47.2731, "share_under_60s": 0.953},
        {"source": "W_Completeren aanvraag-SCHEDULE", "target": "W_Completeren aanvraag-START", "n": 4508, "p10_seconds": 28.166100000000007, "p25_seconds": 633.0405, "median_seconds": 6399.7575, "p75_seconds": 44682.3065, "p90_seconds": 69334.2631, "share_under_60s": 0.155},
        {"source": "W_Afhandelen leads-SCHEDULE", "target": "W_Afhandelen leads-START", "n": 4405, "p10_seconds": 72.86820000000004, "p25_seconds": 571.685, "median_seconds": 2538.547, "p75_seconds": 35730.268, "p90_seconds": 53224.030999999995, "share_under_60s": 0.092},
        {"source": "A_FINALIZED-COMPLETE", "target": "O_SELECTED-COMPLETE", "n": 4289, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "A_ACCEPTED-COMPLETE", "target": "A_FINALIZED-COMPLETE", "n": 4266, "p10_seconds": 63.848, "p25_seconds": 118.72775, "median_seconds": 184.13150000000002, "p75_seconds": 280.47425, "p90_seconds": 409.93, "share_under_60s": 0.092},
        {"source": "O_SENT_BACK-COMPLETE", "target": "W_Valideren aanvraag-SCHEDULE", "n": 3432, "p10_seconds": 0.141, "p25_seconds": 0.15275, "median_seconds": 0.18, "p75_seconds": 0.274, "p90_seconds": 0.374, "share_under_60s": 0.999},
        {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "A_DECLINED-COMPLETE", "n": 3429, "p10_seconds": 31.669400000000003, "p25_seconds": 34.088, "median_seconds": 37.411, "p75_seconds": 41.491, "p90_seconds": 47.095200000000006, "share_under_60s": 0.98},
        {"source": "W_Valideren aanvraag-SCHEDULE", "target": "W_Nabellen offertes-COMPLETE", "n": 3254, "p10_seconds": 1.115, "p25_seconds": 1.229, "median_seconds": 1.4505, "p75_seconds": 1.893, "p90_seconds": 2.228, "share_under_60s": 1.0},
        {"source": "W_Nabellen offertes-COMPLETE", "target": "W_Valideren aanvraag-START", "n": 3164, "p10_seconds": 6703.307900000002, "p25_seconds": 89236.39275, "median_seconds": 189494.1955, "p75_seconds": 348573.559, "p90_seconds": 441072.17850000004, "share_under_60s": 0.049},
        {"source": "W_Nabellen offertes-START", "target": "O_SENT_BACK-COMPLETE", "n": 3120, "p10_seconds": 10.182, "p25_seconds": 12.3975, "median_seconds": 21.0825, "p75_seconds": 42.8485, "p90_seconds": 89.14909999999999, "share_under_60s": 0.832},
        {"source": "W_Valideren aanvraag-START", "target": "W_Valideren aanvraag-COMPLETE", "n": 2881, "p10_seconds": 10.801, "p25_seconds": 45.634, "median_seconds": 310.438, "p75_seconds": 1040.096, "p90_seconds": 2204.752, "share_under_60s": 0.282},
        {"source": "W_Valideren aanvraag-COMPLETE", "target": "W_Valideren aanvraag-START", "n": 2851, "p10_seconds": 24.091, "p25_seconds": 238.427, "median_seconds": 2016.402, "p75_seconds": 13280.25, "p90_seconds": 86082.238, "share_under_60s": 0.146},
        {"source": "W_Completeren aanvraag-SCHEDULE", "target": "W_Afhandelen leads-COMPLETE", "n": 2515, "p10_seconds": 0.899, "p25_seconds": 0.988, "median_seconds": 1.148, "p75_seconds": 1.367, "p90_seconds": 1.8546, "share_under_60s": 1.0},
        {"source": "W_Afhandelen leads-COMPLETE", "target": "W_Completeren aanvraag-START", "n": 2438, "p10_seconds": 17.3691, "p25_seconds": 858.7674999999999, "median_seconds": 5769.2875, "p75_seconds": 15620.251, "p90_seconds": 47555.64350000001, "share_under_60s": 0.154},
        {"source": "W_Afhandelen leads-START", "target": "A_PREACCEPTED-COMPLETE", "n": 2349, "p10_seconds": 35.382200000000005, "p25_seconds": 65.219, "median_seconds": 107.408, "p75_seconds": 184.817, "p90_seconds": 346.8740000000001, "share_under_60s": 0.213},
        {"source": "W_Valideren aanvraag-COMPLETE", "target": "W_Nabellen incomplete dossiers-START", "n": 2166, "p10_seconds": 25.928, "p25_seconds": 331.6065, "median_seconds": 2857.8005000000003, "p75_seconds": 7837.031, "p90_seconds": 13917.101, "share_under_60s": 0.158},
        {"source": "W_Nabellen incomplete dossiers-SCHEDULE", "target": "W_Valideren aanvraag-COMPLETE", "n": 2154, "p10_seconds": 1.4473, "p25_seconds": 1.6102500000000002, "median_seconds": 1.847, "p75_seconds": 2.269, "p90_seconds": 2.7245000000000004, "share_under_60s": 1.0},
        {"source": "A_DECLINED-COMPLETE", "target": "W_Afhandelen leads-COMPLETE", "n": 2026, "p10_seconds": 2.838, "p25_seconds": 3.08, "median_seconds": 3.412, "p75_seconds": 3.9145, "p90_seconds": 4.458, "share_under_60s": 0.999},
        {"source": "O_CANCELLED-COMPLETE", "target": "O_SELECTED-COMPLETE", "n": 2011, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "W_Valideren aanvraag-START", "target": "W_Nabellen incomplete dossiers-SCHEDULE", "n": 1924, "p10_seconds": 256.97200000000004, "p25_seconds": 627.48, "median_seconds": 1177.7265, "p75_seconds": 1810.88725, "p90_seconds": 2496.666, "share_under_60s": 0.028},
        {"source": "W_Afhandelen leads-START", "target": "A_DECLINED-COMPLETE", "n": 1798, "p10_seconds": 31.1872, "p25_seconds": 68.05325, "median_seconds": 111.291, "p75_seconds": 194.37924999999998, "p90_seconds": 370.6664999999999, "share_under_60s": 0.214},
        {"source": "W_Valideren aanvraag-SCHEDULE", "target": "W_Nabellen incomplete dossiers-COMPLETE", "n": 1727, "p10_seconds": 1.4856, "p25_seconds": 1.649, "median_seconds": 1.895, "p75_seconds": 2.293, "p90_seconds": 2.7488, "share_under_60s": 1.0},
        {"source": "W_Nabellen incomplete dossiers-COMPLETE", "target": "W_Valideren aanvraag-START", "n": 1719, "p10_seconds": 699.4614, "p25_seconds": 2120.2299999999996, "median_seconds": 5024.022, "p75_seconds": 46929.641, "p90_seconds": 78412.2382, "share_under_60s": 0.031},
        {"source": "A_APPROVED-COMPLETE", "target": "A_REGISTERED-COMPLETE", "n": 1662, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "A_CANCELLED-COMPLETE", "target": "O_CANCELLED-COMPLETE", "n": 1639, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "W_Nabellen offertes-SCHEDULE", "target": "W_Nabellen offertes-COMPLETE", "n": 1619, "p10_seconds": 1.22, "p25_seconds": 1.3115, "median_seconds": 1.618, "p75_seconds": 2.149, "p90_seconds": 2.4658, "share_under_60s": 0.999},
        {"source": "W_Nabellen offertes-START", "target": "O_CANCELLED-COMPLETE", "n": 1616, "p10_seconds": 31.9515, "p25_seconds": 48.48525, "median_seconds": 106.2115, "p75_seconds": 217.0865, "p90_seconds": 409.0865, "share_under_60s": 0.317},
        {"source": "W_Nabellen incomplete dossiers-START", "target": "W_Valideren aanvraag-SCHEDULE", "n": 1514, "p10_seconds": 24.7373, "p25_seconds": 36.484, "median_seconds": 64.6275, "p75_seconds": 200.2605, "p90_seconds": 567.5725000000001, "share_under_60s": 0.471},
        {"source": "A_ACTIVATED-COMPLETE", "target": "A_APPROVED-COMPLETE", "n": 1466, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "O_ACCEPTED-COMPLETE", "target": "A_ACTIVATED-COMPLETE", "n": 1285, "p10_seconds": 0.001, "p25_seconds": 0.001, "median_seconds": 0.001, "p75_seconds": 0.001, "p90_seconds": 0.001, "share_under_60s": 1.0},
        {"source": "W_Afhandelen leads-START", "target": "W_Afhandelen leads-COMPLETE", "n": 1260, "p10_seconds": 8.897400000000001, "p25_seconds": 56.97725, "median_seconds": 141.1065, "p75_seconds": 508.14525000000003, "p90_seconds": 2625.330200000001, "share_under_60s": 0.259},
        {"source": "A_REGISTERED-COMPLETE", "target": "O_ACCEPTED-COMPLETE", "n": 1126, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "W_Afhandelen leads-COMPLETE", "target": "W_Afhandelen leads-START", "n": 1067, "p10_seconds": 3.8134, "p25_seconds": 16.207500000000003, "median_seconds": 363.885, "p75_seconds": 2180.117, "p90_seconds": 8339.338399999997, "share_under_60s": 0.335},
        {"source": "A_DECLINED-COMPLETE", "target": "W_Completeren aanvraag-COMPLETE", "n": 1052, "p10_seconds": 2.9611, "p25_seconds": 3.1895, "median_seconds": 3.534, "p75_seconds": 4.04, "p90_seconds": 4.627399999999999, "share_under_60s": 0.999},
        {"source": "A_REGISTERED-COMPLETE", "target": "W_Valideren aanvraag-COMPLETE", "n": 1009, "p10_seconds": 6.4536, "p25_seconds": 7.027, "median_seconds": 7.763, "p75_seconds": 8.498, "p90_seconds": 9.366600000000002, "share_under_60s": 0.999},
        {"source": "O_CANCELLED-COMPLETE", "target": "W_Nabellen offertes-COMPLETE", "n": 993, "p10_seconds": 1.837, "p25_seconds": 2.093, "median_seconds": 2.396, "p75_seconds": 2.924, "p90_seconds": 3.4406000000000008, "share_under_60s": 0.998},
        {"source": "W_Completeren aanvraag-START", "target": "A_DECLINED-COMPLETE", "n": 990, "p10_seconds": 167.59320000000002, "p25_seconds": 312.663, "median_seconds": 469.317, "p75_seconds": 682.7525, "p90_seconds": 950.6196000000001, "share_under_60s": 0.024},
        {"source": "W_Nabellen offertes-START", "target": "A_CANCELLED-COMPLETE", "n": 913, "p10_seconds": 72.685, "p25_seconds": 111.148, "median_seconds": 156.879, "p75_seconds": 240.343, "p90_seconds": 403.3836000000003, "share_under_60s": 0.081},
        {"source": "W_Valideren aanvraag-START", "target": "A_ACTIVATED-COMPLETE", "n": 862, "p10_seconds": 141.36260000000001, "p25_seconds": 317.4815, "median_seconds": 770.573, "p75_seconds": 1332.394, "p90_seconds": 1957.7283999999997, "share_under_60s": 0.019},
        {"source": "O_ACCEPTED-COMPLETE", "target": "W_Valideren aanvraag-COMPLETE", "n": 862, "p10_seconds": 6.3086, "p25_seconds": 6.952, "median_seconds": 7.6685, "p75_seconds": 8.3765, "p90_seconds": 9.1902, "share_under_60s": 1.0},
        {"source": "A_DECLINED-COMPLETE", "target": "O_DECLINED-COMPLETE", "n": 801, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "A_CANCELLED-COMPLETE", "target": "W_Completeren aanvraag-COMPLETE", "n": 728, "p10_seconds": 1.3039, "p25_seconds": 1.463, "median_seconds": 1.6655, "p75_seconds": 2.12525, "p90_seconds": 2.648900000000003, "share_under_60s": 0.999},
        {"source": "A_FINALIZED-COMPLETE", "target": "O_CREATED-COMPLETE", "n": 726, "p10_seconds": 0.908, "p25_seconds": 1.053, "median_seconds": 1.2515, "p75_seconds": 1.5545, "p90_seconds": 2.1820000000000004, "share_under_60s": 1.0},
        {"source": "W_Nabellen offertes-COMPLETE", "target": "W_Nabellen offertes-COMPLETE", "n": 712, "p10_seconds": 48200.957500000004, "p25_seconds": 98554.79725, "median_seconds": 306248.4655, "p75_seconds": 737130.382, "p90_seconds": 1197428.6763, "share_under_60s": 0.0},
        {"source": "O_SELECTED-COMPLETE", "target": "A_FINALIZED-COMPLETE", "n": 709, "p10_seconds": 0.001, "p25_seconds": 0.001, "median_seconds": 0.001, "p75_seconds": 0.001, "p90_seconds": 0.001, "share_under_60s": 1.0}
    ],
    "global_median_duration_seconds": 47.217,
    "process_abstraction": {
        "dfg_edges": [
            {"source": "W_Nabellen offertes-COMPLETE", "target": "W_Nabellen offertes-START", "frequency": 16831, "median_duration_seconds": 165899.213},
            {"source": "W_Completeren aanvraag-START", "target": "W_Completeren aanvraag-COMPLETE", "frequency": 16077, "median_duration_seconds": 98.527},
            {"source": "W_Nabellen offertes-START", "target": "W_Nabellen offertes-COMPLETE", "frequency": 16023, "median_duration_seconds": 114.822},
            {"source": "W_Completeren aanvraag-COMPLETE", "target": "W_Completeren aanvraag-START", "frequency": 15568, "median_duration_seconds": 14749.843},
            {"source": "A_SUBMITTED-COMPLETE", "target": "A_PARTLYSUBMITTED-COMPLETE", "frequency": 13087, "median_duration_seconds": 0.273},
            {"source": "W_Nabellen incomplete dossiers-COMPLETE", "target": "W_Nabellen incomplete dossiers-START", "frequency": 9065, "median_duration_seconds": 10755.049},
            {"source": "W_Nabellen incomplete dossiers-START", "target": "W_Nabellen incomplete dossiers-COMPLETE", "frequency": 8927, "median_duration_seconds": 138.258},
            {"source": "A_PREACCEPTED-COMPLETE", "target": "W_Completeren aanvraag-SCHEDULE", "frequency": 7266, "median_duration_seconds": 0.38},
            {"source": "O_CREATED-COMPLETE", "target": "O_SENT-COMPLETE", "frequency": 7030, "median_duration_seconds": 0.035},
            {"source": "O_SENT-COMPLETE", "target": "W_Nabellen offertes-SCHEDULE", "frequency": 6633, "median_duration_seconds": 0.119},
            {"source": "O_SELECTED-COMPLETE", "target": "O_CREATED-COMPLETE", "frequency": 6303, "median_duration_seconds": 1.291},
            {"source": "W_Nabellen offertes-SCHEDULE", "target": "W_Completeren aanvraag-COMPLETE", "frequency": 5014, "median_duration_seconds": 1.3860000000000001},
            {"source": "W_Completeren aanvraag-COMPLETE", "target": "W_Nabellen offertes-START", "frequency": 4913, "median_duration_seconds": 217.278},
            {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "A_PREACCEPTED-COMPLETE", "frequency": 4838, "median_duration_seconds": 38.917500000000004},
            {"source": "W_Completeren aanvraag-START", "target": "A_ACCEPTED-COMPLETE", "frequency": 4772, "median_duration_seconds": 368.76250000000005},
            {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "W_Afhandelen leads-SCHEDULE", "frequency": 4728, "median_duration_seconds": 22.6655},
            {"source": "W_Completeren aanvraag-SCHEDULE", "target": "W_Completeren aanvraag-START", "frequency": 4508, "median_duration_seconds": 6399.7575},
            {"source": "W_Afhandelen leads-SCHEDULE", "target": "W_Afhandelen leads-START", "frequency": 4405, "median_duration_seconds": 2538.547},
            {"source": "A_FINALIZED-COMPLETE", "target": "O_SELECTED-COMPLETE", "frequency": 4289, "median_duration_seconds": 0.0},
            {"source": "A_ACCEPTED-COMPLETE", "target": "A_FINALIZED-COMPLETE", "frequency": 4266, "median_duration_seconds": 184.13150000000002},
            {"source": "O_SENT_BACK-COMPLETE", "target": "W_Valideren aanvraag-SCHEDULE", "frequency": 3432, "median_duration_seconds": 0.18},
            {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "A_DECLINED-COMPLETE", "frequency": 3429, "median_duration_seconds": 37.411},
            {"source": "W_Valideren aanvraag-SCHEDULE", "target": "W_Nabellen offertes-COMPLETE", "frequency": 3254, "median_duration_seconds": 1.4505},
            {"source": "W_Nabellen offertes-COMPLETE", "target": "W_Valideren aanvraag-START", "frequency": 3164, "median_duration_seconds": 189494.1955},
            {"source": "W_Nabellen offertes-START", "target": "O_SENT_BACK-COMPLETE", "frequency": 3120, "median_duration_seconds": 21.0825},
            {"source": "W_Valideren aanvraag-START", "target": "W_Valideren aanvraag-COMPLETE", "frequency": 2881, "median_duration_seconds": 310.438},
            {"source": "W_Valideren aanvraag-COMPLETE", "target": "W_Valideren aanvraag-START", "frequency": 2851, "median_duration_seconds": 2016.402},
            {"source": "W_Completeren aanvraag-SCHEDULE", "target": "W_Afhandelen leads-COMPLETE", "frequency": 2515, "median_duration_seconds": 1.148},
            {"source": "W_Afhandelen leads-COMPLETE", "target": "W_Completeren aanvraag-START", "frequency": 2438, "median_duration_seconds": 5769.2875},
            {"source": "W_Afhandelen leads-START", "target": "A_PREACCEPTED-COMPLETE", "frequency": 2349, "median_duration_seconds": 107.408},
            {"source": "W_Valideren aanvraag-COMPLETE", "target": "W_Nabellen incomplete dossiers-START", "frequency": 2166, "median_duration_seconds": 2857.8005000000003},
            {"source": "W_Nabellen incomplete dossiers-SCHEDULE", "target": "W_Valideren aanvraag-COMPLETE", "frequency": 2154, "median_duration_seconds": 1.847},
            {"source": "A_DECLINED-COMPLETE", "target": "W_Afhandelen leads-COMPLETE", "frequency": 2026, "median_duration_seconds": 3.412},
            {"source": "O_CANCELLED-COMPLETE", "target": "O_SELECTED-COMPLETE", "frequency": 2011, "median_duration_seconds": 0.0},
            {"source": "W_Valideren aanvraag-START", "target": "W_Nabellen incomplete dossiers-SCHEDULE", "frequency": 1924, "median_duration_seconds": 1177.7265},
            {"source": "W_Afhandelen leads-START", "target": "A_DECLINED-COMPLETE", "frequency": 1798, "median_duration_seconds": 111.291},
            {"source": "W_Valideren aanvraag-SCHEDULE", "target": "W_Nabellen incomplete dossiers-COMPLETE", "frequency": 1727, "median_duration_seconds": 1.895},
            {"source": "W_Nabellen incomplete dossiers-COMPLETE", "target": "W_Valideren aanvraag-START", "frequency": 1719, "median_duration_seconds": 5024.022},
            {"source": "A_APPROVED-COMPLETE", "target": "A_REGISTERED-COMPLETE", "frequency": 1662, "median_duration_seconds": 0.0},
            {"source": "A_CANCELLED-COMPLETE", "target": "O_CANCELLED-COMPLETE", "frequency": 1639, "median_duration_seconds": 0.0},
            {"source": "W_Nabellen offertes-SCHEDULE", "target": "W_Nabellen offertes-COMPLETE", "frequency": 1619, "median_duration_seconds": 1.6179999999999999},
            {"source": "W_Nabellen offertes-START", "target": "O_CANCELLED-COMPLETE", "frequency": 1616, "median_duration_seconds": 106.2115},
            {"source": "W_Nabellen incomplete dossiers-START", "target": "W_Valideren aanvraag-SCHEDULE", "frequency": 1514, "median_duration_seconds": 64.6275},
            {"source": "A_ACTIVATED-COMPLETE", "target": "A_APPROVED-COMPLETE", "frequency": 1466, "median_duration_seconds": 0.0},
            {"source": "O_ACCEPTED-COMPLETE", "target": "A_ACTIVATED-COMPLETE", "frequency": 1285, "median_duration_seconds": 0.001},
            {"source": "W_Afhandelen leads-START", "target": "W_Afhandelen leads-COMPLETE", "frequency": 1260, "median_duration_seconds": 141.1065},
            {"source": "A_REGISTERED-COMPLETE", "target": "O_ACCEPTED-COMPLETE", "frequency": 1126, "median_duration_seconds": 0.0},
            {"source": "W_Afhandelen leads-COMPLETE", "target": "W_Afhandelen leads-START", "frequency": 1067, "median_duration_seconds": 363.885},
            {"source": "A_DECLINED-COMPLETE", "target": "W_Completeren aanvraag-COMPLETE", "frequency": 1052, "median_duration_seconds": 3.534},
            {"source": "A_REGISTERED-COMPLETE", "target": "W_Valideren aanvraag-COMPLETE", "frequency": 1009, "median_duration_seconds": 7.763},
            {"source": "O_CANCELLED-COMPLETE", "target": "W_Nabellen offertes-COMPLETE", "frequency": 993, "median_duration_seconds": 2.396},
            {"source": "W_Completeren aanvraag-START", "target": "A_DECLINED-COMPLETE", "frequency": 990, "median_duration_seconds": 469.317},
            {"source": "W_Nabellen offertes-START", "target": "A_CANCELLED-COMPLETE", "frequency": 913, "median_duration_seconds": 156.879},
            {"source": "W_Valideren aanvraag-START", "target": "A_ACTIVATED-COMPLETE", "frequency": 862, "median_duration_seconds": 770.573},
            {"source": "O_ACCEPTED-COMPLETE", "target": "W_Valideren aanvraag-COMPLETE", "frequency": 862, "median_duration_seconds": 7.6685},
            {"source": "A_DECLINED-COMPLETE", "target": "O_DECLINED-COMPLETE", "frequency": 801, "median_duration_seconds": 0.0},
            {"source": "A_CANCELLED-COMPLETE", "target": "W_Completeren aanvraag-COMPLETE", "frequency": 728, "median_duration_seconds": 1.6655000000000002},
            {"source": "A_FINALIZED-COMPLETE", "target": "O_CREATED-COMPLETE", "frequency": 726, "median_duration_seconds": 1.2515},
            {"source": "W_Nabellen offertes-COMPLETE", "target": "W_Nabellen offertes-COMPLETE", "frequency": 712, "median_duration_seconds": 306248.4655},
            {"source": "O_SELECTED-COMPLETE", "target": "A_FINALIZED-COMPLETE", "frequency": 709, "median_duration_seconds": 0.001}
        ],
        "frequent_variants": [
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"], "count": 3429},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"], "count": 1464},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"], "count": 212},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"], "count": 209},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"], "count": 192},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 164},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 130},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 108},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 90},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 78},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 72},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 57},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START", "A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"], "count": 47},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 44},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 41},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 38},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 35},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"], "count": 35},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE", "O_CREATED-COMPLETE", "O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START", "A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"], "count": 34},
            {"variant": ["A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START", "A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"], "count": 30}
        ]
    }
}

DURATION_MAP = {(d["source"], d["target"]): d for d in PROCESS_CONTEXT["duration_profiles_top"]}
DFG_FREQ = {(d["source"], d["target"]): d["frequency"] for d in PROCESS_CONTEXT["process_abstraction"]["dfg_edges"]}
VARIANTS = [(v["variant"], v["count"]) for v in PROCESS_CONTEXT["process_abstraction"]["frequent_variants"]]
GLOBAL_MEDIAN = PROCESS_CONTEXT["global_median_duration_seconds"]


def detect_precision(ts_str):
    m = re.search(r"\.(\d+)$", str(ts_str))
    return len(m.group(1)) if m else 0


def format_ts(ts, precision):
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    frac = base.split(".")[1][:precision]
    return base.split(".")[0] + "." + frac


def parse_ts(x):
    return pd.to_datetime(x)


def td_seconds(x):
    return x.total_seconds()


def small_step(precision):
    if precision <= 0:
        return pd.Timedelta(seconds=1)
    return pd.Timedelta(microseconds=10 ** (6 - precision))


def profile_seconds(src, tgt):
    prof = DURATION_MAP.get((src, tgt))
    if prof:
        if prof["share_under_60s"] >= 0.5:
            return prof["p10_seconds"] if prof["p10_seconds"] > 0 else prof["p25_seconds"]
        return prof["p25_seconds"]
    return GLOBAL_MEDIAN


def edge_exists(src, tgt):
    return (src, tgt) in DFG_FREQ


def edge_score(src, tgt):
    return math.log1p(DFG_FREQ.get((src, tgt), 0))


def duration_fit_score(src, tgt, gap_seconds):
    if gap_seconds is None or gap_seconds < 0:
        return -1e9
    prof = DURATION_MAP.get((src, tgt))
    if not prof:
        target = GLOBAL_MEDIAN
        return -abs(math.log1p(gap_seconds) - math.log1p(target))
    center = prof["p25_seconds"]
    if prof["share_under_60s"] >= 0.5 and prof["p10_seconds"] >= 0:
        center = max(prof["p10_seconds"], 0.0)
    if center == 0:
        center = 0.001
    return -abs(math.log1p(gap_seconds) - math.log1p(center))


def subseq_variant_score(activities):
    best = 0
    for variant, count in VARIANTS:
        for i in range(0, len(variant) - len(activities) + 1):
            if variant[i:i + len(activities)] == activities:
                best = max(best, count)
                break
    return math.log1p(best)


def local_ordering_score(prev_act, act1, act2, next_act):
    score = 0.0
    if prev_act is not None:
        score += edge_score(prev_act, act1)
    score += edge_score(act1, act2)
    if next_act is not None:
        score += edge_score(act2, next_act)
    score += subseq_variant_score([x for x in [prev_act, act1, act2, next_act] if x is not None])
    return score


def candidate_valid_order(prev_act, act1, act2, next_act):
    if prev_act is not None and not edge_exists(prev_act, act1):
        return False
    if not edge_exists(act1, act2):
        return False
    if next_act is not None and not edge_exists(act2, next_act):
        return False
    return True


def compute_case_bounds(g):
    parsed = g["_parsed_ts"]
    return parsed.min(), parsed.max()


def find_clean_anchor_left(g, idx):
    sub = g.loc[:idx - 1] if idx > 0 else g.iloc[0:0]
    sub = sub[sub["error_category"] == "CLEAN"]
    if len(sub) == 0:
        return None
    return sub.iloc[-1].name, sub.iloc[-1]["_parsed_ts"]


def find_clean_anchor_right(g, idx):
    sub = g.loc[idx + 1:] if idx + 1 < len(g) else g.iloc[0:0]
    sub = sub[sub["error_category"] == "CLEAN"]
    if len(sub) == 0:
        return None
    return sub.iloc[0].name, sub.iloc[0]["_parsed_ts"]


def clamp(ts, lo, hi):
    if lo is not None and ts < lo:
        ts = lo
    if hi is not None and ts > hi:
        ts = hi
    return ts


def estimate_from_anchors(prev_row, curr_row, next_row, left_bound, right_bound, precision, observed_ts):
    estimates = []
    if prev_row is not None:
        sec = profile_seconds(prev_row["Activity"], curr_row["Activity"])
        estimates.append(prev_row["_candidate_ts"] + pd.to_timedelta(sec, unit="s"))
    if next_row is not None:
        sec = profile_seconds(curr_row["Activity"], next_row["Activity"])
        estimates.append(next_row["_candidate_ts"] - pd.to_timedelta(sec, unit="s"))
    if not estimates:
        base = observed_ts
    elif len(estimates) == 1:
        base = estimates[0]
    else:
        base = estimates[0] + (estimates[1] - estimates[0]) / 2
    ts = clamp(base, left_bound, right_bound)
    step = small_step(precision)
    if ts == observed_ts:
        if right_bound is not None and ts + step <= right_bound:
            ts = ts + step
        elif left_bound is not None and ts - step >= left_bound:
            ts = ts - step
        else:
            ts = ts + step
    ts = clamp(ts, left_bound, right_bound)
    return ts


def score_time_candidate(prev_row, curr_row, next_row, ts):
    score = 0.0
    if prev_row is not None:
        gap = td_seconds(ts - prev_row["_candidate_ts"])
        score += duration_fit_score(prev_row["Activity"], curr_row["Activity"], gap)
    if next_row is not None:
        gap = td_seconds(next_row["_candidate_ts"] - ts)
        score += duration_fit_score(curr_row["Activity"], next_row["Activity"], gap)
    return score


def choose_displaced_fix(g, i, precision):
    prev_row = g.iloc[i - 1] if i - 1 >= 0 else None
    row1 = g.iloc[i]
    row2 = g.iloc[i + 1]
    next_row = g.iloc[i + 2] if i + 2 < len(g) else None

    case_start, case_end = compute_case_bounds(g)

    candidates = []

    # keep row1, move row2
    row1_keep_ts = row1["_parsed_ts"]
    left_bound = max([x for x in [row1_keep_ts, case_start] if x is not None])
    right_bound = case_end
    if next_row is not None:
        right_bound = min(right_bound, next_row["_candidate_ts"])
    est2 = estimate_from_anchors(
        {"Activity": row1["Activity"], "_candidate_ts": row1_keep_ts},
        {"Activity": row2["Activity"]},
        {"Activity": next_row["Activity"], "_candidate_ts": next_row["_candidate_ts"]} if next_row is not None else None,
        left_bound,
        right_bound,
        precision,
        row2["_parsed_ts"],
    )
    valid_a = est2 >= row1_keep_ts and est2 >= case_start and est2 <= case_end and (next_row is None or est2 <= next_row["_candidate_ts"])
    if valid_a:
        move = abs(td_seconds(est2 - row2["_parsed_ts"]))
        score = score_time_candidate(
            {"Activity": row1["Activity"], "_candidate_ts": row1_keep_ts},
            {"Activity": row2["Activity"]},
            {"Activity": next_row["Activity"], "_candidate_ts": next_row["_candidate_ts"]} if next_row is not None else None,
            est2,
        )
        if prev_row is not None:
            score += duration_fit_score(prev_row["Activity"], row1["Activity"], td_seconds(row1_keep_ts - prev_row["_candidate_ts"]))
        candidates.append(("move_second", row1_keep_ts, est2, score, move, 1))

    # keep row2, move row1
    row2_keep_ts = row2["_parsed_ts"]
    left_bound = case_start
    if prev_row is not None:
        left_bound = max(left_bound, prev_row["_candidate_ts"])
    right_bound = min([x for x in [row2_keep_ts, case_end] if x is not None])
    est1 = estimate_from_anchors(
        {"Activity": prev_row["Activity"], "_candidate_ts": prev_row["_candidate_ts"]} if prev_row is not None else None,
        {"Activity": row1["Activity"]},
        {"Activity": row2["Activity"], "_candidate_ts": row2_keep_ts},
        left_bound,
        right_bound,
        precision,
        row1["_parsed_ts"],
    )
    valid_b = est1 <= row2_keep_ts and est1 >= case_start and est1 <= case_end and (prev_row is None or est1 >= prev_row["_candidate_ts"])
    if valid_b:
        move = abs(td_seconds(est1 - row1["_parsed_ts"]))
        score = score_time_candidate(
            {"Activity": prev_row["Activity"], "_candidate_ts": prev_row["_candidate_ts"]} if prev_row is not None else None,
            {"Activity": row1["Activity"]},
            {"Activity": row2["Activity"], "_candidate_ts": row2_keep_ts},
            est1,
        )
        if next_row is not None:
            score += duration_fit_score(row2["Activity"], next_row["Activity"], td_seconds(next_row["_candidate_ts"] - row2_keep_ts))
        candidates.append(("move_first", est1, row2_keep_ts, score, move, 1))

    if not candidates:
        step = small_step(precision)
        est1 = clamp(row1["_parsed_ts"], case_start, min(case_end, row2["_parsed_ts"]))
        if est1 == row1["_parsed_ts"]:
            est1 = clamp(est1 - step, case_start, min(case_end, row2["_parsed_ts"]))
        return "move_first", est1, row2["_parsed_ts"]

    candidates.sort(key=lambda x: (-x[3], x[4], x[5]))
    best = candidates[0]
    return best[0], best[1], best[2]


def repair_case(g):
    g = g.copy().reset_index(drop=True)
    g["_orig_ts"] = g["Timestamp"].astype(str)
    g["_parsed_ts"] = pd.to_datetime(g["Timestamp"])
    g["_candidate_ts"] = g["_parsed_ts"]
    g["_out_ts"] = g["_orig_ts"]
    g["_modified"] = False

    precision = max(detect_precision(x) for x in g["_orig_ts"].tolist())
    step = small_step(precision)
    exchanged_segments = 0
    displaced_segments = 0

    for i in range(len(g) - 1):
        if g.loc[i, "_candidate_ts"] <= g.loc[i + 1, "_candidate_ts"]:
            continue
        if not (g.loc[i, "error_category"] != "CLEAN" or g.loc[i + 1, "error_category"] != "CLEAN"):
            continue

        prev_act = g.loc[i - 1, "Activity"] if i - 1 >= 0 else None
        act1 = g.loc[i, "Activity"]
        act2 = g.loc[i + 1, "Activity"]
        next_act = g.loc[i + 2, "Activity"] if i + 2 < len(g) else None

        unswapped_valid = candidate_valid_order(prev_act, act1, act2, next_act)
        swapped_valid = candidate_valid_order(prev_act, act2, act1, next_act)

        left_anchor = g.loc[i - 1, "_candidate_ts"] if i - 1 >= 0 else compute_case_bounds(g)[0]
        right_anchor = g.loc[i + 2, "_candidate_ts"] if i + 2 < len(g) else compute_case_bounds(g)[1]

        t1_obs = g.loc[i, "_parsed_ts"]
        t2_obs = g.loc[i + 1, "_parsed_ts"]

        swapped_monotone = (left_anchor <= t2_obs <= t1_obs <= right_anchor)

        swap_score = -1e18
        if swapped_valid and swapped_monotone:
            swap_score = local_ordering_score(prev_act, act2, act1, next_act)
            if prev_act is not None:
                swap_score += duration_fit_score(prev_act, act2, td_seconds(t2_obs - left_anchor))
            swap_score += duration_fit_score(act2, act1, td_seconds(t1_obs - t2_obs))
            if next_act is not None:
                swap_score += duration_fit_score(act1, next_act, td_seconds(right_anchor - t1_obs))

        displaced_score = -1e17
        disp_choice, disp_t1, disp_t2 = choose_displaced_fix(g, i, precision)
        if unswapped_valid:
            displaced_score = local_ordering_score(prev_act, act1, act2, next_act)
            if prev_act is not None:
                displaced_score += duration_fit_score(prev_act, act1, td_seconds(disp_t1 - left_anchor))
            displaced_score += duration_fit_score(act1, act2, td_seconds(disp_t2 - disp_t1))
            if next_act is not None:
                displaced_score += duration_fit_score(act2, next_act, td_seconds(right_anchor - disp_t2))
            moved = (abs(td_seconds(disp_t1 - t1_obs)) if disp_choice == "move_first" else abs(td_seconds(disp_t2 - t2_obs)))
            displaced_score -= moved / 3600.0

        if swap_score > displaced_score:
            if g.loc[i, "error_category"] != "CLEAN":
                g.loc[i, "_candidate_ts"] = t2_obs
                g.loc[i, "_out_ts"] = g.loc[i + 1, "_orig_ts"]
                g.loc[i, "_modified"] = True
            if g.loc[i + 1, "error_category"] != "CLEAN":
                g.loc[i + 1, "_candidate_ts"] = t1_obs
                g.loc[i + 1, "_out_ts"] = g.loc[i, "_orig_ts"]
                g.loc[i + 1, "_modified"] = True
            exchanged_segments += 1
        else:
            if disp_choice == "move_first":
                if g.loc[i, "error_category"] != "CLEAN":
                    g.loc[i, "_candidate_ts"] = disp_t1
                    g.loc[i, "_out_ts"] = format_ts(disp_t1, precision)
                    g.loc[i, "_modified"] = True
                if g.loc[i + 1, "error_category"] != "CLEAN":
                    g.loc[i + 1, "_candidate_ts"] = t2_obs
                    g.loc[i + 1, "_out_ts"] = g.loc[i + 1, "_orig_ts"]
                    g.loc[i + 1, "_modified"] = False
            else:
                if g.loc[i, "error_category"] != "CLEAN":
                    g.loc[i, "_candidate_ts"] = t1_obs
                    g.loc[i, "_out_ts"] = g.loc[i, "_orig_ts"]
                    g.loc[i, "_modified"] = False
                if g.loc[i + 1, "error_category"] != "CLEAN":
                    g.loc[i + 1, "_candidate_ts"] = disp_t2
                    g.loc[i + 1, "_out_ts"] = format_ts(disp_t2, precision)
                    g.loc[i + 1, "_modified"] = True
            displaced_segments += 1

    # self-heal monotonicity and anchor constraints without touching CLEAN rows
    case_start, case_end = compute_case_bounds(g)
    for i in range(len(g)):
        if g.loc[i, "error_category"] == "CLEAN":
            g.loc[i, "_candidate_ts"] = g.loc[i, "_parsed_ts"]
            g.loc[i, "_out_ts"] = g.loc[i, "_orig_ts"]

    for i in range(1, len(g)):
        if g.loc[i, "_candidate_ts"] < g.loc[i - 1, "_candidate_ts"]:
            if g.loc[i, "error_category"] != "CLEAN":
                new_ts = g.loc[i - 1, "_candidate_ts"] + step
                if i + 1 < len(g):
                    upper = g.loc[i + 1, "_candidate_ts"]
                    if new_ts > upper:
                        new_ts = upper
                new_ts = clamp(new_ts, case_start, case_end)
                if new_ts == g.loc[i, "_parsed_ts"]:
                    new_ts = clamp(new_ts + step, case_start, case_end)
                g.loc[i, "_candidate_ts"] = new_ts
                g.loc[i, "_out_ts"] = format_ts(new_ts, precision)
                g.loc[i, "_modified"] = True
            elif g.loc[i - 1, "error_category"] != "CLEAN":
                new_ts = g.loc[i, "_candidate_ts"] - step
                lower = g.loc[i - 2, "_candidate_ts"] if i - 2 >= 0 else case_start
                if new_ts < lower:
                    new_ts = lower
                new_ts = clamp(new_ts, case_start, case_end)
                if new_ts == g.loc[i - 1, "_parsed_ts"]:
                    new_ts = clamp(new_ts - step, case_start, case_end)
                g.loc[i - 1, "_candidate_ts"] = new_ts
                g.loc[i - 1, "_out_ts"] = format_ts(new_ts, precision)
                g.loc[i - 1, "_modified"] = True

    for i in range(len(g)):
        if g.loc[i, "error_category"] != "CLEAN":
            left_bound = case_start
            right_bound = case_end
            left_anchor = find_clean_anchor_left(g, i)
            right_anchor = find_clean_anchor_right(g, i)
            if left_anchor is not None:
                left_bound = max(left_bound, left_anchor[1])
            if right_anchor is not None:
                right_bound = min(right_bound, right_anchor[1])
            ts = clamp(g.loc[i, "_candidate_ts"], left_bound, right_bound)
            if i > 0 and ts < g.loc[i - 1, "_candidate_ts"]:
                ts = g.loc[i - 1, "_candidate_ts"] + step
            if i + 1 < len(g) and ts > g.loc[i + 1, "_candidate_ts"]:
                ts = g.loc[i + 1, "_candidate_ts"] - step
            ts = clamp(ts, left_bound, right_bound)
            if ts == g.loc[i, "_parsed_ts"]:
                if ts + step <= right_bound and (i + 1 == len(g) or ts + step <= g.loc[i + 1, "_candidate_ts"]):
                    ts = ts + step
                elif ts - step >= left_bound and (i == 0 or ts - step >= g.loc[i - 1, "_candidate_ts"]):
                    ts = ts - step
                else:
                    alt = estimate_from_anchors(
                        {"Activity": g.loc[i - 1, "Activity"], "_candidate_ts": g.loc[i - 1, "_candidate_ts"]} if i > 0 else None,
                        {"Activity": g.loc[i, "Activity"]},
                        {"Activity": g.loc[i + 1, "Activity"], "_candidate_ts": g.loc[i + 1, "_candidate_ts"]} if i + 1 < len(g) else None,
                        left_bound,
                        right_bound,
                        precision,
                        g.loc[i, "_parsed_ts"],
                    )
                    ts = alt
            g.loc[i, "_candidate_ts"] = ts
            g.loc[i, "_out_ts"] = format_ts(ts, precision)

    return g, exchanged_segments, displaced_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str)
    except Exception:
        df = pd.read_csv(args.input)

    original_columns = list(df.columns)
    for col in ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]:
        if col not in df.columns:
            df[col] = ""

    df["_orig_index"] = range(len(df))
    out_parts = []
    exchanged_total = 0
    displaced_total = 0

    for _, g in df.groupby("Case", sort=False, group_keys=False):
        repaired, ex_n, dis_n = repair_case(g)
        exchanged_total += ex_n
        displaced_total += dis_n
        out_parts.append(repaired)

    out = pd.concat(out_parts, axis=0).sort_values("_orig_index").reset_index(drop=True)

    rows_still_equal = 0
    for i in range(len(out)):
        if out.loc[i, "error_category"] == "CLEAN":
            out.loc[i, "Timestamp"] = out.loc[i, "_orig_ts"]
        else:
            out.loc[i, "Timestamp"] = out.loc[i, "_out_ts"]
            if str(out.loc[i, "Timestamp"]) == str(out.loc[i, "_orig_ts"]):
                rows_still_equal += 1

    # final monotonicity pass per case, repair only corrupted rows
    for case, idxs in out.groupby("Case", sort=False).groups.items():
        idxs = list(idxs)
        precision = max(detect_precision(out.loc[j, "_orig_ts"]) for j in idxs)
        step = small_step(precision)
        parsed = {j: parse_ts(out.loc[j, "Timestamp"]) for j in idxs}
        case_start = min(parsed.values())
        case_end = max(parsed.values())

        changed = True
        guard = 0
        while changed and guard < len(idxs) * 3:
            changed = False
            guard += 1
            for pos in range(1, len(idxs)):
                j_prev = idxs[pos - 1]
                j = idxs[pos]
                if parsed[j] < parsed[j_prev]:
                    if out.loc[j, "error_category"] != "CLEAN":
                        new_ts = parsed[j_prev] + step
                        if pos + 1 < len(idxs):
                            nxt = idxs[pos + 1]
                            if new_ts > parsed[nxt]:
                                new_ts = parsed[nxt]
                        if new_ts == parse_ts(out.loc[j, "_orig_ts"]):
                            new_ts = new_ts + step
                        new_ts = clamp(new_ts, case_start, case_end)
                        parsed[j] = new_ts
                        out.loc[j, "Timestamp"] = format_ts(new_ts, precision)
                        changed = True
                    elif out.loc[j_prev, "error_category"] != "CLEAN":
                        new_ts = parsed[j] - step
                        if pos - 2 >= 0:
                            lower = parsed[idxs[pos - 2]]
                            if new_ts < lower:
                                new_ts = lower
                        if new_ts == parse_ts(out.loc[j_prev, "_orig_ts"]):
                            new_ts = new_ts - step
                        new_ts = clamp(new_ts, case_start, case_end)
                        parsed[j_prev] = new_ts
                        out.loc[j_prev, "Timestamp"] = format_ts(new_ts, precision)
                        changed = True

    # recompute rows_still_equal after final adjustments
    rows_still_equal = int(((out["error_category"] != "CLEAN") & (out["Timestamp"].astype(str) == out["_orig_ts"].astype(str))).sum())

    # preserve byte-identical non-timestamp columns and row order/count
    final_df = out[original_columns].copy()

    try:
        final_df.to_csv(args.output, index=False)
    except Exception:
        safe_cols = [c for c in original_columns if c in final_df.columns]
        final_df[safe_cols].to_csv(args.output, index=False)

    print(f"exchanged_segments={exchanged_total} displaced_segments={displaced_total} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    main()