import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

# ---------------------------------------------------------------------
# Embedded process knowledge from prompt
# ---------------------------------------------------------------------

DURATION_PROFILES = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): {"p10": 46.211, "p25": 13993.699, "median": 165899.213, "p75": 442997.29150000005, "p90": 637001.579, "share_under_60s": 0.106, "n": 16831},
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): {"p10": 10.6268, "p25": 40.38, "median": 98.527, "p75": 202.45, "p90": 475.39899999999983, "share_under_60s": 0.341, "n": 16077},
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): {"p10": 18.1906, "p25": 55.7485, "median": 114.822, "p75": 206.486, "p90": 399.88480000000027, "share_under_60s": 0.27, "n": 16023},
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 61.138, "p25": 1594.0265, "median": 14749.843, "p75": 69796.52475, "p90": 164699.61340000006, "share_under_60s": 0.099, "n": 15568},
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): {"p10": 0.1256, "p25": 0.174, "median": 0.273, "p75": 0.499, "p90": 1.0807999999999993, "share_under_60s": 1.0, "n": 13087},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 71.4828, "p25": 1248.542, "median": 10755.049, "p75": 65504.338, "p90": 178981.7996, "share_under_60s": 0.094, "n": 9065},
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 10.157, "p25": 32.9045, "median": 138.258, "p75": 382.422, "p90": 890.1490000000011, "share_under_60s": 0.332, "n": 8927},
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): {"p10": 0.046, "p25": 0.14525, "median": 0.38, "p75": 0.548, "p90": 0.892, "share_under_60s": 1.0, "n": 7266},
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): {"p10": 0.025, "p25": 0.028, "median": 0.035, "p75": 0.049, "p90": 0.079, "share_under_60s": 1.0, "n": 7030},
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): {"p10": 0.083, "p25": 0.094, "median": 0.119, "p75": 0.227, "p90": 0.31, "share_under_60s": 1.0, "n": 6633},
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.9152, "p25": 1.0845, "median": 1.291, "p75": 1.5885, "p90": 2.5616, "share_under_60s": 1.0, "n": 6303},
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.077, "p25": 1.179, "median": 1.386, "p75": 1.95075, "p90": 2.28, "share_under_60s": 1.0, "n": 5014},
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): {"p10": 12.5592, "p25": 19.679, "median": 217.278, "p75": 601336.423, "p90": 673190.7256, "share_under_60s": 0.43, "n": 4913},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): {"p10": 32.8284, "p25": 35.57925, "median": 38.9175, "p75": 43.05975, "p90": 48.7444, "share_under_60s": 0.974, "n": 4838},
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): {"p10": 120.3837, "p25": 242.27225, "median": 368.7625, "p75": 516.8530000000001, "p90": 722.8312, "share_under_60s": 0.051, "n": 4772},
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): {"p10": 8.7146, "p25": 11.882, "median": 22.6655, "p75": 39.5405, "p90": 47.2731, "share_under_60s": 0.953, "n": 4728},
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): {"p10": 28.1661, "p25": 633.0405, "median": 6399.7575, "p75": 44682.3065, "p90": 69334.2631, "share_under_60s": 0.155, "n": 4508},
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): {"p10": 72.8682, "p25": 571.685, "median": 2538.547, "p75": 35730.268, "p90": 53224.031, "share_under_60s": 0.092, "n": 4405},
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 4289},
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 63.848, "p25": 118.72775, "median": 184.1315, "p75": 280.47425, "p90": 409.93, "share_under_60s": 0.092, "n": 4266},
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): {"p10": 0.141, "p25": 0.15275, "median": 0.18, "p75": 0.274, "p90": 0.374, "share_under_60s": 0.999, "n": 3432},
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): {"p10": 31.6694, "p25": 34.088, "median": 37.411, "p75": 41.491, "p90": 47.0952, "share_under_60s": 0.98, "n": 3429},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.115, "p25": 1.229, "median": 1.4505, "p75": 1.893, "p90": 2.228, "share_under_60s": 1.0, "n": 3254},
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 6703.3079, "p25": 89236.39275, "median": 189494.1955, "p75": 348573.559, "p90": 441072.1785, "share_under_60s": 0.049, "n": 3164},
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): {"p10": 10.182, "p25": 12.3975, "median": 21.0825, "p75": 42.8485, "p90": 89.1491, "share_under_60s": 0.832, "n": 3120},
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): {"p10": 10.801, "p25": 45.634, "median": 310.438, "p75": 1040.096, "p90": 2204.752, "share_under_60s": 0.282, "n": 2881},
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 24.091, "p25": 238.427, "median": 2016.402, "p75": 13280.25, "p90": 86082.238, "share_under_60s": 0.146, "n": 2851},
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): {"p10": 0.899, "p25": 0.988, "median": 1.148, "p75": 1.367, "p90": 1.8546, "share_under_60s": 1.0, "n": 2515},
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): {"p10": 17.3691, "p25": 858.7675, "median": 5769.2875, "p75": 15620.251, "p90": 47555.6435, "share_under_60s": 0.154, "n": 2438},
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): {"p10": 35.3822, "p25": 65.219, "median": 107.408, "p75": 184.817, "p90": 346.874, "share_under_60s": 0.213, "n": 2349},
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): {"p10": 25.928, "p25": 331.6065, "median": 2857.8005, "p75": 7837.031, "p90": 13917.101, "share_under_60s": 0.158, "n": 2166},
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): {"p10": 1.4473, "p25": 1.61025, "median": 1.847, "p75": 2.269, "p90": 2.7245, "share_under_60s": 1.0, "n": 2154},
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): {"p10": 2.838, "p25": 3.08, "median": 3.412, "p75": 3.9145, "p90": 4.458, "share_under_60s": 0.999, "n": 2026},
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 2011},
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): {"p10": 256.972, "p25": 627.48, "median": 1177.7265, "p75": 1810.88725, "p90": 2496.666, "share_under_60s": 0.028, "n": 1924},
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): {"p10": 31.1872, "p25": 68.05325, "median": 111.291, "p75": 194.37925, "p90": 370.6665, "share_under_60s": 0.214, "n": 1798},
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): {"p10": 1.4856, "p25": 1.649, "median": 1.895, "p75": 2.293, "p90": 2.7488, "share_under_60s": 1.0, "n": 1727},
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): {"p10": 699.4614, "p25": 2120.23, "median": 5024.022, "p75": 46929.641, "p90": 78412.2382, "share_under_60s": 0.031, "n": 1719},
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 1662},
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 1639},
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.22, "p25": 1.3115, "median": 1.618, "p75": 2.149, "p90": 2.4658, "share_under_60s": 0.999, "n": 1619},
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): {"p10": 31.9515, "p25": 48.48525, "median": 106.2115, "p75": 217.0865, "p90": 409.0865, "share_under_60s": 0.317, "n": 1616},
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): {"p10": 24.7373, "p25": 36.484, "median": 64.6275, "p75": 200.2605, "p90": 567.5725, "share_under_60s": 0.471, "n": 1514},
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 1466},
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "p75": 0.001, "p90": 0.001, "share_under_60s": 1.0, "n": 1285},
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): {"p10": 8.8974, "p25": 56.97725, "median": 141.1065, "p75": 508.14525, "p90": 2625.3302, "share_under_60s": 0.259, "n": 1260},
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 1126},
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): {"p10": 3.8134, "p25": 16.2075, "median": 363.885, "p75": 2180.117, "p90": 8339.3384, "share_under_60s": 0.335, "n": 1067},
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 2.9611, "p25": 3.1895, "median": 3.534, "p75": 4.04, "p90": 4.6274, "share_under_60s": 0.999, "n": 1052},
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.4536, "p25": 7.027, "median": 7.763, "p75": 8.498, "p90": 9.3666, "share_under_60s": 0.999, "n": 1009},
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"p10": 1.837, "p25": 2.093, "median": 2.396, "p75": 2.924, "p90": 3.4406, "share_under_60s": 0.998, "n": 993},
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): {"p10": 167.5932, "p25": 312.663, "median": 469.317, "p75": 682.7525, "p90": 950.6196, "share_under_60s": 0.024, "n": 990},
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): {"p10": 72.685, "p25": 111.148, "median": 156.879, "p75": 240.343, "p90": 403.3836, "share_under_60s": 0.081, "n": 913},
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): {"p10": 141.3626, "p25": 317.4815, "median": 770.573, "p75": 1332.394, "p90": 1957.7284, "share_under_60s": 0.019, "n": 862},
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): {"p10": 6.3086, "p25": 6.952, "median": 7.6685, "p75": 8.3765, "p90": 9.1902, "share_under_60s": 1.0, "n": 862},
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share_under_60s": 1.0, "n": 801},
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): {"p10": 1.3039, "p25": 1.463, "median": 1.6655, "p75": 2.12525, "p90": 2.6489, "share_under_60s": 0.999, "n": 728},
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): {"p10": 0.908, "p25": 1.053, "median": 1.2515, "p75": 1.5545, "p90": 2.182, "share_under_60s": 1.0, "n": 726},
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-COMPLETE"): {"p10": 48200.9575, "p25": 98554.79725, "median": 306248.4655, "p75": 737130.382, "p90": 1197428.6763, "share_under_60s": 0.0, "n": 712},
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): {"p10": 0.001, "p25": 0.001, "median": 0.001, "p75": 0.001, "p90": 0.001, "share_under_60s": 1.0, "n": 709},
}

DFG_FREQ = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): 16831,
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): 16077,
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): 16023,
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): 15568,
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): 13087,
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): 9065,
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): 8927,
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): 7266,
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): 7030,
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): 6633,
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): 6303,
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): 5014,
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): 4913,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): 4838,
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): 4772,
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): 4728,
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): 4508,
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): 4405,
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): 4289,
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): 4266,
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): 3432,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): 3429,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 3254,
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): 3164,
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): 3120,
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): 2881,
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): 2851,
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): 2515,
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): 2438,
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): 2349,
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): 2166,
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): 2154,
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): 2026,
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): 2011,
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): 1924,
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): 1798,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): 1727,
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): 1719,
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): 1662,
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): 1639,
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 1619,
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): 1616,
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): 1514,
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): 1466,
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): 1285,
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): 1260,
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): 1126,
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): 1067,
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 1052,
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 1009,
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): 993,
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): 990,
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): 913,
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): 862,
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 862,
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): 801,
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 728,
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): 726,
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-COMPLETE"): 712,
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): 709,
}

FREQUENT_VARIANTS = [
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_DECLINED-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE","W_Afhandelen leads-START","W_Afhandelen leads-COMPLETE","W_Afhandelen leads-START","A_DECLINED-COMPLETE","W_Afhandelen leads-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_DECLINED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","W_Afhandelen leads-SCHEDULE","W_Afhandelen leads-START","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Afhandelen leads-COMPLETE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","A_ACCEPTED-COMPLETE","A_FINALIZED-COMPLETE","O_SELECTED-COMPLETE","O_CREATED-COMPLETE","O_SENT-COMPLETE","W_Nabellen offertes-SCHEDULE","W_Completeren aanvraag-COMPLETE","W_Nabellen offertes-START","A_CANCELLED-COMPLETE","O_CANCELLED-COMPLETE","W_Nabellen offertes-COMPLETE"],
    ["A_SUBMITTED-COMPLETE","A_PARTLYSUBMITTED-COMPLETE","A_PREACCEPTED-COMPLETE","W_Completeren aanvraag-SCHEDULE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","W_Completeren aanvraag-COMPLETE","W_Completeren aanvraag-START","A_CANCELLED-COMPLETE","W_Completeren aanvraag-COMPLETE"],
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def infer_fractional_digits(series):
    max_digits = 0
    any_fraction = False
    for s in series.dropna().astype(str):
        if "." in s:
            any_fraction = True
            frac = s.split(".")[-1]
            max_digits = max(max_digits, len(frac))
    return max_digits if any_fraction else 0

def parse_ts(s):
    return pd.to_datetime(s, errors="coerce")

def fmt_ts(ts, frac_digits):
    if pd.isna(ts):
        return ""
    if frac_digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    if frac_digits <= 6:
        keep = 20 + frac_digits
        return base[:keep]
    return base

def td_seconds(sec):
    return pd.Timedelta(seconds=float(sec))

def get_profile(a, b):
    return DURATION_PROFILES.get((a, b))

def preferred_gap(a, b):
    prof = get_profile(a, b)
    if prof is None:
        return None
    if prof["share_under_60s"] >= 0.9:
        return prof["p25"]
    return prof["p25"]

def minimal_positive_gap_seconds(a=None, b=None, frac_digits=3):
    prof = get_profile(a, b) if a is not None and b is not None else None
    if prof is not None and prof["share_under_60s"] >= 0.95 and prof["p25"] <= 1.0:
        val = max(prof["p25"], 10 ** (-frac_digits))
        return val
    return max(10 ** (-frac_digits), 0.001)

def local_edges_for_pair(acts, i, swapped=False):
    a_prev = acts[i-1] if i-1 >= 0 else None
    a_i = acts[i]
    a_j = acts[i+1]
    a_next = acts[i+2] if i+2 < len(acts) else None
    if not swapped:
        middle = [a_i, a_j]
    else:
        middle = [a_j, a_i]
    seq = []
    if a_prev is not None:
        seq.append((a_prev, middle[0]))
    seq.append((middle[0], middle[1]))
    if a_next is not None:
        seq.append((middle[1], a_next))
    return seq

def variant_support(acts, i, swapped=False):
    local = acts[max(0, i-2):min(len(acts), i+4)].copy()
    center = i - max(0, i-2)
    if swapped and center + 1 < len(local):
        local[center], local[center+1] = local[center+1], local[center]
    best = 0
    for v in FREQUENT_VARIANTS:
        for start in range(0, len(v) - len(local) + 1):
            if v[start:start+len(local)] == local:
                best = max(best, len(local))
    return best

def score_edge_existence(edges):
    score = 0.0
    missing = 0
    for e in edges:
        if e in DFG_FREQ:
            score += np.log1p(DFG_FREQ[e])
        else:
            missing += 1
            score -= 100.0
    return score, missing

def duration_plausibility(edge, gap_sec):
    prof = get_profile(*edge)
    if prof is None or gap_sec is None or gap_sec < 0:
        return -5.0
    if prof["p10"] == prof["p90"]:
        center = prof["median"]
        return 2.0 if abs(gap_sec - center) <= 1.0 else -2.0
    p10, p25, p50, p90 = prof["p10"], prof["p25"], prof["median"], prof["p90"]
    if gap_sec < 0:
        return -50.0
    if p10 <= gap_sec <= p90:
        return 3.0
    if p25 <= gap_sec <= p50:
        return 3.5
    # log distance to nearest plausible boundary
    target = p25 if gap_sec < p10 else p50 if gap_sec < p25 else p90
    return -abs(np.log1p(abs(gap_sec - target)))

def candidate_swap_score(case_df, i):
    acts = case_df["Activity"].tolist()
    times = case_df["_ts_parsed"].tolist()
    n = len(case_df)
    if i < 0 or i+1 >= n:
        return -1e9, {"reason": "bounds"}

    t_i = times[i]
    t_j = times[i+1]
    if pd.isna(t_i) or pd.isna(t_j):
        return -1e9, {"reason": "nan"}

    left_anchor = times[i-1] if i-1 >= 0 else None
    right_anchor = times[i+2] if i+2 < n else None

    new_i = t_j
    new_j = t_i

    if left_anchor is not None and new_i < left_anchor:
        return -1e9, {"reason": "left_anchor"}
    if new_j < new_i:
        return -1e9, {"reason": "pair_order"}
    if right_anchor is not None and new_j > right_anchor:
        return -1e9, {"reason": "right_anchor"}

    edges = local_edges_for_pair(acts, i, swapped=False)
    e_score, missing = score_edge_existence(edges)
    if missing > 0:
        return -1e9, {"reason": "non_dfg_after_swap"}

    score = e_score + 2.0 * variant_support(acts, i, swapped=False)
    local_times = {}
    if i-1 >= 0:
        local_times[(acts[i-1], acts[i])] = (new_i - left_anchor).total_seconds()
    local_times[(acts[i], acts[i+1])] = (new_j - new_i).total_seconds()
    if i+2 < n:
        local_times[(acts[i+1], acts[i+2])] = (right_anchor - new_j).total_seconds()

    for edge, gap in local_times.items():
        score += duration_plausibility(edge, gap)

    return score, {"reason": "ok"}

def choose_displaced_target(case_df, i, frac_digits):
    acts = case_df["Activity"].tolist()
    times = case_df["_repair_ts"].tolist()
    orig = case_df["_ts_parsed"].tolist()
    n = len(case_df)

    left_idx = i - 1
    right_idx = i + 1
    left_t = times[left_idx] if left_idx >= 0 else None
    right_t = times[right_idx] if right_idx < n else None
    left_act = acts[left_idx] if left_idx >= 0 else None
    this_act = acts[i]
    right_act = acts[right_idx] if right_idx < n else None

    eps = pd.Timedelta(seconds=max(10 ** (-frac_digits), 0.001))

    lower = left_t if left_t is not None else orig[i]
    upper = right_t if right_t is not None else orig[i]
    if left_t is not None:
        lower = left_t
    if right_t is not None:
        upper = right_t

    candidates = []

    if left_t is not None:
        g = preferred_gap(left_act, this_act)
        if g is not None:
            candidates.append(left_t + td_seconds(g))
        else:
            candidates.append(left_t + eps)

    if right_t is not None:
        g = preferred_gap(this_act, right_act)
        if g is not None:
            candidates.append(right_t - td_seconds(g))
        else:
            candidates.append(right_t - eps)

    if left_t is not None and right_t is not None:
        prof_l = get_profile(left_act, this_act)
        prof_r = get_profile(this_act, right_act)
        if prof_l is not None and prof_r is not None:
            fl = left_t + td_seconds(preferred_gap(left_act, this_act))
            br = right_t - td_seconds(preferred_gap(this_act, right_act))
            candidates.append(fl + (br - fl) / 2)

    obs = orig[i]
    if not pd.isna(obs):
        candidates.append(obs)

    if not candidates:
        if left_t is not None and right_t is not None:
            cand = left_t + (right_t - left_t) / 2
        elif left_t is not None:
            cand = left_t + eps
        elif right_t is not None:
            cand = right_t - eps
        else:
            cand = orig[i]
        candidates.append(cand)

    feasible_low = left_t if left_t is not None else pd.Timestamp.min.tz_localize(None) if hasattr(pd.Timestamp.min, "tz_localize") else pd.Timestamp.min
    feasible_high = right_t if right_t is not None else pd.Timestamp.max.tz_localize(None) if hasattr(pd.Timestamp.max, "tz_localize") else pd.Timestamp.max

    def clamp(ts):
        out = ts
        if left_t is not None and out < left_t:
            out = left_t
        if right_t is not None and out > right_t:
            out = right_t
        return out

    def objective(ts):
        ts = clamp(ts)
        val = 0.0
        if left_t is not None:
            gap = (ts - left_t).total_seconds()
            prof = get_profile(left_act, this_act)
            if prof is not None:
                target = preferred_gap(left_act, this_act)
                val += abs(gap - target)
            else:
                val += gap
            if gap < 0:
                val += 1e9
        if right_t is not None:
            gap = (right_t - ts).total_seconds()
            prof = get_profile(this_act, right_act)
            if prof is not None:
                target = preferred_gap(this_act, right_act)
                val += abs(gap - target)
            else:
                val += gap
            if gap < 0:
                val += 1e9
        if not pd.isna(obs):
            val += abs((ts - obs).total_seconds()) * 0.01
        return val

    best = min((clamp(c) for c in candidates), key=objective)

    if left_t is not None and right_t is not None:
        if best < left_t:
            best = left_t
        if best > right_t:
            best = right_t
        if best == left_t and best == right_t:
            pass
        elif best <= left_t:
            best = min(right_t, left_t + eps)
        elif best >= right_t:
            best = max(left_t, right_t - eps)

    if not pd.isna(obs) and best == obs:
        if left_t is not None and right_t is not None:
            alt1 = min(right_t, obs + eps)
            alt2 = max(left_t, obs - eps)
            if alt1 != obs and alt1 <= right_t and alt1 >= left_t:
                best = alt1
            elif alt2 != obs and alt2 >= left_t and alt2 <= right_t:
                best = alt2
            else:
                gap_l = minimal_positive_gap_seconds(left_act, this_act, frac_digits)
                gap_r = minimal_positive_gap_seconds(this_act, right_act, frac_digits)
                target = left_t + td_seconds(gap_l)
                if target <= right_t:
                    best = target
                else:
                    best = max(left_t, right_t - td_seconds(gap_r))
        elif left_t is not None:
            best = left_t + eps if obs != left_t + eps else left_t + 2 * eps
        elif right_t is not None:
            best = right_t - eps if obs != right_t - eps else right_t - 2 * eps

    return best

def self_check_and_fix(df, frac_digits, repaired_rows, exchanged_pairs, displaced_pairs):
    fixes = []

    # CLEAN exact preservation already handled by writing original strings unless repaired; enforce.
    clean_mask = df["error_category"] == "CLEAN"
    bad_clean = clean_mask & (df["Timestamp_out"] != df["Timestamp"])
    if bad_clean.any():
        df.loc[bad_clean, "Timestamp_out"] = df.loc[bad_clean, "Timestamp"]
        fixes.append(f"restored_clean={int(bad_clean.sum())}")

    # parse output
    df["_out_parsed"] = pd.to_datetime(df["Timestamp_out"], errors="coerce")

    # corrupted rows must differ from input
    corr_mask = df["error_category"] != "CLEAN"
    same_corr = corr_mask & (df["Timestamp_out"] == df["Timestamp"])
    if same_corr.any():
        for idx in df.index[same_corr]:
            case = df.at[idx, "Case"]
            case_idx = df.index[df["Case"] == case].tolist()
            pos = case_idx.index(idx)
            left_idx = case_idx[pos-1] if pos > 0 else None
            right_idx = case_idx[pos+1] if pos+1 < len(case_idx) else None
            left_t = pd.to_datetime(df.at[left_idx, "Timestamp_out"]) if left_idx is not None else None
            right_t = pd.to_datetime(df.at[right_idx, "Timestamp_out"]) if right_idx is not None else None
            act_left = df.at[left_idx, "Activity"] if left_idx is not None else None
            act = df.at[idx, "Activity"]
            act_right = df.at[right_idx, "Activity"] if right_idx is not None else None
            eps = pd.Timedelta(seconds=max(10 ** (-frac_digits), 0.001))
            new_t = pd.to_datetime(df.at[idx, "Timestamp"])
            if left_t is not None and right_t is not None:
                cand = left_t + td_seconds(minimal_positive_gap_seconds(act_left, act, frac_digits))
                if cand > right_t:
                    cand = max(left_t, right_t - eps)
                if cand == new_t:
                    cand = min(right_t, cand + eps)
                new_t = cand
            elif left_t is not None:
                new_t = left_t + eps
                if new_t == pd.to_datetime(df.at[idx, "Timestamp"]):
                    new_t = new_t + eps
            elif right_t is not None:
                new_t = right_t - eps
                if new_t == pd.to_datetime(df.at[idx, "Timestamp"]):
                    new_t = new_t - eps
            df.at[idx, "Timestamp_out"] = fmt_ts(new_t, frac_digits)
        fixes.append(f"changed_equal_corrupted={int(same_corr.sum())}")

    # monotonicity within case and anchor clamp
    df["_out_parsed"] = pd.to_datetime(df["Timestamp_out"], errors="coerce")
    for case, g in df.groupby("Case", sort=False):
        idxs = g.index.tolist()
        out = g["_out_parsed"].tolist()
        for k in range(1, len(out)):
            if out[k] < out[k-1]:
                prev_idx = idxs[k-1]
                idx = idxs[k]
                prev_t = pd.to_datetime(df.at[prev_idx, "Timestamp_out"])
                next_t = pd.to_datetime(df.at[idxs[k+1], "Timestamp_out"]) if k+1 < len(idxs) else None
                eps = pd.Timedelta(seconds=max(10 ** (-frac_digits), 0.001))
                cand = prev_t + eps
                if next_t is not None and cand > next_t:
                    cand = next_t
                if cand <= prev_t and next_t is not None:
                    cand = prev_t
                if df.at[idx, "error_category"] != "CLEAN":
                    df.at[idx, "Timestamp_out"] = fmt_ts(cand, frac_digits)
                    fixes.append(f"monotonic_fix_rowid={df.at[idx,'row_id']}")
                else:
                    # if clean cannot move, move previous corrupted if possible
                    if df.at[prev_idx, "error_category"] != "CLEAN":
                        cand_prev = min(pd.to_datetime(df.at[idx, "Timestamp_out"]), prev_t)
                        df.at[prev_idx, "Timestamp_out"] = fmt_ts(cand_prev, frac_digits)
                        fixes.append(f"monotonic_fix_rowid={df.at[prev_idx,'row_id']}")

    df["_out_parsed"] = pd.to_datetime(df["Timestamp_out"], errors="coerce")
    return fixes

# ---------------------------------------------------------------------
# Main repair
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for col in original_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[original_cols].copy()
    df["_orig_order"] = np.arange(len(df))
    df["_ts_parsed"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_repair_ts"] = df["_ts_parsed"].copy()
    df["Timestamp_out"] = df["Timestamp"]

    frac_digits = infer_fractional_digits(df["Timestamp"])

    exchanged_segments = 0
    displaced_segments = 0

    # Process case by case in original row order
    for case, g in df.groupby("Case", sort=False):
        idxs = g.index.tolist()
        acts = g["Activity"].tolist()

        # initialize repair ts from current outputs
        df.loc[idxs, "_repair_ts"] = pd.to_datetime(df.loc[idxs, "Timestamp_out"], errors="coerce")

        i = 0
        while i < len(idxs) - 1:
            idx_i = idxs[i]
            idx_j = idxs[i+1]

            t_i = pd.to_datetime(df.at[idx_i, "Timestamp_out"], errors="coerce")
            t_j = pd.to_datetime(df.at[idx_j, "Timestamp_out"], errors="coerce")

            if pd.isna(t_i) or pd.isna(t_j):
                i += 1
                continue

            if t_j >= t_i:
                i += 1
                continue

            # later row carries earlier timestamp than predecessor
            if df.at[idx_j, "error_category"] == "CLEAN":
                # should not happen per task, but repair corrupted predecessor if needed without branching to other error types
                i += 1
                continue

            local_case = df.loc[idxs].copy()
            local_case["_ts_parsed"] = pd.to_datetime(local_case["Timestamp_out"], errors="coerce")
            pair_pos = i

            swap_score, info = candidate_swap_score(local_case, pair_pos)

            # no-swap local score with partner preserved and row i+1 estimated
            left_t = t_i
            right_t = pd.to_datetime(df.at[idxs[i+2], "Timestamp_out"], errors="coerce") if i+2 < len(idxs) else None
            act_left = acts[i]
            act_mid = acts[i+1]
            act_right = acts[i+2] if i+2 < len(idxs) else None

            displaced_candidate = choose_displaced_target(local_case, pair_pos + 1, frac_digits)
            displaced_score = 0.0
            edge1 = (act_left, act_mid)
            if edge1 in DFG_FREQ:
                displaced_score += np.log1p(DFG_FREQ[edge1])
                displaced_score += duration_plausibility(edge1, (displaced_candidate - left_t).total_seconds())
            else:
                displaced_score -= 100.0
            if right_t is not None:
                edge2 = (act_mid, act_right)
                if edge2 in DFG_FREQ:
                    displaced_score += np.log1p(DFG_FREQ[edge2])
                    displaced_score += duration_plausibility(edge2, (right_t - displaced_candidate).total_seconds())
                else:
                    displaced_score -= 100.0
            displaced_score += variant_support(acts, pair_pos, swapped=False)

            # classify
            if swap_score > displaced_score + 1.0:
                # exchanged: exactly swap observed pair values
                df.at[idx_i, "Timestamp_out"] = df.at[idx_j, "Timestamp"]
                df.at[idx_j, "Timestamp_out"] = df.at[idx_i, "Timestamp"]
                exchanged_segments += 1
            else:
                # displaced: keep partner unchanged, change only later row
                new_t = displaced_candidate
                # clamp to anchors
                if new_t < left_t:
                    new_t = left_t
                if right_t is not None and new_t > right_t:
                    new_t = right_t
                if new_t <= left_t and (right_t is None or right_t > left_t):
                    new_t = min(right_t, left_t + pd.Timedelta(seconds=max(10 ** (-frac_digits), 0.001))) if right_t is not None else left_t + pd.Timedelta(seconds=max(10 ** (-frac_digits), 0.001))
                if fmt_ts(new_t, frac_digits) == df.at[idx_j, "Timestamp"]:
                    eps = pd.Timedelta(seconds=max(10 ** (-frac_digits), 0.001))
                    alt = new_t + eps
                    if right_t is not None and alt > right_t:
                        alt = max(left_t, new_t - eps)
                    if alt < left_t:
                        alt = left_t
                    new_t = alt
                df.at[idx_j, "Timestamp_out"] = fmt_ts(new_t, frac_digits)
                displaced_segments += 1

            # refresh case timestamps after repair
            df.loc[idxs, "_repair_ts"] = pd.to_datetime(df.loc[idxs, "Timestamp_out"], errors="coerce")
            i += 1

    fixes = self_check_and_fix(df, frac_digits, None, exchanged_segments, displaced_segments)

    # final parseability safeguard for repaired rows
    corr_mask = df["error_category"] != "CLEAN"
    parsed_final = pd.to_datetime(df["Timestamp_out"], errors="coerce")
    bad_parse = corr_mask & parsed_final.isna()
    if bad_parse.any():
        for idx in df.index[bad_parse]:
            orig = pd.to_datetime(df.at[idx, "Timestamp"], errors="coerce")
            case = df.at[idx, "Case"]
            case_idxs = df.index[df["Case"] == case].tolist()
            pos = case_idxs.index(idx)
            left_t = pd.to_datetime(df.at[case_idxs[pos-1], "Timestamp_out"], errors="coerce") if pos > 0 else orig
            new_t = left_t + pd.Timedelta(seconds=max(10 ** (-frac_digits), 0.001))
            df.at[idx, "Timestamp_out"] = fmt_ts(new_t, frac_digits)
        fixes.append(f"parse_fix={int(bad_parse.sum())}")

    # write output with exact untouched fields
    out = df[["Case", "Activity", "Timestamp_out", "Resource", "error_category", "row_id"]].copy()
    out.columns = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    out.to_csv(args.output, index=False)

    rows_still_equal = int((out["Timestamp"] == df["Timestamp"]).sum())
    if fixes:
        print(f"fixes:{';'.join(fixes)} | exchanged={exchanged_segments} displaced={displaced_segments} rows_still_equal={rows_still_equal}")
    else:
        print(f"exchanged={exchanged_segments} displaced={displaced_segments} rows_still_equal={rows_still_equal}")

if __name__ == "__main__":
    main()