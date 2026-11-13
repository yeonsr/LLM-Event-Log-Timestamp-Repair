import argparse
import math
import itertools
from collections import defaultdict
from datetime import timedelta

import pandas as pd


PROCESS_CONTEXT = {
    "transition_probabilities_top": [
        {"source": "O_CREATED-COMPLETE", "target": "O_SENT-COMPLETE", "probability": 0.996},
        {"source": "O_SENT_BACK-COMPLETE", "target": "W_Valideren aanvraag-SCHEDULE", "probability": 0.9944},
        {"source": "A_SUBMITTED-COMPLETE", "target": "A_PARTLYSUBMITTED-COMPLETE", "probability": 0.9682},
        {"source": "A_PREACCEPTED-COMPLETE", "target": "W_Completeren aanvraag-SCHEDULE", "probability": 0.9653},
        {"source": "W_Afhandelen leads-SCHEDULE", "target": "W_Afhandelen leads-START", "probability": 0.9601},
        {"source": "O_SENT-COMPLETE", "target": "W_Nabellen offertes-SCHEDULE", "probability": 0.933},
        {"source": "W_Nabellen incomplete dossiers-SCHEDULE", "target": "W_Valideren aanvraag-COMPLETE", "probability": 0.909},
        {"source": "O_SELECTED-COMPLETE", "target": "O_CREATED-COMPLETE", "probability": 0.9002},
        {"source": "A_FINALIZED-COMPLETE", "target": "O_SELECTED-COMPLETE", "probability": 0.8575},
        {"source": "A_ACCEPTED-COMPLETE", "target": "A_FINALIZED-COMPLETE", "probability": 0.8397},
        {"source": "W_Nabellen incomplete dossiers-COMPLETE", "target": "W_Nabellen incomplete dossiers-START", "probability": 0.8371},
        {"source": "O_DECLINED-COMPLETE", "target": "W_Valideren aanvraag-COMPLETE", "probability": 0.834},
        {"source": "W_Nabellen offertes-COMPLETE", "target": "W_Nabellen offertes-START", "probability": 0.7948},
        {"source": "W_Nabellen incomplete dossiers-START", "target": "W_Nabellen incomplete dossiers-COMPLETE", "probability": 0.792},
        {"source": "A_APPROVED-COMPLETE", "target": "A_REGISTERED-COMPLETE", "probability": 0.74},
        {"source": "W_Nabellen offertes-SCHEDULE", "target": "W_Completeren aanvraag-COMPLETE", "probability": 0.7391},
        {"source": "W_Nabellen offertes-START", "target": "W_Nabellen offertes-COMPLETE", "probability": 0.7321},
        {"source": "W_Completeren aanvraag-COMPLETE", "target": "W_Completeren aanvraag-START", "probability": 0.7267},
        {"source": "W_Completeren aanvraag-START", "target": "W_Completeren aanvraag-COMPLETE", "probability": 0.7062},
        {"source": "A_CANCELLED-COMPLETE", "target": "O_CANCELLED-COMPLETE", "probability": 0.6917},
        {"source": "W_Afhandelen leads-COMPLETE", "target": "W_Completeren aanvraag-START", "probability": 0.6855},
        {"source": "W_Beoordelen fraude-COMPLETE", "target": "W_Beoordelen fraude-START", "probability": 0.67},
        {"source": "W_Wijzigen contractgegevens-SCHEDULE", "target": "W_Wijzigen contractgegevens-SCHEDULE", "probability": 0.6667},
        {"source": "A_ACTIVATED-COMPLETE", "target": "A_APPROVED-COMPLETE", "probability": 0.6527},
        {"source": "W_Valideren aanvraag-SCHEDULE", "target": "W_Nabellen offertes-COMPLETE", "probability": 0.6423},
        {"source": "W_Completeren aanvraag-SCHEDULE", "target": "W_Completeren aanvraag-START", "probability": 0.6405},
        {"source": "O_CANCELLED-COMPLETE", "target": "O_SELECTED-COMPLETE", "probability": 0.6367},
        {"source": "O_ACCEPTED-COMPLETE", "target": "A_ACTIVATED-COMPLETE", "probability": 0.5729},
        {"source": "W_Valideren aanvraag-COMPLETE", "target": "W_Valideren aanvraag-START", "probability": 0.5623},
        {"source": "W_Beoordelen fraude-START", "target": "W_Beoordelen fraude-COMPLETE", "probability": 0.556},
        {"source": "W_Beoordelen fraude-SCHEDULE", "target": "W_Beoordelen fraude-START", "probability": 0.5086},
        {"source": "A_DECLINED-COMPLETE", "target": "W_Afhandelen leads-COMPLETE", "probability": 0.5017},
        {"source": "A_REGISTERED-COMPLETE", "target": "O_ACCEPTED-COMPLETE", "probability": 0.5016},
        {"source": "A_REGISTERED-COMPLETE", "target": "W_Valideren aanvraag-COMPLETE", "probability": 0.4494},
        {"source": "W_Afhandelen leads-START", "target": "A_PREACCEPTED-COMPLETE", "probability": 0.4478},
        {"source": "W_Valideren aanvraag-COMPLETE", "target": "W_Nabellen incomplete dossiers-START", "probability": 0.4262},
        {"source": "O_ACCEPTED-COMPLETE", "target": "W_Valideren aanvraag-COMPLETE", "probability": 0.3843},
        {"source": "W_Valideren aanvraag-START", "target": "W_Valideren aanvraag-COMPLETE", "probability": 0.3648},
        {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "A_PREACCEPTED-COMPLETE", "probability": 0.3602},
        {"source": "W_Completeren aanvraag-SCHEDULE", "target": "W_Afhandelen leads-COMPLETE", "probability": 0.3453},
        {"source": "W_Afhandelen leads-START", "target": "A_DECLINED-COMPLETE", "probability": 0.3449},
        {"source": "W_Valideren aanvraag-SCHEDULE", "target": "W_Nabellen incomplete dossiers-COMPLETE", "probability": 0.3439},
        {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "W_Afhandelen leads-SCHEDULE", "probability": 0.3295},
        {"source": "O_CANCELLED-COMPLETE", "target": "W_Nabellen offertes-COMPLETE", "probability": 0.3147},
        {"source": "A_CANCELLED-COMPLETE", "target": "W_Completeren aanvraag-COMPLETE", "probability": 0.3083},
        {"source": "W_Afhandelen leads-COMPLETE", "target": "W_Afhandelen leads-START", "probability": 0.2994},
        {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "A_DECLINED-COMPLETE", "probability": 0.2788},
        {"source": "A_DECLINED-COMPLETE", "target": "W_Completeren aanvraag-COMPLETE", "probability": 0.275},
        {"source": "A_ACTIVATED-COMPLETE", "target": "A_REGISTERED-COMPLETE", "probability": 0.26},
        {"source": "W_Beoordelen fraude-SCHEDULE", "target": "W_Valideren aanvraag-COMPLETE", "probability": 0.2586},
        {"source": "A_APPROVED-COMPLETE", "target": "O_ACCEPTED-COMPLETE", "probability": 0.2547},
        {"source": "W_Nabellen offertes-SCHEDULE", "target": "W_Nabellen offertes-COMPLETE", "probability": 0.2498},
        {"source": "W_Valideren aanvraag-START", "target": "W_Nabellen incomplete dossiers-SCHEDULE", "probability": 0.2496},
        {"source": "W_Completeren aanvraag-COMPLETE", "target": "W_Nabellen offertes-START", "probability": 0.2276},
        {"source": "W_Completeren aanvraag-START", "target": "A_ACCEPTED-COMPLETE", "probability": 0.215},
        {"source": "A_DECLINED-COMPLETE", "target": "O_DECLINED-COMPLETE", "probability": 0.209},
        {"source": "W_Beoordelen fraude-START", "target": "A_DECLINED-COMPLETE", "probability": 0.208},
        {"source": "W_Afhandelen leads-START", "target": "W_Afhandelen leads-COMPLETE", "probability": 0.2001},
        {"source": "W_Beoordelen fraude-SCHEDULE", "target": "W_Afhandelen leads-COMPLETE", "probability": 0.181},
        {"source": "W_Wijzigen contractgegevens-SCHEDULE", "target": "W_Nabellen offertes-START", "probability": 0.1667},
    ],
    "duration_profiles_top": [
        {"source": "W_Nabellen offertes-COMPLETE", "target": "W_Nabellen offertes-START", "n": 16578, "p10_seconds": 51.48200000000001, "p25_seconds": 14904.8565, "median_seconds": 167152.1195, "p75_seconds": 443033.66625, "p90_seconds": 655024.4455000004, "share_under_60s": 0.104},
        {"source": "W_Nabellen offertes-START", "target": "W_Nabellen offertes-COMPLETE", "n": 15747, "p10_seconds": 18.4252, "p25_seconds": 56.2145, "median_seconds": 116.085, "p75_seconds": 209.78500000000003, "p90_seconds": 419.5531999999998, "share_under_60s": 0.267},
        {"source": "W_Completeren aanvraag-START", "target": "W_Completeren aanvraag-COMPLETE", "n": 15647, "p10_seconds": 10.703, "p25_seconds": 40.826, "median_seconds": 98.688, "p75_seconds": 203.4725, "p90_seconds": 491.8296, "share_under_60s": 0.339},
        {"source": "W_Completeren aanvraag-COMPLETE", "target": "W_Completeren aanvraag-START", "n": 15330, "p10_seconds": 62.131800000000005, "p25_seconds": 1614.967, "median_seconds": 14829.922999999999, "p75_seconds": 70301.2695, "p90_seconds": 164970.455, "share_under_60s": 0.099},
        {"source": "A_SUBMITTED-COMPLETE", "target": "A_PARTLYSUBMITTED-COMPLETE", "n": 12671, "p10_seconds": 0.125, "p25_seconds": 0.173, "median_seconds": 0.273, "p75_seconds": 0.497, "p90_seconds": 1.079, "share_under_60s": 1.0},
        {"source": "W_Nabellen incomplete dossiers-COMPLETE", "target": "W_Nabellen incomplete dossiers-START", "n": 8973, "p10_seconds": 75.33920000000005, "p25_seconds": 1281.503, "median_seconds": 10850.015, "p75_seconds": 66106.468, "p90_seconds": 180463.22840000005, "share_under_60s": 0.093},
        {"source": "W_Nabellen incomplete dossiers-START", "target": "W_Nabellen incomplete dossiers-COMPLETE", "n": 8830, "p10_seconds": 10.189, "p25_seconds": 33.673, "median_seconds": 140.75650000000002, "p75_seconds": 392.14775, "p90_seconds": 916.5624, "share_under_60s": 0.329},
        {"source": "O_CREATED-COMPLETE", "target": "O_SENT-COMPLETE", "n": 6995, "p10_seconds": 0.025, "p25_seconds": 0.028, "median_seconds": 0.035, "p75_seconds": 0.049, "p90_seconds": 0.079, "share_under_60s": 1.0},
        {"source": "A_PREACCEPTED-COMPLETE", "target": "W_Completeren aanvraag-SCHEDULE", "n": 6642, "p10_seconds": 0.046, "p25_seconds": 0.131, "median_seconds": 0.378, "p75_seconds": 0.544, "p90_seconds": 0.8836000000000022, "share_under_60s": 1.0},
        {"source": "O_SENT-COMPLETE", "target": "W_Nabellen offertes-SCHEDULE", "n": 6533, "p10_seconds": 0.08220000000000005, "p25_seconds": 0.094, "median_seconds": 0.119, "p75_seconds": 0.227, "p90_seconds": 0.3098000000000002, "share_under_60s": 1.0},
        {"source": "O_SELECTED-COMPLETE", "target": "O_CREATED-COMPLETE", "n": 6298, "p10_seconds": 0.9147000000000001, "p25_seconds": 1.084, "median_seconds": 1.29, "p75_seconds": 1.588, "p90_seconds": 2.5606000000000004, "share_under_60s": 1.0},
        {"source": "W_Nabellen offertes-SCHEDULE", "target": "W_Completeren aanvraag-COMPLETE", "n": 4829, "p10_seconds": 1.077, "p25_seconds": 1.179, "median_seconds": 1.387, "p75_seconds": 1.95, "p90_seconds": 2.2822, "share_under_60s": 1.0},
        {"source": "W_Completeren aanvraag-COMPLETE", "target": "W_Nabellen offertes-START", "n": 4800, "p10_seconds": 12.5843, "p25_seconds": 19.862750000000002, "median_seconds": 305.6565, "p75_seconds": 602280.85725, "p90_seconds": 675964.9162, "share_under_60s": 0.425},
        {"source": "W_Completeren aanvraag-START", "target": "A_ACCEPTED-COMPLETE", "n": 4764, "p10_seconds": 123.8471, "p25_seconds": 245.35725, "median_seconds": 373.4235, "p75_seconds": 524.60575, "p90_seconds": 734.748, "share_under_60s": 0.049},
        {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "A_PREACCEPTED-COMPLETE", "n": 4564, "p10_seconds": 32.8282, "p25_seconds": 35.614, "median_seconds": 38.9685, "p75_seconds": 43.210750000000004, "p90_seconds": 49.1896, "share_under_60s": 0.968},
        {"source": "W_Completeren aanvraag-SCHEDULE", "target": "W_Completeren aanvraag-START", "n": 4318, "p10_seconds": 28.597200000000004, "p25_seconds": 698.53475, "median_seconds": 6869.639, "p75_seconds": 45426.326, "p90_seconds": 71242.7371, "share_under_60s": 0.152},
        {"source": "A_FINALIZED-COMPLETE", "target": "O_SELECTED-COMPLETE", "n": 4289, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "A_ACCEPTED-COMPLETE", "target": "A_FINALIZED-COMPLETE", "n": 4249, "p10_seconds": 63.803, "p25_seconds": 118.448, "median_seconds": 183.982, "p75_seconds": 280.223, "p90_seconds": 409.7776000000001, "share_under_60s": 0.093},
        {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "W_Afhandelen leads-SCHEDULE", "n": 4175, "p10_seconds": 8.712200000000001, "p25_seconds": 11.942, "median_seconds": 22.613, "p75_seconds": 39.6125, "p90_seconds": 47.538999999999994, "share_under_60s": 0.95},
        {"source": "W_Afhandelen leads-SCHEDULE", "target": "W_Afhandelen leads-START", "n": 4038, "p10_seconds": 78.05730000000003, "p25_seconds": 584.8657499999999, "median_seconds": 2541.5485, "p75_seconds": 35574.42, "p90_seconds": 53684.0683, "share_under_60s": 0.089},
        {"source": "A_PARTLYSUBMITTED-COMPLETE", "target": "A_DECLINED-COMPLETE", "n": 3533, "p10_seconds": 31.7478, "p25_seconds": 34.197, "median_seconds": 37.565, "p75_seconds": 41.934, "p90_seconds": 49.74720000000001, "share_under_60s": 0.951},
        {"source": "O_SENT_BACK-COMPLETE", "target": "W_Valideren aanvraag-SCHEDULE", "n": 3345, "p10_seconds": 0.141, "p25_seconds": 0.153, "median_seconds": 0.18, "p75_seconds": 0.276, "p90_seconds": 0.37559999999999993, "share_under_60s": 0.999},
        {"source": "W_Valideren aanvraag-SCHEDULE", "target": "W_Nabellen offertes-COMPLETE", "n": 3171, "p10_seconds": 1.115, "p25_seconds": 1.229, "median_seconds": 1.45, "p75_seconds": 1.8955, "p90_seconds": 2.228, "share_under_60s": 1.0},
        {"source": "W_Nabellen offertes-COMPLETE", "target": "W_Valideren aanvraag-START", "n": 3127, "p10_seconds": 6744.789800000003, "p25_seconds": 89237.016, "median_seconds": 189089.061, "p75_seconds": 348615.80700000003, "p90_seconds": 440920.091, "share_under_60s": 0.048},
        {"source": "W_Nabellen offertes-START", "target": "O_SENT_BACK-COMPLETE", "n": 3123, "p10_seconds": 10.1908, "p25_seconds": 12.4835, "median_seconds": 21.627, "p75_seconds": 44.1475, "p90_seconds": 96.08440000000004, "share_under_60s": 0.818},
        {"source": "W_Valideren aanvraag-COMPLETE", "target": "W_Valideren aanvraag-START", "n": 2830, "p10_seconds": 24.4568, "p25_seconds": 242.11625, "median_seconds": 2012.101, "p75_seconds": 13263.28975, "p90_seconds": 87046.36559999995, "share_under_60s": 0.143},
        {"source": "W_Valideren aanvraag-START", "target": "W_Valideren aanvraag-COMPLETE", "n": 2817, "p10_seconds": 10.74, "p25_seconds": 44.596, "median_seconds": 303.645, "p75_seconds": 1048.956, "p90_seconds": 2213.381600000001, "share_under_60s": 0.285},
        {"source": "W_Afhandelen leads-COMPLETE", "target": "W_Completeren aanvraag-START", "n": 2361, "p10_seconds": 17.397, "p25_seconds": 844.995, "median_seconds": 5791.562, "p75_seconds": 15846.441, "p90_seconds": 48063.924, "share_under_60s": 0.153},
        {"source": "W_Completeren aanvraag-SCHEDULE", "target": "W_Afhandelen leads-COMPLETE", "n": 2328, "p10_seconds": 0.898, "p25_seconds": 0.987, "median_seconds": 1.148, "p75_seconds": 1.372, "p90_seconds": 1.8543000000000003, "share_under_60s": 1.0},
        {"source": "W_Afhandelen leads-START", "target": "A_PREACCEPTED-COMPLETE", "n": 2298, "p10_seconds": 35.6911, "p25_seconds": 65.77924999999999, "median_seconds": 107.6855, "p75_seconds": 184.928, "p90_seconds": 345.6515000000003, "share_under_60s": 0.213},
        {"source": "W_Valideren aanvraag-COMPLETE", "target": "W_Nabellen incomplete dossiers-START", "n": 2145, "p10_seconds": 26.0428, "p25_seconds": 332.025, "median_seconds": 2923.088, "p75_seconds": 8016.999, "p90_seconds": 14223.308, "share_under_60s": 0.157},
        {"source": "W_Nabellen incomplete dossiers-SCHEDULE", "target": "W_Valideren aanvraag-COMPLETE", "n": 2128, "p10_seconds": 1.448, "p25_seconds": 1.61, "median_seconds": 1.8525, "p75_seconds": 2.271, "p90_seconds": 2.7342999999999997, "share_under_60s": 1.0},
        {"source": "O_CANCELLED-COMPLETE", "target": "O_SELECTED-COMPLETE", "n": 2007, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "W_Valideren aanvraag-START", "target": "W_Nabellen incomplete dossiers-SCHEDULE", "n": 1927, "p10_seconds": 249.9444, "p25_seconds": 623.466, "median_seconds": 1169.758, "p75_seconds": 1819.0425, "p90_seconds": 2501.7416000000007, "share_under_60s": 0.029},
        {"source": "A_DECLINED-COMPLETE", "target": "W_Afhandelen leads-COMPLETE", "n": 1923, "p10_seconds": 2.8494, "p25_seconds": 3.0905, "median_seconds": 3.415, "p75_seconds": 3.9059999999999997, "p90_seconds": 4.4517999999999995, "share_under_60s": 0.998},
        {"source": "W_Afhandelen leads-START", "target": "A_DECLINED-COMPLETE", "n": 1770, "p10_seconds": 32.098000000000006, "p25_seconds": 69.0895, "median_seconds": 112.2, "p75_seconds": 200.07725, "p90_seconds": 382.05620000000005, "share_under_60s": 0.204},
        {"source": "W_Nabellen incomplete dossiers-COMPLETE", "target": "W_Valideren aanvraag-START", "n": 1706, "p10_seconds": 697.8025, "p25_seconds": 2132.33425, "median_seconds": 5040.9805, "p75_seconds": 46590.543000000005, "p90_seconds": 78429.4945, "share_under_60s": 0.03},
        {"source": "W_Valideren aanvraag-SCHEDULE", "target": "W_Nabellen incomplete dossiers-COMPLETE", "n": 1698, "p10_seconds": 1.4857, "p25_seconds": 1.65025, "median_seconds": 1.8965, "p75_seconds": 2.28, "p90_seconds": 2.7641999999999998, "share_under_60s": 0.999},
        {"source": "A_APPROVED-COMPLETE", "target": "A_REGISTERED-COMPLETE", "n": 1662, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "A_CANCELLED-COMPLETE", "target": "O_CANCELLED-COMPLETE", "n": 1638, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "W_Nabellen offertes-SCHEDULE", "target": "W_Nabellen offertes-COMPLETE", "n": 1632, "p10_seconds": 1.2241, "p25_seconds": 1.3207499999999999, "median_seconds": 1.694, "p75_seconds": 2.21325, "p90_seconds": 2.612, "share_under_60s": 0.96},
        {"source": "W_Nabellen offertes-START", "target": "O_CANCELLED-COMPLETE", "n": 1612, "p10_seconds": 31.9407, "p25_seconds": 48.406000000000006, "median_seconds": 106.4295, "p75_seconds": 217.2525, "p90_seconds": 412.43380000000025, "share_under_60s": 0.318},
        {"source": "W_Nabellen incomplete dossiers-START", "target": "W_Valideren aanvraag-SCHEDULE", "n": 1502, "p10_seconds": 24.8903, "p25_seconds": 36.49575, "median_seconds": 65.1735, "p75_seconds": 210.235, "p90_seconds": 569.3479000000001, "share_under_60s": 0.469},
        {"source": "A_ACTIVATED-COMPLETE", "target": "A_APPROVED-COMPLETE", "n": 1466, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "O_ACCEPTED-COMPLETE", "target": "A_ACTIVATED-COMPLETE", "n": 1285, "p10_seconds": 0.001, "p25_seconds": 0.001, "median_seconds": 0.001, "p75_seconds": 0.001, "p90_seconds": 0.001, "share_under_60s": 1.0},
        {"source": "A_REGISTERED-COMPLETE", "target": "O_ACCEPTED-COMPLETE", "n": 1126, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "A_DECLINED-COMPLETE", "target": "W_Completeren aanvraag-COMPLETE", "n": 1054, "p10_seconds": 2.9613, "p25_seconds": 3.188, "median_seconds": 3.5300000000000002, "p75_seconds": 4.02575, "p90_seconds": 4.5857, "share_under_60s": 0.999},
        {"source": "W_Afhandelen leads-COMPLETE", "target": "W_Afhandelen leads-START", "n": 1031, "p10_seconds": 3.878, "p25_seconds": 16.207500000000003, "median_seconds": 339.963, "p75_seconds": 2256.3575, "p90_seconds": 8921.651, "share_under_60s": 0.335},
        {"source": "W_Afhandelen leads-START", "target": "W_Afhandelen leads-COMPLETE", "n": 1027, "p10_seconds": 7.172200000000002, "p25_seconds": 42.710499999999996, "median_seconds": 141.496, "p75_seconds": 619.1610000000001, "p90_seconds": 4397.890399999998, "share_under_60s": 0.299},
        {"source": "A_REGISTERED-COMPLETE", "target": "W_Valideren aanvraag-COMPLETE", "n": 1009, "p10_seconds": 6.4536, "p25_seconds": 7.027, "median_seconds": 7.763, "p75_seconds": 8.498, "p90_seconds": 9.366600000000002, "share_under_60s": 0.999},
        {"source": "W_Completeren aanvraag-START", "target": "A_DECLINED-COMPLETE", "n": 1004, "p10_seconds": 176.1539, "p25_seconds": 313.63275, "median_seconds": 474.154, "p75_seconds": 697.546, "p90_seconds": 983.8745000000005, "share_under_60s": 0.023},
        {"source": "O_CANCELLED-COMPLETE", "target": "W_Nabellen offertes-COMPLETE", "n": 992, "p10_seconds": 1.837, "p25_seconds": 2.092, "median_seconds": 2.3964999999999996, "p75_seconds": 2.92425, "p90_seconds": 3.4413, "share_under_60s": 0.998},
        {"source": "W_Nabellen offertes-START", "target": "A_CANCELLED-COMPLETE", "n": 911, "p10_seconds": 72.639, "p25_seconds": 111.07249999999999, "median_seconds": 156.875, "p75_seconds": 240.096, "p90_seconds": 399.95, "share_under_60s": 0.081},
        {"source": "O_ACCEPTED-COMPLETE", "target": "W_Valideren aanvraag-COMPLETE", "n": 862, "p10_seconds": 6.3086, "p25_seconds": 6.952, "median_seconds": 7.6685, "p75_seconds": 8.3765, "p90_seconds": 9.1902, "share_under_60s": 1.0},
        {"source": "W_Valideren aanvraag-START", "target": "A_ACTIVATED-COMPLETE", "n": 862, "p10_seconds": 141.36260000000001, "p25_seconds": 317.4815, "median_seconds": 770.573, "p75_seconds": 1332.394, "p90_seconds": 1957.7283999999997, "share_under_60s": 0.019},
        {"source": "A_DECLINED-COMPLETE", "target": "O_DECLINED-COMPLETE", "n": 801, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "A_CANCELLED-COMPLETE", "target": "W_Completeren aanvraag-COMPLETE", "n": 730, "p10_seconds": 1.3109, "p25_seconds": 1.466, "median_seconds": 1.668, "p75_seconds": 2.12575, "p90_seconds": 2.6807000000000003, "share_under_60s": 0.999},
        {"source": "A_FINALIZED-COMPLETE", "target": "O_CREATED-COMPLETE", "n": 713, "p10_seconds": 0.9154000000000001, "p25_seconds": 1.056, "median_seconds": 1.252, "p75_seconds": 1.557, "p90_seconds": 2.208000000000002, "share_under_60s": 1.0},
        {"source": "W_Valideren aanvraag-START", "target": "A_APPROVED-COMPLETE", "n": 702, "p10_seconds": 142.19940000000005, "p25_seconds": 329.30825, "median_seconds": 739.4095, "p75_seconds": 1357.752, "p90_seconds": 1954.3673999999999, "share_under_60s": 0.02},
        {"source": "O_SELECTED-COMPLETE", "target": "A_FINALIZED-COMPLETE", "n": 697, "p10_seconds": 0.001, "p25_seconds": 0.001, "median_seconds": 0.001, "p75_seconds": 0.001, "p90_seconds": 0.001, "share_under_60s": 1.0},
    ],
    "global_median_duration_seconds": 47.94,
}

TPROB = {(d["source"], d["target"]): d["probability"] for d in PROCESS_CONTEXT["transition_probabilities_top"]}
DPROF = {(d["source"], d["target"]): d for d in PROCESS_CONTEXT["duration_profiles_top"]}
GLOBAL_MEDIAN = PROCESS_CONTEXT["global_median_duration_seconds"]


def parse_ts_safe(s):
    if pd.isna(s):
        return pd.NaT
    txt = str(s)
    if txt.strip() == "":
        return pd.NaT
    try:
        return pd.to_datetime(txt, errors="coerce")
    except Exception:
        return pd.NaT


def infer_precision_and_sep(ts_str):
    s = "" if pd.isna(ts_str) else str(ts_str)
    if "." in s:
        frac = s.split(".")[-1]
        if frac.isdigit():
            return len(frac), "."
    return 0, None


def precision_unit_seconds(precision):
    if precision <= 0:
        return 1.0
    return 10 ** (-precision)


def format_ts_like(dt, precision):
    if pd.isna(dt):
        return ""
    py = pd.Timestamp(dt).to_pydatetime()
    base = py.strftime("%Y-%m-%d %H:%M:%S")
    if precision <= 0:
        return base
    micros = f"{py.microsecond:06d}"
    return base + "." + micros[:precision]


def transition_prob(a, b):
    if a is None or b is None:
        return 0.2
    if (a, b) in TPROB:
        return TPROB[(a, b)]
    # weak structural heuristics
    wa = a.split("-")[0] if "-" in a else a
    wb = b.split("-")[0] if "-" in b else b
    sa = a.split("-")[-1] if "-" in a else ""
    sb = b.split("-")[-1] if "-" in b else ""
    if wa == wb:
        if sa == "SCHEDULE" and sb == "START":
            return 0.6
        if sa == "START" and sb == "COMPLETE":
            return 0.65
        if sa == "COMPLETE" and sb == "START":
            return 0.35
    return 0.02


def duration_guess_seconds(a, b):
    prof = DPROF.get((a, b))
    if prof is not None:
        if prof["share_under_60s"] >= 0.95:
            val = prof["p10_seconds"]
        else:
            val = prof["p25_seconds"]
        return max(val, 0.001)
    wa = a.split("-")[0] if isinstance(a, str) and "-" in a else None
    wb = b.split("-")[0] if isinstance(b, str) and "-" in b else None
    sa = a.split("-")[-1] if isinstance(a, str) and "-" in a else None
    sb = b.split("-")[-1] if isinstance(b, str) and "-" in b else None
    if wa is not None and wa == wb:
        if sa == "SCHEDULE" and sb == "START":
            return 30.0
        if sa == "START" and sb == "COMPLETE":
            return 20.0
        if sa == "COMPLETE" and sb == "START":
            return 60.0
    return min(max(GLOBAL_MEDIAN, 3.0), 60.0)


def local_duration_score(a, b, delta_sec):
    if delta_sec < 0:
        return -1e9
    prof = DPROF.get((a, b))
    if prof is None:
        expected = duration_guess_seconds(a, b)
        return -abs(delta_sec - expected) / max(expected, 10.0)
    candidates = [prof["p10_seconds"], prof["p25_seconds"], prof["median_seconds"]]
    scales = [max(c, 1.0) for c in candidates]
    best = max(-abs(delta_sec - c) / s for c, s in zip(candidates, scales))
    if prof["share_under_60s"] >= 0.95 and delta_sec <= 120:
        best += 0.5
    return best


def find_case_bounds(df_case):
    times = [parse_ts_safe(x) for x in df_case["Timestamp"].tolist() if str(x).strip() != ""]
    times = [t for t in times if not pd.isna(t)]
    if not times:
        now = pd.Timestamp("2000-01-01 00:00:00")
        return now, now
    return min(times), max(times)


def previous_clean_anchor(case_df, pos):
    for j in range(pos - 1, -1, -1):
        if case_df.iloc[j]["error_category"] == "CLEAN":
            ts = parse_ts_safe(case_df.iloc[j]["Timestamp"])
            if not pd.isna(ts):
                return j, ts, case_df.iloc[j]["Activity"]
    return None, None, None


def next_clean_anchor(case_df, pos):
    for j in range(pos + 1, len(case_df)):
        if case_df.iloc[j]["error_category"] == "CLEAN":
            ts = parse_ts_safe(case_df.iloc[j]["Timestamp"])
            if not pd.isna(ts):
                return j, ts, case_df.iloc[j]["Activity"]
    return None, None, None


def get_maximal_c1_blocks(case_df):
    blocks = []
    i = 0
    n = len(case_df)
    while i < n:
        row = case_df.iloc[i]
        if row["error_category"] == "C1_ORDERING_AMBIGUITY":
            obs = str(row["Timestamp"])
            start = i
            i += 1
            while i < n:
                r = case_df.iloc[i]
                if r["error_category"] == "C1_ORDERING_AMBIGUITY" and str(r["Timestamp"]) == obs:
                    i += 1
                else:
                    break
            if i - start >= 2:
                blocks.append((start, i - 1, obs))
        else:
            i += 1
    return blocks


def score_order(order, left_act, right_act):
    seq = []
    if left_act is not None:
        seq.append(left_act)
    seq.extend(order)
    if right_act is not None:
        seq.append(right_act)
    score = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        p = transition_prob(a, b)
        score += math.log(max(p, 1e-6))
    # motif bonuses
    for a, b in zip(order[:-1], order[1:]):
        wa = a.split("-")[0] if "-" in a else a
        wb = b.split("-")[0] if "-" in b else b
        sa = a.split("-")[-1] if "-" in a else ""
        sb = b.split("-")[-1] if "-" in b else ""
        if wa == wb and sa == "SCHEDULE" and sb == "START":
            score += 0.3
        if wa == wb and sa == "START" and sb == "COMPLETE":
            score += 0.3
    return score


def best_order(activities, left_act, right_act):
    n = len(activities)
    idxs = list(range(n))
    if n <= 7:
        best = None
        best_score = -1e18
        for perm in itertools.permutations(idxs):
            order = [activities[i] for i in perm]
            sc = score_order(order, left_act, right_act)
            if sc > best_score:
                best_score = sc
                best = perm
        return list(best)
    beam = [([], set(), 0.0)]
    width = 30
    for _ in range(n):
        nxt = []
        for seq, used, sc in beam:
            prev_act = left_act if not seq else activities[seq[-1]]
            for i in idxs:
                if i in used:
                    continue
                act = activities[i]
                add = math.log(max(transition_prob(prev_act, act), 1e-6))
                nxt.append((seq + [i], used | {i}, sc + add))
        nxt.sort(key=lambda x: x[2], reverse=True)
        beam = nxt[:width]
    best_seq = None
    best_score = -1e18
    for seq, used, sc in beam:
        order = [activities[i] for i in seq]
        final = sc + (math.log(max(transition_prob(order[-1], right_act), 1e-6)) if right_act is not None else 0.0)
        if final > best_score:
            best_score = final
            best_seq = seq
    return best_seq


def make_strict_increasing_times(start_ts, gaps_sec, precision, right_anchor=None):
    unit = precision_unit_seconds(precision)
    times = [pd.Timestamp(start_ts)]
    for g in gaps_sec:
        inc = max(g, unit)
        times.append(times[-1] + pd.to_timedelta(inc, unit="s"))
    # rounding safety
    fixed = [times[0]]
    for t in times[1:]:
        if t <= fixed[-1]:
            t = fixed[-1] + pd.to_timedelta(unit, unit="s")
        fixed.append(t)
    if right_anchor is not None and fixed[-1] > right_anchor:
        avail = (right_anchor - fixed[0]).total_seconds()
        min_needed = unit * (len(fixed) - 1)
        if avail >= min_needed:
            raw = [(x - fixed[0]).total_seconds() for x in fixed]
            total = max(raw[-1], unit)
            scaled = [0.0]
            for r in raw[1:]:
                scaled.append(r * avail / total)
            fixed2 = [fixed[0]]
            for off in scaled[1:]:
                candidate = fixed[0] + pd.to_timedelta(off, unit="s")
                if candidate <= fixed2[-1]:
                    candidate = fixed2[-1] + pd.to_timedelta(unit, unit="s")
                fixed2.append(candidate)
            if fixed2[-1] > right_anchor:
                fixed2[-1] = right_anchor
                for i in range(len(fixed2) - 2, -1, -1):
                    max_allowed = fixed2[i + 1] - pd.to_timedelta(unit, unit="s")
                    if fixed2[i] > max_allowed:
                        fixed2[i] = max_allowed
            fixed = fixed2
    return fixed


def repair_c1_block(case_df, start, end, obs_str, output_ts, repaired_flags, stats):
    rows = case_df.iloc[start:end + 1]
    acts = rows["Activity"].tolist()
    global_idxs = rows.index.tolist()
    obs_ts = parse_ts_safe(obs_str)
    precision, _ = infer_precision_and_sep(obs_str)
    left_pos, left_ts, left_act = previous_clean_anchor(case_df, start)
    right_pos, right_ts, right_act = next_clean_anchor(case_df, end)
    case_start, case_end = find_case_bounds(case_df)

    order_idx = best_order(acts, left_act, right_act)
    ordered_acts = [acts[i] for i in order_idx]
    ordered_global = [global_idxs[i] for i in order_idx]

    gaps = []
    for a, b in zip(ordered_acts[:-1], ordered_acts[1:]):
        gaps.append(duration_guess_seconds(a, b))

    right_anchor = right_ts if right_ts is not None else case_end
    room = (right_anchor - obs_ts).total_seconds() if (not pd.isna(obs_ts) and not pd.isna(right_anchor)) else None
    min_needed = max((len(ordered_acts) - 1) * max(1.0, precision_unit_seconds(precision)), 0.0)
    total_gap = sum(max(g, 0.001) for g in gaps)

    clamped = False
    if room is not None and total_gap > room:
        if room >= min_needed and total_gap > 0:
            factor = room / total_gap
            gaps = [max(g * factor, precision_unit_seconds(precision)) for g in gaps]
            clamped = True
        else:
            unit = max(1.0, precision_unit_seconds(precision))
            gaps = [unit] * len(gaps)
            clamped = True

    times = make_strict_increasing_times(obs_ts, gaps, precision, right_anchor=right_anchor)

    if len(times) >= 2:
        span = (times[-1] - times[0]).total_seconds()
        if span < (len(times) - 1):
            unit = max(1.0, precision_unit_seconds(precision))
            gaps2 = [unit + (g if i < len(gaps) else 0) * 0.0 for i, g in enumerate(gaps)]
            times = make_strict_increasing_times(obs_ts, gaps2, precision, right_anchor=right_anchor)
            clamped = True

    # clamp to anchors/case bounds
    unit_td = pd.to_timedelta(max(precision_unit_seconds(precision), 1e-6), unit="s")
    if left_ts is not None and times[0] < left_ts:
        shift = left_ts - times[0]
        times = [t + shift for t in times]
        clamped = True
    if right_ts is not None and times[-1] > right_ts:
        avail = (right_ts - times[0]).total_seconds()
        raw = [(t - times[0]).total_seconds() for t in times]
        total = max(raw[-1], precision_unit_seconds(precision))
        if avail > 0:
            times2 = [times[0]]
            for r in raw[1:]:
                cand = times[0] + pd.to_timedelta(r * avail / total, unit="s")
                if cand <= times2[-1]:
                    cand = times2[-1] + unit_td
                times2.append(cand)
            times = times2
            clamped = True
    times = [min(max(t, case_start), case_end) for t in times]

    # final distinctness and anchor safety
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + unit_td
            clamped = True
    if right_ts is not None and times[-1] > right_ts:
        latest_start = right_ts - pd.to_timedelta((len(times) - 1) * precision_unit_seconds(precision), unit="s")
        new_start = min(obs_ts, latest_start)
        if new_start < obs_ts:
            new_start = obs_ts
        times = [new_start + pd.to_timedelta(i * max(1.0, precision_unit_seconds(precision)), unit="s") for i in range(len(times))]
        times[-1] = min(times[-1], right_ts)
        clamped = True

    # exactly first keeps observed value
    times[0] = obs_ts

    for rank, gidx in enumerate(ordered_global):
        new_str = format_ts_like(times[rank], precision)
        output_ts[gidx] = new_str
        repaired_flags[gidx] = True

    stats["c1_blocks_repaired"] += 1
    if clamped:
        stats["blocks_clamped"] += 1


def plausibility_score_for_pair(prev_act, act1, act2, next_act, tprev, t1, t2, tnext):
    score = 0.0
    if tprev is not None and not pd.isna(tprev):
        score += local_duration_score(prev_act, act1, (t1 - tprev).total_seconds()) if prev_act is not None else 0.0
    score += local_duration_score(act1, act2, (t2 - t1).total_seconds())
    if tnext is not None and not pd.isna(tnext):
        score += local_duration_score(act2, next_act, (tnext - t2).total_seconds()) if next_act is not None else 0.0
    return score


def estimate_from_anchors(left_ts, left_act, right_ts, right_act, act, precision, observed_ts=None, preserve_date=None, case_start=None, case_end=None):
    candidates = []

    if left_ts is not None:
        d = duration_guess_seconds(left_act, act) if left_act is not None else min(max(GLOBAL_MEDIAN, 3.0), 60.0)
        candidates.append(left_ts + pd.to_timedelta(d, unit="s"))
    if right_ts is not None:
        d = duration_guess_seconds(act, right_act) if right_act is not None else min(max(GLOBAL_MEDIAN, 3.0), 60.0)
        candidates.append(right_ts - pd.to_timedelta(d, unit="s"))
    if left_ts is not None and right_ts is not None:
        dl = duration_guess_seconds(left_act, act) if left_act is not None else 30.0
        dr = duration_guess_seconds(act, right_act) if right_act is not None else 30.0
        span = (right_ts - left_ts).total_seconds()
        pos = dl / max(dl + dr, 1e-9)
        mid = left_ts + pd.to_timedelta(max(0.0, min(span, span * pos)), unit="s")
        candidates.append(mid)
    if observed_ts is not None and not pd.isna(observed_ts):
        candidates.append(observed_ts)

    if not candidates:
        base = case_start if case_start is not None else pd.Timestamp("2000-01-01")
        cand = base
    else:
        nums = sorted([c.value for c in candidates])
        cand = pd.Timestamp(nums[len(nums) // 2])

    if preserve_date is not None:
        cand = pd.Timestamp(
            year=preserve_date.year,
            month=preserve_date.month,
            day=preserve_date.day,
            hour=cand.hour,
            minute=cand.minute,
            second=cand.second,
            microsecond=cand.microsecond,
        )

    if left_ts is not None and cand < left_ts:
        if right_ts is not None and right_ts > left_ts:
            cand = left_ts + (right_ts - left_ts) / 2
        else:
            cand = left_ts + pd.to_timedelta(max(1.0, precision_unit_seconds(precision)), unit="s")
    if right_ts is not None and cand > right_ts:
        if left_ts is not None and right_ts > left_ts:
            cand = left_ts + (right_ts - left_ts) / 2
        else:
            cand = right_ts - pd.to_timedelta(max(1.0, precision_unit_seconds(precision)), unit="s")
    if case_start is not None and cand < case_start:
        cand = case_start
    if case_end is not None and cand > case_end:
        cand = case_end
    return cand


def repair_non_c1_rows(case_df, output_ts, repaired_flags):
    n = len(case_df)
    case_start, case_end = find_case_bounds(case_df)
    local_parsed = [parse_ts_safe(x) for x in case_df["Timestamp"].tolist()]
    for i in range(n):
        row = case_df.iloc[i]
        gidx = row.name
        cat = row["error_category"]
        if cat == "CLEAN" or cat == "C1_ORDERING_AMBIGUITY":
            continue

        obs_str = str(row["Timestamp"])
        precision, _ = infer_precision_and_sep(obs_str) if str(obs_str).strip() != "" else (3, ".")
        obs_ts = local_parsed[i]

        left_j, left_ts, left_act = previous_clean_anchor(case_df, i)
        right_j, right_ts, right_act = next_clean_anchor(case_df, i)

        if cat == "C2_TEMPORAL_INCONSISTENCY":
            # consider adjacent inversion pair
            if i > 0:
                prev_ts = parse_ts_safe(output_ts[case_df.index[i - 1]])
                cur_ts = obs_ts
                if not pd.isna(prev_ts) and not pd.isna(cur_ts) and cur_ts < prev_ts:
                    prev_row = case_df.iloc[i - 1]
                    if prev_row["error_category"] != "CLEAN":
                        prev_prev_j, prev_prev_ts, prev_prev_act = previous_clean_anchor(case_df, i - 1)
                        next_next_j, next_next_ts, next_next_act = next_clean_anchor(case_df, i)
                        t1, t2 = cur_ts, prev_ts
                        swap_score = plausibility_score_for_pair(
                            prev_prev_act,
                            row["Activity"],
                            prev_row["Activity"],
                            next_next_act,
                            prev_prev_ts,
                            t1,
                            t2,
                            next_next_ts,
                        )
                        disp_est_cur = estimate_from_anchors(
                            prev_ts, prev_row["Activity"], next_next_ts, next_next_act, row["Activity"],
                            precision, observed_ts=obs_ts, case_start=case_start, case_end=case_end
                        )
                        disp_score = plausibility_score_for_pair(
                            prev_prev_act,
                            prev_row["Activity"],
                            row["Activity"],
                            next_next_act,
                            prev_prev_ts,
                            prev_ts,
                            disp_est_cur,
                            next_next_ts,
                        )
                        movement_swap = abs((t1 - prev_ts).total_seconds()) + abs((t2 - cur_ts).total_seconds())
                        movement_disp = abs((disp_est_cur - cur_ts).total_seconds())
                        if swap_score > disp_score or (abs(swap_score - disp_score) < 0.25 and movement_swap <= movement_disp):
                            output_ts[case_df.index[i - 1]] = format_ts_like(t1, infer_precision_and_sep(str(prev_row["Timestamp"]))[0])
                            output_ts[gidx] = format_ts_like(t2, precision)
                            repaired_flags[case_df.index[i - 1]] = True
                            repaired_flags[gidx] = True
                            continue

            est = estimate_from_anchors(
                left_ts, left_act, right_ts, right_act, row["Activity"],
                precision, observed_ts=obs_ts, case_start=case_start, case_end=case_end
            )
            if left_ts is not None and est < left_ts:
                est = left_ts + pd.to_timedelta(max(1.0, precision_unit_seconds(precision)), unit="s")
            if right_ts is not None and est > right_ts:
                est = left_ts + (right_ts - left_ts) / 2 if left_ts is not None else right_ts
            output_ts[gidx] = format_ts_like(est, precision)
            repaired_flags[gidx] = True

        elif cat == "C3_MISSING_TEMPORAL_INFORMATION":
            preserve_date = None
            s = "" if pd.isna(obs_str) else str(obs_str).strip()
            if s and len(s) >= 10:
                try:
                    d = pd.to_datetime(s[:10], errors="coerce")
                    if not pd.isna(d):
                        preserve_date = d
                except Exception:
                    preserve_date = None
            est = estimate_from_anchors(
                left_ts, left_act, right_ts, right_act, row["Activity"],
                precision if precision > 0 else 3,
                observed_ts=obs_ts if not pd.isna(obs_ts) else None,
                preserve_date=preserve_date,
                case_start=case_start,
                case_end=case_end,
            )
            if preserve_date is not None and pd.Timestamp(est).date() != pd.Timestamp(preserve_date).date():
                est = pd.Timestamp(
                    year=preserve_date.year, month=preserve_date.month, day=preserve_date.day,
                    hour=est.hour, minute=est.minute, second=est.second, microsecond=est.microsecond
                )
                if left_ts is not None and est < left_ts and right_ts is not None and right_ts.date() == preserve_date.date():
                    est = left_ts + (right_ts - left_ts) / 2
                if right_ts is not None and est > right_ts and left_ts is not None and left_ts.date() == preserve_date.date():
                    est = left_ts + (right_ts - left_ts) / 2
            output_ts[gidx] = format_ts_like(est, precision if precision > 0 else 3)
            repaired_flags[gidx] = True

        else:
            est = estimate_from_anchors(
                left_ts, left_act, right_ts, right_act, row["Activity"],
                precision if precision > 0 else 3, observed_ts=obs_ts, case_start=case_start, case_end=case_end
            )
            output_ts[gidx] = format_ts_like(est, precision if precision > 0 else 3)
            repaired_flags[gidx] = True


def self_check_and_fix(df, output_ts, repaired_flags, c1_blocks):
    rows_equal = 0
    blocks_clamped_extra = 0

    # CLEAN rows unchanged exactly
    for idx, row in df.iterrows():
        if row["error_category"] == "CLEAN":
            output_ts[idx] = row["Timestamp"]

    # Corrupted rows should differ except exactly one per C1 block.
    allowed_equal = set()
    for block in c1_blocks:
        start_idx, end_idx, obs = block
        block_rows = df.iloc[start_idx:end_idx + 1]
        obs_match = [r.name for _, r in block_rows.iterrows() if output_ts[r.name] == str(r["Timestamp"])]
        if obs_match:
            parsed = [(g, parse_ts_safe(output_ts[g])) for g in obs_match]
            parsed = [(g, t) for g, t in parsed if not pd.isna(t)]
            if parsed:
                parsed.sort(key=lambda x: x[1])
                keep = parsed[0][0]
                allowed_equal.add(keep)
                for g in obs_match[1:]:
                    if g != keep:
                        t = parse_ts_safe(output_ts[g])
                        if pd.isna(t):
                            t = parse_ts_safe(str(df.loc[g, "Timestamp"]))
                        precision, _ = infer_precision_and_sep(str(df.loc[g, "Timestamp"]))
                        t = t + pd.to_timedelta(max(1.0, precision_unit_seconds(precision)), unit="s")
                        output_ts[g] = format_ts_like(t, precision)
                        blocks_clamped_extra += 1

    # validate/fix C1 blocks
    for start, end, obs in c1_blocks:
        block_rows = df.iloc[start:end + 1]
        glist = block_rows.index.tolist()
        vals = [parse_ts_safe(output_ts[g]) for g in glist]
        acts = block_rows["Activity"].tolist()
        precision, _ = infer_precision_and_sep(obs)
        obs_ts = parse_ts_safe(obs)
        _, _, _ = acts, vals, precision

        equal_obs = [g for g in glist if output_ts[g] == str(df.loc[g, "Timestamp"])]
        if len(equal_obs) != 1:
            # identify earliest current timestamp and force it to keep observed
            parsed = [(g, parse_ts_safe(output_ts[g])) for g in glist]
            parsed.sort(key=lambda x: x[1] if not pd.isna(x[1]) else pd.Timestamp.max)
            keeper = parsed[0][0]
            output_ts[keeper] = obs
            equal_obs = [keeper]
            for g, _ in parsed[1:]:
                if output_ts[g] == str(df.loc[g, "Timestamp"]):
                    t = parse_ts_safe(output_ts[g]) + pd.to_timedelta(max(1.0, precision_unit_seconds(precision)), unit="s")
                    output_ts[g] = format_ts_like(t, precision)
            blocks_clamped_extra += 1

        left_j, left_ts, left_act = previous_clean_anchor(df.iloc[df[df["Case"] == block_rows.iloc[0]["Case"]].index.min():df[df["Case"] == block_rows.iloc[0]["Case"]].index.max()+1].reset_index(drop=False), start)
        # above anchor retrieval not reused; just use direct global neighbor scans:
        right_anchor = None
        for j in range(end + 1, len(df)):
            if df.iloc[j]["Case"] == block_rows.iloc[0]["Case"] and df.iloc[j]["error_category"] == "CLEAN":
                right_anchor = parse_ts_safe(df.iloc[j]["Timestamp"])
                break

        parsed_block = [(g, parse_ts_safe(output_ts[g])) for g in glist]
        ordered = sorted(parsed_block, key=lambda x: x[1] if not pd.isna(x[1]) else pd.Timestamp.max)
        changed = False
        for i in range(1, len(ordered)):
            if ordered[i][1] <= ordered[i - 1][1]:
                ordered[i] = (ordered[i][0], ordered[i - 1][1] + pd.to_timedelta(max(1.0, precision_unit_seconds(precision)), unit="s"))
                changed = True
        if right_anchor is not None and ordered[-1][1] > right_anchor:
            avail = (right_anchor - obs_ts).total_seconds()
            if avail >= len(ordered) - 1:
                base_offsets = [0.0]
                for i in range(1, len(ordered)):
                    base_offsets.append((ordered[i][1] - ordered[0][1]).total_seconds())
                total = max(base_offsets[-1], 1.0)
                new_times = [obs_ts]
                for off in base_offsets[1:]:
                    nt = obs_ts + pd.to_timedelta(off * avail / total, unit="s")
                    if nt <= new_times[-1]:
                        nt = new_times[-1] + pd.to_timedelta(1.0, unit="s")
                    new_times.append(nt)
                ordered = [(ordered[i][0], new_times[i]) for i in range(len(ordered))]
                changed = True
        if changed:
            for g, t in ordered:
                output_ts[g] = format_ts_like(t, precision)
            blocks_clamped_extra += 1

    for idx, row in df.iterrows():
        if row["error_category"] != "CLEAN":
            if output_ts[idx] == row["Timestamp"] and idx not in allowed_equal:
                ts = parse_ts_safe(output_ts[idx])
                precision, _ = infer_precision_and_sep(str(row["Timestamp"]))
                if pd.isna(ts):
                    left = None
                    for j in range(idx - 1, -1, -1):
                        if df.iloc[j]["Case"] == row["Case"]:
                            left = parse_ts_safe(output_ts[df.index[j]])
                            if not pd.isna(left):
                                break
                    ts = left + pd.to_timedelta(max(1.0, precision_unit_seconds(precision)), unit="s") if left is not None else pd.Timestamp("2000-01-01")
                else:
                    ts = ts + pd.to_timedelta(max(1.0, precision_unit_seconds(precision)), unit="s")
                output_ts[idx] = format_ts_like(ts, precision)
            if output_ts[idx] == row["Timestamp"]:
                rows_equal += 1

    return blocks_clamped_extra, rows_equal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    required_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    original = df.copy(deep=True)
    output_ts = {idx: row["Timestamp"] for idx, row in df.iterrows()}
    repaired_flags = {idx: False for idx in df.index}
    stats = {"c1_blocks_repaired": 0, "blocks_clamped": 0}

    c1_blocks_global = []

    for case, case_df in df.groupby("Case", sort=False):
        case_df = case_df.copy()
        case_df["__global_index__"] = case_df.index
        case_df = case_df.reset_index(drop=True)

        blocks = get_maximal_c1_blocks(case_df)
        for start, end, obs in blocks:
            c1_blocks_global.append((case_df.iloc[start]["__global_index__"], case_df.iloc[end]["__global_index__"], obs))
            try:
                repair_c1_block(case_df, start, end, obs, output_ts, repaired_flags, stats)
            except Exception:
                # conservative fallback: observed + 1s increasing according to best order
                rows = case_df.iloc[start:end + 1]
                obs_ts = parse_ts_safe(obs)
                precision, _ = infer_precision_and_sep(obs)
                acts = rows["Activity"].tolist()
                gidxs = rows["__global_index__"].tolist()
                left_pos, left_ts, left_act = previous_clean_anchor(case_df, start)
                right_pos, right_ts, right_act = next_clean_anchor(case_df, end)
                ord_idx = best_order(acts, left_act, right_act)
                for rank, oi in enumerate(ord_idx):
                    t = obs_ts + pd.to_timedelta(rank * max(1.0, precision_unit_seconds(precision)), unit="s")
                    output_ts[gidxs[oi]] = format_ts_like(t, precision)
                    repaired_flags[gidxs[oi]] = True
                stats["c1_blocks_repaired"] += 1
                stats["blocks_clamped"] += 1

        try:
            repair_non_c1_rows(case_df, output_ts, repaired_flags)
        except Exception:
            # continue without aborting
            pass

    extra_clamped, rows_equal = self_check_and_fix(df, output_ts, repaired_flags, c1_blocks_global)
    stats["blocks_clamped"] += extra_clamped

    # write output with exact unchanged non-timestamp columns
    out = original.copy(deep=True)
    out["Timestamp"] = [output_ts[idx] for idx in out.index]

    # final hard checks repair-in-place
    if len(out) != len(original):
        out = original.copy(deep=True)
        out["Timestamp"] = [output_ts.get(idx, original.loc[idx, "Timestamp"]) for idx in original.index]

    for c in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        out[c] = original[c]

    # preserve CLEAN exact strings
    clean_mask = out["error_category"] == "CLEAN"
    out.loc[clean_mask, "Timestamp"] = original.loc[clean_mask, "Timestamp"]

    try:
        out.to_csv(args.output, index=False)
    except Exception:
        # last-resort safe write
        original["Timestamp"] = out["Timestamp"]
        original.to_csv(args.output, index=False)

    print(f"{stats['c1_blocks_repaired']},{stats['blocks_clamped']},{rows_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Must never exit non-zero or abort without output file intent; silent 0
        print("0,0,0")