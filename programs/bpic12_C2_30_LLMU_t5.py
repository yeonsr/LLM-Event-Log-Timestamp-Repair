import argparse
import itertools
import math
import re
from collections import defaultdict, Counter

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
        {"source": "W_Beoordelen fraude-SCHEDULE", "target": "W_Valideren aanvraag-COMPLETE", "probability": 0.2586},
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


def detect_precision(strings):
    frac = 0
    for s in strings:
        if pd.isna(s):
            continue
        m = re.search(r"\.(\d+)$", str(s).strip())
        if m:
            frac = max(frac, len(m.group(1)))
    return frac


def parse_ts(s):
    if s is None:
        return pd.NaT
    s2 = str(s).strip()
    if s2 == "":
        return pd.NaT
    try:
        return pd.to_datetime(s2)
    except Exception:
        return pd.NaT


def is_date_only_string(s):
    if s is None or pd.isna(s):
        return False
    s = str(s).strip()
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s))


def format_ts(ts, frac_digits):
    if pd.isna(ts):
        return ""
    if frac_digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    return base[:- (6 - frac_digits)] if frac_digits < 6 else base


class Context:
    def __init__(self, ctx):
        self.prob = {}
        for r in ctx["transition_probabilities_top"]:
            self.prob[(r["source"], r["target"])] = float(r["probability"])
        self.dur = {}
        for r in ctx["duration_profiles_top"]:
            self.dur[(r["source"], r["target"])] = r
        self.global_median = float(ctx["global_median_duration_seconds"])

    def trans_prob(self, a, b):
        if a is None or b is None:
            return 0.05
        return self.prob.get((a, b), 1e-6)

    def logp(self, a, b):
        return math.log(max(self.trans_prob(a, b), 1e-12))

    def gap_estimate(self, a, b):
        prof = self.dur.get((a, b))
        if prof:
            if prof["share_under_60s"] >= 0.8:
                g = prof["p10_seconds"]
            else:
                g = prof["p25_seconds"]
            g = max(float(g), 0.001)
            return g
        p = self.trans_prob(a, b)
        if p > 0.5:
            return min(5.0, self.global_median / 10.0)
        if p > 0.1:
            return min(30.0, self.global_median)
        return self.global_median

    def plausible_score(self, a, b, gap_seconds):
        prof = self.dur.get((a, b))
        if gap_seconds is None or gap_seconds < 0:
            return -50.0
        if prof:
            p10 = float(prof["p10_seconds"])
            p25 = float(prof["p25_seconds"])
            p50 = float(prof["median_seconds"])
            p90 = float(prof["p90_seconds"])
            center = self.gap_estimate(a, b)
            scale = max(p25, center, 1.0)
            if gap_seconds <= p90 * 2 + 1:
                return -abs(gap_seconds - center) / scale + math.log(max(prof["n"], 1)) / 20.0
            return -10.0 - (gap_seconds - p90) / max(p90, 1.0)
        center = self.gap_estimate(a, b)
        return -abs(gap_seconds - center) / max(center, 1.0) + self.logp(a, b)


def previous_clean_idx(case_df, pos):
    for j in range(pos - 1, -1, -1):
        if case_df.iloc[j]["error_category"] == "CLEAN":
            return j
    return None


def next_clean_idx(case_df, pos):
    for j in range(pos + 1, len(case_df)):
        if case_df.iloc[j]["error_category"] == "CLEAN":
            return j
    return None


def previous_parsed_idx(case_df, pos):
    for j in range(pos - 1, -1, -1):
        if not pd.isna(case_df.iloc[j]["parsed_ts"]):
            return j
    return None


def next_parsed_idx(case_df, pos):
    for j in range(pos + 1, len(case_df)):
        if not pd.isna(case_df.iloc[j]["parsed_ts"]):
            return j
    return None


def candidate_permutations(items, max_exact=720, beam_width=80, ctx=None, left_act=None, right_act=None):
    k = len(items)
    idxs = list(range(k))
    if math.factorial(k) <= max_exact:
        perms = list(itertools.permutations(idxs))
        return perms

    remaining = set(idxs)
    beam = [([], remaining, 0.0)]
    for _ in range(k):
        new_beam = []
        for seq, rem, score in beam:
            prev_act = left_act if not seq else items[seq[-1]]["Activity"]
            for i in rem:
                act = items[i]["Activity"]
                s = score + (ctx.logp(prev_act, act) if ctx else 0.0)
                new_seq = seq + [i]
                new_rem = set(rem)
                new_rem.remove(i)
                new_beam.append((new_seq, new_rem, s))
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]
    finals = []
    for seq, rem, score in beam:
        last_act = items[seq[-1]]["Activity"]
        score2 = score + (ctx.logp(last_act, right_act) if ctx else 0.0)
        finals.append((tuple(seq), score2))
    finals.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in finals[:beam_width]]


def choose_c1_order(block_rows, left_act, right_act, ctx):
    perms = candidate_permutations(block_rows, ctx=ctx, left_act=left_act, right_act=right_act)
    best = None
    best_score = -1e99
    for perm in perms:
        ordered = [block_rows[i] for i in perm]
        score = 0.0
        if left_act is not None:
            score += ctx.logp(left_act, ordered[0]["Activity"]) * 1.3
        for a, b in zip(ordered[:-1], ordered[1:]):
            score += ctx.logp(a["Activity"], b["Activity"]) * 1.5
            g = ctx.gap_estimate(a["Activity"], b["Activity"])
            score += -0.02 * min(g, 600.0)
        if right_act is not None:
            score += ctx.logp(ordered[-1]["Activity"], right_act) * 1.1

        score += -0.001 * sum(abs(r["_orig_pos"] - rank) for rank, r in enumerate(ordered))
        if score > best_score:
            best_score = score
            best = ordered
    return best


def assign_c1_times(case_df, block_positions, ctx, frac_digits, clamped_counter):
    first_pos = block_positions[0]
    last_pos = block_positions[-1]
    shared_ts = case_df.iloc[first_pos]["parsed_ts"]
    block_rows = []
    for rel, pos in enumerate(block_positions):
        row = case_df.iloc[pos].to_dict()
        row["_case_pos"] = pos
        row["_orig_pos"] = rel
        block_rows.append(row)

    left_idx = previous_clean_idx(case_df, first_pos)
    right_idx = next_clean_idx(case_df, last_pos)
    left_act = case_df.iloc[left_idx]["Activity"] if left_idx is not None else None
    right_act = case_df.iloc[right_idx]["Activity"] if right_idx is not None else None
    right_ts = case_df.iloc[right_idx]["parsed_ts"] if right_idx is not None else pd.NaT
    left_ts = case_df.iloc[left_idx]["parsed_ts"] if left_idx is not None else pd.NaT

    ordered = choose_c1_order(block_rows, left_act, right_act, ctx)

    min_step = pd.Timedelta(seconds=1)
    gaps = []
    for a, b in zip(ordered[:-1], ordered[1:]):
        gs = ctx.gap_estimate(a["Activity"], b["Activity"])
        gaps.append(max(gs, 1.0))
    cum = [0.0]
    for g in gaps:
        cum.append(cum[-1] + g)

    available_forward = None
    if not pd.isna(right_ts):
        available_forward = max((right_ts - shared_ts).total_seconds(), 0.0)

    place_as_end = False
    if not pd.isna(left_ts) and not pd.isna(shared_ts) and not pd.isna(right_ts):
        left_room = max((shared_ts - left_ts).total_seconds(), 0.0)
        right_room = max((right_ts - shared_ts).total_seconds(), 0.0)
        if right_room < len(ordered) - 1 and left_room > right_room * 10:
            place_as_end = True

    assigned = {}
    changed_clamp = False

    if place_as_end and left_idx is not None:
        rev_order = list(reversed(ordered))
        rev_gaps = []
        for a, b in zip(rev_order[:-1], rev_order[1:]):
            rev_gaps.append(max(ctx.gap_estimate(a["Activity"], b["Activity"]), 1.0))
        cum_back = [0.0]
        for g in rev_gaps:
            cum_back.append(cum_back[-1] + g)
        available_back = max((shared_ts - left_ts).total_seconds(), 0.0)
        min_span = max(len(ordered) - 1, 1)
        natural_span = cum_back[-1]
        target_span = natural_span
        if available_back < natural_span:
            target_span = max(min_span, available_back)
            changed_clamp = True
        elif natural_span < min_span:
            target_span = min_span
            changed_clamp = True
        scale = target_span / natural_span if natural_span > 0 else 1.0
        scaled = [c * scale for c in cum_back]
        scaled[-1] = target_span
        times_rev = [shared_ts - pd.to_timedelta(off, unit="s") for off in scaled]
        times = list(reversed(times_rev))
        for row, ts in zip(ordered, times):
            assigned[row["_case_pos"]] = ts
    else:
        min_span = max(len(ordered) - 1, 1)
        natural_span = cum[-1]
        target_span = natural_span
        if available_forward is not None:
            if natural_span > available_forward:
                target_span = max(min_span, available_forward)
                changed_clamp = True
            elif natural_span < min_span:
                target_span = min_span
                changed_clamp = True
        else:
            if natural_span < min_span:
                target_span = min_span
                changed_clamp = True
        scale = target_span / natural_span if natural_span > 0 else 1.0
        scaled = [c * scale for c in cum]
        scaled[0] = 0.0
        if len(scaled) > 1:
            for i in range(1, len(scaled)):
                scaled[i] = max(scaled[i], scaled[i - 1] + i * 0 + 1.0 if scaled[i] <= scaled[i - 1] else scaled[i])
            if available_forward is not None and scaled[-1] > available_forward:
                factor = available_forward / scaled[-1] if scaled[-1] > 0 else 1.0
                scaled = [x * factor for x in scaled]
                changed_clamp = True
            for i in range(1, len(scaled)):
                if scaled[i] - scaled[i - 1] < 1.0:
                    scaled[i] = scaled[i - 1] + 1.0
        if available_forward is not None and len(scaled) > 1 and scaled[-1] > available_forward:
            tail = available_forward
            scaled = [0.0] + [0.0] * (len(scaled) - 2) + [tail]
            if len(scaled) > 2:
                for i in range(1, len(scaled) - 1):
                    scaled[i] = min(max(float(i), scaled[i - 1] + 1.0), tail - (len(scaled) - 1 - i))
            changed_clamp = True
        times = [shared_ts + pd.to_timedelta(off, unit="s") for off in scaled]
        for row, ts in zip(ordered, times):
            assigned[row["_case_pos"]] = ts

    if changed_clamp:
        clamped_counter[0] += 1

    observed_string = case_df.iloc[first_pos]["Timestamp"]
    observed_positions = [pos for pos, ts in assigned.items() if ts == shared_ts]
    if len(observed_positions) != 1:
        keeper = ordered[0]["_case_pos"]
        assigned[keeper] = shared_ts
        others = [r["_case_pos"] for r in ordered[1:]]
        t = shared_ts
        for pos in others:
            t = t + min_step
            if right_idx is not None and t > right_ts:
                t = right_ts
            assigned[pos] = t

    if right_idx is not None:
        if max(assigned.values()) > right_ts:
            ordered_pos = [r["_case_pos"] for r in ordered]
            tlist = [assigned[p] for p in ordered_pos]
            span = max((right_ts - shared_ts).total_seconds(), float(len(ordered_pos) - 1))
            newt = [shared_ts + pd.to_timedelta(i * span / max(len(ordered_pos)-1, 1), unit="s") for i in range(len(ordered_pos))]
            for i in range(1, len(newt)):
                if newt[i] <= newt[i-1]:
                    newt[i] = newt[i-1] + min_step
            if newt[-1] > right_ts:
                newt[-1] = right_ts
                for i in range(len(newt)-2, -1, -1):
                    if newt[i] >= newt[i+1]:
                        newt[i] = newt[i+1] - min_step
            for p, ts in zip(ordered_pos, newt):
                assigned[p] = ts
            clamped_counter[0] += 1

    return assigned


def estimate_from_anchors(left_ts, right_ts, left_act, act, right_act, ctx):
    g_left = ctx.gap_estimate(left_act, act) if left_act is not None else ctx.global_median
    g_right = ctx.gap_estimate(act, right_act) if right_act is not None else ctx.global_median
    if left_ts is not None and not pd.isna(left_ts) and right_ts is not None and not pd.isna(right_ts):
        room = (right_ts - left_ts).total_seconds()
        if room < 0:
            return left_ts
        frac = g_left / max(g_left + g_right, 1e-9)
        cand = left_ts + pd.to_timedelta(room * frac, unit="s")
        if cand < left_ts:
            cand = left_ts
        if cand > right_ts:
            cand = left_ts + pd.to_timedelta(min(room, max(g_left, 1.0)), unit="s")
        return cand
    if left_ts is not None and not pd.isna(left_ts):
        return left_ts + pd.to_timedelta(max(g_left, 1.0), unit="s")
    if right_ts is not None and not pd.isna(right_ts):
        return right_ts - pd.to_timedelta(max(g_right, 1.0), unit="s")
    return pd.NaT


def repair_c2(case_df, pos, ctx):
    left_idx = previous_parsed_idx(case_df, pos)
    right_idx = next_parsed_idx(case_df, pos + 1)
    i = pos
    j = pos + 1
    t1 = case_df.iloc[i]["parsed_ts"]
    t2 = case_df.iloc[j]["parsed_ts"]

    left_ts = case_df.iloc[left_idx]["parsed_ts"] if left_idx is not None else pd.NaT
    right_ts = case_df.iloc[right_idx]["parsed_ts"] if right_idx is not None else pd.NaT
    left_act = case_df.iloc[left_idx]["Activity"] if left_idx is not None else None
    a1 = case_df.iloc[i]["Activity"]
    a2 = case_df.iloc[j]["Activity"]
    right_act = case_df.iloc[right_idx]["Activity"] if right_idx is not None else None

    def local_score(x1, x2):
        s = 0.0
        valid = True
        if left_idx is not None:
            gap = (x1 - left_ts).total_seconds()
            if gap < 0:
                valid = False
            s += ctx.plausible_score(left_act, a1, gap)
        gap12 = (x2 - x1).total_seconds()
        if gap12 < 0:
            valid = False
        s += ctx.plausible_score(a1, a2, gap12)
        if right_idx is not None:
            gap = (right_ts - x2).total_seconds()
            if gap < 0:
                valid = False
            s += ctx.plausible_score(a2, right_act, gap)
        if not valid:
            s -= 200.0
        return s

    current_score = local_score(t1, t2)
    swap_score = local_score(t2, t1)

    if t2 >= t1 and swap_score <= current_score:
        return {}

    if swap_score > current_score + 2.0 and (t2 <= t1 or swap_score > current_score + 5.0):
        return {i: t2, j: t1}

    candidates = {}
    cand1 = estimate_from_anchors(left_ts if left_idx is not None else pd.NaT, t2, left_act, a1, a2, ctx)
    cand2 = estimate_from_anchors(t1, right_ts if right_idx is not None else pd.NaT, a2, a2, right_act, ctx)

    score_fix1 = local_score(cand1, t2) if not pd.isna(cand1) else -1e99
    score_fix2 = local_score(t1, cand2) if not pd.isna(cand2) else -1e99

    if score_fix1 >= score_fix2:
        candidates[i] = cand1
    else:
        candidates[j] = cand2

    return candidates


def repair_c3(case_df, pos, ctx):
    left_idx = previous_clean_idx(case_df, pos)
    right_idx = next_clean_idx(case_df, pos)
    left_ts = case_df.iloc[left_idx]["parsed_ts"] if left_idx is not None else pd.NaT
    right_ts = case_df.iloc[right_idx]["parsed_ts"] if right_idx is not None else pd.NaT
    left_act = case_df.iloc[left_idx]["Activity"] if left_idx is not None else None
    right_act = case_df.iloc[right_idx]["Activity"] if right_idx is not None else None
    act = case_df.iloc[pos]["Activity"]
    raw = case_df.iloc[pos]["Timestamp"]

    if is_date_only_string(raw):
        day = pd.to_datetime(str(raw).strip()).date()
        day_start = pd.Timestamp(f"{day} 00:00:00")
        day_end = pd.Timestamp(f"{day} 23:59:59.999999")
        L = max(left_ts, day_start) if not pd.isna(left_ts) else day_start
        R = min(right_ts, day_end) if not pd.isna(right_ts) else day_end
        cand = estimate_from_anchors(L, R, left_act, act, right_act, ctx)
        if pd.isna(cand):
            cand = L
        if cand < L:
            cand = L
        if cand > R:
            room = (R - L).total_seconds()
            cand = L + pd.to_timedelta(max(room / 2.0, 0), unit="s")
        return cand

    cand = estimate_from_anchors(left_ts, right_ts, left_act, act, right_act, ctx)
    if pd.isna(cand):
        prev_idx = previous_parsed_idx(case_df, pos)
        next_idx = next_parsed_idx(case_df, pos)
        if prev_idx is not None and next_idx is not None:
            cand = case_df.iloc[prev_idx]["parsed_ts"] + (case_df.iloc[next_idx]["parsed_ts"] - case_df.iloc[prev_idx]["parsed_ts"]) / 2
        elif prev_idx is not None:
            cand = case_df.iloc[prev_idx]["parsed_ts"] + pd.to_timedelta(ctx.global_median, unit="s")
        elif next_idx is not None:
            cand = case_df.iloc[next_idx]["parsed_ts"] - pd.to_timedelta(ctx.global_median, unit="s")
        else:
            cand = pd.Timestamp("2000-01-01 00:00:00")
    if not pd.isna(left_ts) and cand < left_ts:
        cand = left_ts
    if not pd.isna(right_ts) and cand > right_ts:
        if not pd.isna(left_ts) and right_ts >= left_ts:
            cand = left_ts + (right_ts - left_ts) / 2
        else:
            cand = right_ts
    return cand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_columns = list(df.columns)
    for c in ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]:
        if c not in df.columns:
            df[c] = ""

    df["_orig_index"] = range(len(df))
    df["_orig_ts_str"] = df["Timestamp"]
    df["parsed_ts"] = df["Timestamp"].apply(parse_ts)

    frac_digits = detect_precision(df["Timestamp"].tolist())
    ctx = Context(PROCESS_CONTEXT)

    output_ts = df["_orig_ts_str"].copy()
    repaired_mask = pd.Series(False, index=df.index)
    c1_blocks_repaired = 0
    clamped_counter = [0]

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_idx = list(idxs)
        case_df = df.loc[case_idx].copy().reset_index()
        n = len(case_df)

        p = 0
        while p < n:
            cat = case_df.iloc[p]["error_category"]
            if cat == "C1_ORDERING_AMBIGUITY":
                ts = case_df.iloc[p]["Timestamp"]
                block = [p]
                q = p + 1
                while q < n and case_df.iloc[q]["error_category"] == "C1_ORDERING_AMBIGUITY" and case_df.iloc[q]["Timestamp"] == ts:
                    block.append(q)
                    q += 1
                if len(block) >= 2:
                    assigned = assign_c1_times(case_df, block, ctx, frac_digits, clamped_counter)
                    for local_pos, ts_new in assigned.items():
                        global_idx = case_df.iloc[local_pos]["index"]
                        output_ts.loc[global_idx] = format_ts(ts_new, frac_digits)
                        repaired_mask.loc[global_idx] = True
                    c1_blocks_repaired += 1
                else:
                    global_idx = case_df.iloc[p]["index"]
                    base = case_df.iloc[p]["parsed_ts"]
                    if pd.isna(base):
                        base = repair_c3(case_df, p, ctx)
                    else:
                        prev_idx = previous_clean_idx(case_df, p)
                        next_idx = next_clean_idx(case_df, p)
                        L = case_df.iloc[prev_idx]["parsed_ts"] if prev_idx is not None else pd.NaT
                        R = case_df.iloc[next_idx]["parsed_ts"] if next_idx is not None else pd.NaT
                        if not pd.isna(R) and base >= R:
                            base = min(R - pd.Timedelta(seconds=1), base - pd.Timedelta(seconds=1))
                        elif not pd.isna(L) and base <= L:
                            base = max(L + pd.Timedelta(seconds=1), base + pd.Timedelta(seconds=1))
                        else:
                            base = base + pd.Timedelta(seconds=1)
                    output_ts.loc[global_idx] = format_ts(base, frac_digits)
                    repaired_mask.loc[global_idx] = True
                p = q
                continue
            p += 1

        p = 0
        while p < n - 1:
            if case_df.iloc[p]["error_category"] == "C2_TEMPORAL_INCONSISTENCY" and case_df.iloc[p + 1]["error_category"] == "C2_TEMPORAL_INCONSISTENCY":
                t1 = case_df.iloc[p]["parsed_ts"]
                t2 = case_df.iloc[p + 1]["parsed_ts"]
                if not pd.isna(t1) and not pd.isna(t2) and t2 < t1:
                    repaired = repair_c2(case_df, p, ctx)
                    for local_pos, ts_new in repaired.items():
                        global_idx = case_df.iloc[local_pos]["index"]
                        if pd.isna(ts_new):
                            continue
                        output_ts.loc[global_idx] = format_ts(ts_new, frac_digits)
                        repaired_mask.loc[global_idx] = True
                p += 2
                continue
            p += 1

        for p in range(n):
            if case_df.iloc[p]["error_category"] == "C3_MISSING_TEMPORAL_INFORMATION":
                ts_new = repair_c3(case_df, p, ctx)
                global_idx = case_df.iloc[p]["index"]
                output_ts.loc[global_idx] = format_ts(ts_new, frac_digits)
                repaired_mask.loc[global_idx] = True

        for p in range(n):
            if case_df.iloc[p]["error_category"] == "C2_TEMPORAL_INCONSISTENCY":
                global_idx = case_df.iloc[p]["index"]
                if not repaired_mask.loc[global_idx]:
                    ts_parsed = parse_ts(output_ts.loc[global_idx])
                    if pd.isna(ts_parsed):
                        ts_new = repair_c3(case_df, p, ctx)
                    else:
                        left_idx = previous_clean_idx(case_df, p)
                        right_idx = next_clean_idx(case_df, p)
                        left_ts = case_df.iloc[left_idx]["parsed_ts"] if left_idx is not None else pd.NaT
                        right_ts = case_df.iloc[right_idx]["parsed_ts"] if right_idx is not None else pd.NaT
                        if not pd.isna(left_ts) and ts_parsed < left_ts:
                            ts_new = estimate_from_anchors(left_ts, right_ts, case_df.iloc[left_idx]["Activity"] if left_idx is not None else None, case_df.iloc[p]["Activity"], case_df.iloc[right_idx]["Activity"] if right_idx is not None else None, ctx)
                        elif not pd.isna(right_ts) and ts_parsed > right_ts:
                            ts_new = estimate_from_anchors(left_ts, right_ts, case_df.iloc[left_idx]["Activity"] if left_idx is not None else None, case_df.iloc[p]["Activity"], case_df.iloc[right_idx]["Activity"] if right_idx is not None else None, ctx)
                        else:
                            ts_new = ts_parsed + pd.Timedelta(seconds=1)
                    output_ts.loc[global_idx] = format_ts(ts_new, frac_digits)
                    repaired_mask.loc[global_idx] = True

    df["Timestamp"] = output_ts

    try:
        parsed_out = pd.to_datetime(df["Timestamp"], errors="coerce")
    except Exception:
        parsed_out = df["Timestamp"].apply(parse_ts)

    for idx, row in df.iterrows():
        if row["error_category"] == "CLEAN":
            df.at[idx, "Timestamp"] = row["_orig_ts_str"]

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        idxs = list(idxs)
        case_slice = df.loc[idxs].copy()
        case_parsed = pd.to_datetime(case_slice["Timestamp"], errors="coerce")
        valid_ts = case_parsed.dropna()
        if len(valid_ts) == 0:
            continue
        case_start = valid_ts.min()
        case_end = valid_ts.max()

        for local_i, idx in enumerate(idxs):
            if df.at[idx, "error_category"] == "CLEAN":
                continue
            ts = parse_ts(df.at[idx, "Timestamp"])
            if pd.isna(ts):
                prev_idx = idxs[local_i - 1] if local_i > 0 else None
                next_idx = idxs[local_i + 1] if local_i + 1 < len(idxs) else None
                L = parse_ts(df.at[prev_idx, "Timestamp"]) if prev_idx is not None else case_start
                R = parse_ts(df.at[next_idx, "Timestamp"]) if next_idx is not None else case_end
                if pd.isna(L):
                    L = case_start
                if pd.isna(R):
                    R = case_end
                ts = L + (R - L) / 2
            if ts < case_start:
                ts = case_start
            if ts > case_end:
                ts = case_end
            df.at[idx, "Timestamp"] = format_ts(ts, frac_digits)

        local = case_slice.reset_index()
        p = 0
        while p < len(local):
            if local.iloc[p]["error_category"] == "C1_ORDERING_AMBIGUITY":
                observed = local.iloc[p]["_orig_ts_str"]
                block = [p]
                q = p + 1
                while q < len(local) and local.iloc[q]["error_category"] == "C1_ORDERING_AMBIGUITY" and local.iloc[q]["_orig_ts_str"] == observed:
                    block.append(q)
                    q += 1
                if len(block) >= 2:
                    block_global = [local.iloc[x]["index"] for x in block]
                    tvals = [parse_ts(df.at[g, "Timestamp"]) for g in block_global]
                    observed_ts = parse_ts(observed)
                    eq_obs = [i for i, t in enumerate(tvals) if t == observed_ts]
                    if len(eq_obs) != 1:
                        min_i = min(range(len(tvals)), key=lambda z: tvals[z] if not pd.isna(tvals[z]) else pd.Timestamp.max)
                        for kx, g in enumerate(block_global):
                            if kx == min_i:
                                df.at[g, "Timestamp"] = observed
                            else:
                                base = observed_ts + pd.Timedelta(seconds=kx if kx < min_i else kx + 1)
                                df.at[g, "Timestamp"] = format_ts(base, frac_digits)
                    tvals = [parse_ts(df.at[g, "Timestamp"]) for g in block_global]
                    sorted_unique = True
                    for a, b in zip(sorted(tvals), sorted(tvals)[1:]):
                        if not pd.isna(a) and not pd.isna(b) and not (b > a):
                            sorted_unique = False
                    if not sorted_unique or len(set(tvals)) != len(tvals):
                        order = sorted(range(len(block_global)), key=lambda z: (tvals[z] if not pd.isna(tvals[z]) else pd.Timestamp.max))
                        for rank, z in enumerate(order):
                            g = block_global[z]
                            if rank == 0:
                                df.at[g, "Timestamp"] = observed
                            else:
                                df.at[g, "Timestamp"] = format_ts(observed_ts + pd.Timedelta(seconds=rank), frac_digits)
                    right_clean_local = None
                    for rr in range(q, len(local)):
                        if local.iloc[rr]["error_category"] == "CLEAN":
                            right_clean_local = rr
                            break
                    if right_clean_local is not None:
                        right_ts = parse_ts(local.iloc[right_clean_local]["Timestamp"])
                        repaired_vals = [parse_ts(df.at[g, "Timestamp"]) for g in block_global]
                        repaired_vals_sorted = sorted(repaired_vals)
                        if repaired_vals_sorted[-1] > right_ts:
                            span = max((right_ts - observed_ts).total_seconds(), len(block_global) - 1)
                            times = [observed_ts + pd.to_timedelta(i * span / max(len(block_global)-1, 1), unit="s") for i in range(len(block_global))]
                            for i in range(1, len(times)):
                                if times[i] <= times[i-1]:
                                    times[i] = times[i-1] + pd.Timedelta(seconds=1)
                            if times[-1] > right_ts:
                                times[-1] = right_ts
                                for i in range(len(times)-2, -1, -1):
                                    if times[i] >= times[i+1]:
                                        times[i] = times[i+1] - pd.Timedelta(seconds=1)
                            order = sorted(block_global, key=lambda g: parse_ts(df.at[g, "Timestamp"]))
                            for g, ts in zip(order, times):
                                if ts < observed_ts:
                                    ts = observed_ts
                                df.at[g, "Timestamp"] = format_ts(ts, frac_digits)
                            clamped_counter[0] += 1
                p = q
                continue
            p += 1

    still_equal = 0
    for idx, row in df.iterrows():
        if row["error_category"] != "CLEAN" and row["Timestamp"] == row["_orig_ts_str"]:
            still_equal += 1

    out = df[original_columns].copy()
    out.to_csv(args.output, index=False)

    print(f"{c1_blocks_repaired},{clamped_counter[0]},{still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            import csv
            import argparse as _argparse
            p = _argparse.ArgumentParser()
            p.add_argument("--input", required=True)
            p.add_argument("--output", required=True)
            a = p.parse_args()
            df = pd.read_csv(a.input, dtype=str, keep_default_na=False)
            df.to_csv(a.output, index=False)
            print("0,0,0")
        except Exception:
            print("0,0,0")