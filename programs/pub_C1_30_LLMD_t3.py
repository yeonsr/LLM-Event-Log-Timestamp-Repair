import argparse
import itertools
import math
import re
from collections import Counter, defaultdict

import pandas as pd


TRANSITION_PROBABILITIES_TOP = [
    {"source": "Deliver to customer", "target": "EVENT 8 END", "probability": 1.0},
    {"source": "Signal food complete", "target": "Bring food", "probability": 0.9985},
    {"source": "Bring food", "target": "Deliver to customer", "probability": 0.9972},
    {"source": "Prepare main course", "target": "Signal food complete", "probability": 0.9913},
    {"source": "start", "target": "Take order", "probability": 0.9708},
    {"source": "Bring drinks", "target": "Review drinks", "probability": 0.9554},
    {"source": "Take order", "target": "Review order", "probability": 0.9464},
    {"source": "prepare drinks", "target": "Bring drinks", "probability": 0.8979},
    {"source": "Review order", "target": "prepare drinks", "probability": 0.7727},
    {"source": "Prepare starter", "target": "prepare drinks", "probability": 0.531},
    {"source": "Review drinks", "target": "Prepare main course", "probability": 0.4767},
    {"source": "Review drinks", "target": "prepare drinks", "probability": 0.3939},
    {"source": "Prepare starter", "target": "Bring drinks", "probability": 0.2706},
    {"source": "Review order", "target": "Prepare starter", "probability": 0.1765},
    {"source": "Prepare starter", "target": "Prepare main course", "probability": 0.086},
    {"source": "Review drinks", "target": "Signal food complete", "probability": 0.0681},
    {"source": "prepare drinks", "target": "Prepare starter", "probability": 0.0596},
    {"source": "Prepare starter", "target": "Review drinks", "probability": 0.0583},
    {"source": "Prepare starter", "target": "Signal food complete", "probability": 0.0481},
    {"source": "Review drinks", "target": "Prepare starter", "probability": 0.0392},
    {"source": "Take order", "target": "Review drinks", "probability": 0.0218},
    {"source": "Review order", "target": "Review drinks", "probability": 0.0181},
    {"source": "prepare drinks", "target": "Bring food", "probability": 0.0155},
    {"source": "Review drinks", "target": "Bring food", "probability": 0.0149},
    {"source": "Review order", "target": "Prepare main course", "probability": 0.0133},
    {"source": "Bring drinks", "target": "Signal food complete", "probability": 0.0129},
    {"source": "start", "target": "Bring drinks", "probability": 0.0124},
    {"source": "Take order", "target": "Bring drinks", "probability": 0.0124},
    {"source": "prepare drinks", "target": "Signal food complete", "probability": 0.0121},
    {"source": "Bring drinks", "target": "Prepare starter", "probability": 0.0119},
    {"source": "Take order", "target": "Prepare main course", "probability": 0.0117},
    {"source": "Bring drinks", "target": "Bring food", "probability": 0.011},
    {"source": "Review order", "target": "Signal food complete", "probability": 0.0107},
    {"source": "start", "target": "Review drinks", "probability": 0.0092},
    {"source": "Review order", "target": "Bring drinks", "probability": 0.0087},
    {"source": "prepare drinks", "target": "Prepare main course", "probability": 0.0068},
    {"source": "prepare drinks", "target": "Review drinks", "probability": 0.0062},
    {"source": "Prepare starter", "target": "Bring food", "probability": 0.006},
    {"source": "start", "target": "prepare drinks", "probability": 0.005},
    {"source": "Take order", "target": "prepare drinks", "probability": 0.0049},
    {"source": "Prepare main course", "target": "Bring drinks", "probability": 0.0046},
    {"source": "Bring drinks", "target": "Bring drinks", "probability": 0.0046},
    {"source": "Review drinks", "target": "Review drinks", "probability": 0.004},
    {"source": "Bring drinks", "target": "Prepare main course", "probability": 0.0034},
    {"source": "Prepare main course", "target": "prepare drinks", "probability": 0.0031},
    {"source": "Take order", "target": "Prepare starter", "probability": 0.0027},
    {"source": "start", "target": "Prepare starter", "probability": 0.0026},
    {"source": "Bring food", "target": "Bring drinks", "probability": 0.0022},
    {"source": "prepare drinks", "target": "prepare drinks", "probability": 0.0019},
    {"source": "Review drinks", "target": "Deliver to customer", "probability": 0.0019},
    {"source": "Review drinks", "target": "Bring drinks", "probability": 0.0013},
    {"source": "Prepare main course", "target": "Review drinks", "probability": 0.001},
    {"source": "Bring drinks", "target": "prepare drinks", "probability": 0.0009},
    {"source": "Signal food complete", "target": "prepare drinks", "probability": 0.0009},
    {"source": "Signal food complete", "target": "Bring drinks", "probability": 0.0006},
    {"source": "Bring food", "target": "prepare drinks", "probability": 0.0006},
]

DURATION_PROFILES_TOP = [
    {"source": "Bring drinks", "target": "Review drinks", "n": 7062, "p10_seconds": 6.1307, "p25_seconds": 17.0105, "median_seconds": 41.052, "p75_seconds": 83.472, "p90_seconds": 136.7487, "share_under_60s": 0.631},
    {"source": "prepare drinks", "target": "Bring drinks", "n": 6704, "p10_seconds": 28.8126, "p25_seconds": 79.71675, "median_seconds": 195.9745, "p75_seconds": 413.75325, "p90_seconds": 700.3471, "share_under_60s": 0.198},
    {"source": "Deliver to customer", "target": "EVENT 8 END", "n": 5000, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "Bring food", "target": "Deliver to customer", "n": 4986, "p10_seconds": 12.5235, "p25_seconds": 34.57875, "median_seconds": 85.062, "p75_seconds": 168.381, "p90_seconds": 278.243, "share_under_60s": 0.389},
    {"source": "start", "target": "Take order", "n": 4854, "p10_seconds": 12.910100000000002, "p25_seconds": 35.795249999999996, "median_seconds": 89.148, "p75_seconds": 180.67575, "p90_seconds": 315.96629999999993, "share_under_60s": 0.378},
    {"source": "Signal food complete", "target": "Bring food", "n": 4682, "p10_seconds": 33.0276, "p25_seconds": 90.3885, "median_seconds": 214.9425, "p75_seconds": 423.9705, "p90_seconds": 718.4200000000002, "share_under_60s": 0.176},
    {"source": "Take order", "target": "Review order", "n": 4594, "p10_seconds": 6.1912, "p25_seconds": 17.985, "median_seconds": 41.872, "p75_seconds": 82.97149999999999, "p90_seconds": 138.1861, "share_under_60s": 0.636},
    {"source": "Prepare main course", "target": "Signal food complete", "n": 3866, "p10_seconds": 85943.671, "p25_seconds": 253686.04975, "median_seconds": 513669.015, "p75_seconds": 685817.92925, "p90_seconds": 845950.272, "share_under_60s": 0.001},
    {"source": "Review drinks", "target": "Prepare main course", "n": 3563, "p10_seconds": 83921.2662, "p25_seconds": 243350.55099999998, "median_seconds": 500815.099, "p75_seconds": 680990.933, "p90_seconds": 788371.1142000001, "share_under_60s": 0.001},
    {"source": "Review order", "target": "prepare drinks", "n": 3550, "p10_seconds": 25.252, "p25_seconds": 68.49925, "median_seconds": 173.713, "p75_seconds": 352.967, "p90_seconds": 606.4154, "share_under_60s": 0.226},
    {"source": "Review drinks", "target": "prepare drinks", "n": 2944, "p10_seconds": 30.721300000000003, "p25_seconds": 85.6735, "median_seconds": 209.56900000000002, "p75_seconds": 415.462, "p90_seconds": 694.3028000000007, "share_under_60s": 0.182},
    {"source": "Prepare starter", "target": "prepare drinks", "n": 883, "p10_seconds": 30.3582, "p25_seconds": 89.143, "median_seconds": 206.2, "p75_seconds": 397.074, "p90_seconds": 656.9610000000001, "share_under_60s": 0.19},
    {"source": "Review order", "target": "Prepare starter", "n": 811, "p10_seconds": 15.008, "p25_seconds": 42.552, "median_seconds": 97.24, "p75_seconds": 200.8065, "p90_seconds": 351.984, "share_under_60s": 0.354},
    {"source": "Review drinks", "target": "Signal food complete", "n": 509, "p10_seconds": 7451.2538, "p25_seconds": 175593.462, "median_seconds": 445342.856, "p75_seconds": 686397.671, "p90_seconds": 797873.2921999999, "share_under_60s": 0.002},
    {"source": "Prepare starter", "target": "Bring drinks", "n": 450, "p10_seconds": 29.029200000000007, "p25_seconds": 86.16725, "median_seconds": 203.727, "p75_seconds": 427.8865, "p90_seconds": 688.7044000000003, "share_under_60s": 0.189},
    {"source": "prepare drinks", "target": "Prepare starter", "n": 445, "p10_seconds": 15.8778, "p25_seconds": 40.819, "median_seconds": 100.292, "p75_seconds": 192.881, "p90_seconds": 296.64940000000007, "share_under_60s": 0.362},
    {"source": "Review drinks", "target": "Prepare starter", "n": 293, "p10_seconds": 17.944400000000005, "p25_seconds": 63.83, "median_seconds": 151.424, "p75_seconds": 322.423, "p90_seconds": 679.4376000000004, "share_under_60s": 0.232},
    {"source": "Prepare starter", "target": "Prepare main course", "n": 143, "p10_seconds": 76289.61880000001, "p25_seconds": 192412.7035, "median_seconds": 445174.842, "p75_seconds": 624995.3195, "p90_seconds": 770926.678, "share_under_60s": 0.0},
    {"source": "prepare drinks", "target": "Bring food", "n": 116, "p10_seconds": 39201.883, "p25_seconds": 406754.14800000004, "median_seconds": 785107.1799999999, "p75_seconds": 1309999.6535, "p90_seconds": 1599036.5720000002, "share_under_60s": 0.009},
    {"source": "Review drinks", "target": "Bring food", "n": 111, "p10_seconds": 161178.253, "p25_seconds": 593624.664, "median_seconds": 857938.338, "p75_seconds": 1377516.8425, "p90_seconds": 1634321.944, "share_under_60s": 0.009},
    {"source": "Take order", "target": "Review drinks", "n": 106, "p10_seconds": 257.7665, "p25_seconds": 389.71375, "median_seconds": 707.2185, "p75_seconds": 1038.79475, "p90_seconds": 1354.652, "share_under_60s": 0.0},
    {"source": "Prepare starter", "target": "Review drinks", "n": 97, "p10_seconds": 11.519400000000001, "p25_seconds": 22.425, "median_seconds": 56.303, "p75_seconds": 132.229, "p90_seconds": 367.86520000000047, "share_under_60s": 0.505},
    {"source": "Bring drinks", "target": "Signal food complete", "n": 95, "p10_seconds": 179812.44900000002, "p25_seconds": 593579.078, "median_seconds": 883902.239, "p75_seconds": 1299383.9915, "p90_seconds": 1510712.6246000004, "share_under_60s": 0.0},
    {"source": "prepare drinks", "target": "Signal food complete", "n": 90, "p10_seconds": 84583.4036, "p25_seconds": 431151.8905, "median_seconds": 982509.744, "p75_seconds": 1344037.3537499998, "p90_seconds": 1629650.5926, "share_under_60s": 0.011},
    {"source": "Bring drinks", "target": "Prepare starter", "n": 88, "p10_seconds": 4.3939, "p25_seconds": 15.899249999999999, "median_seconds": 33.688, "p75_seconds": 75.99025, "p90_seconds": 139.0768, "share_under_60s": 0.693},
    {"source": "Review order", "target": "Review drinks", "n": 83, "p10_seconds": 303.721, "p25_seconds": 381.153, "median_seconds": 593.835, "p75_seconds": 954.486, "p90_seconds": 1402.8865999999996, "share_under_60s": 0.0},
    {"source": "Bring drinks", "target": "Bring food", "n": 81, "p10_seconds": 102484.185, "p25_seconds": 366203.855, "median_seconds": 696631.885, "p75_seconds": 1367714.419, "p90_seconds": 1616933.672, "share_under_60s": 0.0},
    {"source": "Prepare starter", "target": "Signal food complete", "n": 80, "p10_seconds": 88210.8159, "p25_seconds": 276729.8375, "median_seconds": 519007.66500000004, "p75_seconds": 688787.4967499999, "p90_seconds": 944907.5365, "share_under_60s": 0.0},
    {"source": "start", "target": "Bring drinks", "n": 62, "p10_seconds": 275.2757, "p25_seconds": 422.42224999999996, "median_seconds": 695.196, "p75_seconds": 915.41325, "p90_seconds": 1413.8303999999998, "share_under_60s": 0.0},
    {"source": "Review order", "target": "Prepare main course", "n": 61, "p10_seconds": 943.183, "p25_seconds": 84467.297, "median_seconds": 444605.508, "p75_seconds": 620958.677, "p90_seconds": 700252.983, "share_under_60s": 0.0},
    {"source": "Take order", "target": "Bring drinks", "n": 60, "p10_seconds": 296.2082, "p25_seconds": 433.589, "median_seconds": 624.287, "p75_seconds": 992.78675, "p90_seconds": 1341.9016, "share_under_60s": 0.0},
    {"source": "Take order", "target": "Prepare main course", "n": 57, "p10_seconds": 101957.3608, "p25_seconds": 258966.497, "median_seconds": 513968.537, "p75_seconds": 624777.344, "p90_seconds": 777773.7006, "share_under_60s": 0.0},
    {"source": "prepare drinks", "target": "Prepare main course", "n": 51, "p10_seconds": 71.18, "p25_seconds": 281.63, "median_seconds": 160626.596, "p75_seconds": 523929.66000000003, "p90_seconds": 681005.825, "share_under_60s": 0.078},
    {"source": "Review order", "target": "Signal food complete", "n": 49, "p10_seconds": 152688.05200000005, "p25_seconds": 587731.024, "median_seconds": 862530.877, "p75_seconds": 1300062.102, "p90_seconds": 1476447.8158, "share_under_60s": 0.0},
    {"source": "start", "target": "Review drinks", "n": 46, "p10_seconds": 281.845, "p25_seconds": 455.605, "median_seconds": 733.1669999999999, "p75_seconds": 1084.1665, "p90_seconds": 1306.389, "share_under_60s": 0.0},
    {"source": "prepare drinks", "target": "Review drinks", "n": 46, "p10_seconds": 368.894, "p25_seconds": 508.01550000000003, "median_seconds": 656.5765, "p75_seconds": 1133.5845, "p90_seconds": 1545.7044999999998, "share_under_60s": 0.0},
    {"source": "Review order", "target": "Bring drinks", "n": 40, "p10_seconds": 298.81200000000007, "p25_seconds": 538.94575, "median_seconds": 784.107, "p75_seconds": 1205.8220000000001, "p90_seconds": 1648.7174, "share_under_60s": 0.0},
    {"source": "Bring drinks", "target": "Bring drinks", "n": 34, "p10_seconds": 237.8929, "p25_seconds": 350.94375, "median_seconds": 571.9159999999999, "p75_seconds": 863.97725, "p90_seconds": 1168.7469999999998, "share_under_60s": 0.0},
    {"source": "Review drinks", "target": "Review drinks", "n": 30, "p10_seconds": 283.804, "p25_seconds": 471.11474999999996, "median_seconds": 695.4390000000001, "p75_seconds": 1119.0275, "p90_seconds": 2202.1487, "share_under_60s": 0.0},
    {"source": "start", "target": "prepare drinks", "n": 25, "p10_seconds": 255.112, "p25_seconds": 319.999, "median_seconds": 413.649, "p75_seconds": 582.288, "p90_seconds": 738.0338, "share_under_60s": 0.0},
    {"source": "Bring drinks", "target": "Prepare main course", "n": 25, "p10_seconds": 53.07200000000001, "p25_seconds": 21489.537, "median_seconds": 438128.715, "p75_seconds": 669828.116, "p90_seconds": 685551.208, "share_under_60s": 0.12},
    {"source": "Take order", "target": "prepare drinks", "n": 24, "p10_seconds": 214.6615, "p25_seconds": 304.47475, "median_seconds": 736.587, "p75_seconds": 1131.26225, "p90_seconds": 1339.0379, "share_under_60s": 0.0},
    {"source": "Prepare main course", "target": "Bring drinks", "n": 18, "p10_seconds": 28.227300000000003, "p25_seconds": 68.60625, "median_seconds": 129.8, "p75_seconds": 190.83625, "p90_seconds": 340.4634000000001, "share_under_60s": 0.222},
    {"source": "Review drinks", "target": "Deliver to customer", "n": 14, "p10_seconds": 13.1735, "p25_seconds": 26.957, "median_seconds": 70.7255, "p75_seconds": 122.75300000000001, "p90_seconds": 151.1391, "share_under_60s": 0.357},
    {"source": "prepare drinks", "target": "prepare drinks", "n": 14, "p10_seconds": 430.3684, "p25_seconds": 495.2955, "median_seconds": 639.1405, "p75_seconds": 705.744, "p90_seconds": 1295.039900000001, "share_under_60s": 0.0},
    {"source": "start", "target": "Prepare starter", "n": 13, "p10_seconds": 275.65099999999995, "p25_seconds": 370.046, "median_seconds": 404.992, "p75_seconds": 456.3, "p90_seconds": 517.9136, "share_under_60s": 0.0},
    {"source": "Take order", "target": "Prepare starter", "n": 13, "p10_seconds": 327.2842, "p25_seconds": 389.43, "median_seconds": 558.771, "p75_seconds": 1311.424, "p90_seconds": 2316.6824000000006, "share_under_60s": 0.0},
    {"source": "Prepare main course", "target": "prepare drinks", "n": 12, "p10_seconds": 94.8512, "p25_seconds": 110.78, "median_seconds": 178.7615, "p75_seconds": 259.93625, "p90_seconds": 354.04850000000005, "share_under_60s": 0.0},
    {"source": "Bring food", "target": "Bring drinks", "n": 11, "p10_seconds": 50.942, "p25_seconds": 152.058, "median_seconds": 229.533, "p75_seconds": 512.9984999999999, "p90_seconds": 626.086, "share_under_60s": 0.182},
    {"source": "Review drinks", "target": "Bring drinks", "n": 10, "p10_seconds": 434.9222, "p25_seconds": 666.72275, "median_seconds": 1267.9125, "p75_seconds": 1729.7255, "p90_seconds": 1950.6003999999998, "share_under_60s": 0.0},
    {"source": "Prepare starter", "target": "Bring food", "n": 10, "p10_seconds": 167409.7799, "p25_seconds": 293248.85774999997, "median_seconds": 525665.88, "p75_seconds": 931247.942, "p90_seconds": 1290839.1252, "share_under_60s": 0.0},
    {"source": "Bring drinks", "target": "prepare drinks", "n": 7, "p10_seconds": 691.6136, "p25_seconds": 983.6489999999999, "median_seconds": 1736.597, "p75_seconds": 2019.6970000000001, "p90_seconds": 2129.3134, "share_under_60s": 0.0},
    {"source": "Signal food complete", "target": "prepare drinks", "n": 4, "p10_seconds": 65.31860000000002, "p25_seconds": 153.056, "median_seconds": 228.6855, "p75_seconds": 267.4535, "p90_seconds": 288.84020000000004, "share_under_60s": 0.25},
    {"source": "Prepare main course", "target": "Review drinks", "n": 4, "p10_seconds": 2.2898, "p25_seconds": 3.6739999999999995, "median_seconds": 35.4335, "p75_seconds": 86.17125000000001, "p90_seconds": 121.71630000000002, "share_under_60s": 0.5},
    {"source": "Bring food", "target": "prepare drinks", "n": 3, "p10_seconds": 164.92839999999998, "p25_seconds": 166.546, "median_seconds": 169.242, "p75_seconds": 176.463, "p90_seconds": 180.7956, "share_under_60s": 0.0},
    {"source": "Signal food complete", "target": "Bring drinks", "n": 3, "p10_seconds": 150.102, "p25_seconds": 179.118, "median_seconds": 227.478, "p75_seconds": 255.185, "p90_seconds": 271.8092, "share_under_60s": 0.0},
]

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
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
]

GLOBAL_MEDIAN_DURATION_SECONDS = 117.1945


def build_maps():
    edge_prob = {}
    for x in TRANSITION_PROBABILITIES_TOP:
        edge_prob[(x["source"], x["target"])] = x["probability"]

    edge_dur = {}
    for x in DURATION_PROFILES_TOP:
        edge_dur[(x["source"], x["target"])] = x
    return edge_prob, edge_dur


EDGE_PROB, EDGE_DUR = build_maps()


def infer_precision_and_format(ts_series):
    sample = None
    for s in ts_series:
        if isinstance(s, str) and s:
            sample = s
            break
    frac_digits = 0
    if sample and "." in sample:
        frac = sample.split(".")[-1]
        frac_digits = len(frac)
    unit_ns = 1_000_000_000
    if frac_digits > 0:
        unit_ns = 10 ** (9 - frac_digits)
    return frac_digits, unit_ns


def format_ts(ts, frac_digits):
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits <= 0:
        return base
    ns = ts.nanosecond
    micro = ts.microsecond
    total_ns = micro * 1000 + ns
    frac = f"{total_ns:09d}"[:frac_digits]
    return f"{base}.{frac}"


def subsequence_contiguous_score(seq, left_act, right_act):
    best = 0
    ext = []
    if left_act is not None:
        ext.append(left_act)
    ext.extend(seq)
    if right_act is not None:
        ext.append(right_act)
    m = len(seq)
    for v in FREQUENT_VARIANTS:
        for start in range(len(v) - m + 1):
            if v[start:start + m] == list(seq):
                best = max(best, 50)
        for start in range(len(v) - len(ext) + 1):
            if v[start:start + len(ext)] == ext:
                best = max(best, 120)
    return best


def multiset_variant_orders(acts):
    target = Counter(acts)
    out = []
    seen = set()
    for v in FREQUENT_VARIANTS:
        n = len(acts)
        for i in range(len(v) - n + 1):
            chunk = v[i:i + n]
            if Counter(chunk) == target:
                tup = tuple(chunk)
                if tup not in seen:
                    out.append(list(chunk))
                    seen.add(tup)
    return out


def edge_exists(a, b):
    return (a, b) in EDGE_PROB


def edge_score(a, b):
    if (a, b) not in EDGE_PROB:
        return -9.0
    p = EDGE_PROB[(a, b)]
    return math.log(max(p, 1e-12)) * 10.0 + 20.0


def concurrency_small_gap(a, b):
    p1 = EDGE_PROB.get((a, b), 0.0)
    p2 = EDGE_PROB.get((b, a), 0.0)
    if p1 > 0 and p2 > 0:
        ratio = min(p1, p2) / max(p1, p2)
        if ratio >= 0.5:
            return True
    return False


def duration_gap_seconds(a, b, precision_seconds):
    prof = EDGE_DUR.get((a, b))
    if prof is None:
        return max(precision_seconds * 1.7, 0.137)
    if concurrency_small_gap(a, b):
        return max(precision_seconds * 1.3, 0.113)
    share = prof["share_under_60s"]
    if share >= 0.5:
        g = prof["p10_seconds"]
    else:
        g = prof["p25_seconds"]
    if g <= 0:
        g = precision_seconds * 1.5
    return max(g, precision_seconds * 1.1)


def permutations_or_candidates(acts):
    n = len(acts)
    if n <= 7:
        return [list(p) for p in itertools.permutations(acts)]
    cands = multiset_variant_orders(acts)
    if cands:
        return cands[:200]
    uniq = []
    seen = set()
    for p in itertools.permutations(acts[:7]):
        candidate = list(p) + acts[7:]
        t = tuple(candidate)
        if t not in seen:
            uniq.append(candidate)
            seen.add(t)
        if len(uniq) >= 200:
            break
    return uniq


def score_order(order, left_act, right_act):
    s = subsequence_contiguous_score(order, left_act, right_act)

    internal_missing = 0
    for a, b in zip(order[:-1], order[1:]):
        if edge_exists(a, b):
            s += edge_score(a, b)
        else:
            internal_missing += 1
            s -= 200

    if left_act is not None:
        if edge_exists(left_act, order[0]):
            s += edge_score(left_act, order[0]) * 0.6 + 8
        else:
            s -= 20
    if right_act is not None:
        if edge_exists(order[-1], right_act):
            s += edge_score(order[-1], right_act) * 0.6 + 8
        else:
            s -= 20

    if internal_missing == 0:
        s += 50
    return s


def choose_order(activities, left_act, right_act):
    candidates = []
    var_orders = multiset_variant_orders(activities)
    if var_orders:
        candidates.extend(var_orders)
    candidates.extend(permutations_or_candidates(activities))

    best = None
    best_score = -10**18
    seen = set()
    for cand in candidates:
        t = tuple(cand)
        if t in seen:
            continue
        seen.add(t)
        sc = score_order(cand, left_act, right_act)
        if sc > best_score:
            best_score = sc
            best = cand

    if best is None:
        best = list(activities)
    return best


def make_nonuniform_positive_gaps(order, precision_seconds):
    gaps = []
    for i, (a, b) in enumerate(zip(order[:-1], order[1:])):
        g = duration_gap_seconds(a, b, precision_seconds)
        wiggle = 1.0 + 0.03 * ((i % 3) - 1)
        g = max(g * wiggle, precision_seconds * (1.05 + 0.01 * i))
        gaps.append(g)
    return gaps


def allocate_times_forward(observed_ts, right_anchor_ts, case_end_ts, k, order, frac_digits):
    precision_seconds = 10 ** (-frac_digits) if frac_digits > 0 else 1.0
    unit_ns = int(round(precision_seconds * 1_000_000_000))
    if unit_ns <= 0:
        unit_ns = 1

    upper = right_anchor_ts if right_anchor_ts is not None else case_end_ts
    room_ns = max(0, int((upper.value - observed_ts.value)))
    min_required_ns = (k - 1) * unit_ns

    gaps_sec = make_nonuniform_positive_gaps(order, precision_seconds)
    gaps_ns = [max(unit_ns, int(round(g * 1_000_000_000))) for g in gaps_sec]

    if k == 1:
        return [observed_ts], False

    changed = False
    if room_ns < min_required_ns:
        changed = True
        room_ns = min_required_ns

    target_total_ns = sum(gaps_ns)
    max_total_ns = room_ns

    if target_total_ns > max_total_ns:
        changed = True
        if target_total_ns <= 0:
            gaps_ns = [unit_ns] * (k - 1)
        else:
            scaled = []
            for i, g in enumerate(gaps_ns):
                weight = g / target_total_ns
                val = max(unit_ns, int(math.floor(weight * max_total_ns)))
                scaled.append(val)
            gaps_ns = scaled

            current = sum(gaps_ns)
            deficit = max_total_ns - current
            i = 0
            while deficit > 0 and len(gaps_ns) > 0:
                add = min(deficit, unit_ns * (1 + (i % 2)))
                gaps_ns[i % len(gaps_ns)] += add
                deficit -= add
                i += 1

            excess = sum(gaps_ns) - max_total_ns
            i = len(gaps_ns) - 1
            while excess > 0 and i >= 0:
                removable = max(0, gaps_ns[i] - unit_ns)
                rem = min(removable, excess)
                gaps_ns[i] -= rem
                excess -= rem
                i -= 1

    if sum(gaps_ns) < min_required_ns:
        changed = True
        base = [unit_ns + ((i % 3) * unit_ns) for i in range(k - 1)]
        total = sum(base)
        if total > max_total_ns and max_total_ns >= min_required_ns:
            base = [unit_ns] * (k - 1)
            extra = max_total_ns - min_required_ns
            i = 0
            while extra > 0:
                add = min(extra, unit_ns * (1 + (i % 2)))
                base[i % len(base)] += add
                extra -= add
                i += 1
        gaps_ns = base

    times = [observed_ts]
    cur = observed_ts.value
    for g in gaps_ns:
        cur += g
        times.append(pd.Timestamp(cur))
    return times, changed


def force_within_bounds_strict(times, lower, upper, frac_digits):
    precision_seconds = 10 ** (-frac_digits) if frac_digits > 0 else 1.0
    unit_ns = int(round(precision_seconds * 1_000_000_000))
    if unit_ns <= 0:
        unit_ns = 1
    changed = False

    vals = [t.value for t in times]
    if lower is not None:
        low = lower.value
        if vals[0] < low:
            changed = True
            vals[0] = low
            for i in range(1, len(vals)):
                vals[i] = max(vals[i], vals[i - 1] + unit_ns)

    if upper is not None:
        up = upper.value
        if vals[-1] > up:
            changed = True
            vals[-1] = up
            for i in range(len(vals) - 2, -1, -1):
                vals[i] = min(vals[i], vals[i + 1] - unit_ns)

    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            changed = True
            vals[i] = vals[i - 1] + unit_ns

    if upper is not None and vals[-1] > upper.value:
        changed = True
        span_needed = (len(vals) - 1) * unit_ns
        start = min(vals[0], upper.value - span_needed)
        vals = [start + i * unit_ns for i in range(len(vals))]

    if lower is not None and vals[0] < lower.value:
        changed = True
        vals = [max(v, lower.value + i * unit_ns) for i, v in enumerate([vals[0]] + vals[1:])]
        vals = [vals[0]] + [max(vals[i], vals[i-1] + unit_ns) for i in range(1, len(vals))]

    return [pd.Timestamp(v) for v in vals], changed


def find_blocks(df):
    blocks = []
    i = 0
    n = len(df)
    while i < n:
        if df.at[i, "error_category"] != "CLEAN":
            case = df.at[i, "Case"]
            ts = df.at[i, "Timestamp"]
            j = i + 1
            while j < n and df.at[j, "Case"] == case and df.at[j, "error_category"] != "CLEAN" and df.at[j, "Timestamp"] == ts:
                j += 1
            if j - i >= 2:
                blocks.append((i, j - 1))
            i = j
        else:
            i += 1
    return blocks


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
    frac_digits, unit_ns = infer_precision_and_format(df["Timestamp"].tolist())
    precision_seconds = 10 ** (-frac_digits) if frac_digits > 0 else 1.0

    parsed_ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_parsed_ts"] = parsed_ts
    df["_out_ts"] = df["Timestamp"]

    blocks = find_blocks(df)
    blocks_repaired = 0
    blocks_clamped = 0

    for start, end in blocks:
        try:
            block = df.iloc[start:end + 1].copy()
            case = block.iloc[0]["Case"]
            observed_str = block.iloc[0]["Timestamp"]
            observed_ts = pd.to_datetime(observed_str)
            idxs = list(range(start, end + 1))
            activities = block["Activity"].tolist()

            left_idx = start - 1
            left_act = None
            left_ts = None
            while left_idx >= 0 and df.at[left_idx, "Case"] == case:
                if df.at[left_idx, "error_category"] == "CLEAN":
                    left_act = df.at[left_idx, "Activity"]
                    left_ts = df.at[left_idx, "_parsed_ts"]
                    break
                left_idx -= 1

            right_idx = end + 1
            right_act = None
            right_ts = None
            while right_idx < len(df) and df.at[right_idx, "Case"] == case:
                if df.at[right_idx, "error_category"] == "CLEAN":
                    right_act = df.at[right_idx, "Activity"]
                    right_ts = df.at[right_idx, "_parsed_ts"]
                    break
                right_idx += 1

            case_mask = df["Case"] == case
            case_ts = df.loc[case_mask, "_parsed_ts"].dropna()
            case_start = case_ts.min()
            case_end = case_ts.max()

            order = choose_order(activities, left_act, right_act)
            times_exec, clamped1 = allocate_times_forward(observed_ts, right_ts, case_end, len(order), order, frac_digits)

            lower_bound = observed_ts
            if left_ts is not None and lower_bound < left_ts:
                lower_bound = left_ts
            upper_bound = right_ts if right_ts is not None else case_end

            times_exec, clamped2 = force_within_bounds_strict(times_exec, lower_bound, upper_bound, frac_digits)

            if times_exec[0] != observed_ts:
                times_exec[0] = observed_ts
                times_exec, clamped3 = force_within_bounds_strict(times_exec, observed_ts, upper_bound, frac_digits)
            else:
                clamped3 = False

            if any(times_exec[i] >= times_exec[i + 1] for i in range(len(times_exec) - 1)):
                vals = [observed_ts.value + i * unit_ns for i in range(len(times_exec))]
                if upper_bound is not None and vals[-1] > upper_bound.value:
                    start_val = max(observed_ts.value, upper_bound.value - (len(times_exec) - 1) * unit_ns)
                    vals = [start_val + i * unit_ns for i in range(len(times_exec))]
                    vals[0] = observed_ts.value
                    for i in range(1, len(vals)):
                        vals[i] = max(vals[i], vals[i-1] + unit_ns)
                times_exec = [pd.Timestamp(v) for v in vals]
                clamped4 = True
            else:
                clamped4 = False

            activity_to_positions = defaultdict(list)
            for pos, act in enumerate(order):
                activity_to_positions[act].append(pos)

            row_to_execpos = {}
            temp_count = defaultdict(int)
            for idx in idxs:
                act = df.at[idx, "Activity"]
                c = temp_count[act]
                row_to_execpos[idx] = activity_to_positions[act][c]
                temp_count[act] += 1

            for idx in idxs:
                exec_pos = row_to_execpos[idx]
                out_str = format_ts(times_exec[exec_pos], frac_digits)
                df.at[idx, "_out_ts"] = out_str

            blocks_repaired += 1
            if clamped1 or clamped2 or clamped3 or clamped4:
                blocks_clamped += 1

        except Exception:
            try:
                observed_ts = pd.to_datetime(df.at[start, "Timestamp"])
                case = df.at[start, "Case"]
                case_mask = df["Case"] == case
                case_end = df.loc[case_mask, "_parsed_ts"].dropna().max()
                right_idx = end + 1
                right_ts = None
                while right_idx < len(df) and df.at[right_idx, "Case"] == case:
                    if df.at[right_idx, "error_category"] == "CLEAN":
                        right_ts = df.at[right_idx, "_parsed_ts"]
                        break
                    right_idx += 1
                upper = right_ts if right_ts is not None else case_end
                k = end - start + 1
                vals = [observed_ts.value + i * unit_ns for i in range(k)]
                if upper is not None and vals[-1] > upper.value:
                    base = min(observed_ts.value, upper.value - (k - 1) * unit_ns)
                    vals = [base + i * unit_ns for i in range(k)]
                    vals[0] = observed_ts.value
                    for i in range(1, k):
                        vals[i] = max(vals[i], vals[i-1] + unit_ns)
                for off, idx in enumerate(range(start, end + 1)):
                    df.at[idx, "_out_ts"] = format_ts(pd.Timestamp(vals[off]), frac_digits)
                blocks_repaired += 1
                blocks_clamped += 1
            except Exception:
                pass

    rows_equal_to_input = 0

    for start, end in blocks:
        idxs = list(range(start, end + 1))
        observed = orig.at[start, "Timestamp"]
        out_vals = [df.at[idx, "_out_ts"] for idx in idxs]

        parsed = [pd.to_datetime(x) for x in out_vals]
        acts = [df.at[idx, "Activity"] for idx in idxs]

        left_idx = start - 1
        left_ts = None
        right_ts = None
        case = df.at[start, "Case"]
        while left_idx >= 0 and df.at[left_idx, "Case"] == case:
            if df.at[left_idx, "error_category"] == "CLEAN":
                left_ts = pd.to_datetime(df.at[left_idx, "Timestamp"])
                break
            left_idx -= 1
        right_idx = end + 1
        while right_idx < len(df) and df.at[right_idx, "Case"] == case:
            if df.at[right_idx, "error_category"] == "CLEAN":
                right_ts = pd.to_datetime(df.at[right_idx, "Timestamp"])
                break
            right_idx += 1
        case_mask = df["Case"] == case
        case_ts = pd.to_datetime(df.loc[case_mask, "Timestamp"], errors="coerce")
        case_start = case_ts.min()
        case_end = case_ts.max()

        order = choose_order(acts, df.at[left_idx, "Activity"] if left_ts is not None else None, df.at[right_idx, "Activity"] if right_ts is not None and right_idx < len(df) else None)
        activity_to_positions = defaultdict(list)
        for pos, act in enumerate(order):
            activity_to_positions[act].append(pos)
        temp_count = defaultdict(int)
        exec_positions = []
        for idx in idxs:
            act = df.at[idx, "Activity"]
            c = temp_count[act]
            exec_positions.append(activity_to_positions[act][c])
            temp_count[act] += 1

        exec_time = {}
        for idx, ep, pt in zip(idxs, exec_positions, parsed):
            exec_time[ep] = pt

        ordered_times = [exec_time[i] for i in range(len(idxs))]
        fix_needed = False

        for i in range(len(ordered_times) - 1):
            if not (ordered_times[i] < ordered_times[i + 1]):
                fix_needed = True

        count_observed = sum(1 for x in out_vals if x == observed)
        if count_observed != 1:
            fix_needed = True
        else:
            obs_row_pos = [i for i, x in enumerate(out_vals) if x == observed][0]
            if exec_positions[obs_row_pos] != 0:
                fix_needed = True

        room = None
        if right_ts is not None:
            room = int((right_ts.value - pd.to_datetime(observed).value))
            span = int((max(parsed).value - min(parsed).value))
            if span > room:
                fix_needed = True
        span_sec = (max(parsed).value - min(parsed).value) / 1e9
        if len(idxs) >= 2 and span_sec < (len(idxs) - 1) * 1.0:
            fix_needed = True

        for pt in parsed:
            if pt < case_start or pt > case_end:
                fix_needed = True
            if left_ts is not None and right_ts is not None:
                if pt < left_ts or pt > right_ts:
                    fix_needed = True

        if fix_needed:
            acts2 = [df.at[idx, "Activity"] for idx in idxs]
            left_act = df.at[left_idx, "Activity"] if left_ts is not None else None
            right_act = df.at[right_idx, "Activity"] if right_ts is not None and right_idx < len(df) else None
            order2 = choose_order(acts2, left_act, right_act)
            observed_ts = pd.to_datetime(observed)
            times_exec, _ = allocate_times_forward(observed_ts, right_ts, case_end, len(order2), order2, frac_digits)
            times_exec, _ = force_within_bounds_strict(times_exec, observed_ts, right_ts if right_ts is not None else case_end, frac_digits)

            activity_to_positions = defaultdict(list)
            for pos, act in enumerate(order2):
                activity_to_positions[act].append(pos)
            temp_count = defaultdict(int)
            for idx in idxs:
                act = df.at[idx, "Activity"]
                c = temp_count[act]
                ep = activity_to_positions[act][c]
                temp_count[act] += 1
                df.at[idx, "_out_ts"] = format_ts(times_exec[ep], frac_digits)
            blocks_clamped += 1

    for i in range(len(df)):
        if df.at[i, "_out_ts"] == orig.at[i, "Timestamp"]:
            rows_equal_to_input += 1

    out = orig.copy(deep=True)
    out["Timestamp"] = df["_out_ts"]

    try:
        out.to_csv(args.output, index=False)
    except Exception:
        out.to_csv(args.output, index=False)

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={blocks_clamped} rows_still_equal_to_input={rows_equal_to_input}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0")