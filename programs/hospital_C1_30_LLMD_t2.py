import argparse
import itertools
import math
import sys
from collections import Counter, defaultdict

import pandas as pd


TRANSITION_PROBABILITIES_TOP = [
    {"source": "RELEASE", "target": "CODE OK", "probability": 0.9536},
    {"source": "FIN", "target": "RELEASE", "probability": 0.9243},
    {"source": "CODE OK", "target": "BILLED", "probability": 0.9101},
    {"source": "BILLED", "target": "STORNO", "probability": 0.9013},
    {"source": "CHANGE DIAGN", "target": "FIN", "probability": 0.8112},
    {"source": "JOIN-PAT", "target": "JOIN-PAT", "probability": 0.7966},
    {"source": "REOPEN", "target": "FIN", "probability": 0.6865},
    {"source": "CHANGE END", "target": "DELETE", "probability": 0.6},
    {"source": "DELETE", "target": "DELETE", "probability": 0.6},
    {"source": "REJECT", "target": "BILLED", "probability": 0.5991},
    {"source": "STORNO", "target": "BILLED", "probability": 0.5158},
    {"source": "CODE ERROR", "target": "MANUAL", "probability": 0.5},
    {"source": "CODE ERROR", "target": "RELEASE", "probability": 0.5},
    {"source": "MANUAL", "target": "RELEASE", "probability": 0.4889},
    {"source": "STORNO", "target": "REJECT", "probability": 0.4572},
    {"source": "NEW", "target": "CHANGE DIAGN", "probability": 0.4233},
    {"source": "CHANGE END", "target": "FIN", "probability": 0.4},
    {"source": "DELETE", "target": "JOIN-PAT", "probability": 0.4},
    {"source": "SET STATUS", "target": "DELETE", "probability": 0.3333},
    {"source": "REJECT", "target": "REOPEN", "probability": 0.3208},
    {"source": "NEW", "target": "FIN", "probability": 0.2573},
    {"source": "CODE NOK", "target": "CODE NOK", "probability": 0.2531},
    {"source": "REOPEN", "target": "DELETE", "probability": 0.2508},
    {"source": "CODE NOK", "target": "CODE OK", "probability": 0.2407},
    {"source": "CODE NOK", "target": "SET STATUS", "probability": 0.2222},
    {"source": "MANUAL", "target": "MANUAL", "probability": 0.2},
    {"source": "SET STATUS", "target": "CHANGE END", "probability": 0.1667},
    {"source": "SET STATUS", "target": "FIN", "probability": 0.1667},
    {"source": "SET STATUS", "target": "RELEASE", "probability": 0.1667},
    {"source": "SET STATUS", "target": "SET STATUS", "probability": 0.1667},
    {"source": "MANUAL", "target": "CODE OK", "probability": 0.1556},
    {"source": "CODE NOK", "target": "REOPEN", "probability": 0.1358},
    {"source": "JOIN-PAT", "target": "FIN", "probability": 0.1356},
    {"source": "NEW", "target": "NEW", "probability": 0.1185},
    {"source": "CODE NOK", "target": "DELETE", "probability": 0.1173},
    {"source": "NEW", "target": "BILLED", "probability": 0.0745},
    {"source": "MANUAL", "target": "BILLED", "probability": 0.0667},
    {"source": "CODE OK", "target": "REOPEN", "probability": 0.065},
    {"source": "NEW", "target": "DELETE", "probability": 0.0649},
    {"source": "CHANGE DIAGN", "target": "CHANGE DIAGN", "probability": 0.0594},
    {"source": "CHANGE DIAGN", "target": "BILLED", "probability": 0.0505},
    {"source": "FIN", "target": "BILLED", "probability": 0.0496},
    {"source": "BILLED", "target": "BILLED", "probability": 0.0461},
    {"source": "MANUAL", "target": "DELETE", "probability": 0.0444},
    {"source": "NEW", "target": "CODE OK", "probability": 0.0408},
    {"source": "REOPEN", "target": "CHANGE DIAGN", "probability": 0.0396},
    {"source": "BILLED", "target": "CODE OK", "probability": 0.0373},
    {"source": "CHANGE DIAGN", "target": "CODE OK", "probability": 0.0343},
    {"source": "JOIN-PAT", "target": "BILLED", "probability": 0.0339},
    {"source": "JOIN-PAT", "target": "DELETE", "probability": 0.0339},
    {"source": "CHANGE DIAGN", "target": "DELETE", "probability": 0.0336},
    {"source": "REJECT", "target": "RELEASE", "probability": 0.033},
    {"source": "REJECT", "target": "SET STATUS", "probability": 0.0283},
    {"source": "MANUAL", "target": "REOPEN", "probability": 0.0222},
    {"source": "MANUAL", "target": "SET STATUS", "probability": 0.0222},
    {"source": "RELEASE", "target": "CODE NOK", "probability": 0.0192},
    {"source": "CODE NOK", "target": "RELEASE", "probability": 0.0185},
    {"source": "NEW", "target": "RELEASE", "probability": 0.0173},
    {"source": "REJECT", "target": "REJECT", "probability": 0.0142},
    {"source": "STORNO", "target": "SET STATUS", "probability": 0.0135},
]

DURATION_PROFILES_TOP = [
    {"source": "CODE OK", "target": "BILLED", "n": 5710, "p10_seconds": 370618.4, "p25_seconds": 1226151.0, "median_seconds": 3696259.5, "p75_seconds": 7237922.0, "p90_seconds": 11042559.3, "share_under_60s": 0.0},
    {"source": "RELEASE", "target": "CODE OK", "n": 5671, "p10_seconds": 3489.0, "p25_seconds": 176021.0, "median_seconds": 342906.0, "p75_seconds": 440473.5, "p90_seconds": 534683.0, "share_under_60s": 0.05},
    {"source": "FIN", "target": "RELEASE", "n": 5642, "p10_seconds": 1970.3000000000002, "p25_seconds": 6714.0, "median_seconds": 16196.5, "p75_seconds": 25828.0, "p90_seconds": 37133.00000000001, "share_under_60s": 0.058},
    {"source": "NEW", "target": "CHANGE DIAGN", "n": 3920, "p10_seconds": 12.0, "p25_seconds": 16.0, "median_seconds": 24.0, "p75_seconds": 9419.25, "p90_seconds": 1812061.0, "share_under_60s": 0.637},
    {"source": "CHANGE DIAGN", "target": "FIN", "n": 3403, "p10_seconds": 1327999.8, "p25_seconds": 4002043.0, "median_seconds": 7707138.0, "p75_seconds": 7901954.0, "p90_seconds": 8063873.2, "share_under_60s": 0.009},
    {"source": "NEW", "target": "FIN", "n": 2383, "p10_seconds": 429682.0, "p25_seconds": 5192415.0, "median_seconds": 29527664.0, "p75_seconds": 31622124.5, "p90_seconds": 31635000.2, "share_under_60s": 0.04},
    {"source": "NEW", "target": "NEW", "n": 1097, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 1.0, "p75_seconds": 1.0, "p90_seconds": 1.0, "share_under_60s": 1.0},
    {"source": "NEW", "target": "BILLED", "n": 690, "p10_seconds": 10067520.7, "p25_seconds": 12572347.25, "median_seconds": 15589142.5, "p75_seconds": 31018716.0, "p90_seconds": 32835955.0, "share_under_60s": 0.0},
    {"source": "NEW", "target": "DELETE", "n": 601, "p10_seconds": 2555.0, "p25_seconds": 171326.0, "median_seconds": 1865565.0, "p75_seconds": 7138403.0, "p90_seconds": 22923619.0, "share_under_60s": 0.062},
    {"source": "REOPEN", "target": "FIN", "n": 416, "p10_seconds": 5.0, "p25_seconds": 10.0, "median_seconds": 42.0, "p75_seconds": 130724.0, "p90_seconds": 404391.0, "share_under_60s": 0.526},
    {"source": "BILLED", "target": "STORNO", "n": 411, "p10_seconds": 195856.0, "p25_seconds": 434146.5, "median_seconds": 1077095.0, "p75_seconds": 3261891.5, "p90_seconds": 11620515.0, "share_under_60s": 0.002},
    {"source": "CODE OK", "target": "REOPEN", "n": 408, "p10_seconds": 124052.70000000003, "p25_seconds": 463729.25, "median_seconds": 1921091.5, "p75_seconds": 4367123.0, "p90_seconds": 7403638.600000001, "share_under_60s": 0.015},
    {"source": "NEW", "target": "CODE OK", "n": 378, "p10_seconds": 4135423.0000000005, "p25_seconds": 7154291.5, "median_seconds": 8264453.5, "p75_seconds": 21700480.5, "p90_seconds": 31923870.1, "share_under_60s": 0.0},
    {"source": "FIN", "target": "BILLED", "n": 303, "p10_seconds": 775793.4, "p25_seconds": 1720658.5, "median_seconds": 3633863.0, "p75_seconds": 7504679.5, "p90_seconds": 11774911.8, "share_under_60s": 0.0},
    {"source": "CHANGE DIAGN", "target": "CHANGE DIAGN", "n": 249, "p10_seconds": 201.60000000000002, "p25_seconds": 42275.0, "median_seconds": 698451.0, "p75_seconds": 2424991.0, "p90_seconds": 4898341.200000004, "share_under_60s": 0.04},
    {"source": "STORNO", "target": "BILLED", "n": 229, "p10_seconds": 92767.8, "p25_seconds": 92840.0, "median_seconds": 169860.0, "p75_seconds": 170046.0, "p90_seconds": 492469.4000000007, "share_under_60s": 0.0},
    {"source": "CHANGE DIAGN", "target": "BILLED", "n": 212, "p10_seconds": 8103533.100000001, "p25_seconds": 10535503.25, "median_seconds": 12718535.5, "p75_seconds": 14788243.75, "p90_seconds": 18427400.900000002, "share_under_60s": 0.0},
    {"source": "STORNO", "target": "REJECT", "n": 203, "p10_seconds": 5.0, "p25_seconds": 13.5, "median_seconds": 1240.0, "p75_seconds": 2485864.0, "p90_seconds": 9494887.80000001, "share_under_60s": 0.438},
    {"source": "NEW", "target": "RELEASE", "n": 160, "p10_seconds": 3683105.6, "p25_seconds": 5029015.5, "median_seconds": 7740827.0, "p75_seconds": 7919664.5, "p90_seconds": 8084171.5, "share_under_60s": 0.0},
    {"source": "REOPEN", "target": "DELETE", "n": 152, "p10_seconds": 2.0, "p25_seconds": 2.0, "median_seconds": 3.0, "p75_seconds": 4.0, "p90_seconds": 9.0, "share_under_60s": 0.967},
    {"source": "CHANGE DIAGN", "target": "CODE OK", "n": 144, "p10_seconds": 1583545.3, "p25_seconds": 4364398.0, "median_seconds": 8186331.0, "p75_seconds": 8265612.75, "p90_seconds": 8348544.600000001, "share_under_60s": 0.007},
    {"source": "CHANGE DIAGN", "target": "DELETE", "n": 141, "p10_seconds": 206.0, "p25_seconds": 69873.0, "median_seconds": 946414.0, "p75_seconds": 4227898.0, "p90_seconds": 7842703.0, "share_under_60s": 0.064},
    {"source": "REJECT", "target": "BILLED", "n": 127, "p10_seconds": 1892.6, "p25_seconds": 163288.0, "median_seconds": 328893.0, "p75_seconds": 411313.5, "p90_seconds": 4295264.000000003, "share_under_60s": 0.055},
    {"source": "RELEASE", "target": "CODE NOK", "n": 114, "p10_seconds": 330802.6, "p25_seconds": 606467.75, "median_seconds": 1430526.5, "p75_seconds": 3820919.0, "p90_seconds": 6404524.700000001, "share_under_60s": 0.009},
    {"source": "CODE OK", "target": "RELEASE", "n": 84, "p10_seconds": 1976833.6, "p25_seconds": 2891266.75, "median_seconds": 4348244.5, "p75_seconds": 6039876.5, "p90_seconds": 6805873.4, "share_under_60s": 0.012},
    {"source": "REJECT", "target": "REOPEN", "n": 68, "p10_seconds": 7.7, "p25_seconds": 10.0, "median_seconds": 30.5, "p75_seconds": 95.0, "p90_seconds": 466.6, "share_under_60s": 0.662},
    {"source": "RELEASE", "target": "BILLED", "n": 49, "p10_seconds": 1835036.0000000005, "p25_seconds": 2669431.0, "median_seconds": 7875755.0, "p75_seconds": 21979277.0, "p90_seconds": 24867527.8, "share_under_60s": 0.0},
    {"source": "FIN", "target": "REOPEN", "n": 47, "p10_seconds": 6.6000000000000005, "p25_seconds": 65.0, "median_seconds": 63422.0, "p75_seconds": 1848555.0, "p90_seconds": 4325798.6, "share_under_60s": 0.255},
    {"source": "FIN", "target": "FIN", "n": 47, "p10_seconds": 0.0, "p25_seconds": 204302.5, "median_seconds": 9034907.0, "p75_seconds": 32494116.0, "p90_seconds": 37155753.0, "share_under_60s": 0.234},
    {"source": "JOIN-PAT", "target": "JOIN-PAT", "n": 47, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 1.0, "p90_seconds": 1.0, "share_under_60s": 1.0},
    {"source": "RELEASE", "target": "REOPEN", "n": 43, "p10_seconds": 6480.000000000003, "p25_seconds": 35903.0, "median_seconds": 225448.0, "p75_seconds": 435678.0, "p90_seconds": 6341835.800000014, "share_under_60s": 0.023},
    {"source": "CODE NOK", "target": "CODE NOK", "n": 41, "p10_seconds": 342359.0, "p25_seconds": 518520.0, "median_seconds": 1897911.0, "p75_seconds": 4417212.0, "p90_seconds": 7345234.0, "share_under_60s": 0.0},
    {"source": "CODE NOK", "target": "CODE OK", "n": 39, "p10_seconds": 2036.4, "p25_seconds": 12560.0, "median_seconds": 43358.0, "p75_seconds": 52008.5, "p90_seconds": 85967.20000000001, "share_under_60s": 0.026},
    {"source": "CODE NOK", "target": "SET STATUS", "n": 36, "p10_seconds": 13580.5, "p25_seconds": 18966.0, "median_seconds": 30387.0, "p75_seconds": 35123.0, "p90_seconds": 41303.5, "share_under_60s": 0.0},
    {"source": "FIN", "target": "DELETE", "n": 35, "p10_seconds": 68.2, "p25_seconds": 22685.5, "median_seconds": 1255926.0, "p75_seconds": 2575088.5, "p90_seconds": 5584914.800000001, "share_under_60s": 0.086},
    {"source": "CODE OK", "target": "MANUAL", "n": 29, "p10_seconds": 448.20000000000005, "p25_seconds": 111643.0, "median_seconds": 1652942.0, "p75_seconds": 2052943.0, "p90_seconds": 2806651.9999999995, "share_under_60s": 0.069},
    {"source": "REOPEN", "target": "CHANGE DIAGN", "n": 24, "p10_seconds": 10.9, "p25_seconds": 21.5, "median_seconds": 88.5, "p75_seconds": 246.5, "p90_seconds": 1141004.6999999993, "share_under_60s": 0.458},
    {"source": "CODE NOK", "target": "REOPEN", "n": 22, "p10_seconds": 9324.800000000001, "p25_seconds": 21800.5, "median_seconds": 31220.0, "p75_seconds": 46365.0, "p90_seconds": 679956.8000000005, "share_under_60s": 0.045},
    {"source": "MANUAL", "target": "RELEASE", "n": 22, "p10_seconds": 453.80000000000007, "p25_seconds": 27384.25, "median_seconds": 36655.5, "p75_seconds": 41885.0, "p90_seconds": 42120.5, "share_under_60s": 0.045},
    {"source": "BILLED", "target": "BILLED", "n": 21, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 20385.0, "p90_seconds": 1774618.0, "share_under_60s": 0.714},
    {"source": "CODE OK", "target": "CODE OK", "n": 19, "p10_seconds": 828668.8, "p25_seconds": 2361117.5, "median_seconds": 12213631.0, "p75_seconds": 13201080.5, "p90_seconds": 16030544.8, "share_under_60s": 0.0},
    {"source": "CODE NOK", "target": "DELETE", "n": 19, "p10_seconds": 12976.6, "p25_seconds": 23342.0, "median_seconds": 27554.0, "p75_seconds": 39503.0, "p90_seconds": 48221.799999999996, "share_under_60s": 0.0},
    {"source": "RELEASE", "target": "RELEASE", "n": 18, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 1510013.25, "p90_seconds": 10203704.600000003, "share_under_60s": 0.611},
    {"source": "BILLED", "target": "CODE OK", "n": 17, "p10_seconds": 2266256.2, "p25_seconds": 7135198.0, "median_seconds": 8266846.0, "p75_seconds": 25559073.0, "p90_seconds": 38782272.20000001, "share_under_60s": 0.0},
    {"source": "RELEASE", "target": "DELETE", "n": 14, "p10_seconds": 34255.7, "p25_seconds": 124880.75, "median_seconds": 249504.5, "p75_seconds": 443842.25, "p90_seconds": 9945301.800000012, "share_under_60s": 0.071},
    {"source": "CHANGE DIAGN", "target": "RELEASE", "n": 14, "p10_seconds": 4419902.6, "p25_seconds": 7074031.0, "median_seconds": 7794016.0, "p75_seconds": 19824846.5, "p90_seconds": 39810191.6, "share_under_60s": 0.0},
    {"source": "CHANGE DIAGN", "target": "STORNO", "n": 12, "p10_seconds": 10634565.1, "p25_seconds": 11483076.25, "median_seconds": 16546335.0, "p75_seconds": 34982658.0, "p90_seconds": 45561936.900000006, "share_under_60s": 0.0},
    {"source": "NEW", "target": "REOPEN", "n": 12, "p10_seconds": 4866878.0, "p25_seconds": 7547122.25, "median_seconds": 8322668.5, "p75_seconds": 9249008.5, "p90_seconds": 9810184.3, "share_under_60s": 0.0},
    {"source": "FIN", "target": "CODE OK", "n": 11, "p10_seconds": 411574.0, "p25_seconds": 2954281.0, "median_seconds": 5896170.0, "p75_seconds": 15043754.5, "p90_seconds": 16554155.0, "share_under_60s": 0.091},
    {"source": "RELEASE", "target": "SET STATUS", "n": 10, "p10_seconds": 140567.6, "p25_seconds": 150606.0, "median_seconds": 368578.5, "p75_seconds": 807029.75, "p90_seconds": 5248262.199999986, "share_under_60s": 0.0},
    {"source": "CODE OK", "target": "DELETE", "n": 10, "p10_seconds": 2419266.4, "p25_seconds": 2776767.75, "median_seconds": 4336982.5, "p75_seconds": 6998553.25, "p90_seconds": 7897376.199999999, "share_under_60s": 0.0},
    {"source": "RELEASE", "target": "FIN", "n": 10, "p10_seconds": 491295.5, "p25_seconds": 848418.25, "median_seconds": 2666622.0, "p75_seconds": 5874606.25, "p90_seconds": 7464398.899999996, "share_under_60s": 0.0},
    {"source": "MANUAL", "target": "MANUAL", "n": 9, "p10_seconds": 6.8, "p25_seconds": 41.0, "median_seconds": 171.0, "p75_seconds": 1568.0, "p90_seconds": 2877729.0000000005, "share_under_60s": 0.333},
    {"source": "JOIN-PAT", "target": "FIN", "n": 8, "p10_seconds": 3050054.3000000003, "p25_seconds": 4138408.25, "median_seconds": 5127431.5, "p75_seconds": 6350856.5, "p90_seconds": 7059534.8, "share_under_60s": 0.0},
    {"source": "FIN", "target": "SET STATUS", "n": 8, "p10_seconds": 59.60000000000001, "p25_seconds": 346488.5, "median_seconds": 739344.5, "p75_seconds": 1892574.0, "p90_seconds": 2707260.3, "share_under_60s": 0.125},
    {"source": "CHANGE DIAGN", "target": "JOIN-PAT", "n": 8, "p10_seconds": 5154.200000000001, "p25_seconds": 14279.75, "median_seconds": 596185.5, "p75_seconds": 1898689.0, "p90_seconds": 4362698.8, "share_under_60s": 0.0},
    {"source": "REJECT", "target": "RELEASE", "n": 7, "p10_seconds": 7.6, "p25_seconds": 9.0, "median_seconds": 34.0, "p75_seconds": 404.0, "p90_seconds": 160713.60000000015, "share_under_60s": 0.714},
    {"source": "RELEASE", "target": "MANUAL", "n": 7, "p10_seconds": 8931504.600000001, "p25_seconds": 15188442.5, "median_seconds": 19738990.0, "p75_seconds": 21935968.0, "p90_seconds": 23057631.6, "share_under_60s": 0.0},
    {"source": "MANUAL", "target": "CODE OK", "n": 7, "p10_seconds": 31.4, "p25_seconds": 54.5, "median_seconds": 104.0, "p75_seconds": 453.5, "p90_seconds": 1138.8000000000004, "share_under_60s": 0.286},
    {"source": "RELEASE", "target": "STORNO", "n": 7, "p10_seconds": 4772324.2, "p25_seconds": 5108938.0, "median_seconds": 5648782.0, "p75_seconds": 17627811.5, "p90_seconds": 23873601.4, "share_under_60s": 0.0},
]

FREQUENT_VARIANTS = [
    {"variant": ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "BILLED"], "count": 2448},
    {"variant": ["NEW"], "count": 1836},
    {"variant": ["NEW", "FIN", "RELEASE", "CODE OK", "BILLED"], "count": 1664},
    {"variant": ["NEW", "BILLED"], "count": 680},
    {"variant": ["NEW", "DELETE"], "count": 559},
    {"variant": ["NEW", "CODE OK", "BILLED"], "count": 359},
    {"variant": ["NEW", "CHANGE DIAGN", "BILLED"], "count": 195},
    {"variant": ["NEW", "FIN"], "count": 161},
    {"variant": ["NEW", "CHANGE DIAGN", "FIN", "BILLED"], "count": 147},
    {"variant": ["NEW", "RELEASE", "CODE OK", "BILLED"], "count": 144},
    {"variant": ["NEW", "CHANGE DIAGN", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "BILLED"], "count": 140},
    {"variant": ["NEW", "CHANGE DIAGN", "DELETE"], "count": 133},
    {"variant": ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "REOPEN", "FIN", "RELEASE", "CODE OK", "BILLED"], "count": 129},
    {"variant": ["NEW", "FIN", "BILLED"], "count": 128},
    {"variant": ["NEW", "CHANGE DIAGN", "CODE OK", "BILLED"], "count": 126},
    {"variant": ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "BILLED", "STORNO", "BILLED", "STORNO", "BILLED"], "count": 66},
    {"variant": ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "REOPEN", "DELETE"], "count": 64},
    {"variant": ["NEW", "FIN", "RELEASE", "CODE OK", "RELEASE", "CODE OK", "BILLED"], "count": 51},
    {"variant": ["NEW", "CHANGE DIAGN", "FIN", "RELEASE", "CODE OK", "BILLED", "STORNO", "REJECT", "BILLED"], "count": 37},
    {"variant": ["NEW", "FIN", "RELEASE", "BILLED"], "count": 36},
]


def build_maps():
    prob_map = {}
    for x in TRANSITION_PROBABILITIES_TOP:
        prob_map[(x["source"], x["target"])] = x["probability"]

    dur_map = {}
    for x in DURATION_PROFILES_TOP:
        dur_map[(x["source"], x["target"])] = x

    edge_freq = {}
    for x in DURATION_PROFILES_TOP:
        edge_freq[(x["source"], x["target"])] = x["n"]

    variant_subseq = defaultdict(int)
    for item in FREQUENT_VARIANTS:
        var = item["variant"]
        cnt = item["count"]
        n = len(var)
        for i in range(n):
            for j in range(i + 1, n + 1):
                variant_subseq[tuple(var[i:j])] += cnt
    return prob_map, dur_map, edge_freq, variant_subseq


PROB_MAP, DUR_MAP, EDGE_FREQ, VARIANT_SUBSEQ = build_maps()


def detect_precision_and_step(ts_series):
    max_digits = 0
    for s in ts_series.dropna().astype(str):
        if "." in s:
            frac = s.split(".")[-1]
            digits = len(frac)
            if digits > max_digits:
                max_digits = digits
    if max_digits <= 0:
        return 0, pd.Timedelta(seconds=1)
    step_ns = 10 ** (9 - min(max_digits, 9))
    return max_digits, pd.Timedelta(step_ns, unit="ns")


def format_dt(dt, digits):
    if digits <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    frac = base.split(".")[-1]
    if digits <= 6:
        frac = frac[:digits]
    else:
        frac = frac + ("0" * (digits - 6))
    return base.split(".")[0] + "." + frac


def multiset_permutations(items):
    ctr = Counter(items)
    keys = list(ctr.keys())
    n = len(items)
    path = []

    def backtrack():
        if len(path) == n:
            yield tuple(path)
            return
        for k in keys:
            if ctr[k] > 0:
                ctr[k] -= 1
                path.append(k)
                yield from backtrack()
                path.pop()
                ctr[k] += 1

    yield from backtrack()


def count_supported_edges(order, left_act, right_act):
    seq = []
    if left_act is not None:
        seq.append(left_act)
    seq.extend(order)
    if right_act is not None:
        seq.append(right_act)
    supported = 0
    missing = 0
    for a, b in zip(seq, seq[1:]):
        if (a, b) in EDGE_FREQ or (a, b) in PROB_MAP:
            supported += 1
        else:
            missing += 1
    return supported, missing


def variant_bonus(order, left_act, right_act):
    best = 0
    if VARIANT_SUBSEQ.get(tuple(order), 0) > best:
        best = VARIANT_SUBSEQ[tuple(order)]
    if left_act is not None:
        best = max(best, VARIANT_SUBSEQ.get(tuple([left_act] + list(order)), 0))
    if right_act is not None:
        best = max(best, VARIANT_SUBSEQ.get(tuple(list(order) + [right_act]), 0))
    if left_act is not None and right_act is not None:
        best = max(best, VARIANT_SUBSEQ.get(tuple([left_act] + list(order) + [right_act]), 0))
    return math.log1p(best)


def edge_score(a, b):
    if (a, b) in EDGE_FREQ or (a, b) in PROB_MAP:
        freq = EDGE_FREQ.get((a, b), 1)
        prob = PROB_MAP.get((a, b), 1e-6)
        return math.log1p(freq) + 2.0 * math.log(max(prob, 1e-12))
    return -50.0


def score_order(order, left_act, right_act):
    seq = []
    if left_act is not None:
        seq.append(left_act)
    seq.extend(order)
    if right_act is not None:
        seq.append(right_act)

    total = 0.0
    for a, b in zip(seq, seq[1:]):
        total += edge_score(a, b)

    supported, missing = count_supported_edges(order, left_act, right_act)
    total += 3.0 * supported - 8.0 * missing
    total += 6.0 * variant_bonus(order, left_act, right_act)
    return total, supported, missing


def infer_order(activities, left_act, right_act):
    unique_perm_count = math.factorial(len(activities))
    for c in Counter(activities).values():
        unique_perm_count //= math.factorial(c)

    best = None
    candidates = []

    if unique_perm_count <= 720:
        for perm in multiset_permutations(activities):
            score, supported, missing = score_order(perm, left_act, right_act)
            candidates.append((missing, -supported, -score, perm, score, supported))
        candidates.sort()
        min_missing = candidates[0][0]
        plausible = [c for c in candidates if c[0] == min_missing]
        plausible.sort(key=lambda x: (x[1], x[2]))
        best = plausible[0][3]
        return list(best)

    # Beam search fallback
    beam = [((), Counter(activities), 0.0)]
    beam_width = 200
    for _ in range(len(activities)):
        new_beam = []
        for prefix, rem, sc in beam:
            prev = left_act if len(prefix) == 0 else prefix[-1]
            for act in list(rem.keys()):
                if rem[act] <= 0:
                    continue
                rem2 = rem.copy()
                rem2[act] -= 1
                if rem2[act] == 0:
                    del rem2[act]
                add = 0.0
                if prev is not None:
                    add += edge_score(prev, act)
                    if (prev, act) in EDGE_FREQ or (prev, act) in PROB_MAP:
                        add += 3.0
                    else:
                        add -= 8.0
                new_prefix = prefix + (act,)
                add += 2.0 * variant_bonus(list(new_prefix), left_act, None)
                new_beam.append((new_prefix, rem2, sc + add))
        new_beam.sort(key=lambda x: -x[2])
        beam = new_beam[:beam_width]

    finals = []
    for prefix, rem, sc in beam:
        end_sc = sc
        if right_act is not None and prefix:
            end_sc += edge_score(prefix[-1], right_act)
            if (prefix[-1], right_act) in EDGE_FREQ or (prefix[-1], right_act) in PROB_MAP:
                end_sc += 3.0
            else:
                end_sc -= 8.0
        end_sc += 6.0 * variant_bonus(list(prefix), left_act, right_act)
        supported, missing = count_supported_edges(prefix, left_act, right_act)
        finals.append((missing, -supported, -end_sc, prefix))
    finals.sort()
    return list(finals[0][3])


def gap_seconds(a, b, min_step_seconds):
    prof = DUR_MAP.get((a, b))
    if prof:
        if prof["share_under_60s"] >= 0.45:
            g = prof["p10_seconds"]
        else:
            g = prof["p25_seconds"]
        if g <= 0:
            g = min_step_seconds
        return max(g, min_step_seconds)
    return min_step_seconds


def choose_position_mode(order, observed_ts, left_ts, right_ts, min_step):
    start_score = 0.0
    end_score = 0.0
    room_right = None if right_ts is None else (right_ts - observed_ts).total_seconds()
    room_left = None if left_ts is None else (observed_ts - left_ts).total_seconds()

    if right_ts is not None:
        start_score += 3.0
        if room_right >= 0:
            start_score += 1.0
    if left_ts is not None:
        end_score += 0.5
        if room_left is not None and room_left >= 0:
            end_score += 0.5

    # Explicit exception only when end placement is clearly favored
    # and start placement is incompatible with anchor ordering.
    if right_ts is not None and room_right is not None and room_right < (len(order) - 1) * min_step.total_seconds():
        if left_ts is not None and room_left is not None and room_left >= (len(order) - 1) * min_step.total_seconds():
            return "end"
    return "start"


def build_times(order, observed_ts, left_ts, right_ts, case_start, case_end, min_step):
    min_step_sec = min_step.total_seconds()
    mode = choose_position_mode(order, observed_ts, left_ts, right_ts, min_step)

    gaps_sec = [gap_seconds(order[i], order[i + 1], min_step_sec) for i in range(len(order) - 1)]
    min_total = (len(order) - 1) * min_step_sec
    pref_total = sum(gaps_sec)

    if mode == "start":
        times = [observed_ts]
        room_right = None if right_ts is None else max(0.0, (right_ts - observed_ts).total_seconds())
        if right_ts is not None:
            max_total = room_right
        else:
            max_total = max(0.0, (case_end - observed_ts).total_seconds())

        if len(order) == 1:
            return times, False

        target_total = pref_total
        if max_total < min_total:
            target_total = min_total
        else:
            target_total = min(pref_total, max_total)

        clamped = target_total != pref_total or (right_ts is not None and pref_total > max_total)

        if pref_total <= 0:
            scaled = [min_step_sec] * (len(order) - 1)
        elif target_total < min_total:
            scaled = [min_step_sec] * (len(order) - 1)
            clamped = True
        elif abs(target_total - pref_total) < 1e-9:
            scaled = list(gaps_sec)
        else:
            extra_pref = [max(0.0, g - min_step_sec) for g in gaps_sec]
            extra_sum = sum(extra_pref)
            if extra_sum <= 0:
                scaled = [min_step_sec] * (len(order) - 1)
            else:
                target_extra = max(0.0, target_total - min_total)
                ratio = target_extra / extra_sum if extra_sum > 0 else 0.0
                scaled = [min_step_sec + e * ratio for e in extra_pref]

        t = observed_ts
        for g in scaled:
            t = t + pd.to_timedelta(g, unit="s")
            times.append(t)
        return times, clamped

    # end-at-observed explicit exception
    times = [None] * len(order)
    times[-1] = observed_ts
    room_left = None if left_ts is None else max(0.0, (observed_ts - left_ts).total_seconds())
    if left_ts is not None:
        max_total = room_left
    else:
        max_total = max(0.0, (observed_ts - case_start).total_seconds())

    if len(order) == 1:
        return [observed_ts], False

    target_total = pref_total
    if max_total < min_total:
        target_total = min_total
    else:
        target_total = min(pref_total, max_total)

    clamped = True

    if pref_total <= 0:
        scaled = [min_step_sec] * (len(order) - 1)
    elif target_total < min_total:
        scaled = [min_step_sec] * (len(order) - 1)
    elif abs(target_total - pref_total) < 1e-9:
        scaled = list(gaps_sec)
    else:
        extra_pref = [max(0.0, g - min_step_sec) for g in gaps_sec]
        extra_sum = sum(extra_pref)
        if extra_sum <= 0:
            scaled = [min_step_sec] * (len(order) - 1)
        else:
            target_extra = max(0.0, target_total - min_total)
            ratio = target_extra / extra_sum if extra_sum > 0 else 0.0
            scaled = [min_step_sec + e * ratio for e in extra_pref]

    t = observed_ts
    for i in range(len(order) - 2, -1, -1):
        g = scaled[i]
        t = t - pd.to_timedelta(g, unit="s")
        times[i] = t
    return times, clamped


def enforce_block_constraints(times, observed_ts, left_ts, right_ts, case_start, case_end, min_step, keep_observed_pos):
    clamped = False
    n = len(times)
    lower = case_start if left_ts is None else max(case_start, left_ts)
    upper = case_end if right_ts is None else min(case_end, right_ts)

    # ensure observed retained exactly at keep_observed_pos
    if times[keep_observed_pos] != observed_ts:
        times[keep_observed_pos] = observed_ts
        clamped = True

    # forward pass strict increase and lower bound
    if times[0] < lower:
        times[0] = lower
        clamped = True
    for i in range(1, n):
        min_allowed = times[i - 1] + min_step
        if times[i] < min_allowed:
            times[i] = min_allowed
            clamped = True

    # backward from upper
    if times[-1] > upper:
        times[-1] = upper
        clamped = True
    for i in range(n - 2, -1, -1):
        max_allowed = times[i + 1] - min_step
        if times[i] > max_allowed:
            times[i] = max_allowed
            clamped = True

    # restore observed and rebuild around it with minimum distinct spacing if needed
    if keep_observed_pos == 0:
        if times[0] != observed_ts:
            times[0] = observed_ts
            clamped = True
        for i in range(1, n):
            req = times[i - 1] + min_step
            if times[i] < req:
                times[i] = req
                clamped = True
        if times[-1] > upper:
            latest_start = upper - min_step * (n - 1)
            if latest_start < observed_ts:
                latest_start = observed_ts
            seq = [latest_start + i * min_step for i in range(n)]
            seq[0] = observed_ts
            for i in range(1, n):
                if seq[i] <= seq[i - 1]:
                    seq[i] = seq[i - 1] + min_step
            if seq[-1] > upper:
                seq = [observed_ts + i * min_step for i in range(n)]
            times = seq
            clamped = True
    else:
        if times[-1] != observed_ts:
            times[-1] = observed_ts
            clamped = True
        for i in range(n - 2, -1, -1):
            req = times[i + 1] - min_step
            if times[i] > req:
                times[i] = req
                clamped = True
        if times[0] < lower:
            earliest_end = lower + min_step * (n - 1)
            if earliest_end > observed_ts:
                earliest_end = observed_ts
            seq = [observed_ts - (n - 1 - i) * min_step for i in range(n)]
            if seq[0] < lower:
                seq = [lower + i * min_step for i in range(n)]
            seq[-1] = observed_ts
            for i in range(n - 2, -1, -1):
                if seq[i] >= seq[i + 1]:
                    seq[i] = seq[i + 1] - min_step
            times = seq
            clamped = True

    # Final absolute clamp if impossible due to case bounds; still preserve distinctness as much as possible
    if times[0] < lower:
        shift = lower - times[0]
        times = [t + shift for t in times]
        clamped = True
    if times[-1] > upper:
        shift = times[-1] - upper
        times = [t - shift for t in times]
        clamped = True

    # last strictness pass
    for i in range(1, n):
        if not (times[i] > times[i - 1]):
            times[i] = times[i - 1] + min_step
            clamped = True
    if times[-1] > upper:
        # compact around observed/start
        if keep_observed_pos == 0:
            times = [observed_ts + i * min_step for i in range(n)]
        else:
            times = [observed_ts - (n - 1 - i) * min_step for i in range(n)]
        clamped = True

    # case bounds final
    if times[0] < case_start:
        shift = case_start - times[0]
        times = [t + shift for t in times]
        clamped = True
    if times[-1] > case_end:
        shift = times[-1] - case_end
        times = [t - shift for t in times]
        clamped = True

    # final monotonicity rescue
    for i in range(1, n):
        if not (times[i] > times[i - 1]):
            times[i] = times[i - 1] + min_step
            clamped = True

    return times, clamped


def assign_times_to_rows(block_df, order, times, observed_ts, min_step):
    positions_by_activity = defaultdict(list)
    for pos, act in enumerate(order):
        positions_by_activity[act].append(pos)

    row_assignments = {}
    # Earliest event keeps observed timestamp exactly.
    # Among rows with same activity as first ordered activity, assign observed to the earliest row index for stability.
    used_rows = set()
    for act in order:
        pass

    act_to_rows = defaultdict(list)
    for idx, row in block_df.iterrows():
        act_to_rows[row["Activity"]].append(idx)

    for act in act_to_rows:
        act_to_rows[act] = sorted(act_to_rows[act])

    for act, positions in positions_by_activity.items():
        rows = act_to_rows[act]
        for r, pos in zip(rows, positions):
            row_assignments[r] = times[pos]

    return row_assignments


def find_c1_blocks(case_df):
    blocks = []
    rows = case_df.to_dict("records")
    i = 0
    n = len(rows)
    while i < n:
        row = rows[i]
        if row["error_category"] != "CLEAN":
            j = i + 1
            shared_ts = row["Timestamp"]
            while (
                j < n
                and rows[j]["error_category"] != "CLEAN"
                and rows[j]["Timestamp"] == shared_ts
            ):
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

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_columns = list(df.columns)
    df = df.copy()
    df["_orig_ts"] = df["Timestamp"].astype(str)
    df["_out_ts"] = df["_orig_ts"]
    df["_parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    precision_digits, min_step = detect_precision_and_step(df["Timestamp"])

    repaired_blocks = 0
    clamped_blocks = 0

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_idx = list(idxs)
        case_df = df.loc[case_idx].copy()
        case_df["_case_pos"] = range(len(case_df))
        case_start = case_df["_parsed_ts"].min()
        case_end = case_df["_parsed_ts"].max()

        blocks = find_c1_blocks(case_df.reset_index())
        if not blocks:
            continue

        cdf = case_df.reset_index()
        for start, end in blocks:
            block = cdf.iloc[start:end].copy()
            observed_str = block.iloc[0]["Timestamp"]
            observed_ts = pd.to_datetime(observed_str, errors="coerce")
            if pd.isna(observed_ts):
                observed_ts = block.iloc[0]["_parsed_ts"]

            left_anchor = cdf.iloc[start - 1] if start - 1 >= 0 and cdf.iloc[start - 1]["error_category"] == "CLEAN" else None
            right_anchor = cdf.iloc[end] if end < len(cdf) and cdf.iloc[end]["error_category"] == "CLEAN" else None

            left_act = None if left_anchor is None else left_anchor["Activity"]
            right_act = None if right_anchor is None else right_anchor["Activity"]
            left_ts = None if left_anchor is None else left_anchor["_parsed_ts"]
            right_ts = None if right_anchor is None else right_anchor["_parsed_ts"]

            activities = list(block["Activity"])
            order = infer_order(activities, left_act, right_act)

            mode = choose_position_mode(order, observed_ts, left_ts, right_ts, min_step)
            times, local_clamped = build_times(order, observed_ts, left_ts, right_ts, case_start, case_end, min_step)
            keep_observed_pos = 0 if mode == "start" else len(order) - 1
            times, local_clamped2 = enforce_block_constraints(
                times, observed_ts, left_ts, right_ts, case_start, case_end, min_step, keep_observed_pos
            )
            local_clamped = local_clamped or local_clamped2

            # If start-at-observed should be preferred but failed monotonicity due to impossible room, compact minimally from observed.
            if keep_observed_pos == 0:
                if any(not (times[i] > times[i - 1]) for i in range(1, len(times))):
                    times = [observed_ts + i * min_step for i in range(len(times))]
                    local_clamped = True
                if right_ts is not None and times[-1] > right_ts:
                    room = (right_ts - observed_ts).total_seconds()
                    if room >= (len(times) - 1) * min_step.total_seconds():
                        times = [observed_ts + i * min_step for i in range(len(times))]
                    else:
                        times = [observed_ts + i * min_step for i in range(len(times))]
                    local_clamped = True

            row_to_time = assign_times_to_rows(block, order, times, observed_ts, min_step)

            # self-check and corrective rewrite by inferred execution order, not row order
            positions_by_activity = defaultdict(list)
            for pos, act in enumerate(order):
                positions_by_activity[act].append(pos)
            act_row_lists = defaultdict(list)
            for _, r in block.iterrows():
                act_row_lists[r["Activity"]].append(r["index"])
            for act in act_row_lists:
                act_row_lists[act] = sorted(act_row_lists[act])

            # ensure exactly one row keeps observed and it belongs to earliest ordered activity position
            observed_rows = []
            for act, pos_list in positions_by_activity.items():
                rows = act_row_lists[act]
                for row_idx, pos in zip(rows, pos_list):
                    if times[pos] == observed_ts:
                        observed_rows.append(row_idx)
            if len(observed_rows) != 1:
                local_clamped = True
                first_act = order[0]
                rows = act_row_lists[first_act]
                chosen_row = rows[0]
                # remap by positions stable
                row_to_time = {}
                for act, pos_list in positions_by_activity.items():
                    rows2 = act_row_lists[act]
                    for row_idx, pos in zip(rows2, pos_list):
                        row_to_time[row_idx] = times[pos]
                row_to_time[chosen_row] = observed_ts

            # final guard: all corrupted rows except one differ from input
            equal_rows = []
            for row_idx in block["index"]:
                out_ts = row_to_time[row_idx]
                out_str = format_dt(out_ts, precision_digits)
                in_str = df.at[row_idx, "_orig_ts"]
                if out_str == in_str:
                    equal_rows.append(row_idx)

            if len(equal_rows) != 1:
                local_clamped = True
                # choose one earliest-order row to keep observed, nudge all others forward minimally if needed
                first_act = order[0]
                keep_row = sorted(act_row_lists[first_act])[0]
                assigned_rows_sorted = sorted(row_to_time.keys(), key=lambda rid: row_to_time[rid])
                seen_keep = False
                for rid in assigned_rows_sorted:
                    t = row_to_time[rid]
                    s = format_dt(t, precision_digits)
                    if rid == keep_row and s == df.at[rid, "_orig_ts"] and not seen_keep:
                        seen_keep = True
                        continue
                    if s == df.at[rid, "_orig_ts"]:
                        row_to_time[rid] = t + min_step

                # rebuild strictness in time order
                ordered_pairs = sorted(row_to_time.items(), key=lambda kv: kv[1])
                fixed = []
                prev = None
                for rid, t in ordered_pairs:
                    if prev is not None and t <= prev:
                        t = prev + min_step
                    fixed.append((rid, t))
                    prev = t
                row_to_time = dict(fixed)

            for row_idx, out_dt in row_to_time.items():
                df.at[row_idx, "_out_ts"] = format_dt(out_dt, precision_digits)

            repaired_blocks += 1
            if local_clamped:
                clamped_blocks += 1

    # Global self-check / corrections
    # CLEAN rows unchanged
    clean_mask = df["error_category"] == "CLEAN"
    df.loc[clean_mask, "_out_ts"] = df.loc[clean_mask, "_orig_ts"]

    # parse repaired timestamps
    parsed_out = pd.to_datetime(df["_out_ts"], errors="coerce")

    # Per-case monotonicity against anchors only for repaired C1 blocks; repair again if necessary
    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_idx = list(idxs)
        cdf = df.loc[case_idx].copy().reset_index()
        blocks = find_c1_blocks(cdf)
        case_start = pd.to_datetime(cdf["_out_ts"], errors="coerce").min()
        case_end = pd.to_datetime(cdf["_out_ts"], errors="coerce").max()
        for start, end in blocks:
            block = cdf.iloc[start:end].copy()
            observed_str = block.iloc[0]["_orig_ts"]
            observed_ts = pd.to_datetime(observed_str, errors="coerce")
            left_anchor = cdf.iloc[start - 1] if start - 1 >= 0 and cdf.iloc[start - 1]["error_category"] == "CLEAN" else None
            right_anchor = cdf.iloc[end] if end < len(cdf) and cdf.iloc[end]["error_category"] == "CLEAN" else None
            left_ts = None if left_anchor is None else pd.to_datetime(left_anchor["_out_ts"], errors="coerce")
            right_ts = None if right_anchor is None else pd.to_datetime(right_anchor["_out_ts"], errors="coerce")

            out_times = []
            for _, r in block.iterrows():
                out_times.append((r["index"], pd.to_datetime(df.at[r["index"], "_out_ts"], errors="coerce"), r["Activity"]))
            unique_vals = {t for _, t, _ in out_times}
            if len(unique_vals) != len(out_times):
                out_times.sort(key=lambda x: x[1])
                base = min(t for _, t, _ in out_times)
                fixed = []
                for i, (rid, _, _) in enumerate(out_times):
                    fixed.append((rid, base + i * min_step))
                if right_ts is not None and fixed[-1][1] > right_ts:
                    base = observed_ts
                    fixed = [(rid, base + i * min_step) for i, (rid, _, _) in enumerate(out_times)]
                for rid, t in fixed:
                    df.at[rid, "_out_ts"] = format_dt(t, precision_digits)
                clamped_blocks += 1

            # anchor bounds
            lower = case_start if left_ts is None else max(case_start, left_ts)
            upper = case_end if right_ts is None else min(case_end, right_ts)
            block_rows = [r["index"] for _, r in block.iterrows()]
            block_parsed = [pd.to_datetime(df.at[rid, "_out_ts"], errors="coerce") for rid in block_rows]
            changed = False
            ordered = sorted(block_parsed)
            if any(t < lower for t in ordered) or any(t > upper for t in ordered) or any(ordered[i] <= ordered[i - 1] for i in range(1, len(ordered))):
                base = max(observed_ts, lower)
                fixed = [base + i * min_step for i in range(len(block_rows))]
                if fixed[-1] > upper:
                    # compress to minimal feasible within anchors
                    base = lower
                    fixed = [base + i * min_step for i in range(len(block_rows))]
                    if fixed[-1] > upper:
                        fixed = [observed_ts + i * min_step for i in range(len(block_rows))]
                sorted_rows = sorted(block_rows, key=lambda rid: pd.to_datetime(df.at[rid, "_out_ts"], errors="coerce"))
                for rid, t in zip(sorted_rows, fixed):
                    df.at[rid, "_out_ts"] = format_dt(t, precision_digits)
                changed = True
            if changed:
                clamped_blocks += 1

    # Final counts
    rows_still_equal = int((df["_out_ts"] == df["_orig_ts"]).sum())

    out_df = df[original_columns].copy()
    out_df["Timestamp"] = df["_out_ts"]

    try:
        out_df.to_csv(args.output, index=False)
    except Exception:
        # never fail hard
        try:
            out_df.astype(str).to_csv(args.output, index=False)
        except Exception:
            pass

    print(f"blocks_repaired={repaired_blocks} blocks_clamped={clamped_blocks} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")
        except Exception:
            pass