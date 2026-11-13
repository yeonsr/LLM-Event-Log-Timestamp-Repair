import argparse
import itertools
import math
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

GLOBAL_MEDIAN_DURATION_SECONDS = 431590.0

TRANS_PROB = {(d["source"], d["target"]): d["probability"] for d in TRANSITION_PROBABILITIES_TOP}
DUR = {(d["source"], d["target"]): d for d in DURATION_PROFILES_TOP}
DFG_EDGES = set(DUR.keys()) | set(TRANS_PROB.keys())


def detect_precision(ts: str):
    if "." in ts:
        frac = ts.split(".")[-1]
        return len(frac), "%Y-%m-%d %H:%M:%S.%f"
    return 0, "%Y-%m-%d %H:%M:%S"


def format_ts(dt: pd.Timestamp, frac_digits: int):
    if frac_digits <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    keep = 19 + 1 + frac_digits
    return base[:keep]


def parse_ts(s: str):
    return pd.to_datetime(s)


def edge_exists(a, b):
    return (a, b) in DFG_EDGES


def edge_score(a, b):
    if (a, b) in TRANS_PROB:
        return 1000.0 * TRANS_PROB[(a, b)]
    if (a, b) in DUR:
        return 50.0
    return -5000.0


def variant_support(order, left_act, right_act):
    seq = list(order)
    best = 0
    for item in FREQUENT_VARIANTS:
        var = item["variant"]
        cnt = item["count"]
        n = len(seq)
        for i in range(len(var) - n + 1):
            if var[i:i + n] == seq:
                bonus = cnt
                if left_act is not None and i > 0 and var[i - 1] == left_act:
                    bonus += cnt * 0.5
                if right_act is not None and i + n < len(var) and var[i + n] == right_act:
                    bonus += cnt * 0.5
                if bonus > best:
                    best = bonus
    return best


def candidate_orders(activities):
    counts = Counter(activities)
    uniq = sorted(counts.keys())
    if len(activities) <= 8 and len(set(itertools.permutations(activities))) <= 5000:
        seen = set()
        out = []
        for p in itertools.permutations(activities):
            if p not in seen:
                seen.add(p)
                out.append(list(p))
        return out
    perms = []
    for p in itertools.permutations(uniq):
        expanded = []
        for x in p:
            expanded.extend([x] * counts[x])
        perms.append(expanded)
    return perms


def score_order(order, left_act, right_act):
    internal_missing = sum(1 for a, b in zip(order[:-1], order[1:]) if not edge_exists(a, b))
    if len(order) >= 2 and internal_missing >= max(2, len(order) - 1):
        return -1e12
    s = 0.0
    if left_act is not None:
        s += edge_score(left_act, order[0]) * 1.5
    if right_act is not None:
        s += edge_score(order[-1], right_act) * 1.5
    for a, b in zip(order[:-1], order[1:]):
        s += edge_score(a, b) * 2.0
    s += variant_support(order, left_act, right_act) * 0.2
    return s


def choose_gap(a, b, min_step_sec):
    prof = DUR.get((a, b))
    if prof is None:
        return max(min_step_sec, 2.0, 7.0, min_step_sec + (1 if min_step_sec < 2 else 0))
    if prof["share_under_60s"] >= 0.4:
        g = prof["p10_seconds"]
    elif prof["share_under_60s"] >= 0.15:
        g = prof["p25_seconds"]
    else:
        g = prof["p25_seconds"]
    if g <= 0:
        g = 1.0
    return max(float(g), float(min_step_sec))


def is_concurrent_like(a, b):
    ab = TRANS_PROB.get((a, b), 0.0)
    ba = TRANS_PROB.get((b, a), 0.0)
    if ab > 0 and ba > 0 and min(ab, ba) / max(ab, ba) >= 0.5:
        return True
    dab = DUR.get((a, b))
    dba = DUR.get((b, a))
    if dab and dba:
        if dab.get("share_under_60s", 0) >= 0.4 and dba.get("share_under_60s", 0) >= 0.4:
            return True
    return False


def assign_times(order, observed_ts, right_anchor_ts, case_end_ts, frac_digits):
    precision_step_ns = 10 ** (9 - frac_digits) if frac_digits > 0 else 10 ** 9
    min_step = max(1.0, precision_step_ns / 1e9)

    increments = []
    for a, b in zip(order[:-1], order[1:]):
        if is_concurrent_like(a, b):
            g = min_step
        else:
            g = choose_gap(a, b, min_step)
        increments.append(float(g))

    avail_end = right_anchor_ts if right_anchor_ts is not None else case_end_ts
    room = max(0.0, (avail_end - observed_ts).total_seconds())
    min_required = (len(order) - 1) * min_step

    if room < min_required:
        increments = [min_step] * (len(order) - 1)
    else:
        total = sum(increments)
        if total > room:
            scale = room / total if total > 0 else 1.0
            scaled = [max(min_step, g * scale) for g in increments]
            total2 = sum(scaled)
            if total2 > room:
                extra = total2 - room
                reducible = [g - min_step for g in scaled]
                red_sum = sum(max(0.0, x) for x in reducible)
                if red_sum > 0:
                    scaled2 = []
                    for g, r in zip(scaled, reducible):
                        rr = extra * (max(0.0, r) / red_sum)
                        scaled2.append(max(min_step, g - rr))
                    scaled = scaled2
            increments = scaled

    times = [observed_ts]
    cur = observed_ts
    for g in increments:
        cur = cur + pd.to_timedelta(g, unit="s")
        times.append(cur)

    max_last = avail_end
    if times[-1] > max_last and len(times) > 1:
        room = max(0.0, (max_last - observed_ts).total_seconds())
        if room >= min_required:
            weights = increments[:]
            if sum(weights) <= 0:
                weights = [1.0] * len(increments)
            total = sum(weights)
            cums = []
            acc = 0.0
            for w in weights:
                acc += w
                cums.append(acc)
            times = [observed_ts]
            for c in cums:
                offset = room * (c / total)
                times.append(observed_ts + pd.to_timedelta(offset, unit="s"))

    return times


def respaced_distinct(times, lower_ts, upper_ts, frac_digits):
    precision_step_ns = 10 ** (9 - frac_digits) if frac_digits > 0 else 10 ** 9
    step = pd.to_timedelta(precision_step_ns, unit="ns")
    out = [times[0]]
    for i in range(1, len(times)):
        t = times[i]
        if t <= out[-1]:
            t = out[-1] + step
        out.append(t)
    if upper_ts is not None and out[-1] > upper_ts:
        k = len(out)
        min_required = step * (k - 1)
        if lower_ts + min_required <= upper_ts:
            out = [lower_ts + step * i for i in range(k)]
        else:
            out = [upper_ts - step * (k - 1 - i) for i in range(k)]
            if out[0] < lower_ts:
                out = [lower_ts + step * i for i in range(k)]
    return out


def map_times_to_rows(block_df, order, times):
    act_to_rows = defaultdict(list)
    for idx, row in block_df.iterrows():
        act_to_rows[row["Activity"]].append(idx)
    for act in act_to_rows:
        act_to_rows[act].sort()
    act_to_times = defaultdict(list)
    for act, t in zip(order, times):
        act_to_times[act].append(t)
    mapping = {}
    for act, rows in act_to_rows.items():
        ts = sorted(act_to_times[act])
        for r, t in zip(rows, ts):
            mapping[r] = t
    return mapping


def repair_block(df, block_idx, case_indices, case_start_ts, case_end_ts, frac_digits):
    block = df.loc[block_idx]
    observed_str = block.iloc[0]["Timestamp"]
    observed_ts = parse_ts(observed_str)
    first_pos = case_indices.index(block_idx[0])
    last_pos = case_indices.index(block_idx[-1])

    left_anchor_idx = None
    right_anchor_idx = None
    for p in range(first_pos - 1, -1, -1):
        idx = case_indices[p]
        if df.at[idx, "error_category"] == "CLEAN":
            left_anchor_idx = idx
            break
    for p in range(last_pos + 1, len(case_indices)):
        idx = case_indices[p]
        if df.at[idx, "error_category"] == "CLEAN":
            right_anchor_idx = idx
            break

    left_act = df.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None else None
    right_act = df.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None else None
    left_ts = parse_ts(df.at[left_anchor_idx, "Timestamp"]) if left_anchor_idx is not None else case_start_ts
    right_ts = parse_ts(df.at[right_anchor_idx, "Timestamp"]) if right_anchor_idx is not None else case_end_ts

    activities = list(block["Activity"])
    orders = candidate_orders(activities)
    scored = [(score_order(o, left_act, right_act), o) for o in orders]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_order = scored[0][1] if scored else activities[:]

    times = assign_times(best_order, observed_ts, right_ts, case_end_ts, frac_digits)
    clamped = 0

    lower = max(case_start_ts, left_ts)
    upper = min(case_end_ts, right_ts)
    fixed_times = []
    for i, t in enumerate(times):
        if t < lower:
            t = lower
            clamped = 1
        if t > upper:
            t = upper
            clamped = 1
        fixed_times.append(t)
    times = fixed_times
    times[0] = observed_ts
    times = respaced_distinct(times, observed_ts, upper, frac_digits)

    if len(times) >= 2:
        precision_step_ns = 10 ** (9 - frac_digits) if frac_digits > 0 else 10 ** 9
        min_span = pd.to_timedelta(precision_step_ns * (len(times) - 1), unit="ns")
        if times[-1] - times[0] < min_span:
            times = [observed_ts + pd.to_timedelta(precision_step_ns * i, unit="ns") for i in range(len(times))]
            times = respaced_distinct(times, observed_ts, upper, frac_digits)
            clamped = 1

    row_map = map_times_to_rows(block, best_order, times)
    earliest_idx = min(row_map.keys(), key=lambda i: row_map[i])
    return row_map, earliest_idx, clamped


def find_c1_blocks(df):
    blocks = []
    i = 0
    n = len(df)
    while i < n:
        if df.iloc[i]["error_category"] == "CLEAN":
            i += 1
            continue
        case = df.iloc[i]["Case"]
        ts = df.iloc[i]["Timestamp"]
        j = i + 1
        while j < n and df.iloc[j]["error_category"] != "CLEAN" and df.iloc[j]["Case"] == case and df.iloc[j]["Timestamp"] == ts:
            j += 1
        block_idx = list(df.index[i:j])
        if len(block_idx) >= 2:
            blocks.append(block_idx)
        i = j
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

    original_timestamps = df["Timestamp"].copy()
    out_ts = df["Timestamp"].copy()

    frac_digits = 0
    for s in df["Timestamp"]:
        d, _ = detect_precision(s)
        frac_digits = max(frac_digits, d)

    case_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        case_to_indices[row["Case"]].append(idx)

    case_bounds = {}
    for case, idxs in case_to_indices.items():
        parsed = [parse_ts(df.at[i, "Timestamp"]) for i in idxs]
        case_bounds[case] = (min(parsed), max(parsed))

    blocks = find_c1_blocks(df)
    repaired_blocks = 0
    clamped_blocks = 0

    for block_idx in blocks:
        case = df.at[block_idx[0], "Case"]
        case_start_ts, case_end_ts = case_bounds[case]
        row_map, earliest_idx, clamped = repair_block(df, block_idx, case_to_indices[case], case_start_ts, case_end_ts, frac_digits)
        repaired_blocks += 1
        clamped_blocks += clamped
        observed_str = df.at[block_idx[0], "Timestamp"]
        observed_ts = parse_ts(observed_str)

        for idx in block_idx:
            t = row_map[idx]
            if idx == earliest_idx:
                out_ts.at[idx] = observed_str
            else:
                if t == observed_ts:
                    precision_step_ns = 10 ** (9 - frac_digits) if frac_digits > 0 else 10 ** 9
                    t = t + pd.to_timedelta(precision_step_ns, unit="ns")
                out_ts.at[idx] = format_ts(t, frac_digits)

    # Self-check and corrective pass
    for block_idx in blocks:
        observed_str = original_timestamps.at[block_idx[0]]
        observed_ts = parse_ts(observed_str)
        case = df.at[block_idx[0], "Case"]
        case_start_ts, case_end_ts = case_bounds[case]
        case_indices = case_to_indices[case]
        first_pos = case_indices.index(block_idx[0])
        last_pos = case_indices.index(block_idx[-1])

        left_anchor_idx = None
        right_anchor_idx = None
        for p in range(first_pos - 1, -1, -1):
            idx = case_indices[p]
            if df.at[idx, "error_category"] == "CLEAN":
                left_anchor_idx = idx
                break
        for p in range(last_pos + 1, len(case_indices)):
            idx = case_indices[p]
            if df.at[idx, "error_category"] == "CLEAN":
                right_anchor_idx = idx
                break

        lower = parse_ts(df.at[left_anchor_idx, "Timestamp"]) if left_anchor_idx is not None else case_start_ts
        upper = parse_ts(df.at[right_anchor_idx, "Timestamp"]) if right_anchor_idx is not None else case_end_ts
        lower = max(lower, case_start_ts)
        upper = min(upper, case_end_ts)

        parsed = []
        for idx in block_idx:
            try:
                parsed.append(parse_ts(out_ts.at[idx]))
            except Exception:
                parsed.append(observed_ts)

        same_count = sum(1 for idx in block_idx if out_ts.at[idx] == original_timestamps.at[idx])
        eq_observed = [idx for idx in block_idx if parse_ts(out_ts.at[idx]) == observed_ts]

        need_fix = False
        if len(set(parsed)) != len(parsed):
            need_fix = True
        if len(eq_observed) != 1:
            need_fix = True
        if any(t < lower or t > upper for t in parsed):
            need_fix = True
        if upper >= observed_ts:
            if max(parsed) - min(parsed) > (upper - observed_ts):
                need_fix = True
        if len(block_idx) >= 2:
            precision_step_ns = 10 ** (9 - frac_digits) if frac_digits > 0 else 10 ** 9
            min_span = pd.to_timedelta(precision_step_ns * (len(block_idx) - 1), unit="ns")
            if max(parsed) - min(parsed) < min_span:
                need_fix = True
        if same_count < 1:
            need_fix = True

        if need_fix:
            clamped_blocks += 1
            precision_step_ns = 10 ** (9 - frac_digits) if frac_digits > 0 else 10 ** 9
            step = pd.to_timedelta(precision_step_ns, unit="ns")
            room = upper - observed_ts
            if room < step * (len(block_idx) - 1):
                start = max(lower, upper - step * (len(block_idx) - 1))
            else:
                start = observed_ts
            repaired = [start + step * i for i in range(len(block_idx))]
            if repaired[0] != observed_ts and observed_ts >= lower and observed_ts <= upper and upper - observed_ts >= step * (len(block_idx) - 1):
                repaired = [observed_ts + step * i for i in range(len(block_idx))]
            parsed_rows = list(block_idx)
            activities = [df.at[i, "Activity"] for i in parsed_rows]
            best_order = None
            orders = candidate_orders(activities)
            scored = [(score_order(o, None, None), o) for o in orders]
            scored.sort(key=lambda x: x[0], reverse=True)
            best_order = scored[0][1] if scored else activities
            act_to_rows = defaultdict(list)
            for idx in parsed_rows:
                act_to_rows[df.at[idx, "Activity"]].append(idx)
            for act in act_to_rows:
                act_to_rows[act].sort()
            act_to_times = defaultdict(list)
            for act, t in zip(best_order, repaired):
                act_to_times[act].append(t)
            row_map = {}
            for act, rows in act_to_rows.items():
                tslist = act_to_times[act]
                for r, t in zip(rows, tslist):
                    row_map[r] = t
            earliest_idx = min(row_map.keys(), key=lambda i: row_map[i])
            for idx in block_idx:
                if idx == earliest_idx:
                    out_ts.at[idx] = observed_str
                else:
                    out_ts.at[idx] = format_ts(row_map[idx], frac_digits)

    df["Timestamp"] = out_ts

    rows_still_equal = int((df["Timestamp"] == original_timestamps).sum())

    try:
        df.to_csv(args.output, index=False)
    except Exception:
        # Last-resort safe write
        safe_df = df.copy()
        safe_df.to_csv(args.output, index=False)

    print(f"blocks_repaired={repaired_blocks} blocks_clamped={clamped_blocks} rows_still_equal_to_input={rows_still_equal}")


if __name__ == "__main__":
    main()