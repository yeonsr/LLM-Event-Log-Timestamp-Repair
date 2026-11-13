import argparse
import csv
import math
import itertools
from collections import defaultdict, Counter
import pandas as pd

ERROR_CAT = "C1_ORDERING_AMBIGUITY"

PROCESS = {
    "transition_probabilities_top": [
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
        {"source": "STORNO", "target": "SET STATUS", "probability": 0.0135}
    ],
    "duration_profiles_top": [
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
        {"source": "RELEASE", "target": "STORNO", "n": 7, "p10_seconds": 4772324.2, "p25_seconds": 5108938.0, "median_seconds": 5648782.0, "p75_seconds": 17627811.5, "p90_seconds": 23873601.4, "share_under_60s": 0.0}
    ],
    "block_span_profile_seconds": {
        "2": {"n": 33782, "p10": 17.0, "p25": 11477.75, "median": 431590.0, "p75": 5503867.75, "p90": 11720432.200000001},
        "3": {"n": 25619, "p10": 42085.200000000004, "p25": 443774.0, "median": 3707174.0, "p75": 7916961.0, "p90": 14720249.8},
        "4": {"n": 18864, "p10": 229470.80000000005, "p25": 1886430.5, "median": 6224724.5, "p75": 8275661.0, "p90": 19222264.900000002},
        "5": {"n": 12996, "p10": 315736.5, "p25": 4189951.75, "median": 8268695.0, "p75": 14078886.5, "p90": 31054433.5}
    },
    "global_median_duration_seconds": 431590.0,
    "process_abstraction": {
        "dfg_edges": [
            {"source": "CODE OK", "target": "BILLED", "frequency": 5710, "median_duration_seconds": 3696259.5},
            {"source": "RELEASE", "target": "CODE OK", "frequency": 5671, "median_duration_seconds": 342906.0},
            {"source": "FIN", "target": "RELEASE", "frequency": 5642, "median_duration_seconds": 16196.5},
            {"source": "NEW", "target": "CHANGE DIAGN", "frequency": 3920, "median_duration_seconds": 24.0},
            {"source": "CHANGE DIAGN", "target": "FIN", "frequency": 3403, "median_duration_seconds": 7707138.0},
            {"source": "NEW", "target": "FIN", "frequency": 2383, "median_duration_seconds": 29527664.0},
            {"source": "NEW", "target": "NEW", "frequency": 1097, "median_duration_seconds": 1.0},
            {"source": "NEW", "target": "BILLED", "frequency": 690, "median_duration_seconds": 15589142.5},
            {"source": "NEW", "target": "DELETE", "frequency": 601, "median_duration_seconds": 1865565.0},
            {"source": "REOPEN", "target": "FIN", "frequency": 416, "median_duration_seconds": 42.0},
            {"source": "BILLED", "target": "STORNO", "frequency": 411, "median_duration_seconds": 1077095.0},
            {"source": "CODE OK", "target": "REOPEN", "frequency": 408, "median_duration_seconds": 1921091.5},
            {"source": "NEW", "target": "CODE OK", "frequency": 378, "median_duration_seconds": 8264453.5},
            {"source": "FIN", "target": "BILLED", "frequency": 303, "median_duration_seconds": 3633863.0},
            {"source": "CHANGE DIAGN", "target": "CHANGE DIAGN", "frequency": 249, "median_duration_seconds": 698451.0},
            {"source": "STORNO", "target": "BILLED", "frequency": 229, "median_duration_seconds": 169860.0},
            {"source": "CHANGE DIAGN", "target": "BILLED", "frequency": 212, "median_duration_seconds": 12718535.5},
            {"source": "STORNO", "target": "REJECT", "frequency": 203, "median_duration_seconds": 1240.0},
            {"source": "NEW", "target": "RELEASE", "frequency": 160, "median_duration_seconds": 7740827.0},
            {"source": "REOPEN", "target": "DELETE", "frequency": 152, "median_duration_seconds": 3.0},
            {"source": "CHANGE DIAGN", "target": "CODE OK", "frequency": 144, "median_duration_seconds": 8186331.0},
            {"source": "CHANGE DIAGN", "target": "DELETE", "frequency": 141, "median_duration_seconds": 946414.0},
            {"source": "REJECT", "target": "BILLED", "frequency": 127, "median_duration_seconds": 328893.0},
            {"source": "RELEASE", "target": "CODE NOK", "frequency": 114, "median_duration_seconds": 1430526.5},
            {"source": "CODE OK", "target": "RELEASE", "frequency": 84, "median_duration_seconds": 4348244.5},
            {"source": "REJECT", "target": "REOPEN", "frequency": 68, "median_duration_seconds": 30.5},
            {"source": "RELEASE", "target": "BILLED", "frequency": 49, "median_duration_seconds": 7875755.0},
            {"source": "FIN", "target": "REOPEN", "frequency": 47, "median_duration_seconds": 63422.0},
            {"source": "FIN", "target": "FIN", "frequency": 47, "median_duration_seconds": 9034907.0},
            {"source": "JOIN-PAT", "target": "JOIN-PAT", "frequency": 47, "median_duration_seconds": 0.0},
            {"source": "RELEASE", "target": "REOPEN", "frequency": 43, "median_duration_seconds": 225448.0},
            {"source": "CODE NOK", "target": "CODE NOK", "frequency": 41, "median_duration_seconds": 1897911.0},
            {"source": "CODE NOK", "target": "CODE OK", "frequency": 39, "median_duration_seconds": 43358.0},
            {"source": "CODE NOK", "target": "SET STATUS", "frequency": 36, "median_duration_seconds": 30387.0},
            {"source": "FIN", "target": "DELETE", "frequency": 35, "median_duration_seconds": 1255926.0},
            {"source": "CODE OK", "target": "MANUAL", "frequency": 29, "median_duration_seconds": 1652942.0},
            {"source": "REOPEN", "target": "CHANGE DIAGN", "frequency": 24, "median_duration_seconds": 88.5},
            {"source": "MANUAL", "target": "RELEASE", "frequency": 22, "median_duration_seconds": 36655.5},
            {"source": "CODE NOK", "target": "REOPEN", "frequency": 22, "median_duration_seconds": 31220.0},
            {"source": "BILLED", "target": "BILLED", "frequency": 21, "median_duration_seconds": 0.0},
            {"source": "CODE OK", "target": "CODE OK", "frequency": 19, "median_duration_seconds": 12213631.0},
            {"source": "CODE NOK", "target": "DELETE", "frequency": 19, "median_duration_seconds": 27554.0},
            {"source": "RELEASE", "target": "RELEASE", "frequency": 18, "median_duration_seconds": 0.0},
            {"source": "BILLED", "target": "CODE OK", "frequency": 17, "median_duration_seconds": 8266846.0},
            {"source": "CHANGE DIAGN", "target": "RELEASE", "frequency": 14, "median_duration_seconds": 7794016.0},
            {"source": "RELEASE", "target": "DELETE", "frequency": 14, "median_duration_seconds": 249504.5},
            {"source": "CHANGE DIAGN", "target": "STORNO", "frequency": 12, "median_duration_seconds": 16546335.0},
            {"source": "NEW", "target": "REOPEN", "frequency": 12, "median_duration_seconds": 8322668.5},
            {"source": "FIN", "target": "CODE OK", "frequency": 11, "median_duration_seconds": 5896170.0},
            {"source": "RELEASE", "target": "FIN", "frequency": 10, "median_duration_seconds": 2666622.0},
            {"source": "CODE OK", "target": "DELETE", "frequency": 10, "median_duration_seconds": 4336982.5},
            {"source": "RELEASE", "target": "SET STATUS", "frequency": 10, "median_duration_seconds": 368578.5},
            {"source": "MANUAL", "target": "MANUAL", "frequency": 9, "median_duration_seconds": 171.0},
            {"source": "CHANGE DIAGN", "target": "JOIN-PAT", "frequency": 8, "median_duration_seconds": 596185.5},
            {"source": "JOIN-PAT", "target": "FIN", "frequency": 8, "median_duration_seconds": 5127431.5},
            {"source": "FIN", "target": "SET STATUS", "frequency": 8, "median_duration_seconds": 739344.5},
            {"source": "RELEASE", "target": "MANUAL", "frequency": 7, "median_duration_seconds": 19738990.0},
            {"source": "RELEASE", "target": "STORNO", "frequency": 7, "median_duration_seconds": 5648782.0},
            {"source": "MANUAL", "target": "CODE OK", "frequency": 7, "median_duration_seconds": 104.0},
            {"source": "REJECT", "target": "RELEASE", "frequency": 7, "median_duration_seconds": 34.0}
        ],
        "start_activities": {"NEW": 9999},
        "end_activities": {"BILLED": 6896, "NEW": 1836, "DELETE": 980, "FIN": 170, "CODE OK": 18, "SET STATUS": 79, "MANUAL": 6, "RELEASE": 6, "JOIN-PAT": 6, "CODE NOK": 1, "CHANGE DIAGN": 1},
        "frequent_variants": [
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
            {"variant": ["NEW", "FIN", "RELEASE", "BILLED"], "count": 36}
        ]
    }
}

DFG_FREQ = {}
DFG_SET = set()
for e in PROCESS["process_abstraction"]["dfg_edges"]:
    DFG_FREQ[(e["source"], e["target"])] = e["frequency"]
    DFG_SET.add((e["source"], e["target"]))

TPROB = {}
for e in PROCESS["transition_probabilities_top"]:
    TPROB[(e["source"], e["target"])] = e["probability"]

DUR = {}
for d in PROCESS["duration_profiles_top"]:
    DUR[(d["source"], d["target"])] = d

FREQUENT_VARIANTS = PROCESS["process_abstraction"]["frequent_variants"]
BLOCK_SPAN = PROCESS["block_span_profile_seconds"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()

def detect_precision(ts):
    if "." in ts:
        frac = ts.split(".")[-1]
        return len(frac)
    return 0

def precision_unit_seconds(precision_digits):
    if precision_digits <= 0:
        return 1.0
    return 10 ** (-precision_digits)

def fmt_ts(dt, precision_digits):
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    if precision_digits <= 0:
        return base
    micros = dt.microsecond
    frac = f"{micros:06d}"[:precision_digits]
    return base + "." + frac

def parse_ts(s):
    return pd.to_datetime(s)

def safe_log(x):
    return math.log(max(x, 1e-12))

def contiguous_variant_bonus(order, left_act, right_act):
    acts = list(order)
    bonus = 0.0
    seqs = []
    if left_act is not None:
        seqs.append([left_act] + acts)
    if right_act is not None:
        seqs.append(acts + [right_act])
    if left_act is not None and right_act is not None:
        seqs.append([left_act] + acts + [right_act])
    seqs.append(acts)
    for fv in FREQUENT_VARIANTS:
        var = fv["variant"]
        cnt = fv["count"]
        for seq in seqs:
            m = len(seq)
            if m > len(var):
                continue
            for i in range(len(var) - m + 1):
                if var[i:i+m] == seq:
                    bonus += math.log1p(cnt) * (1.0 + 0.2 * m)
                    break
    return bonus

def pair_order_evidence(a, b):
    ab = DFG_FREQ.get((a, b), 0)
    ba = DFG_FREQ.get((b, a), 0)
    return ab, ba

def edge_exists(a, b):
    return (a, b) in DFG_SET or (a, b) in TPROB

def score_order(order, left_act, right_act):
    unsupported = 0
    score = 0.0
    pairs = []
    if left_act is not None:
        pairs.append((left_act, order[0], True))
    for i in range(len(order) - 1):
        pairs.append((order[i], order[i+1], False))
    if right_act is not None:
        pairs.append((order[-1], right_act, True))
    for a, b, is_anchor in pairs:
        if edge_exists(a, b):
            freq = DFG_FREQ.get((a, b), 1)
            prob = TPROB.get((a, b), 1e-6)
            score += safe_log(freq + 1.0) + 2.0 * safe_log(prob + 1e-9)
            if is_anchor:
                score += 0.5
        else:
            unsupported += 1
            score -= 50.0 if not is_anchor else 20.0
    for i in range(len(order) - 1):
        a, b = order[i], order[i+1]
        ab, ba = pair_order_evidence(a, b)
        if ab > 0 and ba > 0:
            score -= 0.2 * abs(safe_log(ab + 1) - safe_log(ba + 1))
    score += contiguous_variant_bonus(order, left_act, right_act)
    return unsupported, score

def unique_permutations(items):
    seen = set()
    for p in itertools.permutations(items):
        if p not in seen:
            seen.add(p)
            yield p

def infer_order(block_rows, left_act, right_act):
    activities = [r["Activity"] for r in block_rows]
    n = len(activities)
    perms = []
    if n <= 7:
        perms = list(unique_permutations(activities))
    else:
        counts = Counter(activities)
        fixed = []
        remaining = list(activities)
        while remaining:
            a = remaining.pop(0)
            fixed.append(a)
        perms = [tuple(fixed)]
    scored = []
    for p in perms:
        unsupported, score = score_order(p, left_act, right_act)
        scored.append((unsupported, -score, p))
    scored.sort()
    best = scored[0][2]

    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        best_score = -scored[0][1]
        close = [p for u, ns, p in scored if u == scored[0][0] and (-ns) >= best_score - 1.0]
        if close:
            best = close[0]
    return list(best)

def choose_edge_gap_seconds(a, b, min_unit):
    d = DUR.get((a, b))
    if d:
        if d["share_under_60s"] >= 0.4:
            g = d["p10_seconds"]
        else:
            g = d["p25_seconds"]
        if g is None or g <= 0:
            g = max(min_unit, 1.0)
        return max(g, min_unit)
    if (a, b) in DFG_SET:
        return max(5.0, min_unit)
    ab = DFG_FREQ.get((a, b), 0)
    ba = DFG_FREQ.get((b, a), 0)
    if ab > 0 and ba > 0:
        return max(min_unit, 1.0)
    return max(7.0, min_unit)

def build_nonuniform_gaps(order, min_unit):
    gaps = []
    for i in range(len(order) - 1):
        base = choose_edge_gap_seconds(order[i], order[i+1], min_unit)
        ab = DFG_FREQ.get((order[i], order[i+1]), 0)
        ba = DFG_FREQ.get((order[i+1], order[i]), 0)
        if ab > 0 and ba > 0 and min(ab, ba) / max(ab, ba) > 0.35:
            base = min(base, max(min_unit, 2.0 * min_unit))
        jitter = (i % 3 + 1) * min_unit
        gaps.append(max(base + jitter, min_unit))
    return gaps

def assign_times_forward(observed_dt, order, right_bound_dt, precision_digits):
    min_unit = precision_unit_seconds(precision_digits)
    k = len(order)
    if k == 1:
        return [observed_dt]
    gaps = build_nonuniform_gaps(order, min_unit)
    min_total = (k - 1) * max(1.0, min_unit)
    avail = (right_bound_dt - observed_dt).total_seconds() if right_bound_dt is not None else None

    if avail is not None:
        if avail < min_total:
            avail = min_total
            right_bound_dt = observed_dt + pd.to_timedelta(avail, unit="s")
        total = sum(gaps)
        target_cap = avail
        if total > target_cap:
            base_min = [max(1.0, min_unit) + (i % 2) * min_unit for i in range(k - 1)]
            minsum = sum(base_min)
            if target_cap <= minsum:
                gaps = base_min
            else:
                extra = [max(0.0, g - b) for g, b in zip(gaps, base_min)]
                extrasum = sum(extra)
                if extrasum <= 0:
                    gaps = base_min
                else:
                    scale = (target_cap - minsum) / extrasum
                    gaps = [b + e * scale for b, e in zip(base_min, extra)]
        total = sum(gaps)
        if total > target_cap:
            gaps = [max(1.0, min_unit) for _ in gaps]

    times = [observed_dt]
    cur = observed_dt
    for g in gaps:
        cur = cur + pd.to_timedelta(g, unit="s")
        times.append(cur)

    times = enforce_strictly_increasing(times, precision_digits)
    if right_bound_dt is not None and times[-1] > right_bound_dt:
        times = compress_to_right_bound(times, observed_dt, right_bound_dt, precision_digits)
    return times

def assign_times_backward(left_bound_dt, observed_dt, order, precision_digits):
    min_unit = precision_unit_seconds(precision_digits)
    k = len(order)
    if k == 1:
        return [observed_dt]
    rev = list(reversed(order))
    gaps = build_nonuniform_gaps(rev, min_unit)
    avail = (observed_dt - left_bound_dt).total_seconds() if left_bound_dt is not None else None
    min_total = (k - 1) * max(1.0, min_unit)
    if avail is not None and avail < min_total:
        left_bound_dt = observed_dt - pd.to_timedelta(min_total, unit="s")
        avail = min_total
    if avail is not None:
        total = sum(gaps)
        if total > avail:
            base_min = [max(1.0, min_unit) + (i % 2) * min_unit for i in range(k - 1)]
            minsum = sum(base_min)
            if avail <= minsum:
                gaps = base_min
            else:
                extra = [max(0.0, g - b) for g, b in zip(gaps, base_min)]
                extrasum = sum(extra)
                scale = 0.0 if extrasum <= 0 else (avail - minsum) / extrasum
                gaps = [b + e * scale for b, e in zip(base_min, extra)]
    times_rev = [observed_dt]
    cur = observed_dt
    for g in gaps:
        cur = cur - pd.to_timedelta(g, unit="s")
        times_rev.append(cur)
    times = list(reversed(times_rev))
    times = enforce_strictly_increasing(times, precision_digits)
    if left_bound_dt is not None and times[0] < left_bound_dt:
        times = shift_right_from_left_bound(times, left_bound_dt, observed_dt, precision_digits)
    return times

def enforce_strictly_increasing(times, precision_digits):
    unit = precision_unit_seconds(precision_digits)
    fixed = [times[0]]
    for i in range(1, len(times)):
        t = times[i]
        min_allowed = fixed[-1] + pd.to_timedelta(unit, unit="s")
        if t <= fixed[-1]:
            t = min_allowed
        elif t < min_allowed:
            t = min_allowed
        fixed.append(t)
    return fixed

def compress_to_right_bound(times, left_anchor, right_bound, precision_digits):
    unit = precision_unit_seconds(precision_digits)
    k = len(times)
    min_total = (k - 1) * unit
    avail = (right_bound - left_anchor).total_seconds()
    if avail < min_total:
        right_bound = left_anchor + pd.to_timedelta(min_total, unit="s")
    offsets = [(t - left_anchor).total_seconds() for t in times]
    total = max(offsets[-1], min_total)
    target = (right_bound - left_anchor).total_seconds()
    if total <= 0:
        total = min_total
    scale = min(1.0, target / total) if total > 0 else 1.0
    new_offsets = [0.0]
    for i in range(1, k):
        raw = offsets[i] * scale
        min_off = new_offsets[i-1] + unit
        new_offsets.append(max(raw, min_off))
    if new_offsets[-1] > target:
        new_offsets = [i * unit for i in range(k)]
        if new_offsets[-1] > target:
            target = new_offsets[-1]
    return [left_anchor + pd.to_timedelta(off, unit="s") for off in new_offsets]

def shift_right_from_left_bound(times, left_bound, observed_dt, precision_digits):
    unit = precision_unit_seconds(precision_digits)
    k = len(times)
    shifted = [left_bound + pd.to_timedelta(i * unit, unit="s") for i in range(k)]
    if shifted[-1] > observed_dt:
        shifted[-1] = observed_dt
        for i in range(k - 2, -1, -1):
            max_allowed = shifted[i+1] - pd.to_timedelta(unit, unit="s")
            if shifted[i] > max_allowed:
                shifted[i] = max_allowed
    return enforce_strictly_increasing(shifted, precision_digits)

def map_times_to_rows(block_rows, inferred_order, inferred_times, observed_ts_str):
    act_to_queue = defaultdict(list)
    for idx, act in enumerate(inferred_order):
        act_to_queue[act].append(inferred_times[idx])
    assignments = {}
    kept_observed_row = None
    used_first = False
    for row in block_rows:
        act = row["Activity"]
        t = act_to_queue[act].pop(0)
        assignments[row["_idx"]] = t
        if not used_first and t == inferred_times[0] and row["Timestamp"] == observed_ts_str:
            kept_observed_row = row["_idx"]
            used_first = True
    if kept_observed_row is None:
        first_act = inferred_order[0]
        for row in block_rows:
            if row["Activity"] == first_act:
                kept_observed_row = row["_idx"]
                break
        first_time = inferred_times[0]
        for row in block_rows:
            if row["_idx"] == kept_observed_row:
                assignments[row["_idx"]] = first_time
                break
        remaining_rows = [r for r in block_rows if r["_idx"] != kept_observed_row]
        remaining_times = inferred_times[1:]
        rem_map = defaultdict(list)
        for act, t in zip(inferred_order[1:], remaining_times):
            rem_map[act].append(t)
        for row in remaining_rows:
            assignments[row["_idx"]] = rem_map[row["Activity"]].pop(0)
    return assignments, kept_observed_row

def block_room_right(rows_case, start_idx, observed_dt, case_end_dt):
    for j in range(start_idx, len(rows_case)):
        r = rows_case[j]
        if r["error_category"] == "CLEAN":
            return parse_ts(r["Timestamp"])
    return case_end_dt

def nearest_left_clean(rows_case, end_idx):
    for j in range(end_idx, -1, -1):
        r = rows_case[j]
        if r["error_category"] == "CLEAN":
            return r
    return None

def nearest_right_clean(rows_case, start_idx):
    for j in range(start_idx, len(rows_case)):
        r = rows_case[j]
        if r["error_category"] == "CLEAN":
            return r
    return None

def get_case_bounds(rows_case):
    clean_ts = [parse_ts(r["Timestamp"]) for r in rows_case if r["error_category"] == "CLEAN"]
    if clean_ts:
        return min(clean_ts), max(clean_ts)
    all_ts = [parse_ts(r["Timestamp"]) for r in rows_case]
    return min(all_ts), max(all_ts)

def find_c1_blocks(rows):
    blocks = []
    i = 0
    n = len(rows)
    while i < n:
        r = rows[i]
        if r["error_category"] != ERROR_CAT:
            i += 1
            continue
        case = r["Case"]
        ts = r["Timestamp"]
        j = i + 1
        while j < n and rows[j]["Case"] == case and rows[j]["error_category"] == ERROR_CAT and rows[j]["Timestamp"] == ts:
            j += 1
        blocks.append((i, j))
        i = j
    return blocks

def repair_block(rows, case_to_indices, case_groups, start, end, repaired_ts, clamped_counter):
    block_rows = rows[start:end]
    case = block_rows[0]["Case"]
    observed_str = block_rows[0]["Timestamp"]
    observed_dt = parse_ts(observed_str)
    precision_digits = max(detect_precision(r["Timestamp"]) for r in block_rows)
    case_rows = case_groups[case]
    pos0 = case_to_indices[case][start]
    pos1 = case_to_indices[case][end - 1]
    left_clean = nearest_left_clean(case_rows, pos0 - 1)
    right_clean = nearest_right_clean(case_rows, pos1 + 1)
    case_start, case_end = get_case_bounds(case_rows)

    left_act = left_clean["Activity"] if left_clean is not None else None
    right_act = right_clean["Activity"] if right_clean is not None else None
    left_dt = parse_ts(left_clean["Timestamp"]) if left_clean is not None else case_start
    right_dt = parse_ts(right_clean["Timestamp"]) if right_clean is not None else case_end

    order = infer_order(block_rows, left_act, right_act)

    k = len(block_rows)
    unit = precision_unit_seconds(precision_digits)
    min_span = (k - 1) * max(1.0, unit)
    right_room = (right_dt - observed_dt).total_seconds() if right_dt is not None else None
    left_room = (observed_dt - left_dt).total_seconds() if left_dt is not None else None

    use_backward = False
    if right_room is not None and right_room < min_span and left_room is not None and left_room >= min_span:
        anchor_bad = right_act is not None and not edge_exists(order[-1], right_act)
        left_good = left_act is None or edge_exists(left_act, order[0])
        if left_good:
            use_backward = True

    if use_backward:
        times = assign_times_backward(left_dt, observed_dt, order, precision_digits)
        if times[-1] != observed_dt:
            times[-1] = observed_dt
            for i in range(len(times) - 2, -1, -1):
                max_allowed = times[i+1] - pd.to_timedelta(unit, unit="s")
                if times[i] >= max_allowed:
                    times[i] = max_allowed
            times = enforce_strictly_increasing(times, precision_digits)
    else:
        times = assign_times_forward(observed_dt, order, right_dt, precision_digits)
        if times[0] != observed_dt:
            times[0] = observed_dt
            times = enforce_strictly_increasing(times, precision_digits)
            if right_dt is not None and times[-1] > right_dt:
                times = compress_to_right_bound(times, observed_dt, right_dt, precision_digits)

    if right_dt is not None and times[-1] > right_dt:
        times = compress_to_right_bound(times, times[0], right_dt, precision_digits)
        clamped_counter[0] += 1
    if left_dt is not None and times[0] < left_dt:
        times = shift_right_from_left_bound(times, left_dt, observed_dt if use_backward else right_dt, precision_digits)
        clamped_counter[0] += 1

    assignments, kept_idx = map_times_to_rows(block_rows, order, times, observed_str)

    ordered_rows = []
    row_buckets = defaultdict(list)
    for r in block_rows:
        row_buckets[r["Activity"]].append(r)
    for act, t in zip(order, times):
        row = row_buckets[act].pop(0)
        ordered_rows.append((row["_idx"], t))
    earliest_idx = ordered_rows[0][0]

    if kept_idx != earliest_idx:
        earliest_time = times[0]
        current_time_at_earliest = assignments[earliest_idx]
        assignments[kept_idx] = current_time_at_earliest
        assignments[earliest_idx] = earliest_time

    for r in block_rows:
        repaired_ts[r["_idx"]] = fmt_ts(assignments[r["_idx"]], precision_digits)

def self_check_and_fix(rows, repaired_ts, blocks):
    clamped = 0
    rows_equal = 0
    for start, end in blocks:
        block = rows[start:end]
        obs = block[0]["Timestamp"]
        precision_digits = max(detect_precision(r["Timestamp"]) for r in block)
        unit = precision_unit_seconds(precision_digits)
        pair_list = []
        for r in block:
            out = repaired_ts.get(r["_idx"], r["Timestamp"])
            dt = parse_ts(out)
            pair_list.append((r["_idx"], dt, out, r["Timestamp"]))
        equal_rows = [x for x in pair_list if x[2] == x[3]]
        rows_equal += len(equal_rows)

        sorted_by_time = sorted(pair_list, key=lambda x: (x[1], x[0]))
        if len([x for x in pair_list if x[2] == obs]) != 1 or sorted_by_time[0][2] != obs:
            target_idx = sorted_by_time[0][0]
            repaired_ts[target_idx] = obs
            base_dt = parse_ts(obs)
            later = sorted(sorted_by_time[1:], key=lambda x: (x[1], x[0]))
            cur = base_dt
            for idx, _, _, _ in later:
                cur = cur + pd.to_timedelta(max(1.0, unit), unit="s")
                repaired_ts[idx] = fmt_ts(cur, precision_digits)
            clamped += 1

        repaired_dts = [parse_ts(repaired_ts.get(r["_idx"], r["Timestamp"])) for r in block]
        order_idx = sorted(range(len(repaired_dts)), key=lambda i: repaired_dts[i])
        sorted_dts = [repaired_dts[i] for i in order_idx]
        changed = False
        for i in range(1, len(sorted_dts)):
            if not (sorted_dts[i] > sorted_dts[i-1]):
                sorted_dts[i] = sorted_dts[i-1] + pd.to_timedelta(max(1.0, unit), unit="s")
                changed = True
        if changed:
            sorted_rows = [block[i]["_idx"] for i in order_idx]
            for ridx, dt in zip(sorted_rows, sorted_dts):
                repaired_ts[ridx] = fmt_ts(dt, precision_digits)
            clamped += 1

    rows_equal = 0
    for row in rows:
        out = repaired_ts.get(row["_idx"], row["Timestamp"])
        if row["error_category"] == ERROR_CAT and out == row["Timestamp"]:
            rows_equal += 1
    return clamped, rows_equal

def main():
    args = parse_args()
    rows = []
    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, r in enumerate(reader):
            rr = dict(r)
            rr["_idx"] = i
            rows.append(rr)

    case_groups = defaultdict(list)
    case_to_indices = defaultdict(dict)
    for i, r in enumerate(rows):
        case_groups[r["Case"]].append(r)
    for case, group in case_groups.items():
        for pos, r in enumerate(group):
            case_to_indices[case][r["_idx"]] = pos

    repaired_ts = {}
    blocks = find_c1_blocks(rows)
    clamped_counter = [0]

    for start, end in blocks:
        try:
            repair_block(rows, case_to_indices, case_groups, start, end, repaired_ts, clamped_counter)
        except Exception:
            block = rows[start:end]
            observed = block[0]["Timestamp"]
            observed_dt = parse_ts(observed)
            precision_digits = max(detect_precision(r["Timestamp"]) for r in block)
            unit = precision_unit_seconds(precision_digits)
            acts = [r["Activity"] for r in block]
            order = acts[:]
            times = [observed_dt + pd.to_timedelta(i * max(1.0, unit), unit="s") for i in range(len(block))]
            first_row = block[0]["_idx"]
            for idx, t in zip([r["_idx"] for r in block], times):
                repaired_ts[idx] = fmt_ts(t, precision_digits)
            repaired_ts[first_row] = observed
            clamped_counter[0] += 1

    extra_clamped, rows_equal = self_check_and_fix(rows, repaired_ts, blocks)
    total_clamped = clamped_counter[0] + extra_clamped

    out_rows = []
    for r in rows:
        out = dict(r)
        out.pop("_idx", None)
        if r["error_category"] == "CLEAN":
            out["Timestamp"] = r["Timestamp"]
        else:
            out["Timestamp"] = repaired_ts.get(r["_idx"], r["Timestamp"])
            try:
                parse_ts(out["Timestamp"])
            except Exception:
                out["Timestamp"] = r["Timestamp"]
        out_rows.append(out)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"blocks_repaired={len(blocks)} blocks_clamped={total_clamped} rows_still_equal_to_input={rows_equal}")

if __name__ == "__main__":
    main()