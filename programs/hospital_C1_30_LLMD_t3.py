import argparse
import itertools
import json
import math
from collections import Counter, defaultdict

import pandas as pd


PROCESS_CONTEXT = r'''
{
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
'''


def load_context():
    ctx = json.loads(PROCESS_CONTEXT)
    prob = {}
    for x in ctx["transition_probabilities_top"]:
        prob[(x["source"], x["target"])] = x["probability"]
    dur = {}
    for x in ctx["duration_profiles_top"]:
        dur[(x["source"], x["target"])] = x
    freq = {}
    for x in ctx["process_abstraction"]["dfg_edges"]:
        freq[(x["source"], x["target"])] = x["frequency"]
    variants = [(v["variant"], v["count"]) for v in ctx["process_abstraction"]["frequent_variants"]]
    return prob, dur, freq, variants


TRANS_PROB, DUR_PROF, DFG_FREQ, VARIANTS = load_context()


def detect_precision(ts: str):
    if "." in ts:
        frac = ts.split(".")[-1]
        return len(frac)
    return 0


def format_ts(ts: pd.Timestamp, precision: int):
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    s = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
    return s[:-6 + precision]


def unit_ns_for_precision(precision: int) -> int:
    if precision <= 0:
        return 1_000_000_000
    if precision <= 6:
        return 10 ** (9 - precision * 3) if precision in (1, 2, 3) else 10 ** (6 - precision) * 1000
    return 1


def unit_delta(precision: int) -> pd.Timedelta:
    if precision == 0:
        return pd.Timedelta(seconds=1)
    if precision <= 6:
        return pd.Timedelta(microseconds=10 ** (6 - precision))
    return pd.Timedelta(nanoseconds=1)


def pair_gap_seconds(a, b, precision):
    prof = DUR_PROF.get((a, b))
    if prof:
        share = prof.get("share_under_60s", 0.0)
        gap = prof["p10_seconds"] if share >= 0.4 else prof["p25_seconds"]
        gap = max(gap, unit_delta(precision).total_seconds())
        return gap
    return None


def contiguous_subsequence_bonus(order, left_act, right_act):
    seq = []
    if left_act is not None:
        seq.append(left_act)
    seq.extend(order)
    if right_act is not None:
        seq.append(right_act)
    bonus = 0.0
    for var, count in VARIANTS:
        m = len(seq)
        n = len(var)
        if m > n:
            continue
        for i in range(n - m + 1):
            if var[i:i + m] == seq:
                bonus += math.log1p(count) * 3.0
                break
        core = order
        mc = len(core)
        for i in range(n - mc + 1):
            if var[i:i + mc] == core:
                bonus += math.log1p(count) * 1.5
                break
    return bonus


def score_order(order, left_act, right_act):
    impossible_internal = 0
    internal_score = 0.0
    existing_internal = 0
    for a, b in zip(order[:-1], order[1:]):
        f = DFG_FREQ.get((a, b), 0)
        p = TRANS_PROB.get((a, b), 0.0)
        if f <= 0:
            impossible_internal += 1
            internal_score -= 30.0
        else:
            existing_internal += 1
            internal_score += math.log1p(f) + 2.0 * math.log(max(p, 1e-9))
    anchor_score = 0.0
    if left_act is not None:
        f = DFG_FREQ.get((left_act, order[0]), 0)
        p = TRANS_PROB.get((left_act, order[0]), 0.0)
        anchor_score += math.log1p(f) + math.log(max(p, 1e-9)) if f > 0 else -6.0
    if right_act is not None:
        f = DFG_FREQ.get((order[-1], right_act), 0)
        p = TRANS_PROB.get((order[-1], right_act), 0.0)
        anchor_score += math.log1p(f) + math.log(max(p, 1e-9)) if f > 0 else -6.0
    bonus = contiguous_subsequence_bonus(order, left_act, right_act)
    return {
        "order": order,
        "impossible_internal": impossible_internal,
        "existing_internal": existing_internal,
        "score": internal_score + anchor_score + bonus,
    }


def enumerate_orders(activities, left_act, right_act):
    idxs_by_act = defaultdict(list)
    for i, a in enumerate(activities):
        idxs_by_act[a].append(i)
    unique_perms = set(itertools.permutations(activities))
    scored = [score_order(list(p), left_act, right_act) for p in unique_perms]
    min_imp = min(x["impossible_internal"] for x in scored)
    candidates = [x for x in scored if x["impossible_internal"] == min_imp]
    best = max(candidates, key=lambda x: (x["score"], x["existing_internal"]))
    return best["order"]


def build_gaps(order, precision):
    base_unit = unit_delta(precision).total_seconds()
    gaps = []
    for i, (a, b) in enumerate(zip(order[:-1], order[1:])):
        g = pair_gap_seconds(a, b, precision)
        if g is None:
            g = base_unit * (2 ** i)
        if g <= 0:
            g = base_unit * (i + 1)
        gaps.append(float(g))
    return gaps


def enforce_strict_times(times, start_ts, lower_ts, upper_ts, precision):
    unit = unit_delta(precision)
    n = len(times)
    out = list(times)
    if out[0] < start_ts:
        out[0] = start_ts
    for i in range(1, n):
        min_allowed = out[i - 1] + unit
        if out[i] < min_allowed:
            out[i] = min_allowed
    if lower_ts is not None and out[0] < lower_ts:
        out[0] = lower_ts
        for i in range(1, n):
            min_allowed = out[i - 1] + unit
            if out[i] < min_allowed:
                out[i] = min_allowed
    if upper_ts is not None and out[-1] > upper_ts:
        out[-1] = upper_ts
        for i in range(n - 2, -1, -1):
            max_allowed = out[i + 1] - unit
            if out[i] > max_allowed:
                out[i] = max_allowed
        if out[0] < start_ts:
            out[0] = start_ts
            for i in range(1, n):
                min_allowed = out[i - 1] + unit
                if out[i] < min_allowed:
                    out[i] = min_allowed
    return out


def place_block(order, observed_ts, right_anchor_ts, left_anchor_ts, case_end_ts, precision):
    k = len(order)
    if k == 1:
        return [observed_ts]
    gaps = build_gaps(order, precision)
    unit = unit_delta(precision)
    min_gap_sec = unit.total_seconds()
    min_total_sec = (k - 1) * min_gap_sec

    if right_anchor_ts is not None:
        room_sec = max(0.0, (right_anchor_ts - observed_ts).total_seconds())
        max_total_sec = max(0.0, room_sec)
    else:
        upper = case_end_ts if case_end_ts is not None else observed_ts + pd.Timedelta(days=3650)
        max_total_sec = max(0.0, (upper - observed_ts).total_seconds())

    raw_total = sum(gaps)
    if raw_total <= 0:
        raw_total = min_total_sec

    if max_total_sec < min_total_sec:
        max_total_sec = min_total_sec

    scale = 1.0
    if raw_total > max_total_sec:
        scale = max_total_sec / raw_total if raw_total > 0 else 1.0

    scaled = [max(g * scale, min_gap_sec) for g in gaps]
    total = sum(scaled)

    if total > max_total_sec and total > 0:
        excess = total - max_total_sec
        adjustable = [max(0.0, g - min_gap_sec) for g in scaled]
        adj_sum = sum(adjustable)
        if adj_sum > 0:
            scaled = [
                g - excess * ((g - min_gap_sec) / adj_sum) if g > min_gap_sec else g
                for g in scaled
            ]
        else:
            scaled = [min_gap_sec] * len(gaps)

    scaled = [max(g, min_gap_sec) for g in scaled]

    cum = [observed_ts]
    cur = observed_ts
    for i, g in enumerate(scaled):
        step = pd.to_timedelta(g, unit="s")
        cur = cur + step
        min_required = cum[-1] + unit
        if cur < min_required:
            cur = min_required
        cum.append(cur)

    upper_ts = right_anchor_ts if right_anchor_ts is not None else case_end_ts
    lower_ts = max([x for x in [left_anchor_ts, observed_ts] if x is not None])
    cum = enforce_strict_times(cum, observed_ts, lower_ts, upper_ts, precision)
    return cum


def choose_row_assignment(block_df, ordered_activities, ordered_times):
    rows_by_act = defaultdict(list)
    for idx, row in block_df.iterrows():
        rows_by_act[row["Activity"]].append(idx)
    for act in rows_by_act:
        rows_by_act[act] = sorted(rows_by_act[act])

    times_by_act = defaultdict(list)
    for act, t in zip(ordered_activities, ordered_times):
        times_by_act[act].append(t)

    assignment = {}
    for act in times_by_act:
        row_idxs = rows_by_act[act]
        ts_list = times_by_act[act]
        for ridx, ts in zip(row_idxs, ts_list):
            assignment[ridx] = ts
    return assignment


def find_blocks(df):
    blocks = []
    n = len(df)
    i = 0
    while i < n:
        row = df.iloc[i]
        if row["error_category"] == "CLEAN":
            i += 1
            continue
        case = row["Case"]
        ts = row["Timestamp"]
        j = i + 1
        while j < n:
            r = df.iloc[j]
            if r["Case"] == case and r["error_category"] != "CLEAN" and r["Timestamp"] == ts:
                j += 1
            else:
                break
        blocks.append((i, j))
        i = j
    return blocks


def previous_clean_index(df, start):
    case = df.iloc[start]["Case"]
    for i in range(start - 1, -1, -1):
        if df.iloc[i]["Case"] != case:
            break
        if df.iloc[i]["error_category"] == "CLEAN":
            return i
    return None


def next_clean_index(df, end_excl):
    case = df.iloc[end_excl - 1]["Case"]
    for i in range(end_excl, len(df)):
        if df.iloc[i]["Case"] != case:
            break
        if df.iloc[i]["error_category"] == "CLEAN":
            return i
    return None


def case_bounds(df):
    bounds = {}
    for case, g in df.groupby("Case", sort=False):
        parsed = pd.to_datetime(g["Timestamp"], errors="coerce")
        bounds[case] = (parsed.min(), parsed.max())
    return bounds


def self_check_and_fix_block(df, out_ts, start, end, precision, clamped_counter):
    block = df.iloc[start:end]
    observed_str = block.iloc[0]["Timestamp"]
    observed_ts = pd.to_datetime(observed_str)
    case = block.iloc[0]["Case"]
    left_i = previous_clean_index(df, start)
    right_i = next_clean_index(df, end)
    left_ts = pd.to_datetime(df.at[left_i, "Timestamp"]) if left_i is not None else None
    right_ts = pd.to_datetime(df.at[right_i, "Timestamp"]) if right_i is not None else None
    unit = unit_delta(precision)

    repaired_idxs = list(block.index)
    repaired_times = [(idx, pd.to_datetime(out_ts[idx])) for idx in repaired_idxs]
    earliest_val = min(t for _, t in repaired_times)
    keepers = [idx for idx, t in repaired_times if t == observed_ts]

    need_fix = False
    if len(set(t for _, t in repaired_times)) != len(repaired_times):
        need_fix = True
    if len(keepers) != 1:
        need_fix = True
    if earliest_val != observed_ts:
        need_fix = True
    if right_ts is not None:
        if max(t for _, t in repaired_times) > right_ts:
            need_fix = True
    if left_ts is not None:
        if min(t for _, t in repaired_times) < left_ts:
            need_fix = True

    if need_fix:
        clamped_counter[0] += 1
        ordered = sorted(repaired_times, key=lambda x: (x[1], x[0]))
        fixed = [observed_ts]
        for i in range(1, len(ordered)):
            fixed.append(fixed[-1] + unit)
        if right_ts is not None and fixed[-1] > right_ts:
            last = right_ts
            fixed[-1] = last
            for i in range(len(fixed) - 2, -1, -1):
                cand = fixed[i + 1] - unit
                if cand < observed_ts + unit * i:
                    cand = observed_ts + unit * i
                fixed[i] = cand
            fixed[0] = observed_ts
            for i in range(1, len(fixed)):
                if fixed[i] <= fixed[i - 1]:
                    fixed[i] = fixed[i - 1] + unit
        if left_ts is not None and fixed[0] < left_ts:
            fixed[0] = max(left_ts, observed_ts)
            for i in range(1, len(fixed)):
                fixed[i] = max(fixed[i], fixed[i - 1] + unit)
            if right_ts is not None and fixed[-1] > right_ts:
                fixed[-1] = right_ts
                for i in range(len(fixed) - 2, -1, -1):
                    fixed[i] = min(fixed[i], fixed[i + 1] - unit)
                if fixed[0] != observed_ts:
                    fixed[0] = max(observed_ts, left_ts)
                    for i in range(1, len(fixed)):
                        if fixed[i] <= fixed[i - 1]:
                            fixed[i] = fixed[i - 1] + unit
        for (idx, _), ts in zip(ordered, fixed):
            out_ts[idx] = format_ts(ts, precision)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.DataFrame(columns=["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])

    if len(df) == 0:
        df.to_csv(args.output, index=False)
        print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")
        return

    original_ts = df["Timestamp"].copy()
    out_ts = original_ts.copy()

    precision = 0
    for s in df["Timestamp"].tolist():
        precision = max(precision, detect_precision(s))

    c_bounds = case_bounds(df)
    blocks = find_blocks(df)
    blocks_repaired = 0
    clamped_counter = [0]

    for start, end in blocks:
        block = df.iloc[start:end]
        case = block.iloc[0]["Case"]
        observed_str = block.iloc[0]["Timestamp"]
        observed_ts = pd.to_datetime(observed_str)
        acts = block["Activity"].tolist()

        left_i = previous_clean_index(df, start)
        right_i = next_clean_index(df, end)

        left_act = df.at[left_i, "Activity"] if left_i is not None else None
        right_act = df.at[right_i, "Activity"] if right_i is not None else None
        left_ts = pd.to_datetime(df.at[left_i, "Timestamp"]) if left_i is not None else None
        right_ts = pd.to_datetime(df.at[right_i, "Timestamp"]) if right_i is not None else None
        case_start, case_end = c_bounds[case]

        order = enumerate_orders(acts, left_act, right_act)
        ordered_times = place_block(order, observed_ts, right_ts, left_ts, case_end, precision)
        assignment = choose_row_assignment(block, order, ordered_times)

        for idx in block.index:
            new_ts = assignment[idx]
            out_ts.at[idx] = format_ts(new_ts, precision)

        self_check_and_fix_block(df, out_ts, start, end, precision, clamped_counter)
        blocks_repaired += 1

    df["Timestamp"] = out_ts

    # Final global safety pass: preserve exact original strings for CLEAN rows.
    clean_mask = df["error_category"] == "CLEAN"
    df.loc[clean_mask, "Timestamp"] = original_ts.loc[clean_mask]

    # Ensure every corrupted row except one per block changed.
    for start, end in blocks:
        block_idx = list(df.iloc[start:end].index)
        obs = original_ts.iloc[start]
        same_idxs = [idx for idx in block_idx if df.at[idx, "Timestamp"] == original_ts.at[idx]]
        if len(same_idxs) != 1:
            observed_ts = pd.to_datetime(obs)
            unit = unit_delta(precision)
            parsed = [(idx, pd.to_datetime(df.at[idx, "Timestamp"])) for idx in block_idx]
            keeper = min(parsed, key=lambda x: (x[1], x[0]))[0]
            ordered = sorted(parsed, key=lambda x: (x[1], x[0]))
            current = observed_ts
            for idx, _ in ordered:
                if idx == keeper:
                    df.at[idx, "Timestamp"] = format_ts(observed_ts, precision)
                    current = observed_ts
                else:
                    current = current + unit
                    if format_ts(current, precision) == original_ts.at[idx]:
                        current = current + unit
                    df.at[idx, "Timestamp"] = format_ts(current, precision)

    rows_still_equal = int((df["Timestamp"] == original_ts).sum())

    try:
        df.to_csv(args.output, index=False)
    except Exception:
        try:
            pd.DataFrame(df).to_csv(args.output, index=False)
        except Exception:
            pass

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={clamped_counter[0]} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    main()