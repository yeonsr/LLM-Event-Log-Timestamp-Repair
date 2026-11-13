import argparse
import csv
import math
import itertools
from collections import defaultdict, Counter
import pandas as pd

ERROR_CAT = "C1_ORDERING_AMBIGUITY"

PROCESS = {
    "transition_probabilities_top": [
        {"source": "Deliver card", "target": "EVENT 13 END", "probability": 1.0},
        {"source": "EVENT 13 END", "target": "EVENT 13 END", "probability": 1.0},
        {"source": "Notify accept", "target": "Deliver card", "probability": 1.0},
        {"source": "time out", "target": "EVENT 13 END", "probability": 1.0},
        {"source": "Perform checks", "target": "Make decision", "probability": 0.9919},
        {"source": "review request received", "target": "Check for completeness", "probability": 0.9708},
        {"source": "info received", "target": "Check for completeness", "probability": 0.9662},
        {"source": "Request info", "target": "info received", "probability": 0.962},
        {"source": "New online application received", "target": "Check for completeness", "probability": 0.91},
        {"source": "notify reject", "target": "time out", "probability": 0.5226},
        {"source": "Check for completeness", "target": "Request info", "probability": 0.5124},
        {"source": "Make decision", "target": "notify reject", "probability": 0.5079},
        {"source": "Make decision", "target": "Notify accept", "probability": 0.4883},
        {"source": "notify reject", "target": "review request received", "probability": 0.465},
        {"source": "Check for completeness", "target": "Perform checks", "probability": 0.4649},
        {"source": "New online application received", "target": "Notify accept", "probability": 0.0276},
        {"source": "New online application received", "target": "Make decision", "probability": 0.0234},
        {"source": "New online application received", "target": "notify reject", "probability": 0.0164},
        {"source": "info received", "target": "Make decision", "probability": 0.0106},
        {"source": "New online application received", "target": "info received", "probability": 0.0106},
        {"source": "Check for completeness", "target": "Notify accept", "probability": 0.0085},
        {"source": "Request info", "target": "Perform checks", "probability": 0.0084},
        {"source": "info received", "target": "info received", "probability": 0.0084},
        {"source": "Request info", "target": "Make decision", "probability": 0.0079},
        {"source": "review request received", "target": "Make decision", "probability": 0.0076},
        {"source": "Request info", "target": "Request info", "probability": 0.0068},
        {"source": "New online application received", "target": "Perform checks", "probability": 0.0066},
        {"source": "Check for completeness", "target": "Make decision", "probability": 0.0059},
        {"source": "review request received", "target": "Notify accept", "probability": 0.0057},
        {"source": "review request received", "target": "Perform checks", "probability": 0.0057},
        {"source": "Request info", "target": "Check for completeness", "probability": 0.0054},
        {"source": "Check for completeness", "target": "notify reject", "probability": 0.0049},
        {"source": "Request info", "target": "Notify accept", "probability": 0.0047},
        {"source": "Request info", "target": "notify reject", "probability": 0.0047},
        {"source": "info received", "target": "notify reject", "probability": 0.0047},
        {"source": "info received", "target": "Perform checks", "probability": 0.0043},
        {"source": "info received", "target": "Notify accept", "probability": 0.0039},
        {"source": "notify reject", "target": "Make decision", "probability": 0.0036},
        {"source": "New online application received", "target": "Request info", "probability": 0.0034},
        {"source": "review request received", "target": "Request info", "probability": 0.0032},
        {"source": "review request received", "target": "notify reject", "probability": 0.0032},
        {"source": "notify reject", "target": "Perform checks", "probability": 0.0027},
        {"source": "Perform checks", "target": "Check for completeness", "probability": 0.0026},
        {"source": "Perform checks", "target": "review request received", "probability": 0.0026},
        {"source": "review request received", "target": "info received", "probability": 0.0025},
        {"source": "New online application received", "target": "review request received", "probability": 0.002},
        {"source": "Perform checks", "target": "Request info", "probability": 0.0019},
        {"source": "notify reject", "target": "Check for completeness", "probability": 0.0018},
        {"source": "Check for completeness", "target": "info received", "probability": 0.0016},
        {"source": "notify reject", "target": "notify reject", "probability": 0.0015},
        {"source": "info received", "target": "Request info", "probability": 0.0014},
        {"source": "review request received", "target": "review request received", "probability": 0.0013},
        {"source": "Make decision", "target": "info received", "probability": 0.0012},
        {"source": "Check for completeness", "target": "Check for completeness", "probability": 0.0011},
        {"source": "Perform checks", "target": "Perform checks", "probability": 0.001},
        {"source": "Make decision", "target": "Make decision", "probability": 0.001},
        {"source": "notify reject", "target": "Notify accept", "probability": 0.0009},
        {"source": "notify reject", "target": "Request info", "probability": 0.0009},
        {"source": "notify reject", "target": "info received", "probability": 0.0009},
        {"source": "Make decision", "target": "Perform checks", "probability": 0.0008},
    ],
    "duration_profiles_top": [
        {"source": "Check for completeness", "target": "Request info", "n": 6227, "p10_seconds": 61.931400000000004, "p25_seconds": 170.11149999999998, "median_seconds": 418.728, "p75_seconds": 830.2675, "p90_seconds": 1373.2918000000009, "share_under_60s": 0.097},
        {"source": "Request info", "target": "info received", "n": 6080, "p10_seconds": 10.0, "p25_seconds": 10.0, "median_seconds": 10.0, "p75_seconds": 10.0, "p90_seconds": 10.0, "share_under_60s": 0.996},
        {"source": "info received", "target": "Check for completeness", "n": 6008, "p10_seconds": 63.73900000000001, "p25_seconds": 176.05, "median_seconds": 420.682, "p75_seconds": 839.701, "p90_seconds": 1404.5438000000004, "share_under_60s": 0.095},
        {"source": "Perform checks", "target": "Make decision", "n": 5745, "p10_seconds": 62.8388, "p25_seconds": 171.483, "median_seconds": 422.924, "p75_seconds": 835.934, "p90_seconds": 1392.8122, "share_under_60s": 0.095},
        {"source": "Check for completeness", "target": "Perform checks", "n": 5650, "p10_seconds": 61.0229, "p25_seconds": 171.5965, "median_seconds": 420.37149999999997, "p75_seconds": 832.8305, "p90_seconds": 1416.3387000000023, "share_under_60s": 0.098},
        {"source": "New online application received", "target": "Check for completeness", "n": 4550, "p10_seconds": 65.3893, "p25_seconds": 171.7955, "median_seconds": 426.0135, "p75_seconds": 834.95025, "p90_seconds": 1385.6881999999998, "share_under_60s": 0.092},
        {"source": "Deliver card", "target": "EVENT 13 END", "n": 3276, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "Notify accept", "target": "Deliver card", "n": 3276, "p10_seconds": 59.4585, "p25_seconds": 176.35075, "median_seconds": 425.762, "p75_seconds": 851.3035, "p90_seconds": 1370.5795, "share_under_60s": 0.101},
        {"source": "Make decision", "target": "notify reject", "n": 3088, "p10_seconds": 63.79740000000002, "p25_seconds": 174.02075000000002, "median_seconds": 426.0405, "p75_seconds": 834.3485000000001, "p90_seconds": 1427.2486000000006, "share_under_60s": 0.095},
        {"source": "Make decision", "target": "Notify accept", "n": 2969, "p10_seconds": 65.5868, "p25_seconds": 174.826, "median_seconds": 428.03, "p75_seconds": 867.335, "p90_seconds": 1436.6726, "share_under_60s": 0.089},
        {"source": "time out", "target": "EVENT 13 END", "n": 1724, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "notify reject", "target": "time out", "n": 1724, "p10_seconds": 10.0, "p25_seconds": 10.0, "median_seconds": 10.0, "p75_seconds": 10.0, "p90_seconds": 10.0, "share_under_60s": 1.0},
        {"source": "notify reject", "target": "review request received", "n": 1534, "p10_seconds": 10.0, "p25_seconds": 10.0, "median_seconds": 10.0, "p75_seconds": 10.0, "p90_seconds": 10.0, "share_under_60s": 1.0},
        {"source": "review request received", "target": "Check for completeness", "n": 1527, "p10_seconds": 78.02420000000001, "p25_seconds": 188.8875, "median_seconds": 441.348, "p75_seconds": 841.8565, "p90_seconds": 1413.9960000000003, "share_under_60s": 0.075},
        {"source": "New online application received", "target": "Notify accept", "n": 138, "p10_seconds": 944.0129000000001, "p25_seconds": 1469.49875, "median_seconds": 2020.112, "p75_seconds": 2906.0237500000003, "p90_seconds": 3878.7889999999998, "share_under_60s": 0.0},
        {"source": "New online application received", "target": "Make decision", "n": 117, "p10_seconds": 634.1, "p25_seconds": 963.121, "median_seconds": 1661.293, "p75_seconds": 2178.854, "p90_seconds": 3090.3764, "share_under_60s": 0.0},
        {"source": "Check for completeness", "target": "Notify accept", "n": 103, "p10_seconds": 711.8126, "p25_seconds": 1094.916, "median_seconds": 1780.137, "p75_seconds": 2325.416, "p90_seconds": 3107.586599999999, "share_under_60s": 0.0},
        {"source": "New online application received", "target": "notify reject", "n": 82, "p10_seconds": 1132.4149, "p25_seconds": 1538.03275, "median_seconds": 2159.5775, "p75_seconds": 2824.4955, "p90_seconds": 3498.6097, "share_under_60s": 0.0},
        {"source": "Check for completeness", "target": "Make decision", "n": 72, "p10_seconds": 1072.0155, "p25_seconds": 1536.0985, "median_seconds": 2023.867, "p75_seconds": 2917.63875, "p90_seconds": 3699.0501, "share_under_60s": 0.0},
        {"source": "info received", "target": "Make decision", "n": 66, "p10_seconds": 721.489, "p25_seconds": 1122.1032500000001, "median_seconds": 1868.1909999999998, "p75_seconds": 2613.01725, "p90_seconds": 3118.5855, "share_under_60s": 0.0},
        {"source": "Check for completeness", "target": "notify reject", "n": 60, "p10_seconds": 945.7493000000001, "p25_seconds": 1361.11625, "median_seconds": 1757.751, "p75_seconds": 2556.73125, "p90_seconds": 3239.0085000000004, "share_under_60s": 0.0},
        {"source": "Request info", "target": "Perform checks", "n": 53, "p10_seconds": 403.0504, "p25_seconds": 643.92, "median_seconds": 928.08, "p75_seconds": 1395.19, "p90_seconds": 2110.2466, "share_under_60s": 0.0},
        {"source": "New online application received", "target": "info received", "n": 53, "p10_seconds": 452.3138, "p25_seconds": 639.021, "median_seconds": 857.047, "p75_seconds": 1303.053, "p90_seconds": 1811.6944000000012, "share_under_60s": 0.0},
        {"source": "info received", "target": "info received", "n": 52, "p10_seconds": 613.9782, "p25_seconds": 895.2945, "median_seconds": 1302.5635, "p75_seconds": 1835.0997499999999, "p90_seconds": 2340.9239, "share_under_60s": 0.0},
        {"source": "Request info", "target": "Make decision", "n": 50, "p10_seconds": 669.1557, "p25_seconds": 855.869, "median_seconds": 1523.8755, "p75_seconds": 2372.5625, "p90_seconds": 3437.5409000000004, "share_under_60s": 0.0},
        {"source": "Request info", "target": "Request info", "n": 43, "p10_seconds": 569.524, "p25_seconds": 796.317, "median_seconds": 1164.851, "p75_seconds": 1761.0320000000002, "p90_seconds": 2732.683, "share_under_60s": 0.0},
        {"source": "Request info", "target": "Check for completeness", "n": 34, "p10_seconds": 803.7977000000001, "p25_seconds": 952.73675, "median_seconds": 1889.409, "p75_seconds": 2738.841, "p90_seconds": 4159.568299999999, "share_under_60s": 0.0},
        {"source": "New online application received", "target": "Perform checks", "n": 33, "p10_seconds": 1091.8104, "p25_seconds": 1559.19, "median_seconds": 2363.648, "p75_seconds": 3302.949, "p90_seconds": 3864.007, "share_under_60s": 0.0},
        {"source": "Request info", "target": "Notify accept", "n": 30, "p10_seconds": 1429.9083, "p25_seconds": 1610.589, "median_seconds": 2396.886, "p75_seconds": 3179.76125, "p90_seconds": 3601.6131, "share_under_60s": 0.0},
        {"source": "Request info", "target": "notify reject", "n": 30, "p10_seconds": 1539.1919, "p25_seconds": 1979.3754999999999, "median_seconds": 2454.9030000000002, "p75_seconds": 2997.263, "p90_seconds": 3804.9144, "share_under_60s": 0.0},
        {"source": "info received", "target": "notify reject", "n": 29, "p10_seconds": 1059.5184, "p25_seconds": 1682.395, "median_seconds": 2806.157, "p75_seconds": 3442.502, "p90_seconds": 3679.6308, "share_under_60s": 0.0},
        {"source": "info received", "target": "Perform checks", "n": 27, "p10_seconds": 809.8986, "p25_seconds": 1087.906, "median_seconds": 1874.355, "p75_seconds": 2956.759, "p90_seconds": 4336.3136, "share_under_60s": 0.0},
        {"source": "info received", "target": "Notify accept", "n": 24, "p10_seconds": 1006.2042, "p25_seconds": 1510.50375, "median_seconds": 2230.734, "p75_seconds": 3852.069, "p90_seconds": 4100.492, "share_under_60s": 0.0},
        {"source": "Check for completeness", "target": "info received", "n": 19, "p10_seconds": 817.942, "p25_seconds": 1530.7269999999999, "median_seconds": 1910.593, "p75_seconds": 2581.938, "p90_seconds": 2788.1602, "share_under_60s": 0.0},
        {"source": "New online application received", "target": "Request info", "n": 17, "p10_seconds": 809.1351999999999, "p25_seconds": 1687.418, "median_seconds": 2254.706, "p75_seconds": 2700.696, "p90_seconds": 4232.184200000001, "share_under_60s": 0.0},
        {"source": "Perform checks", "target": "Check for completeness", "n": 15, "p10_seconds": 1023.0088000000001, "p25_seconds": 1420.4865, "median_seconds": 1995.413, "p75_seconds": 2965.38, "p90_seconds": 3838.7202, "share_under_60s": 0.0},
        {"source": "Perform checks", "target": "review request received", "n": 15, "p10_seconds": 378.41200000000003, "p25_seconds": 555.4955, "median_seconds": 705.0, "p75_seconds": 849.476, "p90_seconds": 1327.2799999999997, "share_under_60s": 0.0},
        {"source": "Check for completeness", "target": "Check for completeness", "n": 13, "p10_seconds": 1452.7526, "p25_seconds": 1939.982, "median_seconds": 2577.014, "p75_seconds": 3296.051, "p90_seconds": 3589.3102000000003, "share_under_60s": 0.0},
        {"source": "notify reject", "target": "Make decision", "n": 12, "p10_seconds": 785.8718, "p25_seconds": 1341.0605, "median_seconds": 1518.5795, "p75_seconds": 1958.461, "p90_seconds": 2274.8716, "share_under_60s": 0.0},
        {"source": "review request received", "target": "Make decision", "n": 12, "p10_seconds": 727.5060000000001, "p25_seconds": 964.28475, "median_seconds": 2227.617, "p75_seconds": 2260.87725, "p90_seconds": 3134.5752, "share_under_60s": 0.0},
        {"source": "Perform checks", "target": "Request info", "n": 11, "p10_seconds": 1288.529, "p25_seconds": 1794.608, "median_seconds": 2625.632, "p75_seconds": 3426.251, "p90_seconds": 4208.01, "share_under_60s": 0.0},
        {"source": "New online application received", "target": "review request received", "n": 10, "p10_seconds": 783.2576, "p25_seconds": 1569.0394999999999, "median_seconds": 1845.21, "p75_seconds": 2201.5355, "p90_seconds": 2416.4853999999996, "share_under_60s": 0.0},
        {"source": "notify reject", "target": "Perform checks", "n": 9, "p10_seconds": 318.1018, "p25_seconds": 699.898, "median_seconds": 874.95, "p75_seconds": 1271.62, "p90_seconds": 1995.7800000000004, "share_under_60s": 0.0},
        {"source": "review request received", "target": "Notify accept", "n": 9, "p10_seconds": 2066.8628, "p25_seconds": 2191.667, "median_seconds": 2296.639, "p75_seconds": 4048.847, "p90_seconds": 4780.2734, "share_under_60s": 0.0},
        {"source": "Check for completeness", "target": "review request received", "n": 9, "p10_seconds": 786.9502, "p25_seconds": 1083.927, "median_seconds": 1363.124, "p75_seconds": 2554.657, "p90_seconds": 2954.3854, "share_under_60s": 0.0},
        {"source": "review request received", "target": "Perform checks", "n": 9, "p10_seconds": 1338.8826, "p25_seconds": 1593.822, "median_seconds": 2249.376, "p75_seconds": 3393.616, "p90_seconds": 4267.802, "share_under_60s": 0.0},
        {"source": "info received", "target": "Request info", "n": 9, "p10_seconds": 824.7008, "p25_seconds": 1305.603, "median_seconds": 1993.345, "p75_seconds": 3112.501, "p90_seconds": 4079.4476, "share_under_60s": 0.0},
        {"source": "EVENT 13 END", "target": "EVENT 13 END", "n": 8, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "Make decision", "target": "info received", "n": 7, "p10_seconds": 615.6284, "p25_seconds": 944.3135, "median_seconds": 1863.824, "p75_seconds": 1941.4715, "p90_seconds": 2142.2018000000003, "share_under_60s": 0.0},
        {"source": "notify reject", "target": "Check for completeness", "n": 6, "p10_seconds": 1769.1770000000001, "p25_seconds": 2725.29575, "median_seconds": 3075.0545, "p75_seconds": 3917.4522500000003, "p90_seconds": 4349.0205000000005, "share_under_60s": 0.0},
        {"source": "Make decision", "target": "Make decision", "n": 6, "p10_seconds": 1349.042, "p25_seconds": 1693.34325, "median_seconds": 2015.0765000000001, "p75_seconds": 2417.1235, "p90_seconds": 2768.227, "share_under_60s": 0.0},
        {"source": "Perform checks", "target": "Perform checks", "n": 6, "p10_seconds": 1054.435, "p25_seconds": 1415.281, "median_seconds": 2294.3645, "p75_seconds": 2990.205, "p90_seconds": 5170.2335, "share_under_60s": 0.0},
        {"source": "review request received", "target": "notify reject", "n": 5, "p10_seconds": 1334.3154, "p25_seconds": 1360.086, "median_seconds": 1469.711, "p75_seconds": 2384.126, "p90_seconds": 3042.9542, "share_under_60s": 0.0},
        {"source": "review request received", "target": "Request info", "n": 5, "p10_seconds": 1863.9608, "p25_seconds": 2128.058, "median_seconds": 2501.573, "p75_seconds": 3114.888, "p90_seconds": 4176.5796, "share_under_60s": 0.0},
        {"source": "Make decision", "target": "Request info", "n": 5, "p10_seconds": 1027.2532, "p25_seconds": 1078.57, "median_seconds": 1325.858, "p75_seconds": 1553.253, "p90_seconds": 2857.4970000000003, "share_under_60s": 0.0},
        {"source": "notify reject", "target": "notify reject", "n": 5, "p10_seconds": 1475.2649999999999, "p25_seconds": 2749.563, "median_seconds": 2967.996, "p75_seconds": 3278.084, "p90_seconds": 3452.2106, "share_under_60s": 0.0},
        {"source": "Make decision", "target": "Perform checks", "n": 5, "p10_seconds": 783.9354, "p25_seconds": 796.233, "median_seconds": 823.397, "p75_seconds": 1650.51, "p90_seconds": 1824.5034, "share_under_60s": 0.0},
        {"source": "review request received", "target": "info received", "n": 4, "p10_seconds": 1391.3396, "p25_seconds": 1447.0055, "median_seconds": 1983.33, "p75_seconds": 2496.487, "p90_seconds": 2510.4514, "share_under_60s": 0.0},
        {"source": "notify reject", "target": "info received", "n": 3, "p10_seconds": 652.6522, "p25_seconds": 721.4095, "median_seconds": 836.005, "p75_seconds": 1576.566, "p90_seconds": 2020.9026000000001, "share_under_60s": 0.0},
        {"source": "notify reject", "target": "Request info", "n": 3, "p10_seconds": 389.1752, "p25_seconds": 613.13, "median_seconds": 986.388, "p75_seconds": 1967.2504999999999, "p90_seconds": 2555.768, "share_under_60s": 0.0},
    ],
    "process_abstraction": {
        "dfg_edges": [
            {"source": "Check for completeness", "target": "Request info", "frequency": 6227, "median_duration_seconds": 418.728},
            {"source": "Request info", "target": "info received", "frequency": 6080, "median_duration_seconds": 10.0},
            {"source": "info received", "target": "Check for completeness", "frequency": 6008, "median_duration_seconds": 420.682},
            {"source": "Perform checks", "target": "Make decision", "frequency": 5745, "median_duration_seconds": 422.924},
            {"source": "Check for completeness", "target": "Perform checks", "frequency": 5650, "median_duration_seconds": 420.37149999999997},
            {"source": "New online application received", "target": "Check for completeness", "frequency": 4550, "median_duration_seconds": 426.0135},
            {"source": "Notify accept", "target": "Deliver card", "frequency": 3276, "median_duration_seconds": 425.762},
            {"source": "Deliver card", "target": "EVENT 13 END", "frequency": 3276, "median_duration_seconds": 0.0},
            {"source": "Make decision", "target": "notify reject", "frequency": 3088, "median_duration_seconds": 426.0405},
            {"source": "Make decision", "target": "Notify accept", "frequency": 2969, "median_duration_seconds": 428.03},
            {"source": "notify reject", "target": "time out", "frequency": 1724, "median_duration_seconds": 10.0},
            {"source": "time out", "target": "EVENT 13 END", "frequency": 1724, "median_duration_seconds": 0.0},
            {"source": "notify reject", "target": "review request received", "frequency": 1534, "median_duration_seconds": 10.0},
            {"source": "review request received", "target": "Check for completeness", "frequency": 1527, "median_duration_seconds": 441.348},
            {"source": "New online application received", "target": "Notify accept", "frequency": 138, "median_duration_seconds": 2020.112},
            {"source": "New online application received", "target": "Make decision", "frequency": 117, "median_duration_seconds": 1661.293},
            {"source": "Check for completeness", "target": "Notify accept", "frequency": 103, "median_duration_seconds": 1780.137},
            {"source": "New online application received", "target": "notify reject", "frequency": 82, "median_duration_seconds": 2159.5775},
            {"source": "Check for completeness", "target": "Make decision", "frequency": 72, "median_duration_seconds": 2023.867},
            {"source": "info received", "target": "Make decision", "frequency": 66, "median_duration_seconds": 1868.1909999999998},
            {"source": "Check for completeness", "target": "notify reject", "frequency": 60, "median_duration_seconds": 1757.751},
            {"source": "New online application received", "target": "info received", "frequency": 53, "median_duration_seconds": 857.047},
            {"source": "Request info", "target": "Perform checks", "frequency": 53, "median_duration_seconds": 928.08},
            {"source": "info received", "target": "info received", "frequency": 52, "median_duration_seconds": 1302.5635},
            {"source": "Request info", "target": "Make decision", "frequency": 50, "median_duration_seconds": 1523.8755},
            {"source": "Request info", "target": "Request info", "frequency": 43, "median_duration_seconds": 1164.851},
            {"source": "Request info", "target": "Check for completeness", "frequency": 34, "median_duration_seconds": 1889.409},
            {"source": "New online application received", "target": "Perform checks", "frequency": 33, "median_duration_seconds": 2363.648},
            {"source": "Request info", "target": "notify reject", "frequency": 30, "median_duration_seconds": 2454.9030000000002},
            {"source": "Request info", "target": "Notify accept", "frequency": 30, "median_duration_seconds": 2396.886},
            {"source": "info received", "target": "notify reject", "frequency": 29, "median_duration_seconds": 2806.157},
            {"source": "info received", "target": "Perform checks", "frequency": 27, "median_duration_seconds": 1874.355},
            {"source": "info received", "target": "Notify accept", "frequency": 24, "median_duration_seconds": 2230.734},
            {"source": "Check for completeness", "target": "info received", "frequency": 19, "median_duration_seconds": 1910.593},
            {"source": "New online application received", "target": "Request info", "frequency": 17, "median_duration_seconds": 2254.706},
            {"source": "Perform checks", "target": "Check for completeness", "frequency": 15, "median_duration_seconds": 1995.413},
            {"source": "Perform checks", "target": "review request received", "frequency": 15, "median_duration_seconds": 705.0},
            {"source": "Check for completeness", "target": "Check for completeness", "frequency": 13, "median_duration_seconds": 2577.014},
            {"source": "notify reject", "target": "Make decision", "frequency": 12, "median_duration_seconds": 1518.5795},
            {"source": "review request received", "target": "Make decision", "frequency": 12, "median_duration_seconds": 2227.617},
            {"source": "Perform checks", "target": "Request info", "frequency": 11, "median_duration_seconds": 2625.632},
            {"source": "New online application received", "target": "review request received", "frequency": 10, "median_duration_seconds": 1845.21},
            {"source": "Check for completeness", "target": "review request received", "frequency": 9, "median_duration_seconds": 1363.124},
            {"source": "review request received", "target": "Notify accept", "frequency": 9, "median_duration_seconds": 2296.639},
            {"source": "info received", "target": "Request info", "frequency": 9, "median_duration_seconds": 1993.345},
            {"source": "notify reject", "target": "Perform checks", "frequency": 9, "median_duration_seconds": 874.95},
            {"source": "review request received", "target": "Perform checks", "frequency": 9, "median_duration_seconds": 2249.376},
            {"source": "EVENT 13 END", "target": "EVENT 13 END", "frequency": 8, "median_duration_seconds": 0.0},
            {"source": "Make decision", "target": "info received", "frequency": 7, "median_duration_seconds": 1863.824},
            {"source": "notify reject", "target": "Check for completeness", "frequency": 6, "median_duration_seconds": 3075.0545},
            {"source": "Make decision", "target": "Make decision", "frequency": 6, "median_duration_seconds": 2015.0765000000001},
            {"source": "Perform checks", "target": "Perform checks", "frequency": 6, "median_duration_seconds": 2294.3645},
            {"source": "notify reject", "target": "notify reject", "frequency": 5, "median_duration_seconds": 2967.996},
            {"source": "Make decision", "target": "Perform checks", "frequency": 5, "median_duration_seconds": 823.397},
            {"source": "review request received", "target": "Request info", "frequency": 5, "median_duration_seconds": 2501.573},
            {"source": "review request received", "target": "notify reject", "frequency": 5, "median_duration_seconds": 1469.711},
            {"source": "Make decision", "target": "Request info", "frequency": 5, "median_duration_seconds": 1325.858},
            {"source": "review request received", "target": "info received", "frequency": 4, "median_duration_seconds": 1983.33},
            {"source": "notify reject", "target": "info received", "frequency": 3, "median_duration_seconds": 836.005},
            {"source": "notify reject", "target": "Request info", "frequency": 3, "median_duration_seconds": 986.388},
        ],
        "frequent_variants": [
            {"variant": ["New online application received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 947},
            {"variant": ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], "count": 531},
            {"variant": ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 437},
            {"variant": ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], "count": 215},
            {"variant": ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 202},
            {"variant": ["New online application received", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 138},
            {"variant": ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], "count": 120},
            {"variant": ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 106},
            {"variant": ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 89},
            {"variant": ["New online application received", "notify reject", "time out", "EVENT 13 END"], "count": 75},
            {"variant": ["New online application received", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 75},
            {"variant": ["New online application received", "Check for completeness", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 72},
            {"variant": ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 61},
            {"variant": ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], "count": 54},
            {"variant": ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 49},
            {"variant": ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], "count": 49},
            {"variant": ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 47},
            {"variant": ["New online application received", "Make decision", "notify reject", "time out", "EVENT 13 END"], "count": 36},
            {"variant": ["New online application received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "time out", "EVENT 13 END"], "count": 36},
            {"variant": ["New online application received", "Check for completeness", "Perform checks", "Make decision", "notify reject", "review request received", "Check for completeness", "Request info", "info received", "Check for completeness", "Request info", "info received", "Check for completeness", "Perform checks", "Make decision", "Notify accept", "Deliver card", "EVENT 13 END"], "count": 36},
        ],
    },
}

TPROB = {(x["source"], x["target"]): x["probability"] for x in PROCESS["transition_probabilities_top"]}
DFG = {(x["source"], x["target"]): x["frequency"] for x in PROCESS["process_abstraction"]["dfg_edges"]}
DUR = {(x["source"], x["target"]): x for x in PROCESS["duration_profiles_top"]}
VARIANTS = [(v["variant"], v["count"]) for v in PROCESS["process_abstraction"]["frequent_variants"]]

def detect_precision(ts):
    if "." not in ts:
        return 0
    frac = ts.split(".")[-1]
    frac = "".join(ch for ch in frac if ch.isdigit())
    return len(frac)

def floor_to_precision(ts, precision):
    if precision <= 0:
        return ts.floor("S")
    unit = {1: "100ms", 2: "10ms", 3: "1ms", 4: "100us", 5: "10us", 6: "1us"}.get(precision, "1us")
    return ts.floor(unit)

def format_ts(ts, precision):
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if precision <= 0:
        return base
    frac = f"{ts.microsecond:06d}"[:precision]
    return f"{base}.{frac}"

def min_step_seconds(precision):
    return 1.0 if precision <= 0 else 10 ** (-precision)

def contiguous_subseq_score(seq):
    best = 0
    tup = tuple(seq)
    for var, count in VARIANTS:
        n = len(tup)
        for i in range(len(var) - n + 1):
            if tuple(var[i:i+n]) == tup:
                best += count
    return best

def edge_exists(a, b):
    return (a, b) in DFG

def edge_log_score(a, b):
    if (a, b) in DFG:
        return math.log(DFG[(a, b)] + 1.0) + math.log(TPROB.get((a, b), 1e-6) + 1e-12)
    return -1e9

def internal_valid(order):
    return all(edge_exists(order[i], order[i+1]) for i in range(len(order)-1))

def score_order(order, left_act, right_act, require_left=False, require_right=False):
    if not internal_valid(order):
        return None
    if require_left and left_act is not None and not edge_exists(left_act, order[0]):
        return None
    if require_right and right_act is not None and not edge_exists(order[-1], right_act):
        return None
    score = 0.0
    full = list(order)
    if left_act is not None:
        full = [left_act] + full
    if right_act is not None:
        full = full + [right_act]
    score += 1000000.0 * contiguous_subseq_score(full)
    score += 10000.0 * contiguous_subseq_score(order)
    for i in range(len(order)-1):
        score += edge_log_score(order[i], order[i+1])
    if left_act is not None and edge_exists(left_act, order[0]):
        score += 0.5 * edge_log_score(left_act, order[0])
    if right_act is not None and edge_exists(order[-1], right_act):
        score += 0.5 * edge_log_score(order[-1], right_act)
    return score

def best_order(activities, left_act, right_act):
    idxs = list(range(len(activities)))
    perms = list(itertools.permutations(idxs))
    levels = [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ]
    best = None
    best_meta = None
    for require_left, require_right in levels:
        local_best = None
        local_meta = None
        for p in perms:
            order = [activities[i] for i in p]
            sc = score_order(order, left_act, right_act, require_left=require_left, require_right=require_right)
            if sc is None:
                continue
            if local_best is None or sc > local_best:
                local_best = sc
                local_meta = p
        if local_meta is not None:
            best = local_best
            best_meta = local_meta
            break
    if best_meta is None:
        best_meta = tuple(range(len(activities)))
    return list(best_meta)

def gap_for_edge(a, b):
    prof = DUR.get((a, b))
    if prof is not None:
        if prof.get("share_under_60s", 0.0) >= 0.8:
            g = prof.get("p10_seconds", 10.0)
        else:
            g = prof.get("p25_seconds", 60.0)
        return max(g, 1.0)
    if (a, b) in DFG:
        return 60.0
    if a == b:
        return 1.0
    return 10.0

def build_times(observed_dt, ordered_acts, right_anchor_dt, precision):
    k = len(ordered_acts)
    if k == 1:
        return [floor_to_precision(observed_dt, precision)], False
    gaps = [gap_for_edge(ordered_acts[i], ordered_acts[i+1]) for i in range(k-1)]
    min_step = min_step_seconds(precision)
    total_min = (k - 1) * min_step
    clamped = False

    if right_anchor_dt is not None:
        room = (right_anchor_dt - observed_dt).total_seconds()
        if room < total_min:
            room = total_min
            clamped = True
        span = sum(gaps)
        if span > room:
            factor = room / span if span > 0 else 1.0
            gaps = [g * factor for g in gaps]
            clamped = True

    gaps = [max(g, min_step) for g in gaps]

    if right_anchor_dt is not None:
        room = (right_anchor_dt - observed_dt).total_seconds()
        if room < sum(gaps):
            excess = sum(gaps) - room
            reducible = sum(max(0.0, g - min_step) for g in gaps)
            if reducible > 0:
                newg = []
                for g in gaps:
                    extra = max(0.0, g - min_step)
                    reduce_by = excess * (extra / reducible)
                    newg.append(max(min_step, g - reduce_by))
                gaps = newg
                clamped = True
            if sum(gaps) > room + 1e-9:
                gaps = [min_step] * (k - 1)
                clamped = True

    times = [observed_dt]
    cur = observed_dt
    for g in gaps:
        cur = cur + pd.to_timedelta(g, unit="s")
        times.append(cur)

    times = [floor_to_precision(t, precision) for t in times]

    for i in range(1, len(times)):
        if not (times[i] > times[i-1]):
            times[i] = floor_to_precision(times[i-1] + pd.to_timedelta(min_step, unit="s"), precision)
            clamped = True

    if right_anchor_dt is not None and len(times) > 1 and times[-1] > right_anchor_dt:
        latest_start = right_anchor_dt - pd.to_timedelta((k - 1) * min_step, unit="s")
        if latest_start < observed_dt:
            latest_start = observed_dt
        times = [floor_to_precision(latest_start + pd.to_timedelta(i * min_step, unit="s"), precision) for i in range(k)]
        times[0] = floor_to_precision(observed_dt, precision)
        for i in range(1, k):
            if not (times[i] > times[i-1]):
                times[i] = floor_to_precision(times[i-1] + pd.to_timedelta(min_step, unit="s"), precision)
        if right_anchor_dt is not None and times[-1] > right_anchor_dt:
            end = floor_to_precision(right_anchor_dt, precision)
            start = end - pd.to_timedelta((k - 1) * min_step, unit="s")
            if start < observed_dt:
                start = observed_dt
            times = [floor_to_precision(start + pd.to_timedelta(i * min_step, unit="s"), precision) for i in range(k)]
        clamped = True

    return times, clamped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    for r in rows:
        r["_orig_ts"] = r["Timestamp"]
        r["_out_ts"] = r["Timestamp"]
        r["_dt"] = pd.to_datetime(r["Timestamp"])
        r["_prec"] = detect_precision(r["Timestamp"])

    case_to_idxs = defaultdict(list)
    for i, r in enumerate(rows):
        case_to_idxs[r["Case"]].append(i)

    repaired_blocks = 0
    clamped_blocks = 0

    for case, idxs in case_to_idxs.items():
        idxs_sorted = idxs
        clean_dts = [rows[i]["_dt"] for i in idxs_sorted]
        case_start = min(clean_dts)
        case_end = max(clean_dts)

        j = 0
        while j < len(idxs_sorted):
            gi = idxs_sorted[j]
            row = rows[gi]
            if row["error_category"] == "CLEAN":
                j += 1
                continue

            block = [gi]
            shared = row["Timestamp"]
            j2 = j + 1
            while j2 < len(idxs_sorted):
                gj = idxs_sorted[j2]
                rj = rows[gj]
                if rj["error_category"] != "CLEAN" and rj["Timestamp"] == shared:
                    block.append(gj)
                    j2 += 1
                else:
                    break

            obs_dt = rows[block[0]]["_dt"]
            precision = max(rows[x]["_prec"] for x in block)

            left_idx = idxs_sorted[j - 1] if j - 1 >= 0 else None
            while left_idx is not None and rows[left_idx]["error_category"] != "CLEAN":
                pos = idxs_sorted.index(left_idx)
                left_idx = idxs_sorted[pos - 1] if pos - 1 >= 0 else None

            right_idx = idxs_sorted[j2] if j2 < len(idxs_sorted) else None
            while right_idx is not None and rows[right_idx]["error_category"] != "CLEAN":
                pos = idxs_sorted.index(right_idx)
                right_idx = idxs_sorted[pos + 1] if pos + 1 < len(idxs_sorted) else None

            left_act = rows[left_idx]["Activity"] if left_idx is not None else None
            right_act = rows[right_idx]["Activity"] if right_idx is not None else None
            right_anchor_dt = rows[right_idx]["_dt"] if right_idx is not None else case_end

            activities = [rows[x]["Activity"] for x in block]
            perm = best_order(activities, left_act, right_act)
            ordered_block_idxs = [block[p] for p in perm]
            ordered_acts = [rows[x]["Activity"] for x in ordered_block_idxs]

            if right_anchor_dt is not None and right_anchor_dt <= obs_dt:
                right_anchor_dt = obs_dt + pd.to_timedelta((len(block) - 1) * min_step_seconds(precision), unit="s")
                if right_anchor_dt > case_end:
                    right_anchor_dt = case_end

            times, clamped = build_times(obs_dt, ordered_acts, right_anchor_dt, precision)

            if times[0] < case_start:
                shift = case_start - times[0]
                times = [t + shift for t in times]
                clamped = True
            if times[-1] > case_end:
                shift = times[-1] - case_end
                times = [t - shift for t in times]
                clamped = True

            min_step = min_step_seconds(precision)
            for ii in range(1, len(times)):
                if not (times[ii] > times[ii-1]):
                    times[ii] = floor_to_precision(times[ii-1] + pd.to_timedelta(min_step, unit="s"), precision)
                    clamped = True

            if right_idx is not None:
                ra = rows[right_idx]["_dt"]
                if times[-1] > ra:
                    end = ra
                    start = obs_dt
                    if (end - start).total_seconds() < (len(times) - 1) * min_step:
                        end = start + pd.to_timedelta((len(times) - 1) * min_step, unit="s")
                    times = [floor_to_precision(start + pd.to_timedelta(i * min_step, unit="s"), precision) for i in range(len(times))]
                    clamped = True

            for pos, bi in enumerate(ordered_block_idxs):
                rows[bi]["_out_ts"] = format_ts(times[pos], precision)

            repaired_blocks += 1
            if clamped:
                clamped_blocks += 1

            j = j2

    # self-check and repair pass
    rows_equal_input = 0

    for case, idxs in case_to_idxs.items():
        j = 0
        while j < len(idxs):
            gi = idxs[j]
            if rows[gi]["error_category"] == "CLEAN":
                if rows[gi]["_out_ts"] != rows[gi]["_orig_ts"]:
                    rows[gi]["_out_ts"] = rows[gi]["_orig_ts"]
                j += 1
                continue

            shared = rows[gi]["_orig_ts"]
            block = [gi]
            j2 = j + 1
            while j2 < len(idxs):
                gj = idxs[j2]
                if rows[gj]["error_category"] != "CLEAN" and rows[gj]["_orig_ts"] == shared:
                    block.append(gj)
                    j2 += 1
                else:
                    break

            obs_dt = rows[block[0]]["_dt"]
            precision = max(rows[x]["_prec"] for x in block)
            min_step = min_step_seconds(precision)

            out_dts = [pd.to_datetime(rows[x]["_out_ts"]) for x in block]
            equal_count = sum(rows[x]["_out_ts"] == rows[x]["_orig_ts"] for x in block)

            keepers = [x for x in block if rows[x]["_out_ts"] == rows[x]["_orig_ts"]]
            if len(keepers) != 1:
                acts = [rows[x]["Activity"] for x in block]
                left_idx = idxs[j - 1] if j - 1 >= 0 else None
                right_idx = idxs[j2] if j2 < len(idxs) else None
                left_act = rows[left_idx]["Activity"] if left_idx is not None and rows[left_idx]["error_category"] == "CLEAN" else None
                right_act = rows[right_idx]["Activity"] if right_idx is not None and rows[right_idx]["error_category"] == "CLEAN" else None
                perm = best_order(acts, left_act, right_act)
                ordered_block_idxs = [block[p] for p in perm]
                right_anchor_dt = pd.to_datetime(rows[right_idx]["_orig_ts"]) if right_idx is not None and rows[right_idx]["error_category"] == "CLEAN" else max(pd.to_datetime(rows[x]["_orig_ts"]) for x in idxs)
                times, _ = build_times(obs_dt, [rows[x]["Activity"] for x in ordered_block_idxs], right_anchor_dt, precision)
                for pos, bi in enumerate(ordered_block_idxs):
                    rows[bi]["_out_ts"] = format_ts(times[pos], precision)
                out_dts = [pd.to_datetime(rows[x]["_out_ts"]) for x in block]

            changed = False
            dt_pairs = [(x, pd.to_datetime(rows[x]["_out_ts"])) for x in block]
            keep_idx = None
            for x, dtv in dt_pairs:
                if rows[x]["_out_ts"] == rows[x]["_orig_ts"]:
                    keep_idx = x
                    break
            if keep_idx is None:
                keep_idx = block[0]
                rows[keep_idx]["_out_ts"] = rows[keep_idx]["_orig_ts"]
                changed = True

            others = [x for x in block if x != keep_idx]
            assigned = {keep_idx: pd.to_datetime(rows[keep_idx]["_out_ts"])}
            cur = assigned[keep_idx]
            for x in sorted(others, key=lambda z: pd.to_datetime(rows[z]["_out_ts"])):
                nxt = pd.to_datetime(rows[x]["_out_ts"])
                if not (nxt > cur):
                    nxt = floor_to_precision(cur + pd.to_timedelta(min_step, unit="s"), precision)
                    rows[x]["_out_ts"] = format_ts(nxt, precision)
                    changed = True
                assigned[x] = nxt
                cur = nxt

            right_idx = idxs[j2] if j2 < len(idxs) and rows[idxs[j2]]["error_category"] == "CLEAN" else None
            if right_idx is not None:
                right_anchor_dt = pd.to_datetime(rows[right_idx]["_orig_ts"])
                last_dt = max(pd.to_datetime(rows[x]["_out_ts"]) for x in block)
                if last_dt > right_anchor_dt:
                    ordered = sorted(block, key=lambda z: pd.to_datetime(rows[z]["_out_ts"]))
                    latest_start = right_anchor_dt - pd.to_timedelta((len(block) - 1) * min_step, unit="s")
                    if latest_start < obs_dt:
                        latest_start = obs_dt
                    seq = [floor_to_precision(latest_start + pd.to_timedelta(i * min_step, unit="s"), precision) for i in range(len(block))]
                    seq[0] = floor_to_precision(obs_dt, precision)
                    for i in range(1, len(seq)):
                        if not (seq[i] > seq[i-1]):
                            seq[i] = floor_to_precision(seq[i-1] + pd.to_timedelta(min_step, unit="s"), precision)
                    for pos, x in enumerate(ordered):
                        rows[x]["_out_ts"] = format_ts(seq[pos], precision)
                    changed = True

            if changed:
                clamped_blocks += 1

            j = j2

    for r in rows:
        if r["error_category"] != "CLEAN" and r["_out_ts"] == r["_orig_ts"]:
            rows_equal_input += 1
        try:
            pd.to_datetime(r["_out_ts"])
        except Exception:
            r["_out_ts"] = r["_orig_ts"]
            if r["error_category"] != "CLEAN" and r["_out_ts"] == r["_orig_ts"]:
                pass

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r[k] for k in fieldnames}
            out["Timestamp"] = r["_out_ts"]
            writer.writerow(out)

    print(f"blocks_repaired={repaired_blocks} blocks_clamped={clamped_blocks} rows_still_equal_to_input={rows_equal_input}")

if __name__ == "__main__":
    main()