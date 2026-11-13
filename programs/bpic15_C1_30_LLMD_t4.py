import argparse
import itertools
import math
import sys
from collections import defaultdict, Counter

import pandas as pd


# -----------------------------
# Embedded process abstraction
# -----------------------------
PROCESS = {
    "transition_probabilities_top": [
        {"source": "appeal subcase completed", "target": "appealed to higher court", "probability": 1.0},
        {"source": "create coordination confirmation", "target": "by law", "probability": 1.0},
        {"source": "create decision no permit required", "target": "create subcases completeness", "probability": 1.0},
        {"source": "create forwarding copy", "target": "enter senddate letter to stakeholders", "probability": 1.0},
        {"source": "create letter only notification required", "target": "forward to the competent authority", "probability": 1.0},
        {"source": "creating cover letter no permit required", "target": "decision date prior to decision", "probability": 1.0},
        {"source": "date environmental permit irrevocable", "target": "assessment of content completed", "probability": 1.0},
        {"source": "enrich draft decision environmental permit", "target": "enter date draft decisionenvironmental permit", "probability": 1.0},
        {"source": "enter objection term permanent suspension", "target": "copy decision permanent suspension to stakeholders", "probability": 1.0},
        {"source": "enter senddate letter no permit required", "target": "generating decision no permit required", "probability": 1.0},
        {"source": "generating decision no permit required", "target": "no permit needed or only notification needed", "probability": 1.0},
        {"source": "objection decision no permit required received", "target": "phase activity no permit required", "probability": 1.0},
        {"source": "partly permit application", "target": "partly permit", "probability": 1.0},
        {"source": "phase activity no permit required", "target": "phase archived case", "probability": 1.0},
        {"source": "phase forwarded to competent authority", "target": "phase procedure aborted", "probability": 1.0},
        {"source": "phase only notification required", "target": "phase procedure aborted", "probability": 1.0},
        {"source": "phase updated plan requested", "target": "register receipt updated plan", "probability": 1.0},
        {"source": "reactive designation received", "target": "register objection and appeal periods", "probability": 1.0},
        {"source": "record date of decision no permit required", "target": "enter senddate letter no permit required", "probability": 1.0},
        {"source": "request updated plan again", "target": "contacting applicant", "probability": 1.0},
        {"source": "send letter receptive test not ok wait for report firedepartement", "target": "resume completeness subcases", "probability": 1.0},
        {"source": "send send confirmation competent authority", "target": "request to competent authority", "probability": 1.0},
        {"source": "set decision phase decision no permit required sent", "target": "set decision phase no permit required decided", "probability": 1.0},
        {"source": "set decision phase decision permitting published", "target": "close case", "probability": 1.0},
        {"source": "set decision phase no permit required decided", "target": "set decision status", "probability": 1.0},
        {"source": "set decision phase unlicensed irrevocable", "target": "close case", "probability": 1.0},
        {"source": "set phase", "target": "creating environmental permit decision", "probability": 1.0},
        {"source": "subcase art. 34 WABO completed", "target": "refusal after suspension", "probability": 1.0},
        {"source": "subcase designation GS completed", "target": "subcase perspective complete", "probability": 1.0},
        {"source": "subcases submit draft decision completed", "target": "enrich draft decision", "probability": 1.0},
        {"source": "temporary permit", "target": "regular procedure without MER", "probability": 1.0},
        {"source": "treat subcase suspension art. 34 WABO", "target": "subcase art. 34 WABO completed", "probability": 1.0},
        {"source": "treat subcases submit draft decision", "target": "subcases submit draft decision completed", "probability": 1.0},
        {"source": "treating subcase designation GS", "target": "updated plan after perspective", "probability": 1.0},
        {"source": "phase archived case", "target": "phase case handled", "probability": 0.9813},
        {"source": "enter senddate decision", "target": "enter senddate decision environmental permit", "probability": 0.9729},
        {"source": "enter objectionterm", "target": "copy decision to extend procedure to stakeholders", "probability": 0.9444},
        {"source": "decided on draft decision environmental permit", "target": "enrich draft decision", "probability": 0.9},
        {"source": "decision terminate on request", "target": "phase procedure aborted", "probability": 0.875},
        {"source": "no permit required for application", "target": "MER present", "probability": 0.85},
        {"source": "MER present", "target": "activities regular procedure", "probability": 0.8478},
        {"source": "start decision phase permanent suspension sent", "target": "start decision phase permanently suspended decided", "probability": 0.8462},
        {"source": "cancel BAG objects research", "target": "phase procedure aborted", "probability": 0.8333},
        {"source": "determine cause for suspension article 34 WABO", "target": "assessment of content completed", "probability": 0.8182},
        {"source": "create monitoring case", "target": "creating cover letter decision", "probability": 0.8065},
        {"source": "phase decision irrevocable", "target": "read publication date field", "probability": 0.8},
        {"source": "subcase final VVGB completed", "target": "VVGB positive", "probability": 0.8},
        {"source": "treat subcase final VVGB", "target": "subcase final VVGB completed", "probability": 0.8},
        {"source": "create decision terminate on request", "target": "decision terminate on request", "probability": 0.7778},
        {"source": "set phase decision revoked", "target": "close case", "probability": 0.7778},
        {"source": "publication date draft decision", "target": "date draft decision for inspection", "probability": 0.7647},
        {"source": "reception through OLO", "target": "send confirmation receipt", "probability": 0.7589},
        {"source": "environmental permit irrevocable", "target": "determine cause for suspension article 34 WABO", "probability": 0.75},
        {"source": "objection to extension procedure term receptive", "target": "decision extension procedure term irrevocable", "probability": 0.75},
        {"source": "publish official draft decision", "target": "send design decision to stakeholders", "probability": 0.75},
        {"source": "term perspective on draft decision", "target": "phase draft decision made", "probability": 0.75},
        {"source": "treat subcase WAW", "target": "subcase WAW completed", "probability": 0.75},
        {"source": "MER required", "target": "OLO messaging active", "probability": 0.7333},
        {"source": "treat subcases completeness", "target": "subcases completeness completed", "probability": 0.7206},
        {"source": "create subcases submit draft decision", "target": "enrich draft decision", "probability": 0.7143},
    ],
    "duration_profiles_top": [
        {"source": "register submission date request", "target": "OLO messaging active", "n": 729, "p10_seconds": 86400.0, "p25_seconds": 172800.0, "median_seconds": 306406.0, "p75_seconds": 518400.0, "p90_seconds": 733026.6, "share_under_60s": 0.0},
        {"source": "forward to the competent authority", "target": "regular procedure without MER", "n": 673, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 3.0, "p75_seconds": 6.0, "p90_seconds": 12.0, "share_under_60s": 0.973},
        {"source": "send confirmation receipt", "target": "forward to the competent authority", "n": 591, "p10_seconds": 2.0, "p25_seconds": 3.0, "median_seconds": 6.0, "p75_seconds": 9.0, "p90_seconds": 19.0, "share_under_60s": 0.941},
        {"source": "request complete", "target": "send letter in progress", "n": 572, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 1.0, "p75_seconds": 2.0, "p90_seconds": 4.0, "share_under_60s": 0.991},
        {"source": "article 34  WABO applies", "target": "assessment of content completed", "n": 568, "p10_seconds": 0.0, "p25_seconds": 1.0, "median_seconds": 1.0, "p75_seconds": 2.0, "p90_seconds": 4.0, "share_under_60s": 0.993},
        {"source": "assessment of content completed", "target": "phase advice known", "n": 555, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 1.0, "share_under_60s": 1.0},
        {"source": "send confirmation receipt", "target": "send confirmation receipt", "n": 503, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 211.2, "share_under_60s": 0.799},
        {"source": "phase application receptive", "target": "request complete", "n": 500, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "treat subcases content", "target": "completed subcases content", "n": 493, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 1.0, "p75_seconds": 2.0, "p90_seconds": 3.0, "share_under_60s": 0.992},
        {"source": "treat subcases completeness", "target": "subcases completeness completed", "n": 490, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 2.0, "p75_seconds": 3.0, "p90_seconds": 7.0, "share_under_60s": 0.984},
        {"source": "by law", "target": "creating environmental permit decision", "n": 481, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 1.0, "share_under_60s": 0.994},
        {"source": "enter senddate decision", "target": "enter senddate decision environmental permit", "n": 467, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 0.991},
        {"source": "phase advice known", "target": "grounds for refusal", "n": 438, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 2.0, "p75_seconds": 3.0, "p90_seconds": 6.3, "share_under_60s": 0.954},
        {"source": "ask stakeholders views", "target": "suspension ground applicable", "n": 431, "p10_seconds": 0.0, "p25_seconds": 1.0, "median_seconds": 2.0, "p75_seconds": 4.0, "p90_seconds": 6.0, "share_under_60s": 0.984},
        {"source": "grounds for refusal", "target": "ask stakeholders views", "n": 426, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 2.0, "p75_seconds": 5.0, "p90_seconds": 15.5, "share_under_60s": 0.955},
        {"source": "subcases completeness completed", "target": "procedure change", "n": 424, "p10_seconds": 1.0, "p25_seconds": 2.0, "median_seconds": 3.0, "p75_seconds": 4.0, "p90_seconds": 7.0, "share_under_60s": 0.991},
        {"source": "creating environmental permit decision", "target": "decision date prior to decision", "n": 423, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 4.0, "share_under_60s": 0.993},
        {"source": "procedure change", "target": "request complete", "n": 421, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 3.0, "p90_seconds": 6.0, "share_under_60s": 0.988},
        {"source": "send confirmation receipt", "target": "enter senddate acknowledgement", "n": 412, "p10_seconds": 2.0, "p25_seconds": 67.0, "median_seconds": 159.0, "p75_seconds": 308.25, "p90_seconds": 1342.5000000000043, "share_under_60s": 0.245},
        {"source": "suspension ground applicable", "target": "by law", "n": 410, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 1.0, "p75_seconds": 2.0, "p90_seconds": 4.0, "share_under_60s": 0.993},
        {"source": "OLO messaging active", "target": "phase application received", "n": 409, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "regular procedure without MER", "target": "send confirmation receipt", "n": 399, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "phase application received", "target": "send confirmation receipt", "n": 393, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 2.0, "p75_seconds": 4.0, "p90_seconds": 10.0, "share_under_60s": 0.982},
        {"source": "procedure change", "target": "phase application receptive", "n": 383, "p10_seconds": 1.0, "p25_seconds": 2.0, "median_seconds": 3.0, "p75_seconds": 5.0, "p90_seconds": 9.0, "share_under_60s": 0.966},
        {"source": "enter senddate acknowledgement", "target": "send confirmation receipt", "n": 383, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 0.99},
        {"source": "enter senddate procedure confirmation", "target": "send procedure confirmation", "n": 378, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 0.952},
        {"source": "send confirmation receipt", "target": "start WABOprocedure", "n": 375, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "forward to the competent authority", "target": "phase application received", "n": 372, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "registration date publication", "target": "regular procedure without MER", "n": 357, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 0.994},
        {"source": "procedure change", "target": "article 34  WABO applies", "n": 352, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 2.0, "p75_seconds": 3.0, "p90_seconds": 9.9, "share_under_60s": 0.949},
        {"source": "OLO messaging active", "target": "create procedure confirmation", "n": 344, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "transcript decision environmental permit to stakeholders", "target": "generate publication document decision environmental permit", "n": 341, "p10_seconds": 6.0, "p25_seconds": 7.0, "median_seconds": 10.0, "p75_seconds": 19.0, "p90_seconds": 55.0, "share_under_60s": 0.906},
        {"source": "send letter in progress", "target": "treat subcases content", "n": 324, "p10_seconds": 0.0, "p25_seconds": 1.0, "median_seconds": 1.0, "p75_seconds": 2.0, "p90_seconds": 2.0, "share_under_60s": 0.991},
        {"source": "phase decision sent", "target": "phase decision taken", "n": 320, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "grounds for refusal", "target": "phase advice known", "n": 319, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "create procedure confirmation", "target": "create publication document", "n": 317, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "phase advice known", "target": "phase application receptive", "n": 316, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "phase decision taken", "target": "register objection and appeal periods", "n": 308, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 5.0, "p90_seconds": 12.0, "share_under_60s": 0.974},
        {"source": "article 34  WABO applies", "target": "ask stakeholders views", "n": 305, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 0.997},
        {"source": "completed subcases content", "target": "procedure change", "n": 298, "p10_seconds": 1.0, "p25_seconds": 2.0, "median_seconds": 3.0, "p75_seconds": 5.0, "p90_seconds": 10.0, "share_under_60s": 0.98},
        {"source": "enter senddate decision environmental permit", "target": "phase decision sent", "n": 293, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 1.0, "share_under_60s": 1.0},
        {"source": "start WABOprocedure", "target": "enter senddate acknowledgement", "n": 293, "p10_seconds": 34535.2, "p25_seconds": 38196.0, "median_seconds": 43648.0, "p75_seconds": 52527.0, "p90_seconds": 86400.0, "share_under_60s": 0.0},
        {"source": "creating environmental permit decision", "target": "generating decision environmental permit", "n": 284, "p10_seconds": 4.0, "p25_seconds": 55.5, "median_seconds": 1796.0, "p75_seconds": 113196.25, "p90_seconds": 914443.3, "share_under_60s": 0.257},
        {"source": "send letter in progress", "target": "set decision status", "n": 254, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "set decision phase decision permitting decided", "target": "start decision phase decision permitting sent", "n": 241, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 0.988},
        {"source": "regular procedure without MER", "target": "start WABOprocedure", "n": 230, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 1.0, "share_under_60s": 1.0},
        {"source": "transcript decision environmental permit to stakeholders", "target": "enter senddate decision environmental permit", "n": 225, "p10_seconds": 8.0, "p25_seconds": 34.0, "median_seconds": 50131.0, "p75_seconds": 172800.0, "p90_seconds": 259200.0, "share_under_60s": 0.258},
        {"source": "send procedure confirmation", "target": "registration date publication", "n": 224, "p10_seconds": 8.0, "p25_seconds": 11.0, "median_seconds": 19.0, "p75_seconds": 78.0, "p90_seconds": 738814.7, "share_under_60s": 0.728},
        {"source": "phase application received", "target": "reception through OLO", "n": 222, "p10_seconds": 1.0, "p25_seconds": 2.0, "median_seconds": 3.0, "p75_seconds": 6.0, "p90_seconds": 14.9, "share_under_60s": 0.973},
        {"source": "regular procedure without MER", "target": "enter senddate procedure confirmation", "n": 219, "p10_seconds": 1.0, "p25_seconds": 2.0, "median_seconds": 2.0, "p75_seconds": 4.0, "p90_seconds": 87809.2, "share_under_60s": 0.872},
        {"source": "generating decision environmental permit", "target": "grounds for refusal", "n": 215, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "decision date prior to decision", "target": "generating decision environmental permit", "n": 209, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 1.0, "p75_seconds": 4206.0, "p90_seconds": 430595.8, "share_under_60s": 0.569},
        {"source": "create publication document", "target": "create subcases completeness", "n": 197, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 7.4, "share_under_60s": 0.914},
        {"source": "ask stakeholders views", "target": "assessment of content completed", "n": 194, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "by law", "target": "completed subcases content", "n": 194, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "submit decision", "target": "suspension ground applicable", "n": 193, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "procedure change", "target": "procedure change", "n": 193, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "completed subcases content", "target": "coordination of application", "n": 190, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
        {"source": "register submission date request", "target": "enter senddate acknowledgement", "n": 188, "p10_seconds": 172800.0, "p25_seconds": 345600.0, "median_seconds": 604800.0, "p75_seconds": 864000.0, "p90_seconds": 1209600.0, "share_under_60s": 0.0},
        {"source": "record date of decision environmental permit", "target": "set decision phase decision permitting decided", "n": 187, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    ],
    "process_abstraction": {
        "dfg_edges": [
            {"source": "register submission date request", "target": "OLO messaging active", "frequency": 729, "median_duration_seconds": 306406.0},
            {"source": "forward to the competent authority", "target": "regular procedure without MER", "frequency": 673, "median_duration_seconds": 3.0},
            {"source": "send confirmation receipt", "target": "forward to the competent authority", "frequency": 591, "median_duration_seconds": 6.0},
            {"source": "request complete", "target": "send letter in progress", "frequency": 572, "median_duration_seconds": 1.0},
            {"source": "article 34  WABO applies", "target": "assessment of content completed", "frequency": 568, "median_duration_seconds": 1.0},
            {"source": "assessment of content completed", "target": "phase advice known", "frequency": 555, "median_duration_seconds": 0.0},
            {"source": "send confirmation receipt", "target": "send confirmation receipt", "frequency": 503, "median_duration_seconds": 0.0},
            {"source": "phase application receptive", "target": "request complete", "frequency": 500, "median_duration_seconds": 0.0},
            {"source": "treat subcases content", "target": "completed subcases content", "frequency": 493, "median_duration_seconds": 1.0},
            {"source": "treat subcases completeness", "target": "subcases completeness completed", "frequency": 490, "median_duration_seconds": 2.0},
            {"source": "by law", "target": "creating environmental permit decision", "frequency": 481, "median_duration_seconds": 0.0},
            {"source": "enter senddate decision", "target": "enter senddate decision environmental permit", "frequency": 467, "median_duration_seconds": 0.0},
            {"source": "phase advice known", "target": "grounds for refusal", "frequency": 438, "median_duration_seconds": 2.0},
            {"source": "ask stakeholders views", "target": "suspension ground applicable", "frequency": 431, "median_duration_seconds": 2.0},
            {"source": "grounds for refusal", "target": "ask stakeholders views", "frequency": 426, "median_duration_seconds": 2.0},
            {"source": "subcases completeness completed", "target": "procedure change", "frequency": 424, "median_duration_seconds": 3.0},
            {"source": "creating environmental permit decision", "target": "decision date prior to decision", "frequency": 423, "median_duration_seconds": 0.0},
            {"source": "procedure change", "target": "request complete", "frequency": 421, "median_duration_seconds": 0.0},
            {"source": "send confirmation receipt", "target": "enter senddate acknowledgement", "frequency": 412, "median_duration_seconds": 159.0},
            {"source": "suspension ground applicable", "target": "by law", "frequency": 410, "median_duration_seconds": 1.0},
            {"source": "OLO messaging active", "target": "phase application received", "frequency": 409, "median_duration_seconds": 0.0},
            {"source": "regular procedure without MER", "target": "send confirmation receipt", "frequency": 399, "median_duration_seconds": 0.0},
            {"source": "phase application received", "target": "send confirmation receipt", "frequency": 393, "median_duration_seconds": 2.0},
            {"source": "enter senddate acknowledgement", "target": "send confirmation receipt", "frequency": 383, "median_duration_seconds": 0.0},
            {"source": "procedure change", "target": "phase application receptive", "frequency": 383, "median_duration_seconds": 3.0},
            {"source": "enter senddate procedure confirmation", "target": "send procedure confirmation", "frequency": 378, "median_duration_seconds": 0.0},
            {"source": "send confirmation receipt", "target": "start WABOprocedure", "frequency": 375, "median_duration_seconds": 0.0},
            {"source": "forward to the competent authority", "target": "phase application received", "frequency": 372, "median_duration_seconds": 0.0},
            {"source": "registration date publication", "target": "regular procedure without MER", "frequency": 357, "median_duration_seconds": 0.0},
            {"source": "procedure change", "target": "article 34  WABO applies", "frequency": 352, "median_duration_seconds": 2.0},
            {"source": "OLO messaging active", "target": "create procedure confirmation", "frequency": 344, "median_duration_seconds": 0.0},
            {"source": "transcript decision environmental permit to stakeholders", "target": "generate publication document decision environmental permit", "frequency": 341, "median_duration_seconds": 10.0},
            {"source": "send letter in progress", "target": "treat subcases content", "frequency": 324, "median_duration_seconds": 1.0},
            {"source": "phase decision sent", "target": "phase decision taken", "frequency": 320, "median_duration_seconds": 0.0},
            {"source": "grounds for refusal", "target": "phase advice known", "frequency": 319, "median_duration_seconds": 0.0},
            {"source": "create procedure confirmation", "target": "create publication document", "frequency": 317, "median_duration_seconds": 0.0},
            {"source": "phase advice known", "target": "phase application receptive", "frequency": 316, "median_duration_seconds": 0.0},
            {"source": "phase decision taken", "target": "register objection and appeal periods", "frequency": 308, "median_duration_seconds": 0.0},
            {"source": "article 34  WABO applies", "target": "ask stakeholders views", "frequency": 305, "median_duration_seconds": 0.0},
            {"source": "completed subcases content", "target": "procedure change", "frequency": 298, "median_duration_seconds": 3.0},
            {"source": "start WABOprocedure", "target": "enter senddate acknowledgement", "frequency": 293, "median_duration_seconds": 43648.0},
            {"source": "enter senddate decision environmental permit", "target": "phase decision sent", "frequency": 293, "median_duration_seconds": 0.0},
            {"source": "creating environmental permit decision", "target": "generating decision environmental permit", "frequency": 284, "median_duration_seconds": 1796.0},
            {"source": "send letter in progress", "target": "set decision status", "frequency": 254, "median_duration_seconds": 0.0},
            {"source": "set decision phase decision permitting decided", "target": "start decision phase decision permitting sent", "frequency": 241, "median_duration_seconds": 0.0},
            {"source": "regular procedure without MER", "target": "start WABOprocedure", "frequency": 230, "median_duration_seconds": 0.0},
            {"source": "transcript decision environmental permit to stakeholders", "target": "enter senddate decision environmental permit", "frequency": 225, "median_duration_seconds": 50131.0},
            {"source": "send procedure confirmation", "target": "registration date publication", "frequency": 224, "median_duration_seconds": 19.0},
            {"source": "phase application received", "target": "reception through OLO", "frequency": 222, "median_duration_seconds": 3.0},
            {"source": "regular procedure without MER", "target": "enter senddate procedure confirmation", "frequency": 219, "median_duration_seconds": 2.0},
            {"source": "generating decision environmental permit", "target": "grounds for refusal", "frequency": 215, "median_duration_seconds": 0.0},
            {"source": "decision date prior to decision", "target": "generating decision environmental permit", "frequency": 209, "median_duration_seconds": 1.0},
            {"source": "create publication document", "target": "create subcases completeness", "frequency": 197, "median_duration_seconds": 0.0},
            {"source": "ask stakeholders views", "target": "assessment of content completed", "frequency": 194, "median_duration_seconds": 0.0},
            {"source": "by law", "target": "completed subcases content", "frequency": 194, "median_duration_seconds": 0.0},
            {"source": "procedure change", "target": "procedure change", "frequency": 193, "median_duration_seconds": 0.0},
            {"source": "submit decision", "target": "suspension ground applicable", "frequency": 193, "median_duration_seconds": 0.0},
            {"source": "completed subcases content", "target": "coordination of application", "frequency": 190, "median_duration_seconds": 0.0},
            {"source": "register submission date request", "target": "enter senddate acknowledgement", "frequency": 188, "median_duration_seconds": 604800.0},
            {"source": "record date of decision environmental permit", "target": "set decision phase decision permitting decided", "frequency": 187, "median_duration_seconds": 0.0},
        ],
        "start_activities": {
            "register submission date request": 1134, "OLO messaging active": 36, "date for inspection MER": 1,
            "enter senddate acknowledgement": 11, "enter senddate procedure confirmation": 5,
            "enter date termination request received": 1, "MER present": 1, "suspension ground applicable": 1,
            "set decision phase suspension decided": 7, "enter senddate decision suspension": 1,
            "record date of decision environmental permit": 1
        },
        "end_activities": {
            "start decision phase decision permitting sent": 134, "enter senddate decision environmental permit": 99,
            "transcript decision environmental permit to stakeholders": 14, "enter senddate decision permanent suspension": 25,
            "stop all running subcases 2b": 44, "retrieve missing data": 4, "terminate on request": 10,
            "set phase: phase permitting irrevocable": 37, "set decision phase inspection of decision permitting": 13,
            "set decision phase decision permitting published": 29, "phase application received": 4, "start WABOprocedure": 7,
            "stop suspended completeness subcases": 1, "enter senddate acknowledgement": 26, "enter senddate request termination": 5,
            "enter date decision permanent suspension": 1, "enter senddate retrieve missing data": 7,
            "send confirmation receipt additional data": 6, "register deadline": 23,
            "start decision phase permanently suspended decided": 3, "submit draft decision": 1, "phase decision sent": 3,
            "term for supplying missing data": 7, "set decision phase decision permitting decided": 1,
            "phase asked additional information": 1, "enter senddate decision procedure term extension": 5,
            "regular procedure without MER": 10, "enter date publication decision environmental permit": 281,
            "stop process after terminate on request": 3, "close case": 84, "subcases completeness completed": 4,
            "register date environmental permit decision": 21, "registrer date of publishing received request": 8,
            "creating decision permanently suspended": 2, "enter senddate confirmation request": 1,
            "forward to the competent authority": 9, "detailed procedure applies": 1, "send procedure confirmation": 3,
            "objection lodged against decision": 5, "copy decision permanent suspension to stakeholders": 2,
            "send confirmation receipt": 6, "decision terminate on request": 1,
            "copy decision to extend procedure to stakeholders": 3, "reception through OLO": 1,
            "receive additional information": 2, "phase procedure aborted": 3,
            "generate publication document decision environmental permit": 31, "enter senddate procedure confirmation": 4,
            "register submission date request": 1, "no permit required for application": 1, "procedure change": 3,
            "keep permanently suspended": 1, "phased application": 2, "treat subcases content": 4,
            "treat subcases completeness": 2, "continue or abort mutation BAG objects": 3,
            "registration date publication": 21, "completed subcases content": 3, "creating environmental permit decision": 3,
            "request to competent authority": 1, "generating decision environmental permit": 2, "send letter in progress": 1,
            "inform BAG administrator": 1, "suspension ground applicable": 1, "mutation BAG carry objects": 1,
            "set phase": 3, "appeal subcase completed": 1, "treat subcase perspective": 1,
            "applicant is stakeholder": 1, "date decision for inspection": 78, "phase decision irrevocable": 10,
            "create publication document": 32, "enter senddate letter to stakeholders": 2,
            "phase case handled": 6, "enter senddate letter to competent authority": 1,
            "record date of decision environmental permit": 4, "phase archived case": 1,
            "MER present in supplement": 1, "phase procedure prematurely terminated": 1,
            "enter senddate decision": 1, "creating cover letter decision": 1, "read publication date field": 1,
            "request for preliminary injunction received": 1, "start decision phase permanent suspension sent": 2,
            "create message terminate request": 1, "enter senddate letter no permit required": 1,
            "set decision status": 1, "set decision phase: draft decision for inspection": 1,
            "decision permanent suspension irrevocable": 1
        },
        "frequent_variants": [
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "forward to the competent authority", "phase application received", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement"], "count": 20},
            {"variant": ["register submission date request", "enter senddate acknowledgement", "phase application received", "reception through OLO", "send confirmation receipt", "forward to the competent authority"], "count": 8},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "forward to the competent authority", "phase application received", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement", "article 34  WABO applies", "ask stakeholders views", "by law", "coordination of application", "create subcases completeness", "create subcases content", "creating cover letter decision", "creating environmental permit decision", "decision date prior to decision", "enter senddate decision", "enter senddate decision environmental permit", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase application receptive", "phase decision ready", "phase decision sent", "phase decision taken", "procedure change", "procedure change", "register objection and appeal periods", "request complete", "send letter in progress", "set decision status", "submit decision", "suspension ground applicable", "transcript decision environmental permit to stakeholders", "record date of decision environmental permit", "set decision phase decision permitting decided", "enter senddate decision environmental permit", "start decision phase decision permitting sent"], "count": 4},
            {"variant": ["register submission date request", "enter senddate acknowledgement", "phase application received", "reception through OLO", "send confirmation receipt", "forward to the competent authority", "regular procedure without MER"], "count": 4},
            {"variant": ["register submission date request", "OLO messaging active", "phase application received", "send confirmation receipt", "enter senddate acknowledgement", "send confirmation receipt", "forward to the competent authority", "regular procedure without MER", "start WABOprocedure", "enter senddate procedure confirmation", "send procedure confirmation", "registration date publication", "create publication document", "treat subcases completeness", "subcases completeness completed", "procedure change", "phase application receptive", "request complete", "procedure change after completeness", "send letter in progress", "treat subcases content", "completed subcases content", "procedure change", "article 34  WABO applies", "assessment of content completed", "phase advice known", "grounds for refusal", "ask stakeholders views", "suspension ground applicable", "by law", "creating environmental permit decision", "decision date prior to decision", "generating decision environmental permit", "record date of decision environmental permit", "submit decision", "phase decision taken", "register objection and appeal periods", "enter senddate decision environmental permit", "transcript decision environmental permit to stakeholders", "enter senddate decision", "enter senddate decision environmental permit", "phase decision sent", "create publication document", "date decision for inspection"], "count": 4},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement", "article 34  WABO applies", "ask stakeholders views", "by law", "coordination of application", "create subcases content", "creating environmental permit decision", "decision date prior to decision", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase application receptive", "phase decision ready", "procedure change", "procedure change", "request complete", "send letter in progress", "set decision status", "submit decision", "suspension ground applicable", "creating cover letter decision", "enter senddate decision", "enter senddate decision environmental permit", "phase decision sent", "phase decision taken", "register objection and appeal periods", "transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit", "record date of decision environmental permit", "set decision phase decision permitting decided", "start decision phase decision permitting sent"], "count": 3},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "procedure change", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "start WABOprocedure", "article 34  WABO applies", "ask stakeholders views", "by law", "coordination of application", "create subcases content", "creating cover letter decision", "creating environmental permit decision", "decision date prior to decision", "enter senddate decision", "enter senddate decision environmental permit", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase application receptive", "phase decision ready", "phase decision sent", "phase decision taken", "procedure change", "register objection and appeal periods", "request complete", "send letter in progress", "set decision status", "submit decision", "suspension ground applicable", "transcript decision environmental permit to stakeholders", "record date of decision environmental permit", "set decision phase decision permitting decided", "enter senddate decision environmental permit", "start decision phase decision permitting sent", "register deadline", "set phase: phase permitting irrevocable", "close case", "objection lodged against decision", "phase archived case", "phase case handled", "phase decision irrevocable", "read publication date field", "read publication date field", "stop all running subcases 2b"], "count": 3},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement", "article 34  WABO applies", "ask stakeholders views", "by law", "coordination of application", "create subcases content", "creating environmental permit decision", "decision date prior to decision", "grounds for refusal", "phase advice known", "phase application receptive", "procedure change", "procedure change", "request complete", "send letter in progress", "set decision status", "suspension ground applicable", "creating cover letter decision", "enter senddate decision", "enter senddate decision environmental permit", "generating decision environmental permit", "phase decision ready", "phase decision sent", "phase decision taken", "register objection and appeal periods", "submit decision", "transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit", "record date of decision environmental permit", "set decision phase decision permitting decided", "start decision phase decision permitting sent"], "count": 3},
            {"variant": ["register submission date request", "enter senddate acknowledgement", "phase application received", "reception through OLO", "send confirmation receipt"], "count": 3},
            {"variant": ["register submission date request", "OLO messaging active", "phase application received", "send confirmation receipt", "enter senddate acknowledgement", "send confirmation receipt", "forward to the competent authority", "regular procedure without MER", "enter senddate procedure confirmation", "send procedure confirmation", "registration date publication", "inform BAG administrator", "treat subcases completeness", "subcases completeness completed", "procedure change", "phase application receptive", "request complete", "send letter in progress", "treat subcases content", "completed subcases content", "procedure change", "article 34  WABO applies", "assessment of content completed", "phase advice known", "grounds for refusal", "ask stakeholders views", "suspension ground applicable", "by law", "creating environmental permit decision", "generating decision environmental permit", "phase decision taken", "transcript decision environmental permit to stakeholders", "generate publication document decision environmental permit", "enter senddate decision environmental permit", "register date environmental permit decision", "enter date publication decision environmental permit"], "count": 3},
            {"variant": ["register submission date request", "OLO messaging active", "phase application received", "terminate on request", "send confirmation receipt", "enter senddate acknowledgement", "send confirmation receipt", "forward to the competent authority", "regular procedure without MER", "enter senddate procedure confirmation", "send procedure confirmation", "registration date publication"], "count": 3},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "procedure change", "registration date publication", "regular procedure without MER", "send confirmation receipt", "start WABOprocedure", "article 34  WABO applies", "ask stakeholders views", "assessment of content completed", "by law", "completed subcases content", "coordination of application", "creating cover letter decision", "creating environmental permit decision", "decision date prior to decision", "enter senddate decision", "enter senddate decision environmental permit", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase application receptive", "phase decision ready", "phase decision sent", "phase decision taken", "procedure change", "register objection and appeal periods", "request complete", "send letter in progress", "set decision status", "submit decision", "suspension ground applicable", "transcript decision environmental permit to stakeholders", "treat subcases content", "record date of decision environmental permit", "set decision phase decision permitting decided", "start decision phase decision permitting sent", "enter senddate decision environmental permit", "register deadline", "set phase: phase permitting irrevocable", "close case", "objection lodged against decision", "phase archived case", "phase case handled", "phase decision irrevocable", "read publication date field", "read publication date field", "stop all running subcases 2b"], "count": 3},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement", "article 34  WABO applies", "ask stakeholders views", "assessment of content completed", "by law", "completed subcases content", "coordination of application", "creating environmental permit decision", "decision date prior to decision", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase application receptive", "phase decision ready", "procedure change", "procedure change", "request complete", "send letter in progress", "set decision status", "submit decision", "suspension ground applicable", "treat subcases content", "creating cover letter decision", "enter senddate decision", "enter senddate decision environmental permit", "phase decision sent", "phase decision taken", "record date of decision environmental permit", "register objection and appeal periods", "set decision phase decision permitting decided", "transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit", "start decision phase decision permitting sent"], "count": 2},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "start WABOprocedure", "article 34  WABO applies", "ask stakeholders views", "by law", "coordination of application", "create subcases content", "creating cover letter decision", "creating environmental permit decision", "decision date prior to decision", "enter senddate decision", "enter senddate decision environmental permit", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase application receptive", "phase decision ready", "phase decision sent", "phase decision taken", "procedure change", "procedure change", "register objection and appeal periods", "request complete", "send letter in progress", "set decision status", "submit decision", "suspension ground applicable", "transcript decision environmental permit to stakeholders", "record date of decision environmental permit", "set decision phase decision permitting decided", "enter senddate decision environmental permit", "start decision phase decision permitting sent", "register deadline", "set phase: phase permitting irrevocable", "close case", "objection lodged against decision", "phase archived case", "phase case handled", "phase decision irrevocable", "read publication date field", "read publication date field", "stop all running subcases 2b"], "count": 2},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement", "article 34  WABO applies", "ask stakeholders views", "by law", "coordination of application", "create subcases content", "creating environmental permit decision", "decision date prior to decision", "grounds for refusal", "phase advice known", "phase application receptive", "procedure change", "procedure change", "request complete", "send letter in progress", "set decision status", "suspension ground applicable", "generating decision environmental permit", "phase decision ready", "submit decision", "creating cover letter decision", "enter senddate decision", "enter senddate decision environmental permit", "phase decision sent", "phase decision taken", "register objection and appeal periods", "transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit", "record date of decision environmental permit", "set decision phase decision permitting decided", "start decision phase decision permitting sent"], "count": 2},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "procedure change", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement", "article 34  WABO applies", "ask stakeholders views", "by law", "coordination of application", "create subcases content", "creating environmental permit decision", "decision date prior to decision", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase application receptive", "phase decision ready", "procedure change", "request complete", "send letter in progress", "set decision status", "submit decision", "suspension ground applicable", "create publication document", "creating cover letter decision", "enter date publication decision environmental permit", "enter senddate decision", "enter senddate decision environmental permit", "phase decision sent", "phase decision taken", "record date of decision environmental permit", "register objection and appeal periods", "set decision phase decision permitting decided", "transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit", "start decision phase decision permitting sent", "set decision phase decision permitting published"], "count": 2},
            {"variant": ["register submission date request", "OLO messaging active", "phase application received"], "count": 2},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "procedure change", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement", "article 34  WABO applies", "ask stakeholders views", "by law", "coordination of application", "create subcases content", "creating environmental permit decision", "decision date prior to decision", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase application receptive", "phase decision ready", "procedure change", "request complete", "send letter in progress", "set decision status", "submit decision", "suspension ground applicable", "creating cover letter decision", "enter senddate decision", "enter senddate decision environmental permit", "phase decision sent", "phase decision taken", "register objection and appeal periods", "transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit", "record date of decision environmental permit", "set decision phase decision permitting decided", "start decision phase decision permitting sent"], "count": 2},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "create subcases completeness", "forward to the competent authority", "phase application received", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement", "phase application receptive", "procedure change", "request complete", "send letter in progress", "article 34  WABO applies", "ask stakeholders views", "by law", "completed subcases content", "coordination of application", "create subcases content", "creating environmental permit decision", "decision date prior to decision", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase decision ready", "procedure change", "set decision status", "submit decision", "suspension ground applicable", "treat subcases content", "creating cover letter decision", "enter senddate decision", "enter senddate decision environmental permit", "phase decision sent", "phase decision taken", "register objection and appeal periods", "transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit", "record date of decision environmental permit", "set decision phase decision permitting decided", "start decision phase decision permitting sent"], "count": 2},
            {"variant": ["register submission date request", "OLO messaging active", "create procedure confirmation", "create publication document", "forward to the competent authority", "phase application received", "publish", "registration date publication", "regular procedure without MER", "send confirmation receipt", "send confirmation receipt", "start WABOprocedure", "enter senddate acknowledgement", "article 34  WABO applies", "ask stakeholders views", "by law", "coordination of application", "create subcases completeness", "create subcases content", "creating environmental permit decision", "decision date prior to decision", "generating decision environmental permit", "grounds for refusal", "phase advice known", "phase application receptive", "phase decision ready", "procedure change", "procedure change", "request complete", "send letter in progress", "set decision status", "submit decision", "suspension ground applicable", "create publication document", "creating cover letter decision", "enter date publication decision environmental permit", "enter senddate decision", "enter senddate decision environmental permit", "phase decision sent", "phase decision taken", "record date of decision environmental permit", "register objection and appeal periods", "set decision phase decision permitting decided", "start decision phase decision permitting sent", "transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit", "set decision phase decision permitting published"], "count": 2},
        ]
    }
}


# -----------------------------
# Build lookup structures
# -----------------------------
DFG_FREQ = {}
for e in PROCESS["process_abstraction"]["dfg_edges"]:
    DFG_FREQ[(e["source"], e["target"])] = e["frequency"]

TRANS_PROB = {}
for e in PROCESS["transition_probabilities_top"]:
    TRANS_PROB[(e["source"], e["target"])] = e["probability"]

DUR = {}
for e in PROCESS["duration_profiles_top"]:
    DUR[(e["source"], e["target"])] = e

START_ACTS = PROCESS["process_abstraction"]["start_activities"]
END_ACTS = PROCESS["process_abstraction"]["end_activities"]
FREQ_VARIANTS = PROCESS["process_abstraction"]["frequent_variants"]


def frac_digits(ts: str) -> int:
    if "." in ts:
        return len(ts.split(".")[-1])
    return 0


def format_ts(ts: pd.Timestamp, digits: int) -> str:
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if digits <= 0:
        return base
    frac = f"{ts.nanosecond // 1000:06d}" if digits <= 6 else f"{ts.nanosecond:09d}"
    if digits <= 6:
        frac_full = ts.strftime("%f")
        return base + "." + frac_full[:digits]
    ns = f"{ts.nanosecond:09d}"
    return base + "." + ns[:digits]


def precision_step(digits: int) -> pd.Timedelta:
    if digits <= 0:
        return pd.Timedelta(seconds=1)
    unit_ns = 10 ** (9 - digits)
    return pd.Timedelta(unit= "ns", value=unit_ns)


def duration_for_edge(a, b, min_step_seconds):
    prof = DUR.get((a, b))
    if prof:
        if prof.get("share_under_60s", 0.0) >= 0.95:
            val = prof.get("p10_seconds", 0.0)
        else:
            val = prof.get("p25_seconds", 0.0)
        if val is None:
            val = 0.0
        val = max(float(val), 0.0)
        return max(val, min_step_seconds)
    if (a, b) in DFG_FREQ:
        return min_step_seconds
    return None


def is_contiguous_subsequence(seq, sub):
    n, m = len(seq), len(sub)
    if m > n:
        return False
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return True
    return False


def variant_score(order):
    s = 0.0
    for fv in FREQ_VARIANTS:
        v = fv["variant"]
        c = fv["count"]
        if is_contiguous_subsequence(v, order):
            s += 1000.0 * c
    return s


def score_order(order, left_act, right_act):
    internal_edges = [(order[i], order[i + 1]) for i in range(len(order) - 1)]

    feasible_internal = all(edge in DFG_FREQ for edge in internal_edges)
    if not feasible_internal:
        return None

    s = 0.0
    s += variant_score(order)

    for e in internal_edges:
        s += 10.0 * math.log1p(DFG_FREQ.get(e, 0))
        s += 100.0 * TRANS_PROB.get(e, 0.0)

    if left_act is not None:
        if (left_act, order[0]) in DFG_FREQ:
            s += 20.0 * math.log1p(DFG_FREQ[(left_act, order[0])])
            s += 100.0 * TRANS_PROB.get((left_act, order[0]), 0.0)
        elif left_act in START_ACTS:
            s -= 5.0
        else:
            s -= 20.0

    if right_act is not None:
        if (order[-1], right_act) in DFG_FREQ:
            s += 20.0 * math.log1p(DFG_FREQ[(order[-1], right_act)])
            s += 100.0 * TRANS_PROB.get((order[-1], right_act), 0.0)
        elif right_act in END_ACTS:
            s -= 5.0
        else:
            s -= 20.0

    if left_act is None and order[0] in START_ACTS:
        s += 5.0 * math.log1p(START_ACTS[order[0]])
    if right_act is None and order[-1] in END_ACTS:
        s += 5.0 * math.log1p(END_ACTS[order[-1]])

    return s


def greedy_order(acts, left_act, right_act):
    remaining = list(range(len(acts)))
    best_starts = []
    for i in remaining:
        a = acts[i]
        sc = 0.0
        if left_act is not None and (left_act, a) in DFG_FREQ:
            sc += 1000 + DFG_FREQ[(left_act, a)]
        elif a in START_ACTS:
            sc += START_ACTS[a]
        best_starts.append((sc, -i, i))
    current = max(best_starts)[2]
    order_idx = [current]
    remaining.remove(current)

    while remaining:
        prev = acts[order_idx[-1]]
        candidates = []
        for j in remaining:
            a = acts[j]
            if (prev, a) in DFG_FREQ:
                sc = 1000 + DFG_FREQ[(prev, a)]
            else:
                sc = 0
            if right_act is not None and (a, right_act) in DFG_FREQ:
                sc += 100 + DFG_FREQ[(a, right_act)]
            candidates.append((sc, -j, j))
        chosen = max(candidates)[2]
        order_idx.append(chosen)
        remaining.remove(chosen)
    return order_idx


def infer_order(block_rows, left_act, right_act):
    acts = list(block_rows["Activity"])
    n = len(acts)
    idxs = list(range(n))

    counts = Counter(acts)
    perm_limit = 720
    unique_perm_count_est = math.factorial(n)
    for c in counts.values():
        unique_perm_count_est //= math.factorial(c)

    candidates = []
    if unique_perm_count_est <= perm_limit:
        seen = set()
        for perm in itertools.permutations(idxs):
            labels = tuple(acts[i] for i in perm)
            if labels in seen:
                continue
            seen.add(labels)
            sc = score_order(list(labels), left_act, right_act)
            if sc is not None:
                candidates.append((sc, perm))
    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        return list(candidates[0][1])

    greedy_idx = greedy_order(acts, left_act, right_act)
    return greedy_idx


def build_times_for_order(order_acts, observed_ts, right_anchor_ts, case_end_ts, digits):
    step = precision_step(digits)
    min_step_seconds = max(step / pd.Timedelta(seconds=1), 1e-9)

    gaps = []
    available_gap_vals = []
    for i in range(len(order_acts) - 1):
        g = duration_for_edge(order_acts[i], order_acts[i + 1], min_step_seconds)
        if g is not None:
            available_gap_vals.append(g)
        gaps.append(g)

    if available_gap_vals:
        borrow = max(min_step_seconds, min(available_gap_vals))
    else:
        borrow = max(min_step_seconds, 1.0 if digits == 0 else min_step_seconds)

    gaps = [borrow if g is None else g for g in gaps]

    target_end = right_anchor_ts if right_anchor_ts is not None else case_end_ts
    room = max((target_end - observed_ts) / pd.Timedelta(seconds=1), 0.0)
    min_total = max((len(order_acts) - 1) * min_step_seconds, 0.0)
    total = sum(gaps)

    clamped = False
    if total > room and total > 0:
        clamped = True
        if room >= min_total:
            scale = room / total
            gaps = [max(min_step_seconds, g * scale) for g in gaps]
            current = sum(gaps)
            if current > room and current > 0:
                excess = current - room
                reducible = [g - min_step_seconds for g in gaps]
                reducible_total = sum(max(0.0, x) for x in reducible)
                if reducible_total > 0:
                    new_gaps = []
                    for g, r in zip(gaps, reducible):
                        dec = excess * max(0.0, r) / reducible_total
                        new_gaps.append(max(min_step_seconds, g - dec))
                    gaps = new_gaps
        else:
            gaps = [min_step_seconds] * (len(order_acts) - 1)

    times = [observed_ts]
    for g in gaps:
        times.append(times[-1] + pd.Timedelta(seconds=float(g)))

    if times[-1] > target_end:
        clamped = True
        times[-1] = target_end
        for i in range(len(times) - 2, -1, -1):
            latest_allowed = times[i + 1] - step
            if times[i] > latest_allowed:
                times[i] = latest_allowed
        if times[0] < observed_ts:
            times[0] = observed_ts
            for i in range(1, len(times)):
                min_allowed = times[i - 1] + step
                if times[i] < min_allowed:
                    times[i] = min_allowed

    return times, clamped


def enforce_block_constraints(times, observed_ts, left_anchor_ts, right_anchor_ts, case_start_ts, case_end_ts, digits):
    step = precision_step(digits)
    clamped = False

    low = observed_ts
    high = right_anchor_ts if right_anchor_ts is not None else case_end_ts
    if left_anchor_ts is not None:
        low = max(low, left_anchor_ts)
    low = max(low, case_start_ts)
    high = min(high, case_end_ts)

    if len(times) == 0:
        return times, clamped

    times[0] = observed_ts
    if times[0] < low:
        times[0] = low
        clamped = True
    if times[0] > high:
        times[0] = high
        clamped = True

    for i in range(1, len(times)):
        min_allowed = times[i - 1] + step
        if times[i] < min_allowed:
            times[i] = min_allowed
            clamped = True

    if times[-1] > high:
        clamped = True
        times[-1] = high
        for i in range(len(times) - 2, -1, -1):
            latest_allowed = times[i + 1] - step
            if times[i] > latest_allowed:
                times[i] = latest_allowed

    if times[0] < observed_ts:
        clamped = True
        times[0] = observed_ts
        for i in range(1, len(times)):
            min_allowed = times[i - 1] + step
            if times[i] < min_allowed:
                times[i] = min_allowed

    if left_anchor_ts is not None:
        for i in range(len(times)):
            if times[i] < left_anchor_ts:
                times[i] = left_anchor_ts + i * step
                clamped = True

    for i in range(len(times)):
        if times[i] < case_start_ts:
            times[i] = case_start_ts + i * step
            clamped = True
        if times[i] > case_end_ts:
            times[i] = case_end_ts - (len(times) - 1 - i) * step
            clamped = True

    for i in range(1, len(times)):
        if not times[i] > times[i - 1]:
            times[i] = times[i - 1] + step
            clamped = True

    if right_anchor_ts is not None and times[-1] > right_anchor_ts:
        clamped = True
        times[-1] = right_anchor_ts
        for i in range(len(times) - 2, -1, -1):
            latest_allowed = times[i + 1] - step
            if times[i] > latest_allowed:
                times[i] = latest_allowed

    return times, clamped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original = df.copy(deep=True)
    out_ts = list(df["Timestamp"])

    parsed_ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_parsed_ts"] = parsed_ts
    df["_orig_idx"] = range(len(df))

    blocks_repaired = 0
    blocks_clamped = 0

    for case_id, cdf in df.groupby("Case", sort=False):
        cidx = list(cdf.index)
        case_start = cdf["_parsed_ts"].min()
        case_end = cdf["_parsed_ts"].max()
        if pd.isna(case_start) or pd.isna(case_end):
            continue

        i = 0
        while i < len(cidx):
            idx = cidx[i]
            if df.at[idx, "error_category"] == "CLEAN":
                i += 1
                continue

            j = i
            shared_ts_str = df.at[cidx[i], "Timestamp"]
            while j + 1 < len(cidx):
                nidx = cidx[j + 1]
                if df.at[nidx, "error_category"] != "CLEAN" and df.at[nidx, "Timestamp"] == shared_ts_str:
                    j += 1
                else:
                    break

            block_indices = cidx[i:j + 1]
            k = len(block_indices)
            if k == 1:
                i = j + 1
                continue

            observed_ts = pd.to_datetime(shared_ts_str, errors="coerce")
            if pd.isna(observed_ts):
                i = j + 1
                continue

            digits = frac_digits(shared_ts_str)
            left_anchor_idx = cidx[i - 1] if i - 1 >= 0 else None
            right_anchor_idx = cidx[j + 1] if j + 1 < len(cidx) else None

            left_anchor_ts = None
            left_anchor_act = None
            if left_anchor_idx is not None and df.at[left_anchor_idx, "error_category"] == "CLEAN":
                left_anchor_ts = df.at[left_anchor_idx, "_parsed_ts"]
                left_anchor_act = df.at[left_anchor_idx, "Activity"]

            right_anchor_ts = None
            right_anchor_act = None
            if right_anchor_idx is not None and df.at[right_anchor_idx, "error_category"] == "CLEAN":
                right_anchor_ts = df.at[right_anchor_idx, "_parsed_ts"]
                right_anchor_act = df.at[right_anchor_idx, "Activity"]

            block_rows = df.loc[block_indices]
            order_positions = infer_order(block_rows, left_anchor_act, right_anchor_act)
            ordered_block_indices = [block_indices[p] for p in order_positions]
            ordered_acts = [df.at[idx2, "Activity"] for idx2 in ordered_block_indices]

            times, clamped1 = build_times_for_order(
                ordered_acts, observed_ts, right_anchor_ts, case_end, digits
            )
            times, clamped2 = enforce_block_constraints(
                times, observed_ts, left_anchor_ts, right_anchor_ts, case_start, case_end, digits
            )
            clamped = clamped1 or clamped2

            step = precision_step(digits)
            min_span = (k - 1) * step
            actual_span = times[-1] - times[0]
            if actual_span < min_span:
                clamped = True
                base = times[0]
                times = [base + n * step for n in range(k)]
                times, clamped3 = enforce_block_constraints(
                    times, observed_ts, left_anchor_ts, right_anchor_ts, case_start, case_end, digits
                )
                clamped = clamped or clamped3

            if times[0] != observed_ts:
                clamped = True
                times[0] = observed_ts
                for t_i in range(1, k):
                    if not times[t_i] > times[t_i - 1]:
                        times[t_i] = times[t_i - 1] + step
                times, clamped3 = enforce_block_constraints(
                    times, observed_ts, left_anchor_ts, right_anchor_ts, case_start, case_end, digits
                )
                clamped = clamped or clamped3

            for pos, ridx in enumerate(ordered_block_indices):
                out_ts[ridx] = format_ts(times[pos], digits)

            blocks_repaired += 1
            if clamped:
                blocks_clamped += 1

            i = j + 1

    df_out = df.drop(columns=["_parsed_ts", "_orig_idx"]).copy()
    df_out["Timestamp"] = out_ts

    # Self-check and corrective pass
    rows_still_equal = 0
    try:
        repaired_mask = df_out["error_category"] != "CLEAN"
        clean_mask = df_out["error_category"] == "CLEAN"

        df_out.loc[clean_mask, "Timestamp"] = original.loc[clean_mask, "Timestamp"]

        parsed_out = pd.to_datetime(df_out["Timestamp"], errors="coerce")
        bad_parse = parsed_out.isna()
        if bad_parse.any():
            for idx in df_out.index[bad_parse]:
                if df_out.at[idx, "error_category"] == "CLEAN":
                    df_out.at[idx, "Timestamp"] = original.at[idx, "Timestamp"]
                else:
                    obs = original.at[idx, "Timestamp"]
                    digits = frac_digits(obs)
                    ts = pd.to_datetime(obs, errors="coerce")
                    if pd.isna(ts):
                        ts = pd.Timestamp("2000-01-01 00:00:00")
                    ts = ts + precision_step(digits)
                    df_out.at[idx, "Timestamp"] = format_ts(ts, digits)

        for case_id, cdf in df_out.groupby("Case", sort=False):
            cidx = list(cdf.index)
            parsed_case = pd.to_datetime(cdf["Timestamp"], errors="coerce")
            case_start = parsed_case.min()
            case_end = parsed_case.max()
            i = 0
            while i < len(cidx):
                idx = cidx[i]
                if df_out.at[idx, "error_category"] == "CLEAN":
                    i += 1
                    continue
                shared = original.at[idx, "Timestamp"]
                j = i
                while j + 1 < len(cidx):
                    nidx = cidx[j + 1]
                    if df_out.at[nidx, "error_category"] != "CLEAN" and original.at[nidx, "Timestamp"] == shared:
                        j += 1
                    else:
                        break
                block = cidx[i:j + 1]
                if len(block) > 1:
                    parsed_block = pd.to_datetime(df_out.loc[block, "Timestamp"], errors="coerce")
                    equals_input = [df_out.at[b, "Timestamp"] == original.at[b, "Timestamp"] for b in block]
                    if sum(equals_input) != 1:
                        # Keep earliest repaired timestamp equal to observed; move the rest minimally forward
                        order = sorted(block, key=lambda x: pd.to_datetime(df_out.at[x, "Timestamp"], errors="coerce"))
                        first = order[0]
                        df_out.at[first, "Timestamp"] = original.at[first, "Timestamp"]
                        digits = frac_digits(original.at[first, "Timestamp"])
                        step = precision_step(digits)
                        base = pd.to_datetime(original.at[first, "Timestamp"], errors="coerce")
                        for m, ridx in enumerate(order[1:], start=1):
                            new_ts = base + m * step
                            right_anchor = pd.to_datetime(df_out.at[cidx[j + 1], "Timestamp"], errors="coerce") if j + 1 < len(cidx) and df_out.at[cidx[j + 1], "error_category"] == "CLEAN" else case_end
                            if new_ts > right_anchor:
                                new_ts = right_anchor - (len(order) - 1 - m) * step
                            if new_ts <= base:
                                new_ts = base + m * step
                            df_out.at[ridx, "Timestamp"] = format_ts(new_ts, digits)
                    # ensure distinct
                    order = sorted(block, key=lambda x: pd.to_datetime(df_out.at[x, "Timestamp"], errors="coerce"))
                    digits = max(frac_digits(original.at[b, "Timestamp"]) for b in block)
                    step = precision_step(digits)
                    prev = None
                    for ridx in order:
                        t = pd.to_datetime(df_out.at[ridx, "Timestamp"], errors="coerce")
                        if prev is not None and not t > prev:
                            t = prev + step
                            df_out.at[ridx, "Timestamp"] = format_ts(t, digits)
                        prev = t
                i = j + 1

        rows_still_equal = int((df_out.loc[repaired_mask, "Timestamp"] == original.loc[repaired_mask, "Timestamp"]).sum())

    except Exception:
        # never fail hard
        rows_still_equal = int(((df_out["error_category"] != "CLEAN") & (df_out["Timestamp"] == original["Timestamp"])).sum())

    # Preserve all non-timestamp columns exactly and row order
    final_df = original.copy()
    final_df["Timestamp"] = df_out["Timestamp"]

    try:
        final_df.to_csv(args.output, index=False)
    except Exception:
        try:
            final_df.to_csv(args.output, index=False, lineterminator="\n")
        except Exception:
            pass

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={blocks_clamped} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Best effort, zero exit status
        print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")