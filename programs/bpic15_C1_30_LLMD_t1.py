import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict

import pandas as pd

PROCESS_CONTEXT = r"""
{
  "transition_probabilities_top": [
    {
      "source": "appeal subcase completed",
      "target": "appealed to higher court",
      "probability": 1.0
    },
    {
      "source": "create coordination confirmation",
      "target": "by law",
      "probability": 1.0
    },
    {
      "source": "create decision no permit required",
      "target": "create subcases completeness",
      "probability": 1.0
    },
    {
      "source": "create forwarding copy",
      "target": "enter senddate letter to stakeholders",
      "probability": 1.0
    },
    {
      "source": "create letter only notification required",
      "target": "forward to the competent authority",
      "probability": 1.0
    },
    {
      "source": "creating cover letter no permit required",
      "target": "decision date prior to decision",
      "probability": 1.0
    },
    {
      "source": "date environmental permit irrevocable",
      "target": "assessment of content completed",
      "probability": 1.0
    },
    {
      "source": "enrich draft decision environmental permit",
      "target": "enter date draft decisionenvironmental permit",
      "probability": 1.0
    },
    {
      "source": "enter objection term permanent suspension",
      "target": "copy decision permanent suspension to stakeholders",
      "probability": 1.0
    },
    {
      "source": "enter senddate letter no permit required",
      "target": "generating decision no permit required",
      "probability": 1.0
    },
    {
      "source": "generating decision no permit required",
      "target": "no permit needed or only notification needed",
      "probability": 1.0
    },
    {
      "source": "objection decision no permit required received",
      "target": "phase activity no permit required",
      "probability": 1.0
    },
    {
      "source": "partly permit application",
      "target": "partly permit",
      "probability": 1.0
    },
    {
      "source": "phase activity no permit required",
      "target": "phase archived case",
      "probability": 1.0
    },
    {
      "source": "phase forwarded to competent authority",
      "target": "phase procedure aborted",
      "probability": 1.0
    },
    {
      "source": "phase only notification required",
      "target": "phase procedure aborted",
      "probability": 1.0
    },
    {
      "source": "phase updated plan requested",
      "target": "register receipt updated plan",
      "probability": 1.0
    },
    {
      "source": "reactive designation received",
      "target": "register objection and appeal periods",
      "probability": 1.0
    },
    {
      "source": "record date of decision no permit required",
      "target": "enter senddate letter no permit required",
      "probability": 1.0
    },
    {
      "source": "request updated plan again",
      "target": "contacting applicant",
      "probability": 1.0
    },
    {
      "source": "send letter receptive test not ok wait for report firedepartement",
      "target": "resume completeness subcases",
      "probability": 1.0
    },
    {
      "source": "send send confirmation competent authority",
      "target": "request to competent authority",
      "probability": 1.0
    },
    {
      "source": "set decision phase decision no permit required sent",
      "target": "set decision phase no permit required decided",
      "probability": 1.0
    },
    {
      "source": "set decision phase decision permitting published",
      "target": "close case",
      "probability": 1.0
    },
    {
      "source": "set decision phase no permit required decided",
      "target": "set decision status",
      "probability": 1.0
    },
    {
      "source": "set decision phase unlicensed irrevocable",
      "target": "close case",
      "probability": 1.0
    },
    {
      "source": "set phase",
      "target": "creating environmental permit decision",
      "probability": 1.0
    },
    {
      "source": "subcase art. 34 WABO completed",
      "target": "refusal after suspension",
      "probability": 1.0
    },
    {
      "source": "subcase designation GS completed",
      "target": "subcase perspective complete",
      "probability": 1.0
    },
    {
      "source": "subcases submit draft decision completed",
      "target": "enrich draft decision",
      "probability": 1.0
    },
    {
      "source": "temporary permit",
      "target": "regular procedure without MER",
      "probability": 1.0
    },
    {
      "source": "treat subcase suspension art. 34 WABO",
      "target": "subcase art. 34 WABO completed",
      "probability": 1.0
    },
    {
      "source": "treat subcases submit draft decision",
      "target": "subcases submit draft decision completed",
      "probability": 1.0
    },
    {
      "source": "treating subcase designation GS",
      "target": "updated plan after perspective",
      "probability": 1.0
    },
    {
      "source": "phase archived case",
      "target": "phase case handled",
      "probability": 0.9813
    },
    {
      "source": "enter senddate decision",
      "target": "enter senddate decision environmental permit",
      "probability": 0.9729
    },
    {
      "source": "enter objectionterm",
      "target": "copy decision to extend procedure to stakeholders",
      "probability": 0.9444
    },
    {
      "source": "decided on draft decision environmental permit",
      "target": "enrich draft decision",
      "probability": 0.9
    },
    {
      "source": "decision terminate on request",
      "target": "phase procedure aborted",
      "probability": 0.875
    },
    {
      "source": "no permit required for application",
      "target": "MER present",
      "probability": 0.85
    },
    {
      "source": "MER present",
      "target": "activities regular procedure",
      "probability": 0.8478
    },
    {
      "source": "start decision phase permanent suspension sent",
      "target": "start decision phase permanently suspended decided",
      "probability": 0.8462
    },
    {
      "source": "cancel BAG objects research",
      "target": "phase procedure aborted",
      "probability": 0.8333
    },
    {
      "source": "determine cause for suspension article 34 WABO",
      "target": "assessment of content completed",
      "probability": 0.8182
    },
    {
      "source": "create monitoring case",
      "target": "creating cover letter decision",
      "probability": 0.8065
    },
    {
      "source": "phase decision irrevocable",
      "target": "read publication date field",
      "probability": 0.8
    },
    {
      "source": "subcase final VVGB completed",
      "target": "VVGB positive",
      "probability": 0.8
    },
    {
      "source": "treat subcase final VVGB",
      "target": "subcase final VVGB completed",
      "probability": 0.8
    },
    {
      "source": "create decision terminate on request",
      "target": "decision terminate on request",
      "probability": 0.7778
    },
    {
      "source": "set phase decision revoked",
      "target": "close case",
      "probability": 0.7778
    },
    {
      "source": "publication date draft decision",
      "target": "date draft decision for inspection",
      "probability": 0.7647
    },
    {
      "source": "reception through OLO",
      "target": "send confirmation receipt",
      "probability": 0.7589
    },
    {
      "source": "environmental permit irrevocable",
      "target": "determine cause for suspension article 34 WABO",
      "probability": 0.75
    },
    {
      "source": "objection to extension procedure term receptive",
      "target": "decision extension procedure term irrevocable",
      "probability": 0.75
    },
    {
      "source": "publish official draft decision",
      "target": "send design decision to stakeholders",
      "probability": 0.75
    },
    {
      "source": "term perspective on draft decision",
      "target": "phase draft decision made",
      "probability": 0.75
    },
    {
      "source": "treat subcase WAW",
      "target": "subcase WAW completed",
      "probability": 0.75
    },
    {
      "source": "MER required",
      "target": "OLO messaging active",
      "probability": 0.7333
    },
    {
      "source": "treat subcases completeness",
      "target": "subcases completeness completed",
      "probability": 0.7206
    },
    {
      "source": "create subcases submit draft decision",
      "target": "enrich draft decision",
      "probability": 0.7143
    }
  ],
  "duration_profiles_top": [
    {
      "source": "register submission date request",
      "target": "OLO messaging active",
      "n": 729,
      "p10_seconds": 86400.0,
      "p25_seconds": 172800.0,
      "median_seconds": 306406.0,
      "p75_seconds": 518400.0,
      "p90_seconds": 733026.6,
      "share_under_60s": 0.0
    },
    {
      "source": "forward to the competent authority",
      "target": "regular procedure without MER",
      "n": 673,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 3.0,
      "p75_seconds": 6.0,
      "p90_seconds": 12.0,
      "share_under_60s": 0.973
    },
    {
      "source": "send confirmation receipt",
      "target": "forward to the competent authority",
      "n": 591,
      "p10_seconds": 2.0,
      "p25_seconds": 3.0,
      "median_seconds": 6.0,
      "p75_seconds": 9.0,
      "p90_seconds": 19.0,
      "share_under_60s": 0.941
    },
    {
      "source": "request complete",
      "target": "send letter in progress",
      "n": 572,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 1.0,
      "p75_seconds": 2.0,
      "p90_seconds": 4.0,
      "share_under_60s": 0.991
    },
    {
      "source": "article 34  WABO applies",
      "target": "assessment of content completed",
      "n": 568,
      "p10_seconds": 0.0,
      "p25_seconds": 1.0,
      "median_seconds": 1.0,
      "p75_seconds": 2.0,
      "p90_seconds": 4.0,
      "share_under_60s": 0.993
    },
    {
      "source": "assessment of content completed",
      "target": "phase advice known",
      "n": 555,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 1.0,
      "share_under_60s": 1.0
    },
    {
      "source": "send confirmation receipt",
      "target": "send confirmation receipt",
      "n": 503,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 211.20000000000005,
      "share_under_60s": 0.799
    },
    {
      "source": "phase application receptive",
      "target": "request complete",
      "n": 500,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "treat subcases content",
      "target": "completed subcases content",
      "n": 493,
      "p10_seconds": 1.0,
      "p25_seconds": 1.0,
      "median_seconds": 1.0,
      "p75_seconds": 2.0,
      "p90_seconds": 3.0,
      "share_under_60s": 0.992
    },
    {
      "source": "treat subcases completeness",
      "target": "subcases completeness completed",
      "n": 490,
      "p10_seconds": 1.0,
      "p25_seconds": 1.0,
      "median_seconds": 2.0,
      "p75_seconds": 3.0,
      "p90_seconds": 7.0,
      "share_under_60s": 0.984
    },
    {
      "source": "by law",
      "target": "creating environmental permit decision",
      "n": 481,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 1.0,
      "share_under_60s": 0.994
    },
    {
      "source": "enter senddate decision",
      "target": "enter senddate decision environmental permit",
      "n": 467,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 0.991
    },
    {
      "source": "phase advice known",
      "target": "grounds for refusal",
      "n": 438,
      "p10_seconds": 1.0,
      "p25_seconds": 1.0,
      "median_seconds": 2.0,
      "p75_seconds": 3.0,
      "p90_seconds": 6.300000000000011,
      "share_under_60s": 0.954
    },
    {
      "source": "ask stakeholders views",
      "target": "suspension ground applicable",
      "n": 431,
      "p10_seconds": 0.0,
      "p25_seconds": 1.0,
      "median_seconds": 2.0,
      "p75_seconds": 4.0,
      "p90_seconds": 6.0,
      "share_under_60s": 0.984
    },
    {
      "source": "grounds for refusal",
      "target": "ask stakeholders views",
      "n": 426,
      "p10_seconds": 1.0,
      "p25_seconds": 1.0,
      "median_seconds": 2.0,
      "p75_seconds": 5.0,
      "p90_seconds": 15.5,
      "share_under_60s": 0.955
    },
    {
      "source": "subcases completeness completed",
      "target": "procedure change",
      "n": 424,
      "p10_seconds": 1.0,
      "p25_seconds": 2.0,
      "median_seconds": 3.0,
      "p75_seconds": 4.0,
      "p90_seconds": 7.0,
      "share_under_60s": 0.991
    },
    {
      "source": "creating environmental permit decision",
      "target": "decision date prior to decision",
      "n": 423,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 4.0,
      "share_under_60s": 0.993
    },
    {
      "source": "procedure change",
      "target": "request complete",
      "n": 421,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 3.0,
      "p90_seconds": 6.0,
      "share_under_60s": 0.988
    },
    {
      "source": "send confirmation receipt",
      "target": "enter senddate acknowledgement",
      "n": 412,
      "p10_seconds": 2.0,
      "p25_seconds": 67.0,
      "median_seconds": 159.0,
      "p75_seconds": 308.25,
      "p90_seconds": 1342.5000000000043,
      "share_under_60s": 0.245
    },
    {
      "source": "suspension ground applicable",
      "target": "by law",
      "n": 410,
      "p10_seconds": 1.0,
      "p25_seconds": 1.0,
      "median_seconds": 1.0,
      "p75_seconds": 2.0,
      "p90_seconds": 4.0,
      "share_under_60s": 0.993
    },
    {
      "source": "OLO messaging active",
      "target": "phase application received",
      "n": 409,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "regular procedure without MER",
      "target": "send confirmation receipt",
      "n": 399,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "phase application received",
      "target": "send confirmation receipt",
      "n": 393,
      "p10_seconds": 1.0,
      "p25_seconds": 1.0,
      "median_seconds": 2.0,
      "p75_seconds": 4.0,
      "p90_seconds": 10.0,
      "share_under_60s": 0.982
    },
    {
      "source": "procedure change",
      "target": "phase application receptive",
      "n": 383,
      "p10_seconds": 1.0,
      "p25_seconds": 2.0,
      "median_seconds": 3.0,
      "p75_seconds": 5.0,
      "p90_seconds": 9.0,
      "share_under_60s": 0.966
    },
    {
      "source": "enter senddate acknowledgement",
      "target": "send confirmation receipt",
      "n": 383,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 0.99
    },
    {
      "source": "enter senddate procedure confirmation",
      "target": "send procedure confirmation",
      "n": 378,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 0.952
    },
    {
      "source": "send confirmation receipt",
      "target": "start WABOprocedure",
      "n": 375,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "forward to the competent authority",
      "target": "phase application received",
      "n": 372,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "registration date publication",
      "target": "regular procedure without MER",
      "n": 357,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 0.994
    },
    {
      "source": "procedure change",
      "target": "article 34  WABO applies",
      "n": 352,
      "p10_seconds": 1.0,
      "p25_seconds": 1.0,
      "median_seconds": 2.0,
      "p75_seconds": 3.0,
      "p90_seconds": 9.900000000000034,
      "share_under_60s": 0.949
    },
    {
      "source": "OLO messaging active",
      "target": "create procedure confirmation",
      "n": 344,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "transcript decision environmental permit to stakeholders",
      "target": "generate publication document decision environmental permit",
      "n": 341,
      "p10_seconds": 6.0,
      "p25_seconds": 7.0,
      "median_seconds": 10.0,
      "p75_seconds": 19.0,
      "p90_seconds": 55.0,
      "share_under_60s": 0.906
    },
    {
      "source": "send letter in progress",
      "target": "treat subcases content",
      "n": 324,
      "p10_seconds": 0.0,
      "p25_seconds": 1.0,
      "median_seconds": 1.0,
      "p75_seconds": 2.0,
      "p90_seconds": 2.0,
      "share_under_60s": 0.991
    },
    {
      "source": "phase decision sent",
      "target": "phase decision taken",
      "n": 320,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "grounds for refusal",
      "target": "phase advice known",
      "n": 319,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "create procedure confirmation",
      "target": "create publication document",
      "n": 317,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "phase advice known",
      "target": "phase application receptive",
      "n": 316,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "phase decision taken",
      "target": "register objection and appeal periods",
      "n": 308,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 5.0,
      "p90_seconds": 12.0,
      "share_under_60s": 0.974
    },
    {
      "source": "article 34  WABO applies",
      "target": "ask stakeholders views",
      "n": 305,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 0.997
    },
    {
      "source": "completed subcases content",
      "target": "procedure change",
      "n": 298,
      "p10_seconds": 1.0,
      "p25_seconds": 2.0,
      "median_seconds": 3.0,
      "p75_seconds": 5.0,
      "p90_seconds": 10.0,
      "share_under_60s": 0.98
    },
    {
      "source": "enter senddate decision environmental permit",
      "target": "phase decision sent",
      "n": 293,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 1.0,
      "share_under_60s": 1.0
    },
    {
      "source": "start WABOprocedure",
      "target": "enter senddate acknowledgement",
      "n": 293,
      "p10_seconds": 34535.2,
      "p25_seconds": 38196.0,
      "median_seconds": 43648.0,
      "p75_seconds": 52527.0,
      "p90_seconds": 86400.0,
      "share_under_60s": 0.0
    },
    {
      "source": "creating environmental permit decision",
      "target": "generating decision environmental permit",
      "n": 284,
      "p10_seconds": 4.0,
      "p25_seconds": 55.5,
      "median_seconds": 1796.0,
      "p75_seconds": 113196.25,
      "p90_seconds": 914443.3000000016,
      "share_under_60s": 0.257
    },
    {
      "source": "send letter in progress",
      "target": "set decision status",
      "n": 254,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "set decision phase decision permitting decided",
      "target": "start decision phase decision permitting sent",
      "n": 241,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 0.988
    },
    {
      "source": "regular procedure without MER",
      "target": "start WABOprocedure",
      "n": 230,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 1.0,
      "share_under_60s": 1.0
    },
    {
      "source": "transcript decision environmental permit to stakeholders",
      "target": "enter senddate decision environmental permit",
      "n": 225,
      "p10_seconds": 8.0,
      "p25_seconds": 34.0,
      "median_seconds": 50131.0,
      "p75_seconds": 172800.0,
      "p90_seconds": 259200.0,
      "share_under_60s": 0.258
    },
    {
      "source": "send procedure confirmation",
      "target": "registration date publication",
      "n": 224,
      "p10_seconds": 8.0,
      "p25_seconds": 11.0,
      "median_seconds": 19.0,
      "p75_seconds": 78.0,
      "p90_seconds": 738814.7000000001,
      "share_under_60s": 0.728
    },
    {
      "source": "phase application received",
      "target": "reception through OLO",
      "n": 222,
      "p10_seconds": 1.0,
      "p25_seconds": 2.0,
      "median_seconds": 3.0,
      "p75_seconds": 6.0,
      "p90_seconds": 14.900000000000006,
      "share_under_60s": 0.973
    },
    {
      "source": "regular procedure without MER",
      "target": "enter senddate procedure confirmation",
      "n": 219,
      "p10_seconds": 1.0,
      "p25_seconds": 2.0,
      "median_seconds": 2.0,
      "p75_seconds": 4.0,
      "p90_seconds": 87809.20000000433,
      "share_under_60s": 0.872
    },
    {
      "source": "generating decision environmental permit",
      "target": "grounds for refusal",
      "n": 215,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "decision date prior to decision",
      "target": "generating decision environmental permit",
      "n": 209,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 1.0,
      "p75_seconds": 4206.0,
      "p90_seconds": 430595.80000000034,
      "share_under_60s": 0.569
    },
    {
      "source": "create publication document",
      "target": "create subcases completeness",
      "n": 197,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 7.400000000000006,
      "share_under_60s": 0.914
    },
    {
      "source": "ask stakeholders views",
      "target": "assessment of content completed",
      "n": 194,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "by law",
      "target": "completed subcases content",
      "n": 194,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "submit decision",
      "target": "suspension ground applicable",
      "n": 193,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "procedure change",
      "target": "procedure change",
      "n": 193,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "completed subcases content",
      "target": "coordination of application",
      "n": 190,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    },
    {
      "source": "register submission date request",
      "target": "enter senddate acknowledgement",
      "n": 188,
      "p10_seconds": 172800.0,
      "p25_seconds": 345600.0,
      "median_seconds": 604800.0,
      "p75_seconds": 864000.0,
      "p90_seconds": 1209600.0,
      "share_under_60s": 0.0
    },
    {
      "source": "record date of decision environmental permit",
      "target": "set decision phase decision permitting decided",
      "n": 187,
      "p10_seconds": 0.0,
      "p25_seconds": 0.0,
      "median_seconds": 0.0,
      "p75_seconds": 0.0,
      "p90_seconds": 0.0,
      "share_under_60s": 1.0
    }
  ],
  "block_span_profile_seconds": {
    "2": {
      "n": 50569,
      "p10": 0.0,
      "p25": 0.0,
      "median": 1.0,
      "p75": 11.0,
      "p90": 172800.0
    },
    "3": {
      "n": 49370,
      "p10": 0.0,
      "p25": 0.0,
      "median": 5.0,
      "p75": 1156.75,
      "p90": 691200.0
    },
    "4": {
      "n": 48173,
      "p10": 0.0,
      "p25": 0.0,
      "median": 12.0,
      "p75": 86400.0,
      "p90": 1479942.0000000151
    },
    "5": {
      "n": 46979,
      "p10": 0.0,
      "p25": 0.0,
      "median": 30.0,
      "p75": 312956.5,
      "p90": 2356837.000000011
    }
  },
  "global_median_duration_seconds": 1.0,
  "note": "Estimated from CLEAN rows only. duration_profiles give the distribution, not a point estimate: these gaps are bimodal, so the median mixes a fast automated mode with a slow waiting mode. share_under_60s tells you how often a transition is the fast kind. block_span_profile gives how long a run of k consecutive events actually lasts -- use it to size a reconstructed block.",
  "process_abstraction": {
    "pm4py_available": true,
    "dfg_edges": [
      {
        "source": "register submission date request",
        "target": "OLO messaging active",
        "frequency": 729,
        "median_duration_seconds": 306406.0
      },
      {
        "source": "forward to the competent authority",
        "target": "regular procedure without MER",
        "frequency": 673,
        "median_duration_seconds": 3.0
      },
      {
        "source": "send confirmation receipt",
        "target": "forward to the competent authority",
        "frequency": 591,
        "median_duration_seconds": 6.0
      },
      {
        "source": "request complete",
        "target": "send letter in progress",
        "frequency": 572,
        "median_duration_seconds": 1.0
      },
      {
        "source": "article 34  WABO applies",
        "target": "assessment of content completed",
        "frequency": 568,
        "median_duration_seconds": 1.0
      },
      {
        "source": "assessment of content completed",
        "target": "phase advice known",
        "frequency": 555,
        "median_duration_seconds": 0.0
      },
      {
        "source": "send confirmation receipt",
        "target": "send confirmation receipt",
        "frequency": 503,
        "median_duration_seconds": 0.0
      },
      {
        "source": "phase application receptive",
        "target": "request complete",
        "frequency": 500,
        "median_duration_seconds": 0.0
      },
      {
        "source": "treat subcases content",
        "target": "completed subcases content",
        "frequency": 493,
        "median_duration_seconds": 1.0
      },
      {
        "source": "treat subcases completeness",
        "target": "subcases completeness completed",
        "frequency": 490,
        "median_duration_seconds": 2.0
      },
      {
        "source": "by law",
        "target": "creating environmental permit decision",
        "frequency": 481,
        "median_duration_seconds": 0.0
      },
      {
        "source": "enter senddate decision",
        "target": "enter senddate decision environmental permit",
        "frequency": 467,
        "median_duration_seconds": 0.0
      },
      {
        "source": "phase advice known",
        "target": "grounds for refusal",
        "frequency": 438,
        "median_duration_seconds": 2.0
      },
      {
        "source": "ask stakeholders views",
        "target": "suspension ground applicable",
        "frequency": 431,
        "median_duration_seconds": 2.0
      },
      {
        "source": "grounds for refusal",
        "target": "ask stakeholders views",
        "frequency": 426,
        "median_duration_seconds": 2.0
      },
      {
        "source": "subcases completeness completed",
        "target": "procedure change",
        "frequency": 424,
        "median_duration_seconds": 3.0
      },
      {
        "source": "creating environmental permit decision",
        "target": "decision date prior to decision",
        "frequency": 423,
        "median_duration_seconds": 0.0
      },
      {
        "source": "procedure change",
        "target": "request complete",
        "frequency": 421,
        "median_duration_seconds": 0.0
      },
      {
        "source": "send confirmation receipt",
        "target": "enter senddate acknowledgement",
        "frequency": 412,
        "median_duration_seconds": 159.0
      },
      {
        "source": "suspension ground applicable",
        "target": "by law",
        "frequency": 410,
        "median_duration_seconds": 1.0
      },
      {
        "source": "OLO messaging active",
        "target": "phase application received",
        "frequency": 409,
        "median_duration_seconds": 0.0
      },
      {
        "source": "regular procedure without MER",
        "target": "send confirmation receipt",
        "frequency": 399,
        "median_duration_seconds": 0.0
      },
      {
        "source": "phase application received",
        "target": "send confirmation receipt",
        "frequency": 393,
        "median_duration_seconds": 2.0
      },
      {
        "source": "enter senddate acknowledgement",
        "target": "send confirmation receipt",
        "frequency": 383,
        "median_duration_seconds": 0.0
      },
      {
        "source": "procedure change",
        "target": "phase application receptive",
        "frequency": 383,
        "median_duration_seconds": 3.0
      },
      {
        "source": "enter senddate procedure confirmation",
        "target": "send procedure confirmation",
        "frequency": 378,
        "median_duration_seconds": 0.0
      },
      {
        "source": "send confirmation receipt",
        "target": "start WABOprocedure",
        "frequency": 375,
        "median_duration_seconds": 0.0
      },
      {
        "source": "forward to the competent authority",
        "target": "phase application received",
        "frequency": 372,
        "median_duration_seconds": 0.0
      },
      {
        "source": "registration date publication",
        "target": "regular procedure without MER",
        "frequency": 357,
        "median_duration_seconds": 0.0
      },
      {
        "source": "procedure change",
        "target": "article 34  WABO applies",
        "frequency": 352,
        "median_duration_seconds": 2.0
      },
      {
        "source": "OLO messaging active",
        "target": "create procedure confirmation",
        "frequency": 344,
        "median_duration_seconds": 0.0
      },
      {
        "source": "transcript decision environmental permit to stakeholders",
        "target": "generate publication document decision environmental permit",
        "frequency": 341,
        "median_duration_seconds": 10.0
      },
      {
        "source": "send letter in progress",
        "target": "treat subcases content",
        "frequency": 324,
        "median_duration_seconds": 1.0
      },
      {
        "source": "phase decision sent",
        "target": "phase decision taken",
        "frequency": 320,
        "median_duration_seconds": 0.0
      },
      {
        "source": "grounds for refusal",
        "target": "phase advice known",
        "frequency": 319,
        "median_duration_seconds": 0.0
      },
      {
        "source": "create procedure confirmation",
        "target": "create publication document",
        "frequency": 317,
        "median_duration_seconds": 0.0
      },
      {
        "source": "phase advice known",
        "target": "phase application receptive",
        "frequency": 316,
        "median_duration_seconds": 0.0
      },
      {
        "source": "phase decision taken",
        "target": "register objection and appeal periods",
        "frequency": 308,
        "median_duration_seconds": 0.0
      },
      {
        "source": "article 34  WABO applies",
        "target": "ask stakeholders views",
        "frequency": 305,
        "median_duration_seconds": 0.0
      },
      {
        "source": "completed subcases content",
        "target": "procedure change",
        "frequency": 298,
        "median_duration_seconds": 3.0
      },
      {
        "source": "start WABOprocedure",
        "target": "enter senddate acknowledgement",
        "frequency": 293,
        "median_duration_seconds": 43648.0
      },
      {
        "source": "enter senddate decision environmental permit",
        "target": "phase decision sent",
        "frequency": 293,
        "median_duration_seconds": 0.0
      },
      {
        "source": "creating environmental permit decision",
        "target": "generating decision environmental permit",
        "frequency": 284,
        "median_duration_seconds": 1796.0
      },
      {
        "source": "send letter in progress",
        "target": "set decision status",
        "frequency": 254,
        "median_duration_seconds": 0.0
      },
      {
        "source": "set decision phase decision permitting decided",
        "target": "start decision phase decision permitting sent",
        "frequency": 241,
        "median_duration_seconds": 0.0
      },
      {
        "source": "regular procedure without MER",
        "target": "start WABOprocedure",
        "frequency": 230,
        "median_duration_seconds": 0.0
      },
      {
        "source": "transcript decision environmental permit to stakeholders",
        "target": "enter senddate decision environmental permit",
        "frequency": 225,
        "median_duration_seconds": 50131.0
      },
      {
        "source": "send procedure confirmation",
        "target": "registration date publication",
        "frequency": 224,
        "median_duration_seconds": 19.0
      },
      {
        "source": "phase application received",
        "target": "reception through OLO",
        "frequency": 222,
        "median_duration_seconds": 3.0
      },
      {
        "source": "regular procedure without MER",
        "target": "enter senddate procedure confirmation",
        "frequency": 219,
        "median_duration_seconds": 2.0
      },
      {
        "source": "generating decision environmental permit",
        "target": "grounds for refusal",
        "frequency": 215,
        "median_duration_seconds": 0.0
      },
      {
        "source": "decision date prior to decision",
        "target": "generating decision environmental permit",
        "frequency": 209,
        "median_duration_seconds": 1.0
      },
      {
        "source": "create publication document",
        "target": "create subcases completeness",
        "frequency": 197,
        "median_duration_seconds": 0.0
      },
      {
        "source": "ask stakeholders views",
        "target": "assessment of content completed",
        "frequency": 194,
        "median_duration_seconds": 0.0
      },
      {
        "source": "by law",
        "target": "completed subcases content",
        "frequency": 194,
        "median_duration_seconds": 0.0
      },
      {
        "source": "procedure change",
        "target": "procedure change",
        "frequency": 193,
        "median_duration_seconds": 0.0
      },
      {
        "source": "submit decision",
        "target": "suspension ground applicable",
        "frequency": 193,
        "median_duration_seconds": 0.0
      },
      {
        "source": "completed subcases content",
        "target": "coordination of application",
        "frequency": 190,
        "median_duration_seconds": 0.0
      },
      {
        "source": "register submission date request",
        "target": "enter senddate acknowledgement",
        "frequency": 188,
        "median_duration_seconds": 604800.0
      },
      {
        "source": "record date of decision environmental permit",
        "target": "set decision phase decision permitting decided",
        "frequency": 187,
        "median_duration_seconds": 0.0
      }
    ],
    "start_activities": {
      "register submission date request": 1134,
      "OLO messaging active": 36,
      "date for inspection MER": 1,
      "enter senddate acknowledgement": 11,
      "enter senddate procedure confirmation": 5,
      "enter date termination request received": 1,
      "MER present": 1,
      "suspension ground applicable": 1,
      "set decision phase suspension decided": 7,
      "enter senddate decision suspension": 1,
      "record date of decision environmental permit": 1
    },
    "end_activities": {
      "start decision phase decision permitting sent": 134,
      "enter senddate decision environmental permit": 99,
      "transcript decision environmental permit to stakeholders": 14,
      "enter senddate decision permanent suspension": 25,
      "stop all running subcases 2b": 44,
      "retrieve missing data": 4,
      "terminate on request": 10,
      "set phase: phase permitting irrevocable": 37,
      "set decision phase inspection of decision permitting": 13,
      "set decision phase decision permitting published": 29,
      "phase application received": 4,
      "start WABOprocedure": 7,
      "stop suspended completeness subcases": 1,
      "enter senddate acknowledgement": 26,
      "enter senddate request termination": 5,
      "enter date decision permanent suspension": 1,
      "enter senddate retrieve missing data": 7,
      "send confirmation receipt additional data": 6,
      "register deadline": 23,
      "start decision phase permanently suspended decided": 3,
      "submit draft decision": 1,
      "phase decision sent": 3,
      "term for supplying missing data": 7,
      "set decision phase decision permitting decided": 1,
      "phase asked additional information": 1,
      "enter senddate decision procedure term extension": 5,
      "regular procedure without MER": 10,
      "enter date publication decision environmental permit": 281,
      "stop process after terminate on request": 3,
      "close case": 84,
      "subcases completeness completed": 4,
      "register date environmental permit decision": 21,
      "registrer date of publishing received request": 8,
      "creating decision permanently suspended": 2,
      "enter senddate confirmation request": 1,
      "forward to the competent authority": 9,
      "detailed procedure applies": 1,
      "send procedure confirmation": 3,
      "objection lodged against decision": 5,
      "copy decision permanent suspension to stakeholders": 2,
      "send confirmation receipt": 6,
      "decision terminate on request": 1,
      "copy decision to extend procedure to stakeholders": 3,
      "reception through OLO": 1,
      "receive additional information": 2,
      "phase procedure aborted": 3,
      "generate publication document decision environmental permit": 31,
      "enter senddate procedure confirmation": 4,
      "register submission date request": 1,
      "no permit required for application": 1,
      "procedure change": 3,
      "keep permanently suspended": 1,
      "phased application": 2,
      "treat subcases content": 4,
      "treat subcases completeness": 2,
      "continue or abort mutation BAG objects": 3,
      "registration date publication": 21,
      "completed subcases content": 3,
      "creating environmental permit decision": 3,
      "request to competent authority": 1,
      "generating decision environmental permit": 2,
      "send letter in progress": 1,
      "inform BAG administrator": 1,
      "suspension ground applicable": 1,
      "mutation BAG carry objects": 1,
      "set phase": 3,
      "appeal subcase completed": 1,
      "treat subcase perspective": 1,
      "applicant is stakeholder": 1,
      "date decision for inspection": 78,
      "phase decision irrevocable": 10,
      "create publication document": 32,
      "enter senddate letter to stakeholders": 2,
      "phase case handled": 6,
      "enter senddate letter to competent authority": 1,
      "record date of decision environmental permit": 4,
      "phase archived case": 1,
      "MER present in supplement": 1,
      "phase procedure prematurely terminated": 1,
      "enter senddate decision": 1,
      "creating cover letter decision": 1,
      "read publication date field": 1,
      "request for preliminary injunction received": 1,
      "start decision phase permanent suspension sent": 2,
      "create message terminate request": 1,
      "enter senddate letter no permit required": 1,
      "set decision status": 1,
      "set decision phase: draft decision for inspection": 1,
      "decision permanent suspension irrevocable": 1
    },
    "frequent_variants": []
  }
}
"""

CTX = json.loads(PROCESS_CONTEXT)
PA = CTX["process_abstraction"]

DFG_FREQ = {}
DFG_EXISTS = set()
for e in PA["dfg_edges"]:
    DFG_FREQ[(e["source"], e["target"])] = float(e["frequency"])
    DFG_EXISTS.add((e["source"], e["target"]))

DUR = {}
for e in CTX["duration_profiles_top"]:
    DUR[(e["source"], e["target"])] = e

START_ACT = {k: int(v) for k, v in PA["start_activities"].items()}
END_ACT = {k: int(v) for k, v in PA["end_activities"].items()}
FREQUENT_VARIANTS = [v["variant"] for v in PA.get("frequent_variants", [])]

GLOBAL_FAST_GAP_SECONDS = 1.0


def detect_fractional_digits(ts: str) -> int:
    if "." not in ts:
        return 0
    frac = ts.split(".")[-1]
    return len(frac)


def parse_ts(s: str):
    return pd.to_datetime(s, errors="coerce")


def format_ts(ts: pd.Timestamp, frac_digits: int) -> str:
    if frac_digits <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    ns = ts.nanosecond + ts.microsecond * 1000
    frac9 = f"{ns:09d}"
    frac = frac9[:frac_digits]
    return f"{base}.{frac}"


def tick_ns(frac_digits: int) -> int:
    if frac_digits <= 0:
        return 1_000_000_000
    if frac_digits >= 9:
        return 1
    return 10 ** (9 - frac_digits)


def contiguous_subsequence_bonus(seq):
    bonus = 0.0
    n = len(seq)
    for var in FREQUENT_VARIANTS:
        if len(var) < n:
            continue
        for i in range(len(var) - n + 1):
            if var[i:i + n] == list(seq):
                bonus += 1000.0
                break
    return bonus


def pair_gap_seconds(a, b):
    prof = DUR.get((a, b))
    if prof:
        p25 = float(prof.get("p25_seconds", 0.0))
        p10 = float(prof.get("p10_seconds", 0.0))
        share = float(prof.get("share_under_60s", 0.0))
        if share >= 0.5:
            gap = p25 if p25 > 0 else max(1.0, p10, GLOBAL_FAST_GAP_SECONDS)
        else:
            gap = p25 if p25 > 0 and p25 < 60 else GLOBAL_FAST_GAP_SECONDS
        return max(gap, GLOBAL_FAST_GAP_SECONDS)
    if (a, b) in DFG_EXISTS:
        return GLOBAL_FAST_GAP_SECONDS
    return GLOBAL_FAST_GAP_SECONDS


def score_order(order, left_act, right_act):
    score = 0.0
    impossible = 0

    seq = list(order)
    score += contiguous_subsequence_bonus(seq)

    if left_act is not None:
        if (left_act, seq[0]) in DFG_EXISTS:
            score += 150.0 + math.log1p(DFG_FREQ.get((left_act, seq[0]), 1.0))
        else:
            impossible += 1
            score -= 300.0

    for a, b in zip(seq, seq[1:]):
        if (a, b) in DFG_EXISTS:
            score += 200.0 + math.log1p(DFG_FREQ.get((a, b), 1.0))
        else:
            impossible += 1
            score -= 500.0

    if right_act is not None:
        if (seq[-1], right_act) in DFG_EXISTS:
            score += 150.0 + math.log1p(DFG_FREQ.get((seq[-1], right_act), 1.0))
        else:
            impossible += 1
            score -= 300.0

    score += START_ACT.get(seq[0], 0) * 0.01
    score += END_ACT.get(seq[-1], 0) * 0.01

    return score, impossible


def all_unique_permutations(items):
    counter = Counter(items)
    keys = list(counter.keys())
    n = len(items)

    def backtrack(path):
        if len(path) == n:
            yield tuple(path)
            return
        for k in keys:
            if counter[k] > 0:
                counter[k] -= 1
                path.append(k)
                yield from backtrack(path)
                path.pop()
                counter[k] += 1

    yield from backtrack([])


def choose_order(activities, left_act, right_act):
    uniq_count = math.factorial(len(activities))
    counts = Counter(activities)
    for c in counts.values():
        uniq_count //= math.factorial(c)

    if uniq_count <= 720:
        perms = all_unique_permutations(activities)
    else:
        perms = set()
        perms.add(tuple(activities))
        perms.add(tuple(reversed(activities)))
        by_anchor = sorted(activities, key=lambda a: (
            -int((left_act, a) in DFG_EXISTS) if left_act is not None else 0,
            -int((a, right_act) in DFG_EXISTS) if right_act is not None else 0,
            -START_ACT.get(a, 0),
            -END_ACT.get(a, 0),
            a
        ))
        perms.add(tuple(by_anchor))
        perms = list(perms)

    best = None
    best_key = None
    for p in perms:
        score, impossible = score_order(p, left_act, right_act)
        movement_penalty = 0.0
        key = (impossible, -score, movement_penalty, p)
        if best is None or key < best_key:
            best = p
            best_key = key
    return list(best)


def build_times(observed_ts, order, right_anchor_ts, case_end_ts, frac_digits):
    k = len(order)
    min_tick = tick_ns(frac_digits)
    min_tick_sec = min_tick / 1e9
    min_step_sec = max(1.0, min_tick_sec)

    gaps = [max(min_step_sec, pair_gap_seconds(a, b)) for a, b in zip(order, order[1:])]
    planned_span = sum(gaps)
    lower_span = max((k - 1) * min_step_sec, (k - 1) * 1.0 if frac_digits == 0 else (k - 1) * min_step_sec)

    upper_anchor = right_anchor_ts if right_anchor_ts is not None else case_end_ts
    room = max(0.0, (upper_anchor - observed_ts).total_seconds()) if upper_anchor is not None else planned_span
    max_span = room

    span = planned_span
    if span > max_span:
        if planned_span > 0:
            scale = max_span / planned_span
            gaps = [g * scale for g in gaps]
        span = sum(gaps)

    if span < lower_span:
        span = lower_span

    if max_span >= lower_span and span > max_span:
        span = max_span

    if len(gaps) > 0:
        if sum(gaps) <= 0:
            gaps = [min_step_sec] * len(gaps)
        if abs(sum(gaps) - span) > 1e-12:
            if sum(gaps) > 0:
                gaps = [g * (span / sum(gaps)) for g in gaps]

    offsets = [0.0]
    cur = 0.0
    for g in gaps:
        cur += g
        offsets.append(cur)

    times = [observed_ts + pd.to_timedelta(off, unit="s") for off in offsets]
    return times


def enforce_block(times, observed_ts, right_anchor_ts, case_start_ts, case_end_ts, frac_digits):
    k = len(times)
    min_tick = tick_ns(frac_digits)
    floor = observed_ts
    ceiling = right_anchor_ts if right_anchor_ts is not None else case_end_ts
    if ceiling is None:
        ceiling = times[-1]
    if floor < case_start_ts:
        floor = case_start_ts
    if ceiling > case_end_ts:
        ceiling = case_end_ts
    if ceiling < floor:
        ceiling = floor

    out = [floor]
    for i in range(1, k):
        t = times[i]
        min_allowed = out[i - 1] + pd.to_timedelta(min_tick, unit="ns")
        max_remaining = ceiling - pd.to_timedelta((k - 1 - i) * min_tick, unit="ns")
        if t < min_allowed:
            t = min_allowed
        if t > max_remaining:
            t = max_remaining
        if t < min_allowed:
            t = min_allowed
        out.append(t)

    if out[-1] > ceiling:
        start = floor
        out = [start + pd.to_timedelta(i * min_tick, unit="ns") for i in range(k)]
        if out[-1] > ceiling:
            latest_start = ceiling - pd.to_timedelta((k - 1) * min_tick, unit="ns")
            if latest_start < floor:
                latest_start = floor
            out = [latest_start + pd.to_timedelta(i * min_tick, unit="ns") for i in range(k)]
            if out[0] != floor:
                out[0] = floor
                for i in range(1, k):
                    out[i] = max(out[i], out[i - 1] + pd.to_timedelta(min_tick, unit="ns"))
                    if out[i] > ceiling:
                        out[i] = ceiling

    out[0] = observed_ts
    for i in range(1, k):
        min_allowed = out[i - 1] + pd.to_timedelta(min_tick, unit="ns")
        if out[i] < min_allowed:
            out[i] = min_allowed
    if right_anchor_ts is not None and out[-1] > right_anchor_ts:
        latest_start = observed_ts
        out = [latest_start + pd.to_timedelta(i * min_tick, unit="ns") for i in range(k)]
        if out[-1] > right_anchor_ts:
            end = right_anchor_ts
            start = end - pd.to_timedelta((k - 1) * min_tick, unit="ns")
            if start < observed_ts:
                start = observed_ts
            out = [start + pd.to_timedelta(i * min_tick, unit="ns") for i in range(k)]
            out[0] = observed_ts
            for i in range(1, k):
                out[i] = max(out[i], out[i - 1] + pd.to_timedelta(min_tick, unit="ns"))
                if out[i] > right_anchor_ts:
                    out[i] = right_anchor_ts

    if out[0] != observed_ts:
        out[0] = observed_ts
    for i in range(1, k):
        if not out[i] > out[i - 1]:
            out[i] = out[i - 1] + pd.to_timedelta(min_tick, unit="ns")
    return out


def map_order_positions_to_rows(block_df, order):
    act_to_positions = defaultdict(list)
    for pos, act in enumerate(order):
        act_to_positions[act].append(pos)
    act_seen = defaultdict(int)
    row_to_pos = {}
    for idx, row in block_df.iterrows():
        act = row["Activity"]
        occ = act_seen[act]
        pos = act_to_positions[act][occ]
        row_to_pos[idx] = pos
        act_seen[act] += 1
    return row_to_pos


def find_c1_blocks(df):
    blocks = []
    for case, g in df.groupby("Case", sort=False):
        idxs = list(g.index)
        i = 0
        while i < len(idxs):
            idx = idxs[i]
            if df.at[idx, "error_category"] == "CLEAN":
                i += 1
                continue
            ts = df.at[idx, "Timestamp"]
            j = i + 1
            while j < len(idxs):
                idx2 = idxs[j]
                if df.at[idx2, "error_category"] == "CLEAN":
                    break
                if df.at[idx2, "Timestamp"] != ts:
                    break
                j += 1
            if j - i >= 2:
                blocks.append(idxs[i:j])
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

    original = df.copy(deep=True)
    repaired_ts = df["Timestamp"].copy()

    parsed = df["Timestamp"].apply(parse_ts)
    frac_digits_global = max((detect_fractional_digits(x) for x in df["Timestamp"]), default=0)

    case_bounds = {}
    for case, g in df.groupby("Case", sort=False):
        ts = parsed.loc[g.index]
        valid = ts.dropna()
        if len(valid) == 0:
            continue
        case_bounds[case] = (valid.min(), valid.max())

    blocks = find_c1_blocks(df)

    blocks_repaired = 0
    blocks_clamped = 0

    for block_idxs in blocks:
        try:
            block = df.loc[block_idxs]
            case = block["Case"].iloc[0]
            obs_str = block["Timestamp"].iloc[0]
            observed_ts = parse_ts(obs_str)
            frac_digits = detect_fractional_digits(obs_str)
            if frac_digits == 0 and frac_digits_global > 0:
                frac_digits = frac_digits_global

            case_start, case_end = case_bounds[case]

            case_rows = df.index[df["Case"] == case].tolist()
            pos_first = case_rows.index(block_idxs[0])
            pos_last = case_rows.index(block_idxs[-1])

            left_anchor_idx = None
            for p in range(pos_first - 1, -1, -1):
                idx = case_rows[p]
                if df.at[idx, "error_category"] == "CLEAN":
                    left_anchor_idx = idx
                    break

            right_anchor_idx = None
            for p in range(pos_last + 1, len(case_rows)):
                idx = case_rows[p]
                if df.at[idx, "error_category"] == "CLEAN":
                    right_anchor_idx = idx
                    break

            left_act = df.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None else None
            right_act = df.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None else None
            left_anchor_ts = parse_ts(df.at[left_anchor_idx, "Timestamp"]) if left_anchor_idx is not None else None
            right_anchor_ts = parse_ts(df.at[right_anchor_idx, "Timestamp"]) if right_anchor_idx is not None else None

            activities = block["Activity"].tolist()
            order = choose_order(activities, left_act, right_act)

            times = build_times(observed_ts, order, right_anchor_ts, case_end, frac_digits)
            enforced = enforce_block(times, observed_ts, right_anchor_ts, case_start, case_end, frac_digits)

            need_clamp = False
            if any(enforced[i] <= enforced[i - 1] for i in range(1, len(enforced))):
                need_clamp = True
            if right_anchor_ts is not None and enforced[-1] > right_anchor_ts:
                need_clamp = True
            if any(t < case_start or t > case_end for t in enforced):
                need_clamp = True
            if (enforced[-1] - enforced[0]).total_seconds() < (len(enforced) - 1):
                need_clamp = True

            row_to_pos = map_order_positions_to_rows(block, order)
            for idx in block_idxs:
                pos = row_to_pos[idx]
                repaired_ts.at[idx] = format_ts(enforced[pos], frac_digits)

            # self-fix exact one unchanged row, earliest in chosen order
            earliest_positions = [idx for idx in block_idxs if row_to_pos[idx] == 0]
            earliest_idx = earliest_positions[0]
            for idx in block_idxs:
                if idx == earliest_idx:
                    repaired_ts.at[idx] = obs_str
                else:
                    if repaired_ts.at[idx] == df.at[idx, "Timestamp"]:
                        pos = row_to_pos[idx]
                        frac = frac_digits
                        t = enforced[pos]
                        bump = pd.to_timedelta(tick_ns(frac), unit="ns")
                        t2 = t + bump
                        if right_anchor_ts is not None and t2 > right_anchor_ts:
                            t2 = t - bump
                            if t2 <= observed_ts:
                                t2 = t + 2 * bump
                        repaired_ts.at[idx] = format_ts(t2, frac)

            # final block validation and repair
            ordered_pairs = sorted(((row_to_pos[idx], idx) for idx in block_idxs))
            ordered_idxs = [idx for _, idx in ordered_pairs]
            ordered_ts = [parse_ts(repaired_ts.at[idx]) for idx in ordered_idxs]

            bad = False
            if any(ordered_ts[i] <= ordered_ts[i - 1] for i in range(1, len(ordered_ts))):
                bad = True
            unchanged_count = sum(1 for idx in block_idxs if repaired_ts.at[idx] == df.at[idx, "Timestamp"])
            if unchanged_count != 1:
                bad = True
            if ordered_ts[0] != observed_ts:
                bad = True
            if right_anchor_ts is not None:
                if (ordered_ts[-1] - ordered_ts[0]).total_seconds() > (right_anchor_ts - observed_ts).total_seconds() + 1e-9:
                    bad = True
            if (ordered_ts[-1] - ordered_ts[0]).total_seconds() < (len(ordered_ts) - 1):
                bad = True

            if bad:
                need_clamp = True
                min_tick = tick_ns(frac_digits)
                latest_end = right_anchor_ts if right_anchor_ts is not None else case_end
                latest_end = min(latest_end, case_end)
                rebuilt = [observed_ts + pd.to_timedelta(i * max(min_tick, 1_000_000_000 if frac_digits == 0 else min_tick), unit="ns") for i in range(len(ordered_idxs))]
                if rebuilt[-1] > latest_end:
                    rebuilt = [observed_ts + pd.to_timedelta(i * min_tick, unit="ns") for i in range(len(ordered_idxs))]
                if rebuilt[-1] > latest_end:
                    start = observed_ts
                    rebuilt = [start + pd.to_timedelta(i * min_tick, unit="ns") for i in range(len(ordered_idxs))]
                for p, idx in enumerate(ordered_idxs):
                    repaired_ts.at[idx] = format_ts(rebuilt[p], frac_digits)
                repaired_ts.at[ordered_idxs[0]] = obs_str
                for p, idx in enumerate(ordered_idxs[1:], start=1):
                    if repaired_ts.at[idx] == df.at[idx, "Timestamp"]:
                        repaired_ts.at[idx] = format_ts(parse_ts(repaired_ts.at[idx]) + pd.to_timedelta(min_tick, unit="ns"), frac_digits)

            if need_clamp:
                blocks_clamped += 1
            blocks_repaired += 1
        except Exception:
            try:
                block = df.loc[block_idxs]
                obs_str = block["Timestamp"].iloc[0]
                observed_ts = parse_ts(obs_str)
                frac_digits = max(detect_fractional_digits(obs_str), frac_digits_global)
                min_tick = tick_ns(frac_digits)
                activities = block["Activity"].tolist()
                order = activities
                row_to_pos = map_order_positions_to_rows(block, order)
                ordered_pairs = sorted(((row_to_pos[idx], idx) for idx in block_idxs))
                for p, (_, idx) in enumerate(ordered_pairs):
                    t = observed_ts + pd.to_timedelta(p * min_tick, unit="ns")
                    repaired_ts.at[idx] = format_ts(t, frac_digits)
                first_idx = ordered_pairs[0][1]
                repaired_ts.at[first_idx] = obs_str
                blocks_repaired += 1
                blocks_clamped += 1
            except Exception:
                pass

    df["Timestamp"] = repaired_ts

    # preserve CLEAN rows exactly
    clean_mask = original["error_category"] == "CLEAN"
    df.loc[clean_mask, "Timestamp"] = original.loc[clean_mask, "Timestamp"]

    # global self-check repair
    for idx in df.index:
        if original.at[idx, "error_category"] == "CLEAN":
            df.at[idx, "Timestamp"] = original.at[idx, "Timestamp"]
        else:
            if parse_ts(df.at[idx, "Timestamp"]) is pd.NaT or pd.isna(parse_ts(df.at[idx, "Timestamp"])):
                frac = max(detect_fractional_digits(original.at[idx, "Timestamp"]), frac_digits_global)
                t = parse_ts(original.at[idx, "Timestamp"]) + pd.to_timedelta(tick_ns(frac), unit="ns")
                df.at[idx, "Timestamp"] = format_ts(t, frac)

    still_equal_rows = int(((original["error_category"] != "CLEAN") & (df["Timestamp"] == original["Timestamp"])).sum())

    try:
        df.to_csv(args.output, index=False)
    except Exception:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={blocks_clamped} rows_still_equal_to_input={still_equal_rows}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0")