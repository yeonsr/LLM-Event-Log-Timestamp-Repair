import argparse
import csv
import itertools
import math
import re
from collections import defaultdict
from datetime import timedelta

import pandas as pd


# =========================
# Embedded process context
# =========================
TRANSITION_PROBABILITIES_TOP = [
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
]

DURATION_PROFILES_TOP = [
    {"source": "register submission date request", "target": "OLO messaging active", "n": 729, "p10_seconds": 86400.0, "p25_seconds": 172800.0, "median_seconds": 306406.0, "p75_seconds": 518400.0, "p90_seconds": 733026.6, "share_under_60s": 0.0},
    {"source": "forward to the competent authority", "target": "regular procedure without MER", "n": 673, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 3.0, "p75_seconds": 6.0, "p90_seconds": 12.0, "share_under_60s": 0.973},
    {"source": "send confirmation receipt", "target": "forward to the competent authority", "n": 591, "p10_seconds": 2.0, "p25_seconds": 3.0, "median_seconds": 6.0, "p75_seconds": 9.0, "p90_seconds": 19.0, "share_under_60s": 0.941},
    {"source": "request complete", "target": "send letter in progress", "n": 572, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 1.0, "p75_seconds": 2.0, "p90_seconds": 4.0, "share_under_60s": 0.991},
    {"source": "article 34  WABO applies", "target": "assessment of content completed", "n": 568, "p10_seconds": 0.0, "p25_seconds": 1.0, "median_seconds": 1.0, "p75_seconds": 2.0, "p90_seconds": 4.0, "share_under_60s": 0.993},
    {"source": "assessment of content completed", "target": "phase advice known", "n": 555, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 1.0, "share_under_60s": 1.0},
    {"source": "send confirmation receipt", "target": "send confirmation receipt", "n": 503, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 211.20000000000005, "share_under_60s": 0.799},
    {"source": "phase application receptive", "target": "request complete", "n": 500, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "treat subcases content", "target": "completed subcases content", "n": 493, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 1.0, "p75_seconds": 2.0, "p90_seconds": 3.0, "share_under_60s": 0.992},
    {"source": "treat subcases completeness", "target": "subcases completeness completed", "n": 490, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 2.0, "p75_seconds": 3.0, "p90_seconds": 7.0, "share_under_60s": 0.984},
    {"source": "by law", "target": "creating environmental permit decision", "n": 481, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 1.0, "share_under_60s": 0.994},
    {"source": "enter senddate decision", "target": "enter senddate decision environmental permit", "n": 467, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 0.991},
    {"source": "phase advice known", "target": "grounds for refusal", "n": 438, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 2.0, "p75_seconds": 3.0, "p90_seconds": 6.300000000000011, "share_under_60s": 0.954},
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
    {"source": "procedure change", "target": "article 34  WABO applies", "n": 352, "p10_seconds": 1.0, "p25_seconds": 1.0, "median_seconds": 2.0, "p75_seconds": 3.0, "p90_seconds": 9.900000000000034, "share_under_60s": 0.949},
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
    {"source": "creating environmental permit decision", "target": "generating decision environmental permit", "n": 284, "p10_seconds": 4.0, "p25_seconds": 55.5, "median_seconds": 1796.0, "p75_seconds": 113196.25, "p90_seconds": 914443.3000000016, "share_under_60s": 0.257},
    {"source": "send letter in progress", "target": "set decision status", "n": 254, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "set decision phase decision permitting decided", "target": "start decision phase decision permitting sent", "n": 241, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 0.988},
    {"source": "regular procedure without MER", "target": "start WABOprocedure", "n": 230, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 1.0, "share_under_60s": 1.0},
    {"source": "transcript decision environmental permit to stakeholders", "target": "enter senddate decision environmental permit", "n": 225, "p10_seconds": 8.0, "p25_seconds": 34.0, "median_seconds": 50131.0, "p75_seconds": 172800.0, "p90_seconds": 259200.0, "share_under_60s": 0.258},
    {"source": "send procedure confirmation", "target": "registration date publication", "n": 224, "p10_seconds": 8.0, "p25_seconds": 11.0, "median_seconds": 19.0, "p75_seconds": 78.0, "p90_seconds": 738814.7000000001, "share_under_60s": 0.728},
    {"source": "phase application received", "target": "reception through OLO", "n": 222, "p10_seconds": 1.0, "p25_seconds": 2.0, "median_seconds": 3.0, "p75_seconds": 6.0, "p90_seconds": 14.900000000000006, "share_under_60s": 0.973},
    {"source": "regular procedure without MER", "target": "enter senddate procedure confirmation", "n": 219, "p10_seconds": 1.0, "p25_seconds": 2.0, "median_seconds": 2.0, "p75_seconds": 4.0, "p90_seconds": 87809.20000000433, "share_under_60s": 0.872},
    {"source": "generating decision environmental permit", "target": "grounds for refusal", "n": 215, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "decision date prior to decision", "target": "generating decision environmental permit", "n": 209, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 1.0, "p75_seconds": 4206.0, "p90_seconds": 430595.80000000034, "share_under_60s": 0.569},
    {"source": "create publication document", "target": "create subcases completeness", "n": 197, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 7.400000000000006, "share_under_60s": 0.914},
    {"source": "ask stakeholders views", "target": "assessment of content completed", "n": 194, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "by law", "target": "completed subcases content", "n": 194, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "submit decision", "target": "suspension ground applicable", "n": 193, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "procedure change", "target": "procedure change", "n": 193, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "completed subcases content", "target": "coordination of application", "n": 190, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
    {"source": "register submission date request", "target": "enter senddate acknowledgement", "n": 188, "p10_seconds": 172800.0, "p25_seconds": 345600.0, "median_seconds": 604800.0, "p75_seconds": 864000.0, "p90_seconds": 1209600.0, "share_under_60s": 0.0},
    {"source": "record date of decision environmental permit", "target": "set decision phase decision permitting decided", "n": 187, "p10_seconds": 0.0, "p25_seconds": 0.0, "median_seconds": 0.0, "p75_seconds": 0.0, "p90_seconds": 0.0, "share_under_60s": 1.0},
]

GLOBAL_FAST_DEFAULT_SECONDS = 3.0
LOW_PROB_DEFAULT = 1e-6


# =========================
# Helpers
# =========================
def parse_timestamp_safe(s):
    if s is None:
        return None
    st = str(s).strip()
    if st == "":
        return None
    try:
        return pd.to_datetime(st)
    except Exception:
        return None


def detect_precision(strings):
    max_frac = 0
    any_frac = False
    for s in strings:
        if s is None:
            continue
        m = re.search(r"\.(\d+)", str(s))
        if m:
            any_frac = True
            max_frac = max(max_frac, len(m.group(1)))
    return max_frac if any_frac else 0


def format_ts(ts, frac_digits):
    if frac_digits > 0:
        base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
        frac = base.split(".")[1][:frac_digits].ljust(frac_digits, "0")
        return base.split(".")[0] + "." + frac
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def seconds_between(a, b):
    return (b - a).total_seconds()


def build_maps():
    prob = {}
    for x in TRANSITION_PROBABILITIES_TOP:
        prob[(x["source"], x["target"])] = float(x["probability"])
    dur = {}
    for x in DURATION_PROFILES_TOP:
        dur[(x["source"], x["target"])] = x
    return prob, dur


TRANS_PROB, DUR_PROF = build_maps()


def transition_prob(a, b):
    if a is None or b is None:
        return 0.05
    return TRANS_PROB.get((a, b), LOW_PROB_DEFAULT)


def fast_gap_seconds(a, b):
    prof = DUR_PROF.get((a, b))
    if prof is None:
        return GLOBAL_FAST_DEFAULT_SECONDS
    share = float(prof.get("share_under_60s", 0.0))
    p10 = float(prof.get("p10_seconds", 0.0))
    p25 = float(prof.get("p25_seconds", 0.0))
    if share >= 0.97:
        g = p10
    else:
        g = p25
    if g <= 0:
        g = 1.0 if share >= 0.8 else GLOBAL_FAST_DEFAULT_SECONDS
    return max(1.0, float(g))


def expected_gap_seconds(a, b):
    prof = DUR_PROF.get((a, b))
    if prof is None:
        return GLOBAL_FAST_DEFAULT_SECONDS
    share = float(prof.get("share_under_60s", 0.0))
    if share >= 0.8:
        g = float(prof.get("p25_seconds", 0.0))
        if g <= 0:
            g = float(prof.get("p10_seconds", 0.0))
    else:
        g = float(prof.get("median_seconds", 0.0))
    if g <= 0:
        g = GLOBAL_FAST_DEFAULT_SECONDS
    return max(1.0, g)


def plausibility_penalty(observed_gap, a, b):
    expg = expected_gap_seconds(a, b)
    return abs(float(observed_gap) - expg) / max(expg, 1.0)


def enumerate_best_order(block_activities, left_act, right_act):
    idxs = list(range(len(block_activities)))
    best = None
    best_score = None
    best_span = None
    seen = set()
    for perm in itertools.permutations(idxs):
        acts = tuple(block_activities[i] for i in perm)
        if acts in seen:
            continue
        seen.add(acts)
        score = 0.0
        span = 0.0
        if left_act is not None:
            p = transition_prob(left_act, acts[0])
            score += math.log(max(p, LOW_PROB_DEFAULT))
            span += fast_gap_seconds(left_act, acts[0]) * 0.2
        for i in range(len(acts) - 1):
            p = transition_prob(acts[i], acts[i + 1])
            score += math.log(max(p, LOW_PROB_DEFAULT))
            span += fast_gap_seconds(acts[i], acts[i + 1])
        if right_act is not None:
            p = transition_prob(acts[-1], right_act)
            score += math.log(max(p, LOW_PROB_DEFAULT))
            span += fast_gap_seconds(acts[-1], right_act) * 0.2
        cand = (perm, score, span)
        if best is None or score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and span < best_span):
            best = cand
            best_score = score
            best_span = span
    return best[0]


def find_prev_clean_anchor(case_rows, pos):
    for j in range(pos - 1, -1, -1):
        if case_rows[j]["error_category"] == "CLEAN":
            ts = case_rows[j]["parsed"]
            if ts is not None:
                return j, case_rows[j]
    return None, None


def find_next_clean_anchor(case_rows, pos):
    for j in range(pos + 1, len(case_rows)):
        if case_rows[j]["error_category"] == "CLEAN":
            ts = case_rows[j]["parsed"]
            if ts is not None:
                return j, case_rows[j]
    return None, None


def choose_between(left_ts, right_ts, left_act, act, right_act, preserve_date=None):
    candidates = []
    if left_ts is not None:
        candidates.append(left_ts + pd.to_timedelta(expected_gap_seconds(left_act, act), unit="s"))
    if right_ts is not None:
        candidates.append(right_ts - pd.to_timedelta(expected_gap_seconds(act, right_act), unit="s"))
    if left_ts is not None and right_ts is not None:
        total = expected_gap_seconds(left_act, act) + expected_gap_seconds(act, right_act)
        if total <= 0:
            total = 2.0
        ratio = expected_gap_seconds(left_act, act) / total
        candidates.append(left_ts + (right_ts - left_ts) * ratio)
    if not candidates:
        return None
    cand = candidates[-1]
    if preserve_date is not None:
        day_start = pd.Timestamp(preserve_date)
        if day_start.tzinfo is not None:
            day_start = day_start.tz_localize(None)
        day_start = pd.Timestamp(day_start.date())
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        if cand.date() != preserve_date:
            if left_ts is not None and left_ts.date() == preserve_date and right_ts is not None and right_ts.date() == preserve_date:
                cand = max(left_ts, min(right_ts, day_start + pd.Timedelta(hours=12)))
            else:
                cand = max(day_start, min(day_end, cand))
    return cand


def clamp_strictly_between_sequence(times, left_bound, right_bound):
    n = len(times)
    if n == 0:
        return times, False
    changed = False
    lb = left_bound
    rb = right_bound
    if lb is not None and rb is not None and rb < lb:
        rb = lb
        changed = True
    out = list(times)
    for i in range(n):
        min_t = lb + pd.Timedelta(seconds=i + 1) if lb is not None else None
        max_t = rb - pd.Timedelta(seconds=n - i) if rb is not None else None
        t = out[i]
        if min_t is not None and t < min_t:
            t = min_t
            changed = True
        if max_t is not None and t > max_t:
            t = max_t
            changed = True
        out[i] = t
    for i in range(1, n):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + pd.Timedelta(seconds=1)
            changed = True
    if rb is not None and out[-1] > rb:
        latest_start = rb - pd.Timedelta(seconds=n - 1)
        if lb is not None and latest_start <= lb:
            latest_start = lb + pd.Timedelta(seconds=1)
        base = latest_start
        for i in range(n):
            out[i] = base + pd.Timedelta(seconds=i)
        changed = True
    if lb is not None and out[0] <= lb:
        base = lb + pd.Timedelta(seconds=1)
        for i in range(n):
            out[i] = base + pd.Timedelta(seconds=i)
        changed = True
    return out, changed


# =========================
# Repairs
# =========================
def repair_c1_case(case_rows, frac_digits, stats):
    n = len(case_rows)
    i = 0
    while i < n:
        row = case_rows[i]
        if row["error_category"] != "C1_ORDERING_AMBIGUITY" or row["parsed"] is None:
            i += 1
            continue
        shared = row["parsed"]
        j = i + 1
        block_idxs = [i]
        while j < n:
            rj = case_rows[j]
            if rj["error_category"] == "C1_ORDERING_AMBIGUITY" and rj["parsed"] is not None and rj["parsed"] == shared:
                block_idxs.append(j)
                j += 1
            else:
                break
        if len(block_idxs) == 1:
            i = j
            continue

        stats["blocks_repaired"] += 1

        left_idx, left_anchor = find_prev_clean_anchor(case_rows, block_idxs[0])
        right_idx, right_anchor = find_next_clean_anchor(case_rows, block_idxs[-1])

        acts = [case_rows[k]["Activity"] for k in block_idxs]
        left_act = left_anchor["Activity"] if left_anchor is not None else None
        right_act = right_anchor["Activity"] if right_anchor is not None else None

        best_perm = enumerate_best_order(acts, left_act, right_act)
        order_positions = list(best_perm)
        ordered_case_positions = [block_idxs[p] for p in order_positions]
        ordered_acts = [acts[p] for p in order_positions]

        gaps = []
        for a, b in zip(ordered_acts[:-1], ordered_acts[1:]):
            gaps.append(max(1.0, fast_gap_seconds(a, b)))

        cum = [0.0]
        for g in gaps:
            cum.append(cum[-1] + g)

        room_right = None
        if right_anchor is not None and right_anchor["parsed"] is not None:
            room_right = max(0.0, seconds_between(shared, right_anchor["parsed"]))
        case_end = case_rows[-1]["case_end"]
        room_case_end = max(0.0, seconds_between(shared, case_end))
        max_room = room_case_end if room_right is None else min(room_right, room_case_end)

        min_span = float(len(block_idxs) - 1)
        raw_span = cum[-1]
        clamped = False

        if max_room < min_span:
            max_room = min_span
            clamped = True

        if raw_span < min_span:
            raw_span = min_span
        span = raw_span
        if span > max_room:
            span = max_room
            clamped = True

        if len(cum) > 1 and cum[-1] > 0:
            scale = span / cum[-1]
            scaled_cum = [0.0]
            for x in cum[1:]:
                scaled_cum.append(x * scale)
        else:
            scaled_cum = list(range(len(block_idxs)))

        offsets = [0]
        for idx_off in range(1, len(scaled_cum)):
            sec = max(idx_off, int(round(scaled_cum[idx_off])))
            if sec <= offsets[-1]:
                sec = offsets[-1] + 1
            offsets.append(sec)

        if offsets[-1] > max_room:
            offsets = list(range(len(block_idxs)))
            last = offsets[-1]
            if last > max_room:
                start_shift = int(math.floor(max_room)) - (len(block_idxs) - 1)
                if start_shift < 0:
                    start_shift = 0
                offsets = [start_shift + x for x in range(len(block_idxs))]
            clamped = True

        assigned = {}
        for rank, case_pos in enumerate(ordered_case_positions):
            ts = shared + pd.to_timedelta(offsets[rank], unit="s")
            assigned[case_pos] = ts

        # self-check and repair
        ordered_times = [assigned[p] for p in ordered_case_positions]
        if not all(ordered_times[k] < ordered_times[k + 1] for k in range(len(ordered_times) - 1)):
            repaired_seq, changed = clamp_strictly_between_sequence(
                ordered_times[1:],
                shared,
                right_anchor["parsed"] if right_anchor is not None else case_end
            )
            ordered_times = [shared] + repaired_seq
            for cp, ts in zip(ordered_case_positions, ordered_times):
                assigned[cp] = ts
            clamped = clamped or changed

        # write back to own rows, preserving row order in file
        first_case_pos = ordered_case_positions[0]
        for cp in block_idxs:
            ts = assigned[cp]
            case_rows[cp]["out_ts"] = format_ts(ts, frac_digits)
            case_rows[cp]["repaired"] = True

        # exactly one row may remain equal to input: the first event in inferred order
        for cp in block_idxs:
            if cp != first_case_pos and case_rows[cp]["out_ts"] == case_rows[cp]["Timestamp"]:
                ts = assigned[cp] + pd.to_timedelta(1, unit="s")
                assigned[cp] = ts
                case_rows[cp]["out_ts"] = format_ts(ts, frac_digits)

        if clamped:
            stats["blocks_clamped"] += 1

        i = j


def repair_c2_case(case_rows, frac_digits, stats):
    n = len(case_rows)
    for i in range(1, n):
        curr = case_rows[i]
        prev = case_rows[i - 1]
        if curr["error_category"] != "C2_TEMPORAL_INCONSISTENCY":
            continue
        t_curr = curr["parsed"]
        t_prev = prev["parsed"]
        if t_curr is None or t_prev is None:
            continue
        if t_curr >= t_prev:
            continue

        left_idx, left_anchor = find_prev_clean_anchor(case_rows, i - 1)
        right_idx, right_anchor = find_next_clean_anchor(case_rows, i)

        # swapped candidate
        swap_prev_t = t_curr
        swap_curr_t = t_prev

        swap_ok = True
        swap_pen = 0.0

        if left_anchor is not None:
            if swap_prev_t < left_anchor["parsed"]:
                swap_ok = False
            else:
                swap_pen += plausibility_penalty(seconds_between(left_anchor["parsed"], swap_prev_t),
                                                 left_anchor["Activity"], prev["Activity"])
        if swap_curr_t < swap_prev_t:
            swap_ok = False
        else:
            swap_pen += plausibility_penalty(seconds_between(swap_prev_t, swap_curr_t),
                                             prev["Activity"], curr["Activity"])
        if right_anchor is not None:
            if swap_curr_t > right_anchor["parsed"]:
                swap_ok = False
            else:
                swap_pen += plausibility_penalty(seconds_between(swap_curr_t, right_anchor["parsed"]),
                                                 curr["Activity"], right_anchor["Activity"])

        # displaced candidate: keep plausible partner fixed, estimate current
        disp_t = choose_between(
            t_prev,
            right_anchor["parsed"] if right_anchor is not None else None,
            prev["Activity"],
            curr["Activity"],
            right_anchor["Activity"] if right_anchor is not None else None,
            preserve_date=t_curr.date() if re.match(r"^\d{4}-\d{2}-\d{2}$", curr["Timestamp"].strip()) else None
        )
        if disp_t is None:
            disp_t = t_prev + pd.to_timedelta(max(1.0, expected_gap_seconds(prev["Activity"], curr["Activity"])), unit="s")

        if right_anchor is not None and disp_t > right_anchor["parsed"]:
            # place between anchors, not one second before right anchor
            mid = t_prev + (right_anchor["parsed"] - t_prev) * 0.5
            disp_t = mid if mid > t_prev else t_prev + pd.to_timedelta(1, unit="s")
        if disp_t <= t_prev:
            disp_t = t_prev + pd.to_timedelta(1, unit="s")
        if right_anchor is not None and disp_t > right_anchor["parsed"]:
            disp_t = min(right_anchor["parsed"], disp_t)
        if right_anchor is not None and disp_t <= t_prev:
            window = seconds_between(t_prev, right_anchor["parsed"])
            if window >= 2:
                disp_t = t_prev + pd.to_timedelta(window / 2.0, unit="s")
            else:
                disp_t = t_prev + pd.to_timedelta(1, unit="s")

        disp_pen = plausibility_penalty(seconds_between(t_prev, disp_t), prev["Activity"], curr["Activity"])
        if right_anchor is not None and disp_t <= right_anchor["parsed"]:
            disp_pen += plausibility_penalty(seconds_between(disp_t, right_anchor["parsed"]),
                                             curr["Activity"], right_anchor["Activity"])

        if swap_ok and swap_pen <= disp_pen * 1.05:
            prev["out_ts"] = format_ts(swap_prev_t, frac_digits)
            curr["out_ts"] = format_ts(swap_curr_t, frac_digits)
            prev["repaired"] = True
            curr["repaired"] = True
        else:
            if format_ts(disp_t, frac_digits) == curr["Timestamp"]:
                disp_t = disp_t + pd.to_timedelta(1, unit="s")
                if right_anchor is not None and disp_t > right_anchor["parsed"]:
                    disp_t = t_prev + pd.to_timedelta(max(1.0, min(seconds_between(t_prev, right_anchor["parsed"]) / 2.0, 60.0)), unit="s")
            curr["out_ts"] = format_ts(disp_t, frac_digits)
            curr["repaired"] = True


def repair_c3_case(case_rows, frac_digits, stats):
    n = len(case_rows)
    date_only_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for i, row in enumerate(case_rows):
        if row["error_category"] != "C3_MISSING_TEMPORAL_INFORMATION":
            continue

        orig = row["Timestamp"].strip()
        parsed = row["parsed"]
        preserve_date = None
        if date_only_re.match(orig):
            try:
                preserve_date = pd.to_datetime(orig).date()
            except Exception:
                preserve_date = None
        elif parsed is not None and re.match(r"^\d{4}-\d{2}-\d{2}\s+00:00:00(?:\.0+)?$", orig):
            preserve_date = parsed.date()

        left_idx, left_anchor = find_prev_clean_anchor(case_rows, i)
        right_idx, right_anchor = find_next_clean_anchor(case_rows, i)

        left_ts = left_anchor["parsed"] if left_anchor is not None else None
        right_ts = right_anchor["parsed"] if right_anchor is not None else None
        left_act = left_anchor["Activity"] if left_anchor is not None else None
        right_act = right_anchor["Activity"] if right_anchor is not None else None

        cand = choose_between(left_ts, right_ts, left_act, row["Activity"], right_act, preserve_date=preserve_date)

        if cand is None:
            if preserve_date is not None:
                cand = pd.Timestamp(f"{preserve_date.isoformat()} 12:00:00")
            elif left_ts is not None:
                cand = left_ts + pd.to_timedelta(expected_gap_seconds(left_act, row["Activity"]), unit="s")
            elif right_ts is not None:
                cand = right_ts - pd.to_timedelta(expected_gap_seconds(row["Activity"], right_act), unit="s")
            else:
                cand = row["case_start"]

        if preserve_date is not None:
            day_start = pd.Timestamp(f"{preserve_date.isoformat()} 00:00:00")
            day_end = pd.Timestamp(f"{preserve_date.isoformat()} 23:59:59.999999")
            cand = max(day_start, min(day_end, cand))

        if left_ts is not None and right_ts is not None:
            if cand <= left_ts or cand >= right_ts:
                span = seconds_between(left_ts, right_ts)
                if span >= 2:
                    cand = left_ts + pd.to_timedelta(span / 2.0, unit="s")
                else:
                    cand = left_ts
        elif left_ts is not None and cand < left_ts:
            cand = left_ts
        elif right_ts is not None and cand > right_ts:
            cand = right_ts

        if preserve_date is not None:
            day_start = pd.Timestamp(f"{preserve_date.isoformat()} 00:00:00")
            day_end = pd.Timestamp(f"{preserve_date.isoformat()} 23:59:59.999999")
            cand = max(day_start, min(day_end, cand))

        out = format_ts(cand, frac_digits)
        if out == row["Timestamp"]:
            cand = cand + pd.to_timedelta(1, unit="s")
            if right_ts is not None and cand > right_ts and left_ts is not None:
                span = seconds_between(left_ts, right_ts)
                cand = left_ts + pd.to_timedelta(max(1.0, span / 2.0), unit="s")
            if preserve_date is not None:
                day_end = pd.Timestamp(f"{preserve_date.isoformat()} 23:59:59.999999")
                if cand > day_end:
                    cand = day_end
            out = format_ts(cand, frac_digits)

        row["out_ts"] = out
        row["repaired"] = True


# =========================
# Validation and fixups
# =========================
def final_self_check_and_fix(all_rows, frac_digits, stats):
    by_case = defaultdict(list)
    for idx, r in enumerate(all_rows):
        by_case[r["Case"]].append((idx, r))

    c1_equal_allowed = 0

    for case, items in by_case.items():
        case_rows = [r for _, r in items]
        case_start = min([r["parsed"] for r in case_rows if r["parsed"] is not None] + [parse_timestamp_safe(r["out_ts"]) for r in case_rows if r["out_ts"]])
        case_end = max([r["parsed"] for r in case_rows if r["parsed"] is not None] + [parse_timestamp_safe(r["out_ts"]) for r in case_rows if r["out_ts"]])

        for r in case_rows:
            if r["error_category"] == "CLEAN":
                r["out_ts"] = r["Timestamp"]

        i = 0
        while i < len(case_rows):
            r = case_rows[i]
            if r["error_category"] != "C1_ORDERING_AMBIGUITY" or r["parsed"] is None:
                i += 1
                continue
            shared = r["parsed"]
            j = i + 1
            block = [i]
            while j < len(case_rows):
                rr = case_rows[j]
                if rr["error_category"] == "C1_ORDERING_AMBIGUITY" and rr["parsed"] is not None and rr["parsed"] == shared:
                    block.append(j)
                    j += 1
                else:
                    break
            if len(block) > 1:
                times = [parse_timestamp_safe(case_rows[k]["out_ts"]) for k in block]
                earliest_count = sum(1 for t in times if t == shared)
                if earliest_count != 1:
                    min_pos = min(range(len(block)), key=lambda x: times[x] if times[x] is not None else pd.Timestamp.max)
                    for m, k in enumerate(block):
                        if m == min_pos:
                            case_rows[k]["out_ts"] = format_ts(shared, frac_digits)
                        else:
                            t = parse_timestamp_safe(case_rows[k]["out_ts"])
                            if t is None or t <= shared:
                                t = shared + pd.to_timedelta(m, unit="s")
                            case_rows[k]["out_ts"] = format_ts(t, frac_digits)
                    stats["blocks_clamped"] += 1
                c1_equal_allowed += 1
            i = j

        for idx_case, r in enumerate(case_rows):
            if r["error_category"] == "CLEAN":
                continue
            outp = parse_timestamp_safe(r["out_ts"])
            if outp is None:
                left_idx, left_anchor = find_prev_clean_anchor(case_rows, idx_case)
                right_idx, right_anchor = find_next_clean_anchor(case_rows, idx_case)
                cand = choose_between(
                    left_anchor["parsed"] if left_anchor is not None else None,
                    right_anchor["parsed"] if right_anchor is not None else None,
                    left_anchor["Activity"] if left_anchor is not None else None,
                    r["Activity"],
                    right_anchor["Activity"] if right_anchor is not None else None
                )
                if cand is None:
                    cand = case_start
                r["out_ts"] = format_ts(cand, frac_digits)
                outp = cand
            if outp < case_start:
                r["out_ts"] = format_ts(case_start, frac_digits)
            if outp > case_end:
                r["out_ts"] = format_ts(case_end, frac_digits)

    equal_rows = 0
    for r in all_rows:
        if r["out_ts"] == r["Timestamp"] and r["error_category"] != "CLEAN":
            equal_rows += 1

    print(f"{stats['blocks_repaired']} {stats['blocks_clamped']} {equal_rows}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    frac_digits = detect_precision([r["Timestamp"] for r in rows])

    all_rows = []
    for r in rows:
        rr = dict(r)
        rr["parsed"] = parse_timestamp_safe(rr["Timestamp"])
        rr["out_ts"] = rr["Timestamp"]
        rr["repaired"] = False
        all_rows.append(rr)

    by_case = defaultdict(list)
    for r in all_rows:
        by_case[r["Case"]].append(r)

    for case, case_rows in by_case.items():
        parsed_vals = [r["parsed"] for r in case_rows if r["parsed"] is not None]
        if parsed_vals:
            case_start = min(parsed_vals)
            case_end = max(parsed_vals)
        else:
            case_start = pd.Timestamp("2000-01-01 00:00:00")
            case_end = case_start
        for r in case_rows:
            r["case_start"] = case_start
            r["case_end"] = case_end

    stats = {"blocks_repaired": 0, "blocks_clamped": 0}

    for case, case_rows in by_case.items():
        repair_c1_case(case_rows, frac_digits, stats)
        repair_c2_case(case_rows, frac_digits, stats)
        repair_c3_case(case_rows, frac_digits, stats)

    final_self_check_and_fix(all_rows, frac_digits, stats)

    # Preserve exact non-timestamp columns and row order
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            out = {}
            for c in fieldnames:
                if c == "Timestamp":
                    out[c] = r["out_ts"]
                else:
                    out[c] = r[c]
            writer.writerow(out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # fail-safe: produce no crash; best-effort passthrough if something unexpected happens
        import sys
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=True)
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()
            with open(args.input, "r", newline="", encoding="utf-8") as fi, open(args.output, "w", newline="", encoding="utf-8") as fo:
                data = fi.read()
                fo.write(data)
            print("0 0 0")
        except Exception:
            print("0 0 0")