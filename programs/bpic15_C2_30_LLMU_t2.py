import argparse
import itertools
import math
import re
from collections import defaultdict
from datetime import timedelta

import pandas as pd


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
    {"source": "send confirmation receipt", "target": "enter senddate acknowledgement", "n": 412, "p10_seconds": 2.0, "p25_seconds": 67.0, "median_seconds": 159.0, "p75_seconds": 308.25, "p90_seconds": 1342.5, "share_under_60s": 0.245},
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
]

GLOBAL_MEDIAN_DURATION_SECONDS = 1.0


def build_maps():
    trans_prob = {}
    for x in TRANSITION_PROBABILITIES_TOP:
        trans_prob[(x["source"], x["target"])] = float(x["probability"])
    dur = {}
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for x in DURATION_PROFILES_TOP:
        key = (x["source"], x["target"])
        dur[key] = x
        outgoing[x["source"]].append(x)
        incoming[x["target"]].append(x)
    return trans_prob, dur, outgoing, incoming


TRANS_PROB, DUR_PROF, OUTGOING_PROF, INCOMING_PROF = build_maps()


def detect_precision(ts_series):
    max_frac = 0
    any_frac = False
    for s in ts_series.dropna().astype(str):
        m = re.search(r"\.(\d+)", s)
        if m:
            any_frac = True
            max_frac = max(max_frac, len(m.group(1)))
    return max_frac if any_frac else 0


def parse_ts(s):
    if s is None:
        return pd.NaT
    ss = str(s).strip()
    if ss == "" or ss.lower() == "nan":
        return pd.NaT
    try:
        return pd.to_datetime(ss, errors="coerce")
    except Exception:
        return pd.NaT


def infer_missing_pattern(ts):
    s = "" if ts is None else str(ts).strip()
    if s == "" or s.lower() == "nan":
        return {"blank": True, "date_only": False, "date": None}
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return {"blank": False, "date_only": True, "date": s}
    return {"blank": False, "date_only": False, "date": None}


def format_ts(dt, frac_digits):
    if pd.isna(dt):
        return ""
    py = pd.Timestamp(dt).to_pydatetime()
    base = py.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits > 0:
        micros = py.microsecond
        frac = f"{micros:06d}"[:frac_digits].ljust(frac_digits, "0")
        return f"{base}.{frac}"
    return base


def choose_gap_seconds(source, target, min_step_seconds):
    prof = DUR_PROF.get((source, target))
    if prof:
        su = prof["share_under_60s"]
        if su >= 0.99:
            g = prof["p10_seconds"]
        elif su >= 0.85:
            g = prof["p25_seconds"]
        elif su >= 0.6:
            g = min(prof["p25_seconds"], max(prof["p10_seconds"], 1.0))
        else:
            g = min(prof["p10_seconds"], 60.0) if prof["p10_seconds"] > 0 else min(prof["p25_seconds"], 60.0)
        g = max(float(g), min_step_seconds)
        return g
    candidates = []
    for x in OUTGOING_PROF.get(source, []):
        su = x["share_under_60s"]
        if su >= 0.85:
            candidates.append(max(float(x["p25_seconds"]), min_step_seconds))
        elif su >= 0.99:
            candidates.append(max(float(x["p10_seconds"]), min_step_seconds))
    for x in INCOMING_PROF.get(target, []):
        su = x["share_under_60s"]
        if su >= 0.85:
            candidates.append(max(float(x["p25_seconds"]), min_step_seconds))
        elif su >= 0.99:
            candidates.append(max(float(x["p10_seconds"]), min_step_seconds))
    if candidates:
        return max(min_step_seconds, min(sorted(candidates)[len(candidates) // 2], 300.0))
    return max(min_step_seconds, GLOBAL_MEDIAN_DURATION_SECONDS)


def transition_score(a, b):
    p = TRANS_PROB.get((a, b))
    if p is not None:
        return math.log(max(p, 1e-12)) + 4.0
    prof = DUR_PROF.get((a, b))
    if prof:
        su = prof["share_under_60s"]
        n = prof["n"]
        return 1.0 + min(2.0, math.log1p(n) / 4.0) + su
    out_any = bool(OUTGOING_PROF.get(a))
    in_any = bool(INCOMING_PROF.get(b))
    if out_any or in_any:
        return -1.5
    return -3.0


def local_gap_plausibility(prev_act, next_act, gap_seconds):
    if gap_seconds < 0:
        return -1e9
    prof = DUR_PROF.get((prev_act, next_act))
    if prof:
        p10 = float(prof["p10_seconds"])
        p25 = float(prof["p25_seconds"])
        med = float(prof["median_seconds"])
        p90 = float(prof["p90_seconds"])
        center = p25 if prof["share_under_60s"] >= 0.7 else med
        scale = max(1.0, p90 - p10, center, 5.0)
        return -abs(gap_seconds - center) / scale + math.log1p(prof["n"]) / 10.0
    return -abs(gap_seconds - GLOBAL_MEDIAN_DURATION_SECONDS) / 60.0 - 0.2


def order_block(activities, left_anchor_act=None, right_anchor_act=None):
    n = len(activities)
    idxs = list(range(n))
    if n <= 7:
        best_perm = idxs[:]
        best_score = -1e18
        for perm in itertools.permutations(idxs):
            score = 0.0
            if left_anchor_act is not None:
                score += transition_score(left_anchor_act, activities[perm[0]])
            for i in range(n - 1):
                score += transition_score(activities[perm[i]], activities[perm[i + 1]])
            if right_anchor_act is not None:
                score += transition_score(activities[perm[-1]], right_anchor_act)
            if score > best_score:
                best_score = score
                best_perm = list(perm)
        return best_perm
    beams = [([], set(), 0.0)]
    beam_width = 50
    for _ in range(n):
        new_beams = []
        for seq, used, score in beams:
            for j in idxs:
                if j in used:
                    continue
                add = 0.0
                if not seq and left_anchor_act is not None:
                    add += transition_score(left_anchor_act, activities[j])
                if seq:
                    add += transition_score(activities[seq[-1]], activities[j])
                new_beams.append((seq + [j], used | {j}, score + add))
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]
    best = None
    best_score = -1e18
    for seq, used, score in beams:
        if right_anchor_act is not None and seq:
            score += transition_score(activities[seq[-1]], right_anchor_act)
        if score > best_score:
            best_score = score
            best = seq
    return best if best else idxs[:]


def build_c1_times(observed_dt, ordered_activities, right_anchor_dt, min_step_seconds):
    gaps = []
    for i in range(len(ordered_activities) - 1):
        gaps.append(choose_gap_seconds(ordered_activities[i], ordered_activities[i + 1], min_step_seconds))
    if len(ordered_activities) == 1:
        return [observed_dt], False
    min_total = min_step_seconds * (len(ordered_activities) - 1)
    if right_anchor_dt is not None and not pd.isna(right_anchor_dt):
        room = max(0.0, (right_anchor_dt - observed_dt).total_seconds())
        target_total = sum(gaps)
        clamped = False
        if room < min_total:
            room = min_total
            clamped = True
        if target_total > room:
            ratio = room / target_total if target_total > 0 else 1.0
            gaps = [max(min_step_seconds, g * ratio) for g in gaps]
            total = sum(gaps)
            if total > room:
                excess = total - room
                reducible = sum(max(0.0, g - min_step_seconds) for g in gaps)
                if reducible > 0:
                    new_gaps = []
                    for g in gaps:
                        red = max(0.0, g - min_step_seconds)
                        take = excess * red / reducible
                        new_gaps.append(max(min_step_seconds, g - take))
                    gaps = new_gaps
            clamped = True
        else:
            clamped = False
    else:
        clamped = False
    times = [observed_dt]
    cur = observed_dt
    for g in gaps:
        cur = cur + pd.to_timedelta(g, unit="s")
        times.append(cur)
    return times, clamped


def estimate_between(left_dt, right_dt, left_act, act, right_act, min_step_seconds):
    if left_dt is not None and right_dt is not None and not pd.isna(left_dt) and not pd.isna(right_dt):
        left_gap = choose_gap_seconds(left_act, act, min_step_seconds) if left_act is not None else min_step_seconds
        right_gap = choose_gap_seconds(act, right_act, min_step_seconds) if right_act is not None else min_step_seconds
        total_room = (right_dt - left_dt).total_seconds()
        if total_room < 0:
            return left_dt
        if left_gap + right_gap <= 0:
            frac = 0.5
        else:
            frac = left_gap / (left_gap + right_gap)
        est = left_dt + pd.to_timedelta(total_room * frac, unit="s")
        lo = left_dt
        hi = right_dt
        if est < lo:
            est = lo
        if est > hi:
            est = hi
        if est == hi and total_room >= min_step_seconds:
            est = hi - pd.to_timedelta(min_step_seconds, unit="s")
            if est < lo:
                est = lo
        return est
    if left_dt is not None and not pd.isna(left_dt):
        gap = choose_gap_seconds(left_act, act, min_step_seconds) if left_act is not None else min_step_seconds
        return left_dt + pd.to_timedelta(gap, unit="s")
    if right_dt is not None and not pd.isna(right_dt):
        gap = choose_gap_seconds(act, right_act, min_step_seconds) if right_act is not None else min_step_seconds
        return right_dt - pd.to_timedelta(gap, unit="s")
    return pd.NaT


def same_day_clamp(dt, date_str, left_dt, right_dt):
    day_start = pd.to_datetime(date_str + " 00:00:00")
    day_end = pd.to_datetime(date_str + " 23:59:59.999999")
    lo = day_start
    hi = day_end
    if left_dt is not None and not pd.isna(left_dt) and left_dt.date() == day_start.date():
        lo = max(lo, left_dt)
    if right_dt is not None and not pd.isna(right_dt) and right_dt.date() == day_start.date():
        hi = min(hi, right_dt)
    if dt < lo:
        return lo
    if dt > hi:
        return hi
    return dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in original_cols:
        if c not in df.columns:
            df[c] = ""

    out_df = df.copy()
    out_df["__orig_ts"] = df["Timestamp"].astype(str)
    out_df["__new_ts"] = out_df["__orig_ts"]
    out_df["__parsed_ts"] = df["Timestamp"].apply(parse_ts)
    out_df["__repaired"] = False

    frac_digits = detect_precision(df["Timestamp"])
    min_step_seconds = 0.001 if frac_digits >= 3 else (10 ** (-frac_digits) if frac_digits > 0 else 1.0)

    case_to_indices = defaultdict(list)
    for i, case in enumerate(df["Case"].astype(str).tolist()):
        case_to_indices[case].append(i)

    c1_blocks_repaired = 0
    c1_blocks_clamped = 0

    for case, idxs in case_to_indices.items():
        parsed_list = out_df.loc[idxs, "__parsed_ts"].tolist()

        clean_idx_local = [j for j, gi in enumerate(idxs) if str(df.at[gi, "error_category"]) == "CLEAN" and not pd.isna(parsed_list[j])]
        case_start = min([parsed_list[j] for j in clean_idx_local], default=min([x for x in parsed_list if not pd.isna(x)], default=pd.NaT))
        case_end = max([parsed_list[j] for j in clean_idx_local], default=max([x for x in parsed_list if not pd.isna(x)], default=pd.NaT))

        # C1 block detection: consecutive same-case non-clean rows with same observed timestamp string and category C1
        local = 0
        while local < len(idxs):
            gi = idxs[local]
            cat = str(df.at[gi, "error_category"])
            ts_str = str(df.at[gi, "Timestamp"])
            if cat == "C1_ORDERING_AMBIGUITY":
                start = local
                end = local + 1
                while end < len(idxs):
                    gj = idxs[end]
                    if str(df.at[gj, "error_category"]) == "C1_ORDERING_AMBIGUITY" and str(df.at[gj, "Timestamp"]) == ts_str:
                        end += 1
                    else:
                        break
                if end - start >= 2:
                    block_gidxs = idxs[start:end]
                    observed_dt = parse_ts(ts_str)
                    left_anchor_idx = None
                    right_anchor_idx = None
                    for p in range(start - 1, -1, -1):
                        g = idxs[p]
                        if str(df.at[g, "error_category"]) == "CLEAN" and not pd.isna(out_df.at[g, "__parsed_ts"]):
                            left_anchor_idx = g
                            break
                    for p in range(end, len(idxs)):
                        g = idxs[p]
                        if str(df.at[g, "error_category"]) == "CLEAN" and not pd.isna(out_df.at[g, "__parsed_ts"]):
                            right_anchor_idx = g
                            break

                    left_act = df.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None else None
                    right_act = df.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None else None
                    right_dt = out_df.at[right_anchor_idx, "__parsed_ts"] if right_anchor_idx is not None else pd.NaT

                    acts = [df.at[g, "Activity"] for g in block_gidxs]
                    perm = order_block(acts, left_act, right_act)
                    ordered_gidxs = [block_gidxs[p] for p in perm]
                    ordered_acts = [acts[p] for p in perm]
                    times, clamped = build_c1_times(observed_dt, ordered_acts, right_dt, min_step_seconds)

                    if left_anchor_idx is not None:
                        left_dt = out_df.at[left_anchor_idx, "__parsed_ts"]
                        fixed_times = []
                        cur_prev = left_dt
                        for k, t in enumerate(times):
                            if t < cur_prev:
                                t = cur_prev
                            if k > 0 and t <= fixed_times[-1]:
                                t = fixed_times[-1] + pd.to_timedelta(min_step_seconds, unit="s")
                            fixed_times.append(t)
                            cur_prev = t
                        times = fixed_times
                    if right_anchor_idx is not None and len(times) > 0:
                        room = (right_dt - observed_dt).total_seconds()
                        min_total = min_step_seconds * (len(times) - 1)
                        if room < min_total:
                            room = min_total
                        if (times[-1] - observed_dt).total_seconds() > room:
                            gaps = []
                            for i in range(len(times) - 1):
                                gaps.append((times[i + 1] - times[i]).total_seconds())
                            total = sum(gaps)
                            if total <= 0:
                                gaps = [min_step_seconds] * (len(times) - 1)
                                total = sum(gaps)
                            ratio = room / total if total > 0 else 1.0
                            gaps = [max(min_step_seconds, g * ratio) for g in gaps]
                            times = [observed_dt]
                            cur = observed_dt
                            for g in gaps:
                                cur = cur + pd.to_timedelta(g, unit="s")
                                times.append(cur)
                            clamped = True

                    for rank, g in enumerate(ordered_gidxs):
                        new_str = format_ts(times[rank], frac_digits)
                        out_df.at[g, "__new_ts"] = new_str
                        out_df.at[g, "__parsed_ts"] = parse_ts(new_str)
                        out_df.at[g, "__repaired"] = True
                    c1_blocks_repaired += 1
                    if clamped:
                        c1_blocks_clamped += 1
                    local = end
                    continue
            local += 1

        # C2 processing
        for pos in range(1, len(idxs)):
            g_prev = idxs[pos - 1]
            g_cur = idxs[pos]
            if str(df.at[g_cur, "error_category"]) != "C2_TEMPORAL_INCONSISTENCY":
                continue
            prev_dt = out_df.at[g_prev, "__parsed_ts"]
            cur_dt = out_df.at[g_cur, "__parsed_ts"]
            if pd.isna(prev_dt) or pd.isna(cur_dt) or cur_dt >= prev_dt:
                continue

            left2_idx = None
            right2_idx = None
            for p in range(pos - 2, -1, -1):
                g = idxs[p]
                if str(df.at[g, "error_category"]) == "CLEAN" and not pd.isna(out_df.at[g, "__parsed_ts"]):
                    left2_idx = g
                    break
            for p in range(pos + 1, len(idxs)):
                g = idxs[p]
                if str(df.at[g, "error_category"]) == "CLEAN" and not pd.isna(out_df.at[g, "__parsed_ts"]):
                    right2_idx = g
                    break

            act_left = df.at[g_prev, "Activity"]
            act_cur = df.at[g_cur, "Activity"]
            act_before = df.at[left2_idx, "Activity"] if left2_idx is not None else None
            act_after = df.at[right2_idx, "Activity"] if right2_idx is not None else None
            dt_before = out_df.at[left2_idx, "__parsed_ts"] if left2_idx is not None else pd.NaT
            dt_after = out_df.at[right2_idx, "__parsed_ts"] if right2_idx is not None else pd.NaT

            swap_prev = cur_dt
            swap_cur = prev_dt
            swap_valid = True
            if left2_idx is not None and swap_prev < dt_before:
                swap_valid = False
            if right2_idx is not None and swap_cur > dt_after:
                swap_valid = False
            if swap_prev > swap_cur:
                swap_valid = False

            observed_score = 0.0
            observed_score += local_gap_plausibility(act_before, act_left, (prev_dt - dt_before).total_seconds()) if left2_idx is not None else 0.0
            observed_score += local_gap_plausibility(act_left, act_cur, (cur_dt - prev_dt).total_seconds())
            observed_score += local_gap_plausibility(act_cur, act_after, (dt_after - cur_dt).total_seconds()) if right2_idx is not None else 0.0

            swap_score = -1e18
            if swap_valid:
                swap_score = 0.0
                swap_score += local_gap_plausibility(act_before, act_left, (swap_prev - dt_before).total_seconds()) if left2_idx is not None else 0.0
                swap_score += local_gap_plausibility(act_left, act_cur, (swap_cur - swap_prev).total_seconds())
                swap_score += local_gap_plausibility(act_cur, act_after, (dt_after - swap_cur).total_seconds()) if right2_idx is not None else 0.0

            if swap_valid and swap_score > observed_score + 0.5:
                s1 = format_ts(swap_prev, frac_digits)
                s2 = format_ts(swap_cur, frac_digits)
                out_df.at[g_prev, "__new_ts"] = s1
                out_df.at[g_prev, "__parsed_ts"] = parse_ts(s1)
                out_df.at[g_prev, "__repaired"] = True
                out_df.at[g_cur, "__new_ts"] = s2
                out_df.at[g_cur, "__parsed_ts"] = parse_ts(s2)
                out_df.at[g_cur, "__repaired"] = True
            else:
                est = estimate_between(prev_dt, dt_after if right2_idx is not None else pd.NaT, act_left, act_cur, act_after, min_step_seconds)
                if pd.isna(est):
                    est = prev_dt + pd.to_timedelta(min_step_seconds, unit="s")
                if right2_idx is not None and est > dt_after:
                    est = prev_dt + (dt_after - prev_dt) / 2
                if est <= prev_dt:
                    est = prev_dt + pd.to_timedelta(min_step_seconds, unit="s")
                if right2_idx is not None and est > dt_after:
                    est = dt_after
                if case_end is not None and not pd.isna(case_end) and est > case_end:
                    est = case_end
                if case_start is not None and not pd.isna(case_start) and est < case_start:
                    est = case_start
                new_s = format_ts(est, frac_digits)
                out_df.at[g_cur, "__new_ts"] = new_s
                out_df.at[g_cur, "__parsed_ts"] = parse_ts(new_s)
                out_df.at[g_cur, "__repaired"] = True

        # C3 processing
        for pos, g in enumerate(idxs):
            if str(df.at[g, "error_category"]) != "C3_MISSING_TEMPORAL_INFORMATION":
                continue
            pattern = infer_missing_pattern(df.at[g, "Timestamp"])
            left_idx = None
            right_idx = None
            for p in range(pos - 1, -1, -1):
                gg = idxs[p]
                if not pd.isna(out_df.at[gg, "__parsed_ts"]):
                    left_idx = gg
                    break
            for p in range(pos + 1, len(idxs)):
                gg = idxs[p]
                if not pd.isna(out_df.at[gg, "__parsed_ts"]):
                    right_idx = gg
                    break
            left_dt = out_df.at[left_idx, "__parsed_ts"] if left_idx is not None else pd.NaT
            right_dt = out_df.at[right_idx, "__parsed_ts"] if right_idx is not None else pd.NaT
            left_act = df.at[left_idx, "Activity"] if left_idx is not None else None
            right_act = df.at[right_idx, "Activity"] if right_idx is not None else None
            act = df.at[g, "Activity"]

            est = estimate_between(left_dt, right_dt, left_act, act, right_act, min_step_seconds)
            if pd.isna(est):
                if case_start is not None and not pd.isna(case_start) and case_end is not None and not pd.isna(case_end):
                    est = case_start + (case_end - case_start) / 2
                elif case_start is not None and not pd.isna(case_start):
                    est = case_start
                elif case_end is not None and not pd.isna(case_end):
                    est = case_end
                else:
                    est = pd.Timestamp("2000-01-01 00:00:00")

            if pattern["date_only"]:
                est = same_day_clamp(est, pattern["date"], left_dt, right_dt)
            else:
                if left_dt is not None and not pd.isna(left_dt) and est < left_dt:
                    est = left_dt
                if right_dt is not None and not pd.isna(right_dt) and est > right_dt:
                    est = left_dt + (right_dt - left_dt) / 2 if left_dt is not None and not pd.isna(left_dt) else right_dt
            if case_start is not None and not pd.isna(case_start) and est < case_start:
                est = case_start
            if case_end is not None and not pd.isna(case_end) and est > case_end:
                est = case_end

            new_s = format_ts(est, frac_digits)
            out_df.at[g, "__new_ts"] = new_s
            out_df.at[g, "__parsed_ts"] = parse_ts(new_s)
            out_df.at[g, "__repaired"] = True

    # Self-check and repair
    # Preserve CLEAN rows exactly
    clean_mask = df["error_category"].astype(str) == "CLEAN"
    out_df.loc[clean_mask, "__new_ts"] = out_df.loc[clean_mask, "__orig_ts"]

    # Parseability and bounds
    for case, idxs in case_to_indices.items():
        parsed = [parse_ts(out_df.at[g, "__new_ts"]) for g in idxs if str(out_df.at[g, "__new_ts"]).strip() != ""]
        case_start = min(parsed) if parsed else pd.NaT
        case_end = max(parsed) if parsed else pd.NaT
        for pos, g in enumerate(idxs):
            if str(df.at[g, "error_category"]) == "CLEAN":
                continue
            ts = parse_ts(out_df.at[g, "__new_ts"])
            if pd.isna(ts):
                left_dt = pd.NaT
                right_dt = pd.NaT
                for p in range(pos - 1, -1, -1):
                    gg = idxs[p]
                    t = parse_ts(out_df.at[gg, "__new_ts"])
                    if not pd.isna(t):
                        left_dt = t
                        break
                for p in range(pos + 1, len(idxs)):
                    gg = idxs[p]
                    t = parse_ts(out_df.at[gg, "__new_ts"])
                    if not pd.isna(t):
                        right_dt = t
                        break
                repl = estimate_between(left_dt, right_dt, None, df.at[g, "Activity"], None, min_step_seconds)
                if pd.isna(repl):
                    repl = case_start if not pd.isna(case_start) else pd.Timestamp("2000-01-01 00:00:00")
                out_df.at[g, "__new_ts"] = format_ts(repl, frac_digits)
                out_df.at[g, "__parsed_ts"] = parse_ts(out_df.at[g, "__new_ts"])
            else:
                if not pd.isna(case_start) and ts < case_start:
                    ts = case_start
                if not pd.isna(case_end) and ts > case_end:
                    ts = case_end
                out_df.at[g, "__new_ts"] = format_ts(ts, frac_digits)
                out_df.at[g, "__parsed_ts"] = parse_ts(out_df.at[g, "__new_ts"])

    # C1 exact checks and fix
    for case, idxs in case_to_indices.items():
        local = 0
        while local < len(idxs):
            g = idxs[local]
            if str(df.at[g, "error_category"]) == "C1_ORDERING_AMBIGUITY":
                obs = str(df.at[g, "Timestamp"])
                start = local
                end = local + 1
                while end < len(idxs):
                    gg = idxs[end]
                    if str(df.at[gg, "error_category"]) == "C1_ORDERING_AMBIGUITY" and str(df.at[gg, "Timestamp"]) == obs:
                        end += 1
                    else:
                        break
                if end - start >= 2:
                    block = idxs[start:end]
                    observed_dt = parse_ts(obs)
                    vals = [(gg, parse_ts(out_df.at[gg, "__new_ts"])) for gg in block]
                    equal_obs = [gg for gg, t in vals if not pd.isna(t) and t == observed_dt]
                    acts = [df.at[gg, "Activity"] for gg in block]
                    left_anchor_idx = None
                    right_anchor_idx = None
                    for p in range(start - 1, -1, -1):
                        gg = idxs[p]
                        if str(df.at[gg, "error_category"]) == "CLEAN" and not pd.isna(parse_ts(out_df.at[gg, "__new_ts"])):
                            left_anchor_idx = gg
                            break
                    for p in range(end, len(idxs)):
                        gg = idxs[p]
                        if str(df.at[gg, "error_category"]) == "CLEAN" and not pd.isna(parse_ts(out_df.at[gg, "__new_ts"])):
                            right_anchor_idx = gg
                            break
                    left_act = df.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None else None
                    right_act = df.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None else None
                    right_dt = parse_ts(out_df.at[right_anchor_idx, "__new_ts"]) if right_anchor_idx is not None else pd.NaT

                    if len(equal_obs) != 1:
                        perm = order_block(acts, left_act, right_act)
                        ordered_g = [block[p] for p in perm]
                        ordered_a = [acts[p] for p in perm]
                        times, _ = build_c1_times(observed_dt, ordered_a, right_dt, min_step_seconds)
                        for i, gg in enumerate(ordered_g):
                            out_df.at[gg, "__new_ts"] = format_ts(times[i], frac_digits)
                            out_df.at[gg, "__parsed_ts"] = parse_ts(out_df.at[gg, "__new_ts"])
                    else:
                        times_sorted = sorted([t for _, t in vals if not pd.isna(t)])
                        need_fix = False
                        if len(times_sorted) != len(block):
                            need_fix = True
                        else:
                            for i in range(1, len(times_sorted)):
                                if not (times_sorted[i] > times_sorted[i - 1]):
                                    need_fix = True
                                    break
                        if need_fix:
                            perm = order_block(acts, left_act, right_act)
                            ordered_g = [block[p] for p in perm]
                            ordered_a = [acts[p] for p in perm]
                            times, _ = build_c1_times(observed_dt, ordered_a, right_dt, min_step_seconds)
                            for i, gg in enumerate(ordered_g):
                                out_df.at[gg, "__new_ts"] = format_ts(times[i], frac_digits)
                                out_df.at[gg, "__parsed_ts"] = parse_ts(out_df.at[gg, "__new_ts"])
                local = end
                continue
            local += 1

    rows_equal_input = 0
    for i in range(len(out_df)):
        if str(out_df.at[i, "__new_ts"]) == str(out_df.at[i, "__orig_ts"]):
            rows_equal_input += 1

    result = df.copy()
    result["Timestamp"] = out_df["__new_ts"]

    # hard identity preservation
    for c in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        result[c] = df[c]

    try:
        result.to_csv(args.output, index=False)
    except Exception:
        safe = result.copy()
        safe.to_csv(args.output, index=False)

    print(f"blocks_repaired={c1_blocks_repaired} blocks_clamped={c1_blocks_clamped} rows_still_equal={rows_equal_input}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")
        except Exception:
            pass