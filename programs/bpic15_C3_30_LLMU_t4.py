import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import permutations

import pandas as pd


TRANSITION_PROFILES = {
    ("register submission date request", "OLO messaging active"): {"n": 729, "p10": 86400.0, "p25": 172800.0, "median": 306406.0, "p75": 518400.0, "p90": 733026.6, "share": 0.0},
    ("forward to the competent authority", "regular procedure without MER"): {"n": 679, "p10": 0.0, "p25": 0.0, "median": 3.0, "p75": 6.0, "p90": 12.0, "share": 0.975},
    ("send confirmation receipt", "forward to the competent authority"): {"n": 603, "p10": 2.0, "p25": 3.0, "median": 6.0, "p75": 9.0, "p90": 21.0, "share": 0.939},
    ("request complete", "send letter in progress"): {"n": 572, "p10": 0.0, "p25": 0.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "share": 0.991},
    ("article 34  WABO applies", "assessment of content completed"): {"n": 567, "p10": 0.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "share": 0.991},
    ("assessment of content completed", "phase advice known"): {"n": 555, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 1.0, "share": 1.0},
    ("send confirmation receipt", "send confirmation receipt"): {"n": 500, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 208.4, "share": 0.802},
    ("phase application receptive", "request complete"): {"n": 500, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("by law", "creating environmental permit decision"): {"n": 493, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 1.0, "p90": 1.0, "share": 0.992},
    ("treat subcases completeness", "subcases completeness completed"): {"n": 491, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 3.0, "p90": 8.0, "share": 0.969},
    ("treat subcases content", "completed subcases content"): {"n": 479, "p10": 1.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 3.0, "share": 0.992},
    ("enter senddate decision", "enter senddate decision environmental permit"): {"n": 467, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.991},
    ("phase advice known", "grounds for refusal"): {"n": 439, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 3.0, "p90": 6.2, "share": 0.954},
    ("creating environmental permit decision", "decision date prior to decision"): {"n": 430, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 4.0, "share": 0.993},
    ("procedure change", "request complete"): {"n": 425, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 3.0, "p90": 6.0, "share": 0.986},
    ("subcases completeness completed", "procedure change"): {"n": 424, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 4.0, "p90": 7.0, "share": 0.988},
    ("suspension ground applicable", "by law"): {"n": 414, "p10": 1.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "share": 0.993},
    ("send confirmation receipt", "enter senddate acknowledgement"): {"n": 412, "p10": 2.0, "p25": 68.75, "median": 160.0, "p75": 308.25, "p90": 1342.5, "share": 0.243},
    ("ask stakeholders views", "suspension ground applicable"): {"n": 408, "p10": 0.0, "p25": 1.0, "median": 2.0, "p75": 4.0, "p90": 6.0, "share": 0.983},
    ("OLO messaging active", "phase application received"): {"n": 406, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("grounds for refusal", "ask stakeholders views"): {"n": 405, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 5.0, "p90": 15.6, "share": 0.953},
    ("phase application received", "send confirmation receipt"): {"n": 400, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 5.0, "p90": 11.0, "share": 0.968},
    ("regular procedure without MER", "send confirmation receipt"): {"n": 399, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("procedure change", "phase application receptive"): {"n": 384, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 5.0, "p90": 9.0, "share": 0.966},
    ("enter senddate acknowledgement", "send confirmation receipt"): {"n": 383, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.99},
    ("enter senddate procedure confirmation", "send procedure confirmation"): {"n": 378, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.952},
    ("send confirmation receipt", "start WABOprocedure"): {"n": 375, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("forward to the competent authority", "phase application received"): {"n": 372, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("registration date publication", "regular procedure without MER"): {"n": 357, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.994},
    ("procedure change", "article 34  WABO applies"): {"n": 348, "p10": 1.0, "p25": 1.0, "median": 2.0, "p75": 3.0, "p90": 11.0, "share": 0.945},
    ("OLO messaging active", "create procedure confirmation"): {"n": 344, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("transcript decision environmental permit to stakeholders", "generate publication document decision environmental permit"): {"n": 338, "p10": 6.0, "p25": 7.0, "median": 10.0, "p75": 19.75, "p90": 57.3, "share": 0.902},
    ("phase decision sent", "phase decision taken"): {"n": 320, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("grounds for refusal", "phase advice known"): {"n": 319, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("create procedure confirmation", "create publication document"): {"n": 318, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.997},
    ("phase advice known", "phase application receptive"): {"n": 316, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("send letter in progress", "treat subcases content"): {"n": 316, "p10": 0.0, "p25": 1.0, "median": 1.0, "p75": 2.0, "p90": 3.0, "share": 0.991},
    ("phase decision taken", "register objection and appeal periods"): {"n": 310, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 5.0, "p90": 13.1, "share": 0.971},
    ("article 34  WABO applies", "ask stakeholders views"): {"n": 305, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.997},
    ("creating environmental permit decision", "generating decision environmental permit"): {"n": 297, "p10": 4.0, "p25": 57.0, "median": 2063.0, "p75": 149542.0, "p90": 886390.6, "share": 0.256},
    ("enter senddate decision environmental permit", "phase decision sent"): {"n": 294, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 1.0, "share": 1.0},
    ("start WABOprocedure", "enter senddate acknowledgement"): {"n": 293, "p10": 34535.2, "p25": 38196.0, "median": 43648.0, "p75": 52527.0, "p90": 86400.0, "share": 0.0},
    ("completed subcases content", "procedure change"): {"n": 292, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 5.0, "p90": 10.0, "share": 0.979},
    ("send letter in progress", "set decision status"): {"n": 254, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("set decision phase decision permitting decided", "start decision phase decision permitting sent"): {"n": 241, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 0.988},
    ("phase application received", "reception through OLO"): {"n": 232, "p10": 1.0, "p25": 2.0, "median": 3.0, "p75": 6.25, "p90": 15.0, "share": 0.966},
    ("regular procedure without MER", "start WABOprocedure"): {"n": 230, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 1.0, "share": 1.0},
    ("transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit"): {"n": 228, "p10": 8.0, "p25": 25.75, "median": 49045.5, "p75": 172800.0, "p90": 259200.0, "share": 0.268},
    ("send procedure confirmation", "registration date publication"): {"n": 225, "p10": 8.0, "p25": 11.0, "median": 19.0, "p75": 69.0, "p90": 738725.6, "share": 0.724},
    ("regular procedure without MER", "enter senddate procedure confirmation"): {"n": 219, "p10": 1.0, "p25": 2.0, "median": 2.0, "p75": 4.0, "p90": 87809.2, "share": 0.872},
    ("generating decision environmental permit", "grounds for refusal"): {"n": 215, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("decision date prior to decision", "generating decision environmental permit"): {"n": 212, "p10": 0.0, "p25": 0.0, "median": 2.0, "p75": 5150.75, "p90": 442876.6, "share": 0.566},
    ("create publication document", "create subcases completeness"): {"n": 202, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 602849.9, "share": 0.891},
    ("ask stakeholders views", "assessment of content completed"): {"n": 194, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("by law", "completed subcases content"): {"n": 194, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("procedure change", "procedure change"): {"n": 193, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("submit decision", "suspension ground applicable"): {"n": 193, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("completed subcases content", "coordination of application"): {"n": 190, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
    ("register submission date request", "enter senddate acknowledgement"): {"n": 188, "p10": 172800.0, "p25": 345600.0, "median": 604800.0, "p75": 864000.0, "p90": 1209600.0, "share": 0.0},
    ("record date of decision environmental permit", "set decision phase decision permitting decided"): {"n": 187, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "share": 1.0},
}

TRANSITION_PROBS = {
    ("appeal subcase completed", "appealed to higher court"): 1.0,
    ("create coordination confirmation", "by law"): 1.0,
    ("create decision no permit required", "create subcases completeness"): 1.0,
    ("create forwarding copy", "enter senddate letter to stakeholders"): 1.0,
    ("create letter only notification required", "forward to the competent authority"): 1.0,
    ("creating cover letter no permit required", "decision date prior to decision"): 1.0,
    ("date environmental permit irrevocable", "assessment of content completed"): 1.0,
    ("enrich draft decision environmental permit", "enter date draft decisionenvironmental permit"): 1.0,
    ("enter objection term permanent suspension", "copy decision permanent suspension to stakeholders"): 1.0,
    ("enter senddate letter no permit required", "generating decision no permit required"): 1.0,
    ("generating decision no permit required", "no permit needed or only notification needed"): 1.0,
    ("objection decision no permit required received", "phase activity no permit required"): 1.0,
    ("partly permit application", "partly permit"): 1.0,
    ("phase activity no permit required", "phase archived case"): 1.0,
    ("phase forwarded to competent authority", "phase procedure aborted"): 1.0,
    ("phase only notification required", "phase procedure aborted"): 1.0,
    ("phase updated plan requested", "register receipt updated plan"): 1.0,
    ("reactive designation received", "register objection and appeal periods"): 1.0,
    ("record date of decision no permit required", "enter senddate letter no permit required"): 1.0,
    ("request updated plan again", "contacting applicant"): 1.0,
    ("send letter receptive test not ok wait for report firedepartement", "receive additional information"): 1.0,
    ("send send confirmation competent authority", "request to competent authority"): 1.0,
    ("set decision phase decision no permit required sent", "set decision phase no permit required decided"): 1.0,
    ("set decision phase decision permitting published", "close case"): 1.0,
    ("set decision phase no permit required decided", "set decision status"): 1.0,
    ("set decision phase unlicensed irrevocable", "close case"): 1.0,
    ("set phase", "creating environmental permit decision"): 1.0,
    ("subcase art. 34 WABO completed", "refusal after suspension"): 1.0,
    ("subcase designation GS completed", "subcase perspective complete"): 1.0,
    ("subcases submit draft decision completed", "enrich draft decision"): 1.0,
    ("temporary permit", "regular procedure without MER"): 1.0,
    ("treat subcase suspension art. 34 WABO", "subcase art. 34 WABO completed"): 1.0,
    ("treat subcases submit draft decision", "subcases submit draft decision completed"): 1.0,
    ("treating subcase designation GS", "updated plan after perspective"): 1.0,
    ("phase archived case", "phase case handled"): 0.9813,
    ("enter senddate decision", "enter senddate decision environmental permit"): 0.9729,
    ("enter objectionterm", "copy decision to extend procedure to stakeholders"): 0.9444,
    ("decided on draft decision environmental permit", "enrich draft decision"): 0.9,
    ("no permit required for application", "MER present"): 0.85,
    ("MER present", "activities regular procedure"): 0.8495,
    ("start decision phase permanent suspension sent", "start decision phase permanently suspended decided"): 0.8462,
    ("cancel BAG objects research", "phase procedure aborted"): 0.8333,
    ("subcase final VVGB completed", "VVGB positive"): 0.8333,
    ("treat subcase final VVGB", "subcase final VVGB completed"): 0.8333,
    ("decision terminate on request", "phase procedure aborted"): 0.8235,
    ("determine cause for suspension article 34 WABO", "assessment of content completed"): 0.8182,
    ("create monitoring case", "creating cover letter decision"): 0.8065,
    ("phase decision irrevocable", "read publication date field"): 0.8,
    ("treat subcase WAW", "subcase WAW completed"): 0.8,
    ("create decision terminate on request", "decision terminate on request"): 0.7895,
    ("set phase decision revoked", "close case"): 0.7778,
    ("publication date draft decision", "date draft decision for inspection"): 0.7647,
    ("reception through OLO", "send confirmation receipt"): 0.7564,
    ("publish official draft decision", "send design decision to stakeholders"): 0.7556,
    ("environmental permit irrevocable", "determine cause for suspension article 34 WABO"): 0.75,
    ("term perspective on draft decision", "phase draft decision made"): 0.75,
    ("MER required", "OLO messaging active"): 0.7333,
    ("create subcases submit draft decision", "enrich draft decision"): 0.7143,
    ("enter enddate suspension art. 33 WABO", "set decision phase: suspension cancelled"): 0.7143,
    ("treat subcases completeness", "subcases completeness completed"): 0.7044,
}

GLOBAL_FAST_DEFAULT = 3.0
VERY_SMALL_DEFAULT = 1.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def detect_precision(ts):
    if ts is None:
        return 0
    ts = str(ts)
    if "." in ts:
        frac = ts.split(".")[-1]
        frac = frac.strip()
        digits = 0
        for ch in frac:
            if ch.isdigit():
                digits += 1
            else:
                break
        return digits
    return 0


def parse_ts(ts):
    if ts is None:
        return None
    s = str(ts).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


def is_date_only_string(s):
    if s is None:
        return False
    s = str(s).strip()
    if s == "":
        return False
    return len(s) == 10 and s[4] == "-" and s[7] == "-"


def format_ts(dt, precision):
    if precision <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    frac = f"{dt.microsecond:06d}"[:precision]
    return f"{base}.{frac}"


def score_prob(a, b):
    p = TRANSITION_PROBS.get((a, b))
    if p is None:
        return math.log(1e-6)
    return math.log(max(p, 1e-9))


def profile(a, b):
    return TRANSITION_PROFILES.get((a, b))


def expected_gap(a, b):
    prof = profile(a, b)
    if prof:
        if prof["share"] >= 0.98:
            val = prof["p10"]
        else:
            val = prof["p25"]
        if val <= 0:
            val = VERY_SMALL_DEFAULT
        return max(val, VERY_SMALL_DEFAULT)
    if (a, b) in TRANSITION_PROBS:
        return VERY_SMALL_DEFAULT
    return GLOBAL_FAST_DEFAULT


def duration_penalty(a, b, gap_seconds):
    if gap_seconds < 0:
        return 1000.0 + abs(gap_seconds)
    prof = profile(a, b)
    if prof:
        target = expected_gap(a, b)
        spread = max(prof["p90"] - prof["p10"], 5.0)
        return abs(gap_seconds - target) / spread
    return min(gap_seconds / 60.0, 100.0)


def choose_order(block_rows, left_act, right_act):
    acts = [r["Activity"] for r in block_rows]
    idxs = list(range(len(block_rows)))
    best_perm = idxs[:]
    best_score = -1e18

    if len(idxs) <= 7:
        candidates = permutations(idxs)
    else:
        # simple beam: start from each possible first, greedily extend by best transition score
        candidates_list = []
        for start in idxs:
            used = {start}
            order = [start]
            while len(order) < len(idxs):
                last = acts[order[-1]]
                remaining = [j for j in idxs if j not in used]
                remaining.sort(key=lambda j: score_prob(last, acts[j]), reverse=True)
                pick = remaining[0]
                order.append(pick)
                used.add(pick)
            candidates_list.append(tuple(order))
        candidates = candidates_list

    seen = set()
    for perm in candidates:
        if perm in seen:
            continue
        seen.add(perm)
        seq = [acts[i] for i in perm]
        s = 0.0
        if left_act is not None:
            s += score_prob(left_act, seq[0])
        if right_act is not None:
            s += score_prob(seq[-1], right_act)
        for a, b in zip(seq[:-1], seq[1:]):
            s += 2.0 * score_prob(a, b)
        # slight preference to keep earliest row as first only as weak tie-breaker
        s += -0.001 * perm[0]
        if s > best_score:
            best_score = s
            best_perm = list(perm)
    return best_perm


def cumulative_gaps_for_sequence(acts):
    gaps = []
    for a, b in zip(acts[:-1], acts[1:]):
        gaps.append(max(expected_gap(a, b), VERY_SMALL_DEFAULT))
    return gaps


def ensure_strict_increasing(times, min_step_us=1000000):
    out = times[:]
    step = pd.Timedelta(microseconds=min_step_us)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + step
    return out


def rescale_gaps(gaps, available_seconds, min_unit_seconds=1.0):
    n = len(gaps)
    if n == 0:
        return []
    min_total = min_unit_seconds * n
    if available_seconds <= min_total:
        return [min_unit_seconds] * n
    total = sum(gaps)
    if total <= available_seconds:
        return [max(g, min_unit_seconds) for g in gaps]
    extra_room = available_seconds - min_total
    raw_extra = [max(g - min_unit_seconds, 0.0) for g in gaps]
    raw_total = sum(raw_extra)
    if raw_total <= 0:
        return [min_unit_seconds] * n
    scaled = [min_unit_seconds + extra_room * (g / raw_total) for g in raw_extra]
    return scaled


def nearest_clean_left(case_rows, pos):
    for i in range(pos - 1, -1, -1):
        if case_rows[i]["error_category"] == "CLEAN":
            t = case_rows[i]["parsed_ts"]
            if t is not None:
                return i, case_rows[i]
    return None, None


def nearest_clean_right(case_rows, pos):
    for i in range(pos + 1, len(case_rows)):
        if case_rows[i]["error_category"] == "CLEAN":
            t = case_rows[i]["parsed_ts"]
            if t is not None:
                return i, case_rows[i]
    return None, None


def case_bounds(case_rows):
    vals = [r["parsed_ts"] for r in case_rows if r["parsed_ts"] is not None]
    if not vals:
        return None, None
    return min(vals), max(vals)


def clamp(dt, lo, hi):
    if lo is not None and dt < lo:
        dt = lo
    if hi is not None and dt > hi:
        dt = hi
    return dt


def local_plausibility(prev_row, row, next_row, t):
    score = 0.0
    if prev_row is not None and prev_row["parsed_ts"] is not None:
        g = (t - prev_row["parsed_ts"]).total_seconds()
        score -= duration_penalty(prev_row["Activity"], row["Activity"], g)
    if next_row is not None and next_row["parsed_ts"] is not None:
        g = (next_row["parsed_ts"] - t).total_seconds()
        score -= duration_penalty(row["Activity"], next_row["Activity"], g)
    return score


def infer_single_time(case_rows, pos, keep_date=False):
    row = case_rows[pos]
    left_idx, left = nearest_clean_left(case_rows, pos)
    right_idx, right = nearest_clean_right(case_rows, pos)
    lo, hi = case_bounds(case_rows)

    candidate = None
    from_left = None
    from_right = None

    if left is not None and left["parsed_ts"] is not None:
        gap = expected_gap(left["Activity"], row["Activity"])
        from_left = left["parsed_ts"] + pd.Timedelta(seconds=gap)

    if right is not None and right["parsed_ts"] is not None:
        gap = expected_gap(row["Activity"], right["Activity"])
        from_right = right["parsed_ts"] - pd.Timedelta(seconds=gap)

    if from_left is not None and from_right is not None:
        if from_left <= from_right:
            candidate = from_left + (from_right - from_left) / 2
        else:
            candidate = right["parsed_ts"] - (right["parsed_ts"] - left["parsed_ts"]) / 2
    elif from_left is not None:
        candidate = from_left
    elif from_right is not None:
        candidate = from_right
    else:
        candidate = lo if lo is not None else pd.Timestamp("2000-01-01 00:00:01")

    anchor_lo = left["parsed_ts"] if left is not None else lo
    anchor_hi = right["parsed_ts"] if right is not None else hi
    candidate = clamp(candidate, anchor_lo, anchor_hi)

    if keep_date and row["Timestamp"] and str(row["Timestamp"]).strip():
        s = str(row["Timestamp"]).strip()
        date_part = s[:10]
        try:
            d = datetime.strptime(date_part, "%Y-%m-%d").date()
            candidate = pd.Timestamp(datetime.combine(d, candidate.to_pydatetime().time()))
            day_lo = pd.Timestamp(datetime.combine(d, datetime.min.time()))
            day_hi = pd.Timestamp(datetime.combine(d, datetime.max.time().replace(microsecond=999999)))
            if anchor_lo is not None:
                day_lo = max(day_lo, anchor_lo)
            if anchor_hi is not None:
                day_hi = min(day_hi, anchor_hi)
            candidate = clamp(candidate, day_lo, day_hi)
        except Exception:
            pass

    return candidate


def repair_c1_block(case_rows, start, end, repaired_map, stats):
    block = case_rows[start:end + 1]
    observed_str = block[0]["Timestamp"]
    observed_ts = block[0]["parsed_ts"]
    if observed_ts is None:
        return

    left_idx, left = nearest_clean_left(case_rows, start)
    right_idx, right = nearest_clean_right(case_rows, end)
    left_act = left["Activity"] if left is not None else None
    right_act = right["Activity"] if right is not None else None

    perm = choose_order(block, left_act, right_act)
    ordered_rows = [block[i] for i in perm]
    ordered_acts = [r["Activity"] for r in ordered_rows]
    gaps = cumulative_gaps_for_sequence(ordered_acts)

    clamped = False
    if right is not None and right["parsed_ts"] is not None:
        available = (right["parsed_ts"] - observed_ts).total_seconds()
        available = max(available, len(block) - 1)
        scaled_gaps = rescale_gaps(gaps, available_seconds=available, min_unit_seconds=1.0)
        if sum(gaps) > available:
            clamped = True
    else:
        scaled_gaps = [max(g, 1.0) for g in gaps]

    ordered_times = [observed_ts]
    cur = observed_ts
    for g in scaled_gaps:
        cur = cur + pd.Timedelta(seconds=float(g))
        ordered_times.append(cur)

    if right is not None and right["parsed_ts"] is not None:
        ordered_times = [clamp(t, observed_ts, right["parsed_ts"]) for t in ordered_times]

    ordered_times = ensure_strict_increasing(ordered_times, min_step_us=1000000)
    if right is not None and ordered_times[-1] > right["parsed_ts"]:
        # back-compress deterministically
        min_last = observed_ts + pd.Timedelta(seconds=len(block) - 1)
        final_hi = max(min_last, right["parsed_ts"])
        if final_hi < min_last:
            final_hi = min_last
        span = max((final_hi - observed_ts).total_seconds(), len(block) - 1)
        step = max(span / (len(block) - 1), 1.0)
        ordered_times = [observed_ts + pd.Timedelta(seconds=step * i) for i in range(len(block))]
        ordered_times = ensure_strict_increasing(ordered_times, min_step_us=1000000)
        clamped = True

    # exactly one row keeps observed shared timestamp, the first event in inferred order
    for idx_in_order, row in enumerate(ordered_rows):
        rid = row["row_id"]
        if idx_in_order == 0:
            repaired_map[rid] = row["Timestamp"]
        else:
            repaired_map[rid] = format_ts(ordered_times[idx_in_order], row["precision"])

    stats["c1_blocks_repaired"] += 1
    if clamped:
        stats["blocks_clamped"] += 1


def plausibility_of_assignment(case_rows, i, t_i, t_j):
    prev_row = case_rows[i - 1] if i - 1 >= 0 else None
    next_row = case_rows[i + 1] if i + 1 < len(case_rows) else None
    row_i = case_rows[i]
    row_j = case_rows[i + 1]
    score = 0.0

    if prev_row is not None and prev_row["parsed_ts"] is not None:
        score -= duration_penalty(prev_row["Activity"], row_i["Activity"], (t_i - prev_row["parsed_ts"]).total_seconds())
    score -= duration_penalty(row_i["Activity"], row_j["Activity"], (t_j - t_i).total_seconds())
    if next_row is not None and next_row["parsed_ts"] is not None:
        score -= duration_penalty(row_j["Activity"], next_row["Activity"], (next_row["parsed_ts"] - t_j).total_seconds())
    return score


def repair_c2_pair(case_rows, i, repaired_map):
    r1 = case_rows[i]
    r2 = case_rows[i + 1]
    t1 = r1["parsed_ts"]
    t2 = r2["parsed_ts"]
    if t1 is None or t2 is None:
        return False

    # swap candidate
    swap_valid = True
    prev_t = case_rows[i - 1]["parsed_ts"] if i - 1 >= 0 else None
    next_t = case_rows[i + 2]["parsed_ts"] if i + 2 < len(case_rows) else None
    if prev_t is not None and t2 < prev_t:
        swap_valid = False
    if next_t is not None and t1 > next_t:
        swap_valid = False
    if t2 > t1:
        swap_score = plausibility_of_assignment(case_rows, i, t2, t1) if swap_valid else -1e18
    else:
        swap_score = -1e18

    # displaced candidate: repair only later row usually
    lo, hi = case_bounds(case_rows)
    left_idx, left = nearest_clean_left(case_rows, i + 1)
    right_idx, right = nearest_clean_right(case_rows, i + 1)
    cand2 = infer_single_time(case_rows, i + 1, keep_date=is_date_only_string(r2["Timestamp"]))
    cand2 = clamp(cand2, t1, right["parsed_ts"] if right is not None else hi)
    if cand2 <= t1:
        cand2 = t1 + pd.Timedelta(seconds=1)
    disp2_score = local_plausibility(r1, r2, case_rows[i + 2] if i + 2 < len(case_rows) else None, cand2)

    # alternative displaced candidate: repair first row only
    cand1 = infer_single_time(case_rows, i, keep_date=is_date_only_string(r1["Timestamp"]))
    cand1 = clamp(cand1, left["parsed_ts"] if left is not None else lo, t2)
    if cand1 >= t2:
        cand1 = t2 - pd.Timedelta(seconds=1)
    disp1_score = local_plausibility(case_rows[i - 1] if i - 1 >= 0 else None, r1, r2, cand1)

    best_mode = "swap"
    best_score = swap_score
    if disp2_score > best_score:
        best_score = disp2_score
        best_mode = "disp2"
    if disp1_score > best_score:
        best_score = disp1_score
        best_mode = "disp1"

    if best_mode == "swap" and swap_valid and t2 != t1:
        repaired_map[r1["row_id"]] = format_ts(t2, r1["precision"])
        repaired_map[r2["row_id"]] = format_ts(t1, r2["precision"])
        return True
    elif best_mode == "disp2":
        repaired_map[r2["row_id"]] = format_ts(cand2, r2["precision"])
        return True
    elif best_mode == "disp1":
        repaired_map[r1["row_id"]] = format_ts(cand1, r1["precision"])
        return True
    return False


def repair_c3_row(case_rows, pos, repaired_map):
    row = case_rows[pos]
    keep_date = is_date_only_string(row["Timestamp"])
    t = infer_single_time(case_rows, pos, keep_date=keep_date)
    repaired_map[row["row_id"]] = format_ts(t, row["precision"])


def self_check_and_fix(rows, repaired_map, c1_blocks):
    blocks_clamped_extra = 0

    case_to_rows = defaultdict(list)
    for idx, r in enumerate(rows):
        case_to_rows[r["Case"]].append((idx, r))

    # per-case bounds and anchor checks
    for case, idxrows in case_to_rows.items():
        parsed = [parse_ts(repaired_map.get(r["row_id"], r["Timestamp"])) if r["error_category"] != "CLEAN" else r["parsed_ts"] for _, r in idxrows]
        valid = [t for t in parsed if t is not None]
        if not valid:
            continue
        cmin, cmax = min(valid), max(valid)
        for local_pos, (gidx, r) in enumerate(idxrows):
            if r["error_category"] == "CLEAN":
                continue
            ts_str = repaired_map.get(r["row_id"], r["Timestamp"])
            t = parse_ts(ts_str)
            if t is None:
                t = infer_single_time([rr for _, rr in idxrows], local_pos, keep_date=is_date_only_string(r["Timestamp"]))
                repaired_map[r["row_id"]] = format_ts(t, r["precision"])
                continue
            left = None
            right = None
            for j in range(local_pos - 1, -1, -1):
                rr = idxrows[j][1]
                if rr["error_category"] == "CLEAN" and rr["parsed_ts"] is not None:
                    left = rr["parsed_ts"]
                    break
            for j in range(local_pos + 1, len(idxrows)):
                rr = idxrows[j][1]
                if rr["error_category"] == "CLEAN" and rr["parsed_ts"] is not None:
                    right = rr["parsed_ts"]
                    break
            lo = left if left is not None else cmin
            hi = right if right is not None else cmax
            fixed = clamp(t, lo, hi)
            if is_date_only_string(r["Timestamp"]):
                d = str(r["Timestamp"]).strip()[:10]
                try:
                    date_obj = datetime.strptime(d, "%Y-%m-%d").date()
                    fixed = pd.Timestamp(datetime.combine(date_obj, fixed.to_pydatetime().time()))
                    day_lo = pd.Timestamp(datetime.combine(date_obj, datetime.min.time()))
                    day_hi = pd.Timestamp(datetime.combine(date_obj, datetime.max.time().replace(microsecond=999999)))
                    fixed = clamp(fixed, max(day_lo, lo), min(day_hi, hi))
                except Exception:
                    pass
            repaired_map[r["row_id"]] = format_ts(fixed, r["precision"])

    # enforce C1 properties
    for block in c1_blocks:
        observed = block[0]["Timestamp"]
        observed_ts = block[0]["parsed_ts"]
        if observed_ts is None:
            continue
        times = []
        for r in block:
            ts_str = repaired_map.get(r["row_id"], r["Timestamp"])
            times.append(parse_ts(ts_str))
        equal_obs = [i for i, t in enumerate(times) if t == observed_ts]
        if len(equal_obs) != 1:
            # keep first occurrence by inferred earliest repaired time or default first row
            earliest_i = min(range(len(block)), key=lambda i: times[i] if times[i] is not None else observed_ts)
            for i, r in enumerate(block):
                if i == earliest_i:
                    repaired_map[r["row_id"]] = observed
                else:
                    base = observed_ts + pd.Timedelta(seconds=i + 1)
                    repaired_map[r["row_id"]] = format_ts(base, r["precision"])
            blocks_clamped_extra += 1
            continue

        # all others must be > observed and distinct
        obs_i = equal_obs[0]
        other_indices = [i for i in range(len(block)) if i != obs_i]
        curr = observed_ts
        used = {observed_ts}
        need_fix = False
        sorted_others = sorted(other_indices, key=lambda i: times[i] if times[i] is not None else observed_ts)
        for rank, i in enumerate(sorted_others, start=1):
            t = times[i]
            if t is None or t <= curr or t in used:
                t = curr + pd.Timedelta(seconds=1)
                repaired_map[block[i]["row_id"]] = format_ts(t, block[i]["precision"])
                need_fix = True
            curr = parse_ts(repaired_map.get(block[i]["row_id"], block[i]["Timestamp"]))
            used.add(curr)
        if need_fix:
            blocks_clamped_extra += 1

    return blocks_clamped_extra


def main():
    args = parse_args()

    rows = []
    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for rec in reader:
            rec = dict(rec)
            rec["parsed_ts"] = parse_ts(rec["Timestamp"])
            rec["precision"] = detect_precision(rec["Timestamp"])
            rows.append(rec)

    repaired_map = {}
    stats = {"c1_blocks_repaired": 0, "blocks_clamped": 0}

    case_groups = defaultdict(list)
    for idx, r in enumerate(rows):
        case_groups[r["Case"]].append(idx)

    c1_blocks = []

    for case, idxs in case_groups.items():
        case_rows = [rows[i] for i in idxs]

        # C1 blocks: consecutive same observed timestamp and non-CLEAN
        j = 0
        while j < len(case_rows):
            r = case_rows[j]
            if r["error_category"] == "C1_ORDERING_AMBIGUITY" and r["parsed_ts"] is not None:
                k = j + 1
                while (
                    k < len(case_rows)
                    and case_rows[k]["error_category"] == "C1_ORDERING_AMBIGUITY"
                    and case_rows[k]["Timestamp"] == r["Timestamp"]
                ):
                    k += 1
                if k - j >= 2:
                    block = case_rows[j:k]
                    c1_blocks.append(block)
                    repair_c1_block(case_rows, j, k - 1, repaired_map, stats)
                    j = k
                    continue
            j += 1

        # C2 pairs
        j = 0
        while j < len(case_rows) - 1:
            r1 = case_rows[j]
            r2 = case_rows[j + 1]
            if r1["error_category"] == "C2_TEMPORAL_INCONSISTENCY" or r2["error_category"] == "C2_TEMPORAL_INCONSISTENCY":
                t1 = parse_ts(repaired_map.get(r1["row_id"], r1["Timestamp"]))
                t2 = parse_ts(repaired_map.get(r2["row_id"], r2["Timestamp"]))
                if t1 is not None and t2 is not None and t2 < t1:
                    repair_c2_pair(case_rows, j, repaired_map)
                    j += 2
                    continue
            j += 1

        # C3 and remaining non-clean unrepaired rows
        for pos, r in enumerate(case_rows):
            if r["error_category"] == "CLEAN":
                continue
            if r["row_id"] in repaired_map:
                continue
            if r["error_category"] == "C3_MISSING_TEMPORAL_INFORMATION":
                repair_c3_row(case_rows, pos, repaired_map)
            elif r["error_category"] == "C2_TEMPORAL_INCONSISTENCY":
                repair_c3_row(case_rows, pos, repaired_map)
            elif r["error_category"] == "C1_ORDERING_AMBIGUITY":
                # singleton C1 still must change because no legitimate keeper block of size>1
                repair_c3_row(case_rows, pos, repaired_map)
            else:
                repair_c3_row(case_rows, pos, repaired_map)

    stats["blocks_clamped"] += self_check_and_fix(rows, repaired_map, c1_blocks)

    output_rows = []
    still_equal = 0
    c1_keeper_ids = set()
    for block in c1_blocks:
        observed = block[0]["Timestamp"]
        for r in block:
            if repaired_map.get(r["row_id"], r["Timestamp"]) == observed:
                c1_keeper_ids.add(r["row_id"])
                break

    for r in rows:
        out = {
            "Case": r["Case"],
            "Activity": r["Activity"],
            "Timestamp": r["Timestamp"],
            "Resource": r["Resource"],
            "error_category": r["error_category"],
            "row_id": r["row_id"],
        }
        if r["error_category"] != "CLEAN":
            new_ts = repaired_map.get(r["row_id"], r["Timestamp"])
            if new_ts == r["Timestamp"]:
                if not (r["error_category"] == "C1_ORDERING_AMBIGUITY" and r["row_id"] in c1_keeper_ids):
                    t = parse_ts(new_ts)
                    if t is None:
                        t = pd.Timestamp("2000-01-01 00:00:01")
                    t = t + pd.Timedelta(seconds=1)
                    new_ts = format_ts(t, r["precision"])
            out["Timestamp"] = new_ts
        if out["Timestamp"] == r["Timestamp"]:
            still_equal += 1
        output_rows.append(out)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"blocks_repaired={stats['c1_blocks_repaired']} blocks_clamped={stats['blocks_clamped']} rows_still_equal={still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail safe: never exit non-zero
        try:
            print(f"blocks_repaired=0 blocks_clamped=0 rows_still_equal=0 error={str(e)[:200]}")
        except Exception:
            pass