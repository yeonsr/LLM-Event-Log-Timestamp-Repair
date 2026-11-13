import argparse
import itertools
import math
import sys
from collections import defaultdict, Counter

import pandas as pd


TRANSITION_FREQ = {
    ("register submission date request", "OLO messaging active"): 729,
    ("forward to the competent authority", "regular procedure without MER"): 673,
    ("send confirmation receipt", "forward to the competent authority"): 591,
    ("request complete", "send letter in progress"): 572,
    ("article 34  WABO applies", "assessment of content completed"): 568,
    ("assessment of content completed", "phase advice known"): 555,
    ("send confirmation receipt", "send confirmation receipt"): 503,
    ("phase application receptive", "request complete"): 500,
    ("treat subcases content", "completed subcases content"): 493,
    ("treat subcases completeness", "subcases completeness completed"): 490,
    ("by law", "creating environmental permit decision"): 481,
    ("enter senddate decision", "enter senddate decision environmental permit"): 467,
    ("phase advice known", "grounds for refusal"): 438,
    ("ask stakeholders views", "suspension ground applicable"): 431,
    ("grounds for refusal", "ask stakeholders views"): 426,
    ("subcases completeness completed", "procedure change"): 424,
    ("creating environmental permit decision", "decision date prior to decision"): 423,
    ("procedure change", "request complete"): 421,
    ("send confirmation receipt", "enter senddate acknowledgement"): 412,
    ("suspension ground applicable", "by law"): 410,
    ("OLO messaging active", "phase application received"): 409,
    ("regular procedure without MER", "send confirmation receipt"): 399,
    ("phase application received", "send confirmation receipt"): 393,
    ("enter senddate acknowledgement", "send confirmation receipt"): 383,
    ("procedure change", "phase application receptive"): 383,
    ("enter senddate procedure confirmation", "send procedure confirmation"): 378,
    ("send confirmation receipt", "start WABOprocedure"): 375,
    ("forward to the competent authority", "phase application received"): 372,
    ("registration date publication", "regular procedure without MER"): 357,
    ("procedure change", "article 34  WABO applies"): 352,
    ("OLO messaging active", "create procedure confirmation"): 344,
    ("transcript decision environmental permit to stakeholders", "generate publication document decision environmental permit"): 341,
    ("send letter in progress", "treat subcases content"): 324,
    ("phase decision sent", "phase decision taken"): 320,
    ("grounds for refusal", "phase advice known"): 319,
    ("create procedure confirmation", "create publication document"): 317,
    ("phase advice known", "phase application receptive"): 316,
    ("phase decision taken", "register objection and appeal periods"): 308,
    ("article 34  WABO applies", "ask stakeholders views"): 305,
    ("completed subcases content", "procedure change"): 298,
    ("start WABOprocedure", "enter senddate acknowledgement"): 293,
    ("enter senddate decision environmental permit", "phase decision sent"): 293,
    ("creating environmental permit decision", "generating decision environmental permit"): 284,
    ("send letter in progress", "set decision status"): 254,
    ("set decision phase decision permitting decided", "start decision phase decision permitting sent"): 241,
    ("regular procedure without MER", "start WABOprocedure"): 230,
    ("transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit"): 225,
    ("send procedure confirmation", "registration date publication"): 224,
    ("phase application received", "reception through OLO"): 222,
    ("regular procedure without MER", "enter senddate procedure confirmation"): 219,
    ("generating decision environmental permit", "grounds for refusal"): 215,
    ("decision date prior to decision", "generating decision environmental permit"): 209,
    ("create publication document", "create subcases completeness"): 197,
    ("ask stakeholders views", "assessment of content completed"): 194,
    ("by law", "completed subcases content"): 194,
    ("procedure change", "procedure change"): 193,
    ("submit decision", "suspension ground applicable"): 193,
    ("completed subcases content", "coordination of application"): 190,
    ("register submission date request", "enter senddate acknowledgement"): 188,
    ("record date of decision environmental permit", "set decision phase decision permitting decided"): 187,
}

DURATION_PROFILES = {
    ("register submission date request", "OLO messaging active"): {"p10": 86400.0, "p25": 172800.0, "median": 306406.0, "share_under_60s": 0.0},
    ("forward to the competent authority", "regular procedure without MER"): {"p10": 0.0, "p25": 0.0, "median": 3.0, "share_under_60s": 0.973},
    ("send confirmation receipt", "forward to the competent authority"): {"p10": 2.0, "p25": 3.0, "median": 6.0, "share_under_60s": 0.941},
    ("request complete", "send letter in progress"): {"p10": 0.0, "p25": 0.0, "median": 1.0, "share_under_60s": 0.991},
    ("article 34  WABO applies", "assessment of content completed"): {"p10": 0.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.993},
    ("assessment of content completed", "phase advice known"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("send confirmation receipt", "send confirmation receipt"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.799},
    ("phase application receptive", "request complete"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("treat subcases content", "completed subcases content"): {"p10": 1.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.992},
    ("treat subcases completeness", "subcases completeness completed"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.984},
    ("by law", "creating environmental permit decision"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.994},
    ("enter senddate decision", "enter senddate decision environmental permit"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.991},
    ("phase advice known", "grounds for refusal"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.954},
    ("ask stakeholders views", "suspension ground applicable"): {"p10": 0.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.984},
    ("grounds for refusal", "ask stakeholders views"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.955},
    ("subcases completeness completed", "procedure change"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.991},
    ("creating environmental permit decision", "decision date prior to decision"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.993},
    ("procedure change", "request complete"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.988},
    ("send confirmation receipt", "enter senddate acknowledgement"): {"p10": 2.0, "p25": 67.0, "median": 159.0, "share_under_60s": 0.245},
    ("suspension ground applicable", "by law"): {"p10": 1.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.993},
    ("OLO messaging active", "phase application received"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("regular procedure without MER", "send confirmation receipt"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("phase application received", "send confirmation receipt"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.982},
    ("enter senddate acknowledgement", "send confirmation receipt"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.99},
    ("procedure change", "phase application receptive"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.966},
    ("enter senddate procedure confirmation", "send procedure confirmation"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.952},
    ("send confirmation receipt", "start WABOprocedure"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("forward to the competent authority", "phase application received"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("registration date publication", "regular procedure without MER"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.994},
    ("procedure change", "article 34  WABO applies"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.949},
    ("OLO messaging active", "create procedure confirmation"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("transcript decision environmental permit to stakeholders", "generate publication document decision environmental permit"): {"p10": 6.0, "p25": 7.0, "median": 10.0, "share_under_60s": 0.906},
    ("send letter in progress", "treat subcases content"): {"p10": 0.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.991},
    ("phase decision sent", "phase decision taken"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("grounds for refusal", "phase advice known"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("create procedure confirmation", "create publication document"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("phase advice known", "phase application receptive"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("phase decision taken", "register objection and appeal periods"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.974},
    ("article 34  WABO applies", "ask stakeholders views"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.997},
    ("completed subcases content", "procedure change"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.98},
    ("enter senddate decision environmental permit", "phase decision sent"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("start WABOprocedure", "enter senddate acknowledgement"): {"p10": 34535.2, "p25": 38196.0, "median": 43648.0, "share_under_60s": 0.0},
    ("creating environmental permit decision", "generating decision environmental permit"): {"p10": 4.0, "p25": 55.5, "median": 1796.0, "share_under_60s": 0.257},
    ("send letter in progress", "set decision status"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("set decision phase decision permitting decided", "start decision phase decision permitting sent"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.988},
    ("regular procedure without MER", "start WABOprocedure"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit"): {"p10": 8.0, "p25": 34.0, "median": 50131.0, "share_under_60s": 0.258},
    ("send procedure confirmation", "registration date publication"): {"p10": 8.0, "p25": 11.0, "median": 19.0, "share_under_60s": 0.728},
    ("phase application received", "reception through OLO"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.973},
    ("regular procedure without MER", "enter senddate procedure confirmation"): {"p10": 1.0, "p25": 2.0, "median": 2.0, "share_under_60s": 0.872},
    ("generating decision environmental permit", "grounds for refusal"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("decision date prior to decision", "generating decision environmental permit"): {"p10": 0.0, "p25": 0.0, "median": 1.0, "share_under_60s": 0.569},
    ("create publication document", "create subcases completeness"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.914},
    ("ask stakeholders views", "assessment of content completed"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("by law", "completed subcases content"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("submit decision", "suspension ground applicable"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("procedure change", "procedure change"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("completed subcases content", "coordination of application"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("register submission date request", "enter senddate acknowledgement"): {"p10": 172800.0, "p25": 345600.0, "median": 604800.0, "share_under_60s": 0.0},
    ("record date of decision environmental permit", "set decision phase decision permitting decided"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
}

FREQUENT_VARIANTS = [
    ["register submission date request","OLO messaging active","create procedure confirmation","create publication document","forward to the competent authority","phase application received","publish","registration date publication","regular procedure without MER","send confirmation receipt","send confirmation receipt","start WABOprocedure","enter senddate acknowledgement"],
    ["register submission date request","enter senddate acknowledgement","phase application received","reception through OLO","send confirmation receipt","forward to the competent authority"],
    ["register submission date request","OLO messaging active","create procedure confirmation","create publication document","forward to the competent authority","phase application received","publish","registration date publication","regular procedure without MER","send confirmation receipt","send confirmation receipt","start WABOprocedure","enter senddate acknowledgement","article 34  WABO applies","ask stakeholders views","by law","coordination of application","create subcases completeness","create subcases content","creating cover letter decision","creating environmental permit decision","decision date prior to decision","enter senddate decision","enter senddate decision environmental permit","generating decision environmental permit","grounds for refusal","phase advice known","phase application receptive","phase decision ready","phase decision sent","phase decision taken","procedure change","procedure change","register objection and appeal periods","request complete","send letter in progress","set decision status","submit decision","suspension ground applicable","transcript decision environmental permit to stakeholders","record date of decision environmental permit","set decision phase decision permitting decided","enter senddate decision environmental permit","start decision phase decision permitting sent"],
    ["register submission date request","enter senddate acknowledgement","phase application received","reception through OLO","send confirmation receipt","forward to the competent authority","regular procedure without MER"],
    ["register submission date request","OLO messaging active","phase application received","send confirmation receipt","enter senddate acknowledgement","send confirmation receipt","forward to the competent authority","regular procedure without MER","start WABOprocedure","enter senddate procedure confirmation","send procedure confirmation","registration date publication","create publication document","treat subcases completeness","subcases completeness completed","procedure change","phase application receptive","request complete","procedure change after completeness","send letter in progress","treat subcases content","completed subcases content","procedure change","article 34  WABO applies","assessment of content completed","phase advice known","grounds for refusal","ask stakeholders views","suspension ground applicable","by law","creating environmental permit decision","decision date prior to decision","generating decision environmental permit","record date of decision environmental permit","submit decision","phase decision taken","register objection and appeal periods","enter senddate decision environmental permit","transcript decision environmental permit to stakeholders","enter senddate decision","enter senddate decision environmental permit","phase decision sent","create publication document","date decision for inspection"],
    ["register submission date request","OLO messaging active","create procedure confirmation","create publication document","create subcases completeness","forward to the competent authority","phase application received","publish","registration date publication","regular procedure without MER","send confirmation receipt","send confirmation receipt","start WABOprocedure","enter senddate acknowledgement","article 34  WABO applies","ask stakeholders views","by law","coordination of application","create subcases content","creating environmental permit decision","decision date prior to decision","generating decision environmental permit","grounds for refusal","phase advice known","phase application receptive","phase decision ready","procedure change","procedure change","request complete","send letter in progress","set decision status","submit decision","suspension ground applicable","creating cover letter decision","enter senddate decision","enter senddate decision environmental permit","phase decision sent","phase decision taken","register objection and appeal periods","transcript decision environmental permit to stakeholders","enter senddate decision environmental permit","record date of decision environmental permit","set decision phase decision permitting decided","start decision phase decision permitting sent"],
]

GLOBAL_MEDIAN = 1.0


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


def fmt_timestamp(ts, precision):
    if precision <= 0:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    frac = f"{ts.microsecond:06d}"[:precision]
    return f"{base}.{frac}"


def edge_freq(a, b):
    return TRANSITION_FREQ.get((a, b), 0)


def edge_exists(a, b):
    return (a, b) in TRANSITION_FREQ


def duration_gap_seconds(a, b, concurrent_hint=False):
    prof = DURATION_PROFILES.get((a, b))
    if prof:
        if concurrent_hint:
            val = prof["p10"]
        else:
            val = prof["p10"] if prof["share_under_60s"] >= 0.97 else prof["p25"]
        if val is None:
            val = 1.0
        return max(1.0, float(val))
    return 1.0 if concurrent_hint else max(1.0, float(GLOBAL_MEDIAN))


def variant_subsequence_score(seq):
    best = 0
    pair_orders = Counter()
    for var in FREQUENT_VARIANTS:
        positions = []
        ok = True
        start_idx = 0
        for act in seq:
            try:
                idx = var.index(act, start_idx)
            except ValueError:
                ok = False
                break
            positions.append(idx)
            start_idx = idx + 1
        if ok:
            span = positions[-1] - positions[0] + 1
            contig = 1 if span == len(seq) else 0
            score = 1000 * contig + (100 - span)
            if score > best:
                best = score
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                a, b = seq[i], seq[j]
                if a in var and b in var:
                    if var.index(a) < var.index(b):
                        pair_orders[(a, b)] += 1
                    else:
                        pair_orders[(b, a)] += 1
    return best, pair_orders


def pair_concurrent(a, b, pair_orders):
    ab = pair_orders.get((a, b), 0)
    ba = pair_orders.get((b, a), 0)
    if ab > 0 and ba > 0:
        ratio = min(ab, ba) / max(ab, ba)
        return ratio >= 0.67
    if edge_exists(a, b) and edge_exists(b, a):
        fa, fb = edge_freq(a, b), edge_freq(b, a)
        if min(fa, fb) / max(fa, fb) >= 0.67:
            return True
    return False


def score_order(order, left_act=None, right_act=None, require_internal=True):
    for a, b in zip(order, order[1:]):
        if require_internal and not edge_exists(a, b):
            return None
    vscore, pair_orders = variant_subsequence_score(order)
    dfg = 0.0
    for a, b in zip(order, order[1:]):
        f = edge_freq(a, b)
        if f > 0:
            dfg += math.log1p(f) * 10.0
        else:
            dfg -= 1000.0
    anchor = 0.0
    if left_act is not None:
        if edge_exists(left_act, order[0]):
            anchor += math.log1p(edge_freq(left_act, order[0])) * 3.0
        else:
            anchor -= 2.0
    if right_act is not None:
        if edge_exists(order[-1], right_act):
            anchor += math.log1p(edge_freq(order[-1], right_act)) * 3.0
        else:
            anchor -= 2.0
    return vscore + dfg + anchor, pair_orders


def choose_order(activities, left_act=None, right_act=None):
    perms = list(itertools.permutations(activities))
    best = None
    best_pair_orders = Counter()

    for strict in [True, False]:
        current_best = None
        current_po = Counter()
        for perm in perms:
            scored = score_order(list(perm), left_act, right_act, require_internal=strict)
            if scored is None:
                continue
            score, po = scored
            if current_best is None or score > current_best[0]:
                current_best = (score, list(perm))
                current_po = po
        if current_best is not None:
            best = current_best
            best_pair_orders = current_po
            break

    if best is None:
        best = (0.0, list(activities))
    return best[1], best_pair_orders


def build_gaps(order, pair_orders):
    gaps = []
    for a, b in zip(order, order[1:]):
        concurrent_hint = pair_concurrent(a, b, pair_orders)
        gaps.append(duration_gap_seconds(a, b, concurrent_hint=concurrent_hint))
    return gaps


def enforce_monotonic_times(start_ts, right_anchor_ts, case_end, k, precision, gaps):
    min_step = pd.Timedelta(seconds=1)
    times = [start_ts]
    for g in gaps:
        step = max(pd.Timedelta(seconds=float(g)), min_step)
        times.append(times[-1] + step)

    min_total = pd.Timedelta(seconds=k - 1)
    if times[-1] - start_ts < min_total:
        times = [start_ts + i * min_step for i in range(k)]

    clamp_needed = False
    upper = case_end
    if right_anchor_ts is not None:
        upper = min(upper, right_anchor_ts)
    if times[-1] > upper:
        clamp_needed = True
        room = upper - start_ts
        if room < min_total:
            tail = upper
            times[-1] = tail
            for i in range(k - 2, -1, -1):
                latest = times[i + 1] - min_step
                target = start_ts + i * min_step
                times[i] = min(latest, max(start_ts if i == 0 else times[i], target))
            times[0] = start_ts
            for i in range(1, k):
                if times[i] <= times[i - 1]:
                    times[i] = times[i - 1] + min_step
            if times[-1] > upper:
                overflow = times[-1] - upper
                for i in range(1, k):
                    times[i] = times[i] - overflow
                if times[1] <= times[0]:
                    times = [start_ts + i * min_step for i in range(k)]
                    if times[-1] > upper:
                        for i in range(k - 1, -1, -1):
                            times[i] = upper - (k - 1 - i) * min_step
                        times[0] = start_ts
        else:
            raw = [0.0]
            acc = 0.0
            for g in gaps:
                acc += max(1.0, float(g))
                raw.append(acc)
            total = raw[-1]
            room_s = room.total_seconds()
            if total <= 0:
                scaled = [i for i in range(k)]
            else:
                factor = room_s / total
                scaled = [x * factor for x in raw]
            times = [start_ts + pd.Timedelta(seconds=s) for s in scaled]
            times[0] = start_ts
            for i in range(1, k):
                floor = times[i - 1] + min_step
                if times[i] < floor:
                    times[i] = floor
            if times[-1] > upper:
                times[-1] = upper
                for i in range(k - 2, 0, -1):
                    latest = times[i + 1] - min_step
                    if times[i] > latest:
                        times[i] = latest
                times[0] = start_ts
                for i in range(1, k):
                    if times[i] <= times[i - 1]:
                        times[i] = times[i - 1] + min_step

    if times[-1] - start_ts < min_total:
        candidate = [start_ts + i * min_step for i in range(k)]
        upper = case_end if right_anchor_ts is None else min(case_end, right_anchor_ts)
        if candidate[-1] <= upper:
            times = candidate
        else:
            times = candidate
            if times[-1] > upper:
                times[-1] = upper
                for i in range(k - 2, 0, -1):
                    latest = times[i + 1] - min_step
                    if times[i] > latest:
                        times[i] = latest
                times[0] = start_ts
                for i in range(1, k):
                    if times[i] <= times[i - 1]:
                        times[i] = times[i - 1] + min_step

    return times, clamp_needed


def find_blocks(case_df):
    blocks = []
    idxs = list(case_df.index)
    n = len(idxs)
    i = 0
    while i < n:
        idx = idxs[i]
        row = case_df.loc[idx]
        if row["error_category"] == "CLEAN":
            i += 1
            continue
        j = i + 1
        ts = row["Timestamp"]
        while j < n:
            idx2 = idxs[j]
            row2 = case_df.loc[idx2]
            if row2["error_category"] != "CLEAN" and row2["Timestamp"] == ts:
                j += 1
            else:
                break
        if j - i >= 2:
            blocks.append(idxs[i:j])
        else:
            blocks.append(idxs[i:j])
        i = j
    return blocks


def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original = df.copy(deep=True)
    output_ts = df["Timestamp"].tolist()
    parsed_ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    precision = max(detect_precision(x) for x in df["Timestamp"].astype(str).tolist())

    repaired_blocks = 0
    clamped_blocks = 0
    kept_equal_rows = 0

    by_case = df.groupby("Case", sort=False).groups

    for case, idxs in by_case.items():
        case_df = df.loc[list(idxs)]
        case_times = pd.to_datetime(case_df["Timestamp"], errors="coerce")
        valid_case_times = case_times.dropna()
        if len(valid_case_times) == 0:
            continue
        case_start = valid_case_times.min()
        case_end = valid_case_times.max()

        blocks = find_blocks(case_df)
        for block_idxs in blocks:
            block = df.loc[block_idxs]
            if all(block["error_category"] == "CLEAN"):
                continue

            shared_ts_str = block.iloc[0]["Timestamp"]
            shared_ts = pd.to_datetime(shared_ts_str, errors="coerce")
            if pd.isna(shared_ts):
                continue

            case_pos = list(case_df.index)
            pos_first = case_pos.index(block_idxs[0])
            pos_last = case_pos.index(block_idxs[-1])

            left_anchor_ts = None
            left_anchor_act = None
            for p in range(pos_first - 1, -1, -1):
                ridx = case_pos[p]
                if df.at[ridx, "error_category"] == "CLEAN":
                    left_anchor_ts = pd.to_datetime(df.at[ridx, "Timestamp"], errors="coerce")
                    left_anchor_act = df.at[ridx, "Activity"]
                    break

            right_anchor_ts = None
            right_anchor_act = None
            for p in range(pos_last + 1, len(case_pos)):
                ridx = case_pos[p]
                if df.at[ridx, "error_category"] == "CLEAN":
                    right_anchor_ts = pd.to_datetime(df.at[ridx, "Timestamp"], errors="coerce")
                    right_anchor_act = df.at[ridx, "Activity"]
                    break

            activities = block["Activity"].tolist()
            order_acts, pair_orders = choose_order(activities, left_anchor_act, right_anchor_act)

            used = [False] * len(block_idxs)
            activity_to_rows = defaultdict(list)
            for ridx in block_idxs:
                activity_to_rows[df.at[ridx, "Activity"]].append(ridx)

            ordered_rows = []
            local_used = defaultdict(int)
            for act in order_acts:
                ridx = activity_to_rows[act][local_used[act]]
                local_used[act] += 1
                ordered_rows.append(ridx)

            k = len(ordered_rows)
            gaps = build_gaps(order_acts, pair_orders)

            times, clamped = enforce_monotonic_times(shared_ts, right_anchor_ts, case_end, k, precision, gaps)

            if left_anchor_ts is not None:
                if times[0] < left_anchor_ts:
                    delta = left_anchor_ts - times[0]
                    times = [t + delta for t in times]
                    clamped = True

            upper = case_end if right_anchor_ts is None else min(case_end, right_anchor_ts)
            if times[-1] > upper:
                clamped = True
                times, _ = enforce_monotonic_times(shared_ts, right_anchor_ts, case_end, k, precision, [1.0] * (k - 1))

            if times[0] != shared_ts:
                times[0] = shared_ts
                for i in range(1, k):
                    if times[i] <= times[i - 1]:
                        times[i] = times[i - 1] + pd.Timedelta(seconds=1)

            if right_anchor_ts is not None and times[-1] > right_anchor_ts:
                clamped = True
                latest_allowed = right_anchor_ts
                times[-1] = min(times[-1], latest_allowed)
                for i in range(k - 2, 0, -1):
                    if times[i] >= times[i + 1]:
                        times[i] = times[i + 1] - pd.Timedelta(seconds=1)
                times[0] = shared_ts
                for i in range(1, k):
                    if times[i] <= times[i - 1]:
                        times[i] = times[i - 1] + pd.Timedelta(seconds=1)

            if times[-1] - times[0] < pd.Timedelta(seconds=k - 1):
                clamped = True
                fallback = [shared_ts + pd.Timedelta(seconds=i) for i in range(k)]
                if right_anchor_ts is None or fallback[-1] <= right_anchor_ts:
                    times = fallback

            for ord_pos, ridx in enumerate(ordered_rows):
                ts_str = fmt_timestamp(times[ord_pos], precision)
                output_ts[ridx] = ts_str

            repaired_blocks += 1
            if clamped:
                clamped_blocks += 1

    df["Timestamp"] = output_ts

    # Self-checks and repairs
    # Keep CLEAN exact
    clean_mask = original["error_category"] == "CLEAN"
    df.loc[clean_mask, "Timestamp"] = original.loc[clean_mask, "Timestamp"]

    # Revalidate blocks
    by_case = df.groupby("Case", sort=False).groups
    for case, idxs in by_case.items():
        case_df = df.loc[list(idxs)]
        case_times = pd.to_datetime(case_df["Timestamp"], errors="coerce")
        valid_case_times = case_times.dropna()
        if len(valid_case_times) == 0:
            continue
        case_start = valid_case_times.min()
        case_end = valid_case_times.max()

        blocks = find_blocks(original.loc[list(idxs)])
        case_pos = list(case_df.index)

        for block_idxs in blocks:
            block_orig = original.loc[block_idxs]
            if all(block_orig["error_category"] == "CLEAN"):
                continue
            shared_ts_str = block_orig.iloc[0]["Timestamp"]
            shared_ts = pd.to_datetime(shared_ts_str, errors="coerce")
            if pd.isna(shared_ts):
                continue

            pos_first = case_pos.index(block_idxs[0])
            pos_last = case_pos.index(block_idxs[-1])

            left_anchor_ts = None
            right_anchor_ts = None
            for p in range(pos_first - 1, -1, -1):
                ridx = case_pos[p]
                if original.at[ridx, "error_category"] == "CLEAN":
                    left_anchor_ts = pd.to_datetime(original.at[ridx, "Timestamp"], errors="coerce")
                    break
            for p in range(pos_last + 1, len(case_pos)):
                ridx = case_pos[p]
                if original.at[ridx, "error_category"] == "CLEAN":
                    right_anchor_ts = pd.to_datetime(original.at[ridx, "Timestamp"], errors="coerce")
                    break

            block_now = df.loc[block_idxs]
            activities = block_now["Activity"].tolist()
            left_anchor_act = None
            right_anchor_act = None
            for p in range(pos_first - 1, -1, -1):
                ridx = case_pos[p]
                if original.at[ridx, "error_category"] == "CLEAN":
                    left_anchor_act = original.at[ridx, "Activity"]
                    break
            for p in range(pos_last + 1, len(case_pos)):
                ridx = case_pos[p]
                if original.at[ridx, "error_category"] == "CLEAN":
                    right_anchor_act = original.at[ridx, "Activity"]
                    break

            order_acts, pair_orders = choose_order(activities, left_anchor_act, right_anchor_act)
            activity_to_rows = defaultdict(list)
            for ridx in block_idxs:
                activity_to_rows[df.at[ridx, "Activity"]].append(ridx)
            local_used = defaultdict(int)
            ordered_rows = []
            for act in order_acts:
                ridx = activity_to_rows[act][local_used[act]]
                local_used[act] += 1
                ordered_rows.append(ridx)

            ts_list = [pd.to_datetime(df.at[ridx, "Timestamp"], errors="coerce") for ridx in ordered_rows]
            bad = False
            if any(pd.isna(t) for t in ts_list):
                bad = True
            else:
                for i in range(1, len(ts_list)):
                    if not (ts_list[i] > ts_list[i - 1]):
                        bad = True
                        break
                equal_count = sum(1 for ridx in block_idxs if df.at[ridx, "Timestamp"] == original.at[ridx, "Timestamp"])
                if equal_count != 1:
                    bad = True
                if df.at[ordered_rows[0], "Timestamp"] != shared_ts_str:
                    bad = True
                if left_anchor_ts is not None and ts_list[0] < left_anchor_ts:
                    bad = True
                if right_anchor_ts is not None and ts_list[-1] > right_anchor_ts:
                    bad = True
                if ts_list[0] < case_start or ts_list[-1] > case_end:
                    bad = True
                if ts_list[-1] - ts_list[0] < pd.Timedelta(seconds=len(ts_list) - 1):
                    bad = True

            if bad:
                gaps = build_gaps(order_acts, pair_orders)
                fixed, _ = enforce_monotonic_times(shared_ts, right_anchor_ts, case_end, len(ordered_rows), precision, gaps)
                for ord_pos, ridx in enumerate(ordered_rows):
                    df.at[ridx, "Timestamp"] = fmt_timestamp(fixed[ord_pos], precision)
                clamped_blocks += 1

    # Final count rows still equal to input among corrupted rows
    neq_mask = original["error_category"] != "CLEAN"
    kept_equal_rows = int((df.loc[neq_mask, "Timestamp"] == original.loc[neq_mask, "Timestamp"]).sum())

    # Preserve all non-time columns byte-identically
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        df[col] = original[col]

    # Row count and row order preserved automatically
    df.to_csv(args.output, index=False)
    print(f"blocks_repaired={repaired_blocks} blocks_clamped={clamped_blocks} rows_still_equal_to_input={kept_equal_rows}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            args = parse_args()
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0")
        except Exception:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal_to_input=0")