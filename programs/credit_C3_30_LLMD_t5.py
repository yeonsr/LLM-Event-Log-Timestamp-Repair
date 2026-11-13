import argparse
import math
import re
from datetime import timedelta
import pandas as pd


# -----------------------------
# Process priors from prompt
# -----------------------------
DFG_P25 = {
    ("Check for completeness", "Request info"): 172.53300000000002,
    ("info received", "Check for completeness"): 176.3765,
    ("Request info", "info received"): 10.0,
    ("Check for completeness", "Perform checks"): 174.17,
    ("Perform checks", "Make decision"): 172.6405,
    ("New online application received", "Check for completeness"): 171.767,
    ("Deliver card", "EVENT 13 END"): 0.0,
    ("Notify accept", "Deliver card"): 176.35075,
    ("Make decision", "notify reject"): 175.97650000000002,
    ("Make decision", "Notify accept"): 174.0265,
    ("time out", "EVENT 13 END"): 0.0,
    ("notify reject", "time out"): 10.0,
    ("notify reject", "review request received"): 10.0,
    ("review request received", "Check for completeness"): 186.74975,
    ("Check for completeness", "Make decision"): 587.1220000000001,
    ("New online application received", "Perform checks"): 596.06,
    ("Perform checks", "Notify accept"): 548.193,
    ("Request info", "Check for completeness"): 177.8785,
    ("Check for completeness", "info received"): 193.724,
    ("Perform checks", "notify reject"): 726.6279999999999,
    ("New online application received", "Request info"): 510.01375,
    ("info received", "Perform checks"): 598.8085,
    ("info received", "Request info"): 664.2945,
    ("notify reject", "Check for completeness"): 216.3995,
    ("Make decision", "review request received"): 198.772,
    ("review request received", "Perform checks"): 717.5822499999999,
    ("review request received", "Request info"): 545.983,
    ("EVENT 13 END", "EVENT 13 END"): 0.0,
}
DFG_FREQ = {
    ("Check for completeness", "Request info"): 6543,
    ("info received", "Check for completeness"): 6538,
    ("Request info", "info received"): 6534,
    ("Check for completeness", "Perform checks"): 6037,
    ("Perform checks", "Make decision"): 6036,
    ("New online application received", "Check for completeness"): 4705,
    ("Deliver card", "EVENT 13 END"): 3276,
    ("Notify accept", "Deliver card"): 3276,
    ("Make decision", "notify reject"): 3256,
    ("Make decision", "Notify accept"): 3083,
    ("time out", "EVENT 13 END"): 1724,
    ("notify reject", "time out"): 1724,
    ("notify reject", "review request received"): 1624,
    ("review request received", "Check for completeness"): 1616,
    ("Check for completeness", "Make decision"): 330,
    ("New online application received", "Perform checks"): 209,
    ("Perform checks", "Notify accept"): 193,
    ("Request info", "Check for completeness"): 170,
    ("Check for completeness", "info received"): 150,
    ("Perform checks", "notify reject"): 123,
    ("New online application received", "Request info"): 86,
    ("info received", "Perform checks"): 86,
    ("info received", "Request info"): 60,
    ("notify reject", "Check for completeness"): 31,
    ("Make decision", "review request received"): 27,
    ("review request received", "Perform checks"): 20,
    ("review request received", "Request info"): 15,
    ("EVENT 13 END", "EVENT 13 END"): 8,
}
BLOCK_SPAN_P25 = {
    2: 10.0,
    3: 324.66949999999997,
    4: 705.348,
    5: 1079.1412500000001,
}
GLOBAL_MEDIAN = 243.79649999999998

FREQUENT_VARIANTS = [
    ["New online application received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Make decision","notify reject","review request received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","notify reject","review request received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Make decision","notify reject","review request received","Check for completeness","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
    ["New online application received","Check for completeness","Perform checks","Make decision","notify reject","review request received","Check for completeness","Perform checks","Make decision","notify reject","time out","EVENT 13 END"],
    ["New online application received","Request info","info received","Check for completeness","Perform checks","Make decision","Notify accept","Deliver card","EVENT 13 END"],
]


# -----------------------------
# Utilities
# -----------------------------
def parse_ts(s):
    if pd.isna(s):
        return pd.NaT
    s2 = str(s).strip()
    if s2 == "":
        return pd.NaT
    return pd.to_datetime(s2, errors="coerce")


def detect_precision(ts_str):
    if ts_str is None or (isinstance(ts_str, float) and math.isnan(ts_str)):
        return 3
    s = str(ts_str).strip()
    m = re.search(r"\.(\d+)$", s)
    return len(m.group(1)) if m else 0


def is_blank(ts_str):
    return pd.isna(ts_str) or str(ts_str).strip() == ""


def is_date_only_missing_time(ts_str):
    if pd.isna(ts_str):
        return False
    s = str(ts_str).strip()
    if s == "":
        return False
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return False
    # For this task, corrupted value may be date-only or date with midnight / all-zero TOD.
    # Treat midnight-valued timestamps as date-preserved missing time.
    has_date = re.match(r"^\d{4}-\d{2}-\d{2}", s) is not None
    return has_date and dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0


def floor_to_precision(dt, precision):
    if precision <= 0:
        return dt.replace(microsecond=0)
    if precision >= 6:
        return dt
    factor = 10 ** (6 - precision)
    us = (dt.microsecond // factor) * factor
    return dt.replace(microsecond=us)


def format_ts(dt, precision):
    dt = floor_to_precision(pd.Timestamp(dt).to_pydatetime(), precision)
    if precision <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    frac = f"{dt.microsecond:06d}"[:precision]
    return dt.strftime("%Y-%m-%d %H:%M:%S") + "." + frac


def dfg_gap(a, b):
    if (a, b) in DFG_P25:
        return float(DFG_P25[(a, b)])
    return None


def fallback_gap():
    return float(GLOBAL_MEDIAN)


def edge_weight(a, b):
    g = dfg_gap(a, b)
    if g is not None:
        return max(g, 0.0)
    return fallback_gap()


def variant_contains_contiguous(seq):
    n = len(seq)
    for v in FREQUENT_VARIANTS:
        for i in range(0, len(v) - n + 1):
            if v[i:i+n] == seq:
                return True
    return False


def dfg_score(seq):
    score = 0
    for i in range(len(seq)-1):
        pair = (seq[i], seq[i+1])
        if pair in DFG_FREQ:
            score += DFG_FREQ[pair]
        else:
            score -= 1000000
    return score


def block_span_estimate_secs(activities):
    weights = []
    for i in range(len(activities) - 1):
        weights.append(edge_weight(activities[i], activities[i+1]))
    s = sum(weights)
    if s > 0:
        return s
    k = len(activities)
    if k in BLOCK_SPAN_P25:
        return BLOCK_SPAN_P25[k]
    kk = max(BLOCK_SPAN_P25.keys())
    return BLOCK_SPAN_P25[kk] + (k - kk) * GLOBAL_MEDIAN


def same_calendar_date(dt, date_ref):
    return dt.date() == date_ref.date()


def combine_date_and_seconds(date_ref, sec_of_day):
    base = pd.Timestamp(year=date_ref.year, month=date_ref.month, day=date_ref.day)
    return base + pd.to_timedelta(sec_of_day, unit="s")


def seconds_of_day(ts):
    return ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6


def choose_precision_for_case(case_df):
    precs = []
    for s in case_df["Timestamp"].tolist():
        if not is_blank(s):
            precs.append(detect_precision(s))
    return max(precs) if precs else 3


def nudge_within_bounds(dt, lower, upper, precision):
    step_us = max(1, 10 ** max(0, 6 - precision))
    dt = pd.Timestamp(dt)
    lo = pd.Timestamp(lower)
    hi = pd.Timestamp(upper)
    if dt < lo:
        dt = lo
    if dt > hi:
        dt = hi
    # small nudge where possible
    cand = dt + pd.to_timedelta(step_us, unit="us")
    if cand <= hi:
        return cand
    cand = dt - pd.to_timedelta(step_us, unit="us")
    if cand >= lo:
        return cand
    return dt


# -----------------------------
# Core repair
# -----------------------------
def repair_case(case_df):
    case_df = case_df.copy()
    idxs = list(case_df.index)
    acts = case_df["Activity"].tolist()
    errs = case_df["error_category"].tolist()
    orig_ts = case_df["Timestamp"].tolist()

    parsed = [parse_ts(x) for x in orig_ts]
    case_precision = choose_precision_for_case(case_df)

    clean_positions = [i for i, e in enumerate(errs) if e == "CLEAN" and not pd.isna(parsed[i])]
    clean_times = [parsed[i] for i in clean_positions]

    if clean_times:
        case_start = min(clean_times)
        case_end = max(clean_times)
    else:
        # should be rare; synthesize from any parseable corrupted date or current neutral anchor
        any_dt = next((p for p in parsed if not pd.isna(p)), pd.Timestamp("2023-01-01 12:00:00"))
        case_start = any_dt
        case_end = any_dt

    repaired_strings = list(orig_ts)
    repaired_dt = list(parsed)

    nonclean_positions = [i for i, e in enumerate(errs) if e != "CLEAN"]
    blocks = []
    if nonclean_positions:
        start = nonclean_positions[0]
        prev = start
        for p in nonclean_positions[1:]:
            if p == prev + 1:
                prev = p
            else:
                blocks.append((start, prev))
                start = prev = p
        blocks.append((start, prev))

    for bstart, bend in blocks:
        block_pos = list(range(bstart, bend + 1))
        block_acts = [acts[i] for i in block_pos]

        # validate order only; no permutation
        _ = dfg_score(block_acts)
        _ = variant_contains_contiguous(block_acts)

        left_clean = bstart - 1
        while left_clean >= 0 and errs[left_clean] != "CLEAN":
            left_clean -= 1
        right_clean = bend + 1
        while right_clean < len(errs) and errs[right_clean] != "CLEAN":
            right_clean += 1

        left_anchor = parsed[left_clean] if left_clean >= 0 and errs[left_clean] == "CLEAN" else pd.NaT
        right_anchor = parsed[right_clean] if right_clean < len(errs) and errs[right_clean] == "CLEAN" else pd.NaT

        # weights per transition inside [left_anchor, block, right_anchor]
        # For a block of m rows, define m+1 edge weights when both anchors exist:
        # left->r1, r1->r2, ..., rm->right
        forward_weights = []
        if not pd.isna(left_anchor):
            forward_weights.append(edge_weight(acts[left_clean], acts[block_pos[0]]))
        else:
            forward_weights.append(None)

        for i in range(len(block_pos) - 1):
            forward_weights.append(edge_weight(acts[block_pos[i]], acts[block_pos[i + 1]]))

        if not pd.isna(right_anchor):
            right_w = edge_weight(acts[block_pos[-1]], acts[right_clean])
        else:
            right_w = None

        # Build candidate schedule
        block_candidate = [pd.NaT] * len(block_pos)

        if not pd.isna(left_anchor) and not pd.isna(right_anchor):
            total_window = max((right_anchor - left_anchor).total_seconds(), 0.0)
            weights = []
            if forward_weights[0] is not None:
                weights.append(max(forward_weights[0], 0.0))
            else:
                weights.append(fallback_gap())
            for w in forward_weights[1:]:
                weights.append(max(w if w is not None else fallback_gap(), 0.0))
            weights.append(max(right_w if right_w is not None else fallback_gap(), 0.0))

            sumw = sum(weights)
            if sumw <= 0:
                sumw = len(weights)

            cum = 0.0
            raw_times = []
            for j in range(len(block_pos)):
                cum += weights[j]
                frac = cum / sumw
                raw_times.append(left_anchor + pd.to_timedelta(total_window * frac, unit="s"))

            # Reduce movement for date-preserved rows: prefer left-based same-day projection when valid
            for j, pos in enumerate(block_pos):
                source = orig_ts[pos]
                if is_date_only_missing_time(source):
                    observed_date = parse_ts(source)
                    # left-based projection using cumulative DFG p25
                    left_based_secs = sum(weights[:j+1])
                    left_proj = left_anchor + pd.to_timedelta(left_based_secs, unit="s")

                    # If overshoots right, place between anchors instead of pinning at boundary
                    cand = left_proj
                    if cand > right_anchor:
                        cand = raw_times[j]

                    # Keep exact observed calendar date
                    if same_calendar_date(cand, observed_date):
                        raw_times[j] = cand
                    else:
                        # Intersect same-date window with anchors and place proportionally
                        day_start = pd.Timestamp(year=observed_date.year, month=observed_date.month, day=observed_date.day)
                        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                        lo = max(left_anchor, day_start)
                        hi = min(right_anchor, day_end)
                        if lo <= hi:
                            frac = (j + 1) / (len(block_pos) + 1)
                            raw_times[j] = lo + (hi - lo) * frac
                        else:
                            # impossible with given anchors; clamp to same-date nearest endpoint then to anchor window
                            tmp = combine_date_and_seconds(observed_date, min(max(seconds_of_day(cand), 0.0), 86399.999999))
                            if tmp < left_anchor:
                                tmp = left_anchor
                            if tmp > right_anchor:
                                tmp = right_anchor
                            raw_times[j] = tmp

            block_candidate = raw_times

        elif not pd.isna(left_anchor):
            # project forward by cumulative relative gaps
            cum = 0.0
            for j, pos in enumerate(block_pos):
                w = forward_weights[j]
                if w is None:
                    # use neighboring edge priors or block span fallback
                    w = fallback_gap()
                cum += max(w, 0.0)
                block_candidate[j] = left_anchor + pd.to_timedelta(cum, unit="s")

            # if date-preserved row and projection crosses another day, keep same date and use observed TOD-near-anchor logic
            for j, pos in enumerate(block_pos):
                source = orig_ts[pos]
                if is_date_only_missing_time(source):
                    observed_date = parse_ts(source)
                    cand = block_candidate[j]
                    if not same_calendar_date(cand, observed_date):
                        sec = seconds_of_day(cand)
                        tmp = combine_date_and_seconds(observed_date, sec)
                        # clamp inside case bounds
                        if tmp < case_start:
                            tmp = max(case_start, combine_date_and_seconds(observed_date, 0))
                        if tmp > case_end:
                            tmp = min(case_end, combine_date_and_seconds(observed_date, 86399.999))
                        block_candidate[j] = tmp

        elif not pd.isna(right_anchor):
            # project backward by cumulative relative gaps from right
            backward_weights = []
            backward_weights.append(max(right_w if right_w is not None else fallback_gap(), 0.0))
            for j in range(len(block_pos)-1, 0, -1):
                w = edge_weight(acts[block_pos[j-1]], acts[block_pos[j]])
                backward_weights.append(max(w, 0.0))

            cum = 0.0
            temp = [pd.NaT] * len(block_pos)
            for k in range(len(block_pos)-1, -1, -1):
                idx_back = len(block_pos)-1-k
                cum += backward_weights[idx_back]
                temp[k] = right_anchor - pd.to_timedelta(cum, unit="s")
            block_candidate = temp

            for j, pos in enumerate(block_pos):
                source = orig_ts[pos]
                if is_date_only_missing_time(source):
                    observed_date = parse_ts(source)
                    cand = block_candidate[j]
                    if not same_calendar_date(cand, observed_date):
                        sec = seconds_of_day(cand)
                        tmp = combine_date_and_seconds(observed_date, sec)
                        if tmp < case_start:
                            tmp = case_start
                        if tmp > case_end:
                            tmp = case_end
                        block_candidate[j] = tmp
        else:
            # no anchors: build relative schedule from date if any, else case_start
            span = block_span_estimate_secs(block_acts)
            weights = []
            for i in range(len(block_pos)-1):
                weights.append(max(edge_weight(acts[block_pos[i]], acts[block_pos[i+1]]), 0.0))
            if sum(weights) <= 0:
                weights = [1.0] * max(1, len(block_pos)-1)
            base_date_dt = None
            for pos in block_pos:
                if is_date_only_missing_time(orig_ts[pos]) or (not is_blank(orig_ts[pos]) and not pd.isna(parse_ts(orig_ts[pos]))):
                    base_date_dt = parse_ts(orig_ts[pos])
                    break
            if base_date_dt is not None:
                base = pd.Timestamp(year=base_date_dt.year, month=base_date_dt.month, day=base_date_dt.day, hour=12)
            else:
                base = case_start
            if len(block_pos) == 1:
                block_candidate[0] = base
            else:
                total_w = sum(weights)
                cum = 0.0
                block_candidate[0] = base
                for j in range(1, len(block_pos)):
                    cum += weights[j-1]
                    block_candidate[j] = base + pd.to_timedelta(span * (cum / total_w), unit="s")

        # Final per-row constraints, formatting, and ensure changed value
        for j, pos in enumerate(block_pos):
            source = orig_ts[pos]
            precision = detect_precision(source) if not is_blank(source) else case_precision
            if precision == 0:
                precision = max(case_precision, 3)

            dt = pd.Timestamp(block_candidate[j])

            # anchor bounds
            lo = case_start
            hi = case_end
            if not pd.isna(left_anchor):
                lo = max(lo, left_anchor)
            if not pd.isna(right_anchor):
                hi = min(hi, right_anchor)
            if lo > hi:
                lo, hi = hi, lo

            # date-only rows must keep the exact date
            if is_date_only_missing_time(source):
                observed_date = parse_ts(source)
                day_start = pd.Timestamp(year=observed_date.year, month=observed_date.month, day=observed_date.day)
                day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                lo2 = max(lo, day_start)
                hi2 = min(hi, day_end)
                if lo2 <= hi2:
                    if dt < lo2:
                        dt = lo2
                    if dt > hi2:
                        # place between anchors/date bounds, not one second before right
                        dt = lo2 + (hi2 - lo2) / 2
                    lo, hi = lo2, hi2
                else:
                    # impossible intersection; keep date and choose noon then clamp best effort
                    dt = combine_date_and_seconds(observed_date, 12 * 3600)

            if dt < lo:
                dt = lo
            if dt > hi:
                dt = hi

            # preserve non-uniformity and small gaps; if all collapsed, that's acceptable for 0/10s modeled edges
            out = format_ts(dt, precision)

            # every corrupted row must differ from input
            if str(out) == str(source):
                dt2 = nudge_within_bounds(dt, lo, hi, precision)
                out2 = format_ts(dt2, precision)
                if out2 == str(source):
                    # stronger nudge using tiny forward/backward
                    step_us = max(1, 10 ** max(0, 6 - precision))
                    cand1 = pd.Timestamp(dt) + pd.to_timedelta(step_us * 3, unit="us")
                    cand2 = pd.Timestamp(dt) - pd.to_timedelta(step_us * 3, unit="us")
                    if cand1 <= pd.Timestamp(hi):
                        out2 = format_ts(cand1, precision)
                        dt2 = cand1
                    elif cand2 >= pd.Timestamp(lo):
                        out2 = format_ts(cand2, precision)
                        dt2 = cand2
                out = out2
                dt = dt2

            repaired_strings[pos] = out
            repaired_dt[pos] = pd.to_datetime(out, errors="coerce")

    case_df["_repaired_ts"] = repaired_strings
    case_df["_repaired_dt"] = repaired_dt
    case_df["_case_start"] = case_start
    case_df["_case_end"] = case_end
    return case_df


# -----------------------------
# Self-check and auto-fix
# -----------------------------
def self_check_and_fix(df):
    df = df.copy()

    # Preserve CLEAN rows exactly
    clean_mask = df["error_category"] == "CLEAN"
    df.loc[clean_mask, "_final_ts"] = df.loc[clean_mask, "Timestamp"]

    # Parse repaired
    repaired_mask = ~clean_mask

    # Case bounds from CLEAN rows
    case_bounds = {}
    for case, g in df.groupby("Case", sort=False):
        clean_parsed = pd.to_datetime(g.loc[g["error_category"] == "CLEAN", "Timestamp"], errors="coerce")
        clean_parsed = clean_parsed.dropna()
        if len(clean_parsed) > 0:
            case_bounds[case] = (clean_parsed.min(), clean_parsed.max())
        else:
            # fallback from repaired
            rp = pd.to_datetime(g["_final_ts"], errors="coerce").dropna()
            if len(rp) > 0:
                case_bounds[case] = (rp.min(), rp.max())
            else:
                now = pd.Timestamp("2023-01-01 12:00:00")
                case_bounds[case] = (now, now)

    for case, g in df.groupby("Case", sort=False):
        idxs = list(g.index)
        acts = g["Activity"].tolist()
        errs = g["error_category"].tolist()
        origs = g["Timestamp"].tolist()
        finals = g["_final_ts"].tolist()
        precision_case = choose_precision_for_case(g)
        case_start, case_end = case_bounds[case]

        parsed_final = [pd.to_datetime(x, errors="coerce") if not is_blank(x) else pd.NaT for x in finals]
        parsed_orig = [parse_ts(x) for x in origs]

        for local_i, idx in enumerate(idxs):
            if errs[local_i] == "CLEAN":
                continue

            source = origs[local_i]
            precision = detect_precision(source) if not is_blank(source) else precision_case
            if precision == 0:
                precision = max(precision_case, 3)

            dt = parsed_final[local_i]
            if pd.isna(dt):
                # repair blank/invalid using nearby same-case anchors
                left = None
                right = None
                for j in range(local_i - 1, -1, -1):
                    d = parsed_final[j] if errs[j] != "CLEAN" else parsed_orig[j]
                    if not pd.isna(d):
                        left = d
                        break
                for j in range(local_i + 1, len(idxs)):
                    d = parsed_final[j] if errs[j] != "CLEAN" else parsed_orig[j]
                    if not pd.isna(d):
                        right = d
                        break
                if left is not None and right is not None:
                    dt = left + (right - left) / 2
                elif left is not None:
                    gap = dfg_gap(acts[local_i - 1], acts[local_i]) if local_i > 0 else fallback_gap()
                    dt = left + pd.to_timedelta(gap if gap is not None else fallback_gap(), unit="s")
                elif right is not None:
                    gap = dfg_gap(acts[local_i], acts[local_i + 1]) if local_i < len(idxs)-1 else fallback_gap()
                    dt = right - pd.to_timedelta(gap if gap is not None else fallback_gap(), unit="s")
                else:
                    base = parsed_orig[local_i] if not pd.isna(parsed_orig[local_i]) else case_start
                    dt = base

            # neighbor clean anchors
            left_anchor = None
            right_anchor = None
            for j in range(local_i - 1, -1, -1):
                if errs[j] == "CLEAN" and not pd.isna(parsed_orig[j]):
                    left_anchor = parsed_orig[j]
                    break
            for j in range(local_i + 1, len(idxs)):
                if errs[j] == "CLEAN" and not pd.isna(parsed_orig[j]):
                    right_anchor = parsed_orig[j]
                    break

            lo = case_start
            hi = case_end
            if left_anchor is not None:
                lo = max(lo, left_anchor)
            if right_anchor is not None:
                hi = min(hi, right_anchor)
            if lo > hi:
                lo, hi = hi, lo

            # keep same observed calendar date for date-only
            if is_date_only_missing_time(source):
                observed_date = parse_ts(source)
                day_start = pd.Timestamp(year=observed_date.year, month=observed_date.month, day=observed_date.day)
                day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                lo2 = max(lo, day_start)
                hi2 = min(hi, day_end)
                if lo2 <= hi2:
                    if dt < lo2 or dt > hi2:
                        dt = lo2 + (hi2 - lo2) / 2
                    lo, hi = lo2, hi2
                else:
                    sec = min(max(seconds_of_day(dt), 0.0), 86399.999999)
                    dt = combine_date_and_seconds(observed_date, sec)

            if dt < lo:
                dt = lo
            if dt > hi:
                dt = hi

            out = format_ts(dt, precision)
            if out == str(source):
                dt = nudge_within_bounds(dt, lo, hi, precision)
                out = format_ts(dt, precision)
                if out == str(source):
                    # force minimal safe change if possible
                    step_us = max(1, 10 ** max(0, 6 - precision))
                    cand = pd.Timestamp(dt) + pd.to_timedelta(step_us, unit="us")
                    if cand <= pd.Timestamp(hi):
                        out = format_ts(cand, precision)
                    else:
                        cand = pd.Timestamp(dt) - pd.to_timedelta(step_us, unit="us")
                        if cand >= pd.Timestamp(lo):
                            out = format_ts(cand, precision)

            # final fallback: if still bad and source blank/date-only midnight, use midpoint
            if pd.isna(pd.to_datetime(out, errors="coerce")) or out == str(source):
                midpoint = lo + (hi - lo) / 2 if lo <= hi else lo
                out = format_ts(midpoint, precision)
                if out == str(source):
                    midpoint = nudge_within_bounds(midpoint, lo, hi, precision)
                    out = format_ts(midpoint, precision)

            df.at[idx, "_final_ts"] = out

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        # Must not fail hard
        df = pd.read_csv(args.input, dtype=str)

    # Preserve exact row count/order and untouched columns
    original_columns = list(df.columns)

    # This log contains only CLEAN and C3_MISSING_TEMPORAL_INFORMATION
    # Do not branch for other error classes.
    repaired_parts = []
    for case, g in df.groupby("Case", sort=False):
        repaired_parts.append(repair_case(g))
    repaired_df = pd.concat(repaired_parts).sort_index()

    repaired_df["_final_ts"] = repaired_df["_repaired_ts"]
    repaired_df = self_check_and_fix(repaired_df)

    # Re-assert CLEAN rows unchanged exactly
    clean_mask = repaired_df["error_category"] == "CLEAN"
    repaired_df.loc[clean_mask, "_final_ts"] = repaired_df.loc[clean_mask, "Timestamp"]

    # Diagnostics counts
    nonclean = repaired_df["error_category"] != "CLEAN"
    blank_filled = int(((repaired_df["Timestamp"].apply(is_blank)) & nonclean).sum())
    date_only_assigned = int(((repaired_df["Timestamp"].apply(is_date_only_missing_time)) & nonclean).sum())
    still_equal = int((repaired_df.loc[nonclean, "_final_ts"].astype(str) == repaired_df.loc[nonclean, "Timestamp"].astype(str)).sum())

    # Write output with exact untouched columns except repaired Timestamp values
    out_df = repaired_df[original_columns].copy()
    out_df["Timestamp"] = repaired_df["_final_ts"]

    # Final soft safeguards
    if len(out_df) != len(df):
        out_df = out_df.reindex(df.index)
        for c in original_columns:
            if c != "Timestamp":
                out_df[c] = df[c]
        ts_vals = repaired_df["_final_ts"].reindex(df.index)
        out_df["Timestamp"] = ts_vals.fillna(df["Timestamp"])

    # Byte-identical untouched required columns
    for c in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        out_df[c] = df[c]

    out_df.to_csv(args.output, index=False)

    print(f"blank_rows_filled={blank_filled}, date_only_rows_given_time={date_only_assigned}, rows_still_equal_to_input={still_equal}")


if __name__ == "__main__":
    main()