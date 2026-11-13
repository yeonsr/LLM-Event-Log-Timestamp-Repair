#!/usr/bin/env python3
import argparse
import math
import re
from datetime import timedelta

import pandas as pd


ERROR_CAT = "C3_MISSING_TEMPORAL_INFORMATION"


# ---------------------------------------------------------------------
# Process priors extracted from the prompt
# Only data needed by the repair logic is embedded.
# ---------------------------------------------------------------------
DURATION_PROFILES = {
    ("register submission date request", "OLO messaging active"): {"p10": 86400.0, "p25": 172800.0, "median": 306406.0, "share_under_60s": 0.0},
    ("forward to the competent authority", "regular procedure without MER"): {"p10": 0.0, "p25": 0.0, "median": 3.0, "share_under_60s": 0.975},
    ("send confirmation receipt", "forward to the competent authority"): {"p10": 2.0, "p25": 3.0, "median": 6.0, "share_under_60s": 0.939},
    ("request complete", "send letter in progress"): {"p10": 0.0, "p25": 0.0, "median": 1.0, "share_under_60s": 0.991},
    ("article 34  WABO applies", "assessment of content completed"): {"p10": 0.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.991},
    ("assessment of content completed", "phase advice known"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("send confirmation receipt", "send confirmation receipt"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.802},
    ("phase application receptive", "request complete"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("by law", "creating environmental permit decision"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.992},
    ("treat subcases completeness", "subcases completeness completed"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.969},
    ("treat subcases content", "completed subcases content"): {"p10": 1.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.992},
    ("enter senddate decision", "enter senddate decision environmental permit"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.991},
    ("phase advice known", "grounds for refusal"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.954},
    ("creating environmental permit decision", "decision date prior to decision"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.993},
    ("procedure change", "request complete"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.986},
    ("subcases completeness completed", "procedure change"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.988},
    ("suspension ground applicable", "by law"): {"p10": 1.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.993},
    ("send confirmation receipt", "enter senddate acknowledgement"): {"p10": 2.0, "p25": 68.75, "median": 160.0, "share_under_60s": 0.243},
    ("ask stakeholders views", "suspension ground applicable"): {"p10": 0.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.983},
    ("OLO messaging active", "phase application received"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("grounds for refusal", "ask stakeholders views"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.953},
    ("phase application received", "send confirmation receipt"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.968},
    ("regular procedure without MER", "send confirmation receipt"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("procedure change", "phase application receptive"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.966},
    ("enter senddate acknowledgement", "send confirmation receipt"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.99},
    ("enter senddate procedure confirmation", "send procedure confirmation"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.952},
    ("send confirmation receipt", "start WABOprocedure"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("forward to the competent authority", "phase application received"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("registration date publication", "regular procedure without MER"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.994},
    ("procedure change", "article 34  WABO applies"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.945},
    ("OLO messaging active", "create procedure confirmation"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("transcript decision environmental permit to stakeholders", "generate publication document decision environmental permit"): {"p10": 6.0, "p25": 7.0, "median": 10.0, "share_under_60s": 0.902},
    ("phase decision sent", "phase decision taken"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("grounds for refusal", "phase advice known"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("create procedure confirmation", "create publication document"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.997},
    ("phase advice known", "phase application receptive"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("send letter in progress", "treat subcases content"): {"p10": 0.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.991},
    ("phase decision taken", "register objection and appeal periods"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.971},
    ("article 34  WABO applies", "ask stakeholders views"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.997},
    ("creating environmental permit decision", "generating decision environmental permit"): {"p10": 4.0, "p25": 57.0, "median": 2063.0, "share_under_60s": 0.256},
    ("enter senddate decision environmental permit", "phase decision sent"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("start WABOprocedure", "enter senddate acknowledgement"): {"p10": 34535.2, "p25": 38196.0, "median": 43648.0, "share_under_60s": 0.0},
    ("completed subcases content", "procedure change"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.979},
    ("send letter in progress", "set decision status"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("set decision phase decision permitting decided", "start decision phase decision permitting sent"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.988},
    ("phase application received", "reception through OLO"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.966},
    ("regular procedure without MER", "start WABOprocedure"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit"): {"p10": 8.0, "p25": 25.75, "median": 49045.5, "share_under_60s": 0.268},
    ("send procedure confirmation", "registration date publication"): {"p10": 8.0, "p25": 11.0, "median": 19.0, "share_under_60s": 0.724},
    ("regular procedure without MER", "enter senddate procedure confirmation"): {"p10": 1.0, "p25": 2.0, "median": 2.0, "share_under_60s": 0.872},
    ("generating decision environmental permit", "grounds for refusal"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("decision date prior to decision", "generating decision environmental permit"): {"p10": 0.0, "p25": 0.0, "median": 2.0, "share_under_60s": 0.566},
    ("create publication document", "create subcases completeness"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.891},
    ("ask stakeholders views", "assessment of content completed"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("by law", "completed subcases content"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("procedure change", "procedure change"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("submit decision", "suspension ground applicable"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("completed subcases content", "coordination of application"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("register submission date request", "enter senddate acknowledgement"): {"p10": 172800.0, "p25": 345600.0, "median": 604800.0, "share_under_60s": 0.0},
    ("record date of decision environmental permit", "set decision phase decision permitting decided"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
}

BLOCK_SPAN_PROFILE = {
    2: {"p10": 0.0, "p25": 0.0, "median": 1.0, "p75": 11.0, "p90": 172800.0},
    3: {"p10": 0.0, "p25": 0.0, "median": 5.0, "p75": 1116.0, "p90": 691200.0},
    4: {"p10": 0.0, "p25": 0.0, "median": 12.0, "p75": 83427.0, "p90": 1468800.0},
    5: {"p10": 0.0, "p25": 0.0, "median": 31.0, "p75": 310365.0, "p90": 2338362.8},
}

GLOBAL_FALLBACK_SECONDS = 1.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def is_blank_ts(s):
    return pd.isna(s) or str(s).strip() == ""


def parse_ts_safe(s):
    try:
        if is_blank_ts(s):
            return pd.NaT
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def detect_precision(series):
    max_frac = 0
    for s in series.dropna().astype(str):
        m = re.search(r"\.(\d+)$", s.strip())
        if m:
            max_frac = max(max_frac, len(m.group(1)))
    return max_frac


def classify_observed_timestamp(s):
    if is_blank_ts(s):
        return "blank"
    st = str(s).strip()
    dt = parse_ts_safe(st)
    if pd.isna(dt):
        return "blank"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", st):
        return "date_only"
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$", st)
    if m:
        hh, mm, ss = int(m.group(2)), int(m.group(3)), int(m.group(4))
        if hh == 0 and mm == 0 and ss == 0:
            return "date_only"
        return "full"
    return "full"


def fmt_timestamp(ts, frac_digits):
    if frac_digits > 0:
        base = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
        return base[:- (6 - frac_digits)] if frac_digits < 6 else base
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def same_day_bounds(date_ts):
    d = pd.Timestamp(date_ts).normalize()
    return d, d + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)


def get_edge_stat(a, b, which="p25"):
    prof = DURATION_PROFILES.get((a, b))
    if prof is not None:
        val = prof.get(which, None)
        if val is not None:
            return float(val)
    return None


def get_pair_gap_seconds(prev_act, next_act):
    p = get_edge_stat(prev_act, next_act, "p25")
    if p is not None:
        return p
    p = get_edge_stat(prev_act, next_act, "p10")
    if p is not None:
        return p
    p = get_edge_stat(prev_act, next_act, "median")
    if p is not None:
        return min(p, 5.0)
    return GLOBAL_FALLBACK_SECONDS


def get_local_gap_left(prev_act, cur_act):
    p = get_edge_stat(prev_act, cur_act, "p25")
    if p is not None:
        return p
    p = get_edge_stat(prev_act, cur_act, "p10")
    if p is not None:
        return p
    p = get_edge_stat(prev_act, cur_act, "median")
    if p is not None:
        return min(p, 5.0)
    return GLOBAL_FALLBACK_SECONDS


def get_local_gap_right(cur_act, next_act):
    p = get_edge_stat(cur_act, next_act, "p25")
    if p is not None:
        return p
    p = get_edge_stat(cur_act, next_act, "p10")
    if p is not None:
        return p
    p = get_edge_stat(cur_act, next_act, "median")
    if p is not None:
        return min(p, 5.0)
    return GLOBAL_FALLBACK_SECONDS


def estimate_nonuniform_offsets(activities):
    n = len(activities)
    if n == 1:
        return [0.0]
    gaps = []
    for i in range(n - 1):
        gaps.append(get_pair_gap_seconds(activities[i], activities[i + 1]))
    offsets = [0.0]
    acc = 0.0
    for g in gaps:
        acc += max(0.0, float(g))
        offsets.append(acc)
    return offsets


def choose_within_window(left_t, right_t, left_est=None, right_est=None):
    if left_t is not None and right_t is not None:
        if left_est is not None and right_est is not None:
            cand = left_est if abs((left_est - left_t).total_seconds()) <= abs((right_t - right_est).total_seconds()) else right_est
            if cand < left_t or cand > right_t:
                return left_t + (right_t - left_t) / 2
            return cand
        if left_est is not None:
            if left_est <= right_t:
                return max(left_t, left_est)
            return left_t + (right_t - left_t) / 2
        if right_est is not None:
            if right_est >= left_t:
                return min(right_t, right_est)
            return left_t + (right_t - left_t) / 2
        return left_t + (right_t - left_t) / 2
    return left_est if left_est is not None else right_est


def repair_block(case_df, idxs, frac_digits):
    n = len(idxs)
    activities = [case_df.loc[i, "Activity"] for i in idxs]
    orig_ts = [case_df.loc[i, "Timestamp"] for i in idxs]

    left_idx = None
    for j in range(idxs[0] - 1, case_df.index.min() - 1, -1):
        if case_df.loc[j, "error_category"] == "CLEAN":
            t = parse_ts_safe(case_df.loc[j, "Timestamp"])
            if not pd.isna(t):
                left_idx = j
                break

    right_idx = None
    for j in range(idxs[-1] + 1, case_df.index.max() + 1):
        if case_df.loc[j, "error_category"] == "CLEAN":
            t = parse_ts_safe(case_df.loc[j, "Timestamp"])
            if not pd.isna(t):
                right_idx = j
                break

    left_t = parse_ts_safe(case_df.loc[left_idx, "Timestamp"]) if left_idx is not None else None
    right_t = parse_ts_safe(case_df.loc[right_idx, "Timestamp"]) if right_idx is not None else None
    left_act = case_df.loc[left_idx, "Activity"] if left_idx is not None else None
    right_act = case_df.loc[right_idx, "Activity"] if right_idx is not None else None

    offsets = estimate_nonuniform_offsets(activities)
    total_span = offsets[-1] if offsets else 0.0
    if total_span == 0.0 and n > 1:
        total_span = BLOCK_SPAN_PROFILE.get(min(n, 5), {}).get("p25", 0.0)

    repaired = {}

    for pos, idx in enumerate(idxs):
        kind = classify_observed_timestamp(orig_ts[pos])
        observed_dt = parse_ts_safe(orig_ts[pos])

        candidate = None

        if kind == "date_only" and not pd.isna(observed_dt):
            day_start, day_end = same_day_bounds(observed_dt)

            l_anchor = left_t
            r_anchor = right_t
            if l_anchor is not None and l_anchor.normalize() != observed_dt.normalize():
                l_anchor = None
            if r_anchor is not None and r_anchor.normalize() != observed_dt.normalize():
                r_anchor = None

            left_est = None
            if l_anchor is not None:
                if pos == 0:
                    gap = get_local_gap_left(left_act, activities[pos]) if left_act is not None else 0.0
                    left_est = l_anchor + pd.Timedelta(seconds=gap)
                else:
                    left_est = day_start + pd.Timedelta(seconds=offsets[pos])

            right_est = None
            if r_anchor is not None:
                if pos == n - 1:
                    gap = get_local_gap_right(activities[pos], right_act) if right_act is not None else 0.0
                    right_est = r_anchor - pd.Timedelta(seconds=gap)
                else:
                    rem = offsets[-1] - offsets[pos]
                    right_est = r_anchor - pd.Timedelta(seconds=rem)

            candidate = choose_within_window(l_anchor, r_anchor, left_est, right_est)

            if candidate is None:
                if l_anchor is not None:
                    candidate = l_anchor + pd.Timedelta(seconds=(get_local_gap_left(left_act, activities[pos]) if left_act else 0.0))
                elif r_anchor is not None:
                    candidate = r_anchor - pd.Timedelta(seconds=(get_local_gap_right(activities[pos], right_act) if right_act else 0.0))
                else:
                    candidate = day_start + pd.Timedelta(hours=12)

            if candidate < day_start:
                candidate = day_start
            if candidate > day_end:
                candidate = day_end

        else:
            left_est = None
            if left_t is not None:
                if pos == 0:
                    gap = get_local_gap_left(left_act, activities[pos]) if left_act is not None else 0.0
                    left_est = left_t + pd.Timedelta(seconds=gap)
                else:
                    left_est = left_t + pd.Timedelta(seconds=offsets[pos])

            right_est = None
            if right_t is not None:
                if pos == n - 1:
                    gap = get_local_gap_right(activities[pos], right_act) if right_act is not None else 0.0
                    right_est = right_t - pd.Timedelta(seconds=gap)
                else:
                    rem = offsets[-1] - offsets[pos]
                    right_est = right_t - pd.Timedelta(seconds=rem)

            candidate = choose_within_window(left_t, right_t, left_est, right_est)

            if candidate is None:
                if left_t is not None:
                    candidate = left_t + pd.Timedelta(seconds=max(1.0, offsets[pos] if pos < len(offsets) else GLOBAL_FALLBACK_SECONDS))
                elif right_t is not None:
                    candidate = right_t - pd.Timedelta(seconds=max(1.0, (offsets[-1] - offsets[pos]) if offsets else GLOBAL_FALLBACK_SECONDS))
                else:
                    candidate = pd.Timestamp("2000-01-01 12:00:00")

        repaired[idx] = candidate

    if left_t is not None and right_t is not None:
        for idx in idxs:
            if repaired[idx] < left_t:
                repaired[idx] = left_t
            if repaired[idx] > right_t:
                repaired[idx] = right_t

    out_strings = {}
    prev_ts = left_t
    for pos, idx in enumerate(idxs):
        cand = repaired[idx]
        if prev_ts is not None and cand < prev_ts:
            cand = prev_ts
        if pos < n - 1:
            nxt_idx = idxs[pos + 1]
            if repaired[nxt_idx] < cand:
                repaired[nxt_idx] = cand
        out_strings[idx] = fmt_timestamp(cand, frac_digits)
        prev_ts = cand

    return out_strings


def main():
    args = parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    original = df.copy(deep=True)

    frac_digits = detect_precision(df["Timestamp"].replace("", pd.NA))

    repaired_mask = df["error_category"] != "CLEAN"
    repaired_strings = {}

    blank_filled = 0
    date_only_filled = 0

    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        case_idxs = list(case_idx)
        case_idxs.sort()

        block = []
        for idx in case_idxs:
            if df.loc[idx, "error_category"] != "CLEAN":
                block.append(idx)
            else:
                if block:
                    out = repair_block(df, block, frac_digits)
                    for k, v in out.items():
                        repaired_strings[k] = v
                    block = []
        if block:
            out = repair_block(df, block, frac_digits)
            for k, v in out.items():
                repaired_strings[k] = v

    for idx in df.index:
        if df.loc[idx, "error_category"] == "CLEAN":
            continue
        old = df.loc[idx, "Timestamp"]
        kind = classify_observed_timestamp(old)
        new = repaired_strings.get(idx, old)
        if kind == "blank":
            blank_filled += 1
        elif kind == "date_only":
            date_only_filled += 1
        df.loc[idx, "Timestamp"] = new

    # -----------------------------------------------------------------
    # Self-checks and repairs
    # -----------------------------------------------------------------
    # 1. preserve non-timestamp columns byte-identically
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        df[col] = original[col]

    # 2. CLEAN rows exact timestamp unchanged
    clean_mask = df["error_category"] == "CLEAN"
    df.loc[clean_mask, "Timestamp"] = original.loc[clean_mask, "Timestamp"]

    # 3. every repaired row parseable and different from input
    for idx in df.index[repaired_mask]:
        new_s = df.loc[idx, "Timestamp"]
        old_s = original.loc[idx, "Timestamp"]
        new_t = parse_ts_safe(new_s)

        if pd.isna(new_t):
            # rebuild from nearest valid same-case anchor
            case = df.loc[idx, "Case"]
            case_rows = df.index[df["Case"] == case].tolist()
            pos = case_rows.index(idx)
            left_t = None
            right_t = None
            for j in range(pos - 1, -1, -1):
                t = parse_ts_safe(df.loc[case_rows[j], "Timestamp"])
                if not pd.isna(t):
                    left_t = t
                    break
            for j in range(pos + 1, len(case_rows)):
                t = parse_ts_safe(df.loc[case_rows[j], "Timestamp"])
                if not pd.isna(t):
                    right_t = t
                    break
            if left_t is not None and right_t is not None:
                new_t = left_t + (right_t - left_t) / 2
            elif left_t is not None:
                new_t = left_t + pd.Timedelta(seconds=GLOBAL_FALLBACK_SECONDS)
            elif right_t is not None:
                new_t = right_t - pd.Timedelta(seconds=GLOBAL_FALLBACK_SECONDS)
            else:
                new_t = pd.Timestamp("2000-01-01 12:00:00")
            df.loc[idx, "Timestamp"] = fmt_timestamp(new_t, frac_digits)

        if df.loc[idx, "Timestamp"] == old_s:
            t = parse_ts_safe(df.loc[idx, "Timestamp"])
            if pd.isna(t):
                t = pd.Timestamp("2000-01-01 12:00:00")
            t = t + pd.Timedelta(microseconds=max(1, 10 ** max(0, 6 - frac_digits)))
            df.loc[idx, "Timestamp"] = fmt_timestamp(t, frac_digits)

    # 4. date-only rows stay on same date
    for idx in df.index[repaired_mask]:
        old_s = original.loc[idx, "Timestamp"]
        if classify_observed_timestamp(old_s) == "date_only":
            old_t = parse_ts_safe(old_s)
            new_t = parse_ts_safe(df.loc[idx, "Timestamp"])
            if not pd.isna(old_t) and not pd.isna(new_t) and old_t.normalize() != new_t.normalize():
                day_start, day_end = same_day_bounds(old_t)
                if new_t < day_start:
                    new_t = day_start
                elif new_t > day_end:
                    new_t = day_end
                else:
                    secs = (new_t - new_t.normalize()).total_seconds()
                    new_t = day_start + pd.Timedelta(seconds=secs)
                    if new_t > day_end:
                        new_t = day_end
                if df.loc[idx, "Timestamp"] == original.loc[idx, "Timestamp"]:
                    new_t = min(day_end, new_t + pd.Timedelta(seconds=1))
                df.loc[idx, "Timestamp"] = fmt_timestamp(new_t, frac_digits)

    # 5. within same-case anchor windows and case bounds
    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        case_idxs = list(case_idx)
        case_idxs.sort()

        parsed_case = {i: parse_ts_safe(df.loc[i, "Timestamp"]) for i in case_idxs}
        valid_ts = [parsed_case[i] for i in case_idxs if not pd.isna(parsed_case[i])]
        if valid_ts:
            case_start = min(valid_ts)
            case_end = max(valid_ts)
        else:
            case_start = pd.Timestamp("2000-01-01 00:00:00")
            case_end = pd.Timestamp("2000-01-01 23:59:59")

        # nearest clean anchors
        left_clean = {}
        last = None
        for i in case_idxs:
            if df.loc[i, "error_category"] == "CLEAN":
                t = parse_ts_safe(df.loc[i, "Timestamp"])
                if not pd.isna(t):
                    last = t
            left_clean[i] = last
        right_clean = {}
        last = None
        for i in reversed(case_idxs):
            if df.loc[i, "error_category"] == "CLEAN":
                t = parse_ts_safe(df.loc[i, "Timestamp"])
                if not pd.isna(t):
                    last = t
            right_clean[i] = last

        prev_t = None
        for i in case_idxs:
            t = parse_ts_safe(df.loc[i, "Timestamp"])
            if pd.isna(t):
                t = case_start
            if i in df.index[repaired_mask]:
                lo = case_start
                hi = case_end
                if left_clean[i] is not None:
                    lo = max(lo, left_clean[i])
                if right_clean[i] is not None:
                    hi = min(hi, right_clean[i])
                if t < lo:
                    t = lo
                if t > hi:
                    t = hi
                if prev_t is not None and t < prev_t:
                    t = prev_t
                if classify_observed_timestamp(original.loc[i, "Timestamp"]) == "date_only":
                    old_t = parse_ts_safe(original.loc[i, "Timestamp"])
                    ds, de = same_day_bounds(old_t)
                    if t < ds:
                        t = ds
                    if t > de:
                        t = de
                if fmt_timestamp(t, frac_digits) == original.loc[i, "Timestamp"]:
                    step_us = max(1, 10 ** max(0, 6 - frac_digits))
                    t = min(hi, t + pd.Timedelta(microseconds=step_us))
                    if classify_observed_timestamp(original.loc[i, "Timestamp"]) == "date_only":
                        old_t = parse_ts_safe(original.loc[i, "Timestamp"])
                        ds, de = same_day_bounds(old_t)
                        if t > de:
                            t = max(ds, de - pd.Timedelta(microseconds=step_us))
                df.loc[i, "Timestamp"] = fmt_timestamp(t, frac_digits)
            prev_t = parse_ts_safe(df.loc[i, "Timestamp"])

    still_equal = int((df.loc[repaired_mask, "Timestamp"].values == original.loc[repaired_mask, "Timestamp"].values).sum())

    df.to_csv(args.output, index=False)
    print(f"blank_rows_filled={blank_filled} date_only_rows_timed={date_only_filled} rows_still_equal_to_input={still_equal}")


if __name__ == "__main__":
    main()