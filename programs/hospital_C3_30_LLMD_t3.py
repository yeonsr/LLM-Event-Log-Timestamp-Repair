import argparse
import math
import re
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd


DURATION_PROFILES = {
    ("FIN", "RELEASE"): {"p10": 2115.6, "p25": 6885.25, "median": 16217.0, "share_under_60s": 0.048},
    ("RELEASE", "CODE OK"): {"p10": 3665.1, "p25": 177283.0, "median": 343757.5, "share_under_60s": 0.043},
    ("CODE OK", "BILLED"): {"p10": 384101.4, "p25": 1317610.5, "median": 3783677.0, "share_under_60s": 0.0},
    ("NEW", "CHANGE DIAGN"): {"p10": 12.0, "p25": 16.0, "median": 25.0, "share_under_60s": 0.636},
    ("CHANGE DIAGN", "FIN"): {"p10": 1524200.0, "p25": 4012910.0, "median": 7708691.0, "share_under_60s": 0.006},
    ("NEW", "FIN"): {"p10": 431237.0, "p25": 5458135.0, "median": 24051621.5, "share_under_60s": 0.033},
    ("NEW", "NEW"): {"p10": 1.0, "p25": 1.0, "median": 1.0, "share_under_60s": 1.0},
    ("NEW", "DELETE"): {"p10": 2443.7, "p25": 156903.5, "median": 1814597.5, "share_under_60s": 0.061},
    ("FIN", "CODE OK"): {"p10": 78171.0, "p25": 251406.0, "median": 371464.0, "share_under_60s": 0.027},
    ("RELEASE", "BILLED"): {"p10": 736717.8, "p25": 1459903.25, "median": 3963059.0, "share_under_60s": 0.0},
    ("REOPEN", "FIN"): {"p10": 6.0, "p25": 10.0, "median": 53.0, "share_under_60s": 0.512},
    ("CODE OK", "REOPEN"): {"p10": 126745.6, "p25": 459994.0, "median": 1916316.0, "share_under_60s": 0.013},
    ("BILLED", "STORNO"): {"p10": 258883.1, "p25": 434157.75, "median": 1077097.0, "share_under_60s": 0.002},
    ("CHANGE DIAGN", "RELEASE"): {"p10": 2299488.4, "p25": 4247154.0, "median": 7732648.0, "share_under_60s": 0.003},
    ("CHANGE DIAGN", "CHANGE DIAGN"): {"p10": 206.2, "p25": 36627.0, "median": 766567.0, "share_under_60s": 0.041},
    ("NEW", "RELEASE"): {"p10": 448298.6, "p25": 7864724.0, "median": 31216839.5, "share_under_60s": 0.014},
    ("STORNO", "BILLED"): {"p10": 92763.2, "p25": 92835.0, "median": 169855.0, "share_under_60s": 0.0},
    ("STORNO", "REJECT"): {"p10": 5.7, "p25": 15.75, "median": 1242.0, "share_under_60s": 0.428},
    ("REOPEN", "DELETE"): {"p10": 2.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.974},
    ("REJECT", "BILLED"): {"p10": 1710.8, "p25": 160511.5, "median": 325498.5, "share_under_60s": 0.062},
    ("RELEASE", "CODE NOK"): {"p10": 331371.0, "p25": 599392.0, "median": 1473039.0, "share_under_60s": 0.008},
    ("CHANGE DIAGN", "DELETE"): {"p10": 203.4, "p25": 69873.0, "median": 961787.0, "share_under_60s": 0.064},
    ("CODE OK", "RELEASE"): {"p10": 1990550.0, "p25": 2961133.0, "median": 4592168.5, "share_under_60s": 0.01},
    ("REJECT", "REOPEN"): {"p10": 7.7, "p25": 10.75, "median": 38.5, "share_under_60s": 0.647},
    ("RELEASE", "REOPEN"): {"p10": 19776.0, "p25": 48101.0, "median": 292036.0, "share_under_60s": 0.033},
    ("JOIN-PAT", "JOIN-PAT"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0},
    ("CODE NOK", "CODE OK"): {"p10": 3809.4, "p25": 18932.0, "median": 49700.0, "share_under_60s": 0.023},
    ("CODE NOK", "SET STATUS"): {"p10": 9180.2, "p25": 17044.5, "median": 25733.5, "share_under_60s": 0.0},
    ("CODE NOK", "CODE NOK"): {"p10": 345704.0, "p25": 590782.0, "median": 2352460.0, "share_under_60s": 0.0},
    ("FIN", "REOPEN"): {"p10": 5.8, "p25": 49.5, "median": 13837.0, "share_under_60s": 0.275},
    ("FIN", "FIN"): {"p10": 0.0, "p25": 0.0, "median": 31430606.0, "share_under_60s": 0.351},
    ("CODE OK", "MANUAL"): {"p10": 487.0, "p25": 469502.0, "median": 1652942.0, "share_under_60s": 0.061},
    ("FIN", "DELETE"): {"p10": 66.0, "p25": 57817.25, "median": 1337180.0, "share_under_60s": 0.094},
    ("BILLED", "BILLED"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.517},
    ("CODE NOK", "REOPEN"): {"p10": 12467.6, "p25": 22825.5, "median": 31343.5, "share_under_60s": 0.036},
    ("MANUAL", "RELEASE"): {"p10": 289.0, "p25": 27209.5, "median": 36319.0, "share_under_60s": 0.038},
    ("REOPEN", "CHANGE DIAGN"): {"p10": 8.2, "p25": 20.0, "median": 42.0, "share_under_60s": 0.52},
    ("RELEASE", "RELEASE"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.55},
    ("BILLED", "CODE OK"): {"p10": 2311645.3, "p25": 7165299.25, "median": 11056150.0, "share_under_60s": 0.0},
    ("CODE OK", "CODE OK"): {"p10": 940142.6, "p25": 2474552.0, "median": 5718020.0, "share_under_60s": 0.0},
    ("CODE NOK", "DELETE"): {"p10": 12976.6, "p25": 21498.5, "median": 27554.0, "share_under_60s": 0.0},
    ("CODE OK", "DELETE"): {"p10": 806324.1, "p25": 2109509.5, "median": 2712903.0, "share_under_60s": 0.0},
    ("CODE OK", "FIN"): {"p10": 183503.0, "p25": 938728.75, "median": 1682164.5, "share_under_60s": 0.062},
    ("CODE OK", "STORNO"): {"p10": 1925302.0, "p25": 3485959.5, "median": 5570562.0, "share_under_60s": 0.0},
    ("REOPEN", "RELEASE"): {"p10": 11.9, "p25": 7614.0, "median": 215921.5, "share_under_60s": 0.214},
    ("FIN", "CODE NOK"): {"p10": 653233.0, "p25": 989636.0, "median": 1645922.0, "share_under_60s": 0.0},
    ("RELEASE", "DELETE"): {"p10": 33312.8, "p25": 123510.0, "median": 137503.0, "share_under_60s": 0.077},
    ("RELEASE", "SET STATUS"): {"p10": 145901.2, "p25": 197064.0, "median": 428082.0, "share_under_60s": 0.0},
    ("MANUAL", "MANUAL"): {"p10": 6.8, "p25": 41.0, "median": 171.0, "share_under_60s": 0.333},
    ("REJECT", "RELEASE"): {"p10": 7.7, "p25": 9.5, "median": 35.0, "share_under_60s": 0.75},
    ("JOIN-PAT", "FIN"): {"p10": 3050054.3, "p25": 4138408.25, "median": 5127431.5, "share_under_60s": 0.0},
    ("CHANGE DIAGN", "JOIN-PAT"): {"p10": 5154.2, "p25": 14279.75, "median": 596185.5, "share_under_60s": 0.0},
    ("MANUAL", "CODE OK"): {"p10": 31.8, "p25": 54.0, "median": 90.0, "share_under_60s": 0.25},
    ("STORNO", "SET STATUS"): {"p10": 12.5, "p25": 16.25, "median": 48.5, "share_under_60s": 0.625},
    ("RELEASE", "FIN"): {"p10": 1529460.8, "p25": 2625254.5, "median": 6350998.0, "share_under_60s": 0.0},
    ("STORNO", "STORNO"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.857},
    ("BILLED", "REJECT"): {"p10": 136092.0, "p25": 1764092.0, "median": 7959808.0, "share_under_60s": 0.0},
    ("RELEASE", "MANUAL"): {"p10": 9625657.0, "p25": 17681111.5, "median": 20815914.0, "share_under_60s": 0.0},
    ("NEW", "JOIN-PAT"): {"p10": 1159796.5, "p25": 1457865.0, "median": 9127796.5, "share_under_60s": 0.0},
    ("FIN", "BILLED"): {"p10": 10045555.5, "p25": 17956271.75, "median": 20730728.0, "share_under_60s": 0.0},
}

BLOCK_SPAN_P25 = {2: 12215.0, 3: 460038.5, 4: 2342423.0, 5: 4326198.0}
GLOBAL_MEDIAN = 405039.0


def detect_fractional_digits(ts: str) -> int:
    if ts is None:
        return 0
    m = re.search(r"\.(\d+)", str(ts).strip())
    return len(m.group(1)) if m else 0


def detect_date_only(ts: str) -> bool:
    if ts is None:
        return False
    s = str(ts).strip()
    if s == "":
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]00:00:00(?:\.0+)?", s):
        return True
    return False


def parse_ts_safe(ts: str):
    try:
        if ts is None:
            return pd.NaT
        s = str(ts).strip()
        if s == "":
            return pd.NaT
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def same_day_bounds(date_ts: pd.Timestamp):
    start = pd.Timestamp(year=date_ts.year, month=date_ts.month, day=date_ts.day)
    end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return start, end


def choose_edge_seconds(a1, a2):
    prof = DURATION_PROFILES.get((a1, a2))
    if prof is None:
        return None
    if prof["share_under_60s"] >= 0.2:
        return prof["p25"]
    return prof["p10"]


def quantize_timestamp(ts: pd.Timestamp, frac_digits: int):
    if frac_digits <= 0:
        return ts.round("S")
    if frac_digits <= 6:
        unit = {1: "100ms", 2: "10ms", 3: "1ms", 4: "100us", 5: "10us", 6: "1us"}[frac_digits]
        return ts.round(unit)
    return ts.round("1us")


def format_timestamp(ts: pd.Timestamp, frac_digits: int):
    ts = quantize_timestamp(ts, frac_digits)
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if frac_digits <= 0:
        return base
    frac = f"{ts.microsecond:06d}"
    if frac_digits <= 6:
        return base + "." + frac[:frac_digits]
    return base + "." + frac + ("0" * (frac_digits - 6))


def tick_timedelta(frac_digits: int):
    if frac_digits <= 0:
        return pd.Timedelta(seconds=1)
    if frac_digits <= 6:
        micros = max(1, int(round(10 ** (6 - frac_digits))))
        return pd.Timedelta(microseconds=micros)
    return pd.Timedelta(microseconds=1)


def non_uniform_weights(n):
    vals = []
    for i in range(n):
        vals.append(1.0 + 0.17 * ((i % 3) + 1) + 0.03 * (i + 1))
    return vals


def place_between(left, right, alpha=0.5):
    if left > right:
        return right + (left - right) * alpha
    return left + (right - left) * alpha


def repair_block(case_df, idxs, out_ts, orig_ts, frac_digits_default, stats):
    acts = case_df["Activity"].tolist()
    ts_parsed = case_df["_parsed"].tolist()
    orig = case_df["Timestamp"].tolist()
    errs = case_df["error_category"].tolist()

    n = len(case_df)
    s = idxs[0]
    e = idxs[-1]

    left_clean = None
    for i in range(s - 1, -1, -1):
        if errs[i] == "CLEAN" and pd.notna(ts_parsed[i]):
            left_clean = (i, ts_parsed[i], acts[i])
            break

    right_clean = None
    for i in range(e + 1, n):
        if errs[i] == "CLEAN" and pd.notna(ts_parsed[i]):
            right_clean = (i, ts_parsed[i], acts[i])
            break

    case_times = [t for t in ts_parsed if pd.notna(t)]
    case_start = min(case_times) if case_times else None
    case_end = max(case_times) if case_times else None

    edge_weights = []
    for j in range(s - 1, e + 1):
        if 0 <= j < n - 1:
            sec = choose_edge_seconds(acts[j], acts[j + 1])
            edge_weights.append(sec if sec is not None else None)

    block_len = len(idxs)
    total_weight = 0.0
    weight_list = []
    for k in range(block_len):
        left_edge_idx = s - 1 + k
        sec = edge_weights[k] if 0 <= k < len(edge_weights) else None
        if sec is None or sec < 0:
            sec = None
        if sec is None:
            sec = non_uniform_weights(block_len)[k]
        weight_list.append(max(float(sec), 0.000001))
        total_weight += weight_list[-1]

    if left_clean and right_clean:
        left_t = left_clean[1]
        right_t = right_clean[1]
        actual_gap = max((right_t - left_t).total_seconds(), 0.0)
        span = actual_gap
    else:
        span = sum(weight_list)
        cap = BLOCK_SPAN_P25.get(block_len + 1)
        if cap is not None:
            span = min(span, cap)
        if span <= 0:
            span = min(BLOCK_SPAN_P25.get(block_len + 1, GLOBAL_MEDIAN), GLOBAL_MEDIAN)

    cumulative = []
    run = 0.0
    for w in weight_list:
        run += w
        cumulative.append(run)

    for pos, idx in enumerate(idxs):
        frac_digits = max(frac_digits_default, detect_fractional_digits(orig[idx]))
        tick = tick_timedelta(frac_digits)
        raw = str(orig[idx]) if orig[idx] is not None else ""
        is_blank = raw.strip() == ""
        is_date_only = detect_date_only(raw)
        observed_dt = parse_ts_safe(raw)

        allowed_start = case_start if case_start is not None else pd.Timestamp("1900-01-01")
        allowed_end = case_end if case_end is not None else pd.Timestamp("2100-01-01")

        if is_date_only and pd.notna(observed_dt):
            d0, d1 = same_day_bounds(observed_dt)
            allowed_start = max(allowed_start, d0)
            allowed_end = min(allowed_end, d1)

        if left_clean and right_clean:
            allowed_start = max(allowed_start, left_clean[1])
            allowed_end = min(allowed_end, right_clean[1])

        forward_candidate = None
        if left_clean:
            if total_weight > 0:
                frac = cumulative[pos] / total_weight
                forward_candidate = left_clean[1] + pd.to_timedelta(span * frac, unit="s")
            else:
                forward_candidate = left_clean[1] + tick

        backward_candidate = None
        if right_clean:
            rem = sum(weight_list[pos + 1:]) if pos + 1 < len(weight_list) else 0.0
            if total_weight > 0:
                frac = rem / total_weight
                backward_candidate = right_clean[1] - pd.to_timedelta(span * frac, unit="s")
            else:
                backward_candidate = right_clean[1] - tick

        if is_date_only and pd.notna(observed_dt):
            noon = observed_dt.normalize() + pd.Timedelta(hours=12)
        else:
            noon = None

        candidate = None
        if is_date_only and pd.notna(observed_dt):
            if forward_candidate is not None and backward_candidate is not None:
                candidate = place_between(forward_candidate, backward_candidate, 0.5)
            elif forward_candidate is not None:
                candidate = place_between(forward_candidate, noon if noon is not None else forward_candidate, 0.75)
            elif backward_candidate is not None:
                candidate = place_between(noon if noon is not None else backward_candidate, backward_candidate, 0.25)
            else:
                candidate = noon
        else:
            if forward_candidate is not None and backward_candidate is not None:
                candidate = place_between(forward_candidate, backward_candidate, 0.5)
            elif forward_candidate is not None:
                candidate = forward_candidate
            elif backward_candidate is not None:
                candidate = backward_candidate
            elif case_start is not None and case_end is not None:
                candidate = place_between(case_start, case_end, 0.5)
            else:
                candidate = pd.Timestamp("2000-01-01 12:00:00")

        if candidate < allowed_start:
            candidate = place_between(allowed_start, allowed_end, 0.15) if allowed_start <= allowed_end else allowed_start
        if candidate > allowed_end:
            candidate = place_between(allowed_start, allowed_end, 0.85) if allowed_start <= allowed_end else allowed_end

        if allowed_start <= allowed_end:
            candidate = min(max(candidate, allowed_start), allowed_end)

        if pos > 0:
            prev_idx = idxs[pos - 1]
            prev_dt = parse_ts_safe(out_ts[prev_idx])
            if pd.notna(prev_dt):
                min_here = prev_dt
                if allowed_start <= allowed_end:
                    min_here = max(min_here, allowed_start)
                if candidate < min_here:
                    candidate = min_here

        if pos < len(idxs) - 1 and right_clean:
            if candidate > right_clean[1]:
                candidate = place_between(allowed_start, right_clean[1], 0.5)

        if is_date_only and pd.notna(observed_dt):
            d0, d1 = same_day_bounds(observed_dt)
            candidate = min(max(candidate, d0), d1)
            if left_clean and candidate < left_clean[1]:
                candidate = max(candidate, left_clean[1])
            if right_clean and candidate > right_clean[1]:
                candidate = min(candidate, right_clean[1])

        repaired = format_timestamp(candidate, frac_digits)
        if repaired == raw:
            bumped = candidate + tick
            if allowed_start <= allowed_end:
                bumped = min(max(bumped, allowed_start), allowed_end)
            repaired2 = format_timestamp(bumped, frac_digits)
            if repaired2 == raw:
                bumped = candidate - tick
                if allowed_start <= allowed_end:
                    bumped = min(max(bumped, allowed_start), allowed_end)
                repaired2 = format_timestamp(bumped, frac_digits)
            repaired = repaired2

        out_ts[idx] = repaired
        if is_blank:
            stats["blank_filled"] += 1
        if is_date_only:
            stats["date_only_filled"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    original_df = df.copy(deep=True)

    df["_parsed"] = df["Timestamp"].apply(parse_ts_safe)

    frac_digits_default = 0
    for ts in df["Timestamp"].tolist():
        frac_digits_default = max(frac_digits_default, detect_fractional_digits(ts))

    out_ts = df["Timestamp"].tolist()
    stats = {"blank_filled": 0, "date_only_filled": 0}

    for case_id, grp in df.groupby("Case", sort=False):
        case_idxs = grp.index.tolist()
        case_df = df.loc[case_idxs].reset_index(drop=True)

        bad_positions = [i for i, e in enumerate(case_df["error_category"].tolist()) if e != "CLEAN"]
        if not bad_positions:
            continue

        blocks = []
        start = bad_positions[0]
        prev = bad_positions[0]
        for p in bad_positions[1:]:
            if p == prev + 1:
                prev = p
            else:
                blocks.append(list(range(start, prev + 1)))
                start = p
                prev = p
        blocks.append(list(range(start, prev + 1)))

        for block in blocks:
            global_block = [case_idxs[p] for p in block]
            repair_block(case_df, block, out_ts, df["Timestamp"].tolist(), frac_digits_default, stats)
            for local_pos, global_idx in zip(block, global_block):
                out_ts[global_idx] = out_ts[case_idxs[local_pos]]

    repaired_df = df.copy()
    repaired_df["Timestamp"] = out_ts

    # self-check and repair in-place
    still_equal = 0
    repaired_df["_out_parsed"] = repaired_df["Timestamp"].apply(parse_ts_safe)

    for case_id, grp in repaired_df.groupby("Case", sort=False):
        idxs = grp.index.tolist()
        case_start = None
        case_end = None
        clean_parsed = []
        for i in idxs:
            t = repaired_df.at[i, "_out_parsed"]
            if pd.notna(t):
                clean_parsed.append(t)
        if clean_parsed:
            case_start = min(clean_parsed)
            case_end = max(clean_parsed)

        for pos, i in enumerate(idxs):
            err = repaired_df.at[i, "error_category"]
            orig_ts = original_df.at[i, "Timestamp"]
            out_ts_i = repaired_df.at[i, "Timestamp"]
            out_dt = repaired_df.at[i, "_out_parsed"]

            if err == "CLEAN":
                repaired_df.at[i, "Timestamp"] = orig_ts
                repaired_df.at[i, "_out_parsed"] = parse_ts_safe(orig_ts)
                continue

            if (str(out_ts_i).strip() == "") or pd.isna(out_dt):
                prev_clean = None
                next_clean = None
                for j in range(pos - 1, -1, -1):
                    gi = idxs[j]
                    if original_df.at[gi, "error_category"] == "CLEAN":
                        t = parse_ts_safe(original_df.at[gi, "Timestamp"])
                        if pd.notna(t):
                            prev_clean = t
                            break
                for j in range(pos + 1, len(idxs)):
                    gi = idxs[j]
                    if original_df.at[gi, "error_category"] == "CLEAN":
                        t = parse_ts_safe(original_df.at[gi, "Timestamp"])
                        if pd.notna(t):
                            next_clean = t
                            break
                frac_digits = max(frac_digits_default, detect_fractional_digits(orig_ts))
                tick = tick_timedelta(frac_digits)
                cand = None
                if prev_clean is not None and next_clean is not None:
                    cand = place_between(prev_clean, next_clean, 0.5)
                elif prev_clean is not None:
                    cand = prev_clean + tick
                elif next_clean is not None:
                    cand = next_clean - tick
                elif case_start is not None and case_end is not None:
                    cand = place_between(case_start, case_end, 0.5)
                else:
                    cand = pd.Timestamp("2000-01-01 12:00:00")
                if detect_date_only(orig_ts):
                    obs = parse_ts_safe(orig_ts)
                    if pd.notna(obs):
                        d0, d1 = same_day_bounds(obs)
                        cand = min(max(cand, d0), d1)
                repaired_df.at[i, "Timestamp"] = format_timestamp(cand, frac_digits)
                repaired_df.at[i, "_out_parsed"] = parse_ts_safe(repaired_df.at[i, "Timestamp"])
                out_dt = repaired_df.at[i, "_out_parsed"]

            # anchor bounds
            left_anchor = None
            right_anchor = None
            for j in range(pos - 1, -1, -1):
                gi = idxs[j]
                if original_df.at[gi, "error_category"] == "CLEAN":
                    t = parse_ts_safe(original_df.at[gi, "Timestamp"])
                    if pd.notna(t):
                        left_anchor = t
                        break
            for j in range(pos + 1, len(idxs)):
                gi = idxs[j]
                if original_df.at[gi, "error_category"] == "CLEAN":
                    t = parse_ts_safe(original_df.at[gi, "Timestamp"])
                    if pd.notna(t):
                        right_anchor = t
                        break

            frac_digits = max(frac_digits_default, detect_fractional_digits(orig_ts))
            tick = tick_timedelta(frac_digits)
            low = case_start if case_start is not None else out_dt
            high = case_end if case_end is not None else out_dt
            if left_anchor is not None:
                low = max(low, left_anchor)
            if right_anchor is not None:
                high = min(high, right_anchor)

            if detect_date_only(orig_ts):
                obs = parse_ts_safe(orig_ts)
                if pd.notna(obs):
                    d0, d1 = same_day_bounds(obs)
                    low = max(low, d0)
                    high = min(high, d1)

            if low > high:
                high = low

            out_dt = min(max(out_dt, low), high)

            if detect_date_only(orig_ts):
                obs = parse_ts_safe(orig_ts)
                if pd.notna(obs) and out_dt.date() != obs.date():
                    d0, d1 = same_day_bounds(obs)
                    out_dt = min(max(out_dt, d0), d1)
                    out_dt = min(max(out_dt, low), high)

            new_s = format_timestamp(out_dt, frac_digits)
            if new_s == orig_ts:
                alt = out_dt + tick
                alt = min(max(alt, low), high)
                new_s2 = format_timestamp(alt, frac_digits)
                if new_s2 == orig_ts:
                    alt = out_dt - tick
                    alt = min(max(alt, low), high)
                    new_s2 = format_timestamp(alt, frac_digits)
                new_s = new_s2

            repaired_df.at[i, "Timestamp"] = new_s
            repaired_df.at[i, "_out_parsed"] = parse_ts_safe(new_s)

            if repaired_df.at[i, "Timestamp"] == orig_ts:
                if left_anchor is not None and right_anchor is not None and left_anchor < right_anchor:
                    mid = place_between(left_anchor, right_anchor, 0.5)
                elif left_anchor is not None:
                    mid = left_anchor + tick
                elif right_anchor is not None:
                    mid = right_anchor - tick
                else:
                    mid = out_dt + tick
                if detect_date_only(orig_ts):
                    obs = parse_ts_safe(orig_ts)
                    if pd.notna(obs):
                        d0, d1 = same_day_bounds(obs)
                        mid = min(max(mid, d0), d1)
                mid = min(max(mid, low), high)
                repaired_df.at[i, "Timestamp"] = format_timestamp(mid, frac_digits)
                repaired_df.at[i, "_out_parsed"] = parse_ts_safe(repaired_df.at[i, "Timestamp"])

            if repaired_df.at[i, "Timestamp"] == orig_ts:
                still_equal += 1

    # final output columns unchanged except Timestamp
    final_df = original_df.copy()
    final_df["Timestamp"] = repaired_df["Timestamp"]

    final_df.to_csv(args.output, index=False)

    print(f"blank_rows_filled={stats['blank_filled']} date_only_rows_timed={stats['date_only_filled']} rows_still_equal_to_input={still_equal}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass