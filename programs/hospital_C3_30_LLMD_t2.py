import argparse
import math
import re
from datetime import timedelta

import pandas as pd


# ---------------------------------------------------------------------
# Fixed process knowledge from prompt
# ---------------------------------------------------------------------
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

BLOCK_SPAN = {
    2: {"p10": 18.0, "p25": 12215.0, "median": 405039.0},
    3: {"p10": 95866.1, "p25": 460038.5, "median": 3831559.5},
    4: {"p10": 384922.4, "p25": 2342423.0, "median": 6968368.0},
    5: {"p10": 440341.0, "p25": 4326198.0, "median": 8294963.0},
}
GLOBAL_MEDIAN = 405039.0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def detect_precision_and_sep(ts_series):
    for s in ts_series:
        if pd.isna(s):
            continue
        s = str(s)
        m = re.search(r"(\.\d+)$", s.strip())
        if m:
            return len(m.group(1)) - 1, "."
    return 0, None


def parse_ts(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return pd.NaT
    txt = str(s).strip()
    if txt == "":
        return pd.NaT
    return pd.to_datetime(txt, errors="coerce")


def is_blank(s):
    return s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == ""


def is_date_only_or_midnight_damage(s):
    if is_blank(s):
        return False
    txt = str(s).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", txt):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]00:00:00(?:\.0+)?", txt):
        return True
    return False


def surviving_date(s):
    if is_blank(s):
        return None
    txt = str(s).strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", txt)
    return pd.to_datetime(m.group(1)).date() if m else None


def edge_gap_seconds(a, b):
    prof = DURATION_PROFILES.get((a, b))
    if prof:
        if prof["share_under_60s"] >= 0.4:
            return max(prof["p10"], 0.001)
        return max(prof["p25"], 0.001)
    return None


def relative_default_seconds():
    return GLOBAL_MEDIAN


def fmt_timestamp(dt, precision):
    if precision <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    frac = f"{dt.microsecond:06d}"[:precision]
    return f"{base}.{frac}"


def quantize(dt, precision):
    if precision <= 0:
        return dt.replace(microsecond=0)
    factor = 10 ** (6 - precision)
    us = int(round(dt.microsecond / factor) * factor)
    if us >= 1000000:
        dt = dt + timedelta(seconds=1)
        us = 0
    return dt.replace(microsecond=us)


def day_bounds(d, precision):
    start = pd.Timestamp(f"{d} 00:00:00")
    if precision <= 0:
        end = pd.Timestamp(f"{d} 23:59:59")
    else:
        unit = 10 ** (6 - precision)
        end = pd.Timestamp(f"{d} 23:59:59.999999")
        end = end - pd.Timedelta(microseconds=(10 ** (6 - precision) - 1))
    return start, end


def clamp(dt, lo, hi):
    if lo is not None and dt < lo:
        dt = lo
    if hi is not None and dt > hi:
        dt = hi
    return dt


def make_strictly_increasing(times, precision, lo=None, hi=None):
    eps = pd.Timedelta(microseconds=max(1, 10 ** (6 - precision)) if precision > 0 else 1000000)
    out = []
    prev = None
    for t in times:
        tt = t
        if prev is not None and tt <= prev:
            tt = prev + eps
        out.append(tt)
        prev = tt
    if hi is not None and out and out[-1] > hi:
        overflow = out[-1] - hi
        out = [t - overflow for t in out]
    if lo is not None and out and out[0] < lo:
        under = lo - out[0]
        out = [t + under for t in out]
    # second pass
    out2 = []
    prev = None
    for t in out:
        tt = clamp(t, lo, hi)
        if prev is not None and tt <= prev:
            tt = prev + eps
            tt = clamp(tt, lo, hi)
        out2.append(tt)
        prev = tt
    return [quantize(t, precision) for t in out2]


# ---------------------------------------------------------------------
# Repair logic
# ---------------------------------------------------------------------
def repair_block(case_df, block_positions, precision):
    idxs = list(block_positions)
    acts = case_df.loc[idxs, "Activity"].tolist()
    orig_ts = case_df.loc[idxs, "Timestamp"].tolist()

    left_idx = None
    for p in range(idxs[0] - 1, case_df.index.min() - 1, -1):
        if case_df.at[p, "error_category"] == "CLEAN":
            left_idx = p
            break

    right_idx = None
    for p in range(idxs[-1] + 1, case_df.index.max() + 1):
        if case_df.at[p, "error_category"] == "CLEAN":
            right_idx = p
            break

    left_ts = parse_ts(case_df.at[left_idx, "Timestamp"]) if left_idx is not None else pd.NaT
    right_ts = parse_ts(case_df.at[right_idx, "Timestamp"]) if right_idx is not None else pd.NaT
    left_act = case_df.at[left_idx, "Activity"] if left_idx is not None else None
    right_act = case_df.at[right_idx, "Activity"] if right_idx is not None else None

    case_clean = case_df[case_df["error_category"] == "CLEAN"].copy()
    case_start = pd.to_datetime(case_clean["Timestamp"], errors="coerce").min() if len(case_clean) else pd.NaT
    case_end = pd.to_datetime(case_clean["Timestamp"], errors="coerce").max() if len(case_clean) else pd.NaT

    n = len(idxs)
    weights = []

    seq_acts = []
    if left_act is not None:
        seq_acts.append(left_act)
    seq_acts.extend(acts)
    if right_act is not None:
        seq_acts.append(right_act)

    for i in range(len(seq_acts) - 1):
        g = edge_gap_seconds(seq_acts[i], seq_acts[i + 1])
        if g is None:
            g = BLOCK_SPAN.get(n, {}).get("p10", None)
        if g is None:
            g = min(relative_default_seconds(), 3600.0)
        weights.append(max(g, 0.001))

    if n >= 2:
        for i in range(n - 1):
            if i < len(weights) - 1:
                if weights[i] == weights[i + 1]:
                    weights[i + 1] *= 1.07

    repaired = [None] * n

    date_constraints = {}
    for j, ts in enumerate(orig_ts):
        if is_date_only_or_midnight_damage(ts):
            d = surviving_date(ts)
            if d is not None:
                date_constraints[j] = d

    # Single-row strategy
    if n == 1:
        j = 0
        target = None
        has_left = pd.notna(left_ts)
        has_right = pd.notna(right_ts)

        if has_left:
            gap = edge_gap_seconds(left_act, acts[0])
            if gap is None:
                gap = relative_default_seconds()
            target = left_ts + pd.Timedelta(seconds=gap)

        if has_left and has_right:
            if target is None:
                target = left_ts + (right_ts - left_ts) / 2
            if target > right_ts:
                target = left_ts + (right_ts - left_ts) * 0.5
            target = clamp(target, left_ts, right_ts)
        elif (not has_left) and has_right:
            gap = edge_gap_seconds(acts[0], right_act)
            if gap is None:
                gap = relative_default_seconds()
            target = right_ts - pd.Timedelta(seconds=gap)
            target = clamp(target, case_start if pd.notna(case_start) else None, right_ts)
        else:
            if target is None:
                if pd.notna(case_start):
                    target = case_start
                else:
                    target = pd.Timestamp("2000-01-01 12:00:00")

        if j in date_constraints:
            d = date_constraints[j]
            dlo, dhi = day_bounds(d, precision)
            target = pd.Timestamp.combine(d, target.time())
            target = clamp(target, dlo, dhi)
            if has_left and has_right:
                feasible_lo = max(left_ts, dlo)
                feasible_hi = min(right_ts, dhi)
                if feasible_lo <= feasible_hi:
                    if target < feasible_lo or target > feasible_hi:
                        target = feasible_lo + (feasible_hi - feasible_lo) / 2
                else:
                    target = dlo + pd.Timedelta(hours=12)

        target = clamp(target, case_start if pd.notna(case_start) else None, case_end if pd.notna(case_end) else None)
        if has_left and has_right:
            target = clamp(target, left_ts, right_ts)
        repaired[0] = quantize(target, precision)
        return repaired

    # Multi-row strategy
    has_left = pd.notna(left_ts)
    has_right = pd.notna(right_ts)

    if has_left and has_right:
        total_window = (right_ts - left_ts).total_seconds()
        cum = []
        s = 0.0
        for w in weights[:-1]:
            s += w
            cum.append(s)
        denom = sum(weights)
        if denom <= 0:
            denom = float(n + 1)
            cum = list(range(1, n + 1))
        times = [left_ts + pd.Timedelta(seconds=total_window * (c / denom)) for c in cum]

        for j in range(n):
            if j in date_constraints:
                d = date_constraints[j]
                dlo, dhi = day_bounds(d, precision)
                feasible_lo = max(left_ts, dlo)
                feasible_hi = min(right_ts, dhi)
                if feasible_lo <= feasible_hi:
                    candidate = times[j]
                    if candidate < feasible_lo or candidate > feasible_hi:
                        candidate = feasible_lo + (feasible_hi - feasible_lo) / 2
                    times[j] = candidate
                else:
                    times[j] = left_ts + (right_ts - left_ts) * ((j + 1) / (n + 1))

        times = make_strictly_increasing(times, precision, lo=left_ts, hi=right_ts)
        repaired = times
    else:
        # one-sided anchor placement using cumulative non-uniform gaps
        if has_left:
            times = []
            t = left_ts
            for j in range(n):
                gap = weights[j] if j < len(weights) else relative_default_seconds()
                t = t + pd.Timedelta(seconds=gap)
                times.append(t)
            hi = case_end if pd.notna(case_end) else None
            times = make_strictly_increasing(times, precision, lo=left_ts, hi=hi)
            repaired = times
        elif has_right:
            rev_times = []
            t = right_ts
            for j in range(n - 1, -1, -1):
                wi = j + 1
                gap = weights[wi] if wi < len(weights) else relative_default_seconds()
                t = t - pd.Timedelta(seconds=gap)
                rev_times.append(t)
            times = list(reversed(rev_times))
            lo = case_start if pd.notna(case_start) else None
            times = make_strictly_increasing(times, precision, lo=lo, hi=right_ts)
            repaired = times
        else:
            base = pd.Timestamp("2000-01-01 00:00:00")
            times = []
            t = base
            for j in range(n):
                t = t + pd.Timedelta(seconds=max(weights[j] if j < len(weights) else 1.0, 1.0))
                times.append(t)
            repaired = make_strictly_increasing(times, precision)

        for j in range(n):
            if j in date_constraints:
                d = date_constraints[j]
                dlo, dhi = day_bounds(d, precision)
                rep = repaired[j]
                rep = pd.Timestamp.combine(d, rep.time())
                rep = clamp(rep, dlo, dhi)
                repaired[j] = quantize(rep, precision)

        repaired = make_strictly_increasing(
            repaired,
            precision,
            lo=(left_ts if has_left else (case_start if pd.notna(case_start) else None)),
            hi=(right_ts if has_right else (case_end if pd.notna(case_end) else None)),
        )

    return repaired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    orig = df.copy()

    precision, _ = detect_precision_and_sep(df["Timestamp"].tolist())
    df["Timestamp_out"] = df["Timestamp"]

    blank_filled = 0
    date_only_filled = 0

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_df = df.loc[list(idxs)].copy()
        case_positions = list(case_df.index)
        bad_mask = case_df["error_category"] != "CLEAN"
        bad_positions = [p for p in case_positions if df.at[p, "error_category"] != "CLEAN"]

        i = 0
        while i < len(bad_positions):
            start = bad_positions[i]
            block = [start]
            j = i + 1
            while j < len(bad_positions) and bad_positions[j] == bad_positions[j - 1] + 1:
                block.append(bad_positions[j])
                j += 1

            repaired_dts = repair_block(case_df, block, precision)

            for pos, new_dt in zip(block, repaired_dts):
                old_ts = df.at[pos, "Timestamp"]
                if is_blank(old_ts):
                    blank_filled += 1
                elif is_date_only_or_midnight_damage(old_ts):
                    date_only_filled += 1
                new_str = fmt_timestamp(new_dt.to_pydatetime() if hasattr(new_dt, "to_pydatetime") else new_dt, precision)
                if new_str == str(old_ts):
                    q = pd.to_datetime(new_str) + pd.Timedelta(microseconds=(10 ** (6 - precision) if precision > 0 else 1000000))
                    new_dt2 = q
                    new_str = fmt_timestamp(new_dt2.to_pydatetime() if hasattr(new_dt2, "to_pydatetime") else new_dt2, precision)
                df.at[pos, "Timestamp_out"] = new_str

            i = j

    # -----------------------------------------------------------------
    # Self-check and repair-in-place
    # -----------------------------------------------------------------
    still_equal = 0

    for idx, row in df.iterrows():
        if row["error_category"] == "CLEAN":
            df.at[idx, "Timestamp_out"] = orig.at[idx, "Timestamp"]
            continue

        candidate = df.at[idx, "Timestamp_out"]
        parsed = parse_ts(candidate)
        if pd.isna(parsed):
            d = surviving_date(orig.at[idx, "Timestamp"])
            if d is not None:
                parsed = pd.Timestamp(f"{d} 12:00:00")
            else:
                # same-case anchor fallback
                case_rows = df[df["Case"] == row["Case"]]
                clean_rows = case_rows[case_rows["error_category"] == "CLEAN"]
                if len(clean_rows):
                    parsed = pd.to_datetime(clean_rows.iloc[0]["Timestamp"])
                else:
                    parsed = pd.Timestamp("2000-01-01 12:00:00")
            df.at[idx, "Timestamp_out"] = fmt_timestamp(quantize(parsed, precision), precision)

    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_idx = list(idxs)
        case_rows = df.loc[case_idx]
        clean_rows = case_rows[case_rows["error_category"] == "CLEAN"]
        case_start = pd.to_datetime(clean_rows["Timestamp"], errors="coerce").min() if len(clean_rows) else pd.NaT
        case_end = pd.to_datetime(clean_rows["Timestamp"], errors="coerce").max() if len(clean_rows) else pd.NaT

        for local_pos, idx in enumerate(case_idx):
            if df.at[idx, "error_category"] == "CLEAN":
                continue

            t = parse_ts(df.at[idx, "Timestamp_out"])
            left_anchor = None
            right_anchor = None
            for p in range(local_pos - 1, -1, -1):
                i2 = case_idx[p]
                if df.at[i2, "error_category"] == "CLEAN":
                    left_anchor = parse_ts(df.at[i2, "Timestamp"])
                    break
            for p in range(local_pos + 1, len(case_idx)):
                i2 = case_idx[p]
                if df.at[i2, "error_category"] == "CLEAN":
                    right_anchor = parse_ts(df.at[i2, "Timestamp"])
                    break

            lo = case_start if pd.notna(case_start) else None
            hi = case_end if pd.notna(case_end) else None
            if left_anchor is not None and pd.notna(left_anchor):
                lo = left_anchor if lo is None else max(lo, left_anchor)
            if right_anchor is not None and pd.notna(right_anchor):
                hi = right_anchor if hi is None else min(hi, right_anchor)

            d = surviving_date(orig.at[idx, "Timestamp"])
            if d is not None:
                dlo, dhi = day_bounds(d, precision)
                lo = dlo if lo is None else max(lo, dlo)
                hi = dhi if hi is None else min(hi, dhi)

            if lo is not None and hi is not None and lo > hi:
                # impossible intersection: respect date if present, else anchors
                if d is not None:
                    t = pd.Timestamp(f"{d} 12:00:00")
                elif left_anchor is not None and right_anchor is not None:
                    t = left_anchor + (right_anchor - left_anchor) / 2
                elif left_anchor is not None:
                    t = left_anchor
                elif right_anchor is not None:
                    t = right_anchor
            else:
                t = clamp(t, lo, hi)

            t = quantize(t, precision)
            out_str = fmt_timestamp(t, precision)
            if out_str == orig.at[idx, "Timestamp"]:
                if lo is not None and hi is not None and lo < hi:
                    mid = lo + (hi - lo) / 2
                    t = quantize(mid, precision)
                    out_str = fmt_timestamp(t, precision)
                if out_str == orig.at[idx, "Timestamp"]:
                    bump = pd.Timedelta(microseconds=(10 ** (6 - precision) if precision > 0 else 1000000))
                    t2 = t + bump
                    t2 = clamp(t2, lo, hi)
                    out_str = fmt_timestamp(quantize(t2, precision), precision)

            df.at[idx, "Timestamp_out"] = out_str
            if df.at[idx, "Timestamp_out"] == orig.at[idx, "Timestamp"]:
                still_equal += 1

    out = orig.copy()
    out["Timestamp"] = df["Timestamp_out"]

    # enforce byte-identical untouched columns and row order automatically by writing out as-is
    out.to_csv(args.output, index=False)

    print(f"blank_rows_filled={blank_filled} date_only_rows_given_time={date_only_filled} rows_still_equal_to_input={still_equal}")


if __name__ == "__main__":
    main()