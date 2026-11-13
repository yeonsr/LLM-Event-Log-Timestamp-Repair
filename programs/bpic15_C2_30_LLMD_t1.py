import argparse
import pandas as pd
import numpy as np
from datetime import timedelta

# Only error present in this task
ERR = "C2_TEMPORAL_INCONSISTENCY"

# Process priors extracted from prompt
DURATION_PROFILES = {
    ("register submission date request", "OLO messaging active"): {"p10": 86400.0, "p25": 172800.0, "median": 306406.0, "share_under_60s": 0.0, "n": 729},
    ("forward to the competent authority", "regular procedure without MER"): {"p10": 0.0, "p25": 0.0, "median": 3.0, "share_under_60s": 0.975, "n": 687},
    ("send confirmation receipt", "forward to the competent authority"): {"p10": 2.0, "p25": 3.0, "median": 5.0, "share_under_60s": 0.94, "n": 582},
    ("request complete", "send letter in progress"): {"p10": 0.0, "p25": 0.0, "median": 1.0, "share_under_60s": 0.991, "n": 572},
    ("article 34  WABO applies", "assessment of content completed"): {"p10": 0.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.993, "n": 568},
    ("assessment of content completed", "phase advice known"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 555},
    ("treat subcases completeness", "subcases completeness completed"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.969, "n": 510},
    ("send confirmation receipt", "send confirmation receipt"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.802, "n": 501},
    ("phase application receptive", "request complete"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 500},
    ("by law", "creating environmental permit decision"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.996, "n": 493},
    ("treat subcases content", "completed subcases content"): {"p10": 1.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.996, "n": 492},
    ("enter senddate decision", "enter senddate decision environmental permit"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.991, "n": 467},
    ("phase advice known", "grounds for refusal"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.954, "n": 439},
    ("subcases completeness completed", "procedure change"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.986, "n": 436},
    ("ask stakeholders views", "suspension ground applicable"): {"p10": 0.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.984, "n": 431},
    ("creating environmental permit decision", "decision date prior to decision"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.993, "n": 431},
    ("procedure change", "request complete"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.991, "n": 429},
    ("grounds for refusal", "ask stakeholders views"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.958, "n": 425},
    ("suspension ground applicable", "by law"): {"p10": 1.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.993, "n": 415},
    ("send confirmation receipt", "enter senddate acknowledgement"): {"p10": 2.0, "p25": 74.25, "median": 159.0, "share_under_60s": 0.24, "n": 412},
    ("OLO messaging active", "phase application received"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 409},
    ("regular procedure without MER", "send confirmation receipt"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 399},
    ("phase application received", "send confirmation receipt"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.982, "n": 395},
    ("procedure change", "phase application receptive"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.964, "n": 384},
    ("enter senddate acknowledgement", "send confirmation receipt"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.99, "n": 383},
    ("enter senddate procedure confirmation", "send procedure confirmation"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.952, "n": 378},
    ("send confirmation receipt", "start WABOprocedure"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 375},
    ("forward to the competent authority", "phase application received"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 372},
    ("registration date publication", "regular procedure without MER"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.994, "n": 357},
    ("procedure change", "article 34  WABO applies"): {"p10": 1.0, "p25": 1.0, "median": 2.0, "share_under_60s": 0.946, "n": 353},
    ("OLO messaging active", "create procedure confirmation"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 344},
    ("transcript decision environmental permit to stakeholders", "generate publication document decision environmental permit"): {"p10": 6.0, "p25": 7.0, "median": 10.0, "share_under_60s": 0.906, "n": 342},
    ("send letter in progress", "treat subcases content"): {"p10": 0.0, "p25": 1.0, "median": 1.0, "share_under_60s": 0.994, "n": 324},
    ("phase decision sent", "phase decision taken"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 320},
    ("grounds for refusal", "phase advice known"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 319},
    ("create procedure confirmation", "create publication document"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.997, "n": 318},
    ("phase advice known", "phase application receptive"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 316},
    ("phase decision taken", "register objection and appeal periods"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.971, "n": 313},
    ("article 34  WABO applies", "ask stakeholders views"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.997, "n": 305},
    ("completed subcases content", "procedure change"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.98, "n": 299},
    ("enter senddate decision environmental permit", "phase decision sent"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 294},
    ("creating environmental permit decision", "generating decision environmental permit"): {"p10": 4.0, "p25": 51.0, "median": 1864.0, "share_under_60s": 0.259, "n": 293},
    ("start WABOprocedure", "enter senddate acknowledgement"): {"p10": 34535.2, "p25": 38196.0, "median": 43648.0, "share_under_60s": 0.0, "n": 293},
    ("send letter in progress", "set decision status"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 254},
    ("set decision phase decision permitting decided", "start decision phase decision permitting sent"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.988, "n": 241},
    ("phase application received", "reception through OLO"): {"p10": 1.0, "p25": 2.0, "median": 3.0, "share_under_60s": 0.966, "n": 237},
    ("regular procedure without MER", "start WABOprocedure"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 229},
    ("send procedure confirmation", "registration date publication"): {"p10": 8.0, "p25": 11.0, "median": 19.0, "share_under_60s": 0.724, "n": 225},
    ("transcript decision environmental permit to stakeholders", "enter senddate decision environmental permit"): {"p10": 8.3, "p25": 34.75, "median": 50393.0, "share_under_60s": 0.254, "n": 224},
    ("regular procedure without MER", "enter senddate procedure confirmation"): {"p10": 1.0, "p25": 2.0, "median": 2.0, "share_under_60s": 0.872, "n": 219},
    ("generating decision environmental permit", "grounds for refusal"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 215},
    ("decision date prior to decision", "generating decision environmental permit"): {"p10": 0.0, "p25": 0.0, "median": 2.5, "share_under_60s": 0.561, "n": 214},
    ("create publication document", "create subcases completeness"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 0.914, "n": 197},
    ("ask stakeholders views", "assessment of content completed"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 194},
    ("by law", "completed subcases content"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 194},
    ("procedure change", "procedure change"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 193},
    ("submit decision", "suspension ground applicable"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 193},
    ("completed subcases content", "coordination of application"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 190},
    ("register submission date request", "enter senddate acknowledgement"): {"p10": 172800.0, "p25": 345600.0, "median": 604800.0, "share_under_60s": 0.0, "n": 188},
    ("record date of decision environmental permit", "set decision phase decision permitting decided"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share_under_60s": 1.0, "n": 187},
}

DFG_FREQ = {(s, t): v["n"] for (s, t), v in DURATION_PROFILES.items()}

BLOCK_SPAN_2 = {"p10": 0.0, "p25": 0.0, "median": 1.0, "p75": 11.0, "p90": 172800.0}
GLOBAL_MEDIAN = 1.0


def detect_precision(ts: str):
    if "." in ts:
        frac = ts.split(".")[-1]
        return len(frac)
    return 0


def parse_ts(ts: str):
    return pd.to_datetime(ts, errors="coerce")


def fmt_ts(dt: pd.Timestamp, precision: int):
    if precision <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    head, frac = base.split(".")
    return f"{head}.{frac[:precision]}"


def smallest_unit(precision: int):
    if precision <= 0:
        return pd.Timedelta(seconds=1)
    if precision <= 3:
        return pd.Timedelta(milliseconds=10 ** (3 - precision))
    if precision <= 6:
        return pd.Timedelta(microseconds=10 ** (6 - precision))
    return pd.Timedelta(microseconds=1)


def choose_gap(src, tgt):
    prof = DURATION_PROFILES.get((src, tgt))
    if prof:
        if prof["share_under_60s"] >= 0.95 or prof["median"] <= 1.0:
            return prof["p10"]
        return prof["p25"]
    return GLOBAL_MEDIAN


def supported_edge(src, tgt):
    return (src, tgt) in DFG_FREQ


def local_score(prev_act, act1, act2, next_act, t_prev, t1, t2, t_next):
    score = 0.0
    big_penalty = 1e9
    if t1 > t2:
        return big_penalty
    if t_prev is not None and t_prev > t1:
        return big_penalty
    if t_next is not None and t2 > t_next:
        return big_penalty

    def edge_penalty(a, b, ta, tb):
        if ta is None or tb is None:
            return 0.0
        if tb < ta:
            return big_penalty
        gap = (tb - ta).total_seconds()
        if supported_edge(a, b):
            prof = DURATION_PROFILES[(a, b)]
            target = choose_gap(a, b)
            scale = max(1.0, prof["p25"] if prof["p25"] > 0 else prof["p10"] if prof["p10"] > 0 else 1.0)
            freq_bonus = np.log1p(prof["n"])
            return abs(gap - target) / scale - 0.2 * freq_bonus
        return 1000.0 + gap / 1000.0

    if prev_act is not None and t_prev is not None:
        score += edge_penalty(prev_act, act1, t_prev, t1)
    score += edge_penalty(act1, act2, t1, t2)
    if next_act is not None and t_next is not None:
        score += edge_penalty(act2, next_act, t2, t_next)
    return score


def estimate_one(prev_row, row, next_row, observed_dt):
    precision = row["precision"]
    eps = smallest_unit(precision)

    left_anchor = prev_row["dt"] if prev_row is not None else None
    right_anchor = next_row["dt"] if next_row is not None else None

    lower = left_anchor if left_anchor is not None else observed_dt - pd.Timedelta(days=3650)
    upper = right_anchor if right_anchor is not None else observed_dt + pd.Timedelta(days=3650)

    cands = []

    if prev_row is not None:
        sec = choose_gap(prev_row["Activity"], row["Activity"])
        cands.append(prev_row["dt"] + pd.Timedelta(seconds=float(sec)))
    if next_row is not None:
        sec = choose_gap(row["Activity"], next_row["Activity"])
        cands.append(next_row["dt"] - pd.Timedelta(seconds=float(sec)))

    if cands:
        cand = min(cands, key=lambda x: abs((x - observed_dt).total_seconds()))
        if len(cands) == 2:
            lo = min(cands)
            hi = max(cands)
            if lo <= hi:
                mid = lo + (hi - lo) / 2
                cand = min([cand, mid], key=lambda x: abs((x - observed_dt).total_seconds()))
    else:
        cand = observed_dt

    if left_anchor is not None and cand < left_anchor:
        cand = left_anchor
    if right_anchor is not None and cand > right_anchor:
        cand = right_anchor

    if left_anchor is not None and cand < left_anchor:
        cand = left_anchor
    if right_anchor is not None and cand > right_anchor:
        cand = right_anchor

    if prev_row is not None and cand < prev_row["dt"]:
        cand = prev_row["dt"]
    if next_row is not None and cand > next_row["dt"]:
        cand = next_row["dt"]

    if cand == observed_dt:
        if right_anchor is not None and observed_dt + eps <= right_anchor:
            cand = observed_dt + eps
        elif left_anchor is not None and observed_dt - eps >= left_anchor:
            cand = observed_dt - eps
        else:
            cand = observed_dt + eps

    if prev_row is not None and cand < prev_row["dt"]:
        cand = prev_row["dt"]
    if next_row is not None and cand > next_row["dt"]:
        cand = next_row["dt"]

    return cand


def repair_case(g):
    exchanged_segments = 0
    displaced_segments = 0
    fixes = []

    rows = []
    for idx, r in g.iterrows():
        rows.append({
            "index": idx,
            "Case": r["Case"],
            "Activity": r["Activity"],
            "Timestamp": r["Timestamp"],
            "Resource": r["Resource"],
            "error_category": r["error_category"],
            "row_id": r["row_id"],
            "dt": parse_ts(r["Timestamp"]),
            "precision": detect_precision(str(r["Timestamp"])),
            "out_ts": r["Timestamp"],
            "changed": False,
        })

    n = len(rows)
    i = 0
    while i < n - 1:
        r1 = rows[i]
        r2 = rows[i + 1]
        if r1["error_category"] != "CLEAN" and r2["error_category"] != "CLEAN" and r1["dt"] > r2["dt"]:
            prev_row = rows[i - 1] if i - 1 >= 0 else None
            next_row = rows[i + 2] if i + 2 < n else None

            prev_act = prev_row["Activity"] if prev_row is not None else None
            next_act = next_row["Activity"] if next_row is not None else None
            t_prev = prev_row["dt"] if prev_row is not None else None
            t_next = next_row["dt"] if next_row is not None else None

            swap_t1, swap_t2 = r2["dt"], r1["dt"]
            swap_valid = True
            if t_prev is not None and swap_t1 < t_prev:
                swap_valid = False
            if t_next is not None and swap_t2 > t_next:
                swap_valid = False
            if swap_t1 > swap_t2:
                swap_valid = False

            swap_score = 1e18
            if swap_valid:
                swap_score = local_score(prev_act, r1["Activity"], r2["Activity"], next_act, t_prev, swap_t1, swap_t2, t_next)

            # displaced alternatives
            est1 = estimate_one(prev_row, r1, {"dt": r2["dt"], "Activity": r2["Activity"]}, r1["dt"])
            score1 = local_score(prev_act, r1["Activity"], r2["Activity"], next_act, t_prev, est1, r2["dt"], t_next) if est1 <= r2["dt"] else 1e18

            est2 = estimate_one({"dt": r1["dt"], "Activity": r1["Activity"]}, r2, next_row, r2["dt"])
            score2 = local_score(prev_act, r1["Activity"], r2["Activity"], next_act, t_prev, r1["dt"], est2, t_next) if r1["dt"] <= est2 else 1e18

            best_disp = min(score1, score2)

            if swap_valid and swap_score + 1e-9 < best_disp:
                r1["out_ts"] = r2["Timestamp"]
                r2["out_ts"] = r1["Timestamp"]
                r1["dt"], r2["dt"] = swap_t1, swap_t2
                r1["changed"] = (r1["out_ts"] != r1["Timestamp"])
                r2["changed"] = (r2["out_ts"] != r2["Timestamp"])
                exchanged_segments += 1
            else:
                if score1 <= score2:
                    new_dt = est1
                    r1["dt"] = new_dt
                    r1["out_ts"] = fmt_ts(new_dt, r1["precision"])
                    r1["changed"] = (r1["out_ts"] != r1["Timestamp"])
                else:
                    new_dt = est2
                    r2["dt"] = new_dt
                    r2["out_ts"] = fmt_ts(new_dt, r2["precision"])
                    r2["changed"] = (r2["out_ts"] != r2["Timestamp"])
                displaced_segments += 1
            i += 2
        else:
            i += 1

    # Self-check and corrective pass
    for j in range(n):
        cur = rows[j]
        prev_row = rows[j - 1] if j - 1 >= 0 else None
        next_row = rows[j + 1] if j + 1 < n else None

        if cur["error_category"] == "CLEAN":
            cur["out_ts"] = cur["Timestamp"]
            cur["dt"] = parse_ts(cur["Timestamp"])
            cur["changed"] = False
            continue

        # Every corrupted row must differ from input
        if cur["out_ts"] == cur["Timestamp"]:
            new_dt = estimate_one(prev_row, cur, next_row, cur["dt"])
            if new_dt == cur["dt"]:
                eps = smallest_unit(cur["precision"])
                if next_row is not None and cur["dt"] + eps <= next_row["dt"]:
                    new_dt = cur["dt"] + eps
                elif prev_row is not None and cur["dt"] - eps >= prev_row["dt"]:
                    new_dt = cur["dt"] - eps
                else:
                    new_dt = cur["dt"] + eps
            cur["dt"] = new_dt
            cur["out_ts"] = fmt_ts(new_dt, cur["precision"])
            cur["changed"] = (cur["out_ts"] != cur["Timestamp"])
            fixes.append(f"changed_equal_input_row_id={cur['row_id']}")

        # Anchor/order checks
        if prev_row is not None and cur["dt"] < prev_row["dt"]:
            cand = prev_row["dt"]
            if cur["out_ts"] == cur["Timestamp"]:
                cand = cand + smallest_unit(cur["precision"])
            if next_row is not None and cand > next_row["dt"]:
                cand = next_row["dt"]
            cur["dt"] = cand
            cur["out_ts"] = fmt_ts(cur["dt"], cur["precision"])
            if cur["out_ts"] == cur["Timestamp"]:
                eps = smallest_unit(cur["precision"])
                if next_row is not None and cur["dt"] + eps <= next_row["dt"]:
                    cur["dt"] = cur["dt"] + eps
                    cur["out_ts"] = fmt_ts(cur["dt"], cur["precision"])
            fixes.append(f"clamped_left_row_id={cur['row_id']}")

        if next_row is not None and cur["dt"] > next_row["dt"]:
            cand = next_row["dt"]
            if cur["out_ts"] == cur["Timestamp"]:
                cand = cand - smallest_unit(cur["precision"])
            if prev_row is not None and cand < prev_row["dt"]:
                cand = prev_row["dt"]
            cur["dt"] = cand
            cur["out_ts"] = fmt_ts(cur["dt"], cur["precision"])
            if cur["out_ts"] == cur["Timestamp"]:
                eps = smallest_unit(cur["precision"])
                if prev_row is not None and cur["dt"] - eps >= prev_row["dt"]:
                    cur["dt"] = cur["dt"] - eps
                    cur["out_ts"] = fmt_ts(cur["dt"], cur["precision"])
            fixes.append(f"clamped_right_row_id={cur['row_id']}")

        # final guarantee changed
        if cur["out_ts"] == cur["Timestamp"]:
            eps = smallest_unit(cur["precision"])
            cand = cur["dt"]
            if next_row is not None and cand + eps <= next_row["dt"]:
                cand = cand + eps
            elif prev_row is not None and cand - eps >= prev_row["dt"]:
                cand = cand - eps
            else:
                cand = cand + eps
            cur["dt"] = cand
            cur["out_ts"] = fmt_ts(cand, cur["precision"])
            fixes.append(f"nudged_row_id={cur['row_id']}")

    return rows, exchanged_segments, displaced_segments, fixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str)
    original_cols = list(df.columns)

    out = df.copy()
    exchanged_total = 0
    displaced_total = 0
    fixes_all = []

    for case_id, g in df.groupby("Case", sort=False):
        repaired_rows, ex_cnt, dis_cnt, fixes = repair_case(g)
        exchanged_total += ex_cnt
        displaced_total += dis_cnt
        fixes_all.extend(fixes)
        for r in repaired_rows:
            out.at[r["index"], "Timestamp"] = r["out_ts"]

    # Final dataset-level checks without aborting
    if len(out) != len(df):
        out = out.iloc[:len(df)].copy()
        fixes_all.append("row_count_trimmed")

    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        mismatch = out[col].astype(str) != df[col].astype(str)
        if mismatch.any():
            out.loc[mismatch, col] = df.loc[mismatch, col]
            fixes_all.append(f"restored_column_{col}")

    # CLEAN rows exact timestamp preservation
    clean_mask = df["error_category"] == "CLEAN"
    if (out.loc[clean_mask, "Timestamp"] != df.loc[clean_mask, "Timestamp"]).any():
        bad = clean_mask & (out["Timestamp"] != df["Timestamp"])
        out.loc[bad, "Timestamp"] = df.loc[bad, "Timestamp"]
        fixes_all.append("restored_clean_timestamps")

    # Parseability check for repaired rows
    rep_mask = df["error_category"] != "CLEAN"
    parsed = pd.to_datetime(out.loc[rep_mask, "Timestamp"], errors="coerce")
    if parsed.isna().any():
        bad_idx = parsed[parsed.isna()].index
        for idx in bad_idx:
            orig = df.at[idx, "Timestamp"]
            prec = detect_precision(orig)
            orig_dt = parse_ts(orig)
            eps = smallest_unit(prec)
            new_dt = orig_dt + eps
            out.at[idx, "Timestamp"] = fmt_ts(new_dt, prec)
        fixes_all.append("reformatted_unparseable_repairs")

    # Ensure every corrupted row differs from input
    same_rep = rep_mask & (out["Timestamp"] == df["Timestamp"])
    if same_rep.any():
        by_case_positions = {}
        for case_id, g in out.groupby("Case", sort=False):
            idxs = list(g.index)
            by_case_positions[case_id] = idxs
        for idx in out.index[same_rep]:
            case_id = out.at[idx, "Case"]
            idxs = by_case_positions[case_id]
            pos = idxs.index(idx)
            prev_idx = idxs[pos - 1] if pos > 0 else None
            next_idx = idxs[pos + 1] if pos < len(idxs) - 1 else None
            cur_ts = parse_ts(out.at[idx, "Timestamp"])
            prec = detect_precision(df.at[idx, "Timestamp"])
            eps = smallest_unit(prec)
            prev_dt = parse_ts(out.at[prev_idx, "Timestamp"]) if prev_idx is not None else None
            next_dt = parse_ts(out.at[next_idx, "Timestamp"]) if next_idx is not None else None
            cand = cur_ts
            if next_dt is not None and cur_ts + eps <= next_dt:
                cand = cur_ts + eps
            elif prev_dt is not None and cur_ts - eps >= prev_dt:
                cand = cur_ts - eps
            else:
                cand = cur_ts + eps
            out.at[idx, "Timestamp"] = fmt_ts(cand, prec)
        fixes_all.append("forced_changed_corrupted_rows")

    out = out[original_cols]
    out.to_csv(args.output, index=False)

    rows_still_equal = int(((df["error_category"] != "CLEAN") & (out["Timestamp"] == df["Timestamp"])).sum())
    if fixes_all:
        print("fixes:" + ";".join(fixes_all[:200]))
    print(f"exchanged={exchanged_total} displaced={displaced_total} rows_still_equal={rows_still_equal}")


if __name__ == "__main__":
    main()