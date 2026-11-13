import argparse
import csv
import itertools
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
import pandas as pd


TRANSITION_PROBS = {
    ("Deliver to customer", "EVENT 8 END"): 1.0,
    ("Signal food complete", "Bring food"): 0.9985,
    ("Bring food", "Deliver to customer"): 0.9972,
    ("Prepare main course", "Signal food complete"): 0.9913,
    ("start", "Take order"): 0.9708,
    ("Bring drinks", "Review drinks"): 0.9554,
    ("Take order", "Review order"): 0.9464,
    ("prepare drinks", "Bring drinks"): 0.8979,
    ("Review order", "prepare drinks"): 0.7727,
    ("Prepare starter", "prepare drinks"): 0.5310,
    ("Review drinks", "Prepare main course"): 0.4767,
    ("Review drinks", "prepare drinks"): 0.3939,
    ("Prepare starter", "Bring drinks"): 0.2706,
    ("Review order", "Prepare starter"): 0.1765,
    ("Prepare starter", "Prepare main course"): 0.0860,
    ("Review drinks", "Signal food complete"): 0.0681,
    ("prepare drinks", "Prepare starter"): 0.0596,
    ("Prepare starter", "Review drinks"): 0.0583,
    ("Prepare starter", "Signal food complete"): 0.0481,
    ("Review drinks", "Prepare starter"): 0.0392,
    ("Take order", "Review drinks"): 0.0218,
    ("Review order", "Review drinks"): 0.0181,
    ("prepare drinks", "Bring food"): 0.0155,
    ("Review drinks", "Bring food"): 0.0149,
    ("Review order", "Prepare main course"): 0.0133,
    ("Bring drinks", "Signal food complete"): 0.0129,
    ("start", "Bring drinks"): 0.0124,
    ("Take order", "Bring drinks"): 0.0124,
    ("prepare drinks", "Signal food complete"): 0.0121,
    ("Bring drinks", "Prepare starter"): 0.0119,
    ("Take order", "Prepare main course"): 0.0117,
    ("Bring drinks", "Bring food"): 0.0110,
    ("Review order", "Signal food complete"): 0.0107,
    ("start", "Review drinks"): 0.0092,
    ("Review order", "Bring drinks"): 0.0087,
    ("prepare drinks", "Prepare main course"): 0.0068,
    ("prepare drinks", "Review drinks"): 0.0062,
    ("Prepare starter", "Bring food"): 0.0060,
    ("start", "prepare drinks"): 0.0050,
    ("Take order", "prepare drinks"): 0.0049,
    ("Prepare main course", "Bring drinks"): 0.0046,
    ("Bring drinks", "Bring drinks"): 0.0046,
    ("Review drinks", "Review drinks"): 0.0040,
    ("Bring drinks", "Prepare main course"): 0.0034,
    ("Prepare main course", "prepare drinks"): 0.0031,
    ("Take order", "Prepare starter"): 0.0027,
    ("start", "Prepare starter"): 0.0026,
    ("Bring food", "Bring drinks"): 0.0022,
    ("prepare drinks", "prepare drinks"): 0.0019,
    ("Review drinks", "Deliver to customer"): 0.0019,
    ("Review drinks", "Bring drinks"): 0.0013,
    ("Prepare main course", "Review drinks"): 0.0010,
    ("Bring drinks", "prepare drinks"): 0.0009,
    ("Signal food complete", "prepare drinks"): 0.0009,
    ("Signal food complete", "Bring drinks"): 0.0006,
    ("Bring food", "prepare drinks"): 0.0006,
}

DURATION_PROFILES = {
    ("Bring drinks", "Review drinks"): {"p10": 6.1307, "p25": 17.0105, "median": 41.052, "share": 0.631},
    ("prepare drinks", "Bring drinks"): {"p10": 28.8126, "p25": 79.71675, "median": 195.9745, "share": 0.198},
    ("Deliver to customer", "EVENT 8 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share": 1.0},
    ("Bring food", "Deliver to customer"): {"p10": 12.5235, "p25": 34.57875, "median": 85.062, "share": 0.389},
    ("start", "Take order"): {"p10": 12.9101, "p25": 35.79525, "median": 89.148, "share": 0.378},
    ("Signal food complete", "Bring food"): {"p10": 33.0276, "p25": 90.3885, "median": 214.9425, "share": 0.176},
    ("Take order", "Review order"): {"p10": 6.1912, "p25": 17.985, "median": 41.872, "share": 0.636},
    ("Prepare main course", "Signal food complete"): {"p10": 85943.671, "p25": 253686.04975, "median": 513669.015, "share": 0.001},
    ("Review drinks", "Prepare main course"): {"p10": 83921.2662, "p25": 243350.551, "median": 500815.099, "share": 0.001},
    ("Review order", "prepare drinks"): {"p10": 25.252, "p25": 68.49925, "median": 173.713, "share": 0.226},
    ("Review drinks", "prepare drinks"): {"p10": 30.7213, "p25": 85.6735, "median": 209.569, "share": 0.182},
    ("Prepare starter", "prepare drinks"): {"p10": 30.3582, "p25": 89.143, "median": 206.2, "share": 0.19},
    ("Review order", "Prepare starter"): {"p10": 15.008, "p25": 42.552, "median": 97.24, "share": 0.354},
    ("Review drinks", "Signal food complete"): {"p10": 7451.2538, "p25": 175593.462, "median": 445342.856, "share": 0.002},
    ("Prepare starter", "Bring drinks"): {"p10": 29.0292, "p25": 86.16725, "median": 203.727, "share": 0.189},
    ("prepare drinks", "Prepare starter"): {"p10": 15.8778, "p25": 40.819, "median": 100.292, "share": 0.362},
    ("Review drinks", "Prepare starter"): {"p10": 17.9444, "p25": 63.83, "median": 151.424, "share": 0.232},
    ("Prepare starter", "Prepare main course"): {"p10": 76289.6188, "p25": 192412.7035, "median": 445174.842, "share": 0.0},
    ("prepare drinks", "Bring food"): {"p10": 39201.883, "p25": 406754.148, "median": 785107.18, "share": 0.009},
    ("Review drinks", "Bring food"): {"p10": 161178.253, "p25": 593624.664, "median": 857938.338, "share": 0.009},
    ("Take order", "Review drinks"): {"p10": 257.7665, "p25": 389.71375, "median": 707.2185, "share": 0.0},
    ("Prepare starter", "Review drinks"): {"p10": 11.5194, "p25": 22.425, "median": 56.303, "share": 0.505},
    ("Bring drinks", "Signal food complete"): {"p10": 179812.449, "p25": 593579.078, "median": 883902.239, "share": 0.0},
    ("prepare drinks", "Signal food complete"): {"p10": 84583.4036, "p25": 431151.8905, "median": 982509.744, "share": 0.011},
    ("Bring drinks", "Prepare starter"): {"p10": 4.3939, "p25": 15.89925, "median": 33.688, "share": 0.693},
    ("Review order", "Review drinks"): {"p10": 303.721, "p25": 381.153, "median": 593.835, "share": 0.0},
    ("Bring drinks", "Bring food"): {"p10": 102484.185, "p25": 366203.855, "median": 696631.885, "share": 0.0},
    ("Prepare starter", "Signal food complete"): {"p10": 88210.8159, "p25": 276729.8375, "median": 519007.665, "share": 0.0},
    ("start", "Bring drinks"): {"p10": 275.2757, "p25": 422.42225, "median": 695.196, "share": 0.0},
    ("Review order", "Prepare main course"): {"p10": 943.183, "p25": 84467.297, "median": 444605.508, "share": 0.0},
    ("Take order", "Bring drinks"): {"p10": 296.2082, "p25": 433.589, "median": 624.287, "share": 0.0},
    ("Take order", "Prepare main course"): {"p10": 101957.3608, "p25": 258966.497, "median": 513968.537, "share": 0.0},
    ("prepare drinks", "Prepare main course"): {"p10": 71.18, "p25": 281.63, "median": 160626.596, "share": 0.078},
    ("Review order", "Signal food complete"): {"p10": 152688.052, "p25": 587731.024, "median": 862530.877, "share": 0.0},
    ("start", "Review drinks"): {"p10": 281.845, "p25": 455.605, "median": 733.167, "share": 0.0},
    ("prepare drinks", "Review drinks"): {"p10": 368.894, "p25": 508.0155, "median": 656.5765, "share": 0.0},
    ("Review order", "Bring drinks"): {"p10": 298.812, "p25": 538.94575, "median": 784.107, "share": 0.0},
    ("Bring drinks", "Bring drinks"): {"p10": 237.8929, "p25": 350.94375, "median": 571.916, "share": 0.0},
    ("Review drinks", "Review drinks"): {"p10": 283.804, "p25": 471.11475, "median": 695.439, "share": 0.0},
    ("start", "prepare drinks"): {"p10": 255.112, "p25": 319.999, "median": 413.649, "share": 0.0},
    ("Bring drinks", "Prepare main course"): {"p10": 53.072, "p25": 21489.537, "median": 438128.715, "share": 0.12},
    ("Take order", "prepare drinks"): {"p10": 214.6615, "p25": 304.47475, "median": 736.587, "share": 0.0},
    ("Prepare main course", "Bring drinks"): {"p10": 28.2273, "p25": 68.60625, "median": 129.8, "share": 0.222},
    ("Review drinks", "Deliver to customer"): {"p10": 13.1735, "p25": 26.957, "median": 70.7255, "share": 0.357},
    ("prepare drinks", "prepare drinks"): {"p10": 430.3684, "p25": 495.2955, "median": 639.1405, "share": 0.0},
    ("start", "Prepare starter"): {"p10": 275.651, "p25": 370.046, "median": 404.992, "share": 0.0},
    ("Take order", "Prepare starter"): {"p10": 327.2842, "p25": 389.43, "median": 558.771, "share": 0.0},
    ("Prepare main course", "prepare drinks"): {"p10": 94.8512, "p25": 110.78, "median": 178.7615, "share": 0.0},
    ("Bring food", "Bring drinks"): {"p10": 50.942, "p25": 152.058, "median": 229.533, "share": 0.182},
    ("Review drinks", "Bring drinks"): {"p10": 434.9222, "p25": 666.72275, "median": 1267.9125, "share": 0.0},
    ("Prepare starter", "Bring food"): {"p10": 167409.7799, "p25": 293248.85775, "median": 525665.88, "share": 0.0},
    ("Bring drinks", "prepare drinks"): {"p10": 691.6136, "p25": 983.649, "median": 1736.597, "share": 0.0},
    ("Signal food complete", "prepare drinks"): {"p10": 65.3186, "p25": 153.056, "median": 228.6855, "share": 0.25},
    ("Prepare main course", "Review drinks"): {"p10": 2.2898, "p25": 3.674, "median": 35.4335, "share": 0.5},
    ("Bring food", "prepare drinks"): {"p10": 164.9284, "p25": 166.546, "median": 169.242, "share": 0.0},
    ("Signal food complete", "Bring drinks"): {"p10": 150.102, "p25": 179.118, "median": 227.478, "share": 0.0},
}

FREQUENT_VARIANTS = [
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Prepare starter", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Prepare starter", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "prepare drinks", "Bring drinks", "Review drinks", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Prepare starter", "prepare drinks", "Bring drinks", "Review drinks", "prepare drinks", "Bring drinks", "Review drinks", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
    ["start", "Take order", "Review order", "Review drinks", "Prepare main course", "Signal food complete", "Bring food", "Deliver to customer", "EVENT 8 END"],
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def detect_precision(ts: str) -> int:
    if "." in ts:
        frac = ts.split(".")[-1]
        return len(frac)
    return 0


def round_datetime_to_precision(dt: pd.Timestamp, precision: int) -> pd.Timestamp:
    if precision <= 0:
        return dt.round("S")
    if precision <= 6:
        unit = {1: "100ms", 2: "10ms", 3: "1ms", 4: "100us", 5: "10us", 6: "1us"}[precision]
        return dt.round(unit)
    return dt


def format_ts(dt: pd.Timestamp, precision: int) -> str:
    dt = round_datetime_to_precision(dt, precision)
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    if precision == 0:
        return base
    frac = f"{dt.microsecond:06d}"
    if precision <= 6:
        return base + "." + frac[:precision]
    return base + "." + frac.ljust(precision, "0")


def timedelta_seconds(sec: float) -> pd.Timedelta:
    micros = int((Decimal(str(sec)) * Decimal("1000000")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return pd.Timedelta(microseconds=micros)


def get_gap_seconds(a: str, b: str) -> float:
    prof = DURATION_PROFILES.get((a, b))
    if prof:
        if prof["share"] >= 0.5:
            return max(1.0, prof["p10"])
        return max(1.0, prof["p25"])
    return 1.0


def edge_exists(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return True
    return (a, b) in TRANSITION_PROBS


def transition_score(a: Optional[str], b: Optional[str]) -> float:
    if a is None or b is None:
        return 0.0
    p = TRANSITION_PROBS.get((a, b))
    if p is None:
        return -20.0
    return 10.0 * p


def contiguous_subsequence_boost(left_act: Optional[str], seq: List[str], right_act: Optional[str]) -> float:
    pattern = []
    if left_act is not None:
        pattern.append(left_act)
    pattern.extend(seq)
    if right_act is not None:
        pattern.append(right_act)
    best = 0.0
    for var in FREQUENT_VARIANTS:
        m = len(pattern)
        if m > len(var):
            continue
        for i in range(len(var) - m + 1):
            if var[i:i + m] == pattern:
                best = max(best, 15.0)
    if best > 0:
        return best
    inner_best = 0.0
    for var in FREQUENT_VARIANTS:
        m = len(seq)
        if m > len(var):
            continue
        for i in range(len(var) - m + 1):
            if var[i:i + m] == seq:
                inner_best = max(inner_best, 8.0)
    return inner_best


def score_order(order: List[int], acts: List[str], left_act: Optional[str], right_act: Optional[str]) -> Tuple[float, int]:
    seq = [acts[i] for i in order]
    unknown = 0
    score = 0.0
    if left_act is not None:
        if not edge_exists(left_act, seq[0]):
            unknown += 1
        score += transition_score(left_act, seq[0])
    for a, b in zip(seq, seq[1:]):
        if not edge_exists(a, b):
            unknown += 1
        score += transition_score(a, b)
    if right_act is not None:
        if not edge_exists(seq[-1], right_act):
            unknown += 1
        score += transition_score(seq[-1], right_act)
    score += contiguous_subsequence_boost(left_act, seq, right_act)
    return score, unknown


def infer_order(acts: List[str], left_act: Optional[str], right_act: Optional[str]) -> List[int]:
    n = len(acts)
    idxs = list(range(n))
    perms = itertools.permutations(idxs)
    best = None
    best_key = None
    for perm in perms:
        score, unknown = score_order(list(perm), acts, left_act, right_act)
        key = (unknown, -score)
        if best is None or key < best_key:
            best = list(perm)
            best_key = key
    return best if best is not None else idxs


def build_times_forward(observed: pd.Timestamp,
                        ordered_acts: List[str],
                        right_anchor_time: Optional[pd.Timestamp],
                        precision: int) -> Tuple[List[pd.Timestamp], bool]:
    k = len(ordered_acts)
    min_step = pd.Timedelta(seconds=1)
    times = [observed]
    for a, b in zip(ordered_acts, ordered_acts[1:]):
        gap = max(1.0, get_gap_seconds(a, b))
        times.append(times[-1] + timedelta_seconds(gap))

    clamped = False
    if right_anchor_time is not None and times[-1] > right_anchor_time:
        clamped = True
        room = (right_anchor_time - observed).total_seconds()
        min_required = max(k - 1, 0)
        if room < min_required:
            room = min_required
        room = max(room, min_required)
        candidate_gaps = []
        for a, b in zip(ordered_acts, ordered_acts[1:]):
            candidate_gaps.append(max(1.0, get_gap_seconds(a, b)))
        total = sum(candidate_gaps)
        if total <= 0:
            scaled_gaps = [1.0] * (k - 1)
        else:
            scale = room / total
            scaled_gaps = [max(1.0, g * scale) for g in candidate_gaps]
            while sum(scaled_gaps) > room + 1e-9:
                changed = False
                for i in range(len(scaled_gaps)):
                    if scaled_gaps[i] > 1.0:
                        scaled_gaps[i] = max(1.0, scaled_gaps[i] - 0.001)
                        changed = True
                        if sum(scaled_gaps) <= room + 1e-9:
                            break
                if not changed:
                    break
        times = [observed]
        for g in scaled_gaps:
            times.append(times[-1] + timedelta_seconds(g))

    rounded = []
    for i, t in enumerate(times):
        rt = round_datetime_to_precision(t, precision)
        if i > 0 and rt <= rounded[-1]:
            rt = rounded[-1] + min_step
            rt = round_datetime_to_precision(rt, precision)
            if rt <= rounded[-1]:
                rt = rounded[-1] + min_step
        rounded.append(rt)

    if right_anchor_time is not None and rounded[-1] > right_anchor_time:
        clamped = True
        latest_start = right_anchor_time - pd.Timedelta(seconds=(k - 1))
        if latest_start < observed:
            latest_start = observed
        rounded = [observed]
        for _ in range(1, k):
            rounded.append(rounded[-1] + pd.Timedelta(seconds=1))
        if rounded[-1] > right_anchor_time:
            delta = rounded[-1] - right_anchor_time
            rounded = [t - delta for t in rounded]
            if rounded[0] < observed:
                rounded[0] = observed
                for i in range(1, k):
                    if rounded[i] <= rounded[i-1]:
                        rounded[i] = rounded[i-1] + pd.Timedelta(seconds=1)
    return rounded, clamped


def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    df = df[original_cols].copy()
    df["parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    global_precision = max(detect_precision(ts) for ts in df["Timestamp"].astype(str).tolist())
    df["out_ts"] = df["Timestamp"]

    blocks_repaired = 0
    blocks_clamped = 0

    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        case_df = df.loc[idxs]
        case_start = case_df["parsed_ts"].min()
        case_end = case_df["parsed_ts"].max()

        i = 0
        while i < len(idxs):
            idx = idxs[i]
            if df.at[idx, "error_category"] == "CLEAN":
                i += 1
                continue

            shared = df.at[idx, "Timestamp"]
            block = [idx]
            j = i + 1
            while j < len(idxs):
                idx2 = idxs[j]
                if df.at[idx2, "error_category"] != "CLEAN" and df.at[idx2, "Timestamp"] == shared:
                    block.append(idx2)
                    j += 1
                else:
                    break

            observed_ts = pd.to_datetime(shared, errors="coerce")
            left_anchor_idx = idxs[i - 1] if i - 1 >= 0 else None
            right_anchor_idx = idxs[j] if j < len(idxs) else None

            left_act = df.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None and df.at[left_anchor_idx, "error_category"] == "CLEAN" else None
            right_act = df.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None and df.at[right_anchor_idx, "error_category"] == "CLEAN" else None
            right_anchor_time = df.at[right_anchor_idx, "parsed_ts"] if right_anchor_idx is not None and df.at[right_anchor_idx, "error_category"] == "CLEAN" else case_end

            acts = [df.at[b, "Activity"] for b in block]
            order = infer_order(acts, left_act, right_act)
            ordered_acts = [acts[o] for o in order]
            ordered_rows = [block[o] for o in order]

            new_times, clamped = build_times_forward(observed_ts, ordered_acts, right_anchor_time, global_precision)
            if clamped:
                blocks_clamped += 1

            if len(new_times) >= 2:
                min_span = pd.Timedelta(seconds=len(new_times) - 1)
                actual_span = new_times[-1] - new_times[0]
                if actual_span < min_span:
                    for z in range(1, len(new_times)):
                        if new_times[z] <= new_times[z - 1]:
                            new_times[z] = new_times[z - 1] + pd.Timedelta(seconds=1)

            if right_anchor_time is not None and new_times[-1] > right_anchor_time:
                clamped = True
                blocks_clamped += 1
                new_times = [observed_ts + pd.Timedelta(seconds=s) for s in range(len(block))]
                if new_times[-1] > right_anchor_time:
                    shift = new_times[-1] - right_anchor_time
                    new_times = [t - shift for t in new_times]
                    if new_times[0] < observed_ts:
                        new_times = [observed_ts + pd.Timedelta(seconds=s) for s in range(len(block))]

            for pos, row_idx in enumerate(ordered_rows):
                dt = new_times[pos]
                if dt < case_start:
                    dt = case_start
                if dt > case_end:
                    dt = case_end
                if pos > 0 and dt <= new_times[pos - 1]:
                    dt = new_times[pos - 1] + pd.Timedelta(seconds=1)
                new_times[pos] = dt

            for pos, row_idx in enumerate(ordered_rows):
                df.at[row_idx, "out_ts"] = format_ts(new_times[pos], global_precision)

            blocks_repaired += 1
            i = j

    rows_equal = 0

    for case_id, case_idx in df.groupby("Case", sort=False).groups.items():
        idxs = list(case_idx)
        i = 0
        while i < len(idxs):
            idx = idxs[i]
            if df.at[idx, "error_category"] == "CLEAN":
                if df.at[idx, "out_ts"] != df.at[idx, "Timestamp"]:
                    df.at[idx, "out_ts"] = df.at[idx, "Timestamp"]
                i += 1
                continue

            shared = df.at[idx, "Timestamp"]
            block = [idx]
            j = i + 1
            while j < len(idxs):
                idx2 = idxs[j]
                if df.at[idx2, "error_category"] != "CLEAN" and df.at[idx2, "Timestamp"] == shared:
                    block.append(idx2)
                    j += 1
                else:
                    break

            parsed_out = [pd.to_datetime(df.at[b, "out_ts"], errors="coerce") for b in block]
            acts = [df.at[b, "Activity"] for b in block]
            left_anchor_idx = idxs[i - 1] if i - 1 >= 0 else None
            right_anchor_idx = idxs[j] if j < len(idxs) else None
            left_act = df.at[left_anchor_idx, "Activity"] if left_anchor_idx is not None and df.at[left_anchor_idx, "error_category"] == "CLEAN" else None
            right_act = df.at[right_anchor_idx, "Activity"] if right_anchor_idx is not None and df.at[right_anchor_idx, "error_category"] == "CLEAN" else None

            order = infer_order(acts, left_act, right_act)
            ordered_rows = [block[o] for o in order]
            ordered_times = [pd.to_datetime(df.at[r, "out_ts"], errors="coerce") for r in ordered_rows]
            observed_ts = pd.to_datetime(shared, errors="coerce")
            right_anchor_time = df.at[right_anchor_idx, "parsed_ts"] if right_anchor_idx is not None and df.at[right_anchor_idx, "error_category"] == "CLEAN" else df.loc[idxs, "parsed_ts"].max()

            need_fix = False
            if ordered_times[0] != observed_ts:
                need_fix = True
            for a, b in zip(ordered_times, ordered_times[1:]):
                if not (b > a):
                    need_fix = True
                    break
            unchanged = sum(1 for b in block if df.at[b, "out_ts"] == df.at[b, "Timestamp"])
            if unchanged != 1:
                need_fix = True
            if (ordered_times[-1] - ordered_times[0]).total_seconds() < max(len(block) - 1, 0):
                need_fix = True
            if right_anchor_time is not None and ordered_times[-1] > right_anchor_time:
                need_fix = True

            if need_fix:
                ordered_acts = [acts[o] for o in order]
                fixed_times = [observed_ts + pd.Timedelta(seconds=s) for s in range(len(block))]
                if right_anchor_time is not None and fixed_times[-1] > right_anchor_time:
                    room = max(int((right_anchor_time - observed_ts).total_seconds()), len(block) - 1)
                    fixed_times = [observed_ts + pd.Timedelta(seconds=min(s, room)) for s in range(len(block))]
                    for z in range(1, len(fixed_times)):
                        if fixed_times[z] <= fixed_times[z - 1]:
                            fixed_times[z] = fixed_times[z - 1] + pd.Timedelta(seconds=1)
                for pos, row_idx in enumerate(ordered_rows):
                    df.at[row_idx, "out_ts"] = format_ts(fixed_times[pos], global_precision)
                blocks_clamped += 1

            i = j

    rows_equal = int((df["out_ts"] == df["Timestamp"]).sum())

    out_df = df[["Case", "Activity", "out_ts", "Resource", "error_category", "row_id"]].copy()
    out_df.columns = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"])
        for row in out_df.itertuples(index=False, name=None):
            writer.writerow(row)

    print(f"blocks_repaired={blocks_repaired} blocks_clamped={blocks_clamped} rows_still_equal_to_input={rows_equal}")


if __name__ == "__main__":
    main()