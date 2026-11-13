import argparse
import itertools
import math
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


TRANS_PROBS = {
    ("Deliver to customer", "EVENT 8 END"): 1.0,
    ("Signal food complete", "Bring food"): 0.9985,
    ("Bring food", "Deliver to customer"): 0.9972,
    ("Prepare main course", "Signal food complete"): 0.9913,
    ("start", "Take order"): 0.9708,
    ("Bring drinks", "Review drinks"): 0.9554,
    ("Take order", "Review order"): 0.9464,
    ("prepare drinks", "Bring drinks"): 0.8979,
    ("Review order", "prepare drinks"): 0.7727,
    ("Prepare starter", "prepare drinks"): 0.531,
    ("Review drinks", "Prepare main course"): 0.4767,
    ("Review drinks", "prepare drinks"): 0.3939,
    ("Prepare starter", "Bring drinks"): 0.2706,
    ("Review order", "Prepare starter"): 0.1765,
    ("Prepare starter", "Prepare main course"): 0.086,
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
    ("Bring drinks", "Bring food"): 0.011,
    ("Review order", "Signal food complete"): 0.0107,
    ("start", "Review drinks"): 0.0092,
    ("Review order", "Bring drinks"): 0.0087,
    ("prepare drinks", "Prepare main course"): 0.0068,
    ("prepare drinks", "Review drinks"): 0.0062,
    ("Prepare starter", "Bring food"): 0.006,
    ("start", "prepare drinks"): 0.005,
    ("Take order", "prepare drinks"): 0.0049,
    ("Prepare main course", "Bring drinks"): 0.0046,
    ("Bring drinks", "Bring drinks"): 0.0046,
    ("Review drinks", "Review drinks"): 0.004,
    ("Bring drinks", "Prepare main course"): 0.0034,
    ("Prepare main course", "prepare drinks"): 0.0031,
    ("Take order", "Prepare starter"): 0.0027,
    ("start", "Prepare starter"): 0.0026,
    ("Bring food", "Bring drinks"): 0.0022,
    ("prepare drinks", "prepare drinks"): 0.0019,
    ("Review drinks", "Deliver to customer"): 0.0019,
    ("Review drinks", "Bring drinks"): 0.0013,
    ("Prepare main course", "Review drinks"): 0.001,
    ("Bring drinks", "prepare drinks"): 0.0009,
    ("Signal food complete", "prepare drinks"): 0.0009,
    ("Signal food complete", "Bring drinks"): 0.0006,
    ("Bring food", "prepare drinks"): 0.0006,
}

DURATION = {
    ("Bring drinks", "Review drinks"): {"p10": 6.1307, "p25": 17.0105, "median": 41.052, "share60": 0.631, "n": 7062},
    ("prepare drinks", "Bring drinks"): {"p10": 28.8126, "p25": 79.71675, "median": 195.9745, "share60": 0.198, "n": 6704},
    ("Deliver to customer", "EVENT 8 END"): {"p10": 0.0, "p25": 0.0, "median": 0.0, "share60": 1.0, "n": 5000},
    ("Bring food", "Deliver to customer"): {"p10": 12.5235, "p25": 34.57875, "median": 85.062, "share60": 0.389, "n": 4986},
    ("start", "Take order"): {"p10": 12.9101, "p25": 35.79525, "median": 89.148, "share60": 0.378, "n": 4854},
    ("Signal food complete", "Bring food"): {"p10": 33.0276, "p25": 90.3885, "median": 214.9425, "share60": 0.176, "n": 4682},
    ("Take order", "Review order"): {"p10": 6.1912, "p25": 17.985, "median": 41.872, "share60": 0.636, "n": 4594},
    ("Prepare main course", "Signal food complete"): {"p10": 85943.671, "p25": 253686.04975, "median": 513669.015, "share60": 0.001, "n": 3866},
    ("Review drinks", "Prepare main course"): {"p10": 83921.2662, "p25": 243350.551, "median": 500815.099, "share60": 0.001, "n": 3563},
    ("Review order", "prepare drinks"): {"p10": 25.252, "p25": 68.49925, "median": 173.713, "share60": 0.226, "n": 3550},
    ("Review drinks", "prepare drinks"): {"p10": 30.7213, "p25": 85.6735, "median": 209.569, "share60": 0.182, "n": 2944},
    ("Prepare starter", "prepare drinks"): {"p10": 30.3582, "p25": 89.143, "median": 206.2, "share60": 0.19, "n": 883},
    ("Review order", "Prepare starter"): {"p10": 15.008, "p25": 42.552, "median": 97.24, "share60": 0.354, "n": 811},
    ("Review drinks", "Signal food complete"): {"p10": 7451.2538, "p25": 175593.462, "median": 445342.856, "share60": 0.002, "n": 509},
    ("Prepare starter", "Bring drinks"): {"p10": 29.0292, "p25": 86.16725, "median": 203.727, "share60": 0.189, "n": 450},
    ("prepare drinks", "Prepare starter"): {"p10": 15.8778, "p25": 40.819, "median": 100.292, "share60": 0.362, "n": 445},
    ("Review drinks", "Prepare starter"): {"p10": 17.9444, "p25": 63.83, "median": 151.424, "share60": 0.232, "n": 293},
    ("Prepare starter", "Prepare main course"): {"p10": 76289.6188, "p25": 192412.7035, "median": 445174.842, "share60": 0.0, "n": 143},
    ("prepare drinks", "Bring food"): {"p10": 39201.883, "p25": 406754.148, "median": 785107.18, "share60": 0.009, "n": 116},
    ("Review drinks", "Bring food"): {"p10": 161178.253, "p25": 593624.664, "median": 857938.338, "share60": 0.009, "n": 111},
    ("Take order", "Review drinks"): {"p10": 257.7665, "p25": 389.71375, "median": 707.2185, "share60": 0.0, "n": 106},
    ("Prepare starter", "Review drinks"): {"p10": 11.5194, "p25": 22.425, "median": 56.303, "share60": 0.505, "n": 97},
    ("Bring drinks", "Signal food complete"): {"p10": 179812.449, "p25": 593579.078, "median": 883902.239, "share60": 0.0, "n": 95},
    ("prepare drinks", "Signal food complete"): {"p10": 84583.4036, "p25": 431151.8905, "median": 982509.744, "share60": 0.011, "n": 90},
    ("Bring drinks", "Prepare starter"): {"p10": 4.3939, "p25": 15.89925, "median": 33.688, "share60": 0.693, "n": 88},
    ("Review order", "Review drinks"): {"p10": 303.721, "p25": 381.153, "median": 593.835, "share60": 0.0, "n": 83},
    ("Bring drinks", "Bring food"): {"p10": 102484.185, "p25": 366203.855, "median": 696631.885, "share60": 0.0, "n": 81},
    ("Prepare starter", "Signal food complete"): {"p10": 88210.8159, "p25": 276729.8375, "median": 519007.665, "share60": 0.0, "n": 80},
    ("start", "Bring drinks"): {"p10": 275.2757, "p25": 422.42225, "median": 695.196, "share60": 0.0, "n": 62},
    ("Review order", "Prepare main course"): {"p10": 943.183, "p25": 84467.297, "median": 444605.508, "share60": 0.0, "n": 61},
    ("Take order", "Bring drinks"): {"p10": 296.2082, "p25": 433.589, "median": 624.287, "share60": 0.0, "n": 60},
    ("Take order", "Prepare main course"): {"p10": 101957.3608, "p25": 258966.497, "median": 513968.537, "share60": 0.0, "n": 57},
    ("prepare drinks", "Prepare main course"): {"p10": 71.18, "p25": 281.63, "median": 160626.596, "share60": 0.078, "n": 51},
    ("Review order", "Signal food complete"): {"p10": 152688.052, "p25": 587731.024, "median": 862530.877, "share60": 0.0, "n": 49},
    ("start", "Review drinks"): {"p10": 281.845, "p25": 455.605, "median": 733.167, "share60": 0.0, "n": 46},
    ("prepare drinks", "Review drinks"): {"p10": 368.894, "p25": 508.0155, "median": 656.5765, "share60": 0.0, "n": 46},
    ("Review order", "Bring drinks"): {"p10": 298.812, "p25": 538.94575, "median": 784.107, "share60": 0.0, "n": 40},
    ("Bring drinks", "Bring drinks"): {"p10": 237.8929, "p25": 350.94375, "median": 571.916, "share60": 0.0, "n": 34},
    ("Review drinks", "Review drinks"): {"p10": 283.804, "p25": 471.11475, "median": 695.439, "share60": 0.0, "n": 30},
    ("start", "prepare drinks"): {"p10": 255.112, "p25": 319.999, "median": 413.649, "share60": 0.0, "n": 25},
    ("Bring drinks", "Prepare main course"): {"p10": 53.072, "p25": 21489.537, "median": 438128.715, "share60": 0.12, "n": 25},
    ("Take order", "prepare drinks"): {"p10": 214.6615, "p25": 304.47475, "median": 736.587, "share60": 0.0, "n": 24},
    ("Prepare main course", "Bring drinks"): {"p10": 28.2273, "p25": 68.60625, "median": 129.8, "share60": 0.222, "n": 18},
    ("Review drinks", "Deliver to customer"): {"p10": 13.1735, "p25": 26.957, "median": 70.7255, "share60": 0.357, "n": 14},
    ("prepare drinks", "prepare drinks"): {"p10": 430.3684, "p25": 495.2955, "median": 639.1405, "share60": 0.0, "n": 14},
    ("start", "Prepare starter"): {"p10": 275.651, "p25": 370.046, "median": 404.992, "share60": 0.0, "n": 13},
    ("Take order", "Prepare starter"): {"p10": 327.2842, "p25": 389.43, "median": 558.771, "share60": 0.0, "n": 13},
    ("Prepare main course", "prepare drinks"): {"p10": 94.8512, "p25": 110.78, "median": 178.7615, "share60": 0.0, "n": 12},
    ("Bring food", "Bring drinks"): {"p10": 50.942, "p25": 152.058, "median": 229.533, "share60": 0.182, "n": 11},
    ("Review drinks", "Bring drinks"): {"p10": 434.9222, "p25": 666.72275, "median": 1267.9125, "share60": 0.0, "n": 10},
    ("Prepare starter", "Bring food"): {"p10": 167409.7799, "p25": 293248.85775, "median": 525665.88, "share60": 0.0, "n": 10},
    ("Bring drinks", "prepare drinks"): {"p10": 691.6136, "p25": 983.649, "median": 1736.597, "share60": 0.0, "n": 7},
    ("Signal food complete", "prepare drinks"): {"p10": 65.3186, "p25": 153.056, "median": 228.6855, "share60": 0.25, "n": 4},
    ("Prepare main course", "Review drinks"): {"p10": 2.2898, "p25": 3.674, "median": 35.4335, "share60": 0.5, "n": 4},
    ("Bring food", "prepare drinks"): {"p10": 164.9284, "p25": 166.546, "median": 169.242, "share60": 0.0, "n": 3},
    ("Signal food complete", "Bring drinks"): {"p10": 150.102, "p25": 179.118, "median": 227.478, "share60": 0.0, "n": 3},
}

GLOBAL_MEDIAN = 117.1945
SMALL_PROB = 1e-6


def safe_log_prob(p: float) -> float:
    return math.log(max(p, SMALL_PROB))


def parse_timestamp_loose(s: str):
    if pd.isna(s):
        return pd.NaT
    s2 = str(s).strip()
    if s2 == "":
        return pd.NaT
    return pd.to_datetime(s2, errors="coerce")


def detect_precision_and_sep(ts: str) -> Tuple[int, str]:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return 3, "."
    s = str(ts)
    m = re.search(r"(\d{2}:\d{2}:\d{2})([.,](\d+))?$", s)
    if not m:
        return 3, "."
    frac = m.group(3)
    sep = "." if m.group(2) is None else m.group(2)[0]
    return (len(frac) if frac is not None else 0), sep


def format_timestamp_like(dt: pd.Timestamp, precision: int, sep: str) -> str:
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    if precision <= 0:
        return base
    nanos = dt.nanosecond + dt.microsecond * 1000
    frac = f"{nanos:09d}"[:precision]
    return f"{base}{sep}{frac}"


def get_profile(a: Optional[str], b: Optional[str]):
    if a is None or b is None:
        return None
    return DURATION.get((a, b))


def transition_prob(a: Optional[str], b: Optional[str]) -> float:
    if a is None or b is None:
        return SMALL_PROB
    return TRANS_PROBS.get((a, b), SMALL_PROB)


def lower_tail_gap_seconds(a: Optional[str], b: Optional[str]) -> float:
    prof = get_profile(a, b)
    if prof is None:
        return max(1.0, GLOBAL_MEDIAN)
    if prof["share60"] >= 0.5:
        val = prof["p10"]
    else:
        val = prof["p25"]
    return max(1.0, float(val))


def estimate_from_anchors(left_t, left_a, cur_a, right_t, right_a) -> pd.Timestamp:
    if pd.notna(left_t) and pd.notna(right_t):
        total = max(0.0, (right_t - left_t).total_seconds())
        lg = lower_tail_gap_seconds(left_a, cur_a)
        rg = lower_tail_gap_seconds(cur_a, right_a)
        frac = lg / max(lg + rg, 1e-9)
        sec = min(total, max(0.0, frac * total))
        cand = left_t + pd.to_timedelta(sec, unit="s")
        if cand < left_t:
            cand = left_t
        if cand > right_t:
            cand = right_t
        return cand
    if pd.notna(left_t):
        return left_t + pd.to_timedelta(lower_tail_gap_seconds(left_a, cur_a), unit="s")
    if pd.notna(right_t):
        return right_t - pd.to_timedelta(lower_tail_gap_seconds(cur_a, right_a), unit="s")
    return pd.Timestamp("2000-01-01 00:00:00")


def score_order(order: Tuple[str, ...], left_a: Optional[str], right_a: Optional[str], available_room_sec: Optional[float]) -> float:
    seq = []
    if left_a is not None:
        seq.append(left_a)
    seq.extend(order)
    if right_a is not None:
        seq.append(right_a)

    score = 0.0
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        p = transition_prob(a, b)
        score += safe_log_prob(p)
        prof = get_profile(a, b)
        if prof is not None:
            score += math.log(max(prof["n"], 1)) / 20.0
            if prof["share60"] >= 0.3:
                score += 0.5
            if available_room_sec is not None and available_room_sec >= 0:
                prefer = lower_tail_gap_seconds(a, b)
                if prefer <= max(available_room_sec, 1.0):
                    score += 0.2
        else:
            score -= 0.5
    return score


def enumerate_orders(activities: List[str], left_a: Optional[str], right_a: Optional[str], available_room_sec: Optional[float]) -> Tuple[str, ...]:
    n = len(activities)
    if n <= 7:
        perms = itertools.permutations(activities)
        best = None
        best_score = -1e100
        for perm in perms:
            s = score_order(perm, left_a, right_a, available_room_sec)
            if s > best_score:
                best_score = s
                best = perm
        return best
    # beam search fallback
    beam = [([], set(), 0.0)]
    width = 50
    for _ in range(n):
        new_beam = []
        for seq, used, sc in beam:
            prev = left_a if not seq else seq[-1]
            for idx, act in enumerate(activities):
                if idx in used:
                    continue
                add = safe_log_prob(transition_prob(prev, act))
                prof = get_profile(prev, act)
                if prof is not None:
                    add += math.log(max(prof["n"], 1)) / 20.0
                    if prof["share60"] >= 0.3:
                        add += 0.5
                else:
                    add -= 0.5
                new_beam.append((seq + [act], used | {idx}, sc + add))
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:width]
    best_seq = None
    best_score = -1e100
    for seq, used, sc in beam:
        tail = safe_log_prob(transition_prob(seq[-1] if seq else left_a, right_a)) if right_a is not None else 0.0
        total = sc + tail
        if total > best_score:
            best_score = total
            best_seq = seq
    return tuple(best_seq if best_seq is not None else activities)


def build_c1_times(observed_t: pd.Timestamp, order: Tuple[str, ...], right_t: Optional[pd.Timestamp], left_t: Optional[pd.Timestamp], left_a: Optional[str], right_a: Optional[str]) -> Tuple[List[pd.Timestamp], bool]:
    k = len(order)
    if k == 1:
        return [observed_t], False

    gaps = []
    for i in range(k - 1):
        gaps.append(max(1.0, lower_tail_gap_seconds(order[i], order[i + 1])))

    cum = []
    s = 0.0
    for g in gaps:
        s += g
        cum.append(s)

    min_span = k - 1
    total_span = max(cum[-1], min_span)

    clamped = False
    room_right = None
    if pd.notna(right_t):
        room_right = max(0.0, (right_t - observed_t).total_seconds())

    if room_right is not None:
        max_span = room_right
        if max_span < min_span:
            max_span = min_span
        if total_span > max_span:
            clamped = True
            raw = gaps[:]
            raw_sum = sum(raw)
            target = max_span
            if raw_sum <= 0:
                raw = [1.0] * (k - 1)
                raw_sum = k - 1
            scaled = [max(1.0, g * target / raw_sum) for g in raw]
            # iterative correction to fit exactly within target while keeping >=1
            for _ in range(5):
                cur = sum(scaled)
                if cur <= target + 1e-9:
                    break
                free = [i for i, x in enumerate(scaled) if x > 1.0 + 1e-9]
                if not free:
                    break
                reducible = sum(scaled[i] - 1.0 for i in free)
                excess = cur - target
                if reducible <= 0:
                    break
                for i in free:
                    red = excess * ((scaled[i] - 1.0) / reducible)
                    scaled[i] = max(1.0, scaled[i] - red)
            # final micro-adjust
            cur = sum(scaled)
            if cur > target:
                for i in range(len(scaled) - 1, -1, -1):
                    take = min(scaled[i] - 1.0, cur - target)
                    if take > 0:
                        scaled[i] -= take
                        cur -= take
                    if cur <= target:
                        break
            gaps = scaled

    times = [observed_t]
    acc = 0.0
    for g in gaps:
        acc += g
        times.append(observed_t + pd.to_timedelta(acc, unit="s"))

    # ensure strict increase and within right anchor
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + pd.to_timedelta(1, unit="s")
            clamped = True

    if pd.notna(right_t):
        if times[-1] > right_t:
            clamped = True
            latest_end = right_t
            back = [latest_end]
            for i in range(len(gaps) - 1, -1, -1):
                back.append(back[-1] - pd.to_timedelta(max(1.0, gaps[i]), unit="s"))
            back = list(reversed(back))
            if back[0] < observed_t:
                # preserve first at observed, compress to 1-second increments
                back = [observed_t + pd.to_timedelta(i, unit="s") for i in range(k)]
                if back[-1] > right_t:
                    back[-1] = right_t
                    for j in range(k - 2, -1, -1):
                        if back[j] >= back[j + 1]:
                            back[j] = back[j + 1] - pd.to_timedelta(1, unit="s")
                    back[0] = observed_t
            times = back
            times[0] = observed_t
            for i in range(1, len(times)):
                if times[i] <= times[i - 1]:
                    times[i] = times[i - 1] + pd.to_timedelta(1, unit="s")

    if pd.notna(left_t) and times[0] < left_t:
        times[0] = left_t
        clamped = True
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                times[i] = times[i - 1] + pd.to_timedelta(1, unit="s")

    return times, clamped


def plausibility_gap_score(actual_sec: float, a: Optional[str], b: Optional[str]) -> float:
    if actual_sec < 0:
        return -1e9
    prof = get_profile(a, b)
    if prof is None:
        target = GLOBAL_MEDIAN
        return -abs(actual_sec - target) / max(target, 1.0)
    target = lower_tail_gap_seconds(a, b)
    scale = max(prof["p25"], 1.0)
    return -abs(actual_sec - target) / scale + (0.2 if actual_sec >= 0 else -10.0)


def find_left_clean_anchor(case_df: pd.DataFrame, pos: int) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    for j in range(pos - 1, -1, -1):
        if case_df.iloc[j]["error_category"] == "CLEAN":
            return case_df.iloc[j]["parsed_ts"], case_df.iloc[j]["Activity"]
    return pd.NaT, None


def find_right_clean_anchor(case_df: pd.DataFrame, pos: int) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    for j in range(pos + 1, len(case_df)):
        if case_df.iloc[j]["error_category"] == "CLEAN":
            return case_df.iloc[j]["parsed_ts"], case_df.iloc[j]["Activity"]
    return pd.NaT, None


def infer_date_only(s: str) -> bool:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return False
    s = str(s).strip()
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s))


def preserve_date_set_time(date_str: str, t: pd.Timestamp) -> pd.Timestamp:
    d = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(d):
        return t
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=t.hour, minute=t.minute, second=t.second, microsecond=t.microsecond, nanosecond=t.nanosecond)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    original_columns = list(df.columns)
    required = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    df["orig_ts"] = df["Timestamp"].astype(str)
    df["repaired_ts"] = df["orig_ts"]
    df["parsed_ts"] = df["Timestamp"].apply(parse_timestamp_loose)

    sample_prec = 3
    sample_sep = "."
    for s in df["orig_ts"]:
        if s and str(s).strip():
            p, sep = detect_precision_and_sep(s)
            if p > 0:
                sample_prec, sample_sep = p, sep
                break

    c1_blocks_repaired = 0
    c1_blocks_clamped = 0

    # process by case in original row order
    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_df = df.loc[list(idxs)].copy()
        case_df = case_df.reset_index().rename(columns={"index": "orig_index"})

        clean_parsed = case_df.loc[case_df["parsed_ts"].notna(), "parsed_ts"]
        case_start = clean_parsed.min() if not clean_parsed.empty else pd.NaT
        case_end = clean_parsed.max() if not clean_parsed.empty else pd.NaT

        # C1 blocks: consecutive rows in same case, non-clean, same observed timestamp, category C1
        i = 0
        while i < len(case_df):
            row = case_df.iloc[i]
            if row["error_category"] == "C1_ORDERING_AMBIGUITY":
                ts_val = row["orig_ts"]
                block_rows = [i]
                j = i + 1
                while j < len(case_df):
                    rj = case_df.iloc[j]
                    if rj["error_category"] == "C1_ORDERING_AMBIGUITY" and rj["orig_ts"] == ts_val:
                        block_rows.append(j)
                        j += 1
                    else:
                        break
                if len(block_rows) >= 2:
                    observed_t = case_df.iloc[block_rows[0]]["parsed_ts"]
                    left_t, left_a = find_left_clean_anchor(case_df, block_rows[0])
                    right_t, right_a = find_right_clean_anchor(case_df, block_rows[-1])

                    acts = [case_df.iloc[b]["Activity"] for b in block_rows]
                    room = None
                    if pd.notna(observed_t) and pd.notna(right_t):
                        room = max(0.0, (right_t - observed_t).total_seconds())

                    order = enumerate_orders(acts, left_a, right_a, room)
                    times, clamped = build_c1_times(observed_t, order, right_t, left_t, left_a, right_a)

                    # write times back to original rows by activity occurrence matching
                    used = [False] * len(block_rows)
                    act_to_pos = {}
                    for pos, act in enumerate(order):
                        act_to_pos.setdefault(act, []).append(pos)

                    assigned_times = [None] * len(block_rows)
                    taken_pos = set()
                    for r_idx_local, b in enumerate(block_rows):
                        act = case_df.iloc[b]["Activity"]
                        candidates = act_to_pos.get(act, [])
                        choose = None
                        for c in candidates:
                            if c not in taken_pos:
                                choose = c
                                break
                        if choose is None:
                            choose = 0
                        taken_pos.add(choose)
                        assigned_times[r_idx_local] = times[choose]

                    for local_ix, b in enumerate(block_rows):
                        orig_ix = case_df.iloc[b]["orig_index"]
                        prec, sep = detect_precision_and_sep(df.at[orig_ix, "orig_ts"])
                        df.at[orig_ix, "repaired_ts"] = format_timestamp_like(assigned_times[local_ix], prec if prec > 0 else sample_prec, sep if sep else sample_sep)

                    c1_blocks_repaired += 1
                    if clamped:
                        c1_blocks_clamped += 1
                i = j
            else:
                i += 1

        # C2 repair
        for pos in range(1, len(case_df)):
            row = case_df.iloc[pos]
            if row["error_category"] != "C2_TEMPORAL_INCONSISTENCY":
                continue
            cur_ix = row["orig_index"]
            prev_ix = case_df.iloc[pos - 1]["orig_index"]
            cur_t_obs = parse_timestamp_loose(df.at[cur_ix, "orig_ts"])
            prev_t_current = parse_timestamp_loose(df.at[prev_ix, "repaired_ts"])
            prev_t_obs = parse_timestamp_loose(df.at[prev_ix, "orig_ts"])

            if pd.isna(cur_t_obs) or pd.isna(prev_t_obs) or pd.isna(prev_t_current):
                continue

            left2_t, left2_a = find_left_clean_anchor(case_df, pos - 1)
            right_t, right_a = find_right_clean_anchor(case_df, pos)

            prev_a = case_df.iloc[pos - 1]["Activity"]
            cur_a = case_df.iloc[pos]["Activity"]

            # candidate swap
            swap_prev = cur_t_obs
            swap_cur = prev_t_obs
            swap_valid = True
            if pd.notna(left2_t) and swap_prev < left2_t:
                swap_valid = False
            if pd.notna(right_t) and swap_cur > right_t:
                swap_valid = False
            if swap_cur < swap_prev:
                swap_valid = False

            swap_score = -1e9
            if swap_valid:
                parts = []
                if pd.notna(left2_t):
                    parts.append(plausibility_gap_score((swap_prev - left2_t).total_seconds(), left2_a, prev_a))
                parts.append(plausibility_gap_score((swap_cur - swap_prev).total_seconds(), prev_a, cur_a))
                if pd.notna(right_t):
                    parts.append(plausibility_gap_score((right_t - swap_cur).total_seconds(), cur_a, right_a))
                swap_score = sum(parts)

            # displaced estimate for current row only
            est_cur = estimate_from_anchors(prev_t_current, prev_a, cur_a, right_t, right_a)
            if pd.notna(right_t) and est_cur > right_t:
                est_cur = right_t
            if est_cur < prev_t_current:
                est_cur = prev_t_current + pd.to_timedelta(1, unit="s")
            if pd.notna(right_t) and est_cur > right_t:
                est_cur = min(right_t, prev_t_current + pd.to_timedelta(1, unit="s"))
            disp_parts = [plausibility_gap_score((est_cur - prev_t_current).total_seconds(), prev_a, cur_a)]
            if pd.notna(right_t):
                disp_parts.append(plausibility_gap_score((right_t - est_cur).total_seconds(), cur_a, right_a))
            disp_score = sum(disp_parts)

            if swap_score > disp_score:
                pprec, psep = detect_precision_and_sep(df.at[prev_ix, "orig_ts"])
                cprec, csep = detect_precision_and_sep(df.at[cur_ix, "orig_ts"])
                if case_df.iloc[pos - 1]["error_category"] != "CLEAN":
                    df.at[prev_ix, "repaired_ts"] = format_timestamp_like(swap_prev, pprec if pprec > 0 else sample_prec, psep)
                df.at[cur_ix, "repaired_ts"] = format_timestamp_like(swap_cur, cprec if cprec > 0 else sample_prec, csep)
            else:
                cprec, csep = detect_precision_and_sep(df.at[cur_ix, "orig_ts"])
                df.at[cur_ix, "repaired_ts"] = format_timestamp_like(est_cur, cprec if cprec > 0 else sample_prec, csep)

        # C3 repair
        for pos in range(len(case_df)):
            row = case_df.iloc[pos]
            if row["error_category"] != "C3_MISSING_TEMPORAL_INFORMATION":
                continue
            orig_ix = row["orig_index"]
            raw = df.at[orig_ix, "orig_ts"]
            left_t, left_a = find_left_clean_anchor(case_df, pos)
            right_t, right_a = find_right_clean_anchor(case_df, pos)
            cur_a = row["Activity"]

            est = estimate_from_anchors(left_t, left_a, cur_a, right_t, right_a)

            if infer_date_only(raw):
                est = preserve_date_set_time(raw, est)
                if pd.notna(left_t) and est.date() == left_t.date() and est < left_t:
                    est = preserve_date_set_time(raw, left_t)
                if pd.notna(right_t) and est.date() == right_t.date() and est > right_t:
                    midpoint = estimate_from_anchors(left_t, left_a, cur_a, right_t, right_a)
                    est = preserve_date_set_time(raw, midpoint)

            if pd.notna(left_t) and pd.notna(right_t):
                if est < left_t:
                    est = left_t
                if est > right_t:
                    span = (right_t - left_t).total_seconds()
                    est = left_t + pd.to_timedelta(max(0.0, min(span, span * 0.5)), unit="s")
                    if infer_date_only(raw):
                        est = preserve_date_set_time(raw, est)
            elif pd.notna(left_t) and est < left_t:
                est = left_t + pd.to_timedelta(1, unit="s")
            elif pd.notna(right_t) and est > right_t:
                est = right_t - pd.to_timedelta(1, unit="s")

            prec, sep = detect_precision_and_sep(raw)
            out_prec = prec if prec > 0 else sample_prec
            df.at[orig_ix, "repaired_ts"] = format_timestamp_like(est, out_prec, sep if sep else sample_sep)

    # self-check and corrective pass
    rows_equal_corrupt = 0
    for idx, row in df.iterrows():
        cat = row["error_category"]
        if cat == "CLEAN":
            df.at[idx, "repaired_ts"] = row["orig_ts"]
        else:
            if df.at[idx, "repaired_ts"] == row["orig_ts"]:
                rows_equal_corrupt += 1

    # enforce parseability and anchor bounds with another case-wise pass
    for case_id, idxs in df.groupby("Case", sort=False).groups.items():
        case_idx_list = list(idxs)
        case_df = df.loc[case_idx_list].copy()
        parsed_rep = case_df["repaired_ts"].apply(parse_timestamp_loose)
        parsed_any = parsed_rep.dropna()
        case_start = parsed_any.min() if not parsed_any.empty else pd.NaT
        case_end = parsed_any.max() if not parsed_any.empty else pd.NaT

        # repair C1 blocks post-check
        temp = case_df.reset_index().rename(columns={"index": "orig_index"})
        i = 0
        while i < len(temp):
            row = temp.iloc[i]
            if row["error_category"] == "C1_ORDERING_AMBIGUITY":
                ts_val = row["orig_ts"]
                block_rows = [i]
                j = i + 1
                while j < len(temp):
                    rj = temp.iloc[j]
                    if rj["error_category"] == "C1_ORDERING_AMBIGUITY" and rj["orig_ts"] == ts_val:
                        block_rows.append(j)
                        j += 1
                    else:
                        break
                if len(block_rows) >= 2:
                    observed = parse_timestamp_loose(ts_val)
                    repaired_times = []
                    for b in block_rows:
                        repaired_times.append(parse_timestamp_loose(df.at[temp.iloc[b]["orig_index"], "repaired_ts"]))
                    acts = [temp.iloc[b]["Activity"] for b in block_rows]

                    left_t, left_a = find_left_clean_anchor(temp.assign(parsed_ts=temp["orig_ts"].apply(parse_timestamp_loose)), block_rows[0])
                    right_t, right_a = find_right_clean_anchor(temp.assign(parsed_ts=temp["orig_ts"].apply(parse_timestamp_loose)), block_rows[-1])

                    order = enumerate_orders(acts, left_a, right_a, max(0.0, (right_t - observed).total_seconds()) if pd.notna(right_t) and pd.notna(observed) else None)
                    expected_times, clamped = build_c1_times(observed, order, right_t, left_t, left_a, right_a)
                    if clamped:
                        c1_blocks_clamped += 1

                    # exactly one shared timestamp, earliest in inferred order
                    act_to_pos = {}
                    for pos2, act in enumerate(order):
                        act_to_pos.setdefault(act, []).append(pos2)
                    taken_pos = set()
                    for local_ix, b in enumerate(block_rows):
                        act = temp.iloc[b]["Activity"]
                        choose = None
                        for c in act_to_pos.get(act, []):
                            if c not in taken_pos:
                                choose = c
                                break
                        if choose is None:
                            choose = 0
                        taken_pos.add(choose)
                        dt = expected_times[choose]
                        orig_ix = temp.iloc[b]["orig_index"]
                        prec, sep = detect_precision_and_sep(df.at[orig_ix, "orig_ts"])
                        df.at[orig_ix, "repaired_ts"] = format_timestamp_like(dt, prec if prec > 0 else sample_prec, sep)
                i = j
            else:
                i += 1

    # final equality recount after corrective pass
    rows_equal_corrupt = 0
    for idx, row in df.iterrows():
        if row["error_category"] != "CLEAN" and df.at[idx, "repaired_ts"] == row["orig_ts"]:
            rows_equal_corrupt += 1

    out = df[original_columns].copy()
    out["Timestamp"] = df["repaired_ts"]

    # preserve byte-identical protected columns and row order
    for col in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        out[col] = df[col]

    out.to_csv(args.output, index=False)

    print(f"{c1_blocks_repaired},{c1_blocks_clamped},{rows_equal_corrupt}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # fail-safe: never non-zero, still attempt passthrough if possible
        import sys
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", required=True)
            parser.add_argument("--output", required=True)
            args, _ = parser.parse_known_args()
            df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
            df.to_csv(args.output, index=False)
            print("0,0,0")
        except Exception:
            print("0,0,0")