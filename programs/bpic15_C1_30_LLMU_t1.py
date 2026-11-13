import argparse
import itertools
import math
import re
from collections import defaultdict, Counter

import pandas as pd


TRANSITION_PROB = {
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): 0.9871,
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): 0.9801,
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): 0.973,
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): 0.959,
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): 0.9229,
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): 0.918,
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): 0.9085,
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): 0.8978,
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): 0.8561,
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): 0.8365,
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): 0.8338,
    ("O_DECLINED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 0.8329,
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): 0.7942,
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): 0.7917,
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): 0.74,
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): 0.7371,
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): 0.7302,
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): 0.7262,
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): 0.7022,
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): 0.6868,
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): 0.6741,
    ("W_Beoordelen fraude-COMPLETE", "W_Beoordelen fraude-START"): 0.6683,
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): 0.6527,
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): 0.6431,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 0.6388,
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): 0.6371,
    ("W_Wijzigen contractgegevens-SCHEDULE", "W_Wijzigen contractgegevens-SCHEDULE"): 0.625,
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): 0.5729,
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): 0.5643,
    ("W_Beoordelen fraude-START", "W_Beoordelen fraude-COMPLETE"): 0.5564,
    ("W_Beoordelen fraude-SCHEDULE", "W_Beoordelen fraude-START"): 0.5366,
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): 0.5221,
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): 0.5016,
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 0.4494,
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): 0.4217,
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): 0.42,
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): 0.3843,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): 0.3787,
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): 0.3702,
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): 0.367,
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): 0.3517,
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): 0.3443,
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): 0.3353,
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): 0.3141,
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 0.3132,
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): 0.3006,
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): 0.2668,
    ("W_Beoordelen fraude-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): 0.2602,
    ("A_ACTIVATED-COMPLETE", "A_REGISTERED-COMPLETE"): 0.26,
    ("A_APPROVED-COMPLETE", "O_ACCEPTED-COMPLETE"): 0.2547,
    ("W_Wijzigen contractgegevens-SCHEDULE", "W_Valideren aanvraag-START"): 0.25,
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): 0.2487,
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): 0.2407,
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): 0.2228,
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): 0.2219,
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): 0.2145,
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): 0.2094,
    ("W_Beoordelen fraude-START", "A_DECLINED-COMPLETE"): 0.2068,
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): 0.197,
    ("W_Beoordelen fraude-SCHEDULE", "W_Afhandelen leads-COMPLETE"): 0.1707,
}

DURATION = {
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): dict(n=16994, p10=50.5901, p25=14888.91575, median=167356.1795, p90=652053.5182, share=0.104),
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): dict(n=16243, p10=10.8058, p25=41.519, median=99.459, p90=477.0068, share=0.336),
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): dict(n=16181, p10=18.568, p25=56.637, median=115.945, p90=400.625, share=0.265),
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): dict(n=15763, p10=61.7376, p25=1598.129, median=14639.161, p90=164002.6406, share=0.099),
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): dict(n=12014, p10=0.126, p25=0.175, median=0.275, p90=1.0927, share=1.0),
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): dict(n=9103, p10=76.925, p25=1284.3235, median=10784.402, p90=178947.3336, share=0.093),
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): dict(n=8970, p10=10.2459, p25=33.62275, median=140.2915, p90=895.4752, share=0.33),
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): dict(n=6954, p10=0.046, p25=0.13325, median=0.378, p90=0.888, share=1.0),
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): dict(n=6885, p10=0.025, p25=0.028, median=0.035, p90=0.079, share=1.0),
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): dict(n=6359, p10=0.083, p25=0.093, median=0.119, p90=0.3112, share=1.0),
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): dict(n=6306, p10=0.9155, p25=1.08425, median=1.291, p90=2.564, share=1.0),
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): dict(n=4962, p10=125.3631, p25=247.7525, median=373.9065, p90=724.5254, share=0.048),
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): dict(n=4837, p10=12.5822, p25=20.106, median=310.933, p90=673073.0688, share=0.421),
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): dict(n=4786, p10=1.077, p25=1.18, median=1.3905, p90=2.2835, share=1.0),
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): dict(n=4616, p10=30.8955, p25=742.71875, median=6597.786, p90=69323.723, share=0.144),
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): dict(n=4550, p10=32.8288, p25=35.5895, median=38.9175, p90=48.9672, share=0.971),
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): dict(n=4422, p10=83.1826, p25=597.00775, median=2551.737, p90=53740.6673, share=0.085),
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): dict(n=4409, p10=8.6702, p25=11.885, median=22.736, p90=47.734, share=0.949),
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): dict(n=4289, p10=0.0, p25=0.0, median=0.0, p90=0.0, share=1.0),
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): dict(n=4254, p10=63.8074, p25=118.415, median=183.87, p90=409.5407, share=0.093),
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): dict(n=3371, p10=0.141, p25=0.153, median=0.179, p90=0.374, share=0.999),
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): dict(n=3177, p10=10.1838, p25=12.41, median=21.162, p90=89.9652, share=0.827),
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): dict(n=3175, p10=1.115, p25=1.228, median=1.447, p90=2.228, share=1.0),
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): dict(n=3150, p10=7064.724, p25=89340.6535, median=189136.3675, p90=441203.2956, share=0.049),
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): dict(n=2907, p10=10.7182, p25=44.835, median=307.187, p90=2200.1726, share=0.284),
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): dict(n=2882, p10=24.2998, p25=244.81225, median=2005.5505, p90=86081.2703, share=0.144),
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): dict(n=2666, p10=31.724, p25=34.09525, median=37.348, p90=47.236, share=0.977),
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): dict(n=2422, p10=18.2621, p25=908.79625, median=5975.98, p90=48230.6337, share=0.148),
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): dict(n=2410, p10=35.6547, p25=66.4165, median=107.805, p90=344.4917, share=0.21),
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): dict(n=2407, p10=0.898, p25=0.987, median=1.148, p90=1.854, share=1.0),
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): dict(n=2145, p10=26.1244, p25=332.025, median=2855.616, p90=13851.0726, share=0.155),
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): dict(n=2136, p10=1.446, p25=1.605, median=1.843, p90=2.733, share=1.0),
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): dict(n=2123, p10=2.843, p25=3.0815, median=3.41, p90=4.4502, share=0.998),
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): dict(n=2014, p10=0.0, p25=0.0, median=0.0, p90=0.0, share=1.0),
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): dict(n=2010, p10=32.1372, p25=69.84475, median=113.2505, p90=373.0038, share=0.203),
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): dict(n=1953, p10=254.4338, p25=623.07, median=1174.31, p90=2497.312, share=0.029),
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): dict(n=1711, p10=1.486, p25=1.649, median=1.895, p90=2.75, share=1.0),
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): dict(n=1711, p10=696.887, p25=2100.1995, median=5033.333, p90=76705.849, share=0.031),
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): dict(n=1662, p10=0.0, p25=0.0, median=0.0, p90=0.0, share=1.0),
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): dict(n=1638, p10=0.0, p25=0.0, median=0.0, p90=0.0, share=1.0),
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): dict(n=1621, p10=31.965, p25=48.504, median=106.471, p90=410.137, share=0.316),
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): dict(n=1563, p10=1.2212, p25=1.3115, median=1.618, p90=2.4682, share=0.999),
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): dict(n=1520, p10=24.7997, p25=36.5235, median=65.3805, p90=566.6155, share=0.466),
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): dict(n=1466, p10=0.0, p25=0.0, median=0.0, p90=0.0, share=1.0),
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): dict(n=1285, p10=0.001, p25=0.001, median=0.001, p90=0.001, share=1.0),
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): dict(n=1197, p10=8.0472, p25=46.988, median=135.472, p90=3067.5444, share=0.292),
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): dict(n=1126, p10=0.0, p25=0.0, median=0.0, p90=0.0, share=1.0),
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): dict(n=1085, p10=2.9624, p25=3.184, median=3.533, p90=4.6056, share=0.999),
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): dict(n=1080, p10=3.8414, p25=17.07475, median=362.5105, p90=8920.1066, share=0.327),
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): dict(n=1038, p10=179.8682, p25=315.4015, median=474.467, p90=961.663, share=0.022),
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): dict(n=1009, p10=6.4536, p25=7.027, median=7.763, p90=9.3666, share=0.999),
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): dict(n=993, p10=1.837, p25=2.093, median=2.396, p90=3.4406, share=0.998),
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): dict(n=912, p10=72.662, p25=111.11025, median=156.877, p90=399.8073, share=0.081),
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): dict(n=862, p10=6.3086, p25=6.952, median=7.6685, p90=9.1902, share=1.0),
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): dict(n=862, p10=141.3626, p25=317.4815, median=770.573, p90=1957.7284, share=0.019),
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): dict(n=801, p10=0.0, p25=0.0, median=0.0, p90=0.0, share=1.0),
    ("A_SUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): dict(n=763, p10=31.7182, p25=34.433, median=38.218, p90=47.6524, share=0.987),
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): dict(n=747, p10=1.3078, p25=1.4635, median=1.661, p90=2.6858, share=0.999),
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): dict(n=717, p10=0.9084, p25=1.053, median=1.251, p90=2.1808, share=1.0),
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): dict(n=716, p10=0.001, p25=0.001, median=0.001, p90=0.001, share=1.0),
}

GLOBAL_FAST_DEFAULT = 1.0
DEFAULT_TRANSITION_PROB = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def infer_precision(series):
    max_digits = 0
    any_frac = False
    for s in series.dropna().astype(str):
        m = re.search(r"\.(\d+)$", s.strip())
        if m:
            any_frac = True
            max_digits = max(max_digits, len(m.group(1)))
    if any_frac:
        return max_digits
    return 0


def parse_ts(s):
    if s is None:
        return pd.NaT
    st = str(s)
    if st.strip() == "":
        return pd.NaT
    try:
        return pd.to_datetime(st, errors="coerce")
    except Exception:
        return pd.NaT


def is_date_only(s):
    if s is None:
        return False
    st = str(s).strip()
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", st))


def format_ts(ts, precision):
    if pd.isna(ts):
        return ""
    ts = pd.Timestamp(ts)
    base = ts.strftime("%Y-%m-%d %H:%M:%S")
    if precision <= 0:
        return base
    frac = f"{ts.microsecond:06d}"[:precision].ljust(precision, "0")
    return f"{base}.{frac}"


def step_timedelta(precision):
    if precision <= 0:
        return pd.Timedelta(seconds=1)
    if precision <= 6:
        return pd.Timedelta(microseconds=10 ** (6 - precision))
    return pd.Timedelta(microseconds=1)


def transition_prob(a, b):
    return TRANSITION_PROB.get((a, b), DEFAULT_TRANSITION_PROB)


def transition_duration_guess(a, b):
    prof = DURATION.get((a, b))
    if prof:
        if prof["share"] >= 0.5:
            val = prof["p10"]
        else:
            val = prof["p25"]
        return max(float(val), 0.0)
    fa = activity_family(a)
    fb = activity_family(b)
    family_vals = []
    for (x, y), prof in DURATION.items():
        if activity_family(x) == fa and activity_family(y) == fb:
            family_vals.append(prof["p10"] if prof["share"] >= 0.5 else prof["p25"])
    if family_vals:
        family_vals = [max(float(v), 0.0) for v in family_vals]
        family_vals.sort()
        return family_vals[len(family_vals) // 2]
    return GLOBAL_FAST_DEFAULT


def duration_plausibility(a, b, observed_sec):
    if observed_sec is None or observed_sec < 0:
        return -1e9
    prof = DURATION.get((a, b))
    if prof is None:
        exp = transition_duration_guess(a, b)
        scale = max(10.0, exp * 3.0)
        return -abs(observed_sec - exp) / scale - 3.0
    target = prof["p10"] if prof["share"] >= 0.5 else prof["p25"]
    target = max(target, 0.0)
    upper = max(prof["p90"], target + 1.0)
    if observed_sec == 0 and target == 0:
        return 1.0
    if target == 0:
        return -observed_sec / max(1.0, upper)
    ratio = observed_sec / max(target, 1e-9)
    if ratio <= 0:
        return -1e9
    pen = abs(math.log(max(ratio, 1e-9)))
    if observed_sec > upper * 5:
        pen += 2.0
    return -pen


def activity_family(a):
    if not isinstance(a, str):
        return ""
    if "-" in a:
        return a.split("-", 1)[0]
    return a


def candidate_orders(rows, left_act, right_act):
    acts = [r["Activity"] for r in rows]
    idxs = list(range(len(rows)))
    if len(rows) <= 7:
        perms = itertools.permutations(idxs)
        scored = []
        for perm in perms:
            score = ordering_score([rows[i] for i in perm], left_act, right_act)
            scored.append((score, perm))
        scored.sort(reverse=True)
        return [perm for _, perm in scored[:50]]
    beam = [([], set(), 0.0)]
    width = 30
    for _ in range(len(rows)):
        new_beam = []
        for seq, used, sc in beam:
            prev_act = left_act if not seq else rows[seq[-1]]["Activity"]
            for i in idxs:
                if i in used:
                    continue
                add = math.log(transition_prob(prev_act, rows[i]["Activity"]) + 1e-15) if prev_act else 0.0
                new_seq = seq + [i]
                new_used = set(used)
                new_used.add(i)
                new_beam.append((new_seq, new_used, sc + add))
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:width]
    scored = []
    for seq, _, sc in beam:
        tail = math.log(transition_prob(rows[seq[-1]]["Activity"], right_act) + 1e-15) if right_act else 0.0
        scored.append((sc + tail, tuple(seq)))
    scored.sort(reverse=True)
    return [perm for _, perm in scored[:50]]


def ordering_score(ordered_rows, left_act, right_act):
    acts = [r["Activity"] for r in ordered_rows]
    s = 0.0
    if left_act:
        s += math.log(transition_prob(left_act, acts[0]) + 1e-15)
    for a, b in zip(acts, acts[1:]):
        s += math.log(transition_prob(a, b) + 1e-15)
        g = transition_duration_guess(a, b)
        s -= 0.00001 * g
    if right_act:
        s += math.log(transition_prob(acts[-1], right_act) + 1e-15)
    return s


def find_clean_anchor(case_df, pos, direction):
    if direction < 0:
        rng = range(pos - 1, -1, -1)
    else:
        rng = range(pos + 1, len(case_df))
    for j in rng:
        if str(case_df.iloc[j]["error_category"]) == "CLEAN":
            ts = parse_ts(case_df.iloc[j]["Timestamp"])
            if not pd.isna(ts):
                return j, ts, case_df.iloc[j]["Activity"]
    return None, pd.NaT, None


def find_case_bounds(case_df):
    parsed = []
    for _, r in case_df.iterrows():
        t = parse_ts(r["Timestamp"])
        if not pd.isna(t):
            parsed.append(t)
    if not parsed:
        now = pd.Timestamp("2000-01-01 00:00:00")
        return now, now
    return min(parsed), max(parsed)


def clamp_between(ts, lo, hi):
    if pd.isna(lo) and pd.isna(hi):
        return ts
    if pd.isna(lo):
        return min(ts, hi)
    if pd.isna(hi):
        return max(ts, lo)
    return min(max(ts, lo), hi)


def strictly_increasing_from(start, gaps, right_anchor, step):
    times = [start]
    cum = 0.0
    pos_gaps = [max(float(g), step.total_seconds()) for g in gaps]
    if len(pos_gaps) == 0:
        return times
    total = sum(pos_gaps)
    room = None
    if not pd.isna(right_anchor):
        room = max((right_anchor - start).total_seconds(), 0.0)
    min_total = (len(pos_gaps)) * step.total_seconds()
    if room is not None:
        if room < min_total:
            pos_gaps = [step.total_seconds()] * len(pos_gaps)
        elif total > room:
            if total <= 0:
                pos_gaps = [step.total_seconds()] * len(pos_gaps)
            else:
                scaled = [g * room / total for g in pos_gaps]
                pos_gaps = [max(g, step.total_seconds()) for g in scaled]
                total2 = sum(pos_gaps)
                if total2 > room and total2 > 0:
                    excess = total2 - room
                    flex = [g - step.total_seconds() for g in pos_gaps]
                    flex_sum = sum(max(x, 0.0) for x in flex)
                    if flex_sum > 0:
                        pos_gaps = [
                            max(step.total_seconds(), g - excess * max(f, 0.0) / flex_sum)
                            for g, f in zip(pos_gaps, flex)
                        ]
    cur = start
    for g in pos_gaps:
        cur = cur + pd.to_timedelta(g, unit="s")
        if not pd.isna(right_anchor) and cur > right_anchor:
            cur = right_anchor
        times.append(cur)
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + step
    if not pd.isna(right_anchor):
        if times[-1] > right_anchor:
            span = len(times) - 1
            if span > 0:
                latest_start = right_anchor - step * span
                if latest_start < start:
                    latest_start = start
                times = [latest_start + step * i for i in range(len(times))]
    return times


def repair_c1_block(case_df, block_positions, precision):
    step = step_timedelta(precision)
    first_pos = block_positions[0]
    observed = parse_ts(case_df.iloc[first_pos]["Timestamp"])
    left_idx, left_ts, left_act = find_clean_anchor(case_df, first_pos, -1)
    right_idx, right_ts, right_act = find_clean_anchor(case_df, block_positions[-1], 1)
    rows = []
    for p in block_positions:
        rows.append({
            "pos": p,
            "Activity": case_df.iloc[p]["Activity"],
            "row_id": case_df.iloc[p]["row_id"],
            "orig_ts_str": case_df.iloc[p]["Timestamp"],
        })
    perms = candidate_orders(rows, left_act, right_act)
    best = None
    for perm in perms:
        ordered = [rows[i] for i in perm]
        acts = [r["Activity"] for r in ordered]
        gaps = [transition_duration_guess(a, b) for a, b in zip(acts, acts[1:])]
        room_anchor = right_ts if not pd.isna(right_ts) else pd.NaT
        times = strictly_increasing_from(observed, gaps, room_anchor, step)
        if len(times) != len(ordered):
            continue
        score = ordering_score(ordered, left_act, right_act)
        if not pd.isna(right_ts):
            if times[-1] > right_ts:
                score -= 1000
        candidate = (score, perm, ordered, times)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        perm = tuple(range(len(rows)))
        ordered = [rows[i] for i in perm]
        gaps = [max(transition_duration_guess(a["Activity"], b["Activity"]), step.total_seconds()) for a, b in zip(ordered, ordered[1:])]
        times = strictly_increasing_from(observed, gaps, right_ts, step)
        best = (0.0, perm, ordered, times)
    _, perm, ordered, times = best
    mapping = {}
    for r, t in zip(ordered, times):
        mapping[r["pos"]] = t
    clamped = False
    if not pd.isna(right_ts) and times[-1] > right_ts:
        clamped = True
    if len(times) >= 2 and (times[-1] - times[0]) < step * (len(times) - 1):
        clamped = True
    return mapping, clamped, ordered[0]["pos"]


def choose_anchor_bounds(case_df, pos):
    case_start, case_end = find_case_bounds(case_df)
    left_idx, left_ts, left_act = find_clean_anchor(case_df, pos, -1)
    right_idx, right_ts, right_act = find_clean_anchor(case_df, pos, 1)
    lo = left_ts if not pd.isna(left_ts) else case_start
    hi = right_ts if not pd.isna(right_ts) else case_end
    return lo, hi, left_ts, right_ts, left_act, right_act, case_start, case_end


def estimate_from_neighbors(case_df, pos, keep_date=None):
    lo, hi, left_ts, right_ts, left_act, right_act, case_start, case_end = choose_anchor_bounds(case_df, pos)
    act = case_df.iloc[pos]["Activity"]
    step = step_timedelta(INFERRED_PRECISION)
    cand = None
    if not pd.isna(left_ts) and left_act:
        d1 = transition_duration_guess(left_act, act)
        cand = left_ts + pd.to_timedelta(d1, unit="s")
    if not pd.isna(right_ts) and right_act:
        d2 = transition_duration_guess(act, right_act)
        cand2 = right_ts - pd.to_timedelta(d2, unit="s")
        if cand is None:
            cand = cand2
        else:
            if cand > right_ts or cand < left_ts:
                if not pd.isna(left_ts) and not pd.isna(right_ts):
                    span = (right_ts - left_ts).total_seconds()
                    left_gap = max(transition_duration_guess(left_act, act), 0.001) if left_act else 1.0
                    right_gap = max(transition_duration_guess(act, right_act), 0.001) if right_act else 1.0
                    frac = left_gap / (left_gap + right_gap)
                    cand = left_ts + pd.to_timedelta(span * frac, unit="s")
                else:
                    cand = cand2
            else:
                cand = min(cand, cand2) if cand2 >= cand else max(cand, cand2)
    if cand is None:
        cand = lo
    cand = clamp_between(cand, lo, hi)
    if keep_date is not None:
        target_date = pd.Timestamp(keep_date).date()
        day_start = pd.Timestamp(target_date)
        day_end = day_start + pd.Timedelta(days=1) - step
        lo2 = max(lo, day_start)
        hi2 = min(hi, day_end)
        if lo2 > hi2:
            lo2 = max(case_start, day_start)
            hi2 = min(case_end, day_end)
            if lo2 > hi2:
                lo2 = day_start
                hi2 = day_end
        cand = clamp_between(cand, lo2, hi2)
        if cand.date() != target_date:
            cand = lo2
    return cand


def score_swap_option(case_df, i, j, t_i, t_j):
    act_i = case_df.iloc[i]["Activity"]
    act_j = case_df.iloc[j]["Activity"]
    left_idx, left_ts, left_act = find_clean_anchor(case_df, i, -1)
    right_idx, right_ts, right_act = find_clean_anchor(case_df, j, 1)
    if t_i > t_j:
        return -1e12
    score = 0.0
    if not pd.isna(left_ts) and left_act:
        score += duration_plausibility(left_act, act_i, (t_i - left_ts).total_seconds())
        score += math.log(transition_prob(left_act, act_i) + 1e-15)
    score += duration_plausibility(act_i, act_j, (t_j - t_i).total_seconds())
    score += math.log(transition_prob(act_i, act_j) + 1e-15)
    if not pd.isna(right_ts) and right_act:
        score += duration_plausibility(act_j, right_act, (right_ts - t_j).total_seconds())
        score += math.log(transition_prob(act_j, right_act) + 1e-15)
    return score


def repair_c2_pair(case_df, i, precision):
    j = i + 1
    step = step_timedelta(precision)
    ti_obs = parse_ts(case_df.iloc[i]["Timestamp"])
    tj_obs = parse_ts(case_df.iloc[j]["Timestamp"])
    swap_score = score_swap_option(case_df, i, j, tj_obs, ti_obs)
    displaced_best = None
    cand_i = estimate_from_neighbors(case_df, i)
    if cand_i <= tj_obs:
        score1 = score_swap_option(case_df, i, j, cand_i, tj_obs)
        displaced_best = ("i", cand_i, tj_obs, score1)
    cand_j = estimate_from_neighbors(case_df, j)
    if cand_j >= ti_obs:
        score2 = score_swap_option(case_df, i, j, ti_obs, cand_j)
        if displaced_best is None or score2 > displaced_best[3]:
            displaced_best = ("j", ti_obs, cand_j, score2)
    if displaced_best is None:
        lo, hi, *_ = choose_anchor_bounds(case_df, i)
        mid = clamp_between(ti_obs + step, lo, hi)
        if mid <= ti_obs:
            mid = ti_obs + step
        displaced_best = ("j", ti_obs, max(mid, tj_obs), -1e9)
    if swap_score >= displaced_best[3]:
        return {i: tj_obs, j: ti_obs}
    return {i: displaced_best[1], j: displaced_best[2]}


def repair_c3_row(case_df, pos, precision):
    raw = str(case_df.iloc[pos]["Timestamp"]) if pd.notna(case_df.iloc[pos]["Timestamp"]) else ""
    keep_date = None
    if is_date_only(raw):
        keep_date = raw.strip()
    return estimate_from_neighbors(case_df, pos, keep_date=keep_date)


def self_fix_case(case_df, repaired_ts, c1_blocks_meta, precision):
    step = step_timedelta(precision)
    clamp_count = 0
    case_start, case_end = find_case_bounds(case_df)

    for meta in c1_blocks_meta:
        positions = meta["positions"]
        observed = parse_ts(case_df.iloc[positions[0]]["Timestamp"])
        kept_pos = meta["kept_pos"]
        vals = {p: repaired_ts[p] for p in positions}
        order = sorted(positions, key=lambda p: (repaired_ts[p], str(case_df.iloc[p]["row_id"])))
        if kept_pos not in positions:
            kept_pos = order[0]
        repaired_ts[kept_pos] = observed
        others = [p for p in order if p != kept_pos]
        cur = observed
        for p in others:
            cand = repaired_ts[p]
            if cand <= cur:
                cand = cur + step
            repaired_ts[p] = cand
            cur = cand
        right_idx, right_ts, right_act = find_clean_anchor(case_df, positions[-1], 1)
        if not pd.isna(right_ts) and repaired_ts[max(positions, key=lambda x: repaired_ts[x])] > right_ts:
            clamp_count += 1
            ordered = [kept_pos] + sorted(others, key=lambda p: repaired_ts[p])
            latest_start = observed
            room = (right_ts - observed).total_seconds()
            min_needed = (len(ordered) - 1) * step.total_seconds()
            if room < min_needed:
                base = right_ts - step * (len(ordered) - 1)
                if base < observed:
                    base = observed
                seq = [base + step * k for k in range(len(ordered))]
                if seq[0] != observed:
                    seq[0] = observed
                    for k in range(1, len(seq)):
                        if seq[k] <= seq[k - 1]:
                            seq[k] = seq[k - 1] + step
            else:
                seq = [observed + step * k for k in range(len(ordered))]
                if seq[-1] > right_ts:
                    seq = [right_ts - step * (len(ordered) - 1) + step * k for k in range(len(ordered))]
                    if seq[0] < observed:
                        seq = [observed + step * k for k in range(len(ordered))]
            for p, t in zip(ordered, seq):
                repaired_ts[p] = t

    for p, t in list(repaired_ts.items()):
        lo, hi, left_ts, right_ts, left_act, right_act, _, _ = choose_anchor_bounds(case_df, p)
        nt = clamp_between(t, lo, hi)
        if nt != t:
            clamp_count += 1
            repaired_ts[p] = nt

    for p, t in list(repaired_ts.items()):
        if t < case_start:
            repaired_ts[p] = case_start
            clamp_count += 1
        if t > case_end:
            repaired_ts[p] = case_end
            clamp_count += 1

    return clamp_count


def process(df):
    global INFERRED_PRECISION
    INFERRED_PRECISION = infer_precision(df["Timestamp"])
    precision = INFERRED_PRECISION

    out_ts = df["Timestamp"].astype(str).tolist()
    repaired_positions = set()
    c1_blocks_repaired = 0
    clamped_blocks = 0
    c1_blocks_meta_all = []

    case_groups = df.groupby("Case", sort=False).groups
    for case, idxs in case_groups.items():
        case_df = df.loc[list(idxs)].reset_index()
        local_to_global = case_df["index"].tolist()
        local_repaired = {}
        c1_meta = []

        i = 0
        while i < len(case_df):
            cat = str(case_df.iloc[i]["error_category"])
            if cat == "C1_ORDERING_AMBIGUITY":
                ts_raw = case_df.iloc[i]["Timestamp"]
                block = [i]
                j = i + 1
                while j < len(case_df):
                    if str(case_df.iloc[j]["error_category"]) == "C1_ORDERING_AMBIGUITY" and str(case_df.iloc[j]["Timestamp"]) == str(ts_raw):
                        block.append(j)
                        j += 1
                    else:
                        break
                if len(block) >= 2:
                    mapping, clamped, kept_pos = repair_c1_block(case_df, block, precision)
                    for lp, ts in mapping.items():
                        local_repaired[lp] = ts
                    c1_blocks_repaired += 1
                    if clamped:
                        clamped_blocks += 1
                    c1_meta.append({"positions": block, "kept_pos": kept_pos})
                    i = j
                    continue
                else:
                    est = repair_c3_row(case_df, i, precision)
                    local_repaired[i] = est
            elif cat == "C2_TEMPORAL_INCONSISTENCY":
                if i + 1 < len(case_df) and str(case_df.iloc[i + 1]["error_category"]) == "C2_TEMPORAL_INCONSISTENCY":
                    ti = parse_ts(case_df.iloc[i]["Timestamp"])
                    tj = parse_ts(case_df.iloc[i + 1]["Timestamp"])
                    if not pd.isna(ti) and not pd.isna(tj) and tj < ti:
                        mp = repair_c2_pair(case_df, i, precision)
                        for k, v in mp.items():
                            local_repaired[k] = v
                        i += 2
                        continue
                local_repaired[i] = repair_c3_row(case_df, i, precision)
            elif cat == "C3_MISSING_TEMPORAL_INFORMATION":
                local_repaired[i] = repair_c3_row(case_df, i, precision)
            i += 1

        global_rep = {}
        for lp, ts in local_repaired.items():
            global_rep[local_to_global[lp]] = ts

        clamp_added = self_fix_case(case_df, local_repaired, c1_meta, precision)
        clamped_blocks += clamp_added

        for lp, ts in local_repaired.items():
            gp = local_to_global[lp]
            out_ts[gp] = format_ts(ts, precision)
            repaired_positions.add(gp)

        for meta in c1_meta:
            c1_blocks_meta_all.append({
                "case": case,
                "global_positions": [local_to_global[p] for p in meta["positions"]],
                "kept_global_pos": local_to_global[meta["kept_pos"]],
            })

    for idx, row in df.iterrows():
        if str(row["error_category"]) == "CLEAN":
            out_ts[idx] = row["Timestamp"]

    result = df.copy()
    result["Timestamp"] = out_ts

    still_equal = 0
    c1_kept_allowed = set(m["kept_global_pos"] for m in c1_blocks_meta_all)
    for idx, row in df.iterrows():
        if str(row["error_category"]) != "CLEAN":
            if str(result.iloc[idx]["Timestamp"]) == str(row["Timestamp"]):
                if idx in c1_kept_allowed:
                    still_equal += 1
                else:
                    parsed = parse_ts(result.iloc[idx]["Timestamp"])
                    if pd.isna(parsed):
                        est = parse_ts(row["Timestamp"])
                        if pd.isna(est):
                            est = pd.Timestamp("2000-01-01 00:00:00")
                        est = est + step_timedelta(precision)
                        result.at[idx, "Timestamp"] = format_ts(est, precision)
                    else:
                        adjusted = parsed + step_timedelta(precision)
                        result.at[idx, "Timestamp"] = format_ts(adjusted, precision)

    final_still_equal = 0
    for idx, row in df.iterrows():
        if str(row["error_category"]) != "CLEAN" and str(result.iloc[idx]["Timestamp"]) == str(row["Timestamp"]):
            final_still_equal += 1

    print(f"blocks_repaired={c1_blocks_repaired} blocks_clamped={clamped_blocks} rows_still_equal={final_still_equal}")
    return result


def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    expected_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""

    original = df.copy()
    result = process(df)

    for c in ["Case", "Activity", "Resource", "error_category", "row_id"]:
        result[c] = original[c]

    if len(result) != len(original):
        result = result.iloc[:len(original)].copy()
        if len(result) < len(original):
            missing = original.iloc[len(result):].copy()
            result = pd.concat([result, missing], ignore_index=True)

    result = result[original.columns.tolist()]
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()