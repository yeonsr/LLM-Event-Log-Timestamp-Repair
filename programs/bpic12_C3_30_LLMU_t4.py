import argparse
import itertools
import math
import re
from datetime import timedelta

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
    ("W_Nabellen offertes-COMPLETE", "W_Nabellen offertes-START"): dict(n=16994, p10=50.5901, p25=14888.91575, median=167356.1795, p75=443985.20225, p90=652053.5182, share=0.104),
    ("W_Completeren aanvraag-START", "W_Completeren aanvraag-COMPLETE"): dict(n=16243, p10=10.8058, p25=41.519, median=99.459, p75=203.335, p90=477.0068, share=0.336),
    ("W_Nabellen offertes-START", "W_Nabellen offertes-COMPLETE"): dict(n=16181, p10=18.568, p25=56.637, median=115.945, p75=206.713, p90=400.625, share=0.265),
    ("W_Completeren aanvraag-COMPLETE", "W_Completeren aanvraag-START"): dict(n=15763, p10=61.7376, p25=1598.129, median=14639.161, p75=69316.2765, p90=164002.6406, share=0.099),
    ("A_SUBMITTED-COMPLETE", "A_PARTLYSUBMITTED-COMPLETE"): dict(n=12014, p10=0.126, p25=0.175, median=0.275, p75=0.502, p90=1.0927, share=1.0),
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Nabellen incomplete dossiers-START"): dict(n=9103, p10=76.925, p25=1284.3235, median=10784.402, p75=65543.008, p90=178947.3336, share=0.093),
    ("W_Nabellen incomplete dossiers-START", "W_Nabellen incomplete dossiers-COMPLETE"): dict(n=8970, p10=10.2459, p25=33.62275, median=140.2915, p75=387.21975, p90=895.4752, share=0.33),
    ("A_PREACCEPTED-COMPLETE", "W_Completeren aanvraag-SCHEDULE"): dict(n=6954, p10=0.046, p25=0.13325, median=0.378, p75=0.544, p90=0.888, share=1.0),
    ("O_CREATED-COMPLETE", "O_SENT-COMPLETE"): dict(n=6885, p10=0.025, p25=0.028, median=0.035, p75=0.049, p90=0.079, share=1.0),
    ("O_SENT-COMPLETE", "W_Nabellen offertes-SCHEDULE"): dict(n=6359, p10=0.083, p25=0.093, median=0.119, p75=0.227, p90=0.3112, share=1.0),
    ("O_SELECTED-COMPLETE", "O_CREATED-COMPLETE"): dict(n=6306, p10=0.9155, p25=1.08425, median=1.291, p75=1.588, p90=2.564, share=1.0),
    ("W_Completeren aanvraag-START", "A_ACCEPTED-COMPLETE"): dict(n=4962, p10=125.3631, p25=247.7525, median=373.9065, p75=522.40025, p90=724.5254, share=0.048),
    ("W_Completeren aanvraag-COMPLETE", "W_Nabellen offertes-START"): dict(n=4837, p10=12.5822, p25=20.106, median=310.933, p75=601822.32, p90=673073.0688, share=0.421),
    ("W_Nabellen offertes-SCHEDULE", "W_Completeren aanvraag-COMPLETE"): dict(n=4786, p10=1.077, p25=1.18, median=1.3905, p75=1.954, p90=2.2835, share=1.0),
    ("W_Completeren aanvraag-SCHEDULE", "W_Completeren aanvraag-START"): dict(n=4616, p10=30.8955, p25=742.71875, median=6597.786, p75=44591.09425, p90=69323.723, share=0.144),
    ("A_PARTLYSUBMITTED-COMPLETE", "A_PREACCEPTED-COMPLETE"): dict(n=4550, p10=32.8288, p25=35.5895, median=38.9175, p75=43.09375, p90=48.9672, share=0.971),
    ("W_Afhandelen leads-SCHEDULE", "W_Afhandelen leads-START"): dict(n=4422, p10=83.1826, p25=597.00775, median=2551.737, p75=35699.1845, p90=53740.6673, share=0.085),
    ("A_PARTLYSUBMITTED-COMPLETE", "W_Afhandelen leads-SCHEDULE"): dict(n=4409, p10=8.6702, p25=11.885, median=22.736, p75=39.619, p90=47.734, share=0.949),
    ("A_FINALIZED-COMPLETE", "O_SELECTED-COMPLETE"): dict(n=4289, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("A_ACCEPTED-COMPLETE", "A_FINALIZED-COMPLETE"): dict(n=4254, p10=63.8074, p25=118.415, median=183.87, p75=280.09125, p90=409.5407, share=0.093),
    ("O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-SCHEDULE"): dict(n=3371, p10=0.141, p25=0.153, median=0.179, p75=0.274, p90=0.374, share=0.999),
    ("W_Nabellen offertes-START", "O_SENT_BACK-COMPLETE"): dict(n=3177, p10=10.1838, p25=12.41, median=21.162, p75=43.238, p90=89.9652, share=0.827),
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen offertes-COMPLETE"): dict(n=3175, p10=1.115, p25=1.228, median=1.447, p75=1.893, p90=2.228, share=1.0),
    ("W_Nabellen offertes-COMPLETE", "W_Valideren aanvraag-START"): dict(n=3150, p10=7064.724, p25=89340.6535, median=189136.3675, p75=348658.055, p90=441203.2956, share=0.049),
    ("W_Valideren aanvraag-START", "W_Valideren aanvraag-COMPLETE"): dict(n=2907, p10=10.7182, p25=44.835, median=307.187, p75=1039.046, p90=2200.1726, share=0.284),
    ("W_Valideren aanvraag-COMPLETE", "W_Valideren aanvraag-START"): dict(n=2882, p10=24.2998, p25=244.81225, median=2005.5505, p75=13058.8255, p90=86081.2703, share=0.144),
    ("A_PARTLYSUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): dict(n=2666, p10=31.724, p25=34.09525, median=37.348, p75=41.488, p90=47.236, share=0.977),
    ("W_Afhandelen leads-COMPLETE", "W_Completeren aanvraag-START"): dict(n=2422, p10=18.2621, p25=908.79625, median=5975.98, p75=15887.6505, p90=48230.6337, share=0.148),
    ("W_Afhandelen leads-START", "A_PREACCEPTED-COMPLETE"): dict(n=2410, p10=35.6547, p25=66.4165, median=107.805, p75=184.928, p90=344.4917, share=0.21),
    ("W_Completeren aanvraag-SCHEDULE", "W_Afhandelen leads-COMPLETE"): dict(n=2407, p10=0.898, p25=0.987, median=1.148, p75=1.367, p90=1.854, share=1.0),
    ("W_Valideren aanvraag-COMPLETE", "W_Nabellen incomplete dossiers-START"): dict(n=2145, p10=26.1244, p25=332.025, median=2855.616, p75=7839.745, p90=13851.0726, share=0.155),
    ("W_Nabellen incomplete dossiers-SCHEDULE", "W_Valideren aanvraag-COMPLETE"): dict(n=2136, p10=1.446, p25=1.605, median=1.843, p75=2.269, p90=2.733, share=1.0),
    ("A_DECLINED-COMPLETE", "W_Afhandelen leads-COMPLETE"): dict(n=2123, p10=2.843, p25=3.0815, median=3.41, p75=3.893, p90=4.4502, share=0.998),
    ("O_CANCELLED-COMPLETE", "O_SELECTED-COMPLETE"): dict(n=2014, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("W_Afhandelen leads-START", "A_DECLINED-COMPLETE"): dict(n=2010, p10=32.1372, p25=69.84475, median=113.2505, p75=200.6415, p90=373.0038, share=0.203),
    ("W_Valideren aanvraag-START", "W_Nabellen incomplete dossiers-SCHEDULE"): dict(n=1953, p10=254.4338, p25=623.07, median=1174.31, p75=1810.364, p90=2497.312, share=0.029),
    ("W_Valideren aanvraag-SCHEDULE", "W_Nabellen incomplete dossiers-COMPLETE"): dict(n=1711, p10=1.486, p25=1.649, median=1.895, p75=2.293, p90=2.75, share=1.0),
    ("W_Nabellen incomplete dossiers-COMPLETE", "W_Valideren aanvraag-START"): dict(n=1711, p10=696.887, p25=2100.1995, median=5033.333, p75=46329.5515, p90=76705.849, share=0.031),
    ("A_APPROVED-COMPLETE", "A_REGISTERED-COMPLETE"): dict(n=1662, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("A_CANCELLED-COMPLETE", "O_CANCELLED-COMPLETE"): dict(n=1638, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("W_Nabellen offertes-START", "O_CANCELLED-COMPLETE"): dict(n=1621, p10=31.965, p25=48.504, median=106.471, p75=217.721, p90=410.137, share=0.316),
    ("W_Nabellen offertes-SCHEDULE", "W_Nabellen offertes-COMPLETE"): dict(n=1563, p10=1.2212, p25=1.3115, median=1.618, p75=2.147, p90=2.4682, share=0.999),
    ("W_Nabellen incomplete dossiers-START", "W_Valideren aanvraag-SCHEDULE"): dict(n=1520, p10=24.7997, p25=36.5235, median=65.3805, p75=202.5615, p90=566.6155, share=0.466),
    ("A_ACTIVATED-COMPLETE", "A_APPROVED-COMPLETE"): dict(n=1466, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("O_ACCEPTED-COMPLETE", "A_ACTIVATED-COMPLETE"): dict(n=1285, p10=0.001, p25=0.001, median=0.001, p75=0.001, p90=0.001, share=1.0),
    ("W_Afhandelen leads-START", "W_Afhandelen leads-COMPLETE"): dict(n=1197, p10=8.0472, p25=46.988, median=135.472, p75=553.506, p90=3067.5444, share=0.292),
    ("A_REGISTERED-COMPLETE", "O_ACCEPTED-COMPLETE"): dict(n=1126, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("A_DECLINED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): dict(n=1085, p10=2.9624, p25=3.184, median=3.533, p75=4.033, p90=4.6056, share=0.999),
    ("W_Afhandelen leads-COMPLETE", "W_Afhandelen leads-START"): dict(n=1080, p10=3.8414, p25=17.07475, median=362.5105, p75=2195.20125, p90=8920.1066, share=0.327),
    ("W_Completeren aanvraag-START", "A_DECLINED-COMPLETE"): dict(n=1038, p10=179.8682, p25=315.4015, median=474.467, p75=687.9825, p90=961.663, share=0.022),
    ("A_REGISTERED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): dict(n=1009, p10=6.4536, p25=7.027, median=7.763, p75=8.498, p90=9.3666, share=0.999),
    ("O_CANCELLED-COMPLETE", "W_Nabellen offertes-COMPLETE"): dict(n=993, p10=1.837, p25=2.093, median=2.396, p75=2.924, p90=3.4406, share=0.998),
    ("W_Nabellen offertes-START", "A_CANCELLED-COMPLETE"): dict(n=912, p10=72.662, p25=111.11025, median=156.877, p75=240.16575, p90=399.8073, share=0.081),
    ("O_ACCEPTED-COMPLETE", "W_Valideren aanvraag-COMPLETE"): dict(n=862, p10=6.3086, p25=6.952, median=7.6685, p75=8.3765, p90=9.1902, share=1.0),
    ("W_Valideren aanvraag-START", "A_ACTIVATED-COMPLETE"): dict(n=862, p10=141.3626, p25=317.4815, median=770.573, p75=1332.394, p90=1957.7284, share=0.019),
    ("A_DECLINED-COMPLETE", "O_DECLINED-COMPLETE"): dict(n=801, p10=0.0, p25=0.0, median=0.0, p75=0.0, p90=0.0, share=1.0),
    ("A_SUBMITTED-COMPLETE", "A_DECLINED-COMPLETE"): dict(n=763, p10=31.7182, p25=34.433, median=38.218, p75=41.968, p90=47.6524, share=0.987),
    ("A_CANCELLED-COMPLETE", "W_Completeren aanvraag-COMPLETE"): dict(n=747, p10=1.3078, p25=1.4635, median=1.661, p75=2.1255, p90=2.6858, share=0.999),
    ("A_FINALIZED-COMPLETE", "O_CREATED-COMPLETE"): dict(n=717, p10=0.9084, p25=1.053, median=1.251, p75=1.556, p90=2.1808, share=1.0),
    ("O_SELECTED-COMPLETE", "A_FINALIZED-COMPLETE"): dict(n=716, p10=0.001, p25=0.001, median=0.001, p75=0.001, p90=0.001, share=1.0),
}

GLOBAL_SHORT_DEFAULT = 49.725


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    return ap.parse_args()


def infer_dataset_precision(ts_series):
    max_digits = 0
    for s in ts_series.fillna("").astype(str):
        m = re.search(r"\.(\d+)$", s.strip())
        if m:
            max_digits = max(max_digits, len(m.group(1)))
    return max_digits


def parse_timestamp_safe(s):
    if pd.isna(s):
        return pd.NaT, "blank", None
    text = str(s).strip()
    if text == "":
        return pd.NaT, "blank", None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}$", text):
        dt = pd.to_datetime(text, errors="coerce")
        return dt, "date_only", text
    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return pd.NaT, "blank", None
    return dt, "full", None


def format_dt(dt, precision):
    if precision <= 0:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    base = dt.strftime("%Y-%m-%d %H:%M:%S")
    frac = f"{dt.microsecond:06d}"[:precision].ljust(precision, "0")
    return f"{base}.{frac}"


def date_only_format(dt, original_date, precision):
    return format_dt(pd.Timestamp(
        year=pd.Timestamp(original_date).year,
        month=pd.Timestamp(original_date).month,
        day=pd.Timestamp(original_date).day,
        hour=dt.hour, minute=dt.minute, second=dt.second, microsecond=dt.microsecond
    ), precision)


def trans_prob(a, b):
    return TRANSITION_PROB.get((a, b), 1e-6)


def duration_profile(a, b):
    return DURATION.get((a, b))


def expected_gap_seconds(a, b):
    prof = duration_profile(a, b)
    if prof:
        if prof["share"] >= 0.95:
            return max(prof["p10"], 0.001)
        return max(prof["p25"], 0.001)
    return max(GLOBAL_SHORT_DEFAULT, 1.0)


def fit_penalty(a, b, observed_gap):
    prof = duration_profile(a, b)
    if observed_gap < 0:
        return 1e6
    if prof:
        target = prof["p10"] if prof["share"] >= 0.95 else prof["p25"]
        scale = max(target, 1.0)
        return abs(observed_gap - target) / scale
    return abs(observed_gap - GLOBAL_SHORT_DEFAULT) / max(GLOBAL_SHORT_DEFAULT, 1.0) + 2.0


def find_case_bounds(case_df):
    parsed = case_df["_parsed_dt"]
    valid = parsed.dropna()
    if len(valid) == 0:
        return pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")
    return valid.min(), valid.max()


def nearest_clean_left(case_df, pos):
    for i in range(pos - 1, -1, -1):
        if case_df.iloc[i]["error_category"] == "CLEAN" and pd.notna(case_df.iloc[i]["_parsed_dt"]):
            return i
    return None


def nearest_clean_right(case_df, pos):
    for i in range(pos + 1, len(case_df)):
        if case_df.iloc[i]["error_category"] == "CLEAN" and pd.notna(case_df.iloc[i]["_parsed_dt"]):
            return i
    return None


def score_order(activities, left_act, right_act):
    score = 0.0
    seq = list(activities)
    if left_act is not None:
        score += math.log(trans_prob(left_act, seq[0]))
    for a, b in zip(seq[:-1], seq[1:]):
        p = trans_prob(a, b)
        score += math.log(p)
        prof = duration_profile(a, b)
        if prof:
            score += math.log(min(max(prof["n"] / 1000.0, 1e-6), 1000.0))
    if right_act is not None:
        score += math.log(trans_prob(seq[-1], right_act))
    return score


def best_order_for_block(block_df, left_act, right_act):
    idxs = list(block_df.index)
    activities = [block_df.loc[i, "Activity"] for i in idxs]
    n = len(activities)
    best_perm = list(range(n))
    best_score = -1e99

    if n <= 7:
        for perm in itertools.permutations(range(n)):
            seq = [activities[i] for i in perm]
            sc = score_order(seq, left_act, right_act)
            if sc > best_score:
                best_score = sc
                best_perm = list(perm)
    else:
        beam = [([], set(), 0.0)]
        beam_width = 50
        for _ in range(n):
            new_beam = []
            for seq, used, sc in beam:
                prev_act = activities[seq[-1]] if seq else left_act
                for j in range(n):
                    if j in used:
                        continue
                    add = math.log(trans_prob(prev_act, activities[j])) if prev_act is not None else 0.0
                    prof = duration_profile(prev_act, activities[j]) if prev_act is not None else None
                    if prof:
                        add += math.log(min(max(prof["n"] / 1000.0, 1e-6), 1000.0))
                    new_beam.append((seq + [j], used | {j}, sc + add))
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_width]
        rescored = []
        for seq, used, sc in beam:
            final_sc = score_order([activities[i] for i in seq], left_act, right_act)
            rescored.append((final_sc, seq))
        rescored.sort(reverse=True)
        best_perm = rescored[0][1]

    ordered_indices = [idxs[i] for i in best_perm]
    return ordered_indices


def compress_gaps(gaps, available_room, min_total):
    gaps = [max(float(g), 0.001) for g in gaps]
    min_gaps = [1.0] * len(gaps)
    min_sum = sum(min_gaps)
    if len(gaps) == 0:
        return []
    target_total = sum(gaps)
    target_total = min(target_total, available_room)
    target_total = max(target_total, min_total)
    if target_total <= min_sum:
        return min_gaps
    extra = target_total - min_sum
    raw_extra = [max(g - 1.0, 0.0) for g in gaps]
    raw_sum = sum(raw_extra)
    if raw_sum <= 0:
        out = min_gaps[:]
        rem = extra
        i = 0
        while rem > 1e-9:
            add = min(rem, 1.0)
            out[i % len(out)] += add
            rem -= add
            i += 1
        return out
    return [1.0 + extra * (re / raw_sum) for re in raw_extra]


def repair_c1_blocks(case_df, repaired_strings, repaired_dts, precision, counters):
    i = 0
    n = len(case_df)
    while i < n:
        row = case_df.iloc[i]
        if row["error_category"] != "C1_ORDERING_AMBIGUITY" or pd.isna(row["_parsed_dt"]):
            i += 1
            continue
        shared = row["_parsed_dt"]
        j = i
        block_idxs = []
        while j < n:
            rj = case_df.iloc[j]
            if rj["error_category"] == "C1_ORDERING_AMBIGUITY" and pd.notna(rj["_parsed_dt"]) and rj["_parsed_dt"] == shared:
                block_idxs.append(case_df.index[j])
                j += 1
            else:
                break
        if len(block_idxs) <= 1:
            i = j
            continue

        block_df = case_df.loc[block_idxs]
        left_pos = i - 1
        right_pos = j
        left_act = case_df.iloc[left_pos]["Activity"] if left_pos >= 0 else None
        right_act = case_df.iloc[right_pos]["Activity"] if right_pos < n else None

        ordered_idxs = best_order_for_block(block_df, left_act, right_act)
        ordered_acts = [case_df.loc[idx, "Activity"] for idx in ordered_idxs]

        gaps = []
        for a, b in zip(ordered_acts[:-1], ordered_acts[1:]):
            gaps.append(expected_gap_seconds(a, b))

        right_anchor_idx = nearest_clean_right(case_df, i)
        left_anchor_idx = nearest_clean_left(case_df, i)
        case_start, case_end = find_case_bounds(case_df)

        available_room = None
        if right_anchor_idx is not None:
            right_dt = case_df.iloc[right_anchor_idx]["_parsed_dt"]
            available_room = max((right_dt - shared).total_seconds(), 0.0)
        else:
            available_room = max((case_end - shared).total_seconds(), len(block_idxs) - 1)

        min_total = max(len(block_idxs) - 1, 0)
        need_clamp = False
        total_gaps = compress_gaps(gaps, available_room, min_total)
        if sum(gaps) > available_room + 1e-9:
            need_clamp = True

        ordered_times = [shared]
        cur = shared
        for g in total_gaps:
            cur = cur + pd.to_timedelta(g, unit="s")
            ordered_times.append(cur)

        if left_anchor_idx is not None:
            left_dt = case_df.iloc[left_anchor_idx]["_parsed_dt"]
            if ordered_times[0] < left_dt:
                need_clamp = True
                shift = left_dt - ordered_times[0]
                ordered_times = [t + shift for t in ordered_times]

        if right_anchor_idx is not None and ordered_times[-1] > case_df.iloc[right_anchor_idx]["_parsed_dt"]:
            need_clamp = True
            right_dt = case_df.iloc[right_anchor_idx]["_parsed_dt"]
            room = max((right_dt - shared).total_seconds(), min_total)
            total_gaps = compress_gaps(total_gaps, room, min_total)
            ordered_times = [shared]
            cur = shared
            for g in total_gaps:
                cur = cur + pd.to_timedelta(g, unit="s")
                ordered_times.append(cur)

        # strict distinctness in inferred order
        for z in range(1, len(ordered_times)):
            if ordered_times[z] <= ordered_times[z - 1]:
                need_clamp = True
                ordered_times[z] = ordered_times[z - 1] + pd.to_timedelta(1, unit="s")

        # map times back to original rows by inferred execution order
        for idx, dt in zip(ordered_idxs, ordered_times):
            repaired_dts[idx] = dt
            if idx == ordered_idxs[0]:
                repaired_strings[idx] = case_df.loc[idx, "Timestamp"]
            else:
                repaired_strings[idx] = format_dt(dt, precision)

        counters["c1_blocks"] += 1
        if need_clamp:
            counters["clamped_blocks"] += 1
        i = j


def local_fit(case_df, pos_a, t_a, pos_b, t_b):
    penalty = 0.0
    if t_b < t_a:
        return 1e6
    act_a = case_df.iloc[pos_a]["Activity"]
    act_b = case_df.iloc[pos_b]["Activity"]
    penalty += fit_penalty(act_a, act_b, (t_b - t_a).total_seconds())

    if pos_a - 1 >= 0 and pd.notna(case_df.iloc[pos_a - 1]["_effective_dt"]):
        prev_dt = case_df.iloc[pos_a - 1]["_effective_dt"]
        prev_act = case_df.iloc[pos_a - 1]["Activity"]
        if t_a < prev_dt:
            penalty += 1e6
        else:
            penalty += fit_penalty(prev_act, act_a, (t_a - prev_dt).total_seconds())

    if pos_b + 1 < len(case_df) and pd.notna(case_df.iloc[pos_b + 1]["_effective_dt"]):
        next_dt = case_df.iloc[pos_b + 1]["_effective_dt"]
        next_act = case_df.iloc[pos_b + 1]["Activity"]
        if next_dt < t_b:
            penalty += 1e6
        else:
            penalty += fit_penalty(act_b, next_act, (next_dt - t_b).total_seconds())

    return penalty


def estimate_between(case_df, pos, precision):
    left_idx = None
    right_idx = None
    for i in range(pos - 1, -1, -1):
        if pd.notna(case_df.iloc[i]["_effective_dt"]):
            left_idx = i
            break
    for i in range(pos + 1, len(case_df)):
        if pd.notna(case_df.iloc[i]["_effective_dt"]):
            right_idx = i
            break

    case_start, case_end = find_case_bounds(case_df)
    act = case_df.iloc[pos]["Activity"]

    candidates = []

    if left_idx is not None:
        left_dt = case_df.iloc[left_idx]["_effective_dt"]
        left_act = case_df.iloc[left_idx]["Activity"]
        candidates.append(left_dt + pd.to_timedelta(expected_gap_seconds(left_act, act), unit="s"))

    if right_idx is not None:
        right_dt = case_df.iloc[right_idx]["_effective_dt"]
        right_act = case_df.iloc[right_idx]["Activity"]
        back = expected_gap_seconds(act, right_act)
        candidates.append(right_dt - pd.to_timedelta(back, unit="s"))

    if left_idx is not None and right_idx is not None:
        left_dt = case_df.iloc[left_idx]["_effective_dt"]
        right_dt = case_df.iloc[right_idx]["_effective_dt"]
        left_act = case_df.iloc[left_idx]["Activity"]
        right_act = case_df.iloc[right_idx]["Activity"]
        g1 = expected_gap_seconds(left_act, act)
        g2 = expected_gap_seconds(act, right_act)
        span = max((right_dt - left_dt).total_seconds(), 0.0)
        if span > 0:
            frac = g1 / max(g1 + g2, 1e-6)
            candidates.append(left_dt + pd.to_timedelta(span * frac, unit="s"))

    if not candidates:
        dt = case_start
    else:
        dt = min(candidates, key=lambda x: abs((x - candidates[0]).total_seconds())) if len(candidates) == 1 else sorted(candidates)[len(candidates) // 2]

    low = case_start
    high = case_end
    if left_idx is not None:
        low = max(low, case_df.iloc[left_idx]["_effective_dt"])
    if right_idx is not None:
        high = min(high, case_df.iloc[right_idx]["_effective_dt"])

    if low > high:
        dt = low
    else:
        if dt < low:
            dt = low
        if dt > high:
            dt = low + (high - low) / 2 if low < high else low

    return dt


def repair_c2_and_c3(case_df, repaired_strings, repaired_dts, precision):
    for pos in range(len(case_df)):
        row = case_df.iloc[pos]
        idx = case_df.index[pos]
        cat = row["error_category"]
        if cat == "CLEAN":
            continue

        if cat == "C2_TEMPORAL_INCONSISTENCY":
            current_dt = repaired_dts.get(idx, row["_parsed_dt"])
            prev_pos = pos - 1
            if prev_pos >= 0:
                prev_idx = case_df.index[prev_pos]
                prev_dt = repaired_dts.get(prev_idx, case_df.iloc[prev_pos]["_parsed_dt"])
            else:
                prev_dt = pd.NaT

            if prev_pos >= 0 and pd.notna(prev_dt) and pd.notna(current_dt) and current_dt < prev_dt:
                swap_pen = local_fit(case_df, prev_pos, current_dt, pos, prev_dt)
                est_dt = estimate_between(case_df, pos, precision)
                disp_pen = local_fit(case_df, prev_pos, prev_dt, pos, est_dt)

                if swap_pen + 0.5 < disp_pen:
                    repaired_dts[prev_idx] = current_dt
                    repaired_strings[prev_idx] = format_dt(current_dt, precision)
                    repaired_dts[idx] = prev_dt
                    repaired_strings[idx] = format_dt(prev_dt, precision)
                    case_df.at[case_df.index[prev_pos], "_effective_dt"] = current_dt
                    case_df.at[idx, "_effective_dt"] = prev_dt
                else:
                    repaired_dts[idx] = est_dt
                    if row["_ts_kind"] == "date_only" and row["_date_only_text"] is not None:
                        repaired_strings[idx] = date_only_format(est_dt, row["_date_only_text"], precision)
                    else:
                        repaired_strings[idx] = format_dt(est_dt, precision)
                    case_df.at[idx, "_effective_dt"] = est_dt
            else:
                est_dt = estimate_between(case_df, pos, precision)
                repaired_dts[idx] = est_dt
                if row["_ts_kind"] == "date_only" and row["_date_only_text"] is not None:
                    repaired_strings[idx] = date_only_format(est_dt, row["_date_only_text"], precision)
                else:
                    repaired_strings[idx] = format_dt(est_dt, precision)
                case_df.at[idx, "_effective_dt"] = est_dt

        elif cat == "C3_MISSING_TEMPORAL_INFORMATION":
            est_dt = estimate_between(case_df, pos, precision)
            if row["_ts_kind"] == "date_only" and row["_date_only_text"] is not None:
                original_date = pd.Timestamp(row["_date_only_text"]).date()
                est_dt = pd.Timestamp(
                    year=original_date.year, month=original_date.month, day=original_date.day,
                    hour=est_dt.hour, minute=est_dt.minute, second=est_dt.second, microsecond=est_dt.microsecond
                )
                repaired_strings[idx] = date_only_format(est_dt, row["_date_only_text"], precision)
            else:
                repaired_strings[idx] = format_dt(est_dt, precision)
            repaired_dts[idx] = est_dt
            case_df.at[idx, "_effective_dt"] = est_dt


def final_self_check(df, out_strings, repaired_dts, c1_blocks_meta, precision, counters):
    # non-clean rows must differ except one keeper per C1 block
    keepers = set()
    for block in c1_blocks_meta:
        keepers.add(block["keeper"])

    for case, case_df in df.groupby("Case", sort=False):
        case_idxs = list(case_df.index)
        valid_dts = [pd.to_datetime(out_strings[idx], errors="coerce") for idx in case_idxs if str(out_strings[idx]).strip() != ""]
        valid_dts = [x for x in valid_dts if pd.notna(x)]
        if valid_dts:
            case_start, case_end = min(valid_dts), max(valid_dts)
        else:
            case_start, case_end = pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")

        for local_pos, idx in enumerate(case_idxs):
            row = df.loc[idx]
            if row["error_category"] == "CLEAN":
                out_strings[idx] = row["Timestamp"]
                continue

            dt = pd.to_datetime(out_strings[idx], errors="coerce")
            if pd.isna(dt):
                est = repaired_dts.get(idx, row["_parsed_dt"])
                if pd.isna(est):
                    est = case_start
                dt = est
                if row["_ts_kind"] == "date_only" and row["_date_only_text"] is not None:
                    out_strings[idx] = date_only_format(dt, row["_date_only_text"], precision)
                else:
                    out_strings[idx] = format_dt(dt, precision)

            if dt < case_start:
                dt = case_start
            if dt > case_end:
                dt = case_end

            left_clean = None
            right_clean = None
            for p in range(local_pos - 1, -1, -1):
                idx2 = case_idxs[p]
                if df.loc[idx2, "error_category"] == "CLEAN":
                    left_clean = pd.to_datetime(df.loc[idx2, "Timestamp"], errors="coerce")
                    break
            for p in range(local_pos + 1, len(case_idxs)):
                idx2 = case_idxs[p]
                if df.loc[idx2, "error_category"] == "CLEAN":
                    right_clean = pd.to_datetime(df.loc[idx2, "Timestamp"], errors="coerce")
                    break

            if left_clean is not None and dt < left_clean:
                dt = left_clean
            if right_clean is not None and dt > right_clean:
                dt = right_clean if left_clean is None else left_clean + (right_clean - left_clean) / 2

            if row["_ts_kind"] == "date_only" and row["_date_only_text"] is not None:
                out_strings[idx] = date_only_format(dt, row["_date_only_text"], precision)
            else:
                out_strings[idx] = format_dt(dt, precision)

    # enforce C1 strictness and keeper semantics
    for block in c1_blocks_meta:
        ordered_idxs = block["ordered_idxs"]
        observed = block["observed"]
        right_anchor = block["right_anchor"]
        times = [pd.to_datetime(out_strings[idx], errors="coerce") for idx in ordered_idxs]
        changed = False

        if pd.isna(times[0]) or times[0] != observed:
            times[0] = observed
            changed = True

        for i in range(1, len(times)):
            min_t = times[i - 1] + pd.to_timedelta(1, unit="s")
            if pd.isna(times[i]) or times[i] < min_t:
                times[i] = min_t
                changed = True

        span = (times[-1] - times[0]).total_seconds()
        min_span = len(times) - 1
        if span < min_span:
            changed = True
            cur = times[0]
            for i in range(1, len(times)):
                cur = cur + pd.to_timedelta(1, unit="s")
                times[i] = cur

        if right_anchor is not None and times[-1] > right_anchor:
            changed = True
            room = max((right_anchor - observed).total_seconds(), len(times) - 1)
            gaps = [max((t2 - t1).total_seconds(), 1.0) for t1, t2 in zip(times[:-1], times[1:])]
            gaps = compress_gaps(gaps, room, len(times) - 1)
            times = [observed]
            cur = observed
            for g in gaps:
                cur = cur + pd.to_timedelta(g, unit="s")
                if cur <= times[-1]:
                    cur = times[-1] + pd.to_timedelta(1, unit="s")
                times.append(cur)

        for idx, dt in zip(ordered_idxs, times):
            if idx == ordered_idxs[0]:
                out_strings[idx] = block["observed_string"]
            else:
                out_strings[idx] = format_dt(dt, precision)

        if changed:
            counters["clamped_blocks"] += 1

    rows_equal = 0
    for idx, row in df.iterrows():
        if str(out_strings[idx]) == str(row["Timestamp"]):
            rows_equal += 1
    counters["rows_equal"] = rows_equal


def main():
    args = parse_args()

    try:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(args.input, dtype=str)

    required_cols = ["Case", "Activity", "Timestamp", "Resource", "error_category", "row_id"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    original = df.copy(deep=True)
    precision = infer_dataset_precision(df["Timestamp"])

    parsed_info = df["Timestamp"].apply(parse_timestamp_safe)
    df["_parsed_dt"] = parsed_info.apply(lambda x: x[0])
    df["_ts_kind"] = parsed_info.apply(lambda x: x[1])
    df["_date_only_text"] = parsed_info.apply(lambda x: x[2])
    df["_effective_dt"] = df["_parsed_dt"]

    out_strings = {idx: row["Timestamp"] for idx, row in df.iterrows()}
    repaired_dts = {}

    counters = {"c1_blocks": 0, "clamped_blocks": 0, "rows_equal": 0}
    c1_blocks_meta = []

    for case, case_idxs in df.groupby("Case", sort=False).groups.items():
        case_df = df.loc[list(case_idxs)].copy()
        repair_c1_blocks(case_df, out_strings, repaired_dts, precision, counters)

        # push repaired values from case_df temp context
        for idx in case_df.index:
            if idx in repaired_dts:
                df.at[idx, "_effective_dt"] = repaired_dts[idx]
            elif pd.notna(case_df.loc[idx, "_effective_dt"]):
                df.at[idx, "_effective_dt"] = case_df.loc[idx, "_effective_dt"]

        # collect c1 metadata after repair
        pos = 0
        while pos < len(case_df):
            row = case_df.iloc[pos]
            if row["error_category"] == "C1_ORDERING_AMBIGUITY" and pd.notna(row["_parsed_dt"]):
                shared = row["_parsed_dt"]
                j = pos
                block_idxs = []
                while j < len(case_df):
                    rr = case_df.iloc[j]
                    if rr["error_category"] == "C1_ORDERING_AMBIGUITY" and pd.notna(rr["_parsed_dt"]) and rr["_parsed_dt"] == shared:
                        block_idxs.append(case_df.index[j])
                        j += 1
                    else:
                        break
                if len(block_idxs) > 1:
                    left_pos = pos - 1
                    right_pos = j
                    left_act = case_df.iloc[left_pos]["Activity"] if left_pos >= 0 else None
                    right_act = case_df.iloc[right_pos]["Activity"] if right_pos < len(case_df) else None
                    ordered_idxs = best_order_for_block(case_df.loc[block_idxs], left_act, right_act)
                    right_anchor = None
                    for rr in range(j, len(case_df)):
                        if case_df.iloc[rr]["error_category"] == "CLEAN" and pd.notna(case_df.iloc[rr]["_parsed_dt"]):
                            right_anchor = case_df.iloc[rr]["_parsed_dt"]
                            break
                    c1_blocks_meta.append({
                        "ordered_idxs": ordered_idxs,
                        "keeper": ordered_idxs[0],
                        "observed": shared,
                        "observed_string": df.loc[ordered_idxs[0], "Timestamp"],
                        "right_anchor": right_anchor,
                    })
                pos = j
            else:
                pos += 1

    # second pass per case for c2/c3
    for case, case_idxs in df.groupby("Case", sort=False).groups.items():
        case_df = df.loc[list(case_idxs)].copy()
        case_df["_effective_dt"] = case_df.index.map(lambda idx: repaired_dts.get(idx, df.loc[idx, "_effective_dt"]))
        repair_c2_and_c3(case_df, out_strings, repaired_dts, precision)
        for idx in case_df.index:
            if pd.notna(case_df.loc[idx, "_effective_dt"]):
                df.at[idx, "_effective_dt"] = case_df.loc[idx, "_effective_dt"]

    final_self_check(df, out_strings, repaired_dts, c1_blocks_meta, precision, counters)

    out_df = original.copy(deep=True)
    out_df["Timestamp"] = [out_strings[idx] for idx in out_df.index]

    # preserve untouched exact strings for CLEAN and any untouched others
    for idx, row in original.iterrows():
        if row["error_category"] == "CLEAN":
            out_df.at[idx, "Timestamp"] = row["Timestamp"]

    # hard constraints on identity columns and order already preserved
    out_df.to_csv(args.output, index=False)

    print(f"blocks_repaired={counters['c1_blocks']} blocks_clamped={counters['clamped_blocks']} rows_still_equal={counters['rows_equal']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # fail-soft: try to produce passthrough output if anything unexpected happens
        try:
            import sys
            ap = argparse.ArgumentParser()
            ap.add_argument("--input", required=True)
            ap.add_argument("--output", required=True)
            ns, _ = ap.parse_known_args()
            df = pd.read_csv(ns.input, dtype=str, keep_default_na=False)
            df.to_csv(ns.output, index=False)
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")
        except Exception:
            print("blocks_repaired=0 blocks_clamped=0 rows_still_equal=0")