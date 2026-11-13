# Strategies implemented by the synthesized programs

## Judge: gpt-5

Share of the 148 programs implementing each strategy, overall and by error category.

| Strategy | All | C1 | C2 | C3 |
|---|---|---|---|---|
| perm | 67% | 100% | 50% | 51% |
| beam | 45% | 53% | 40% | 41% |
| transprob | 80% | 100% | 86% | 55% |
| rightanch | 99% | 100% | 98% | 100% |
| r_ceiling | 91% | 96% | 90% | 88% |
| r_evidence | 98% | 98% | 96% | 100% |
| median | 24% | 16% | 30% | 27% |
| mean | 1% | 0% | 0% | 2% |
| lognorm | 14% | 12% | 14% | 14% |
| uniform | 27% | 27% | 20% | 35% |
| rescale | 80% | 100% | 50% | 92% |
| firstts | 67% | 100% | 50% | 51% |
| monotone | 99% | 100% | 100% | 98% |
| fallback | 100% | 100% | 100% | 100% |

## Judge: gpt-4o

Share of the 148 programs implementing each strategy, overall and by error category.

| Strategy | All | C1 | C2 | C3 |
|---|---|---|---|---|
| perm | 67% | 100% | 50% | 51% |
| beam | 40% | 41% | 38% | 41% |
| transprob | 83% | 100% | 90% | 59% |
| rightanch | 99% | 100% | 98% | 100% |
| r_ceiling | 95% | 98% | 94% | 94% |
| r_evidence | 76% | 65% | 70% | 92% |
| median | 84% | 78% | 90% | 86% |
| mean | 1% | 0% | 2% | 2% |
| lognorm | 35% | 43% | 26% | 37% |
| uniform | 20% | 27% | 14% | 18% |
| rescale | 79% | 100% | 50% | 88% |
| firstts | 57% | 88% | 44% | 41% |
| monotone | 97% | 100% | 98% | 94% |
| fallback | 97% | 94% | 98% | 100% |

## Agreement between the two judges

| strategy | gpt-4o | gpt-5 | agreement | gpt-4o_only | gpt-5_only |
|---|---|---|---|---|---|
| perm | 0.669 | 0.669 | 1.000 | 0 | 0 |
| beam | 0.399 | 0.446 | 0.953 | 0 | 7 |
| transprob | 0.831 | 0.804 | 0.959 | 5 | 1 |
| rightanch | 0.993 | 0.993 | 0.986 | 1 | 1 |
| r_ceiling | 0.953 | 0.912 | 0.892 | 11 | 5 |
| r_evidence | 0.757 | 0.980 | 0.764 | 1 | 34 |
| median | 0.845 | 0.243 | 0.385 | 90 | 1 |
| mean | 0.014 | 0.007 | 0.993 | 1 | 0 |
| lognorm | 0.351 | 0.135 | 0.770 | 33 | 1 |
| uniform | 0.196 | 0.270 | 0.750 | 13 | 24 |
| rescale | 0.791 | 0.804 | 0.973 | 1 | 3 |
| firstts | 0.574 | 0.669 | 0.905 | 0 | 14 |
| monotone | 0.973 | 0.993 | 0.980 | 0 | 3 |
| fallback | 0.973 | 1.000 | 0.973 | 0 | 4 |

## Two-sided anchor solving, asked narrowly (judge: gpt-5)

The `r_evidence` row above asks whether a program uses the right anchor as a constraint at all, and reads broadly: laying the block out from the left anchor and then rescaling it proportionally to fit the interval satisfies it, and that is what STAT does too. This second pass asks only for the mechanism that separates MAXL from STAT -- solving for the gaps under an equality constraint with a per-transition duration model, or fitting the transition into the right anchor's own activity -- and explicitly excludes clipping, clamping and proportional rescaling. It is the narrow figure that the discussion of MAXL relies on.

| Scope | Programs | Solve jointly | Share |
|---|---|---|---|
| all | 148 | 28 | 19% |
| C1 | 49 | 0 | 0% |
| C2 | 50 | 12 | 24% |
| C3 | 49 | 16 | 33% |
| LLMU / C1 | 25 | 0 | 0% |
| LLMU / C2 | 25 | 0 | 0% |
| LLMU / C3 | 25 | 0 | 0% |
| LLMD / C1 | 24 | 0 | 0% |
| LLMD / C2 | 25 | 12 | 48% |
| LLMD / C3 | 24 | 16 | 67% |

The split by arm is the sharpest thing here: no LLMU program solves jointly anywhere, and every instance is LLMD, the arm whose prompt carries the directly-follows graph and the activity duration profiles. On C1 -- the category the comparison with MAXL is about -- neither arm does it at all.

The programs it finds, with the line cited:

| Log | Cat | Arm | Trial | Evidence |
|---|---|---|---|---|
| bpic12 | C2 | LLMD | 1 | target = preferred_gap(this_act, right_act) |
| bpic12 | C2 | LLMD | 5 | d = choose_duration_seconds(act, right_a) |
| bpic12 | C3 | LLMD | 1 | scaled = [g / s * total_window for g in gaps] |
| bpic12 | C3 | LLMD | 3 | last = choose_edge_gap(acts[-1], right_act) |
| bpic15 | C2 | LLMD | 2 | g = prof_gap(acts[idx_local], acts[right_idx]) |
| bpic15 | C2 | LLMD | 3 | pen += gap_penalty(act, act_right, (right_ts - c).total_seconds()) |
| bpic15 | C2 | LLMD | 4 | score += duration_penalty(act, next_act, max(0.0, (t_next - c).total_seconds())) |
| bpic15 | C3 | LLMD | 2 | step = effective_gap * (w / total_w) |
| bpic15 | C3 | LLMD | 3 | total_window = (right_anchor - left_anchor).total_seconds() |
| credit | C2 | LLMD | 2 | cost += plausibility_cost(td_seconds(right_ts - cand), act, right_act) |
| credit | C2 | LLMD | 5 |         if right_t is not None:
            score += gap_penalty((right_t - c).total_seconds(), act, right_act) |
| credit | C3 | LLMD | 2 | positions.append(left_t + (right_t - left_t) * cum) |
| credit | C3 | LLMD | 3 | last_w = edge_weight_seconds(acts[-1], next_act) if n and next_act is not None else None |
| credit | C3 | LLMD | 4 | cum = edge_prior_seconds(seq_acts[-1], right_act) |
| credit | C3 | LLMD | 5 | weights.append(max(right_w if right_w is not None else fallback_gap(), 0.0)) |
| hospital | C2 | LLMD | 1 | sec = choose_stat((act, next_act)) if edge_exists(act, next_act) else GLOBAL_MEDIAN |
| hospital | C2 | LLMD | 3 | e = right_t - pd.to_timedelta(duration_prior(act, right_a), unit="s") |
| hospital | C2 | LLMD | 4 | cands.append(prev_t + (next_t - prev_t) * (left_gap / total)) |
| hospital | C3 | LLMD | 1 | gap = choose_edge_gap_seconds(act, nxt) |
| hospital | C3 | LLMD | 2 | times = [left_ts + pd.Timedelta(seconds=total_window * (c / denom)) for c in cum] |
| hospital | C3 | LLMD | 3 | backward_candidate = right_clean[1] - pd.to_timedelta(span * frac, unit="s") |
| hospital | C3 | LLMD | 4 | rev_gaps.append(choose_gap(acts[i], current_next, n + 1, i)) |
| hospital | C3 | LLMD | 5 | b = case_rows.iloc[right_anchor_idx]["Activity"] if right_anchor_idx is not None else None |
| pub | C2 | LLMD | 2 | score += profile_score(act, right_act, (right_ts - cand).total_seconds()) |
| pub | C2 | LLMD | 3 | right_est = (estimate_from_right(r2["Activity"], next_row["Activity"], next_row["parsed_ts"])[0], choose_gap_seconds(r2["Activity"], next_row["Activity"])) |
| pub | C3 | LLMD | 2 | chain_acts = ([prev_act] if prev_act is not None else []) + activities + ([next_act] if next_act is not None else []) |
| pub | C3 | LLMD | 4 | t = left_ts + pd.Timedelta(seconds=left_pad + usable * frac) |
| pub | C3 | LLMD | 5 | t_fit = left_anchor + p * interval |
