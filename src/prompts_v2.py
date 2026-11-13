# -*- coding: utf-8 -*-
"""Rewritten repair prompts, authored from the run diagnostics.

What the measurements say is actually wrong, and what each change addresses:

C1  M2 reaches exact_match 0.949 and noop_rate 0.352 -- identical to M1 on both.
    Ordering and anchoring are already solved. The entire remaining gap
    (105.1 vs 83.9) is INTERNAL SPACING. Yet the old prompt steers spacing the
    wrong way: it says "prefer p10/p25 for intra-block spacing" while the
    baseline that wins uses the per-pair MEDIAN (CFG C1_DURATION_STAT).
    Diagnostics also show programs emitting negative durations (118-1342 per
    run) and "56% of blocks have a span outside p10-p90".
      -> name the single decision explicitly (which row is first)
      -> give the spacing rule that the measurements support, and make the
         program CALIBRATE rather than take my word for it
      -> forbid negative durations as a hard post-condition

C2  The old prompt describes EXCHANGED vs DISPLACED but gives no decision
    procedure, so each program invents one. Result: trial spread 118-417 on
    pub, with 3 of 6 cells landing exactly on M0_null (no effective repair).
      -> give an explicit, computable discriminator with a tie-break
      -> state that swap is exactly invertible and must be preferred when it
         fits, because it is the only branch that can be exactly right

C3  Already beats M1. Changes are minimal: keep what works, state the
    blank-filling requirement that G5 now enforces.

Everything else -- hard constraints, repair budget, process context, output
format -- is unchanged from the original, so this remains a prompt-content
change and not a change of task.
"""

from prompts_percat import (HEADER, CONSTRAINTS, BUDGET_C1, BUDGET_OTHER,
                            CONTEXT_BLOCK, OUTPUT_FORMAT, CATEGORY_LABEL,
                            DIRECT_HEADER, DIRECT_TAIL)

# =====================================================================
# C1
# =====================================================================
C1_BODY_V2 = r"""
C1_ORDERING_AMBIGUITY
  Several consecutive events in one case carry one identical timestamp. That
  shared value is NOT invented: it is one of the block's own timestamps, the
  earliest one, kept when the others were overwritten by the batch write.

  THE WHOLE TASK REDUCES TO ONE DECISION PER BLOCK:
  which of the block's rows is the FIRST event in execution order.
  That row keeps the observed value unchanged. Every other row is placed
  forward from it. Choose that row wrongly and the entire block is wrong, even
  though your output will still look well-formed -- it will contain the
  observed value, the timestamps will be distinct, and nothing will flag it.

  Step 1  ORDER THE BLOCK.
     The rows are NOT in execution order; they were shuffled when the block was
     recorded. Enumerate the permutations of the block's activities (k is 2-4,
     so this is at most 24 candidates). Score each one by the summed log
     transition probability of the chain
          left_anchor_activity -> ... block activities ... -> right_anchor_activity
     using transition_probabilities from the context. Include BOTH anchors in
     the chain; they carry most of the signal for short blocks. Take the
     highest-scoring permutation.
     If your positions come out in the order the rows appear in the file, you
     have not ordered anything -- that order is random by construction.

  Step 2  PLACE THE FIRST EVENT.
     The row ranked first in step 1 keeps the observed shared timestamp
     EXACTLY. Do not shift it by even one second.

  Step 3  SPACE THE REST.
     For each consecutive pair (a, b) in your inferred order, take the duration
     from the per-pair statistics. Use the MEDIAN, not p10 or p25.
     This is measured, not assumed: the statistical baseline on these logs uses
     the per-pair median and is the most accurate method known on this task,
     and runs that compressed blocks toward p10 scored worse. If you want to
     verify it yourself, compute both layouts and keep the one whose resulting
     within-block gaps have a distribution closer to the gaps between
     consecutive CLEAN events of the same activity pairs. Do not use
     block_span_profile to size a block: it is measured on ordinary consecutive
     events, most of which are separated by waiting time, whereas a form-based
     block happened inside one working session.

  Step 4  FIT THE ROOM.
     Let room = right_anchor - observed_value. If the sum of your gaps exceeds
     room, scale all gaps by the same factor so they fit inside 95% of room.
     Never truncate the last gap alone. If there is no right anchor, use the
     gaps as they are.

  Step 5  WRITE BACK.
     Each event's timestamp goes onto ITS OWN ROW, wherever that row sits in
     the file. The row order never changes; the ordering lives in the values.
"""

SELFCHECK_C1_V2 = r"""
=====================================================================
POST-CONDITIONS -- your program must verify these and repair violations
=====================================================================
When a check fails, fix the candidate: re-space it, rescale it, clamp it.
Never fall back to the observed value, and never leave a row unrepaired.

  1. Every C1 block has k DISTINCT, strictly increasing timestamps.
  2. Exactly one row per block carries the observed shared value, and it is the
     EARLIEST of the block.
  3. NO NEGATIVE DURATION ANYWHERE IN THE CASE after repair -- not inside the
     block, and not between the block and either anchor. Walk every repaired
     case end to end and assert the timestamps are non-decreasing. Previous
     programs emitted hundreds of negative durations here and still passed
     every other check.
  4. Every repaired row lies within [left_anchor, right_anchor].
  5. Every corrupted row differs from its input, except the one row in rule 2.

Print one line to stdout: blocks repaired, blocks clamped, negative durations
remaining (must be 0), rows still equal to their input (should be close to the
number of C1 blocks, one per block, and no more).

=====================================================================
WORKED EXAMPLE
=====================================================================
Anchors: E4 "Submit" at 09:00:00 (clean), E8 "Close" at 11:00:00 (clean).
Block, three rows, all reading 09:12:00:
    row 5  B
    row 6  D
    row 7  A
Transition statistics: Submit->A frequent, A->B median 90 s, B->D median 240 s,
D->Close frequent; D->A rare.

Step 1  Score the permutations of (B, D, A) in the chain Submit -> ... -> Close.
        The order A, B, D scores highest.
Step 2  Row 7 (A) ranks first, so row 7 KEEPS 09:12:00.
Step 3  A->B median 90 s, B->D median 240 s.
Step 4  room = 11:00:00 - 09:12:00 = 6,480 s. 90 + 240 = 330 s fits; no scaling.
Step 5  row 5 (B) = 09:13:30, row 6 (D) = 09:17:30, row 7 (A) = 09:12:00.

Two of the three rows changed, the block is increasing and distinct, nothing
left the anchors, and no negative duration was created.

Failures observed in practice, all of which pass a naive self-check:
  - leaving all three at 09:12:00
  - rebuilding from 09:00:00, so no row keeps the observed value
  - giving the observed value to row 5 instead of row 7 -- well-formed output,
    and the whole block is wrong
  - spacing evenly at 09:12, 09:47, 10:22 because a span profile said a
    three-event run "should" last an hour
"""

# =====================================================================
# C2
# =====================================================================
C2_BODY_V2 = r"""
C2_TEMPORAL_INCONSISTENCY
  A later event carries an earlier timestamp than its predecessor. Two
  different faults produce this and they need OPPOSITE repairs, so classify
  first and repair second. Do not blend the two.

  EXCHANGED. The two values were swapped. Both rows are wrong, and putting
    each value back on the other row is EXACTLY right -- this is the only
    branch in this whole task where a repair can be perfect, so it is worth
    testing for carefully.
  DISPLACED. One row was pushed backwards by an arbitrary offset. Its partner
    is CORRECT and must not be touched. The true value was destroyed, so it has
    to be estimated.

  THE DISCRIMINATOR -- compute both candidates, then choose:

    Let the pair be rows (i, j) in file order with timestamps (t_i, t_j),
    activities (a_i, a_j), and clean anchors L (time l) and R (time r).

    Candidate SWAP: assign t_j to row i and t_i to row j.
      Reject it outright if the swap does not satisfy
          l <= min(t_i,t_j)  and  max(t_i,t_j) <= r
      Otherwise score it: for each of the two resulting consecutive gaps, take
      the per-pair duration statistics for that activity pair and measure how
      typical the gap is -- for example |log(gap) - log(median_for_that_pair)|,
      summed over the two gaps. Lower is better.

    Candidate ESTIMATE: keep the earlier-positioned row as observed and
      re-estimate the other from its predecessor plus that pair's median
      duration, clamped into (l, r). Score it the same way.

    Choose the lower-scoring candidate. When the two scores are within a factor
    of two of each other, PREFER SWAP: it is exactly invertible when it is
    right, while an estimate is never exactly right.

  Do not assume either branch. A program that always sorts destroys the correct
  row of every displaced pair; a program that always estimates throws away the
  only exact repair available in this task.
"""

SELFCHECK_C2_V2 = r"""
=====================================================================
POST-CONDITIONS -- your program must verify these and repair violations
=====================================================================
  1. Every corrupted row has a new value; none is returned as it arrived.
  2. The repaired pair is non-decreasing with respect to each other and to both
     anchors, and both lie within [left_anchor, right_anchor].
  3. For a pair classified EXCHANGED, the two output values are exactly the two
     input values with the rows swapped. No new value was invented.
  4. For a row classified DISPLACED, its partner's timestamp is byte-identical
     to its input.
  5. No negative duration anywhere in the case after repair.

Print one line: pairs classified exchanged, pairs classified displaced, rows
still equal to their input, negative durations remaining (must be 0).
If "exchanged" is 0 or equals the total, your discriminator is not discriminating.

=====================================================================
WORKED EXAMPLE
=====================================================================
Anchors: E2 at 09:00:00, E7 at 09:30:00. Median durations: X->Y 60 s,
Y->Z 900 s.

Case A. Rows 4 (X) and 5 (Y) read 09:20:00 and 09:05:00.
  SWAP -> row4 09:05:00, row5 09:20:00. Both inside the anchors. Resulting
  gaps are typical for X->Y and Y->Z. Score low.
  ESTIMATE -> keep 09:20:00 on row 4 and push row 5 to 09:21:00; that leaves
  the earlier part of the interval empty and the Y->Z gap atypically long.
  Score higher.
  CHOOSE SWAP. Both rows change, no value invented.

Case B. Rows 4 (X) and 5 (Y) read 09:25:00 and 09:02:00.
  SWAP -> row 5 would take 09:25:00, making the X->Y gap 23 minutes against a
  60 s median. Score high.
  ESTIMATE -> row 4 is correct at 09:25:00; row 5 becomes 09:26:00. Score low.
  CHOOSE ESTIMATE. Only row 5 changes; row 4 is untouched.

Failures observed in practice: sorting every pair unconditionally, which
destroys row 4 in case B; returning a row unchanged because neither branch felt
certain, which is scored as a non-repair.
"""

# =====================================================================
# C3
# =====================================================================
C3_BODY_V2 = r"""
C3_MISSING_TEMPORAL_INFORMATION
  Two sub-cases, and they are not interchangeable.

  DATE-ONLY. The date survived; the time of day is gone (the value reads
    midnight). KEEP THAT DATE. Infer only the time of day, from the left
    anchor plus the per-pair median duration. If that estimate would land on
    another calendar day, clamp it to the last moment of the correct day
    rather than moving the event. Never change the date.

  BLANK. There is no timestamp at all. You MUST produce one. Estimate from the
    left anchor plus the per-pair median duration for that transition, and
    clamp into (left_anchor, right_anchor). If the estimate overruns the right
    anchor, place the event between the two anchors in proportion to its
    position among the missing events, NOT one second before the right anchor.
    Leaving a blank row blank is not a conservative choice: a row with no
    timestamp is dropped from the evaluation entirely, so declining to answer
    removes the row rather than scoring neutrally on it.
"""

SELFCHECK_C3_V2 = r"""
=====================================================================
POST-CONDITIONS -- your program must verify these and repair violations
=====================================================================
  1. EVERY corrupted row has a parseable timestamp. Zero blanks remain.
  2. For a row whose date survived, the repaired value falls on that same
     calendar date.
  3. Every repaired row lies within [left_anchor, right_anchor].
  4. No negative duration anywhere in the case after repair.
  5. Every corrupted row differs from its input value.

Print one line: blank rows filled, date-only rows given a time of day, rows
still blank (must be 0), negative durations remaining (must be 0).

=====================================================================
WORKED EXAMPLE
=====================================================================
Anchors: E3 at 2015-03-04 09:00:00, E6 at 2015-03-04 17:00:00.

Date-only. Row 4 reads 2015-03-04 00:00:00. The E3->row4 median is 1,800 s, so
the repair is 2015-03-04 09:30:00 -- same date, inside the anchors.

Blank. Row 5 has no value. Left anchor is row 4 at 09:30:00 and the median for
that transition is 6,300 s, giving 2015-03-04 11:15:00, inside the anchors.

Failures observed in practice: moving the date-only row to 2015-03-05 because
the estimate ran past midnight; pinning the blank row at 16:59:59; leaving the
blank row blank.
"""

_BODY = {"C1": C1_BODY_V2, "C2": C2_BODY_V2, "C3": C3_BODY_V2}
_SELF = {"C1": SELFCHECK_C1_V2, "C2": SELFCHECK_C2_V2, "C3": SELFCHECK_C3_V2}
_BUDGET = {"C1": BUDGET_C1, "C2": BUDGET_OTHER, "C3": BUDGET_OTHER}

PREAMBLE_V2 = r"""
=====================================================================
WHAT THE CORRUPTED VALUE IS
=====================================================================
The corrupted timestamp is not noise. It is a damaged but informative
observation, and discarding it is the most expensive mistake available here.
Read the following before designing anything.
"""

DURATION_STATS_V2 = r"""
=====================================================================
USING THE DURATION STATISTICS
=====================================================================
Each activity pair carries n, p10, p25, median, p75, p90 and share_under_60s,
estimated from CLEAN rows only. These gaps are bimodal: a fast mode inside one
working session and a slow mode waiting on an external party.

Use the MEDIAN as the default spacing statistic. share_under_60s tells you
which mode a transition usually belongs to and is useful for sanity-checking a
layout, but on these logs the median is what the most accurate known method
uses, and layouts compressed toward p10 scored worse. A pair with n below about
5 has an unreliable median; fall back to the global median for those.
"""


def codegen_prompt(cat):
    return (HEADER % CATEGORY_LABEL[cat]
            + PREAMBLE_V2 + _BODY[cat]
            + CONSTRAINTS + _BUDGET[cat] + DURATION_STATS_V2
            + CONTEXT_BLOCK + _SELF[cat] + OUTPUT_FORMAT)


_DIRECT_BODY_V2 = {
    "C1": C1_BODY_V2.replace("Step 5  WRITE BACK.",
                             "Step 5  REPORT.").replace(
        "Each event's timestamp goes onto ITS OWN ROW, wherever that row sits\n"
        "     in the file. The row order never changes; the ordering lives in the values.",
        "Return position 1..k along your inferred order and offset_seconds from\n"
        "     reference_timestamp. The row you rank first gets offset 0."),
    "C2": C2_BODY_V2,
    "C3": C3_BODY_V2,
}


def direct_prompt(cat):
    return (DIRECT_HEADER % CATEGORY_LABEL[cat]
            + _DIRECT_BODY_V2[cat] + DIRECT_TAIL)
