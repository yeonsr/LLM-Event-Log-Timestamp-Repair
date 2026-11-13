# -*- coding: utf-8 -*-
"""Category-specific prompts for M2 / M3 / M4.

DESIGN RULE: these are the ORIGINAL unified prompt, SPLIT. Shared sections are
reproduced verbatim from PROMPT_BASE / DIRECT_PROMPT; each category keeps only
its own error description. Nothing was added to make a category easier, with
one declared exception:

  the unified prompt carries a self-check list and a worked example that are
  both C1-only. Giving C1 an example and C2/C3 none would make the split arm
  unbalanced, so a C2 and a C3 example were written. They restate the
  corresponding C2/C3 paragraphs of the unified prompt and introduce no
  strategy that was not already there. This is the one place the split arm is
  not a strict subset of the unified arm, and it must be stated in the paper.

Everything else -- hard constraints, repair budget, duration statistics,
process context, output format -- is byte-identical across the three.
"""

# =====================================================================
# shared blocks, verbatim from PROMPT_BASE
# =====================================================================

HEADER = r"""
You are an expert in process mining, temporal event-log repair, and executable
Python program synthesis.

Input CSV columns: Case, Activity, Timestamp, Resource, error_category, row_id
Detection is already done. Repair ONLY rows whose error_category is not CLEAN.

This log contains exactly ONE kind of error: %s. No other error
category appears in it. Do not write branches for errors that are not here.
"""

PREAMBLE = r"""
=====================================================================
WHAT THE CORRUPTED VALUES ACTUALLY ARE
=====================================================================

Read this before designing anything. Methods fail on this task by treating the
corrupted timestamp as noise. It is not noise; it is a damaged but informative
observation, and the repair must start from it.
"""

C1_BODY = r"""
C1_ORDERING_AMBIGUITY
  Several consecutive events in one case carry one identical timestamp.
  That shared value is not invented: it is ONE OF THE BLOCK'S OWN TIMESTAMPS,
  kept when the others were overwritten by the batch write. So exactly one event
  in the block is ALREADY CORRECT and the rest are displaced from it.
  Therefore:
    - Do NOT rebuild the block from the left anchor by adding durations. That
      discards the one value you know to be right and drifts the whole block.
    - Decide the ordering first. Then place the block so that it STARTS at the
      observed shared timestamp and extends forward, unless the surrounding
      evidence clearly says the shared value is the block's end.
    - The first event in your inferred order keeps the observed value exactly.
    - Size the block from the room that exists between the observed value and
      the right anchor. Do NOT size it from block_span_profile: that profile is
      measured on uncorrupted consecutive events, most of which are separated by
      waiting time, whereas a form-based block by definition happened inside one
      working session. Using it to size a block, or to validate one, will make
      every block look wrong and push you into doing nothing.
    - Leaving the block collapsed is the one outcome that is always wrong. k
      identical timestamps must become k distinct ones.
"""

C2_BODY = r"""
C2_TEMPORAL_INCONSISTENCY
  A later event carries an earlier timestamp than its predecessor.
  Two different faults produce this, and they need different repairs. Decide
  which one you are looking at before acting:
    - EXCHANGED. The two values were swapped, so both rows are wrong and the
      pair is exactly recoverable: put each value back on the other row. The
      giveaway is that the two timestamps together still fit the surrounding
      gaps -- sorting them restores a plausible spacing.
    - DISPLACED. One row was pushed backwards by an arbitrary offset. Its
      partner is CORRECT and must not be touched. Sorting cannot recover this,
      because the true value was destroyed rather than moved elsewhere; you have
      to estimate it from the anchors and the duration statistics, exactly as
      you would for a missing value.
  Do not assume the exchanged case. Check whether swapping the pair produces
  durations consistent with the duration profiles for those transitions, and
  fall back to estimation when it does not.
"""

C3_BODY = r"""
C3_MISSING_TEMPORAL_INFORMATION
  Blank, or date-only with the time of day gone.
  If a date is present, KEEP THAT DATE and infer only the time of day. Never move
  the event to another day. When both anchors exist and the imputed value would
  run past the right anchor, place it between the anchors rather than pinning it
  one second before the right anchor.
"""

CONSTRAINTS = r"""
=====================================================================
HARD CONSTRAINTS
=====================================================================
1. Output row count MUST equal input row count.
2. Case, Activity, Resource, error_category, row_id MUST be byte-identical.
3. Row order MUST NOT change. Express ordering through timestamp VALUES only.
4. error_category == CLEAN rows MUST keep their exact input timestamp.
5. Repaired timestamps MUST parse with pandas.to_datetime and keep the input's
   precision (milliseconds when the input has them).
6. Uniform or evenly spaced interpolation inside a block is FORBIDDEN.
7. FORBIDDEN fallbacks: 1970-01-01, the dataset minimum, a timestamp from
   another case, an elapsed-seconds value used as an absolute time.
8. Durations are RELATIVE intervals: add or subtract them from a same-case anchor.
9. For every repaired row: case_start <= t <= case_end, and when both clean
   anchors exist, left_anchor <= t <= right_anchor.
"""

BUDGET_C1 = r"""
=====================================================================
REPAIR BUDGET
=====================================================================
Among repairs that are all valid, prefer the one that moves rows least. A
repair landing within seconds of the truth for most rows beats one that is
spectacular on a few and wrong by hours on the rest.

This is a tie-breaker between valid repairs. It is NOT a licence to do nothing.
Returning the observed value for a corrupted row is not the minimal repair, it
is the absence of one: the observed value IS the corruption. Every row whose
error_category is not CLEAN must leave your program with a different value from
the one it arrived with, with one exception -- the single row in each C1 block
that legitimately keeps the shared timestamp because it is the block's first
event.
"""

BUDGET_OTHER = r"""
=====================================================================
REPAIR BUDGET
=====================================================================
Among repairs that are all valid, prefer the one that moves rows least. A
repair landing within seconds of the truth for most rows beats one that is
spectacular on a few and wrong by hours on the rest.

This is a tie-breaker between valid repairs. It is NOT a licence to do nothing.
Returning the observed value for a corrupted row is not the minimal repair, it
is the absence of one: the observed value IS the corruption. Every row whose
error_category is not CLEAN must leave your program with a different value from
the one it arrived with.
"""

DURATION_STATS = r"""
=====================================================================
USING THE DURATION STATISTICS
=====================================================================
These gaps are bimodal: a fast mode (same working session, seconds) and a slow
mode (waiting for an external party, hours or days). A per-pair MEDIAN mixes the
two and is usually far too large for events inside one form-based block, which
by construction happened in one session. Use share_under_60s to tell which mode
a transition belongs to, and prefer p10/p25 for intra-block spacing.
"""

CONTEXT_BLOCK = r"""
=====================================================================
PROCESS CONTEXT (from CLEAN rows only)
=====================================================================

<<CONTEXT>>
"""

OUTPUT_FORMAT = r"""
=====================================================================
OUTPUT FORMAT
=====================================================================
Return ONE complete Python script in a single ```python code block, taking
--input and --output arguments. No explanation outside the code block.
"""

# =====================================================================
# per-category self-checks and worked examples
# =====================================================================

SELFCHECK_C1 = r"""
=====================================================================
SELF-CHECK BEFORE WRITING THE OUTPUT
=====================================================================
Your program must verify its own output. When a check fails, REPAIR THE
CANDIDATE -- clamp it, rescale it, re-space it -- and never fall back to the
observed value.

  - inside every repaired C1 block the timestamps are strictly increasing and
    all distinct
  - exactly one row of each C1 block carries the observed shared timestamp, and
    it is the earliest of the block
  - the block's total span is at most the distance from the observed value to
    the right anchor, and at least (k-1) seconds
  - no repaired row sits outside its anchors
  - every corrupted row differs from its input value, except the one C1 row
    described above

Print one line to stdout with the count of blocks repaired, blocks that needed
clamping, and rows still equal to their input. If that last number is not close
to the number of C1 blocks, your program is not doing its job.

=====================================================================
THE ROW ORDER INSIDE A CORRUPTED BLOCK IS MEANINGLESS
=====================================================================
When a form-based block was recorded, the events were written in whatever order
the system happened to persist them. The rows you receive are NOT in execution
order, and treating them as if they were is the most common way to fail this
task.

If your program walks a block from its first row to its last and assigns
increasing timestamps as it goes, it has separated the block without ordering
it, and the result is a confidently wrong sequence. That is worse than leaving
the block collapsed, because the collapsed value at least sits at the true time
of the block's first event.

  1. score the candidate orderings of the block's activities against the
     transition statistics, including the left and right anchor activities
  2. take the best-scoring ordering
  3. lay the timestamps out along THAT ordering
  4. write each event's timestamp back onto ITS OWN ROW, wherever that row sits
     in the file

Step 4 keeps the row structure intact while the ordering lives in the values.
If, after your repair, every block's timestamps increase from its first row to
its last, you skipped steps 1 to 3.

=====================================================================
WORKED EXAMPLE
=====================================================================
Anchors: E4 at 09:00:00 (clean), E8 at 11:00:00 (clean).
Corrupted block, three rows, all reading 09:12:00:
    row 5  activity B
    row 6  activity D
    row 7  activity A
Transition statistics say A->B is frequent with p25 = 90 s, B->D is frequent
with p25 = 240 s, and D->A is rare.

Correct output: infer the order A, B, D. Row 7 (A) is the block's first event,
so it KEEPS 09:12:00. Row 5 (B) becomes 09:13:30 and row 6 (D) becomes
09:17:30. The span, 330 s, is well inside the 108-minute room to the right
anchor, all three values are distinct and increasing, and two of the three rows
changed.

Wrong outputs, all of which have been observed in practice: leaving all three at
09:12:00; rebuilding from 09:00:00 so that no row keeps the observed value;
spacing them evenly at 09:12, 09:47, 10:22 because a span profile said a
three-event run "should" last an hour.
"""

SELFCHECK_C2 = r"""
=====================================================================
SELF-CHECK BEFORE WRITING THE OUTPUT
=====================================================================
Your program must verify its own output. When a check fails, REPAIR THE
CANDIDATE -- clamp it, re-estimate it -- and never fall back to the observed
value.

  - no repaired row sits outside its anchors
  - the repaired pair is in non-decreasing order with respect to each other and
    to both anchors
  - every corrupted row differs from its input value
  - for a pair you classified as EXCHANGED, the two output values are exactly
    the two input values, with the rows swapped -- no new value was invented
  - for a row you classified as DISPLACED, its partner's timestamp is unchanged

Print one line to stdout with the count of segments classified as exchanged, the
count classified as displaced, and rows still equal to their input.

=====================================================================
WORKED EXAMPLE
=====================================================================
Anchors: E2 at 09:00:00 (clean), E7 at 09:30:00 (clean).

Case A -- exchanged. Rows 4 and 5 read 09:20:00 and 09:05:00. Swapping them
gives 09:05:00 and 09:20:00, both inside the anchors, and both resulting gaps
sit near the median for their transitions. Output the swap: row 4 becomes
09:05:00 and row 5 becomes 09:20:00. No value was invented, and both rows
changed.

Case B -- displaced. Rows 4 and 5 read 09:25:00 and 09:02:00. Swapping would
put 09:25:00 on row 5, whose transition from row 4 normally takes 60 s, not
23 minutes, so the swap does not fit the duration profiles. Row 4 is correct and
must not be touched. Estimate row 5 from row 4 plus the p25 of that transition:
09:26:00. Only row 5 changed.

Wrong outputs: sorting every pair unconditionally, which destroys row 4 in
case B; leaving a row on its observed value because neither branch felt
certain.
"""

SELFCHECK_C3 = r"""
=====================================================================
SELF-CHECK BEFORE WRITING THE OUTPUT
=====================================================================
Your program must verify its own output. When a check fails, REPAIR THE
CANDIDATE -- clamp it, re-estimate it -- and never fall back to the observed
value.

  - every corrupted row has a parseable timestamp; none is left blank
  - no repaired row sits outside its anchors
  - for a row whose date survived, the repaired value falls on THAT SAME
    calendar date
  - every corrupted row differs from its input value

Print one line to stdout with the count of blank rows filled, the count of
date-only rows given a time of day, and rows still equal to their input.

=====================================================================
WORKED EXAMPLE
=====================================================================
Anchors: E3 at 2015-03-04 09:00:00 (clean), E6 at 2015-03-04 17:00:00 (clean).

Case A -- date-only. Row 4 reads 2015-03-04 00:00:00: the date survived, the
time of day did not. Keep 2015-03-04 and infer only the time. The transition
from E3 to row 4 has p25 = 1,800 s, so the repaired value is
2015-03-04 09:30:00 -- same date, inside the anchors.

Case B -- blank. Row 5 has no timestamp at all. Both anchors exist, so estimate
from the left anchor plus the transition statistic, and if that overshoots the
right anchor, place it between the two anchors rather than one second before the
right one. Row 5 becomes 2015-03-04 11:15:00.

Wrong outputs: moving case A to 2015-03-05 because the estimated duration ran
past midnight; pinning case B at 16:59:59; leaving a blank row blank, which is
not a repair.
"""

CATEGORY_LABEL = {
    "C1": "C1_ORDERING_AMBIGUITY",
    "C2": "C2_TEMPORAL_INCONSISTENCY",
    "C3": "C3_MISSING_TEMPORAL_INFORMATION",
}

_BODY = {"C1": C1_BODY, "C2": C2_BODY, "C3": C3_BODY}
_SELF = {"C1": SELFCHECK_C1, "C2": SELFCHECK_C2, "C3": SELFCHECK_C3}
_BUDGET = {"C1": BUDGET_C1, "C2": BUDGET_OTHER, "C3": BUDGET_OTHER}


def codegen_prompt(cat):
    """The M2/M3 prompt for one category. Keeps <<CONTEXT>> for substitution."""
    return (HEADER % CATEGORY_LABEL[cat]
            + PREAMBLE + _BODY[cat]
            + CONSTRAINTS + _BUDGET[cat] + DURATION_STATS
            + CONTEXT_BLOCK + _SELF[cat] + OUTPUT_FORMAT)


# =====================================================================
# M4 - direct repair, same split applied to DIRECT_PROMPT
# =====================================================================

DIRECT_HEADER = r"""
You are an expert in process mining and temporal event-log timestamp repair.

You repair corrupted event-log segments directly. You are NOT writing code.

Every segment you are given has error_category %s. No other
error category appears in this log.

=====================================================================
WHY YOU ANSWER IN OFFSETS
=====================================================================
You never write an absolute timestamp. For each corrupted row you return an
integer POSITION in the segment's inferred order and an OFFSET IN SECONDS from
that segment's reference_timestamp, which is given to you. The caller builds the
timestamp. This removes every failure mode that comes from writing dates by
hand: wrong day, wrong year, lost milliseconds, a value from another case, an
elapsed-seconds number mistaken for an absolute time.

Return ONLY valid JSON, no markdown, no prose:

{"repairs": [{"segment_id": 3, "row_id": 123, "position": 1,
              "offset_seconds": 0, "reason": "short reason"}]}

- exactly one object for every row_id in corrupted_rows of every segment
- position is 1..k inside its segment, each value used once
- offset_seconds is a non-negative integer, measured from reference_timestamp
- offsets must be strictly increasing with position, and all distinct
- offset_seconds must not exceed max_offset_seconds for that segment

=====================================================================
WHAT THE CORRUPTED VALUES ARE
=====================================================================
"""

DIRECT_C1 = r"""
C1_ORDERING_AMBIGUITY
  k consecutive events collapsed onto ONE timestamp. That timestamp is not
  invented: it is the block's own earliest value, so the event you rank first is
  already correct. reference_timestamp is that value, so the row you give
  position 1 must get offset_seconds = 0, and the others get positive offsets.
  The rows are NOT in execution order -- they were shuffled when the block
  was recorded. Infer the order from the transition statistics supplied
  with the segment, and do not assume position 1 is the first row listed.
  If your positions come out as 1,2,3,... in the order the rows are
  given, you have not ordered anything.
  Never leave the block collapsed: every offset in a block must differ.
  Never space a block evenly.
"""

DIRECT_C2 = r"""
C2_TEMPORAL_INCONSISTENCY
  Either two adjacent values were EXCHANGED -- both rows wrong, and swapping
  them back is exact -- or one row was DISPLACED backwards while its partner
  stayed correct. Check whether swapping produces durations consistent with the
  segment's transition statistics; if it does not, estimate the displaced value
  from the anchors instead. reference_timestamp is the left anchor.
"""

DIRECT_C3 = r"""
C3_MISSING_TEMPORAL_INFORMATION
  Blank, or date-only. When the date survives, reference_timestamp is midnight
  of that date and max_offset_seconds is 86399: keep the day and infer only the
  time of day. When the value is blank, reference_timestamp is the left anchor.
"""

DIRECT_TAIL = r"""
=====================================================================
HOW TO CHOOSE AN OFFSET
=====================================================================
Each segment carries transition_stats for the activity pairs that actually occur
in it: n, p10, p25, median, p75, p90 and share_under_60s. These gaps are
bimodal. A high share_under_60s means the transition normally happens inside one
working session, so use p10 or p25. A low one means it normally waits, so the
median is the right scale. A per-pair median applied inside a C1 block is
usually far too large, because the block collapsed precisely because those
events happened together.

Sanity: the sum of the gaps you assign inside a segment must fit in
max_offset_seconds. If your first choice does not fit, scale the gaps down
proportionally rather than truncating the last one.
"""

_DIRECT_BODY = {"C1": DIRECT_C1, "C2": DIRECT_C2, "C3": DIRECT_C3}


def direct_prompt(cat):
    return (DIRECT_HEADER % CATEGORY_LABEL[cat]
            + _DIRECT_BODY[cat] + DIRECT_TAIL)


# =====================================================================
# Addendum for sub-second logs (bpic12).
#
# Three of four bpic12_C3_30 cells produced no data at all, not because the
# repair was poor but because the generated program raised
#   ValueError("CLEAN row timestamp changed")
# from its own self-check and exited with a non-zero status. bpic12 is the
# only log with millisecond precision (%Y-%m-%d %H:%M:%S.%f); re-formatting a
# parsed timestamp changes a CLEAN row's string by a trailing digit, the
# program reads that as a constraint violation, and aborts.
#
# This addendum does not change the repair strategy. It removes a crash: it
# tells the program to copy untouched rows verbatim rather than re-render
# them, and to repair rather than abort when a self-check fails -- which is
# what the main prompt already asks for in prose.
# =====================================================================

PRECISION_ADDENDUM = r"""

=====================================================================
TIMESTAMP FORMATTING AND FAILURE HANDLING -- READ THIS LAST
=====================================================================
This log records timestamps with SUB-SECOND precision.

1. Do NOT re-render timestamps you did not repair. Carry the original string
   through unchanged, exactly as it appeared in the input file. Parse a
   timestamp when you need to compute with it, but write back the ORIGINAL
   STRING for every row you did not repair, including every CLEAN row. A
   round-trip through a datetime and back is not safe here: it can add or drop
   a trailing fractional digit and silently rewrite rows you never intended to
   touch.

2. For rows you DID repair, emit the same format the input uses, keeping the
   fractional seconds.

3. Your program MUST NOT raise, exit non-zero, or call sys.exit() when a
   self-check fails. A program that aborts produces no output and is scored as
   a total failure -- strictly worse than an imperfect repair. When a check
   fails: fix the offending rows (clamp them, re-space them, restore the
   original string) and continue. Report what you fixed on stdout. The only
   acceptable exit status is 0.
"""
