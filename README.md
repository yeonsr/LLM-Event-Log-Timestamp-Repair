# LLM-Generated Programs for Event Log Timestamp Repair

Code, prompts, synthesized programs and complete result tables for the paper.

The paper reports the part of these results its argument rests on. This
repository holds all of it: every metric for every method, log, error category
and pollution level, together with the programs the LLM arms produced and the
generation logs behind them.

## Layout

```
src/        the implementation
  core.py            error injection, the baselines, the enforcement layer, the metrics
  llm.py             the synthesis pipeline: prompting, gates G1-G5, the feedback loop
  baselines.py       DEPU, LSTM, MAXL, DISS
  stat.py            STAT
  prompts_percat.py  per-category prompt templates
  prompts_v2.py      base prompt templates

run/        entry points, one per family of methods
  run_baselines.py   DEPU, LSTM, MAXL, DISS over every dataset
  run_stat.py        B0 and STAT
  run_lstm.py        the LSTM baseline on its own
  run_llm.py         LLME, LLMU and LLMD

results/    the complete results
  tables/            every result as a readable table -- start at results/tables/README.md
  *.csv              the same numbers, machine-readable
  judgements/        every judge verdict with the line of code it cites

analysis/   the scripts that produce everything in results/
  make_result_tables.py   renders results/tables/ from the CSVs
  strategy_judge.py       runs a judge over programs/
  judge_compare.py        agreement between two judges, and evidence grounding
  strategy_regression.py  the strategy vector against log10(MedAE / B0)
  strategy_scan.py        a static pattern scan, kept as a cross-check
  make_latex_tables.py    the two tables of the paper, in LaTeX

programs/   148 synthesized repair programs, one per reported cell and trial
data/       how to obtain or regenerate the event logs (not shipped; see data/README.md)
```

Set `TSREPAIR_ROOT` to the checkout if you run the scripts from elsewhere; it
defaults to the current directory.

## Method labels

The CSVs use the internal names; the paper uses short labels.

| CSV | Paper | What it is |
|---|---|---|
| `M0_null` | B0 | returns the log untouched |
| `B1_conforti` | DEPU | dependency ordering with uniform redistribution |
| `B2_lstm` | LSTM | LSTM duration prediction |
| `B3_bridge` | MAXL | maximum-likelihood bridge |
| `B4_sampling` | DISS | distributional sampling |
| `M1` | STAT | per-pair median durations plus an ordering search |
| `M4_direct_chunk50` | LLME | direct estimation, no program synthesis |
| `M2_codegen` | LLMU | one program synthesized per log, applied unchanged |
| `M3_codegen_pm4py` | LLMD | a program synthesized per dataset, with PM4Py process statistics |

## The result tables

`results/tables/` holds all of it laid out for reading; the CSVs are the same
data for loading.

| File | Rows |
|---|---|
| `results/results_all_logs.csv` | the baselines and LLME: 17 metrics for every dataset and scope |
| `results/b0_and_stat.csv` | B0 and STAT, scored by the same pipeline |
| `results/llm_per_synthesis.csv` | LLMU and LLMD, one row per individual synthesis |
| `results/generation_summary.csv` | the 270 generation logs condensed to one row per candidate |
| `results/strategy_judgements_gpt-5.csv` | the strategies each program implements, as judged by GPT-5 |
| `results/strategy_judgements_gpt-4o.csv` | the same, by a second judge, for the agreement check |
| `results/judge_agreement.csv` | per-strategy agreement between the two judges |
| `results/maxl_solve_gpt-5.csv` | a second, narrower judging pass on two-sided anchor solving |
| `results/strategy_regression_*.csv` | the strategy vector regressed on log10(MedAE / B0) |
| `results/strategy_features.csv` | a static pattern scan of the same programs, kept as a cross-check |
| `results/judgements/<model>/*.json` | every verdict with the line of code it cites |
| `results/judgements_maxl/<model>/*.json` | the same for the narrow anchor question |

The paper reports the subset with `scope == "ALL_POLLUTED"` at the 30 % pollution
level. The files also contain material the paper does not discuss:

- **10 % pollution** (`*_10` datasets) alongside the reported 30 %.
- **The exchange variant of C2** (`C2SWAP_*`). The paper reports the
  displacement variant only, because an exchange of two adjacent timestamps is
  exactly invertible by sorting and several baselines score a median error of
  0 s on it, which separates the methods not at all.
- **Per-defect scopes** `C3_BLANK` and `C3_DATEONLY`, which split C3 into its
  two injected modes.
- **`NAT_C1` and `COARSE_C4`** are *not* further injected categories. They label
  defects already present in the raw logs — naturally collapsed blocks, for
  which no ground truth exists, and date-only fields — and are excluded from the
  error statistics for that reason.

`analysis/make_result_tables.py` regenerates everything under
`results/tables/`. `analysis/strategy_judge.py --model <name>` re-runs a judge
over `programs/` (verdicts are cached per model, so a repeat run costs
nothing), `analysis/judge_compare.py gpt-4o gpt-5` reports their agreement and
how much of each judge's cited evidence is verbatim in the source, and
`analysis/strategy_regression.py <model>` refits the regression. The
pattern-based `analysis/strategy_scan.py` is kept as an independent
cross-check; it under-detects any strategy written as arithmetic rather than
as a recognisable call, so the judged labels are the ones reported.

## Two questions about the right anchor

The strategy list asks whether a program uses the right anchor as a constraint
at all. That reads broadly: a program that lays the block out from the left
anchor and then rescales it proportionally to fit the interval satisfies it,
and so does STAT. Asked that way, 98 % of programs qualify.

`analysis/judge_maxl_solve.py` asks the narrow question that the comparison
with MAXL actually turns on -- solving for the gaps under an equality
constraint with a per-transition duration model, or fitting the transition
into the right anchor's own activity -- and excludes clipping, clamping and
proportional rescaling explicitly. Asked that way, 28 of 148 programs qualify,
none of them on C1 and none of them from LLMU.

Both figures are published because the gap between them is the finding: the
broad property is near-universal and tells you little, while the mechanism
that explains MAXL's advantage is absent exactly where MAXL wins.

## Notes on the released results

These are stated here so the files can be read correctly against the paper.

- **Syntheses per cell.** The design is five independent syntheses per
  log × category × arm. Seven of the thirty cells hold four;
  `results/llm_per_synthesis.csv` shows exactly which.
- **The model.** All reported LLM runs used `gpt-4o`. `run/run_llm.py --model`
  sets it at run time and overrides the default in `src/core.py`, so the
  authoritative record is the `model` column of
  `results/llm_per_synthesis.csv`, which carries `gpt-4o` on all 286 rows.
- **LLME on C3.** On BPIC12 and BPIC15 the direct-estimation arm returned the
  C3 input unchanged, reproducing B0 on every metric at a coverage of about
  0.50. Its error statistics on those two cells record a refusal to repair, not
  an accurate one. `coverage` identifies such cases.
- **B0 and coverage.** B0 leaves missing timestamps missing, so on C3 it is
  scored on roughly half the damaged rows and its error columns are not on the
  same footing as the others.

## Reproducing

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...          # only needed for the LLM arms
python run/run_stat.py             # B0, STAT
python run/run_baselines.py        # DEPU, LSTM, MAXL, DISS
python run/run_llm.py --model <name>
python analysis/make_result_tables.py
```

The baselines are deterministic apart from LSTM and DISS, which are seeded (42).
The LLM arms are not reproducible exactly: each run synthesizes a new program.

## Data

The event logs are not shipped. Three are public and one script prepares them;
the two simulated logs and the injection procedure are described in
`data/README.md`. Injection is seeded (42), so the damaged datasets are
reproducible from the clean logs.

