# Event logs

The logs are not shipped with the repository: the injected datasets alone run to
several hundred megabytes, and three of the five sources are public and should
be cited at their own DOIs rather than redistributed here.

## The five logs

| Log | Source |
|---|---|
| Hospital billing | <https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741> |
| BPIC 2012 | <http://dx.doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f> |
| BPIC 2015 | <http://dx.doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17> |
| Credit | simulated from a loan application process |
| Pub | simulated from a process of serving food in a pub |

Place the prepared CSVs under `data/raw/`. `src/core.py` holds a `LOGS`
dictionary giving, for each log, the expected filename, the column renaming, the
timestamp format, and the `leak` columns that must be dropped. Those `leak`
columns matter: several of them (a row index, a start timestamp, a variant id)
encode the true event order, and leaving any of them in hands the answer to the
repair methods.

## Regenerating the damaged datasets

Injection is deterministic given the clean log and `CFG["SEED"] = 42`, so the
datasets the paper uses are reproducible rather than redistributed:

- 30 % of cases damaged (also 10 %, generated but not reported in the paper)
- one error per damaged case, one category per dataset
- C1 blocks of length 2, 3 or 4, collapsed onto the block's first timestamp,
  with the rows of the block shuffled
- C2 in displacement mode (the exchange variant is emitted separately as
  `C2SWAP_*`)
- C3 in both modes: a wholly absent timestamp and one reduced to its date

The settings live in the `CFG` dictionary at the top of `src/core.py`.
