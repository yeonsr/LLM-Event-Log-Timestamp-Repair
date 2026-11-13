# -*- coding: utf-8 -*-
"""LLM-as-a-judge over the synthesized repair programs.

Each program in programs/ is shown to the model together with a strategy list
fixed in advance, and the model reports, for every strategy, whether the code
implements it and the line that shows it. The list is fixed so that every run
is scored on the same axes; a strategy may be present alongside others, so the
labels are not mutually exclusive.

The judge is the same model used for the repair experiments. Judging is
resumable: a program whose verdict is already on disk is not sent again.

    python paper/strategy_judge.py              # judge everything not yet judged
    python paper/strategy_judge.py --limit 5    # a small trial first
"""
import os, io, re, csv, json, glob, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PROGRAMS = os.path.join(ROOT, 'programs')
# The judge is a different model from the one that wrote the programs, so a
# program is never scored by its own author. Verdicts are cached per model:
# the cache key includes the model name, because a cache keyed on the program
# alone would silently return another model's verdict after a model change.
DEFAULT_MODEL = 'gpt-5'
CACHE_ROOT = os.path.join(ROOT, 'results', 'judgements')
OUT_TMPL = os.path.join(ROOT, 'results', 'strategy_judgements_%s.csv')

STRATEGIES = [
    ('perm',       'enumerates the orderings of a block exhaustively, '
                   'e.g. by generating all permutations'),
    ('beam',       'keeps a bounded set of candidate orderings instead of all '
                   'of them, e.g. a beam or top-k search'),
    ('transprob',  'scores a candidate ordering using directly-follows counts '
                   'or transition probabilities between activity labels'),
    ('rightanch',  'reads the right anchor, that is the first clean event '
                   'after the damaged segment, at all'),
    ('r_ceiling',  'uses the right anchor ONLY as a bound: the block is laid '
                   'out forward from the left anchor and then clipped, '
                   'clamped or shrunk so it fits before the right anchor'),
    ('r_evidence', 'uses the right anchor as a CONSTRAINT that shapes the '
                   'estimate: the gaps are solved or distributed so that they '
                   'sum to the span between the two anchors, or the transition '
                   'into the right anchor is itself modelled'),
    ('median',     'uses a median duration statistic for spacing'),
    ('mean',       'uses a mean duration statistic for spacing'),
    ('lognorm',    'works on a log scale or fits a lognormal-like duration '
                   'model'),
    ('uniform',    'spreads the events of a block uniformly across the '
                   'available interval'),
    ('rescale',    'rescales the gaps when their sum overruns the room between '
                   'the anchors'),
    ('firstts',    'treats the surviving collapsed timestamp as belonging to '
                   'the FIRST event of the block'),
    ('monotone',   'explicitly enforces that timestamps increase within a case'),
    ('fallback',   'carries an explicit fallback path for inputs its main '
                   'logic does not handle'),
]
KEYS = [k for k, _ in STRATEGIES]

SYSTEM = (
    "You are analysing Python programs that repair timestamps in process "
    "event logs. You report only what the code demonstrably does. You never "
    "infer a strategy from a comment, a docstring or a variable name alone: "
    "the executable code must perform it. If the evidence is not there, the "
    "strategy is absent."
)

SPEC = '\n'.join('- %s: %s' % (k, d) for k, d in STRATEGIES)

TEMPLATE = """Here is a repair program.

```python
{src}
```

For each strategy below, say whether this program implements it.

{spec}

Note that `r_ceiling` and `r_evidence` are different uses of the same anchor and
both may be present or absent independently.

Reply with JSON only, no prose, in exactly this shape:

{{"perm": {{"present": true, "evidence": "<the single most telling line of code, verbatim, or empty>"}},
 ... one entry for every strategy key ...}}
"""

_lock = threading.Lock()
_done = 0


def judge_one(path, client, total, model, cache_dir):
    global _done
    name = os.path.basename(path)[:-3]
    cached = os.path.join(cache_dir, name + '.json')
    if os.path.exists(cached):
        with _lock:
            _done += 1
        return name, json.load(io.open(cached, encoding='utf-8')), True

    src = io.open(path, encoding='utf-8', errors='replace').read()
    prompt = TEMPLATE.format(src=src, spec=SPEC)
    last = None
    for attempt in (1, 2, 3):
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": prompt}],
            )
            txt = resp.output_text.strip()
            m = re.search(r'\{.*\}', txt, re.S)
            verdict = json.loads(m.group(0) if m else txt)
            verdict = {k: bool(verdict.get(k, {}).get('present', False))
                       if isinstance(verdict.get(k), dict)
                       else bool(verdict.get(k, False))
                       for k in KEYS} | {
                '_evidence': {k: (verdict[k].get('evidence', '')[:200]
                                  if isinstance(verdict.get(k), dict) else '')
                              for k in KEYS}}
            json.dump(verdict, io.open(cached, 'w', encoding='utf-8'))
            with _lock:
                _done += 1
                print('  [%3d/%d] %s' % (_done, total, name), flush=True)
            return name, verdict, False
        except Exception as e:
            last = e
            time.sleep(2 * attempt)
    with _lock:
        _done += 1
        print('  [%3d/%d] %s  FAILED: %s' % (_done, total, name, last), flush=True)
    return name, None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    a = ap.parse_args()
    model = a.model
    cache_dir = os.path.join(CACHE_ROOT, model.replace('/', '_'))
    out_csv = OUT_TMPL % model.replace('/', '_')

    from openai import OpenAI
    client = OpenAI()
    os.makedirs(cache_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(PROGRAMS, '*.py')))
    if a.limit:
        paths = paths[:a.limit]
    print('judging %d programs with %s, %d workers' % (len(paths), model, a.workers))

    rows, cached_n, failed = [], 0, []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(judge_one, p, client, len(paths), model, cache_dir)
                for p in paths]
        for f in as_completed(futs):
            name, verdict, was_cached = f.result()
            cached_n += was_cached
            if verdict is None:
                failed.append(name)
                continue
            m = re.match(r'^(\w+?)_(C\d)_\d+_(\w+?)_t(\d+)$', name)
            if not m:
                continue
            r = dict(log=m.group(1), category=m.group(2),
                     method=m.group(3), trial=m.group(4), judge_model=model)
            r.update({k: int(verdict[k]) for k in KEYS})
            rows.append(r)

    rows.sort(key=lambda r: (r['log'], r['category'], r['method'], r['trial']))
    with io.open(out_csv, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(
            fh, fieldnames=['log', 'category', 'method', 'trial', 'judge_model'] + KEYS)
        w.writeheader()
        w.writerows(rows)

    print('\nwrote %s  (%d programs, %d served from cache, %d failed)'
          % (os.path.relpath(out_csv, ROOT), len(rows), cached_n, len(failed)))
    if failed:
        print('  failed:', ', '.join(failed[:8]))
    if rows:
        print('\n%-12s %6s' % ('strategy', 'share'))
        for k in KEYS:
            print('%-12s %5.0f%%' % (k, 100.0 * sum(r[k] for r in rows) / len(rows)))


if __name__ == '__main__':
    main()
