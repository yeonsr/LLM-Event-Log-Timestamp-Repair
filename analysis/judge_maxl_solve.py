# -*- coding: utf-8 -*-
"""A second, narrower judging pass on one distinction only.

The general strategy list asks whether a program uses the right anchor as a
constraint. That question is too coarse for Section 6.2, which turns on the
specific mechanism that separates MAXL from STAT: solving for the gaps under
an equality constraint with a per-transition duration model, rather than
laying the block out forward and then rescaling it to fit. Proportional
rescaling satisfies the loose reading of "constrained by the span" while being
exactly the behaviour STAT also has, so it has to be excluded explicitly.

    python analysis/judge_maxl_solve.py [--model gpt-5] [--limit N]
"""
import os, io, re, csv, json, glob, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PROGRAMS = os.path.join(ROOT, 'programs')
DEFAULT_MODEL = 'gpt-5'

SYSTEM = (
    "You analyse Python programs that repair timestamps in process event logs. "
    "You answer only from executable code, never from comments, docstrings or "
    "variable names. When the evidence is not in the code, the answer is no."
)

QUESTION = """Here is a repair program.

```python
{src}
```

Answer one question about how it uses the RIGHT anchor, that is the first
clean event after the damaged segment.

Does the program SOLVE for the gaps jointly against the two anchors, using a
per-transition duration model? That means one of:
  - it optimises or solves for durations subject to their sum equalling the
    span between the anchors, weighting transitions by their own fitted
    parameters (a likelihood, a variance, a per-pair sigma), or
  - it fits the transition INTO the right anchor's own activity label from
    that activity pair's statistics, so the right anchor participates in the
    modelled chain.

It is NOT this if the program merely:
  - lays the block out forward from the left anchor and then clips, clamps,
    truncates or proportionally rescales the result so it ends before the
    right anchor,
  - divides the interval uniformly or in proportion to sampled or median
    gaps without any per-transition weighting,
  - reads the right anchor only to bound or validate the output.

Those are the behaviours STAT also has; only the first description counts.

Reply with JSON only:
{{"solves": true or false, "evidence": "<the single most telling line of code, verbatim, or empty>", "why": "<one short sentence>"}}
"""

_lock = threading.Lock()
_n = 0


def judge(path, client, total, model, cache):
    global _n
    name = os.path.basename(path)[:-3]
    cf = os.path.join(cache, name + '.json')
    if os.path.exists(cf):
        with _lock:
            _n += 1
        return name, json.load(io.open(cf, encoding='utf-8'))
    src = io.open(path, encoding='utf-8', errors='replace').read()
    for attempt in (1, 2, 3):
        try:
            r = client.responses.create(
                model=model,
                input=[{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": QUESTION.format(src=src)}])
            t = r.output_text.strip()
            m = re.search(r'\{.*\}', t, re.S)
            v = json.loads(m.group(0) if m else t)
            v = {'solves': bool(v.get('solves')),
                 'evidence': str(v.get('evidence', ''))[:200],
                 'why': str(v.get('why', ''))[:200]}
            json.dump(v, io.open(cf, 'w', encoding='utf-8'))
            with _lock:
                _n += 1
                print('  [%3d/%d] %-34s %s' % (_n, total, name,
                                               'SOLVES' if v['solves'] else '-'),
                      flush=True)
            return name, v
        except Exception as e:
            last = e
            time.sleep(2 * attempt)
    with _lock:
        _n += 1
    print('  [%3d/%d] %-34s FAILED %s' % (_n, total, name, last), flush=True)
    return name, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI()
    cache = os.path.join(ROOT, 'results', 'judgements_maxl', a.model)
    os.makedirs(cache, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(PROGRAMS, '*.py')))
    if a.limit:
        paths = paths[:a.limit]
    print('narrow anchor question over %d programs with %s' % (len(paths), a.model))
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(judge, p, client, len(paths), a.model, cache)
                for p in paths]
        for f in as_completed(futs):
            name, v = f.result()
            if v is None:
                continue
            m = re.match(r'^(\w+?)_(C\d)_\d+_(\w+?)_t(\d+)$', name)
            if m:
                rows.append(dict(log=m.group(1), category=m.group(2),
                                 method=m.group(3), trial=m.group(4),
                                 judge_model=a.model,
                                 maxl_solve=int(v['solves']),
                                 evidence=v['evidence'], why=v['why']))
    rows.sort(key=lambda r: (r['log'], r['category'], r['method'], r['trial']))
    out = os.path.join(ROOT, 'results', 'maxl_solve_%s.csv' % a.model)
    with io.open(out, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    s = sum(r['maxl_solve'] for r in rows)
    print('\n%d of %d programs solve jointly against both anchors  (%.0f%%)'
          % (s, n, 100.0 * s / n))
    for c in ('C1', 'C2', 'C3'):
        sub = [r for r in rows if r['category'] == c]
        if sub:
            print('  %s: %.0f%%' % (c, 100.0 * sum(r['maxl_solve'] for r in sub)
                                    / len(sub)))
    print('wrote %s' % os.path.relpath(out, ROOT))


if __name__ == '__main__':
    main()
