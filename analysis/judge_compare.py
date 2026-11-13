# -*- coding: utf-8 -*-
"""Compare two judges over the same programs.

The programs were written by one model and are judged by another, so the
obvious question is how much the verdict depends on which judge is asked.
This reports, per strategy, what each judge found and how often they agree,
and checks how much of each judge's cited evidence occurs verbatim in the
program it was judging.

    python paper/judge_compare.py gpt-4o gpt-5
"""
import os, io, sys, json, glob, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
KEY = ['log', 'category', 'method', 'trial']

a_name = sys.argv[1] if len(sys.argv) > 2 else 'gpt-4o'
b_name = sys.argv[2] if len(sys.argv) > 2 else 'gpt-5'


def load(m):
    p = os.path.join(R, 'results', 'strategy_judgements_%s.csv' % m)
    return pd.read_csv(p)


def grounding(m):
    """share of cited evidence lines that occur verbatim in the source"""
    ok = tot = 0
    for f in sorted(glob.glob(os.path.join(R, 'results', 'judgements', m, '*.json'))):
        name = os.path.basename(f)[:-5]
        sp = os.path.join(R, 'programs', name + '.py')
        if not os.path.exists(sp):
            continue
        src = io.open(sp, encoding='utf-8', errors='replace').read()
        v = json.load(io.open(f, encoding='utf-8'))
        for k, e in v.get('_evidence', {}).items():
            if not v.get(k):
                continue
            e = (e or '').strip()
            tot += 1
            if e and (e in src or e.split('\n')[0].strip() in src):
                ok += 1
    return ok, tot


A, B = load(a_name), load(b_name)
KEYS = [c for c in A.columns if c not in KEY + ['judge_model']]
m = A.merge(B, on=KEY, suffixes=('_a', '_b'))
print('programs judged by both: %d\n' % len(m))

print('%-12s %9s %9s %9s   %s' % ('strategy', a_name, b_name, 'agree', 'split'))
rows = []
for k in KEYS:
    x, y = m[k + '_a'], m[k + '_b']
    ag = float((x == y).mean())
    rows.append((k, x.mean(), y.mean(), ag,
                 int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())))
for k, pa, pb, ag, oa, ob in sorted(rows, key=lambda r: r[3]):
    print('%-12s %8.0f%% %8.0f%% %8.0f%%   %s only %2d / %s only %2d'
          % (k, 100 * pa, 100 * pb, 100 * ag, a_name, oa, b_name, ob))

mean_ag = sum(r[3] for r in rows) / len(rows)
print('\nmean agreement across %d strategies: %.0f%%' % (len(rows), 100 * mean_ag))
firm = [r[0] for r in rows if r[3] >= 0.85]
soft = [r[0] for r in rows if r[3] < 0.70]
print('agree at 85%% or better : %s' % (', '.join(firm) if firm else 'none'))
print('below 70%%               : %s' % (', '.join(soft) if soft else 'none'))

print('\nevidence grounded verbatim in the judged source:')
for name in (a_name, b_name):
    ok, tot = grounding(name)
    print('  %-8s %d of %d cited lines (%.0f%%)'
          % (name, ok, tot, 100.0 * ok / tot if tot else 0))

out = os.path.join(R, 'results', 'judge_agreement.csv')
pd.DataFrame(rows, columns=['strategy', a_name, b_name, 'agreement',
                            a_name + '_only', b_name + '_only']).to_csv(out, index=False)
print('\nwrote %s' % os.path.relpath(out, R))
