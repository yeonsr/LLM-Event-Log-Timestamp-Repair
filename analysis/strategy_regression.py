# -*- coding: utf-8 -*-
"""Does any strategy predict repair quality?

Regresses the judge's one-hot strategy vector on

    y = log10( MedAE(run) / MedAE(B0 on the same log and category) )

so that y < 0 means the run beat leaving the log untouched. One row per
synthesized program. Strategies that are constant across the runs carry no
variance and are reported separately rather than fitted.

    python paper/strategy_regression.py
"""
import os, io, sys, csv, numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
ARM = {'LLMU': 'M2_codegen', 'LLMD': 'M3_codegen_pm4py'}

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gpt-5'
j = pd.read_csv(os.path.join(R, 'results',
                             'strategy_judgements_%s.csv' % MODEL))
s = pd.read_csv(os.path.join(R, 'results', 'llm_per_synthesis.csv'))
b = pd.read_csv(os.path.join(R, 'results', 'b0_and_stat.csv'))
s = s[s.scope.eq('ALL_POLLUTED')]
b = b[b.scope.eq('ALL_POLLUTED') & b.method.eq('M0_null')]

KEYS = [c for c in j.columns
        if c not in ('log', 'category', 'method', 'trial', 'judge_model')]

j['method_csv'] = j.method.map(ARM)
m = j.merge(s[['log', 'category', 'method', 'trial', 'MedAE_rep']],
            left_on=['log', 'category', 'method_csv', 'trial'],
            right_on=['log', 'category', 'method', 'trial'],
            how='inner', suffixes=('', '_s'))
m = m.merge(b[['log', 'category', 'MedAE_rep']].rename(
    columns={'MedAE_rep': 'MedAE_b0'}), on=['log', 'category'], how='left')
m = m[(m.MedAE_rep > 0) & (m.MedAE_b0 > 0)].copy()
m['y'] = np.log10(m.MedAE_rep / m.MedAE_b0)

print('runs matched to a MedAE and a B0 reference: %d of %d judged'
      % (len(m), len(j)))
print('y = log10(MedAE / B0):  mean %.3f, sd %.3f, %d of %d runs beat B0\n'
      % (m.y.mean(), m.y.std(), int((m.y < 0).sum()), len(m)))

const = [k for k in KEYS if m[k].nunique() < 2]
use = [k for k in KEYS if k not in const]
if const:
    print('constant across every run, so not fitted:')
    for k in const:
        print('  %-12s always %s' % (k, 'present' if m[k].iloc[0] else 'absent'))
    print()

X = np.column_stack([np.ones(len(m))] + [m[k].to_numpy(float) for k in use])
y = m.y.to_numpy(float)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
n, p = X.shape
dof = n - p
sse = float(resid @ resid)
sst = float(((y - y.mean()) ** 2).sum())
r2 = 1 - sse / sst
adj = 1 - (1 - r2) * (n - 1) / dof
se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * sse / dof)
t = beta / se
pv = 2 * (1 - stats.t.cdf(np.abs(t), dof))

print('%-12s %9s %8s %8s %9s' % ('term', 'coef', 'se', 't', 'p'))
print('%-12s %9.3f %8.3f %8.2f %9.3f' % ('(intercept)', beta[0], se[0], t[0], pv[0]))
rows = sorted(zip(use, beta[1:], se[1:], t[1:], pv[1:]), key=lambda r: r[4])
for k, bb, ss, tt, pp in rows:
    star = ' *' if pp < 0.05 else ''
    print('%-12s %9.3f %8.3f %8.2f %9.3f%s' % (k, bb, ss, tt, pp, star))

sig = [k for k, _, _, _, pp in rows if pp < 0.05]
print('\nn = %d, %d fitted strategies, R2 = %.3f, adj. R2 = %.3f'
      % (n, len(use), r2, adj))
print('significant at 0.05: %s' % (', '.join(sig) if sig else 'none'))

with io.open(os.path.join(R, 'results', 'strategy_regression_%s.csv' % MODEL), 'w',
             encoding='utf-8', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['term', 'coef', 'se', 't', 'p'])
    w.writerow(['(intercept)', beta[0], se[0], t[0], pv[0]])
    for k, bb, ss, tt, pp in rows:
        w.writerow([k, bb, ss, tt, pp])
    w.writerow([])
    w.writerow(['n', n]); w.writerow(['r2', r2]); w.writerow(['adj_r2', adj])
    for k in const:
        w.writerow(['constant:' + k, int(m[k].iloc[0])])
print('wrote results/strategy_regression_%s.csv' % MODEL)
