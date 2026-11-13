# -*- coding: utf-8 -*-
"""Render every result in the study as Markdown tables under results/tables/.

The CSVs remain the machine-readable source; these are the same numbers laid
out so they can be read in the browser without opening anything.

    python analysis/make_result_tables.py
"""
import os, io, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
RES = os.path.join(R, 'results')
OUT = os.path.join(RES, 'tables')
os.makedirs(OUT, exist_ok=True)

LOGS = [('Credit', 'credit'), ('Pub', 'pub'), ('Hospital', 'hospital'),
        ('BPIC12', 'bpic12'), ('BPIC15', 'bpic15')]
NAME = {'M0_null': 'B0', 'B1_conforti': 'DEPU', 'B2_lstm': 'LSTM',
        'B3_bridge': 'MAXL', 'B4_sampling': 'DISS', 'M1': 'STAT',
        'M4_direct_chunk50': 'LLME', 'M2_codegen': 'LLMU',
        'M3_codegen_pm4py': 'LLMD'}
ORDER = ['B0', 'DEPU', 'LSTM', 'MAXL', 'DISS', 'STAT', 'LLME', 'LLMU', 'LLMD']

MET = [('coverage', 3), ('MedAE_rep', 1), ('MAE_rep', 0), ('P90_rep', 0),
       ('RMSE_rep', 0), ('improved_rate', 3), ('unchanged_rate', 3),
       ('regression_rate', 3), ('acc@1s', 3), ('acc@10s', 3), ('acc@60s', 3),
       ('acc@600s', 3), ('acc@3600s', 3), ('acc@86400s', 3),
       ('exact_match', 4), ('pairwise_acc', 4), ('position_acc', 4)]

o = pd.read_csv(os.path.join(RES, 'results_all_logs.csv'))
b = pd.read_csv(os.path.join(RES, 'b0_and_stat.csv'))
d = pd.read_csv(os.path.join(RES, 'llm_per_synthesis.csv'))


def fmt(v, nd):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    return '{:,.0f}'.format(v) if abs(v) >= 1e4 else '{:,.{}f}'.format(v, nd)


def md(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    out += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(out)


def num(frame, col):
    if col not in frame.columns or not len(frame):
        return np.nan
    v = pd.to_numeric(frame[col], errors='coerce').dropna()
    return float(v.median()) if len(v) else np.nan


index = []


def dataset_table(variant, scope='ALL_POLLUTED'):
    """One table per injected dataset: every method, every metric, every log."""
    cat = variant.split('_')[0]
    rows = []
    for lab, lg in LOGS:
        sub = o[o.dataset.eq('%s_%s' % (lg, variant)) & o.scope.eq(scope)]
        # Match the exact dataset. b0_and_stat.csv holds the 30 % datasets
        # only, so filtering on the category alone would put the 30 % numbers
        # into the 10 % tables; where it has no row the method is read from
        # results_all_logs like every other one.
        ds = '%s_%s' % (lg, variant)
        bb = b[b.dataset.eq(ds) & b.scope.eq(scope)]
        # llm_per_synthesis.csv covers the 30 % per-category runs only.
        dd = d[d.log.eq(lg) & d.category.eq(cat) & d.scope.eq(scope)] \
            if variant in ('C1_30', 'C2_30', 'C3_30') else d.iloc[0:0]
        for name in ORDER:
            key = [k for k, v in NAME.items() if v == name][0]
            if name in ('B0', 'STAT'):
                fr = bb[bb.method.eq(key)]
                if not len(fr):
                    fr = sub[sub.method.eq(key)]
            elif name in ('LLMU', 'LLMD'):
                fr = dd[dd.method.eq(key)]
                if not len(fr):
                    fr = sub[sub.method.eq(key)]
            else:
                fr = sub[sub.method.eq(key)]
            if not len(fr):
                continue
            rows.append([lab, name] + [fmt(num(fr, m), nd) for m, nd in MET])
        if rows and rows[-1][0] == lab:
            for r in rows[::-1]:
                if r[0] == lab:
                    continue
                break
    if not rows:
        return
    # blank the repeated log name so the blocks read clearly
    seen = set()
    for r in rows:
        if r[0] in seen:
            r[0] = ''
        else:
            seen.add(r[0])
    title = '%s, %s%% pollution' % (cat, variant.split('_')[-1]) \
        if '_' in variant else variant
    body = ['# %s' % title, '',
            'Scope `%s`. Every method against every metric, one block per log. '
            'For LLMU and LLMD each value is the median over that cell\'s '
            'syntheses; `results/tables/llm_per_synthesis.md` gives the '
            'individual runs.' % scope, '',
            md(['Log', 'Method'] + [m for m, _ in MET], rows), '']
    fn = '%s.md' % variant
    io.open(os.path.join(OUT, fn), 'w', encoding='utf-8').write('\n'.join(body))
    index.append((fn, title, len(rows)))


for variant in ['C1_30', 'C2_30', 'C3_30', 'C1_10', 'C2_10', 'C3_10',
                'C2SWAP_30', 'C2SWAP_10', 'MIX_30']:
    dataset_table(variant)

# ---- the C3 sub-scopes, which split the category into its two injected modes
for scope, lab in (('C3_BLANK', 'C3, blank timestamps only'),
                   ('C3_DATEONLY', 'C3, date-only timestamps only')):
    rows = []
    for llab, lg in LOGS:
        sub = o[o.dataset.eq('%s_C3_30' % lg) & o.scope.eq(scope)]
        for name in ORDER:
            key = [k for k, v in NAME.items() if v == name][0]
            fr = sub[sub.method.eq(key)]
            if len(fr):
                rows.append([llab, name] + [fmt(num(fr, m), nd) for m, nd in MET])
    if rows:
        seen = set()
        for r in rows:
            r[0] = '' if r[0] in seen else (seen.add(r[0]) or r[0])
        body = ['# %s' % lab, '', 'Scope `%s` on the 30%% datasets.' % scope, '',
                md(['Log', 'Method'] + [m for m, _ in MET], rows), '']
        fn = '%s.md' % scope
        io.open(os.path.join(OUT, fn), 'w', encoding='utf-8').write('\n'.join(body))
        index.append((fn, lab, len(rows)))

# ---- every individual synthesis
rows = []
for lab, lg in LOGS:
    for cat in ('C1', 'C2', 'C3'):
        for arm, key in (('LLMU', 'M2_codegen'), ('LLMD', 'M3_codegen_pm4py')):
            fr = d[d.log.eq(lg) & d.category.eq(cat) & d.method.eq(key)
                   & d.scope.eq('ALL_POLLUTED')].sort_values('trial')
            for _, r in fr.iterrows():
                rows.append([lab, cat, arm, str(int(r.trial))]
                            + [fmt(pd.to_numeric(r.get(m), errors='coerce'), nd)
                               for m, nd in MET])
body = ['# Individual syntheses', '',
        'One row per synthesized program: LLMU and LLMD, each trial scored '
        'separately. Seven of the thirty cells hold four syntheses rather '
        'than five.', '',
        md(['Log', 'Cat', 'Arm', 'Trial'] + [m for m, _ in MET], rows), '']
io.open(os.path.join(OUT, 'llm_per_synthesis.md'), 'w',
        encoding='utf-8').write('\n'.join(body))
index.append(('llm_per_synthesis.md', 'Individual syntheses', len(rows)))

# ---- the judges
parts = ['# Strategies implemented by the synthesized programs', '']
for jm in ('gpt-5', 'gpt-4o'):
    p = os.path.join(RES, 'strategy_judgements_%s.csv' % jm)
    if not os.path.exists(p):
        continue
    j = pd.read_csv(p)
    keys = [c for c in j.columns
            if c not in ('log', 'category', 'method', 'trial', 'judge_model')]
    parts += ['## Judge: %s' % jm, '',
              'Share of the 148 programs implementing each strategy, overall '
              'and by error category.', '']
    rows = [[k, fmt(100 * j[k].mean(), 0) + '%']
            + [fmt(100 * j[j.category.eq(c)][k].mean(), 0) + '%'
               for c in ('C1', 'C2', 'C3')] for k in keys]
    parts += [md(['Strategy', 'All', 'C1', 'C2', 'C3'], rows), '']
p = os.path.join(RES, 'judge_agreement.csv')
if os.path.exists(p):
    a = pd.read_csv(p)
    parts += ['## Agreement between the two judges', '',
              md(list(a.columns),
                 [[fmt(x, 3) if isinstance(x, float) else str(x) for x in r]
                  for r in a.values]), '']
io.open(os.path.join(OUT, 'strategies.md'), 'w',
        encoding='utf-8').write('\n'.join(parts))
index.append(('strategies.md', 'Strategies and judge agreement', 0))

# ---- the regressions
parts = ['# Does any strategy predict repair quality?', '',
         'The strategy vector regressed on `log10(MedAE / B0)`, once per '
         'judge. A negative coefficient means the strategy is associated with '
         'a lower error.', '']
for jm in ('gpt-5', 'gpt-4o'):
    p = os.path.join(RES, 'strategy_regression_%s.csv' % jm)
    if not os.path.exists(p):
        continue
    raw = [l.rstrip('\n').split(',') for l in io.open(p, encoding='utf-8')]
    tbl = [r for r in raw if len(r) == 5 and r[0] != 'term']
    tail = [r for r in raw if len(r) == 2]
    parts += ['## Judge: %s' % jm, '',
              md(['term', 'coef', 'se', 't', 'p'],
                 [[r[0]] + [fmt(float(x), 3) for x in r[1:]] for r in tbl]), '',
              '  '.join('**%s** %s' % (k, v) for k, v in tail), '']
io.open(os.path.join(OUT, 'regression.md'), 'w',
        encoding='utf-8').write('\n'.join(parts))
index.append(('regression.md', 'Strategy regression', 0))

# ---- index
lines = ['# Result tables', '',
         'Every number in the study, laid out for reading. The CSVs one level '
         'up are the same data in machine-readable form.', '',
         '| Table | What it holds | Rows |', '|---|---|---|']
for fn, title, n in index:
    lines.append('| [`%s`](%s) | %s | %s |' % (fn, fn, title, n or ''))
lines += ['', 'Method labels: B0 null, DEPU dependency ordering, LSTM duration '
          'prediction, MAXL maximum-likelihood bridge, DISS distributional '
          'sampling, STAT per-pair medians, LLME direct estimation, LLMU and '
          'LLMD the two synthesis arms.', '']
io.open(os.path.join(OUT, 'README.md'), 'w', encoding='utf-8').write('\n'.join(lines))

print('wrote %d tables to results/tables/' % (len(index) + 1))
for fn, title, n in index:
    print('  %-24s %-38s %s rows' % (fn, title, n or '-'))
