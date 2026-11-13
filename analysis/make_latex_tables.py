# -*- coding: utf-8 -*-
"""Two merged tables for the submission.

A. table_merged_cats.tex  -- replaces Tables 3, 4 and 5 with a single table.
   C1 is reported with MedAE, exact_match and position_acc; C2 and C3 with
   MedAE only. One row per log and method.

B. table_merged_best.tex  -- replaces Tables 9 and 10 with a single table.
   Keeps Table 9's comparison (each cell against the strongest baseline
   available for it) but runs it over every metric rather than MedAE alone,
   and keeps Table 10's per-category medians.
"""
import os, numpy as np, pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.dirname(os.path.abspath(__file__))
LOGS = [('Credit', 'credit'), ('Pub', 'pub'), ('Hospital', 'hospital'),
        ('BPIC12', 'bpic12'), ('BPIC15', 'bpic15')]
CATS = ['C1', 'C2', 'C3']
BS, SEP = chr(92), chr(92) * 2

o = pd.read_csv(R + '/results/results_all_logs.csv'); o = o[o.scope.eq('ALL_POLLUTED')]
m = pd.read_csv(R + '/results/b0_and_stat.csv')
m = m[m.scope.eq('ALL_POLLUTED')]
d = pd.read_csv(R + '/results/llm_per_synthesis.csv')
d = d[d.scope.eq('ALL_POLLUTED')]

SRC = {'B0': ('m', 'M0_null'), 'DEPU': ('o', 'B1_conforti'),
       'LSTM': ('o', 'B2_lstm'), 'MAXL': ('o', 'B3_bridge'),
       'DISS': ('o', 'B4_sampling'), 'STAT': ('m', 'M1'),
       'LLME': ('o', 'M4_direct_chunk50'),
       'LLMU': ('d', 'M2_codegen'), 'LLMD': ('d', 'M3_codegen_pm4py')}
MULTI = {'LLMU', 'LLMD'}
# DEPU, LSTM and DISS repair same-timestamp errors only; MAXL also has a C3
# bridge. Everything else reduces to STAT, which the table says rather than
# repeating the number.
FALLBACK = {('DEPU', 'C2'), ('LSTM', 'C2'), ('DISS', 'C2'), ('MAXL', 'C2'),
            ('DEPU', 'C3'), ('LSTM', 'C3'), ('DISS', 'C3')}


def series(name, cat, log, metric):
    src, key = SRC[name]
    if src == 'o':
        t = o[(o.dataset.eq('%s_%s_30' % (log, cat))) & (o.method.eq(key))]
    elif src == 'm':
        t = m[(m.log.eq(log)) & (m.category.eq(cat)) & (m.method.eq(key))]
    else:
        t = d[(d.log.eq(log)) & (d.category.eq(cat)) & (d.method.eq(key))]
    if not len(t) or metric not in t.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(t[metric], errors='coerce').dropna()


def val(name, cat, log, metric):
    s = series(name, cat, log, metric)
    return float(s.median()) if len(s) else np.nan


def fmt(v, nd):
    if not np.isfinite(v):
        return '--'
    return '{:,.0f}'.format(v) if abs(v) >= 1e4 else '{:,.{}f}'.format(v, nd)


# =====================================================================  A
COLS = [('C1', 'MedAE_rep', 'MedAE (s)', 1, -1),
        ('C1', 'exact_match', 'exact', 4, +1),
        ('C1', 'position_acc', 'position', 4, +1),
        ('C2', 'MedAE_rep', 'MedAE (s)', 1, -1),
        ('C3', 'MedAE_rep', 'MedAE (s)', 1, -1)]
ORDER = ['B0', 'DEPU', 'LSTM', 'MAXL', 'DISS', 'STAT', 'LLME', 'LLMU', 'LLMD']

best = {}
for lab, lg in LOGS:
    for cat, mk, _, _, dr in COLS:
        cand = []
        for n in ORDER:
            if (n, cat) in FALLBACK:
                continue
            if n == 'B0' and cat == 'C3':
                continue            # scored on about half the rows
            v = val(n, cat, lg, mk)
            if np.isfinite(v):
                cand.append((v, n))
        if cand:
            tgt = min(v for v, _ in cand) if dr < 0 else max(v for v, _ in cand)
            best[(lg, cat, mk)] = {n for v, n in cand if abs(v - tgt) < 1e-12}

L = [BS + 'begin{table}[t]' + BS + 'centering' + BS + 'small',
     BS + 'setlength{' + BS + 'tabcolsep}{4pt}',
     BS + 'caption{Repair accuracy for the three error categories in one table. '
     'C1 is reported with the median absolute error and the two ordering '
     'statistics, C2 and C3 with the median absolute error alone, since '
     'ordering is defined only for C1. For LLMU and LLMD the bracket is the '
     '[min, max] over that cell\'s syntheses and the central value their '
     'median. DEPU, LSTM and DISS repair same-timestamp errors only and MAXL '
     'has no C2 procedure, so those cells reduce to STAT and are marked '
     '$=$STAT. B0 is omitted on C3, where it leaves blanks blank and is scored '
     'on about half the damaged rows. The best value in each log and column is '
     'bold.}' + BS + 'label{tab:bycat}',
     BS + 'begin{tabular}{ll' + 'r' * len(COLS) + '}' + BS + 'toprule',
     '& & ' + BS + 'multicolumn{3}{c}{C1} & C2 & C3 ' + SEP
     + BS + 'cmidrule(lr){3-5}' + BS + 'cmidrule(lr){6-6}' + BS + 'cmidrule(lr){7-7}',
     'Log & Method & ' + ' & '.join(l for _, _, l, _, _ in COLS) + ' ' + SEP
     + BS + 'midrule']

for li, (lab, lg) in enumerate(LOGS):
    for j, name in enumerate(ORDER):
        cells = []
        for cat, mk, _, nd, dr in COLS:
            if (name, cat) in FALLBACK:
                cells.append('$=$STAT')
                continue
            if name == 'B0' and cat == 'C3':
                cells.append('--')
                continue
            s = fmt(val(name, cat, lg, mk), nd)
            if name in best.get((lg, cat, mk), set()):
                s = BS + 'textbf{' + s + '}'
            if mk == 'MedAE_rep' and name in MULTI:
                sr = series(name, cat, lg, mk)
                if len(sr) and (sr.max() - sr.min()) > 1e-9:
                    s += (' {' + BS + 'scriptsize[%s, %s]}'
                          % (fmt(float(sr.min()), nd), fmt(float(sr.max()), nd)))
            cells.append(s)
        L.append('%s & %s & %s %s' % (lab if j == 0 else '', name,
                                      ' & '.join(cells), SEP))
    if li < len(LOGS) - 1:
        L.append(BS + 'addlinespace[3pt]')
L += [BS + 'bottomrule' + BS + 'end{tabular}' + BS + 'end{table}']
open(os.path.join(OUT, 'table_results_by_category.tex'), 'w', encoding='utf-8').write('\n'.join(L))
print('table_results_by_category.tex : %d rows, %d metric columns'
      % (len(ORDER) * len(LOGS), len(COLS)))

# =====================================================================  B
# Table 9's baseline pool: B0, DEPU, LSTM, MAXL and STAT. DISS is excluded as
# an instrument rather than a method under test, LLME as an LLM arm.
POOL = ['B0', 'DEPU', 'LSTM', 'MAXL', 'STAT']
MET = [('Central error', 'MedAE_rep', 'MedAE (s)', 1, -1),
       ('', 'MAE_rep', 'MAE (s)', 0, -1),
       ('Tail', 'P90_rep', 'P90 (s)', 0, -1),
       ('', 'RMSE_rep', 'RMSE (s)', 0, -1),
       ('Row outcome', 'improved_rate', 'improved rate', 3, +1),
       ('', 'regression_rate', 'regression rate', 3, -1),
       ('Within tolerance', 'acc@1s', 'acc@1s', 3, +1),
       ('', 'acc@60s', 'acc@60s', 3, +1),
       ('', 'acc@600s', 'acc@600s', 3, +1),
       ('', 'acc@3600s', 'acc@3600s', 3, +1),
       ('Ordering (C1)', 'exact_match', 'exact match', 4, +1),
       ('', 'position_acc', 'position acc', 4, +1)]

rows = []
for grp, mk, lab, nd, dr in MET:
    cells, wb, wm, nc = [], 0, 0, 0
    for cat in CATS:
        bb, lb, lm = [], [], []
        for lab2, lg in LOGS:
            pool = [val(n, cat, lg, mk) for n in POOL
                    if (n, cat) not in FALLBACK and not (n == 'B0' and cat == 'C3')]
            pool = [v for v in pool if np.isfinite(v)]
            if not pool:
                continue
            base = min(pool) if dr < 0 else max(pool)
            tr = pd.concat([series(a, cat, lg, mk) for a in ('LLMU', 'LLMD')])
            if not len(tr):
                continue
            bst = float(tr.min()) if dr < 0 else float(tr.max())
            meds = [val(a, cat, lg, mk) for a in ('LLMU', 'LLMD')]
            meds = [v for v in meds if np.isfinite(v)]
            med = (min(meds) if dr < 0 else max(meds)) if meds else np.nan
            bb.append(base); lb.append(bst); lm.append(med)
            nc += 1
            if (bst < base if dr < 0 else bst > base):
                wb += 1
            if np.isfinite(med) and (med < base if dr < 0 else med > base):
                wm += 1
        if not bb:
            cells += ['--', '--', '--']
        else:
            cells += [fmt(float(np.median(bb)), nd), fmt(float(np.median(lb)), nd),
                      fmt(float(np.median(lm)), nd)]
    rows.append((grp, lab, cells, wb, wm, nc))

K = [BS + 'begin{table}[t]' + BS + 'centering' + BS + 'small',
     BS + 'setlength{' + BS + 'tabcolsep}{3pt}',
     BS + 'caption{Every metric in the study, each cell set against the '
     'strongest baseline available for it. \\emph{Base} is the best of B0, '
     'DEPU, LSTM, MAXL and STAT on that cell; \\emph{Best} and \\emph{Med} are '
     'the best and the median synthesis result of LLMU and LLMD. The three '
     'values are medians over the five logs of a category. The last two '
     'columns count the log--category cells in which the synthesis arms beat '
     'that baseline, on their best trial and on their median, out of the cells '
     'where the metric is defined. DISS is excluded from the baseline pool as '
     'an instrument rather than a method under test, and LLME as an LLM arm. '
     'Ordering statistics are defined for C1 only.}' + BS + 'label{tab:vsbase}',
     BS + 'begin{tabular}{ll' + 'r' * 9 + 'cc}' + BS + 'toprule',
     '& & ' + BS + 'multicolumn{3}{c}{C1} & ' + BS + 'multicolumn{3}{c}{C2} & '
     + BS + 'multicolumn{3}{c}{C3} & ' + BS
     + 'multicolumn{2}{c}{wins} ' + SEP
     + BS + 'cmidrule(lr){3-5}' + BS + 'cmidrule(lr){6-8}'
     + BS + 'cmidrule(lr){9-11}' + BS + 'cmidrule(lr){12-13}',
     'Group & Metric & ' + ' & '.join(['Base', 'Best', 'Med'] * 3)
     + ' & best & med ' + SEP + BS + 'midrule']
for grp, lab, cells, wb, wm, nc in rows:
    K.append('%s & %s & %s & %d/%d & %d/%d %s'
             % (grp, lab, ' & '.join(cells), wb, nc, wm, nc, SEP))
K += [BS + 'bottomrule' + BS + 'end{tabular}' + BS + 'end{table}']
open(os.path.join(OUT, 'table_vs_best_baseline.tex'), 'w', encoding='utf-8').write('\n'.join(K))
print('table_vs_best_baseline.tex : %d metric rows' % len(rows))
print()
print('%-16s %-15s %7s %7s' % ('group', 'metric', 'best', 'median'))
for grp, lab, cells, wb, wm, nc in rows:
    print('%-16s %-15s %7s %7s' % (grp, lab, '%d/%d' % (wb, nc), '%d/%d' % (wm, nc)))
