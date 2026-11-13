# -*- coding: utf-8 -*-
"""Which strategies do the synthesized programs actually implement?

Scans the winning program of every reported per-category run and records which
of a fixed set of code-level features it contains. The feature list is the one
the paper needs for Section 6.3; detection is by pattern, so a feature means
"the program contains code that does this", not an LLM's opinion of it.
"""
import os, io, re, glob, csv, collections

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

FEATURES = [
    ('perm',      r'itertools\.permutations|permutations\(',
     'enumerates orderings of the block exhaustively'),
    ('beam',      r'beam|heapq\.nlargest|top_?k',
     'keeps a bounded set of partial orderings instead of all of them'),
    ('transprob', r'trans_prob|transition_prob|directly_follows|dfg|follows_count',
     'scores an ordering with directly-follows or transition statistics'),
    ('rightanch', r'right_?anchor|t_?R\b|next_clean|succ_ts|anchor_end',
     'reads the right anchor at all'),
    ('r_ceiling', r'clip\(|clamp|min\(.*right|cap_|upper',
     'right anchor as a CEILING, bounding a forward layout (STAT-like)'),
    ('r_evidence', r'sum\(.*\)\s*==\s*W|total[_ ]?span|W\s*=\s*t_?R\s*-\s*t_?L',
     'right anchor as EVIDENCE, gaps fitted to the total span (MAXL-like)'),
    ('clamp',     r'\bclip\(|clamp|min\(max\(|np\.minimum\(.*np\.maximum',
     'clamps repaired values into the anchor interval'),
    ('median',    r'\.median\(|np\.median|median_duration',
     'uses a median duration statistic'),
    ('mean',      r'\.mean\(\)|np\.mean',
     'uses a mean duration statistic'),
    ('lognorm',   r'lognorm|np\.log1p|np\.log\(|sigma',
     'works on a log scale or fits a lognormal-like duration model'),
    ('uniform',   r'linspace|uniform|equal_?spac|/ *\(?n *\+ *1',
     'spreads the block uniformly across the interval'),
    ('rescale',   r'scale_?factor|rescal|\* *\(?W */|ratio *\*',
     'rescales gaps when their sum overruns the available room'),
    ('firstts',   r'first|iloc\[0\]|head\(1\)|idxmin',
     'treats the surviving value as the first event of the block'),
    ('monotone',  r'cummax|is_monotonic|sort_values\(.*_ts|np\.maximum\.accumulate',
     'enforces monotonicity within the case'),
    ('fallback',  r'FALLBACK|except\s|try:',
     'carries an explicit fallback path'),
]

rows, counts = [], collections.Counter()
paths = sorted(glob.glob(R + '/programs/*.py'))
for p in paths:
    # programs/<log>_<cat>_<pollution>_<ARM>_t<N>.py
    m = re.match(r'^(\w+?)_(C\d)_\d+_(\w+?)_t(\d+)$',
                 os.path.basename(p)[:-3])
    if not m:
        continue
    log, cat, method, trial = m.group(1), m.group(2), m.group(3), m.group(4)
    try:
        src = io.open(p, encoding='utf-8', errors='replace').read()
    except Exception:
        continue
    rec = dict(log=log, category=cat, method=method, trial=trial,
               n_lines=src.count('\n') + 1)
    for key, pat, _ in FEATURES:
        hit = int(bool(re.search(pat, src, re.I)))
        rec[key] = hit
        if hit:
            counts[key] += 1
    rows.append(rec)

cols = (['log', 'category', 'method', 'trial', 'n_lines']
        + [k for k, _, _ in FEATURES])
with io.open(os.path.join(OUT, 'strategy_features.csv'), 'w',
             encoding='utf-8', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

n = len(rows)
print('scanned %d winning programs from the reported per-category runs' % n)
print('median program length: %d lines\n'
      % sorted(r['n_lines'] for r in rows)[n // 2] if n else 0)
print('%-10s %6s  %s' % ('feature', 'share', 'description'))
for key, _, desc in FEATURES:
    print('%-10s %5.0f%%  %s' % (key, 100.0 * counts[key] / n if n else 0, desc))

print('\nby category (share of programs containing the feature):')
print('%-10s %7s %7s %7s' % ('feature', 'C1', 'C2', 'C3'))
for key, _, _ in FEATURES:
    line = '%-10s' % key
    for c in ('C1', 'C2', 'C3'):
        sub = [r for r in rows if r['category'] == c]
        line += ' %6.0f%%' % (100.0 * sum(r[key] for r in sub) / len(sub)
                              if sub else 0)
    print(line)
