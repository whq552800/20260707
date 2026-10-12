"""Intervals for the persistence results (2026-10-12).
(1) Spearman rank correlation between 1990-2000 and 2011-2023 period residuals for each outcome, with 95% intervals from 2000
location resamples (percentile); seed 20261012. (2) Percentages of early high-tail and low-tail records found in the same tail in
2011-2023, and of locations with opposite incidence and mortality directions in 2011-2023, with Wilson 95% intervals.
Output: PERSISTENCE_INTERVALS.csv."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
A = Path(__file__).resolve().parent.parent; HERE = Path(__file__).resolve().parent
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
E0, L0 = '1990–2000', '2011–2023'; rng = np.random.default_rng(20261012); B = 2000
OUT = {(302, 1): 'Diarrhoeal mortality', (322, 1): 'LRI mortality', (332, 1): 'Meningitis mortality',
       (302, 6): 'Diarrhoeal incidence', (322, 6): 'LRI incidence', (332, 6): 'Meningitis incidence'}
def rho(x, y): return np.corrcoef(rankdata(x), rankdata(y))[0, 1]
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return 100 * (c - h), 100 * (c + h)
rows = []
for (c, m), o in OUT.items():
    w = D[(D.cause_id == c) & (D.measure_id == m)].pivot(index='location_id', columns='period', values='residual')[[E0, L0]].dropna()
    x, y = w[E0].values, w[L0].values; idx = rng.integers(0, len(x), (B, len(x)))
    b = np.array([rho(x[i], y[i]) for i in idx])
    rows.append(dict(item='rank_corr', outcome=o, n=len(x), est=rho(x, y), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
j = D[D.period == E0].set_index(['location_id', 'cause_id', 'measure_id']).category.to_frame('e').join(D[D.period == L0].set_index(['location_id', 'cause_id', 'measure_id']).category.rename('l'))
for t, lab in [('稳健高尾', 'stay_high_tail'), ('稳健低尾', 'stay_low_tail')]:
    s = j[j.e == t]; k, n = int((s.l == t).sum()), len(s); lo, hi = wilson(k, n)
    rows.append(dict(item=lab, outcome='all', n=n, k=k, est=100 * k / n, lo=lo, hi=hi))
L = D[D.period == L0]
for c, o in [(302, 'Diarrhoeal disease'), (322, 'LRI'), (332, 'Meningitis')]:
    s = L[L.cause_id == c].assign(s=lambda x: x.direction.map({'pos': 1, 'neg': -1})).pivot(index='location_id', columns='measure_id', values='s').dropna()
    k, n = int((s[1] != s[6]).sum()), len(s); lo, hi = wilson(k, n)
    rows.append(dict(item='opposite_direction', outcome=o, n=n, k=k, est=100 * k / n, lo=lo, hi=hi))
R = pd.DataFrame(rows); R.to_csv(HERE / 'PERSISTENCE_INTERVALS.csv', index=False, encoding='utf-8-sig'); print(R.round(3).to_string())
