"""2026-10-20 (reviewer request): for each pre-specified change pair, the change over time in all three infections of the same
measure, side by side, with pairwise differences (joint location bootstrap, 2000 replicates, seed 20261020).
Factor change: 2001-2010 to 2011-2023, favourable direction, per SD of change (as change_20261019). Outcome: change in the period
residual. OLS with SDI-stratum intercepts, all locations. Values: % change in the burden ratio per 1-SD widening of the shortfall;
differences are on the log scale converted to % (ratio of ratios). Pairs in which the factor is a GBD covariate of that outcome
are flagged (covariate=True). Output: PER_INFECTION_CHANGE.csv, PER_INFECTION_DIFF.csv"""
import sys, runpy
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
g = runpy.run_path(str(A / 'change_20261019' / 'change.py'), run_name='lib') if False else None
# reuse definitions without running change.py's loops
src = (A / 'change_20261019' / 'change.py').read_text().split("KEY = {v: k for k, v in OUT.items()}")[0]
src = src.replace("H = Path(__file__).resolve().parent; A = H.parent", "")
ns = {'A': A, 'Path': Path, 'np': np, 'pd': pd}; exec(src, ns)
dfac, dres, T, noncov, OUT = ns['dfac'], ns['dres'], ns['T'], ns['noncov'], ns['OUT']
KEY = {v: k for k, v in OUT.items()}
rng = np.random.default_rng(20261020); NB = 2000
P2, P3 = '2001–2010', '2011–2023'
pc = lambda b: 100 * (np.exp(-b) - 1)
pairs = [('Air pollution', ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev'], 'incidence', 'LRI'),
         ('Drinking water', ['basic_water_pct'], 'mortality', 'LRI'),
         ('Clean cooking', ['clean_cooking_pct'], 'mortality', 'Meningitis')]
DZ = {'Diarrhoeal': 302, 'LRI': 322, 'Meningitis': 332}
rows, drows = [], []
for lab, fs, meas, target in pairs:
    x = dfac(fs, P2, P3); m = 6 if meas == 'incidence' else 1
    ys = {d: dres((c, m), P2, P3) for d, c in DZ.items()}
    dd = pd.concat([x.rename('x'), T.rename('band')] + [ys[d].rename(d) for d in DZ], axis=1).dropna()
    S = pd.get_dummies(dd.band).astype(float).values; X = np.column_stack([S, dd.x.values])
    Yv = dd[list(DZ)].values
    b = np.linalg.lstsq(X, Yv, rcond=None)[0][-1]
    B = []
    for _ in range(NB):
        ii = rng.integers(0, len(dd), len(dd)); B.append(np.linalg.lstsq(X[ii], Yv[ii], rcond=None)[0][-1])
    B = np.array(B)
    for j, d in enumerate(DZ):
        o = f'{d} {meas}'
        rows.append(dict(factor=lab, outcome=o, target=(d == target), covariate=not all(noncov(f, o) for f in fs), n=len(dd),
                         pct=pc(b[j]), lo=np.percentile(pc(B[:, j]), 2.5), hi=np.percentile(pc(B[:, j]), 97.5)))
    jt = list(DZ).index(target)
    for j, d in enumerate(DZ):
        if j == jt: continue
        diff = -(B[:, jt] - B[:, j]); est = -(b[jt] - b[j])
        drows.append(dict(factor=lab, measure=meas, target=target, other=d, diff_pct=100 * (np.exp(est) - 1),
                          lo=100 * (np.exp(np.percentile(diff, 2.5)) - 1), hi=100 * (np.exp(np.percentile(diff, 97.5)) - 1)))
R = pd.DataFrame(rows); D = pd.DataFrame(drows)
R.to_csv(H / 'PER_INFECTION_CHANGE.csv', index=False); D.to_csv(H / 'PER_INFECTION_DIFF.csv', index=False)
pd.set_option('display.width', 200); print(R.round(1).to_string()); print(D.round(1).to_string())
