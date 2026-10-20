"""2026-10-19: within-country, between-outcome models — where, within a country, does the disadvantage fall?

Country background (all that is common to a country's outcomes) is removed by centring each country's residuals:
  (a) disease contrast: each mortality residual minus the mean of the country's three mortality residuals (same for incidence),
      i.e. y_{c,o} - mean_o' y_{c,o'} = country fixed effect model with outcome effects;
  (b) mortality-incidence contrast: mortality residual minus incidence residual of the same disease (departure in deaths per
      episode relative to the reference).
Each contrast is regressed on the five domain indices of factor_models_20261017 (2011-2023; 1-SD shortfall), within SDI strata,
with 2000 location bootstrap replicates (seed 20261019) resampling locations with all their outcomes. Values: 100*(exp(-b)-1),
the percentage by which a 1-SD shortfall raises this disease's ratio relative to the country's other infections (a), or raises
mortality relative to incidence of the same disease (b). In (a) the three coefficients of a domain sum to zero on the log scale
by construction, so only differences between diseases are identified.
A priori expectations (set before running, from the literature cited in the manuscript): air pollution -> LRI (a), basic
services -> diarrhoeal disease (a), immunisation -> LRI and meningitis rather than diarrhoea (a; no rotavirus vaccine in the
data), health system -> mortality relative to incidence (b).
Output: WC_DISEASE.csv, WC_FATALITY.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
rng = np.random.default_rng(20261019); NB = 2000
F = pd.read_csv(A / 'factor_models_20261017' / 'FM_DATA.csv').set_index('location_id').dropna()   # 190 locations with all five domains (2026-10-20 rule)
DN = ['Basic services', 'Immunisation', 'Nutrition', 'Health system', 'Air pollution']
DZ = ['Diarrhoeal', 'LRI', 'Meningitis']
Y = {}
for m in ['mortality', 'incidence']:
    cols = [f'{d} {m}' for d in DZ]; mu = F[cols].mean(axis=1)
    for c in cols: Y[('disease', c)] = F[c] - mu
for d in DZ: Y[('fatality', d)] = F[f'{d} mortality'] - F[f'{d} incidence']
pc = lambda b: 100 * (np.exp(-b) - 1)
BANDS = ['Low and low-middle', 'Middle and high-middle', 'High', 'All']
def fit(X, y): return np.linalg.lstsq(X, y, rcond=None)[0]
out = {'disease': [], 'fatality': []}
for band in BANDS:
    s = F if band == 'All' else F[F.band == band]
    if band == 'All':
        S = pd.get_dummies(s.band).astype(float).values; X = np.column_stack([S, s[DN].values]); k0 = S.shape[1]
    else:
        X = np.column_stack([np.ones(len(s)), s[DN].values]); k0 = 1
    Ys = {k: v.loc[s.index].values for k, v in Y.items()}
    est = {k: fit(X, y)[k0:] for k, y in Ys.items()}
    boot = {k: [] for k in Ys}
    for _ in range(NB):
        ii = rng.integers(0, len(s), len(s))
        for k, y in Ys.items(): boot[k].append(fit(X[ii], y[ii])[k0:])
    for (kind, name), b in est.items():
        B = pc(np.array(boot[(kind, name)]))
        for j, d in enumerate(DN):
            out[kind].append(dict(band=band, n=len(s), outcome=name, domain=d, pct=pc(b[j]), lo=np.percentile(B[:, j], 2.5), hi=np.percentile(B[:, j], 97.5)))
pd.DataFrame(out['disease']).to_csv(H / 'WC_DISEASE.csv', index=False)
pd.DataFrame(out['fatality']).to_csv(H / 'WC_FATALITY.csv', index=False)
print('done')
