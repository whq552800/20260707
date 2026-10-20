"""2026-10-20 (reviewer request): the UHC service coverage index is a composite that includes immunisation, sanitation and
care-seeking indicators, overlapping the other domains. Refit (i) the between-country stratum models and (ii) the cross-infection
contrast and deaths-per-episode models with the health-system domain built without the UHC index (health expenditure, hospital
beds, nurses and midwives, physicians). Same locations as the main models (190 with all five domains), same bootstrap scheme
(2000 location replicates, seed 20261020). Output: NOUHC_BETWEEN.csv, NOUHC_WITHIN.csv"""
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
rng = np.random.default_rng(20261020); NB = 2000
P = '2011–2023'
DOM = {'Basic services': ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct'], 'Immunisation': ['dtp3_pct', 'hib3_pct', 'pcv3_pct'],
       'Nutrition': ['stunting_u5_pct', 'low_birthweight_pct', 'undernourishment_pct'],
       'Health system': ['health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000', 'physicians_per1000'],
       'Air pollution': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']}
DN = list(DOM); FAC = [f for v in DOM.values() for f in v]
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv'); F = F[(F.period == P) & F.factor.isin(FAC) & (F.status != '不可评价')]
Z = F.pivot(index='location_id', columns='factor', values='favourable_residual').reindex(columns=FAC); Z = Z / Z.std()
X = pd.DataFrame({d: Z[fs].mean(axis=1) for d, fs in DOM.items()}); X = X / X.std()
M = pd.read_csv(A / 'factor_models_20261017' / 'FM_DATA.csv').set_index('location_id').dropna()   # main analysis set (190)
OUTS = ['Diarrhoeal incidence', 'Diarrhoeal mortality', 'LRI incidence', 'LRI mortality', 'Meningitis incidence', 'Meningitis mortality']
dat = X.loc[M.index].join(M[OUTS + ['band']]).dropna()
print('locations', len(dat), dat.band.value_counts().to_dict())
DZ = ['Diarrhoeal', 'LRI', 'Meningitis']; Y = {}
for o in OUTS: Y[('between', o)] = dat[o]
for m in ['mortality', 'incidence']:
    cols = [f'{d} {m}' for d in DZ]; mu = dat[cols].mean(axis=1)
    for c in cols: Y[('contrast', c)] = dat[c] - mu
for d in DZ: Y[('per_episode', d)] = dat[f'{d} mortality'] - dat[f'{d} incidence']
pc = lambda b: 100 * (np.exp(-b) - 1)
rows = []
for band in ['Low and low-middle', 'Middle and high-middle', 'High']:
    s = dat[dat.band == band]; Xm = np.column_stack([np.ones(len(s)), s[DN].values])
    Ys = np.column_stack([v.loc[s.index].values for v in Y.values()]); keys = list(Y)
    b = np.linalg.lstsq(Xm, Ys, rcond=None)[0][1:]
    B = np.array([np.linalg.lstsq(Xm[ii], Ys[ii], rcond=None)[0][1:] for ii in (rng.integers(0, len(s), len(s)) for _ in range(NB))])
    for k, (kind, name) in enumerate(keys):
        for j, d in enumerate(DN):
            v = pc(B[:, j, k]); rows.append(dict(band=band, n=len(s), kind=kind, outcome=name, domain=d, pct=pc(b[j, k]), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
R = pd.DataFrame(rows)
R[R.kind == 'between'].to_csv(H / 'NOUHC_BETWEEN.csv', index=False); R[R.kind != 'between'].to_csv(H / 'NOUHC_WITHIN.csv', index=False)
pd.set_option('display.width', 200)
sig = R[(R.lo > 0) | (R.hi < 0)]
print(sig.round(1).to_string())
