"""2026-10-20: does the immunisation result depend on which vaccines form the domain? The within-country models (cross-infection
contrasts and deaths per episode) and the between-country models are refitted with the immunisation domain built from DTP3 only,
or from Hib3 and PCV3 only; other domains as in the main analysis (health system with the UHC index). Same 190 locations, 2000
location resamples (seed 20261020). Output: VACCINE_SPLIT.csv (immunisation coefficients only)."""
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
P = '2011–2023'; NB = 2000
BASE = {'Basic services': ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct'],
        'Nutrition': ['stunting_u5_pct', 'low_birthweight_pct', 'undernourishment_pct'],
        'Health system': ['uhc_service_index', 'health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000', 'physicians_per1000'],
        'Air pollution': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']}
VAR = {'All three vaccines (main)': ['dtp3_pct', 'hib3_pct', 'pcv3_pct'], 'DTP3 only': ['dtp3_pct'], 'Hib3 and PCV3 only': ['hib3_pct', 'pcv3_pct']}
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv'); F = F[(F.period == P) & (F.status != '不可评价')]
Z = F.pivot(index='location_id', columns='factor', values='favourable_residual'); Z = Z / Z.std()
M = pd.read_csv(A / 'factor_models_20261017' / 'FM_DATA.csv').set_index('location_id').dropna()
OUTS = ['Diarrhoeal incidence', 'Diarrhoeal mortality', 'LRI incidence', 'LRI mortality', 'Meningitis incidence', 'Meningitis mortality']
DZ = ['Diarrhoeal', 'LRI', 'Meningitis']
pc = lambda b: 100 * (np.exp(-b) - 1)
rows = []
for vname, vf in VAR.items():
    rng = np.random.default_rng(20261020)
    dom = {**{d: Z[fs].mean(axis=1) for d, fs in BASE.items()}, 'Immunisation': Z[vf].mean(axis=1)}
    X = pd.DataFrame(dom)[['Basic services', 'Immunisation', 'Nutrition', 'Health system', 'Air pollution']]; X = X / X.std()
    dat = X.loc[M.index].join(M[OUTS + ['band']])
    Y = {}
    for o in OUTS: Y[('between', o)] = dat[o]
    for m in ['mortality', 'incidence']:
        cols = [f'{d} {m}' for d in DZ]; mu = dat[cols].mean(axis=1)
        for c in cols: Y[('contrast', c)] = dat[c] - mu
    for d in DZ: Y[('per_episode', d)] = dat[f'{d} mortality'] - dat[f'{d} incidence']
    for band in ['Low and low-middle', 'High']:
        s = dat[dat.band == band].dropna(); keys = list(Y)
        Xm = np.column_stack([np.ones(len(s)), s[list(X.columns)].values]); Ys = np.column_stack([Y[k].loc[s.index].values for k in keys])
        b = np.linalg.lstsq(Xm, Ys, rcond=None)[0][2]       # immunisation coefficient (column order: intercept, basic, immun, ...)
        B = np.array([np.linalg.lstsq(Xm[ii], Ys[ii], rcond=None)[0][2] for ii in (rng.integers(0, len(s), len(s)) for _ in range(NB))])
        for k, (kind, o) in enumerate(keys):
            v = pc(B[:, k]); rows.append(dict(vaccines=vname, band=band, n=len(s), kind=kind, outcome=o, pct=pc(b[k]), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
R = pd.DataFrame(rows); R.to_csv(H / 'VACCINE_SPLIT.csv', index=False)
pd.set_option('display.width', 200)
print(R[(R.band == 'Low and low-middle') & (R.kind == 'contrast')].round(1).pivot_table(index='outcome', columns='vaccines', values=['pct', 'lo', 'hi']).round(1).to_string())
