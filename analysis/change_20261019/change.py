"""2026-10-19: do changes in a country's shortfalls go with changes in its burden departures? (within-country, over time)

Only factor-outcome pairs in which the factor is NOT a covariate of the GBD model for that outcome (COVARIATE_OVERLAP.csv) are
used, because for covariate pairs a change in the factor changes the GBD estimate mechanically.
Pre-specified primary pairs (written before running):
  P1 air pollution (mean of standardised ambient PM2.5 and second-hand smoke departures) -> LRI incidence
  P2 drinking water -> LRI mortality
  P3 clean cooking -> meningitis mortality
Secondary: every single non-covariate factor x outcome pair (exploratory).
Change: departure in 2011-2023 minus departure in 2001-2010 (main; Hib3/PCV3/nurses/undernourishment better covered) and minus
1990-2000 (sensitivity). Factor changes are in the favourable direction and divided by their SD across locations, so a negative
value is a widening shortfall. Outcome: change in the period residual (a) as is and (b) minus the change in the country's mean
residual for the same measure across the three infections (disease contrast, removing country-wide change). OLS with SDI-stratum
intercepts (strata by 2011-2023 mean SDI), all locations and the low and low-middle stratum; 2000 location bootstrap replicates
(seed 20261019). Reported: % change in the outcome ratio per 1-SD widening of the shortfall, 100*(exp(-b)-1).
Outputs: CH_PRIMARY.csv, CH_GRID.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
rng = np.random.default_rng(20261019); NB = 2000
P1, P2, P3 = '1990–2000', '2001–2010', '2011–2023'
OUT = {(302, 1): 'Diarrhoeal mortality', (302, 6): 'Diarrhoeal incidence', (322, 1): 'LRI mortality', (322, 6): 'LRI incidence', (332, 1): 'Meningitis mortality', (332, 6): 'Meningitis incidence'}
COVCOL = {'Diarrhoeal incidence': 'Diarrhoea incidence', 'Diarrhoeal mortality': 'Diarrhoea mortality', 'LRI incidence': 'LRI incidence', 'LRI mortality': 'LRI mortality', 'Meningitis incidence': 'Meningitis incidence', 'Meningitis mortality': 'Meningitis mortality'}
FAC = ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct', 'stunting_u5_pct', 'low_birthweight_pct', 'undernourishment_pct',
       'uhc_service_index', 'health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000', 'physicians_per1000', 'gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']
COV = pd.read_csv(A / 'gbd_coupling_20261011' / 'COVARIATE_OVERLAP.csv').set_index('factor')
noncov = lambda f, o: f not in COV.index or pd.isna(COV.loc[f, COVCOL[o]])
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
R = D.pivot_table(index='location_id', columns=['cause_id', 'measure_id', 'period'], values='residual')
def dres(o_key, a, b): return R[(*o_key, b)] - R[(*o_key, a)]
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
F = F[F.factor.isin(FAC) & (F.status != '不可评价')].pivot_table(index='location_id', columns=['factor', 'period'], values='favourable_residual')
T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv'); T = T[T.period == P3].set_index('location_id').tier.map(
    {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '中等SDI': 'Middle and high-middle', '中高SDI': 'Middle and high-middle', '高SDI': 'High'})
def dfac(fs, a, b):
    parts = []
    for f in fs:
        if (f, a) not in F.columns or (f, b) not in F.columns: return None
        d = F[(f, b)] - F[(f, a)]; parts.append(d / d.std())
    x = pd.concat(parts, axis=1).mean(axis=1, skipna=False); return x / x.std()
pc = lambda b: 100 * (np.exp(-b) - 1)
def run(x, y, band):
    d = pd.concat([x.rename('x'), y.rename('y'), T.rename('band')], axis=1).dropna()
    if band != 'All': d = d[d.band == band]
    if len(d) < 25: return None
    S = pd.get_dummies(d.band).astype(float).values if band == 'All' else np.ones((len(d), 1))
    X = np.column_stack([S, d.x.values]); y_ = d.y.values
    b = np.linalg.lstsq(X, y_, rcond=None)[0][-1]; bb = []
    for _ in range(NB):
        ii = rng.integers(0, len(d), len(d)); bb.append(np.linalg.lstsq(X[ii], y_[ii], rcond=None)[0][-1])
    bb = pc(np.array(bb))
    return dict(n=len(d), pct=pc(b), lo=np.percentile(bb, 2.5), hi=np.percentile(bb, 97.5))
def outcome_change(o_key, a, b, contrast):
    y = dres(o_key, a, b)
    if contrast:
        m = o_key[1]; y = y - pd.concat([dres((c, m), a, b) for c in (302, 322, 332)], axis=1).mean(axis=1)
    return y
KEY = {v: k for k, v in OUT.items()}
prim = [('P1 air pollution -> LRI incidence', ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev'], 'LRI incidence'),
        ('P2 drinking water -> LRI mortality', ['basic_water_pct'], 'LRI mortality'),
        ('P3 clean cooking -> meningitis mortality', ['clean_cooking_pct'], 'Meningitis mortality')]
rows = []
for lab, fs, o in prim:
    assert all(noncov(f, o) for f in fs), lab
    for a in (P2, P1):
        x = dfac(fs, a, P3)
        for contrast in (False, True):
            for band in ('All', 'Low and low-middle'):
                r = run(x, outcome_change(KEY[o], a, P3, contrast), band)
                if r: rows.append(dict(pair=lab, change_from=a, contrast=contrast, band=band, **r))
pd.DataFrame(rows).to_csv(H / 'CH_PRIMARY.csv', index=False)
rows = []
for o in OUT.values():
    for f in FAC:
        if not noncov(f, o): continue
        x = dfac([f], P2, P3)
        if x is None: continue
        for contrast in (False, True):
            r = run(x, outcome_change(KEY[o], P2, P3, contrast), 'All')
            if r: rows.append(dict(outcome=o, factor=f, contrast=contrast, **r))
pd.DataFrame(rows).to_csv(H / 'CH_GRID.csv', index=False)
print('done')
