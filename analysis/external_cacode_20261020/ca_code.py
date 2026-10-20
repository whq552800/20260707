"""2026-10-20: external check with UN IGME / WHO-UNICEF CA CODE cause-specific under-5 mortality (2000-2024).

Data (not redistributed; downloaded 2026-10-19 by the authors from the UNICEF SDMX flow CME_CAUSE_OF_DEATH 1.0, series CA CODE,
consistent with UN IGME all-cause estimates): deaths per 1000 live births from diarrhoea, LRI and meningitis/encephalitis, both
sexes, age 0-4, 194 countries. Path given by env var CACODE_WIDE or ./input/UN_IGME_under5_three_causes_wide.csv.
All-cause U5MR is recovered as rate / fraction x 100 (median over the three causes with fraction > 0); other-cause = U5MR minus the
three causes. Locations are matched to GBD by name (4 renamed).
Each CA CODE outcome gets the same SDI-year reference as the GBD analysis (GBD SDI; natural spline with the GBD knots, year effects,
spline x centred year; 2000-2023, equal weights; zero rates dropped). Periods: early 2000-2010, late 2011-2023.
Checks: (1) agreement of CA CODE and GBD late-period residuals; (2) early-late persistence; (3) share of variance shared with
other-cause mortality and Nigeria's general share of 2023 excess deaths; (4) within-country mortality contrasts on GBD-based domain
indices (low and low-middle SDI); (5) within-country change for the pre-specified mortality pairs (drinking water -> LRI, clean
cooking -> meningitis), per cross-sectional SD. Bootstrap: 2000 location resamples, seed 20261020.
Outputs: CA_AGREEMENT.csv, CA_LAYERS.csv, CA_WITHIN.csv, CA_CHANGE.csv
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
sys.path.insert(0, str(A / 'independent_reproduction_20261009'))
from paths import PANEL
from statlib import rankdata
rng = np.random.default_rng(20261020); NB = 2000
src = Path(os.environ.get('CACODE_WIDE', H / 'input' / 'UN_IGME_under5_three_causes_wide.csv'))
W = pd.read_csv(src, encoding='utf-8-sig')
REN = {'Netherlands (Kingdom of the)': 'Netherlands', 'Republic of Türkiye': 'Türkiye', 'State of Palestine': 'Palestine', 'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom'}
W['location'] = W.country.replace(REN)
pan = pd.read_csv(PANEL, usecols=['location_id', 'location', 'year', 'sdi', 'cause_id', 'measure_id', 'gbd_rate_per_100k'])
LOC = pan[['location_id', 'location']].drop_duplicates()
W = W.merge(LOC, on='location', how='inner'); W = W[W.year <= 2023]
CZ = {'diarrhea': 302, 'lri': 322, 'meningitis_encephalitis': 332}
fr = pd.concat([W[f'{c}_rate_per_1000_live_births_estimate'] / W[f'{c}_fraction_pct_estimate'] * 100 for c in CZ], axis=1).where(lambda x: np.isfinite(x))
W['u5mr'] = fr.median(axis=1)
W['other'] = W.u5mr - sum(W[f'{c}_rate_per_1000_live_births_estimate'] for c in CZ)
sdi = pan.drop_duplicates(['location_id', 'year'])[['location_id', 'year', 'sdi']]
W = W.merge(sdi, on=['location_id', 'year'])
KN = np.quantile(pan.drop_duplicates(['location_id', 'year']).sdi, [.25, .5, .75])
def ns_basis(x, knots, bk=(0.0, 1.0)):
    k = np.r_[bk[0], knots, bk[1]]; K = len(k)
    def dk(j): return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])
    return np.column_stack([x] + [dk(j) - dk(K - 2) for j in range(K - 2)])
def resid(df, col):
    g = df[df[col] > 0][['location_id', 'year', 'sdi', col]].sort_values(['location_id', 'year']).reset_index(drop=True)
    B = ns_basis(g.sdi.values, KN); tc = ((g.year - 2011.5) / 23).values[:, None]
    YF = pd.get_dummies(g.year, drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(g)), YF, B, B * tc]); y = np.log(g[col].values)
    b = np.linalg.pinv(X.T @ X) @ X.T @ y; g['res'] = y - X @ b; g['fit'] = X @ b
    g['period'] = np.where(g.year <= 2010, 'early', 'late')
    return g
OUTC = {'Diarrhoeal mortality': 'diarrhea_rate_per_1000_live_births_estimate', 'LRI mortality': 'lri_rate_per_1000_live_births_estimate',
        'Meningitis mortality': 'meningitis_encephalitis_rate_per_1000_live_births_estimate', 'Other-cause mortality': 'other'}
FIT = {o: resid(W, c) for o, c in OUTC.items()}
PR = {o: g.groupby(['location_id', 'period']).res.mean().unstack() for o, g in FIT.items()}
def rho(x, y):
    m = x.notna() & y.notna(); return np.corrcoef(rankdata(x[m].values), rankdata(y[m].values))[0, 1], int(m.sum())
def boot_ci(f, idx):
    v = [f(rng.choice(idx, len(idx))) for _ in range(NB)]; return np.percentile(v, [2.5, 97.5])
# (1) agreement with GBD and (2) persistence
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv'); D = D[D.measure_id == 1]
GL = D[D.period == '2011–2023'].pivot(index='location_id', columns='cause_id', values='residual')
GBDO = pd.read_csv(A / 'gbd_envelope_20261012' / 'GBD2023_U5_ALLCAUSE_DEATH_RATE.csv')
rows = []
for o, c in [('Diarrhoeal mortality', 302), ('LRI mortality', 322), ('Meningitis mortality', 332)]:
    r1, n1 = rho(PR[o]['late'], GL[c].reindex(PR[o].index)); r2, n2 = rho(PR[o]['early'], PR[o]['late'])
    rows.append(dict(outcome=o, rho_ca_vs_gbd_late=r1, n_agree=n1, rho_early_late_ca=r2, n_persist=n2))
r2, n2 = rho(PR['Other-cause mortality']['early'], PR['Other-cause mortality']['late']); rows.append(dict(outcome='Other-cause mortality', rho_early_late_ca=r2, n_persist=n2))
pd.DataFrame(rows).to_csv(H / 'CA_AGREEMENT.csv', index=False)
# (3) layers
rows = []
for o in ['Diarrhoeal mortality', 'LRI mortality', 'Meningitis mortality']:
    d = pd.concat([PR[o]['late'].rename('y'), PR['Other-cause mortality']['late'].rename('x')], axis=1).dropna()
    def r2f(ii):
        s = d.iloc[ii]; X = np.column_stack([np.ones(len(s)), s.x]); e = s.y - X @ np.linalg.lstsq(X, s.y, rcond=None)[0]; return 1 - e.var() / s.y.var()
    est = r2f(np.arange(len(d))); lo, hi = boot_ci(r2f, np.arange(len(d)))
    rows.append(dict(outcome=o, n=len(d), r2_shared=est, lo=lo, hi=hi))
# Nigeria 2023: share of log excess ratio predicted by the other-cause residual (point)
ng = {}
oth23 = FIT['Other-cause mortality'].query('year == 2023').set_index('location_id').res
for o in ['Diarrhoeal mortality', 'LRI mortality', 'Meningitis mortality']:
    g = FIT[o].query('year == 2023').set_index('location_id'); d = pd.concat([g.res.rename('r'), oth23.rename('x')], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d.x]); b = np.linalg.lstsq(X, d.r, rcond=None)[0]
    if 214 in d.index: ng[o] = dict(resid_2023=d.loc[214, 'r'], general_part=X[d.index.get_loc(214)] @ b)
for o, v in ng.items(): rows.append(dict(outcome='Nigeria 2023: ' + o, r2_shared=np.nan, nigeria_log_ratio=v['resid_2023'], nigeria_general_part=v['general_part']))
pd.DataFrame(rows).to_csv(H / 'CA_LAYERS.csv', index=False)
# (4) within-country mortality contrasts on GBD-based domains
FMD = pd.read_csv(A / 'factor_models_20261017' / 'FM_DATA.csv').set_index('location_id').dropna()
DN = ['Basic services', 'Immunisation', 'Nutrition', 'Health system', 'Air pollution']
M = ['Diarrhoeal mortality', 'LRI mortality', 'Meningitis mortality']
L = pd.concat([PR[o]['late'].rename(o) for o in M], axis=1); L = L.sub(L.mean(axis=1), axis=0)
pc = lambda b: 100 * (np.exp(-b) - 1)
rows = []
for band in ['Low and low-middle', 'All']:
    s = FMD[DN + ['band']].join(L, how='inner').dropna(subset=M)
    if band != 'All': s = s[s.band == band]
    S = pd.get_dummies(s.band).astype(float).values if band == 'All' else np.ones((len(s), 1)); k0 = S.shape[1]
    X = np.column_stack([S, s[DN].values])
    for o in M:
        y = s[o].values; b = np.linalg.lstsq(X, y, rcond=None)[0][k0:]
        bb = np.array([np.linalg.lstsq(X[ii], y[ii], rcond=None)[0][k0:] for ii in (rng.integers(0, len(s), len(s)) for _ in range(NB))])
        for j, d in enumerate(DN):
            rows.append(dict(band=band, n=len(s), outcome=o, domain=d, pct=pc(b[j]), lo=np.percentile(pc(bb[:, j]), 2.5), hi=np.percentile(pc(bb[:, j]), 97.5)))
pd.DataFrame(rows).to_csv(H / 'CA_WITHIN.csv', index=False)
# (5) within-country change, pre-specified mortality pairs
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
F = F[F.status != '不可评价'].pivot_table(index='location_id', columns=['factor', 'period'], values='favourable_residual')
T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv'); T = T[T.period == '2011–2023'].set_index('location_id').tier
rows = []
for lab, f, o in [('Drinking water -> LRI mortality', 'basic_water_pct', 'LRI mortality'), ('Clean cooking -> meningitis mortality', 'clean_cooking_pct', 'Meningitis mortality')]:
    lvl = F[(f, '2011–2023')]; sd = lvl.std(); dx = (F[(f, '2011–2023')] - F[(f, '2001–2010')]) / sd
    for contrast in (False, True):
        if contrast:
            yl = L[o]; Ls = pd.concat([PR[m]['early'].rename(m) for m in M], axis=1); Ls = Ls.sub(Ls.mean(axis=1), axis=0); dy = L[o] - Ls[o]
        else:
            yl = PR[o]['late']; dy = PR[o]['late'] - PR[o]['early']
        for kind, x, y in [('between', lvl / sd, yl), ('within', dx, dy)]:
            d = pd.concat([x.rename('x'), y.rename('y'), T.rename('band')], axis=1).dropna()
            S = pd.get_dummies(d.band).astype(float).values; X = np.column_stack([S, d.x.values])
            b = np.linalg.lstsq(X, d.y.values, rcond=None)[0][-1]
            bb = np.array([np.linalg.lstsq(X[ii], d.y.values[ii], rcond=None)[0][-1] for ii in (rng.integers(0, len(d), len(d)) for _ in range(NB))])
            rows.append(dict(pair=lab, contrast=contrast, kind=kind, n=len(d), pct=pc(b), lo=np.percentile(pc(bb), 2.5), hi=np.percentile(pc(bb), 97.5)))
pd.DataFrame(rows).to_csv(H / 'CA_CHANGE.csv', index=False)
print('done', W.location_id.nunique(), 'locations')
