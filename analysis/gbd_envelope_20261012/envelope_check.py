"""All-cause envelope check (2026-10-12, third independent review).

GBD rescales cause-specific deaths to a common all-cause mortality envelope, so departures in all-cause under-5 mortality
could make cause-specific mortality residuals correlate across diseases. Input: GBD 2023 all-cause death rate per 100 000
among children younger than 5 years, both sexes, for the 204 locations in 1990-2023 (GBD2023_U5_ALLCAUSE_DEATH_RATE.csv,
retrieved from the GBD 2023 results database on 2026-10-12).
1. The all-cause rate is given the same SDI-year reference as the disease outcomes; its period residual is the envelope residual.
2. Cross-disease Spearman correlations of mortality residuals are recomputed (a) after removing the envelope residual from each
   mortality residual by linear regression, and (b) for cause fractions (cause-specific / all-cause rate), each fitted with
   the same reference. Incidence residuals are shown unchanged and, in (a), also after removing the envelope residual.
   Intervals: 1000 location resamples (seed 20261015) for the mortality-minus-incidence difference.
3. Early-late rank correlations of the cause-fraction residuals describe persistence net of the envelope.
4. (Fourth independent review) Because all-cause mortality contains the three causes, the adjustment is repeated with the
   residual of all other causes (all-cause minus the three study causes), applied symmetrically to mortality and incidence;
   and raw cross-disease correlations are recomputed after subtracting United Nations subregion means from every residual.
Outputs: ENVELOPE_PERIOD_RESIDUALS.csv, ENVELOPE_CORR.csv, ENVELOPE_PERSISTENCE.csv, ENVELOPE_OTHER_CAUSES.csv,
   ENVELOPE_PERSISTENCE_WITHIN_SUBREGION.csv (early-late rank correlations after subtracting subregion means),
   ENVELOPE_OTHER_WITHIN.csv (fifth review: other-cause adjustment within subregions; persistence of other-cause mortality)"""
import sys, numpy as np, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent; A = H.parent
sys.path.insert(0, str(A / 'independent_reproduction_20261009'))
from paths import PANEL
from statlib import rankdata

P3 = ['1990–2000', '2001–2010', '2011–2023']; PAIRS = [(302, 322), (302, 332), (322, 332)]
def period_of(y): return P3[0] if y <= 2000 else (P3[1] if y <= 2010 else P3[2])
def ns_basis(x, knots, bk=(0.0, 1.0)):
    k = np.r_[bk[0], knots, bk[1]]; K = len(k)
    def dk(j): return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])
    return np.column_stack([x] + [dk(j) - dk(K - 2) for j in range(K - 2)])
def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = ~(np.isnan(x) | np.isnan(y))
    return np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1]

pan = pd.read_csv(PANEL, usecols=['location_id', 'cause_id', 'measure_id', 'year', 'sdi', 'gbd_rate_per_100k'])
KN = np.quantile(pan.drop_duplicates(['location_id', 'year']).sdi, [.25, .5, .75])
env = pd.read_csv(H / 'GBD2023_U5_ALLCAUSE_DEATH_RATE.csv')
sdi = pan.drop_duplicates(['location_id', 'year'])[['location_id', 'year', 'sdi']]

def period_resid(g, ycol):
    """Same reference as the disease outcomes: spline of SDI, year effects, spline x centred year; equal weights."""
    g = g.sort_values(['location_id', 'year']).reset_index(drop=True)
    B = ns_basis(g.sdi.values, KN); tc = ((g.year - 2006.5) / 33).values[:, None]
    YF = pd.get_dummies(g.year, drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(g)), YF, B, B * tc]); y = g[ycol].values
    b = np.linalg.pinv(X.T @ X) @ X.T @ y; g['res'] = y - X @ b; g['period'] = g.year.map(period_of)
    return g.groupby(['location_id', 'period']).res.mean()

e = env.merge(sdi, on=['location_id', 'year']); e['y'] = np.log(e.allcause_rate_per_100k)
R = {'env': period_resid(e, 'y')}
for c in (302, 322, 332):
    for m in (1, 6):
        g = pan[(pan.cause_id == c) & (pan.measure_id == m)].copy(); g['y'] = np.log(g.gbd_rate_per_100k)
        R[f'{c}_{m}'] = period_resid(g, 'y')
    g = pan[(pan.cause_id == c) & (pan.measure_id == 1)].merge(env[['location_id', 'year', 'allcause_rate_per_100k']], on=['location_id', 'year'])
    g['y'] = np.log(g.gbd_rate_per_100k / g.allcause_rate_per_100k); R[f'{c}_frac'] = period_resid(g, 'y')
R = pd.DataFrame(R).reset_index(); R.to_csv(H / 'ENVELOPE_PERIOD_RESIDUALS.csv', index=False, encoding='utf-8-sig')

def stats(d):
    o = {}
    E = np.column_stack([np.ones(len(d)), d.env.values])
    adj = lambda v: v - E @ np.linalg.lstsq(E, v, rcond=None)[0]
    o['mort_raw'] = np.mean([rho(d[f'{a}_1'], d[f'{b}_1']) for a, b in PAIRS])
    o['inc_raw'] = np.mean([rho(d[f'{a}_6'], d[f'{b}_6']) for a, b in PAIRS])
    o['mort_env_removed'] = np.mean([rho(adj(d[f'{a}_1'].values), adj(d[f'{b}_1'].values)) for a, b in PAIRS])
    o['inc_env_removed'] = np.mean([rho(adj(d[f'{a}_6'].values), adj(d[f'{b}_6'].values)) for a, b in PAIRS])
    o['cause_fraction'] = np.mean([rho(d[f'{a}_frac'], d[f'{b}_frac']) for a, b in PAIRS])
    o['diff_raw'] = o['mort_raw'] - o['inc_raw']; o['diff_env_removed'] = o['mort_env_removed'] - o['inc_raw']
    o['diff_both_removed'] = o['mort_env_removed'] - o['inc_env_removed']; o['diff_cause_fraction'] = o['cause_fraction'] - o['inc_raw']
    return o
rng = np.random.default_rng(20261015); rows = []
for p in P3:
    d = R[R.period == p].dropna().reset_index(drop=True)
    est = stats(d); boot = pd.DataFrame([stats(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(1000)])
    rec = dict(period=p, n=len(d), **{k: round(v, 6) for k, v in est.items()})
    for c in (302, 322, 332): rec[f'rho_env_mort_{c}'] = round(rho(d.env, d[f'{c}_1']), 6)
    for k in ['diff_raw', 'diff_env_removed', 'diff_both_removed', 'diff_cause_fraction']:
        rec[k + '_lo'], rec[k + '_hi'] = np.round(boot[k].quantile([.025, .975]).values, 6)
    rows.append(rec)
O = pd.DataFrame(rows); O.to_csv(H / 'ENVELOPE_CORR.csv', index=False, encoding='utf-8-sig')
W = R.set_index(['location_id', 'period']).unstack('period')
rows = [dict(series=s, rho_early_late=round(rho(W[(s, P3[0])], W[(s, P3[2])]), 6)) for s in ['env'] + [f'{c}_{m}' for c in (302, 322, 332) for m in (1, 6)] + [f'{c}_frac' for c in (302, 322, 332)]]
pd.DataFrame(rows).to_csv(H / 'ENVELOPE_PERSISTENCE.csv', index=False, encoding='utf-8-sig')
print(O.round(3).T.to_string()); print(pd.DataFrame(rows).to_string())

# ---- 4. all other causes as adjustor (symmetric), and within-subregion correlations
m3 = pan[pan.measure_id == 1].groupby(['location_id', 'year']).gbd_rate_per_100k.sum().rename('three').reset_index()
o = env.merge(m3, on=['location_id', 'year']).merge(sdi, on=['location_id', 'year'])
o['y'] = np.log(o.allcause_rate_per_100k - o.three); assert np.isfinite(o.y).all()
R2 = R.merge(period_resid(o, 'y').rename('other').reset_index(), on=['location_id', 'period'])
reg = pd.read_csv(A / 'substantive_findings_20261010' / 'A_PROFILES_WITH_REGION.csv').drop_duplicates('location_id').set_index('location_id').region
R2['region'] = R2.location_id.map(reg)
R2.to_csv(H / 'ENVELOPE_PERIOD_RESIDUALS_WITH_OTHER.csv', index=False, encoding='utf-8-sig')
def stats2(d):
    E = np.column_stack([np.ones(len(d)), d.other.values]); adj = lambda v: v - E @ np.linalg.lstsq(E, v, rcond=None)[0]
    o = {}
    for m, lab in [(1, 'mort'), (6, 'inc')]:
        o[f'{lab}_other_removed'] = np.mean([rho(adj(d[f'{a}_{m}'].values), adj(d[f'{b}_{m}'].values)) for a, b in PAIRS])
        dm = {c: d[f'{c}_{m}'] - d.groupby('region')[f'{c}_{m}'].transform('mean') for c in (302, 322, 332)}
        o[f'{lab}_within_subregion'] = np.mean([rho(dm[a], dm[b]) for a, b in PAIRS])
    o['diff_other_removed'] = o['mort_other_removed'] - o['inc_other_removed']
    o['diff_within_subregion'] = o['mort_within_subregion'] - o['inc_within_subregion']
    return o
rng = np.random.default_rng(20261016); rows = []
for p in P3:
    d = R2[R2.period == p].dropna().reset_index(drop=True)
    est = stats2(d); boot = pd.DataFrame([stats2(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(1000)])
    rec = dict(period=p, n=len(d), rho_other_env=round(rho(d.other, d.env), 6), **{k: round(v, 6) for k, v in est.items()})
    for c in (302, 322, 332): rec[f'rho_other_mort_{c}'] = round(rho(d.other, d[f'{c}_1']), 6)
    for k in ['diff_other_removed', 'diff_within_subregion']:
        rec[k + '_lo'], rec[k + '_hi'] = np.round(boot[k].quantile([.025, .975]).values, 6)
    rows.append(rec)
O2 = pd.DataFrame(rows); O2.to_csv(H / 'ENVELOPE_OTHER_CAUSES.csv', index=False, encoding='utf-8-sig'); print(O2.round(3).T.to_string())
# within-subregion persistence: early-late rank correlation after subtracting subregion means in each period
rows = []
for s in [f'{c}_{m}' for c in (302, 322, 332) for m in (1, 6)]:
    w = R2.assign(dm=R2[s] - R2.groupby(['period', 'region'])[s].transform('mean')).pivot(index='location_id', columns='period', values='dm')
    rows.append(dict(series=s, rho_early_late_within_subregion=round(rho(w[P3[0]], w[P3[2]]), 6)))
pd.DataFrame(rows).to_csv(H / 'ENVELOPE_PERSISTENCE_WITHIN_SUBREGION.csv', index=False, encoding='utf-8-sig'); print(pd.DataFrame(rows).to_string())

# (fifth review) within-subregion contrast after removing other-cause mortality; persistence of other-cause mortality as a benchmark
rows = []
for p_ in P3:
    d = R2[R2.period == p_].dropna().reset_index(drop=True)
    E = np.column_stack([np.ones(len(d)), d.other.values]); adj = lambda v: v - E @ np.linalg.lstsq(E, v, rcond=None)[0]
    for c in (302, 322, 332):
        for m in (1, 6): d[f'a{c}_{m}'] = adj(d[f'{c}_{m}'].values)
    def st(x):
        o = {}
        for m, lab in [(1, 'mort'), (6, 'inc')]:
            dm = {c: x[f'a{c}_{m}'] - x.groupby('region')[f'a{c}_{m}'].transform('mean') for c in (302, 322, 332)}
            o[lab] = np.mean([rho(dm[a], dm[b]) for a, b in PAIRS])
        o['diff'] = o['mort'] - o['inc']; return o
    est = st(d); bt = pd.DataFrame([st(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(1000)])
    rows.append(dict(period=p_, mort_other_removed_within=round(est['mort'], 6), inc_other_removed_within=round(est['inc'], 6), diff=round(est['diff'], 6),
                     diff_lo=round(bt['diff'].quantile(.025), 6), diff_hi=round(bt['diff'].quantile(.975), 6)))
Wd = R2.pivot(index='location_id', columns='period', values='other')
rows.append(dict(period='other-cause early-late rho', mort_other_removed_within=round(rho(Wd[P3[0]], Wd[P3[2]]), 6)))
pd.DataFrame(rows).to_csv(H / 'ENVELOPE_OTHER_WITHIN.csv', index=False, encoding='utf-8-sig'); print(pd.DataFrame(rows).round(3).to_string())
