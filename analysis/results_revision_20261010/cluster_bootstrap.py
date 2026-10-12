"""Subregion-cluster bootstrap for the two main interval estimates (2026-10-12, sixth independent review).
GBD borrows strength within regions, so locations in the same United Nations subregion may not be independent. Here whole
subregions are resampled with replacement (2000 resamples, seed 20261018) for:
(1) the outcome-specific factor contrast of S1 Text Table S (equal-weight mean over the comparable outcome-factor pairs of the
    high-minus-low-tail difference in the percentage with an adverse departure, 2011-2023; low and low-middle versus high SDI);
(2) the difference in mean cross-disease Spearman correlation between mortality and incidence residuals, as analysed and after
    removing the residual of under-5 mortality from other causes (gbd_envelope_20261012).
Output: CLUSTER_BOOTSTRAP.csv"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '../independent_reproduction_20261009'); from statlib import rankdata
pr = pd.read_csv('../independent_reproduction_20261009/REPRO_COUNTRY_DISEASE_PROFILES.csv')
F = pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv'); T = pd.read_csv('../SDI_PERIOD_TIERS.csv')
reg = pd.read_csv('../substantive_findings_20261010/A_PROFILES_WITH_REGION.csv').drop_duplicates('location_id').set_index('location_id').region
LATE = '2011–2023'; JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']; NB = 2000; rng = np.random.default_rng(20261018)
REL = {'LRI': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct', 'stunting_u5_pct', 'low_birthweight_pct'],
       '腹泻病': ['basic_water_pct', 'basic_sanitation_pct', 'stunting_u5_pct'], '脑膜炎': ['dtp3_pct', 'hib3_pct', 'pcv3_pct']}
BAND = {'低SDI': 'low', '中低SDI': 'low', '高SDI': 'high'}
TL = T[T.period == LATE].assign(band=lambda d: d.tier.map(BAND))
R = pr[pr.p3_category.isin(['稳健高尾', '稳健低尾'])][['location_id', 'dz', 'p3_category']].copy()
R['tail'] = R.p3_category.map({'稳健高尾': 'high', '稳健低尾': 'low'}); R['band'] = R.location_id.map(TL.set_index('location_id').band)
pairs_all = [(dz, f) for dz in R.dz.unique() for f in REL[dz.replace('发病', '').replace('死亡', '')]]
L = pd.concat([R[R.dz == dz].assign(factor=f) for dz, f in pairs_all], ignore_index=True).merge(F[F.period == LATE][['location_id', 'factor', 'status']], on=['location_id', 'factor'], how='left')
L['evaluable'] = L.status.isin(JUDGE); L['adverse'] = L.status == '明确不利'
def comp(b):
    s = L[L.band == b]
    return {(dz, f) for dz, f in pairs_all if s[(s.dz == dz) & (s.factor == f) & (s['tail'] == 'high')].evaluable.sum() >= 3 and s[(s.dz == dz) & (s.factor == f) & (s['tail'] == 'low')].evaluable.sum() >= 3}
pairs = sorted(comp('low') & comp('high'))
locs = TL[TL.band.isin(['low', 'high'])].location_id.values; ix = {l: i for i, l in enumerate(locs)}
regions = np.array([reg.get(l, 'NA') for l in locs]); ureg = np.unique(regions); rix = np.array([np.where(ureg == r)[0][0] for r in regions])
M = {}
for b in ['low', 'high']:
    A = {t: np.zeros((len(locs), len(pairs))) for t in ('high', 'low')}; E = {t: np.zeros((len(locs), len(pairs))) for t in ('high', 'low')}
    s = L[L.band == b]
    for j, (dz, f) in enumerate(pairs):
        q = s[(s.dz == dz) & (s.factor == f)]
        for t in ('high', 'low'):
            qt = q[q['tail'] == t]; ii = qt.location_id.map(ix).values; A[t][ii, j] = qt.adverse.values; E[t][ii, j] = qt.evaluable.values
    M[b] = (A, E)
def grad(w):
    out = {}
    for b in ['low', 'high']:
        A, E = M[b]
        with np.errstate(invalid='ignore', divide='ignore'):
            out[b] = np.nanmean(100 * (w @ A['high']) / (w @ E['high']) - 100 * (w @ A['low']) / (w @ E['low']), axis=-1)
    return out['low'], out['high'], out['low'] - out['high']
est = grad(np.ones(len(locs)))
W = np.array([np.bincount(rng.integers(0, len(ureg), len(ureg)), minlength=len(ureg))[rix] for _ in range(NB)]).astype(float)
bt = grad(W)
rows = [dict(estimate_set='factor contrast, 2011-2023', quantity=q, n_pairs=len(pairs), n_clusters=len(ureg), estimate=e, lo=np.nanquantile(x, .025), hi=np.nanquantile(x, .975))
        for q, e, x in zip(['low and low-middle SDI', 'high SDI', 'difference between strata'], est, bt)]
# (2) correlations
Rp = pd.read_csv('../gbd_envelope_20261012/ENVELOPE_PERIOD_RESIDUALS_WITH_OTHER.csv')
PAIRS = [(302, 322), (302, 332), (322, 332)]
def rho(x, y): return np.corrcoef(rankdata(x), rankdata(y))[0, 1]
for p in ['1990–2000', '2001–2010', '2011–2023']:
    d = Rp[Rp.period == p].drop(columns='region').dropna().reset_index(drop=True); d['region'] = d.location_id.map(reg).fillna('NA'); ur = d.region.unique()
    def st(dd):
        Em = np.column_stack([np.ones(len(dd)), dd.other.values]); adj = lambda v: v - Em @ np.linalg.lstsq(Em, v, rcond=None)[0]
        raw = np.mean([rho(dd[f'{a}_1'], dd[f'{b}_1']) for a, b in PAIRS]) - np.mean([rho(dd[f'{a}_6'], dd[f'{b}_6']) for a, b in PAIRS])
        oth = np.mean([rho(adj(dd[f'{a}_1'].values), adj(dd[f'{b}_1'].values)) for a, b in PAIRS]) - np.mean([rho(adj(dd[f'{a}_6'].values), adj(dd[f'{b}_6'].values)) for a, b in PAIRS])
        return raw, oth
    e0 = st(d); grp = {r: d.index[d.region == r].values for r in ur}
    B = np.array([st(d.loc[np.concatenate([grp[r] for r in rng.choice(ur, len(ur))])].reset_index(drop=True)) for _ in range(1000)])
    for k, lab in enumerate(['mortality minus incidence, as analysed', 'mortality minus incidence, other causes removed']):
        rows.append(dict(estimate_set=f'cross-disease correlation, {p}', quantity=lab, n_clusters=len(ur), estimate=e0[k], lo=np.quantile(B[:, k], .025), hi=np.quantile(B[:, k], .975)))
O = pd.DataFrame(rows); O[['estimate', 'lo', 'hi']] = O[['estimate', 'lo', 'hi']].round(6)
O.to_csv('CLUSTER_BOOTSTRAP.csv', index=False, encoding='utf-8-sig'); print(O.round(3).to_string())
