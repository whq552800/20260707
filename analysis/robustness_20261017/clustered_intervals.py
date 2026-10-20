"""2026-10-17: intervals that respect dependence between outcomes of the same location (response to an external critique).

(1) Domain profiles pooled over the three mortality or three incidence outcomes (main text, Fig 3B): the earlier intervals
    resampled locations independently in each stratum-outcome cell, which ignores that the same location appears in the tails
    of several outcomes. Here locations are resampled with replacement within each SDI stratum, and every resampled location
    brings its tail memberships for all outcomes and its factor departures. Point estimates are unchanged.
(2) Retention of early tail records (Fig 2C): the Wilson interval treated location-outcome records as independent; here locations
    are resampled, each bringing all its records.
(3) Change in the spread of period residuals between 1990-2000 and 2011-2023 (rank stability does not show whether the size of
    departures shrank): interquartile range (IQR) and SD of the period residuals of all locations, by outcome, with locations
    resampled and the SDI-year reference refitted in every replicate (each location brings all its years).
2000 replicates for (1) and (2), 1000 for (3); percentile intervals; seed 20261017.
Outputs: PROFILE_DOMAINS_POOLED_CLUSTERED.csv, TAIL_RETENTION_CLUSTERED.csv, DISPERSION_CHANGE.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
sys.path.insert(0, str(A / 'independent_reproduction_20261009'))
from paths import PANEL
rng = np.random.default_rng(20261017)
D0 = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
OUT = {(302, 6): 'Diarrhoeal incidence', (302, 1): 'Diarrhoeal mortality', (322, 6): 'LRI incidence', (322, 1): 'LRI mortality', (332, 6): 'Meningitis incidence', (332, 1): 'Meningitis mortality'}

# ---------------- (1) domain profiles, location bootstrap within stratum
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv'); T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')
BAND = {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '中等SDI': 'Middle and high-middle', '中高SDI': 'Middle and high-middle', '高SDI': 'High'}
DOM = {'Basic services': ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct'], 'Immunisation': ['dtp3_pct', 'hib3_pct', 'pcv3_pct'],
       'Nutrition': ['stunting_u5_pct', 'low_birthweight_pct', 'undernourishment_pct'],
       'Health system': ['uhc_service_index', 'health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000', 'physicians_per1000'],
       'Air pollution': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']}
FAC = [f for v in DOM.values() for f in v]; P = '2011–2023'
JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']
D = D0[D0.period == P].merge(T[T.period == P][['location_id', 'tier']], on='location_id'); D['band'] = D.tier.map(BAND)
Fe = F[(F.period == P) & F.status.isin(JUDGE) & F.factor.isin(FAC)]
ADV = Fe.assign(a=(Fe.status == '明确不利').astype(float)).pivot(index='location_id', columns='factor', values='a').reindex(columns=FAC)
B1 = 2000
def cell(cat, adv):
    """cat: (n,) categories of the sampled locations for one outcome; adv: (n, factors). Returns diff (factors,) or NaN."""
    h, l = adv[cat == '稳健高尾'], adv[cat == '稳健低尾']
    nh, nl = (~np.isnan(h)).sum(0), (~np.isnan(l)).sum(0)
    with np.errstate(all='ignore'):
        d = 100 * (np.nansum(h, 0) / nh - np.nansum(l, 0) / nl)
    d[(nh < 3) | (nl < 3)] = np.nan
    if len(h) < 3 or len(l) < 3: d[:] = np.nan
    return d
rows = []
for b in dict.fromkeys(BAND.values()):
    x = D[D.band == b]; locs = np.array(sorted(x.location_id.unique()))
    CAT = x.pivot(index='location_id', columns=['cause_id', 'measure_id'], values='category').reindex(locs)
    AD = ADV.reindex(locs).values
    for t, mm in [('Mortality', 1), ('Incidence', 6)]:
        outs = [(c, mm) for c in (302, 322, 332)]
        def stat(ix):
            dd = np.array([cell(CAT[o].values[ix], AD[ix]) for o in outs])   # outcomes x factors
            res = {}
            for dom, fs in DOM.items():
                k = [FAC.index(f) for f in fs]
                with np.errstate(all='ignore'):
                    per_out = np.nanmean(dd[:, k], 1)            # domain mean within each outcome cell
                    res[dom] = np.nanmean(per_out)               # mean over outcomes
            return res
        est = stat(np.arange(len(locs)))
        bs = [stat(rng.integers(0, len(locs), len(locs))) for _ in range(B1)]
        for dom in DOM:
            r = np.array([z[dom] for z in bs], float)
            rows.append(dict(band=b, outcome_type=t, domain=dom, diff_pp=est[dom], lo=np.nanpercentile(r, 2.5), hi=np.nanpercentile(r, 97.5)))
PD = pd.DataFrame(rows); PD.round(2).to_csv(H / 'PROFILE_DOMAINS_POOLED_CLUSTERED.csv', index=False)
old = pd.read_csv(A / 'results_revision_20261010' / 'PROFILE_DOMAINS_POOLED.csv')
cmp = PD.merge(old, on=['band', 'outcome_type', 'domain'], suffixes=('', '_old'))
assert np.allclose(cmp.diff_pp, cmp.diff_pp_old, atol=0.01), 'point estimates differ from the main analysis'
print(cmp[['band', 'outcome_type', 'domain', 'diff_pp', 'lo', 'hi', 'lo_old', 'hi_old']].round(1).to_string())

# ---------------- (2) tail retention, location bootstrap
W = D0.pivot_table(index=['location_id', 'cause_id', 'measure_id'], columns='period', values='category', aggfunc='first').reset_index()
locs = np.array(sorted(W.location_id.unique())); g = {l: W[W.location_id == l] for l in locs}
def ret(ix, tail):
    s = pd.concat([g[locs[i]] for i in ix]); s = s[s['1990–2000'] == tail]
    return (s['2011–2023'] == tail).mean(), len(s)
rr = []
for tail, nm in [('稳健高尾', 'High tail'), ('稳健低尾', 'Low tail')]:
    e, n = ret(np.arange(len(locs)), tail)
    bs = [ret(rng.integers(0, len(locs), len(locs)), tail)[0] for _ in range(B1)]
    rr.append(dict(tail=nm, records=n, retained_pct=100 * e, lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5)))
RT = pd.DataFrame(rr); RT.round(2).to_csv(H / 'TAIL_RETENTION_CLUSTERED.csv', index=False); print(RT.round(1).to_string())

# ---------------- (3) dispersion change with refit
d = pd.read_csv(PANEL); d = d[d.rate_valid & (d.gbd_rate_per_100k > 0)].copy(); d['y'] = np.log(d.gbd_rate_per_100k)
def ns_basis(x, knots, bk=(0.0, 1.0)):
    k = np.r_[bk[0], knots, bk[1]]; K = len(k)
    def dk(j): return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])
    return np.column_stack([x] + [dk(j) - dk(K - 2) for j in range(K - 2)])
knots = np.quantile(d.drop_duplicates(['location_id', 'year']).sdi, [0.25, 0.5, 0.75])
PER = {'1990–2000': (1990, 2000), '2011–2023': (2011, 2023)}
iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)
out = []; B3 = 1000
for (c, m), oname in OUT.items():
    gg = d[(d.cause_id == c) & (d.measure_id == m)].sort_values(['location_id', 'year']).reset_index(drop=True)
    Bm = ns_basis(gg.sdi.values, knots); tc = ((gg.year - 2006.5) / 33).values[:, None]
    YF = pd.get_dummies(gg.year, prefix='y', drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(gg)), YF, Bm, Bm * tc]); y = gg.y.values
    L = gg.location_id.values; ul = np.unique(L); rows_of = {l: np.where(L == l)[0] for l in ul}
    early = (gg.year <= 2000).values; late = (gg.year >= 2011).values
    def disp(lsel):
        idx = np.concatenate([rows_of[l] for l in lsel]); inst = np.concatenate([[i] * len(rows_of[l]) for i, l in enumerate(lsel)])
        bb, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None); e = y[idx] - X[idx] @ bb
        res = {}
        for pn, msk in (('early', early[idx]), ('late', late[idx])):
            pr = pd.Series(e[msk]).groupby(inst[msk]).mean().values
            res[pn] = (iqr(pr), pr.std(ddof=1))
        return res
    e0 = disp(ul)
    bs = [disp(rng.choice(ul, len(ul), replace=True)) for _ in range(B3)]
    for j, nm in [(0, 'IQR'), (1, 'SD')]:
        de = e0['late'][j] - e0['early'][j]; db = np.array([z['late'][j] - z['early'][j] for z in bs])
        rt = np.array([z['late'][j] / z['early'][j] for z in bs])
        out.append(dict(outcome=oname, measure=nm, early=e0['early'][j], late=e0['late'][j], change=de, change_lo=np.percentile(db, 2.5), change_hi=np.percentile(db, 97.5),
                        ratio=e0['late'][j] / e0['early'][j], ratio_lo=np.percentile(rt, 2.5), ratio_hi=np.percentile(rt, 97.5)))
DC = pd.DataFrame(out); DC.round(3).to_csv(H / 'DISPERSION_CHANGE.csv', index=False); print(DC.round(2).to_string())
