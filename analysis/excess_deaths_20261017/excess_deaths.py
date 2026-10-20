"""2026-10-17: under-5 deaths in excess of the SDI–year reference (route 3 after the second desk rejection).

Expected deaths in a location–year = reference rate (exp of the fitted log rate of the main SDI–year reference model,
the same model that defines residuals, tails and reference ratios) × population younger than 5 years.
Excess deaths = observed GBD deaths − expected deaths. The reference is the average burden of locations at the same SDI
in the same year, not the lowest achievable burden, so excess is a shortfall against development peers, not a count of
preventable deaths.

Quantities (mortality outcomes of diarrhoeal disease, LRI and meningitis):
  * excess deaths in 2023 among locations persistently in the high tail for that outcome (same tail in 1990–2000,
    2001–2010 and 2011–2023; membership taken from the main classification and held fixed);
  * the same summed over 2011–2023;
  * excess in 2023 summed over all locations whose deaths exceeded the reference (positive part only), as context;
  * the persistent-tail excess in 2023 without Nigeria (location 214), which contributes about three quarters of it;
  * share of all under-5 deaths from each disease in 2023 that occurred in persistent high-tail locations.
Intervals: 2000 draws combining (i) the GBD uncertainty of each location–year rate (log-normal, scaled to the 95% UI,
as in the gap analysis) and (ii) the uncertainty of the reference model (coefficients drawn from a normal distribution
with the location-clustered CR1 covariance). Seed 20261017.
Outputs: EXCESS_SUMMARY.csv, EXCESS_BY_LOCATION_2023.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'independent_reproduction_20261009'))
from paths import PANEL

d = pd.read_csv(PANEL)
d = d[d.rate_valid & (d.gbd_rate_per_100k > 0)].copy()
d['y'] = np.log(d.gbd_rate_per_100k)
DZ = {302: 'Diarrhoeal disease', 322: 'LRI', 332: 'Meningitis'}

def ns_basis(x, knots, bk=(0.0, 1.0)):   # identical to repro_disease.py
    k = np.r_[bk[0], knots, bk[1]]; K = len(k)
    def dk(j): return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])
    return np.column_stack([x] + [dk(j) - dk(K - 2) for j in range(K - 2)])

knots = np.quantile(d.drop_duplicates(['location_id', 'year']).sdi, [0.25, 0.5, 0.75])
P = pd.read_csv(H.parent / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
AF = pd.read_csv(H.parent / 'independent_reproduction_20261009' / 'REPRO_ANNUAL_FIT.csv')
rng = np.random.default_rng(20261017); NDRAW = 2000; Z = 1.959964
# R4 (2026-10-18): the 2011-2023 sum also gets an interval with GBD errors fully correlated over years within a location
# (one standard-normal deviate per location and draw), from a separate random stream so that the other results are unchanged;
# independent errors across years understate the uncertainty of a multi-year sum.
rng2 = np.random.default_rng(20261018)

summ, byloc = [], []
for c in (302, 322, 332):
    g = d[(d.cause_id == c) & (d.measure_id == 1)].sort_values(['location_id', 'year']).reset_index(drop=True)
    B = ns_basis(g.sdi.values, knots); tc = ((g.year - 2006.5) / 33).values[:, None]
    YF = pd.get_dummies(g.year, prefix='y', drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(g)), YF, B, B * tc]); y = g.y.values
    XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b; n, k = X.shape
    G = g.location_id.nunique(); meat = np.zeros((k, k))
    for _, idx in g.groupby('location_id').indices.items():
        s = X[idx].T @ e[idx]; meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    # check against the main analysis
    chk = g[['location_id', 'year']].assign(f=X @ b).merge(AF[(AF.cause_id == c) & (AF.measure_id == 1)], on=['location_id', 'year'])
    assert np.allclose(chk.f, chk.fitted_log_rate, atol=1e-8), 'reference fit differs from the main analysis'
    pop = g.under5_population.values
    obs = g.gbd_number.values
    sdlog = (np.log(g.gbd_rate_upper_per_100k.clip(lower=1e-12)) - np.log(g.gbd_rate_lower_per_100k.clip(lower=1e-12))).values / (2 * Z)
    exp_ = np.exp(X @ b) / 1e5 * pop
    cat = P[(P.cause_id == c) & (P.measure_id == 1)].pivot(index='location_id', columns='period', values='category')
    pers = set(cat.index[(cat == '稳健高尾').all(axis=1)]); globals().setdefault('PERS', []).append(pers)
    hi23 = set(cat.index[cat['2011–2023'] == '稳健高尾'])
    m23 = (g.year == 2023).values; mlate = (g.year >= 2011).values
    mp = g.location_id.isin(pers).values
    mng = (g.location_id == 214).values   # Nigeria, which contributes most of the excess
    li = g.location_id.astype('category').cat.codes.values; nl = li.max() + 1
    def stats(o, ex):
        exc = o - ex
        return dict(pers_2023=exc[m23 & mp].sum(), pers_late=exc[mlate & mp].sum(), pers_2023_exNGA=exc[m23 & mp & ~mng].sum(),
                    pos_2023=np.clip(exc[m23], 0, None).sum(), deaths_2023=o[m23].sum(), pers_deaths_2023=o[m23 & mp].sum(),
                    pers_exp_2023=ex[m23 & mp].sum())
    point = stats(obs, exp_)
    L = np.linalg.cholesky(V + 1e-12 * np.eye(k))
    draws = []
    for _ in range(NDRAW):
        bd = b + L @ rng.standard_normal(k)
        od = obs * np.exp(rng.standard_normal(len(g)) * sdlog)
        dd_ = stats(od, np.exp(X @ bd) / 1e5 * pop)
        oc = obs * np.exp(rng2.standard_normal(nl)[li] * sdlog)
        dd_['pers_late_corr'] = (oc - np.exp(X @ bd) / 1e5 * pop)[mlate & mp].sum()
        draws.append(dd_)
    D = pd.DataFrame(draws)
    point['pers_late_corr'] = point['pers_late']
    for key, v in point.items():
        lo, hi = np.percentile(D[key], [2.5, 97.5])
        summ.append(dict(disease=DZ[c], quantity=key, estimate=v, lower=lo, upper=hi, n_persistent=len(pers)))
    sh = D.pers_deaths_2023 / D.deaths_2023
    summ.append(dict(disease=DZ[c], quantity='share_deaths_2023_in_persistent', estimate=point['pers_deaths_2023'] / point['deaths_2023'],
                     lower=np.percentile(sh, 2.5), upper=np.percentile(sh, 97.5), n_persistent=len(pers)))
    sx = D.pers_2023 / D.pers_deaths_2023
    summ.append(dict(disease=DZ[c], quantity='excess_share_of_deaths_in_persistent_2023', estimate=point['pers_2023'] / point['pers_deaths_2023'],
                     lower=np.percentile(sx, 2.5), upper=np.percentile(sx, 97.5), n_persistent=len(pers)))
    gl = g[m23].assign(expected=exp_[m23], excess=obs[m23] - exp_[m23], persistent_high=lambda t: t.location_id.isin(pers),
                       high_2011_2023=lambda t: t.location_id.isin(hi23))
    byloc.append(gl[['location_id', 'location', 'cause_id', 'sdi', 'under5_population', 'gbd_number', 'expected', 'excess', 'persistent_high', 'high_2011_2023']].assign(disease=DZ[c]))
    totals = D[['pers_2023', 'pers_late', 'pos_2023', 'pers_2023_exNGA', 'deaths_2023', 'pers_deaths_2023', 'pers_late_corr']]
    globals().setdefault('TOT', []).append(totals.values)
# all three diseases combined (same draw index across diseases; draws are independent between diseases)
T = np.sum(TOT, axis=0)
for j, key in enumerate(['pers_2023', 'pers_late', 'pos_2023', 'pers_2023_exNGA']):
    est = sum(r['estimate'] for r in summ if r['quantity'] == key)
    summ.append(dict(disease='All three', quantity=key, estimate=est, lower=np.percentile(T[:, j], 2.5), upper=np.percentile(T[:, j], 97.5)))
tp = sum(r['estimate'] for r in summ if r['quantity'] == 'pers_2023' and r['disease'] != 'All three'); td = sum(r['estimate'] for r in summ if r['quantity'] == 'deaths_2023' and r['disease'] != 'All three')
tx = sum(r['estimate'] for r in summ if r['quantity'] == 'pers_2023_exNGA' and r['disease'] != 'All three')
summ.append(dict(disease='All three', quantity='share_of_all_deaths_2023', estimate=tp / td, lower=np.percentile(T[:, 0] / T[:, 4], 2.5), upper=np.percentile(T[:, 0] / T[:, 4], 97.5)))
ng = T[:, 0] - T[:, 3]
pdt = sum(r['estimate'] for r in summ if r['quantity'] == 'pers_deaths_2023' and r['disease'] != 'All three')
summ.append(dict(disease='All three', quantity='share_deaths_2023_in_persistent', estimate=pdt / td, lower=np.percentile(T[:, 5] / T[:, 4], 2.5), upper=np.percentile(T[:, 5] / T[:, 4], 97.5)))
summ.append(dict(disease='All three', quantity='nigeria_2023', estimate=tp - tx, lower=np.percentile(ng, 2.5), upper=np.percentile(ng, 97.5)))
summ.append(dict(disease='All three', quantity='pers_late_corr', estimate=sum(r['estimate'] for r in summ if r['quantity'] == 'pers_late' and r['disease'] != 'All three'),
                 lower=np.percentile(T[:, 6], 2.5), upper=np.percentile(T[:, 6], 97.5)))
summ.append(dict(disease='All three', quantity='n_locations_persistent_any_mortality', estimate=len(set().union(*PERS)), lower=np.nan, upper=np.nan))
S = pd.DataFrame(summ); S.to_csv(H / 'EXCESS_SUMMARY.csv', index=False)
BL = pd.concat(byloc); BL.to_csv(H / 'EXCESS_BY_LOCATION_2023.csv', index=False)
pd.set_option('display.width', 200); print(S.to_string(float_format=lambda v: f'{v:,.3f}' if abs(v) < 2 else f'{v:,.0f}'))
for dz, q in BL[BL.persistent_high].groupby('disease'):
    print(dz, q.sort_values('excess', ascending=False).head(8)[['location', 'excess', 'gbd_number']].round(0).to_string(index=False))

# Check: Nigeria's 2023 excess with the reference refitted without Nigeria (does Nigeria pull its own reference?)
rows = []
for c in (302, 322, 332):
    g = d[(d.cause_id == c) & (d.measure_id == 1)].sort_values(['location_id', 'year']).reset_index(drop=True)
    B = ns_basis(g.sdi.values, knots); tc = ((g.year - 2006.5) / 33).values[:, None]
    YF = pd.get_dummies(g.year, prefix='y', drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(g)), YF, B, B * tc]); y = g.y.values; m = (g.location_id != 214).values
    b1 = np.linalg.pinv(X[m].T @ X[m]) @ X[m].T @ y[m]; b0 = np.linalg.pinv(X.T @ X) @ X.T @ y
    k = ((g.location_id == 214) & (g.year == 2023)).values; o = g.gbd_number.values[k][0]; pop = g.under5_population.values[k][0]
    rows.append(dict(disease=DZ[c], observed=o, excess_main=o - np.exp(X[k] @ b0)[0] / 1e5 * pop, excess_without_nigeria_in_reference=o - np.exp(X[k] @ b1)[0] / 1e5 * pop))
pd.DataFrame(rows).to_csv(H / 'EXCESS_NIGERIA_LOO.csv', index=False); print(pd.DataFrame(rows).round(0).to_string(index=False))
