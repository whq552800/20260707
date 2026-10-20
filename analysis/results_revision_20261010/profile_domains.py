"""Domain summary of the factor profiles (2026-10-12, restructuring).
For each SDI stratum and outcome in 2011-2023, the difference between high-tail and low-tail locations in the percentage with an adverse
departure is averaged with equal weight over the evaluable factors of a domain (at least 3 evaluable locations in each tail).
Domains: basic services (drinking water, sanitation, clean cooking fuels); immunisation (DTP3, Hib3, PCV3); nutrition (stunting,
low birthweight, undernourishment); health system (UHC service coverage index, health expenditure per person, hospital beds, nurses and
midwives); air pollution (ambient PM2.5, second-hand smoke). Also averaged over the three mortality and the three incidence outcomes.
95% intervals: 2000 replicates resampling high-tail and low-tail locations with replacement within each stratum-outcome cell
(the same resampled locations for every factor in that cell); percentile intervals. Seed 20261012.
Output: PROFILE_DOMAINS.csv (cell level) and PROFILE_DOMAINS_POOLED.csv (stratum x outcome type x domain)."""
import numpy as np, pandas as pd
D = pd.read_csv('../independent_reproduction_20261009/REPRO_DISEASE_PERIOD.csv')
F = pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv'); T = pd.read_csv('../SDI_PERIOD_TIERS.csv')
BAND = {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '中等SDI': 'Middle and high-middle', '中高SDI': 'Middle and high-middle', '高SDI': 'High'}
OUT = {(302, 6): 'Diarrhoeal incidence', (302, 1): 'Diarrhoeal mortality', (322, 6): 'LRI incidence', (322, 1): 'LRI mortality', (332, 6): 'Meningitis incidence', (332, 1): 'Meningitis mortality'}
DOM = {'Basic services': ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct'], 'Immunisation': ['dtp3_pct', 'hib3_pct', 'pcv3_pct'],
       'Nutrition': ['stunting_u5_pct', 'low_birthweight_pct', 'undernourishment_pct'],
       'Health system': ['uhc_service_index', 'health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000', 'physicians_per1000'],
       'Air pollution': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']}
FAC = [f for v in DOM.values() for f in v]
BANDS = list(dict.fromkeys(BAND.values()))
JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']; P = '2011–2023'; B = 2000; rng = np.random.default_rng(20261012)
D = D[D.period == P].merge(T[T.period == P][['location_id', 'tier']], on='location_id'); D['band'] = D.tier.map(BAND)
Fe = F[(F.period == P) & F.status.isin(JUDGE) & F.factor.isin(FAC)]
ADV = Fe.assign(a=(Fe.status == '明确不利').astype(float)).pivot(index='location_id', columns='factor', values='a').reindex(columns=FAC)

def cell_diffs(H, L):
    """H, L: arrays (replicates x locations x factors) of adverse indicators with NaN; returns (replicates x factors) diff in pp, NaN if <3."""
    nh, nl = (~np.isnan(H)).sum(1), (~np.isnan(L)).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = 100 * (np.nansum(H, 1) / nh - np.nansum(L, 1) / nl)
    d[(nh < 3) | (nl < 3)] = np.nan; return d

rows, reps = [], {}
for (c, m), o in OUT.items():
    for b in BANDS:
        x = D[(D.cause_id == c) & (D.measure_id == m) & (D.band == b)]
        hi = ADV.reindex(x[x.category == '稳健高尾'].location_id).values; lo = ADV.reindex(x[x.category == '稳健低尾'].location_id).values
        if len(hi) < 3 or len(lo) < 3: continue
        est = cell_diffs(hi[None], lo[None])[0]
        ih = rng.integers(0, len(hi), (B, len(hi))); il = rng.integers(0, len(lo), (B, len(lo)))
        bd = cell_diffs(hi[ih], lo[il])
        for dom, fs in DOM.items():
            k = [FAC.index(f) for f in fs]
            e = np.nanmean(est[k]) if np.isfinite(est[k]).any() else np.nan
            with np.errstate(all='ignore'):
                r = np.nanmean(bd[:, k], 1)
            reps[(b, o, dom)] = r
            rows.append(dict(band=b, outcome=o, outcome_type='Mortality' if m == 1 else 'Incidence', domain=dom, n_high=len(hi), n_low=len(lo),
                             n_factors=int(np.isfinite(est[k]).sum()), diff_pp=e, lo=np.nanpercentile(r, 2.5), hi=np.nanpercentile(r, 97.5)))
C = pd.DataFrame(rows); C.round(2).to_csv('PROFILE_DOMAINS.csv', index=False, encoding='utf-8-sig')
pool = []
for b in BANDS:
    for t in ['Mortality', 'Incidence']:
        for dom in DOM:
            ks = [k for k in reps if k[0] == b and k[2] == dom and C[(C.band == b) & (C.outcome == k[1]) & (C.domain == dom)].outcome_type.iloc[0] == t]
            e = C[(C.band == b) & (C.outcome_type == t) & (C.domain == dom)].diff_pp.mean()
            with np.errstate(all='ignore'):
                r = np.nanmean(np.vstack([reps[k] for k in ks]), 0)
            pool.append(dict(band=b, outcome_type=t, domain=dom, diff_pp=e, lo=np.nanpercentile(r, 2.5), hi=np.nanpercentile(r, 97.5)))
Pq = pd.DataFrame(pool); Pq.round(2).to_csv('PROFILE_DOMAINS_POOLED.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(C.pivot_table(index=['band', 'outcome'], columns='domain', values='diff_pp').round(0).to_string())
print(Pq.assign(s=Pq.apply(lambda r: f'{r.diff_pp:.0f} ({r.lo:.0f},{r.hi:.0f})', axis=1)).pivot_table(index=['band', 'outcome_type'], columns='domain', values='s', aggfunc='first').to_string())
