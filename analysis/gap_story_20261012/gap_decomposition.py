"""Low-SDI versus high-SDI gaps in under-5 incidence, mortality and case fatality, 1990-2023, with uncertainty intervals (2026-10-12).
Groups: GBD 2023 SDI tiers (low; high) assigned once from the mean SDI of 1990-2000 and held fixed ("early-fixed", main analysis);
groups assigned from the mean SDI of 2011-2023 and held fixed ("late-fixed") are a supplementary comparison.
Group rate = population-weighted rate = sum(cases or deaths) / sum(under-5 population). Case fatality = deaths per 1000 incident cases.
Rate ratio (RR) = low-group rate / high-group rate; relative change in the gap = RR(2023) / RR(1990); absolute gap (RD) = low - high.
Intervals (95%): 2000 replicates, each (i) resampling locations with replacement within each group, the same locations for every year,
and (ii) perturbing each location-year rate on the log scale by z x SE, with SE = (ln upper - ln lower) / 3.92 from the GBD 95% UI and
z drawn independently for each location, year and outcome (independence between years and between incidence and deaths is conservative
for changes and for case fatality). Percentile intervals. Seed 20261012.
Outputs: GAP_ANNUAL.csv (year-by-year RR with interval for each design and outcome, including case fatality),
GAP_SUMMARY.csv (1990 and 2023 rates, RR, RD, change in RR, group declines, with intervals)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
A = Path(__file__).resolve().parent.parent; HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(A / 'independent_reproduction_20261009'), str(A)]
from paths import PANEL
B = 2000; rng = np.random.default_rng(20261012)
d = pd.read_csv(PANEL, usecols=['location_id', 'cause_id', 'measure_id', 'year', 'under5_population', 'gbd_rate_per_100k',
                                'gbd_rate_lower_per_100k', 'gbd_rate_upper_per_100k'])
T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')
DESIGNS = {'early-fixed': T[T.period == '1990–2000'].set_index('location_id').tier, 'late-fixed': T[T.period == '2011–2023'].set_index('location_id').tier}
DZ = {302: 'Diarrhoeal disease', 322: 'LRI', 332: 'Meningitis'}
YEARS = np.arange(1990, 2024)

def arrays(c, m, locs):
    x = d[(d.cause_id == c) & (d.measure_id == m) & d.location_id.isin(locs)]
    r = x.pivot(index='location_id', columns='year', values='gbd_rate_per_100k').reindex(index=locs, columns=YEARS)
    lo = x.pivot(index='location_id', columns='year', values='gbd_rate_lower_per_100k').reindex(index=locs, columns=YEARS).clip(lower=1e-12)
    hi = x.pivot(index='location_id', columns='year', values='gbd_rate_upper_per_100k').reindex(index=locs, columns=YEARS).clip(lower=1e-12)
    p = x.pivot(index='location_id', columns='year', values='under5_population').reindex(index=locs, columns=YEARS)
    se = ((np.log(hi) - np.log(lo)) / 3.92).fillna(0).clip(lower=0)
    return r.values, se.values, p.values

def group_counts(r, se, p, idx, z):
    """r, se, p: locations x years. idx: replicate x n resampled row indices; z: replicate x n x years. Returns cases (rep x years), pop."""
    rr = r[idx] * np.exp(z * se[idx]) if z is not None else r[idx]
    return (rr * p[idx] / 1e5).sum(axis=1), p[idx].sum(axis=1)

def q(a, axis=0): return np.nanpercentile(a, 2.5, axis=axis), np.nanpercentile(a, 97.5, axis=axis)

annual, summ = [], []
for des, tier in DESIGNS.items():
    G = {g: tier[tier == g].index.values for g in ['低SDI', '高SDI']}
    idx = {g: rng.integers(0, len(v), size=(B, len(v))) for g, v in G.items()}
    for c in DZ:
        est, rep = {}, {}
        for m in (1, 6):
            for g, locs in G.items():
                r, se, p = arrays(c, m, locs)
                n0, p0 = (r * p / 1e5).sum(0), p.sum(0)
                est[(m, g)] = (n0, p0)
                z = rng.standard_normal((B, len(locs), len(YEARS)))
                rep[(m, g)] = group_counts(r, se, p, idx[g], z)
        for lab, f in [('Mortality', lambda E, g: 1e5 * E[(1, g)][0] / E[(1, g)][1]),
                       ('Incidence', lambda E, g: 1e5 * E[(6, g)][0] / E[(6, g)][1]),
                       ('Case fatality', lambda E, g: 1e3 * E[(1, g)][0] / E[(6, g)][0])]:
            lo0, hi0 = f(est, '低SDI'), f(est, '高SDI'); lob, hib = f(rep, '低SDI'), f(rep, '高SDI')
            RR0, RRb = lo0 / hi0, lob / hib
            ql, qh = q(RRb)
            for k, y in enumerate(YEARS):
                annual.append(dict(design=des, disease=DZ[c], measure=lab, year=y, low=lo0[k], high=hi0[k], RR=RR0[k], RR_lo=ql[k], RR_hi=qh[k]))
            i0, i1 = 0, len(YEARS) - 1
            rec = dict(design=des, disease=DZ[c], measure=lab, n_low=len(G['低SDI']), n_high=len(G['高SDI']))
            for nm, v0, vb in [('low_1990', lo0[i0], lob[:, i0]), ('low_2023', lo0[i1], lob[:, i1]), ('high_1990', hi0[i0], hib[:, i0]), ('high_2023', hi0[i1], hib[:, i1]),
                               ('RR_1990', RR0[i0], RRb[:, i0]), ('RR_2023', RR0[i1], RRb[:, i1]), ('RR_change', RR0[i1] / RR0[i0], RRb[:, i1] / RRb[:, i0]),
                               ('RD_1990', lo0[i0] - hi0[i0], lob[:, i0] - hib[:, i0]), ('RD_2023', lo0[i1] - hi0[i1], lob[:, i1] - hib[:, i1]),
                               ('decline_low_pct', 100 * (1 - lo0[i1] / lo0[i0]), 100 * (1 - lob[:, i1] / lob[:, i0])),
                               ('decline_high_pct', 100 * (1 - hi0[i1] / hi0[i0]), 100 * (1 - hib[:, i1] / hib[:, i0]))]:
                a, b = q(vb); rec[nm] = v0; rec[nm + '_lo'] = a; rec[nm + '_hi'] = b
            rec['P_RR_change_gt1'] = float((RRb[:, i1] / RRb[:, i0] > 1).mean())
            summ.append(rec)
pd.DataFrame(annual).to_csv(HERE / 'GAP_ANNUAL.csv', index=False, encoding='utf-8-sig')
S = pd.DataFrame(summ); S.to_csv(HERE / 'GAP_SUMMARY.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
for _, r in S.iterrows():
    print(f"{r.design:11s} {r.disease:18s} {r.measure:13s} RR {r.RR_1990:7.1f} ({r.RR_1990_lo:.1f}-{r.RR_1990_hi:.1f}) -> {r.RR_2023:7.1f} ({r.RR_2023_lo:.1f}-{r.RR_2023_hi:.1f}); "
          f"x{r.RR_change:.2f} ({r.RR_change_lo:.2f}-{r.RR_change_hi:.2f}); decline low {r.decline_low_pct:.0f} ({r.decline_low_pct_lo:.0f}-{r.decline_low_pct_hi:.0f}) high {r.decline_high_pct:.0f} ({r.decline_high_pct_lo:.0f}-{r.decline_high_pct_hi:.0f})")
