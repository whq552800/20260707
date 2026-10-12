"""Domain-level trajectories of the factor profiles (2026-10-12, restructuring).
Location-outcome records are grouped by tail membership in 2001-2010 and 2011-2023 (stayed in the high tail; left the high tail;
entered the high tail; stayed in the low tail), with SDI stratum taken from 2001-2010. For each group and domain, the percentage of
record-factor pairs with an adverse departure is computed in 2001-2010 and in 2011-2023 over pairs judged in both periods, and the change
(late minus middle, percentage points). 95% intervals: 2000 resamples of locations (2026-10-12 revision: a resampled location brings all its
records in every group, so records of the same location are not treated as independent).
Seed 20261012. Domains as in profile_domains.py; PCV3 is excluded because it was evaluable in few locations in 2001-2010.
Contrasts of the change between groups (left minus stayed in the high tail; entered the high tail minus stayed in the low tail), with
intervals from the same location resamples, are written to PROFILE_TRAJECTORY_CONTRASTS.csv.
Output: PROFILE_TRAJECTORY_DOMAINS.csv."""
import numpy as np, pandas as pd
D = pd.read_csv('../independent_reproduction_20261009/REPRO_DISEASE_PERIOD.csv')
F = pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv'); T = pd.read_csv('../SDI_PERIOD_TIERS.csv')
BAND = {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '中等SDI': 'Middle and high-middle', '中高SDI': 'Middle and high-middle', '高SDI': 'High'}
DOM = {'Basic services': ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct'], 'Immunisation': ['dtp3_pct', 'hib3_pct'],
       'Nutrition': ['stunting_u5_pct', 'low_birthweight_pct', 'undernourishment_pct'],
       'Health system': ['uhc_service_index', 'health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000'],
       'Air pollution': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']}
JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']; M, L = '2001–2010', '2011–2023'; B = 2000; rng = np.random.default_rng(20261012)
W = D.pivot_table(index=['location_id', 'cause_id', 'measure_id'], columns='period', values='category', aggfunc='first').reset_index()
def grp(r):
    hm, hl = r[M] == '稳健高尾', r[L] == '稳健高尾'
    if hm and hl: return 'Stayed in high tail'
    if hm: return 'Left high tail'
    if hl: return 'Entered high tail'
    if r[M] == '稳健低尾' and r[L] == '稳健低尾': return 'Stayed in low tail'
W['group'] = W.apply(grp, axis=1); W = W.dropna(subset=['group'])
W['band'] = W.location_id.map(T[T.period == M].set_index('location_id').tier.map(BAND))
Fj = F[F.status.isin(JUDGE)].assign(a=lambda x: (x.status == '明确不利').astype(float))
locs = W.location_id.unique(); li = {l: i for i, l in enumerate(locs)}
CNT = np.vstack([np.bincount(rng.integers(0, len(locs), len(locs)), minlength=len(locs)) for _ in range(B)])
rows, BS = [], {}
for dom, fs in DOM.items():
    A = Fj[Fj.factor.isin(fs)].pivot_table(index=['location_id', 'factor'], columns='period', values='a')[[M, L]].dropna().reset_index()
    agg = A.groupby('location_id').agg(sm=(M, 'sum'), sl=(L, 'sum'), n=(M, 'count'))
    for (b, g), s in list(W.groupby(['band', 'group'])) + [(('All strata', g), s) for g, s in W.groupby('group')]:
        x = s.merge(agg, left_on='location_id', right_index=True)
        if len(x) < 3: continue
        t = x.groupby('location_id')[['sm', 'sl', 'n']].sum(); w = CNT[:, [li[l] for l in t.index]]
        pm, pl = 100 * t.sm.sum() / t.n.sum(), 100 * t.sl.sum() / t.n.sum()
        with np.errstate(invalid='ignore', divide='ignore'):
            bm = 100 * (w @ t.sm.values) / (w @ t.n.values); bl = 100 * (w @ t.sl.values) / (w @ t.n.values)
        BS[(b, g, dom)] = bl - bm
        rows.append(dict(band=b, group=g, domain=dom, n_records=len(x), n_locations=len(t), n_pairs=int(t.n.sum()), pct_adv_mid=pm, pct_adv_late=pl, change_pp=pl - pm,
                         change_lo=np.nanpercentile(bl - bm, 2.5), change_hi=np.nanpercentile(bl - bm, 97.5),
                         late_lo=np.nanpercentile(bl, 2.5), late_hi=np.nanpercentile(bl, 97.5)))
R = pd.DataFrame(rows); R.round(2).to_csv('PROFILE_TRAJECTORY_DOMAINS.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
con = []
for (b, g, dom), v in BS.items():
    for g1, g0 in [('Left high tail', 'Stayed in high tail'), ('Entered high tail', 'Stayed in low tail')]:
        if g == g1 and (b, g0, dom) in BS:
            e = R[(R.band == b) & (R.group == g1) & (R.domain == dom)].change_pp.iloc[0] - R[(R.band == b) & (R.group == g0) & (R.domain == dom)].change_pp.iloc[0]
            dd = v - BS[(b, g0, dom)]
            con.append(dict(band=b, contrast=f'{g1} minus {g0}', domain=dom, diff_change_pp=e, lo=np.nanpercentile(dd, 2.5), hi=np.nanpercentile(dd, 97.5)))
Cn = pd.DataFrame(con); Cn.round(2).to_csv('PROFILE_TRAJECTORY_CONTRASTS.csv', index=False, encoding='utf-8-sig')
print(Cn[Cn.band == 'All strata'].round(1).to_string())
