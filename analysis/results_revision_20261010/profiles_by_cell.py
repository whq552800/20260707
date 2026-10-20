"""Factor profiles of high-tail and low-tail locations for each SDI stratum, disease and outcome (2026-10-12, restructuring).
1. PROFILE_CELLS.csv: for every period, SDI stratum (low and low-middle; middle and high-middle; high; by the period's mean SDI),
   outcome, and factor (16 factors, excluding the SDI components income, secondary enrolment, and fertility), the numbers of
   high-tail and low-tail locations with the factor evaluable and with an adverse or favourable departure, the difference in the
   percentage adverse (high minus low) with a Newcombe interval, and a two-sided Fisher exact P value.
2. PROFILE_TRAJECTORIES.csv: location-outcome records grouped by tail membership in 2001-2010 and 2011-2023 (stayed in the high
   tail, left the high tail, entered the high tail, stayed in the low tail), by SDI stratum in 2001-2010; for each group and factor,
   the percentage with an adverse departure in each period, and the mean change in the factor departure (favourable direction).
   The middle and late periods are used because most factor series begin in 2000."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '../independent_reproduction_20261009'); from statlib import fisher_exact
D = pd.read_csv('../independent_reproduction_20261009/REPRO_DISEASE_PERIOD.csv')
F = pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv'); T = pd.read_csv('../SDI_PERIOD_TIERS.csv')
BAND = {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '中等SDI': 'Middle and high-middle', '中高SDI': 'Middle and high-middle', '高SDI': 'High'}
OUT = {(302, 6): 'Diarrhoeal incidence', (302, 1): 'Diarrhoeal mortality', (322, 6): 'LRI incidence', (322, 1): 'LRI mortality', (332, 6): 'Meningitis incidence', (332, 1): 'Meningitis mortality'}
FAC = ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct', 'stunting_u5_pct', 'low_birthweight_pct',
       'undernourishment_pct', 'uhc_service_index', 'health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000', 'physicians_per1000',
       'gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']
JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']; P3 = ['1990–2000', '2001–2010', '2011–2023']
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2; l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2); d = p1 - p2
    return 100 * (d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 100 * (d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))
D = D.merge(T[['location_id', 'period', 'tier']], on=['location_id', 'period']); D['band'] = D.tier.map(BAND)
Fe = F[F.status.isin(JUDGE)]
rows = []
for p in P3:
    fp = Fe[Fe.period == p].set_index(['factor', 'location_id']).status
    for (c, m), o in OUT.items():
        for b in BAND.values():
            d = D[(D.cause_id == c) & (D.measure_id == m) & (D.period == p) & (D.band == b)]
            hi = d[d.category == '稳健高尾'].location_id.values; lo = d[d.category == '稳健低尾'].location_id.values
            for f in FAC:
                s = fp.loc[f] if f in fp.index.get_level_values(0) else pd.Series(dtype=object)
                sh = s.reindex(hi).dropna(); sl = s.reindex(lo).dropna()
                ah, al = int((sh == '明确不利').sum()), int((sl == '明确不利').sum()); fh, fl = int((sh == '明确有利').sum()), int((sl == '明确有利').sum())
                rec = dict(period=p, band=b, outcome=o, factor=f, n_high_tail=len(hi), n_low_tail=len(lo), eval_high=len(sh), eval_low=len(sl),
                           adv_high=ah, adv_low=al, fav_high=fh, fav_low=fl)
                if len(sh) >= 3 and len(sl) >= 3:
                    rec['pct_adv_high'] = 100 * ah / len(sh); rec['pct_adv_low'] = 100 * al / len(sl)
                    rec['diff_pp'] = rec['pct_adv_high'] - rec['pct_adv_low']; rec['diff_lo'], rec['diff_hi'] = newcombe(ah, len(sh), al, len(sl))
                    rec['p_fisher'] = fisher_exact(ah, len(sh) - ah, al, len(sl) - al)
                rows.append(rec)
C = pd.DataFrame(rows).drop_duplicates(['period', 'band', 'outcome', 'factor'])
C.round(4).to_csv('PROFILE_CELLS.csv', index=False, encoding='utf-8-sig')
# trajectories
M, L = '2001–2010', '2011–2023'
W = D.pivot_table(index=['location_id', 'cause_id', 'measure_id'], columns='period', values='category', aggfunc='first').reset_index()
def grp(r):
    hm, hl = r[M] == '稳健高尾', r[L] == '稳健高尾'
    if hm and hl: return 'Stayed in high tail'
    if hm and not hl: return 'Left high tail'
    if not hm and hl: return 'Entered high tail'
    if r[M] == '稳健低尾' and r[L] == '稳健低尾': return 'Stayed in low tail'
W['group'] = W.apply(grp, axis=1); W = W.dropna(subset=['group'])
W['band'] = W.location_id.map(T[T.period == M].set_index('location_id').tier.map(BAND)); W['outcome'] = [OUT[(c, m)] for c, m in zip(W.cause_id, W.measure_id)]
A = F.assign(adv=np.where(F.status.isin(JUDGE), (F.status == '明确不利').astype(float), np.nan)).pivot_table(index=['location_id', 'factor'], columns='period', values='adv')
R = F.pivot_table(index=['location_id', 'factor'], columns='period', values='favourable_residual')
rows = []
for (b, o, g), s in W.groupby(['band', 'outcome', 'group']):
    for f in FAC:
        a = A.xs(f, level='factor').reindex(s.location_id) if f in A.index.get_level_values(1) else None
        r = R.xs(f, level='factor').reindex(s.location_id)
        both = a[[M, L]].dropna() if a is not None else pd.DataFrame()
        rows.append(dict(band=b, outcome=o, group=g, factor=f, n=len(s), n_both=len(both),
                         pct_adv_mid=100 * both[M].mean() if len(both) else np.nan, pct_adv_late=100 * both[L].mean() if len(both) else np.nan,
                         mean_change_departure=(r[L] - r[M]).mean()))
pd.DataFrame(rows).round(4).to_csv('PROFILE_TRAJECTORIES.csv', index=False, encoding='utf-8-sig')
print(len(C), len(rows))
