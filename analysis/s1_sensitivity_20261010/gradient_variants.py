"""Outcome-specific tail contrast (S1 Text Table S summary) under the sensitivity analyses (2026-10-12, fourth independent review).
For each disease-side variant (as in run_variants.py), each factor-rule variant (cache_S1-S9), and the 200 GBD-uncertainty
draws (same random sequence as run_ui_mc.py), recomputes the equal-weight mean over outcome-factor pairs of the percentage
of robust high-tail locations with the factor clearly adverse minus that of robust low-tail locations, in 2011-2023, in the
low and low-middle and the high SDI strata, for all disease-relevant factors except the UHC index and for water and sanitation.
Pairs need at least 3 evaluable locations in each tail in both strata. Output: S1_GRADIENT_VARIANTS.csv"""
import numpy as np, pandas as pd, sys
from engine import *
from fit import fit_all, PANEL, oof_period
F0 = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
main = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv'); LATE = '2011–2023'
BAND = {'低SDI': 'low', '中低SDI': 'low', '高SDI': 'high'}
band = T[T.period == LATE].set_index('location_id').tier.map(BAND)
REL = {322: ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct', 'stunting_u5_pct', 'low_birthweight_pct'],
       302: ['basic_water_pct', 'basic_sanitation_pct', 'stunting_u5_pct'], 332: ['dtp3_pct', 'hib3_pct', 'pcv3_pct']}
WS = {'basic_water_pct', 'basic_sanitation_pct'}
JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']

def gradient(r, F):
    r = r[(r.period == LATE) & r.category.isin(['稳健高尾', '稳健低尾'])].copy(); r['band'] = r.location_id.map(band)
    f = F[(F.period == LATE) & F.status.isin(JUDGE)][['location_id', 'factor', 'status']]
    pairs = {}
    for (c, m), g in r.groupby(['cause_id', 'measure_id']):
        for fac in REL[c]:
            x = g.merge(f[f.factor == fac], on='location_id')
            d = {}
            for b in ['low', 'high']:
                h, l = x[(x.band == b) & (x.category == '稳健高尾')], x[(x.band == b) & (x.category == '稳健低尾')]
                if len(h) < 3 or len(l) < 3: break
                d[b] = 100 * ((h.status == '明确不利').mean() - (l.status == '明确不利').mean())
            if len(d) == 2: pairs[(c, m, fac)] = d
    P = pd.DataFrame(pairs).T; W = P[[k[2] in WS for k in P.index]]
    return dict(n_pairs=len(P), low=P['low'].mean(), high=P['high'].mean(), diff=(P['low'] - P['high']).mean(),
                ws_low=W['low'].mean(), ws_high=W['high'].mean(), ws_diff=(W['low'] - W['high']).mean())

rows = [dict(variant='main', **gradient(main, F0))]
SP = pd.read_csv(A / 'reference_spec_sensitivity_20261010' / 'SPEC_PERIOD_RESIDUALS.csv')
LAB = {'腹泻病死亡': (302, 1), 'LRI死亡': (322, 1), '脑膜炎死亡': (332, 1), '腹泻病发病': (302, 6), 'LRI发病': (322, 6), '脑膜炎发病': (332, 6)}
SP['cause_id'] = SP.outcome.map(lambda o: LAB[o][0]); SP['measure_id'] = SP.outcome.map(lambda o: LAB[o][1])
V = {f'spec_{s}': SP[SP.spec == s].drop(columns=['spec', 'outcome']) for s in 'BCD'}
V['df3'] = classify(fit_all(df=3)); V['df5'] = classify(fit_all(df=5)); V['pop_weighted'] = classify(fit_all(weight=True))
u = PANEL.groupby('location_id').under5_population.mean(); V['excl_small'] = classify(fit_all(sub=set(u[u >= 1e5].index)))
V['loo'] = classify(oof_period())
V['tertile_tails'] = classify(main.drop(columns='category'), 1 / 3, 2 / 3); V['quintile_tails'] = classify(main.drop(columns='category'), 0.2, 0.8)
for k, r in V.items(): rows.append(dict(variant=k, **gradient(r, F0)))
for i in range(1, 10):
    Fi = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / f'cache_S{i}.csv'); rows.append(dict(variant=f'factor_S{i}', **gradient(main, Fi)))
# GBD uncertainty draws: same random sequence as run_ui_mc.py
lo = PANEL.gbd_rate_lower_per_100k.clip(lower=1e-12); hi = PANEL.gbd_rate_upper_per_100k.clip(lower=1e-12)
PANEL['se'] = ((np.log(hi) - np.log(lo)) / 3.92).fillna(0).clip(lower=0)
grp = PANEL.groupby(['cause_id', 'measure_id', 'location_id', 'period']).ngroup().values
rng = np.random.default_rng(20261010); N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
for i in range(N):
    z = rng.standard_normal(grp.max() + 1)
    pert = lambda g: z[grp[g.index]] * PANEL.se.values[g.index]
    rows.append(dict(variant=f'ui_draw_{i}', **gradient(classify(fit_all(perturb=pert)), F0)))
    if i % 50 == 0: print(i, flush=True)
O = pd.DataFrame(rows); O.to_csv(HERE / 'S1_GRADIENT_VARIANTS.csv', index=False, encoding='utf-8-sig')
print(O[~O.variant.str.startswith('ui_')].round(1).to_string())
U = O[O.variant.str.startswith('ui_')]
print('UI draws median (2.5-97.5):', {k: np.round(U[k].quantile([.5, .025, .975]).values, 1).tolist() for k in ['low', 'high', 'diff', 'ws_diff']})
