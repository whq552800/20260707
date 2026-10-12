"""Tail contrasts grouped by GBD covariate status (2026-10-12, fourth independent review).
For the ten disease-relevant factors and three further health-system factors (health expenditure, hospital beds, nurses and
midwives) with all six outcomes in 2011-2023, each outcome-factor pair is classed by whether the factor, or a closely related
construct, is a covariate of the GBD 2023 model for that outcome (gbd_coupling_20261011/COVARIATE_OVERLAP.csv: 'same' = direct,
'related' = related construct, blank = none). Within the low and low-middle and the high SDI strata, the high-minus-low-tail
difference in the percentage of locations with the factor clearly adverse is averaged with equal weight over pairs in each
class (pairs need at least 3 evaluable locations in each tail). Intervals: 2000 location resamples within strata (seed 20261017).
The LRI incidence model, unlike those for diarrhoeal and meningitis incidence, does not read in cause-specific mortality, so its pairs
are shown separately. Output: OUTCOME_FACTOR_COVSTATUS.csv"""
import numpy as np, pandas as pd
pr = pd.read_csv('../independent_reproduction_20261009/REPRO_COUNTRY_DISEASE_PROFILES.csv')
F = pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv')
T = pd.read_csv('../SDI_PERIOD_TIERS.csv'); OV = pd.read_csv('../gbd_coupling_20261011/COVARIATE_OVERLAP.csv').set_index('factor').fillna('')
LATE = '2011–2023'; JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']; MIN_N = 3; NB = 2000
BAND = {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '高SDI': 'High'}
FACT = ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct', 'stunting_u5_pct', 'low_birthweight_pct',
        'gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev', 'health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000']
OCOL = {'LRI发病': 'LRI incidence', 'LRI死亡': 'LRI mortality', '腹泻病发病': 'Diarrhoea incidence', '腹泻病死亡': 'Diarrhoea mortality', '脑膜炎发病': 'Meningitis incidence', '脑膜炎死亡': 'Meningitis mortality'}
def status(dz, f):
    v = OV.loc[f, OCOL[dz]]
    return 'direct' if v.startswith('same') else ('related' if v.startswith('related') else 'none')
TL = T[T.period == LATE].assign(band=lambda d: d.tier.map(BAND))
R = pr[pr.p3_category.isin(['稳健高尾', '稳健低尾'])][['location_id', 'dz', 'p3_category']].copy()
R['tail'] = R.p3_category.map({'稳健高尾': 'high', '稳健低尾': 'low'}); R['band'] = R.location_id.map(TL.set_index('location_id').band)
L = pd.concat([R.assign(factor=f) for f in FACT], ignore_index=True).merge(F[F.period == LATE][['location_id', 'factor', 'status']], on=['location_id', 'factor'], how='left')
L['evaluable'] = L.status.isin(JUDGE); L['adverse'] = L.status == '明确不利'
rng = np.random.default_rng(20261017); rows = []
for b in ['Low and low-middle', 'High']:
    locs = TL[TL.band == b].location_id.values; ix = {l: i for i, l in enumerate(locs)}; sub = L[L.band == b]
    pairs = [(dz, f) for dz in OCOL for f in FACT
             if sub[(sub.dz == dz) & (sub.factor == f) & (sub['tail'] == 'high')].evaluable.sum() >= MIN_N
             and sub[(sub.dz == dz) & (sub.factor == f) & (sub['tail'] == 'low')].evaluable.sum() >= MIN_N]
    A = {t: np.zeros((len(locs), len(pairs))) for t in ('high', 'low')}; E = {t: np.zeros((len(locs), len(pairs))) for t in ('high', 'low')}
    for j, (dz, f) in enumerate(pairs):
        s = sub[(sub.dz == dz) & (sub.factor == f)]
        for t in ('high', 'low'):
            st = s[s['tail'] == t]; ii = st.location_id.map(ix).values; A[t][ii, j] = st.adverse.values; E[t][ii, j] = st.evaluable.values
    def pdiff(w):
        with np.errstate(invalid='ignore', divide='ignore'):
            return 100 * (w @ A['high']) / (w @ E['high']) - 100 * (w @ A['low']) / (w @ E['low'])
    est = pdiff(np.ones(len(locs))); Wb = rng.multinomial(len(locs), np.full(len(locs), 1 / len(locs)), size=NB).astype(float); bt = pdiff(Wb)
    cls = np.array([status(dz, f) for dz, f in pairs]); mort = np.array([dz.endswith('死亡') for dz, f in pairs])
    for lab, sel in [('direct covariate', cls == 'direct'), ('related covariate', cls == 'related'), ('not a covariate', cls == 'none'),
                     ('direct or related, mortality outcomes', (cls != 'none') & mort), ('not a covariate, mortality outcomes', (cls == 'none') & mort),
                     ('not a covariate, incidence outcomes', (cls == 'none') & ~mort),
                     ('LRI incidence, all pairs (model uses no cause-specific mortality)', np.array([dz == 'LRI发病' for dz, f in pairs]))]:
        x = np.nanmean(bt[:, sel], 1)
        rows.append(dict(band=b, set=lab, n_pairs=int(sel.sum()), estimate=np.nanmean(est[sel]), lo=np.nanquantile(x, .025), hi=np.nanquantile(x, .975),
                         pairs='; '.join(f'{OCOL[dz]} × {f}' for (dz, f), s in zip(pairs, sel) if s)))
O = pd.DataFrame(rows); O[['estimate', 'lo', 'hi']] = O[['estimate', 'lo', 'hi']].round(6)
O.to_csv('OUTCOME_FACTOR_COVSTATUS.csv', index=False, encoding='utf-8-sig'); print(O.drop(columns='pairs').round(1).to_string())
