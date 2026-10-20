"""2026-10-17 (R4): multivariable models of burden departures on factor departures, the factor-centred core of the paper.

Unit: location, period 2011-2023 (main). Outcome: period residual of each of the six disease outcomes (log observed minus log
SDI-year reference; positive = burden higher than expected for development). Exposures: departures of the 15 factors from their
own SDI-year references, in the favourable direction (negative = shortfall), standardised by their SD across locations in the
period, and averaged within five domains (basic services, immunisation, nutrition, health system, air pollution) over the factors
available for a location. Domain indices are rescaled to SD 1 so that coefficients compare a 1-SD shortfall across domains.

Models (OLS), for each outcome, within each SDI stratum (low and low-middle; middle and high-middle; high; by mean SDI in
2011-2023) and for all locations with stratum intercepts:   residual = a + sum_d b_d * domain_d + e.
Reported as the percentage change in the burden ratio per 1-SD shortfall: 100 * (exp(-b_d) - 1).
Intervals: 2000 location bootstrap replicates; the same resampled locations are used for all outcomes and domains, so
differences between outcomes (shared vs specific) and Shapley shares have joint intervals. Seed 20261017.
Shapley decomposition of R^2 over the five domains (exact, 2^5 subsets).
Model dependence (GBD covariates): a variant builds each domain only from factors that are not covariates of the GBD model of the
outcome (COVARIATE_OVERLAP.csv); incidence models of LRI and diarrhoeal disease use almost none of the factors.
Deaths associated with shortfalls: for each location and mortality outcome in 2023, the log reduction predicted if every domain
shortfall (negative index) were raised to zero, using the stratum model, sum_d max(-b_d, 0) * max(-domain_d, 0) (only domains whose
coefficient indicates lower burden with more favourable conditions contribute); deaths associated =
GBD deaths x (1 - exp(-reduction)); summed over all locations and over persistent high-tail locations, by domain (domain-alone
counterfactual) and for all domains together.
Outputs: FM_COEF.csv, FM_CONTRAST.csv, FM_SHAPLEY.csv, FM_COEF_NONCOV.csv, FM_DEATHS.csv, FM_DATA.csv."""
import sys, itertools, math
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
sys.path.insert(0, str(A / 'independent_reproduction_20261009'))
from paths import PANEL
rng = np.random.default_rng(20261017); NB = 2000
P = '2011–2023'
OUT = {(302, 6): 'Diarrhoeal incidence', (302, 1): 'Diarrhoeal mortality', (322, 6): 'LRI incidence', (322, 1): 'LRI mortality', (332, 6): 'Meningitis incidence', (332, 1): 'Meningitis mortality'}
COVCOL = {'Diarrhoeal incidence': 'Diarrhoea incidence', 'Diarrhoeal mortality': 'Diarrhoea mortality', 'LRI incidence': 'LRI incidence', 'LRI mortality': 'LRI mortality', 'Meningitis incidence': 'Meningitis incidence', 'Meningitis mortality': 'Meningitis mortality'}
DOM = {'Basic services': ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct'], 'Immunisation': ['dtp3_pct', 'hib3_pct', 'pcv3_pct'],
       'Nutrition': ['stunting_u5_pct', 'low_birthweight_pct', 'undernourishment_pct'],
       'Health system': ['uhc_service_index', 'health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000', 'physicians_per1000'],   # physicians added 2026-10-20
       'Air pollution': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']}
DN = list(DOM); FAC = [f for v in DOM.values() for f in v]
BAND = {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '中等SDI': 'Middle and high-middle', '中高SDI': 'Middle and high-middle', '高SDI': 'High'}
BANDS = ['Low and low-middle', 'Middle and high-middle', 'High']

D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv'); D = D[D.period == P]
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv'); F = F[(F.period == P) & F.factor.isin(FAC) & (F.status != '不可评价')]
T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv'); T = T[T.period == P].set_index('location_id').tier.map(BAND)
COV = pd.read_csv(A / 'gbd_coupling_20261011' / 'COVARIATE_OVERLAP.csv').set_index('factor')
Z = F.pivot(index='location_id', columns='factor', values='favourable_residual').reindex(columns=FAC)
Z = Z / Z.std()                                                  # 1 = one SD of departures across locations
def domains(fac_ok=None):
    out = {}
    for d, fs in DOM.items():
        fs2 = [f for f in fs if fac_ok is None or f in fac_ok]
        out[d] = Z[fs2].mean(axis=1) if fs2 else pd.Series(np.nan, index=Z.index)
    X = pd.DataFrame(out)
    return X / X.std()
X0 = domains()
Y = D.pivot(index='location_id', columns=['cause_id', 'measure_id'], values='residual'); Y.columns = [OUT[c] for c in Y.columns]
dat = X0.join(Y, how='inner').join(T.rename('band'), how='inner')
dat.to_csv(H / 'FM_DATA.csv')

def ols(Xm, y):
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None); e = y - Xm @ b
    return b, 1 - e.var() / y.var()
def design(df, cols, band):
    M = df[cols].values
    if band == 'All':
        S = pd.get_dummies(df.band).reindex(columns=BANDS, fill_value=0).astype(float).values
        return np.column_stack([S, M]), S.shape[1]
    return np.column_stack([np.ones(len(df)), M]), 1
def shapley(df, cols, y, band):
    r2 = {}
    for k in range(len(cols) + 1):
        for sub in itertools.combinations(range(len(cols)), k):
            Xm, _ = design(df, [cols[i] for i in sub], band)
            r2[sub] = ols(Xm, y)[1] if Xm.shape[1] else 0.0
    base = r2[()]; n = len(cols); sh = np.zeros(n)
    for i in range(n):
        for sub, v in r2.items():
            if i in sub: continue
            w = math.factorial(len(sub)) * math.factorial(n - len(sub) - 1) / math.factorial(n)
            sh[i] += w * (r2[tuple(sorted(sub + (i,)))] - v)
    return sh, r2[tuple(range(n))] - base

def fit_all(df, Xcols_by_outcome, do_shap=True):
    res = {}
    for band in BANDS + ['All']:
        sub = df if band == 'All' else df[df.band == band]
        for o in OUT.values():
            cols = Xcols_by_outcome(o)
            s = sub.dropna(subset=cols + [o])
            Xm, k0 = design(s, cols, band); b, r2 = ols(Xm, s[o].values)
            sh = shapley(s, cols, s[o].values, band) if do_shap else (None, None)
            res[(band, o)] = dict(b=b[k0:], n=len(s), r2=r2, shap=sh[0], r2dom=sh[1])
    return res

cols0 = lambda o: DN
est = fit_all(dat, cols0)
locs = dat.index.values
boot = []
for r in range(NB):
    idx = rng.choice(locs, len(locs), replace=True)
    bd = dat.loc[idx].reset_index()
    boot.append(fit_all(bd, cols0, do_shap=(r < 1000)))
    if r % 200 == 0: print('boot', r, flush=True)

pc = lambda b: 100 * (np.exp(-b) - 1)     # % change in burden ratio per 1-SD shortfall
rows, crow, srow = [], [], []
for (band, o), e in est.items():
    B_ = np.array([bt[(band, o)]['b'] for bt in boot])
    for j, d in enumerate(DN):
        v = pc(B_[:, j])
        rows.append(dict(band=band, outcome=o, domain=d, n=e['n'], r2=e['r2'], pct_per_sd=pc(e['b'][j]), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5),
                         p_two=2 * min((v > 0).mean(), (v < 0).mean())))
    S_ = np.array([bt[(band, o)]['shap'] for bt in boot[:1000]])
    for j, d in enumerate(DN):
        srow.append(dict(band=band, outcome=o, domain=d, shapley_r2=e['shap'][j], lo=np.percentile(S_[:, j], 2.5), hi=np.percentile(S_[:, j], 97.5),
                         share=e['shap'][j] / e['r2dom'] if e['r2dom'] > 0 else np.nan, r2_domains=e['r2dom']))
C = pd.DataFrame(rows); C.to_csv(H / 'FM_COEF.csv', index=False); pd.DataFrame(srow).to_csv(H / 'FM_SHAPLEY.csv', index=False)
# shared vs specific: pairwise differences between the three mortality (and three incidence) outcomes, same domain and stratum
for band in BANDS + ['All']:
    for kind in ['mortality', 'incidence']:
        os_ = [o for o in OUT.values() if o.endswith(kind)]
        for d_i, d in enumerate(DN):
            for a_, b_ in itertools.combinations(os_, 2):
                diff = np.array([pc(bt[(band, a_)]['b'][d_i]) - pc(bt[(band, b_)]['b'][d_i]) for bt in boot])
                e = pc(est[(band, a_)]['b'][d_i]) - pc(est[(band, b_)]['b'][d_i])
                crow.append(dict(band=band, domain=d, a=a_, b=b_, diff=e, lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5)))
pd.DataFrame(crow).to_csv(H / 'FM_CONTRAST.csv', index=False)

# model dependence: domains rebuilt from factors that are not GBD covariates of the outcome
nonc = {o: [f for f in FAC if f not in COV.index or pd.isna(COV.loc[f, COVCOL[o]])] for o in OUT.values()}   # factors absent from the overlap table are not GBD covariates
rows = []
for o in OUT.values():
    Xn = domains(set(nonc[o])); keep = [d for d in DN if Xn[d].notna().sum() > 30]
    dd = Xn[keep].join(Y[[o]], how='inner').join(T.rename('band'), how='inner')
    for band in BANDS + ['All']:
        s = (dd if band == 'All' else dd[dd.band == band]).dropna()
        Xm, k0 = design(s, keep, band); b, r2 = ols(Xm, s[o].values)
        bb = []
        for _ in range(1000):
            ii = rng.integers(0, len(s), len(s)); ss = s.iloc[ii]
            Xb, _k = design(ss, keep, band); bb.append(ols(Xb, ss[o].values)[0][k0:])
        bb = np.array(bb)
        for j, d in enumerate(keep):
            v = pc(bb[:, j])
            rows.append(dict(band=band, outcome=o, domain=d, factors='; '.join(f for f in DOM[d] if f in nonc[o]), n=len(s), pct_per_sd=pc(b[k0 + j]), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
pd.DataFrame(rows).to_csv(H / 'FM_COEF_NONCOV.csv', index=False)

# deaths associated with domain shortfalls, 2023
pan = pd.read_csv(PANEL); pan = pan[(pan.year == 2023) & (pan.measure_id == 1) & pan.rate_valid]
DP = D.pivot(index='location_id', columns=['cause_id', 'measure_id'], values='category')
Pers = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv').pivot_table(index='location_id', columns=['cause_id', 'measure_id', 'period'], values='category', aggfunc='first')
rows = []
def deaths(estd, c, o):
    g = pan[pan.cause_id == c].set_index('location_id')[['gbd_number']].join(dat[DN + ['band']], how='inner').dropna()
    pers = set(Pers.index[(Pers[(c, 1)] == '稳健高尾').all(axis=1)])
    out = {}
    for which in DN + ['All domains']:
        red = np.zeros(len(g))
        for band in BANDS:
            m = (g.band == band).values; b = estd[(band, o)]['b']
            for j, d in enumerate(DN):
                if which in (d, 'All domains'):
                    red[m] += np.clip(-b[j], 0, None) * np.clip(-g[d].values[m], 0, None)   # b<0: favourable departures lower the burden
        av = g.gbd_number.values * (1 - np.exp(-red))
        out[(which, 'All locations')] = av.sum(); out[(which, 'Persistent high tail')] = av[g.index.isin(pers)].sum()
    out[('deaths', 'All locations')] = g.gbd_number.sum()
    return out
ALLB = {}
for c, o in [(302, 'Diarrhoeal mortality'), (322, 'LRI mortality'), (332, 'Meningitis mortality')]:
    e = deaths(est, c, o); bs = [deaths(bt, c, o) for bt in boot[:1000]]; ALLB[o] = (e, bs)
    for k, v in e.items():
        arr = np.array([z[k] for z in bs])
        rows.append(dict(outcome=o, domain=k[0], locations=k[1], deaths=v, lo=np.percentile(arr, 2.5), hi=np.percentile(arr, 97.5)))
for k in ALLB['LRI mortality'][0]:
    v = sum(ALLB[o][0][k] for o in ALLB); arr = np.sum([[z[k] for z in ALLB[o][1]] for o in ALLB], axis=0)
    rows.append(dict(outcome='All three', domain=k[0], locations=k[1], deaths=v, lo=np.percentile(arr, 2.5), hi=np.percentile(arr, 97.5)))
FD = pd.DataFrame(rows)
tot = FD.groupby(['domain', 'locations'])[['deaths']].sum().reset_index()   # point totals over the three diseases
FD.to_csv(H / 'FM_DEATHS.csv', index=False)
pd.set_option('display.width', 220)
print(C[C.band != 'All'].pivot_table(index=['band', 'outcome'], columns='domain', values='pct_per_sd').round(1).to_string())
print(FD.round(0).to_string())
