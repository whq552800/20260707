"""Checks for coupling inside the GBD estimates (added in revision, 2026-10-11).

GBD 2023 cause-of-death models (CODEm) for the three diseases use several of this study's factors as covariates, and the
nonfatal meningitis model reads in cause-specific mortality. Two consequences are examined here:
1. Shared covariates could make mortality residuals correlate across diseases. We recompute the cross-disease correlations
   after removing, by linear regression, the residuals of the observed factors that the mortality models share
   (water, sanitation, clean cooking, DTP3, Hib3, UHC index, ambient PM2.5, second-hand smoke), and within SDI strata.
   In high-SDI locations, vital registration dominates the mortality estimates and covariates carry less weight.
2. Which factor-outcome pairs are covariates in GBD 2023 models (COVARIATE_OVERLAP.csv, from the GBD 2023 methods appendices).
Outputs: COUPLING_PARTIAL_CORR.csv, COUPLING_STRATA_CORR.csv, COVARIATE_OVERLAP.csv"""
import sys, numpy as np, pandas as pd
from pathlib import Path
H = Path(__file__).resolve().parent; A = H.parent
sys.path.insert(0, str(A / 'independent_reproduction_20261009'))
from statlib import rankdata

def rho(x, y):   # Spearman correlation without the p-value (faster for resampling)
    x, y = np.asarray(x, float), np.asarray(y, float); m = ~(np.isnan(x) | np.isnan(y))
    return np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1]
P3 = ['1990–2000', '2001–2010', '2011–2023']
PAIRS = [(302, 322), (302, 332), (322, 332)]
SHARED = ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'uhc_service_index', 'gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
F = F[F.status != '不可评价']
T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')

def mean_rho(df, m, cols=None):
    return np.mean([rho(df[f'{a}_{m}'] if cols is None else cols[(a, m)], df[f'{b}_{m}'] if cols is None else cols[(b, m)]) for a, b in PAIRS])

rows = []; rng = np.random.default_rng(20261011)
for p in P3:
    R = D[D.period == p].pivot_table(index='location_id', columns=['cause_id', 'measure_id'], values='residual')
    R.columns = [f'{c}_{m}' for c, m in R.columns]
    fac = [f for f in SHARED if F[(F.period == p) & (F.factor == f)].location_id.nunique() >= 150]   # Hib3 is sparse before 2001
    X = F[(F.period == p) & F.factor.isin(fac)].pivot_table(index='location_id', columns='factor', values='resid_orig')
    df = R.join(X, how='inner').dropna().reset_index(drop=True)
    Xm = np.column_stack([np.ones(len(df))] + [df[f].values for f in fac])
    def stats(idx):
        d = df.iloc[idx]; Xi = Xm[idx]; out = {}
        for m, lab in [(1, 'mort'), (6, 'inc')]:
            res = {(c, m): d[f'{c}_{m}'].values - Xi @ np.linalg.lstsq(Xi, d[f'{c}_{m}'].values, rcond=None)[0] for c in (302, 322, 332)}
            out[f'{lab}_raw'] = mean_rho(d, m); out[f'{lab}_partial'] = mean_rho(None, m, res)
        out['diff_raw'] = out['mort_raw'] - out['inc_raw']; out['diff_partial'] = out['mort_partial'] - out['inc_partial']
        return out
    est = stats(np.arange(len(df)))
    boot = pd.DataFrame([stats(rng.integers(0, len(df), len(df))) for _ in range(1000)])
    rec = dict(period=p, n=len(df), factors=';'.join(fac), **{k: round(v, 6) for k, v in est.items()})
    for k in ['diff_raw', 'diff_partial']: rec[k + '_lo'], rec[k + '_hi'] = np.round(boot[k].quantile([.025, .975]).values, 6)
    rows.append(rec)
pd.DataFrame(rows).to_csv(H / 'COUPLING_PARTIAL_CORR.csv', index=False, encoding='utf-8-sig')

rows = []
for p in P3:
    R = D[D.period == p].pivot_table(index='location_id', columns=['cause_id', 'measure_id'], values='residual')
    R.columns = [f'{c}_{m}' for c, m in R.columns]
    for lab, tiers in [('Low and low-middle SDI', ['低SDI', '中低SDI']), ('Middle and high-middle SDI', ['中等SDI', '中高SDI']), ('High SDI', ['高SDI'])]:
        s = set(T[(T.period == p) & T.tier.isin(tiers)].location_id); d = R.loc[R.index.isin(s)]
        rows.append(dict(period=p, stratum=lab, n=len(d), mort_mean_rho=round(mean_rho(d, 1), 6), inc_mean_rho=round(mean_rho(d, 6), 6)))
pd.DataFrame(rows).to_csv(H / 'COUPLING_STRATA_CORR.csv', index=False, encoding='utf-8-sig')

# Covariates in GBD 2023 models (CoD appendix: LRI p.571, diarrhoea p.430-431, meningitis p.592; nonfatal appendix: LRI p.834,
# diarrhoea p.472, meningitis p.887-890). 'same' = same or near-identical indicator; 'related' = related construct.
OV = [  # factor, LRI mortality, diarrhoea mortality, meningitis mortality, LRI incidence, diarrhoea incidence, meningitis incidence
    ('basic_water_pct', '', 'same (improved water)', 'same (water access)', '', '', ''),
    ('basic_sanitation_pct', 'related (sanitation SEV)', 'same (sanitation access)', 'same (sanitation access)', '', '', ''),
    ('clean_cooking_pct', 'related (household air pollution SEV; indoor air pollution)', '', '', '', '', ''),
    ('dtp3_pct', 'same (DTP3, lagged)', '', 'related (maternal care and immunisation composite, which includes DTP3)', '', '', ''),
    ('hib3_pct', 'same (Hib3, lagged)', '', 'same (Hib3, herd-effect transformed)', '', '', 'same (Hib3)'),
    ('pcv3_pct', 'same (PCV3, lagged)', '', 'same (PCV3, lagged)', '', '', 'same (PCV3)'),
    ('stunting_u5_pct', 'related (stunting SEV)', '', '', '', '', ''),
    ('undernourishment_pct', 'related (child underweight and wasting SEV)', '', 'related (child underweight SEV)', '', '', ''),
    ('low_birthweight_pct', '', '', '', '', '', ''),
    ('uhc_service_index', 'related (HAQ index)', 'related (HAQ index)', 'related (HAQ index)', 'related (HAQ, excess mortality)', 'related (HAQ, excess mortality)', 'related (HAQ, excess mortality)'),
    ('gbd_ambient_pm25_ug_m3', 'same (outdoor PM2.5; ambient PM SEV)', '', '', '', '', ''),
    ('secondhand_smoke_sev', 'same (second-hand smoking prevalence)', '', '', '', '', ''),
    ('secondary_enrolment_pct', 'related (maternal education)', 'related (maternal education)', 'related (maternal education)', '', '', ''),
    ('income_ppp_pc', '', 'related (lag-distributed income)', 'related (lag-distributed income)', '', '', ''),
    ('health_expenditure_pc_current_usd', '', '', '', '', '', ''),
    ('hospital_beds_per1000', '', '', '', '', '', ''),
    ('nurses_midwives_per1000', '', '', '', '', '', ''),
    ('fertility_births_per_woman', 'related (SDI component)', 'related (SDI component)', 'related (SDI component)', 'related (SDI component)', 'related (SDI component)', 'related (SDI component)')]
pd.DataFrame(OV, columns=['factor', 'LRI mortality', 'Diarrhoea mortality', 'Meningitis mortality', 'LRI incidence', 'Diarrhoea incidence', 'Meningitis incidence']).to_csv(
    H / 'COVARIATE_OVERLAP.csv', index=False, encoding='utf-8-sig')
print(pd.read_csv(H / 'COUPLING_PARTIAL_CORR.csv').to_string()); print(pd.read_csv(H / 'COUPLING_STRATA_CORR.csv').to_string())
