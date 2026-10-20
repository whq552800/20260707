"""2026-10-19: how much of each infection's departure from the SDI-year reference is shared with general under-5 mortality?

Question: is the excess burden of LRI, diarrhoeal disease and meningitis a disease-specific phenomenon, or one expression of a
country's overall child-mortality position relative to development?

General component: the period (and 2023) residual of under-5 mortality from all OTHER causes (GBD 2023 all-cause minus the three
study causes), given the same SDI-year reference as the disease outcomes (natural spline of SDI, year effects, spline x centred
year). For each disease outcome and period, residual = a + beta * other + specific (OLS across locations); the general part is
a + beta * other, the specific part is the remainder. This is a statistical split: the other-cause residual is itself a GBD estimate
and shares the all-cause envelope, the covariates and the data systems with the study causes, so "general" means shared with
other child mortality in GBD, not proven to be a single real mechanism.

Outputs
  GS_SHARE.csv        beta, R2 (share of between-location variance shared with other-cause mortality) by outcome and period,
                      2000 location bootstrap replicates (seed 20261019); also within SDI strata.
  GS_PERSIST.csv      early-late (1990-2000 vs 2011-2023) Spearman correlation of the general and specific parts.
  GS_EXCESS_2023.csv  2023 excess deaths in persistent high-tail locations split into general and specific parts
                      (symmetric two-factor split of exp(general + specific) - 1), point estimates.
  GS_FACTORS.csv      factor models (as in factor_models_20261017) refitted on the other-cause residual and on the specific
                      parts, 2011-2023, by SDI stratum, 1000 bootstrap replicates.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
sys.path.insert(0, str(A / 'independent_reproduction_20261009'))
from paths import PANEL
from statlib import rankdata
rng = np.random.default_rng(20261019)
P3 = ['1990–2000', '2001–2010', '2011–2023']
def period_of(y): return P3[0] if y <= 2000 else (P3[1] if y <= 2010 else P3[2])
def ns_basis(x, knots, bk=(0.0, 1.0)):
    k = np.r_[bk[0], knots, bk[1]]; K = len(k)
    def dk(j): return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])
    return np.column_stack([x] + [dk(j) - dk(K - 2) for j in range(K - 2)])
def rho(x, y): return np.corrcoef(rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float)))[0, 1]

pan = pd.read_csv(PANEL, usecols=['location_id', 'cause_id', 'measure_id', 'year', 'sdi', 'gbd_rate_per_100k'])
KN = np.quantile(pan.drop_duplicates(['location_id', 'year']).sdi, [.25, .5, .75])
env = pd.read_csv(A / 'gbd_envelope_20261012' / 'GBD2023_U5_ALLCAUSE_DEATH_RATE.csv')
three = pan[pan.measure_id == 1].groupby(['location_id', 'year']).gbd_rate_per_100k.sum().rename('three')
o = env.merge(three.reset_index(), on=['location_id', 'year']).merge(pan.drop_duplicates(['location_id', 'year'])[['location_id', 'year', 'sdi']], on=['location_id', 'year'])
o['y'] = np.log(o.allcause_rate_per_100k - o.three)

def fit_resid(g):
    g = g.sort_values(['location_id', 'year']).reset_index(drop=True)
    B = ns_basis(g.sdi.values, KN); tc = ((g.year - 2006.5) / 33).values[:, None]
    YF = pd.get_dummies(g.year, drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(g)), YF, B, B * tc])
    g['res'] = g.y.values - X @ (np.linalg.pinv(X.T @ X) @ X.T @ g.y.values); g['period'] = g.year.map(period_of)
    return g
og = fit_resid(o)
OTH = og.groupby(['location_id', 'period']).res.mean().unstack()          # other-cause period residuals
OTH23 = og[og.year == 2023].set_index('location_id').res

D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
OUT = {(302, 1): 'Diarrhoeal mortality', (302, 6): 'Diarrhoeal incidence', (322, 1): 'LRI mortality', (322, 6): 'LRI incidence', (332, 1): 'Meningitis mortality', (332, 6): 'Meningitis incidence'}
def _piv(p):
    w = D[D.period == p].pivot(index='location_id', columns=['cause_id', 'measure_id'], values='residual'); w.columns = [OUT[c] for c in w.columns]; return w
RES = {p: _piv(p) for p in P3}
T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv'); T = T[T.period == '2011–2023'].set_index('location_id').tier.map(
    {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '中等SDI': 'Middle and high-middle', '中高SDI': 'Middle and high-middle', '高SDI': 'High'})

def split(y, x):
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; gen = X @ b
    return b[1], 1 - np.var(y - gen) / np.var(y), gen, y - gen

# 1. shares
rows = []; NB = 2000
for p in P3:
    for o_ in OUT.values():
        d = pd.concat([RES[p][o_].rename('y'), OTH[p].rename('x'), T.rename('band')], axis=1).dropna(subset=['y', 'x'])
        for band in ['All'] + ['Low and low-middle', 'Middle and high-middle', 'High']:
            s = d if band == 'All' else d[d.band == band]
            if band != 'All' and p != '2011–2023': continue
            b, r2, _, _ = split(s.y.values, s.x.values)
            bb = []
            for _ in range(NB if band == 'All' else 1000):
                ii = rng.integers(0, len(s), len(s)); bb.append(split(s.y.values[ii], s.x.values[ii])[:2])
            bb = np.array(bb)
            rows.append(dict(period=p, outcome=o_, band=band, n=len(s), beta=b, beta_lo=np.percentile(bb[:, 0], 2.5), beta_hi=np.percentile(bb[:, 0], 97.5),
                             r2=r2, r2_lo=np.percentile(bb[:, 1], 2.5), r2_hi=np.percentile(bb[:, 1], 97.5), rho=rho(s.y, s.x)))
S = pd.DataFrame(rows); S.to_csv(H / 'GS_SHARE.csv', index=False)

# 2. persistence of the parts
rows = []
parts = {}
for p in P3:
    for o_ in OUT.values():
        d = pd.concat([RES[p][o_].rename('y'), OTH[p].rename('x')], axis=1).dropna()
        _, _, gen, spec = split(d.y.values, d.x.values); parts[(p, o_)] = pd.DataFrame({'gen': gen, 'spec': spec}, index=d.index)
for o_ in OUT.values():
    a, b = parts[('1990–2000', o_)], parts[('2011–2023', o_)]; j = a.index.intersection(b.index)
    rows.append(dict(outcome=o_, n=len(j), rho_total=rho(a.gen[j] + a.spec[j], b.gen[j] + b.spec[j]), rho_general=rho(a.gen[j], b.gen[j]), rho_specific=rho(a.spec[j], b.spec[j])))
rows.append(dict(outcome='Other-cause mortality', n=OTH.dropna().shape[0], rho_total=rho(OTH.dropna()['1990–2000'], OTH.dropna()['2011–2023'])))
pd.DataFrame(rows).to_csv(H / 'GS_PERSIST.csv', index=False)

# 3. 2023 excess deaths: split
BL = pd.read_csv(A / 'excess_deaths_20261017' / 'EXCESS_BY_LOCATION_2023.csv')
AF = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_ANNUAL_FIT.csv'); AF = AF[(AF.year == 2023) & (AF.measure_id == 1)]
rows = []; LOCROWS = []
for c, dz in [(302, 'Diarrhoeal disease'), (322, 'LRI'), (332, 'Meningitis')]:
    a = AF[AF.cause_id == c].set_index('location_id').residual_log
    d = pd.concat([a.rename('r'), OTH23.rename('x')], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d.x]); bcoef = np.linalg.lstsq(X, d.r.values, rcond=None)[0]
    d['g'] = X @ bcoef; d['s'] = d.r - d.g
    q = BL[BL.disease == dz].set_index('location_id').join(d, how='inner')
    q['tot'] = q.expected * (np.exp(q.r) - 1)                               # equals GBD deaths minus expected
    q['gen'] = q.expected * 0.5 * ((np.exp(q.g) - 1) + (np.exp(q.r) - np.exp(q.s)))
    q['spec'] = q.tot - q.gen
    for lab, m in [('Persistent high tail', q.persistent_high), ('Persistent high tail without Nigeria', q.persistent_high & (q.index != 214))]:
        z = q[m]; rows.append(dict(disease=dz, locations=lab, n=int(m.sum()), excess=z.tot.sum(), general=z.gen.sum(), specific=z.spec.sum(), beta_2023=bcoef[1]))
    LOCROWS.append(q[q.persistent_high][['tot', 'gen', 'spec']].assign(disease=dz))
X3 = pd.DataFrame(rows)
for lab in X3.locations.unique():
    z = X3[X3.locations == lab]; rows.append(dict(disease='All three', locations=lab, n=np.nan, excess=z.excess.sum(), general=z.general.sum(), specific=z.specific.sum()))
X3 = pd.DataFrame(rows); X3['general_share'] = X3.general / X3.excess; X3.to_csv(H / 'GS_EXCESS_2023.csv', index=False)
# 2026-10-20: per location (three infections summed over the persistent high-tail records of each location)
LX = pd.concat(LOCROWS).groupby(level=0)[['tot', 'gen', 'spec']].sum()
LX['dominant'] = np.where(LX.spec > LX.gen, 'infection-specific', 'general'); LX.to_csv(H / 'GS_EXCESS_BY_LOCATION_2023.csv')
o = LX.drop(index=214, errors='ignore'); print('other locations', len(o), o.dominant.value_counts().to_dict(), 'positive excess', int((o.tot > 0).sum()))

# 4. factor models on the other-cause residual and on the specific parts (2011-2023)
FM = pd.read_csv(A / 'factor_models_20261017' / 'FM_DATA.csv').set_index('location_id')
DN = ['Basic services', 'Immunisation', 'Nutrition', 'Health system', 'Air pollution']
Y = {'Other-cause mortality (general)': OTH['2011–2023']}
for o_ in OUT.values(): Y[o_ + ' (specific)'] = parts[('2011–2023', o_)].spec
pc = lambda b: 100 * (np.exp(-b) - 1)
rows = []
for name, y in Y.items():
    d = FM[DN + ['band']].join(y.rename('y'), how='inner').dropna()
    for band in ['Low and low-middle', 'Middle and high-middle', 'High']:
        s = d[d.band == band]; X = np.column_stack([np.ones(len(s)), s[DN].values]); b = np.linalg.lstsq(X, s.y.values, rcond=None)[0]
        bb = []
        for _ in range(1000):
            ii = rng.integers(0, len(s), len(s)); bb.append(np.linalg.lstsq(X[ii], s.y.values[ii], rcond=None)[0][1:])
        bb = np.array(bb)
        for j, dn in enumerate(DN):
            rows.append(dict(outcome=name, band=band, n=len(s), domain=dn, pct_per_sd=pc(b[1 + j]), lo=np.percentile(pc(bb[:, j]), 2.5), hi=np.percentile(pc(bb[:, j]), 97.5)))
pd.DataFrame(rows).to_csv(H / 'GS_FACTORS.csv', index=False)
print('done')
