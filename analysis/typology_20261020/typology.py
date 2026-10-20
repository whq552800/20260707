"""2026-10-20: country patterns of disadvantage, by rules fixed before listing countries (2011-2023 period residuals and domain
indices of factor_models_20261017; contrasts as in within_country_20261019; 'top quartile' over the 190 modelled locations).
  A General child-survival disadvantage: other-cause under-5 mortality residual in the top quartile and >=2 of the 3 infection
    mortality residuals above zero.
  B Air quality - LRI: country LRI contrast (mean of LRI incidence and mortality minus the country's mean of the same measure)
    in the top quartile, and air pollution domain <= -0.5 SD.
  C Case management - diarrhoeal deaths: diarrhoeal mortality contrast in the top quartile, and health-system domain <= -0.5 SD.
  D Lethality - basic services: mean of (mortality - incidence) residuals of diarrhoeal disease and LRI in the top quartile, and
    basic-services domain <= -0.5 SD.
  E Nutrition - LRI at high SDI: high SDI stratum, LRI contrast in the top quartile of that stratum, nutrition domain <= -0.5 SD.
A location may meet more than one pattern. Output: TYPOLOGY.csv (one row per location x pattern, with the defining values)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
sys.path.insert(0, str(A / 'independent_reproduction_20261009'))
src = (A / 'general_specific_20261019' / 'general_specific.py').read_text().split('# 1. shares')[0].replace("H = Path(__file__).resolve().parent; A = H.parent", "")
exec(src)
F = pd.read_csv(A / 'factor_models_20261017' / 'FM_DATA.csv').set_index('location_id').dropna()
L = D[['location_id', 'location']].drop_duplicates().set_index('location_id').location
d = F.join(OTH['2011–2023'].rename('other')).join(L)
M = ['Diarrhoeal mortality', 'LRI mortality', 'Meningitis mortality']; I = ['Diarrhoeal incidence', 'LRI incidence', 'Meningitis incidence']
d['lri_c'] = ((d['LRI incidence'] - d[I].mean(axis=1)) + (d['LRI mortality'] - d[M].mean(axis=1))) / 2
d['dz_c'] = d['Diarrhoeal mortality'] - d[M].mean(axis=1)
d['leth'] = ((d['Diarrhoeal mortality'] - d['Diarrhoeal incidence']) + (d['LRI mortality'] - d['LRI incidence'])) / 2
d['n_mort_pos'] = (d[M] > 0).sum(axis=1)
q = lambda s: s >= s.quantile(.75)
pats = {
 'A General child-survival disadvantage': q(d.other) & (d.n_mort_pos >= 2),
 'B Air quality - LRI': q(d.lri_c) & (d['Air pollution'] <= -0.5),
 'C Case management - diarrhoeal deaths': q(d.dz_c) & (d['Health system'] <= -0.5),
 'D Lethality - basic services': q(d.leth) & (d['Basic services'] <= -0.5),
 'E Nutrition - LRI at high SDI': (d.band == 'High') & (d.lri_c >= d[d.band == 'High'].lri_c.quantile(.75)) & (d.Nutrition <= -0.5)}
rows = []
for p, m in pats.items():
    for lid, r in d[m].iterrows():
        rows.append(dict(pattern=p, location_id=lid, location=r.location, band=r.band, other=r.other, lri_c=r.lri_c, dz_c=r.dz_c, leth=r.leth,
                         **{k: r[k] for k in ['Basic services', 'Immunisation', 'Nutrition', 'Health system', 'Air pollution']}))
T_ = pd.DataFrame(rows); T_.to_csv(H / 'TYPOLOGY.csv', index=False)
for p, g in T_.groupby('pattern'): print(p, len(g), '|', ', '.join(g.location))

# Examples for the main table: within each pattern, the four locations with the most under-5 deaths from the three infections in 2023
BL = pd.read_csv(A / 'excess_deaths_20261017' / 'EXCESS_BY_LOCATION_2023.csv').groupby('location_id').gbd_number.sum()
T_['deaths_2023'] = T_.location_id.map(BL)
EX = T_.sort_values('deaths_2023', ascending=False).groupby('pattern').head(4).sort_values(['pattern', 'deaths_2023'], ascending=[True, False])
EX.to_csv(H / 'TYPOLOGY_EXAMPLES.csv', index=False); T_.to_csv(H / 'TYPOLOGY.csv', index=False)
pd.set_option('display.width', 250); print(EX[['pattern', 'location', 'band', 'deaths_2023', 'other', 'lri_c', 'dz_c', 'leth', 'Basic services', 'Health system', 'Air pollution', 'Nutrition']].round(2).to_string(index=False))
print(T_.groupby('pattern').deaths_2023.sum().round(0))
