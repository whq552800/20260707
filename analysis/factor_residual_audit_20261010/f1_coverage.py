"""F1 因素数据覆盖诊断：每个因素的国家数、年份覆盖、各时期可评价（≥1年，2026-10-20 起；原为≥2年）国家数、取值边界堆积。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parent.parent / 'independent_reproduction_20261009'; sys.path.insert(0, str(R))
from paths import PANEL
from repro_common import period_of
import repro_factor_defs as D
ly = pd.read_csv(PANEL).drop_duplicates(['location_id', 'year'])
ly['period'] = ly.year.map(period_of)
rows = []; yearly = []
for f in D.ALL:
    g = ly[['location_id', 'year', 'period', 'sdi', f]].dropna()
    if f in D.LOG: g = g[g[f] > 0]
    ny = g.groupby('year').location_id.nunique()
    yearly.append(ny.rename(f))
    per = g.groupby(['period', 'location_id']).size().unstack(0).fillna(0)
    row = dict(factor=f, transform='logit' if f in D.LOGIT else ('log' if f in D.LOG else 'log1p'),
               n_obs=len(g), n_loc=g.location_id.nunique(), first_year=g.year.min(), last_year=g.year.max(),
               years_with_data=g.year.nunique(), min_loc_per_year=int(ny.min()), median_loc_per_year=int(ny.median()))
    for p in ['1990–2000', '2001–2010', '2011–2023']:
        row[f'eligible_{p}'] = int((per.get(p, pd.Series(dtype=float)) >= 1).sum())
    if f in D.LOGIT:
        row['pct_at_floor(<=0.1)'] = round(100 * (g[f] <= 0.1).mean(), 1); row['pct_at_ceiling(>=99.9)'] = round(100 * (g[f] >= 99.9).mean(), 1)
        row['pct_ge_99'] = round(100 * (g[f] >= 99).mean(), 1)
    if f in D.LOG1P or f in D.LOG: row['pct_zero'] = round(100 * (g[f] == 0).mean(), 1)
    rows.append(row)
T = pd.DataFrame(rows); T.to_csv('F1_COVERAGE_BY_FACTOR.csv', index=False, encoding='utf-8-sig')
Y = pd.concat(yearly, axis=1).fillna(0).astype(int); Y.to_csv('F1_LOCATIONS_PER_YEAR.csv', encoding='utf-8-sig')
pd.set_option('display.width', 260); print(T.to_string())
print(Y.loc[[1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023]].T.to_string())
