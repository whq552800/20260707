"""探索A：超额负担有多大、集中在哪里；持续尾部是哪些国家（地区、SDI）；高尾差距在扩大还是缩小。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parent.parent / 'independent_reproduction_20261009'; sys.path.insert(0, str(R))
from paths import PANEL
from repro_common import period_of
pd.set_option('display.width', 250)
d = pd.read_csv(PANEL); d = d[d.rate_valid & (d.gbd_rate_per_100k > 0)].copy()
DN = {302: '腹泻病', 322: 'LRI', 332: '脑膜炎'}; ON = {1: '死亡', 6: '发病'}
d['dz'] = d.cause_id.map(DN) + d.measure_id.map(ON); d['period'] = d.year.map(period_of)
d = d.merge(pd.read_csv(R / 'REPRO_ANNUAL_FIT.csv'), on=['location_id', 'cause_id', 'measure_id', 'year'])
d['expected'] = np.exp(d.fitted_log_rate) * d.under5_population / 1e5
d['observed'] = d.gbd_rate_per_100k * d.under5_population / 1e5
d['excess'] = d.observed - d.expected
prof = pd.read_csv(R / 'REPRO_COUNTRY_DISEASE_PROFILES.csv')
d = d.merge(prof[['location_id', 'dz', 'group']], on=['location_id', 'dz'], how='left')
# 1 超额：各期，观测总数 vs 参照期望总数；正超额集中度
rows = []
for (dz, p), g in d.groupby(['dz', 'period']):
    pos = g[g.excess > 0].groupby('location')['excess'].sum().sort_values(ascending=False)
    hi = g[g.group == '持续高尾']
    rows.append(dict(dz=dz, period=p, observed=g.observed.sum(), expected=g.expected.sum(), pos_excess=pos.sum(),
                     pos_excess_share_of_obs=100 * pos.sum() / g.observed.sum(), top5_share_of_pos=100 * pos.head(5).sum() / pos.sum(),
                     top5=', '.join(pos.head(5).index), hi_obs_share=100 * hi.observed.sum() / g.observed.sum(),
                     hi_pop_share=100 * hi.under5_population.sum() / g.under5_population.sum(), hi_excess=hi.excess.sum()))
X = pd.DataFrame(rows); X.to_csv('A1_EXCESS_BY_PERIOD.csv', index=False, encoding='utf-8-sig')
print(X[X.dz.str.endswith('死亡')].round(1).to_string())
# 2 持续尾部的地区与SDI构成
ly = d.drop_duplicates(['location_id']).set_index('location_id')
sdi23 = pd.read_csv(PANEL, usecols=['location_id', 'year', 'sdi']).query('year==2023').drop_duplicates('location_id').set_index('location_id').sdi
prof['region'] = prof.location_id.map(ly.region_proxy); prof['continent'] = prof.location_id.map(ly.continent_proxy); prof['sdi2023'] = prof.location_id.map(sdi23)
for grp in ['持续高尾', '持续低尾']:
    g = prof[prof.group == grp]
    print('\n', grp, 'records', len(g), 'locations', g.location_id.nunique())
    print(g.groupby('dz').location.apply(lambda s: ', '.join(sorted(s))).to_string())
    print(g.drop_duplicates('location_id').continent.value_counts().to_string())
    print('SDI2023 median', g.drop_duplicates('location_id').sdi2023.median().round(3), 'quartiles', g.drop_duplicates('location_id').sdi2023.quantile([.25, .75]).round(3).tolist())
print('all SDI2023 median', sdi23.median().round(3))
# 3 持续高尾组的参照比：扩大还是缩小
for grp in ['持续高尾', '持续低尾']:
    g = prof[prof.group == grp]
    t = g.groupby('dz')[['p1_rate_ratio', 'p2_rate_ratio', 'p3_rate_ratio']].median().round(2)
    w = g.assign(widen=g.p3_rate_ratio > g.p1_rate_ratio).groupby('dz').widen.agg(['sum', 'size'])
    print('\n', grp, '参照比中位数（早/中/晚）与远离参照的记录数'); print(t.join(w).to_string())
prof.to_csv('A_PROFILES_WITH_REGION.csv', index=False, encoding='utf-8-sig')
