"""SDI差距（官方固定区间）。主分析：按1990—2000年SDI均值、依GBD 2023固定参考区间赋级，组成员固定；比较低SDI与高SDI组的人口加权率、率比（RR）和率差（RD）。
敏感性：每个时期按该期SDI均值重新赋级（组成员随时期变化）。另输出五级全部的逐年率。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
A = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(A / 'independent_reproduction_20261009'), str(A)]
from paths import PANEL
from repro_common import period_of
from sdi_tiers import TIERS
pd.set_option('display.width', 250)
d = pd.read_csv(PANEL); d['period'] = d.year.map(period_of)
T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')
DZ = {302: '腹泻病', 322: 'LRI', 332: '脑膜炎'}; ON = {1: '死亡', 6: '发病'}
def rates(df, key):
    g = df.groupby(['cause_id', 'measure_id', 'year', key]).apply(lambda x: 1e5 * x.gbd_number.sum() / x.under5_population.sum(), include_groups=False).unstack(key)
    return g
fixed = T[T.period == '1990–2000'].set_index('location_id').tier
d['tier_fixed'] = d.location_id.map(fixed)
d = d.merge(T[['location_id', 'period', 'tier']].rename(columns={'tier': 'tier_varying'}), on=['location_id', 'period'])
print('固定组成员（1990—2000年均值赋级）国家数：', d.drop_duplicates('location_id').tier_fixed.value_counts().reindex(TIERS).to_dict())
out = []
for key, lab in [('tier_fixed', '主分析：固定组成员'), ('tier_varying', '敏感性：各期重新赋级')]:
    g = rates(d, key)
    g.reset_index().to_csv(f'RATES_BY_TIER_{key}.csv', index=False, encoding='utf-8-sig')
    for (c, m), x in g.groupby(level=[0, 1]):
        x = x.droplevel([0, 1])
        for y in (1990, 2023):
            lo, hi = x.loc[y, '低SDI'], x.loc[y, '高SDI']
            out.append(dict(design=lab, dz=DZ[c] + ON[m], year=y, low=lo, high=hi, RR=lo / hi, RD=lo - hi))
O = pd.DataFrame(out)
W = O.pivot_table(index=['design', 'dz'], columns='year', values=['low', 'high', 'RR', 'RD'])
W[('decline_low', '%')] = 100 * (1 - W[('low', 2023)] / W[('low', 1990)]); W[('decline_high', '%')] = 100 * (1 - W[('high', 2023)] / W[('high', 1990)])
W.to_csv('SDI_GAPS_OFFICIAL_SUMMARY.csv', encoding='utf-8-sig'); print(W.round(1).to_string())
# 方向一致性：与旧口径（1990年地点五等分）比较
old = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_SDI_GAPS_ANNUAL.csv')
old['dz'] = old.cause_id.map(DZ) + old.measure_id.map(ON)
oc = old[old.year.isin([1990, 2023])].pivot_table(index='dz', columns='year', values=['RR', 'RD'])
for dz in oc.index:
    for des in O.design.unique():
        n = W.loc[(des, dz)]
        same_rr = np.sign(n[('RR', 2023)] - n[('RR', 1990)]) == np.sign(oc.loc[dz, ('RR', 2023)] - oc.loc[dz, ('RR', 1990)])
        same_rd = np.sign(n[('RD', 2023)] - n[('RD', 1990)]) == np.sign(oc.loc[dz, ('RD', 2023)] - oc.loc[dz, ('RD', 1990)])
        print(dz, des, 'RR方向与旧口径一致' if same_rr else 'RR方向不同', '；', 'RD方向一致' if same_rd else 'RD方向不同')
