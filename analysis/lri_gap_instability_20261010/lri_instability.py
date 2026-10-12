"""为什么只有LRI死亡率比的方向对分组方式敏感：比较固定组成员（1990—2000年赋级）与各期重新赋级、以及固定为晚期成员三种设计；
分解为低SDI组与高SDI组的成员构成效应；找出主导高SDI组变化的地点。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
A = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(A / 'independent_reproduction_20261009'), str(A)]
from paths import PANEL
d = pd.read_csv(PANEL); T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')
E = T[T.period == '1990–2000'].set_index('location_id').tier; L = T[T.period == '2011–2023'].set_index('location_id').tier
d['tE'] = d.location_id.map(E); d['tL'] = d.location_id.map(L)
DZ = {302: '腹泻病', 322: 'LRI', 332: '脑膜炎'}
def rate(x): return 1e5 * x.gbd_number.sum() / x.under5_population.sum()
out = []
for c in [302, 322, 332]:
    for m in [1, 6]:
        x = d[(d.cause_id == c) & (d.measure_id == m)]
        for y in [1990, 2023]:
            z = x[x.year == y]
            r = dict(dz=DZ[c] + ('死亡' if m == 1 else '发病'), year=y,
                     low_fixedE=rate(z[z.tE == '低SDI']), high_fixedE=rate(z[z.tE == '高SDI']),
                     low_fixedL=rate(z[z.tL == '低SDI']), high_fixedL=rate(z[z.tL == '高SDI']))
            out.append(r)
O = pd.DataFrame(out)
O['RR_fixedE'] = O.low_fixedE / O.high_fixedE; O['RR_fixedL'] = O.low_fixedL / O.high_fixedL
O.to_csv('RR_BY_DESIGN.csv', index=False, encoding='utf-8-sig'); print(O.round(2).to_string())
# 2023年：固定早期成员 vs 晚期成员（=重新赋级的2023年组）的构成效应
rows = []
for c in [302, 322, 332]:
    z = d[(d.cause_id == c) & (d.measure_id == 1) & (d.year == 2023)]
    lo_f, lo_v, hi_f, hi_v = rate(z[z.tE == '低SDI']), rate(z[z.tL == '低SDI']), rate(z[z.tE == '高SDI']), rate(z[z.tL == '高SDI'])
    stay = z[(z.tE == '高SDI') & (z.tL == '高SDI')]; ent = z[(z.tE != '高SDI') & (z.tL == '高SDI')]
    lstay = z[(z.tE == '低SDI') & (z.tL == '低SDI')]; lleave = z[(z.tE == '低SDI') & (z.tL != '低SDI')]
    rows.append(dict(dz=DZ[c], RR_fixed=lo_f / hi_f, RR_var=lo_v / hi_v, dlog_low=np.log(lo_v / lo_f), dlog_high=np.log(hi_v / hi_f),
                     n_high_stay=stay.location_id.nunique(), n_high_entrants=ent.location_id.nunique(), rate_high_stay=rate(stay), rate_high_entrants=rate(ent),
                     pop_share_entrants=ent.under5_population.sum() / (stay.under5_population.sum() + ent.under5_population.sum()),
                     n_low_stay=lstay.location_id.nunique(), n_low_leavers=lleave.location_id.nunique(), rate_low_stay=rate(lstay), rate_low_leavers=rate(lleave)))
C = pd.DataFrame(rows); C['dlogRR'] = np.log(C.RR_var / C.RR_fixed); C.to_csv('COMPOSITION_2023.csv', index=False, encoding='utf-8-sig'); print(C.round(3).T.to_string())
# 高SDI新进入者中LRI死亡贡献最大的地点
z = d[(d.cause_id == 322) & (d.measure_id == 1) & (d.year == 2023) & (d.tE != '高SDI') & (d.tL == '高SDI')].copy()
z['rate'] = 1e5 * z.gbd_number / z.under5_population; z['death_share'] = z.gbd_number / z.gbd_number.sum()
z = z.sort_values('gbd_number', ascending=False)[['location', 'tE', 'rate', 'gbd_number', 'under5_population', 'death_share']]
z.to_csv('HIGH_ENTRANTS_LRI_DEATH_2023.csv', index=False, encoding='utf-8-sig'); print(z.head(15).round(3).to_string())
for c in [302, 332]:
    w = d[(d.cause_id == c) & (d.measure_id == 1) & (d.year == 2023) & (d.tE != '高SDI') & (d.tL == '高SDI')]
    print(DZ[c], 'entrants top:', w.sort_values('gbd_number', ascending=False).location.head(5).tolist())
