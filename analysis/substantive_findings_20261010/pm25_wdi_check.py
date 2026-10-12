"""环境PM2.5数据源的检查（2026-10-11，独立审稿意见）：分析过程中环境PM2.5由世界发展指标（WDI）改为GBD估计。
这里用WDI序列（面板列 wdi_pm25_annual_mean_ug_m3）按与 r4_environment.py 相同的SDI—年份结构取对数去除发展梯度，
重算2011—2023年高SDI地点LRI发病残差与PM2.5残差的相关（全部地点与亚区内，亚区内置换10 000次，种子20261012）。
输出 R5c_PM25_WDI_CHECK.csv。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parent.parent; R = B / 'independent_reproduction_20261009'; sys.path.insert(0, str(R))
from paths import PANEL
from repro_common import ns_basis, period_of
import statlib
ly = pd.read_csv(PANEL).drop_duplicates(['location_id', 'year']); ly['period'] = ly.year.map(period_of)
KN = np.quantile(ly.sdi, [.25, .5, .75]); LATE = '2011–2023'
def env_dev(v):
    g = ly[['location_id', 'year', 'period', 'sdi', v]].dropna(); g = g[g[v] > 0].sort_values(['location_id', 'year']).reset_index(drop=True)
    z = np.log(g[v].values); Bm = ns_basis(g.sdi.values, KN); YF = pd.get_dummies(g.year, drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(g)), YF, Bm, Bm * ((g.year.values - 2006.5) / 33)[:, None]])
    g['dev'] = z - X @ (np.linalg.pinv(X.T @ X) @ X.T @ z)
    return g[g.period == LATE].groupby('location_id').dev.mean()
r = pd.read_csv(R / 'REPRO_DISEASE_PERIOD.csv'); r = r[(r.cause_id == 322) & (r.measure_id == 6) & (r.period == LATE)]
TT = pd.read_csv(B / 'SDI_PERIOD_TIERS.csv'); hi = set(TT[(TT.period == LATE) & (TT.tier == '高SDI')].location_id)
reg = pd.read_csv('A_PROFILES_WITH_REGION.csv').drop_duplicates('location_id').set_index('location_id').region
rng = np.random.default_rng(20261012); rows = []
for src, col in [('GBD', 'gbd_ambient_pm25_ug_m3'), ('WDI', 'wdi_pm25_annual_mean_ug_m3')]:
    e = env_dev(col)
    m = r[r.location_id.isin(hi)].assign(v=lambda d: d.location_id.map(e), region=lambda d: d.location_id.map(reg)).dropna(subset=['v'])
    m = m[m.groupby('region').location_id.transform('size') >= 2]
    rho0 = statlib.spearman(m.v.values, m.residual.values)
    mv = (m.v - m.groupby('region').v.transform('mean')).values; mr = (m.residual - m.groupby('region').residual.transform('mean')).values
    rho1 = statlib.spearman(mv, mr)
    rv, rr = pd.Series(mv).rank().values, pd.Series(mr).rank().values; groups = [np.where(m.region.values == g)[0] for g in np.unique(m.region.values)]; cnt = 0
    for _ in range(10000):
        pv = rv.copy()
        for ix in groups: pv[ix] = rv[rng.permutation(ix)]
        cnt += abs(np.corrcoef(pv, rr)[0, 1]) >= abs(rho1[0]) - 1e-12
    rows.append(dict(source=src, n=len(m), rho_between=rho0[0], p_between=rho0[1], rho_within_region=rho1[0], p_within=rho1[1], p_within_perm=(cnt + 1) / 10001))
O = pd.DataFrame(rows); O.to_csv('R5c_PM25_WDI_CHECK.csv', index=False, encoding='utf-8-sig'); print(O.round(4).to_string(index=False))
