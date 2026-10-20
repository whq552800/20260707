"""R5b 气候信号是否为地区效应：在 region_proxy 内去均值后重算相关；并按地区汇总高SDI国家的LRI发病残差。
2026-10-11（审稿意见R2-m3）：常规Spearman P值把地点当作独立观测。另加两项检查：(1) 亚区内置换——只在同一亚区内打乱环境残差
（10000次，种子20261012），P值为置换相关绝对值不小于观测值的比例；(2) 逐一剔除一个亚区后重算亚区内相关，报告最小值和最大值。
原有各列不变。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parent.parent; R = B / 'independent_reproduction_20261009'; sys.path.insert(0, str(R))
import statlib
pd.set_option('display.width', 250)
r = pd.read_csv(R / 'REPRO_DISEASE_PERIOD.csv'); r = r[(r.cause_id == 322) & (r.measure_id == 6)]
E = pd.read_csv('R4_ENV_PERIOD.csv'); prof = pd.read_csv('A_PROFILES_WITH_REGION.csv')
meta = prof.drop_duplicates('location_id').set_index('location_id')[['sdi2023', 'region']]
r = r.join(meta, on='location_id')
TT = pd.read_csv(B / 'SDI_PERIOD_TIERS.csv'); r = r.merge(TT[['location_id', 'period', 'tier']], on=['location_id', 'period'])
rng = np.random.default_rng(20261012); NPERM = 10000
def rk(a): return pd.Series(a).rank().values
def perm_p(mv, mr, reg):
    rv, rr = rk(mv), rk(mr); obs = np.corrcoef(rv, rr)[0, 1]
    groups = [np.where(reg == g)[0] for g in np.unique(reg)]; cnt = 0
    for _ in range(NPERM):
        pv = rv.copy()
        for ix in groups: pv[ix] = rv[rng.permutation(ix)]
        cnt += abs(np.corrcoef(pv, rr)[0, 1]) >= abs(obs) - 1e-12
    return (cnt + 1) / (NPERM + 1)
out = []
for p in ['1990–2000', '2001–2010', '2011–2023']:
    for sub, sel in [('全部国家', r.sdi2023 > 0), ('高SDI', r.tier == '高SDI')]:
        x = r[(r.period == p) & sel]
        for env in ['气温', '绝对湿度', '寒冷日', '湿润日', '环境PM2.5']:
            e = E[(E.period == p) & (E.env == env)].set_index('location_id').dev
            m = x.assign(v=x.location_id.map(e)).dropna(subset=['v'])
            m = m[m.groupby('region').location_id.transform('size') >= 2]
            rho0 = statlib.spearman(m.v.values, m.residual.values)
            mv = m.v - m.groupby('region').v.transform('mean'); mr = m.residual - m.groupby('region').residual.transform('mean')
            rho1 = statlib.spearman(mv.values, mr.values)
            loso = []
            for g in m.region.unique():
                k = m[m.region != g]; k = k[k.groupby('region').location_id.transform('size') >= 2]
                kv = k.v - k.groupby('region').v.transform('mean'); kr = k.residual - k.groupby('region').residual.transform('mean')
                loso.append(statlib.spearman(kv.values, kr.values)[0])
            out.append(dict(period=p, subset=sub, env=env, n=len(m), n_regions=m.region.nunique(), rho_between=rho0[0], p_between=rho0[1], rho_within_region=rho1[0], p_within=rho1[1],
                            p_within_perm=perm_p(mv.values, mr.values, m.region.values), rho_within_loso_min=min(loso), rho_within_loso_max=max(loso)))
O = pd.DataFrame(out); O.to_csv('R5b_WITHIN_REGION.csv', index=False, encoding='utf-8-sig'); print(O.round(3).to_string(index=False))
late = r[(r.period == '2011–2023') & (r.tier == '高SDI')]
print(late.groupby('region').agg(n=('location', 'size'), median_R=('rate_ratio', 'median'), n_hightail=('category', lambda s: (s == '稳健高尾').sum())).sort_values('median_R', ascending=False).round(2).to_string())
