"""R4 微观环境与气象条件。环境变量先按与因素相同的SDI—年份参照去除发展梯度（取对数或原值后同结构回归），
得到"相对同等发展水平国家"的环境偏离；再与疾病残差、病死成分做 Spearman 相关（各期、各疾病），BH 校正。
预设方向（据已知机制，分析前写定）：
  腹泻：气温+、高温日+、湿润日+、绝对湿度+
  LRI：寒冷日+、气温−、家庭PM2.5+、环境PM2.5+
  脑膜炎：相对湿度−、绝对湿度−、高温日+、环境PM2.5+"""
import sys, numpy as np, pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parent.parent; R = B / 'independent_reproduction_20261009'; sys.path.insert(0, str(R))
from paths import PANEL
from repro_common import ns_basis, period_of
import statlib
pd.set_option('display.width', 250)
ENV = {'era5_tavg_C_pw_u5': ('气温', 'raw'), 'era5_hot_days_tmax_gt30_pw_u5': ('高温日', 'log1p'), 'era5_cold_days_tmin_lt0_pw_u5': ('寒冷日', 'log1p'),
       'era5_ah_g_m3_pw_u5': ('绝对湿度', 'raw'), 'era5_rh_pct_pw_u5': ('相对湿度', 'raw'), 'era5_wet_days_pr_gt1mm_pw_u5': ('湿润日', 'raw'),
       'gbd_ambient_pm25_ug_m3': ('环境PM2.5', 'log'), 'gbd_household_pm25_ug_m3': ('家庭PM2.5', 'log')}
HYP = {'腹泻病': {'气温': 1, '高温日': 1, '湿润日': 1, '绝对湿度': 1}, 'LRI': {'寒冷日': 1, '气温': -1, '家庭PM2.5': 1, '环境PM2.5': 1},
       '脑膜炎': {'相对湿度': -1, '绝对湿度': -1, '高温日': 1, '环境PM2.5': 1}}
ly = pd.read_csv(PANEL).drop_duplicates(['location_id', 'year']); ly['period'] = ly.year.map(period_of)
KN = np.quantile(ly.sdi, [.25, .5, .75])
env = []
for v, (name, tr) in ENV.items():
    g = ly[['location_id', 'year', 'period', 'sdi', v]].dropna().sort_values(['location_id', 'year']).reset_index(drop=True)
    z = g[v].values if tr == 'raw' else (np.log1p(g[v].values) if tr == 'log1p' else np.log(g[v].values))
    Bm = ns_basis(g.sdi.values, KN); YF = pd.get_dummies(g.year, drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(g)), YF, Bm, Bm * ((g.year.values - 2006.5) / 33)[:, None]])
    b = np.linalg.pinv(X.T @ X) @ X.T @ z; g['dev'] = z - X @ b; g['raw'] = g[v]
    env.append(g.groupby(['location_id', 'period']).agg(dev=('dev', 'mean'), raw=('raw', 'mean')).reset_index().assign(env=name))
E = pd.concat(env); E.to_csv('R4_ENV_PERIOD.csv', index=False, encoding='utf-8-sig')
DZ = {302: '腹泻病', 322: 'LRI', 332: '脑膜炎'}; ON = {1: '死亡', 6: '发病'}
r = pd.read_csv(R / 'REPRO_DISEASE_PERIOD.csv'); r['dz'] = r.cause_id.map(DZ); r['m'] = r.measure_id.map(ON)
w = r.pivot_table(index=['location_id', 'dz', 'period'], columns='m', values='residual').reset_index(); w['病死成分'] = w['死亡'] - w['发病']
rows = []
for dz, hyp in HYP.items():
    for name, sign in hyp.items():
        for p in ['1990–2000', '2001–2010', '2011–2023']:
            m = w[(w.dz == dz) & (w.period == p)].merge(E[(E.env == name) & (E.period == p)], on=['location_id', 'period'])
            for outc in ['发病', '死亡', '病死成分']:
                for basis in ['dev', 'raw']:
                    rho, pv, n = statlib.spearman(m[basis].values, m[outc].values)
                    rows.append(dict(dz=dz, env=name, expected=sign, period=p, outcome=outc, basis=basis, rho=rho, p=pv, n=n))
T = pd.DataFrame(rows)
T['q'] = np.nan
for bs in ['dev', 'raw']:
    idx = T.basis == bs; T.loc[idx, 'q'] = statlib.bh(T.loc[idx, 'p'].values)
T['consistent'] = np.sign(T.rho) == T.expected
T.to_csv('R4_ENV_DISEASE_ASSOC.csv', index=False, encoding='utf-8-sig')
show = T[(T.basis == 'dev')].pivot_table(index=['dz', 'env', 'expected', 'outcome'], columns='period', values='rho').round(2)
qq = T[(T.basis == 'dev')].pivot_table(index=['dz', 'env', 'expected', 'outcome'], columns='period', values='q')
print(show.join((qq < 0.05).astype(int).add_suffix(' sig')).to_string())
late = T[(T.basis == 'dev') & (T.period == '2011–2023')]
print('\n晚期，去发展梯度：方向与预设一致', int(late.consistent.sum()), '/', len(late), '；一致且q<0.05：', int((late.consistent & (late.q < 0.05)).sum()), '；相反且q<0.05：', int((~late.consistent & (late.q < 0.05)).sum()))
