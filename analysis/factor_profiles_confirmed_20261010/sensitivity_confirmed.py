"""以确认版为基准的因素敏感性分析（补充表用）。已算过的变体从缓存读取；可分次运行。"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AUD = HERE.parent / 'factor_residual_audit_20261010'; sys.path.insert(0, str(AUD))
from factor_pipeline import factor_periods, downstream
DROP = ()   # 2026-10-20: physicians kept (author rule: use any available data)
VARS = [('S1', '不设"已达普及"规则', HERE / 'cache_S1.csv', dict(drop=DROP)),
        ('S2', '"已达普及"阈值98%', HERE / 'cache_S2.csv', dict(ceiling=98, drop=DROP)),
        ('S3', '"已达普及"阈值99.5%', HERE / 'cache_S3.csv', dict(ceiling=99.5, drop=DROP)),
        ('S4', '在变换尺度上判定方向', HERE / 'cache_S4.csv', dict(scale='model', ceiling=99, drop=DROP)),
        ('S5', '样条节点取全样本SDI四分位', HERE / 'cache_S5.csv', dict(knots='all', ceiling=99, drop=DROP)),
        ('S6', '参照曲线分时期拟合', HERE / 'cache_S6.csv', dict(spec='C', ceiling=99, drop=DROP)),
        ('S7', '参照曲线形状固定', HERE / 'cache_S7.csv', dict(spec='B', ceiling=99, drop=DROP)),
        ('S8', '参照曲线不含时间项', HERE / 'cache_S8.csv', dict(spec='D', ceiling=99, drop=DROP)),
        ('S9', '只有1个观测年份的地点—时期不作评价（旧规则）', HERE / 'cache_S9.csv', dict(ceiling=99, drop=DROP, anchor2000=False, single_year=False))]   # 2026-10-11
only = sys.argv[1:] or [v[0] for v in VARS]
M = pd.read_csv(HERE / 'CONFIRMED_FACTOR_PERIOD.csv')
CL = [('腹泻病死亡', 'basic_sanitation_pct'), ('腹泻病死亡', 'basic_water_pct'), ('腹泻病死亡', 'uhc_service_index'), ('脑膜炎死亡', 'basic_sanitation_pct'), ('脑膜炎死亡', 'basic_water_pct')]
AS = [('腹泻病死亡', 'basic_sanitation_pct', 'late'), ('腹泻病死亡', 'basic_sanitation_pct', 'middle_late'), ('LRI死亡', 'uhc_service_index', 'late'), ('LRI死亡', 'uhc_service_index', 'middle_late')]
res_path = HERE / 'SENSITIVITY_RESULTS.csv'
res = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
for code, label, cache, kw in VARS:
    if code not in only: continue
    if cache.exists(): F = pd.read_csv(cache); F = F[~F.factor.isin(DROP)]
    else: F = factor_periods(**kw); F.to_csv(cache, index=False, encoding='utf-8-sig')
    fp, C, A = downstream(F)
    m = M.merge(F, on=['location_id', 'factor', 'period'], suffixes=('_m', ''))
    m = m[m.status_m.isin(['明确有利', '明确不利', '方向不确定'])]
    rec = dict(code=code, variant=label, status_agree=round(100 * (m.status_m == m.status).mean(), 1),
               n_sig_contrasts=int((C.q < 0.05).sum()))
    for dz, f in CL:
        r = C[(C.dz == dz) & (C.factor == f)].iloc[0]; rec[f'{dz}|{f}'] = f'{r.high_adv}/{r.high_n} vs {r.low_adv}/{r.low_n}; q={r.q:.3f}'
    for dz, f, w in AS:
        r = A[(A.dz == dz) & (A.factor == f) & (A.window == w)].iloc[0]; rec[f'{dz}|{f}|{w}'] = f'{r.rho:.3f} (n={r.n})'
    res = pd.concat([res[res.code != code] if len(res) else res, pd.DataFrame([rec])])
    res.sort_values('code').to_csv(res_path, index=False, encoding='utf-8-sig'); print('done', code)
