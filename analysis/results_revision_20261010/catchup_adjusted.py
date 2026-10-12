"""服务追赶的回归检查（2026-10-11，第二轮独立审稿意见）：二分类比较（part3_trajectories.py）之外，对同一批地点
（2001—2010年明确不利、2011—2023年可评价）以普通最小二乘回归疾病残差的变化（晚期减中期）对因素偏离的变化，
同时调整起点缺口（中期因素偏离）和起点疾病残差，以处理向参照回归。系数为因素偏离每改善10个单位（百分点或指数分）
时疾病残差（对数尺度）的变化，区间为异方差稳健（HC1）95%区间。负值表示缺口收窄伴随相对改善。输出 PART3_CATCHUP_ADJUSTED.csv。"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '../independent_reproduction_20261009'); from statlib import t_ppf
D = pd.read_csv('../independent_reproduction_20261009/REPRO_DISEASE_PERIOD.csv'); F = pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv')
M, L = '2001–2010', '2011–2023'; CAUSE = {'腹泻病': 302, 'LRI': 322}
rows = []
for f, dz in [('basic_water_pct', '腹泻病'), ('basic_sanitation_pct', '腹泻病'), ('uhc_service_index', '腹泻病'), ('dtp3_pct', 'LRI'), ('uhc_service_index', 'LRI'), ('stunting_u5_pct', 'LRI')]:
    for meas, mn in [(1, '死亡'), (6, '发病')]:
        fw = F[F.factor == f].pivot(index='location_id', columns='period', values=['favourable_residual', 'status'])
        fw = fw[(fw[('status', M)] == '明确不利') & fw[('status', L)].isin(['明确不利', '明确有利', '方向不确定', '已达普及'])]
        rw = D[(D.cause_id == CAUSE[dz]) & (D.measure_id == meas)].pivot(index='location_id', columns='period', values='residual').reindex(fw.index)
        d = pd.DataFrame({'dy': rw[L] - rw[M], 'dx': (fw[('favourable_residual', L)] - fw[('favourable_residual', M)]).astype(float),
                          'gap0': fw[('favourable_residual', M)].astype(float), 'y0': rw[M]}).dropna()
        X = np.column_stack([np.ones(len(d)), d.dx / 10, d.gap0 / 10, d.y0]); y = d.dy.values
        XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b; n, k = X.shape
        V = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / (n - k); se = np.sqrt(np.diag(V)); t = t_ppf(.975, n - k)
        rows.append(dict(factor=f, outcome=dz + mn, n=n, coef_per10=b[1], lo=b[1] - t * se[1], hi=b[1] + t * se[1], coef_gap0_per10=b[2], coef_y0=b[3]))
O = pd.DataFrame(rows); O[['coef_per10', 'lo', 'hi', 'coef_gap0_per10', 'coef_y0']] = O[['coef_per10', 'lo', 'hi', 'coef_gap0_per10', 'coef_y0']].round(6)
O.to_csv('PART3_CATCHUP_ADJUSTED.csv', index=False, encoding='utf-8-sig'); print(O.round(3).to_string(index=False))
