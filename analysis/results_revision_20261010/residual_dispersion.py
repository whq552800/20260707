"""残差离散度随时期的变化（2026-10-11，审稿意见：排名稳定不等于差距幅度稳定）。
对每个结局和时期，计算地点时期残差的标准差、四分位距和平均绝对值。输出 RESIDUAL_DISPERSION.csv。"""
import numpy as np, pandas as pd
D = pd.read_csv('../independent_reproduction_20261009/REPRO_DISEASE_PERIOD.csv')
g = D.groupby(['cause_id', 'measure_id', 'period']).residual
O = pd.DataFrame({'sd': g.std(), 'iqr': g.quantile(.75) - g.quantile(.25), 'mean_abs': g.apply(lambda x: x.abs().mean()), 'n': g.size()}).round(6).reset_index()
O.to_csv('RESIDUAL_DISPERSION.csv', index=False, encoding='utf-8-sig'); print(O.pivot_table(index=['cause_id', 'measure_id'], columns='period', values='sd'))

# Between-location variance of the difference between mortality and incidence residuals, with the variances of the
# incidence and mortality residuals themselves (descriptive; 2026-10-11). Note Var(m - i) = Var(m) + Var(i) - 2 Cov(m, i),
# so this quantity is not independent of the incidence-mortality correlation and is not an estimate of case fatality.
W = D.pivot_table(index=['location_id', 'period'], columns=['cause_id', 'measure_id'], values='residual')
rows = [dict(cause_id=c, period=p, var_mort_minus_inc=(W.xs(p, level='period')[(c, 1)] - W.xs(p, level='period')[(c, 6)]).var(),
             var_inc=W.xs(p, level='period')[(c, 6)].var(), var_mort=W.xs(p, level='period')[(c, 1)].var())
        for c in (302, 322, 332) for p in sorted(D.period.unique())]
pd.DataFrame(rows).round(6).to_csv('VARIANCE_MORT_MINUS_INC.csv', index=False, encoding='utf-8-sig')
print(pd.DataFrame(rows).round(4).to_string())
