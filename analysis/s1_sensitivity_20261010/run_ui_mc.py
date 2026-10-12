"""GBD 估计不确定性的传播：每次抽样对每个地点—结局—时期抽一个标准正态 z，该时期各年的对数率加 z×SE（SE 由 GBD 95% UI 换算：(ln上限−ln下限)/3.92），
重新拟合参照模型A并分类，计算正文结论。共 200 次。输出 S1_UI_DRAWS.csv（每次一行）。
2026-10-11 增补（审稿意见R2-M1）：另记录每个地点—结局在各次抽样中为持续高尾/持续低尾、晚期稳健高尾/低尾的次数，
以及每个地点六个结局全部持续高尾的次数，输出 S1_UI_LABEL_FREQ.csv。随机数序列与原来相同，S1_UI_DRAWS.csv 不变。"""
import numpy as np, pandas as pd, sys
from engine import *
from fit import fit_all, PANEL
F0 = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
main = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
KEY = ['location_id', 'cause_id', 'measure_id', 'period']
lo = PANEL.gbd_rate_lower_per_100k.clip(lower=1e-12); hi = PANEL.gbd_rate_upper_per_100k.clip(lower=1e-12)
PANEL['se'] = ((np.log(hi) - np.log(lo)) / 3.92).fillna(0).clip(lower=0)
print('SE median', PANEL.groupby(['cause_id', 'measure_id']).se.median().round(3).to_dict(), flush=True)
grp = PANEL.groupby(['cause_id', 'measure_id', 'location_id', 'period']).ngroup().values
rng = np.random.default_rng(20261010); N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rows = []; cnt = None
for i in range(N):
    z = rng.standard_normal(grp.max() + 1)
    pert = lambda g: z[grp[g.index]] * PANEL.se.values[g.index]
    r = classify(fit_all(perturb=pert))
    m = r.merge(main[KEY + ['category']], on=KEY, suffixes=('', '_m'))
    d = dict(draw=i, class_agree_pct=100 * (m.category == m.category_m).mean(),
             opposite_tail_vs_main=int((((m.category == '稳健高尾') & (m.category_m == '稳健低尾')) | ((m.category == '稳健低尾') & (m.category_m == '稳健高尾'))).sum()))
    d.update(disease_claims(r)); d.update(factor_claims(r, F0)); rows.append(d)
    C = r.pivot_table(index=['cause_id', 'measure_id', 'location_id'], columns='period', values='category', aggfunc='first')
    c = pd.DataFrame({'persist_high': (C == '稳健高尾').all(axis=1).astype(int), 'persist_low': (C == '稳健低尾').all(axis=1).astype(int),
                      'late_high': (C['2011–2023'] == '稳健高尾').astype(int), 'late_low': (C['2011–2023'] == '稳健低尾').astype(int)})
    a6 = c.persist_high.groupby(level='location_id').transform('sum').eq(6).astype(int); c['all6_high'] = a6
    cnt = c if cnt is None else cnt.add(c, fill_value=0)
    if i % 20 == 0: print(i, round(d['class_agree_pct'], 1), flush=True)
pd.DataFrame(rows).to_csv(HERE / 'S1_UI_DRAWS.csv', index=False, encoding='utf-8-sig')
cnt = cnt.astype(int).reset_index(); cnt.insert(3, 'n_draws', N)
cnt = cnt.merge(main[main.period == '2011–2023'][['location_id', 'location']].drop_duplicates(), on='location_id', how='left')
cnt.to_csv(HERE / 'S1_UI_LABEL_FREQ.csv', index=False, encoding='utf-8-sig')
