"""S1 Text 敏感性分析：疾病端变体（参照设定、样条自由度、人口加权、排除小地点、留一地点外推、尾部界值、GBD不确定性）
与因素端变体（S1–S8）下，正文各项结论的数值。输出 S1_DISEASE_VARIANTS.csv（疾病端11个设定）、S1_FACTOR_VARIANTS.csv（因素端S1–S8）。
GBD不确定性的抽样见 run_ui_mc.py。主分析各数值与正文逐项一致（check_main.py）。"""
import numpy as np, pandas as pd, json
from engine import *
from fit import fit_all, PANEL, oof_residuals, oof_period
F0 = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
main = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
KEY = ['location_id', 'cause_id', 'measure_id', 'period']
def agree(r):
    m = r.merge(main[KEY + ['category']], on=KEY, suffixes=('', '_m'))
    sw = (((m.category == '稳健高尾') & (m.category_m == '稳健低尾')) | ((m.category == '稳健低尾') & (m.category_m == '稳健高尾'))).sum()
    return dict(n_records=len(m), class_agree_pct=100 * (m.category == m.category_m).mean(), opposite_tail_vs_main=int(sw))
V = {}
V['main'] = main
SP = pd.read_csv(A / 'reference_spec_sensitivity_20261010' / 'SPEC_PERIOD_RESIDUALS.csv')
LAB = {'腹泻病死亡': (302, 1), 'LRI死亡': (322, 1), '脑膜炎死亡': (332, 1), '腹泻病发病': (302, 6), 'LRI发病': (322, 6), '脑膜炎发病': (332, 6)}
SP['cause_id'] = SP.outcome.map(lambda o: LAB[o][0]); SP['measure_id'] = SP.outcome.map(lambda o: LAB[o][1])
for s in 'BCD': V[f'spec_{s}'] = SP[SP.spec == s].drop(columns=['spec', 'outcome'])
V['df3'] = classify(fit_all(df=3)); V['df5'] = classify(fit_all(df=5))
V['pop_weighted'] = classify(fit_all(weight=True))
u = PANEL.groupby('location_id').under5_population.mean(); keep = set(u[u >= 1e5].index)
V['excl_small'] = classify(fit_all(sub=keep))
# 2026-10-11 R1c: intervals re-estimated in each fold (previously the main-model half-widths were shifted to the out-of-fold residuals)
V['loo'] = classify(oof_period())
V['tertile_tails'] = classify(main.drop(columns='category'), 1 / 3, 2 / 3)
V['quintile_tails'] = classify(main.drop(columns='category'), 0.2, 0.8)
rows = []
for k, r in V.items():
    d = {'variant': k, **agree(r), **disease_claims(r), **factor_claims(r, F0)}
    if k == 'excl_small': d['n_locations'] = len(keep)
    rows.append(d); print(k, round(d['class_agree_pct'], 1), flush=True)
pd.DataFrame(rows).to_csv(HERE / 'S1_DISEASE_VARIANTS.csv', index=False, encoding='utf-8-sig')
# 因素端 S1–S8（疾病分类为主分析）
SR = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'SENSITIVITY_RESULTS.csv')
fr = [{'variant': 'main', **factor_claims(main, F0)}]
for i in range(1, 10):
    Fi = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / f'cache_S{i}.csv'); fr.append({'variant': f'S{i}', **factor_claims(main, Fi)})
FR = pd.DataFrame(fr).merge(SR.rename(columns={'variant': 'variant_label'}).rename(columns={'code': 'variant'}), on='variant', how='left'); FR.to_csv(HERE / 'S1_FACTOR_VARIANTS.csv', index=False, encoding='utf-8-sig')
print(FR.iloc[:, :12].round(1).to_string())
