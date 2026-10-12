"""COVID-19年份的敏感性分析（2026-10-11，独立审稿意见）：去掉2020—2022年后重新拟合六个参照模型并分类
（晚期2011—2023年只用2011—2019年和2023年），重算正文的疾病端结论，与主分析比较。输出 S1_COVID_EXCLUSION.csv（一行）。"""
import numpy as np, pandas as pd
import fit
from engine import *
F0 = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
main = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
KEY = ['location_id', 'cause_id', 'measure_id', 'period']
fit.PANEL = fit.PANEL[~fit.PANEL.year.isin([2020, 2021, 2022])].reset_index(drop=True)
r = classify(fit.fit_all())
m = r.merge(main[KEY + ['category']], on=KEY, suffixes=('', '_m'))
late = m[m.period == '2011–2023']
sw = lambda d: int((((d.category == '稳健高尾') & (d.category_m == '稳健低尾')) | ((d.category == '稳健低尾') & (d.category_m == '稳健高尾'))).sum())
d = dict(variant='exclude 2020–2022', n_records=len(m), class_agree_pct=100 * (m.category == m.category_m).mean(), opposite_tail_vs_main=sw(m),
         late_class_agree_pct=100 * (late.category == late.category_m).mean(), late_opposite=sw(late))
hs = late[(late.cause_id == 322) & (late.measure_id == 6)]
d['highsdi_LRIinc_high_main'] = int(((hs.category_m == '稳健高尾') & hs.location_id.isin(LATE_HIGH)).sum())
d.update(disease_claims(r)); d.update(factor_claims(r, F0))
O = pd.DataFrame([d]); O.to_csv(HERE / 'S1_COVID_EXCLUSION.csv', index=False, encoding='utf-8-sig')
print(O.T.head(45).to_string())
