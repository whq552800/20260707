"""确认版因素残差与画像（2026-10-10 用户确定）：
- 参照设定同疾病端（A：年份FE + SDI样条 + 样条×时间），因素自身样本节点；
- 原单位残差 + delta 区间判定方向；
- 覆盖率类因素（饮水、卫生设施、清洁燃料、DTP3、Hib3、PCV3）观测与参照时期均值均 ≥99% 时记为"已达普及"，不作方向判定，不计入可评价期；
- 剔除医生密度（各国仅1年数据，无可评价时期），共18个因素；
- 环境PM2.5采用GBD来源（gbd_ambient_pm25_ug_m3），2026-10-10由WDI来源改换。
输出：CONFIRMED_FACTOR_PERIOD.csv, CONFIRMED_COUNTRY_FACTOR_PROFILES.csv, CONFIRMED_GROUP_CONTRASTS.csv, CONFIRMED_ASSOCIATIONS.csv"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AUD = HERE.parent / 'factor_residual_audit_20261010'; sys.path.insert(0, str(AUD))
from factor_pipeline import factor_periods, downstream
SETTINGS = dict(spec='A', knots='factor', scale='orig', ceiling=99, drop=('physicians_per1000',))
# 2026-10-11：早期只有2000年一个锚点年份的因素纳入早期评价（anchor2000=True，见 factor_pipeline._anchor_factors）
F = factor_periods(**SETTINGS, anchor2000=True)
F0 = factor_periods(**SETTINGS, anchor2000=False)
# 核对：与旧规则（至少2个观测年份）相比，只在“早期、2000年单一锚点年份”的记录上不同
diff = (F.status != F0.status)
assert (~diff | F.single_year_2000).all() and np.allclose(F.favourable_residual, F0.favourable_residual)
print('2000年锚点记录改为可评价：', int((F.single_year_2000 & F.eligible).sum()), '条；状态', F.loc[diff, 'status'].value_counts().to_dict())
fp, C, A = downstream(F)
F.to_csv(HERE / 'CONFIRMED_FACTOR_PERIOD.csv', index=False, encoding='utf-8-sig')
fp.to_csv(HERE / 'CONFIRMED_COUNTRY_FACTOR_PROFILES.csv', index=False, encoding='utf-8-sig')
C.to_csv(HERE / 'CONFIRMED_GROUP_CONTRASTS.csv', index=False, encoding='utf-8-sig')
A.to_csv(HERE / 'CONFIRMED_ASSOCIATIONS.csv', index=False, encoding='utf-8-sig')
print('factors', F.factor.nunique(), 'records', len(F)); print(F.status.value_counts().to_string())
