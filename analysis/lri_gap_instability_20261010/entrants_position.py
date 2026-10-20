"""新进入高SDI组地点（早期非高SDI、晚期高SDI）与原高SDI地点在晚期的疾病相对位置比较。"""
import pandas as pd, numpy as np
from pathlib import Path
A = Path(__file__).resolve().parent.parent
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv'); T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')
E = T[T.period == '1990–2000'].set_index('location_id').tier; L = T[T.period == '2011–2023'].set_index('location_id').tier
x = D[D.period == '2011–2023'].copy(); x['grp'] = np.where((x.location_id.map(L) == '高SDI') & (x.location_id.map(E) == '高SDI'), '原高SDI',
                                                         np.where(x.location_id.map(L) == '高SDI', '新进入高SDI', None))
x = x[x.grp.notna()]; x['dz'] = x.cause_id.map({302: '腹泻病', 322: 'LRI', 332: '脑膜炎'}) + x.measure_id.map({1: '死亡', 6: '发病'})
S = x.groupby(['dz', 'grp']).agg(n=('location_id', 'size'), median_RR=('rate_ratio', 'median'), high_tail=('category', lambda s: (s == '稳健高尾').sum()),
                                 above_ref=('direction', lambda s: (s == 'pos').sum())).unstack('grp')
S.to_csv('ENTRANTS_LATE_POSITION.csv', encoding='utf-8-sig'); print(S.round(2).to_string())
