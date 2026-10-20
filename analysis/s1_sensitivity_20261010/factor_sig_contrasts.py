"""因素端敏感性 S1–S8：持续高尾与持续低尾“反复不利”比例的 Fisher 检验（BH 校正），列出 q<0.05 的因素—结局，检查是否仍只涉及饮水、卫生设施、清洁烹饪、UHC 和卫生支出。
输出 S1_FACTOR_SIG_CONTRASTS.csv。"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; AN = HERE.parent
sys.path.insert(0, str(AN / 'factor_residual_audit_20261010'))
from factor_pipeline import downstream
BASIC = {'basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct', 'uhc_service_index', 'health_expenditure_pc_current_usd'}
rows = []
for code in ['main'] + [f'S{i}' for i in range(1, 10)]:
    p = AN / 'factor_profiles_confirmed_20261010' / ('CONFIRMED_FACTOR_PERIOD.csv' if code == 'main' else f'cache_{code}.csv')
    F = pd.read_csv(p)
    fp, C, Aso = downstream(F); s = C[C.q < 0.05]
    rows.append(dict(variant=code, n_tests=int(C.q.notna().sum()), n_sig=len(s), all_basic=bool(s.factor.isin(BASIC).all()),
                     sig=';'.join(f'{a}|{b}|q={q:.3f}' for a, b, q in zip(s.dz, s.factor, s.q))))
    print(code, len(s), rows[-1]['all_basic'], flush=True)
pd.DataFrame(rows).to_csv(HERE / 'S1_FACTOR_SIG_CONTRASTS.csv', index=False, encoding='utf-8-sig')
