"""持续高尾/持续低尾记录的现行画像（204地点、GBD 2023固定SDI界值、确认版18因素、已达普及规则）。
输出 PERSISTENT_PROFILES.csv：每条地点—结局一行，含三期参照比、晚期区间（对数尺度）、SDI等级路径、晚期疾病相关因素状态与偏离值（有利方向）。
候选因素与领域定义取自 ../s1_sensitivity_20261010/engine.py（与 results_revision_20261010/part3_typology.py 相同）。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / 's1_sensitivity_20261010'))
from engine import A, REL, PDOM, T
pr = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_COUNTRY_DISEASE_PROFILES.csv')
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
cats = ['p1_category', 'p2_category', 'p3_category']
pr['persist'] = np.where((pr[cats] == '稳健高尾').all(axis=1), 'H', np.where((pr[cats] == '稳健低尾').all(axis=1), 'L', None))
R = pr[pr.persist.notna()].copy()
CID = {'LRI': 322, '腹泻病': 302, '脑膜炎': 332}
R['cause'] = R.dz.str.replace('发病', '').str.replace('死亡', ''); R['cause_id'] = R.cause.map(CID); R['measure'] = np.where(R.dz.str.endswith('死亡'), 'mortality', 'incidence')
tier = T.pivot(index='location_id', columns='period', values='tier'); TN = {'低SDI': 'L', '中低SDI': 'LM', '中等SDI': 'M', '中高SDI': 'HM', '高SDI': 'H'}
R['sdi_path'] = R.location_id.map(lambda i: '→'.join(TN[v] for v in tier.loc[i]))
R['band_late'] = R.location_id.map(lambda i: {'L': 'low', 'LM': 'low', 'M': 'mid', 'HM': 'mid', 'H': 'high'}[TN[tier.loc[i].iloc[-1]]])
FL = F[F.period == '2011–2023'].set_index(['location_id', 'factor'])
def prof(r):
    adv, fav, unc, uni, ne = [], [], [], [], []
    for f in REL[r.cause_id]:
        if (r.location_id, f) not in FL.index: ne.append(f); continue
        q = FL.loc[(r.location_id, f)]; s = q.status; v = f'{f}:{q.favourable_residual:+.1f}'
        {'明确不利': adv, '明确有利': fav, '方向不确定': unc, '已达普及': uni}.get(s, ne).append(v if s in ('明确不利', '明确有利') else f)
    doms = sorted({PDOM[x.split(':')[0]] for x in adv} - {'U'})   # 2026-10-11：UHC 单列，不计入领域
    return pd.Series(dict(adverse=';'.join(adv), favourable=';'.join(fav), uncertain=';'.join(unc), universal=';'.join(uni), not_evaluable=';'.join(ne), n_adv_domains=len(doms), adv_domains=''.join(doms),
                          n_eval_specific=sum(1 for f in REL[r.cause_id] if f != 'uhc_service_index' and (r.location_id, f) in FL.index and FL.loc[(r.location_id, f)].status != '不可评价')))
R = pd.concat([R, R.apply(prof, axis=1)], axis=1)
keep = ['location_id', 'location', 'cause', 'cause_id', 'measure', 'persist', 'sdi_path', 'band_late', 'p1_rate_ratio', 'p2_rate_ratio', 'p3_rate_ratio', 'p3_lower', 'p3_upper',
        'adverse', 'favourable', 'uncertain', 'universal', 'not_evaluable', 'n_eval_specific', 'n_adv_domains', 'adv_domains']
R = R[keep].sort_values(['persist', 'cause', 'measure', 'p3_rate_ratio'], ascending=[True, True, True, False])
R.to_csv(HERE / 'PERSISTENT_PROFILES.csv', index=False, encoding='utf-8-sig')
print(R.groupby(['persist', 'band_late']).size().unstack())
