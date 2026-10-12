"""持续尾部记录“至少一项疾病相关因素明确不利”的可比性检查（S1 Text Table N；审稿意见R2-M2）。
问题：这一事件的发生机会取决于可评价因素的数目，两组可评价因素数不同会使比较失衡。
做法：在每个SDI层×病种内，(1) 报告两组每条记录的平均可评价具体因素数；(2) 取两组中可评价比例均≥90%的具体因素为共同因素集，
只在这些因素都可评价的记录中重算“至少一项明确不利”。另按“全部病种合并”汇总，共同因素集仍按病种分别确定。
输入与 part3_typology.py 相同；不改变其输出。"""
import pandas as pd, numpy as np
pr = pd.read_csv('PART3_PERSISTENT_RECORD_PROFILES.csv')
F = pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv')
LATE = '2011–2023'; JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']; COVER = 0.90
SPEC = {'LRI': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct', 'stunting_u5_pct', 'low_birthweight_pct'],
        '腹泻病': ['basic_water_pct', 'basic_sanitation_pct', 'stunting_u5_pct'], '脑膜炎': ['dtp3_pct', 'hib3_pct', 'pcv3_pct']}
FL = F[F.period == LATE].set_index(['location_id', 'factor']).status
rows = []
for r in pr.itertuples():
    for f in SPEC[r.disease]:
        s = FL.get((r.location_id, f), '不可评价')
        rows.append(dict(location_id=r.location_id, dz=r.dz, disease=r.disease, persist=r.persist, band=r.band, factor=f, evaluable=s in JUDGE, adverse=s == '明确不利'))
L = pd.DataFrame(rows)
out, sets = [], {}
for (band, dz), g in L.groupby(['band', 'disease']):
    cov = g.groupby(['factor', 'persist']).evaluable.mean().unstack()
    if not {'持续高尾', '持续低尾'} <= set(cov.columns): continue
    cs = sorted(cov.index[(cov.min(axis=1) >= COVER)]); sets[(band, dz)] = cs
    rec = g.groupby(['location_id', 'dz', 'persist']).agg(n_eval=('evaluable', 'sum'), any_adv=('adverse', 'any')).reset_index()
    cg = g[g.factor.isin(cs)].groupby(['location_id', 'dz', 'persist']).agg(all_eval=('evaluable', 'all'), any_adv_common=('adverse', 'any')).reset_index()
    rec = rec.merge(cg, on=['location_id', 'dz', 'persist'], how='left')
    for p, s in rec.groupby('persist'):
        e = s[s.n_eval > 0]; c = s[s.all_eval == True]
        out.append(dict(band=band, disease=dz, persist=p, n_records=len(s), mean_eval=s.n_eval.mean(), n_factors=len(SPEC[dz]),
                        any_adv_pct=100 * e.any_adv.mean() if len(e) else np.nan, common_set='; '.join(cs), n_common=len(cs),
                        n_records_common=len(c), any_adv_common_pct=100 * c.any_adv_common.mean() if len(c) else np.nan,
                        _n_any=int(e.any_adv.sum()), _d_any=len(e), _n_c=int(c.any_adv_common.sum()), _d_c=len(c)))
O = pd.DataFrame(out)
# 全部病种合并（与 Fig 3B 相同的口径；共同因素集按病种分别确定）
agg = O.groupby(['band', 'persist']).agg(n_records=('n_records', 'sum'), _n_any=('_n_any', 'sum'), _d_any=('_d_any', 'sum'), _n_c=('_n_c', 'sum'), _d_c=('_d_c', 'sum')).reset_index()
agg['disease'] = 'All'; agg['any_adv_pct'] = 100 * agg._n_any / agg._d_any; agg['n_records_common'] = agg._d_c; agg['any_adv_common_pct'] = 100 * agg._n_c / agg._d_c
O = pd.concat([O, agg], ignore_index=True)
O = O[['band', 'disease', 'persist', 'n_records', 'n_factors', 'mean_eval', 'any_adv_pct', 'n_common', 'common_set', 'n_records_common', 'any_adv_common_pct', '_n_any', '_d_any', '_n_c', '_d_c']]
O = O.rename(columns={'_n_any': 'any_adv_n', '_d_any': 'any_adv_denom', '_n_c': 'common_adv_n', '_d_c': 'common_denom'})
num = ['mean_eval', 'any_adv_pct', 'any_adv_common_pct']; O[num] = O[num].round(6)
O.to_csv('PART3_COMPARABILITY.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(O.round(1).to_string(index=False))

# ---------- 2026-10-11（独立审稿意见）：持续尾部比较的区间。一个地点可贡献最多6条记录，因此按地点整体重抽样（层内，2000次，种子20261013），
# 求“至少一项具体因素不利”和“UHC低于参照”的高尾减低尾差值的95%区间。输出 PART3_SHORTFALL_BOOT.csv。
PR = pd.read_csv('PART3_PERSISTENT_RECORD_PROFILES.csv')
PR['spec_ok'] = PR.n_eval > 0; PR['uhc_ok'] = PR.uhc_status != '不可评价'
rng = np.random.default_rng(20261013); brow = []
def diffs(d):
    out = {}
    for k, ok, v in [('any', 'spec_ok', 'any_specific'), ('uhc', 'uhc_ok', 'uhc_adverse')]:
        x = d[d[ok]]; h = x[x.persist == '持续高尾'][v]; l = x[x.persist == '持续低尾'][v]
        out[k] = 100 * (h.mean() - l.mean()) if len(h) and len(l) else np.nan
    return out
for band, g in PR.groupby('band'):
    locs = g.location_id.unique(); by = {l: g[g.location_id == l] for l in locs}; est = diffs(g); bs = {'any': [], 'uhc': []}
    for _ in range(2000):
        d = pd.concat([by[l] for l in rng.choice(locs, len(locs), replace=True)]); r = diffs(d)
        for k in bs: bs[k].append(r[k])
    for k, lab in [('any', 'any disease-relevant factor'), ('uhc', 'UHC index')]:
        brow.append(dict(band=band, measure=lab, n_locations=len(locs), diff_pp=est[k], lo=np.nanquantile(bs[k], .025), hi=np.nanquantile(bs[k], .975)))
BO = pd.DataFrame(brow); BO[['diff_pp', 'lo', 'hi']] = BO[['diff_pp', 'lo', 'hi']].round(6)
BO.to_csv('PART3_SHORTFALL_BOOT.csv', index=False, encoding='utf-8-sig'); print(BO.round(1).to_string(index=False))
