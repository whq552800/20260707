"""S1 Text 敏感性分析的公共计算：给定一套疾病时期残差/分类（及一套因素状态），计算正文各项结论的数值。
主分析输入时，各数值与正文逐项一致（见 check_main.py）。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
A = HERE.parent   # analysis/
sys.path.insert(0, str(A / 'independent_reproduction_20261009'))
from statlib import spearman
P3 = ['1990–2000', '2001–2010', '2011–2023']; E0, M0, L0 = P3
OUTS = [(302, 1), (322, 1), (332, 1), (302, 6), (322, 6), (332, 6)]
SOUTH_AFRICA = 196
T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')
REG = pd.read_csv(A / 'substantive_findings_20261010' / 'A_PROFILES_WITH_REGION.csv').drop_duplicates('location_id').set_index('location_id').region
ENV = pd.read_csv(A / 'substantive_findings_20261010' / 'R4_ENV_PERIOD.csv')
PM = ENV[(ENV.period == L0) & (ENV.env == '环境PM2.5')].set_index('location_id').dev
BAND = {'低SDI': 'low', '中低SDI': 'low', '中等SDI': 'mid', '中高SDI': 'mid', '高SDI': 'high'}
# 18项因素的领域（与 find_patterns.py 一致；家庭PM2.5不计入）
DOM = {'basic_water_pct': 'WaSH', 'basic_sanitation_pct': 'WaSH', 'clean_cooking_pct': 'Cooking', 'dtp3_pct': 'Immun', 'hib3_pct': 'Immun', 'pcv3_pct': 'Immun',
       'stunting_u5_pct': 'Nutrition', 'low_birthweight_pct': 'Nutrition', 'undernourishment_pct': 'Nutrition', 'uhc_service_index': 'Health', 'health_expenditure_pc_current_usd': 'Health',
       'hospital_beds_per1000': 'Health', 'nurses_midwives_per1000': 'Health', 'gbd_ambient_pm25_ug_m3': 'Pollution', 'secondhand_smoke_sev': 'Pollution',
       'income_ppp_pc': 'Socio', 'secondary_enrolment_pct': 'Socio', 'fertility_births_per_woman': 'Socio'}
JUDGED = ['明确不利', '明确有利', '方向不确定', '已达普及']
# 持续尾部的疾病相关候选因素（与 part3_typology.py 相同）。2026-10-11：UHC指数为综合指数，单独报告，不并入领域（PDOM 中记为 'U'）
PDOM = {'basic_water_pct': 'B', 'basic_sanitation_pct': 'B', 'clean_cooking_pct': 'B', 'uhc_service_index': 'U', 'dtp3_pct': 'I', 'hib3_pct': 'I', 'pcv3_pct': 'I',
        'stunting_u5_pct': 'N', 'low_birthweight_pct': 'N', 'gbd_ambient_pm25_ug_m3': 'X', 'secondhand_smoke_sev': 'X'}
REL = {322: ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct', 'stunting_u5_pct', 'low_birthweight_pct', 'uhc_service_index'],
       302: ['basic_water_pct', 'basic_sanitation_pct', 'stunting_u5_pct', 'uhc_service_index'], 332: ['dtp3_pct', 'hib3_pct', 'pcv3_pct', 'uhc_service_index']}
EARLY_HIGH = set(T[(T.period == E0) & (T.tier == '高SDI')].location_id); LATE_HIGH = set(T[(T.period == L0) & (T.tier == '高SDI')].location_id)
ENTRANTS = LATE_HIGH - EARLY_HIGH
_M = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv').pivot_table(index=['cause_id', 'measure_id', 'location_id'], columns='period', values='category', aggfunc='first')
MAIN_PERSIST = {'high': set(_M.index[(_M == '稳健高尾').all(axis=1)]), 'low': set(_M.index[(_M == '稳健低尾').all(axis=1)])}

def classify(r, lo=0.25, hi=0.75):
    """先区间判方向，再在方向明确者中按分位数分尾；r 需含 cause_id, measure_id, period, residual, direction。"""
    r = r.copy(); key = ['cause_id', 'measure_id', 'period']
    res = r.residual.where(r.direction != 'undetermined')
    q1 = res.groupby([r[k] for k in key]).transform(lambda x: np.nanquantile(x, lo))
    q3 = res.groupby([r[k] for k in key]).transform(lambda x: np.nanquantile(x, hi))
    r['category'] = np.where(r.direction == 'undetermined', '参照不确定', np.where(r.residual < q1, '稳健低尾', np.where(r.residual > q3, '稳健高尾', '明确偏离但居中')))
    return r

def disease_claims(r, has_dir=True):
    out = {}
    W = r.pivot_table(index=['cause_id', 'measure_id', 'location_id'], columns='period', values='residual')
    C = r.pivot_table(index=['cause_id', 'measure_id', 'location_id'], columns='period', values='category', aggfunc='first')
    rh = [spearman(W.loc[o][E0].values, W.loc[o][L0].values)[0] for o in OUTS]
    out['rank_rho_min'], out['rank_rho_max'] = min(rh), max(rh)
    e, l = C[E0], C[L0]
    out['early_high_n'] = int((e == '稳健高尾').sum()); out['early_high_retain_pct'] = 100 * ((e == '稳健高尾') & (l == '稳健高尾')).sum() / (e == '稳健高尾').sum()
    out['early_low_retain_pct'] = 100 * ((e == '稳健低尾') & (l == '稳健低尾')).sum() / (e == '稳健低尾').sum()
    out['tail_swaps'] = int((((e == '稳健高尾') & (l == '稳健低尾')) | ((e == '稳健低尾') & (l == '稳健高尾'))).sum())
    ph = (C == '稳健高尾').all(axis=1); n_out = ph[ph].reset_index().groupby('location_id').size()
    # 2026-10-11：持续尾部名单与主分析的重合（审稿意见：报告标签稳定性，而不仅是高低尾互换）
    pl = (C == '稳健低尾').all(axis=1)
    locs = set(r.location_id)   # 比较限于该设定中出现的地点（排除小地点时）
    for lab, cur, ref in [('high', set(ph[ph].index), {k for k in MAIN_PERSIST['high'] if k[2] in locs}), ('low', set(pl[pl].index), {k for k in MAIN_PERSIST['low'] if k[2] in locs})]:
        out[f'persist_{lab}_n'] = len(cur); out[f'persist_{lab}_retained_pct'] = 100 * len(cur & ref) / len(ref) if ref else np.nan
        out[f'persist_{lab}_new_n'] = len(cur - ref)

    out['persist_high_locations'] = len(n_out); out['persist_high_1to2_pct'] = 100 * (n_out <= 2).mean(); out['persist_high_all6'] = int((n_out == 6).sum())
    out['southafrica_persist_n'] = int(n_out.get(SOUTH_AFRICA, 0))
    rhos = {}
    for m, lab in [(1, 'mort'), (6, 'inc')]:
        v = []
        for p in P3:
            for a, b in [(302, 322), (302, 332), (322, 332)]:
                v.append(spearman(W.loc[(a, m)][p].values, W.loc[(b, m)][p].reindex(W.loc[(a, m)].index).values)[0])
        out[f'cross_{lab}_min'], out[f'cross_{lab}_max'] = min(v), max(v); rhos[lab] = v
    out['cross_mort_gt_inc_periods'] = sum(min(rhos['mort'][3 * i:3 * i + 3]) > max(rhos['inc'][3 * i:3 * i + 3]) for i in range(3))
    out['cross_mort_mean'], out['cross_inc_mean'] = float(np.mean(rhos['mort'])), float(np.mean(rhos['inc']))
    for c in [302, 322, 332]:
        x = W.loc[(c, 6)][L0]; y = W.loc[(c, 1)][L0].reindex(x.index); out[f'incdeath_rho_{c}'] = spearman(x.values, y.values)[0]
    if has_dir:
        Dd = r[r.period == L0].pivot_table(index='location_id', columns=['cause_id', 'measure_id'], values='direction', aggfunc='first')
        ok = (Dd != 'undetermined').all(1); s = Dd[ok]
        out['resolved_all6_n'] = int(ok.sum())
        out['same_dir_mort_pct'] = 100 * ((s[(302, 1)] == s[(322, 1)]) & (s[(322, 1)] == s[(332, 1)])).mean()
        out['same_dir_inc_pct'] = 100 * ((s[(302, 6)] == s[(322, 6)]) & (s[(322, 6)] == s[(332, 6)])).mean()
        for c in [302, 322, 332]:
            z = Dd[[(c, 1), (c, 6)]]; z = z[(z != 'undetermined').all(1)]; out[f'opposite_pct_{c}'] = 100 * (z[(c, 1)] != z[(c, 6)]).mean()
    lc = C.loc[(322, 1)][L0]; ent = [i for i in sorted(ENTRANTS) if i in lc.index]; ori = [i for i in sorted(EARLY_HIGH) if i in lc.index]
    out['entrants_LRIdeath_high'] = int((lc.reindex(ent) == '稳健高尾').sum()); out['entrants_n'] = len(ent)
    out['original_LRIdeath_high'] = int((lc.reindex(ori) == '稳健高尾').sum()); out['original_n'] = len(ori)
    li = C.loc[(322, 6)][L0]; hs = [i for i in sorted(LATE_HIGH) if i in li.index]
    out['highsdi_LRIinc_high'] = int((li.reindex(hs) == '稳健高尾').sum()); out['highsdi_n'] = len(hs)
    q = pd.DataFrame({'res': W.loc[(322, 6)][L0].reindex(hs)}); q['pm'] = PM.reindex(q.index); q['reg'] = REG.reindex(q.index); q = q.dropna()
    q = q[q.groupby('reg').res.transform('size') >= 2]
    out['pm25_within_rho'] = spearman((q.pm - q.groupby('reg').pm.transform('mean')).values, (q.res - q.groupby('reg').res.transform('mean')).values)[0]
    return out

def factor_claims(r, F, catchup=True):
    """r：疾病分类；F：因素状态（CONFIRMED 格式）。返回 Fig 3A/3B 与 Fig 4A 的关键数值。"""
    out = {}
    tails = r[r.category.isin(['稳健高尾', '稳健低尾'])][['location_id', 'cause_id', 'measure_id', 'period', 'category']].merge(T[['location_id', 'period', 'tier']], on=['location_id', 'period'])
    tails['band'] = tails.tier.map(BAND)
    Fj = F[F.status.isin(JUDGED) & F.factor.isin(DOM)][['location_id', 'period', 'factor', 'status']].copy(); Fj['dom'] = Fj.factor.map(DOM)
    m = tails.merge(Fj, on=['location_id', 'period'])
    g = m.groupby(['dom', 'band', 'category']).status.agg(lambda s: 100 * (s == '明确不利').mean()).unstack('category')
    d = g['稳健高尾'] - g['稳健低尾']
    for dm in ['WaSH', 'Immun', 'Cooking', 'Pollution']:
        for b in ['low', 'high']: out[f'diff_{dm}_{b}'] = d.get((dm, b), np.nan)
    C = r.pivot_table(index=['location_id', 'cause_id', 'measure_id'], columns='period', values='category', aggfunc='first')
    per = pd.Series(np.where((C == '稳健高尾').all(axis=1), 'H', np.where((C == '稳健低尾').all(axis=1), 'L', '')), index=C.index); per = per[per != '']
    FL = F[(F.period == L0)].set_index(['location_id', 'factor']).status
    tl = T[T.period == L0].set_index('location_id').tier.map(BAND)
    rows = []
    for (loc, c, mm), p in per.items():
        spec = [f for f in REL[c] if f != 'uhc_service_index']
        ev = [f for f in spec if FL.get((loc, f)) in JUDGED]; u = FL.get((loc, 'uhc_service_index'))
        rows.append((p, tl.get(loc), (any(FL.get((loc, f)) == '明确不利' for f in spec) if ev else np.nan), ((u == '明确不利') if u in JUDGED else np.nan)))
    R = pd.DataFrame(rows, columns=['p', 'band', 'anyspec', 'uhc'])
    for b in ['low', 'high']:
        for p in 'HL':
            x = R[(R.band == b) & (R.p == p)]; out[f'n_{p}_{b}'] = len(x)
            out[f'anyspec_{p}_{b}_pct'] = 100 * x.anyspec.dropna().astype(float).mean(); out[f'uhc_{p}_{b}_pct'] = 100 * x.uhc.dropna().astype(float).mean()
    if catchup:
        D = r.pivot_table(index=['cause_id', 'measure_id', 'location_id'], columns='period', values='residual')
        for f, c, lab in [('basic_water_pct', 302, 'water'), ('basic_sanitation_pct', 302, 'sanit'), ('dtp3_pct', 322, 'dtp3')]:
            fw = F[F.factor == f].pivot(index='location_id', columns='period', values=['favourable_residual', 'status'])
            fw = fw[(fw[('status', M0)] == '明确不利') & fw[('status', L0)].isin(JUDGED)]
            rw = D.loc[(c, 1)]; fw = fw[fw.index.isin(rw.index)]
            nar = fw[('favourable_residual', L0)] > fw[('favourable_residual', M0)]
            imp = (rw[L0] < rw[M0]).reindex(fw.index)
            out[f'catch_{lab}_nar_pct'] = 100 * imp[nar].mean(); out[f'catch_{lab}_not_pct'] = 100 * imp[~nar].mean(); out[f'catch_{lab}_n'] = f'{int(nar.sum())}/{int((~nar).sum())}'
    return out
