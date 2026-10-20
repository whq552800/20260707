"""对 outcome_factor_tails.py（S1 Text 表R、表S）的三项检查（2026-10-11，独立审稿意见）。输出 OUTCOME_FACTOR_CHECKS.csv、OUTCOME_FACTOR_VACC0.csv。
(1) 疾病特异性（阴性对照）：10项具体因素与6个结局的全部组合中，“匹配”组合（该病预先选定的因素）与“不匹配”组合
    （例如腹泻病与疫苗、脑膜炎与饮水）分别等权平均高尾减低尾的不利比例差，并求两者之差；若短板只是普遍的服务不足，两者应相近。
(2) 非GBD协变量因素：低出生体重、卫生支出、医院床位、护士与助产士不是三种疾病GBD死亡模型的协变量（S1 Text表K），
    对三个死亡结局计算同样的差值；若低SDI层的梯度也出现在这些因素上，就不能完全归因于GBD估计从协变量继承。
(3) Hib3与PCV3引入前年份记为0%的敏感性分析（factor_pipeline.factor_periods(vacc0=True)），重算表R中的疫苗行。
另：(4) 低及中低SDI层晚期各服务因素残差之间的Spearman相关，描述不同服务的短板是否同时出现。
区间均为层内按地点有放回重抽样2000次（种子20261013），每个地点带着它在各结局的尾部归属和因素状态。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '../independent_reproduction_20261009'); sys.path.insert(0, '../factor_residual_audit_20261010')
from statlib import fisher_exact, bh, spearman
pr = pd.read_csv('../independent_reproduction_20261009/REPRO_COUNTRY_DISEASE_PROFILES.csv')
F = pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv')
T = pd.read_csv('../SDI_PERIOD_TIERS.csv')
LATE = '2011–2023'; JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']; MIN_N = 3; NB = 2000
BAND = {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '中等SDI': 'Middle and high-middle', '中高SDI': 'Middle and high-middle', '高SDI': 'High'}
SPEC = ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct', 'stunting_u5_pct', 'low_birthweight_pct', 'gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev']
REL = {'LRI': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct', 'stunting_u5_pct', 'low_birthweight_pct'],
       '腹泻病': ['basic_water_pct', 'basic_sanitation_pct', 'stunting_u5_pct'], '脑膜炎': ['dtp3_pct', 'hib3_pct', 'pcv3_pct']}
NONCOV = ['low_birthweight_pct', 'health_expenditure_pc_current_usd', 'hospital_beds_per1000', 'nurses_midwives_per1000', 'physicians_per1000']
OUT_EN = {'LRI发病': 'LRI incidence', 'LRI死亡': 'LRI mortality', '腹泻病发病': 'Diarrhoeal incidence', '腹泻病死亡': 'Diarrhoeal mortality', '脑膜炎发病': 'Meningitis incidence', '脑膜炎死亡': 'Meningitis mortality'}
TL = T[T.period == LATE].assign(band=lambda d: d.tier.map(BAND))
R = pr[pr.p3_category.isin(['稳健高尾', '稳健低尾'])][['location_id', 'dz', 'p3_category']].copy()
R['tail'] = R.p3_category.map({'稳健高尾': 'high', '稳健低尾': 'low'}); R['disease'] = R.dz.str.replace('发病', '').str.replace('死亡', '')
R['band'] = R.location_id.map(TL.set_index('location_id').band)

def long_table(Fp, factors):
    L = pd.concat([R.assign(factor=f) for f in factors], ignore_index=True)
    L = L.merge(Fp[Fp.period == LATE][['location_id', 'factor', 'status']], on=['location_id', 'factor'], how='left')
    L['status'] = L.status.fillna('不可评价'); L['evaluable'] = L.status.isin(JUDGE); L['adverse'] = L.status == '明确不利'
    return L
def mats(L, b, pairs):
    locs = TL[TL.band == b].location_id.values; ix = {l: i for i, l in enumerate(locs)}
    A = {t: np.zeros((len(locs), len(pairs))) for t in ('high', 'low')}; E = {t: np.zeros((len(locs), len(pairs))) for t in ('high', 'low')}
    sub = L[L.band == b]
    for j, (dz, f) in enumerate(pairs):
        s = sub[(sub.dz == dz) & (sub.factor == f)]
        for t in ('high', 'low'):
            st = s[s['tail'] == t]; ii = st.location_id.map(ix).values; A[t][ii, j] = st.adverse.values; E[t][ii, j] = st.evaluable.values
    return len(locs), A, E
def pdiff(w, A, E):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 100 * (w @ A['high']) / (w @ E['high']) - 100 * (w @ A['low']) / (w @ E['low'])
def comparable(L, b, pairs):
    out = []
    for dz, f in pairs:
        s = L[(L.band == b) & (L.dz == dz) & (L.factor == f)]
        if s[s['tail'] == 'high'].evaluable.sum() >= MIN_N and s[s['tail'] == 'low'].evaluable.sum() >= MIN_N: out.append((dz, f))
    return out
rng = np.random.default_rng(20261013); W = {}
for b in ('Low and low-middle', 'High'):
    n = (TL.band == b).sum(); W[b] = rng.multinomial(n, np.full(n, 1 / n), size=NB).astype(float)
q = lambda x: (np.nanquantile(x, .025), np.nanquantile(x, .975))
rows = []
# (1) 匹配与不匹配
LS = long_table(F, SPEC)
for b in W:
    allp = [(dz, f) for dz in OUT_EN for f in SPEC]
    pairs = comparable(LS, b, allp)
    match = np.array([f in REL[dz.replace('发病', '').replace('死亡', '')] for dz, f in pairs])
    n, A, E = mats(LS, b, pairs); est = pdiff(np.ones(n), A, E); bt = pdiff(W[b], A, E)
    m_e, u_e = np.nanmean(est[match]), np.nanmean(est[~match]); m_b, u_b = np.nanmean(bt[:, match], 1), np.nanmean(bt[:, ~match], 1)
    for lab, e, x, k in [('matched pairs', m_e, m_b, match.sum()), ('unmatched pairs', u_e, u_b, (~match).sum()), ('matched minus unmatched', m_e - u_e, m_b - u_b, len(pairs))]:
        rows.append(dict(check='specificity', band=b, set=lab, n_pairs=int(k), estimate=e, lo=q(x)[0], hi=q(x)[1]))
    for lab, fs in [('unmatched: water and sanitation for LRI and meningitis', ['basic_water_pct', 'basic_sanitation_pct']), ('unmatched: vaccines for diarrhoeal disease', ['dtp3_pct', 'hib3_pct', 'pcv3_pct'])]:
        j = [i for i, (dz, f) in enumerate(pairs) if (not match[i]) and f in fs]
        rows.append(dict(check='specificity', band=b, set=lab, n_pairs=len(j), estimate=np.nanmean(est[j]), lo=q(np.nanmean(bt[:, j], 1))[0], hi=q(np.nanmean(bt[:, j], 1))[1]))
# (2) 非GBD协变量因素（死亡结局）
LN = long_table(F, NONCOV); B2 = {}
mort = [dz for dz in OUT_EN if dz.endswith('死亡')]
pairs_b = {b: comparable(LN, b, [(dz, f) for dz in mort for f in NONCOV]) for b in W}
common = sorted(set(pairs_b['Low and low-middle']) & set(pairs_b['High']))
for b in W:
    n, A, E = mats(LN, b, common); est = pdiff(np.ones(n), A, E); bt = pdiff(W[b], A, E)
    B2[b] = (np.nanmean(est), np.nanmean(bt, 1))
    rows.append(dict(check='non-covariate factors, mortality outcomes', band=b, set='; '.join(f'{OUT_EN[dz]} × {f}' for dz, f in common), n_pairs=len(common), estimate=B2[b][0], lo=q(B2[b][1])[0], hi=q(B2[b][1])[1]))
d = B2['Low and low-middle'][1] - B2['High'][1]
rows.append(dict(check='non-covariate factors, mortality outcomes', band='Low and low-middle minus high', set='', n_pairs=len(common), estimate=B2['Low and low-middle'][0] - B2['High'][0], lo=q(d)[0], hi=q(d)[1]))
for b in W:   # 各因素分别
    for f in NONCOV:
        pp = [(dz, f) for dz in mort if (dz, f) in common]
        if not pp: continue
        n, A, E = mats(LN, b, pp); est = pdiff(np.ones(n), A, E)
        rows.append(dict(check='non-covariate factor, by factor', band=b, set=f, n_pairs=len(pp), estimate=np.nanmean(est), lo=np.nan, hi=np.nan))
# (4) 服务短板是否同时出现：低及中低SDI层晚期残差的相关
FL = F[(F.period == LATE) & F.eligible].pivot_table(index='location_id', columns='factor', values='favourable_residual')
low_locs = TL[TL.band == 'Low and low-middle'].location_id
for a, c in [('basic_water_pct', 'dtp3_pct'), ('basic_sanitation_pct', 'dtp3_pct'), ('basic_water_pct', 'uhc_service_index'), ('basic_sanitation_pct', 'uhc_service_index'), ('dtp3_pct', 'uhc_service_index'), ('basic_water_pct', 'basic_sanitation_pct')]:
    x = FL.reindex(low_locs)[[a, c]].dropna(); rho, p, n = spearman(x[a].values, x[c].values)
    rows.append(dict(check='co-occurrence of service shortfalls (Spearman, low and low-middle SDI)', band='Low and low-middle', set=f'{a} × {c}', n_pairs=n, estimate=rho, lo=np.nan, hi=p))
O = pd.DataFrame(rows); O[['estimate', 'lo', 'hi']] = O[['estimate', 'lo', 'hi']].round(6)
O.to_csv('OUTCOME_FACTOR_CHECKS.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); print(O.round(3).to_string(index=False))
# (3) 疫苗引入前记为0%
from factor_pipeline import factor_periods
FV = factor_periods(ceiling=99, drop=('physicians_per1000',), vacc0=True, only=('hib3_pct', 'pcv3_pct'))
LV = long_table(FV, ['hib3_pct', 'pcv3_pct']); LM = long_table(F, ['hib3_pct', 'pcv3_pct'])
vrows = []
for (b, dz, f), g in LV[LV.disease.isin(['LRI', '脑膜炎'])].groupby(['band', 'dz', 'factor']):
    g0 = LM[(LM.band == b) & (LM.dz == dz) & (LM.factor == f)]
    rec = dict(band=b, outcome=OUT_EN[dz], factor=f)
    for lab, gg in [('main', g0), ('vacc0', g)]:
        h, l = gg[gg['tail'] == 'high'], gg[gg['tail'] == 'low']
        rec.update({f'{lab}_adv_high': int(h.adverse.sum()), f'{lab}_eval_high': int(h.evaluable.sum()), f'{lab}_adv_low': int(l.adverse.sum()), f'{lab}_eval_low': int(l.evaluable.sum())})
        ok = rec[f'{lab}_eval_high'] >= MIN_N and rec[f'{lab}_eval_low'] >= MIN_N
        rec[f'{lab}_p'] = fisher_exact(rec[f'{lab}_adv_high'], rec[f'{lab}_eval_high'] - rec[f'{lab}_adv_high'], rec[f'{lab}_adv_low'], rec[f'{lab}_eval_low'] - rec[f'{lab}_adv_low']) if ok else np.nan
    vrows.append(rec)
V = pd.DataFrame(vrows)
st = FV[FV.period == LATE].merge(F[F.period == LATE][['location_id', 'factor', 'status']], on=['location_id', 'factor'], suffixes=('_vacc0', '_main'))
V.to_csv('OUTCOME_FACTOR_VACC0.csv', index=False, encoding='utf-8-sig'); print(V.to_string(index=False))
print('late-period status changes:', st.groupby('factor').apply(lambda x: f"{(x.status_vacc0 != x.status_main).sum()} of {len(x)}").to_dict())
ix = st[st.location.isin(['Indonesia', 'Nigeria'])][['location', 'factor', 'favourable_residual', 'status_main', 'status_vacc0']]; print(ix.to_string(index=False))

# ---------- 2026-10-11（第二轮独立审稿意见）
# (5) 连续分析：不把疾病和因素二分，在每个SDI层内用全部地点计算晚期疾病残差与有利方向因素偏离的Spearman相关
#     （预期为负：因素越差、负担越高），对“结局×疾病相关因素”组合等权平均，层内按地点重抽样2000次（种子20261014）。
# (6) 最小幅度规则：只有偏离至少2个单位（百分点、指数分、μg/m³或SEV单位）且区间不含0时才计为明确不利，重算表S的汇总。
Dd = pd.read_csv('../independent_reproduction_20261009/REPRO_DISEASE_PERIOD.csv'); Dd = Dd[Dd.period == LATE]
DZK = {(322, 6): 'LRI发病', (322, 1): 'LRI死亡', (302, 6): '腹泻病发病', (302, 1): '腹泻病死亡', (332, 6): '脑膜炎发病', (332, 1): '脑膜炎死亡'}
Dd['dz'] = [DZK[(c, m)] for c, m in zip(Dd.cause_id, Dd.measure_id)]
DR = Dd.pivot_table(index='location_id', columns='dz', values='residual')
FE = F[(F.period == LATE) & F.eligible].pivot_table(index='location_id', columns='factor', values='favourable_residual')
rng2 = np.random.default_rng(20261014); crow = {}
def rk(v): return pd.Series(v).rank().values
for b in W:
    locs = TL[TL.band == b].location_id.values
    pairs = [(dz, f) for dz in OUT_EN for f in REL[dz.replace('发病', '').replace('死亡', '')] + ['uhc_service_index']]
    def mean_rho(ix):
        rs = {'spec': [], 'uhc': []}
        for dz, f in pairs:
            x = FE.reindex(locs[ix])[f].values; y = DR.reindex(locs[ix])[dz].values; ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 10 or np.nanstd(x[ok]) == 0: continue
            rs['uhc' if f == 'uhc_service_index' else 'spec'].append(np.corrcoef(rk(x[ok]), rk(y[ok]))[0, 1])
        return np.mean(rs['spec']), np.mean(rs['uhc'])
    est = mean_rho(np.arange(len(locs))); bt = np.array([mean_rho(rng2.integers(0, len(locs), len(locs))) for _ in range(NB)])
    crow[b] = (est, bt)
    for j, lab in enumerate(['disease-relevant factors', 'UHC index']):
        rows.append(dict(check='continuous: mean Spearman rho, disease residual vs factor departure (all locations in stratum)', band=b, set=lab, n_pairs=len(locs), estimate=est[j], lo=np.nanquantile(bt[:, j], .025), hi=np.nanquantile(bt[:, j], .975)))
for j, lab in enumerate(['disease-relevant factors', 'UHC index']):
    d = crow['Low and low-middle'][1][:, j] - crow['High'][1][:, j]
    rows.append(dict(check='continuous: mean Spearman rho, disease residual vs factor departure (all locations in stratum)', band='Low and low-middle minus high', set=lab, n_pairs=np.nan,
                     estimate=crow['Low and low-middle'][0][j] - crow['High'][0][j], lo=np.nanquantile(d, .025), hi=np.nanquantile(d, .975)))
# (6)
F2 = F.copy(); F2.loc[(F2.status == '明确不利') & (F2.favourable_residual > -2), 'status'] = '方向不确定'
L2 = long_table(F2, SPEC); B6 = {}
for b in W:
    pairs = sorted(set(comparable(L2, b, [(dz, f) for dz in OUT_EN for f in REL[dz.replace('发病', '').replace('死亡', '')]])))
    B6[b] = pairs
common6 = sorted(set(B6['Low and low-middle']) & set(B6['High'])); res6 = {}
for b in W:
    n, A, E = mats(L2, b, common6); est = pdiff(np.ones(n), A, E); bt = pdiff(W[b], A, E)
    j = [i for i, (dz, f) in enumerate(common6) if f in ('basic_water_pct', 'basic_sanitation_pct')]
    res6[b] = (np.nanmean(est), np.nanmean(bt, 1), np.nanmean(est[j]), np.nanmean(bt[:, j], 1))
    rows.append(dict(check='minimum magnitude 2 units: high minus low tail, pp', band=b, set='all disease-relevant factors except UHC', n_pairs=len(common6), estimate=res6[b][0], lo=q(res6[b][1])[0], hi=q(res6[b][1])[1]))
    rows.append(dict(check='minimum magnitude 2 units: high minus low tail, pp', band=b, set='water and sanitation', n_pairs=len(j), estimate=res6[b][2], lo=q(res6[b][3])[0], hi=q(res6[b][3])[1]))
for k, lab in [(1, 'all disease-relevant factors except UHC'), (3, 'water and sanitation')]:
    d = res6['Low and low-middle'][k] - res6['High'][k]
    rows.append(dict(check='minimum magnitude 2 units: high minus low tail, pp', band='Low and low-middle minus high', set=lab, n_pairs=len(common6), estimate=res6['Low and low-middle'][k - 1] - res6['High'][k - 1], lo=q(d)[0], hi=q(d)[1]))
rm = int(((F.period == LATE) & (F.status == '明确不利') & (F.favourable_residual > -2) & F.factor.isin(SPEC + ['uhc_service_index'])).sum())
rows.append(dict(check='minimum magnitude 2 units: clearly adverse late-period records reclassified', band='all', set='', n_pairs=rm, estimate=np.nan, lo=np.nan, hi=np.nan))
O = pd.DataFrame(rows); O[['estimate', 'lo', 'hi']] = O[['estimate', 'lo', 'hi']].round(6)
O.to_csv('OUTCOME_FACTOR_CHECKS.csv', index=False, encoding='utf-8-sig')
print(O[O.check.str.startswith(('continuous', 'minimum'))].drop(columns='set').round(3).to_string(index=False))
