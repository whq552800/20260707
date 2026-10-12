"""按疾病结局分别比较2011—2023年稳健高尾与稳健低尾地点中，各预先指定的疾病相关因素“明确不利”的地点比例（S1 Text Table R）。
2026-10-11 新增：Fig 3A 合并了全部18项因素、六个结局和三个时期，同一地点会被重复计数；本步骤改为每个结局单独计数，
只用该结局预先指定的因素（与 part3_typology.py 相同；UHC指数单独列出），每个地点在每行中只计一次，便于直接回答
“某结局高尾地点中有多大比例存在某项短板”。
规则：时期 2011–2023；SDI层按该期SDI均值的GBD 2023固定等级合并为三层；因素状态取同期判定。
可评价 = 明确不利 + 明确有利 + 方向不确定 + 已达普及（与 Fig 3A 相同；已达普及计入分母、不计为不利）；不可评价不计入分母。
比较：比例差（百分点）及 Newcombe 混合得分95%区间；双侧 Fisher 精确检验；同一SDI层内对34个“结局×因素”比较做 BH 校正，
“至少一项具体因素不利”的6个比较另行校正。任一组可评价地点少于3个时不作比较。"""
import pandas as pd, numpy as np
import sys; sys.path.insert(0, '../independent_reproduction_20261009')
from statlib import fisher_exact, bh

pr = pd.read_csv('../independent_reproduction_20261009/REPRO_COUNTRY_DISEASE_PROFILES.csv')
F = pd.read_csv('../factor_profiles_confirmed_20261010/CONFIRMED_FACTOR_PERIOD.csv')
T = pd.read_csv('../SDI_PERIOD_TIERS.csv')
LATE = '2011–2023'
REL = {'LRI': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct',
               'stunting_u5_pct', 'low_birthweight_pct', 'uhc_service_index'],
       '腹泻病': ['basic_water_pct', 'basic_sanitation_pct', 'stunting_u5_pct', 'uhc_service_index'],
       '脑膜炎': ['dtp3_pct', 'hib3_pct', 'pcv3_pct', 'uhc_service_index']}
OUT_EN = {'LRI发病': 'LRI incidence', 'LRI死亡': 'LRI mortality', '腹泻病发病': 'Diarrhoeal incidence',
          '腹泻病死亡': 'Diarrhoeal mortality', '脑膜炎发病': 'Meningitis incidence', '脑膜炎死亡': 'Meningitis mortality'}
BAND = {'低SDI': 'Low and low-middle', '中低SDI': 'Low and low-middle', '中等SDI': 'Middle and high-middle',
        '中高SDI': 'Middle and high-middle', '高SDI': 'High'}
JUDGE = ['明确不利', '明确有利', '方向不确定', '已达普及']
MIN_N = 3

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2; l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2); d = p1 - p2
    return 100 * (d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 100 * (d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))

R = pr[['location_id', 'location', 'dz', 'p3_category', 'p3_rate_ratio']].copy()
R = R[R.p3_category.isin(['稳健高尾', '稳健低尾'])]
R['tail'] = R.p3_category.map({'稳健高尾': 'high', '稳健低尾': 'low'})
R['disease'] = R.dz.str.replace('发病', '').str.replace('死亡', '')
R['outcome'] = R.dz.map(OUT_EN)
R['band'] = R.location_id.map(T[T.period == LATE].set_index('location_id').tier).map(BAND)
FL = F[F.period == LATE][['location_id', 'factor', 'status', 'favourable_residual', 'favourable_lower', 'favourable_upper']]

# 长表：每个尾部地点 × 该结局的预先指定因素
L = pd.concat([R[R.disease == d].assign(factor=f) for d in REL for f in REL[d]], ignore_index=True)
L = L.merge(FL, on=['location_id', 'factor'], how='left'); L['status'] = L.status.fillna('不可评价')
L['evaluable'] = L.status.isin(JUDGE); L['adverse'] = L.status == '明确不利'

def compare(g):
    out = {}
    for t in ('high', 'low'):
        s = g[g['tail'] == t]
        out[f'n_{t}'] = int(s.location_id.nunique()); out[f'eval_{t}'] = int(s.evaluable.sum()); out[f'adv_{t}'] = int(s.adverse.sum())
        out[f'pct_{t}'] = 100 * out[f'adv_{t}'] / out[f'eval_{t}'] if out[f'eval_{t}'] else np.nan
    ok = out['eval_high'] >= MIN_N and out['eval_low'] >= MIN_N
    out['diff_pp'] = out['pct_high'] - out['pct_low'] if ok else np.nan
    lo, hi = newcombe(out['adv_high'], out['eval_high'], out['adv_low'], out['eval_low']) if ok else (np.nan, np.nan)
    out['diff_lo'], out['diff_hi'] = lo, hi
    out['p_fisher'] = fisher_exact(out['adv_high'], out['eval_high'] - out['adv_high'], out['adv_low'], out['eval_low'] - out['adv_low']) if ok else np.nan
    return pd.Series(out)

rows = []
for (band, dz, f), g in L.groupby(['band', 'dz', 'factor'], sort=False):
    rows.append(dict(band=band, dz=dz, outcome=OUT_EN[dz], factor=f, kind='uhc' if f == 'uhc_service_index' else 'factor', **compare(g)))
# 至少一项具体因素明确不利（分母：至少一项具体因素可评价的地点；UHC不计入）
S = L[L.factor != 'uhc_service_index'].groupby(['band', 'dz', 'location_id', 'tail']).agg(evaluable=('evaluable', 'any'), adverse=('adverse', 'any')).reset_index()
for (band, dz), g in S.groupby(['band', 'dz'], sort=False):
    rows.append(dict(band=band, dz=dz, outcome=OUT_EN[dz], factor='any_specific', kind='any', **compare(g)))
O = pd.DataFrame(rows)
O['q_bh'] = np.nan
for (band, kind), idx in O.groupby(['band', O.kind.where(O.kind == 'any', 'factor')]).groups.items():
    O.loc[idx, 'q_bh'] = bh(O.loc[idx, 'p_fisher'].values)
order_b = {'Low and low-middle': 0, 'Middle and high-middle': 1, 'High': 2}
order_o = {v: i for i, v in enumerate(OUT_EN.values())}
O = O.sort_values(['band', 'outcome', 'kind', 'factor'], key=lambda s: s.map(order_b) if s.name == 'band' else (s.map(order_o) if s.name == 'outcome' else s))
num = ['pct_high', 'pct_low', 'diff_pp', 'diff_lo', 'diff_hi', 'p_fisher', 'q_bh']
O[num] = O[num].round(6)
O.to_csv('OUTCOME_FACTOR_TAILS.csv', index=False, encoding='utf-8-sig')
L.sort_values(['dz', 'tail', 'band', 'location', 'factor'])[['location_id', 'location', 'dz', 'outcome', 'tail', 'band', 'p3_rate_ratio', 'factor', 'status',
    'favourable_residual', 'favourable_lower', 'favourable_upper']].round(6).to_csv('OUTCOME_FACTOR_TAILS_LOCATIONS.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
show = O[['band', 'outcome', 'factor', 'adv_high', 'eval_high', 'n_high', 'adv_low', 'eval_low', 'n_low', 'diff_pp', 'diff_lo', 'diff_hi', 'p_fisher', 'q_bh']].copy()
print(show.round(3).to_string(index=False))

# ---------- 2026-10-11（审稿意见R2-M2）：SDI层之间的差别——固定权重、按地点重抽样
# 每个“结局×因素”对等权；只用两层中高尾、低尾可评价地点均≥3的配对，使两层比较的是同一组配对；
# 在每层内按地点有放回重抽样（地点带着它在各结局的尾部归属和因素状态），2000次，种子20261011。
DOMS = {'Water and sanitation': ['basic_water_pct', 'basic_sanitation_pct'], 'Immunisation': ['dtp3_pct', 'hib3_pct', 'pcv3_pct'],
        'Clean cooking': ['clean_cooking_pct'], 'Nutrition': ['stunting_u5_pct', 'low_birthweight_pct'],
        'Exposure': ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev'], 'UHC index': ['uhc_service_index']}
DOMS['All disease-relevant factors'] = [f for k, v in DOMS.items() if k != 'UHC index' for f in v]
TL = T[T.period == LATE].assign(band=lambda d: d.tier.map(BAND))
B_LOW, B_HIGH, NB = 'Low and low-middle', 'High', 2000
FO = O[O.kind != 'any']
ok_pairs = {b: set(map(tuple, FO[(FO.band == b) & (FO.eval_high >= MIN_N) & (FO.eval_low >= MIN_N)][['dz', 'factor']].values)) for b in (B_LOW, B_HIGH)}
common = sorted(ok_pairs[B_LOW] & ok_pairs[B_HIGH])
def band_mats(b):
    locs = TL[TL.band == b].location_id.values; ix = {l: i for i, l in enumerate(locs)}
    A = {t: np.zeros((len(locs), len(common))) for t in ('high', 'low')}; E = {t: np.zeros((len(locs), len(common))) for t in ('high', 'low')}
    sub = L[L.band == b]
    for j, (dz, f) in enumerate(common):
        s = sub[(sub.dz == dz) & (sub.factor == f)]
        for t in ('high', 'low'):
            st = s[s['tail'] == t]; ii = st.location_id.map(ix).values
            A[t][ii, j] = st.adverse.values; E[t][ii, j] = st.evaluable.values
    return len(locs), A, E
def pair_diff(w, A, E):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 100 * (w @ A['high']) / (w @ E['high']) - 100 * (w @ A['low']) / (w @ E['low'])
rng = np.random.default_rng(20261011); est, boot = {}, {}
for b in (B_LOW, B_HIGH):
    n, A, E = band_mats(b)
    est[b] = pair_diff(np.ones(n), A, E)
    W = rng.multinomial(n, np.full(n, 1 / n), size=NB).astype(float)
    boot[b] = pair_diff(W, A, E)                                     # NB × 配对
grows = []
for dom, fs in DOMS.items():
    j = [k for k, (dz, f) in enumerate(common) if f in fs]
    if not j: continue
    e = {b: np.nanmean(est[b][j]) for b in est}; bt = {b: np.nanmean(boot[b][:, j], axis=1) for b in boot}
    dd = bt[B_LOW] - bt[B_HIGH]
    grows.append(dict(domain=dom, n_pairs=len(j), pairs='; '.join(f'{OUT_EN[common[k][0]]} × {common[k][1]}' for k in j),
                      low_band_diff=e[B_LOW], low_lo=np.nanquantile(bt[B_LOW], .025), low_hi=np.nanquantile(bt[B_LOW], .975),
                      high_band_diff=e[B_HIGH], high_lo=np.nanquantile(bt[B_HIGH], .025), high_hi=np.nanquantile(bt[B_HIGH], .975),
                      between_band=e[B_LOW] - e[B_HIGH], between_lo=np.nanquantile(dd, .025), between_hi=np.nanquantile(dd, .975)))
G = pd.DataFrame(grows); nc = [c for c in G.columns if c not in ('domain', 'n_pairs', 'pairs')]; G[nc] = G[nc].round(6)
G.to_csv('OUTCOME_FACTOR_GRADIENT.csv', index=False, encoding='utf-8-sig')
print(G.drop(columns='pairs').round(1).to_string(index=False))
