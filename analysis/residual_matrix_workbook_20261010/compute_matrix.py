"""国家—疾病—因素发展残差矩阵（2026-10-10 现行算法）：计算部分。
疾病：6个参照模型（同正文），时期残差=年度对数残差均值，CR1区间；三期分类=先区间后Q1/Q3（与复现结果逐条核对）；另算1990–2023全程。
因素：确认版18项（环境PM2.5改用GBD来源 gbd_ambient_pm25_ug_m3，覆盖更全）+ 家庭PM2.5（由气候表移入），原单位残差+delta区间，覆盖率类"已达普及"规则（观测与参照均值≥99%），医生密度不纳入。
气候：ERA5六项（不含环境与家庭PM2.5），变换同正文环境分析，原单位残差+delta区间。
输出：MATRIX_DISEASE.csv, MATRIX_FACTOR.csv, MATRIX_CLIMATE.csv, MATRIX_SDI.csv"""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; A = HERE.parent; R = A / 'independent_reproduction_20261009'
sys.path[:0] = [str(R), str(A)]
from paths import PANEL
from repro_common import ns_basis, period_of
from sdi_tiers import tier
import statlib, repro_factor_defs as D
P3 = ['1990–2000', '2001–2010', '2011–2023']; FULL = '1990–2023'
PLEN = {'1990–2000': 11, '2001–2010': 10, '2011–2023': 13, FULL: 34}
pan = pd.read_csv(PANEL); pan['period'] = pan.year.map(period_of)
ly = pan.drop_duplicates(['location_id', 'year']).copy()
KN_ALL = np.quantile(ly.sdi, [.25, .5, .75])

def design(g, kn):
    B = ns_basis(g.sdi.values, kn); YF = pd.get_dummies(g.year, drop_first=True).astype(float).values   # 与确认版相同：以该变量最早年份为基线
    return np.column_stack([np.ones(len(g)), YF, B, B * ((g.year.values - 2006.5) / 33)[:, None]])

def cr1(X, e, loc):
    Xi = np.linalg.pinv(X.T @ X); n, k = X.shape; M = np.zeros((k, k)); G = len(np.unique(loc))
    for l in np.unique(loc):
        idx = loc == l; s = X[idx].T @ e[idx]; M += np.outer(s, s)
    return Xi @ M @ Xi * (G / (G - 1)) * ((n - 1) / (n - k)), statlib.t_ppf(.975, G - 1)

# ---------- 疾病
DZ = {302: '下呼吸道感染', 322: None}
CN_C = {322: '下呼吸道感染', 332: '脑膜炎', 302: '腹泻病'}; CN_M = {6: '发病率', 1: '死亡率'}
d = pan[pan.rate_valid & (pan.gbd_rate_per_100k > 0)].copy(); d['y'] = np.log(d.gbd_rate_per_100k)
rows = []
for (c, m), g in d.groupby(['cause_id', 'measure_id']):
    g = g.sort_values(['location_id', 'year']).reset_index(drop=True); X = design(g, KN_ALL)
    b = np.linalg.pinv(X.T @ X) @ X.T @ g.y.values; e = g.y.values - X @ b; V, t = cr1(X, e, g.location_id.values)
    for per_col in ['period', 'full']:
        key = g.period if per_col == 'period' else pd.Series(FULL, index=g.index)
        for (l, p), idx in g.groupby([g.location_id, key]).indices.items():
            a = X[idx].mean(0); se = np.sqrt(a @ V @ a); r = e[idx].mean()
            rows.append(dict(location_id=l, cause_id=c, measure_id=m, period=p, residual=r, lower=r - t * se, upper=r + t * se))
Dm = pd.DataFrame(rows)
ref = pd.read_csv(R / 'REPRO_DISEASE_PERIOD.csv')
chk = Dm.merge(ref, on=['location_id', 'cause_id', 'measure_id', 'period'], suffixes=('', '_r'))
assert len(chk) == 3672 and np.allclose(chk.residual, chk.residual_r) and np.allclose(chk.lower, chk.lower_r)
Dm = Dm.merge(ref[['location_id', 'cause_id', 'measure_id', 'period', 'category']], on=['location_id', 'cause_id', 'measure_id', 'period'], how='left')
Dm['category'] = Dm.category.fillna('全程不另分类')
prof = pd.read_csv(R / 'REPRO_COUNTRY_DISEASE_PROFILES.csv')
DZK = {'LRI发病': (322, 6), 'LRI死亡': (322, 1), '脑膜炎发病': (332, 6), '脑膜炎死亡': (332, 1), '腹泻病发病': (302, 6), '腹泻病死亡': (302, 1)}
prof['cause_id'] = prof.dz.map(lambda k: DZK[k][0]); prof['measure_id'] = prof.dz.map(lambda k: DZK[k][1])
def traj(r):
    c = [r.p1_category, r.p2_category, r.p3_category]
    if r.group == '持续高尾': return '三个时期持续高尾'
    if r.group == '持续低尾': return '三个时期持续低尾'
    if c[0] == '稳健高尾' and c[2] == '稳健低尾': return '早期高尾转为晚期低尾'
    if c[0] == '稳健低尾' and c[2] == '稳健高尾': return '早期低尾转为晚期高尾'
    return '其他时期组合'
prof['trajectory'] = prof.apply(traj, axis=1)
Dm = Dm.merge(prof[['location_id', 'cause_id', 'measure_id', 'trajectory']], on=['location_id', 'cause_id', 'measure_id'])
Dm['disease'] = Dm.cause_id.map(CN_C); Dm['outcome'] = Dm.measure_id.map(CN_M)
Dm.to_csv(HERE / 'MATRIX_DISEASE.csv', index=False, encoding='utf-8-sig')

# ---------- 连续变量通用拟合
def tf(kind):
    if kind == 'logit':
        return (lambda v: (lambda p: np.log(p / (1 - p)))(np.clip(v, .1, 99.9) / 100), lambda z: 100 / (1 + np.exp(-z)), lambda z: (lambda s: 100 * s * (1 - s))(1 / (1 + np.exp(-z))))
    if kind == 'log': return (np.log, np.exp, np.exp)
    if kind == 'log1p': return (np.log1p, np.expm1, np.exp)
    return (lambda v: v, lambda z: z, lambda z: np.ones_like(z))

def fit_var(col, kind, sign, knots='own', ceiling=False):
    g = ly[['location_id', 'year', 'period', 'sdi', col]].dropna()
    if kind == 'log': g = g[g[col] > 0]
    g = g.sort_values(['location_id', 'year']).reset_index(drop=True)
    fwd, inv, dinv = tf(kind); kn = np.quantile(g.sdi, [.25, .5, .75]) if knots == 'own' else KN_ALL
    X = design(g, kn); z = fwd(g[col].values); b = np.linalg.pinv(X.T @ X) @ X.T @ z; fz = X @ b; e = z - fz
    V, t = cr1(X, e, g.location_id.values); fit = inv(fz); res = g[col].values - fit
    out = []
    for key in ['period', 'full']:
        kk = g.period if key == 'period' else pd.Series(FULL, index=g.index)
        for (l, p), idx in g.groupby([g.location_id, kk]).indices.items():
            grad = (dinv(fz[idx])[:, None] * X[idx]).mean(0); se = np.sqrt(grad @ V @ grad); r = res[idx].mean()
            om, rm = g[col].values[idx].mean(), fit[idx].mean()
            out.append(dict(location_id=l, period=p, n_years=len(idx), residual=r, lower=r - t * se, upper=r + t * se, obs_mean=om, ref_mean=rm,
                            saturated=bool(ceiling and om >= 99 and rm >= 99)))
    o = pd.DataFrame(out)
    # 2026-10-11：早期只有2000年一个锚点年份的序列（所有地点同一年），单年可作早期参照（与 factor_pipeline 一致）
    anchor = g[g.period == P3[0]].year.unique().tolist() == [2000]
    o['single_year_2000'] = anchor & (o.period == P3[0]) & (o.n_years == 1)
    few = (o.n_years < 2) & ~o.single_year_2000
    lo_f = np.where(sign >= 0, o.lower, -o.upper); hi_f = np.where(sign >= 0, o.upper, -o.lower)
    o['status'] = np.where(o.saturated, '已达普及', np.where(few, '有效年份不足',
                  np.where(sign == 0, '无统一方向', np.where(lo_f > 0, '明确有利', np.where(hi_f < 0, '明确不利', '方向不确定')))))
    return o

FACT = [('stunting_u5_pct', '5岁以下儿童生长迟缓率', '5岁以下儿童百分比'), ('hib3_pct', 'Hib疫苗三剂覆盖率', '1岁儿童百分比'), ('secondary_enrolment_pct', '中等教育毛入学率', '百分比，可超过100'),
        ('secondhand_smoke_sev', '二手烟/环境烟草烟雾', 'SEV（0–100分）'), ('income_ppp_pc', '人均GDP（购买力平价）', '2021年不变国际元/人'), ('health_expenditure_pc_current_usd', '人均卫生支出', '当年美元/人'),
        ('low_birthweight_pct', '低出生体重率', '出生百分比'), ('uhc_service_index', '全民健康覆盖服务指数', '0–100指数'), ('basic_sanitation_pct', '基本卫生设施覆盖率', '人口百分比'),
        ('basic_water_pct', '基本饮水服务覆盖率', '人口百分比'), ('fertility_births_per_woman', '总和生育率', '出生数/女性'), ('hospital_beds_per1000', '每千人医院床位数', '床位/千人'),
        ('nurses_midwives_per1000', '每千人护士与助产士数', '人/千人'), ('clean_cooking_pct', '清洁烹饪覆盖率', '人口百分比'), ('gbd_ambient_pm25_ug_m3', '环境PM₂.₅年均暴露', '微克/立方米（GBD估计）'),
        ('dtp3_pct', '百白破疫苗接种覆盖率', '12–23月龄儿童百分比'), ('pcv3_pct', '肺炎球菌结合疫苗覆盖率', '达到当地推荐接种年龄的儿童百分比'), ('undernourishment_pct', '营养不足人口比例', '人口百分比'),
        ('gbd_household_pm25_ug_m3', '家庭PM₂.₅', '微克/立方米（GBD估计）')]
COV = ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct']
fr = []
for col, name, unit in FACT:
    kind = 'logit' if col in D.LOGIT else ('log' if col in D.LOG or col == 'gbd_household_pm25_ug_m3' else 'log1p')
    sign = -1 if (col in D.LOWER_BETTER or col == 'gbd_household_pm25_ug_m3') else 1
    o = fit_var(col, kind, sign, 'own', col in COV); o['factor'] = col; o['name'] = name; o['unit'] = unit; o['rule'] = '越低越有利' if sign < 0 else '越高越有利'; fr.append(o)
Fm = pd.concat(fr)
conf = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
cm = conf.merge(Fm[Fm.period != FULL], on=['location_id', 'factor', 'period'], suffixes=('_c', ''))
MAP = {'不可评价': '有效年份不足'}
assert len(cm) == len(conf) and (cm.status_c.replace(MAP) == cm.status).all() and np.allclose(cm.resid_orig, cm.residual)
Fm.to_csv(HERE / 'MATRIX_FACTOR.csv', index=False, encoding='utf-8-sig')

CLIM = [('era5_tavg_C_pw_u5', '年平均温度', '°C', 'raw', 0, '没有跨三病种统一的单调方向'), ('era5_hot_days_tmax_gt30_pw_u5', '热日（日最高气温>30℃）', '天/年', 'log1p', -1, '相对SDI—年份参照的位置，不是疾病效应'),
        ('era5_cold_days_tmin_lt0_pw_u5', '冷日（日最低气温<0℃）', '天/年', 'log1p', -1, '相对SDI—年份参照的位置，不是疾病效应'), ('era5_ah_g_m3_pw_u5', '绝对湿度', 'g/m³', 'raw', 0, '没有跨三病种统一的单调方向'),
        ('era5_rh_pct_pw_u5', '相对湿度', '%', 'raw', 0, '没有跨三病种统一的单调方向'), ('era5_wet_days_pr_gt1mm_pw_u5', '湿日（日降水>1毫米）', '天/年', 'raw', -1, '相对SDI—年份参照的位置，不是疾病效应')]
cr = []
for col, name, unit, kind, sign, note in CLIM:
    o = fit_var(col, kind, sign, 'all', False); o['factor'] = col; o['name'] = name; o['unit'] = unit
    o['scale'] = {'raw': '原尺度', 'log1p': 'log(1+x)尺度'}[kind]; o['rule'] = '低暴露更有利' if sign < 0 else '无统一单调方向'; o['note'] = note; cr.append(o)
pd.concat(cr).to_csv(HERE / 'MATRIX_CLIMATE.csv', index=False, encoding='utf-8-sig')
s = ly.groupby(['location_id', 'period']).sdi.mean().rename('sdi_mean').reset_index()
s = pd.concat([s, ly.groupby('location_id').sdi.mean().rename('sdi_mean').reset_index().assign(period=FULL)])
s['tier'] = tier(s.sdi_mean)
s.merge(ly.drop_duplicates('location_id')[['location_id', 'location']], on='location_id').to_csv(HERE / 'MATRIX_SDI.csv', index=False, encoding='utf-8-sig')
print('disease', len(Dm), 'factor', len(Fm), 'climate', sum(len(c) for c in cr), 'sdi', len(s))
print(Fm.groupby('period').status.value_counts().unstack(fill_value=0).to_string())
