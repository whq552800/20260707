"""敏感性分析：参照曲线设定对时期残差与分类的影响（补充材料章节用）。
A 现行：全时期一次拟合；年份固定效应 + SDI 自然样条 + 样条×中心化年份交互
B 共同形状：全时期一次拟合；年份固定效应 + SDI 样条（形状不随时间变化）
C 分时期拟合：三个时期各自拟合；时期内年份固定效应 + SDI 样条
D 时间不变：全时期一次拟合；仅 SDI 样条，无时间项
统一：样条节点、204 国等权、时期残差=年度残差均值、CR1 按国家聚类、t(0.975,G-1)、先区间后 Q1/Q3。
输出：SPEC_PERIOD_RESIDUALS.csv, SPEC_CURVES.csv, SPEC_AGREEMENT.csv, SPEC_PERIOD_SUMMARY.csv, SPEC_CLAIMS.csv"""
import sys, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parent.parent / 'independent_reproduction_20261009'; sys.path.insert(0, str(R))
from paths import PANEL
from repro_common import ns_basis
import statlib
d = pd.read_csv(PANEL); d = d[d.rate_valid & (d.gbd_rate_per_100k > 0)].copy(); d['y'] = np.log(d.gbd_rate_per_100k)
PER = [(1990, 2000, '1990–2000'), (2001, 2010, '2001–2010'), (2011, 2023, '2011–2023')]
d['period'] = d.year.map(lambda y: next(p for a, b, p in PER if a <= y <= b))
KN = np.quantile(d.drop_duplicates(['location_id', 'year']).sdi, [.25, .5, .75])
OUT = [(322, 6, 'LRI发病'), (322, 1, 'LRI死亡'), (332, 6, '脑膜炎发病'), (332, 1, '脑膜炎死亡'), (302, 6, '腹泻病发病'), (302, 1, '腹泻病死亡')]
SPECS = 'ABCD'

def design(sdi, year, spec, years):
    B = ns_basis(np.asarray(sdi, float), KN); one = np.ones((len(B), 1))
    if spec == 'D': return np.column_stack([one, B])
    YF = (np.asarray(year)[:, None] == np.array(years[1:])[None, :]).astype(float)
    if spec == 'A': return np.column_stack([one, YF, B, B * ((np.asarray(year) - 2006.5) / 33)[:, None]])
    return np.column_stack([one, YF, B])

def fit(g, spec):
    g = g.sort_values(['location_id', 'year']).reset_index(drop=True); years = sorted(g.year.unique())
    X = design(g.sdi.values, g.year.values, spec, years); y = g.y.values
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ X.T @ y; e = y - X @ b; n, k = X.shape
    G = g.location_id.nunique(); M = np.zeros((k, k))
    for idx in g.groupby('location_id').indices.values():
        s = X[idx].T @ e[idx]; M += np.outer(s, s)
    V = Xi @ M @ Xi * (G / (G - 1)) * ((n - 1) / (n - k)); t = statlib.t_ppf(.975, G - 1)
    res = []
    for (loc, p), idx in g.groupby(['location_id', 'period']).indices.items():
        a = X[idx].mean(0); se = np.sqrt(a @ V @ a); r = e[idx].mean()
        res.append(dict(location_id=loc, location=g.location.values[idx[0]], period=p, sdi=g.sdi.values[idx].mean(), residual=r, lower=r - t * se, upper=r + t * se))
    return res, b, V, t, years, g

rows, curves = [], []
for c, m, lab in OUT:
    g0 = d[(d.cause_id == c) & (d.measure_id == m)]
    for spec in SPECS:
        parts = [(g0[g0.period == p], p) for _, _, p in PER] if spec == 'C' else [(g0, None)]
        for gg, pfit in parts:
            res, b, V, t, years, g = fit(gg, spec)
            rows += [dict(r, outcome=lab, spec=spec) for r in res]
            for a0, z0, p in PER:
                if pfit and p != pfit: continue
                s = g0[(g0.year >= a0) & (g0.year <= z0)].sdi; grid = np.linspace(s.min(), s.max(), 120)
                A = np.mean([design(grid, np.full(len(grid), yy), spec, years) for yy in range(a0, z0 + 1)], axis=0)
                f = A @ b; se = np.sqrt(np.einsum('ij,jk,ik->i', A, V, A))
                curves.append(pd.DataFrame(dict(outcome=lab, spec=spec, period=p, sdi=grid, ref_rate=np.exp(f), ref_lower=np.exp(f - t * se), ref_upper=np.exp(f + t * se))))
r = pd.DataFrame(rows)
r['direction'] = np.where(r.lower > 0, 'pos', np.where(r.upper < 0, 'neg', 'undetermined'))
def cls(r, keys):
    res = r.residual.where(r.direction != 'undetermined'); by = [r[k] for k in keys]
    q1 = res.groupby(by).transform(lambda x: np.nanquantile(x, .25)); q3 = res.groupby(by).transform(lambda x: np.nanquantile(x, .75))
    return r.assign(category=np.where(r.direction == 'undetermined', '参照不确定', np.where(r.residual < q1, '稳健低尾', np.where(r.residual > q3, '稳健高尾', '明确偏离但居中'))), q1=q1, q3=q3)
r = cls(r, ['outcome', 'spec', 'period'])
# 核对：A 与独立复现完全一致
ref = pd.read_csv(R / 'REPRO_DISEASE_PERIOD.csv'); lab = {(c, m): l for c, m, l in OUT}
ref['outcome'] = [lab[(c, m)] for c, m in zip(ref.cause_id, ref.measure_id)]
chk = r[r.spec == 'A'].merge(ref, on=['outcome', 'location_id', 'period'], suffixes=('', '_ref'))
assert len(chk) == 3672 and (chk.category == chk.category_ref).all() and np.allclose(chk.residual, chk.residual_ref)
r.to_csv('SPEC_PERIOD_RESIDUALS.csv', index=False, encoding='utf-8-sig')
pd.concat(curves).to_csv('SPEC_CURVES.csv', index=False, encoding='utf-8-sig')

W = r.pivot_table(index=['outcome', 'location_id', 'period'], columns='spec', values='category', aggfunc='first')
RW = r.pivot_table(index=['outcome', 'location_id', 'period'], columns='spec', values='residual')
TAIL = ['稳健高尾', '稳健低尾']
agr = []
for o in [l for _, _, l in OUT]:
    w, rw = W.loc[o], RW.loc[o]
    for s in 'BCD':
        tail_any = w.A.isin(TAIL) | w[s].isin(TAIL)
        agr.append(dict(outcome=o, spec=s, n=len(w), agreement=(w.A == w[s]).mean(),
            tail_agreement=(w.A[tail_any] == w[s][tail_any]).mean(),
            opposite_tail=int((((w.A == '稳健高尾') & (w[s] == '稳健低尾')) | ((w.A == '稳健低尾') & (w[s] == '稳健高尾'))).sum()),
            spearman=statlib.spearman(rw.A.values, rw[s].values)[0],
            mean_abs_diff=np.abs(rw.A - rw[s]).mean()))
A = pd.DataFrame(agr); A.to_csv('SPEC_AGREEMENT.csv', index=False, encoding='utf-8-sig')
ps = r.groupby(['spec', 'outcome', 'period']).agg(median_residual=('residual', 'median'), n_pos=('direction', lambda x: (x == 'pos').sum()),
    n_neg=('direction', lambda x: (x == 'neg').sum()), n_uncertain=('direction', lambda x: (x == 'undetermined').sum()),
    n_high=('category', lambda x: (x == '稳健高尾').sum()), n_low=('category', lambda x: (x == '稳健低尾').sum())).reset_index()
ps.to_csv('SPEC_PERIOD_SUMMARY.csv', index=False, encoding='utf-8-sig')

# 主文核心结论在各设定下的数值
claims = []
for s in SPECS:
    q = r[r.spec == s]
    P = q.pivot_table(index=['outcome', 'location_id'], columns='period', values='category', aggfunc='first')
    Lo = q.pivot_table(index=['outcome', 'location_id'], columns='period', values='lower'); Up = q.pivot_table(index=['outcome', 'location_id'], columns='period', values='upper')
    ph = (P == '稳健高尾').all(1) & (Lo > 0).all(1); pl = (P == '稳健低尾').all(1) & (Up < 0).all(1)
    e, l = P['1990–2000'], P['2011–2023']
    dth = [o for o in P.index.get_level_values(0).unique() if o.endswith('死亡')]
    phl = ph[ph].reset_index().groupby('location_id').size()
    claims.append(dict(spec=s,
        persistent_high_records=int(ph.sum()), persistent_high_locations=int(ph[ph].index.get_level_values(1).nunique()),
        persistent_low_records=int(pl.sum()), persistent_low_locations=int(pl[pl].index.get_level_values(1).nunique()),
        locations_ge3_persistent_high=int((phl >= 3).sum()),
        early_high=int((e == '稳健高尾').sum()), early_high_still_high_pct=100 * ((e == '稳健高尾') & (l == '稳健高尾')).sum() / (e == '稳健高尾').sum(),
        early_high_to_low=int(((e == '稳健高尾') & (l == '稳健低尾')).sum()),
        early_low=int((e == '稳健低尾').sum()), early_low_still_low_pct=100 * ((e == '稳健低尾') & (l == '稳健低尾')).sum() / (e == '稳健低尾').sum(),
        early_low_to_high=int(((e == '稳健低尾') & (l == '稳健高尾')).sum()),
        uncertain_records=int((q.direction == 'undetermined').sum()),
        median_residual_early=q[q.period == '1990–2000'].residual.median(), median_residual_late=q[q.period == '2011–2023'].residual.median()))
Cl = pd.DataFrame(claims); Cl.to_csv('SPEC_CLAIMS.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(A.round(3).to_string()); print(Cl.round(1).T.to_string())
print(ps.groupby(['spec', 'period']).median_residual.agg(['min', 'max']).round(2).unstack().to_string())
