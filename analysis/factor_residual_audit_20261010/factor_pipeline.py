"""可配置的因素残差流程 + 下游统计（与 repro_factor.py / repro_downstream.py 同逻辑），用于审计各项设定。
选项：
  spec   : 'A'（现行：年份FE+样条×时间）| 'B'（年份FE+共同形状）| 'C'（分时期拟合）| 'D'（无时间项）
  knots  : 'factor'（因素自身样本SDI四分位，现行）| 'all'（全部国家-年份SDI四分位）
  scale  : 'orig'（原单位残差+delta区间，现行）| 'model'（变换尺度残差与区间，与疾病模型一致）
  ceiling: None（现行）| 99（覆盖率类因素：观测与参照时期均值都≥99%时不作方向判定）
  drop   : 需剔除的因素列表"""
import sys, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parent.parent / 'independent_reproduction_20261009'; sys.path.insert(0, str(R))
from paths import PANEL
from repro_common import ns_basis, period_of
import repro_factor_defs as D, statlib
PER = ['1990–2000', '2001–2010', '2011–2023']
COV = ['basic_water_pct', 'basic_sanitation_pct', 'clean_cooking_pct', 'dtp3_pct', 'hib3_pct', 'pcv3_pct']
_ly = None
def ly():
    global _ly
    if _ly is None:
        _ly = pd.read_csv(PANEL).drop_duplicates(['location_id', 'year']); _ly['period'] = _ly.year.map(period_of)
    return _ly
fwd = lambda v, f: (lambda p: np.log(p / (1 - p)))(np.clip(v, .1, 99.9) / 100) if f in D.LOGIT else (np.log(v) if f in D.LOG else np.log1p(v))
inv = lambda z, f: 100 / (1 + np.exp(-z)) if f in D.LOGIT else (np.exp(z) if f in D.LOG else np.expm1(z))
dinv = lambda z, f: (lambda s: 100 * s * (1 - s))(1 / (1 + np.exp(-z))) if f in D.LOGIT else np.exp(z)

def _design(g, spec, kn):
    B = ns_basis(g.sdi.values, kn); one = np.ones((len(g), 1))
    if spec == 'D': return np.column_stack([one, B])
    YF = pd.get_dummies(g.year, drop_first=True).astype(float).values
    if spec == 'A': return np.column_stack([one, YF, B, B * ((g.year.values - 2006.5) / 33)[:, None]])
    return np.column_stack([one, YF, B])

def _fit(g, f, spec, kn, scale, ceiling):
    g = g.sort_values(['location_id', 'year']).reset_index(drop=True)
    X = _design(g, spec, kn); z = fwd(g[f].values, f)
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ X.T @ z; e = z - X @ b; n, k = X.shape; G = g.location_id.nunique(); M = np.zeros((k, k))
    for idx in g.groupby('location_id').indices.values():
        s = X[idx].T @ e[idx]; M += np.outer(s, s)
    V = Xi @ M @ Xi * (G / (G - 1)) * ((n - 1) / (n - k)); t = statlib.t_ppf(.975, G - 1)
    fz = X @ b; fit = inv(fz, f); res_o = g[f].values - fit; sign = -1 if f in D.LOWER_BETTER else 1
    out = []
    for (loc, p), idx in g.groupby(['location_id', 'period']).indices.items():
        if scale == 'orig':
            grad = (dinv(fz[idx], f)[:, None] * X[idx]).mean(0); se = np.sqrt(grad @ V @ grad); r = res_o[idx].mean()
        else:
            a = X[idx].mean(0); se = np.sqrt(a @ V @ a); r = e[idx].mean()
        lo, hi = r - t * se, r + t * se
        flo, fhi = (lo, hi) if sign == 1 else (-hi, -lo)
        obs_m = g[f].values[idx].mean(); ref_m = fit[idx].mean()
        sat = ceiling is not None and f in COV and obs_m >= ceiling and ref_m >= ceiling
        out.append(dict(location_id=loc, location=g.location.values[idx[0]], factor=f, period=p, n_years=len(idx), obs_mean=obs_m, ref_mean=ref_m,
                        resid_orig=res_o[idx].mean(), favourable_residual=sign * r, favourable_lower=flo, favourable_upper=fhi, saturated=sat))
    return out

# 2026-10-11 规则更新：早期（1990–2000）只有2000年一个年份的因素（序列自2000年起：饮水、卫生设施、清洁烹饪、
# 低出生体重、UHC、卫生支出、Hib3），所有地点都以同一年为锚点，地点之间可比，因此可作为早期参照纳入评价；
# 其他情况仍要求同一时期至少2个观测年份。anchor2000=False 可复现旧规则（n_years>=2）。
def _anchor_factors(L):
    e = L[L.period == PER[0]]
    return {f for f in D.ALL if f in e and e[['year', f]].dropna().year.unique().tolist() == [2000]}

# 2026-10-11（审稿意见）：vacc0=True 时，Hib3与PCV3在一国首个有观测的年份之前（国家引入该疫苗之前）记为0%，
# 以检验把引入前年份当作缺失是否影响结论；从未有观测的地点（可能未引入，也可能未向WHO报告）仍为缺失。only 限定只算部分因素。
def _vacc0(L):
    L = L.copy()
    for f in ('hib3_pct', 'pcv3_pct'):
        first = L.dropna(subset=[f]).groupby('location_id').year.min(); start = int(first.min())
        fy = L.location_id.map(first)
        fill = L[f].isna() & fy.notna() & (L.year < fy) & (L.year >= start)
        L.loc[fill, f] = 0.0
    return L

def factor_periods(spec='A', knots='factor', scale='orig', ceiling=None, drop=(), anchor2000=True, vacc0=False, only=None):
    L = _vacc0(ly()) if vacc0 else ly(); kn_all = np.quantile(L.sdi, [.25, .5, .75]); rows = []
    for f in D.ALL:
        if f in drop or (only is not None and f not in only): continue
        g = L[['location_id', 'location', 'year', 'period', 'sdi', f]].dropna()
        if f in D.LOG: g = g[g[f] > 0]
        g = g.reset_index(drop=True)
        kn = np.quantile(g.sdi, [.25, .5, .75]) if knots == 'factor' else kn_all
        if spec == 'C':
            for p in PER:
                gp = g[g.period == p]
                if gp.location_id.nunique() < 20 or gp.year.nunique() < 2: continue   # 该时期数据不足以单独拟合
                rows += _fit(gp, f, 'C', kn, scale, ceiling)
        else:
            rows += _fit(g, f, spec, kn, scale, ceiling)
    r = pd.DataFrame(rows)
    anc = _anchor_factors(L) if anchor2000 else set()
    r['single_year_2000'] = (r.period == PER[0]) & r.factor.isin(anc) & (r.n_years == 1)
    enough = (r.n_years >= 2) | r.single_year_2000
    r['eligible'] = enough & ~r.saturated
    r['status'] = np.where(r.saturated, '已达普及', np.where(~enough, '不可评价',
                  np.where(r.favourable_lower > 0, '明确有利', np.where(r.favourable_upper < 0, '明确不利', '方向不确定'))))
    return r

def downstream(F):
    prof = pd.read_csv(R / 'REPRO_COUNTRY_DISEASE_PROFILES.csv')
    FW = F.pivot_table(index=['location_id', 'factor'], columns='period', values=['status', 'eligible'], aggfunc='first')
    fp = pd.DataFrame(index=FW.index)
    for i, p in enumerate(PER, 1):
        fp[f'p{i}_status'] = FW[('status', p)] if ('status', p) in FW else np.nan
        fp[f'p{i}_elig'] = FW[('eligible', p)].fillna(False).astype(bool) if ('eligible', p) in FW else False
    fp['n_eval'] = fp[[f'p{i}_elig' for i in (1, 2, 3)]].sum(1)
    fp['n_adv'] = sum((fp[f'p{i}_status'] == '明确不利') for i in (1, 2, 3))
    fp['repeated_adverse'] = (fp.n_eval >= 2) & (fp.n_adv >= 2); fp = fp.reset_index()
    rows = []
    for dz in prof.dz.unique():
        g = prof[prof.dz == dz][['location_id', 'group']]
        for f in fp.factor.unique():
            x = fp[(fp.factor == f) & (fp.n_eval >= 2)].merge(g, on='location_id')
            H = x[x.group == '持续高尾']; L = x[x.group == '持续低尾']
            rec = dict(dz=dz, factor=f, high_n=len(H), high_adv=int(H.repeated_adverse.sum()), low_n=len(L), low_adv=int(L.repeated_adverse.sum()), p=np.nan)
            if f not in D.CONTEXT and len(H) >= 5 and len(L) >= 5:
                rec['p'] = statlib.fisher_exact(rec['high_adv'], len(H) - rec['high_adv'], rec['low_adv'], len(L) - rec['low_adv'])
            rows.append(rec)
    C = pd.DataFrame(rows); C['q'] = statlib.bh(C.p.values)
    Dd = pd.read_csv(R / 'REPRO_DISEASE_PERIOD.csv'); DN = {302: '腹泻病', 322: 'LRI', 332: '脑膜炎'}; ON = {1: '死亡', 6: '发病'}
    Dd['dz'] = Dd.cause_id.map(DN) + Dd.measure_id.map(ON)
    Dp = Dd.pivot_table(index=['location_id', 'dz'], columns='period', values='residual').reset_index()
    Fp = F[F.eligible].pivot_table(index=['location_id', 'factor'], columns='period', values='favourable_residual').reset_index()
    for p in PER:
        if p not in Fp: Fp[p] = np.nan
    arows = []
    for dz in Dp.dz.unique():
        dd = Dp[Dp.dz == dz]
        for f in Fp.factor.unique():
            m = Fp[Fp.factor == f].merge(dd, on='location_id', suffixes=('_f', '_d'))
            wins = {'early': (m['1990–2000_f'], m['1990–2000_d']), 'middle': (m['2001–2010_f'], m['2001–2010_d']), 'late': (m['2011–2023_f'], m['2011–2023_d']),
                    'early_late': (m['2011–2023_f'] - m['1990–2000_f'], m['2011–2023_d'] - m['1990–2000_d']),
                    'middle_late': (m['2011–2023_f'] - m['2001–2010_f'], m['2011–2023_d'] - m['2001–2010_d'])}
            for wn, (x, y) in wins.items():
                rho, p, n = statlib.spearman(x, y); ok = n >= 10 and np.nanstd(x) > 0
                arows.append(dict(dz=dz, factor=f, window=wn, n=n, rho=rho if ok else np.nan, p=p if (ok and n >= 20 and f not in D.CONTEXT) else np.nan))
    A = pd.DataFrame(arows); A['q'] = statlib.bh(A.p.values)
    return fp, C, A

if __name__ == '__main__':
    F = factor_periods(anchor2000=False)   # 旧规则复现检查
    ref = pd.read_csv(R / 'REPRO_FACTOR_PERIOD_factor.csv')
    m = ref.merge(F, on=['location_id', 'factor', 'period'], suffixes=('_ref', ''))
    assert len(m) == len(ref) and (m.status_ref == m.status).all() and np.allclose(m.favourable_residual_ref, m.favourable_residual)
    fp, C, A = downstream(F)
    Cr = pd.read_csv(R / 'REPRO_GROUP_CONTRASTS.csv'); Ar = pd.read_csv(R / 'REPRO_ASSOCIATIONS.csv')
    mc = Cr.merge(C, on=['dz', 'factor'], suffixes=('_ref', ''))
    assert (mc.high_adv_ref == mc.high_adv).all() and (mc.low_n_ref == mc.low_n).all() and np.allclose(mc.p_ref.fillna(-1), mc.p.fillna(-1))
    ma = Ar.merge(A, on=['dz', 'factor', 'window'], suffixes=('_ref', ''))
    assert np.allclose(ma.rho_ref.fillna(9), ma.rho.fillna(9)) and np.allclose(ma.q_ref.fillna(9), ma.q.fillna(9))
    print('baseline reproduces current factor statuses, contrasts and associations exactly')
