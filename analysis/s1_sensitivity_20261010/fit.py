"""参照模型A（与 repro_disease.py 相同）的可配置重拟合：样条自由度、加权、地点子集、扰动后的对数率。"""
import numpy as np, pandas as pd
from engine import A, P3
import sys; sys.path.insert(0, str(A / 'independent_reproduction_20261009'))
from statlib import t_ppf
from repro_common import ns_basis
from paths import PANEL as _src   # data/under5_disease_factor_annual_panel_1990_2023.csv
PANEL = pd.read_csv(_src, usecols=['location_id', 'location', 'cause_id', 'measure_id', 'year', 'sdi', 'under5_population', 'gbd_rate_per_100k',
                                          'gbd_rate_lower_per_100k', 'gbd_rate_upper_per_100k', 'rate_valid', 'location_holdout_oof_fold'])
PANEL = PANEL[PANEL.rate_valid & (PANEL.gbd_rate_per_100k > 0)].copy(); PANEL['y'] = np.log(PANEL.gbd_rate_per_100k)
PANEL['period'] = np.where(PANEL.year <= 2000, P3[0], np.where(PANEL.year <= 2010, P3[1], P3[2]))
PANEL = PANEL.sort_values(['cause_id', 'measure_id', 'location_id', 'year']).reset_index(drop=True)
SDI_ALL = PANEL.drop_duplicates(['location_id', 'year']).sdi
TCRIT = {}
def knots_for(df):
    return np.quantile(SDI_ALL, np.linspace(0, 1, df + 1)[1:-1])   # df=4 → 3个内部节点（四分位）
def fit(g, df=4, w=None, y=None):
    """g：单一结局的年度数据（已排序）；返回地点—时期残差、区间。"""
    y = g.y.values if y is None else y
    B = ns_basis(g.sdi.values, knots_for(df)); tc = ((g.year.values - 2006.5) / 33)[:, None]
    yrs = np.sort(g.year.unique()); YF = (g.year.values[:, None] == yrs[None, 1:]).astype(float)
    X = np.column_stack([np.ones(len(g)), YF, B, B * tc])
    sw = np.ones(len(g)) if w is None else np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    XtXi = np.linalg.pinv(Xw.T @ Xw); b = XtXi @ Xw.T @ yw; e = y - X @ b; n, k = X.shape
    loc = g.location_id.values; ul, inv = np.unique(loc, return_inverse=True); G = len(ul)
    S = np.zeros((G, k)); np.add.at(S, inv, Xw * (e * sw)[:, None]); V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    if G not in TCRIT: TCRIT[G] = t_ppf(0.975, G - 1)
    t = TCRIT[G]
    key = pd.MultiIndex.from_arrays([loc, g.period.values]); dfp = pd.DataFrame(X, index=key).groupby(level=[0, 1]).mean()
    r = pd.Series(e, index=key).groupby(level=[0, 1]).mean()
    se = np.sqrt(np.einsum('ij,jk,ik->i', dfp.values, V, dfp.values))
    out = pd.DataFrame({'residual': r.values, 'lower': r.values - t * se, 'upper': r.values + t * se}, index=dfp.index).reset_index()
    out.columns = ['location_id', 'period', 'residual', 'lower', 'upper']
    return out
def fit_all(sub=None, df=4, weight=False, perturb=None):
    res = []
    for (c, m), g in PANEL.groupby(['cause_id', 'measure_id']):
        if sub is not None: g = g[g.location_id.isin(sub)]
        y = None if perturb is None else g.y.values + perturb(g)
        o = fit(g, df=df, w=(g.under5_population.values / g.under5_population.mean()) if weight else None, y=y)
        res.append(o.assign(cause_id=c, measure_id=m))
    r = pd.concat(res, ignore_index=True)
    r['direction'] = np.where(r.lower > 0, 'pos', np.where(r.upper < 0, 'neg', 'undetermined'))
    return r

def oof_residuals():
    """Leave-location-out residuals: five fixed folds of locations (column location_holdout_oof_fold); for each fold the
    reference is refitted on the other folds, with spline knots at the SDI quartiles of the training rows, and the held-out
    locations are predicted. Returns a Series aligned with PANEL."""
    out = []
    for (c, m), g in PANEL.groupby(['cause_id', 'measure_id']):
        yrs = np.sort(g.year.unique())
        def X_(h, kn):
            B = ns_basis(h.sdi.values, kn); tc = ((h.year.values - 2006.5) / 33)[:, None]
            return np.column_stack([np.ones(len(h)), (h.year.values[:, None] == yrs[None, 1:]).astype(float), B, B * tc])
        for k in sorted(g.location_holdout_oof_fold.unique()):
            tr, te = g[g.location_holdout_oof_fold != k], g[g.location_holdout_oof_fold == k]
            kn = np.quantile(tr.sdi, [.25, .5, .75])
            b = np.linalg.lstsq(X_(tr, kn), tr.y.values, rcond=None)[0]
            out.append(pd.Series(te.y.values - X_(te, kn) @ b, index=te.index))
    return pd.concat(out).sort_index()

def oof_period():
    """Leave-location-out period residuals with intervals re-estimated in each fold (2026-10-11).
    For each fold, the reference is refitted on the other folds (knots at the SDI quartiles of the training rows), the
    cluster-robust (CR1) covariance is computed from the training locations, and the held-out location-periods get the
    residual mean and the 95% interval of the fitted reference at their mean design row (t with G_train - 1 df)."""
    res = []
    for (c, m), g in PANEL.groupby(['cause_id', 'measure_id']):
        yrs = np.sort(g.year.unique())
        def X_(h, kn):
            B = ns_basis(h.sdi.values, kn); tc = ((h.year.values - 2006.5) / 33)[:, None]
            return np.column_stack([np.ones(len(h)), (h.year.values[:, None] == yrs[None, 1:]).astype(float), B, B * tc])
        for k in sorted(g.location_holdout_oof_fold.unique()):
            tr, te = g[g.location_holdout_oof_fold != k], g[g.location_holdout_oof_fold == k]
            kn = np.quantile(tr.sdi, [.25, .5, .75]); Xt = X_(tr, kn); y = tr.y.values
            XtXi = np.linalg.pinv(Xt.T @ Xt); b = XtXi @ Xt.T @ y; e = y - Xt @ b; n, kk = Xt.shape
            ul, inv = np.unique(tr.location_id.values, return_inverse=True); G = len(ul)
            S = np.zeros((G, kk)); np.add.at(S, inv, Xt * e[:, None]); V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - kk))
            if G not in TCRIT: TCRIT[G] = t_ppf(0.975, G - 1)
            Xe = X_(te, kn); key = pd.MultiIndex.from_arrays([te.location_id.values, te.period.values])
            dfp = pd.DataFrame(Xe, index=key).groupby(level=[0, 1]).mean()
            r = pd.Series(te.y.values - Xe @ b, index=key).groupby(level=[0, 1]).mean()
            se = np.sqrt(np.einsum('ij,jk,ik->i', dfp.values, V, dfp.values))
            o = pd.DataFrame({'residual': r.values, 'lower': r.values - TCRIT[G] * se, 'upper': r.values + TCRIT[G] * se}, index=dfp.index).reset_index()
            o.columns = ['location_id', 'period', 'residual', 'lower', 'upper']; res.append(o.assign(cause_id=c, measure_id=m))
    r = pd.concat(res, ignore_index=True)
    r['direction'] = np.where(r.lower > 0, 'pos', np.where(r.upper < 0, 'neg', 'undetermined'))
    return r
