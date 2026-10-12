"""Figure 2 (Chinese results draft): country position relative to the SDI–year reference.
Current Methods: period residual = mean of annual log residuals; 95% conditional interval
(CR1, clustered by location); interval-first, then Q1/Q3 among resolved locations.
Usage: python plot_reference_figure.py [cause_id measure_id]   (default 322 6 = LRI incidence)
       python plot_reference_figure.py all                      (all six outcomes -> supplementary/)"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker
HERE = Path(__file__).resolve().parent
REPRO = HERE.parent / 'independent_reproduction_20261009'
sys.path[:0] = [str(REPRO), str(HERE / 'qa_tools'), str(HERE.parent)]
from paths import PANEL
from repro_common import ns_basis
import statlib
from audit_panel_alignment import require_matplotlib_panel_alignment
from figure_export import apply_style, save_figure

apply_style({'font.size': 6.5, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False})
PERIODS = [(1990, 2000, '1990–2000'), (2001, 2010, '2001–2010'), (2011, 2023, '2011–2023')]
CAUSE = {302: 'Diarrhoeal disease', 322: 'Lower respiratory infections', 332: 'Meningitis'}
CAUSE_SLUG = {302: 'diarrhoea', 322: 'LRI', 332: 'meningitis'}
MEAS = {1: 'death', 6: 'incidence'}
CATS = [('稳健高尾', 'High tail', '#B2182B'), ('稳健低尾', 'Low tail', '#2166AC'),
        ('明确偏离但居中', 'Resolved, central', '#7F7F7F'), ('参照不确定', 'Reference-uncertain', '#BDBDBD')]
CAT_EN = {c: e for c, e, _ in CATS}; CAT_COL = {c: k for c, _, k in CATS}
MM = 1 / 25.4

panel = pd.read_csv(PANEL)
panel = panel[panel.rate_valid & (panel.gbd_rate_per_100k > 0)].copy()
panel['y'] = np.log(panel.gbd_rate_per_100k)
KNOTS = np.quantile(panel.drop_duplicates(['location_id', 'year']).sdi, [0.25, 0.5, 0.75])
YEARS = sorted(panel.year.unique())
period_res = pd.read_csv(REPRO / 'REPRO_DISEASE_PERIOD.csv')
ANNUAL = pd.read_csv(REPRO / 'REPRO_ANNUAL_FIT.csv')


def design(sdi, year):
    B = ns_basis(np.asarray(sdi, float), KNOTS)
    tc = ((np.asarray(year) - 2006.5) / 33)[:, None]
    YF = (np.asarray(year)[:, None] == np.array(YEARS[1:])[None, :]).astype(float)
    return np.column_stack([np.ones(len(B)), YF, B, B * tc])


def fit(cid, mid):
    g = panel[(panel.cause_id == cid) & (panel.measure_id == mid)].sort_values(['location_id', 'year'])
    X = design(g.sdi.values, g.year.values); y = g.y.values
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    n, k = X.shape; G = g.location_id.nunique(); meat = np.zeros((k, k))
    for idx in g.groupby('location_id').indices.values():
        s = X[idx].T @ e[idx]; meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    ref = g[['location_id', 'year']].merge(ANNUAL[(ANNUAL.cause_id == cid) & (ANNUAL.measure_id == mid)], on=['location_id', 'year'], how='left')
    assert np.abs(X @ b - ref.fitted_log_rate.values).max() < 1e-8   # same fit as the main reference model
    return g, b, V, statlib.t_ppf(0.975, G - 1)


def set_log_ticks(ax, lo, hi, cands, fmt):
    t = [v for v in cands if lo <= v <= hi]
    ax.set_yticks(t); ax.set_yticklabels([fmt(v) for v in t]); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def build(cid, mid, outdir, stem):
    g, b, V, tcrit = fit(cid, mid)
    pr = period_res[(period_res.cause_id == cid) & (period_res.measure_id == mid)].copy()
    obs = g.groupby(['location_id', g.year.map(lambda yy: next(p for a, z, p in PERIODS if a <= yy <= z))]).y.mean()
    pr['obs_geomean_rate'] = [np.exp(obs[(l, p)]) for l, p in zip(pr.location_id, pr.period)]
    curves = []
    for a, z, p in PERIODS:
        s = g[(g.year >= a) & (g.year <= z)].sdi
        grid = np.linspace(s.min(), s.max(), 160); yrs = np.arange(a, z + 1)
        A = np.mean([design(grid, np.full(len(grid), yy)) for yy in yrs], axis=0)
        f = A @ b; se = np.sqrt(np.einsum('ij,jk,ik->i', A, V, A))
        curves.append(pd.DataFrame(dict(period=p, sdi=grid, ref_rate=np.exp(f),
                      ref_lower=np.exp(f - tcrit * se), ref_upper=np.exp(f + tcrit * se))))
    cur = pd.concat(curves)

    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.865, bottom=0.085, hspace=0.16, wspace=0.12)
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
    ylo = min(pr.obs_geomean_rate.min(), cur.ref_lower.min()) / 1.4
    yhi = max(pr.obs_geomean_rate.max(), cur.ref_upper.max()) * 1.4
    rlo = np.exp(min(pr.lower.min(), pr.residual.min())) / 1.25
    rhi = np.exp(max(pr.upper.max(), pr.residual.max())) * 1.25
    labels = iter('abcdef')
    for c, (_, _, p) in enumerate(PERIODS):
        q = pr[pr.period == p]; cc = cur[cur.period == p]
        ax = axs[0, c]
        ax.fill_between(cc.sdi, cc.ref_lower, cc.ref_upper, color='#222222', alpha=0.15, lw=0, zorder=1)
        ax.plot(cc.sdi, cc.ref_rate, color='#222222', lw=0.9, zorder=2)
        for cat, _, col in CATS[::-1]:
            s = q[q.category == cat]
            hollow = cat == '参照不确定'
            ax.scatter(s.sdi, s.obs_geomean_rate, s=7, lw=0.5, zorder=3,
                       facecolor='none' if hollow else col, edgecolor=col if hollow else 'white')
        ax.set_yscale('log'); ax.set_ylim(ylo, yhi); ax.set_xlim(0, 1)
        ax.set_title(p, fontsize=7, pad=3)
        set_log_ticks(ax, ylo, yhi, [10.0**k * m for k in range(-2, 6) for m in ((1, 3) if np.log10(yhi / ylo) < 2.6 else (1,))], lambda v: f'{v:,.0f}' if v >= 1 else f'{v:g}')
        if c == 0: ax.set_ylabel('Rate per 100,000 (period geometric mean)')
        else: ax.tick_params(labelleft=False)
        ax.set_xticks([0, .25, .5, .75, 1]); ax.tick_params(labelbottom=False)
        ax = axs[1, c]
        qs = q.sort_values('sdi')
        for cat, _, col in CATS[::-1]:
            s = qs[qs.category == cat]
            ax.vlines(s.sdi, np.exp(s.lower), np.exp(s.upper), color=col, lw=0.45, alpha=0.8, zorder=2)
            hollow = cat == '参照不确定'
            ax.scatter(s.sdi, np.exp(s.residual), s=5, lw=0.45, zorder=3,
                       facecolor='none' if hollow else col, edgecolor=col if hollow else col)
        ax.axhline(1, color='#222222', lw=0.6, zorder=1)
        for qq in (q.q1.iloc[0], q.q3.iloc[0]):
            ax.axhline(np.exp(qq), color='#222222', lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.set_yscale('log'); ax.set_ylim(rlo, rhi); ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xlabel('Mean SDI in period')
        if c == 0: ax.set_ylabel('Observed / reference rate (R, 95% CI)')
        else: ax.tick_params(labelleft=False)
        set_log_ticks(ax, rlo, rhi, [2.0**k for k in range(-6, 7) if (np.log2(rhi / rlo) < 7 or k % 2 == 0)], lambda v: f'{v:g}' if v >= 0.1 else f'1/{1/v:g}')
        n = q.category.value_counts()
        ax.text(0.02, 0.98, f"High {n.get('稳健高尾',0)} · Low {n.get('稳健低尾',0)} · Uncertain {n.get('参照不确定',0)}",
                transform=ax.transAxes, fontsize=5.5, va='top', ha='left')
    for ax in axs.ravel():
        ax.text(-0.04 if ax in axs[:, 0] else -0.02, 1.02, next(labels), transform=ax.transAxes,
                fontsize=8, fontweight='bold', ha='right', va='bottom')
    h = [plt.Line2D([], [], marker='o', ls='', ms=3.2, mfc='none' if c == '参照不确定' else k, mec=k, mew=0.6, label=e) for c, e, k in CATS]
    h += [plt.Line2D([], [], color='#222222', lw=0.9, label='Reference (period-average fit, 95% CI)'),
          plt.Line2D([], [], color='#222222', lw=0.5, ls=(0, (3, 2)), label='Q1 / Q3 of resolved residuals')]
    fig.legend(handles=h, loc='upper center', ncol=6, bbox_to_anchor=(0.53, 0.955), fontsize=6, handletextpad=0.3, columnspacing=1.0)
    fig.text(0.01, 0.99, f'{CAUSE[cid]}, under-5 {MEAS[mid]}', fontsize=7, fontweight='bold', ha='left', va='top')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=outdir / f'{stem}.panel_alignment.json',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    save_figure(fig, outdir / stem)
    plt.close(fig)
    cols = ['location_id', 'location', 'cause_id', 'measure_id', 'period', 'sdi', 'n_years', 'obs_geomean_rate',
            'residual', 'lower', 'upper', 'rate_ratio', 'direction', 'category', 'q1', 'q3']
    src = pr[cols].assign(category_en=pr.category.map(CAT_EN), R_lower=np.exp(pr.lower), R_upper=np.exp(pr.upper))
    src.to_csv(outdir / f'{stem}_SOURCE_POINTS.csv', index=False, encoding='utf-8-sig')
    cur.to_csv(outdir / f'{stem}_SOURCE_CURVES.csv', index=False, encoding='utf-8-sig')
    print(stem, pr.groupby('period').category.value_counts().unstack().to_string())


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['all']:
        od = HERE / 'supplementary'; od.mkdir(exist_ok=True)
        for c in (322, 332, 302):
            for m in (6, 1): build(c, m, od, f'FigS_reference_{CAUSE_SLUG[c]}_{MEAS[m]}')
    else:
        c, m = (int(a[0]), int(a[1])) if a else (322, 6)
        build(c, m, HERE, f'Figure2_reference_{CAUSE_SLUG[c]}_{MEAS[m]}')
