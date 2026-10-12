"""S1 Text 图F：正文主要结论在疾病端各敏感性设定下的数值（点＝取值；GBD不确定性行为200次抽样的中位数与2.5–97.5百分位）。
输入：S1_DISEASE_VARIANTS.csv、S1_UI_DRAWS.csv（同目录）。输出：S1Text_FigF_robustness.{jpg,pdf}。"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / 'main_figures_20261010'))
from common import *
import numpy as np, pandas as pd
V = pd.read_csv(HERE / 'S1_DISEASE_VARIANTS.csv').set_index('variant'); U = pd.read_csv(HERE / 'S1_UI_DRAWS.csv')
ORDER = [('main', 'Main analysis'), ('spec_B', 'Fixed curve shape'), ('spec_C', 'Period-specific fits'), ('spec_D', 'No time terms'), ('df3', 'Spline, 3 df'), ('df5', 'Spline, 5 df'),
         ('pop_weighted', 'Population-weighted fit'), ('excl_small', 'Small locations excluded'), ('loo', 'Leave-location-out reference'), ('tertile_tails', 'Tertile tails'),
         ('quintile_tails', 'Quintile tails'), ('ui', 'GBD uncertainty, 200 draws')]
y = np.arange(len(ORDER))[::-1]
def val(k, col):
    if k == 'ui': q = U[col].quantile([0.025, 0.5, 0.975]).values; return q[1], q[0], q[2]
    v = V.loc[k, col]; return v, np.nan, np.nan
PANELS = [('Rank stability, early vs late', 'Spearman ρ (range over six outcomes)', [('rank_rho_min', 'rank_rho_max', '#333333', None)], (0.4, 1.0)),
          ('Early tails still in the same tail', '% of early-tail records', [('early_high_retain_pct', None, HIGH, 'High tail'), ('early_low_retain_pct', None, LOW, 'Low tail')], (40, 80)),
          ('Correlation between diseases', 'Mean Spearman ρ over pairs and periods', [('cross_mort_mean', None, '#333333', 'Mortality'), ('cross_inc_mean', None, '#999999', 'Incidence')], (0, 0.7)),
          ('Incidence vs mortality, same disease', 'Spearman ρ, 2011–2023', [('incdeath_rho_302', None, DZC[302], 'Diarrhoeal'), ('incdeath_rho_322', None, DZC[322], 'LRI'), ('incdeath_rho_332', None, DZC[332], 'Meningitis')], (0, 1.0)),
          ('Water and sanitation, high minus low tail', 'Percentage points', [('diff_WaSH_low', None, '#333333', 'Low & low-middle SDI'), ('diff_WaSH_high', None, '#999999', 'High SDI')], (0, 70)),
          ('Catch-up in drinking water', '% of locations with falling mortality residual', [('catch_water_nar_pct', None, '#1B7837', 'Gap narrowed'), ('catch_water_not_pct', None, 'white', 'Gap not narrowed')], (20, 80))]
fig = plt.figure(figsize=(180 * MM, 132 * MM))
gs = fig.add_gridspec(2, 3, hspace=0.7, wspace=0.12, left=0.19, right=0.985, top=0.92, bottom=0.08)
for k, (title, xl, series, lim) in enumerate(PANELS):
    ax = fig.add_subplot(gs[k // 3, k % 3])
    ax.axhspan(y[0] - 0.5, y[0] + 0.5, color='#F2F2F2', zorder=0)
    for j, (col, col2, c, lab) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * 0.22
        for yi, (key, _) in zip(y, ORDER):
            if col2:   # 区间（范围）
                if key == 'ui':
                    a = U[col].median(); b = U[col2].median()
                else:
                    a, b = V.loc[key, col], V.loc[key, col2]
                ax.plot([a, b], [yi, yi], color=c, lw=1.6, solid_capstyle='round')
                continue
            v, lo, hi = val(key, col)
            if not np.isnan(lo): ax.plot([lo, hi], [yi + off, yi + off], color='black' if c == 'white' else c, lw=0.7)
            ax.scatter(v, yi + off, s=11, color=c, edgecolor='black' if c == 'white' else 'none', lw=0.6, zorder=3, label=lab if yi == y[0] else None)
    v0 = series[0][0]
    ax.set_xlim(*lim); ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_yticks(y)
    ax.set_yticklabels([o[1] for o in ORDER] if k % 3 == 0 else [], fontsize=6)
    ax.set_xlabel(xl, fontsize=6.3); ax.set_title(title, loc='left', fontsize=6.5, pad=11); ax.grid(axis='x', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True)
    if series[0][3]: ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=len(series), fontsize=5.6, handletextpad=0.1, columnspacing=0.6, borderaxespad=0)
    pos = ax.get_position(); fig.text(0.005 if k % 3 == 0 else pos.x0 - 0.02, pos.y1 + 0.03, 'ABCDEF'[k], fontsize=9, fontweight='bold')
save_figure(fig, HERE / 'S1Text_FigF_robustness'); print('ok')
