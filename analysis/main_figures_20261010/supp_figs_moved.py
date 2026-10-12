"""补充图（2026-10-12 从正文移出）：S4 Fig 三个SDI层按记录合并的各领域高尾/低尾“不利偏离”比例（原 Fig 3A）；
S5 Fig 高SDI地点亚区内 LRI 发病残差与环境 PM2.5 残差（原 Fig 3C）；S6 Fig 案例国家轨迹（原 Fig 4C、D）。输出到 analysis/supp_figures_20261010/。"""
from common import *
import numpy as np, pandas as pd
from statlib import spearman
OUTD = A / 'supp_figures_20261010'; OUTD.mkdir(exist_ok=True)
P = A / 'tier_period_tail_factors_20261010'; S = A / 'substantive_findings_20261010'
d = pd.read_csv(P / 'PATTERN_DOMAIN_BY_BAND.csv')
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv'); T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')
E = pd.read_csv(S / 'R4_ENV_PERIOD.csv'); reg = pd.read_csv(S / 'A_PROFILES_WITH_REGION.csv').drop_duplicates('location_id').set_index('location_id').region
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
M2, L2 = '2001–2010', '2011–2023'
# S4
fig = plt.figure(figsize=(180 * MM, 72 * MM))
gs = fig.add_gridspec(1, 1, left=0.17, right=0.98, top=0.86, bottom=0.27)
g1 = gs[0].subgridspec(1, 3, wspace=0.12)
BANDS = [('低+中低', 'Low & low-middle SDI'), ('中等+中高', 'Middle & high-middle SDI'), ('高', 'High SDI')]
DOMS = [('WaSH', 'Water and sanitation'), ('免疫', 'Immunisation'), ('清洁燃料', 'Clean cooking'), ('卫生系统', 'Health system'), ('营养', 'Nutrition'), ('环境暴露', 'Air pollution'), ('社会经济', 'Socioeconomic context')]
y = np.arange(len(DOMS))[::-1]
for k, (b, bl) in enumerate(BANDS):
    ax = fig.add_subplot(g1[0, k]); x = d[d.band == b].set_index('domain')
    hi = np.array([x.loc[c, '高尾不利'] for c, _ in DOMS]); lo = np.array([x.loc[c, '低尾不利'] for c, _ in DOMS])
    for yi, h, l in zip(y, hi, lo): ax.plot([l, h], [yi, yi], color=GREY, lw=1.2, zorder=1)
    ax.scatter(hi, y, color=HIGH, s=14, zorder=2, lw=0, label='High tail'); ax.scatter(lo, y, color=LOW, s=14, zorder=2, lw=0, label='Low tail')
    for yi, h, l in zip(y, hi, lo):
        dv = round(h - l); ax.text(84, yi, '0' if dv == 0 else f'{dv:+d}'.replace('-', '−'), va='center', ha='left', fontsize=6, color=HIGH if dv > 10 else ('black' if dv > -10 else LOW))
    ax.text(88, len(DOMS) - 0.35, 'Δ', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax.set_xlim(0, 94); ax.set_xticks([0, 20, 40, 60, 80]); ax.set_ylim(-0.6, len(DOMS) - 0.4); ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in DOMS] if k == 0 else []); ax.set_xlabel('Records with an adverse departure (%)'); ax.set_title(bl, loc='left')
    ax.grid(axis='x', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True)
    if k == 0: hA, lA = ax.get_legend_handles_labels()
fig.legend(hA, lA, loc='center', bbox_to_anchor=(0.575, 0.05), ncol=2, fontsize=6.3, handletextpad=0.2, columnspacing=1.5)
save_figure(fig, OUTD / 'S4_Fig_pooled_records_by_domain')
# S5
fig = plt.figure(figsize=(95 * MM, 80 * MM)); gs = fig.add_gridspec(1, 1, left=0.17, right=0.97, top=0.9, bottom=0.16)
g2 = [[gs[0, 0]]]
ax = fig.add_subplot(gs[0, 0]); L2 = '2011–2023'
r = D[(D.cause_id == 322) & (D.measure_id == 6) & (D.period == L2)].merge(T[T.period == L2][['location_id', 'tier']], on='location_id')
r = r[r.tier == '高SDI'].copy(); r['region'] = r.location_id.map(reg); r['pm'] = r.location_id.map(E[(E.period == L2) & (E.env == '环境PM2.5')].set_index('location_id').dev)
r = r.dropna(subset=['pm']); r = r[r.groupby('region').location_id.transform('size') >= 2]
r['pm_w'] = r.pm - r.groupby('region').pm.transform('mean'); r['res_w'] = r.residual - r.groupby('region').residual.transform('mean'); ht = r.category == '稳健高尾'
ax.scatter(r.pm_w[~ht], r.res_w[~ht], s=7, color=MID, lw=0, label='Other high-SDI locations'); ax.scatter(r.pm_w[ht], r.res_w[ht], s=9, color=HIGH, lw=0, label='High tail, LRI incidence')
b1, b0 = np.polyfit(r.pm_w, r.res_w, 1); xs = np.linspace(r.pm_w.min(), r.pm_w.max(), 10); ax.plot(xs, b0 + b1 * xs, color='black', lw=0.7)
rho, pv = spearman(r.pm_w.values, r.res_w.values)[:2]
OFF2 = {'Qatar': (-14, 10), 'Kuwait': (8, 4), 'Saudi Arabia': (10, -8), 'Poland': (10, 6), 'Singapore': (-22, 10)}
for nm, (dx, dy) in OFF2.items():
    q = r[r.location == nm].iloc[0]; ax.annotate(nm, (q.pm_w, q.res_w), xytext=(dx, dy), textcoords='offset points', fontsize=5.8, va='center', ha='left' if dx >= 0 else 'right', arrowprops=dict(arrowstyle='-', lw=0.4, color='black', shrinkA=0, shrinkB=1.5))
ax.axhline(0, color=GREY, lw=0.5, zorder=0); ax.axvline(0, color=GREY, lw=0.5, zorder=0)
ax.set_xlabel('Ambient PM2.5 residual, within subregion (log scale)'); ax.set_ylabel('LRI incidence residual, within subregion')
ax.set_title(f'High-SDI locations, 2011–2023 (n={len(r)}, ρ={rho:.2f})', loc='left', fontsize=6.5); ax.legend(loc='lower right', fontsize=5.8, handletextpad=0.2)
save_figure(fig, OUTD / 'S5_Fig_PM25_LRI_incidence_high_SDI')
# S6
fig = plt.figure(figsize=(180 * MM, 76 * MM)); gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.09, right=0.97, top=0.9, bottom=0.25)
GREEN = '#1B7837'
def traj(ax, factor, cause, cases, off, xlab, ylab):
    fs = F[F.factor == factor].pivot(index='location', columns='period', values='favourable_residual')
    dr = D[(D.cause_id == cause) & (D.measure_id == 1)].pivot(index='location', columns='period', values='residual')
    both = fs[[M2, L2]].dropna().index.intersection(dr.index)
    ax.scatter(fs.loc[both, L2], dr.loc[both, L2], s=4, color=GREY, lw=0, zorder=0, label='All locations, 2011–2023')
    for c in cases:
        x0, x1, y0, y1 = fs.loc[c, M2], fs.loc[c, L2], dr.loc[c, M2], dr.loc[c, L2]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='black', lw=0.8, mutation_scale=7))
        ax.scatter(x0, y0, s=8, facecolor='white', edgecolor='black', lw=0.6, zorder=3)
        dx, dy = off.get(c, (3, 0)); ax.annotate(c, (x1, y1), xytext=(dx, dy), textcoords='offset points', fontsize=6, va='center', ha='left' if dx >= 0 else 'right')
    ax.axhline(0, color=GREY, lw=0.5, zorder=0); ax.axvline(0, color=GREY, lw=0.5, zorder=0)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    return [(c, round(fs.loc[c, M2], 1), round(fs.loc[c, L2], 1), round(np.exp(dr.loc[c, M2]), 2), round(np.exp(dr.loc[c, L2]), 2)) for c in cases]
CASES = ['Indonesia', 'China', 'Viet Nam', 'South Africa', 'Nigeria']
ax = fig.add_subplot(gs[0, 0])
tc = traj(ax, 'basic_sanitation_pct', 302, CASES, {'Indonesia': (-4, 7), 'Nigeria': (4, 8), 'South Africa': (3, 0), 'China': (3, -3), 'Viet Nam': (3, 0)},
          'Basic sanitation, observed minus reference (pp)', 'Diarrhoeal mortality residual (log)')
ax.set_title('Sanitation and diarrhoeal mortality', loc='left', fontsize=6.5)
fig.text(0.005, 0.93, 'A', fontsize=9, fontweight='bold')
ax = fig.add_subplot(gs[0, 1])
td = traj(ax, 'dtp3_pct', 322, CASES, {'Indonesia': (-3, -7), 'Nigeria': (4, 6), 'South Africa': (3, 0), 'China': (3, -3), 'Viet Nam': (-4, -6)},
          'DTP3 coverage, observed minus reference (pp)', 'LRI mortality residual (log)')
ax.set_title('DTP3 and LRI mortality', loc='left', fontsize=6.5)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], marker='o', ls='', mfc='white', mec='black', ms=3), Line2D([], [], marker='>', ls='-', color='black', ms=3, lw=0.8),
            Line2D([], [], marker='o', ls='', color=GREY, ms=2.5)], ['2001–2010', '2011–2023 (arrow head)', 'All locations, 2011–2023'],
           loc='lower center', bbox_to_anchor=(0.53, 0.0), ncol=3, fontsize=6, handletextpad=0.3, columnspacing=1.2)
fig.text(0.515, 0.93, 'B', fontsize=9, fontweight='bold')
save_figure(fig, OUTD / 'S6_Fig_case_trajectories')
print('ok', rho)
