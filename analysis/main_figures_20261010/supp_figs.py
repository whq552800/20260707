"""补充图（v5，主图表压缩到5项后移出的内容）。输出到 analysis/supp_figures_20261010/：
S1 Fig 六个结局早期（1990—2000）与晚期（2011—2023）残差散点（原Fig2 A–F）；
S2 Fig 晚期LRI与脑膜炎发病—死亡残差散点（原Fig3 E、F）；
S3 Fig 低及中低SDI层与高SDI层各结局高尾减低尾的不利比例差（原Fig_part2 b、c）。
数据：REPRO_DISEASE_PERIOD.csv；tier_period_tail_factors_20261010/PATTERN_OUTCOME_DOMAIN_BY_BAND.csv。"""
from common import *
import numpy as np, pandas as pd
from statlib import spearman
from matplotlib.lines import Line2D
OUTD = A / 'supp_figures_20261010'; OUTD.mkdir(exist_ok=True)
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
E, L = PER[0], PER[2]
CAT = {'稳健高尾': HIGH, '稳健低尾': LOW, '明确偏离但居中': MID, '参照不确定': '#DDDDDD'}
LAB = {'稳健高尾': 'High tail', '明确偏离但居中': 'Resolved, central', '参照不确定': 'Reference-uncertain', '稳健低尾': 'Low tail'}

# S1
OUT = [(302, 6, 'Diarrhoea incidence'), (322, 6, 'LRI incidence'), (332, 6, 'Meningitis incidence'), (302, 1, 'Diarrhoea mortality'), (322, 1, 'LRI mortality'), (332, 1, 'Meningitis mortality')]
fig = plt.figure(figsize=(150 * MM, 115 * MM))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.45, left=0.09, right=0.98, top=0.93, bottom=0.17)
for k, (c, m, lab) in enumerate(OUT):
    ax = fig.add_subplot(gs[k // 3, k % 3]); x = D[(D.cause_id == c) & (D.measure_id == m)]
    w = x.pivot(index='location_id', columns='period', values='residual'); ce = x[x.period == E].set_index('location_id').category
    for cat in ['参照不确定', '明确偏离但居中', '稳健低尾', '稳健高尾']:
        ids = ce[ce == cat].index; ax.scatter(w.loc[ids, E], w.loc[ids, L], s=4, color=CAT[cat], lw=0)
    lim = np.nanmax(np.abs(w[[E, L]].values)) * 1.05; ax.plot([-lim, lim], [-lim, lim], color='#888888', lw=0.5, ls='--', zorder=0)
    ax.axhline(0, color=GREY, lw=0.4, zorder=0); ax.axvline(0, color=GREY, lw=0.4, zorder=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'{lab}\nρ = {spearman(w[E].values, w[L].values)[0]:.2f}', loc='left', fontsize=6.5); label(ax, 'ABCDEF'[k], x=-0.32, y=1.12)
    if k % 3 == 0: ax.set_ylabel('Residual, 2011–2023')
    if k // 3 == 1: ax.set_xlabel('Residual, 1990–2000')
fig.legend(handles=[Line2D([], [], marker='o', ls='', color=CAT[c], ms=4, label=LAB[c]) for c in ['稳健高尾', '明确偏离但居中', '参照不确定', '稳健低尾']],
           loc='lower center', ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 0.0), title='Position in 1990–2000', title_fontsize=6.5)
save_figure(fig, OUTD / 'S1_Fig_persistence_scatter')

# S2
fig = plt.figure(figsize=(120 * MM, 62 * MM)); gs = fig.add_gridspec(1, 2, wspace=0.4, left=0.14, right=0.98, top=0.88, bottom=0.17)
Ld = D[D.period == L]
for k, c in enumerate([322, 332]):
    ax = fig.add_subplot(gs[0, k]); x = Ld[Ld.cause_id == c].pivot(index='location', columns='measure_id', values=['residual', 'category'])
    xi, yd = x[('residual', 6)].astype(float), x[('residual', 1)].astype(float)
    both = (x[('category', 6)] == '稳健高尾') & (x[('category', 1)] == '稳健高尾'); bothl = (x[('category', 6)] == '稳健低尾') & (x[('category', 1)] == '稳健低尾')
    ax.axhline(0, color=GREY, lw=0.5, zorder=0); ax.axvline(0, color=GREY, lw=0.5, zorder=0)
    ax.scatter(xi[~(both | bothl)], yd[~(both | bothl)], s=5, color=MID, alpha=0.6, lw=0, label='Other locations')
    ax.scatter(xi[both], yd[both], s=7, color=HIGH, lw=0, label='High tail, both'); ax.scatter(xi[bothl], yd[bothl], s=7, color=LOW, lw=0, label='Low tail, both')
    lim = max(abs(xi).max(), abs(yd).max()) * 1.05; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Incidence residual (log)'); ax.set_ylabel('Mortality residual (log)' if k == 0 else '')
    ax.set_title(f'{DZ[c]}, 2011–2023 (ρ={spearman(xi.values, yd.values)[0]:.2f})', loc='left', fontsize=6.5)
    if k == 0: ax.legend(loc='lower right', fontsize=5.8, handletextpad=0.2, borderaxespad=0.1)
    label(ax, 'AB'[k], x=-0.36)
save_figure(fig, OUTD / 'S2_Fig_incidence_mortality_LRI_meningitis')

# S3
o = pd.read_csv(A / 'tier_period_tail_factors_20261010' / 'PATTERN_OUTCOME_DOMAIN_BY_BAND.csv')
DOM6 = [('WaSH', 'Water & sanitation'), ('免疫', 'Immunisation'), ('清洁燃料', 'Clean cooking'), ('卫生系统', 'Health system'), ('营养', 'Nutrition'), ('环境暴露', 'Air pollution')]
OUTS = [('腹泻病发病率', 'Diarrhoea inc.'), ('腹泻病死亡率', 'Diarrhoea death'), ('下呼吸道感染发病率', 'LRI inc.'), ('下呼吸道感染死亡率', 'LRI death'), ('脑膜炎发病率', 'Meningitis inc.'), ('脑膜炎死亡率', 'Meningitis death')]
fig = plt.figure(figsize=(160 * MM, 72 * MM)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.2, left=0.13, right=0.9, top=0.9, bottom=0.27)
for k, (b, bl) in enumerate([('低+中低', 'Low & low-middle SDI'), ('高', 'High SDI')]):
    ax = fig.add_subplot(gs[0, k]); M = np.array([[o[(o.band == b) & (o.outcome_cn == oc) & (o.domain == dm)]['不利差'].iloc[0] for dm, _ in DOM6] for oc, _ in OUTS])
    im = ax.imshow(M, cmap='RdBu_r', vmin=-65, vmax=65, aspect='auto')
    for i in range(6):
        for j in range(6): ax.text(j, i, '0' if round(M[i, j]) == 0 else f'{M[i, j]:+.0f}'.replace('-', '−'), ha='center', va='center', fontsize=6, color='white' if abs(M[i, j]) > 40 else 'black')
    ax.set_xticks(range(6)); ax.set_xticklabels([n for _, n in DOM6], rotation=35, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(6)); ax.set_yticklabels([n for _, n in OUTS] if k == 0 else []); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f'{bl}: high minus low tail (pp)', loc='left', fontsize=6.5); fig.text(0.005 if k == 0 else 0.49, 0.93, 'AB'[k], fontsize=9, fontweight='bold')
cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2])); cb.set_label('Difference in adverse share (pp)'); cb.outline.set_linewidth(0.4)
save_figure(fig, OUTD / 'S3_Fig_outcome_specific_contrasts')
print('ok')
