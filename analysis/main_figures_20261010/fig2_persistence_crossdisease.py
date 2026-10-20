"""Fig 2（v5合并图）：相对位置持续存在；死亡跨病种共有，发病与死亡因病种分离。
A 六个结局早期（1990—2000）与晚期（2011—2023）残差的排名相关；B 早期稳健高尾/低尾记录在晚期的位置；
C 晚期六个结局残差之间的Spearman相关；D 三个时期病种两两之间死亡与发病残差的相关；
E 同一疾病发病—死亡残差相关的时期变化及晚期方向相反比例；F 晚期腹泻病发病与死亡残差（澳大利亚、格陵兰为例）。
早晚期残差散点（原Fig2 A–F）及LRI、脑膜炎发病—死亡散点移入S1 Fig（FigS_persistence_scatter.py、fig_part1.py 输出）。
数据：REPRO_DISEASE_PERIOD.csv。输出：Fig2_persistence_and_cross_disease.{jpg,pdf}，Fig2_SOURCE.csv。"""
from common import *
import numpy as np, pandas as pd
from statlib import spearman
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
D['key'] = D.cause_id.map({302: 'Diar', 322: 'LRI', 332: 'Men'}) + D.measure_id.map({1: ' death', 6: ' inc.'})
E0, L0 = PER[0], PER[2]
src = []
fig = plt.figure(figsize=(180 * MM, 128 * MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.12], hspace=0.9, wspace=0.75, left=0.135, right=0.95, top=0.94, bottom=0.08)

# A 早晚期排名相关
ax = fig.add_subplot(gs[0, 0])
OUT = [(302, 1, 'Diarrhoea, death'), (322, 1, 'LRI, death'), (332, 1, 'Meningitis, death'), (302, 6, 'Diarrhoea, incidence'), (322, 6, 'LRI, incidence'), (332, 6, 'Meningitis, incidence')]
yy = np.arange(len(OUT))[::-1]
for yi, (c, m, lab) in zip(yy, OUT):
    w = D[(D.cause_id == c) & (D.measure_id == m)].pivot(index='location_id', columns='period', values='residual')
    r = spearman(w[E0].values, w[L0].values)[0]; src.append(dict(panel='A', outcome=lab, rho=r))
    ax.plot([0.5, r], [yi, yi], color=GREY, lw=0.8, zorder=1)
    ax.scatter(r, yi, s=20, color=DZC[c], marker='o' if m == 1 else 's', zorder=2, lw=0)
    ax.text(r + 0.02, yi, f'{r:.2f}', va='center', fontsize=5.8)
ax.set_yticks(yy); ax.set_yticklabels([o[2] for o in OUT], fontsize=6); ax.set_xlim(0.5, 1.0); ax.set_ylim(-0.6, len(OUT) - 0.4)
ax.set_xlabel('Spearman ρ, 1990–2000 vs 2011–2023'); ax.set_title('Rank stability of residuals', loc='left')
fig.text(0.008, 0.945, 'A', fontsize=9, fontweight='bold', va='bottom')

# B 早期尾部的晚期去向
ax = fig.add_subplot(gs[0, 1])
CAT = {'稳健高尾': HIGH, '明确偏离但居中': MID, '参照不确定': '#DDDDDD', '稳健低尾': LOW}
LAB = {'稳健高尾': 'High tail', '明确偏离但居中': 'Resolved, central', '参照不确定': 'Reference-uncertain', '稳健低尾': 'Low tail'}
j = D[D.period == E0].set_index(['location_id', 'cause_id', 'measure_id']).category.to_frame('e').join(D[D.period == L0].set_index(['location_id', 'cause_id', 'measure_id']).category.rename('l'))
for xi, start in enumerate(['稳健高尾', '稳健低尾']):
    s = j[j.e == start].l.value_counts().reindex(list(CAT)).fillna(0); n = int(s.sum()); bottom = 0
    src.append(dict(panel='B', start=start, n=n, **{LAB[k]: int(v) for k, v in s.items()}))
    for cat in CAT:
        v = 100 * s[cat] / n
        ax.bar(xi, v, bottom=bottom, color=CAT[cat], width=0.62, edgecolor='white', lw=0.4, label=LAB[cat] if xi == 0 else None)
        if v >= 7: ax.text(xi, bottom + v / 2, f'{v:.0f}', ha='center', va='center', fontsize=6, color='white' if cat in ('稳健高尾', '稳健低尾') else 'black')
        bottom += v
    nb = n
ax.set_xticks([0, 1]); ax.set_xticklabels(['High tail', 'Low tail']); ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 100)
ax.set_xlabel(f'Position in 1990–2000 (n={nb} records each)'); ax.set_ylabel('Position in 2011–2023 (%)')
ax.set_title('Where early tails ended up', loc='left')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=5.6, handlelength=0.9, handletextpad=0.3, labelspacing=0.3, columnspacing=0.8, borderaxespad=0)
label(ax, 'B', x=-0.42, y=1.04)

# C 晚期六结局相关热图
ax = fig.add_subplot(gs[0, 2])
ORDER = ['Diar death', 'LRI death', 'Men death', 'Diar inc.', 'LRI inc.', 'Men inc.']
late = D[D.period == L0].pivot(index='location_id', columns='key', values='residual')[ORDER]
C = np.array([[spearman(late[a].values, late[b].values)[0] for b in ORDER] for a in ORDER]); np.fill_diagonal(C, np.nan)
im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(6):
    for k in range(6):
        if i != k: ax.text(k, i, f'{C[i, k]:.2f}', ha='center', va='center', fontsize=5, color='white' if abs(C[i, k]) > 0.6 else 'black')
ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(ORDER, rotation=45, ha='right', rotation_mode='anchor', fontsize=5.8); ax.set_yticklabels(ORDER, fontsize=5.8)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.tick_params(length=0); cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_ticks([-1, -0.5, 0, 0.5, 1]); cb.outline.set_linewidth(0.4)
ax.set_title('Correlation between outcomes', loc='left')
label(ax, 'C', x=-0.62, y=1.04)

# D 病种两两相关：死亡 vs 发病
ax = fig.add_subplot(gs[1, 0]); pairs = [('Diar', 'LRI'), ('Diar', 'Men'), ('LRI', 'Men')]
for m, col, mk in [(' death', '#333333', 'o'), (' inc.', '#999999', 's')]:
    for k, (a, b) in enumerate(pairs):
        ys = []
        for p in PER:
            w = D[D.period == p].pivot(index='location_id', columns='key', values='residual')
            ys.append(spearman(w[a + m].values, w[b + m].values)[0]); src.append(dict(panel='D', period=p, pair=f'{a}-{b}', measure=m.strip(), rho=ys[-1]))
        ax.plot(range(3), ys, marker=mk, ms=3, lw=0.8, color=col, label=('Mortality' if m == ' death' else 'Incidence') if k == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels(['1990–\n2000', '2001–\n2010', '2011–\n2023']); ax.set_ylim(0, 0.7); ax.set_ylabel('Spearman ρ between diseases')
ax.legend(loc='center right', fontsize=6); ax.set_title('Pairs of diseases', loc='left')
fig.text(0.008, 0.415, 'D', fontsize=9, fontweight='bold', va='bottom')

# E 发病—死亡相关
ax = fig.add_subplot(gs[1, 1]); L = D[D.period == L0]
for c in [302, 322, 332]:
    ys = []
    for p in PER:
        w = D[(D.period == p) & (D.cause_id == c)].pivot(index='location_id', columns='measure_id', values='residual'); ys.append(spearman(w[6].values, w[1].values)[0])
    ax.plot(range(3), ys, marker='o', ms=3, lw=1, color=DZC[c])
    s = L[L.cause_id == c].assign(s=lambda x: x.direction.map({'pos': 1, 'neg': -1})).pivot(index='location_id', columns='measure_id', values='s').dropna()
    opp = (s[1] != s[6]).mean() * 100; src.append(dict(panel='E', disease=DZ[c], rho_by_period=ys, opposite_pct_late=opp, n=len(s)))
    ax.text(2.15, ys[-1], f'{DZ[c].replace(" disease", "")}\n{opp:.0f}% opposite', color=DZC[c], fontsize=5.6, va='center', linespacing=0.95)
ax.set_xticks(range(3)); ax.set_xticklabels(['1990–\n2000', '2001–\n2010', '2011–\n2023']); ax.set_xlim(-0.2, 3.4); ax.set_ylim(0, 1)
ax.set_ylabel('Incidence–death Spearman ρ'); ax.set_title('Incidence vs death, same disease', loc='left')
label(ax, 'E', x=-0.42, y=1.04)

# F 腹泻病发病 vs 死亡
ax = fig.add_subplot(gs[1, 2]); c = 302
x = L[L.cause_id == c].pivot(index='location', columns='measure_id', values=['residual', 'category'])
xi, yd = x[('residual', 6)].astype(float), x[('residual', 1)].astype(float)
both = (x[('category', 6)] == '稳健高尾') & (x[('category', 1)] == '稳健高尾'); bothl = (x[('category', 6)] == '稳健低尾') & (x[('category', 1)] == '稳健低尾')
ax.axhline(0, color=GREY, lw=0.5, zorder=0); ax.axvline(0, color=GREY, lw=0.5, zorder=0)
ax.scatter(xi[~(both | bothl)], yd[~(both | bothl)], s=5, color=MID, alpha=0.6, lw=0, label='Other locations')
ax.scatter(xi[both], yd[both], s=7, color=HIGH, lw=0, label='High tail, both'); ax.scatter(xi[bothl], yd[bothl], s=7, color=LOW, lw=0, label='Low tail, both')
for nm, off in [('Australia', (4, -6)), ('Greenland', (4, 4))]:
    ax.scatter(xi[nm], yd[nm], s=14, facecolor='none', edgecolor='black', lw=0.7); ax.annotate(nm, (xi[nm], yd[nm]), xytext=off, textcoords='offset points', fontsize=6)
lim = max(abs(xi).max(), abs(yd).max()) * 1.05; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Incidence residual (log)'); ax.set_ylabel('Mortality residual (log)')
ax.set_title(f'Diarrhoeal disease, 2011–2023 (ρ={spearman(xi.values, yd.values)[0]:.2f})', loc='left', fontsize=6.5)
ax.legend(loc='lower right', fontsize=5.6, handletextpad=0.2, borderaxespad=0.1)
label(ax, 'F', x=-0.42, y=1.04)
pd.DataFrame(src).to_csv(Path(__file__).with_name('Fig2_SOURCE.csv'), index=False, encoding='utf-8-sig')
save_figure(fig, Path(__file__).with_name('Fig2_persistence_and_cross_disease')); print('ok')
