"""Fig 4（服务追赶与疾病相对位置；2026-10-12 案例轨迹 C、D 移至 S6 Fig，见 supp_figs_moved.py）：
A 中期明确不利地点中，缺口收窄与未收窄者疾病死亡残差下降的比例（5组因素—结局）；
B 中期饮水明确不利的地点：饮水有利残差变化 vs 腹泻病死亡残差变化（中期→晚期），四象限计数；
数据：results_revision_20261010/PART3_CATCHUP_VS_DISEASE.csv；CONFIRMED_FACTOR_PERIOD；REPRO_DISEASE_PERIOD。
输出：Fig4_catchup_and_relative_position.{jpg,pdf} 及 Fig4_SOURCE_panel_B.csv。"""
from common import *
import numpy as np, pandas as pd
from statlib import spearman
V = A / 'results_revision_20261010'
C = pd.read_csv(V / 'PART3_CATCHUP_VS_DISEASE.csv')
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
F = pd.read_csv(A / 'factor_profiles_confirmed_20261010' / 'CONFIRMED_FACTOR_PERIOD.csv')
M2, L2 = '2001–2010', '2011–2023'
GREEN = '#1B7837'
fig = plt.figure(figsize=(180 * MM, 76 * MM))
gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.19, right=0.97, top=0.84, bottom=0.2)

# A
ax = fig.add_subplot(gs[0, 0])
PAIRS = [('basic_water_pct', '腹泻病死亡', 'Water → diarrhoeal death'), ('basic_sanitation_pct', '腹泻病死亡', 'Sanitation → diarrhoeal death'),
         ('uhc_service_index', '腹泻病死亡', 'UHC → diarrhoeal death'), ('dtp3_pct', 'LRI死亡', 'DTP3 → LRI death'), ('uhc_service_index', 'LRI死亡', 'UHC → LRI death')]
yy = np.arange(len(PAIRS))[::-1]
for yi, (f, oc, lab) in zip(yy, PAIRS):
    r = C[(C.factor == f) & (C.outcome == oc)].iloc[0]
    ax.plot([r.pct_improved_if_not, r.pct_improved_if_narrowed], [yi, yi], color=GREY, lw=1.2, zorder=1)
    ax.scatter(r.pct_improved_if_narrowed, yi, color=GREEN, s=18, zorder=2, lw=0, label='Gap narrowed' if yi == yy[0] else None)
    ax.scatter(r.pct_improved_if_not, yi, facecolor='white', edgecolor=GREEN, s=16, zorder=2, lw=0.9, label='Gap not narrowed' if yi == yy[0] else None)
    dtxt = f'{r.diff_pp:+.0f} ({r.diff_lo:.0f} to {r.diff_hi:.0f})'.replace('-', '−')
    ax.text(104, yi, dtxt, va='center', ha='left', fontsize=5.6, color='#333333')
    ax.text(150, yi, f'{r.narrowed}/{r.not_narrowed}', va='center', ha='left', fontsize=5.6, color='#555555')
ax.text(104, len(PAIRS) - 0.45, 'Difference, pp\n(95% CI)', va='bottom', ha='left', fontsize=5.6, color='#333333')
ax.text(150, len(PAIRS) - 0.45, 'n', va='bottom', ha='left', fontsize=5.6, color='#555555')
ax.set_xlim(0, 166); ax.set_xticks([0, 25, 50, 75, 100]); ax.spines['bottom'].set_bounds(0, 100)
ax.set_ylim(-0.6, len(PAIRS) - 0.3); ax.set_yticks(yy); ax.set_yticklabels([p[2] for p in PAIRS], fontsize=6)
ax.set_xlabel('Locations whose mortality residual fell (%)', x=0.3)
ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=1, fontsize=6, handletextpad=0.2, labelspacing=0.3)
fig.text(0.01, 0.92, 'A', fontsize=9, fontweight='bold')

# B
ax = fig.add_subplot(gs[0, 1])
fw = F[F.factor == 'basic_water_pct'].pivot(index='location_id', columns='period', values=['favourable_residual', 'status'])
fw = fw[(fw[('status', M2)] == '明确不利') & fw[('status', L2)].isin(['明确不利', '明确有利', '方向不确定', '已达普及'])]
rw = D[(D.cause_id == 302) & (D.measure_id == 1)].pivot(index='location_id', columns='period', values='residual')
nm = D.drop_duplicates('location_id').set_index('location_id').location
B = pd.DataFrame({'d_water': fw[('favourable_residual', L2)] - fw[('favourable_residual', M2)]})
B['d_death'] = (rw[L2] - rw[M2]).reindex(B.index); B['location'] = B.index.map(nm); B = B.dropna()
nar = B.d_water > 0
ax.scatter(B.d_water[nar], B.d_death[nar], s=9, color=GREEN, lw=0, label='Gap narrowed', zorder=2)
ax.scatter(B.d_water[~nar], B.d_death[~nar], s=9, facecolor='white', edgecolor=GREEN, lw=0.6, label='Gap not narrowed', zorder=2)
ax.axhline(0, color=GREY, lw=0.5, zorder=0); ax.axvline(0, color=GREY, lw=0.5, zorder=0)
xl, yl = ax.get_xlim(), ax.get_ylim()
q = {(1, -1): ((B.d_water > 0) & (B.d_death < 0)).sum(), (1, 1): ((B.d_water > 0) & (B.d_death >= 0)).sum(),
     (-1, -1): ((B.d_water <= 0) & (B.d_death < 0)).sum(), (-1, 1): ((B.d_water <= 0) & (B.d_death >= 0)).sum()}
for (sx, sy), n in q.items():
    ax.text(xl[1] * 0.97 if sx > 0 else xl[0] * 0.97, yl[1] * 0.93 if sy > 0 else yl[0] * 0.93, f'n={n}', ha='right' if sx > 0 else 'left',
            va='top' if sy > 0 else 'bottom', fontsize=6, color='#555555')
rho, pv = spearman(B.d_water.values, B.d_death.values)[:2]
ax.set_xlabel('Change in drinking water, observed minus\nreference (percentage points)')
ax.set_ylabel('Change in diarrhoeal mortality residual (log)')
ax.set_title(f'Water gap (adverse departure) in 2001–2010 (n={len(B)}, ρ={rho:.2f})', loc='left', fontsize=6.5)
B.to_csv(Path(__file__).with_name('Fig4_SOURCE_panel_B.csv'), encoding='utf-8-sig')
fig.text(0.52, 0.92, 'B', fontsize=9, fontweight='bold')

save_figure(fig, Path(__file__).with_name('Fig4_catchup_and_relative_position'))
print('B', len(B), int(nar.sum()), q, round(rho, 3), pv)
