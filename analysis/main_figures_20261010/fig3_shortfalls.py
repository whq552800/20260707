"""Fig 3（短板随发展改变；2026-10-12 按作者决定改版）：
A 2011—2023年按结局计数的高尾减低尾“不利偏离”比例差（S1 Text 表S、表W 的等权汇总），低及中低SDI层与高SDI层并列，含95%区间；
  上半部分按因素领域，最后一行为全部疾病相关因素（不含UHC）；下半部分按因素是否为该结局GBD模型的协变量分组；右侧为两层之差。
B 持续高尾/低尾记录中至少一项疾病相关因素、UHC指数为不利偏离的比例（按晚期SDI层）。
原 A（按记录合并的领域比例）移至 S4 Fig，原 C（高SDI地点PM2.5）移至 S5 Fig（supp_figs_moved.py）。
数据：results_revision_20261010/OUTCOME_FACTOR_GRADIENT.csv、OUTCOME_FACTOR_COVSTATUS.csv、PART3_SHORTFALL_BY_BAND.csv。"""
from common import *
import numpy as np, pandas as pd
V = A / 'results_revision_20261010'
G = pd.read_csv(V / 'OUTCOME_FACTOR_GRADIENT.csv').set_index('domain'); CS = pd.read_csv(V / 'OUTCOME_FACTOR_COVSTATUS.csv')
DARK, LIGHT = '#1A1A1A', '#8C8C8C'
fig = plt.figure(figsize=(180 * MM, 98 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.62, left=0.205, right=0.975, top=0.9, bottom=0.14)
ax = fig.add_subplot(gs[0, 0])
DOM = [('Water and sanitation', 'Water and sanitation'), ('Immunisation', 'Immunisation'), ('Clean cooking', 'Clean cooking'), ('Nutrition', 'Nutrition'),
       ('Exposure', 'Air pollution'), ('UHC index', 'UHC index'), ('All disease-relevant factors', 'All disease-relevant factors')]
rows = []
for k, lab in DOM:
    r = G.loc[k]; rows.append((lab + f' ({int(r.n_pairs)})', (r.low_band_diff, r.low_lo, r.low_hi), (r.high_band_diff, r.high_lo, r.high_hi),
                              f'{r.between_band:.0f} ({r.between_lo:.0f} to {r.between_hi:.0f})'.replace('-', '−'), k == 'All disease-relevant factors'))
CSN = [('direct covariate', 'Direct GBD covariate'), ('not a covariate', 'Not a GBD covariate'), ('LRI incidence, all pairs (model uses no cause-specific mortality)', 'LRI incidence, all factors')]
for k, lab in CSN:
    a = CS[(CS.set == k) & CS.band.str.startswith('Low')].iloc[0]; b = CS[(CS.set == k) & (CS.band == 'High')].iloc[0]
    rows.append((lab + f' ({int(a.n_pairs)})', (a.estimate, a.lo, a.hi), (b.estimate, b.lo, b.hi), '', False))
n1 = len(DOM); y = []; yy = 0.0
for i in range(len(rows)):
    if i == n1: yy -= 0.9
    y.append(yy); yy -= 1
y = np.array(y)
for yi, (lab, lo_, hi_, dtxt, bold) in zip(y, rows):
    for (e, l, h), off, col, mk in [(lo_, 0.17, DARK, 'o'), (hi_, -0.17, LIGHT, 's')]:
        ax.plot([l, h], [yi + off, yi + off], color=col, lw=0.9, zorder=1); ax.scatter([e], [yi + off], color=col, marker=mk, s=13, zorder=2, lw=0)
    if dtxt: ax.text(101, yi, dtxt, va='center', ha='left', fontsize=5.8, fontweight='bold' if bold else 'normal')
ax.axvline(0, color=GREY, lw=0.6, zorder=0)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6)
for t, r in zip(ax.get_yticklabels(), rows):
    if r[4]: t.set_fontweight('bold')
ax.set_xlim(-62, 98); ax.set_xticks([-50, -25, 0, 25, 50, 75]); ax.spines['bottom'].set_bounds(-62, 98)
ax.set_ylim(y.min() - 0.7, 0.75)
ax.text(101, 0.6, 'Difference\nbetween strata', va='bottom', ha='left', fontsize=5.8)
ax.text(-62, y[n1] + 0.75, 'By GBD covariate status', ha='left', va='center', fontsize=6, fontstyle='italic', color='#444444')
ax.set_xlabel('High tail minus low tail, locations with an adverse departure (pp)', x=0.45)
ax.scatter([], [], color=DARK, marker='o', s=13, label='Low and low-middle SDI'); ax.scatter([], [], color=LIGHT, marker='s', s=13, label='High SDI')
ax.legend(loc='lower left', bbox_to_anchor=(-0.01, 1.0), ncol=2, fontsize=6.2, handletextpad=0.2, columnspacing=1.2)
ax.grid(axis='x', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
fig.text(0.005, 0.935, 'A', fontsize=9, fontweight='bold')
# B
SB = pd.read_csv(V / 'PART3_SHORTFALL_BY_BAND.csv')
BANDS = [('低+中低', 'Low & low-middle SDI'), ('中等+中高', 'Middle & high-middle SDI'), ('高', 'High SDI')]
ax = fig.add_subplot(gs[0, 1]); yb, ylab = [], []; ypos = 7.0
for b, bl in BANDS:
    x = SB[SB.band == b].set_index('persist')
    for col, lab, nn in [('any_specific', 'Disease-relevant factors', None), ('uhc_adverse', 'UHC index', 'uhc_evaluable')]:
        h, l = x.loc['持续高尾', col], x.loc['持续低尾', col]
        ax.plot([l, h], [ypos, ypos], color=GREY, lw=1.2, zorder=1)
        ax.scatter([h], [ypos], color=HIGH, s=14, zorder=2, lw=0); ax.scatter([l], [ypos], color=LOW, s=14, zorder=2, lw=0)
        dv = round(h - l); ax.text(104, ypos, ('0' if dv == 0 else f'{dv:+d}'.replace('-', '−')), va='center', ha='left', fontsize=6)
        nh = int(x.loc['持续高尾', 'n'] - x.loc['持续高尾', 'n_eval0']) if nn is None else int(x.loc['持续高尾', nn])
        nl = int(x.loc['持续低尾', 'n'] - x.loc['持续低尾', 'n_eval0']) if nn is None else int(x.loc['持续低尾', nn])
        ax.text(116, ypos, f'{nh}/{nl}', va='center', ha='left', fontsize=5.6, color='#555555'); yb.append(ypos); ylab.append(lab); ypos -= 1
    ax.text(0, ypos + 2.62, bl, ha='left', va='center', fontsize=6.3, fontweight='bold'); ypos -= 1.0
ax.text(108, 7.6, 'Δ', ha='center', va='bottom', fontsize=6.5, fontweight='bold'); ax.text(116, 7.6, 'n', ha='left', va='bottom', fontsize=5.8, color='#555555')
ax.set_yticks(yb); ax.set_yticklabels(ylab, fontsize=6); ax.set_xlim(0, 128); ax.set_xticks([0, 25, 50, 75, 100]); ax.spines['bottom'].set_bounds(0, 100)
ax.set_xlabel('Persistent tail records with an adverse departure (%)', x=0.4); ax.set_ylim(min(yb) - 0.6, 8.0)
ax.scatter([], [], color=HIGH, s=14, label='High tail'); ax.scatter([], [], color=LOW, s=14, label='Low tail')
ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.2, handletextpad=0.2, columnspacing=1.0)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0); ax.grid(axis='x', color='#EEEEEE', lw=0.5); ax.set_axisbelow(True)
fig.text(0.585, 0.935, 'B', fontsize=9, fontweight='bold')
save_figure(fig, Path(__file__).with_name('Fig3_shortfalls_by_development')); print('ok')
