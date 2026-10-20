"""Fig 1 (2026-10-12): the low-high SDI gap narrowed for infection but not for survival.
A-C: ratio of low-SDI to high-SDI group rates, 1990-2023, for incidence, case fatality and mortality of each disease (groups fixed by
mean SDI in 1990-2000), with 95% intervals. D: change in the ratio from 1990 to 2023 (ratio in 2023 / ratio in 1990), groups fixed
in 1990-2000 (filled) and in 2011-2023 (open). E: decline in incidence and case fatality from 1990 to 2023 in the low-SDI and high-SDI
groups (groups fixed in 1990-2000). Data: gap_story_20261012/GAP_ANNUAL.csv and GAP_SUMMARY.csv.
Outputs: Fig1_gap_incidence_case_fatality.{jpg,pdf}, Fig1_SOURCE.csv."""
from common import *
import numpy as np, pandas as pd, sys
G = A / 'gap_story_20261012'
AN = pd.read_csv(G / 'GAP_ANNUAL.csv'); S = pd.read_csv(G / 'GAP_SUMMARY.csv')
fmt = lambda v: f'{v:.0f}' if v >= 100 else f'{v:.1f}'
DIS = ['Diarrhoeal disease', 'LRI', 'Meningitis']; MEAS = ['Incidence', 'Case fatality', 'Mortality']
LAB = {'Case fatality': 'Deaths per\nepisode'}   # 2026-10-17: population ratio of annual deaths to incident episodes, not cohort case fatality
fig = plt.figure(figsize=(180 * MM, 140 * MM))
gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.075, right=0.985, top=0.95, bottom=0.60)
gb = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=1.0, left=0.165, right=0.985, top=0.44, bottom=0.10)
src = []
YT = [1, 2, 5, 10, 20, 50, 100, 200, 500]
for k, dz in enumerate(DIS):
    ax = fig.add_subplot(gs[0, k])
    for m in MEAS:
        x = AN[(AN.design == 'early-fixed') & (AN.disease == dz) & (AN.measure == m)]
        ax.fill_between(x.year, x.RR_lo, x.RR_hi, color=MCOL[m], alpha=0.10 if m == 'Incidence' else 0.15, lw=0)
        ax.plot(x.year, x.RR, color=MCOL[m], lw=1.3, ls='-' if m != 'Incidence' else '--')
        src += [dict(panel='ABC'[k], disease=dz, measure=m, year=r.year, RR=r.RR, lo=r.RR_lo, hi=r.RR_hi) for r in x.itertuples()]
    ax.set_yscale('log'); lo_, hi_ = {0: (1, 600), 1: (1, 150), 2: (1, 100)}[k]
    ax.set_ylim(lo_, hi_); ax.set_yticks([t for t in YT if lo_ <= t <= hi_]); ax.set_yticklabels([str(t) for t in YT if lo_ <= t <= hi_]); ax.minorticks_off()
    ax.set_xlim(1990, 2023); ax.set_xticks([1990, 2000, 2010, 2020])
    if k == 0: ax.set_ylabel('Rate ratio, low-SDI / high-SDI group')
    ax.set_title(dz.replace('Diarrhoeal disease', 'Diarrhoeal disease'), loc='left', color=DZC[[302, 322, 332][k]])
    label(ax, 'ABC'[k], x=-0.24 if k == 0 else -0.17)
    if k == 2:
        for m in MEAS: ax.plot([], [], color=MCOL[m], lw=1.3, ls='-' if m != 'Incidence' else '--', label=LAB.get(m, m).replace('\n', ' '))
        ax.fill_between([], [], [], color='#999999', alpha=0.15, lw=0, label='95% interval')
        fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.53, 0.535), ncol=4, fontsize=6.5, handlelength=2.2)
# D change in the ratio
ax = fig.add_subplot(gb[0, 0]); rows = [(dz, m) for dz in DIS for m in MEAS]; y = np.arange(len(rows))[::-1]
for yi, (dz, m) in zip(y, rows):
    for des, off, fc in [('early-fixed', 0.16, None), ('late-fixed', -0.16, 'white')]:
        r = S[(S.design == des) & (S.disease == dz) & (S.measure == m)].iloc[0]
        ax.plot([r.RR_change_lo, r.RR_change_hi], [yi + off] * 2, color=MCOL[m], lw=1)
        ax.scatter(r.RR_change, yi + off, s=16, color=MCOL[m], facecolor=fc if fc else MCOL[m], lw=0.8, zorder=3)
        src.append(dict(panel='D', disease=dz, measure=m, design=des, change=r.RR_change, lo=r.RR_change_lo, hi=r.RR_change_hi))
    if des == 'late-fixed':
        r = S[(S.design == 'early-fixed') & (S.disease == dz) & (S.measure == m)].iloc[0]
        ax.text(5.2, yi, f'{fmt(r.RR_1990)} → {fmt(r.RR_2023)}', va='center', fontsize=5.8)
ax.axvline(1, color='black', lw=0.6)
for b in [2.5, 5.5]: ax.axhline(b, color='#DDDDDD', lw=0.5)
ax.set_xscale('log'); ax.set_xlim(0.15, 4.4); ax.set_xticks([0.2, 0.5, 1, 2, 4]); ax.set_xticklabels(['0.2', '0.5', '1', '2', '4']); ax.minorticks_off()
ax.set_yticks(y); ax.set_yticklabels([LAB.get(m, m) for _, m in rows]); ax.set_ylim(-0.6, len(rows) - 0.4)
for i, dz in enumerate(DIS):
    ax.text(-0.29, y[3 * i + 1], dz.replace('Diarrhoeal disease', 'Diarrhoeal'), transform=ax.get_yaxis_transform(), rotation=90, va='center', ha='center', fontsize=6.5, color=DZC[[302, 322, 332][i]])
ax.set_xlabel('Ratio in 2023 / ratio in 1990 (log scale)')
ax.text(1.06, 1.01, 'Gap widened →', transform=ax.get_xaxis_transform(), ha='left', va='bottom', fontsize=6)
ax.text(0.94, 1.01, '← Gap narrowed', transform=ax.get_xaxis_transform(), ha='right', va='bottom', fontsize=6)
ax.text(5.2, len(rows) - 0.45, 'Ratio,\n1990 → 2023', fontsize=5.8, va='bottom')
ax.scatter([], [], s=16, color='k', label='Groups fixed in 1990–2000'); ax.scatter([], [], s=16, facecolor='white', edgecolor='k', lw=0.8, label='Groups fixed in 2011–2023')
ax.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.17), ncol=2, fontsize=6)

label(ax, 'D', x=-0.33)
# E declines
ax = fig.add_subplot(gb[0, 1]); rows = [(dz, m) for dz in DIS for m in ['Incidence', 'Case fatality']]; y = np.arange(len(rows))[::-1]
for yi, (dz, m) in zip(y, rows):
    r = S[(S.design == 'early-fixed') & (S.disease == dz) & (S.measure == m)].iloc[0]
    for g, off, fc in [('low', 0.15, 'k'), ('high', -0.15, 'white')]:
        v, lo, hi = r[f'decline_{g}_pct'], r[f'decline_{g}_pct_lo'], r[f'decline_{g}_pct_hi']
        ax.plot([max(lo, -40), hi], [yi + off] * 2, color=MCOL[m], lw=1)
        ax.scatter(v, yi + off, s=16, marker='o' if g == 'low' else 's', facecolor=MCOL[m] if fc == 'k' else 'white', edgecolor=MCOL[m], lw=0.8, zorder=3)
        src.append(dict(panel='E', disease=dz, measure=m, group=g, decline=v, lo=lo, hi=hi))
ax.axvline(0, color='black', lw=0.6)
for b in [1.5, 3.5]: ax.axhline(b, color='#DDDDDD', lw=0.5)
ax.set_yticks(y); ax.set_yticklabels([LAB.get(m, m) for _, m in rows]); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlim(-40, 100); ax.set_xticks([-40, 0, 25, 50, 75, 100])
for i, dz in enumerate(DIS):
    ax.text(-0.6, (y[2 * i] + y[2 * i + 1]) / 2, dz.replace('Diarrhoeal disease', 'Diarrhoeal'), transform=ax.get_yaxis_transform(), rotation=90, va='center', ha='center', fontsize=6.5, color=DZC[[302, 322, 332][i]])
ax.set_xlabel('Decline, 1990 to 2023 (%)')
ax.scatter([], [], s=16, color='k', label='Low-SDI group'); ax.scatter([], [], s=16, marker='s', facecolor='white', edgecolor='k', lw=0.8, label='High-SDI group')
ax.legend(loc='upper left', bbox_to_anchor=(-0.5, -0.17), ncol=2, fontsize=6, columnspacing=0.8, handletextpad=0.2)

label(ax, 'E', x=-0.68)
pd.DataFrame(src).to_csv(Path(__file__).with_name('Fig1_SOURCE.csv'), index=False, encoding='utf-8-sig')
save_figure(fig, Path(__file__).with_name('Fig1_gap_incidence_case_fatality')); print('ok')
