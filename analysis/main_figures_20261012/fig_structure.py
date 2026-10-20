"""R5 (2026-10-20) Fig 2: two layers of burden departures.
A: share of between-location variance in each outcome's 2011-2023 departure shared with under-5 mortality from all other causes
   (R2, 95% interval from 2000 location resamples). B: early-late (1990-2000 vs 2011-2023) Spearman correlation of the total departure,
   its general part and its disease-specific part. C: 2023 excess deaths in persistent high-tail locations split into general and
   specific parts, Nigeria and the other 47 locations. Data: general_specific_20261019/GS_SHARE.csv, GS_PERSIST.csv, GS_EXCESS_2023.csv."""
from common import *
import numpy as np, pandas as pd
G = A / 'general_specific_20261019'
S = pd.read_csv(G / 'GS_SHARE.csv'); S = S[(S.band == 'All') & (S.period == '2011–2023')].set_index('outcome')
P = pd.read_csv(G / 'GS_PERSIST.csv').set_index('outcome'); X = pd.read_csv(G / 'GS_EXCESS_2023.csv')
ROWS = [('Diarrhoeal', 302), ('LRI', 322), ('Meningitis', 332)]
OO = [(f'{d} {k}', c, k) for d, c in ROWS for k in ('mortality', 'incidence')]
fig = plt.figure(figsize=(180 * MM, 70 * MM))
gs = fig.add_gridspec(1, 3, wspace=0.55, left=0.13, right=0.985, top=0.86, bottom=0.2, width_ratios=[1, 1, 0.9])
y = np.arange(len(OO))[::-1]
ax = fig.add_subplot(gs[0, 0])
for yi, (o, c, k) in zip(y, OO):
    r = S.loc[o]; m = k == 'mortality'
    ax.barh(yi, 100 * r.r2, color=DZC[c] if m else 'white', edgecolor=DZC[c], lw=0.8, height=0.65)
    ax.plot([100 * r.r2_lo, 100 * r.r2_hi], [yi, yi], color='black', lw=0.7)
ax.set_yticks(y); ax.set_yticklabels([o for o, _, _ in OO]); ax.set_xlim(0, 60)
ax.set_xlabel('Variance shared with other-cause\nunder-5 mortality (%)'); label(ax, 'A')
ax = fig.add_subplot(gs[0, 1])
for yi, (o, c, k) in zip(y, OO):
    r = P.loc[o]
    ax.plot([r.rho_general], [yi + 0.18], marker='o', mfc='white', mec='#555555', ms=3.6, ls='')
    ax.plot([r.rho_specific], [yi - 0.18], marker='o', color=DZC[c], ms=3.6, ls='')
    ax.plot([r.rho_total], [yi], marker='|', color='black', ms=6, ls='')
ax.set_yticks(y); ax.set_yticklabels([]); ax.tick_params(axis='y', length=0); ax.set_xlim(0.2, 1.0)
for yy in y[:-1] - 0.5: ax.axhline(yy, color='#EEEEEE', lw=0.5)
ax.set_xlabel('Rank correlation,\n1990–2000 vs 2011–2023'); label(ax, 'B')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker='o', mfc='white', mec='#555555', ls='', ms=3.6, label='General part'),
                   Line2D([], [], marker='o', color='#542788', ls='', ms=3.6, label='Disease-specific part'),
                   Line2D([], [], marker='|', color='black', ls='', ms=6, label='Total')], loc='lower left', frameon=False, fontsize=5.8, handletextpad=0.2)
ax = fig.add_subplot(gs[0, 2])
a = X[(X.disease == 'All three') & (X.locations == 'Persistent high tail')].iloc[0]; b = X[(X.disease == 'All three') & (X.locations == 'Persistent high tail without Nigeria')].iloc[0]
ng = (a.general - b.general, a.specific - b.specific); ot = (b.general, b.specific)
for i, (lab, (g, s_)) in enumerate([('Nigeria', ng), ('Other 47\nlocations', ot)]):
    ax.bar(i, g / 1e3, color='#9E9E9E', width=0.6, label='General' if i == 0 else None)
    ax.bar(i, s_ / 1e3, bottom=g / 1e3, color='#542788', width=0.6, label='Disease-specific' if i == 0 else None)
    ax.text(i, (g + s_) / 1e3 + 2, f'{100 * g / (g + s_):.0f}% general', ha='center', fontsize=5.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Nigeria', 'Other 47\nlocations']); ax.set_ylabel('Excess under-5 deaths, 2023 (thousands)'); ax.set_ylim(0, 140)
ax.legend(frameon=False, fontsize=5.8, loc='upper right', bbox_to_anchor=(1.03, 0.78)); label(ax, 'C')
save_figure(fig, 'Fig_structure')
