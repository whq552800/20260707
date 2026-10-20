"""R5 (2026-10-20) Fig 4: where a country's disadvantage falls, and whether it moves.
A: within-country contrasts in the low and low-middle SDI stratum (2011-2023): % by which a 1-SD shortfall in each domain raises one
   infection's burden ratio relative to the country's other infections (top six rows), or raises mortality relative to incidence of the
   same disease (bottom three rows); * 95% interval excludes zero; outlined cells are the expectations written before the analysis.
B, C: for three pre-specified pairs in which the factor is not a covariate of the GBD model of the outcome, the difference in burden
   associated with the same absolute difference in the factor (1 cross-sectional SD), between countries in 2011-2023 and within
   countries between 2001-2010 and 2011-2023; B total burden ratio, C disease contrast (relative to the country's other infections).
Data: within_country_20261019/WC_DISEASE.csv, WC_FATALITY.csv; change_20261019/CH_STAND_MOVE.csv, CH_STAND_MOVE_CONTRAST.csv."""
from common import *
import numpy as np, pandas as pd
from matplotlib.patches import Rectangle
W = A / 'within_country_20261019'; C = A / 'change_20261019'
D = pd.read_csv(W / 'WC_DISEASE.csv'); Fa = pd.read_csv(W / 'WC_FATALITY.csv')
B = 'Low and low-middle'
DN = ['Basic services', 'Immunisation', 'Nutrition', 'Health system', 'Air pollution']
rows = [('Diarrhoeal incidence', D), ('LRI incidence', D), ('Meningitis incidence', D), ('Diarrhoeal mortality', D), ('LRI mortality', D), ('Meningitis mortality', D),
        ('Diarrhoeal', Fa), ('LRI', Fa), ('Meningitis', Fa)]
lab = ['Diarrhoeal incidence', 'LRI incidence', 'Meningitis incidence', 'Diarrhoeal mortality', 'LRI mortality', 'Meningitis mortality',
       'Diarrhoeal: deaths vs episodes', 'LRI: deaths vs episodes', 'Meningitis: deaths vs episodes']
M = np.zeros((len(rows), len(DN))); SIG = np.zeros_like(M, bool)
for i, (o, T_) in enumerate(rows):
    x = T_[(T_.band == B) & (T_.outcome == o)].set_index('domain').reindex(DN)
    M[i] = x.pct; SIG[i] = (x.lo > 0) | (x.hi < 0)
EXP = {('Diarrhoeal incidence', 'Basic services'), ('Diarrhoeal mortality', 'Basic services'), ('LRI incidence', 'Air pollution'), ('LRI mortality', 'Air pollution'),
       ('LRI incidence', 'Immunisation'), ('LRI mortality', 'Immunisation'), ('Meningitis incidence', 'Immunisation'), ('Meningitis mortality', 'Immunisation'),
       ('Diarrhoeal: deaths vs episodes', 'Health system'), ('LRI: deaths vs episodes', 'Health system'), ('Meningitis: deaths vs episodes', 'Health system')}
fig = plt.figure(figsize=(180 * MM, 138 * MM))
ga = fig.add_gridspec(1, 1, left=0.25, right=0.72, top=0.95, bottom=0.46)
gb = fig.add_gridspec(1, 2, left=0.25, right=0.985, top=0.25, bottom=0.08, wspace=0.1)
ax = fig.add_subplot(ga[0, 0])
im = ax.imshow(M, cmap='RdBu_r', vmin=-25, vmax=25, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{M[i, j]:.0f}' + ('*' if SIG[i, j] else ''), ha='center', va='center', fontsize=5.6, color='white' if abs(M[i, j]) > 17 else 'black', fontweight='bold' if SIG[i, j] else 'normal')
        if (lab[i], DN[j]) in EXP: ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec='black', lw=0.9))
ax.axhline(5.5, color='white', lw=2)
ax.set_xticks(range(len(DN))); ax.set_xticklabels(DN, rotation=35, ha='right'); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
label(ax, 'A')
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label('% per 1-SD shortfall, relative', fontsize=6); cb.ax.tick_params(labelsize=5.5)
S = pd.read_csv(C / 'CH_STAND_MOVE.csv').set_index('pair'); K = pd.read_csv(C / 'CH_STAND_MOVE_CONTRAST.csv').set_index('pair')
pairs = list(S.index); short = {'Air pollution -> LRI incidence': 'Air pollution → LRI incidence', 'Drinking water -> LRI mortality': 'Drinking water → LRI mortality',
                                'Clean cooking -> meningitis mortality': 'Clean cooking → meningitis mortality'}
for k, (title, get) in enumerate([('Total burden', lambda p: [(S.loc[p, 'stand'], S.loc[p, 'stand_lo'], S.loc[p, 'stand_hi']), (K.loc[p, 'move_total'], K.loc[p, 'mt_lo'], K.loc[p, 'mt_hi'])]),
                                  ('Relative to the country\'s other infections', lambda p: [(K.loc[p, 'stand_contrast'], K.loc[p, 'sc_lo'], K.loc[p, 'sc_hi']), (K.loc[p, 'move_contrast'], K.loc[p, 'mc_lo'], K.loc[p, 'mc_hi'])])]):
    ax = fig.add_subplot(gb[0, k]); y = np.arange(len(pairs))[::-1]
    for yi, p in zip(y, pairs):
        (a, alo, ahi), (b, blo, bhi) = get(p)
        ax.errorbar([a], [yi + 0.15], xerr=[[a - alo], [ahi - a]], fmt='o', color='#1A1A1A', ms=3, elinewidth=0.7, capsize=0)
        ax.errorbar([b], [yi - 0.15], xerr=[[b - blo], [bhi - b]], fmt='o', mfc='white', mec='#C2410C', ecolor='#C2410C', ms=3, elinewidth=0.7, capsize=0)
    ax.axvline(0, color='black', lw=0.6); ax.set_yticks(y); ax.set_yticklabels([short[p] for p in pairs] if k == 0 else []); ax.set_xlim(-15, 30)
    if k == 1: ax.tick_params(axis='y', length=0)
    label(ax, 'BC'[k])
    ax.set_xlabel('% difference per 1-SD worse conditions')
    if k == 0:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker='o', color='#1A1A1A', ls='', ms=3, label='Between countries, 2011–2023'),
                           Line2D([], [], marker='o', mfc='white', mec='#C2410C', ls='', ms=3, label='Within countries, change 2001–2010 to 2011–2023')],
                  loc='lower left', bbox_to_anchor=(-0.05, 1.12), frameon=False, fontsize=6, ncol=2)
save_figure(fig, 'Fig_allocation_change')
