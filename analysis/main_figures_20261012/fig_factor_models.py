"""R4 (2026-10-17): factor-centred main figure.
A: change in the burden ratio (%) per 1-SD shortfall in each domain, from multivariable models fitted separately for each outcome
   within each SDI stratum (2011-2023), with 95% location-bootstrap intervals; mortality (filled) and incidence (open) by disease colour.
B: share of the variance in burden departures explained by each domain (Shapley decomposition of R^2), by stratum and outcome.
Data: factor_models_20261017/FM_COEF.csv, FM_SHAPLEY.csv. Output: Fig_factor_models.{jpg,pdf}."""
from common import *
import numpy as np, pandas as pd
FMD = A / 'factor_models_20261017'
C = pd.read_csv(FMD / 'FM_COEF.csv'); S = pd.read_csv(FMD / 'FM_SHAPLEY.csv')
BANDS = ['Low and low-middle', 'Middle and high-middle', 'High']; BL = {'Low and low-middle': 'Low and low-middle SDI', 'Middle and high-middle': 'Middle and high-middle SDI', 'High': 'High SDI'}
DN = ['Basic services', 'Immunisation', 'Nutrition', 'Health system', 'Air pollution']
OUTS = [('Diarrhoeal', 302), ('LRI', 322), ('Meningitis', 332)]
fig = plt.figure(figsize=(180 * MM, 165 * MM))
ga = fig.add_gridspec(1, 3, wspace=0.12, left=0.17, right=0.985, top=0.94, bottom=0.47)
gb = fig.add_gridspec(1, 3, wspace=0.12, left=0.17, right=0.985, top=0.33, bottom=0.08)
xs = np.arange(len(DN))[::-1]
for k, b in enumerate(BANDS):
    ax = fig.add_subplot(ga[0, k])
    for i, (dz, c) in enumerate(OUTS):
        for j, (kind, filled) in enumerate([('mortality', True), ('incidence', False)]):
            o = f'{dz} {kind}'; x = C[(C.band == b) & (C.outcome == o)].set_index('domain').reindex(DN)
            y = xs + 0.3 - (i * 2 + j) * 0.12
            ax.errorbar(x.pct_per_sd, y, xerr=[x.pct_per_sd - x.lo, x.hi - x.pct_per_sd], fmt='o' if filled else 's', ms=2.8,
                        mfc=DZC[c] if filled else 'white', mec=DZC[c], ecolor=DZC[c], elinewidth=0.7, capsize=0, lw=0)
    ax.axvline(0, color='black', lw=0.6)
    for yy in xs[:-1] - 0.5: ax.axhline(yy, color='#DDDDDD', lw=0.5)
    ax.set_yticks(xs); ax.set_yticklabels(DN if k == 0 else []); ax.set_xlim(-40, 120)
    ax.set_title(BL[b], loc='left'); ax.set_xlabel('Change in reference ratio per 1-SD shortfall (%); intervals truncated at −40 and 120' if k == 1 else '')
    if k == 0: label(ax, 'A')
    if k != 0: ax.tick_params(axis='y', length=0)
from matplotlib.lines import Line2D
h = [x for dz, c in OUTS for x in (Line2D([], [], marker='o', ls='', color=DZC[c], ms=3.5, label=f'{dz} mortality'),
                                    Line2D([], [], marker='s', ls='', mfc='white', mec=DZC[c], ms=3.5, label=f'{dz} incidence'))]
fig.legend(handles=h, loc='center', bbox_to_anchor=(0.58, 0.395), ncol=3, fontsize=6, frameon=False)
COL = {'Basic services': '#1B9E77', 'Immunisation': '#7570B3', 'Nutrition': '#D95F02', 'Health system': '#E7298A', 'Air pollution': '#666666'}
for k, b in enumerate(BANDS):
    ax = fig.add_subplot(gb[0, k]); rows = [f'{dz} {kind}' for dz, _ in OUTS for kind in ('mortality', 'incidence')]; y = np.arange(len(rows))[::-1]
    for yi, o in zip(y, rows):
        x = S[(S.band == b) & (S.outcome == o)].set_index('domain').reindex(DN); left = 0
        for d in DN:
            v = max(x.loc[d].shapley_r2, 0) * 100; ax.barh(yi, v, left=left, color=COL[d], height=0.7, lw=0); left += v
    ax.set_yticks(y); ax.set_yticklabels([r.replace('Diarrhoeal', 'Diarrh.').replace(' mortality', ' mort.').replace(' incidence', ' inc.') for r in rows] if k == 0 else [])
    ax.set_xlim(0, 80); ax.set_title(BL[b], loc='left'); ax.set_xlabel('Variance in burden departures explained (R², %)' if k == 1 else '')
    if k == 0: label(ax, 'B')
    if k != 0: ax.tick_params(axis='y', length=0)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COL[d], label=d) for d in DN], loc='lower center', bbox_to_anchor=(0.58, -0.005), ncol=5, fontsize=6, frameon=False)
save_figure(fig, Path(__file__).with_name('Fig_factor_models')); print('ok')
