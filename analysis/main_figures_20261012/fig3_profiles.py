"""Fig 3 (2026-10-12): the profile of high-tail locations changed with development, and nutrition was the part that moved.
A: for each SDI stratum and outcome in 2011-2023, difference between high-tail and low-tail locations in the percentage with an adverse
departure for each factor (percentage points); grey, fewer than 3 evaluable locations in a tail; dots, low and low-middle SDI cells whose
Newcombe 95% interval excludes 0. B: the same difference averaged over the factors of each domain and over the three mortality or the three
incidence outcomes, with 95% intervals from 2000 location resamples. C: change from 2001-2010 to 2011-2023 in the percentage of
record-factor pairs with an adverse departure, compared between records that left and stayed in the high tail and between records that
entered the high tail and stayed in the low tail (all strata), with 95% intervals.
Data: results_revision_20261010/PROFILE_CELLS.csv, PROFILE_DOMAINS_POOLED.csv, PROFILE_TRAJECTORY_CONTRASTS.csv.
Outputs: Fig3_profiles_by_development.{jpg,pdf}, Fig3_SOURCE.csv."""
from common import *
import numpy as np, pandas as pd
from matplotlib.colors import TwoSlopeNorm
R = A / 'results_revision_20261010'
C = pd.read_csv(R / 'PROFILE_CELLS.csv'); C = C[C.period == '2011–2023']
PD = pd.read_csv(R / 'PROFILE_DOMAINS_POOLED.csv'); TC = pd.read_csv(R / 'PROFILE_TRAJECTORY_CONTRASTS.csv')
DOM = [('Basic services', [('basic_water_pct', 'Drinking water'), ('basic_sanitation_pct', 'Sanitation'), ('clean_cooking_pct', 'Clean cooking fuels')]),
       ('Immunisation', [('dtp3_pct', 'DTP3'), ('hib3_pct', 'Hib3'), ('pcv3_pct', 'PCV3')]),
       ('Nutrition', [('stunting_u5_pct', 'Stunting'), ('low_birthweight_pct', 'Low birthweight'), ('undernourishment_pct', 'Undernourishment')]),
       ('Health system', [('uhc_service_index', 'UHC index'), ('health_expenditure_pc_current_usd', 'Health expenditure'), ('hospital_beds_per1000', 'Hospital beds'), ('nurses_midwives_per1000', 'Nurses and midwives')]),
       ('Air pollution', [('gbd_ambient_pm25_ug_m3', 'Ambient PM2.5'), ('secondhand_smoke_sev', 'Second-hand smoke')])]
FAC = [f for _, v in DOM for f in v]
BANDS = ['Low and low-middle', 'Middle and high-middle', 'High']; BLAB = ['Low and low-middle SDI', 'Middle and high-middle SDI', 'High SDI']
OUTS = ['Diarrhoeal incidence', 'Diarrhoeal mortality', 'LRI incidence', 'LRI mortality', 'Meningitis incidence', 'Meningitis mortality']
OLAB = ['Diar. inc.', 'Diar. mort.', 'LRI inc.', 'LRI mort.', 'Men. inc.', 'Men. mort.']
src = []
fig = plt.figure(figsize=(180 * MM, 185 * MM))
ga = fig.add_gridspec(1, 1, left=0.20, right=0.89, top=0.935, bottom=0.50)
gb = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.55, left=0.17, right=0.975, top=0.355, bottom=0.115)
ax = fig.add_subplot(ga[0, 0])
M = np.full((len(FAC), 18), np.nan); SIG = np.zeros_like(M, bool); NH = []
for bi, b in enumerate(BANDS):
    for oi, o in enumerate(OUTS):
        j = 6 * bi + oi; x = C[(C.band == b) & (C.outcome == o)].set_index('factor')
        NH.append(f'{int(x.n_high_tail.iloc[0])}/{int(x.n_low_tail.iloc[0])}')
        for i, (f, lab) in enumerate(FAC):
            r = x.loc[f]
            if pd.notna(r.get('diff_pp')):
                M[i, j] = r.diff_pp; SIG[i, j] = (b == BANDS[0]) and (r.diff_lo > 0 or r.diff_hi < 0)
            src.append(dict(panel='A', band=b, outcome=o, factor=lab, diff_pp=r.get('diff_pp'), lo=r.get('diff_lo'), hi=r.get('diff_hi'),
                            pct_adv_high=r.get('pct_adv_high'), pct_adv_low=r.get('pct_adv_low'), eval_high=r.eval_high, eval_low=r.eval_low))
cm = plt.get_cmap('RdBu_r').copy(); cm.set_bad('#A6A6A6')
im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, norm=TwoSlopeNorm(0, -100, 100), aspect='auto')
for i in range(len(FAC)):
    for j in range(18):
        if SIG[i, j]: ax.scatter(j, i, s=5, color='black', lw=0)
for x_ in [5.5, 11.5]: ax.axvline(x_, color='white', lw=2.5)
yb = 0
for dname, fs in DOM[:-1]:
    yb += len(fs); ax.axhline(yb - 0.5, color='white', lw=2.5)
ax.set_yticks(range(len(FAC))); ax.set_yticklabels([l for _, l in FAC])
ax.set_xticks(range(18)); ax.set_xticklabels(OLAB * 3, rotation=60, ha='right', rotation_mode='anchor', fontsize=6)
ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
for bi, bl in enumerate(BLAB): ax.text(6 * bi + 2.5, -1.4, bl, ha='center', va='bottom', fontsize=6.8)
for j, n in enumerate(NH): ax.text(j, len(FAC) - 0.35, n, ha='center', va='top', fontsize=4.8, color='#555555')
ax.text(-0.6, len(FAC) - 0.35, 'n high/low', ha='right', va='top', fontsize=4.8, color='#555555')
yb = 0
for dname, fs in DOM:
    ax.text(-0.26, yb + (len(fs) - 1) / 2, dname, transform=ax.get_yaxis_transform(), rotation=90, ha='center', va='center', fontsize=6.2)
    yb += len(fs)
cax = fig.add_axes([0.905, 0.64, 0.012, 0.25]); cb = fig.colorbar(im, cax=cax); cb.set_ticks([-100, -50, 0, 50, 100]); cb.outline.set_linewidth(0.4)
cb.set_label('High tail minus low tail,\n% with adverse departure', fontsize=6)
ax.set_xlim(-0.5, 17.5); ax.set_ylim(len(FAC) + 0.9, -0.5)
label(ax, 'A', x=-0.27, y=1.07)
# B domain summary
ax = fig.add_subplot(gb[0, 0]); DN = [d for d, _ in DOM]; y0 = np.arange(len(DN))[::-1]
BC = {'Low and low-middle': '#1A1A1A', 'Middle and high-middle': '#7A7A7A', 'High': '#BDBDBD'}
for bi, b in enumerate(BANDS):
    for t, mk, dy in [('Mortality', 'o', 0.09), ('Incidence', 's', -0.09)]:
        for yi, dn in zip(y0, DN):
            r = PD[(PD.band == b) & (PD.outcome_type == t) & (PD.domain == dn)].iloc[0]
            yy = yi + (1 - bi) * 0.27 + dy
            ax.plot([r.lo, r.hi], [yy, yy], color=BC[b], lw=0.9)
            ax.scatter(r.diff_pp, yy, s=13, marker=mk, color=BC[b], facecolor=BC[b] if t == 'Mortality' else 'white', lw=0.8, zorder=3)
            src.append(dict(panel='B', band=b, outcome_type=t, domain=dn, diff_pp=r.diff_pp, lo=r.lo, hi=r.hi))
ax.axvline(0, color='black', lw=0.6)
for yi in y0[:-1]: ax.axhline(yi - 0.5, color='#E0E0E0', lw=0.5)
ax.set_yticks(y0); ax.set_yticklabels(DN); ax.set_ylim(-0.55, len(DN) - 0.45); ax.set_xlim(-50, 80)
ax.set_xlabel('High tail minus low tail, % adverse (pp)')
for b, bl in zip(BANDS, BLAB): ax.plot([], [], color=BC[b], lw=3, label=bl.replace(' SDI', ''))
ax.scatter([], [], marker='o', s=13, color='k', label='Mortality'); ax.scatter([], [], marker='s', s=13, facecolor='white', edgecolor='k', label='Incidence')
ax.legend(loc='upper left', bbox_to_anchor=(-0.42, -0.19), ncol=3, fontsize=5.8, columnspacing=0.9, handlelength=1.2)
ax.set_title('Profiles by domain, 2011–2023', loc='left')
label(ax, 'B', x=-0.42)
# C trajectories
ax = fig.add_subplot(gb[0, 1])
x = TC[TC.band == 'All strata']
for con, col, dy, lab in [('Left high tail minus Stayed in high tail', '#2166AC', 0.13, 'Left vs stayed in high tail'), ('Entered high tail minus Stayed in low tail', '#B2182B', -0.13, 'Entered high tail vs stayed in low tail')]:
    for yi, dn in zip(y0, DN):
        r = x[(x.contrast == con) & (x.domain == dn)].iloc[0]
        ax.plot([r.lo, r.hi], [yi + dy] * 2, color=col, lw=0.9); ax.scatter(r.diff_change_pp, yi + dy, s=13, color=col, zorder=3, lw=0)
        src.append(dict(panel='C', contrast=con, domain=dn, diff_change_pp=r.diff_change_pp, lo=r.lo, hi=r.hi))
    ax.scatter([], [], s=13, color=col, label=lab)
ax.axvline(0, color='black', lw=0.6)
for yi in y0[:-1]: ax.axhline(yi - 0.5, color='#E0E0E0', lw=0.5)
ax.set_yticks(y0); ax.set_yticklabels(DN); ax.set_ylim(-0.55, len(DN) - 0.45); ax.set_xlim(-30, 30)
ax.set_xlabel('Difference in change of % adverse,\n2001–2010 to 2011–2023 (pp)')
ax.legend(loc='upper left', bbox_to_anchor=(-0.55, -0.26), ncol=1, fontsize=5.8)
ax.set_title('Profile change with tail membership', loc='left')
label(ax, 'C', x=-0.62)
pd.DataFrame(src).to_csv(Path(__file__).with_name('Fig3_SOURCE.csv'), index=False, encoding='utf-8-sig')
save_figure(fig, Path(__file__).with_name('Fig3_profiles_by_development')); print('ok')
