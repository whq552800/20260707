"""Supplementary figures in the order of the main text (2026-10-12 restructuring). Output: analysis/supp_figures_20261012/.
S1 Fig (Results 1): group rates of incidence and mortality, groups fixed in 1990-2000 (A, B); rate ratios with groups fixed in
       2011-2023 (C-E), with 95% intervals.
S2 Fig (Results 2): period residuals in 1990-2000 against 2011-2023 for each outcome (copied from supp_figures_20261010/S1_Fig).
S3 Fig (Results 2): correlations between the six outcomes in 2011-2023 (A) and incidence against mortality residuals of each disease (B-D).
S4 Fig (Results 3): percentage of record-factor pairs with an adverse departure in 2001-2010 and 2011-2023 by tail trajectory, domain and
       SDI stratum.
S5 Fig (Results 3): narrowing of a factor gap and concurrent change in the mortality residual (former Fig 4).
S6 Fig (Results 3): trajectories of case countries (copied from supp_figures_20261010/S6_Fig).
S7 Fig (Results 3): LRI incidence and ambient PM2.5 residuals within subregions, high-SDI locations (copied from supp_figures_20261010/S5_Fig)."""
from common import *
import shutil, numpy as np, pandas as pd
from statlib import spearman
OUTS.mkdir(exist_ok=True)
G = A / 'gap_story_20261012'; R = A / 'results_revision_20261010'
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
E0, L0 = PER[0], PER[2]
# ---------- S1 Fig
AN = pd.read_csv(G / 'GAP_ANNUAL.csv'); DIS = ['Diarrhoeal disease', 'LRI', 'Meningitis']; CC = dict(zip(DIS, [302, 322, 332]))
fig = plt.figure(figsize=(180 * MM, 120 * MM))
g1 = fig.add_gridspec(1, 2, wspace=0.35, left=0.08, right=0.83, top=0.93, bottom=0.60)
g2 = fig.add_gridspec(1, 3, wspace=0.32, left=0.08, right=0.985, top=0.40, bottom=0.08)
for k, (m, yl) in enumerate([('Mortality', 'Deaths per 100 000 children <5'), ('Incidence', 'Cases per 100 000 children <5')]):
    ax = fig.add_subplot(g1[0, k])
    for dz in DIS:
        x = AN[(AN.design == 'early-fixed') & (AN.disease == dz) & (AN.measure == m)]
        ax.plot(x.year, x.low, color=DZC[CC[dz]], lw=1.2); ax.plot(x.year, x.high, color=DZC[CC[dz]], lw=1.2, ls='--')
        ax.text(2024, x.low.iloc[-1], dz.replace(' disease', ''), color=DZC[CC[dz]], fontsize=6, va='center')
    ax.set_yscale('log'); ax.set_xlim(1990, 2033); ax.set_xticks([1990, 2000, 2010, 2020]); ax.set_ylabel(yl); ax.set_title(f'{m}, groups fixed in 1990–2000', loc='left')
    label(ax, 'AB'[k], x=-0.22)
ax.plot([], [], color='k', lw=1.2, label='Low-SDI group'); ax.plot([], [], color='k', lw=1.2, ls='--', label='High-SDI group')
ax.legend(loc='upper left', bbox_to_anchor=(1.12, 1.0), fontsize=6)
for k, dz in enumerate(DIS):
    ax = fig.add_subplot(g2[0, k])
    for m in ['Incidence', 'Case fatality', 'Mortality']:
        x = AN[(AN.design == 'late-fixed') & (AN.disease == dz) & (AN.measure == m)]
        ax.fill_between(x.year, x.RR_lo, x.RR_hi, color=MCOL[m], alpha=0.12, lw=0); ax.plot(x.year, x.RR, color=MCOL[m], lw=1.2, ls='--' if m == 'Incidence' else '-', label={'Case fatality': 'Deaths per episode'}.get(m, m))
    ax.set_yscale('log'); ax.set_xlim(1990, 2023); ax.set_xticks([1990, 2000, 2010, 2020]); ax.minorticks_off()
    yt = [1, 2, 5, 10, 20, 50, 100, 200]; lo_, hi_ = ax.get_ylim(); ax.set_ylim(max(lo_, 0.8), hi_); ax.set_yticks([t for t in yt if max(lo_, 0.8) <= t <= hi_]); ax.set_yticklabels([str(t) for t in yt if max(lo_, 0.8) <= t <= hi_])
    if k == 0: ax.set_ylabel('Rate ratio, low / high SDI')
    if k == 2: fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0.855, 0.80), fontsize=6, title='Panels C–E', title_fontsize=6)
    ax.set_title(dz, loc='left', color=DZC[CC[dz]], fontsize=6.5); label(ax, 'CDE'[k], x=-0.24)
fig.text(0.08, 0.475, 'Rate ratios with groups fixed by mean SDI in 2011–2023 (95% intervals)', fontsize=7)
save_figure(fig, OUTS / 'S1_Fig_group_rates_and_late_groups')
# ---------- S2 Fig (copy)
for ext in ('jpg', 'pdf'): shutil.copy(A / 'supp_figures_20261010' / f'S1_Fig_persistence_scatter.{ext}', OUTS / f'S2_Fig_persistence_scatter.{ext}')
# ---------- S3 Fig
fig = plt.figure(figsize=(180 * MM, 60 * MM)); gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.5, left=0.09, right=0.985, top=0.86, bottom=0.2)
D['key'] = D.cause_id.map({302: 'Diar', 322: 'LRI', 332: 'Men'}) + D.measure_id.map({1: ' death', 6: ' inc.'})
ORDER = ['Diar death', 'LRI death', 'Men death', 'Diar inc.', 'LRI inc.', 'Men inc.']
late = D[D.period == L0].pivot(index='location_id', columns='key', values='residual')[ORDER]
ax = fig.add_subplot(gs[0, 0]); Cm = np.array([[spearman(late[a].values, late[b].values)[0] for b in ORDER] for a in ORDER]); np.fill_diagonal(Cm, np.nan)
im = ax.imshow(Cm, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(6):
    for k in range(6):
        if i != k: ax.text(k, i, f'{Cm[i, k]:.2f}', ha='center', va='center', fontsize=4.8, color='white' if abs(Cm[i, k]) > 0.6 else 'black')
ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(ORDER, rotation=45, ha='right', rotation_mode='anchor', fontsize=5.5); ax.set_yticklabels(ORDER, fontsize=5.5)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.tick_params(length=0); label(ax, 'A', x=-0.42, y=1.06)
Ld = D[D.period == L0]
for k, c in enumerate([302, 322, 332]):
    ax = fig.add_subplot(gs[0, k + 1]); x = Ld[Ld.cause_id == c].pivot(index='location', columns='measure_id', values=['residual', 'category'])
    xi, yd = x[('residual', 6)].astype(float), x[('residual', 1)].astype(float)
    both = (x[('category', 6)] == '稳健高尾') & (x[('category', 1)] == '稳健高尾'); bothl = (x[('category', 6)] == '稳健低尾') & (x[('category', 1)] == '稳健低尾')
    ax.axhline(0, color=GREY, lw=0.5, zorder=0); ax.axvline(0, color=GREY, lw=0.5, zorder=0)
    ax.scatter(xi[~(both | bothl)], yd[~(both | bothl)], s=4, color=MID, alpha=0.6, lw=0, label='Other')
    ax.scatter(xi[both], yd[both], s=6, color=HIGH, lw=0, label='High tail, both'); ax.scatter(xi[bothl], yd[bothl], s=6, color=LOW, lw=0, label='Low tail, both')
    if c == 302:
        for nm, off in [('Australia', (4, -6)), ('Greenland', (4, 4))]:
            ax.scatter(xi[nm], yd[nm], s=12, facecolor='none', edgecolor='black', lw=0.6); ax.annotate(nm, (xi[nm], yd[nm]), xytext=off, textcoords='offset points', fontsize=5.5)
    lim = max(abs(xi).max(), abs(yd).max()) * 1.05; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Incidence residual'); ax.set_ylabel('Mortality residual' if k == 0 else '')
    ax.set_title(f'{DZ[c].replace(" disease", "")} (ρ={spearman(xi.values, yd.values)[0]:.2f})', loc='left', fontsize=6.5, color=DZC[c])
    if k == 2: ax.legend(loc='lower right', fontsize=5, handletextpad=0.1, borderaxespad=0.1)
    label(ax, 'BCD'[k], x=-0.42, y=1.06)
save_figure(fig, OUTS / 'S3_Fig_cross_outcome_correlation')
# ---------- S4 Fig
TD = pd.read_csv(R / 'PROFILE_TRAJECTORY_DOMAINS.csv')
GR = [('Stayed in high tail', HIGH, 'o'), ('Left high tail', '#F4A582', 'o'), ('Entered high tail', '#92C5DE', 's'), ('Stayed in low tail', LOW, 's')]
DN = ['Basic services', 'Immunisation', 'Nutrition', 'Health system', 'Air pollution']; BANDS = ['Low and low-middle', 'Middle and high-middle', 'High']
fig = plt.figure(figsize=(180 * MM, 72 * MM)); gs = fig.add_gridspec(1, 3, wspace=0.12, left=0.13, right=0.985, top=0.87, bottom=0.25)
for bi, b in enumerate(BANDS):
    ax = fig.add_subplot(gs[0, bi]); y0 = np.arange(len(DN))[::-1]
    for gi, (g, col, mk) in enumerate(GR):
        for yi, dn in zip(y0, DN):
            r = TD[(TD.band == b) & (TD.group == g) & (TD.domain == dn)]
            if r.empty: continue
            r = r.iloc[0]; yy = yi + 0.3 - gi * 0.2
            ax.annotate('', xy=(r.pct_adv_late, yy), xytext=(r.pct_adv_mid, yy), arrowprops=dict(arrowstyle='->', color=col, lw=0.9, shrinkA=0, shrinkB=0))
            ax.scatter(r.pct_adv_mid, yy, s=6, color=col, marker=mk, lw=0)
    for yi in y0[:-1]: ax.axhline(yi - 0.5, color='#E0E0E0', lw=0.5)
    ax.set_yticks(y0); ax.set_yticklabels(DN if bi == 0 else []); ax.set_ylim(-0.55, len(DN) - 0.45); ax.set_xlim(0, 100)
    ax.set_xlabel('% with adverse departure'); ax.set_title(f'{b} SDI (2001–2010)', loc='left', fontsize=6.5); label(ax, 'ABC'[bi], x=-0.08 if bi else -0.4)
for g, col, mk in GR: ax.plot([], [], color=col, lw=1.2, marker=mk, ms=3, label=g)
fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=6, bbox_to_anchor=(0.55, 0.0))
save_figure(fig, OUTS / 'S4_Fig_profile_trajectories_by_stratum')
# ---------- S5-S7 Fig (copies of existing figures)
for src_, dst in [(A / 'main_figures_20261010' / 'Fig4_catchup_and_relative_position', 'S5_Fig_gap_narrowing_and_relative_position'),
                  (A / 'supp_figures_20261010' / 'S6_Fig_case_trajectories', 'S6_Fig_case_trajectories'),
                  (A / 'supp_figures_20261010' / 'S5_Fig_PM25_LRI_incidence_high_SDI', 'S7_Fig_PM25_LRI_incidence_high_SDI')]:
    for ext in ('jpg', 'pdf'): shutil.copy(f'{src_}.{ext}', OUTS / f'{dst}.{ext}')
print('ok')
