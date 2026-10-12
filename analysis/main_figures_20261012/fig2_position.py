"""Fig 2 (2026-10-12, revised): position relative to development persisted, and persistent tails were specific to disease and outcome.
A: example of the SDI-year reference (LRI incidence, 2011-2023). B: Spearman correlation between 1990-2000 and 2011-2023 residuals with
95% location-bootstrap intervals. C: position in 2011-2023 of records in the high or low tail in 1990-2000. D: number of outcomes per location persistently in the high or low tail.
E: correlation between incidence and mortality residuals of the same disease, with the percentage of locations with opposite directions
in 2011-2023 (Wilson interval). Data: REPRO_DISEASE_PERIOD.csv, figure2_reference source data, gap_story_20261012/PERSISTENCE_INTERVALS.csv,
s1_sensitivity_20261010/S1_BOOT_CORR_SUMMARY.csv. Outputs: Fig2_position_persistence.{jpg,pdf}, Fig2_SOURCE.csv."""
from common import *
import numpy as np, pandas as pd
from statlib import spearman
D = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_DISEASE_PERIOD.csv')
D['key'] = D.cause_id.map({302: 'Diar', 322: 'LRI', 332: 'Men'}) + D.measure_id.map({1: ' death', 6: ' inc.'})
PI = pd.read_csv(A / 'gap_story_20261012' / 'PERSISTENCE_INTERVALS.csv'); BC = pd.read_csv(A / 's1_sensitivity_20261010' / 'S1_BOOT_CORR_SUMMARY.csv', index_col=0)
E0, L0 = PER[0], PER[2]; src = []
fig = plt.figure(figsize=(180 * MM, 140 * MM))
gt = fig.add_gridspec(1, 2, width_ratios=[2.0, 1], wspace=0.55, left=0.075, right=0.975, top=0.95, bottom=0.57)
gb = fig.add_gridspec(1, 3, wspace=0.62, left=0.10, right=0.94, top=0.42, bottom=0.17)
# A reference example
P = pd.read_csv(A / 'figure2_reference_20261010' / 'Figure2_reference_LRI_incidence_SOURCE_POINTS.csv'); Cv = pd.read_csv(A / 'figure2_reference_20261010' / 'Figure2_reference_LRI_incidence_SOURCE_CURVES.csv')
p = P[P.period == L0]; cv = Cv[Cv.period == L0]
ax = fig.add_subplot(gt[0, 0])
ax.fill_between(cv.sdi, cv.ref_lower, cv.ref_upper, color='#D9D9D9', lw=0); ax.plot(cv.sdi, cv.ref_rate, color='black', lw=1.2, label='SDI–year reference (95% CI)')
COL = {'稳健高尾': HIGH, '稳健低尾': LOW, '明确偏离但居中': MID, '参照不确定': 'white'}
for cat, lab in [('参照不确定', 'Reference-uncertain'), ('明确偏离但居中', 'Resolved, central'), ('稳健低尾', 'Low tail'), ('稳健高尾', 'High tail')]:
    q = p[p.category == cat]; ax.scatter(q.sdi, q.obs_geomean_rate, s=10, facecolor=COL[cat], edgecolor=MID if cat == '参照不确定' else 'none', lw=0.5, label=lab, zorder=3)
for nm, lab, off in [('Singapore', 'Singapore (R=2.34)', (-8, -12)), ('Poland', 'Poland (R=2.87)', (6, 8)), ('Australia', 'Australia (R=0.34)', (-8, -14)), ('Bangladesh', 'Bangladesh (R=1.17)', (-10, 22)), ('Viet Nam', 'Viet Nam (R=1.07)', (-25, -28))]:
    q = p[p.location == nm].iloc[0]
    ax.annotate(lab, (q.sdi, q.obs_geomean_rate), xytext=off, textcoords='offset points', fontsize=6, ha='left' if off[0] > 0 else 'right', va='center', arrowprops=dict(arrowstyle='-', lw=0.4, color='#555555'))
for cut in [0.5315638185, 0.6304845024, 0.6766624947, 0.7188265995]: ax.axvline(cut, color='#BBBBBB', lw=0.5, ls=':', zorder=0)
ax.set_yscale('log'); ax.set_yticks([1000, 3000, 10000]); ax.set_yticklabels(['1 000', '3 000', '10 000']); ax.minorticks_off(); ax.set_xlabel('Mean SDI, 2011–2023'); ax.set_ylabel('LRI incidence per 100 000 children <5')
ax.set_title('Position relative to development: LRI incidence, 2011–2023', loc='left'); ax.legend(loc='lower left', fontsize=6)
label(ax, 'A', x=-0.11)
# B rank stability
ax = fig.add_subplot(gt[0, 1])
OUT = [(302, 1, 'Diarrhoeal mortality'), (322, 1, 'LRI mortality'), (332, 1, 'Meningitis mortality'), (302, 6, 'Diarrhoeal incidence'), (322, 6, 'LRI incidence'), (332, 6, 'Meningitis incidence')]
yy = np.arange(len(OUT))[::-1]
for yi, (c, m, lab) in zip(yy, OUT):
    r = PI[(PI.item == 'rank_corr') & (PI.outcome == lab)].iloc[0]; src.append(dict(panel='B', outcome=lab, rho=r.est, lo=r.lo, hi=r.hi))
    ax.plot([r.lo, r.hi], [yi, yi], color=DZC[c], lw=1)
    ax.scatter(r.est, yi, s=18, color=DZC[c], marker='o' if m == 1 else 's', zorder=2, lw=0)
    ax.text(r.hi + 0.015, yi, f'{r.est:.2f}', va='center', fontsize=5.8)
ax.set_yticks(yy); ax.set_yticklabels([o[2] for o in OUT], fontsize=6); ax.set_xlim(0.5, 1.0); ax.set_ylim(-0.6, len(OUT) - 0.4)
ax.set_xlabel('Spearman ρ, 1990–2000 vs 2011–2023'); ax.set_title('Rank stability', loc='left')
label(ax, 'B', x=-0.95)
# C destinations of early tails
ax = fig.add_subplot(gb[0, 0])
CAT = {'稳健高尾': HIGH, '明确偏离但居中': MID, '参照不确定': '#DDDDDD', '稳健低尾': LOW}
LAB = {'稳健高尾': 'High tail', '明确偏离但居中': 'Resolved, central', '参照不确定': 'Reference-uncertain', '稳健低尾': 'Low tail'}
j = D[D.period == E0].set_index(['location_id', 'cause_id', 'measure_id']).category.to_frame('e').join(D[D.period == L0].set_index(['location_id', 'cause_id', 'measure_id']).category.rename('l'))
for xi, start in enumerate(['稳健高尾', '稳健低尾']):
    s = j[j.e == start].l.value_counts().reindex(list(CAT)).fillna(0); n = int(s.sum()); bottom = 0
    src.append(dict(panel='C', start=LAB[start], n=n, **{LAB[k]: int(v) for k, v in s.items()}))
    for cat in CAT:
        v = 100 * s[cat] / n
        ax.bar(xi, v, bottom=bottom, color=CAT[cat], width=0.62, edgecolor='white', lw=0.4, label=LAB[cat] if xi == 0 else None)
        if v >= 7: ax.text(xi, bottom + v / 2, f'{v:.0f}', ha='center', va='center', fontsize=6, color='white' if cat in ('稳健高尾', '稳健低尾') else 'black')
        bottom += v
ax.set_xticks([0, 1]); ax.set_xticklabels(['High tail', 'Low tail']); ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 100)
ax.set_xlabel(f'Position in 1990–2000 (n={n} each)'); ax.set_ylabel('Position in 2011–2023 (%)'); ax.set_title('Where early tails ended up', loc='left')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=5.6, handlelength=0.9, handletextpad=0.3, labelspacing=0.3, columnspacing=0.8, borderaxespad=0)
label(ax, 'C', x=-0.4)
# D persistent tails are specific to outcomes: number of outcomes (of six) persistently in a tail per location
ax = fig.add_subplot(gb[0, 1])
Cc = D.pivot_table(index=['location_id', 'cause_id', 'measure_id'], columns='period', values='category', aggfunc='first')
nh = (Cc == '稳健高尾').all(axis=1).groupby(level=0).sum(); nl = (Cc == '稳健低尾').all(axis=1).groupby(level=0).sum()
k = np.arange(1, 7); hh = [(nh == i).sum() for i in k]; ll = [(nl == i).sum() for i in k]
ax.bar(k - 0.19, hh, 0.36, color=HIGH, label=f'Persistent high tail ({int((nh > 0).sum())} locations)')
ax.bar(k + 0.19, ll, 0.36, color=LOW, label=f'Persistent low tail ({int((nl > 0).sum())} locations)')
for x_, v in zip(k - 0.19, hh): ax.text(x_, v + 0.6, str(v), ha='center', fontsize=5.6)
for x_, v in zip(k + 0.19, ll): ax.text(x_, v + 0.6, str(v), ha='center', fontsize=5.6)
both = int(((nh > 0) & (nl > 0)).sum())
ax.text(0.97, 0.62, f'{both} locations had both\npersistent high-tail and\npersistent low-tail outcomes', transform=ax.transAxes, ha='right', va='top', fontsize=5.6)
ax.set_xticks(k); ax.set_xlabel('Outcomes persistently in the tail (of 6)'); ax.set_ylabel('Locations'); ax.set_ylim(0, 44)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=1, fontsize=5.8); ax.set_title('Persistent tails by number of outcomes', loc='left')
src += [dict(panel='D', n_outcomes=int(i), persistent_high=int(a), persistent_low=int(b)) for i, a, b in zip(k, hh, ll)] + [dict(panel='D', both_high_and_low=both)]
label(ax, 'D', x=-0.42)
# E incidence vs mortality, same disease
ax = fig.add_subplot(gb[0, 2])
for c, dz in [(302, 'Diarrhoeal disease'), (322, 'LRI'), (332, 'Meningitis')]:
    ys = []
    for p_ in PER:
        w = D[(D.period == p_) & (D.cause_id == c)].pivot(index='location_id', columns='measure_id', values='residual'); ys.append(spearman(w[6].values, w[1].values)[0])
    ax.plot(range(3), ys, marker='o', ms=3, lw=1, color=DZC[c])
    r = PI[(PI.item == 'opposite_direction') & (PI.outcome == dz)].iloc[0]
    src.append(dict(panel='E', disease=dz, rho_by_period=ys, opposite_pct_late=r.est, lo=r.lo, hi=r.hi, n=r.n))
    ax.text(2.12, ys[-1], f'{dz.replace(" disease", "")}\n{r.est:.0f}% opposite\n({r.lo:.0f}–{r.hi:.0f})', color=DZC[c], fontsize=5.4, va='center', linespacing=0.95)
ax.set_xticks(range(3)); ax.set_xticklabels(['1990–\n2000', '2001–\n2010', '2011–\n2023']); ax.set_xlim(-0.2, 3.6); ax.set_ylim(0, 1)
ax.set_ylabel('Incidence–mortality Spearman ρ'); ax.set_title('Incidence vs mortality', loc='left')
label(ax, 'E', x=-0.42)
pd.DataFrame(src).to_csv(Path(__file__).with_name('Fig2_SOURCE.csv'), index=False, encoding='utf-8-sig')
save_figure(fig, Path(__file__).with_name('Fig2_position_persistence')); print('ok')
