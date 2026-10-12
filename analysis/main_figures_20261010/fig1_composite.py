"""Figure 1（组合图）：a 低、高SDI组死亡率；b 三种分组方式下的死亡率比；c 2023年分组方式改变带来的构成效应（LRI最大）；
d 参照模型示例（LRI发病，2011—2023年）。数据：PANEL、SDI_PERIOD_TIERS、figure2_reference 源数据、lri_gap_instability 结果。"""
from common import *
import numpy as np, pandas as pd
from paths import PANEL
d = pd.read_csv(PANEL); T = pd.read_csv(A / 'SDI_PERIOD_TIERS.csv')
from repro_common import period_of
d['period'] = d.year.map(period_of)
E = T[T.period == '1990–2000'].set_index('location_id').tier; L = T[T.period == '2011–2023'].set_index('location_id').tier
d['tE'] = d.location_id.map(E); d['tL'] = d.location_id.map(L); d = d.merge(T[['location_id', 'period', 'tier']].rename(columns={'tier': 'tV'}), on=['location_id', 'period'])
dd = d[d.measure_id == 1]
def rates(key):
    g = dd.groupby(['cause_id', 'year', key])[['gbd_number', 'under5_population']].sum(); g = 1e5 * g.gbd_number / g.under5_population
    return g.unstack(key)
R = {k: rates(k) for k in ['tE', 'tL', 'tV']}
src = []
fig = plt.figure(figsize=(180 * MM, 150 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], width_ratios=[1, 1, 1], hspace=0.42, wspace=0.42, left=0.085, right=0.98, top=0.95, bottom=0.08)
yrs = np.arange(1990, 2024)
ax = fig.add_subplot(gs[0, 0])
for c in [302, 322, 332]:
    g = R['tE'].loc[c]; ax.plot(yrs, g['低SDI'], color=DZC[c], lw=1.2); ax.plot(yrs, g['高SDI'], color=DZC[c], lw=1.2, ls='--')
    ax.text(2024, g['低SDI'].iloc[-1], DZ[c].replace(' disease', ''), color=DZC[c], fontsize=6, va='center')
    for y in yrs: src.append(dict(panel='a', cause=DZ[c], year=y, low=g.loc[y, '低SDI'], high=g.loc[y, '高SDI']))
ax.set_yscale('log'); ax.set_yticks([0.5, 1, 10, 100]); ax.set_yticklabels(['0.5', '1', '10', '100']); ax.minorticks_off(); ax.set_ylabel('Deaths per 100 000 children <5'); ax.set_xlim(1990, 2031); ax.set_xticks([1990, 2000, 2010, 2020])
ax.plot([], [], color='k', lw=1, label='Low SDI'); ax.plot([], [], color='k', lw=1, ls='--', label='High SDI'); ax.legend(loc='lower left', fontsize=6)
ax.set_title('Death rates, fixed SDI groups', loc='left'); label(ax, 'A', x=-0.28)
ax = fig.add_subplot(gs[0, 1])
for c in [302, 322, 332]:
    for k, ls, lw in [('tE', '-', 1.3), ('tL', ':', 1.0), ('tV', '--', 1.0)]:
        g = R[k].loc[c]; rr = g['低SDI'] / g['高SDI']; ax.plot(yrs, rr, color=DZC[c], ls=ls, lw=lw)
        for y in yrs: src.append(dict(panel='b', cause=DZ[c], design=k, year=y, RR=rr.loc[y]))
ax.set_yscale('log'); ax.set_yticks([10, 20, 50, 100, 200]); ax.set_yticklabels(['10', '20', '50', '100', '200']); ax.minorticks_off(); ax.set_ylabel('Death rate ratio, low / high SDI'); ax.set_xticks([1990, 2000, 2010, 2020])
for k, ls, lab in [('tE', '-', 'Fixed, 1990–2000 membership'), ('tL', ':', 'Fixed, 2011–2023 membership'), ('tV', '--', 'Reassigned each period')]:
    ax.plot([], [], color='k', ls=ls, lw=1, label=lab)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=5.8, ncol=3, handlelength=2.2, columnspacing=1.0)
for c, yv in [(302, None), (322, None), (332, None)]:
    g = R['tE'].loc[c]; ax.text(2023.5, (g['低SDI'] / g['高SDI']).iloc[-1], DZ[c].replace(' disease', ''), color=DZC[c], fontsize=6, va='center')
ax.set_xlim(1990, 2032)
ax.set_title('Relative gap by grouping design', loc='left'); label(ax, 'B', x=-0.28)
# c 构成效应
C = pd.read_csv(A / 'lri_gap_instability_20261010' / 'COMPOSITION_2023.csv').set_index('dz')
ax = fig.add_subplot(gs[0, 2]); x = np.arange(3); w = 0.26; order = [('腹泻病', 302), ('LRI', 322), ('脑膜炎', 332)]
vals = [[C.loc[k, 'dlog_low'] for k, _ in order], [-C.loc[k, 'dlog_high'] for k, _ in order], [C.loc[k, 'dlogRR'] for k, _ in order]]
labs = ['Low-SDI group change', 'High-SDI group change (−)', 'Net change in log RR']; cols = ['#9E9E9E', '#4D4D4D', 'black']
for j in range(3):
    b = ax.bar(x + (j - 1) * w, vals[j], w, color=cols[j] if j < 2 else [DZC[c] for _, c in order], label=labs[j] if j < 2 else None, edgecolor='white', lw=0.4)
ax.axhline(0, color='black', lw=0.6); ax.set_xticks(x); ax.set_xticklabels(['Diarrhoeal', 'LRI', 'Meningitis'])
ax.set_ylabel('Δ log rate ratio, 2023\n(reassigned minus fixed)'); from matplotlib.patches import Patch
h, l = ax.get_legend_handles_labels(); h.append(Patch(facecolor='white', edgecolor='black', lw=0.6)); l.append('Net change (coloured by disease)')
ax.legend(h, l, loc='upper right', fontsize=5.6, handlelength=1); ax.set_ylim(-1.1, 1.05)
for i, (k, c) in enumerate(order):
    ax.text(i + w, vals[2][i] + (0.04 if vals[2][i] > 0 else -0.04), f'{np.exp(vals[2][i]):.2f}×', ha='center', va='bottom' if vals[2][i] > 0 else 'top', fontsize=5.8)
ax.set_title('Why grouping changes the LRI gap', loc='left'); label(ax, 'C', x=-0.3)
# d 参照示例
P = pd.read_csv(A / 'figure2_reference_20261010' / 'Figure2_reference_LRI_incidence_SOURCE_POINTS.csv'); Cv = pd.read_csv(A / 'figure2_reference_20261010' / 'Figure2_reference_LRI_incidence_SOURCE_CURVES.csv')
p = P[P.period == '2011–2023']; cv = Cv[Cv.period == '2011–2023']
g2 = gs[1, :].subgridspec(1, 2, width_ratios=[2.05, 0.95], wspace=0.48); ax = fig.add_subplot(g2[0, 0])
ax.fill_between(cv.sdi, cv.ref_lower, cv.ref_upper, color='#D9D9D9', lw=0); ax.plot(cv.sdi, cv.ref_rate, color='black', lw=1.2, label='SDI–year reference (95% CI)')
COL = {'稳健高尾': HIGH, '稳健低尾': LOW, '明确偏离但居中': MID, '参照不确定': 'white'}
for cat, lab in [('参照不确定', 'Reference-uncertain'), ('明确偏离但居中', 'Resolved, central'), ('稳健低尾', 'Low tail'), ('稳健高尾', 'High tail')]:
    q = p[p.category == cat]; ax.scatter(q.sdi, q.obs_geomean_rate, s=10, facecolor=COL[cat], edgecolor=MID if cat == '参照不确定' else 'none', lw=0.5, label=lab, zorder=3)
for nm, lab, off in [('Singapore', 'Singapore (R=2.34)', (-8, -12)), ('Poland', 'Poland (R=2.87)', (6, 8)), ('Australia', 'Australia (R=0.34)', (-8, -14)), ('Bangladesh', 'Bangladesh (R=1.17)', (-10, 22)), ('Viet Nam', 'Viet Nam (R=1.07)', (-25, -28))]:
    q = p[p.location == nm].iloc[0]
    ref = np.interp(q.sdi, cv.sdi, cv.ref_rate)
    ax.annotate(lab, (q.sdi, q.obs_geomean_rate), xytext=off, textcoords='offset points', fontsize=6, ha='left' if off[0] > 0 else 'right', va='center',
                arrowprops=dict(arrowstyle='-', lw=0.4, color='#555555'))
for cut in [0.5315638185, 0.6304845024, 0.6766624947, 0.7188265995]: ax.axvline(cut, color='#BBBBBB', lw=0.5, ls=':', zorder=0)
ax.set_yscale('log'); ax.set_yticks([1000, 3000, 10000]); ax.set_yticklabels(['1 000', '3 000', '10 000']); ax.minorticks_off(); ax.set_xlabel('Mean SDI, 2011–2023'); ax.set_ylabel('LRI incidence per 100 000 children <5')
ax.set_title('Example: position relative to the SDI–year reference (LRI incidence, 2011–2023)', loc='left'); ax.legend(loc='lower left', fontsize=6, ncol=1)
fig.text(0.005, 0.485, 'D', fontsize=9, fontweight='bold')
# e 新进入高SDI的地点
S = pd.read_csv(A / 'lri_gap_instability_20261010' / 'ENTRANTS_LATE_POSITION.csv', header=[0, 1], index_col=0)
ax = fig.add_subplot(g2[0, 1]); outs = ['腹泻病死亡', 'LRI死亡', '脑膜炎死亡', 'LRI发病']; yl = ['Diarrhoea death', 'LRI death', 'Meningitis death', 'LRI incidence']
y = np.arange(len(outs))[::-1]
a = [int(S.loc[o, ('high_tail', '原高SDI')]) for o in outs]; b = [int(S.loc[o, ('high_tail', '新进入高SDI')]) for o in outs]
ax.barh(y + 0.18, a, 0.34, color='#BDBDBD', label='Already high SDI in 1990–2000 (n=32)'); ax.barh(y - 0.18, b, 0.34, color=HIGH, label='Entered high SDI by 2011–2023 (n=44)')
for yi, v in zip(y + 0.18, a): ax.text(v + 0.4, yi, f'{v}/32', va='center', fontsize=5.8)
for yi, v in zip(y - 0.18, b): ax.text(v + 0.4, yi, f'{v}/44', va='center', fontsize=5.8)
ax.set_yticks(y); ax.set_yticklabels(yl); ax.set_xlim(0, 25); ax.set_xticks([0, 5, 10, 15, 20, 25]); ax.set_xlabel('Locations in high tail, 2011–2023')
ax.legend(loc='upper center', bbox_to_anchor=(0.35, -0.2), fontsize=5.6, ncol=1); ax.set_title('Late high-SDI locations', loc='left')
fig.text(0.70, 0.485, 'E', fontsize=9, fontweight='bold')
pd.DataFrame(src).to_csv(Path(__file__).with_name('Fig1_SOURCE.csv'), index=False, encoding='utf-8-sig')
save_figure(fig, Path(__file__).with_name('Fig1_sdi_gaps_and_reference_example')); print('done')
