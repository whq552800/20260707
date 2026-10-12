"""跨结局残差相关差异的地点整群自助法（2026-10-11 新增，回应“死亡相关高于发病相关”只比较点估计的问题）。
每次自助：按地点有放回抽取204个地点（重复抽中的地点赋新编号，保持整群结构），六个参照模型在同一抽样上共同重拟合，
再计算：各时期跨病种死亡残差相关均值 − 发病残差相关均值；晚期病内发病—死亡相关的两两差值。共 B 次，种子 20261011。
输出 S1_BOOT_CORR.csv（每次一行）与 S1_BOOT_CORR_SUMMARY.csv（点估计与百分位区间）。"""
import sys, numpy as np, pandas as pd
from engine import P3, HERE
from fit import PANEL, fit
from statlib import spearman
B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
PAIRS = [(302, 322), (302, 332), (322, 332)]
LOCS = np.sort(PANEL.location_id.unique())
GR = {k: g for k, g in PANEL.groupby(['cause_id', 'measure_id'])}

def rho(a, b):
    return spearman(a, b)[0]

def stats(draw):
    """draw：抽中的地点序列（可重复）；返回各统计量。"""
    W = {}
    for (c, m), g in GR.items():
        parts = []
        for new, loc in enumerate(draw):
            x = IDX[(c, m)].get(loc)
            if x is None: continue
            parts.append(x.assign(location_id=new))
        gg = pd.concat(parts, ignore_index=True).sort_values(['location_id', 'year']).reset_index(drop=True)
        o = fit(gg)
        W[(c, m)] = o.pivot(index='location_id', columns='period', values='residual')
    out = {}
    for p in P3:
        mo = [rho(W[(a, 1)][p].values, W[(b, 1)][p].reindex(W[(a, 1)].index).values) for a, b in PAIRS]
        ic = [rho(W[(a, 6)][p].values, W[(b, 6)][p].reindex(W[(a, 6)].index).values) for a, b in PAIRS]
        out[f'mort_minus_inc_{p}'] = np.mean(mo) - np.mean(ic); out[f'mort_mean_{p}'] = np.mean(mo); out[f'inc_mean_{p}'] = np.mean(ic)
    L = P3[2]; idm = {c: rho(W[(c, 6)][L].values, W[(c, 1)][L].reindex(W[(c, 6)].index).values) for c in (302, 322, 332)}
    for c, v in idm.items(): out[f'incdeath_{c}'] = v
    out['incdeath_LRI_minus_diar'] = idm[322] - idm[302]; out['incdeath_men_minus_LRI'] = idm[332] - idm[322]
    return out

IDX = {k: {l: x for l, x in g.groupby('location_id')} for k, g in GR.items()}
if __name__ == '__main__':
    main = stats(LOCS); print({k: round(v, 3) for k, v in main.items()}, flush=True)
    rng = np.random.default_rng(20261011); rows = []
    for i in range(B):
        rows.append({'draw': i, **stats(rng.choice(LOCS, len(LOCS), replace=True))})
        if i % 50 == 0: print(i, flush=True)
    D = pd.DataFrame(rows); D.to_csv(HERE / 'S1_BOOT_CORR.csv', index=False, encoding='utf-8-sig')
    S = pd.DataFrame({'estimate': pd.Series(main), 'lo95': D.drop(columns='draw').quantile(.025), 'hi95': D.drop(columns='draw').quantile(.975),
                      'pct_draws_gt0': (D.drop(columns='draw') > 0).mean() * 100})
    S.to_csv(HERE / 'S1_BOOT_CORR_SUMMARY.csv', encoding='utf-8-sig'); print(S.round(3).to_string())
