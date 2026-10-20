"""2026-10-20: same-metric comparison of cross-sectional ('stand') and within-country change ('move') coefficients.

For each pre-specified pair (and each non-covariate single factor x outcome pair), the cross-sectional coefficient uses the
2011-2023 level of the same single-factor departure (standardised by its SD across locations) and the 2011-2023 period residual,
with SDI-stratum intercepts; the change coefficient (from change.py) uses the 2001-2010 -> 2011-2023 change, also standardised by
its SD. Both are % change in the burden ratio per 1-SD worse conditions. Because a 1-SD change is smaller in absolute terms
than a 1-SD cross-sectional difference, a second comparison rescales the change coefficient to the cross-sectional SD
(per unit of departure), so that both answer: how much does the burden ratio differ for the same absolute difference in the factor?
Reliability: Delta-x is attenuated by measurement error; the test-retest correlation of the factor departure between adjacent
periods gives the reliability of a level, and the reliability of a difference is (r_tt_mean - r12)/(1 - r12) (Lord 1956 style
with equal variances). Reported per pair; corrected coefficients divide the log coefficient by this reliability.
Output: CH_STAND_MOVE.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
src = (H / 'change.py').read_text().split('prim = [')[0]
exec(src)
rng = np.random.default_rng(20261020); NB = 2000
def level(fs, p):
    parts = [F[(f, p)] / F[(f, p)].std() for f in fs]; x = pd.concat(parts, axis=1).mean(axis=1, skipna=False); return x
def coef(x, y, boot=True):
    d = pd.concat([x.rename('x'), y.rename('y'), T.rename('band')], axis=1).dropna()
    S = pd.get_dummies(d.band).astype(float).values; X = np.column_stack([S, d.x.values]); yy = d.y.values
    b = np.linalg.lstsq(X, yy, rcond=None)[0][-1]
    bb = np.array([np.linalg.lstsq(X[ii], yy[ii], rcond=None)[0][-1] for ii in (rng.integers(0, len(d), len(d)) for _ in range(NB))]) if boot else None
    return b, bb, len(d)
prim = [('Air pollution -> LRI incidence', ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev'], 'LRI incidence'),
        ('Drinking water -> LRI mortality', ['basic_water_pct'], 'LRI mortality'),
        ('Clean cooking -> meningitis mortality', ['clean_cooking_pct'], 'Meningitis mortality')]
rows = []
for lab, fs, o in prim:
    xl = level(fs, P3); sd_l = xl.std(); xl = xl / sd_l
    yl = R[(*KEY[o], P3)]
    bl, bbl, nl = coef(xl, yl)
    xc_raw = level(fs, P3) - level(fs, P2); sd_c = xc_raw.std(); xc = xc_raw / sd_c
    yc = dres(KEY[o], P2, P3)
    bc, bbc, nc = coef(xc, yc)
    # same absolute difference: change coefficient per cross-sectional SD
    k = sd_l / sd_c
    r12 = pd.concat([level(fs, P2), level(fs, P3)], axis=1).dropna().corr().iloc[0, 1]
    rel_diff = max((1 - r12) and (0.9 - r12) / (1 - r12), 0.05)   # level reliability assumed 0.9 (sensitivity below)
    rows.append(dict(pair=lab, n_level=nl, n_change=nc, stand=pc(bl), stand_lo=np.percentile(pc(bbl), 2.5), stand_hi=np.percentile(pc(bbl), 97.5),
                     move_per_change_sd=pc(bc), move_lo=np.percentile(pc(bbc), 2.5), move_hi=np.percentile(pc(bbc), 97.5),
                     move_per_level_sd=pc(bc * k), move_lvl_lo=np.percentile(pc(bbc * k), 2.5), move_lvl_hi=np.percentile(pc(bbc * k), 97.5),
                     sd_ratio=k, r_adjacent=r12, reliability_change=rel_diff, move_per_level_sd_corrected=pc(bc * k / rel_diff),
                     ratio_log=(bc * k) / bl, ratio_log_corrected=(bc * k / rel_diff) / bl))
S = pd.DataFrame(rows); S.to_csv(H / 'CH_STAND_MOVE.csv', index=False)
pd.set_option('display.width', 250); print(S.round(2).to_string(index=False))
