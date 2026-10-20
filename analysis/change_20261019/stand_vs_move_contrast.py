"""2026-10-20: as stand_vs_move.py, for the within-country disease contrast (outcome minus the country's mean of the same measure
across the three infections), both coefficients per cross-sectional SD of the factor departure. Output: CH_STAND_MOVE_CONTRAST.csv"""
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
exec((H / 'stand_vs_move.py').read_text().split("\nprim = [('Air")[0])
rng = np.random.default_rng(20261020)
prim = [('Air pollution -> LRI incidence', ['gbd_ambient_pm25_ug_m3', 'secondhand_smoke_sev'], 'LRI incidence'),
        ('Drinking water -> LRI mortality', ['basic_water_pct'], 'LRI mortality'),
        ('Clean cooking -> meningitis mortality', ['clean_cooking_pct'], 'Meningitis mortality')]
def lvl_contrast(o, p):
    k = KEY[o]; m = k[1]; return R[(*k, p)] - pd.concat([R[(c, m, p)] for c in (302, 322, 332)], axis=1).mean(axis=1)
rows = []
for lab, fs, o in prim:
    xl = level(fs, P3); sd_l = xl.std()
    bl, bbl, nl = coef(xl / sd_l, lvl_contrast(o, P3))
    xc = level(fs, P3) - level(fs, P2)
    bc, bbc, nc = coef(xc / sd_l, outcome_change(KEY[o], P2, P3, True))      # change per cross-sectional SD
    bt, bbt, _ = coef(xc / sd_l, outcome_change(KEY[o], P2, P3, False))
    q = lambda v: (pc(v), np.percentile(pc(v), 2.5), np.percentile(pc(v), 97.5))
    rows.append(dict(pair=lab, stand_contrast=pc(bl), sc_lo=np.percentile(pc(bbl), 2.5), sc_hi=np.percentile(pc(bbl), 97.5),
                     move_contrast=pc(bc), mc_lo=np.percentile(pc(bbc), 2.5), mc_hi=np.percentile(pc(bbc), 97.5),
                     move_total=pc(bt), mt_lo=np.percentile(pc(bbt), 2.5), mt_hi=np.percentile(pc(bbt), 97.5),
                     diff_contrast_log=bc - bl, diff_lo=np.percentile(bbc - bbl, 2.5), diff_hi=np.percentile(bbc - bbl, 97.5)))
S = pd.DataFrame(rows); S.to_csv(H / 'CH_STAND_MOVE_CONTRAST.csv', index=False)
pd.set_option('display.width', 250); print(S.round(2).to_string(index=False))
