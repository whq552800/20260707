"""2026-10-20: 2023 excess deaths in persistent high-tail locations with an arithmetic-mean reference.
The main reference is exp(fitted mean log rate), a geometric mean (the typical peer). An arithmetic-mean reference multiplies it by
(a) Duan's smearing factor, the mean of exp(annual residual) over all locations in 2023, or (b) the ratio of total GBD deaths to total
geometric-mean expected deaths over all locations in 2023 (population-weighted). Point estimates. Output: EXCESS_ARITH.csv"""
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; A = H.parent
AF = pd.read_csv(A / 'independent_reproduction_20261009' / 'REPRO_ANNUAL_FIT.csv'); B = pd.read_csv(H / 'EXCESS_BY_LOCATION_2023.csv')
rows = []
for c, dz in [(302, 'Diarrhoeal disease'), (322, 'LRI'), (332, 'Meningitis')]:
    r = AF[(AF.cause_id == c) & (AF.measure_id == 1) & (AF.year == 2023)].residual_log
    fs = np.exp(r).mean(); a = B[B.disease == dz]; fw = a.gbd_number.sum() / a.expected.sum(); b = a[a.persistent_high]
    rows.append(dict(disease=dz, smearing_factor=fs, popweighted_factor=fw, excess_geometric=b.excess.sum(),
                     excess_smearing=(b.gbd_number - b.expected * fs).sum(), excess_popweighted=(b.gbd_number - b.expected * fw).sum()))
X = pd.DataFrame(rows); t = X.drop(columns='disease').sum(); t['disease'] = 'All three'; t['smearing_factor'] = np.nan; t['popweighted_factor'] = np.nan
X = pd.concat([X, t.to_frame().T]); X.to_csv(H / 'EXCESS_ARITH.csv', index=False); print(X.round(2).to_string(index=False))
