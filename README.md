# GBD 2023 Under-Five Childhood Disease Data and Analysis

Public data and reproducible analysis code for annual estimates of incidence and mortality from lower respiratory infections, diarrhoeal disease, and meningitis among children younger than 5 years in 204 countries and territories, 1990–2023.

## Contents

```text
data/        input panel (under5_disease_factor_annual_panel_1990_2023.csv) and DATA_DICTIONARY.md
analysis/    one folder per analysis step; each script writes its outputs into its own folder
run_all.py   runs every step in dependency order
```

All outputs are included, so tables and figures can be inspected without running anything.

## How to reproduce

Python 3.10 or later.

```bash
pip install -r requirements.txt
python run_all.py            # all steps
python run_all.py --quick    # skip the uncertainty-draw and bootstrap steps
python run_all.py --from 9 --to 9   # run a single step or a range
```

Random steps use fixed seeds, so repeated runs give identical results.

## Data sources

| Content | Source |
|---|---|
| Incidence, mortality, population, SDI, ambient and household PM2.5, second-hand smoke | GBD 2023, Institute for Health Metrics and Evaluation |
| Water, sanitation, clean cooking, DTP3, nutrition, health system and socioeconomic indicators | World Bank World Development Indicators |
| Hib3 and PCV3 coverage | WHO Global Health Observatory |
| Climate variables | ERA5 reanalysis, weighted by WorldPop under-5 population |

GBD data are used under the IHME Free-of-Charge Non-commercial User Agreement; follow the terms of each original source. Code is released under the MIT licence.
