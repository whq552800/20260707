# Data dictionary: `under5_disease_factor_annual_panel_1990_2023.csv`

One row per location × disease × outcome × year (unique key `location_id`, `cause_id`, `measure_id`, `year`): 204 locations × 3 diseases × 2 outcomes × 34 years = 41 616 rows. Age group is fixed at `<5 years` and sex at `Both`. Contextual variables are matched by `location_id` × `year`, so each location–year value appears in the six disease–outcome rows of that year. Blank cells mean that no value was available; nothing was imputed, interpolated, extrapolated, or zero-filled.

## Identifiers

| Column | Meaning | Values or unit |
|---|---|---|
| `location_id`, `location` | GBD location identifier and name | 204 countries and territories |
| `cause_id`, `cause` | Disease | 302 diarrhoeal diseases; 322 lower respiratory infections; 332 meningitis |
| `measure_id`, `measure` | Outcome | 1 deaths; 6 incidence |
| `year` | Calendar year | 1990–2023 |
| `age_group`, `sex` | Fixed strata | `<5 years`, `Both` |
| `sdi` | Socio-demographic Index | 0–1 (GBD 2023) |
| `under5_population` | Population younger than 5 years | persons (GBD 2023) |

## Disease estimates (GBD 2023)

| Column | Meaning | Unit |
|---|---|---|
| `gbd_rate_per_100k`, `gbd_rate_lower_per_100k`, `gbd_rate_upper_per_100k` | Incidence or mortality rate with 95% uncertainty interval | per 100 000 |
| `gbd_number`, `gbd_number_lower`, `gbd_number_upper` | Number of cases or deaths with 95% uncertainty interval | persons |
| `rate_valid` | Rate finite and > 0 (records used in log-rate models) | True/False |
| `zero_rate_flag`, `number_valid` | Rate equal to 0; number finite and ≥ 0 | True/False |
| `location_holdout_oof_fold` | Fixed five-fold assignment of locations for the leave-location-out sensitivity analysis (all years of a location in one fold) | 0–4 |

## Contextual factors

| Column | Meaning | Unit | Source (indicator code) |
|---|---|---|---|
| `basic_water_pct` | At least basic drinking water | % of population | WDI (SH.H2O.BASW.ZS; WHO/UNICEF JMP) |
| `basic_sanitation_pct` | At least basic sanitation | % of population | WDI (SH.STA.BASS.ZS; JMP) |
| `clean_cooking_pct` | Access to clean cooking fuels and technologies | % of population | WDI (EG.CFT.ACCS.ZS) |
| `dtp3_pct` | DTP3 coverage | % of children aged 12–23 months | WDI (SH.IMM.IDPT; WUENIC) |
| `hib3_pct` | Hib3 coverage | % | WHO GHO (WHS4_129) |
| `pcv3_pct` | PCV3 coverage | % | WHO GHO (PCV3) |
| `stunting_u5_pct` | Stunting, children younger than 5 years | % | WDI (SH.STA.STNT.ZS; JME) |
| `low_birthweight_pct` | Births < 2500 g | % of births | UNICEF–WHO estimates via WDI |
| `undernourishment_pct` | Prevalence of undernourishment | % of population | WDI (SN.ITK.DEFC.ZS) |
| `uhc_service_index` | UHC service coverage index | 0–100 | WDI (SH_UHC_SCI) |
| `health_expenditure_pc_current_usd` | Current health expenditure per capita | current US$ | WDI (SH.XPD.CHEX.PC.CD) |
| `hospital_beds_per1000` | Hospital beds | per 1000 | WDI (SH.MED.BEDS.ZS) |
| `nurses_midwives_per1000` | Nurses and midwives | per 1000 | WDI (SH.MED.NUMW.P3) |
| `physicians_per1000` | Physicians (not analysed: one year per location, differing between locations) | per 1000 | WDI (SH.MED.PHYS.ZS) |
| `income_ppp_pc` | GDP per capita, PPP | 2021 international $ | WDI (NY.GDP.PCAP.PP.KD) |
| `secondary_enrolment_pct` | Secondary school enrolment, gross | % (can exceed 100) | WDI (SE.SEC.ENRR) |
| `fertility_births_per_woman` | Total fertility rate | births per woman | WDI (SP.DYN.TFRT.IN) |
| `secondhand_smoke_sev`, `_lower`, `_upper` | Second-hand smoke summary exposure value, children younger than 5 years | SEV, 0–100 | GBD 2023 |
| `wdi_pm25_annual_mean_ug_m3` | Ambient PM2.5, WDI series (replaced by the GBD series in the analysis) | μg/m³ | WDI (EN.ATM.PM25.MC.M3) |

## Environment and geography

| Column | Meaning | Unit | Source |
|---|---|---|---|
| `gbd_ambient_pm25_ug_m3` | Ambient PM2.5 (used in the analysis) | μg/m³ | GBD 2023 |
| `gbd_household_pm25_ug_m3` | Household PM2.5 | μg/m³ | GBD 2023 |
| `era5_tavg_C_pw_u5` | Mean 2 m temperature | °C | ERA5, weighted by WorldPop under-5 population |
| `era5_hot_days_tmax_gt30_pw_u5` | Days with maximum temperature > 30 °C | days/year | as above |
| `era5_cold_days_tmin_lt0_pw_u5` | Days with minimum temperature < 0 °C | days/year | as above |
| `era5_ah_g_m3_pw_u5` | Absolute humidity | g/m³ | as above (from temperature and dew point) |
| `era5_rh_pct_pw_u5` | Relative humidity | % | as above |
| `era5_wet_days_pr_gt1mm_pw_u5` | Days with precipitation > 1 mm | days/year | as above |
| `env_pop_u5`, `env_pop_total` | Populations used for weighting | persons | WorldPop |
| `region_proxy` | United Nations M49 geographical subregion | category | United Nations M49 |
| `continent_proxy` | Continent | category | United Nations M49 |

ERA5 variables are missing for Tokelau. The second-hand smoke columns are exposure scores, not disease rates.
