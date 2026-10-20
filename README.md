# GBD 2023 Under-Five Childhood Disease Data and Analysis

Public data and reproducible analysis code for annual estimates of incidence and mortality from lower respiratory infections, diarrhoeal disease, and meningitis among children younger than 5 years in 204 countries and territories, 1990–2023.

The analysis places the incidence and mortality of lower respiratory infections (LRI), diarrhoeal disease, and meningitis in children younger than 5 years against Socio-demographic Index (SDI)–year reference models, classifies each country's departure from its reference in three periods (1990–2000, 2001–2010, 2011–2023), and assesses 18 service, nutrition, exposure, and socioeconomic factors against their own development references. Every number, table, and figure in the manuscript and its supplements can be regenerated from the single input panel in `data/` with `run_all.py`.

## Repository layout

```text
.
├── README.md
├── LICENSE                  MIT licence for the code
├── requirements.txt
├── run_all.py               runs every step in dependency order
├── data/
│   ├── under5_disease_factor_annual_panel_1990_2023.csv   input panel (41 616 rows)
│   └── DATA_DICTIONARY.md                                 variables, units, and sources
└── analysis/                one folder per step; each script writes its outputs into its own folder
    ├── sdi_tiers.py                              SDI levels by period
    ├── independent_reproduction_20261009/        disease reference models and tail classes
    ├── sdi_gaps_official_tiers_20261010/         gaps between SDI groups
    ├── lri_gap_instability_20261010/             effect of group membership
    ├── substantive_findings_20261010/            regions; environmental residuals
    ├── reference_spec_sensitivity_20261010/      alternative reference forms
    ├── factor_residual_audit_20261010/           factor residual engine (factor_pipeline.py); data coverage
    ├── factor_profiles_confirmed_20261010/       factor statuses, contrasts, factor-rule sensitivity
    ├── residual_matrix_workbook_20261010/        location-period residual matrices
    ├── tier_period_tail_factors_20261010/        factor composition of tails by SDI stratum
    ├── results_revision_20261010/                global rates, persistent-tail domains, catch-up
    ├── figure2_reference_20261010/               reference-model figures
    ├── main_figures_20261010/                    R1d figures (partly reused as S Figs); Table 2
    ├── gap_story_20261012/                       gaps in incidence, case fatality and mortality; persistence intervals
    ├── excess_deaths_20261017/                   deaths in excess of the SDI–year reference (persistent high-tail locations)
    ├── robustness_20261017/                      location-clustered intervals; change in the spread of residuals
    ├── factor_models_20261017/                   multivariable models of burden departures on domain shortfalls
    ├── general_specific_20261019/                general vs disease-specific parts of departures
    ├── within_country_20261019/                  within-country contrasts between infections
    ├── change_20261019/                          within-country change in shortfalls and burden; same-scale comparison
    ├── typology_20261020/                        patterns of disadvantage by fixed rules
    ├── main_figures_20261012/                    Figs 1–2, S8–S9 Figs and S1–S7 Figs (current)
    ├── supp_figures_20261012/                    S1–S7 Figs (current)
    ├── s1_sensitivity_20261010/                  sensitivity analyses, uncertainty propagation, bootstrap
    ├── gbd_coupling_20261011/                    checks for coupling inside the GBD estimates
    └── country_profiles_20261011/                persistent-tail country profiles and literature coding
```

Folder names carry the date on which each step was finalised; they are kept so that file references inside the scripts stay unchanged.

## How to reproduce

Python 3.10 or later.

```bash
pip install -r requirements.txt
python run_all.py            # all 59 steps
python run_all.py --quick    # skips the 200-draw uncertainty propagation and the 500-resample bootstrap
python run_all.py --from 13 --to 13   # run a single step (or a range); outputs of earlier steps are already in the repository
```

Factor-rule variants are cached in `factor_profiles_confirmed_20261010/cache_S*.csv`; delete these files to recompute them. Random steps use fixed seeds (20261010 for the uncertainty draws; 20261011 for the correlation bootstrap and the resampling in `outcome_factor_tails.py`; 20261012 for the within-subregion permutations; 20261013 and 20261014 for the resampling in `outcome_factor_checks.py` and `part3_comparability.py`; 20261015 and 20261016 for the resampling in `envelope_check.py`; 20261017 for `covariate_status.py`; 20261018 for `cluster_bootstrap.py`; 20261012 for `gap_decomposition.py`, `persistence_intervals.py`, `profile_domains.py` and `profile_trajectory_domains.py`), so repeated runs give identical results. Each step was rerun from a clean copy of this repository, and the output tables matched the results reported in the manuscript.

All outputs of a run are already included in the repository, so the tables and figures can be inspected without running anything.

## Where each manuscript item comes from

| Manuscript item | Script | Output (in the script's folder) |
|---|---|---|
| Table 1; Fig 1; S1 Text Table E | `gap_story_20261012/gap_decomposition.py` | `GAP_SUMMARY.csv`, `GAP_ANNUAL.csv` |
| Excess deaths (Results; S3 Table) | `excess_deaths_20261017/excess_deaths.py` (seed 20261017) | `EXCESS_SUMMARY.csv`, `EXCESS_BY_LOCATION_2023.csv`, `EXCESS_NIGERIA_LOO.csv` |
| Fig 3B intervals; tail retention; residual spread (Results) | `robustness_20261017/clustered_intervals.py` (seed 20261017) | `PROFILE_DOMAINS_POOLED_CLUSTERED.csv`, `TAIL_RETENTION_CLUSTERED.csv`, `DISPERSION_CHANGE.csv` |
| Fig 2 (R5) | `general_specific_20261019/general_specific.py`; `main_figures_20261012/fig_structure.py` | `GS_SHARE.csv`, `GS_PERSIST.csv`, `GS_EXCESS_2023.csv`, `GS_FACTORS.csv`; `Fig_structure.{jpg,pdf}` |
| Fig 4 (R5) | `within_country_20261019/within_country.py`; `change_20261019/change.py`, `stand_vs_move.py`, `stand_vs_move_contrast.py`; `main_figures_20261012/fig_allocation_change.py` | `WC_DISEASE.csv`, `WC_FATALITY.csv`, `CH_PRIMARY.csv`, `CH_GRID.csv`, `CH_STAND_MOVE.csv`, `CH_STAND_MOVE_CONTRAST.csv`; `Fig_allocation_change.{jpg,pdf}` |
| Table 1 (R5) | `typology_20261020/typology.py` | `TYPOLOGY.csv`, `TYPOLOGY_EXAMPLES.csv` |
| Excess deaths, arithmetic-mean reference (S1 Text) | `excess_deaths_20261017/excess_arith.py` | `EXCESS_ARITH.csv` |
| Fig 3 (R5; Fig 2 in R4); factor models (Results; S1 Text) | `factor_models_20261017/factor_models.py` (2000 location bootstrap replicates, seed 20261017; slow); `main_figures_20261012/fig_factor_models.py` | `FM_COEF.csv`, `FM_CONTRAST.csv`, `FM_SHAPLEY.csv`, `FM_COEF_NONCOV.csv`, `FM_DEATHS.csv`, `FM_DATA.csv`; `Fig_factor_models.{jpg,pdf}` |
| S8 Fig (Fig 1 before R4) | `main_figures_20261012/fig1_gap.py` | `Fig1_gap_incidence_case_fatality.{jpg,pdf}`, `Fig1_SOURCE.csv` |
| Fig 1 (Fig 2 before R4; intervals from `gap_story_20261012/persistence_intervals.py`, `PERSISTENCE_INTERVALS.csv`) | `main_figures_20261012/fig2_position.py` | `Fig2_position_persistence.{jpg,pdf}`, `Fig2_SOURCE.csv` |
| S9 Fig (Fig 3 before R4); S1 Text tail-profile tables | `results_revision_20261010/profiles_by_cell.py`, `profile_domains.py`, `profile_trajectory_domains.py`; `main_figures_20261012/fig3_profiles.py` | `PROFILE_CELLS.csv`, `PROFILE_TRAJECTORIES.csv`, `PROFILE_DOMAINS.csv`, `PROFILE_DOMAINS_POOLED.csv`, `PROFILE_TRAJECTORY_DOMAINS.csv`, `PROFILE_TRAJECTORY_CONTRASTS.csv`; `Fig3_profiles_by_development.{jpg,pdf}` |
| S1–S7 Figs | `main_figures_20261012/supp_figs_20261012.py` (S2, S5–S7 Figs are copies of R1d figures made by the scripts in `main_figures_20261010/`) | `../supp_figures_20261012/` |
| Table 1 (Table 2 before R4; key countries; retention of persistent labels across uncertainty draws; local studies) | `main_figures_20261010/make_tables.py` (run after `s1_sensitivity_20261010/run_ui_mc.py`) | `Table2_EN.csv`, `Table2_ZH.csv` |
| Disease positions (all locations, outcomes, periods) | `independent_reproduction_20261009/repro_disease.py` | `REPRO_DISEASE_PERIOD.csv`; annual fitted values and residuals in `REPRO_ANNUAL_FIT.csv` |
| Persistent tails by location | `independent_reproduction_20261009/repro_downstream.py` | `REPRO_COUNTRY_DISEASE_PROFILES.csv` |
| Factor statuses and contrasts | `factor_profiles_confirmed_20261010/run_confirmed.py` | `CONFIRMED_FACTOR_PERIOD.csv`, `CONFIRMED_GROUP_CONTRASTS.csv`, `CONFIRMED_ASSOCIATIONS.csv` |
| Persistent-tail shortfalls (S1 Text Table V) | `results_revision_20261010/part3_typology.py` | `PART3_SHORTFALL_BY_BAND.csv`, `PART3_SHORTFALL_BY_BAND_DISEASE.csv` |
| Evaluable factors and common factor sets for persistent tails (S1 Text Table V) | `results_revision_20261010/part3_comparability.py` | `PART3_COMPARABILITY.csv` |
| Factor status of tail locations by outcome; difference between SDI strata (S1 Text Tables P and R) | `results_revision_20261010/outcome_factor_tails.py` | `OUTCOME_FACTOR_TAILS.csv`, `OUTCOME_FACTOR_TAILS_LOCATIONS.csv`, `OUTCOME_FACTOR_GRADIENT.csv` |
| Checks of the tail comparison: matched vs unmatched factors, factors that are not GBD covariates, co-occurrence of shortfalls, continuous within-stratum correlations, a minimum-magnitude rule, vaccine years before introduction coded 0% (S1 Text Tables T and U) | `results_revision_20261010/outcome_factor_checks.py` | `OUTCOME_FACTOR_CHECKS.csv`, `OUTCOME_FACTOR_VACC0.csv` |
| Location-clustered intervals for persistent-tail shortfalls (S1 Text Table V) | `results_revision_20261010/part3_comparability.py` | `PART3_SHORTFALL_BOOT.csv` |
| Findings without 2020–2022 (S1 Text) | `s1_sensitivity_20261010/covid_exclusion.py` | `S1_COVID_EXCLUSION.csv` |
| Catch-up regression adjusted for the initial gap and residual (S1 Text Table Y) | `results_revision_20261010/catchup_adjusted.py` | `PART3_CATCHUP_ADJUSTED.csv` |
| All-cause under-5 mortality envelope: cross-disease correlations after removing mortality from other causes, within subregions, and with all-cause and cause fractions (S1 Text Table L) | `gbd_envelope_20261012/envelope_check.py` (input `GBD2023_U5_ALLCAUSE_DEATH_RATE.csv`, GBD 2023 all-cause death rate per 100 000 children younger than 5 years, both sexes, retrieved from the GBD 2023 results database) | `ENVELOPE_OTHER_CAUSES.csv`, `ENVELOPE_CORR.csv`, `ENVELOPE_PERSISTENCE_WITHIN_SUBREGION.csv`, `ENVELOPE_OTHER_WITHIN.csv`, `ENVELOPE_PERIOD_RESIDUALS.csv`, `ENVELOPE_PERSISTENCE.csv` |
| Tail contrasts grouped by GBD covariate status (S1 Text Table Q) | `results_revision_20261010/covariate_status.py` | `OUTCOME_FACTOR_COVSTATUS.csv` |
| Subregion-cluster bootstrap of the main intervals (S1 Text) | `results_revision_20261010/cluster_bootstrap.py` | `CLUSTER_BOOTSTRAP.csv` |
| Outcome-specific tail contrast under alternatives and uncertainty draws (S1 Text Table S) | `s1_sensitivity_20261010/gradient_variants.py` | `S1_GRADIENT_VARIANTS.csv` |
| PM2.5 result with the WDI series (S1 Text) | `substantive_findings_20261010/pm25_wdi_check.py` | `R5c_PM25_WDI_CHECK.csv` |
| Within-subregion correlations, high-SDI LRI incidence (S7 Fig; S1 Text Table W) | `substantive_findings_20261010/r5b_within_region.py` | `R5b_WITHIN_REGION.csv` (with within-subregion permutation P values and leave-one-subregion-out ranges) |
| Catch-up comparison (S5 Fig; S1 Text Table Y) | `results_revision_20261010/part3_trajectories.py` | `PART3_CATCHUP_VS_DISEASE.csv` |
| Dispersion of residuals; variance of the difference between mortality and incidence residuals (S1 Text Table I and text) | `results_revision_20261010/residual_dispersion.py` | `RESIDUAL_DISPERSION.csv`, `VARIANCE_MORT_MINUS_INC.csv` |
| Coupling within the GBD estimates (S1 Text Tables J and K) | `gbd_coupling_20261011/coupling_checks.py` | `COVARIATE_OVERLAP.csv`, `COUPLING_PARTIAL_CORR.csv`, `COUPLING_STRATA_CORR.csv` |
| S1 Text, sensitivity tables | `s1_sensitivity_20261010/run_variants.py`, `run_ui_mc.py`, `boot_corr.py`, `factor_sig_contrasts.py` | `S1_DISEASE_VARIANTS.csv`, `S1_FACTOR_VARIANTS.csv`, `S1_UI_DRAWS.csv`, `S1_UI_LABEL_FREQ.csv`, `S1_BOOT_CORR_SUMMARY.csv`, `S1_FACTOR_SIG_CONTRASTS.csv` |
| S1 Text, Figs A–F | `figure2_reference_20261010/plot_reference_figure.py all`, `s1_sensitivity_20261010/plot_s1_robustness.py` | `figure2_reference_20261010/supplementary/`; `S1Text_FigF_robustness.{jpg,pdf}` |
| S2 Text | `country_profiles_20261011/profiles.py` | `PERSISTENT_PROFILES.csv`; literature coding in `LIT_CONCORDANCE.csv`, `REFS.py` |

## Methods in brief

* **Gaps between development groups.** Low-SDI and high-SDI groups (GBD 2023 cut-offs) fixed by mean SDI in 1990–2000 (and, as a comparison, in 2011–2023). Population-weighted incidence and mortality, and case fatality as deaths per 1000 incident cases, so that the mortality ratio is the product of the incidence and case-fatality ratios. Intervals from 2000 replicates resampling locations within groups and perturbing each location–year rate by its GBD uncertainty interval.
* **Factor profiles.** For three SDI strata and six outcomes in 2011–2023, the percentage of high-tail and low-tail locations with an adverse departure in each of 16 factors, averaged within five domains; change in these percentages between 2001–2010 and 2011–2023 by tail trajectory.
* **Disease references.** For each of six outcomes, ordinary least squares of the log annual rate on a natural cubic spline of SDI (4 df, knots at SDI quartiles), calendar-year fixed effects, and spline × centred year. Location–years are weighted equally. The fitted value is the mean log rate of countries at the same SDI in the same year: a statistical expectation, not a minimum achievable burden.
* **Relative positions.** Period residuals with location-clustered (CR1) 95% confidence intervals. A direction is resolved when the interval excludes 0; among resolved locations, quartiles define robust low and high tails. Locations in the same tail in all three periods are persistent.
* **Factors.** 19 factors (16 analysed plus the three SDI components) fitted with the same reference structure (logit, log, or log(1+x) scale), residuals expressed in original units as observed value minus back-transformed fitted value, with delta-method intervals. A location–period is evaluated whenever it has at least one observed year (series beginning in 2000 use that common year for 1990–2000); a factor is left out only where a location has no observation in the period. Physician density has a single observed year per location. Coverage ≥99% in both observed and reference values is labelled universal. For persistent tails, the late-period status of prespecified disease-relevant factors is summarised as at least one clearly adverse factor; the UHC index, a composite, is reported separately.
* **Factor status of tails by outcome.** For each outcome in 2011–2023 and within each SDI stratum, the proportion of robust high-tail and robust low-tail locations with each prespecified factor clearly adverse (universal coverage counted as not adverse), with Newcombe intervals, Fisher exact tests, and Benjamini–Hochberg q values over the 34 outcome–factor pairs of each stratum. Differences between strata use equal weights for each outcome–factor pair and 2000 location resamples within strata.
* **Catch-up.** Among locations clearly adverse for a factor in 2001–2010, the proportion whose disease residual fell is compared between locations whose gap narrowed and those whose gap did not (Newcombe hybrid score intervals), with Spearman correlations of the two changes (2000 location resamples) and the initial gaps of the two groups.
* **Coupling within the GBD estimates.** Cross-disease correlations are recalculated after removing, by linear regression, the residuals of eight observed factors that are, or stand in for, covariates of one or more GBD mortality models (for example, the UHC index for the Healthcare Access and Quality index), and within SDI strata (1000 location resamples). This tests whether these measured proxies explain the correlation structure; it does not remove the coupling between GBD models.
* **Sensitivity analyses.** Alternative reference forms, spline flexibility, population weighting, exclusion of small locations, leave-location-out references (five fixed folds of locations, given in the panel column `location_holdout_oof_fold`; residuals and their confidence intervals come from the reference refitted without each fold), alternative tail cut-offs, factor-rule variants (including the earlier rule that did not evaluate single-year series in 1990–2000), propagation of GBD uncertainty intervals (200 draws; one perturbation per location, outcome, and period, independent across periods and outcomes; the frequency with which each location kept its persistent label is given in `S1_UI_LABEL_FREQ.csv`), and a location bootstrap for differences between correlations (500 resamples, all six references refitted each time).

Data correction (2026-10-11): in the input panel, `region_proxy` for Nauru was corrected from Polynesia to Micronesia (UN M49). This changes only the all-location within-subregion correlations in `R5b_WITHIN_REGION.csv`; the high-SDI results used in the manuscript are unchanged.

Labels inside the intermediate CSV files are in Chinese for historical reasons: 稳健高尾 = robust high tail, 稳健低尾 = robust low tail, 明确偏离但居中 = resolved but central, 参照不确定 = reference-uncertain, 明确不利 = clearly adverse, 明确有利 = clearly favourable, 方向不确定 = uncertain, 已达普及 = universal, 不可评价 = not evaluable; periods use en dashes (1990–2000).

## Data sources and terms of use

The input panel combines published estimates; no data were imputed, interpolated, or extrapolated.

| Content | Source |
|---|---|
| Incidence and mortality rates, numbers, and 95% uncertainty intervals; SDI; population younger than 5 years; ambient and household PM2.5; second-hand smoke SEV | Global Burden of Disease Study 2023, Institute for Health Metrics and Evaluation (GBD Results Tool) |
| Water, sanitation, clean cooking, DTP3, stunting, low birthweight, undernourishment, UHC index, health expenditure, hospital beds, nurses and midwives, GDP per capita, secondary enrolment, fertility | World Bank World Development Indicators (including WHO/UNICEF JMP, WUENIC, and JME series) |
| Hib3 and PCV3 coverage | WHO Global Health Observatory |
| Temperature, humidity, cold days, wet days | ERA5 reanalysis (Copernicus Climate Change Service), weighted by WorldPop populations younger than 5 years |

GBD data are used under the IHME Free-of-Charge Non-commercial User Agreement; users of the panel should cite GBD 2023 and follow the terms of each original source. The code is released under the MIT licence.

## Citation

If you use these data or code, please cite the manuscript (citation to be added on publication) and the original data sources above.
