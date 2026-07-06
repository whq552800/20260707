# Table catalog

This catalog separates primary manuscript-supporting tables from exploratory or sensitivity tables. Row counts include the header row.

## Primary manuscript and supplement tables

| File | Rows | Role |
|---|---:|---|
| `clean_domain_model_coefficients.csv` | 17 | Primary age-specific environmental-domain association estimates for the four age outcomes and four predefined domains. |
| `clean_domain_model_variants.csv` | 49 | Primary, population-weighted and within-country specifications used to compare estimate direction and scale. |
| `clean_domain_archetype_k_selection.csv` | 6 | K=3-7 profile-selection diagnostics. K=6 is retained as an interpretability compromise, not as a uniquely optimal natural class count. |
| `clean_domain_archetype_loadings_k6.csv` | 7 | Standardized coordinates, labels and exemplar country-years for the six compound environmental-background profiles. |
| `clean_domain_archetype_2023_age_burden_signatures_k6.csv` | 25 | Profile-specific 2023 median LRI incidence summaries by age group. |
| `clean_domain_archetype_membership_k6_country_year.csv` | 6903 | Country-year profile weights, dominant profile labels and age-specific LRI incidence rates. |
| `clean_domain_archetype_trajectory_country_k6.csv` | 204 | Country-level 1990, 2000, 2010 and 2023 profile paths and age-specific LRI declines. |
| `clean_domain_archetype_trajectory_summary_k6.csv` | 7 | Trajectory-cluster summaries, country counts and median age-specific LRI declines. |
| `clean_domain_trajectory_cluster_k_selection.csv` | 6 | K diagnostics for trajectory clustering. |
| `era5important_domain_score_pca_loadings.csv` | 17 | PCA loadings used to construct climate-domain scores from ERA5-derived indicators. |

## Additional candidate and sensitivity tables

| File | Rows | Role |
|---|---:|---|
| `candidate_evidence_credibility_matrix.csv` | 17 | Screening matrix used to judge which candidate analyses were stable enough for manuscript emphasis. |
| `candidate_longterm_quantile_regression.csv` | 65 | Long-term quantile-regression candidates considered during result development. |
| `candidate_ml_grouped_permutation_importance_summary.csv` | 25 | Grouped permutation-importance summaries from candidate machine-learning checks. |
| `candidate_pathway_model_variants.csv` | 49 | Candidate pathway/domain model variants. |
| `candidate_sdi_stratified_pathway_coefficients.csv` | 81 | SDI-stratified candidate pathway coefficients. |

The candidate tables are provided for transparency. The primary interpretation should rely on the domain models, compound profiles and trajectory summaries listed above.
