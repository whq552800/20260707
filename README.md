# LRI environmental-context analysis public materials

This directory contains the machine-readable tables, figures and portable code materials supporting the manuscript:

**Environmental contexts and age-specific lower respiratory infection burden globally, 1990-2023**

## Directory structure

- `tables/`: CSV tables used for the main manuscript and Supplementary Tables S1-S6, plus country-year profile membership and trajectory summaries.
- `figures/`: publication figure files in PNG/PDF/SVG where available.
- `code/`: analysis scripts used to construct the environmental-domain profile analyses and figure outputs. These scripts are provided as source material; workstation-specific Word manuscript builders are intentionally excluded.
- `docs/`: manifest and validation summaries.

## Table guide

Core manuscript and supplement tables:

- `clean_domain_model_coefficients.csv`: primary age-specific environmental-domain association models.
- `clean_domain_model_variants.csv`: primary, population-weighted and within-country model variants.
- `clean_domain_archetype_k_selection.csv`: K=3-7 profile selection diagnostics.
- `clean_domain_archetype_loadings_k6.csv`: coordinates and labels for the six compound background profiles.
- `clean_domain_archetype_2023_age_burden_signatures_k6.csv`: 2023 profile-specific LRI burden summaries.
- `clean_domain_archetype_membership_k6_country_year.csv`: country-year profile weights and dominant profile labels.
- `clean_domain_archetype_trajectory_country_k6.csv`: country-level profile paths and LRI declines from 1990 to 2023.
- `clean_domain_archetype_trajectory_summary_k6.csv`: trajectory-cluster summary table.
- `clean_domain_trajectory_cluster_k_selection.csv`: trajectory-cluster K diagnostics.
- `era5important_domain_score_pca_loadings.csv`: PCA loadings for climate-domain construction.

Additional candidate/sensitivity tables:

- `candidate_evidence_credibility_matrix.csv`
- `candidate_longterm_quantile_regression.csv`
- `candidate_ml_grouped_permutation_importance_summary.csv`
- `candidate_pathway_model_variants.csv`
- `candidate_sdi_stratified_pathway_coefficients.csv`

These candidate tables document exploratory and sensitivity analyses considered during manuscript development. They are included for transparency and should not be read as replacing the primary domain, profile and trajectory results.

## Notes on reproducibility

The public tables are derived from the LRI_ENV 1990-2023 country-year ecological panel and environment-domain analyses. Some original upstream inputs are large or externally licensed/modelled datasets, including GBD/IHME LRI and exposure estimates, ERA5 climate products and WorldPop population grids. Users should obtain upstream data from their official providers and adapt paths locally.

The compound profiles are outcome-independent summaries of HAP-PM, outdoor PM2.5, climate and development coordinates. They should be interpreted as descriptive environmental-background profiles, not deterministic country classes or causal categories.
