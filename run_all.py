"""Run the full analysis in order, from the input panel in data/ to every table and figure.

Usage:  python run_all.py            # everything (the two resampling steps take most of the time)
        python run_all.py --quick    # skip the 200-draw GBD-uncertainty propagation and the 500-resample bootstrap
        python run_all.py --from 9   # start at step 9 (add --to N to stop after step N)

Each step runs in its own folder under analysis/ and writes its outputs there. Steps are listed in
dependency order; see README.md for what each step produces and where it is used in the manuscript."""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A = ROOT / 'analysis'
STEPS = [
    # (folder, script, args, description)
    ('.', 'sdi_tiers.py', [], 'SDI levels by period (GBD 2023 fixed cut-offs)'),
    ('independent_reproduction_20261009', 'repro_disease.py', [], 'Disease reference models, period residuals, intervals, tail classes'),
    ('independent_reproduction_20261009', 'repro_factor.py', [], 'Factor residuals, legacy rules (input to the profile table below)'),
    ('independent_reproduction_20261009', 'repro_downstream.py', [], 'Country disease profiles (persistent tails)'),
    ('independent_reproduction_20261009', 'repro_sdi_gaps.py', [], 'Annual population-weighted rates by SDI group'),
    ('sdi_gaps_official_tiers_20261010', 'sdi_gaps_official.py', [], 'Gaps between low-SDI and high-SDI groups (Table 1, Fig 1A-B)'),
    ('lri_gap_instability_20261010', 'lri_instability.py', [], 'Group rate ratios by grouping and 2023 decomposition (Fig 1B-C)'),
    ('lri_gap_instability_20261010', 'entrants_position.py', [], 'High-SDI entrants versus original members (Fig 1E)'),
    ('substantive_findings_20261010', 'explore_A_excess_and_who.py', [], 'Location profiles with regions'),
    ('substantive_findings_20261010', 'r4_environment.py', [], 'Environmental residuals (PM2.5, climate)'),
    ('substantive_findings_20261010', 'r5b_within_region.py', [], 'Within-subregion correlations with permutation and leave-one-subregion-out checks (S7 Fig, S1 Text Table W)'),
    ('substantive_findings_20261010', 'pm25_wdi_check.py', [], 'Within-subregion PM2.5 correlation with the WDI series (S1 Text)'),
    ('reference_spec_sensitivity_20261010', 'compare_reference_specs.py', [], 'Alternative reference forms B, C, D'),
    ('factor_profiles_confirmed_20261010', 'run_confirmed.py', [], 'Factor residuals, statuses, contrasts and correlations (main analysis)'),
    ('factor_profiles_confirmed_20261010', 'sensitivity_confirmed.py', [], 'Factor-rule sensitivity analyses S1-S9'),
    ('factor_residual_audit_20261010', 'f1_coverage.py', [], 'Factor data coverage (S1 Text Table B)'),
    ('residual_matrix_workbook_20261010', 'compute_matrix.py', [], 'Location-period residual matrices for diseases and factors'),
    ('tier_period_tail_factors_20261010', 'tier_period_tail_factors.py', [], 'Factor status of tail records by SDI level and period'),
    ('tier_period_tail_factors_20261010', 'find_patterns.py', [], 'Domain contrasts by SDI stratum (Fig 3A, S3 Fig)'),
    ('results_revision_20261010', 'verify_intro.py', [], 'Global population-weighted rates (Table 1)'),
    ('results_revision_20261010', 'verify_part1.py', [], 'Persistence and cross-outcome summaries'),
    ('results_revision_20261010', 'part3_typology.py', [], 'Shortfall domains of persistent tails (Fig 3B)'),
    ('results_revision_20261010', 'part3_comparability.py', [], 'Persistent tails: evaluable factors and common factor sets (S1 Text Table V)'),
    ('results_revision_20261010', 'outcome_factor_tails.py', [], 'Factor status of tail locations by outcome; gradient between SDI strata (S1 Text Tables P and R)'),
    ('results_revision_20261010', 'outcome_factor_checks.py', [], 'Checks: matched vs unmatched factors, non-covariate factors, continuous and minimum-magnitude versions, vaccine years before introduction (S1 Text Tables T and U)'),
    ('results_revision_20261010', 'part3_trajectories.py', [], 'Service catch-up versus disease change, with intervals (S5 Fig)'),
    ('results_revision_20261010', 'catchup_adjusted.py', [], 'Catch-up regression adjusted for initial gap and residual (S1 Text Table Y)'),
    ('gbd_envelope_20261012', 'envelope_check.py', [], 'All-cause under-5 mortality envelope: cross-disease correlations after removing it (S1 Text Table L)'),
    ('results_revision_20261010', 'covariate_status.py', [], 'Tail contrasts grouped by GBD covariate status (S1 Text Table Q)'),
    ('results_revision_20261010', 'cluster_bootstrap.py', [], 'Subregion-cluster bootstrap of the main intervals (S1 Text)'),
    ('s1_sensitivity_20261010', 'gradient_variants.py', [], 'Outcome-specific tail contrast under all alternatives and 200 uncertainty draws (slow; S1 Text Table S)'),
    ('results_revision_20261010', 'residual_dispersion.py', [], 'Dispersion of disease residuals by period'),
    ('gbd_coupling_20261011', 'coupling_checks.py', [], 'Checks for coupling inside GBD estimates (S1 Text)'),
    ('figure2_reference_20261010', 'plot_reference_figure.py', [], 'Reference-model example, LRI incidence (Fig 2A)'),
    ('figure2_reference_20261010', 'plot_reference_figure.py', ['all'], 'Reference-model figures for all six outcomes (S1 Text Figs A-E)'),
    ('main_figures_20261010', 'fig1_composite.py', [], 'R1d Fig 1 (superseded)'),
    ('main_figures_20261010', 'fig2_persistence_crossdisease.py', [], 'R1d Fig 2 (superseded)'),
    ('main_figures_20261010', 'fig3_shortfalls.py', [], 'R1d Fig 3 (superseded)'),
    ('main_figures_20261010', 'fig4_catchup.py', [], 'R1d Fig 4 (reused as S5 Fig)'),
    ('main_figures_20261010', 'supp_figs.py', [], 'R1d S1-S3 Fig (S1 Fig reused as S2 Fig)'),
    ('main_figures_20261010', 'supp_figs_moved.py', [], 'R1d S4-S6 Fig (case trajectories reused as S6 Fig, PM2.5 as S7 Fig)'),
    ('s1_sensitivity_20261010', 'check_main.py', [], 'Check that the sensitivity engine reproduces the main findings'),
    ('s1_sensitivity_20261010', 'run_variants.py', [], 'Disease-side sensitivity analyses (S1 Text Tables AA, H and I)'),
    ('s1_sensitivity_20261010', 'covid_exclusion.py', [], 'Disease-side findings without 2020-2022 (S1 Text)'),
    ('s1_sensitivity_20261010', 'factor_sig_contrasts.py', [], 'Significant contrasts under factor-rule variants (S1 Text Table AC)'),
    ('s1_sensitivity_20261010', 'run_ui_mc.py', [], 'Propagation of GBD uncertainty, 200 draws, with label frequencies by location (slow)'),
    ('s1_sensitivity_20261010', 'boot_corr.py', ['500'], 'Location bootstrap of correlation differences, 500 resamples (slow; S1 Text Table H)'),
    ('s1_sensitivity_20261010', 'plot_s1_robustness.py', [], 'S1 Text Fig F'),
    ('gap_story_20261012', 'gap_decomposition.py', [], 'Low/high SDI gaps in incidence, case fatality and mortality with intervals (Table 1, Fig 1, S1 Text)'),
    ('gap_story_20261012', 'persistence_intervals.py', [], 'Intervals for rank stability, tail retention and incidence-mortality direction (Fig 2)'),
    ('results_revision_20261010', 'profiles_by_cell.py', [], 'Factor profiles for 18 SDI stratum x outcome cells and tail trajectories (Fig 3A, S1 Text)'),
    ('results_revision_20261010', 'profile_domains.py', [], 'Domain summaries of the profiles with location-resampling intervals (Fig 3B)'),
    ('results_revision_20261010', 'profile_trajectory_domains.py', [], 'Change in profiles by tail trajectory and between-group contrasts (Fig 3C, S4 Fig)'),
    ('main_figures_20261012', 'fig1_gap.py', [], 'Fig 1'),
    ('main_figures_20261012', 'fig2_position.py', [], 'Fig 2'),
    ('main_figures_20261012', 'fig3_profiles.py', [], 'Fig 3'),
    ('main_figures_20261012', 'supp_figs_20261012.py', [], 'S1-S7 Fig'),
    ('main_figures_20261010', 'make_tables.py', [], 'Table 2 (CSV; uses the uncertainty-draw label frequencies; the R1d Table 1 is superseded by gap_story_20261012)'),
    ('country_profiles_20261011', 'profiles.py', [], 'Factor profiles of persistent-tail countries (S2 Text)'),
]
SLOW = {'run_ui_mc.py', 'boot_corr.py', 'gradient_variants.py'}

if __name__ == '__main__':
    quick = '--quick' in sys.argv
    start = int(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else 1
    stop = int(sys.argv[sys.argv.index('--to') + 1]) if '--to' in sys.argv else len(STEPS)
    for i, (folder, script, args, desc) in enumerate(STEPS, 1):
        if i < start or i > stop or (quick and script in SLOW): continue
        t0 = time.time(); print(f'[{i:02d}/{len(STEPS)}] {folder}/{script}: {desc}', flush=True)
        r = subprocess.run([sys.executable, script, *args], cwd=A / folder, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], sep='\n'); sys.exit(f'Step {i} failed: {folder}/{script}')
        print(f'        done in {time.time() - t0:.0f} s', flush=True)
    print('All steps finished.')
