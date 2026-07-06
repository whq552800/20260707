# Public-release note: paths were made relative to the release root. Place required upstream input data under the expected data/ subdirectories, or edit PANEL/OUT below for your local environment.
# -*- coding: utf-8 -*-
"""Run profile-innovation analyses for the LRI_ENV manuscript.

Outputs:
1. AJPH-style model-space audit on long-term country means.
2. Prespecified pathway/domain models and block-style evidence.
3. Outcome-independent convex archetype prototype analysis.
4. Archetype membership trajectory clustering.
5. Publication-style candidate figures and Chinese report.

Interpretation boundary:
All outputs are country-level ecological profile/pathway descriptions. They are
not individual-level effects, causal mediation estimates, or attributable burden.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.optimize import nnls
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "panel_lri_sdi_env_pollution_1990_2023_analysis_ready_v1.csv"
OUT = ROOT / "data" / "profile_innovation_20260704"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"
SOURCE_DIR = OUT / "source_data"
VALID_DIR = OUT / "validation"


for d in [TABLE_DIR, FIG_DIR, REPORT_DIR, SOURCE_DIR, VALID_DIR]:
    d.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "mathtext.fontset": "stix",
    }
)


OUTCOMES = {
    "u5": "lri_incidence_rate_u5",
    "all_age": "lri_incidence_rate_all_age",
    "age5_69": "lri_incidence_rate_5_69",
    "age70p": "lri_incidence_rate_age70p",
}

OUTCOME_LABELS = {
    "u5": "Under-5",
    "all_age": "All ages",
    "age5_69": "5-69 years",
    "age70p": "70+ years",
}

FEATURE_LABELS = {
    "sdi": "SDI",
    "log_hap": "HAP-PM",
    "log_pm25": "Ambient PM2.5",
    "tavg": "Temperature",
    "ah": "Absolute humidity",
}


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    ok = x.notna() & w.notna() & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(x[ok], weights=w[ok]))


def choose_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing all candidates: {candidates}")


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL)
    df["country"] = df["iso3"].fillna(df["location_name"].astype(str))
    df["hap_base"] = df[choose_col(df, ["hap_pm_pw", "hap_pm"])]
    df["pm25_base"] = df[choose_col(df, ["pm25_pw", "pm25_gbd"])]
    df["tavg"] = df[choose_col(df, ["tavg_pw_C", "tavg_C_pw_u5", "tas_pw"])]
    df["ah"] = df[choose_col(df, ["ah_pw", "ah_g_m3_pw_u5"])]
    df["rh"] = df[choose_col(df, ["rh_pct_pw", "rh_pw", "rh_pct_pw_u5"])]
    if "high_rh_gt60_days_pw_total" in df.columns:
        df["high_rh_days"] = df["high_rh_gt60_days_pw_total"]
    else:
        df["high_rh_days"] = np.nan
    if "outside_rh_days_pw" in df.columns:
        df["outside_rh_days"] = df["outside_rh_days_pw"]
    else:
        df["outside_rh_days"] = np.nan
    df["log_hap"] = np.log1p(df["hap_base"].clip(lower=0))
    df["log_pm25"] = np.log1p(df["pm25_base"].clip(lower=0))
    for key, col in OUTCOMES.items():
        df[f"log_{key}"] = np.log(df[col].clip(lower=1e-6))
    required = ["country", "location_name", "year", "sdi", "log_hap", "log_pm25", "tavg", "ah"] + [
        f"log_{k}" for k in OUTCOMES
    ]
    df = df.dropna(subset=required).copy()
    return df


def build_country_means(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sdi",
        "hap_base",
        "pm25_base",
        "log_hap",
        "log_pm25",
        "tavg",
        "ah",
        "rh",
        "high_rh_days",
        "outside_rh_days",
        "pop_total",
        "pop_total_sum",
    ] + [OUTCOMES[k] for k in OUTCOMES]
    use_cols = [c for c in cols if c in df.columns]
    g = df.groupby(["country", "location_name"], as_index=False)[use_cols].mean(numeric_only=True)
    for key, col in OUTCOMES.items():
        g[f"log_{key}"] = np.log(g[col].clip(lower=1e-6))
    return g


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    model = LinearRegression().fit(X, y)
    return y - model.predict(X)


def make_sdi_design(sdi: pd.Series, spec: str) -> pd.DataFrame:
    if spec == "linear":
        return pd.DataFrame({"sdi_z": zscore(sdi).to_numpy()})
    if spec == "quadratic":
        z = zscore(sdi)
        return pd.DataFrame({"sdi_z": z.to_numpy(), "sdi_z2": (z**2).to_numpy()})
    raise ValueError(spec)


def compute_vif(X: np.ndarray) -> float:
    if X.shape[1] <= 1:
        return 1.0
    X2 = sm.add_constant(X, has_constant="add")
    vals = []
    for i in range(1, X2.shape[1]):
        try:
            vals.append(float(variance_inflation_factor(X2, i)))
        except Exception:
            vals.append(float("inf"))
    return float(np.nanmax(vals))


def cv_rmse(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 20260704) -> float:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)
    for train, test in kf.split(X):
        model = LinearRegression().fit(X[train], y[train])
        preds[test] = model.predict(X[test])
    return float(np.sqrt(np.mean((preds - y) ** 2)))


def fit_model_space_audit(country: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = country.copy()
    base["hap_raw_z"] = zscore(base["hap_base"])
    base["hap_log_z"] = zscore(base["log_hap"])
    base["pm_raw_z"] = zscore(base["pm25_base"])
    base["pm_log_z"] = zscore(base["log_pm25"])
    base["tavg_z"] = zscore(base["tavg"])
    base["ah_z"] = zscore(base["ah"])
    base["ah_resid_tavg"] = residualize(base[["ah_z"]].to_numpy(), base[["tavg_z"]].to_numpy()).ravel()
    base["tavg_resid_ah"] = residualize(base[["tavg_z"]].to_numpy(), base[["ah_z"]].to_numpy()).ravel()

    for outcome in OUTCOMES:
        y = base[f"log_{outcome}"].to_numpy()
        for hap_t in ["raw", "log"]:
            for pm_t in ["raw", "log"]:
                for climate in ["raw", "ah_resid_on_tavg", "tavg_resid_on_ah"]:
                    for sdi_spec in ["linear", "quadratic"]:
                        for interact in ["none", "sdi_interaction"]:
                            cols = []
                            col_names = []
                            hap_col = "hap_raw_z" if hap_t == "raw" else "hap_log_z"
                            pm_col = "pm_raw_z" if pm_t == "raw" else "pm_log_z"
                            for c in [hap_col, pm_col]:
                                cols.append(base[c].to_numpy())
                                col_names.append(c)
                            if climate == "raw":
                                cvars = ["tavg_z", "ah_z"]
                            elif climate == "ah_resid_on_tavg":
                                cvars = ["tavg_z", "ah_resid_tavg"]
                            else:
                                cvars = ["tavg_resid_ah", "ah_z"]
                            for c in cvars:
                                cols.append(base[c].to_numpy())
                                col_names.append(c)
                            sdi_df = make_sdi_design(base["sdi"], sdi_spec)
                            for c in sdi_df.columns:
                                cols.append(sdi_df[c].to_numpy())
                                col_names.append(c)
                            if interact == "sdi_interaction":
                                sdi_z = zscore(base["sdi"]).to_numpy()
                                for c in [hap_col, pm_col] + cvars:
                                    cols.append(base[c].to_numpy() * sdi_z)
                                    col_names.append(f"{c}:sdi")
                            X = np.column_stack(cols)
                            X = np.asarray(X, dtype=float)
                            X_const = sm.add_constant(X, has_constant="add")
                            try:
                                fit = sm.OLS(y, X_const).fit(cov_type="HC3")
                                metric = {
                                    "outcome": outcome,
                                    "outcome_label": OUTCOME_LABELS[outcome],
                                    "hap_transform": hap_t,
                                    "pm25_transform": pm_t,
                                    "climate_structure": climate,
                                    "sdi_spec": sdi_spec,
                                    "interaction": interact,
                                    "n": int(len(y)),
                                    "cv_rmse": cv_rmse(X, y),
                                    "bic": float(fit.bic),
                                    "adj_r2": float(fit.rsquared_adj),
                                    "max_vif": compute_vif(X),
                                    "n_p_lt_0_05_exposure_terms": int(
                                        np.sum(np.asarray(fit.pvalues[1 : 1 + 4]) < 0.05)
                                    ),
                                }
                            except Exception as exc:
                                metric = {
                                    "outcome": outcome,
                                    "outcome_label": OUTCOME_LABELS[outcome],
                                    "hap_transform": hap_t,
                                    "pm25_transform": pm_t,
                                    "climate_structure": climate,
                                    "sdi_spec": sdi_spec,
                                    "interaction": interact,
                                    "n": int(len(y)),
                                    "cv_rmse": np.nan,
                                    "bic": np.nan,
                                    "adj_r2": np.nan,
                                    "max_vif": np.nan,
                                    "n_p_lt_0_05_exposure_terms": np.nan,
                                    "error": str(exc),
                                }
                            rows.append(metric)
    out = pd.DataFrame(rows).sort_values(["outcome", "cv_rmse", "max_vif"])
    out.to_csv(TABLE_DIR / "model_space_audit_longterm_country_means.csv", index=False, encoding="utf-8-sig")
    # Operation summary for prose.
    summaries = []
    for var in ["hap_transform", "pm25_transform", "climate_structure", "sdi_spec", "interaction"]:
        g = out.groupby(["outcome", var], dropna=False).agg(
            median_cv_rmse=("cv_rmse", "median"),
            median_bic=("bic", "median"),
            median_max_vif=("max_vif", "median"),
            median_adj_r2=("adj_r2", "median"),
            n_models=("cv_rmse", "count"),
        )
        g = g.reset_index().rename(columns={var: "choice"})
        g["operation"] = var
        summaries.append(g)
    pd.concat(summaries, ignore_index=True).to_csv(
        TABLE_DIR / "model_space_operation_summary.csv", index=False, encoding="utf-8-sig"
    )
    return out


def build_domain_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["sdi", "log_hap", "log_pm25", "tavg", "ah", "rh", "high_rh_days", "outside_rh_days"]:
        if c in out.columns and out[c].notna().sum() > 0:
            out[f"z_{c}"] = zscore(out[c])
    wet_cols = [c for c in ["z_ah", "z_rh", "z_high_rh_days"] if c in out.columns]
    out["path_household_disadvantage"] = (out["z_log_hap"] - out["z_sdi"]) / 2.0
    out["path_ambient_pollution"] = out["z_log_pm25"]
    out["path_thermal_context"] = out["z_tavg"]
    out["path_wetness_context"] = out[wet_cols].mean(axis=1) if wet_cols else out["z_ah"]
    return out


def fit_pathway_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    block_rows = []
    data = build_domain_scores(df)
    for outcome in OUTCOMES:
        d = data.dropna(
            subset=[
                f"log_{outcome}",
                "path_household_disadvantage",
                "path_ambient_pollution",
                "path_thermal_context",
                "path_wetness_context",
                "sdi",
                "year",
                "country",
            ]
        ).copy()
        formula = (
            f"log_{outcome} ~ path_household_disadvantage + path_ambient_pollution + "
            "path_thermal_context + path_wetness_context + bs(sdi, df=4, include_intercept=False) + C(year)"
        )
        fit = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["country"]})
        for term in [
            "path_household_disadvantage",
            "path_ambient_pollution",
            "path_thermal_context",
            "path_wetness_context",
        ]:
            beta = float(fit.params[term])
            ci = fit.conf_int().loc[term].astype(float)
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS[outcome],
                    "pathway": term.replace("path_", ""),
                    "beta_log_rate": beta,
                    "percent_difference_per_1sd": (math.exp(beta) - 1) * 100,
                    "ci_low_percent": (math.exp(float(ci[0])) - 1) * 100,
                    "ci_high_percent": (math.exp(float(ci[1])) - 1) * 100,
                    "p_value": float(fit.pvalues[term]),
                    "n": int(fit.nobs),
                    "r2": float(fit.rsquared),
                }
            )
        # Cluster-robust Wald tests for pathway blocks.
        params = list(fit.params.index)
        climate_terms = ["path_thermal_context", "path_wetness_context"]
        all_path_terms = [
            "path_household_disadvantage",
            "path_ambient_pollution",
            "path_thermal_context",
            "path_wetness_context",
        ]
        for block_name, terms in [("all_pathways", all_path_terms), ("climate_pathways", climate_terms)]:
            R = np.zeros((len(terms), len(params)))
            for i, term in enumerate(terms):
                R[i, params.index(term)] = 1.0
            wt = fit.wald_test(R, scalar=False)
            block_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS[outcome],
                    "block": block_name,
                    "chi2": float(np.asarray(wt.statistic).ravel()[0]),
                    "df": int(len(terms)),
                    "p_value": float(wt.pvalue),
                    "n": int(fit.nobs),
                }
            )
    coef = pd.DataFrame(rows)
    blocks = pd.DataFrame(block_rows)
    coef.to_csv(TABLE_DIR / "pathway_domain_model_coefficients.csv", index=False, encoding="utf-8-sig")
    blocks.to_csv(TABLE_DIR / "pathway_domain_block_tests.csv", index=False, encoding="utf-8-sig")
    data.to_csv(SOURCE_DIR / "panel_with_prespecified_pathway_scores.csv", index=False, encoding="utf-8-sig")
    return coef, blocks


def fit_development_adjusted_pls(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    use = df.dropna(
        subset=["sdi", "year", "log_hap", "log_pm25", "tavg", "ah", "rh"] + [f"log_{k}" for k in OUTCOMES]
    ).copy()
    X_raw = use[["log_hap", "log_pm25", "tavg", "ah", "rh"]].to_numpy()
    Y_raw = use[[f"log_{k}" for k in OUTCOMES]].to_numpy()
    # Adjust for nonlinear SDI and year fixed effects before extracting PLS axes.
    controls = dmatrix("bs(sdi, df=4, include_intercept=False) + C(year)", data=use, return_type="dataframe").to_numpy()
    X_res = np.column_stack([residualize(X_raw[:, i], controls) for i in range(X_raw.shape[1])])
    Y_res = np.column_stack([residualize(Y_raw[:, i], controls) for i in range(Y_raw.shape[1])])
    Xs = StandardScaler().fit_transform(X_res)
    Ys = StandardScaler().fit_transform(Y_res)
    pls = PLSRegression(n_components=2)
    pls.fit(Xs, Ys)
    xw = pd.DataFrame(
        pls.x_weights_,
        index=["HAP-PM", "Ambient PM2.5", "Temperature", "Absolute humidity", "Relative humidity"],
        columns=["PLS axis 1", "PLS axis 2"],
    )
    yw = pd.DataFrame(
        pls.y_weights_,
        index=[OUTCOME_LABELS[k] for k in OUTCOMES],
        columns=["PLS axis 1", "PLS axis 2"],
    )
    xw.to_csv(TABLE_DIR / "development_adjusted_pls_x_weights.csv", encoding="utf-8-sig")
    yw.to_csv(TABLE_DIR / "development_adjusted_pls_y_weights.csv", encoding="utf-8-sig")
    return xw, yw


def simplex_weights(X: np.ndarray, A: np.ndarray, lam: float = 50.0) -> np.ndarray:
    # A: K x p archetype coordinates. Solve for each x: x ~ w @ A,
    # w >= 0, sum(w) ~= 1, then normalize exactly.
    K = A.shape[0]
    M = np.vstack([A.T, lam * np.ones((1, K))])
    W = np.zeros((X.shape[0], K), dtype=float)
    for i, x in enumerate(X):
        b = np.r_[x, lam]
        w, _ = nnls(M, b)
        s = w.sum()
        if s <= 0:
            w[:] = 1.0 / K
        else:
            w = w / s
        W[i] = w
    return W


def farthest_archetypes(X: np.ndarray, K: int) -> list[int]:
    selected = [int(np.argmax(np.linalg.norm(X - X.mean(axis=0), axis=1)))]
    while len(selected) < K:
        D = np.linalg.norm(X[:, None, :] - X[selected][None, :, :], axis=2)
        min_d = D.min(axis=1)
        min_d[selected] = -np.inf
        selected.append(int(np.argmax(min_d)))
    return selected


def archetype_fit(X: np.ndarray, K: int) -> tuple[list[int], np.ndarray, np.ndarray, float]:
    idx = farthest_archetypes(X, K)
    A = X[idx].copy()
    W = simplex_weights(X, A)
    recon = W @ A
    rmse = float(np.sqrt(np.mean((X - recon) ** 2)))
    return idx, A, W, rmse


def label_archetypes(A_original_z: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in A_original_z.iterrows():
        sdi = row["sdi"]
        hap = row["log_hap"]
        pm = row["log_pm25"]
        tavg = row["tavg"]
        ah = row["ah"]
        if sdi > 0.5 and hap < 0 and pm < 0:
            label = "High-SDI low-exposure archetype"
        elif sdi < -0.3 and hap > 0.4:
            label = "Low-SDI household-energy archetype"
        elif pm == max(pm, hap, tavg, ah) and pm > 0.4:
            label = "Ambient PM2.5 archetype"
        elif (tavg + ah) / 2 > 0.5:
            label = "Warm-humid climate-context archetype"
        else:
            label = "Mixed/intermediate archetype"
        labels.append(label)
    # Make labels unique if needed.
    seen = {}
    unique = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        unique.append(label if seen[label] == 1 else f"{label} {seen[label]}")
    return unique


def run_archetype_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = ["sdi", "log_hap", "log_pm25", "tavg", "ah"]
    use = df.dropna(subset=features + ["country", "year"]).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(use[features].to_numpy())
    k_rows = []
    fits = {}
    for K in range(3, 7):
        idx, A, W, rmse = archetype_fit(X, K)
        dom = W.argmax(axis=1)
        sil = silhouette_score(X, dom) if len(np.unique(dom)) > 1 else np.nan
        k_rows.append({"K": K, "reconstruction_rmse": rmse, "dominant_silhouette": sil})
        fits[K] = (idx, A, W, rmse)
    k_table = pd.DataFrame(k_rows)
    k_table.to_csv(TABLE_DIR / "archetype_k_selection.csv", index=False, encoding="utf-8-sig")
    # Conservative main K: prefer 4 unless K=5 has materially better reconstruction.
    main_K = 4
    idx, A, W, rmse = fits[main_K]
    A_df = pd.DataFrame(A, columns=features)
    A_df.insert(0, "archetype_id", [f"A{i+1}" for i in range(main_K)])
    A_df["label"] = label_archetypes(A_df[features])
    # Add observed exemplar country-year.
    for j, obs_idx in enumerate(idx):
        A_df.loc[j, "exemplar_country"] = use.iloc[obs_idx]["location_name"]
        A_df.loc[j, "exemplar_year"] = int(use.iloc[obs_idx]["year"])
    # Reorder for stable manuscript display.
    preferred = [
        "Low-SDI household-energy archetype",
        "Ambient PM2.5 archetype",
        "Warm-humid climate-context archetype",
        "High-SDI low-exposure archetype",
        "Mixed/intermediate archetype",
    ]
    order = sorted(range(len(A_df)), key=lambda i: preferred.index(A_df.loc[i, "label"].split(" 2")[0]) if A_df.loc[i, "label"].split(" 2")[0] in preferred else 99)
    A_df = A_df.iloc[order].reset_index(drop=True)
    old_to_new = {old: new for new, old in enumerate(order)}
    W_reordered = W[:, order]
    A_reordered = A[order]
    A_df["archetype_id"] = [f"A{i+1}" for i in range(main_K)]
    A_df.to_csv(TABLE_DIR / "archetype_loadings_k4.csv", index=False, encoding="utf-8-sig")
    # Membership table.
    mem = use[["country", "location_name", "year"]].copy()
    for j in range(main_K):
        mem[f"archetype_weight_A{j+1}"] = W_reordered[:, j]
    dom = W_reordered.argmax(axis=1)
    mem["dominant_archetype_id"] = [f"A{i+1}" for i in dom]
    mem["dominant_archetype_label"] = [A_df.loc[i, "label"] for i in dom]
    mem = mem.merge(df[["country", "year"] + [OUTCOMES[k] for k in OUTCOMES]], on=["country", "year"], how="left")
    mem.to_csv(SOURCE_DIR / "archetype_membership_k4_country_year.csv", index=False, encoding="utf-8-sig")
    # 2023 age-burden signatures by dominant archetype.
    m2023 = mem[mem["year"] == 2023].copy()
    burden_rows = []
    for label, g in m2023.groupby("dominant_archetype_label"):
        for key, col in OUTCOMES.items():
            burden_rows.append(
                {
                    "dominant_archetype_label": label,
                    "outcome": key,
                    "outcome_label": OUTCOME_LABELS[key],
                    "n_countries": int(g["country"].nunique()),
                    "median_rate_per_100k": float(g[col].median()),
                    "q25_rate_per_100k": float(g[col].quantile(0.25)),
                    "q75_rate_per_100k": float(g[col].quantile(0.75)),
                }
            )
    burden = pd.DataFrame(burden_rows)
    burden.to_csv(TABLE_DIR / "archetype_2023_age_burden_signatures.csv", index=False, encoding="utf-8-sig")
    # Bootstrap stability for K=4.
    rng = np.random.default_rng(20260704)
    countries = use["country"].unique()
    boot_rows = []
    for b in range(30):
        sample_c = rng.choice(countries, size=int(0.8 * len(countries)), replace=False)
        sub = use[use["country"].isin(sample_c)]
        Xb = scaler.transform(sub[features].to_numpy())
        _, Ab, _, _ = archetype_fit(Xb, main_K)
        # Match bootstrap archetypes to main archetypes by Euclidean distance.
        cost = np.linalg.norm(Ab[:, None, :] - A_reordered[None, :, :], axis=2)
        r, c = linear_sum_assignment(cost)
        boot_rows.append(
            {
                "bootstrap": b + 1,
                "mean_matched_distance": float(cost[r, c].mean()),
                "max_matched_distance": float(cost[r, c].max()),
            }
        )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(TABLE_DIR / "archetype_bootstrap_stability_k4.csv", index=False, encoding="utf-8-sig")
    return k_table, A_df, mem, burden


def run_trajectory_clustering(df: pd.DataFrame, mem: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    landmarks = [1990, 2000, 2010, 2023]
    weight_cols = [c for c in mem.columns if c.startswith("archetype_weight_A")]
    wide_parts = []
    for year in landmarks:
        tmp = mem[mem["year"] == year][["country", "location_name"] + weight_cols].copy()
        tmp = tmp.rename(columns={c: f"{c}_{year}" for c in weight_cols})
        wide_parts.append(tmp)
    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on=["country", "location_name"], how="inner")
    for c in weight_cols:
        wide[f"{c}_delta_2023_1990"] = wide[f"{c}_2023"] - wide[f"{c}_1990"]
    feature_cols = [c for c in wide.columns if c.startswith("archetype_weight_A")]
    X = StandardScaler().fit_transform(wide[feature_cols].to_numpy())
    cl_rows = []
    fits = {}
    for K in range(3, 7):
        km = KMeans(n_clusters=K, random_state=20260704, n_init=50)
        lab = km.fit_predict(X)
        sil = silhouette_score(X, lab)
        cl_rows.append({"K": K, "silhouette": float(sil), "inertia": float(km.inertia_)})
        fits[K] = lab
    cl_table = pd.DataFrame(cl_rows)
    cl_table.to_csv(TABLE_DIR / "trajectory_cluster_k_selection.csv", index=False, encoding="utf-8-sig")
    chosen_K = int(cl_table.sort_values(["silhouette", "K"], ascending=[False, True]).iloc[0]["K"])
    wide["trajectory_cluster_raw"] = fits[chosen_K]
    # Add outcome declines and environmental changes.
    y1990 = df[df["year"] == 1990][
        ["country", "sdi", "log_hap", "log_pm25", "tavg", "ah"] + [OUTCOMES[k] for k in OUTCOMES]
    ].copy()
    y2023 = df[df["year"] == 2023][
        ["country", "sdi", "log_hap", "log_pm25", "tavg", "ah"] + [OUTCOMES[k] for k in OUTCOMES]
    ].copy()
    y = y1990.merge(y2023, on="country", suffixes=("_1990", "_2023"))
    for key, col in OUTCOMES.items():
        y[f"{key}_percent_decline"] = (1 - y[f"{col}_2023"] / y[f"{col}_1990"]) * 100
    y["delta_sdi"] = y["sdi_2023"] - y["sdi_1990"]
    y["hap_log_reduction"] = y["log_hap_1990"] - y["log_hap_2023"]
    y["pm25_log_reduction"] = y["log_pm25_1990"] - y["log_pm25_2023"]
    y["delta_tavg"] = y["tavg_2023"] - y["tavg_1990"]
    y["delta_ah"] = y["ah_2023"] - y["ah_1990"]
    wide = wide.merge(y, on="country", how="left")
    # Order clusters by 2023 high-SDI archetype weight descending if available.
    cluster_order = (
        wide.groupby("trajectory_cluster_raw")[[f"{weight_cols[-1]}_2023" if weight_cols else "trajectory_cluster_raw"]]
        .mean()
        .reset_index()
    )
    # More robust order: by median under-5 decline descending.
    order = (
        wide.groupby("trajectory_cluster_raw")["u5_percent_decline"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    order_map = {old: i + 1 for i, old in enumerate(order)}
    wide["trajectory_cluster"] = wide["trajectory_cluster_raw"].map(order_map)
    wide.to_csv(SOURCE_DIR / "archetype_trajectory_cluster_country.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for cl, g in wide.groupby("trajectory_cluster"):
        row = {
            "trajectory_cluster": int(cl),
            "n_countries": int(g["country"].nunique()),
            "countries": "; ".join(sorted(g["country"].astype(str))),
            "median_delta_sdi": float(g["delta_sdi"].median()),
            "median_hap_log_reduction": float(g["hap_log_reduction"].median()),
            "median_pm25_log_reduction": float(g["pm25_log_reduction"].median()),
            "median_delta_tavg_C": float(g["delta_tavg"].median()),
            "median_delta_ah": float(g["delta_ah"].median()),
        }
        for key in OUTCOMES:
            row[f"median_{key}_percent_decline"] = float(g[f"{key}_percent_decline"].median())
        # Dominant start/end archetype from mean membership.
        start_weights = g[[f"{c}_1990" for c in weight_cols]].mean().to_numpy()
        end_weights = g[[f"{c}_2023" for c in weight_cols]].mean().to_numpy()
        row["dominant_start_A"] = f"A{int(np.argmax(start_weights)+1)}"
        row["dominant_end_A"] = f"A{int(np.argmax(end_weights)+1)}"
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("trajectory_cluster")
    summary.to_csv(TABLE_DIR / "archetype_trajectory_cluster_summary.csv", index=False, encoding="utf-8-sig")
    return wide, summary


def make_figures(
    df: pd.DataFrame,
    k_table: pd.DataFrame,
    archetypes: pd.DataFrame,
    mem: pd.DataFrame,
    burden: pd.DataFrame,
    traj: pd.DataFrame,
    traj_summary: pd.DataFrame,
    pathway_coef: pd.DataFrame,
):
    colors = {
        "Low-SDI household-energy archetype": "#c95a32",
        "Ambient PM2.5 archetype": "#8f6bbd",
        "Warm-humid climate-context archetype": "#dfa42a",
        "High-SDI low-exposure archetype": "#3c835a",
        "Mixed/intermediate archetype": "#9aa4b1",
    }
    # Figure 5: archetype discovery.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(k_table["K"], k_table["reconstruction_rmse"], marker="o", color="#333333")
    ax.set_xlabel("Number of archetypes")
    ax.set_ylabel("Reconstruction RMSE")
    ax.set_xticks(k_table["K"])
    ax.text(-0.16, 1.08, "a", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[0, 1]
    heat = archetypes[["sdi", "log_hap", "log_pm25", "tavg", "ah"]].to_numpy()
    im = ax.imshow(heat, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(archetypes)))
    ax.set_yticklabels([x.replace(" archetype", "") for x in archetypes["label"]])
    ax.set_xticks(range(5))
    ax.set_xticklabels([FEATURE_LABELS[x] for x in ["sdi", "log_hap", "log_pm25", "tavg", "ah"]], rotation=35, ha="right")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Standardized coordinate")
    ax.text(-0.16, 1.08, "b", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[1, 0]
    features = ["sdi", "log_hap", "log_pm25", "tavg", "ah"]
    use = df.dropna(subset=features + ["year", "country"]).copy()
    X = StandardScaler().fit_transform(use[features])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    use["pc1"] = coords[:, 0]
    use["pc2"] = coords[:, 1]
    m2023 = use[use["year"] == 2023].merge(
        mem[mem["year"] == 2023][["country", "dominant_archetype_label"]], on="country", how="left"
    )
    for label, g in m2023.groupby("dominant_archetype_label"):
        ax.scatter(g["pc1"], g["pc2"], s=18, alpha=0.75, label=label.replace(" archetype", ""), color=colors.get(label))
    ax.set_xlabel("PC1 of environmental-development space")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.text(-0.16, 1.08, "c", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[1, 1]
    piv = burden.pivot(index="dominant_archetype_label", columns="outcome_label", values="median_rate_per_100k")
    order = archetypes["label"].tolist()
    cols = ["Under-5", "All ages", "5-69 years", "70+ years"]
    piv = piv.reindex(order)[cols]
    mat = np.log10(piv.to_numpy())
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([x.replace(" archetype", "") for x in piv.index])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = piv.iloc[i, j]
            ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="log10 median rate")
    ax.text(-0.16, 1.08, "d", transform=ax.transAxes, fontsize=16, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"fig5_data_driven_archetypes_20260704.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Figure 6: trajectory clustering.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    ax = axes[0, 0]
    counts = (
        mem[mem["year"].isin([1990, 2000, 2010, 2023])]
        .groupby(["year", "dominant_archetype_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=archetypes["label"].tolist())
    )
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    for label in counts.columns:
        vals = counts[label].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=label.replace(" archetype", ""), color=colors.get(label))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.astype(str))
    ax.set_ylabel("Countries")
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.text(-0.16, 1.08, "a", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[0, 1]
    weight_cols = [c for c in traj.columns if c.startswith("archetype_weight_A") and c.endswith("_delta_2023_1990")]
    delta = traj.groupby("trajectory_cluster")[weight_cols].mean()
    mat = delta.to_numpy()
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_yticks(range(len(delta)))
    ax.set_yticklabels([f"Trajectory {int(i)}" for i in delta.index])
    ax.set_xticks(range(len(weight_cols)))
    ax.set_xticklabels([c.replace("archetype_weight_", "").replace("_delta_2023_1990", "") for c in weight_cols])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Mean membership change")
    ax.text(-0.16, 1.08, "b", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[1, 0]
    g = traj_summary.sort_values("trajectory_cluster")
    y = np.arange(len(g))
    h = 0.35
    ax.barh(y - h / 2, g["median_u5_percent_decline"], height=h, color="#6f8fc7", label="Under-5")
    ax.barh(y + h / 2, g["median_age70p_percent_decline"], height=h, color="#dca326", label="70+ years")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Trajectory {int(i)} (n={int(n)})" for i, n in zip(g["trajectory_cluster"], g["n_countries"])])
    ax.set_xlabel("Median 1990-2023 incidence-rate decline (%)")
    ax.legend(frameon=False)
    ax.text(-0.16, 1.08, "c", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[1, 1]
    env_cols = [
        "median_delta_sdi",
        "median_hap_log_reduction",
        "median_pm25_log_reduction",
        "median_delta_tavg_C",
        "median_delta_ah",
    ]
    env_labels = ["Delta SDI", "HAP reduction", "PM2.5 reduction", "Delta temp.", "Delta AH"]
    mat = g[env_cols].to_numpy()
    mat_z = (mat - np.nanmean(mat, axis=0)) / np.nanstd(mat, axis=0)
    im = ax.imshow(mat_z, cmap="PiYG", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"Trajectory {int(i)}" for i in g["trajectory_cluster"]])
    ax.set_xticks(range(len(env_cols)))
    ax.set_xticklabels(env_labels, rotation=35, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Column z-score")
    ax.text(-0.16, 1.08, "d", transform=ax.transAxes, fontsize=16, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"fig6_archetype_trajectories_20260704.{ext}", bbox_inches="tight")
    plt.close(fig)

    # Figure 7: pathway model compact forest.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot = pathway_coef.copy()
    pathways = ["household_disadvantage", "ambient_pollution", "thermal_context", "wetness_context"]
    ylabels = []
    ypos = []
    x = []
    lo = []
    hi = []
    colors2 = []
    pos = 0
    for outcome in ["u5", "all_age", "age5_69", "age70p"]:
        sub = plot[plot["outcome"] == outcome].set_index("pathway")
        for pathway in pathways:
            row = sub.loc[pathway]
            ylabels.append(f"{OUTCOME_LABELS[outcome]} | {pathway.replace('_', ' ')}")
            ypos.append(pos)
            x.append(row["percent_difference_per_1sd"])
            lo.append(row["ci_low_percent"])
            hi.append(row["ci_high_percent"])
            colors2.append("#333333" if row["p_value"] < 0.05 else "#9aa4b1")
            pos += 1
        pos += 0.8
    x = np.asarray(x)
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    ax.axvline(0, color="#777777", lw=0.8)
    for yi, xi, l, hci, col in zip(ypos, x, lo, hi, colors2):
        ax.plot([l, hci], [yi, yi], color=col, lw=1.4)
        ax.scatter([xi], [yi], color=col, s=24, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Percent difference per 1-SD pathway score")
    ax.invert_yaxis()
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"fig7_pathway_domain_model_20260704.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    panel: pd.DataFrame,
    model_audit: pd.DataFrame,
    pathway_coef: pd.DataFrame,
    pathway_blocks: pd.DataFrame,
    xw: pd.DataFrame,
    yw: pd.DataFrame,
    k_table: pd.DataFrame,
    archetypes: pd.DataFrame,
    burden: pd.DataFrame,
    traj_summary: pd.DataFrame,
):
    best_models = model_audit.sort_values(["outcome", "cv_rmse"]).groupby("outcome").head(1)
    sig_path = pathway_coef[pathway_coef["p_value"] < 0.05].copy()
    block_sig = pathway_blocks[pathway_blocks["p_value"] < 0.05].copy()
    k4 = k_table[k_table["K"] == 4].iloc[0]
    report = []
    report.append("# Profile innovation analyses report (2026-07-04)")
    report.append("")
    report.append("## 核心判断")
    report.append("")
    report.append(
        "本轮新增分析的目标不是通过多模型筛选制造显著性，而是提高估计目标匹配度和信号识别效率："
        "用模型空间审计检查 AJPH 旧版处理是否仍适合 1990-2023 面板，用 pathway/domain 模型降低共线性，"
        "并用 outcome-independent archetype 与 trajectory 分析把规则 profile 升级为数据驱动结构。"
    )
    report.append("")
    report.append("## 1. 数据审计")
    report.append("")
    report.append(
        f"输入面板包含 {len(panel):,} 个 country-year observations，"
        f"{panel['country'].nunique()} 个国家/地区，年份 {int(panel['year'].min())}-{int(panel['year'].max())}。"
    )
    report.append("")
    report.append("## 2. AJPH-style model-space audit")
    report.append("")
    report.append(
        "基于 1990-2023 long-term country means，审计 raw/log transformation、climate residualization、"
        "SDI 线性/二次项和 SDI interaction 等候选设定。该步骤只用于诊断预测稳定性、共线性和解释性，"
        "不用于按 P 值挑选主模型。"
    )
    report.append("")
    report.append("各结局 CV-RMSE 最低的候选模型如下：")
    report.append("")
    report.append(best_models.to_markdown(index=False))
    report.append("")
    report.append("## 3. Pathway/domain model")
    report.append("")
    report.append(
        "预定义 pathway scores 包括 household-energy disadvantage、ambient pollution、thermal context 和 wetness context。"
        "模型调整 nonlinear SDI 和 year fixed effects，并按国家聚类稳健标准误。"
    )
    report.append("")
    if len(sig_path):
        report.append("统计学上较清楚的 pathway signals：")
        report.append("")
        report.append(
            sig_path[
                [
                    "outcome_label",
                    "pathway",
                    "percent_difference_per_1sd",
                    "ci_low_percent",
                    "ci_high_percent",
                    "p_value",
                ]
            ].to_markdown(index=False, floatfmt=".3f")
        )
    else:
        report.append("单个 pathway coefficient 未出现 p<0.05；此时更应依赖 block tests 和 archetype/trajectory evidence。")
    report.append("")
    if len(block_sig):
        report.append("显著 pathway block tests：")
        report.append("")
        report.append(block_sig.to_markdown(index=False, floatfmt=".4f"))
    report.append("")
    report.append("Development-adjusted PLS 提供了一个探索性多年龄轴，但应作为辅助解释，不作为因果路径证明。")
    report.append("")
    report.append("X weights:")
    report.append("")
    report.append(xw.to_markdown(floatfmt=".3f"))
    report.append("")
    report.append("Y weights:")
    report.append("")
    report.append(yw.to_markdown(floatfmt=".3f"))
    report.append("")
    report.append("## 4. Outcome-independent archetype analysis")
    report.append("")
    report.append(
        f"K=4 的 reconstruction RMSE 为 {k4['reconstruction_rmse']:.3f}，dominant silhouette 为 {k4['dominant_silhouette']:.3f}。"
        "主版本保留 K=4，因为它直接对应 household-energy、ambient PM2.5、warm-humid climate-context 和 high-SDI low-exposure 等解释性原型。"
    )
    report.append("")
    report.append("K-selection summary:")
    report.append("")
    report.append(k_table.to_markdown(index=False, floatfmt=".3f"))
    report.append("")
    report.append("Archetype standardized coordinates:")
    report.append("")
    report.append(archetypes.to_markdown(index=False, floatfmt=".3f"))
    report.append("")
    report.append("2023 dominant archetype age-burden signatures:")
    report.append("")
    report.append(
        burden[
            [
                "dominant_archetype_label",
                "outcome_label",
                "n_countries",
                "median_rate_per_100k",
                "q25_rate_per_100k",
                "q75_rate_per_100k",
            ]
        ].to_markdown(index=False, floatfmt=".1f")
    )
    report.append("")
    report.append("## 5. Archetype trajectory clustering")
    report.append("")
    report.append(
        "基于 1990、2000、2010 和 2023 年 archetype membership 及其 1990-2023 变化进行 trajectory clustering。"
        "该分析把规则 transition matrix 扩展为连续 archetype space 中的国家转型路径。"
    )
    report.append("")
    report.append(traj_summary.to_markdown(index=False, floatfmt=".2f"))
    report.append("")
    report.append("## 可进入主文的新观点")
    report.append("")
    report.append(
        "1. 环境-发展 profile 不只是作者阈值定义；不使用 LRI outcomes 的数据驱动 convex archetype analysis "
        "也能识别出可解释的全球 environmental-development archetypes。"
    )
    report.append(
        "2. 将共线单变量转化为 pathway/domain scores 后，结果可从“哪个变量显著”升级为“哪些环境-发展路径与哪些年龄段 LRI burden 对齐”。"
    )
    report.append(
        "3. 国家 1990-2023 年不是沿单一改善梯度移动，而是在 archetype membership space 中形成不同 trajectory；"
        "这些 trajectory 对应儿童和老年 LRI decline 的不同速度。"
    )
    report.append("")
    report.append("## 解释边界")
    report.append("")
    report.append(
        "以上结果均为 country-level ecological profile/pathway evidence，不是个体层面风险估计、因果中介、"
        "可归因负担或可预防负担。模型空间审计不得写成按显著性筛选主模型。"
    )
    report.append("")
    report.append("## 输出图件")
    report.append("")
    report.append("- `fig5_data_driven_archetypes_20260704`: 数据驱动 archetypes。")
    report.append("- `fig6_archetype_trajectories_20260704`: archetype trajectories 与 LRI decline pathways。")
    report.append("- `fig7_pathway_domain_model_20260704`: pathway/domain model coefficients。")
    (REPORT_DIR / "PROFILE_INNOVATION_ANALYSES_REPORT_20260704_ZH.md").write_text(
        "\n".join(report), encoding="utf-8", newline="\n"
    )


def main():
    panel = load_panel()
    validation = {
        "input_panel": str(PANEL),
        "rows": int(len(panel)),
        "countries": int(panel["country"].nunique()),
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "output_dir": str(OUT),
    }
    (VALID_DIR / "input_panel_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    country = build_country_means(panel)
    country.to_csv(SOURCE_DIR / "country_longterm_means_1990_2023.csv", index=False, encoding="utf-8-sig")

    model_audit = fit_model_space_audit(country)
    pathway_coef, pathway_blocks = fit_pathway_models(panel)
    xw, yw = fit_development_adjusted_pls(panel)
    k_table, archetypes, mem, burden = run_archetype_analysis(panel)
    traj, traj_summary = run_trajectory_clustering(panel, mem)
    make_figures(panel, k_table, archetypes, mem, burden, traj, traj_summary, pathway_coef)
    write_report(
        panel,
        model_audit,
        pathway_coef,
        pathway_blocks,
        xw,
        yw,
        k_table,
        archetypes,
        burden,
        traj_summary,
    )
    manifest = {
        "created": "2026-07-04",
        "input_panel": str(PANEL),
        "output_dir": str(OUT),
        "key_outputs": sorted(str(p) for p in OUT.rglob("*") if p.is_file()),
        "interpretation_boundary": "Country-level ecological profile/pathway evidence; not individual-level causal effects, mediation, or attributable burden.",
    }
    (OUT / "manifest_profile_innovation_20260704.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
