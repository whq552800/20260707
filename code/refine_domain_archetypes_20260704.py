# Public-release note: paths were made relative to the release root. Place required upstream input data under the expected data/ subdirectories, or edit PANEL/OUT below for your local environment.
# -*- coding: utf-8 -*-
"""Refine outcome-independent archetype analysis in prespecified domain space.

This script is intentionally separate from
``run_profile_innovation_analyses_20260704.py``.  The first pass showed that
raw-variable archetypes split high-SDI low-exposure settings into cold and warm
variants rather than isolating the warm-humid profile clearly.  Here we move to
prespecified environmental-development domain scores:

1. household-energy disadvantage
2. ambient pollution
3. thermal climate context
4. wetness climate context
5. high-SDI low-exposure development context

No LRI outcome is used to define archetypes or memberships.  LRI outcomes are
summarized only after classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from run_profile_innovation_analyses_20260704 import (
    FIG_DIR,
    OUT,
    OUTCOMES,
    OUTCOME_LABELS,
    SOURCE_DIR,
    TABLE_DIR,
    archetype_fit,
    build_domain_scores,
    load_panel,
    simplex_weights,
    zscore,
)


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


DOMAIN_FEATURES = [
    "household_energy_disadvantage",
    "ambient_pm25_context",
    "thermal_context",
    "wetness_context",
    "high_sdi_low_exposure_context",
]

DOMAIN_LABELS = {
    "household_energy_disadvantage": "Household-energy disadvantage",
    "ambient_pm25_context": "Ambient PM2.5 context",
    "thermal_context": "Thermal context",
    "wetness_context": "Wetness context",
    "high_sdi_low_exposure_context": "High-SDI low-exposure context",
}


def build_domain_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Build prespecified domain scores and standardize them for archetypes."""
    d = build_domain_scores(panel).copy()
    d["household_energy_disadvantage"] = d["path_household_disadvantage"]
    d["ambient_pm25_context"] = d["path_ambient_pollution"]
    d["thermal_context"] = d["path_thermal_context"]
    d["wetness_context"] = d["path_wetness_context"]
    # High values represent advanced development with low household and ambient pollution.
    d["high_sdi_low_exposure_context"] = (
        d["z_sdi"] - d["z_log_hap"] - d["z_log_pm25"]
    ) / 3.0
    for col in DOMAIN_FEATURES:
        d[f"std_{col}"] = zscore(d[col])
    return d


def label_domain_archetypes(A: pd.DataFrame) -> list[str]:
    """Assign concise interpretive labels from archetype domain coordinates."""
    rows = []
    for _, row in A.iterrows():
        vals = {col: float(row[col]) for col in DOMAIN_FEATURES}
        high_sdi = vals["high_sdi_low_exposure_context"]
        household = vals["household_energy_disadvantage"]
        ambient = vals["ambient_pm25_context"]
        thermal = vals["thermal_context"]
        wet = vals["wetness_context"]
        climate = (thermal + wet) / 2.0
        if household > 1.0 and ambient > 1.0 and wet < -1.0 and high_sdi < -1.0:
            label = "Low-SDI high-PM dry domain"
        elif ambient > 0.8 and thermal < -1.0 and wet < -1.0:
            label = "Cold-dry PM2.5 domain"
        elif ambient > 0.8 and thermal > 0.5 and wet < -1.0:
            label = "Hot-dry PM2.5 domain"
        elif household >= max(ambient, thermal, wet, high_sdi) and household > 0.35:
            label = "Low-SDI household-energy domain"
        elif ambient >= max(household, thermal, wet, high_sdi) and ambient > 0.35:
            label = "Ambient PM2.5 domain"
        elif high_sdi >= max(household, ambient, thermal, wet) and high_sdi > 0.35:
            label = "High-SDI low-exposure domain"
        elif climate > 0.35 and thermal > 0.2 and wet > 0.2:
            label = "Warm-humid climate domain"
        elif thermal > 0.5:
            label = "Thermal climate domain"
        else:
            label = "Mixed/intermediate domain"
        rows.append(label)

    seen: dict[str, int] = {}
    out: list[str] = []
    for label in rows:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} {seen[label]}")
    return out


def run_domain_archetypes(panel: pd.DataFrame, K: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = build_domain_frame(panel)
    use = d.dropna(subset=[f"std_{c}" for c in DOMAIN_FEATURES] + ["country", "year"]).copy()
    X = use[[f"std_{c}" for c in DOMAIN_FEATURES]].to_numpy()

    rows = []
    fits = {}
    for k in range(3, 8):
        idx, A, W, rmse = archetype_fit(X, k)
        dom = W.argmax(axis=1)
        sil = silhouette_score(X, dom) if len(np.unique(dom)) > 1 else np.nan
        rows.append({"K": k, "reconstruction_rmse": rmse, "dominant_silhouette": sil})
        fits[k] = (idx, A, W, rmse)
    k_table = pd.DataFrame(rows)
    k_table.to_csv(TABLE_DIR / "domain_archetype_k_selection.csv", index=False, encoding="utf-8-sig")

    idx, A, W, _ = fits[K]
    A_df = pd.DataFrame(A, columns=DOMAIN_FEATURES)
    A_df.insert(0, "archetype_id", [f"D{i+1}" for i in range(K)])
    A_df["label"] = label_domain_archetypes(A_df[DOMAIN_FEATURES])
    for j, obs_idx in enumerate(idx):
        A_df.loc[j, "exemplar_country"] = use.iloc[obs_idx]["location_name"]
        A_df.loc[j, "exemplar_year"] = int(use.iloc[obs_idx]["year"])

    preferred = [
        "Low-SDI household-energy domain",
        "Low-SDI high-PM dry domain",
        "Ambient PM2.5 domain",
        "Cold-dry PM2.5 domain",
        "Hot-dry PM2.5 domain",
        "Warm-humid climate domain",
        "Thermal climate domain",
        "High-SDI low-exposure domain",
        "Mixed/intermediate domain",
    ]

    def rank_label(label: str) -> tuple[int, str]:
        base = label.rsplit(" ", 1)[0] if label.rsplit(" ", 1)[-1].isdigit() else label
        return (preferred.index(base) if base in preferred else 99, label)

    order = sorted(range(len(A_df)), key=lambda i: rank_label(A_df.loc[i, "label"]))
    A_df = A_df.iloc[order].reset_index(drop=True)
    W = W[:, order]
    A_df["archetype_id"] = [f"D{i+1}" for i in range(K)]
    A_df.to_csv(TABLE_DIR / f"domain_archetype_loadings_k{K}.csv", index=False, encoding="utf-8-sig")

    mem = use[["country", "location_name", "year"]].copy()
    for j in range(K):
        mem[f"domain_archetype_weight_D{j+1}"] = W[:, j]
    dom = W.argmax(axis=1)
    mem["dominant_domain_archetype_id"] = [f"D{i+1}" for i in dom]
    mem["dominant_domain_archetype_label"] = [A_df.loc[i, "label"] for i in dom]
    mem = mem.merge(panel[["country", "year"] + [OUTCOMES[k] for k in OUTCOMES]], on=["country", "year"], how="left")
    mem.to_csv(SOURCE_DIR / f"domain_archetype_membership_k{K}_country_year.csv", index=False, encoding="utf-8-sig")

    burden_rows = []
    m2023 = mem[mem["year"] == 2023].copy()
    for label, g in m2023.groupby("dominant_domain_archetype_label"):
        for key, col in OUTCOMES.items():
            burden_rows.append(
                {
                    "dominant_domain_archetype_label": label,
                    "outcome": key,
                    "outcome_label": OUTCOME_LABELS[key],
                    "n_countries": int(g["country"].nunique()),
                    "median_rate_per_100k": float(g[col].median()),
                    "q25_rate_per_100k": float(g[col].quantile(0.25)),
                    "q75_rate_per_100k": float(g[col].quantile(0.75)),
                }
            )
    burden = pd.DataFrame(burden_rows)
    burden.to_csv(TABLE_DIR / f"domain_archetype_2023_age_burden_signatures_k{K}.csv", index=False, encoding="utf-8-sig")

    return k_table, A_df, mem, burden


def macro_domain_label(label: str) -> str:
    """Collapse micro-archetypes into reader-facing macro domains."""
    if label.startswith("Low-SDI household-energy"):
        return "Low-SDI household-energy"
    if label.startswith("Low-SDI high-PM dry"):
        return "Low-SDI high-PM dry compound"
    if label.startswith("Cold-dry PM2.5") or label.startswith("Hot-dry PM2.5") or label.startswith("Ambient PM2.5"):
        return "PM2.5 dry-climate"
    if label.startswith("Warm-humid") or label.startswith("Thermal"):
        return "Warm-humid climate-context"
    if label.startswith("High-SDI low-exposure"):
        return "High-SDI low-exposure"
    return "Mixed/intermediate"


def summarize_macro_domains(panel: pd.DataFrame, mem: pd.DataFrame, K: int) -> pd.DataFrame:
    """Summarize 2023 LRI burden after collapsing micro-archetypes."""
    m = mem[mem["year"] == 2023].copy()
    m["macro_domain"] = m["dominant_domain_archetype_label"].map(macro_domain_label)
    rows = []
    for label, g in m.groupby("macro_domain"):
        micro = "; ".join(sorted(g["dominant_domain_archetype_label"].unique()))
        for key, col in OUTCOMES.items():
            rows.append(
                {
                    "macro_domain": label,
                    "micro_domains": micro,
                    "outcome": key,
                    "outcome_label": OUTCOME_LABELS[key],
                    "n_countries": int(g["country"].nunique()),
                    "median_rate_per_100k": float(g[col].median()),
                    "q25_rate_per_100k": float(g[col].quantile(0.25)),
                    "q75_rate_per_100k": float(g[col].quantile(0.75)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / f"domain_macro_2023_age_burden_signatures_k{K}.csv", index=False, encoding="utf-8-sig")
    return out


def run_domain_trajectory(panel: pd.DataFrame, mem: pd.DataFrame, K: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    landmarks = [1990, 2000, 2010, 2023]
    weight_cols = [c for c in mem.columns if c.startswith("domain_archetype_weight_D")]
    wide = None
    for year in landmarks:
        tmp = mem[mem["year"] == year][["country", "location_name"] + weight_cols].copy()
        tmp = tmp.rename(columns={c: f"{c}_{year}" for c in weight_cols})
        wide = tmp if wide is None else wide.merge(tmp, on=["country", "location_name"], how="inner")
    assert wide is not None
    for c in weight_cols:
        wide[f"{c}_delta_2023_1990"] = wide[f"{c}_2023"] - wide[f"{c}_1990"]
    feature_cols = [c for c in wide.columns if c.startswith("domain_archetype_weight_D")]
    X = StandardScaler().fit_transform(wide[feature_cols].to_numpy())

    rows = []
    fits = {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=20260704, n_init=100)
        lab = km.fit_predict(X)
        rows.append({"K": k, "silhouette": float(silhouette_score(X, lab)), "inertia": float(km.inertia_)})
        fits[k] = lab
    cl_table = pd.DataFrame(rows)
    cl_table.to_csv(TABLE_DIR / "domain_trajectory_cluster_k_selection.csv", index=False, encoding="utf-8-sig")
    chosen_k = int(cl_table.sort_values(["silhouette", "K"], ascending=[False, True]).iloc[0]["K"])
    wide["domain_trajectory_cluster_raw"] = fits[chosen_k]

    y1990 = panel[panel["year"] == 1990][
        ["country", "sdi", "log_hap", "log_pm25", "tavg", "ah"] + [OUTCOMES[k] for k in OUTCOMES]
    ].copy()
    y2023 = panel[panel["year"] == 2023][
        ["country", "sdi", "log_hap", "log_pm25", "tavg", "ah"] + [OUTCOMES[k] for k in OUTCOMES]
    ].copy()
    y = y1990.merge(y2023, on="country", suffixes=("_1990", "_2023"))
    for key, col in OUTCOMES.items():
        y[f"{key}_percent_decline"] = (1.0 - y[f"{col}_2023"] / y[f"{col}_1990"]) * 100.0
    y["delta_sdi"] = y["sdi_2023"] - y["sdi_1990"]
    y["hap_log_reduction"] = y["log_hap_1990"] - y["log_hap_2023"]
    y["pm25_log_reduction"] = y["log_pm25_1990"] - y["log_pm25_2023"]
    y["delta_tavg"] = y["tavg_2023"] - y["tavg_1990"]
    y["delta_ah"] = y["ah_2023"] - y["ah_1990"]
    wide = wide.merge(y, on="country", how="left")

    order = (
        wide.groupby("domain_trajectory_cluster_raw")["u5_percent_decline"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    order_map = {old: i + 1 for i, old in enumerate(order)}
    wide["domain_trajectory_cluster"] = wide["domain_trajectory_cluster_raw"].map(order_map)
    wide.to_csv(SOURCE_DIR / f"domain_archetype_trajectory_country_k{K}.csv", index=False, encoding="utf-8-sig")

    rows2 = []
    for cl, g in wide.groupby("domain_trajectory_cluster"):
        row = {
            "domain_trajectory_cluster": int(cl),
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
        start = g[[f"{c}_1990" for c in weight_cols]].mean().to_numpy()
        end = g[[f"{c}_2023" for c in weight_cols]].mean().to_numpy()
        row["dominant_start_D"] = f"D{int(np.argmax(start) + 1)}"
        row["dominant_end_D"] = f"D{int(np.argmax(end) + 1)}"
        rows2.append(row)
    summary = pd.DataFrame(rows2).sort_values("domain_trajectory_cluster")
    summary.to_csv(TABLE_DIR / f"domain_archetype_trajectory_summary_k{K}.csv", index=False, encoding="utf-8-sig")
    return wide, summary


def make_domain_figures(panel: pd.DataFrame, k_table: pd.DataFrame, archetypes: pd.DataFrame, mem: pd.DataFrame, burden: pd.DataFrame, traj_summary: pd.DataFrame, K: int = 5) -> None:
    colors = {
        "Low-SDI household-energy domain": "#c95a32",
        "Low-SDI high-PM dry domain": "#a63f2b",
        "Ambient PM2.5 domain": "#8f6bbd",
        "Cold-dry PM2.5 domain": "#6d5aa9",
        "Hot-dry PM2.5 domain": "#b77cc4",
        "Warm-humid climate domain": "#dca326",
        "Thermal climate domain": "#dca326",
        "High-SDI low-exposure domain": "#3c835a",
        "Mixed/intermediate domain": "#9aa4b1",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0))

    ax = axes[0, 0]
    ax.plot(k_table["K"], k_table["reconstruction_rmse"], marker="o", color="#333333")
    ax.set_xlabel("Number of archetypes")
    ax.set_ylabel("Reconstruction RMSE")
    ax.set_xticks(k_table["K"])
    ax.text(-0.16, 1.08, "a", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[0, 1]
    mat = archetypes[DOMAIN_FEATURES].to_numpy()
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(archetypes)))
    ax.set_yticklabels([x.replace(" domain", "") for x in archetypes["label"]])
    ax.set_xticks(range(len(DOMAIN_FEATURES)))
    ax.set_xticklabels([DOMAIN_LABELS[c] for c in DOMAIN_FEATURES], rotation=35, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Standardized domain coordinate")
    ax.text(-0.16, 1.08, "b", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[1, 0]
    d = build_domain_frame(panel)
    use = d.dropna(subset=[f"std_{c}" for c in DOMAIN_FEATURES] + ["year", "country"]).copy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(use[[f"std_{c}" for c in DOMAIN_FEATURES]].to_numpy())
    use["pc1"] = coords[:, 0]
    use["pc2"] = coords[:, 1]
    m2023 = use[use["year"] == 2023].merge(
        mem[mem["year"] == 2023][["country", "dominant_domain_archetype_label"]],
        on="country",
        how="left",
    )
    for label, g in m2023.groupby("dominant_domain_archetype_label"):
        ax.scatter(g["pc1"], g["pc2"], s=19, alpha=0.75, label=label.replace(" domain", ""), color=colors.get(label, "#777777"))
    ax.set_xlabel("PC1 of domain-score space")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.text(-0.16, 1.08, "c", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[1, 1]
    piv = burden.pivot(index="dominant_domain_archetype_label", columns="outcome_label", values="median_rate_per_100k")
    order = archetypes["label"].tolist()
    cols = ["Under-5", "All ages", "5-69 years", "70+ years"]
    piv = piv.reindex(order)[cols]
    hm = np.log10(piv.to_numpy())
    im = ax.imshow(hm, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([x.replace(" domain", "") for x in piv.index])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f"{piv.iloc[i, j]:,.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="log10 median rate")
    ax.text(-0.16, 1.08, "d", transform=ax.transAxes, fontsize=16, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"fig8_domain_archetypes_k{K}_20260704.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    counts = (
        mem[mem["year"].isin([1990, 2000, 2010, 2023])]
        .groupby(["year", "dominant_domain_archetype_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=archetypes["label"].tolist())
    )
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    for label in counts.columns:
        vals = counts[label].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=label.replace(" domain", ""), color=colors.get(label, "#777777"))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.astype(str))
    ax.set_ylabel("Countries")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.text(-0.13, 1.08, "a", transform=ax.transAxes, fontsize=16, fontweight="bold")

    ax = axes[1]
    g = traj_summary.sort_values("domain_trajectory_cluster")
    y = np.arange(len(g))
    h = 0.35
    ax.barh(y - h / 2, g["median_u5_percent_decline"], height=h, color="#6f8fc7", label="Under-5")
    ax.barh(y + h / 2, g["median_age70p_percent_decline"], height=h, color="#dca326", label="70+ years")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Trajectory {int(i)} (n={int(n)})" for i, n in zip(g["domain_trajectory_cluster"], g["n_countries"])])
    ax.set_xlabel("Median 1990-2023 incidence-rate decline (%)")
    ax.legend(frameon=False)
    ax.text(-0.13, 1.08, "b", transform=ax.transAxes, fontsize=16, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"fig9_domain_archetype_trajectories_k{K}_20260704.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_domain_report(
    k_table: pd.DataFrame,
    archetypes: pd.DataFrame,
    burden: pd.DataFrame,
    macro_burden: pd.DataFrame,
    traj_summary: pd.DataFrame,
    K: int,
) -> None:
    lines = []
    lines.append(f"# Domain archetype refinement report (K={K}, 2026-07-04)")
    lines.append("")
    lines.append("## 核心判断")
    lines.append("")
    lines.append(
        "第二轮使用预定义 domain scores，而不是原始变量极端点。这样更符合论文问题："
        "先定义环境-发展路径，再在不使用 LRI outcome 的情况下识别原型。"
    )
    lines.append("")
    lines.append("## K selection")
    lines.append("")
    lines.append(k_table.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Domain archetype coordinates")
    lines.append("")
    lines.append(archetypes.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## 2023 age-burden signatures")
    lines.append("")
    lines.append(burden.to_markdown(index=False, floatfmt=".1f"))
    lines.append("")
    lines.append("## Collapsed macro-domain signatures")
    lines.append("")
    lines.append(macro_burden.to_markdown(index=False, floatfmt=".1f"))
    lines.append("")
    lines.append("## Domain trajectory summary")
    lines.append("")
    lines.append(traj_summary.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append("## 解释边界")
    lines.append("")
    lines.append(
        "这些 archetypes 和 trajectories 是 country-level ecological profile evidence。"
        "它们不估计个体风险、因果中介、可归因负担或可预防负担。"
    )
    path = OUT / "reports" / f"DOMAIN_ARCHETYPE_REFINEMENT_REPORT_k{K}_20260704_ZH.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    panel = load_panel()
    K = 6
    k_table, archetypes, mem, burden = run_domain_archetypes(panel, K=K)
    macro_burden = summarize_macro_domains(panel, mem, K=K)
    traj, traj_summary = run_domain_trajectory(panel, mem, K=K)
    make_domain_figures(panel, k_table, archetypes, mem, burden, traj_summary, K=K)
    write_domain_report(k_table, archetypes, burden, macro_burden, traj_summary, K=K)
    manifest_path = OUT / "manifest_domain_archetype_refinement_20260704.json"
    manifest = {
        "created": "2026-07-04",
        "K": K,
        "outputs": [
            str(TABLE_DIR / "domain_archetype_k_selection.csv"),
            str(TABLE_DIR / f"domain_archetype_loadings_k{K}.csv"),
            str(TABLE_DIR / f"domain_archetype_2023_age_burden_signatures_k{K}.csv"),
            str(TABLE_DIR / f"domain_macro_2023_age_burden_signatures_k{K}.csv"),
            str(TABLE_DIR / f"domain_archetype_trajectory_summary_k{K}.csv"),
            str(FIG_DIR / f"fig8_domain_archetypes_k{K}_20260704.png"),
            str(FIG_DIR / f"fig9_domain_archetype_trajectories_k{K}_20260704.png"),
            str(OUT / "reports" / f"DOMAIN_ARCHETYPE_REFINEMENT_REPORT_k{K}_20260704_ZH.md"),
        ],
        "interpretation_boundary": "Outcome-independent country-level ecological profile evidence; not causal mediation, individual risk, or attributable burden.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
