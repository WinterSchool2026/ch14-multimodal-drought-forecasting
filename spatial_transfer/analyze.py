"""Aggregate metrics JSONs into summary tables and plots."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG


def load_all_metrics():
    """Load all metrics JSON files into a DataFrame."""
    metrics_dir = CONFIG["metrics_dir"]
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        row = {
            "model": data["model"],
            "train_region": data["train_region"],
            "test_region": data["test_region"],
            "mean_rmse": data["aggregate"]["mean_rmse"],
            "mean_mae": data["aggregate"]["mean_mae"],
            "r2": data["aggregate"]["r2"],
            "nse": data["aggregate"]["nse"],
            "persistence_rmse": data["baselines"]["persistence"]["mean_rmse"],
            "climatology_rmse": data["baselines"]["climatology"]["mean_rmse"],
            "n_test_cubes": data["n_test_cubes"],
            "n_test_samples": data["n_test_samples"],
            "rmse_per_lt": data["per_leadtime"]["rmse"],
            "mae_per_lt": data["per_leadtime"]["mae"],
        }
        # New NDVI-space metrics
        row["ndvi_r2"] = data["aggregate"].get("ndvi_r2", np.nan)
        row["ndvi_rmse"] = data["aggregate"].get("ndvi_rmse", np.nan)
        row["outperformance_pct"] = data["aggregate"].get("outperformance_pct", np.nan)

        # Skill scores
        row["skill_vs_persistence"] = 1 - row["mean_rmse"] / row["persistence_rmse"] if row["persistence_rmse"] > 0 else 0
        row["skill_vs_climatology"] = 1 - row["mean_rmse"] / row["climatology_rmse"] if row["climatology_rmse"] > 0 else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def summary_table(df):
    """Print and save summary table."""
    cols = ["model", "train_region", "test_region", "ndvi_r2", "ndvi_rmse",
            "outperformance_pct", "mean_rmse", "mean_mae",
            "skill_vs_persistence", "skill_vs_climatology", "n_test_samples"]
    summary = df[cols].copy()
    summary = summary.sort_values(["model", "train_region", "test_region"])

    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    print(summary.to_string(index=False, float_format="%.4f"))

    csv_path = CONFIG["metrics_dir"] / "summary.csv"
    summary.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved to {csv_path}")
    return summary


def transfer_gap_table(df):
    """Compute transfer gap: cross-region metrics minus in-region metrics."""
    print("\n" + "="*120)
    print("TRANSFER GAP TABLE")
    print("="*120)

    rows = []
    for model in df["model"].unique():
        for src_region in CONFIG["regions"]:
            dst_region = [r for r in CONFIG["regions"] if r != src_region][0]
            in_region = df[(df["model"] == model) & (df["train_region"] == src_region) & (df["test_region"] == src_region)]
            cross_region = df[(df["model"] == model) & (df["train_region"] == src_region) & (df["test_region"] == dst_region)]

            if len(in_region) > 0 and len(cross_region) > 0:
                rows.append({
                    "model": model,
                    "train": src_region,
                    "in_ndvi_r2": in_region.iloc[0]["ndvi_r2"],
                    "cross_ndvi_r2": cross_region.iloc[0]["ndvi_r2"],
                    "gap_ndvi_r2": cross_region.iloc[0]["ndvi_r2"] - in_region.iloc[0]["ndvi_r2"],
                    "in_ndvi_rmse": in_region.iloc[0]["ndvi_rmse"],
                    "cross_ndvi_rmse": cross_region.iloc[0]["ndvi_rmse"],
                    "gap_ndvi_rmse": cross_region.iloc[0]["ndvi_rmse"] - in_region.iloc[0]["ndvi_rmse"],
                })

    gap_df = pd.DataFrame(rows)
    print(gap_df.to_string(index=False, float_format="%.4f"))

    csv_path = CONFIG["metrics_dir"] / "transfer_gap.csv"
    gap_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved to {csv_path}")
    return gap_df


def plot_per_leadtime(df):
    """Plot RMSE vs lead time for each experiment condition."""
    conditions = CONFIG["experiment_matrix"]
    models_present = df["model"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle("NDVI Anomaly RMSE per Lead Time (5-day steps)", fontsize=14, fontweight="bold")

    cmap = plt.cm.Set2
    colors = {m: cmap(i) for i, m in enumerate(models_present)}

    for ax, (train_r, test_r) in zip(axes.flat, conditions):
        subset = df[(df["train_region"] == train_r) & (df["test_region"] == test_r)]
        lead_times = np.arange(1, CONFIG["forecast_horizon"] + 1) * 5  # days

        for _, row in subset.iterrows():
            is_in_region = train_r == test_r
            ls = "-" if is_in_region else "--"
            ax.plot(lead_times, row["rmse_per_lt"], marker="o", markersize=3,
                    label=f'{row["model"]} (NDVI R²={row["ndvi_r2"]:.3f})',
                    color=colors.get(row["model"], "gray"), linestyle=ls, linewidth=2)

        # Baselines
        if len(subset) > 0:
            first = subset.iloc[0]
            ax.axhline(first["climatology_rmse"], ls=":", color="gray", alpha=0.6,
                       label="Climatology", linewidth=1.5)

        condition_type = "In-region" if train_r == test_r else "Cross-region"
        ax.set_title(f"{condition_type}: {train_r} → {test_r}", fontsize=12)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel("RMSE (anomaly)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = CONFIG["plots_dir"] / "per_leadtime_rmse.pdf"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


def plot_ndvi_r2_bar(df):
    """Bar chart: NDVI R² by condition, comparing in-region vs cross-region."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models_present = df["model"].unique()
    conditions = CONFIG["experiment_matrix"]
    x = np.arange(len(conditions))
    width = 0.8 / max(len(models_present), 1)
    cmap = plt.cm.Set2

    for i, model in enumerate(models_present):
        r2s = []
        for train_r, test_r in conditions:
            row = df[(df["model"] == model) & (df["train_region"] == train_r) & (df["test_region"] == test_r)]
            r2s.append(row.iloc[0]["ndvi_r2"] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, r2s, width, label=model, color=cmap(i),
                      edgecolor="black", linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0.50, ls=":", color="orange", alpha=0.5, linewidth=1,
               label="R² = 0.50 target")

    ax.set_xticks(x + width * (len(models_present) - 1) / 2)
    ax.set_xticklabels([f"{t} → {te}" for t, te in conditions], fontsize=11)
    ax.set_ylabel("NDVI R²", fontsize=12)
    ax.set_title("Spatial Transfer: NDVI R² by Training/Test Region", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(df["ndvi_r2"].max() * 1.15, 0.75))

    plt.tight_layout()
    path = CONFIG["plots_dir"] / "ndvi_r2_bar.pdf"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


def plot_transfer_heatmap(df):
    """Heatmap: NDVI R² for all model x condition combinations."""
    models_present = sorted(df["model"].unique())
    conditions = CONFIG["experiment_matrix"]

    matrix = np.full((len(models_present), len(conditions)), np.nan)
    for i, model in enumerate(models_present):
        for j, (train_r, test_r) in enumerate(conditions):
            row = df[(df["model"] == model) & (df["train_region"] == train_r) & (df["test_region"] == test_r)]
            if len(row) > 0:
                matrix[i, j] = row.iloc[0]["ndvi_r2"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.8)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([f"{t} → {te}" for t, te in conditions], fontsize=11)
    ax.set_yticks(range(len(models_present)))
    ax.set_yticklabels(models_present, fontsize=11)

    for i in range(len(models_present)):
        for j in range(len(conditions)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if matrix[i, j] < 0.3 else "black")

    plt.colorbar(im, label="NDVI R²")
    ax.set_title("Spatial Transfer: NDVI R² Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = CONFIG["plots_dir"] / "transfer_heatmap.pdf"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


def plot_outperformance(df):
    """Bar chart: outperformance % vs climatology."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models_present = df["model"].unique()
    conditions = CONFIG["experiment_matrix"]
    x = np.arange(len(conditions))
    width = 0.8 / max(len(models_present), 1)
    cmap = plt.cm.Set2

    for i, model in enumerate(models_present):
        vals = []
        for train_r, test_r in conditions:
            row = df[(df["model"] == model) & (df["train_region"] == train_r) & (df["test_region"] == test_r)]
            vals.append(row.iloc[0]["outperformance_pct"] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, vals, width, label=model, color=cmap(i),
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.axhline(50, ls="--", color="gray", alpha=0.5, label="Random (50%)")

    ax.set_xticks(x + width * (len(models_present) - 1) / 2)
    ax.set_xticklabels([f"{t} → {te}" for t, te in conditions], fontsize=11)
    ax.set_ylabel("Outperformance vs Climatology (%)", fontsize=12)
    ax.set_title("Fraction of Samples Where Model Beats Climatology", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = CONFIG["plots_dir"] / "outperformance.pdf"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


def main():
    print("Loading metrics...")
    df = load_all_metrics()

    if len(df) == 0:
        print("No metrics found! Run training and evaluation first.")
        return

    print(f"Loaded {len(df)} metric files")

    summary_table(df)
    transfer_gap_table(df)

    print("\nGenerating plots...")
    CONFIG["plots_dir"].mkdir(parents=True, exist_ok=True)
    plot_per_leadtime(df)
    plot_ndvi_r2_bar(df)
    plot_transfer_heatmap(df)
    plot_outperformance(df)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
