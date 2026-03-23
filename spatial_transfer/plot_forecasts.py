"""Generate forecast vs actual NDVI plots for sample test cubes."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import DeepExtremesDataset, load_cubes_from_cache
from dataset_terramind import TerraMindDataset, load_cubes_with_embeddings
from models import build_model
from evaluate import make_predict_fn


def get_predictions(model_name, train_region, test_region, device, n_samples=6):
    """Load model and get predictions for sample test cubes."""
    splits_path = CONFIG["splits_dir"] / f"{test_region}_split.json"
    with open(splits_path) as f:
        splits = json.load(f)

    cache_dir = CONFIG["cache_dir"] / test_region
    test_ids = [c["id"] for c in splits["test"]]
    use_terramind = model_name == "TerraMindForecaster"

    if use_terramind:
        test_cubes = load_cubes_with_embeddings(cache_dir, test_ids)
        test_ds = TerraMindDataset(test_cubes, CONFIG["context_length"],
                                    CONFIG["forecast_horizon"], CONFIG["n_era5"])
    else:
        test_cubes = load_cubes_from_cache(cache_dir, test_ids)
        test_ds = DeepExtremesDataset(test_cubes, CONFIG["context_length"],
                                       CONFIG["forecast_horizon"], CONFIG["n_era5"])

    ckpt_path = CONFIG["checkpoints_dir"] / f"{train_region}_{model_name}" / "model.pt"
    model = build_model(model_name, CONFIG["context_length"], CONFIG["forecast_horizon"],
                        n_s2=CONFIG["n_s2_bands"], n_era5=CONFIG["n_era5"])
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device).eval()
    predict_fn = make_predict_fn(model, device)

    # Pick evenly-spaced samples
    indices = np.linspace(0, len(test_ds) - 1, n_samples, dtype=int)
    results = []
    with torch.no_grad():
        for idx in indices:
            batch = test_ds[idx]
            # Add batch dimension
            batch_in = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            pred = predict_fn(batch_in).cpu().squeeze(0)  # (T, H, W)
            actual = batch["target_anomaly"] + batch["target_msc"]  # NDVI
            results.append({
                "pred": pred.numpy(),
                "actual": actual.numpy(),
                "msc": batch["target_msc"].numpy(),
            })
    return results


def plot_timeseries(results_by_model, test_region, save_dir):
    """Plot spatially-averaged NDVI time series: forecast vs actual."""
    n_samples = len(list(results_by_model.values())[0])
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    lead_times = np.arange(1, CONFIG["forecast_horizon"] + 1) * 5

    for i, ax in enumerate(axes.flat):
        if i >= n_samples:
            ax.set_visible(False)
            continue

        for model_name, results in results_by_model.items():
            r = results[i]
            pred_mean = r["pred"].mean(axis=(1, 2))
            actual_mean = r["actual"].mean(axis=(1, 2))
            clim_mean = r["msc"].mean(axis=(1, 2))

            if model_name == list(results_by_model.keys())[0]:
                ax.plot(lead_times, actual_mean, "k-", linewidth=2, label="Actual", zorder=5)
                ax.plot(lead_times, clim_mean, ":", color="gray", linewidth=1.5, label="Climatology")

            color = "#2196F3" if "Context" in model_name else "#E65100"
            ax.plot(lead_times, pred_mean, "--", linewidth=1.8, color=color, label=model_name)

        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel("NDVI")
        ax.set_title(f"Sample {i+1}", fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Forecast vs Actual NDVI — Test region: {test_region}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_dir / f"forecast_vs_actual_{test_region}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


def plot_spatial(results_by_model, test_region, save_dir):
    """Plot spatial NDVI maps: actual vs predicted at lead time 10 (50 days)."""
    lt_idx = 9  # lead time step 10 = 50 days
    models = list(results_by_model.keys())
    n_cols = len(models) + 1  # actual + each model
    n_rows = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            if col == 0:
                data = list(results_by_model.values())[0][row]["actual"][lt_idx]
                title = "Actual" if row == 0 else ""
                ax.set_ylabel(f"Sample {row+1}", fontsize=10)
            else:
                model_name = models[col - 1]
                data = results_by_model[model_name][row]["pred"][lt_idx]
                title = model_name if row == 0 else ""

            im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=0.8)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"NDVI at 50-day lead — Test: {test_region}", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.6, label="NDVI")
    plt.tight_layout()
    path = save_dir / f"spatial_forecast_{test_region}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-region", required=True)
    parser.add_argument("--test-region", required=True)
    parser.add_argument("--n-samples", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = CONFIG["plots_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    models = ["ContextUNet", "TerraMindForecaster"]
    results_by_model = {}
    for model_name in models:
        ckpt = CONFIG["checkpoints_dir"] / f"{args.train_region}_{model_name}" / "model.pt"
        if not ckpt.exists():
            print(f"Skipping {model_name} (no checkpoint)")
            continue
        print(f"Getting predictions: {model_name} ({args.train_region} -> {args.test_region})")
        results_by_model[model_name] = get_predictions(
            model_name, args.train_region, args.test_region, device, args.n_samples)

    plot_timeseries(results_by_model, args.test_region, save_dir)
    plot_spatial(results_by_model, args.test_region, save_dir)


if __name__ == "__main__":
    main()
