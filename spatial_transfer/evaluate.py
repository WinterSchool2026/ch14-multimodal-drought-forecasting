"""Evaluate a trained model on a test set and save metrics JSON."""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import DeepExtremesDataset, load_cubes_from_cache
from dataset_terramind import TerraMindDataset, load_cubes_with_embeddings
from metrics import (
    evaluate_per_leadtime,
    compute_r2_nse,
    compute_ndvi_r2,
    compute_outperformance,
    persistence_predict,
    climatology_predict,
)
from models import build_model


def make_predict_fn(model, device):
    model.eval()
    is_terramind = hasattr(model, 'emb_proj')
    def predict_fn(batch):
        if is_terramind:
            return model(
                context_emb=batch["context_emb"],
                context_era5=batch["context_era5"],
                target_era5=batch["target_era5"],
                dem=batch["dem"],
                last_ndvi=batch["last_ndvi"],
                last_s2=batch["last_s2"],
            )
        return model(
            context_s2=batch["context_s2"],
            context_era5=batch["context_era5"],
            target_era5=batch["target_era5"],
            dem=batch["dem"],
        )
    return predict_fn


def evaluate_single(predict_fn, test_loader, device, is_delta_model=False):
    if is_delta_model:
        def anomaly_predict_fn(batch):
            pred_ndvi = predict_fn(batch)
            return pred_ndvi - batch["target_msc"].to(pred_ndvi.device)
        lt = evaluate_per_leadtime(test_loader, anomaly_predict_fn, device)
        r2_nse = compute_r2_nse(test_loader, anomaly_predict_fn, device)
    else:
        lt = evaluate_per_leadtime(test_loader, predict_fn, device)
        r2_nse = compute_r2_nse(test_loader, predict_fn, device)

    ndvi_metrics = compute_ndvi_r2(test_loader, predict_fn, device,
                                   is_delta_model=is_delta_model)
    outperf = compute_outperformance(test_loader, predict_fn, device,
                                     is_delta_model=is_delta_model)
    mean_rmse = float(np.mean(lt["rmse"]))
    mean_mae = float(np.mean(lt["mae"]))
    return {
        "per_leadtime": lt,
        "aggregate": {
            "mean_rmse": mean_rmse,
            "mean_mae": mean_mae,
            "r2": r2_nse["r2"],
            "nse": r2_nse["nse"],
            "ndvi_r2": ndvi_metrics["ndvi_r2"],
            "ndvi_rmse": ndvi_metrics["ndvi_rmse"],
            "outperformance_pct": outperf,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-region", required=True, choices=list(CONFIG["regions"].keys()))
    parser.add_argument("--test-region", required=True, choices=list(CONFIG["regions"].keys()))
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--vi-type", choices=["ndvi", "evi"], default="ndvi")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_path = CONFIG["splits_dir"] / f"{args.test_region}_split.json"
    with open(splits_path) as f:
        splits = json.load(f)

    cache_dir = CONFIG["cache_dir"] / args.test_region
    test_ids = [c["id"] for c in splits["test"]]
    use_terramind = args.model == "TerraMindForecaster"

    if use_terramind:
        print(f"Loading {len(test_ids)} test cubes with TerraMind embeddings...")
        test_cubes = load_cubes_with_embeddings(cache_dir, test_ids)
        test_ds = TerraMindDataset(test_cubes, CONFIG["context_length"], CONFIG["forecast_horizon"], CONFIG["n_era5"],
                                    target_vi=args.vi_type)
    else:
        print(f"Loading {len(test_ids)} test cubes from {cache_dir}...")
        test_cubes = load_cubes_from_cache(cache_dir, test_ids)
        test_ds = DeepExtremesDataset(test_cubes, CONFIG["context_length"], CONFIG["forecast_horizon"], CONFIG["n_era5"],
                                       target_vi=args.vi_type)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test samples: {len(test_ds)}")

    ckpt_path = CONFIG["checkpoints_dir"] / f"{args.train_region}_{args.model}" / "model.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    model = build_model(args.model, CONFIG["context_length"], CONFIG["forecast_horizon"],
                        n_s2=CONFIG["n_s2_bands"], n_era5=CONFIG["n_era5"], vi_type=args.vi_type)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    is_delta = getattr(model, 'delta_prediction', False)
    predict_fn = make_predict_fn(model, device)
    print(f"Evaluating {args.model} (train={args.train_region}, test={args.test_region})...")
    results = evaluate_single(predict_fn, test_loader, device, is_delta_model=is_delta)

    print("Evaluating baselines...")
    persistence_results = evaluate_single(persistence_predict, test_loader, "cpu", is_delta_model=False)
    climatology_results = evaluate_single(climatology_predict, test_loader, "cpu", is_delta_model=False)

    output = {
        "train_region": args.train_region,
        "test_region": args.test_region,
        "model": args.model,
        **results,
        "baselines": {
            "persistence": persistence_results["aggregate"],
            "climatology": climatology_results["aggregate"],
        },
        "n_test_cubes": len(test_ids),
        "n_test_samples": len(test_ds),
    }

    CONFIG["metrics_dir"].mkdir(parents=True, exist_ok=True)
    out_path = CONFIG["metrics_dir"] / f"{args.train_region}_on_{args.test_region}_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved metrics to {out_path}")

    agg = results['aggregate']
    print(f"\n{'='*50}")
    print(f"  Model:            {args.model}")
    print(f"  Train:            {args.train_region} -> {args.test_region}")
    print(f"  NDVI R²:          {agg['ndvi_r2']:.4f}")
    print(f"  NDVI RMSE:        {agg['ndvi_rmse']:.4f}")
    print(f"  Outperformance:   {agg['outperformance_pct']:.1f}%")
    print(f"  Anomaly RMSE:     {agg['mean_rmse']:.4f}")
    print(f"  Anomaly R²:       {agg['r2']:.4f}")
    print(f"  Persistence RMSE: {persistence_results['aggregate']['mean_rmse']:.4f}")
    print(f"  Climatology RMSE: {climatology_results['aggregate']['mean_rmse']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
