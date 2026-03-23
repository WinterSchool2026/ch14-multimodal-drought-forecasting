"""Train one model on one region with early stopping."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import DeepExtremesDataset, load_cubes_from_cache
from dataset_terramind import TerraMindDataset, load_cubes_with_embeddings
from metrics import masked_huber
from models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_target(model, batch):
    if getattr(model, 'delta_prediction', False):
        return batch["target_anomaly"] + batch["target_msc"]
    return batch["target_anomaly"]


def _model_forward(model, batch):
    """Route inputs based on model type."""
    if hasattr(model, 'emb_proj'):  # TerraMindForecaster
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


def train_model(model, train_loader, val_loader, epochs, lr, patience, device, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = _model_forward(model, batch)
            target = _get_target(model, batch)
            loss = masked_huber(pred, target, batch["mask"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = _model_forward(model, batch)
                target = _get_target(model, batch)
                loss = masked_huber(pred, target, batch["mask"])
                val_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter == 0:
            print(f"  [{name}] Epoch {epoch+1:3d}/{epochs}: train={train_loss:.5f}, val={val_loss:.5f}"
                  f"  {'*best*' if patience_counter == 0 else f'(patience {patience_counter}/{patience})'}")

        if patience_counter >= patience:
            print(f"  [{name}] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--region", required=True, choices=list(CONFIG["regions"].keys()))
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--patience", type=int, default=CONFIG["patience"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--vi-type", choices=["ndvi", "evi"], default="ndvi")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    splits_path = CONFIG["splits_dir"] / f"{args.region}_split.json"
    with open(splits_path) as f:
        splits = json.load(f)

    cache_dir = CONFIG["cache_dir"] / args.region
    train_ids = [c["id"] for c in splits["train"]]
    val_ids = [c["id"] for c in splits["val"]]

    use_terramind = args.model == "TerraMindForecaster"

    if use_terramind:
        print(f"Loading {len(train_ids)} train + {len(val_ids)} val cubes with TerraMind embeddings...")
        train_cubes = load_cubes_with_embeddings(cache_dir, train_ids)
        val_cubes = load_cubes_with_embeddings(cache_dir, val_ids)
        train_ds = TerraMindDataset(train_cubes, CONFIG["context_length"], CONFIG["forecast_horizon"], CONFIG["n_era5"],
                                     target_vi=args.vi_type)
        val_ds = TerraMindDataset(val_cubes, CONFIG["context_length"], CONFIG["forecast_horizon"], CONFIG["n_era5"],
                                   target_vi=args.vi_type)
    else:
        print(f"Loading {len(train_ids)} train + {len(val_ids)} val cubes from {cache_dir}...")
        train_cubes = load_cubes_from_cache(cache_dir, train_ids)
        val_cubes = load_cubes_from_cache(cache_dir, val_ids)
        train_ds = DeepExtremesDataset(train_cubes, CONFIG["context_length"], CONFIG["forecast_horizon"], CONFIG["n_era5"],
                                        target_vi=args.vi_type)
        val_ds = DeepExtremesDataset(val_cubes, CONFIG["context_length"], CONFIG["forecast_horizon"], CONFIG["n_era5"],
                                      target_vi=args.vi_type)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    nw = 4 if use_terramind else 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=True)

    model = build_model(args.model, CONFIG["context_length"], CONFIG["forecast_horizon"],
                        n_s2=CONFIG["n_s2_bands"], n_era5=CONFIG["n_era5"], vi_type=args.vi_type)
    model = model.to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({nparams:,} params)")

    name = f"{args.region}_{args.model}"
    history, best_val = train_model(model, train_loader, val_loader, args.epochs, args.lr, args.patience, device, name)
    print(f"\nBest val loss: {best_val:.6f}")

    ckpt_dir = CONFIG["checkpoints_dir"] / f"{args.region}_{args.model}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()
