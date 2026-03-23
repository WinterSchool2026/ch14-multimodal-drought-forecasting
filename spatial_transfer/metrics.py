"""Evaluation metrics and baseline predictors."""

import torch
import numpy as np


def masked_mse(pred, target, mask):
    diff_sq = (pred - target) ** 2
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (diff_sq * mask).sum() / mask.sum()


def masked_huber(pred, target, mask, delta=0.1):
    diff = (pred - target).abs()
    huber = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (huber * mask).sum() / mask.sum()


def masked_mae(pred, target, mask):
    diff_abs = (pred - target).abs()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (diff_abs * mask).sum() / mask.sum()


def evaluate_per_leadtime(loader, predict_fn, device="cpu"):
    """RMSE and MAE per forecast lead time."""
    mse_accum = None
    mae_accum = None
    count_accum = None

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = predict_fn(batch)
            target = batch["target_anomaly"]
            mask = batch["mask"]

            B, F, H, W = target.shape
            if mse_accum is None:
                mse_accum = torch.zeros(F, device=device)
                mae_accum = torch.zeros(F, device=device)
                count_accum = torch.zeros(F, device=device)

            for f in range(F):
                m = mask[:, f]
                if m.sum() > 0:
                    mse_accum[f] += ((pred[:, f] - target[:, f]) ** 2 * m).sum()
                    mae_accum[f] += ((pred[:, f] - target[:, f]).abs() * m).sum()
                    count_accum[f] += m.sum()

    count_accum = count_accum.clamp(min=1)
    rmse = (mse_accum / count_accum).sqrt().cpu().numpy()
    mae = (mae_accum / count_accum).cpu().numpy()
    return {"rmse": rmse.tolist(), "mae": mae.tolist()}


def compute_r2_nse(loader, predict_fn, device="cpu"):
    """R² and NSE on masked pixels (anomaly space)."""
    all_pred, all_target, all_mask = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            pred = predict_fn(batch_dev).cpu()
            all_pred.append(pred)
            all_target.append(batch["target_anomaly"])
            all_mask.append(batch["mask"])

    pred = torch.cat(all_pred, 0)
    target = torch.cat(all_target, 0)
    mask = torch.cat(all_mask, 0)

    p = pred[mask > 0.5].numpy()
    t = target[mask > 0.5].numpy()

    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"r2": r2, "nse": r2}


def compute_ndvi_r2(loader, predict_fn, device="cpu", is_delta_model=False):
    """R² and RMSE on reconstructed NDVI (anomaly + MSC)."""
    all_pred_ndvi, all_target_ndvi, all_mask = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            pred = predict_fn(batch_dev).cpu()

            target_ndvi = batch["target_anomaly"] + batch["target_msc"]
            if is_delta_model:
                pred_ndvi = pred
            else:
                pred_ndvi = pred + batch["target_msc"]

            all_pred_ndvi.append(pred_ndvi)
            all_target_ndvi.append(target_ndvi)
            all_mask.append(batch["mask"])

    pred_ndvi = torch.cat(all_pred_ndvi, 0)
    target_ndvi = torch.cat(all_target_ndvi, 0)
    mask = torch.cat(all_mask, 0)

    p = pred_ndvi[mask > 0.5].numpy()
    t = target_ndvi[mask > 0.5].numpy()

    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    ndvi_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    ndvi_rmse = float(np.sqrt(np.mean((t - p) ** 2)))

    return {"ndvi_r2": ndvi_r2, "ndvi_rmse": ndvi_rmse}


def compute_outperformance(loader, predict_fn, device="cpu", is_delta_model=False):
    """Fraction of samples where model beats climatology."""
    model_wins = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            pred = predict_fn(batch_dev).cpu()
            mask = batch["mask"]
            target_ndvi = batch["target_anomaly"] + batch["target_msc"]

            if is_delta_model:
                pred_ndvi = pred
            else:
                pred_ndvi = pred + batch["target_msc"]

            clim_ndvi = batch["target_msc"]

            B = pred_ndvi.shape[0]
            for i in range(B):
                m = mask[i] > 0.5
                if m.sum() == 0:
                    continue
                model_err = ((pred_ndvi[i][m] - target_ndvi[i][m]) ** 2).mean()
                clim_err = ((clim_ndvi[i][m] - target_ndvi[i][m]) ** 2).mean()
                if model_err < clim_err:
                    model_wins += 1
                total += 1

    return float(model_wins / total * 100) if total > 0 else 0.0


def persistence_predict(batch):
    """Persistence baseline: repeat last observed anomaly for all future steps."""
    if "context_s2" in batch:
        last_s2 = batch["context_s2"][:, -1]
        nir = last_s2[:, 3]
        red = last_s2[:, 2]
        last_vi = ((nir - red) / (nir + red + 1e-8)).clamp(-1, 1)
    else:
        last_vi = batch["last_ndvi"].squeeze(1)  # (B, 64, 64)
    last_anomaly = last_vi - batch["target_msc"][:, 0]
    F = batch["target_anomaly"].shape[1]
    return last_anomaly.unsqueeze(1).expand(-1, F, -1, -1)


def climatology_predict(batch):
    """Climatology baseline: predict zero anomaly."""
    return torch.zeros_like(batch["target_anomaly"])
