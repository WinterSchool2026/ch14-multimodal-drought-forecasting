"""Dataset for TerraMind-based forecasting using pre-extracted embeddings."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TerraMindDataset(Dataset):
    """Loads pre-extracted TerraMind embeddings + ERA5/DEM/targets."""

    def __init__(self, cubes, context_length=10, forecast_horizon=20, n_era5=6,
                 target_vi="ndvi"):
        self.cubes = cubes
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.target_vi = target_vi

        if target_vi == "evi":
            self.anomaly_key = "evi_anomaly"
            self.msc_key = "evi_msc"
            self.filled_key = "evi_filled"
        else:
            self.anomaly_key = "anomaly"
            self.msc_key = "ndvi_msc"
            self.filled_key = "ndvi_filled"

        self.samples = []
        for cube_idx, cube in enumerate(cubes):
            T = cube[self.anomaly_key].shape[0]
            for t in range(context_length, T - forecast_horizon):
                self.samples.append((cube_idx, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cube_idx, t = self.samples[idx]
        cube = self.cubes[cube_idx]

        ctx_start = t - self.context_length
        tgt_end = t + self.forecast_horizon

        # TerraMind embeddings: (T, 768, 14, 14) float16
        context_emb = torch.from_numpy(
            cube["terramind"][ctx_start:t].astype(np.float32)
        )

        # Last observed VI at 64x64 (for delta prediction)
        last_ndvi = torch.from_numpy(
            cube[self.filled_key][t - 1:t]
        ).float()  # (1, 64, 64)

        context_era5 = torch.from_numpy(cube["era5"][ctx_start:t]).float()
        target_era5 = torch.from_numpy(cube["era5"][t:tgt_end]).float()

        # DEM at 64x64 — will be downsampled in model
        dem = torch.from_numpy(cube["dem"]).float().unsqueeze(0)

        target_anomaly = torch.from_numpy(cube[self.anomaly_key][t:tgt_end]).float()
        target_msc = torch.from_numpy(cube[self.msc_key][t:tgt_end]).float()
        mask = torch.from_numpy(cube["qc_mask"][t:tgt_end]).float()

        # Last S2 frame at 64x64 for hybrid decoder
        last_s2 = torch.from_numpy(cube["s2_bands"][t - 1]).float()  # (4, 64, 64)

        return {
            "context_emb": context_emb,     # (10, 768, 14, 14)
            "last_ndvi": last_ndvi,          # (1, 64, 64)
            "last_s2": last_s2,             # (4, 64, 64)
            "context_era5": context_era5,    # (10, 6)
            "target_era5": target_era5,      # (20, 6)
            "dem": dem,                       # (1, 64, 64)
            "target_anomaly": target_anomaly, # (20, 64, 64)
            "target_msc": target_msc,         # (20, 64, 64)
            "mask": mask,                     # (20, 64, 64)
        }


def load_cubes_with_embeddings(cache_dir, cube_ids):
    """Load cached NPZ data + TerraMind embeddings for each cube."""
    from pathlib import Path
    cubes = []
    for cid in cube_ids:
        base_path = Path(cache_dir) / f"{cid}.npz"
        emb_path = Path(cache_dir) / f"{cid}_terramind.npy"

        if not base_path.exists():
            print(f"  Warning: missing cache for {cid}")
            continue
        if not emb_path.exists():
            print(f"  Warning: missing TerraMind embeddings for {cid}")
            continue

        data = dict(np.load(base_path))
        data["terramind"] = np.load(emb_path, mmap_mode='r')  # memory-mapped
        cubes.append(data)
    return cubes
