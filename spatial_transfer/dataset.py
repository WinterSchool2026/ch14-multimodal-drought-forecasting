"""PyTorch Dataset for multi-modal vegetation forecasting."""

import numpy as np
import torch
from torch.utils.data import Dataset


class DeepExtremesDataset(Dataset):
    def __init__(self, cubes, context_length=10, forecast_horizon=20, n_era5=6,
                 augment=False, target_vi="ndvi"):
        self.cubes = cubes
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.n_era5 = n_era5
        self.augment = augment
        self.target_vi = target_vi  # "ndvi" or "evi"

        # Select anomaly/msc keys based on target VI
        if target_vi == "evi":
            self.anomaly_key = "evi_anomaly"
            self.msc_key = "evi_msc"
        else:
            self.anomaly_key = "anomaly"
            self.msc_key = "ndvi_msc"

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

        context_s2 = torch.from_numpy(cube["s2_bands"][ctx_start:t]).float()
        context_era5 = torch.from_numpy(cube["era5"][ctx_start:t]).float()
        target_era5 = torch.from_numpy(cube["era5"][t:tgt_end]).float()
        dem = torch.from_numpy(cube["dem"]).float().unsqueeze(0)
        target_anomaly = torch.from_numpy(cube[self.anomaly_key][t:tgt_end]).float()
        target_msc = torch.from_numpy(cube[self.msc_key][t:tgt_end]).float()
        mask = torch.from_numpy(cube["qc_mask"][t:tgt_end]).float()

        if self.augment:
            if torch.rand(1).item() > 0.5:
                context_s2 = context_s2.flip(-1)
                dem = dem.flip(-1)
                target_anomaly = target_anomaly.flip(-1)
                target_msc = target_msc.flip(-1)
                mask = mask.flip(-1)
            if torch.rand(1).item() > 0.5:
                context_s2 = context_s2.flip(-2)
                dem = dem.flip(-2)
                target_anomaly = target_anomaly.flip(-2)
                target_msc = target_msc.flip(-2)
                mask = mask.flip(-2)
            if torch.rand(1).item() > 0.5:
                context_s2 = context_s2.transpose(-2, -1)
                dem = dem.transpose(-2, -1)
                target_anomaly = target_anomaly.transpose(-2, -1)
                target_msc = target_msc.transpose(-2, -1)
                mask = mask.transpose(-2, -1)

        return {
            "context_s2": context_s2,
            "context_era5": context_era5,
            "dem": dem,
            "target_anomaly": target_anomaly,
            "target_msc": target_msc,
            "target_era5": target_era5,
            "mask": mask,
        }


def load_cubes_from_cache(cache_dir, cube_ids):
    from pathlib import Path
    cubes = []
    for cid in cube_ids:
        path = Path(cache_dir) / f"{cid}.npz"
        if not path.exists():
            print(f"  Warning: missing cache for {cid}")
            continue
        data = dict(np.load(path))
        cubes.append(data)
    return cubes
