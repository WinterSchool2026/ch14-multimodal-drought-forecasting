"""Post-process cached NPZ files to add EVI fields.

Computes EVI from existing s2_bands (B02, B03, B04, B8A) and adds:
  evi_filled, evi_msc, evi_anomaly

EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + L)
where G=2.5, C1=6, C2=7.5, L=1, and reflectance is in [0, 1].

Usage:
    python add_evi.py --region africa
    python add_evi.py --region south_america
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from config import CONFIG


def compute_evi(blue, red, nir):
    """Compute EVI from reflectance arrays in [0, 1] range."""
    num = 2.5 * (nir - red)
    den = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.where(np.abs(den) > 1e-6, num / den, 0.0)
    return np.clip(evi, -1, 1).astype(np.float32)


def compute_msc(vi_filled, n_per_year=73):
    """Compute mean seasonal cycle via day-of-year climatology with smoothing.

    Args:
        vi_filled: (T, H, W) gap-filled VI time series
        n_per_year: timesteps per year (73 for 5-day intervals)
    """
    T, H, W = vi_filled.shape
    # Assign each timestep a day-of-year bin
    doy = np.arange(T) % n_per_year

    # Compute climatology per DOY
    climatology = np.zeros((n_per_year, H, W), dtype=np.float32)
    for d in range(n_per_year):
        mask = doy == d
        if mask.sum() > 0:
            climatology[d] = vi_filled[mask].mean(axis=0)

    # Smooth with circular rolling mean (window=6 = 30 days)
    window = 6
    padded = np.concatenate([climatology[-window:], climatology, climatology[:window]], axis=0)
    kernel = np.ones(2 * window + 1) / (2 * window + 1)
    smoothed = np.zeros_like(climatology)
    for i in range(n_per_year):
        smoothed[i] = padded[i:i + 2 * window + 1].mean(axis=0)

    # Map back to full time series
    msc = smoothed[doy]
    return msc


def add_evi_to_npz(npz_path):
    """Add EVI fields to an existing NPZ file."""
    data = dict(np.load(npz_path))

    if "evi_filled" in data:
        return False  # already has EVI

    s2 = data["s2_bands"]  # (T, 4, H, W) — B02, B03, B04, B8A
    blue = s2[:, 0]  # B02
    red = s2[:, 2]   # B04
    nir = s2[:, 3]   # B8A

    evi = compute_evi(blue, red, nir)  # (T, H, W)

    # Gap-fill: forward fill zeros (cloud-masked pixels are 0 after processing)
    # The s2_bands are already cloud-masked and forward-filled, so EVI should be too
    evi_filled = evi.copy()

    # Compute MSC and anomaly
    evi_msc = compute_msc(evi_filled)
    evi_anomaly = evi_filled - evi_msc

    data["evi_filled"] = evi_filled
    data["evi_msc"] = evi_msc
    data["evi_anomaly"] = evi_anomaly

    np.savez_compressed(npz_path, **data)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    args = parser.parse_args()

    cache_dir = CONFIG["cache_dir"] / args.region
    npz_files = sorted(cache_dir.glob("*.npz"))
    print(f"Processing {len(npz_files)} files in {cache_dir}")

    added, skipped = 0, 0
    for i, path in enumerate(npz_files):
        if add_evi_to_npz(path):
            added += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(npz_files)}] added EVI")
        else:
            skipped += 1

    print(f"Done: {added} updated, {skipped} already had EVI")


if __name__ == "__main__":
    main()
