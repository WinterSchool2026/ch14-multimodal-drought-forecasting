"""Fix cached NPZ files: S2 bands were divided by 10000 incorrectly
(DeepExtremes stores reflectance in 0-1, not 0-10000), and ERA5 ssr_mean
normalization scale was wrong."""

import argparse
import numpy as np
from pathlib import Path
from config import CONFIG

# SSR raw values are ~0-350000 J/m²/day, old scale=30 was way too small
SSR_IDX = 3  # ssr_mean is 4th ERA5 variable
OLD_SSR_SCALE = 30
NEW_SSR_SCALE = 1e6  # normalize to ~0.1-0.3 range


def fix_cube(npz_path):
    """Fix S2 bands and ERA5 in a cached NPZ file."""
    data = dict(np.load(npz_path))
    changed = False

    # Fix S2 bands: multiply by 10000 to undo wrong division
    s2 = data["s2_bands"]
    if s2.max() < 0.01:  # still in wrong scale
        data["s2_bands"] = np.clip(s2 * 10000.0, 0, 1).astype(np.float32)
        changed = True

    # Fix ERA5 ssr_mean: un-normalize with old scale, re-normalize with new
    era5 = data["era5"]
    if era5.shape[1] >= 4 and np.abs(era5[:, SSR_IDX]).max() > 100:
        # ssr was "normalized" with scale=30 but raw values are ~200K
        # Undo: raw = val * old_scale + shift (shift=0 for ssr)
        raw_ssr = era5[:, SSR_IDX] * OLD_SSR_SCALE
        era5[:, SSR_IDX] = raw_ssr / NEW_SSR_SCALE
        data["era5"] = era5
        changed = True

    if changed:
        np.savez_compressed(npz_path, **data)

    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    args = parser.parse_args()

    cache_dir = CONFIG["cache_dir"] / args.region
    npz_files = sorted(cache_dir.glob("*.npz"))
    print(f"Fixing {len(npz_files)} NPZ files in {cache_dir}")

    fixed = 0
    for i, path in enumerate(npz_files):
        if fix_cube(path):
            fixed += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(npz_files)} processed, {fixed} fixed")

    print(f"Done: {fixed}/{len(npz_files)} fixed")

    # Verify
    sample = np.load(npz_files[1])
    print(f"Verification - {npz_files[1].name}:")
    print(f"  S2: mean={sample['s2_bands'].mean():.4f}, max={sample['s2_bands'].max():.4f}")
    era5 = sample["era5"]
    print(f"  ERA5 channels: {[f'{era5[:,i].mean():.4f}' for i in range(6)]}")


if __name__ == "__main__":
    main()
