"""Download minicubes from S3, precompute features, cache as NPZ.

Saves 4 S2 bands (B02,B03,B04,B8A) at 64x64 + DEM + NDVI + anomaly + QC + ERA5.
Optionally saves all 7 available S2 bands at 128x128 for foundation model extraction.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr

from config import CONFIG, ERA5_VARS, ERA5_NORM

S2_FULL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A"]


def precompute_s2_full(ds, qc_mask):
    """Extract all 7 available S2 bands at native 128x128 resolution."""
    qc_xr = xr.DataArray(qc_mask, dims=["time", "y", "x"], coords={"time": ds.time})
    bands = []
    for name in S2_FULL_BANDS:
        band = ds[name].astype(np.float32)
        band_clean = band.where(qc_xr).chunk({"time": -1}).ffill("time")
        band_np = np.nan_to_num(band_clean.values.astype(np.float32), nan=0.0)
        band_np = band_np / 10000.0
        band_np = np.clip(band_np, 0, 1)
        bands.append(band_np)
    return np.stack(bands, axis=1)  # (T, 7, 128, 128)


def precompute_minicube(ds):
    """Process a raw xarray minicube into numpy arrays, center-cropped to 64x64."""
    # Subset to 2017-2021 (indices 73 to 6*73 = 438)
    ds = ds.isel(time=slice(73, 6 * 73))

    # QC mask
    qc_mask = ds.cloudmask_en.values == 0)
    qc_xr = xr.DataArray(qc_mask, dims=["time", "y", "x"],
                          coords={"time": ds.time})

    # S2 bands
    s2_band_names = CONFIG["s2_bands"]
    s2_list = []
    for band_name in s2_band_names:
        band = ds[band_name].astype(np.float32)
        band_clean = band.where(qc_xr).chunk({"time": -1}).ffill("time")
        band_np = np.nan_to_num(band_clean.values.astype(np.float32), nan=0.0)
        band_np = band_np / 10000.0
        band_np = np.clip(band_np, 0, 1)
        s2_list.append(band_np)

    s2_bands = np.stack(s2_list, axis=1)  # (T, 4, H, W)

    # DEM
    if "CopernicusDEM" in ds:
        dem = ds["CopernicusDEM"].values.astype(np.float32)
        if dem.ndim == 3:
            dem = dem[0]  # take first time step (static)
    elif "dem" in ds:
        dem = ds["dem"].values.astype(np.float32)
        if dem.ndim == 3:
            dem = dem[0]
    else:
        dem = np.zeros((ds.dims["y"], ds.dims["x"]), dtype=np.float32)
    dem = np.nan_to_num(dem, nan=0.0)
    dem = (dem - dem.mean()) / max(dem.std(), 1.0)

    # NDVI
    ndvi = ((ds.B8A - ds.B04) / (ds.B8A + ds.B04)).clip(-1, 1)
    ndvi_cloudfree = ndvi.where(qc_xr)
    ndvi_cloudfree = ndvi_cloudfree.chunk({"time": -1})

    # Climatology (Mean Seasonal Cycle)
    ndvi_climatology = (
        ndvi_cloudfree
        .interpolate_na("time", method="linear")
        .resample(time="1D")
        .interpolate("linear")
        .groupby("time.dayofyear")
        .mean()
        .pad(dayofyear=30, mode="wrap")
        .rolling(dayofyear=30, min_periods=1)
        .mean()
        .isel(dayofyear=slice(30, -30))
    )

    ndvi_filled = ndvi_cloudfree.ffill("time")
    ndvi_msc = ndvi_climatology.sel(dayofyear=ndvi_filled.time.dt.dayofyear)
    anomaly = ndvi_filled - ndvi_msc

    # Convert to numpy
    ndvi_filled_np = np.nan_to_num(ndvi_filled.values.astype(np.float32), nan=0.0)
    ndvi_msc_np = np.nan_to_num(ndvi_msc.values.astype(np.float32), nan=0.0)
    anomaly_np = np.nan_to_num(anomaly.values.astype(np.float32), nan=0.0)
    qc_mask_np = qc_mask.astype(np.float32)

    # Center crop to 64x64
    H, W = ndvi_filled_np.shape[1], ndvi_filled_np.shape[2]
    y0 = (H - 64) // 2
    x0 = (W - 64) // 2
    s = np.s_[:, y0:y0+64, x0:x0+64]
    sb = np.s_[:, :, y0:y0+64, x0:x0+64]

    s2_bands = s2_bands[:, :, y0:y0+64, x0:x0+64]
    dem = dem[y0:y0+64, x0:x0+64]
    ndvi_filled_np = ndvi_filled_np[s]
    ndvi_msc_np = ndvi_msc_np[s]
    anomaly_np = anomaly_np[s]
    qc_mask_np = qc_mask_np[s]

    # ERA5
    era5_list = []
    for var in ERA5_VARS:
        if var in ds:
            vals = ds[var].values
            if vals.ndim == 3:
                vals = np.nanmean(vals, axis=(1, 2))
            norm = ERA5_NORM[var]
            vals = (vals - norm["shift"]) / norm["scale"]
            era5_list.append(vals.astype(np.float32))
        else:
            era5_list.append(np.zeros(len(ds.time), dtype=np.float32))

    era5_np = np.stack(era5_list, axis=-1)
    era5_np = np.nan_to_num(era5_np, nan=0.0)

    return {
        "s2_bands": s2_bands,       # (T, 4, 64, 64)
        "dem": dem,                  # (64, 64)
        "ndvi_filled": ndvi_filled_np,
        "ndvi_msc": ndvi_msc_np,
        "anomaly": anomaly_np,
        "qc_mask": qc_mask_np,
        "era5": era5_np,
    }


def download_region(region_name, fs, max_retries=3, save_s2_full=False):
    """Download and cache all cubes for a region."""
    splits_path = CONFIG["splits_dir"] / f"{region_name}_split.json"
    with open(splits_path) as f:
        splits = json.load(f)

    cache_dir = CONFIG["cache_dir"] / region_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_cubes = []
    for split_name, cubes in splits.items():
        for cube in cubes:
            all_cubes.append(cube)

    min_qc = CONFIG["min_valid_qc_fraction"]
    success, fail, skip, reject = 0, 0, 0, 0

    for i, cube in enumerate(all_cubes):
        cube_id = cube["id"]
        cache_path = cache_dir / f"{cube_id}.npz"
        s2full_path = cache_dir / f"{cube_id}_s2full.npy"

        need_base = not cache_path.exists()
        if not need_base:
            try:
                d = np.load(cache_path, allow_pickle=False)
                if "s2_bands" not in d:
                    need_base = True
            except Exception:
                need_base = True

        need_s2full = save_s2_full and not s2full_path.exists()

        if not need_base and not need_s2full:
            skip += 1
            print(f"  [{i+1}/{len(all_cubes)}] {cube_id} — cached")
            continue

        fname = cube["path"]
        for attempt in range(max_retries):
            try:
                print(f"  [{i+1}/{len(all_cubes)}] {cube_id} — downloading (attempt {attempt+1})...")
                store = s3fs.S3Map(root=f"earthnet/deepextremes/{fname}", s3=fs)
                ds_raw = xr.open_zarr(store)
                ds = ds_raw.isel(time=slice(73, 6 * 73))
                qc_mask = np.isin(ds.SCL.values, [4, 5, 6, 7]) & (ds.cloudmask_en.values == 0)

                if need_base:
                    data = precompute_minicube(ds_raw)
                    avg_valid = data["qc_mask"].mean()
                    anom_all_zero = (data["anomaly"] == 0).all()

                    if avg_valid < min_qc:
                        print(f"    REJECT: qc_mask valid fraction = {avg_valid:.3f} < {min_qc}")
                        reject += 1
                        break
                    if anom_all_zero:
                        print(f"    REJECT: anomaly is all zero")
                        reject += 1
                        break

                    np.savez_compressed(cache_path, **data)
                    print(f"    OK: s2={data['s2_bands'].shape}, qc_valid={avg_valid:.3f}")

                if need_s2full:
                    s2_full = precompute_s2_full(ds, qc_mask)
                    np.save(s2full_path, s2_full.astype(np.float16))
                    print(f"    S2 full: {s2_full.shape} saved as float16")

                success += 1
                break

            except Exception as e:
                print(f"    ERROR: {e}")
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"    Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    fail += 1
                    print(f"    FAILED after {max_retries} attempts")

    print(f"\n{region_name} download summary: {success} new, {skip} cached, {reject} rejected, {fail} failed")
    return success, skip, reject, fail


def validate_region(region_name):
    """Validate all cached NPZ files for a region."""
    splits_path = CONFIG["splits_dir"] / f"{region_name}_split.json"
    with open(splits_path) as f:
        splits = json.load(f)

    cache_dir = CONFIG["cache_dir"] / region_name
    ok, missing, bad = 0, 0, 0

    for split_name, cubes in splits.items():
        for cube in cubes:
            path = cache_dir / f"{cube['id']}.npz"
            if not path.exists():
                print(f"  MISSING: {cube['id']} ({split_name})")
                missing += 1
                continue
            data = dict(np.load(path))
            if "s2_bands" not in data:
                print(f"  OLD FORMAT: {cube['id']} ({split_name})")
                bad += 1
                continue
            s2_shape = data["s2_bands"].shape
            if len(s2_shape) != 4 or s2_shape[1] != 4 or s2_shape[2] != 64:
                print(f"  BAD SHAPE: {cube['id']} s2={s2_shape}")
                bad += 1
                continue
            ok += 1

    print(f"{region_name} validation: {ok} OK, {missing} missing, {bad} bad shape")
    return ok, missing, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, choices=list(CONFIG["regions"].keys()))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--s2-full", action="store_true",
                        help="Also save all 7 S2 bands at 128x128 for FM extraction")
    args = parser.parse_args()

    if args.validate_only:
        validate_region(args.region)
        return

    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": CONFIG["s3_endpoint"]},
    )
    download_region(args.region, fs, save_s2_full=args.s2_full)
    print("\nValidating...")
    validate_region(args.region)


if __name__ == "__main__":
    main()
