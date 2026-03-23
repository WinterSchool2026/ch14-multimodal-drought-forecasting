"""Query S3 registry, filter by region, stratified sample, and save splits."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs

from config import CONFIG


def load_registry(fs):
    """Load the DeepExtremes registry from S3 and parse coordinates."""
    with fs.open("earthnet/deepextremes/registry.csv") as f:
        df_reg = pd.read_csv(f, low_memory=False)

    print(f"Full registry: {len(df_reg)} rows")
    df_v13 = df_reg[df_reg["version"].str.strip() == "1.3"].copy()
    print(f"Version 1.3: {len(df_v13)} rows")

    # List all zarr files on S3
    minicube_paths = fs.ls("earthnet/deepextremes/")
    minicube_paths = [p.split("/")[-1] for p in minicube_paths if p.endswith(".zarr")]
    print(f"Zarr files on S3: {len(minicube_paths)}")

    # Parse lon/lat from filenames
    records = []
    for fname in minicube_paths:
        parts = fname.replace(".zarr", "").split("_")
        try:
            lon = float(parts[1])
            lat = float(parts[2])
            mc_id = fname.replace(".zarr", "")
            records.append({"path": fname, "id": mc_id, "lon": lon, "lat": lat})
        except (ValueError, IndexError):
            continue

    df_cubes = pd.DataFrame(records)
    print(f"Parsed {len(df_cubes)} minicubes with valid coordinates")

    # Merge dominant_class from registry
    df_cubes["mc_id"] = df_cubes["id"]
    if "mc_id" in df_v13.columns and "dominant_class" in df_v13.columns:
        df_cubes = df_cubes.merge(
            df_v13[["mc_id", "dominant_class"]].drop_duplicates("mc_id"),
            on="mc_id",
            how="left",
        )
    else:
        df_cubes["dominant_class"] = "Unknown"

    df_cubes["dominant_class"] = df_cubes["dominant_class"].fillna("-")
    return df_cubes


def filter_region(df, region_name):
    """Filter cubes by geographic bounding box."""
    region = CONFIG["regions"][region_name]
    lon_min, lon_max = region["lon_range"]
    lat_min, lat_max = region["lat_range"]
    mask = (
        (df["lon"] >= lon_min) & (df["lon"] <= lon_max) &
        (df["lat"] >= lat_min) & (df["lat"] <= lat_max)
    )
    df_region = df[mask].copy()
    # Remove ocean/unknown
    df_region = df_region[df_region["dominant_class"] != "-"].copy()
    return df_region


def stratified_sample(df, n, seed):
    """Stratified random sample by dominant_class."""
    rng = np.random.RandomState(seed)

    # Class proportions
    class_counts = df["dominant_class"].value_counts()
    total = class_counts.sum()

    sampled = []
    remaining_n = n
    classes = list(class_counts.index)

    # Allocate proportionally
    allocations = {}
    for cls in classes:
        allocations[cls] = max(1, int(round(class_counts[cls] / total * n)))

    # Adjust to exactly n
    total_alloc = sum(allocations.values())
    if total_alloc > n:
        # Remove from largest classes
        for cls in sorted(allocations, key=lambda c: allocations[c], reverse=True):
            if total_alloc <= n:
                break
            allocations[cls] -= 1
            total_alloc -= 1
    elif total_alloc < n:
        for cls in sorted(allocations, key=lambda c: class_counts[c], reverse=True):
            if total_alloc >= n:
                break
            if allocations[cls] < class_counts[cls]:
                allocations[cls] += 1
                total_alloc += 1

    for cls, alloc in allocations.items():
        cls_df = df[df["dominant_class"] == cls]
        actual = min(alloc, len(cls_df))
        idx = rng.choice(cls_df.index, size=actual, replace=False)
        sampled.append(df.loc[idx])

    result = pd.concat(sampled).reset_index(drop=True)
    # If we still need more, sample from remaining
    if len(result) < n:
        remaining = df[~df.index.isin(result.index)]
        extra = remaining.sample(n=n - len(result), random_state=rng)
        result = pd.concat([result, extra]).reset_index(drop=True)

    return result.iloc[:n]


def split_cubes(df, ratios, seed):
    """Split cubes into train/val/test."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(df))
    n = len(df)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def save_split(split_dict, region_name, output_dir):
    """Save split as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for split_name, df in split_dict.items():
        result[split_name] = df[["id", "path", "lon", "lat", "dominant_class"]].to_dict("records")

    path = output_dir / f"{region_name}_split.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {path}")
    return path


def main():
    seed = CONFIG["seed"]
    n_cubes = CONFIG["n_cubes_per_region"]

    print("Connecting to S3...")
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": CONFIG["s3_endpoint"]},
    )

    print("\nLoading registry...")
    df_cubes = load_registry(fs)

    for region_name in CONFIG["regions"]:
        print(f"\n{'='*60}")
        print(f"Region: {region_name}")
        print(f"{'='*60}")

        df_region = filter_region(df_cubes, region_name)
        print(f"Cubes in region (excluding '-' class): {len(df_region)}")
        print(f"Class distribution:\n{df_region['dominant_class'].value_counts().to_string()}")

        if len(df_region) < n_cubes:
            print(f"WARNING: Only {len(df_region)} cubes available, need {n_cubes}")
            n_actual = len(df_region)
        else:
            n_actual = n_cubes

        df_sampled = stratified_sample(df_region, n_actual, seed)
        print(f"\nSampled {len(df_sampled)} cubes")
        print(f"Sampled class distribution:\n{df_sampled['dominant_class'].value_counts().to_string()}")

        splits = split_cubes(df_sampled, CONFIG["split_ratios"], seed)
        for sname, sdf in splits.items():
            print(f"  {sname}: {len(sdf)} cubes")

        save_split(splits, region_name, CONFIG["splits_dir"])

    print("\nDone.")


if __name__ == "__main__":
    main()
