"""Select and split minicubes for 3-continent spatial transfer experiment.

Uses pre-selected broadleaf tree cubes from selected_minicubes.csv.
Samples equal number per continent with stratified train/val/test splits.

Usage:
    python data_select_v2.py --n-per-continent 200
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from config import CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-continent", type=int, default=200)
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = parser.parse_args()

    random.seed(args.seed)

    df = pd.read_csv(CONFIG["base_dir"] / "selected_minicubes.csv")

    # Normalize continent names
    continent_map = {
        "Africa": "africa",
        "South America": "south_america",
        "Central America": "central_america",
    }
    df["continent"] = df["Continent"].map(continent_map)

    splits_dir = CONFIG["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    ratios = CONFIG["split_ratios"]
    n = args.n_per_continent

    for continent_label, continent_key in continent_map.items():
        subset = df[df["Continent"] == continent_label]
        print(f"{continent_key}: {len(subset)} available, sampling {n}")

        if len(subset) < n:
            print(f"  Warning: only {len(subset)} available, using all")
            sampled = subset
        else:
            sampled = subset.sample(n=n, random_state=args.seed)

        cubes = []
        for _, row in sampled.iterrows():
            cubes.append({
                "id": row["id"],
                "path": row["path"],
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "land_cover": row["class"],
            })

        random.shuffle(cubes)
        n_train = int(len(cubes) * ratios["train"])
        n_val = int(len(cubes) * ratios["val"])

        splits = {
            "train": cubes[:n_train],
            "val": cubes[n_train:n_train + n_val],
            "test": cubes[n_train + n_val:],
        }

        out_path = splits_dir / f"{continent_key}_split.json"
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)

        print(f"  train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
