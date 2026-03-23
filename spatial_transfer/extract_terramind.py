"""Extract TerraMind embeddings from full-resolution S2 minicubes.

For each cube, loads 7-band S2 at 128x128, resizes to 224x224,
runs frozen TerraMind backbone, saves (T, 768, 14, 14) embeddings.

Usage:
    python extract_terramind.py --region africa --gpu 2
    python extract_terramind.py --region latam --gpu 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

from config import CONFIG

# DeepExtremes minicubes have 7 of 12 S2 bands
AVAILABLE_BANDS = ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_NARROW"]

# TerraMind normalization stats for the 7 available bands (indices 1,2,3,4,5,6,8 in 12-band order)
BAND_INDICES_IN_12 = [1, 2, 3, 4, 5, 6, 8]
TERRAMIND_MEAN_12 = [
    1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975,
    2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648
]
TERRAMIND_STD_12 = [
    2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
    1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311
]
MEAN_7 = np.array([TERRAMIND_MEAN_12[i] for i in BAND_INDICES_IN_12]).reshape(1, 7, 1, 1)
STD_7 = np.array([TERRAMIND_STD_12[i] for i in BAND_INDICES_IN_12]).reshape(1, 7, 1, 1)


def load_backbone(device):
    backbone = TERRATORCH_BACKBONE_REGISTRY.build(
        'terramind_v1_base',
        pretrained=True,
        modalities=['untok_sen2l2a@224'],
        bands={'untok_sen2l2a@224': AVAILABLE_BANDS},
    )
    backbone.eval()
    backbone.to(device)
    return backbone


def extract_cube(backbone, s2_full, device, batch_size=8):
    """Extract embeddings for all timesteps of one cube.

    Args:
        s2_full: (T, 7, 128, 128) float16 array, reflectance in [0, 1]

    Returns:
        embeddings: (T, 768, 14, 14) float16 array
    """
    T = s2_full.shape[0]
    all_emb = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch = s2_full[start:end].astype(np.float32)

        # Scale [0,1] -> [0, 10000] reflectance
        batch = batch * 10000.0

        # Normalize with TerraMind stats
        batch = (batch - MEAN_7) / STD_7

        # Convert to tensor and resize 128 -> 224
        x = torch.from_numpy(batch).float().to(device)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        input_dict = {'untok_sen2l2a@224': x}
        with torch.no_grad():
            features = backbone(input_dict)
            last = features[-1]  # (B, 196, 768)
            B, N, C = last.shape
            H = W = int(N ** 0.5)
            emb = last.transpose(1, 2).reshape(B, C, H, W)

        all_emb.append(emb.cpu().half().numpy())

    return np.concatenate(all_emb, axis=0)  # (T, 768, 14, 14)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, choices=list(CONFIG["regions"].keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")

    splits_path = CONFIG["splits_dir"] / f"{args.region}_split.json"
    with open(splits_path) as f:
        splits = json.load(f)

    cache_dir = CONFIG["cache_dir"] / args.region
    all_cubes = [c for split_cubes in splits.values() for c in split_cubes]

    print(f"Loading TerraMind backbone...")
    backbone = load_backbone(device)
    print(f"Extracting embeddings for {len(all_cubes)} {args.region} cubes")

    done, fail = 0, 0
    for i, cube in enumerate(all_cubes):
        cube_id = cube["id"]
        s2full_path = cache_dir / f"{cube_id}_s2full.npy"
        out_path = cache_dir / f"{cube_id}_terramind.npy"

        if out_path.exists():
            print(f"  [{i+1}/{len(all_cubes)}] {cube_id} — cached")
            done += 1
            continue

        if not s2full_path.exists():
            print(f"  [{i+1}/{len(all_cubes)}] {cube_id} — MISSING s2full, skipping")
            fail += 1
            continue

        s2_full = np.load(s2full_path)  # (T, 7, 128, 128) float16
        emb = extract_cube(backbone, s2_full, device, args.batch_size)
        np.save(out_path, emb)
        done += 1
        print(f"  [{i+1}/{len(all_cubes)}] {cube_id} — {emb.shape} saved")

    print(f"\nDone: {done} extracted, {fail} failed")


if __name__ == "__main__":
    main()
