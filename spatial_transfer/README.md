# Spatial Transferability of Vegetation Forecasting Models

Does a model trained on African vegetation transfer to Latin America (and vice versa)?
We train several spatio-temporal architectures on one continent and evaluate on the other,
using the DeepExtremes minicube dataset (Sentinel-2 + ERA5).

Based on Benson et al., "Multi-modal Learning for Geospatial Vegetation Forecasting," CVPR 2024.

## Setup

The experiment follows Benson et al.'s protocol:

- **Context:** 10 timesteps (50 days) of 4-band Sentinel-2 (B02, B03, B04, B8A) + ERA5 + DEM
- **Forecast:** 20 timesteps (100 days) of vegetation index anomaly
- **Spatial resolution:** 64x64 pixels at 20m (center-cropped from 128x128)
- **Target:** VI anomaly = vegetation index - Mean Seasonal Cycle (supports NDVI and EVI)

40 minicubes per region, stratified by land cover class and split 60/15/25 into train/val/test.

## Data

Source: DeepExtremes v1.3 zarr archives on S3 

Each cached `.npz` file contains:

| Key          | Shape           | Description                            |
|--------------|-----------------|----------------------------------------|
| `s2_bands`   | (365, 4, 64, 64) | Cloud-filled S2 reflectance [0, 1]   |
| `dem`        | (64, 64)        | Copernicus DEM, z-normalized           |
| `ndvi_filled`| (365, 64, 64)   | Gap-filled vegetation index            |
| `ndvi_msc`   | (365, 64, 64)   | Smoothed seasonal climatology          |
| `anomaly`    | (365, 64, 64)   | VI - MSC                               |
| `qc_mask`    | (365, 64, 64)   | 1 = clear, 0 = cloud/invalid          |
| `era5`       | (365, 6)        | Normalized temperature, precip, etc.   |

365 = 5 years (2017-2021) at 5-day resolution.

## Models

| Name               | Type                      | Params |
|--------------------|---------------------------|--------|
| PixelMLP           | Per-pixel feedforward     | ~180K  |
| PixelLSTM          | Per-pixel LSTM            | ~105K  |
| MiniUNet           | 3-level U-Net, autoregressive | ~1.9M |
| ConvLSTM           | Stacked ConvLSTM          | ~930K  |
| ContextUNet        | ConvGRU encoder + U-Net   | ~4.5M  |
| TerraMindForecaster| TerraMind embeddings + U-Net | ~5M  |

All models predict vegetation index anomaly except ContextUNet and TerraMindForecaster,
which predict the absolute vegetation index as a delta from the last observation.

## Usage

Everything runs from this directory. Edit `config.py` to change hyperparameters or regions.

### 1. Select minicubes and create splits

```
python data_select.py
```

Queries the S3 registry, filters by bounding box, stratified-samples 40 cubes per region,
and writes `outputs/splits/{region}_split.json`.

### 2. Download and preprocess

```
python data_download.py --region africa
python data_download.py --region latam
```

Downloads zarr archives from S3, computes S2 bands / vegetation index / anomaly / ERA5, and caches
as compressed `.npz` under `outputs/cache/{region}/`. Takes ~10-20 min per region.
Cubes with <20% clear pixels are rejected automatically.

Pass `--s2-full` to also save all 7 S2 bands at 128x128 (needed for TerraMind extraction).

To check cached data without re-downloading:

```
python data_download.py --region africa --validate-only
```

### 3. (Optional) Extract TerraMind embeddings

Required only for TerraMindForecaster:

```
python extract_terramind.py --region africa
```

### 4. Train

```
CUDA_VISIBLE_DEVICES=0 python train.py --model ConvLSTM --region africa
```

Trains with AdamW + cosine annealing + early stopping. Checkpoints are saved to
`outputs/checkpoints/{region}_{model}/model.pt`.

Other flags: `--epochs 50`, `--lr 1e-3`, `--patience 10`, `--batch-size 16`.

### 5. Evaluate

```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model ConvLSTM --train-region africa --test-region latam
```

Computes anomaly RMSE/MAE per lead time, R^2, vegetation index R^2, and outperformance vs. climatology.
Results are saved to `outputs/metrics/{train}_on_{test}_{model}.json`.

### 6. Analyze

```
python analyze.py
```

Reads all metrics JSONs, prints summary and transfer-gap tables, and generates
PDF/PNG plots under `outputs/plots/`.

### Run everything at once

```
python run_experiment.py
```

Runs steps 1-6 end-to-end, parallelizing training across 8 GPUs and evaluation across
all 16 train/test combinations.

