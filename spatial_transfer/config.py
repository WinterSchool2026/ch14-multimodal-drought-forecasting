"""Central configuration for the spatial transferability experiment."""

from pathlib import Path

CONFIG = {
    "seed": 42,

    "regions": {
        "africa": {"lon_range": [-18, 52], "lat_range": [-35, 37]},
        "latam": {"lon_range": [-118, -34], "lat_range": [-56, 32]},
        "south_america": {"lon_range": [-82, -34], "lat_range": [-56, 13]},
        "central_america": {"lon_range": [-118, -60], "lat_range": [7, 32]},
    },

    "n_cubes_per_region": 40,
    "split_ratios": {"train": 0.60, "val": 0.15, "test": 0.25},
    "min_valid_qc_fraction": 0.20,

    "s3_endpoint": "https://s3.bgc-jena.mpg.de:9000",
    "s3_bucket": "earthnet/deepextremes/",

    "context_length": 10,       # 50 days
    "forecast_horizon": 20,     # 100 days
    "n_s2_bands": 4,
    "n_era5": 6,
    "n_input_channels": 11,
    "batch_size": 16,
    "epochs": 50,
    "lr": 1e-3,
    "patience": 10,
    "models": ["PixelMLP", "PixelLSTM", "MiniUNet", "ConvLSTM", "ContextUNet", "TerraMindForecaster"],

    "experiment_matrix": [
        ("africa", "africa"),
        ("africa", "latam"),
        ("latam", "latam"),
        ("latam", "africa"),
    ],

    "s2_bands": ["B02", "B03", "B04", "B8A"],

    "base_dir": Path("/mnt/data/benchmark/ch14-multimodal-drought-forecasting/spatial_transfer"),
}

CONFIG["cache_dir"] = CONFIG["base_dir"] / "outputs" / "cache"
CONFIG["splits_dir"] = CONFIG["base_dir"] / "outputs" / "splits"
CONFIG["checkpoints_dir"] = CONFIG["base_dir"] / "outputs" / "checkpoints"
CONFIG["metrics_dir"] = CONFIG["base_dir"] / "outputs" / "metrics"
CONFIG["plots_dir"] = CONFIG["base_dir"] / "outputs" / "plots"

ERA5_VARS = ["t2m_mean", "tp_mean", "sp_mean", "ssr_mean", "e_mean", "pev_mean"]
ERA5_NORM = {
    "t2m_mean":  {"shift": 273.15, "scale": 30},
    "tp_mean":   {"shift": 0,      "scale": 100},
    "sp_mean":   {"shift": 100000, "scale": 10000},
    "ssr_mean":  {"shift": 0,      "scale": 30},
    "e_mean":    {"shift": 0,      "scale": 3},
    "pev_mean":  {"shift": 0,      "scale": 3},
}
