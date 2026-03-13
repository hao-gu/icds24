# Low-Data Deep Transfer Learning for Rainforest Biomass Estimation

A remote sensing research project that uses NASA's GEDI (Global Ecosystem Dynamics Investigation) lidar satellite data alongside ERA5 climate reanalysis data to train neural networks that predict above-ground biomass density (AGBD) in tropical forests.

Two study regions are compared: the **Amazon rainforest** and the **Congo Basin**.

---

## Overview

The project constructs a machine learning pipeline that:
1. Downloads and processes GEDI L4A biomass data from NASA
2. Aggregates measurements to a 0.25° spatial grid
3. Interpolates ERA5 climate variables to match GEDI grid points
4. Trains artificial neural networks (ANNs) to predict biomass from climate and geographic features
5. Compares results across the Amazon and Congo regions

**Target variable:** Above-ground biomass density (AGBD, Mg/ha)

**Input features:** longitude, latitude, LAI, temperature, precipitation, radiation, pressure, wind speed

---

## Repository Structure

```
.
├── src/
│   ├── amazonCodes/          # Data processing and modeling scripts for the Amazon
│   └── congoCodes/           # Data processing and modeling scripts for the Congo Basin
├── Amazon_GEDI_data/         # Raw GEDI L4A HDF5 granules (Amazon region)
└── Congo_GEDI_data/          # Raw GEDI L4A HDF5 granules (Congo region)
```

### `src/amazonCodes/`

| Script | Description |
|--------|-------------|
| `example_data_processing_part1.py` | Downloads GEDI L4A granules via NASA CMR API for the Amazon bounding box |
| `data_aggregate.py` | Reads HDF5 files, extracts AGBD/lat/lon, aggregates to 0.25° grid |
| `interpolate_era5.py` | Interpolates ERA5 climate variables to GEDI grid points |
| `lai_map.py` | Processes LAI (leaf area index) from NetCDF4 files |
| `initial_ANN.py` | Trains a baseline ANN on Amazon data |
| `graphAmazonNPY.py` | Generates publication-ready biomass maps |
| `violinplots.py` | Violin plot comparison of climate variables between Amazon and Congo |
| `GEDI_visualization.py` | Scatter plots of GEDI biomass with geospatial context |

### `src/congoCodes/`

| Script | Description |
|--------|-------------|
| `congo_data_processing.py` | Reads GEDI H5 files and aggregates biomass to 0.25° grid for the Congo |
| `ML_modeling_optuna.py` | Core hyperparameter optimization using Optuna (200+ trials per model) |
| `Congo_ANN.py` | Baseline ANN trained on Congo data |
| `figure2.py` | Publication-ready correlation heatmaps (Amazon vs. Congo) |
| `check_distr.py` | Latitudinal/longitudinal distributions of climate variables |
| `CongoNPYgraphs.py` | Maps Congo GEDI biomass with African continent basemap |
| `heatmap.py` | Utility module for annotated heatmap visualizations |
| `GEDI_visualization.py` | Scatter plots of Congo GEDI biomass |

---

## Data Pipeline

```
GEDI L4A HDF5 files
        ↓
  Beam extraction (8 beams per granule)
  Filter invalid/negative AGBD values
        ↓
  Aggregate to 0.25° grid (mean per cell)
        ↓
  ERA5 NetCDF4 (temp, precip, radiation, pressure, wind)
  LAI NetCDF4
        ↓
  Nearest-neighbor interpolation to GEDI grid points
        ↓
  Feature matrix: [lon, lat, LAI, T, precip, rad, P, wind]
        ↓
  ANN training (TensorFlow/Keras + Optuna tuning)
```

---

## Machine Learning

Models are tuned using [Optuna](https://optuna.org/) with the following search space:

- **Layers:** 5–10
- **Units per layer:** 5–15
- **Activations:** ReLU, Softplus, Sigmoid
- **Optimizers:** Adam, SGD, RMSprop
- **Learning rate:** 1×10⁻⁵ – 1×10⁻³

**Metrics:** MSE, RMSE, R²

Train/test split: 80/20 with stratified sampling.

---

## Data Formats

| Format | Usage |
|--------|-------|
| `.h5` | Raw GEDI L4A satellite granules |
| `.nc` | ERA5 climate data |
| `.npy` / `.npz` | Processed arrays (biomass, climate variables) |
| `.csv` | Training data splits, Optuna trial results |
| `.keras` | Saved trained models |
| `.pkl` | Optuna samplers (for reproducibility) |

---

## Dependencies

- `tensorflow` / `keras`
- `optuna`
- `numpy`, `scipy`, `pandas`
- `matplotlib`, `geopandas`
- `h5py`
- `netCDF4`
- `scikit-learn`
