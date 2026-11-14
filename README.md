# Random-Forest---Crestix-Probe

Random Forest Regressor baseline for **Crestix Probe — Urban Flood Forecasting** on the **LarNO Urban Flood Dataset (UKEA 32×32 subset)**.

---

## Overview

- **Model**: `sklearn.ensemble.RandomForestRegressor` predicting continuous future water depth $h(t + 30\text{ min}, y, x)$ [m] at individual spatial grid cells.
- **Zero Data Leakage**: Strict **event-level** train / validation / test split (`60% / 20% / 20%`). Test events are never used for training or hyperparameter tuning.
- **Validation-Only Tuning**: Hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`) are selected strictly using validation RMSE.
- **Flood / Non-Flood Evaluation**: Binary flood classification scores (Accuracy, Precision, Recall, F1) computed at the common UKEA threshold (`0.03 m`).

---

## Repository Structure

```text
Random-Forest---Crestix-Probe/
├── configs/
│   └── rf_config.yaml               # Dataset, split, search grid, threshold, and output paths
├── src/rf_flood/
│   ├── __init__.py
│   └── dataset.py                   # UKEA 32x32 loader, common feature engineering, event-level splitter
├── artifacts/
│   └── random_forest_model.joblib   # Trained RandomForestRegressor artifact (joblib)
├── outputs/
│   ├── evaluation_metrics.json      # MAE, RMSE, R², flood scores, timings, reproducibility info
│   └── test_predictions.csv         # Held-out predictions with event_id and (grid_y, grid_x)
├── tests/
│   └── test_random_forest.py        # Unit & end-to-end tests
├── train_random_forest.py           # Training, validation tuning, evaluation & artifact export
├── predict.py                       # Prediction CLI & Python API
├── requirements.txt
└── README.md
```

---

## Common Feature Specification

All grid-cell observations at `(event_id, timestep, grid_y, grid_x)` include:
1. **Topographic (DEM)**: `elevation_m`, `slope`, `grad_x`, `grad_y`, `aspect_sin`, `aspect_cos`, `laplacian`, `roughness_3x3`, `tpi`, `is_building_mask`
2. **Meteorological (Rainfall)**: `rain_intensity_mm_5min`, `rain_cum_mm`, `rain_roll_sum_15m`, `rain_roll_sum_30m`, `rain_roll_sum_60m`, `rain_peak_so_far`, `rain_roc`
3. **Hydraulic State & Antecedent Lags**: `depth_t_m`, `depth_lag_5m`, `depth_lag_10m`, `depth_lag_15m`, `depth_roc`, `depth_nbr_mean_3x3`, `depth_nbr_max_3x3`, `hydraulic_head_m`, `hydraulic_grad`
4. **Spatio-Temporal Coordinates**: `norm_y`, `norm_x`, `dist_to_center`, `time_elapsed_min`, `time_fraction`

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train, tune on validation split, evaluate on held-out test events, and save joblib model
python train_random_forest.py --config configs/rf_config.yaml
# Optional: pass real LarNO UKEA event directory
# python train_random_forest.py --config configs/rf_config.yaml --data-dir path/to/ukea_events

# 2. Run inference with saved joblib artifact (outputs event_id, grid_y, grid_x, predicted_depth_m)
python predict.py --model artifacts/random_forest_model.joblib --output outputs/predictions.csv

# 3. Run unit tests
pytest -v
```

