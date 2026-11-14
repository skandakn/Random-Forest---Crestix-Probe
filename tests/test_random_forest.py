"""Unit tests for Random-Forest---Crestix-Probe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from predict import load_rf_artifact, predict_from_dataframe  # noqa: E402
from rf_flood.dataset import (  # noqa: E402
    build_tabular_dataset,
    generate_synthetic_ukea_events,
    split_events_strict,
)
from train_random_forest import compute_metrics, run_training  # noqa: E402


def test_event_split_has_zero_leakage():
    events = generate_synthetic_ukea_events(num_events=10, timesteps=16, seed=42)
    train_ev, val_ev, test_ev = split_events_strict(events, 0.6, 0.2, 0.2, seed=42)
    assert len(train_ev) == 6
    assert len(val_ev) == 2
    assert len(test_ev) == 2
    train_ids = {e.event_id for e in train_ev}
    val_ids = {e.event_id for e in val_ev}
    test_ids = {e.event_id for e in test_ev}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_compute_metrics_regression_and_flood_threshold():
    y_true = np.array([0.00, 0.02, 0.10, 0.50], dtype=np.float32)
    y_pred = np.array([0.00, 0.01, 0.12, 0.48], dtype=np.float32)
    m = compute_metrics(y_true, y_pred, flood_threshold_m=0.03)
    assert m["mae_m"] < 0.02
    assert m["rmse_m"] < 0.02
    assert m["r2"] > 0.99
    assert m["flood_accuracy"] == 1.0
    assert m["flood_f1"] == 1.0


def test_training_and_prediction_roundtrip(tmp_path):
    cfg_path = tmp_path / "rf_test.yaml"
    model_path = tmp_path / "rf.joblib"
    metrics_path = tmp_path / "metrics.json"
    preds_path = tmp_path / "preds.csv"
    cfg_path.write_text(
        f"""
random_seed: 7
dataset:
  grid_rows: 16
  grid_cols: 16
  spatial_resolution_m: 8.0
  temporal_resolution_min: 5.0
  lead_steps: 3
  num_synthetic_events: 5
  timesteps_per_event: 12
splitting:
  strategy: "event_level"
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
evaluation:
  flood_threshold_m: 0.03
  clip_negative_predictions: true
model:
  search_grid:
    - n_estimators: 10
      max_depth: 6
      min_samples_leaf: 2
      max_features: "sqrt"
  n_jobs: 1
outputs:
  model_path: "{model_path.as_posix()}"
  metrics_path: "{metrics_path.as_posix()}"
  predictions_path: "{preds_path.as_posix()}"
""",
        encoding="utf-8",
    )

    summary = run_training(config_path=cfg_path)
    assert model_path.exists()
    assert metrics_path.exists()
    assert preds_path.exists()
    assert summary["timings"]["train_time_sec"] >= 0.0

    artifact = load_rf_artifact(model_path)
    events = generate_synthetic_ukea_events(num_events=1, grid_rows=16, grid_cols=16, timesteps=12, seed=99)
    df, _ = build_tabular_dataset(events, lead_steps=3)
    pred_df, inf_sec = predict_from_dataframe(artifact, df)

    assert inf_sec >= 0.0
    assert {"event_id", "grid_y", "grid_x", "predicted_depth_m", "is_flooded"}.issubset(pred_df.columns)
    assert (pred_df["predicted_depth_m"] >= 0.0).all()

