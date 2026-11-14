"""Train and evaluate the Crestix Probe RandomForestRegressor on LarNO UKEA."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rf_flood.dataset import (  # noqa: E402
    build_tabular_dataset,
    generate_synthetic_ukea_events,
    load_ukea_events,
    split_events_strict,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    flood_threshold_m: float = 0.03,
) -> Dict[str, Any]:
    """Compute continuous regression metrics and thresholded flood/non-flood scores."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    flood_true = (y_true >= flood_threshold_m).astype(int)
    flood_pred = (y_pred >= flood_threshold_m).astype(int)

    return {
        "mae_m": mae,
        "rmse_m": rmse,
        "r2": r2,
        "flood_threshold_m": float(flood_threshold_m),
        "flood_accuracy": float(accuracy_score(flood_true, flood_pred)),
        "flood_precision": float(precision_score(flood_true, flood_pred, zero_division=0)),
        "flood_recall": float(recall_score(flood_true, flood_pred, zero_division=0)),
        "flood_f1": float(f1_score(flood_true, flood_pred, zero_division=0)),
        "num_samples": int(len(y_true)),
    }


def tune_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    search_grid: List[Dict[str, Any]],
    seed: int,
    n_jobs: int,
    clip_nonnegative: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Select best RandomForestRegressor hyperparameters using validation RMSE only."""
    best_params: Dict[str, Any] = search_grid[0]
    best_rmse = float("inf")
    trials: List[Dict[str, Any]] = []

    for candidate in search_grid:
        model = RandomForestRegressor(
            n_estimators=int(candidate.get("n_estimators", 100)),
            max_depth=candidate.get("max_depth", None),
            min_samples_leaf=int(candidate.get("min_samples_leaf", 2)),
            max_features=candidate.get("max_features", "sqrt"),
            random_state=seed,
            n_jobs=n_jobs,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_sec = time.perf_counter() - t0

        val_pred = model.predict(X_val)
        if clip_nonnegative:
            val_pred = np.clip(val_pred, 0.0, None)

        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        val_mae = float(mean_absolute_error(y_val, val_pred))
        trials.append(
            {
                "params": candidate,
                "val_rmse_m": val_rmse,
                "val_mae_m": val_mae,
                "fit_time_sec": fit_sec,
            }
        )
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = candidate

    return best_params, trials


def run_training(
    config_path: Path,
    data_dir: Path | None = None,
) -> Dict[str, Any]:
    """Execute end-to-end event-split training, validation tuning, and test evaluation."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("random_seed", 42))
    ds_cfg = cfg["dataset"]
    split_cfg = cfg["splitting"]
    eval_cfg = cfg["evaluation"]
    model_cfg = cfg["model"]
    out_cfg = cfg["outputs"]

    events = load_ukea_events(data_dir) if data_dir is not None else []
    data_source = f"disk:{data_dir}" if events else "synthetic_ukea_32x32"
    if not events:
        events = generate_synthetic_ukea_events(
            num_events=int(ds_cfg.get("num_synthetic_events", 10)),
            grid_rows=int(ds_cfg.get("grid_rows", 32)),
            grid_cols=int(ds_cfg.get("grid_cols", 32)),
            timesteps=int(ds_cfg.get("timesteps_per_event", 24)),
            seed=seed,
        )

    train_events, val_events, test_events = split_events_strict(
        events,
        train_ratio=float(split_cfg.get("train_ratio", 0.60)),
        val_ratio=float(split_cfg.get("val_ratio", 0.20)),
        test_ratio=float(split_cfg.get("test_ratio", 0.20)),
        seed=seed,
    )

    lead_steps = int(ds_cfg.get("lead_steps", 6))
    dx = float(ds_cfg.get("spatial_resolution_m", 8.0))
    dt_min = float(ds_cfg.get("temporal_resolution_min", 5.0))

    train_df, feature_cols = build_tabular_dataset(train_events, lead_steps, dx, dt_min)
    val_df, _ = build_tabular_dataset(val_events, lead_steps, dx, dt_min)
    test_df, _ = build_tabular_dataset(test_events, lead_steps, dx, dt_min)

    # Leakage assertion: zero event overlap
    assert set(train_df["event_id"]).isdisjoint(set(val_df["event_id"]))
    assert set(train_df["event_id"]).isdisjoint(set(test_df["event_id"]))
    assert set(val_df["event_id"]).isdisjoint(set(test_df["event_id"]))

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["target_depth_m"].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["target_depth_m"].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["target_depth_m"].to_numpy(dtype=np.float32)

    clip_nonneg = bool(eval_cfg.get("clip_negative_predictions", True))
    flood_threshold = float(eval_cfg.get("flood_threshold_m", 0.03))
    n_jobs = int(model_cfg.get("n_jobs", -1))

    t_tune_start = time.perf_counter()
    best_params, tuning_trials = tune_on_validation(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        search_grid=model_cfg["search_grid"],
        seed=seed,
        n_jobs=n_jobs,
        clip_nonnegative=clip_nonneg,
    )
    tuning_time_sec = time.perf_counter() - t_tune_start

    # Fit final model on training split ONLY (test set never touched)
    final_model = RandomForestRegressor(
        n_estimators=int(best_params.get("n_estimators", 100)),
        max_depth=best_params.get("max_depth", None),
        min_samples_leaf=int(best_params.get("min_samples_leaf", 2)),
        max_features=best_params.get("max_features", "sqrt"),
        random_state=seed,
        n_jobs=n_jobs,
    )
    t_train_start = time.perf_counter()
    final_model.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - t_train_start

    # Validation inference
    t_val_inf = time.perf_counter()
    val_pred = final_model.predict(X_val)
    val_inf_sec = time.perf_counter() - t_val_inf
    if clip_nonneg:
        val_pred = np.clip(val_pred, 0.0, None)

    # Test inference (held-out events)
    t_test_inf = time.perf_counter()
    test_pred = final_model.predict(X_test)
    test_inf_sec = time.perf_counter() - t_test_inf
    if clip_nonneg:
        test_pred = np.clip(test_pred, 0.0, None)

    val_metrics = compute_metrics(y_val, val_pred, flood_threshold)
    test_metrics = compute_metrics(y_test, test_pred, flood_threshold)

    # Save model artifact via joblib
    model_path = Path(out_cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_names": feature_cols,
        "config": cfg,
        "best_params": best_params,
        "flood_threshold_m": flood_threshold,
        "clip_negative_predictions": clip_nonneg,
    }
    joblib.dump(artifact, model_path, compress=3)

    # Save test predictions with event_id and grid-cell coordinates (grid_y, grid_x)
    pred_path = Path(out_cfg["predictions_path"])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            "event_id": test_df["event_id"].values,
            "timestep": test_df["timestep"].values,
            "time_min": test_df["time_min"].values,
            "grid_y": test_df["grid_y"].values,
            "grid_x": test_df["grid_x"].values,
            "y_true_m": y_test,
            "y_pred_m": test_pred.astype(np.float32),
            "flood_true": (y_test >= flood_threshold).astype(int),
            "flood_pred": (test_pred >= flood_threshold).astype(int),
        }
    )
    pred_df.to_csv(pred_path, index=False)

    # Top feature importances
    importances = sorted(
        zip(feature_cols, final_model.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    summary = {
        "data_source": data_source,
        "splits": {
            "train_events": [e.event_id for e in train_events],
            "val_events": [e.event_id for e in val_events],
            "test_events": [e.event_id for e in test_events],
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
        },
        "best_hyperparameters": best_params,
        "tuning_trials": tuning_trials,
        "timings": {
            "tuning_time_sec": round(tuning_time_sec, 4),
            "train_time_sec": round(train_time_sec, 4),
            "val_inference_time_sec": round(val_inf_sec, 4),
            "test_inference_time_sec": round(test_inf_sec, 4),
            "test_ms_per_sample": round((test_inf_sec / max(1, len(test_df))) * 1000.0, 6),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "top_10_features": [
            {"feature": name, "importance": round(val, 6)} for name, val in importances[:10]
        ],
        "reproducibility": {
            "random_seed": seed,
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "joblib_version": joblib.__version__,
            "model_artifact": str(model_path),
            "predictions_csv": str(pred_path),
        },
    }

    metrics_path = Path(out_cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RandomForestRegressor on LarNO UKEA 32x32 flood dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rf_config.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional directory of UKEA event folders (dem.npy, h.npy, rainfall.npy).",
    )
    args = parser.parse_args()

    summary = run_training(config_path=args.config, data_dir=args.data_dir)
    tm = summary["test_metrics"]
    timings = summary["timings"]
    print(f"Best params (selected on val): {summary['best_hyperparameters']}")
    print(f"Train time: {timings['train_time_sec']:.3f}s | Test inference: {timings['test_inference_time_sec']:.3f}s")
    print(
        f"Test Metrics -> MAE: {tm['mae_m']:.4f} m | RMSE: {tm['rmse_m']:.4f} m | R2: {tm['r2']:.4f} | "
        f"Flood Acc: {tm['flood_accuracy']:.4f} | Flood F1: {tm['flood_f1']:.4f}"
    )


if __name__ == "__main__":
    main()

