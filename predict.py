"""Prediction interface for the trained Crestix Probe Random Forest model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rf_flood.dataset import (  # noqa: E402
    build_tabular_dataset,
    generate_synthetic_ukea_events,
    load_ukea_events,
)


def load_rf_artifact(model_path: Path) -> Dict[str, Any]:
    """Load saved joblib artifact containing model, feature_names, and thresholds."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def predict_from_dataframe(
    artifact: Dict[str, Any],
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    """
    Predict continuous future water depth and binary flood flags from a tabular DataFrame,
    preserving event_id and grid-cell coordinates (grid_y, grid_x).
    Returns (predictions_df, inference_time_sec).
    """
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    flood_threshold = float(artifact.get("flood_threshold_m", 0.03))
    clip_nonneg = bool(artifact.get("clip_negative_predictions", True))

    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[feature_names].to_numpy(dtype=np.float32)

    t0 = time.perf_counter()
    preds = model.predict(X)
    inference_sec = time.perf_counter() - t0

    if clip_nonneg:
        preds = np.clip(preds, 0.0, None)

    out = pd.DataFrame(
        {
            "event_id": df["event_id"].values if "event_id" in df.columns else "unknown_event",
            "timestep": df["timestep"].values if "timestep" in df.columns else np.arange(len(df)),
            "time_min": df["time_min"].values if "time_min" in df.columns else 0.0,
            "grid_y": df["grid_y"].values if "grid_y" in df.columns else -1,
            "grid_x": df["grid_x"].values if "grid_x" in df.columns else -1,
            "predicted_depth_m": preds.astype(np.float32),
            "is_flooded": (preds >= flood_threshold).astype(int),
        }
    )
    return out, inference_sec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Crestix Probe Random Forest inference on UKEA data."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/random_forest_model.joblib"),
        help="Path to joblib model artifact.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional directory of UKEA event folders (dem.npy, h.npy, rainfall.npy).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional pre-extracted tabular feature CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions.csv"),
        help="Output CSV path for predictions with event_id and grid_y/grid_x.",
    )
    args = parser.parse_args()

    artifact = load_rf_artifact(args.model)
    cfg = artifact.get("config", {})
    ds_cfg = cfg.get("dataset", {})

    if args.input_csv is not None:
        df = pd.read_csv(args.input_csv)
    else:
        events = load_ukea_events(args.input_dir) if args.input_dir is not None else []
        if not events:
            events = generate_synthetic_ukea_events(
                num_events=2,
                grid_rows=int(ds_cfg.get("grid_rows", 32)),
                grid_cols=int(ds_cfg.get("grid_cols", 32)),
                timesteps=int(ds_cfg.get("timesteps_per_event", 24)),
                seed=99,
            )
        df, _ = build_tabular_dataset(
            events,
            lead_steps=int(ds_cfg.get("lead_steps", 6)),
            dx=float(ds_cfg.get("spatial_resolution_m", 8.0)),
            dt_min=float(ds_cfg.get("temporal_resolution_min", 5.0)),
        )

    pred_df, inf_sec = predict_from_dataframe(artifact, df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output, index=False)

    print(
        f"Generated {len(pred_df)} predictions in {inf_sec:.4f}s "
        f"-> saved to {args.output}"
    )


if __name__ == "__main__":
    main()

