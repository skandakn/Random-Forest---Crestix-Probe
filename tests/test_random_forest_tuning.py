"""Unit tests for Random Forest feature builder, metrics, and tuning pipeline."""

from __future__ import annotations

import numpy as np

from random_forest_tuning import (
    compute_flood_detection_metrics,
    compute_regression_metrics,
    run_hyperparameter_search,
)
from rf_feature_builder import FEATURE_NAMES, get_event_splits


def test_event_splits_disjoint_and_test_untouched() -> None:
    splits = get_event_splits(val_count=2)
    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 12
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])


def test_metrics_and_flood_threshold() -> None:
    y_true = np.array([0.0, 0.02, 0.05, 0.40], dtype=np.float32)
    y_pred = np.array([0.0, 0.04, 0.05, 0.30], dtype=np.float32)

    reg = compute_regression_metrics(y_true, y_pred)
    assert reg["mae"] > 0
    assert reg["rmse"] >= reg["mae"]
    assert 0.0 < reg["r2"] <= 1.0

    det = compute_flood_detection_metrics(y_true, y_pred, threshold=0.03)
    # true: [F, F, T, T], pred: [F, T, T, T] -> TP=2, FP=1, FN=0, TN=1
    assert np.isclose(det["precision"], 2 / 3)
    assert np.isclose(det["recall"], 1.0)
    assert np.isclose(det["csi"], 2 / 3)


def test_hyperparameter_search_selects_best_on_val() -> None:
    rng = np.random.RandomState(42)
    X_train = rng.randn(120, len(FEATURE_NAMES)).astype(np.float32)
    y_train = (0.5 * X_train[:, 0] + 0.2 * X_train[:, 1] ** 2).astype(np.float32)
    X_val = rng.randn(60, len(FEATURE_NAMES)).astype(np.float32)
    y_val = (0.5 * X_val[:, 0] + 0.2 * X_val[:, 1] ** 2).astype(np.float32)

    configs = [
        {"n_estimators": 10, "max_depth": 2, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0},
        {"n_estimators": 30, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0},
    ]
    df, best_cfg, model = run_hyperparameter_search(X_train, y_train, X_val, y_val, configs, seed=42)
    assert len(df) == 2
    assert best_cfg["max_depth"] == 10
    assert model.predict(X_val).shape == (60,)

