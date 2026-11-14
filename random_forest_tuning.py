"""Random Forest hyperparameter tuning, overfitting analysis, and error analysis.

Person 5 deliverable for Crestix Probe — Urban Flood Forecasting.
Strictly uses event-level train/validation splits from UKEA_TRAIN_EVENTS (8 events)
and never touches the held-out 12-event test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import yaml

from rf_feature_builder import (
    FEATURE_NAMES,
    UKEA_TRAIN_EVENTS,
    build_tabular_dataset,
    get_event_splits,
)

FLOOD_THRESHOLD_M = 0.03


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if np.var(y_true) > 1e-12 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_flood_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = FLOOD_THRESHOLD_M,
) -> Dict[str, float]:
    true_flood = y_true >= threshold
    pred_flood = y_pred >= threshold

    tp = int(np.sum(true_flood & pred_flood))
    fp = int(np.sum(~true_flood & pred_flood))
    fn = int(np.sum(true_flood & ~pred_flood))
    tn = int(np.sum(~true_flood & ~pred_flood))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "csi": float(csi),
        "accuracy": float(acc),
    }


def build_candidate_configs() -> List[Dict[str, Any]]:
    """Curated hyperparameter grid exploring capacity, regularization, and subsampling."""
    configs: List[Dict[str, Any]] = []

    # 1. Baseline & n_estimators sweep
    for n_est in (50, 100, 200):
        configs.append(
            {
                "n_estimators": n_est,
                "max_depth": 12,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
            }
        )

    # 2. max_depth sweep (underfitting -> balanced -> overfitting)
    for depth in (6, 10, 16, 20, None):
        configs.append(
            {
                "n_estimators": 100,
                "max_depth": depth,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 0.5,
            }
        )

    # 3. Regularization sweep (min_samples_split & min_samples_leaf)
    for min_split in (2, 5, 10):
        for min_leaf in (1, 3, 8):
            configs.append(
                {
                    "n_estimators": 150,
                    "max_depth": 16,
                    "min_samples_split": min_split,
                    "min_samples_leaf": min_leaf,
                    "max_features": 0.5,
                }
            )

    # 4. max_features sweep
    for mf in ("sqrt", 0.5, 0.75, 1.0):
        configs.append(
            {
                "n_estimators": 150,
                "max_depth": 14,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "max_features": mf,
            }
        )

    # Deduplicate while preserving order
    seen = set()
    unique: List[Dict[str, Any]] = []
    for c in configs:
        key = tuple(sorted((k, str(v)) for k, v in c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def run_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    configs: List[Dict[str, Any]],
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], RandomForestRegressor]:
    rows: List[Dict[str, Any]] = []
    best_rmse = float("inf")
    best_cfg: Dict[str, Any] = {}
    best_model: RandomForestRegressor | None = None

    for idx, cfg in enumerate(configs):
        t0 = time.perf_counter()
        model = RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg["max_features"],
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        m_train = compute_regression_metrics(y_train, pred_train)
        m_val = compute_regression_metrics(y_val, pred_val)
        d_val = compute_flood_detection_metrics(y_val, pred_val)

        row = {
            "run_id": idx + 1,
            "n_estimators": cfg["n_estimators"],
            "max_depth": "None" if cfg["max_depth"] is None else cfg["max_depth"],
            "min_samples_split": cfg["min_samples_split"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "max_features": str(cfg["max_features"]),
            "train_mae": m_train["mae"],
            "train_rmse": m_train["rmse"],
            "train_r2": m_train["r2"],
            "val_mae": m_val["mae"],
            "val_rmse": m_val["rmse"],
            "val_r2": m_val["r2"],
            "overfit_rmse_gap": m_val["rmse"] - m_train["rmse"],
            "val_flood_precision": d_val["precision"],
            "val_flood_recall": d_val["recall"],
            "val_flood_f1": d_val["f1"],
            "val_flood_csi": d_val["csi"],
            "val_flood_accuracy": d_val["accuracy"],
            "fit_time_s": fit_time,
        }
        rows.append(row)

        if m_val["rmse"] < best_rmse:
            best_rmse = m_val["rmse"]
            best_cfg = dict(cfg)
            best_model = model

    df = pd.DataFrame(rows).sort_values("val_rmse", ascending=True).reset_index(drop=True)
    assert best_model is not None
    return df, best_cfg, best_model


def plot_feature_importance(
    model: RandomForestRegressor,
    X_val: np.ndarray,
    y_val: np.ndarray,
    dst: Path,
    seed: int = 42,
) -> pd.DataFrame:
    mdi = model.feature_importances_
    perm = permutation_importance(
        model, X_val, y_val, n_repeats=5, random_state=seed, n_jobs=-1, scoring="neg_root_mean_squared_error"
    )
    imp_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "mdi_importance": mdi,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(imp_df["feature"], imp_df["mdi_importance"], color="steelblue")
    axes[0].set_title("Gini / MDI Feature Importance")
    axes[0].set_xlabel("Relative Importance")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(
        imp_df["feature"],
        imp_df["perm_importance_mean"],
        xerr=imp_df["perm_importance_std"],
        color="darkorange",
        capsize=3,
    )
    axes[1].set_title("Validation Permutation Importance (RMSE Increase)")
    axes[1].set_xlabel("Mean RMSE Increase when Permuted (m)")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)
    return imp_df.sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)


def analyze_errors(
    model: RandomForestRegressor,
    X_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
) -> Dict[str, Any]:
    y_pred = model.predict(X_val)
    err = y_pred - y_val
    abs_err = np.abs(err)

    regimes = {
        "dry (<0.03 m)": y_val < 0.03,
        "shallow (0.03-0.30 m)": (y_val >= 0.03) & (y_val < 0.30),
        "deep (>=0.30 m)": y_val >= 0.30,
    }
    regime_stats: Dict[str, Dict[str, Any]] = {}
    for name, mask in regimes.items():
        if np.any(mask):
            m = compute_regression_metrics(y_val[mask], y_pred[mask])
            regime_stats[name] = {
                "count": int(np.sum(mask)),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias": float(np.mean(err[mask])),
            }

    event_stats: Dict[str, Dict[str, Any]] = {}
    for ev in np.unique(groups_val):
        mask = groups_val == ev
        m = compute_regression_metrics(y_val[mask], y_pred[mask])
        d = compute_flood_detection_metrics(y_val[mask], y_pred[mask])
        event_stats[str(ev)] = {
            "count": int(np.sum(mask)),
            **m,
            **d,
            "bias": float(np.mean(err[mask])),
        }

    return {
        "overall": {
            **compute_regression_metrics(y_val, y_pred),
            **compute_flood_detection_metrics(y_val, y_pred),
            "mean_bias_m": float(np.mean(err)),
            "std_error_m": float(np.std(err)),
            "p95_abs_error_m": float(np.percentile(abs_err, 95)),
            "p99_abs_error_m": float(np.percentile(abs_err, 99)),
            "max_abs_error_m": float(np.max(abs_err)),
        },
        "regimes": regime_stats,
        "per_event": event_stats,
    }


def run_event_group_cv(
    data_root: Path,
    best_cfg: Dict[str, Any],
    seed: int = 42,
) -> pd.DataFrame:
    X_all, y_all, groups_all = build_tabular_dataset(data_root, UKEA_TRAIN_EVENTS)
    gkf = GroupKFold(n_splits=4)
    rows: List[Dict[str, Any]] = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups_all), start=1):
        model = RandomForestRegressor(**best_cfg, random_state=seed, n_jobs=-1)
        model.fit(X_all[tr_idx], y_all[tr_idx])
        pred = model.predict(X_all[va_idx])
        m = compute_regression_metrics(y_all[va_idx], pred)
        d = compute_flood_detection_metrics(y_all[va_idx], pred)
        held_out = sorted(set(groups_all[va_idx].tolist()))
        rows.append(
            {
                "fold": fold,
                "val_events": ", ".join(held_out),
                **m,
                **d,
            }
        )
    return pd.DataFrame(rows)


def write_reports(
    out_dir: Path,
    results_df: pd.DataFrame,
    best_cfg: Dict[str, Any],
    imp_df: pd.DataFrame,
    err_analysis: Dict[str, Any],
    cv_df: pd.DataFrame,
    splits: Dict[str, List[str]],
) -> None:
    best_row = results_df.iloc[0]
    underfit_row = results_df[results_df["max_depth"].astype(str) == "6"].iloc[0]
    overfit_row = results_df[results_df["max_depth"].astype(str) == "None"].iloc[0]

    # 1. Error Analysis Report
    err_lines = [
        "# Random Forest Error Analysis Report (UKEA 32x32)",
        "",
        "## 1. Validation Split & Protocol",
        f"- **Training events ({len(splits['train'])}):** {', '.join(splits['train'])}",
        f"- **Validation events ({len(splits['val'])}):** {', '.join(splits['val'])}",
        f"- **Held-out test events ({len(splits['test'])}):** strictly untouched during all tuning and error analysis.",
        f"- **Flood detection threshold:** `{FLOOD_THRESHOLD_M:.2f} m`.",
        "",
        "## 2. Overfitting & Underfitting Investigation",
        "| Regime | Config Summary | Train RMSE (m) | Val RMSE (m) | Gap (Val - Train) | Train R² | Val R² |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| Underfitting (shallow) | depth=6, n_est={int(underfit_row['n_estimators'])} | {underfit_row['train_rmse']:.5f} | {underfit_row['val_rmse']:.5f} | {underfit_row['overfit_rmse_gap']:.5f} | {underfit_row['train_r2']:.4f} | {underfit_row['val_r2']:.4f} |",
        f"| Best Balanced | depth={best_row['max_depth']}, leaf={int(best_row['min_samples_leaf'])}, mf={best_row['max_features']} | {best_row['train_rmse']:.5f} | {best_row['val_rmse']:.5f} | {best_row['overfit_rmse_gap']:.5f} | {best_row['train_r2']:.4f} | {best_row['val_r2']:.4f} |",
        f"| Unconstrained (overfitting) | depth=None, leaf=1, split=2 | {overfit_row['train_rmse']:.5f} | {overfit_row['val_rmse']:.5f} | {overfit_row['overfit_rmse_gap']:.5f} | {overfit_row['train_r2']:.4f} | {overfit_row['val_r2']:.4f} |",
        "",
        "## 3. Shallow vs. Deep Flooding Performance (Validation)",
        "| Water-Depth Regime | Sample Count | MAE (m) | RMSE (m) | Mean Bias (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for rname, rvals in err_analysis["regimes"].items():
        err_lines.append(
            f"| {rname} | {rvals['count']} | {rvals['mae']:.5f} | {rvals['rmse']:.5f} | {rvals['bias']:+.5f} |"
        )

    err_lines.extend(
        [
            "",
            "## 4. Residual Distribution (Validation)",
            f"- **Mean bias:** `{err_analysis['overall']['mean_bias_m']:+.5f} m`",
            f"- **Error std:** `{err_analysis['overall']['std_error_m']:.5f} m`",
            f"- **95th percentile |error|:** `{err_analysis['overall']['p95_abs_error_m']:.5f} m`",
            f"- **99th percentile |error|:** `{err_analysis['overall']['p99_abs_error_m']:.5f} m`",
            f"- **Max |error|:** `{err_analysis['overall']['max_abs_error_m']:.5f} m`",
            "",
            "## 5. Per-Event Generalization (Validation Events)",
            "| Event ID | MAE (m) | RMSE (m) | R² | Precision | Recall | F1 | CSI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ev, ev_vals in err_analysis["per_event"].items():
        err_lines.append(
            f"| `{ev}` | {ev_vals['mae']:.5f} | {ev_vals['rmse']:.5f} | {ev_vals['r2']:.4f} | "
            f"{ev_vals['precision']:.4f} | {ev_vals['recall']:.4f} | {ev_vals['f1']:.4f} | {ev_vals['csi']:.4f} |"
        )

    err_lines.extend(
        [
            "",
            "## 6. Top Feature Importances (Validation Permutation)",
            "| Feature | MDI Importance | Permutation RMSE Increase (m) |",
            "|---|---:|---:|",
        ]
    )
    for _, frow in imp_df.iterrows():
        err_lines.append(
            f"| `{frow['feature']}` | {frow['mdi_importance']:.4f} | {frow['perm_importance_mean']:.5f} ± {frow['perm_importance_std']:.5f} |"
        )

    (out_dir / "error_analysis_report.md").write_text("\n".join(err_lines) + "\n", encoding="utf-8")

    # 2. Final Validation Report
    val_lines = [
        "# Final Random Forest Validation Report (Person 5)",
        "",
        "## 1. Coordination Note",
        "At the time of execution, `Random-Forest---Crestix-Probe` had no prior commits on `main` from Person 4. "
        "To avoid conflicts, all optimization, feature engineering, and validation deliverables are placed in separate, self-contained files (`rf_feature_builder.py`, `random_forest_tuning.py`, `best_rf_config.yaml`).",
        "",
        "## 2. Best Hyperparameter Configuration",
        "```yaml",
        yaml.safe_dump({"random_forest": best_cfg}, sort_keys=False).strip(),
        "```",
        "",
        "## 3. Validation Performance Summary",
        f"- **Configurations evaluated:** `{len(results_df)}` (logged in `hyperparameter_results.csv`)",
        f"- **Validation MAE:** `{best_row['val_mae']:.5f} m`",
        f"- **Validation RMSE:** `{best_row['val_rmse']:.5f} m`",
        f"- **Validation R²:** `{best_row['val_r2']:.4f}`",
        f"- **Flood Detection (threshold = {FLOOD_THRESHOLD_M:.2f} m):** "
        f"Precision=`{best_row['val_flood_precision']:.4f}`, Recall=`{best_row['val_flood_recall']:.4f}`, "
        f"F1=`{best_row['val_flood_f1']:.4f}`, CSI=`{best_row['val_flood_csi']:.4f}`, Accuracy=`{best_row['val_flood_accuracy']:.4f}`",
        "",
        "## 4. 4-Fold Event-Level GroupKFold Cross-Validation (All 8 Train-Pool Events)",
        "| Fold | Held-Out Events | MAE (m) | RMSE (m) | R² | F1 | CSI |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, crow in cv_df.iterrows():
        val_lines.append(
            f"| {int(crow['fold'])} | `{crow['val_events']}` | {crow['mae']:.5f} | {crow['rmse']:.5f} | "
            f"{crow['r2']:.4f} | {crow['f1']:.4f} | {crow['csi']:.4f} |"
        )
    val_lines.extend(
        [
            "",
            f"**Mean 4-Fold Event-CV RMSE:** `{cv_df['rmse'].mean():.5f} ± {cv_df['rmse'].std():.5f} m` | "
            f"**Mean R²:** `{cv_df['r2'].mean():.4f}` | **Mean CSI:** `{cv_df['csi'].mean():.4f}`",
        ]
    )
    (out_dir / "final_validation_report.md").write_text("\n".join(val_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and analyze Random Forest on UKEA 32x32.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../PINO-Crestix-Probe/data/ukea_8m_5min",
        help="Path to LarNO ukea_8m_5min dataset.",
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for deliverables.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    splits = get_event_splits(val_count=2)
    X_train, y_train, _ = build_tabular_dataset(data_root, splits["train"])
    X_val, y_val, groups_val = build_tabular_dataset(data_root, splits["val"])

    print(f"Loaded train={X_train.shape} ({len(splits['train'])} events), val={X_val.shape} ({len(splits['val'])} events)")
    print(f"Held-out test events ({len(splits['test'])}) untouched.")

    configs = build_candidate_configs()
    results_df, best_cfg, best_model = run_hyperparameter_search(
        X_train, y_train, X_val, y_val, configs, seed=args.seed
    )
    results_df.to_csv(out_dir / "hyperparameter_results.csv", index=False)

    with open(out_dir / "best_rf_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"random_forest": best_cfg}, f, sort_keys=False)
    with open(out_dir / "best_rf_config.json", "w", encoding="utf-8") as f:
        json.dump({"random_forest": best_cfg}, f, indent=2)

    imp_df = plot_feature_importance(
        best_model, X_val, y_val, out_dir / "feature_importance.png", seed=args.seed
    )
    err_analysis = analyze_errors(best_model, X_val, y_val, groups_val)
    cv_df = run_event_group_cv(data_root, best_cfg, seed=args.seed)
    write_reports(out_dir, results_df, best_cfg, imp_df, err_analysis, cv_df, splits)

    print("Best config:", best_cfg)
    print(results_df.head(5)[["run_id", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "val_mae", "val_rmse", "val_r2", "val_flood_csi"]].to_string(index=False))


if __name__ == "__main__":
    main()

