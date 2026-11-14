# Final Random Forest Validation Report (Person 5)

## 1. Coordination Note
At the time of execution, `Random-Forest---Crestix-Probe` had no prior commits on `main` from Person 4. To avoid conflicts, all optimization, feature engineering, and validation deliverables are placed in separate, self-contained files (`rf_feature_builder.py`, `random_forest_tuning.py`, `best_rf_config.yaml`).

## 2. Best Hyperparameter Configuration
```yaml
random_forest:
  n_estimators: 100
  max_depth: 6
  min_samples_split: 2
  min_samples_leaf: 1
  max_features: 0.5
```

## 3. Validation Performance Summary
- **Configurations evaluated:** `21` (logged in `hyperparameter_results.csv`)
- **Validation MAE:** `0.04246 m`
- **Validation RMSE:** `0.15085 m`
- **Validation R²:** `-2.2375`
- **Flood Detection (threshold = 0.03 m):** Precision=`0.4338`, Recall=`0.8016`, F1=`0.5629`, CSI=`0.3917`, Accuracy=`0.8632`

## 4. 4-Fold Event-Level GroupKFold Cross-Validation (All 8 Train-Pool Events)
| Fold | Held-Out Events | MAE (m) | RMSE (m) | R² | F1 | CSI |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `r300y_p0.1_d3h_1, r500y_p0.2_d3h_1` | 0.07637 | 0.22649 | 0.3365 | 0.4953 | 0.3292 |
| 2 | `r200y_p0.5_d3h_1, r300y_p0.9_d3h_1` | 0.04246 | 0.15085 | -2.2375 | 0.5629 | 0.3917 |
| 3 | `r200y_p0.3_d3h_1, r300y_p0.6_d3h_1` | 0.01442 | 0.04127 | 0.8205 | 0.7596 | 0.6123 |
| 4 | `r100y_p0.7_d3h_1, r300y_p0.4_d3h_1` | 0.01349 | 0.03425 | 0.8316 | 0.7179 | 0.5599 |

**Mean 4-Fold Event-CV RMSE:** `0.11321 ± 0.09248 m` | **Mean R²:** `-0.0622` | **Mean CSI:** `0.4733`
