# Random Forest Error Analysis Report (UKEA 32x32)

## 1. Validation Split & Protocol
- **Training events (6):** r300y_p0.1_d3h_1, r300y_p0.4_d3h_1, r200y_p0.3_d3h_1, r100y_p0.7_d3h_1, r500y_p0.2_d3h_1, r300y_p0.6_d3h_1
- **Validation events (2):** r200y_p0.5_d3h_1, r300y_p0.9_d3h_1
- **Held-out test events (12):** strictly untouched during all tuning and error analysis.
- **Flood detection threshold:** `0.03 m`.

## 2. Overfitting & Underfitting Investigation
| Regime | Config Summary | Train RMSE (m) | Val RMSE (m) | Gap (Val - Train) | Train R² | Val R² |
|---|---|---:|---:|---:|---:|---:|
| Underfitting (shallow) | depth=6, n_est=100 | 0.05646 | 0.15085 | 0.09439 | 0.9013 | -2.2375 |
| Best Balanced | depth=6, leaf=1, mf=0.5 | 0.05646 | 0.15085 | 0.09439 | 0.9013 | -2.2375 |
| Unconstrained (overfitting) | depth=None, leaf=1, split=2 | 0.00767 | 0.20813 | 0.20046 | 0.9982 | -5.1632 |

## 3. Shallow vs. Deep Flooding Performance (Validation)
| Water-Depth Regime | Sample Count | MAE (m) | RMSE (m) | Mean Bias (m) |
|---|---:|---:|---:|---:|
| dry (<0.03 m) | 8203 | 0.02377 | 0.09609 | +0.02290 |
| shallow (0.03-0.30 m) | 789 | 0.15724 | 0.33369 | +0.12577 |
| deep (>=0.30 m) | 224 | 0.32273 | 0.45368 | +0.29269 |

## 4. Residual Distribution (Validation)
- **Mean bias:** `+0.03826 m`
- **Error std:** `0.14591 m`
- **95th percentile |error|:** `0.16967 m`
- **99th percentile |error|:** `0.87063 m`
- **Max |error|:** `1.01229 m`

## 5. Per-Event Generalization (Validation Events)
| Event ID | MAE (m) | RMSE (m) | R² | Precision | Recall | F1 | CSI |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r200y_p0.5_d3h_1` | 0.00966 | 0.02040 | 0.9411 | 0.7967 | 0.7762 | 0.7864 | 0.6479 |
| `r300y_p0.9_d3h_1` | 0.07526 | 0.21235 | -5.4535 | 0.3043 | 0.8268 | 0.4449 | 0.2861 |

## 6. Top Feature Importances (Validation Permutation)
| Feature | MDI Importance | Permutation RMSE Increase (m) |
|---|---:|---:|
| `dem` | 0.4409 | 0.00621 ± 0.00249 |
| `h_init` | 0.0088 | 0.00000 ± 0.00000 |
| `slope_mag` | 0.0058 | -0.00039 ± 0.00007 |
| `slope_y` | 0.0058 | -0.00058 ± 0.00005 |
| `slope_x` | 0.0114 | -0.00143 ± 0.00017 |
| `rain_t` | 0.0026 | -0.00192 ± 0.00006 |
| `elev_above_local_min` | 0.0413 | -0.00295 ± 0.00026 |
| `rain_prev2` | 0.0251 | -0.01407 ± 0.00057 |
| `rain_prev1` | 0.0339 | -0.01490 ± 0.00047 |
| `t_norm` | 0.0763 | -0.01713 ± 0.00074 |
| `rain_cum` | 0.3483 | -0.06340 ± 0.00172 |
