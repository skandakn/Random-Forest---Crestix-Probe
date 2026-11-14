"""Feature engineering and event-level dataset builder for Random Forest flood forecasting.

Uses the LarNO UKEA 32x32 region with strict event-level splitting:
- Train: 6 official training events
- Validation: 2 official training events (used for all hyperparameter tuning & error analysis)
- Test: 12 official test events (never touched during hyperparameter tuning)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

UKEA_TRAIN_EVENTS: Tuple[str, ...] = (
    "r300y_p0.1_d3h_1",
    "r300y_p0.4_d3h_1",
    "r200y_p0.3_d3h_1",
    "r100y_p0.7_d3h_1",
    "r500y_p0.2_d3h_1",
    "r300y_p0.6_d3h_1",
    "r200y_p0.5_d3h_1",
    "r300y_p0.9_d3h_1",
)

UKEA_TEST_EVENTS: Tuple[str, ...] = (
    "r100y_p0.6_d3h_1",
    "r100y_p0.1_d3h_1",
    "r300y_p0.8_d3h_1",
    "r200y_p0.7_d3h_1",
    "r500y_p0.3_d3h_1",
    "r500y_p0.9_d3h_1",
    "r500y_p0.1_d3h_1",
    "r500y_p0.6_d3h_1",
    "r100y_p0.5_d3h_1",
    "r500y_p0.4_d3h_1",
    "r100y_p0.8_d3h_1",
    "r300y_p0.5_d3h_1",
)

FEATURE_NAMES: List[str] = [
    "dem",
    "slope_x",
    "slope_y",
    "slope_mag",
    "elev_above_local_min",
    "t_norm",
    "h_init",
    "rain_t",
    "rain_prev1",
    "rain_prev2",
    "rain_cum",
]


def get_event_splits(val_count: int = 2) -> Dict[str, List[str]]:
    """Return disjoint event-level train, val, and test splits."""
    train_pool = list(UKEA_TRAIN_EVENTS)
    return {
        "train": train_pool[:-val_count],
        "val": train_pool[-val_count:],
        "test": list(UKEA_TEST_EVENTS),
    }


def _compute_terrain_features(dem_32: np.ndarray, dx: float = 8.0) -> Tuple[np.ndarray, ...]:
    dem_clean = np.nan_to_num(dem_32, nan=float(np.nanmax(dem_32) + 50.0)).astype(np.float32)
    gy, gx = np.gradient(dem_clean, dx, dx)
    slope_mag = np.hypot(gx, gy).astype(np.float32)

    padded = np.pad(dem_clean, 1, mode="edge")
    local_min = np.full_like(dem_clean, np.inf)
    for dy in range(3):
        for dx_i in range(3):
            local_min = np.minimum(local_min, padded[dy : dy + 32, dx_i : dx_i + 32])
    elev_above_local_min = (dem_clean - local_min).astype(np.float32)
    return dem_clean, gx.astype(np.float32), gy.astype(np.float32), slope_mag, elev_above_local_min


def build_tabular_dataset(
    data_root: str | Path,
    event_ids: Sequence[str],
    crop_y: int = 9,
    crop_x: int = 44,
    region_size: int = 32,
    spatial_stride: int = 2,
    temporal_stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, event_groups) tabular arrays for Random Forest from UKEA events.

    Uses spatial/temporal striding by default so hyperparameter sweeps run cleanly
    and fast in memory while covering all events and flood regimes.
    """
    root = Path(data_root)
    dem_path = root / "geodata" / "dem.npy"
    if not dem_path.exists():
        dem_path = root / "geodata" / "ukea_8m_5min" / "dem.npy"
    dem_full = np.load(dem_path, mmap_mode="r")
    dem_32 = np.array(
        dem_full[crop_y : crop_y + region_size, crop_x : crop_x + region_size], copy=True
    )
    dem_clean, sx, sy, smag, elev_diff = _compute_terrain_features(dem_32)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[np.ndarray] = []

    for ev_id in event_ids:
        ev_dir = root / "flood" / ev_id
        if not ev_dir.exists():
            ev_dir = root / "flood" / "ukea_8m_5min" / ev_id

        h_arr = np.array(
            np.load(ev_dir / "h.npy", mmap_mode="r")[
                :, crop_y : crop_y + region_size, crop_x : crop_x + region_size
            ],
            dtype=np.float32,
        )
        r_raw = np.array(
            np.load(ev_dir / "rainfall.npy", mmap_mode="r")[
                :, crop_y : crop_y + region_size, crop_x : crop_x + region_size
            ],
            dtype=np.float32,
        )

        t_steps = h_arr.shape[0]
        if r_raw.shape[0] % t_steps == 0:
            factor = r_raw.shape[0] // t_steps
            rain = r_raw.reshape(t_steps, factor, region_size, region_size).sum(axis=1)
        else:
            rain = r_raw[:t_steps]

        rain_cum = np.cumsum(rain, axis=0)
        h_init = h_arr[0]

        t_indices = range(0, t_steps, temporal_stride)
        for t in t_indices:
            r_t = rain[t, ::spatial_stride, ::spatial_stride]
            r_p1 = rain[max(0, t - 1), ::spatial_stride, ::spatial_stride]
            r_p2 = rain[max(0, t - 2), ::spatial_stride, ::spatial_stride]
            r_c = rain_cum[t, ::spatial_stride, ::spatial_stride]
            h0 = h_init[::spatial_stride, ::spatial_stride]

            d = dem_clean[::spatial_stride, ::spatial_stride]
            gx = sx[::spatial_stride, ::spatial_stride]
            gy = sy[::spatial_stride, ::spatial_stride]
            gm = smag[::spatial_stride, ::spatial_stride]
            ed = elev_diff[::spatial_stride, ::spatial_stride]
            tn = np.full_like(d, float(t) / max(1, t_steps - 1))

            feat_frame = np.stack(
                [d, gx, gy, gm, ed, tn, h0, r_t, r_p1, r_p2, r_c], axis=-1
            ).reshape(-1, len(FEATURE_NAMES))
            target_frame = h_arr[t, ::spatial_stride, ::spatial_stride].reshape(-1)

            xs.append(feat_frame)
            ys.append(target_frame)
            groups.append(np.full(target_frame.shape[0], ev_id, dtype=object))

    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.float32),
        np.concatenate(groups, axis=0),
    )

