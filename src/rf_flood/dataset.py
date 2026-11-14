"""Dataset loading, common UKEA feature engineering, and strict event-level splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, uniform_filter


@dataclass(frozen=True)
class FloodEvent:
    """Single UKEA flood event containing static DEM, rainfall, and water depth."""

    event_id: str
    dem: np.ndarray       # (H, W) in meters
    rainfall: np.ndarray  # (T, H, W) in mm per 5-min step
    depth: np.ndarray     # (T, H, W) in meters


def _crop_to_ukea_32x32(arr: np.ndarray) -> np.ndarray:
    """Crop standard LarNO UKEA (..., 50, 120) arrays to the agreed (..., 32, 32) patch."""
    if arr.shape[-2:] == (32, 32):
        return arr
    if arr.shape[-2] >= 41 and arr.shape[-1] >= 76:
        return arr[..., 9:41, 44:76]
    h_start = max(0, (arr.shape[-2] - 32) // 2)
    w_start = max(0, (arr.shape[-1] - 32) // 2)
    return arr[..., h_start : h_start + 32, w_start : w_start + 32]


def _align_rainfall(rainfall: np.ndarray, target_steps: int) -> np.ndarray:
    """Align high-frequency rainfall sub-steps (e.g., 360) to 5-min depth frames (e.g., 36)."""
    if rainfall.ndim == 1:
        rainfall = rainfall[:, None, None]
    t_r = rainfall.shape[0]
    if t_r == target_steps:
        return rainfall
    if t_r > target_steps and t_r % target_steps == 0:
        factor = t_r // target_steps
        return rainfall.reshape(target_steps, factor, *rainfall.shape[1:]).sum(axis=1)
    indices = np.linspace(0, t_r - 1, target_steps).astype(int)
    return rainfall[indices]


def load_ukea_events(data_dir: Path) -> List[FloodEvent]:
    """Load UKEA flood event folders (dem.npy, h.npy, rainfall.npy) from disk."""
    if not data_dir.exists():
        return []

    events: List[FloodEvent] = []
    for event_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        dem_path = event_dir / "dem.npy"
        h_path = event_dir / "h.npy"
        rain_path = event_dir / "rainfall.npy"
        if not (dem_path.exists() and h_path.exists() and rain_path.exists()):
            continue

        dem = _crop_to_ukea_32x32(np.load(dem_path).astype(np.float32))
        depth = _crop_to_ukea_32x32(np.load(h_path).astype(np.float32))
        rain = _crop_to_ukea_32x32(np.load(rain_path).astype(np.float32))
        rain = _align_rainfall(rain, depth.shape[0])
        if rain.shape[-2:] != dem.shape:
            rain = np.broadcast_to(rain, depth.shape).copy()

        events.append(
            FloodEvent(
                event_id=event_dir.name,
                dem=dem,
                rainfall=rain,
                depth=np.clip(depth, 0.0, None),
            )
        )
    return events


def generate_synthetic_ukea_events(
    num_events: int = 10,
    grid_rows: int = 32,
    grid_cols: int = 32,
    timesteps: int = 24,
    seed: int = 42,
) -> List[FloodEvent]:
    """
    Generate deterministic, physically structured UKEA 32x32 flood events
    for reproducible baseline training and testing when raw UKEA .npy files
    are not mounted locally.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, grid_rows, dtype=np.float32),
        np.linspace(0.0, 1.0, grid_cols, dtype=np.float32),
        indexing="ij",
    )

    # Shared urban catchment DEM with drainage basin & building footprints
    base_dem = (
        14.0
        + 3.2 * (yy - 0.55) ** 2
        + 1.6 * (xx - 0.48) ** 2
        - 1.4 * np.exp(-((xx - 0.45) ** 2 + (yy - 0.55) ** 2) / 0.05)
        + 0.35 * np.sin(xx * np.pi * 4.0) * np.cos(yy * np.pi * 3.0)
    ).astype(np.float32)

    # Mark a few urban building footprints (high elevation)
    building_mask = np.zeros((grid_rows, grid_cols), dtype=bool)
    building_mask[4:8, 5:10] = True
    building_mask[5:9, 18:24] = True
    building_mask[22:26, 8:14] = True
    base_dem[building_mask] += 6.0

    events: List[FloodEvent] = []
    for idx in range(num_events):
        event_id = f"ukea_event_{idx + 1:02d}"
        storm_peak = rng.uniform(4.0, 14.0)  # mm per 5-min step
        peak_step = rng.uniform(0.25, 0.55) * timesteps
        spread = rng.uniform(2.5, 4.5)
        t_arr = np.arange(timesteps, dtype=np.float32)
        hyetograph = storm_peak * np.exp(-0.5 * ((t_arr - peak_step) / spread) ** 2)

        # Mild spatial rainfall variation
        rain = np.zeros((timesteps, grid_rows, grid_cols), dtype=np.float32)
        depth = np.zeros((timesteps, grid_rows, grid_cols), dtype=np.float32)
        dem = base_dem + rng.normal(0.0, 0.03, size=base_dem.shape).astype(np.float32)

        inv_slope_weight = np.clip(16.5 - dem, 0.0, 4.0) / 4.0
        inv_slope_weight[building_mask] = 0.0

        h_curr = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        for t in range(timesteps):
            spatial_mod = 1.0 + 0.15 * np.sin(xx * np.pi * 2.0 + 0.2 * t)
            rain[t] = np.clip(hyetograph[t] * spatial_mod, 0.0, None)
            runoff_m = (rain[t] * 0.001) * (0.45 + 0.85 * inv_slope_weight)
            # Shallow-water diffusive ponding into lower cells + drainage recession
            nbr_mean = uniform_filter(h_curr, size=3, mode="nearest")
            h_next = 0.72 * h_curr + 0.22 * nbr_mean + runoff_m - 0.012 * h_curr
            h_next[building_mask] = 0.0
            h_curr = np.clip(h_next, 0.0, None).astype(np.float32)
            depth[t] = h_curr

        events.append(FloodEvent(event_id=event_id, dem=dem, rainfall=rain, depth=depth))
    return events


def _extract_dem_features(dem: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    """Compute static topographic descriptors from the 2D DEM."""
    dem_clean = dem.astype(np.float32)
    is_building = (~np.isfinite(dem_clean)) | (dem_clean > (np.nanmedian(dem_clean) + 4.0))
    if np.isnan(dem_clean).any():
        dem_clean = np.where(np.isfinite(dem_clean), dem_clean, np.nanmedian(dem_clean))

    grad_y, grad_x = np.gradient(dem_clean, dx, dx)
    slope = np.hypot(grad_x, grad_y)
    aspect = np.arctan2(-grad_y, grad_x)
    laplacian = (
        np.gradient(grad_x, dx, axis=1) + np.gradient(grad_y, dx, axis=0)
    ).astype(np.float32)
    mean_3x3 = uniform_filter(dem_clean, size=3, mode="nearest")
    sq_mean_3x3 = uniform_filter(dem_clean ** 2, size=3, mode="nearest")
    roughness_3x3 = np.sqrt(np.clip(sq_mean_3x3 - mean_3x3 ** 2, 0.0, None))
    tpi = dem_clean - mean_3x3

    return {
        "elevation_m": dem_clean,
        "slope": slope.astype(np.float32),
        "grad_x": grad_x.astype(np.float32),
        "grad_y": grad_y.astype(np.float32),
        "aspect_sin": np.sin(aspect).astype(np.float32),
        "aspect_cos": np.cos(aspect).astype(np.float32),
        "laplacian": laplacian,
        "roughness_3x3": roughness_3x3.astype(np.float32),
        "tpi": tpi.astype(np.float32),
        "is_building_mask": is_building.astype(np.float32),
    }


def build_event_dataframe(
    event: FloodEvent,
    lead_steps: int = 6,
    dx: float = 8.0,
    dt_min: float = 5.0,
) -> pd.DataFrame:
    """Convert a single FloodEvent into a leakage-free tabular DataFrame."""
    timesteps, rows, cols = event.depth.shape
    if timesteps <= lead_steps:
        raise ValueError(
            f"Event {event.event_id} has {timesteps} steps, must exceed lead_steps={lead_steps}"
        )

    topo = _extract_dem_features(event.dem, dx=dx)
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    norm_y = (yy / max(1, rows - 1)).astype(np.float32)
    norm_x = (xx / max(1, cols - 1)).astype(np.float32)
    dist_center = np.hypot(norm_x - 0.5, norm_y - 0.5).astype(np.float32)

    rain_cum = np.cumsum(event.rainfall, axis=0)
    frames: List[pd.DataFrame] = []

    for t in range(timesteps - lead_steps):
        h_t = event.depth[t]
        h_lag1 = event.depth[max(0, t - 1)]
        h_lag2 = event.depth[max(0, t - 2)]
        h_lag3 = event.depth[max(0, t - 3)]
        h_roc = (h_t - h_lag1) / dt_min

        r_t = event.rainfall[t]
        r_prev = event.rainfall[max(0, t - 1)]
        r_roc = (r_t - r_prev) / dt_min
        r_15m = event.rainfall[max(0, t - 2) : t + 1].sum(axis=0)
        r_30m = event.rainfall[max(0, t - 5) : t + 1].sum(axis=0)
        r_60m = event.rainfall[max(0, t - 11) : t + 1].sum(axis=0)
        r_peak = event.rainfall[: t + 1].max(axis=0)

        h_nbr_mean = uniform_filter(h_t, size=3, mode="nearest")
        h_nbr_max = maximum_filter(h_t, size=3, mode="nearest")
        head = topo["elevation_m"] + h_t
        head_gy, head_gx = np.gradient(head, dx, dx)
        head_grad = np.hypot(head_gx, head_gy).astype(np.float32)

        target_h = event.depth[t + lead_steps]

        df_t = pd.DataFrame(
            {
                "event_id": event.event_id,
                "timestep": np.int32(t),
                "time_min": np.float32(t * dt_min),
                "grid_y": yy.ravel().astype(np.int32),
                "grid_x": xx.ravel().astype(np.int32),
                # Topographic
                "elevation_m": topo["elevation_m"].ravel(),
                "slope": topo["slope"].ravel(),
                "grad_x": topo["grad_x"].ravel(),
                "grad_y": topo["grad_y"].ravel(),
                "aspect_sin": topo["aspect_sin"].ravel(),
                "aspect_cos": topo["aspect_cos"].ravel(),
                "laplacian": topo["laplacian"].ravel(),
                "roughness_3x3": topo["roughness_3x3"].ravel(),
                "tpi": topo["tpi"].ravel(),
                "is_building_mask": topo["is_building_mask"].ravel(),
                # Rainfall
                "rain_intensity_mm_5min": r_t.ravel(),
                "rain_cum_mm": rain_cum[t].ravel(),
                "rain_roll_sum_15m": r_15m.ravel(),
                "rain_roll_sum_30m": r_30m.ravel(),
                "rain_roll_sum_60m": r_60m.ravel(),
                "rain_peak_so_far": r_peak.ravel(),
                "rain_roc": r_roc.ravel(),
                # Hydraulic
                "depth_t_m": h_t.ravel(),
                "depth_lag_5m": h_lag1.ravel(),
                "depth_lag_10m": h_lag2.ravel(),
                "depth_lag_15m": h_lag3.ravel(),
                "depth_roc": h_roc.ravel(),
                "depth_nbr_mean_3x3": h_nbr_mean.ravel(),
                "depth_nbr_max_3x3": h_nbr_max.ravel(),
                "hydraulic_head_m": head.ravel(),
                "hydraulic_grad": head_grad.ravel(),
                # Spatio-temporal
                "norm_y": norm_y.ravel(),
                "norm_x": norm_x.ravel(),
                "dist_to_center": dist_center.ravel(),
                "time_elapsed_min": np.float32(t * dt_min),
                "time_fraction": np.float32(t / max(1, timesteps - 1)),
                # Target
                "target_depth_m": target_h.ravel(),
            }
        )
        frames.append(df_t)

    return pd.concat(frames, axis=0, ignore_index=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return strictly ordered feature columns excluding IDs, coordinates, and targets."""
    excluded = {"event_id", "timestep", "time_min", "grid_y", "grid_x", "target_depth_m"}
    return [c for c in df.columns if c not in excluded]


def build_tabular_dataset(
    events: List[FloodEvent],
    lead_steps: int = 6,
    dx: float = 8.0,
    dt_min: float = 5.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build combined tabular DataFrame across all events and return (df, feature_cols)."""
    dfs = [build_event_dataframe(ev, lead_steps=lead_steps, dx=dx, dt_min=dt_min) for ev in events]
    full_df = pd.concat(dfs, axis=0, ignore_index=True)
    return full_df, get_feature_columns(full_df)


def split_events_strict(
    events: List[FloodEvent],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[FloodEvent], List[FloodEvent], List[FloodEvent]]:
    """
    Strictly partition FloodEvents at the event level into disjoint
    (train_events, val_events, test_events).
    """
    if len(events) < 3:
        raise ValueError(f"Need at least 3 events for train/val/test split, got {len(events)}")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(events))
    rng.shuffle(indices)

    n_total = len(events)
    n_test = max(1, int(round(n_total * test_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = max(1, (n_total - 1) // 2)
        n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_ev = [events[i] for i in train_idx]
    val_ev = [events[i] for i in val_idx]
    test_ev = [events[i] for i in test_idx]

    train_ids = {e.event_id for e in train_ev}
    val_ids = {e.event_id for e in val_ev}
    test_ids = {e.event_id for e in test_ev}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    return train_ev, val_ev, test_ev

