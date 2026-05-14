"""
neon_gaze.processing — Gaze-angle computation and blink masking.

Core pipeline step: synchronize IMU pitch with gaze elevation to produce
a combined gaze-angle time series.
"""

import numpy as np
import pandas as pd

from .io import TIMESTAMP_COL


def synchronize_gaze_to_imu(
    imu_df: pd.DataFrame,
    gaze_df: pd.DataFrame,
) -> pd.DataFrame:
    """Downsample gaze data to IMU frame rate via nearest-timestamp matching.

    For each IMU frame, the nearest gaze ``elevation [deg]`` value is found
    using :func:`pandas.merge_asof`.

    Parameters
    ----------
    imu_df : pd.DataFrame
        IMU data with at least ``timestamp [ns]`` and ``pitch [deg]``.
    gaze_df : pd.DataFrame
        Gaze-position data with at least ``timestamp [ns]`` and
        ``elevation [deg]``.

    Returns
    -------
    pd.DataFrame
        The IMU DataFrame augmented with the nearest ``elevation [deg]``.
    """
    imu_sorted = imu_df.sort_values(TIMESTAMP_COL)
    gaze_sorted = gaze_df.sort_values(TIMESTAMP_COL)

    merged = pd.merge_asof(
        imu_sorted,
        gaze_sorted[[TIMESTAMP_COL, "elevation [deg]"]],
        on=TIMESTAMP_COL,
        direction="nearest",
    )
    return merged


def compute_gaze_angle(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Compute gaze angle as pitch + elevation and add a time-in-seconds column.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Output of :func:`synchronize_gaze_to_imu`, containing
        ``pitch [deg]`` and ``elevation [deg]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp [ns]``, ``gaze angle [deg]``,
        ``pitch [deg]``, ``elevation [deg]``, ``time_sec``.
    """
    df = merged_df.copy()
    df["gaze angle [deg]"] = df["pitch [deg]"] + df["elevation [deg]"]

    start_ns = df[TIMESTAMP_COL].iloc[0]
    df["time_sec"] = (df[TIMESTAMP_COL] - start_ns) / 1e9

    result = df[
        [TIMESTAMP_COL, "gaze angle [deg]", "pitch [deg]", "elevation [deg]", "time_sec"]
    ].copy()
    return result


def mask_blinks(
    gaze_angle_df: pd.DataFrame,
    blinks_df: pd.DataFrame,
    cols_to_nan: list[str] | None = None,
) -> pd.DataFrame:
    """Set gaze values to NaN during blink intervals.

    Parameters
    ----------
    gaze_angle_df : pd.DataFrame
        Must contain ``timestamp [ns]``.
    blinks_df : pd.DataFrame
        Must contain ``start timestamp [ns]`` and ``end timestamp [ns]``.
    cols_to_nan : list of str, optional
        Columns to blank during blinks. Defaults to
        ``["gaze angle [deg]", "pitch [deg]", "elevation [deg]"]``.

    Returns
    -------
    pd.DataFrame
        Copy of *gaze_angle_df* with NaNs inserted during blinks.
    """
    if cols_to_nan is None:
        cols_to_nan = ["gaze angle [deg]", "pitch [deg]", "elevation [deg]"]

    df = gaze_angle_df.copy()

    blink_intervals = blinks_df[
        ["start timestamp [ns]", "end timestamp [ns]"]
    ].to_numpy()
    ts = df[TIMESTAMP_COL].to_numpy()

    blink_mask = np.zeros(len(df), dtype=bool)
    for start_ns, end_ns in blink_intervals:
        blink_mask |= (ts >= start_ns) & (ts <= end_ns)

    df.loc[blink_mask, cols_to_nan] = np.nan

    n_blinked = int(blink_mask.sum())
    print(f"Frames inside blinks: {n_blinked} / {len(blink_mask)}")

    return df
