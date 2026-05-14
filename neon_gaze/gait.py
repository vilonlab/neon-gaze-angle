"""
neon_gaze.gait — Gait event detection from IMU accelerometry.

Detects heel-strike and toe-off events using band-pass filtered
vertical acceleration and peak detection.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from .io import TIMESTAMP_COL


@dataclass
class GaitEvents:
    """Container for gait-event detection results."""

    imu_df: pd.DataFrame
    """IMU DataFrame augmented with ``time_sec`` and ``acc_filt`` columns."""

    fs: float
    """Estimated sampling frequency (Hz)."""

    hs_idx: np.ndarray
    """Row indices of detected heel-strike events."""

    to_idx: np.ndarray
    """Row indices of detected toe-off events."""

    hs_times: np.ndarray
    """Heel-strike times in seconds (relative to first frame)."""

    to_times: np.ndarray
    """Toe-off times in seconds (relative to first frame)."""


def detect_gait_events(
    imu_df: pd.DataFrame,
    acc_col: str = "acceleration z [G]",
    low: float = 0.5,
    high: float = 8.0,
    min_step_time: float = 0.3,
    prominence: float = 0.02,
) -> GaitEvents:
    """Detect heel-strike and toe-off events from IMU vertical acceleration.

    Applies a 4th-order Butterworth band-pass filter, then uses
    :func:`scipy.signal.find_peaks` on the filtered (and inverted)
    signal to locate gait events.

    Parameters
    ----------
    imu_df : pd.DataFrame
        IMU data with ``timestamp [ns]`` and *acc_col*.
    acc_col : str
        Column name for vertical acceleration.
    low, high : float
        Band-pass cutoff frequencies in Hz.
    min_step_time : float
        Minimum time between successive events (seconds).
    prominence : float
        Minimum peak prominence for detection.

    Returns
    -------
    GaitEvents
    """
    df = imu_df.copy()
    start_ns = df[TIMESTAMP_COL].iloc[0]
    df["time_sec"] = (df[TIMESTAMP_COL] - start_ns) / 1e9

    timestamps = df[TIMESTAMP_COL].values.astype(np.float64)
    dt = np.diff(timestamps) / 1e9
    fs = 1.0 / np.median(dt)

    acc = df[acc_col].values
    b, a = butter(4, [low / (fs / 2.0), high / (fs / 2.0)], btype="band")
    acc_filt = filtfilt(b, a, acc)
    df["acc_filt"] = acc_filt

    min_distance_samples = int(min_step_time * fs)

    hs_idx, _ = find_peaks(
        acc_filt, prominence=prominence, distance=min_distance_samples
    )
    to_idx, _ = find_peaks(
        -acc_filt, prominence=prominence, distance=min_distance_samples
    )

    hs_times = df.loc[hs_idx, "time_sec"].to_numpy()
    to_times = df.loc[to_idx, "time_sec"].to_numpy()

    return GaitEvents(
        imu_df=df,
        fs=fs,
        hs_idx=hs_idx,
        to_idx=to_idx,
        hs_times=hs_times,
        to_times=to_times,
    )
