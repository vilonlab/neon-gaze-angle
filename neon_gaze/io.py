"""
neon_gaze.io — Data loading and saving utilities for Pupil Neon exports.

Each public function handles one CSV type exported by Pupil Neon's
Neon Player application, returning a pandas DataFrame sorted by timestamp.
"""

import os
import pandas as pd


# ── Column name constants ────────────────────────────────────────────
TIMESTAMP_COL = "timestamp [ns]"


def load_imu(path: str) -> pd.DataFrame:
    """Load an IMU CSV exported by Pupil Neon.

    Expected columns include timestamp [ns], gyro x/y/z, acceleration x/y/z,
    roll, pitch, yaw, and quaternion components.

    Parameters
    ----------
    path : str
        Path to ``imu.csv``.

    Returns
    -------
    pd.DataFrame
        Sorted by timestamp.
    """
    df = pd.read_csv(path)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df


def load_gaze_positions(path: str) -> pd.DataFrame:
    """Load a gaze-positions CSV exported by Pupil Neon.

    Expected columns: timestamp [ns], gaze x [px], gaze y [px],
    azimuth [deg], elevation [deg].

    Parameters
    ----------
    path : str
        Path to ``gaze_positions.csv``.

    Returns
    -------
    pd.DataFrame
        Sorted by timestamp.
    """
    df = pd.read_csv(path)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df


def load_blinks(path: str) -> pd.DataFrame:
    """Load a blinks CSV exported by Pupil Neon.

    Expected columns: start timestamp [ns], end timestamp [ns],
    duration [ms], and others.

    Parameters
    ----------
    path : str
        Path to ``blinks.csv``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def load_annotations(path: str) -> pd.DataFrame:
    """Load an annotations CSV exported by Pupil Neon.

    Expected columns: timestamp [ns], label, and optionally duration [ms].

    Parameters
    ----------
    path : str
        Path to ``annotations.csv``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def load_fixations(path: str) -> pd.DataFrame:
    """Load a fixations CSV exported by Pupil Neon.

    Parameters
    ----------
    path : str
        Path to ``fixations.csv``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def save_gaze_angle(df: pd.DataFrame, path: str) -> None:
    """Save a gaze-angle DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``timestamp [ns]`` and ``gaze angle [deg]``.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
