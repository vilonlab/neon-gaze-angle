"""
neon_gaze.rqa — Recurrence Quantification Analysis for gaze time series.

Provides functions to build recurrence matrices and compute standard RQA
measures (RR, DET, MaxL, ENT, L, LAM, TT) in a 1-D phase space.
"""

import os
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from .segmentation import parse_trial_filename


class RQAResult(NamedTuple):
    """Container for RQA measure values."""

    RR: float
    """Recurrence rate."""
    DET: float
    """Determinism."""
    MaxL: float
    """Maximum diagonal line length."""
    ENT: float
    """Shannon entropy of diagonal line-length distribution (base 2)."""
    L: float
    """Mean diagonal line length."""
    LAM: float
    """Laminarity."""
    TT: float
    """Trapping time (mean vertical line length)."""


# ── Internal helpers ─────────────────────────────────────────────────

def _line_lengths(arr: np.ndarray, l_min: int) -> list[int]:
    """Return lengths of contiguous True runs >= *l_min* in a 1-D bool array."""
    lengths: list[int] = []
    current = 0
    for val in arr:
        if val:
            current += 1
        else:
            if current >= l_min:
                lengths.append(current)
            current = 0
    if current >= l_min:
        lengths.append(current)
    return lengths


# ── Public API ───────────────────────────────────────────────────────

def recurrence_matrix(x: np.ndarray, epsilon: float) -> np.ndarray:
    """Build a 1-D recurrence matrix.

    Parameters
    ----------
    x : array-like
        1-D time series (NaNs are dropped internally).
    epsilon : float
        Recurrence radius.

    Returns
    -------
    np.ndarray
        Boolean (N, N) matrix where ``R[i, j]`` is True when
        ``|x_i - x_j| <= epsilon``.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return np.zeros((0, 0), dtype=bool)

    dist = np.abs(x[:, None] - x[None, :])
    return dist <= epsilon


def compute_rqa(
    x: np.ndarray,
    epsilon: float,
    l_min: int = 2,
) -> RQAResult:
    """Compute RQA measures for a 1-D time series.

    Uses a 1-D phase space (no time-delay embedding).  Recurrence is
    defined as ``|x_i - x_j| <= epsilon``.  The main diagonal
    (self-recurrence) is excluded.

    Parameters
    ----------
    x : array-like
        1-D time series.
    epsilon : float
        Recurrence radius (same units as *x*).
    l_min : int
        Minimum line length for DET, MaxL, ENT, LAM, and TT.

    Returns
    -------
    RQAResult
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return RQAResult(*(np.nan,) * 7)

    R = recurrence_matrix(x, epsilon)
    N = x.size

    if R.size == 0:
        return RQAResult(*(0.0,) * 7)
    np.fill_diagonal(R, False)

    total_points = R.size
    total_rec = int(R.sum())

    if total_rec == 0:
        return RQAResult(*(0.0,) * 7)

    # Recurrence rate
    RR = total_rec / total_points

    # Diagonal line statistics
    diag_lengths: list[int] = []
    for k in range(-N + 1, N):
        diag = np.diagonal(R, offset=k)
        if diag.size == 0:
            continue
        diag_lengths.extend(_line_lengths(diag, l_min))

    if not diag_lengths:
        DET, MaxL, ENT, L = 0.0, 0.0, 0.0, 0.0
    else:
        points_in_diag = sum(diag_lengths)
        DET = points_in_diag / total_rec
        MaxL = float(max(diag_lengths))
        L = float(np.mean(diag_lengths))

        unique, counts = np.unique(diag_lengths, return_counts=True)
        p = counts / counts.sum()
        ENT = float(-np.sum(p * np.log2(p)))

    # Vertical line statistics
    vert_lengths: list[int] = []
    for j in range(N):
        col = R[:, j]
        vert_lengths.extend(_line_lengths(col, l_min))

    if not vert_lengths:
        LAM, TT = 0.0, 0.0
    else:
        points_in_vert = sum(vert_lengths)
        LAM = points_in_vert / total_rec
        TT = float(np.mean(vert_lengths))

    return RQAResult(
        RR=float(RR),
        DET=float(DET),
        MaxL=float(MaxL),
        ENT=float(ENT),
        L=float(L),
        LAM=float(LAM),
        TT=float(TT),
    )


def find_radius_for_target_rr(
    x: np.ndarray,
    target_rr: float,
) -> float:
    """Find the recurrence radius that yields a target recurrence rate.

    The radius is chosen so that the fraction of off-diagonal recurrence
    points (relative to N^2) equals *target_rr*.

    Parameters
    ----------
    x : array-like
        1-D time series (NaN-free, typically z-scored).
    target_rr : float
        Desired recurrence rate (e.g. 0.05).

    Returns
    -------
    float
        Epsilon value, or ``np.nan`` if it cannot be determined.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    N = x.size

    if N < 2 or np.isnan(target_rr):
        return np.nan

    dist = np.abs(x[:, None] - x[None, :])

    off_mask = ~np.eye(N, dtype=bool)
    off_dists = np.sort(dist[off_mask])

    if off_dists.size == 0:
        return np.nan

    total_points = N * N
    k_target = int(np.ceil(target_rr * total_points))
    k_target = max(1, min(k_target, off_dists.size))

    return float(off_dists[k_target - 1])


def build_rqa_summary(
    base_dir: str,
    gaze_col: str = "gaze angle [deg]",
    epsilon: float | None = None,
    target_rr: float | None = None,
    l_min: int = 5,
) -> pd.DataFrame:
    """Compute RQA + descriptive stats for every trial CSV in a directory.

    Provide exactly one of *epsilon* (fixed radius) or *target_rr*
    (RR-locked radius).

    Parameters
    ----------
    base_dir : str
        Folder containing segmented trial CSVs.
    gaze_col : str
        Column name for gaze angle.
    epsilon : float or None
        Fixed recurrence radius (z-score units).
    target_rr : float or None
        Target recurrence rate; radius is found per trial.
    l_min : int
        Minimum line length for RQA.

    Returns
    -------
    pd.DataFrame
        One row per trial with metadata, descriptive stats, ACF, and RQA.
    """
    if (epsilon is None) == (target_rr is None):
        raise ValueError("Provide exactly one of `epsilon` or `target_rr`.")

    rr_locked = target_rr is not None

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory '{base_dir}' not found.")

    csv_files = sorted(
        f for f in os.listdir(base_dir) if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise RuntimeError(f"No CSV files found in '{base_dir}'.")

    rows: list[dict] = []

    for fname in csv_files:
        fpath = os.path.join(base_dir, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            warnings.warn(f"Could not read '{fpath}': {e}")
            continue

        if gaze_col not in df.columns:
            warnings.warn(f"File '{fname}' missing column '{gaze_col}'. Skipping.")
            continue

        gaze = df[gaze_col].to_numpy(dtype=float)
        gaze_clean = gaze[~np.isnan(gaze)]

        if gaze_clean.size == 0:
            warnings.warn(f"No valid gaze data in '{fname}'. Skipping.")
            continue

        mean_gaze = float(np.nanmean(gaze))
        std_gaze = float(np.nanstd(gaze, ddof=1))

        # Z-score for RQA and ACF
        if std_gaze == 0 or np.isnan(std_gaze):
            warnings.warn(f"Zero or NaN std in '{fname}'. Skipping RQA/ACF.")
            acf_mean = acf_auc = np.nan
            rqa = RQAResult(*(np.nan,) * 7)
            rad = np.nan
        else:
            gaze_z = (gaze_clean - gaze_clean.mean()) / gaze_clean.std(ddof=1)

            # Autocorrelation
            s = pd.Series(gaze_z)
            all_lags = list(range(10, 551, 10))
            lags = [lag for lag in all_lags if lag < len(s)]

            if not lags:
                acf_vals = np.array([np.nan])
            else:
                acf_vals = np.array([s.autocorr(lag=lag) for lag in lags])

            acf_mean = float(np.nanmean(acf_vals))
            if not lags or np.all(np.isnan(acf_vals)):
                acf_auc = np.nan
            else:
                acf_auc = float(np.trapz(acf_vals, lags))

            # RQA
            if rr_locked:
                rad = find_radius_for_target_rr(gaze_z, target_rr)
                if np.isnan(rad):
                    rqa = RQAResult(*(np.nan,) * 7)
                else:
                    rqa = compute_rqa(gaze_z, epsilon=rad, l_min=l_min)
            else:
                rad = epsilon
                rqa = compute_rqa(gaze_z, epsilon=epsilon, l_min=l_min)

        meta = parse_trial_filename(fname)

        row: dict = {
            "filename": fname,
            "subject_number": meta["subject_number"],
            "condition": meta["condition"],
            "trial_number": meta["trial_number"],
            "walking_direction": meta["walking_direction"],
            "mean_gaze_angle": mean_gaze,
            "std_gaze_angle": std_gaze,
            "acf_mean": acf_mean,
            "acf_auc": acf_auc,
        }

        if rr_locked:
            row["RAD"] = rad

        row.update(rqa._asdict())
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        by=["subject_number", "trial_number"], na_position="last"
    ).reset_index(drop=True)

    return summary
