"""
neon_gaze.segmentation — Trial segmentation from Pupil Neon annotations.

Parses "Trial Begin" / "Trial End" annotations and splits a continuous
gaze-angle DataFrame into per-trial CSV files.
"""

import os
import re
import warnings

import pandas as pd
import numpy as np

from .io import TIMESTAMP_COL


def add_trial_events(
    df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    expected_trials: int = 20,
) -> pd.DataFrame:
    """Add a ``trial event`` column to *df* using annotation timestamps.

    Each row closest to a "Trial Begin" or "Trial End" annotation gets
    labelled (e.g. ``"Trial 1 Start"``, ``"Trial 3 End"``).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``timestamp [ns]``.
    annotations_df : pd.DataFrame
        Must contain ``timestamp [ns]`` and ``label`` columns, where labels
        include ``"Trial Begin"`` and ``"Trial End"``.
    expected_trials : int
        Expected number of trials; a warning is raised if the count differs.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new ``trial event`` column.
    """
    df = df.copy()

    trial_begins = (
        annotations_df[annotations_df["label"] == "Trial Begin"]
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )
    trial_ends = (
        annotations_df[annotations_df["label"] == "Trial End"]
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )

    n_begins = len(trial_begins)
    n_ends = len(trial_ends)
    n_trials = min(n_begins, n_ends)

    if n_begins != n_ends:
        warnings.warn(
            f"Number of Trial Begin ({n_begins}) and Trial End ({n_ends}) "
            "annotations do not match!"
        )
    if n_trials != expected_trials:
        warnings.warn(
            f"Expected {expected_trials} trials, but found {n_trials} "
            f"(begins={n_begins}, ends={n_ends})."
        )

    df["trial event"] = pd.Series(index=df.index, dtype=object)
    timestamps = df[TIMESTAMP_COL]

    def mark_event(ts_event: int, label: str) -> None:
        idx = (timestamps - ts_event).abs().idxmin()
        existing = df.at[idx, "trial event"]
        if pd.isna(existing) or existing is None:
            df.at[idx, "trial event"] = label
        else:
            df.at[idx, "trial event"] = f"{existing}; {label}"

    for trial_idx in range(n_trials):
        begin_ts = trial_begins.loc[trial_idx, TIMESTAMP_COL]
        end_ts = trial_ends.loc[trial_idx, TIMESTAMP_COL]

        if begin_ts >= end_ts:
            warnings.warn(
                f"Trial {trial_idx + 1} begin timestamp is not before end timestamp!"
            )

        mark_event(begin_ts, f"Trial {trial_idx + 1} Start")
        mark_event(end_ts, f"Trial {trial_idx + 1} End")

    return df


def _get_trial_num(label: str) -> int | None:
    """Extract integer trial number from a label like 'Trial 3 Start'."""
    m = re.search(r"Trial\s+(\d+)", label)
    return int(m.group(1)) if m else None


def segment_and_save_trials(
    df: pd.DataFrame,
    output_dir: str,
    subject_id: str,
    expected_trials: int = 20,
) -> list[str]:
    """Split a trial-annotated DataFrame into per-trial CSV files.

    Odd-numbered trials are labelled Walk-Dir-Up; even-numbered are
    Walk-Dir-Down (matching the experimental protocol).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``trial event`` and ``timestamp [ns]`` columns.
    output_dir : str
        Directory to write trial CSVs into (created if needed).
    subject_id : str
        Prefix for filenames, e.g. ``"Subject-03_Hill-Condition"``.
    expected_trials : int
        Expected trial count; a warning is printed if it differs.

    Returns
    -------
    list of str
        Paths of saved CSV files.
    """
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)

    # Collect start/end indices per trial
    start_indices: dict[int, int] = {}
    end_indices: dict[int, int] = {}

    trial_events = df["trial event"].dropna()
    for idx, cell in trial_events.items():
        for raw_label in str(cell).split(";"):
            label = raw_label.strip()
            trial_num = _get_trial_num(label)
            if trial_num is None:
                continue
            if "Start" in label and trial_num not in start_indices:
                start_indices[trial_num] = idx
            if "End" in label and trial_num not in end_indices:
                end_indices[trial_num] = idx

    all_trials = sorted(set(start_indices.keys()) & set(end_indices.keys()))

    if len(all_trials) != expected_trials:
        print(
            f"WARNING: Expected {expected_trials} trials, "
            f"but found {len(all_trials)} with both Start and End."
        )

    saved_paths: list[str] = []

    for trial_num in all_trials:
        start_idx = start_indices[trial_num]
        end_idx = end_indices[trial_num]

        if end_idx < start_idx:
            print(
                f"WARNING: Trial {trial_num} has End before Start "
                f"(start_idx={start_idx}, end_idx={end_idx}). Skipping."
            )
            continue

        walk_direction = "Down" if trial_num % 2 == 0 else "Up"

        trial_df = df.loc[start_idx:end_idx].copy()
        trial_df["trial"] = trial_num

        trial_str = f"{trial_num:02d}"
        filename = f"{subject_id}_Trial-{trial_str}_Walk-Dir-{walk_direction}.csv"
        filepath = os.path.join(output_dir, filename)

        trial_df.to_csv(filepath, index=False)
        saved_paths.append(filepath)
        print(f"Saved trial {trial_num} to {filepath} ({len(trial_df)} rows).")

    return saved_paths


def parse_trial_filename(fname: str) -> dict:
    """Parse subject, condition, trial, and direction from a trial CSV filename.

    Expected pattern::

        Subject-03_Hill-Condition_Trial-01_Walk-Dir-Up.csv

    Parameters
    ----------
    fname : str
        Filename (basename or full path).

    Returns
    -------
    dict
        Keys: ``subject_number`` (int | None), ``condition`` (str | None),
        ``trial_number`` (int | None), ``walking_direction`` (str | None).
    """
    name = os.path.splitext(os.path.basename(fname))[0]

    m_sub = re.search(r"Subject-(\d+)", name, flags=re.IGNORECASE)
    subject_number = int(m_sub.group(1)) if m_sub else None

    m_cond = re.search(r"([A-Za-z]+)-Condition", name, flags=re.IGNORECASE)
    condition = m_cond.group(1).capitalize() if m_cond else None

    m_trial = re.search(r"Trial-(\d+)", name, flags=re.IGNORECASE)
    trial_number = int(m_trial.group(1)) if m_trial else None

    m_dir = re.search(r"Walk-Dir-([A-Za-z]+)", name, flags=re.IGNORECASE)
    walking_direction = m_dir.group(1).capitalize() if m_dir else None

    if any(
        v is None
        for v in [subject_number, condition, trial_number, walking_direction]
    ):
        warnings.warn(
            f"Could not fully parse filename '{fname}'. "
            f"Parsed: subject={subject_number}, condition={condition}, "
            f"trial={trial_number}, direction={walking_direction}"
        )

    return {
        "subject_number": subject_number,
        "condition": condition,
        "trial_number": trial_number,
        "walking_direction": walking_direction,
    }
