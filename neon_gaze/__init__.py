"""
neon_gaze — Shared utilities for Pupil Neon gaze-angle processing and analysis.

This package provides reusable functions for:
- Loading and synchronizing Pupil Neon eye-tracker exports (io.py)
- Computing gaze angle from IMU pitch + gaze elevation (processing.py)
- Segmenting continuous recordings into per-trial CSVs (segmentation.py)
- Recurrence Quantification Analysis on gaze time series (rqa.py)
- Visualization helpers for gaze data and recurrence plots (plotting.py)
- Interactive linked-brushing recurrence viewer (visualization.py)
"""

__version__ = "0.1.0"
