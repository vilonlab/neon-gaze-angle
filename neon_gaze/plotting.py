"""
neon_gaze.plotting — Visualization helpers for gaze data and recurrence plots.

Provides Plotly-based and Matplotlib-based plotting functions used
across the analysis notebooks.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# ── Plotly-based helpers ─────────────────────────────────────────────

def plot_gaze_angle(
    gaze_angle_df: pd.DataFrame,
    title: str = "Gaze Angle Over Time",
    y_range: tuple[float, float] = (-90, 45),
) -> go.Figure:
    """Scatter-line plot of gaze angle vs time.

    Parameters
    ----------
    gaze_angle_df : pd.DataFrame
        Must contain ``time_sec`` and ``gaze angle [deg]``.
    title : str
        Plot title.
    y_range : tuple
        (min, max) for the y-axis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = px.scatter(
        gaze_angle_df,
        x="time_sec",
        y="gaze angle [deg]",
        title=title,
        labels={
            "gaze angle [deg]": "Gaze Angle (deg)",
            "time_sec": "Time (sec)",
        },
        render_mode="webgl",
    )
    fig.update_traces(mode="lines+markers")
    fig.update_yaxes(range=list(y_range))
    return fig


def plot_gaze_angle_with_gait(
    gaze_angle_df: pd.DataFrame,
    hs_times: np.ndarray,
    to_times: np.ndarray,
    title: str = "Gaze Angle with Gait Events",
    y_range: tuple[float, float] = (-90, 45),
) -> go.Figure:
    """Gaze-angle plot overlaid with heel-strike (red) and toe-off (blue) lines.

    Parameters
    ----------
    gaze_angle_df : pd.DataFrame
        Must contain ``time_sec`` and ``gaze angle [deg]``.
    hs_times, to_times : array-like
        Event times in seconds.
    title : str
        Plot title.
    y_range : tuple
        (min, max) for the y-axis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = plot_gaze_angle(gaze_angle_df, title=title, y_range=y_range)

    y_min = gaze_angle_df["gaze angle [deg]"].min()
    y_max = gaze_angle_df["gaze angle [deg]"].max()

    shapes = []
    for t in hs_times:
        shapes.append(
            dict(
                type="line", xref="x", yref="y",
                x0=t, x1=t, y0=y_min, y1=y_max,
                line=dict(width=2, color="red"), opacity=0.7,
            )
        )
    for t in to_times:
        shapes.append(
            dict(
                type="line", xref="x", yref="y",
                x0=t, x1=t, y0=y_min, y1=y_max,
                line=dict(width=2, color="blue"), opacity=0.7,
            )
        )

    fig.update_layout(shapes=shapes)
    return fig


def plot_gaze_histogram(
    gaze_angle_df: pd.DataFrame,
    title: str = "Histogram of Gaze Angle",
    nbins: int = 60,
    x_range: tuple[float, float] = (-80, 20),
) -> go.Figure:
    """Histogram of gaze angle with mean and +/- 1 SD lines.

    Parameters
    ----------
    gaze_angle_df : pd.DataFrame
        Must contain ``gaze angle [deg]``.
    title : str
        Plot title.
    nbins : int
        Number of bins.
    x_range : tuple
        (min, max) for the x-axis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ga = gaze_angle_df["gaze angle [deg]"]
    mean_ga = ga.mean()
    std_ga = ga.std()

    fig = px.histogram(
        gaze_angle_df,
        x="gaze angle [deg]",
        nbins=nbins,
        title=title,
        labels={"gaze angle [deg]": "Gaze Angle (deg)"},
        marginal="box",
        template="plotly_white",
    )
    fig.update_layout(bargap=0.02)
    fig.update_xaxes(range=list(x_range))

    fig.add_vline(
        x=mean_ga, line=dict(color="red", width=2),
        annotation_text=f"mean={mean_ga:.2f}", annotation_position="top right",
    )
    fig.add_vline(
        x=mean_ga - std_ga, line=dict(color="orange", width=1, dash="dash"),
        annotation_text="-1σ", annotation_position="top left",
    )
    fig.add_vline(
        x=mean_ga + std_ga, line=dict(color="orange", width=1, dash="dash"),
        annotation_text="+1σ", annotation_position="top left",
    )

    return fig


def plot_imu_yaw(
    imu_df: pd.DataFrame,
    title: str = "IMU Yaw Over Time",
) -> go.Figure:
    """Scatter-line plot of IMU yaw vs time.

    Parameters
    ----------
    imu_df : pd.DataFrame
        Must contain ``timestamp [ns]`` and ``yaw [deg]``.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df = imu_df.copy()
    start_ns = df["timestamp [ns]"].iloc[0]
    df["time_sec"] = (df["timestamp [ns]"] - start_ns) / 1e9

    fig = px.scatter(
        df,
        x="time_sec",
        y="yaw [deg]",
        title=title,
        labels={"time_sec": "Time (sec)", "yaw [deg]": "Yaw (deg)"},
        render_mode="webgl",
    )
    fig.update_traces(mode="lines+markers")
    return fig


def plot_filtered_acceleration(
    imu_df: pd.DataFrame,
    hs_times: np.ndarray,
    to_times: np.ndarray,
    title: str = "Filtered IMU Vertical Acceleration",
) -> go.Figure:
    """Plot filtered acceleration with gait-event lines.

    Parameters
    ----------
    imu_df : pd.DataFrame
        Must contain ``time_sec`` and ``acc_filt`` (from
        :func:`neon_gaze.gait.detect_gait_events`).
    hs_times, to_times : array-like
        Event times in seconds.
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = px.scatter(
        imu_df,
        x="time_sec",
        y="acc_filt",
        title=title,
        labels={
            "acc_filt": "Vertical Acceleration (G, filtered)",
            "time_sec": "Time (sec)",
        },
        render_mode="webgl",
    )
    fig.update_traces(mode="lines+markers")

    y_min = imu_df["acc_filt"].min()
    y_max = imu_df["acc_filt"].max()

    shapes = []
    for t in hs_times:
        shapes.append(
            dict(
                type="line", xref="x", yref="y",
                x0=t, x1=t, y0=y_min, y1=y_max,
                line=dict(width=2, color="red"), opacity=0.7,
            )
        )
    for t in to_times:
        shapes.append(
            dict(
                type="line", xref="x", yref="y",
                x0=t, x1=t, y0=y_min, y1=y_max,
                line=dict(width=2, color="blue"), opacity=0.7,
            )
        )

    fig.update_layout(shapes=shapes)
    return fig


# ── Matplotlib-based helpers ─────────────────────────────────────────

def plot_fixation_histograms(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    labels: tuple[str, str] = ("Left", "Right"),
    bins: int = 30,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title_string: str = "Fixation Duration Comparison",
) -> tuple[tuple, tuple[dict, dict]]:
    """Side-by-side histograms of fixation duration for two conditions.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Must contain ``duration [ms]``.
    labels : tuple of str
        Labels for left and right panels.
    bins : int
        Number of histogram bins.
    xlim, ylim : tuple or None
        Axis limits.
    title_string : str
        Shared title suffix.

    Returns
    -------
    tuple
        ``(hist_objects, stats_dicts)``
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    hist_left = axes[0].hist(df_left["duration [ms]"], bins=bins, edgecolor="black")
    axes[0].set_title(f"{labels[0]} - {title_string}")
    axes[0].set_xlabel("Fixation Duration (ms)")
    axes[0].set_ylabel("Frequency")

    hist_right = axes[1].hist(df_right["duration [ms]"], bins=bins, edgecolor="black")
    axes[1].set_title(f"{labels[1]} - {title_string}")
    axes[1].set_xlabel("Fixation Duration (ms)")

    if xlim is not None:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)
    if ylim is not None:
        axes[0].set_ylim(ylim)
        axes[1].set_ylim(ylim)

    plt.tight_layout()
    plt.show()

    def summary(df: pd.DataFrame) -> dict:
        dur = df["duration [ms]"]
        return {
            "mean": dur.mean(),
            "sd": dur.std(),
            "n": len(df),
            "sum": dur.sum(),
        }

    stats_left = summary(df_left)
    stats_right = summary(df_right)

    print(
        f"{labels[0]} - Mean: {stats_left['mean']:.2f}, "
        f"SD: {stats_left['sd']:.2f}, N: {stats_left['n']}, "
        f"Sum: {stats_left['sum']:.0f}"
    )
    print(
        f"{labels[1]} - Mean: {stats_right['mean']:.2f}, "
        f"SD: {stats_right['sd']:.2f}, N: {stats_right['n']}, "
        f"Sum: {stats_right['sum']:.0f}"
    )

    return (hist_left, hist_right), (stats_left, stats_right)


def plot_recurrence_matrix(
    gaze_series: np.ndarray,
    epsilon: float,
    title: str = "Recurrence Plot",
    figsize: tuple[int, int] = (6, 6),
) -> None:
    """Plot a square recurrence matrix for a z-scored gaze series.

    Parameters
    ----------
    gaze_series : array-like
        1-D time series (NaNs are dropped; then z-scored internally).
    epsilon : float
        Recurrence radius in z-score units.
    title : str
        Plot title.
    figsize : tuple
        Matplotlib figure size.
    """
    gaze = np.asarray(gaze_series, dtype=float)
    x = gaze[~np.isnan(gaze)]

    if x.size < 2:
        print("Not enough valid samples for a recurrence plot.")
        return

    mean_x = x.mean()
    std_x = x.std(ddof=1)
    if std_x == 0 or np.isnan(std_x):
        print("Zero or NaN std; cannot z-score.")
        return

    x_z = (x - mean_x) / std_x

    dist = np.abs(x_z[:, None] - x_z[None, :])
    R = dist <= epsilon
    N = x_z.size
    RR = R.mean()

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.imshow(
        R,
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
        extent=[0, N - 1, 0, N - 1],
    )
    ax.set_xlabel("Time index")
    ax.set_ylabel("Time index")
    ax.set_title(f"{title}\nε = {epsilon:.2f} (z), RR = {RR:.3f}")
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, N - 1)
    ax.set_ylim(0, N - 1)
    plt.tight_layout()
    plt.show()
