"""
neon_gaze.visualization — Interactive Bokeh recurrence widget.

Generates a standalone HTML file with linked-brushing between a
z-scored gaze time series and its recurrence plot.
"""

import os

import numpy as np
import pandas as pd

from bokeh.plotting import figure, output_file, save
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool, Span
from bokeh.io import show as bokeh_show
from bokeh.resources import INLINE


GAZE_COL = "gaze angle [deg]"

# JavaScript callback shared by both plot panels
_HIGHLIGHT_FN = r"""
function applyHighlight(target_idx) {
    const z_target = z[target_idx];

    const lc = left.data.color;
    const la = left.data.alpha;
    const ls = left.data.size;
    for (let i = 0; i < lc.length; i++) {
        if (i === target_idx) {
            lc[i] = anchor_color;
            la[i] = 1.0;
            ls[i] = 12;
        } else if (Math.abs(z[i] - z_target) <= eps) {
            lc[i] = hi_color;
            la[i] = 0.95;
            ls[i] = 7;
        } else {
            lc[i] = base_color;
            la[i] = 0.18;
            ls[i] = 4;
        }
    }
    left.change.emit();

    const ri = right.data.i;
    const rj = right.data.j;
    const rc = right.data.color;
    const ra = right.data.alpha;
    const rs = right.data.size;
    for (let k = 0; k < ri.length; k++) {
        if (ri[k] === target_idx || rj[k] === target_idx) {
            rc[k] = hi_color;
            ra[k] = 1.0;
            rs[k] = 5;
        } else {
            rc[k] = base_color;
            ra[k] = 0.10;
            rs[k] = 2;
        }
    }
    right.change.emit();

    vline.location = target_idx;
    hline.location = target_idx;
    vline.visible = true;
    hline.visible = true;

    tline.location = target_idx;
    tline.visible = true;

    let n_partners = 0;
    for (let i = 0; i < z.length; i++) {
        if (i !== target_idx && Math.abs(z[i] - z_target) <= eps) n_partners++;
    }
    info.text =
        "<div style='font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#222;'>" +
        "<b>anchor</b> &nbsp; t = " + target_idx +
        " &nbsp;|&nbsp; z = " + z_target.toFixed(3) +
        " &nbsp;|&nbsp; <b>" + n_partners + "</b> recurrent partners (within ε = " +
        eps.toFixed(3) + ")</div>";
}
"""


def build_linked_recurrence_widget(
    csv_path: str,
    epsilon: float = 0.07,
    output_html: str | None = "recurrence_widget.html",
    show: bool = False,
):
    """Build an interactive linked-brushing recurrence viewer.

    Parameters
    ----------
    csv_path : str
        Path to a trial CSV containing ``gaze angle [deg]``.
    epsilon : float
        Recurrence radius in z-score units.
    output_html : str or None
        If given, save a standalone HTML file.
    show : bool
        If True, also open in the default browser.

    Returns
    -------
    bokeh layout object
    """
    df = pd.read_csv(csv_path)
    if GAZE_COL not in df.columns:
        raise KeyError(f"'{GAZE_COL}' not found in {csv_path}")

    gaze = df[GAZE_COL].to_numpy(dtype=float)
    gaze_clean = gaze[~np.isnan(gaze)]
    if gaze_clean.size < 2:
        raise ValueError(f"Not enough valid data in {csv_path}")

    mu = gaze_clean.mean()
    sd = gaze_clean.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        raise ValueError(f"Zero/NaN std in {csv_path}; cannot z-score.")

    gaze_z = (gaze_clean - mu) / sd
    N = gaze_z.size
    t = np.arange(N)

    # Recurrence matrix (excluding self-recurrence)
    dist = np.abs(gaze_z[:, None] - gaze_z[None, :])
    R = dist <= epsilon
    np.fill_diagonal(R, False)
    rec_i, rec_j = np.where(R)

    n_rec = int(R.sum())
    rr = n_rec / max(R.size - N, 1)

    if len(rec_i) > 1_500_000:
        print(
            f"[warn] {len(rec_i):,} recurrent points; render may be slow. "
            "Consider downsampling or raising epsilon."
        )

    # Data sources
    base_color = "#2b3a55"
    hi_color = "#e63946"
    anchor_color = "#000000"

    left_source = ColumnDataSource(dict(
        idx=t.tolist(), t=t.tolist(), z=gaze_z.tolist(),
        color=[base_color] * N, alpha=[0.55] * N, size=[5] * N,
    ))
    right_source = ColumnDataSource(dict(
        i=rec_i.tolist(), j=rec_j.tolist(),
        color=[base_color] * len(rec_i),
        alpha=[0.35] * len(rec_i),
        size=[2] * len(rec_i),
    ))

    # Left panel: gaze z-score
    p_left = figure(
        width=620, height=540,
        title=f"z-scored gaze · ε = {epsilon}",
        x_axis_label="time index (sample)",
        y_axis_label="gaze angle (z-score)",
        tools="pan,wheel_zoom,box_zoom,reset,tap,save",
        active_scroll="wheel_zoom",
        output_backend="webgl",
        background_fill_color="#fafafa",
    )
    p_left.scatter(
        x="t", y="z", source=left_source,
        color="color", alpha="alpha", size="size", line_color=None,
    )
    p_left.line(
        t.tolist(), gaze_z.tolist(),
        color="#bbbbbb", line_width=0.8, alpha=0.6, level="underlay",
    )

    # Right panel: recurrence plot
    p_right = figure(
        width=560, height=540,
        title=f"recurrence plot · {n_rec:,} pts · RR ≈ {rr:.3f}",
        x_axis_label="i (sample)", y_axis_label="j (sample)",
        tools="pan,wheel_zoom,box_zoom,reset,tap,save",
        active_scroll="wheel_zoom",
        match_aspect=True,
        output_backend="webgl",
        background_fill_color="#fafafa",
    )
    p_right.scatter(
        x="i", y="j", source=right_source,
        color="color", alpha="alpha", size="size", line_color=None,
    )

    # Crosshairs
    vline = Span(location=0, dimension="height", line_color="#e63946",
                 line_dash="dashed", line_width=1.2, visible=False)
    hline = Span(location=0, dimension="width", line_color="#e63946",
                 line_dash="dashed", line_width=1.2, visible=False)
    p_right.add_layout(vline)
    p_right.add_layout(hline)

    tline = Span(location=0, dimension="height", line_color="#000000",
                 line_dash="dotted", line_width=1.2, visible=False)
    p_left.add_layout(tline)

    # Hover tools
    p_left.add_tools(HoverTool(tooltips=[("t", "@t"), ("z", "@z{0.000}")], mode="mouse"))
    p_right.add_tools(HoverTool(tooltips=[("i", "@i"), ("j", "@j")], mode="mouse"))

    # Info banner
    info = Div(text=(
        "<div style='font-family:ui-monospace,Menlo,monospace;"
        "font-size:13px;color:#666;'>"
        "click any point to anchor — partners highlight in both panels"
        "</div>"
    ), width=1180)

    # Linked callbacks
    js_args = dict(
        left=left_source, right=right_source,
        z=gaze_z.tolist(), eps=float(epsilon),
        base_color=base_color, hi_color=hi_color, anchor_color=anchor_color,
        vline=vline, hline=hline, tline=tline, info=info,
    )

    left_cb = CustomJS(args=js_args, code=_HIGHLIGHT_FN + r"""
        const sel = left.selected.indices;
        if (sel.length === 0) return;
        const target_idx = left.data.idx[sel[0]];
        applyHighlight(target_idx);
    """)
    right_cb = CustomJS(args=js_args, code=_HIGHLIGHT_FN + r"""
        const sel = right.selected.indices;
        if (sel.length === 0) return;
        const target_idx = right.data.i[sel[0]];
        applyHighlight(target_idx);
    """)
    left_source.selected.js_on_change("indices", left_cb)
    right_source.selected.js_on_change("indices", right_cb)

    # Header
    header = Div(text=f"""
        <div style='font-family:ui-sans-serif,system-ui,sans-serif;'>
            <h2 style='margin:0 0 4px 0;font-weight:600;letter-spacing:-0.01em;'>
                Linked recurrence viewer
            </h2>
            <div style='color:#555;font-size:13px;'>
                <code>{os.path.basename(csv_path)}</code>
                &nbsp;·&nbsp; N = {N:,}
                &nbsp;·&nbsp; ε = {epsilon}
                &nbsp;·&nbsp; recurrent points = {n_rec:,}
            </div>
        </div>
    """, width=1180)

    layout = column(header, info, row(p_left, p_right))

    if output_html:
        output_file(output_html, title="Linked Recurrence Viewer", mode="inline")
        save(layout, resources=INLINE)
    if show:
        bokeh_show(layout)

    return layout
