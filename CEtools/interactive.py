# -*- coding: utf-8 -*-
"""
Interactive anchor-selection + alignment applet (Panel + Bokeh).

Usage:
    import CEtools as cet
    pseudo_df = cet.run_anchor_alignment_applet(
        samples=unpaired_samples,   # list of column names in norm_df
        times=times,                # 1D array of time (len == rows of norm_df)
        norm_df=norm_df,            # DataFrame with columns for each sample
        y_offset=0.0,               # optional vertical offset between traces
        open_in_new_window=True,    # opens a separate Panel server tab/window
    )
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import threading
import numpy as np
import pandas as pd
import io
import bokeh.plotting
import bokeh.models
from bokeh.palettes import Category10, Category20, Turbo256
from bokeh.models import BoxAnnotation
import panel as pn

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"You are attempting to set `plot\.legend\.",
    category=UserWarning,
    module="bokeh.models.plots"
)

def _ensure_panel():
    try:
        # only call once per process
        if not getattr(pn.state, "_ce_panel_ready", False):
            pn.extension()  # no "bokeh" arg; avoids clash if HoloViews extended first
            pn.state._ce_panel_ready = True
    except Exception:
        pass


def run_anchor_alignment_applet(
    samples: Sequence[str],
    times: np.ndarray,
    norm_df: pd.DataFrame,
    *,
    title: str = "Chromatograms: anchor selection",
    default_targets: str = "0.0, 0.1, 0.4, 0.6, 1.0",
    y_offset: float = 0.0,     # vertical offset between traces for visibility
    palette: Optional[Sequence[str]] = None,
    open_in_new_window: bool = True,  # open via Panel server
) -> pd.DataFrame:
    """
    Opens a separate window/tab. After you click "Align curves", RETURNS a DataFrame
    of pseudotimes for each sample. The window stays open until "Stop Server".

    Parameters
    ----------
    samples : sequence of str
        Column names (samples) to include from `norm_df`.
    times : np.ndarray
        1D time array matching the rows of `norm_df`.
    norm_df : pd.DataFrame
        Each column is a sample; index aligns with `times`.
    title : str
        Title for the selection plot.
    default_targets : str
        Comma-separated initial targets (strictly increasing) shown in the UI.
    y_offset : float
        If > 0, vertically offset each curve to reduce overlap.
    palette : list[str] or None
        Colors to use; auto-picked if None.
    open_in_new_window : bool
        If True, launches a Panel server and opens a new tab/window.

    Returns
    -------
    pd.DataFrame
        Pseudotime values for each sample (columns), indexed by original rows.
    """
    _ensure_panel()
    # --- checks ---
    samples = [str(s) for s in samples]
    for s in samples:
        if s not in norm_df.columns:
            raise KeyError(f"Sample '{s}' not in norm_df.columns")
    if len(times) != len(norm_df.index):
        if len(times) != len(norm_df):
            raise ValueError("len(times) must equal number of rows in norm_df")

    n_samp = len(samples)
    if palette is None:
        if n_samp <= 10:
            colors = list(Category10[10])[:n_samp]
        else:
            idxs = np.linspace(0, 255, num=n_samp, dtype=int)
            colors = [Turbo256[i] for i in idxs]
    else:
        colors = list(palette)

    # --- shared state & result ---
    result_ready = threading.Event()
    result_holder = {"pseudotimes_df": None}  # type: ignore[dict-item]
    server_holder = {"server": None}          # type: ignore[dict-item]

    state = {
        "step": "config",
        "anchor": 0,                      # 0-based anchor index
        "N": None,                        # total anchors
        "anchors_per_sample": {s: [] for s in samples},  # list of (x, y) per anchor
        "selected_range": None,           # (xmin, xmax)
    }

    # --- widgets ---
    n_anchors_input = pn.widgets.IntInput(name="Number of anchors", value=5, start=2, end=20)
    start_btn = pn.widgets.Button(name="Start anchor selection", button_type="primary")
    restart_btn = pn.widgets.Button(name="Restart anchor selection", button_type="danger", disabled=True)
    stop_server_btn = pn.widgets.Button(name="Stop Server", button_type="danger", disabled=True)
    info = pn.pane.Markdown("1) Set the number of anchors, then click **Start anchor selection**.", sizing_mode="stretch_width")

    use_sel_btn = pn.widgets.Button(name="Use selected range for THIS anchor", button_type="primary", disabled=True)
    range_hint = pn.pane.Markdown("", sizing_mode="stretch_width")

    incorrect_samples = pn.widgets.MultiSelect(name="Samples needing correction", options=samples, size=min(10, n_samp))
    refine_sample = pn.widgets.Select(name="Sample to refine", options=[])
    set_sample_btn = pn.widgets.Button(name="Use selection for THIS sample", button_type="warning", disabled=True)
    done_anchor_btn = pn.widgets.Button(name="Done with this anchor → Next", button_type="success", disabled=True)

    targets_input = pn.widgets.TextInput(name=f"Target pseudotime values (comma-separated, length={n_anchors_input.value})", value=default_targets)
    finalize_btn = pn.widgets.Button(name="Align curves", button_type="primary", disabled=True)
    status = pn.pane.Markdown("", sizing_mode="stretch_width")
    
    # Add export widgets (disabled until alignment completes)
    # --- Wide CSV export (time + paired columns per sample) ---
    from packaging.version import Version
    try:
        from panel.widgets import FileDownload
    except Exception:
        FileDownload = None  # fallback later
    
    def _sanitize(name: str) -> str:
        import re
        name = re.sub(r"\s+", "_", str(name).strip())
        name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
        return name
    
    export_name = pn.widgets.TextInput(name="CSV filename", value="aligned_wide.csv")
    export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
    
    def _export_bytesio_wide():
        """
        Wide CSV:
          time, pseudotime__S1, intensity__S1, pseudotime__S2, intensity__S2, ...
        """
        pt_df = state.get("pseudotimes_df", None)
        if pt_df is None:
            return io.BytesIO(b"")
    
        out_cols = {"time": times}  # original time axis
        for s in samples:
            key = _sanitize(s)
            out_cols[f"pseudotime__{key}"] = pt_df[s].values
            out_cols[f"intensity__{key}"]  = norm_df[s].values
    
        out_df = pd.DataFrame(out_cols, index=norm_df.index)
        return io.BytesIO(out_df.to_csv(index=False).encode("utf-8"))

    if (FileDownload is not None) and (Version(pn.__version__) >= Version("0.12.6")):
        export_btn = FileDownload(
            label="Export CSV (wide)",
            filename=export_name.value,   # must be a concrete string on Panel<=0.14
            callback=_export_bytesio_wide,
            auto=False,
            embed=False,
            button_type="primary",
            disabled=True,                # enabled after alignment
        )
    
        def _sync_filename(event):
            export_btn.filename = event.new or "aligned_wide.csv"
        export_name.param.watch(_sync_filename, "value")
    
    else:
        # Fallback: server-side save (if FileDownload unavailable)
        export_btn = pn.widgets.Button(name="Export CSV (wide)", button_type="primary", disabled=True)
    
        def _fallback_save(_):
            buf = _export_bytesio_wide()
            fname = export_name.value or "aligned_wide.csv"
            with open(fname, "wb") as f:
                f.write(buf.getvalue())
            export_status.object = f":white_check_mark: Saved `{fname}` on the server."
        export_btn.on_click(_fallback_save)

    
        
        
    # --- selection figure (LARGE) ---
    fig = bokeh.plotting.figure(
        title=title,
        height=600, width=900,
        x_axis_label="Time",
        y_axis_label="Intensity",
        tools="pan,wheel_zoom,box_zoom,box_select,reset,save,hover",
        active_drag="box_select",
        active_scroll="wheel_zoom",
    )
    fig.hover.tooltips = [("x", "$x{0.000}"), ("y", "$y{0.000}")]
    if fig.legend:
        fig.legend.location = "right"
        fig.legend.click_policy = "mute"
    # Selection highlight box
    sel_box = BoxAnnotation(left=None, right=None, fill_alpha=0.15, fill_color="gray", line_color=None, level="underlay")
    fig.add_layout(sel_box)

    scatter_sources = {}
    anchor_sources = {}

    for i, s in enumerate(samples):
        y = np.asarray(norm_df[s].values, dtype=float)
        y = y + (n_samp - i - 1) * y_offset if y_offset else y
        # set muted_alpha for legend muting
        fig.line(times, y, color=colors[i % len(colors)], line_width=2, alpha=0.9,
                 legend_label=s, muted_alpha=0.1)

        src = bokeh.models.ColumnDataSource(dict(x=times, y=y, sample=[s]*len(times)))
        # invisible selectable points to capture box selection range
        fig.circle("x", "y", source=src, size=4, alpha=0.001, line_alpha=0.001,
                   fill_color=colors[i % len(colors)], line_color=colors[i % len(colors)])
        scatter_sources[s] = src

        a_src = bokeh.models.ColumnDataSource(dict(x=[], y=[]))
        fig.circle("x", "y", source=a_src, size=8, color="red")
        anchor_sources[s] = a_src

    # --- aligned plot container (LARGE) ---
    aligned_fig_pane = pn.pane.Bokeh(height=600, width=900)

    # --- helpers ---
    def _compute_selected_xrange():
        xs = []
        for src in scatter_sources.values():
            sel = src.selected.indices
            if sel:
                xs.extend(np.asarray(src.data["x"])[sel].tolist())
        if not xs:
            return None
        return float(np.min(xs)), float(np.max(xs))

    def _update_selection_box(xr):
        if xr is None:
            sel_box.left = None
            sel_box.right = None
        else:
            sel_box.left, sel_box.right = xr[0], xr[1]

    def _highlight_current_anchor():
        for s in samples:
            xs = [a[0] for a in state["anchors_per_sample"][s]]
            ys = [a[1] for a in state["anchors_per_sample"][s]]
            anchor_sources[s].data = dict(x=xs, y=ys)

    def _auto_pick_anchor_for_all(xmin, xmax):
        for s in samples:
            y = np.asarray(norm_df[s].values, dtype=float)
            if y_offset:
                idx = samples.index(s)
                y = y + (n_samp - idx - 1) * y_offset
            m = (times >= xmin) & (times <= xmax)
            if not np.any(m):
                continue
            i_local = int(np.argmax(y[m]))
            i_global = np.where(m)[0][i_local]
            x_anchor = float(times[i_global]); y_anchor = float(y[i_global])
            while len(state["anchors_per_sample"][s]) < state["anchor"]:
                state["anchors_per_sample"][s].append((np.nan, np.nan))
            if len(state["anchors_per_sample"][s]) == state["anchor"]:
                state["anchors_per_sample"][s].append((x_anchor, y_anchor))
            else:
                state["anchors_per_sample"][s][state["anchor"]] = (x_anchor, y_anchor)

    def _auto_pick_anchor_for_one(sample_name, xmin, xmax):
        y = np.asarray(norm_df[sample_name].values, dtype=float)
        if y_offset:
            idx = samples.index(sample_name)
            y = y + (n_samp - idx - 1) * y_offset
        m = (times >= xmin) & (times <= xmax)
        if not np.any(m):
            return False
        i_local = int(np.argmax(y[m]))
        i_global = np.where(m)[0][i_local]
        x_anchor = float(times[i_global]); y_anchor = float(y[i_global])
        while len(state["anchors_per_sample"][sample_name]) < state["anchor"]:
            state["anchors_per_sample"][sample_name].append((np.nan, np.nan))
        if len(state["anchors_per_sample"][sample_name]) == state["anchor"]:
            state["anchors_per_sample"][sample_name].append((x_anchor, y_anchor))
        else:
            state["anchors_per_sample"][sample_name][state["anchor"]] = (x_anchor, y_anchor)
        return True

    def _all_current_anchors_valid():
        k = state["anchor"]
        for s in samples:
            if len(state["anchors_per_sample"][s]) <= k:
                return False
            x, y = state["anchors_per_sample"][s][k]
            if not (np.isfinite(x) and np.isfinite(y)):
                return False
        return True

    def _clear_all_anchors():
        for s in samples:
            state["anchors_per_sample"][s].clear()
            anchor_sources[s].data = dict(x=[], y=[])
        _update_selection_box(None)

    # --- callbacks ---
    def on_start(_):
        state["N"] = int(n_anchors_input.value)
        state["anchor"] = 0
        state["step"] = "selecting"
        info.object = f"**Anchor 1 / {state['N']}** — Drag a rectangle (Box Select), then click **Use selected range for THIS anchor**."
        use_sel_btn.disabled = False
        done_anchor_btn.disabled = True
        finalize_btn.disabled = True
        restart_btn.disabled = False
        stop_server_btn.disabled = False
        range_hint.object = ""
        _update_selection_box(None)
        targets_input.name = f"Target pseudotime values (comma-separated, length={state['N']})"
        _clear_all_anchors()
    start_btn.on_click(on_start)

    def on_restart(_):
        state["anchor"] = 0
        state["step"] = "config"
        info.object = "Restarted. Set the number of anchors, then click **Start anchor selection**."
        use_sel_btn.disabled = True
        done_anchor_btn.disabled = True
        finalize_btn.disabled = True
        incorrect_samples.value = []
        refine_sample.options = []
        set_sample_btn.disabled = True
        _clear_all_anchors()
    restart_btn.on_click(on_restart)

    def on_stop_server(_):
        srv = server_holder.get("server")
        if srv is not None:
            srv.stop()
            status.object = "Server stopped."
            server_holder["server"] = None
    stop_server_btn.on_click(on_stop_server)

    def on_selection_change(attr, old, new):
        if state["step"] not in ("selecting", "per-sample-refine"):
            return
        xr = _compute_selected_xrange()
        _update_selection_box(xr)
        if xr is None:
            range_hint.object = "_No selection. Drag a rectangle over the curves._"
        else:
            range_hint.object = f"Selected x-range: **[{xr[0]:.4f}, {xr[1]:.4f}]**"
    for src in scatter_sources.values():
        src.selected.on_change("indices", on_selection_change)

    def on_use_selection(_):
        xr = _compute_selected_xrange()
        if xr is None:
            range_hint.object = ":warning: **No selection found**. Drag a rectangle first."
            return
        xmin, xmax = xr
        state["selected_range"] = xr
        _auto_pick_anchor_for_all(xmin, xmax)
        _highlight_current_anchor()
        info.object = f"Anchors proposed for **anchor {state['anchor']+1}**. If any are wrong, pick samples below and refine."
        done_anchor_btn.disabled = not _all_current_anchors_valid()
    use_sel_btn.on_click(on_use_selection)

    def on_incorrect_change(event):
        vals = list(event.new) if event.new else []
        refine_sample.options = vals
        set_sample_btn.disabled = len(vals) == 0
    incorrect_samples.param.watch(on_incorrect_change, "value")

    def on_set_sample(_):
        choice = refine_sample.value
        if not choice:
            status.object = ":warning: Choose a sample to refine."
            return
        xr = _compute_selected_xrange()
        if xr is None:
            status.object = ":warning: Drag a rectangle for the chosen sample."
            return
        ok = _auto_pick_anchor_for_one(choice, xr[0], xr[1])
        if not ok:
            status.object = f":warning: No points in range for **{choice}**."
            return
        _highlight_current_anchor()
        status.object = f":white_check_mark: Updated anchor for **{choice}**."
        done_anchor_btn.disabled = not _all_current_anchors_valid()
    set_sample_btn.on_click(on_set_sample)

    def _parse_targets(txt, N):
        try:
            vals = [float(x.strip()) for x in txt.split(",") if x.strip() != ""]
            if len(vals) != N:
                return None, f"Expected {N} targets, got {len(vals)}."
            if not all(vals[i] < vals[i+1] for i in range(N-1)):
                return None, "Targets must be strictly increasing."
            return vals, ""
        except Exception as e:
            return None, f"Parse error: {e}"

    def on_done_anchor(_):
        if not _all_current_anchors_valid():
            status.object = ":warning: Some anchors are missing; refine or reselect."
            return
        k_next = state["anchor"] + 1
        if k_next >= state["N"]:
            state["step"] = "targets"
            info.object = f"All **{state['N']}** anchors selected. Enter target pseudotime values (comma-separated; length = {state['N']})."
            use_sel_btn.disabled = True
            done_anchor_btn.disabled = True
            finalize_btn.disabled = False
            return
        state["anchor"] = k_next
        incorrect_samples.value = []
        refine_sample.options = []
        set_sample_btn.disabled = True
        info.object = f"**Anchor {k_next+1} / {state['N']}** — Drag a rectangle (Box Select), then click **Use selected range for THIS anchor**."
        range_hint.object = ""
        _update_selection_box(None)
    done_anchor_btn.on_click(on_done_anchor)

    def on_finalize(_):
        targets, msg = _parse_targets(targets_input.value, state["N"])
        if targets is None:
            status.object = f":warning: {msg}"
            return

        anchors_by_sample = {}
        for s in samples:
            pts = state["anchors_per_sample"][s]
            if len(pts) != state["N"]:
                status.object = f":warning: Sample {s} missing anchors."
                return
            anchors_by_sample[s] = [p[0] for p in pts]

        import CEtools as cet

        aligned_fig = bokeh.plotting.figure(
            title="Aligned chromatograms",
            height=600, width=800,
            x_axis_label="pseudotime",
            y_axis_label="Intensity",
            x_range=(0.0, 1.0),  # fixed requested range
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            active_scroll="wheel_zoom",
        )
        
        if aligned_fig.legend:
            aligned_fig.legend.location = "right"
            aligned_fig.legend.click_policy = "mute"
        aligned_fig.hover.tooltips = [("pt", "$x{0.000}"), ("y", "$y{0.000}")]

        pseudotimes_df = pd.DataFrame(index=norm_df.index)

        for i, s in enumerate(samples):
            anchors = anchors_by_sample[s]
            pseudo = cet.pseudotime_transform(times, anchors, targets=targets, clamp=False)
            pseudotimes_df[s] = pseudo

            y = np.asarray(norm_df[s].values, dtype=float)
            if y_offset:
                y = y + (n_samp - i - 1) * y_offset
            aligned_fig.line(x=pseudo, y=y, color=colors[i % len(colors)],
                             line_width=2, alpha=0.9, legend_label=s, muted_alpha=0.1)

        aligned_fig_pane.object = aligned_fig
        
        
        state["pseudotimes_df"] = pseudotimes_df         # <-- ADD: store for export
        export_btn.disabled = False                      # <-- ADD: enable export
        status.object = ":white_check_mark: Alignment complete. You can keep exploring, **Export CSV**, or click **Stop Server** when done."
        state["step"] = "done"

        # return result to caller (keep server alive)
        result_holder["pseudotimes_df"] = pseudotimes_df
        result_ready.set()
    finalize_btn.on_click(on_finalize)

    # --- layout ---
    header = pn.Column(
        pn.pane.Markdown("## Semi-manual anchor selection & alignment", sizing_mode="stretch_width"),
        info,
        pn.Row(n_anchors_input, start_btn, restart_btn, stop_server_btn),
        range_hint,
        sizing_mode="stretch_width",
    )
    selection_controls = pn.WidgetBox(
        pn.Row(use_sel_btn),
        pn.Spacer(height=5),
        pn.pane.Markdown("**If any sample picked the wrong peak, select it below and refine:**"),
        pn.Row(incorrect_samples, refine_sample),
        pn.Row(set_sample_btn, done_anchor_btn),
        sizing_mode="stretch_width",
    )
    targets_controls = pn.WidgetBox(
        pn.pane.Markdown("### Targets"),
        targets_input,
        finalize_btn,
        status,
        sizing_mode="stretch_width",
    )
    app = pn.Column(
        header,
        pn.Row(pn.pane.Bokeh(fig, height=600, width=800), selection_controls),
        pn.layout.Divider(),
        targets_controls,
        pn.layout.Divider(),
        pn.pane.Markdown("### Aligned curves", sizing_mode="stretch_width"),
        aligned_fig_pane,
        pn.Row(export_name, export_btn),    # <-- ADD: filename + export button
        sizing_mode="stretch_width",
    )

    # --- serve externally; keep server up; return df when ready ---
    if open_in_new_window:
        server = pn.serve(app, show=True, start=True, threaded=True, port=0)
        server_holder["server"] = server
        try:
            result_ready.wait()   # returns after "Align curves"
        finally:
            pass  # do not stop server; user clicks "Stop Server"
        return result_holder["pseudotimes_df"]  # type: ignore[return-value]
    else:
        # inline fallback
        pn.panel(app).servable()
        result_ready.wait()
        return result_holder["pseudotimes_df"]  # type: ignore[return-value]

