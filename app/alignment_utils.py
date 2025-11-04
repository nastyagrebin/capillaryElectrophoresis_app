# ===============================
# FILE: app/alignment_utils.py
# ===============================
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable
import io
import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting
import bokeh.models
from bokeh.palettes import Category10, Turbo256
from bokeh.models import BoxAnnotation

OK = "OK:"; WARN = "Warning:"
def ok(m:str)->str: return f"{OK} {m}"
def warn(m:str)->str: return f"{WARN} {m}"

pn.extension()

class AlignmentController:
    """
    Semi-manual anchor selection and alignment (Panel + Bokeh).

    External API:
      - set_input(by_sample: {name: DataFrame(time, intensity)})
      - on_aligned: Optional[Callable[[pd.DataFrame, pd.DataFrame, bool], None]]
        -> called as on_aligned(P, Y, rows_are_traces=False) after "Align curves"
      - add_aligned_listener(fn): legacy hook, receives aligned_by_sample dict for previews/exports
    """
    def __init__(self):
        # ---------- Data in/out ----------
        self.input_by_sample: Dict[str, pd.DataFrame] = {}
        self.pseudotimes_df: Optional[pd.DataFrame] = None  # P (rows=timepoints, cols=samples)

        # ---------- Internal state ----------
        self._colors: List[str] = []
        self._samples: List[str] = []
        self._times: Optional[np.ndarray] = None
        self._anchor_idx: int = 0
        self._N: int = 0
        self._anchors_per_sample: Dict[str, List[Tuple[float,float]]] = {}

        # ---------- UI widgets ----------
        self.n_anchors_input = pn.widgets.IntInput(name="Number of anchors", value=5, start=2, end=20)
        self.start_btn       = pn.widgets.Button(name="Start anchor selection", button_type="primary", disabled=True)
        self.use_sel_btn     = pn.widgets.Button(
            name="Use selected range for THIS anchor (all samples)",
            button_type="primary", disabled=True
        )
        self.incorrect_samples = pn.widgets.MultiSelect(name="Samples needing correction", options=[], size=8, disabled=True)
        self.refine_sample     = pn.widgets.Select(name="Sample to refine", options=[], disabled=True)
        self.set_sample_btn    = pn.widgets.Button(name="Use selection for THIS sample", button_type="warning", disabled=True)
        self.done_anchor_btn   = pn.widgets.Button(name="Done with this anchor → Next", button_type="success", disabled=True)
        self.targets_input     = pn.widgets.TextInput(
            name="Target pseudotime values (comma-separated)",
            value="0.0, 0.1, 0.4, 0.6, 1.0",
            placeholder="e.g. 0.0, 0.25, 0.5, 0.75, 1.0",
            sizing_mode="stretch_width",
        )
        self.finalize_btn      = pn.widgets.Button(name="Align curves", button_type="primary", disabled=True)

        # Messages
        self.info   = pn.pane.Markdown("Load/finish Preprocess → then set anchors.", sizing_mode="stretch_width")
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Figure & selection
        self.fig: Optional[bokeh.plotting.Figure] = None
        self.sel_box: Optional[BoxAnnotation] = None
        self.scatter_sources: Dict[str, bokeh.models.ColumnDataSource] = {}
        self.anchor_sources: Dict[str, bokeh.models.ColumnDataSource]  = {}

        # Aligned preview
        self.aligned_fig: Optional[bokeh.plotting.Figure] = None
        self.aligned_pane = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")

        # CSV export (enabled after alignment)
        self.csv_name = pn.widgets.TextInput(name="Pseudotimes CSV filename", value="pseudotimes_wide.csv", width=260)
        self.csv_download = pn.widgets.FileDownload(
            label="Download Pseudotimes CSV",
            filename=self.csv_name.value, button_type="primary",
            embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True,
        )
        self.export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.csv_name.param.watch(lambda e: setattr(self.csv_download, "filename", e.new or "pseudotimes_wide.csv"), "value")
        self.csv_download.callback = self._csv_bytes

        # ---------- App integration ----------
        self.on_aligned: Optional[Callable[[pd.DataFrame, pd.DataFrame, bool], None]] = None  # P, Y, rows_are_traces
        self._aligned_listeners: List[Callable[[Dict[str, pd.DataFrame]], None]] = []  # legacy preview listeners
        self.aligned_by_sample: Dict[str, pd.DataFrame] = {}

        # ---------- Wire callbacks ----------
        self.start_btn.on_click(self._on_start)
        self.use_sel_btn.on_click(self._on_use_selection)
        self.done_anchor_btn.on_click(self._on_done_anchor)
        self.finalize_btn.on_click(self._on_finalize)
        self.incorrect_samples.param.watch(self._on_incorrect_change, "value")
        self.set_sample_btn.on_click(self._on_set_sample)

        # ---------- Layout ----------
        self._figure_row_placeholder = pn.Row(sizing_mode="stretch_width")
        self._exports_row = pn.Row(pn.Column(self.csv_name, self.csv_download), sizing_mode="stretch_width")
        self.section = pn.Column(
            pn.pane.Markdown("## 3) Alignment"),
            self.info,
            pn.Row(self.n_anchors_input, pn.Spacer(width=10), self.start_btn),
            pn.Row(self.use_sel_btn, pn.Spacer(width=10), self.done_anchor_btn),
            pn.pane.Markdown(
                "_Use **Box Select** to draw a rectangle over a region. Then either apply to **all samples** "
                "or pick **specific samples** to refine below._",
                styles={"color":"#555"}, sizing_mode="stretch_width"
            ),
            pn.Row(self.incorrect_samples, pn.Spacer(width=10), self.refine_sample, pn.Spacer(width=10), self.set_sample_btn),
            pn.layout.Divider(),
            pn.pane.Markdown("### Curves & anchor selection"),
            self._figure_row_placeholder,
            pn.layout.Divider(),
            pn.pane.Markdown("### Targets"),
            self.targets_input,
            pn.Row(self.finalize_btn),
            pn.layout.Divider(),
            pn.pane.Markdown("### Aligned preview"),
            self.aligned_pane,
            self._exports_row,
            self.export_status,
            self.status,
            sizing_mode="stretch_width",
            visible=True,
        )

    # ---------- External API ----------
    def set_input(self, by_sample: Dict[str, pd.DataFrame]) -> None:
        self.input_by_sample = {k: v.copy() for k, v in by_sample.items()}
        self._samples = list(self.input_by_sample.keys())
        self._times = self.input_by_sample[self._samples[0]]["time"].to_numpy()

        n = len(self._samples)
        if n <= 10:
            self._colors = list(Category10[10])[:n]
        else:
            idxs = np.linspace(0, 255, num=n, dtype=int)
            self._colors = [Turbo256[i] for i in idxs]

        self._anchor_idx = 0
        self._N = 0
        self._anchors_per_sample = {s: [] for s in self._samples}
        self.pseudotimes_df = None
        self.aligned_fig = None
        self.aligned_pane.object = None

        self._build_selection_figure()

        self.start_btn.disabled = False
        self.use_sel_btn.disabled = True
        self.done_anchor_btn.disabled = True
        self.finalize_btn.disabled = False

        self.incorrect_samples.options = self._samples
        self.incorrect_samples.disabled = False
        self.refine_sample.options = []
        self.refine_sample.disabled = True
        self.set_sample_btn.disabled = True

        self.targets_input.value = "0.0, 0.1, 0.4, 0.6, 1.0"
        self.targets_input.name  = "Target pseudotime values (comma-separated)"

        self.csv_download.disabled = True
        self.export_status.object = ""

        self.section.visible = True
        self.info.object = (
            "Set the number of anchors, click **Start anchor selection**. For each anchor: "
            "box-select a region and click **Use selected range for THIS anchor (all samples)**. "
            "To correct outliers, pick **Samples needing correction**, then re-select and click "
            "**Use selection for THIS sample**. When all anchors are placed, edit the **Targets** below if needed, then **Align curves**."
        )
        self.status.object = ""

    # ---------- Figure ----------
    def _build_selection_figure(self):
        t = self._times
        p = bokeh.plotting.figure(
            height=420, width=900,
            title="Select anchors by Box Select",
            x_axis_label="time", y_axis_label="intensity",
            tools="pan,box_select,box_zoom,reset,save,hover",
            active_drag="box_select",
        )
        p.legend.location = "right"
        p.legend.click_policy = "hide"
        p.hover.tooltips = [("x", "$x{0.000}"), ("y", "$y{0.000}"), ("sample", "@sample")]

        sel_box = BoxAnnotation(left=None, right=None, fill_alpha=0.15, fill_color="gray", line_color=None, level="underlay")
        p.add_layout(sel_box)
        self.sel_box = sel_box

        self.scatter_sources.clear()
        self.anchor_sources.clear()
        for i, s in enumerate(self._samples):
            df = self.input_by_sample[s]
            y = df["intensity"].to_numpy()
            color = self._colors[i % len(self._colors)]

            p.line(t, y, color=color, line_width=2, alpha=0.9, legend_label=s)

            src = bokeh.models.ColumnDataSource(data=dict(x=t, y=y, sample=[s]*len(t)))
            p.scatter(x="x", y="y", source=src, size=4, alpha=0.001, line_alpha=0.001, fill_color=color, line_color=color)
            self.scatter_sources[s] = src

            a_src = bokeh.models.ColumnDataSource(data=dict(x=[], y=[]))
            p.scatter(x="x", y="y", source=a_src, size=8, color="red")
            self.anchor_sources[s] = a_src

            src.selected.on_change("indices", self._on_selection_change)

        self._figure_row_placeholder.objects = [p]
        self.fig = p

    # ---------- Selection helpers ----------
    def _compute_selected_xrange(self) -> Optional[Tuple[float,float]]:
        xs: List[float] = []
        for src in self.scatter_sources.values():
            sel = src.selected.indices
            if sel:
                xs.extend(np.asarray(src.data["x"])[sel].tolist())
        if not xs:
            return None
        return float(np.min(xs)), float(np.max(xs))

    def _update_sel_box(self, xr: Optional[Tuple[float,float]]):
        if self.sel_box is None:
            return
        if xr is None:
            self.sel_box.left = None; self.sel_box.right = None
        else:
            self.sel_box.left, self.sel_box.right = xr[0], xr[1]

    def _on_selection_change(self, attr, old, new):
        if self._N <= 0:
            return
        xr = self._compute_selected_xrange()
        self._update_sel_box(xr)

    # ---------- Anchor helpers ----------
    def _append_or_replace_anchor(self, sample: str, xy: Tuple[float,float]):
        while len(self._anchors_per_sample[sample]) < self._anchor_idx:
            self._anchors_per_sample[sample].append((np.nan, np.nan))
        if len(self._anchors_per_sample[sample]) == self._anchor_idx:
            self._anchors_per_sample[sample].append(xy)
        else:
            self._anchors_per_sample[sample][self._anchor_idx] = xy

    def _auto_pick_for_all(self, xmin: float, xmax: float):
        t = self._times; assert t is not None
        for s in self._samples:
            df = self.input_by_sample[s]
            y = df["intensity"].to_numpy()
            m = (t >= xmin) & (t <= xmax)
            if not np.any(m):
                self._append_or_replace_anchor(s, (np.nan, np.nan))
                continue
            i_local = int(np.argmax(y[m]))
            i_global = np.where(m)[0][i_local]
            self._append_or_replace_anchor(s, (float(t[i_global]), float(y[i_global])))

    def _auto_pick_for_one(self, sample: str, xmin: float, xmax: float) -> bool:
        t = self._times; assert t is not None
        df = self.input_by_sample[sample]
        y = df["intensity"].to_numpy()
        m = (t >= xmin) & (t <= xmax)
        if not np.any(m):
            return False
        i_local = int(np.argmax(y[m]))
        i_global = np.where(m)[0][i_local]
        self._append_or_replace_anchor(sample, (float(t[i_global]), float(y[i_global])))
        return True

    def _refresh_anchor_dots(self):
        for s in self._samples:
            xs = [a[0] for a in self._anchors_per_sample[s]]
            ys = [a[1] for a in self._anchors_per_sample[s]]
            self.anchor_sources[s].data = dict(x=xs, y=ys)

    def _all_current_valid(self) -> bool:
        k = self._anchor_idx
        for s in self._samples:
            if len(self._anchors_per_sample[s]) <= k: return False
            x,y = self._anchors_per_sample[s][k]
            if not (np.isfinite(x) and np.isfinite(y)): return False
        return True

    # ---------- Buttons logic ----------
    def _on_start(self, _=None):
        self._N = int(self.n_anchors_input.value)
        if self._N < 2:
            self.status.object = warn("Choose at least 2 anchors.")
            return
        for s in self._samples:
            self._anchors_per_sample[s] = []
            self.anchor_sources[s].data = dict(x=[], y=[])
        self._anchor_idx = 0
        self.use_sel_btn.disabled = False
        self.done_anchor_btn.disabled = True
        self.finalize_btn.disabled = True

        targets = ", ".join(f"{v:.3f}" for v in np.linspace(0.0, 1.0, self._N))
        self.targets_input.value = targets
        self.targets_input.name  = f"Target pseudotime values (comma-separated; length={self._N})"

        self.incorrect_samples.disabled = False
        self.refine_sample.disabled = True
        self.set_sample_btn.disabled = True
        self.info.object = f"**Anchor 1 / {self._N}** — Box-select a region, apply to all, then refine specific samples if needed."

    def _on_use_selection(self, _=None):
        xr = self._compute_selected_xrange()
        if xr is None:
            self.status.object = warn("No selection found. Draw a rectangle with Box Select.")
            return
        self._auto_pick_for_all(*xr)
        self._refresh_anchor_dots()
        self.done_anchor_btn.disabled = not self._all_current_valid()
        self.status.object = ok(f"Proposed anchors for anchor {self._anchor_idx+1}. Refine specific samples if needed, then click Next.")
        self.refine_sample.disabled = len(self.incorrect_samples.value or []) == 0
        self.set_sample_btn.disabled = self.refine_sample.disabled

    def _on_incorrect_change(self, event):
        vals = list(event.new) if event.new else []
        self.refine_sample.options = vals
        self.refine_sample.disabled = (len(vals) == 0)
        self.set_sample_btn.disabled = self.refine_sample.disabled

    def _on_set_sample(self, _=None):
        sample = self.refine_sample.value
        if not sample:
            self.status.object = warn("Pick a sample to refine.")
            return
        xr = self._compute_selected_xrange()
        if xr is None:
            self.status.object = warn("Draw a rectangle for the chosen sample.")
            return
        ok_one = self._auto_pick_for_one(sample, *xr)
        if not ok_one:
            self.status.object = warn(f"No points in range for {sample}.")
            return
        self._refresh_anchor_dots()
        self.done_anchor_btn.disabled = not self._all_current_valid()
        self.status.object = ok(f"Updated anchor for {sample}.")

    def _on_done_anchor(self, _=None):
        if not self._all_current_valid():
            self.status.object = warn("Some samples have no anchor in this step.")
            return
        nxt = self._anchor_idx + 1
        if nxt >= self._N:
            self.use_sel_btn.disabled = True
            self.done_anchor_btn.disabled = True
            self.finalize_btn.disabled = False
            self.info.object = "All anchors selected. Edit **Targets** if needed, then click **Align curves**."
            self._update_sel_box(None)
            return
        self._anchor_idx = nxt
        self.info.object = f"**Anchor {self._anchor_idx+1} / {self._N}** — Box-select a region, apply to all, refine as needed, then Next."
        self.done_anchor_btn.disabled = True
        self._update_sel_box(None)

    def _parse_targets(self, txt: str, N: int) -> Optional[List[float]]:
        try:
            vals = [float(s.strip()) for s in txt.split(",") if s.strip()!=""]
            if len(vals) != N: return None
            if not all(vals[i] < vals[i+1] for i in range(N-1)): return None
            return vals
        except Exception:
            return None

    def _on_finalize(self, _=None):
        if self._N < 2:
            self.status.object = warn("Start and finish anchor selection first.")
            return

        targets = self._parse_targets(self.targets_input.value, self._N)
        if targets is None:
            self.status.object = warn(f"Targets must be {self._N} comma-separated, strictly increasing values.")
            return

        try:
            import CEtools as cet
        except Exception as e:
            self.status.object = warn(f"Cannot import CEtools: {e}")
            return

        # A) Build pseudotimes per original index (no resampling)
        t = self._times; assert t is not None
        P = pd.DataFrame(index=self.input_by_sample[self._samples[0]].index)  # pseudotimes_df
        aligned = bokeh.plotting.figure(
            height=420, width=900, title="Aligned chromatograms",
            x_axis_label="pseudotime", y_axis_label="intensity",
            x_range=(0.0, 1.0),
            tools="pan,box_zoom,reset,save,hover",
        )
        aligned.legend.location = "right"
        aligned.legend.click_policy = "hide"
        aligned.hover.tooltips = [("pt","$x{0.000}"), ("y","$y{0.000}")]

        Y_cols = {}  # will become norm_df columns

        for i, s in enumerate(self._samples):
            anchors = [xy[0] for xy in self._anchors_per_sample[s]]
            if len(anchors) != self._N or not np.all(np.isfinite(anchors)):
                self.status.object = warn(f"Sample {s} is missing anchors for one or more steps.")
                return
            pseudo = cet.pseudotime_transform(t, anchors, targets=targets, clamp=False)
            y = self.input_by_sample[s]["intensity"].to_numpy() + 1.0  # keep consistent with plotting
            color = self._colors[i % len(self._colors)]
            aligned.line(x=pseudo, y=y, color=color, line_width=2, alpha=0.9, legend_label=s)
            P[s] = pseudo
            Y_cols[s] = y

        self.pseudotimes_df = P.copy()
        self.aligned_fig = aligned
        self.aligned_pane.object = aligned

        # B) Build aligned dataset on a common pseudotime grid (for preview/export)
        Npts = len(t)
        grid = np.linspace(0.0, 1.0, Npts)
        aligned_by_sample = {}
        for s in self._samples:
            pseudo = P[s].to_numpy()
            y      = self.input_by_sample[s]["intensity"].to_numpy() + 1.0

            order = np.argsort(pseudo)
            pseudo_sorted = pseudo[order]
            y_sorted      = y[order]

            uniq_x, uniq_idx = np.unique(pseudo_sorted, return_index=True)
            uniq_y = y_sorted[uniq_idx]

            y_grid = np.interp(grid, uniq_x, uniq_y)
            aligned_by_sample[s] = pd.DataFrame({"pseudotime": grid, "intensity": y_grid})

        self.aligned_by_sample = aligned_by_sample

        # C) Compose Y (norm_df) with same index/shape as P; columns are samples
        Y = pd.DataFrame({s: Y_cols[s] for s in self._samples}, index=P.index)

        # Notify legacy listeners (previews/exports)
        for fn in self._aligned_listeners:
            try:
                fn(self.aligned_by_sample)
            except Exception:
                pass

        # Notify app with strict contract for NMF
        if callable(self.on_aligned):
            try:
                # rows_are_traces=False: columns are samples
                self.on_aligned(self.pseudotimes_df.copy(), Y.copy(), rows_are_traces=False)
            except Exception:
                # keep UI alive even if bridge fails
                pass

        # Enable CSV export
        self.csv_download.disabled = False
        self.export_status.object = ok("CSV export is ready. Use the **Save** tool in the toolbar for images.")
        self.status.object = ok("Alignment complete. You can export pseudotimes or proceed.")

    # ---------- Public hooks ----------
    def add_aligned_listener(self, fn: Callable[[Dict[str, pd.DataFrame]], None]):
        """Register a callback that receives aligned_by_sample dict when Align finishes."""
        self._aligned_listeners.append(fn)

    # ---------- CSV export ----------
    def _csv_bytes(self):
        """
        Wide CSV: 'time', then for each sample two columns:
          '{sample}_pt' (pseudotime) and '{sample}' (intensity).
        """
        if self.pseudotimes_df is None:
            self.export_status.object = warn("No pseudotimes available. Align first.")
            return io.BytesIO(b"")

        samples = self._samples
        if not samples:
            return io.BytesIO(b"")

        base_time = self.input_by_sample[samples[0]]["time"].to_numpy()
        out = pd.DataFrame({"time": base_time})
        for s in samples:
            out[f"{s}_pt"] = self.pseudotimes_df[s].to_numpy()
            out[s]         = self.input_by_sample[s]["intensity"].to_numpy() + 1.0

        bio = io.BytesIO()
        out.to_csv(bio, index=False)
        bio.seek(0)
        self.export_status.object = ok("Pseudotimes CSV prepared.")
        return bio

# Public factory
def build_alignment_section():
    ctrl = AlignmentController()
    return ctrl.section, ctrl


