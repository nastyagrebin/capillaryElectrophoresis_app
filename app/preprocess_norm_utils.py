# app/preprocess_norm_utils.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting
from bokeh.models import ColumnDataSource, BoxAnnotation, CustomJS
from bokeh.events import SelectionGeometry
from bokeh.palettes import Category10, Turbo256

from common_plot import plot_multi

pn.extension("bokeh")  # safe if already called; Panel guards it

OK = "OK:"
WARN = "Warning:"

def ok(msg: str) -> str:
    return f"{OK} {msg}"

def warn(msg: str) -> str:
    return f"{WARN} {msg}"

def _palette(n: int):
    if n <= 10:
        return list(Category10[10])[:n]
    idxs = np.linspace(0, 255, num=n, dtype=int)
    return [Turbo256[i] for i in idxs]

@dataclass
class NormalizationController:
    """
    Normalization preview via Box Select:
      - For each sample, user drags a box; we find the tallest peak inside [xmin,xmax],
        grow left/right while non-increasing, clamp to selection, and FILL the AUC in red.
      - Visualization is pure JS for responsiveness.
      - On 'Apply normalization', we recompute AUC in Python with the same rule and divide trace.
    """
    # Inputs come from previous stage (e.g., smoothed or raw depending on user path)
    current_by_sample: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Outputs
    normalized_by_sample: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # UI elements
    before_pane: pn.pane.Bokeh = field(default_factory=lambda: pn.pane.Bokeh(height=400, sizing_mode="stretch_width"))
    after_pane: pn.pane.Bokeh = field(default_factory=lambda: pn.pane.Bokeh(height=400, sizing_mode="stretch_width"))
    status: pn.pane.Markdown = field(default_factory=lambda: pn.pane.Markdown("", sizing_mode="stretch_width"))

    apply_btn: pn.widgets.Button = field(default_factory=lambda: pn.widgets.Button(name="Apply normalization", button_type="success", disabled=True))
    skip_btn: pn.widgets.Button = field(default_factory=lambda: pn.widgets.Button(name="Skip normalization (use input data)", button_type="danger", disabled=True))

    export_name: pn.widgets.TextInput = field(default_factory=lambda: pn.widgets.TextInput(name="Normalized CSV filename", value="normalized.csv"))
    export_btn: pn.widgets.FileDownload = field(default_factory=lambda: pn.widgets.FileDownload(label="Export normalized CSV", filename="normalized.csv", button_type="primary", embed=False, auto=False, callback=lambda: None, disabled=True))
    export_status: pn.pane.Markdown = field(default_factory=lambda: pn.pane.Markdown("", sizing_mode="stretch_width"))

    # Internal
    _sel_box: Optional[BoxAnnotation] = None
    _sel_state: ColumnDataSource = field(default_factory=lambda: ColumnDataSource(dict(xmin=[np.nan], xmax=[np.nan])))
    _auc_patch_by_sample: Dict[str, ColumnDataSource] = field(default_factory=dict)

    # Section container for app layout
    section: pn.Column = field(init=False)

    def __post_init__(self):
        self.export_name.param.watch(lambda e: setattr(self.export_btn, "filename", e.new or "normalized.csv"), "value")

        # Build initial figure
        self._render_before_fig()

        # Wire actions
        self.apply_btn.on_click(self._on_apply)
        self.skip_btn.on_click(self._on_skip)
        self.export_btn.callback = self._export_bytes

        # Top-level section
        self.section = pn.Column(
            pn.pane.Markdown("### Normalization (select peak region with Box Select)"),
            self.before_pane,
            pn.Row(self.apply_btn, pn.Spacer(width=8), self.skip_btn),
            pn.layout.Divider(),
            pn.pane.Markdown("**Preview of normalized curves (after Apply):**"),
            self.after_pane,
            pn.Row(self.export_name, self.export_btn),
            self.export_status,
            sizing_mode="stretch_width",
            visible=False,  # unlocked by app once input is available
        )

    # ------------------------ FIGURE + JS wiring ------------------------

    def _render_before_fig(self):
        if not self.current_by_sample:
            self.before_pane.object = None
            self.apply_btn.disabled = True
            self.skip_btn.disabled = True
            return

        n = len(self.current_by_sample)
        colors = _palette(n)

        fig = bokeh.plotting.figure(
            title="Normalization preview — Box select a region; AUC of tallest peak is filled (red)",
            height=400, sizing_mode="stretch_width",
            x_axis_label="time", y_axis_label="intensity",
            tools="pan,wheel_zoom,box_zoom,box_select,reset,save,hover",
            active_drag="box_select",   # UX
            # do not set active_scroll
        )
        fig.legend.click_policy = "hide"
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None

        # Grey selection indicator band
        sel_box = BoxAnnotation(left=None, right=None, fill_alpha=0.15, fill_color="gray", line_color=None, level="underlay")
        fig.add_layout(sel_box)
        self._sel_box = sel_box

        # One source + one AUC patch per sample; invisible points for geometry
        self._auc_patch_by_sample.clear()
        for i, (name, df) in enumerate(self.current_by_sample.items()):
            x = df["time"].to_numpy(); y = df["intensity"].to_numpy()

            src = ColumnDataSource(dict(x=x, y=y))
            fig.line("x","y", source=src, color=colors[i % len(colors)], legend_label=name, line_width=2)

            # Invisible scatter to capture selection; suppress selection visuals
            fig.scatter(
                "x","y", source=src, size=3,
                alpha=0.001, line_alpha=0.001,
                nonselection_fill_alpha=0.0, nonselection_line_alpha=0.0,
                selection_fill_alpha=0.0, selection_line_alpha=0.0,
                hover_fill_alpha=0.0, hover_line_alpha=0.0,
            )

            # Red patch for AUC
            auc_src = ColumnDataSource(dict(px=[], py=[]))
            fig.patch("px","py", source=auc_src, fill_color="red", fill_alpha=0.35, line_color=None)
            self._auc_patch_by_sample[name] = auc_src

            # Per-sample JS callback (robust; no list args)
            cb = CustomJS(args=dict(
                src=src, auc_src=auc_src, selbox=sel_box, sel_state=self._sel_state,
            ), code="""
                const g = cb_obj.geometry || {};
                const x0 = g.x0, x1 = g.x1;

                // Update grey band
                if (x0 == null || x1 == null) {
                    selbox.left = null; selbox.right = null;
                } else {
                    selbox.left = Math.min(x0, x1);
                    selbox.right = Math.max(x0, x1);
                }

                // Clear AUC
                auc_src.data = {px:[], py:[]};

                if (x0 == null || x1 == null) {
                    sel_state.data = {xmin:[NaN], xmax:[NaN]}; sel_state.change.emit();
                    auc_src.change.emit();
                    return;
                }

                const xmin = Math.min(x0, x1), xmax = Math.max(x0, x1);
                sel_state.data = {xmin:[xmin], xmax:[xmax]};
                sel_state.change.emit();

                const x = src.data.x, y = src.data.y;
                const idx = [];
                for (let i=0;i<x.length;i++) if (x[i]>=xmin && x[i]<=xmax) idx.push(i);
                if (idx.length === 0) { auc_src.change.emit(); return; }

                // Apex
                let best = -1, besty = -Infinity;
                for (const i of idx) { if (y[i] > besty) { besty = y[i]; best = i; } }
                if (best < 0) { auc_src.change.emit(); return; }

                // Grow left/right while non-increasing; clamp to selection window
                const eps = 1e-9;
                let L = best, R = best;
                while (L > 0 && y[L-1] <= y[L] + eps) L--;
                while (R < y.length-1 && y[R+1] <= y[R] + eps) R++;

                while (L < best && x[L] < xmin) L++;
                while (R > best && x[R] > xmax) R--;

                if (L >= R) { auc_src.change.emit(); return; }

                // Build polygon to baseline (0)
                const px = [], py = [];
                for (let i=L;i<=R;i++) { px.push(x[i]); py.push(y[i]); }
                for (let i=R;i>=L;i--) { px.push(x[i]); py.push(0); }

                auc_src.data = {px, py};
                auc_src.change.emit();

                // Clear selection highlight
                try { src.selected.indices = []; } catch(_) {}
            """)
            fig.js_on_event(SelectionGeometry, cb)

        self.before_pane.object = fig
        # Enable buttons with data present
        self.apply_btn.disabled = False
        self.skip_btn.disabled = False
        self.status.object = ok("Drag a box on the left figure to preview the AUC that will be used for normalization.")

    # ------------------------ APPLY / SKIP ------------------------

    def _on_apply(self, _=None):
        if not self.current_by_sample:
            self.status.object = warn("No input data to normalize.")
            return

        # Read last selection x-range (same for all samples). If NaN, ask user to select.
        xmin = float(self._sel_state.data.get("xmin", [np.nan])[0])
        xmax = float(self._sel_state.data.get("xmax", [np.nan])[0])
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            self.status.object = warn("Select a region with Box Select before applying normalization.")
            return
        if xmax < xmin:
            xmin, xmax = xmax, xmin

        out: Dict[str, pd.DataFrame] = {}
        aucs: Dict[str, float] = {}

        for name, df in self.current_by_sample.items():
            x = df["time"].to_numpy(dtype=float)
            y = df["intensity"].to_numpy(dtype=float)

            # Indices inside selection
            sel = (x >= xmin) & (x <= xmax)
            idx = np.where(sel)[0]
            if idx.size == 0:
                aucs[name] = np.nan
                out[name] = df.copy()
                continue

            # Apex
            i_rel = int(np.argmax(y[idx]))
            best = int(idx[i_rel])

            # Grow L/R while non-increasing; clamp to selection
            eps = 1e-9
            L = best
            while L > 0 and y[L-1] <= y[L] + eps: L -= 1
            R = best
            n_1 = y.size - 1
            while R < n_1 and y[R+1] <= y[R] + eps: R += 1

            while L < best and x[L] < xmin: L += 1
            while R > best and x[R] > xmax: R -= 1

            if L >= R:
                aucs[name] = np.nan
                out[name] = df.copy()
                continue

            # Trapezoidal AUC above baseline=0 on [L..R]
            xi = x[L:R+1]; yi = y[L:R+1]
            auc = float(np.trapz(np.clip(yi, 0, None), xi))
            aucs[name] = auc if auc > 0 else np.nan
            if np.isfinite(aucs[name]) and aucs[name] > 0:
                yy = y / aucs[name]
                out[name] = pd.DataFrame({"time": x, "intensity": yy})
            else:
                out[name] = df.copy()

        self.normalized_by_sample = out

        # After plot: show normalized curves
        try:
            fig_after = plot_multi(self.normalized_by_sample, title="Normalized curves", xlab="time", ylab="intensity (normalized)")
        except Exception:
            fig_after = bokeh.plotting.figure(height=400, sizing_mode="stretch_width", title="Normalized curves")
            for i, (nm, dfi) in enumerate(self.normalized_by_sample.items()):
                fig_after.line(dfi["time"], dfi["intensity"], legend_label=nm)
            fig_after.legend.click_policy = "hide"
        self.after_pane.object = fig_after

        self.export_btn.disabled = False
        self.export_status.object = ok("Ready to export normalized CSV.")
        self.status.object = ok("Normalization applied. You can proceed to the next tab or export the normalized data.")

    def _on_skip(self, _=None):
        # Keep input dataset as current (no change)
        self.normalized_by_sample = {k: v.copy() for k, v in self.current_by_sample.items()}
        try:
            fig_after = plot_multi(self.normalized_by_sample, title="(Skipped) Using input data as normalized", xlab="time", ylab="intensity")
        except Exception:
            fig_after = bokeh.plotting.figure(height=400, sizing_mode="stretch_width", title="(Skipped) Using input data as normalized")
            for i, (nm, dfi) in enumerate(self.normalized_by_sample.items()):
                fig_after.line(dfi["time"], dfi["intensity"], legend_label=nm)
            fig_after.legend.click_policy = "hide"
        self.after_pane.object = fig_after

        self.export_btn.disabled = False
        self.export_status.object = ok("You can export this dataset (no normalization applied).")
        self.status.object = ok("Skipped normalization. Using input data.")

    # ------------------------ EXPORT ------------------------

    def _export_bytes(self):
        if not self.normalized_by_sample:
            self.export_status.object = warn("No normalized data to export.")
            return None
        # Wide CSV: time + one column per sample
        names = list(self.normalized_by_sample.keys())
        base_time = self.normalized_by_sample[names[0]]["time"].to_numpy()
        merged = pd.DataFrame({"time": base_time})
        for nm in names:
            merged[nm] = self.normalized_by_sample[nm]["intensity"].to_numpy()
        bio = __import__("io").BytesIO()
        merged.to_csv(bio, index=False)
        bio.seek(0)
        return bio

# --- Compatibility shim for older app code ---
from typing import Dict, Tuple
import pandas as pd
import panel as pn

def build_normalization_section(
    initial_by_sample: Dict[str, pd.DataFrame] | None = None,
) -> Tuple[pn.Column, "NormalizationController"]:
    """
    Back-compat factory. Returns (section, controller).
    - initial_by_sample: optional dict{name -> DataFrame(time,intensity)} to pre-load.
    """
    ctrl = NormalizationController(current_by_sample=initial_by_sample or {})
    # Section visibility: show only when data is present
    ctrl.section.visible = bool(ctrl.current_by_sample)
    if ctrl.current_by_sample:
        ctrl._render_before_fig()
        ctrl.apply_btn.disabled = False
        ctrl.skip_btn.disabled = False
    else:
        ctrl.before_pane.object = None
        ctrl.apply_btn.disabled = True
        ctrl.skip_btn.disabled = True
    return ctrl.section, ctrl
