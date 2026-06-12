# ======================================
# FILE: app/preprocess_despike_utils.py
# ======================================
from __future__ import annotations
from typing import Dict
import io
import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting  
from common_plot import plot_multi

OK = "OK:"
WARN = "Warning:"
def ok(m): return f"{OK} {m}"
def warn(m): return f"{WARN} {m}"

class DespikeController:
    def __init__(self):
        # I/O datasets
        self.input_by_sample: Dict[str, pd.DataFrame] = {}
        self.output_by_sample: Dict[str, pd.DataFrame] = {}

        # Widgets (parameters)
        self.window = pn.widgets.IntSlider(name="Neighborhood window (odd)", start=3, end=101, step=2, value=5, width=260)
        self.z_thresh = pn.widgets.FloatSlider(name="Z-score threshold", start=2.0, end=10.0, step=0.1, value=5.0, width=220)
        self.preproc_offset = pn.widgets.FloatSlider(name="Vertical offset", start=0.0, end=10.0, step=0.5, value=0.0, sizing_mode="stretch_width")
        self.asinh_toggle = pn.widgets.Checkbox(name="Use asinh transform", value=True)

        # Actions
        self.preview_btn = pn.widgets.Button(name="Show despiking preview", button_type="primary", disabled=True)
        self.apply_btn   = pn.widgets.Button(name="Apply despiked data", button_type="success", disabled=True)
        self.skip_btn    = pn.widgets.Button(name="Skip despiking (use input)", button_type="danger", disabled=True)

        # Panes
        self.before_pane = pn.pane.Bokeh(sizing_mode="stretch_width")
        self.after_pane  = pn.pane.Bokeh(sizing_mode="stretch_width")

        # Export
        self.export_name = pn.widgets.TextInput(name="Despiked filename", value="despiked_merged.csv", width=300)
        self.export_btn  = pn.widgets.FileDownload(
            label="Export despiked", filename=self.export_name.value,
            button_type="primary", embed=False, auto=False,
            callback=lambda: io.BytesIO(b""), disabled=True
        )
        self.export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.export_name.param.watch(lambda e: setattr(self.export_btn, "filename", e.new or "despiked_merged.csv"), "value")

        # Wiring
        self.preview_btn.on_click(self._on_preview_btn_click)
        self.preproc_offset.param.watch(self._on_offset_change, "value")
        self.asinh_toggle.param.watch(self._on_offset_change, "value")
        self.export_btn.callback = self._export_bytes
        self.apply_btn.on_click(self._on_apply)
        self.skip_btn.on_click(self._on_skip)

        # Layout (with hint text)
        self.section = pn.Column(
            pn.pane.Markdown("### Despiking"),
            pn.Row(self.preview_btn, pn.Spacer(width=10), self.z_thresh, self.window),
            pn.pane.Markdown(
                "_Changes to the controls do not auto-refresh. Click **Show despiking preview** to update the plots._",
                styles={"color": "#555"},
                sizing_mode="stretch_width",
            ),
            pn.Row(self.preproc_offset, self.asinh_toggle),
            pn.Row(self.before_pane, pn.Spacer(width=12), self.after_pane, sizing_mode="stretch_width"),
            pn.Row(self.apply_btn, pn.Spacer(width=8), self.skip_btn),
            pn.Row(self.export_name, self.export_btn),
            self.export_status,
            self.status,
            sizing_mode="stretch_width",
            visible=False,
        )

    # ---------- helpers ----------

    def _remove_single_point_spikes(self, y: np.ndarray, window: int, z_thresh: float) -> np.ndarray:
        """Prefer CEtools.filters.remove_single_point_spikes; else robust fallback."""
        try:
            from CEtools.filters import remove_single_point_spikes  # type: ignore
            return remove_single_point_spikes(y, window=window, z_thresh=z_thresh)
        except Exception:
            y = np.asarray(y, dtype=float)
            n = y.size
            if n == 0: return y.copy()
            w = max(3, int(window) // 2 * 2 + 1)  # odd
            half = w // 2
            out = y.copy()
            for i in range(n):
                lo = max(0, i - half); hi = min(n, i + half + 1)
                win = y[lo:hi]
                med = np.median(win)
                mad = np.median(np.abs(win - med)) + 1e-12
                z = (y[i] - med) / (1.4826 * mad)
                if abs(z) >= z_thresh:
                    out[i] = med
            return out

    # ---------- actions ----------
    def _on_preview_btn_click(self, _=None):
        self._calculate_despiking()
        self._update_plots()

    def _on_offset_change(self, event):
        if self.input_by_sample:
            self._update_plots()

    def _calculate_despiking(self):
        if not self.input_by_sample:
            self.status.object = "No input data. Convert files and unlock preprocessing first."
            return
        z = float(self.z_thresh.value)
        w = int(self.window.value) | 1  # force odd

        out: Dict[str, pd.DataFrame] = {}
        for nm, df in self.input_by_sample.items():
            t = df["time"].to_numpy()
            y = df["intensity"].to_numpy()
            try:
                y2 = self._remove_single_point_spikes(y, window=w, z_thresh=z)
            except Exception as e:
                self.status.object = f"Despike failed for {nm}: {e}"
                y2 = y
            out[nm] = pd.DataFrame({"time": t, "intensity": y2})
        self.output_by_sample = out

    def _update_plots(self):
        if not self.input_by_sample:
            return
            
        off = self.preproc_offset.value
        ash = self.asinh_toggle.value
        p_before = plot_multi(self.input_by_sample, "Before: original", offset=off, asinh=ash)
        
        # If we have calculated output, show it; else show input in 'after' pane too
        current_out = self.output_by_sample if self.output_by_sample else self.input_by_sample
        p_after = plot_multi(current_out, "After: despiked", offset=off, asinh=ash)
        
        p_after.x_range = p_before.x_range
        p_after.y_range = p_before.y_range
        self.before_pane.object = p_before
        self.after_pane.object = p_after

        if self.output_by_sample:
            self.apply_btn.disabled = False
            self.export_btn.disabled = False
            self.export_status.object = ok("Ready to export despiked CSV.")
            self.status.object = "Preview updated. If satisfied, click 'Apply despiked data' to continue."

    def _on_apply(self, _=None):
        if not self.output_by_sample:
            self._on_preview_btn_click()
            if not self.output_by_sample:
                self.status.object = "Nothing to apply."
                return
        self.status.object = "Despiked data applied."

    def _on_skip(self, _=None):
        if not self.input_by_sample:
            self.status.object = "No input data to skip."
            return
        self.output_by_sample = {k: v.copy() for k, v in self.input_by_sample.items()}
        off = self.preproc_offset.value

        p_before = plot_multi(self.input_by_sample, "Before: original", offset=off)
        p_after  = plot_multi(self.output_by_sample, "After: (skipped) using input", offset=off)
        p_after.x_range = p_before.x_range
        p_after.y_range = p_before.y_range
        self.before_pane.object = p_before
        self.after_pane.object = p_after

        self.apply_btn.disabled = False
        self.export_btn.disabled = True  # unchanged dataset
        self.export_status.object = ""
        self.status.object = "Skipped despiking. Using input data."

    def _export_bytes(self):
        if not self.output_by_sample:
            self.export_status.object = "No data to export."
            return None
        names = list(self.output_by_sample.keys())
        base_time = self.output_by_sample[names[0]]["time"].to_numpy()
        merged = pd.DataFrame({"time": base_time})
        for nm in names:
            merged[nm] = self.output_by_sample[nm]["intensity"].to_numpy()
        bio = io.BytesIO()
        merged.to_csv(bio, index=False)
        bio.seek(0)
        return bio


def build_despike_section():
    """Factory returning (section, controller) for app assembly."""
    ctrl = DespikeController()
    return ctrl.section, ctrl
