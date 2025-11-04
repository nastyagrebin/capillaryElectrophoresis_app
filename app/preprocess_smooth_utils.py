# ======================================
# FILE: app/preprocess_smooth_utils.py
# ======================================
from __future__ import annotations
from typing import Dict
import io, zipfile
import numpy as np
import pandas as pd
import panel as pn
from common_plot import plot_multi
from scipy.signal import savgol_filter

OK = "OK:"
WARN = "Warning:"
def ok(m): return f"{OK} {m}"
def warn(m): return f"{WARN} {m}"

class SmoothController:
    def __init__(self):
        self.input_by_sample: Dict[str, pd.DataFrame] = {}
        self.smoothed_by_sample: Dict[str, pd.DataFrame] = {}

        self.window = pn.widgets.IntSlider(name="Savgol window (odd)", start=5, end=101, step=2, value=9)
        self.poly = pn.widgets.IntSlider(name="Polyorder", start=2, end=5, step=1, value=3)
        self.deriv_slider = pn.widgets.IntSlider(name="Derivative order", start=0, end=3, step=1, value=0, width=200)
        self.preview_btn = pn.widgets.Button(name="Show smoothing preview", button_type="primary", disabled=True)
        self.apply_btn   = pn.widgets.Button(name="Apply smoothed data", button_type="success", disabled=True)
        self.skip_btn    = pn.widgets.Button(name="Skip smoothing (use input data)", button_type="danger", disabled=True)


        self.before_pane = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")
        self.after_pane  = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")

        self.export_name = pn.widgets.TextInput(name="Smoothed filename", value="smoothed_merged.csv", width=300)
        self.export_btn  = pn.widgets.FileDownload(label="Export smoothed", filename=self.export_name.value,
                                                   button_type="primary", embed=False, auto=False,
                                                   callback=lambda: io.BytesIO(b""), disabled=True)
        self.export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.export_name.param.watch(lambda e: setattr(self.export_btn, "filename", e.new or "smoothed_merged.csv"), "value")

        self.preview_btn.on_click(self._on_preview)
        self.export_btn.callback = self._export_bytes

        self.section = pn.Column(
            pn.pane.Markdown("### Smoothing"),
            pn.Row(self.preview_btn, pn.Spacer(width=10), self.window, self.poly, self.deriv_slider),
            pn.pane.Markdown(
                "_Changes to the controls do not auto-refresh. Click **Show smoothing preview** to update the plots._",
                styles={"color": "#555"},
                sizing_mode="stretch_width",
            ),
            pn.Row(self.before_pane, pn.Spacer(width=12), self.after_pane, sizing_mode="stretch_width"),
            pn.Row(self.apply_btn, pn.Spacer(width=8), self.skip_btn),
            pn.Row(self.export_name, self.export_btn),
            self.export_status,
            self.status,
            sizing_mode="stretch_width",
            visible=False,
        )

    @staticmethod
    def _safe_savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
        n = y.size
        if n == 0:
            return y.copy()
        w = int(window)
        if w % 2 == 0:
            w -= 1
        minw = poly + 2 if (poly + 2) % 2 == 1 else (poly + 3)
        w = max(w, minw)
        w = min(w, n - 1 if (n - 1) % 2 == 1 else (n - 2))
        if w < 3:
            w = 3
        if w >= n:
            w = n - 1 if (n - 1) % 2 == 1 else n - 2
        return savgol_filter(y, window_length=int(w), polyorder=int(poly), mode="interp")

    def _on_preview(self, _):
        if not self.input_by_sample:
            self.status.object = warn("No input for smoothing. Apply despiked data first.")
            return
        W, P = int(self.window.value), int(self.poly.value)
        self.smoothed_by_sample.clear()
        for s, df in self.input_by_sample.items():
            y = df["intensity"].to_numpy(dtype=float)
            y2 = self._safe_savgol(y, window=W, poly=P)
            self.smoothed_by_sample[s] = pd.DataFrame({"time": df["time"].to_numpy(), "intensity": y2})
        b = plot_multi(self.input_by_sample, "Before smoothing (input)")
        a = plot_multi(self.smoothed_by_sample, "After smoothing")
        a.x_range = b.x_range; a.y_range = b.y_range
        self.before_pane.object = b; self.after_pane.object = a
        self.apply_btn.disabled = False
        self.export_btn.disabled = False
        self.export_status.object = ok("Smoothed data ready to export.")
        self.status.object = ok("Smoothing preview generated.")

    def _export_bytes(self) -> io.BytesIO:
        if not self.smoothed_by_sample:
            self.export_status.object = warn("No smoothed data to export.")
            return io.BytesIO(b"")
        names = list(self.smoothed_by_sample.keys())
        t0 = self.smoothed_by_sample[names[0]]["time"].to_numpy()
        same = all(
            df["time"].to_numpy().shape == t0.shape and np.allclose(df["time"].to_numpy(), t0, rtol=1e-9, atol=1e-12)
            for df in self.smoothed_by_sample.values()
        )
        bio = io.BytesIO()
        if same:
            merged = pd.DataFrame({"time": t0})
            for nm in names:
                merged[nm] = self.smoothed_by_sample[nm]["intensity"].to_numpy()
            merged.to_csv(bio, index=False)
            self.export_status.object = ok("Exporting merged smoothed CSV.")
            bio.seek(0); return bio
        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for nm, df in self.smoothed_by_sample.items():
                zf.writestr(f"{nm}.csv", df.to_csv(index=False))
        self.export_status.object = warn("Time stamps differ; exported per-sample CSVs as ZIP.")
        bio.seek(0); return bio

def build_smoothing_section():
    ctrl = SmoothController()
    return ctrl.section, ctrl
