# app/preprocess_baseline_utils.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting
from bokeh.models import Range1d
from bokeh.palettes import Category10, Turbo256
from common_plot import plot_multi

pn.extension("bokeh")  # safe no-op if already initialized

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

# --- simple robust moving-median baseline (fallback if CEtools not available) ---
def _moving_median_baseline(y: np.ndarray, window: int = 501, pad_mode: str = "reflect") -> np.ndarray:
    w = int(max(1, window | 1))  # ensure odd and >=1
    if w > y.size:
        w = y.size | 1
    half = w // 2
    yp = np.pad(y.astype(float), (half, half), mode=pad_mode)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(yp, w)
        return np.nanmedian(sw, axis=1)
    except Exception:
        # slower fallback
        out = np.empty_like(y, dtype=float)
        for i in range(y.size):
            lo = i
            hi = i + 2*half + 1
            out[i] = np.nanmedian(yp[lo:hi])
        return out

def _subtract_fixed_range_baseline(t: np.ndarray, y: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = (t >= start) & (t <= end)
    if np.any(mask):
        bg = np.nanmean(y[mask])
    else:
        bg = 0.0
    return y - bg

@dataclass
class BaselineController:
    """
    Baseline subtraction stage:
      - Preview: left=original, right=baseline-subtracted (linked axes).
      - Apply: set baseline-subtracted as current dataset.
      - Skip: pass-through.
      - Export: wide CSV of baseline-subtracted curves.
    """
    input_by_sample: Dict[str, pd.DataFrame] = field(default_factory=dict)
    output_by_sample: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # UI
    preview_btn: pn.widgets.Button = field(default_factory=lambda: pn.widgets.Button(name="Show baseline subtraction preview", button_type="primary", disabled=True))
    apply_btn: pn.widgets.Button   = field(default_factory=lambda: pn.widgets.Button(name="Apply baseline subtraction", button_type="success", disabled=True))
    skip_btn: pn.widgets.Button    = field(default_factory=lambda: pn.widgets.Button(name="Skip baseline subtraction", button_type="danger", disabled=True))

    baseline_start: pn.widgets.FloatInput = field(default_factory=lambda: pn.widgets.FloatInput(name="Baseline Start (min)", value=0.0, width=150))
    baseline_end: pn.widgets.FloatInput   = field(default_factory=lambda: pn.widgets.FloatInput(name="Baseline End (min)", value=0.0, width=150))
    preproc_offset: pn.widgets.FloatSlider = field(default_factory=lambda: pn.widgets.FloatSlider(name="Vertical offset", start=0.0, end=10.0, step=0.5, value=0.0, sizing_mode="stretch_width"))
    asinh_toggle: pn.widgets.Checkbox = field(default_factory=lambda: pn.widgets.Checkbox(name="Use asinh transform", value=True))

    before_pane: pn.pane.Bokeh     = field(default_factory=lambda: pn.pane.Bokeh(sizing_mode="stretch_width"))
    after_pane: pn.pane.Bokeh      = field(default_factory=lambda: pn.pane.Bokeh(sizing_mode="stretch_width"))

    export_name: pn.widgets.TextInput   = field(default_factory=lambda: pn.widgets.TextInput(name="CSV filename", value="baseline_subtracted.csv", width=260))
    export_btn: pn.widgets.FileDownload = field(default_factory=lambda: pn.widgets.FileDownload(label="Export CSV", filename="baseline_subtracted.csv", button_type="primary", embed=False, auto=False, callback=lambda: None, disabled=True))
    export_status: pn.pane.Markdown     = field(default_factory=lambda: pn.pane.Markdown("", sizing_mode="stretch_width"))

    status: pn.pane.Markdown        = field(default_factory=lambda: pn.pane.Markdown("", sizing_mode="stretch_width"))

    section: pn.Column = field(init=False)

    def __post_init__(self):
        # wire filename -> download widget
        self.export_name.param.watch(lambda e: setattr(self.export_btn, "filename", e.new or "baseline_subtracted.csv"), "value")

        # wire buttons
        self.preview_btn.on_click(self._on_preview_btn_click)
        self.preproc_offset.param.watch(self._on_offset_change, "value")
        self.asinh_toggle.param.watch(self._on_offset_change, "value")
        self.apply_btn.on_click(self._on_apply)
        self.skip_btn.on_click(self._on_skip)
        self.export_btn.callback = self._export_bytes

        self.section = pn.Column(
            pn.pane.Markdown("### Baseline subtraction"),
            pn.Row(self.preview_btn, pn.Spacer(width=10), self.baseline_start, self.baseline_end),
            pn.pane.Markdown(
                "_Changes to the controls do not auto-refresh. Click **Show baseline subtraction preview** to update the plots._",
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

    def _build_linked_pair(self, left: Dict[str, pd.DataFrame], right: Dict[str, pd.DataFrame]) -> Tuple[bokeh.plotting.Figure, bokeh.plotting.Figure]:
        off = self.preproc_offset.value
        ash = self.asinh_toggle.value
        p1 = plot_multi(left, "Before: original", offset=off, asinh=ash)
        p2 = plot_multi(right, "After: baseline-subtracted", offset=off, asinh=ash)
        # link ranges
        p2.x_range = p1.x_range
        p2.y_range = p1.y_range
        return p1, p2

    # --- actions ---
    def _on_preview_btn_click(self, _=None):
        self._calculate_baseline()
        self._update_plots()

    def _on_offset_change(self, event):
        if self.input_by_sample:
            self._update_plots()

    def _calculate_baseline(self):
        if not self.input_by_sample:
            self.status.object = warn("No input data. Apply or skip smoothing first.")
            return
        t_start = self.baseline_start.value
        t_end   = self.baseline_end.value
        out: Dict[str, pd.DataFrame] = {}
        for nm, df in self.input_by_sample.items():
            t = df["time"].to_numpy()
            y = df["intensity"].to_numpy()
            z = _subtract_fixed_range_baseline(t, y, t_start, t_end)
            out[nm] = pd.DataFrame({"time": t, "intensity": z})
        self.output_by_sample = out

    def _update_plots(self):
        if not self.input_by_sample:
            return
        p_before, p_after = self._build_linked_pair(self.input_by_sample, 
                                                    self.output_by_sample if self.output_by_sample else self.input_by_sample)
        self.before_pane.object = p_before
        self.after_pane.object = p_after

        if self.output_by_sample:
            self.apply_btn.disabled = False
            self.skip_btn.disabled = False
            self.export_btn.disabled = False
            self.export_status.object = ok("Ready to export baseline-subtracted CSV.")
            self.status.object = ok("Baseline preview updated.")

    def _on_apply(self, _=None):
        if not self.output_by_sample:
            # create from current slider setting if preview not pressed
            self._on_preview_btn_click()
            if not self.output_by_sample:
                self.status.object = warn("Nothing to apply.")
                return
        self.status.object = ok("Baseline subtraction applied.")
        # leave self.output_by_sample as the stage output; app will pick it up

    def _on_skip(self, _=None):
        # pass-through
        if not self.input_by_sample:
            self.status.object = warn("No input data to skip.")
            return
        self.output_by_sample = {k: v.copy() for k, v in self.input_by_sample.items()}
        p_before, p_after = self._build_linked_pair(self.input_by_sample, self.output_by_sample)
        self.before_pane.object = p_before
        self.after_pane.object = p_after
        self.apply_btn.disabled = False
        self.export_btn.disabled = False
        self.export_status.object = ok("You can export this dataset (no baseline subtraction applied).")
        self.status.object = ok("Skipped baseline subtraction. Using input data.")

    # --- export ---
    def _export_bytes(self):
        if not self.output_by_sample:
            self.export_status.object = warn("No data to export.")
            return None
        names = list(self.output_by_sample.keys())
        base_time = self.output_by_sample[names[0]]["time"].to_numpy()
        merged = pd.DataFrame({"time": base_time})
        for nm in names:
            merged[nm] = self.output_by_sample[nm]["intensity"].to_numpy()
        bio = __import__("io").BytesIO()
        merged.to_csv(bio, index=False)
        bio.seek(0)
        return bio


# --- Back-compat factory like other stages ---
def build_baseline_section(
    initial_by_sample: Dict[str, pd.DataFrame] | None = None,
) -> tuple[pn.Column, BaselineController]:
    ctrl = BaselineController(input_by_sample=initial_by_sample or {})
    if ctrl.input_by_sample:
        # Default to last 5% of the run if we have data
        s1 = next(iter(ctrl.input_by_sample))
        t_max = ctrl.input_by_sample[s1]["time"].max()
        ctrl.baseline_start.value = t_max * 0.95
        ctrl.baseline_end.value   = t_max
    
    ctrl.section.visible = bool(ctrl.input_by_sample)
    if ctrl.input_by_sample:
        ctrl.preview_btn.disabled = False
        ctrl.skip_btn.disabled = False
    return ctrl.section, ctrl
