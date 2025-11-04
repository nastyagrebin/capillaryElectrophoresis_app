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

def _subtract_baseline(y: np.ndarray, window: int) -> np.ndarray:
    # Prefer CEtools if present; else fallback
    try:
        import CEtools as cet
        b = cet.find_baseline(y, window=window)  # user’s package function
    except Exception:
        b = _moving_median_baseline(y, window=window)
    z = y - b
    # keep negative values (don’t floor), user asked for subtraction preview not clipping
    return z

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

    win: pn.widgets.IntSlider      = field(default_factory=lambda: pn.widgets.IntSlider(name="Baseline window (odd)", start=51, end=2001, step=2, value=501, width=280))

    before_pane: pn.pane.Bokeh     = field(default_factory=lambda: pn.pane.Bokeh(height=380, sizing_mode="stretch_width"))
    after_pane: pn.pane.Bokeh      = field(default_factory=lambda: pn.pane.Bokeh(height=380, sizing_mode="stretch_width"))

    export_name: pn.widgets.TextInput   = field(default_factory=lambda: pn.widgets.TextInput(name="CSV filename", value="baseline_subtracted.csv", width=260))
    export_btn: pn.widgets.FileDownload = field(default_factory=lambda: pn.widgets.FileDownload(label="Export CSV", filename="baseline_subtracted.csv", button_type="primary", embed=False, auto=False, callback=lambda: None, disabled=True))
    export_status: pn.pane.Markdown     = field(default_factory=lambda: pn.pane.Markdown("", sizing_mode="stretch_width"))

    status: pn.pane.Markdown        = field(default_factory=lambda: pn.pane.Markdown("", sizing_mode="stretch_width"))

    section: pn.Column = field(init=False)

    def __post_init__(self):
        # wire filename -> download widget
        self.export_name.param.watch(lambda e: setattr(self.export_btn, "filename", e.new or "baseline_subtracted.csv"), "value")

        # wire buttons
        self.preview_btn.on_click(self._on_preview)
        self.apply_btn.on_click(self._on_apply)
        self.skip_btn.on_click(self._on_skip)
        self.export_btn.callback = self._export_bytes

        self.section = pn.Column(
            pn.pane.Markdown("### Baseline subtraction"),
            pn.Row(self.preview_btn, pn.Spacer(width=10), self.win),
            pn.pane.Markdown(
                "_Changes to the controls do not auto-refresh. Click **Show baseline subtraction preview** to update the plots._",
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

    # --- helpers ---
    @staticmethod
    def _plot_multi(samples: Dict[str, pd.DataFrame], title: str, *, xlab="time", ylab="intensity") -> bokeh.plotting.Figure:
        colors = _palette(len(samples))
        p = bokeh.plotting.figure(
            title=title, height=380, sizing_mode="stretch_width",
            x_axis_label=xlab, y_axis_label=ylab,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            # do not set active_scroll
        )
        for i, (nm, df) in enumerate(samples.items()):
            p.line(df["time"], df["intensity"], color=colors[i % len(colors)], legend_label=nm)
        p.legend.click_policy = "hide"
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        return p

    def _build_linked_pair(self, left: Dict[str, pd.DataFrame], right: Dict[str, pd.DataFrame]) -> Tuple[bokeh.plotting.Figure, bokeh.plotting.Figure]:
        p1 = self._plot_multi(left, "Before: original")
        p2 = self._plot_multi(right, "After: baseline-subtracted")
        # link ranges
        p2.x_range = p1.x_range = p1.x_range if isinstance(p1.x_range, Range1d) else p1.x_range
        p2.y_range = p1.y_range = p1.y_range if isinstance(p1.y_range, Range1d) else p1.y_range
        return p1, p2

    # --- actions ---
    def _on_preview(self, _=None):
        if not self.input_by_sample:
            self.status.object = warn("No input data. Apply or skip smoothing first.")
            return
        W = int(self.win.value) | 1
        out: Dict[str, pd.DataFrame] = {}
        for nm, df in self.input_by_sample.items():
            t = df["time"].to_numpy()
            y = df["intensity"].to_numpy()
            z = _subtract_baseline(y, window=W)
            out[nm] = pd.DataFrame({"time": t, "intensity": z})
        self.output_by_sample = out

        p_before, p_after = self._build_linked_pair(self.input_by_sample, self.output_by_sample)
        self.before_pane.object = p_before
        self.after_pane.object = p_after

        self.apply_btn.disabled = False
        self.skip_btn.disabled = False
        self.export_btn.disabled = False
        self.export_status.object = ok("Ready to export baseline-subtracted CSV.")
        self.status.object = ok("Preview generated.")

    def _on_apply(self, _=None):
        if not self.output_by_sample:
            # create from current slider setting if preview not pressed
            self._on_preview()
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
    ctrl.section.visible = bool(ctrl.input_by_sample)
    if ctrl.input_by_sample:
        ctrl.preview_btn.disabled = False
        ctrl.skip_btn.disabled = False
    return ctrl.section, ctrl
