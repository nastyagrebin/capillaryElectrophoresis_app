# app/snr_utils.py
"""
SNR (Signal-to-Noise Ratio) tab.

  S      = max intensity in the user-selected *signal* region (per sample)
  B      = mean intensity in the user-selected *noise* region
  σ_B    = standard deviation of intensity in the noise region
  SNR    = (S − B) / σ_B
"""
from __future__ import annotations
import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting
from bokeh.models import BoxAnnotation, ColumnDataSource
from bokeh.palettes import Category10, Turbo256

OK = "OK:"; WARN = "Warning:"
def ok(m):   return f"{OK} {m}"
def warn(m): return f"{WARN} {m}"

def _palette(n: int) -> List[str]:
    if n <= 10:
        return list(Category10[10])[:n]
    idxs = np.linspace(0, 255, num=n, dtype=int)
    return [Turbo256[i] for i in idxs]


class SNRController:
    """Controls the SNR calculation tab."""

    def __init__(self):
        self.input_by_sample: Dict[str, pd.DataFrame] = {}
        self._samples: List[str] = []
        self._colors: List[str] = []

        # Selected x-ranges (set after user draws a box-select on the plot)
        self.signal_range: Optional[Tuple[float, float]] = None
        self.noise_range:  Optional[Tuple[float, float]] = None
        self._current_xrange: Optional[Tuple[float, float]] = None

        # References to live BoxAnnotations on the current figure
        self._signal_ann: Optional[BoxAnnotation] = None
        self._noise_ann:  Optional[BoxAnnotation] = None

        # Results
        self.results_df: Optional[pd.DataFrame] = None

        # ---------- Viz Controls ----------
        self.offset_slider = pn.widgets.FloatSlider(
            name="Vertical offset", start=0.0, end=10.0, step=0.5, value=0.0,
            sizing_mode="stretch_width"
        )
        self.asinh_toggle = pn.widgets.Checkbox(name="Use asinh transform", value=True)

        # ---------- Sample Exclusion ----------
        self.exclude_group = pn.widgets.CheckBoxGroup(
            name="Samples to exclude", options=[], value=[], inline=True
        )

        # ---------- Plot ----------
        self.plot_pane = pn.pane.Bokeh(sizing_mode="stretch_width")

        # ---------- Region Buttons ----------
        self.set_signal_btn = pn.widgets.Button(
            name="Set as Signal Region ▶", button_type="primary", disabled=True
        )
        self.set_noise_btn = pn.widgets.Button(
            name="Set as Noise Region ▶", button_type="warning", disabled=True
        )
        self.calc_btn = pn.widgets.Button(
            name="Calculate SNR", button_type="success", disabled=True
        )

        # ---------- Status ----------
        self.status        = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.signal_status = pn.pane.Markdown("Signal region: _not set_", sizing_mode="stretch_width")
        self.noise_status  = pn.pane.Markdown("Noise region: _not set_",  sizing_mode="stretch_width")

        # ---------- Results ----------
        self.results_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=False,
            height=300,
            sizing_mode="stretch_width",
            disabled=True,
        )

        # ---------- Export ----------
        self.csv_name = pn.widgets.TextInput(name="CSV filename", value="snr_results.csv", width=260)
        self.csv_download = pn.widgets.FileDownload(
            label="Download SNR CSV", filename="snr_results.csv",
            button_type="primary", embed=False, auto=False,
            callback=lambda: io.BytesIO(b""),
            disabled=True,
        )
        self.csv_name.param.watch(
            lambda e: setattr(self.csv_download, "filename", e.new or "snr_results.csv"), "value"
        )
        self.csv_download.callback = self._csv_bytes

        # ---------- Wire ----------
        self.offset_slider.param.watch(lambda _: self._rebuild_figure(), "value")
        self.asinh_toggle.param.watch(lambda _:  self._rebuild_figure(), "value")
        self.set_signal_btn.on_click(self._on_set_signal)
        self.set_noise_btn.on_click(self._on_set_noise)
        self.calc_btn.on_click(self._on_calculate)

        # ---------- Layout ----------
        self.section = pn.Column(
            pn.pane.Markdown("## Signal-to-Noise Ratio (SNR)"),
            pn.pane.Markdown(
                "This tab calculates the SNR for each sample using two user-defined regions:\n\n"
                "- **Signal region** (blue shading): the tallest peak height **S** is extracted per sample.\n"
                "- **Noise region** (gray shading): the mean **B** and standard deviation **σ_B** of "
                "the baseline are extracted.\n\n"
                "The SNR is then **SNR = (S − B) / σ_B**.\n\n"
                "Use the **Box Select** tool on the plot to draw a region, then click "
                "**Set as Signal Region** or **Set as Noise Region** to register it. "
                "Samples checked in the exclusion box are skipped during calculation.",
                sizing_mode="stretch_width",
            ),
            pn.layout.Divider(),
            pn.pane.Markdown("### Sample Exclusion"),
            pn.Column(
                pn.pane.Markdown("Check samples to **exclude** from SNR calculation:"),
                self.exclude_group,
                sizing_mode="stretch_width",
                styles={"background": "#f9f9f9", "padding": "10px", "border-radius": "5px"},
            ),
            pn.layout.Divider(),
            pn.Row(self.offset_slider, self.asinh_toggle),
            self.plot_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### Region Selection"),
            pn.pane.Markdown(
                "_Draw a box on the plot with the **Box Select** tool, "
                "then click the button to assign it as Signal (blue) or Noise (gray)._",
                styles={"color": "#555"},
            ),
            pn.Row(self.set_signal_btn, pn.Spacer(width=12), self.set_noise_btn),
            self.signal_status,
            self.noise_status,
            pn.layout.Divider(),
            self.calc_btn,
            self.status,
            pn.layout.Divider(),
            pn.pane.Markdown("### Results"),
            self.results_table,
            pn.Row(self.csv_name, self.csv_download),
            sizing_mode="stretch_width",
            visible=False,
        )

    # ------------------------------------------------------------------
    # External API
    # ------------------------------------------------------------------

    def set_input(self, by_sample: Dict[str, pd.DataFrame]) -> None:
        """Called by app.py whenever new converted data is available."""
        self.input_by_sample = {k: v.copy() for k, v in by_sample.items()}
        self._samples = list(self.input_by_sample.keys())
        n = len(self._samples)
        self._colors = _palette(n)

        # Reset state
        self.signal_range = None
        self.noise_range  = None
        self._current_xrange = None
        self.results_df = None
        self.signal_status.object = "Signal region: _not set_"
        self.noise_status.object  = "Noise region: _not set_"
        self.calc_btn.disabled    = True
        self.csv_download.disabled = True
        self.results_table.value  = pd.DataFrame()
        self.status.object = ""

        self.exclude_group.options = self._samples
        self.exclude_group.value   = []

        self._rebuild_figure()
        self.section.visible = True

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _rebuild_figure(self):
        if not self.input_by_sample:
            return

        off = self.offset_slider.value
        ash = self.asinh_toggle.value
        n_samples = len(self._samples)
        ncols = 6
        legend_rows = math.ceil(n_samples / ncols) if n_samples > 0 else 1
        total_height = 500 + legend_rows * 30

        p = bokeh.plotting.figure(
            height=total_height, sizing_mode="stretch_width",
            title="Sample Chromatograms — use Box Select then assign a region",
            x_axis_label="time (min)",
            y_axis_label="fluorescence (asinh)" if ash else "fluorescence (raw)",
            tools="pan,box_select,box_zoom,wheel_zoom,reset,save,hover",
            active_drag="box_select",
        )
        p.hover.tooltips = [("time", "$x{0.000}"), ("y", "$y{0.000}")]

        for i, s in enumerate(self._samples):
            df  = self.input_by_sample[s]
            t   = df["time"].to_numpy()
            y   = df["intensity"].to_numpy().copy()
            if ash:
                y = np.arcsinh(y)
            y = y - i * off
            color = self._colors[i % len(self._colors)]
            src = ColumnDataSource({"x": t, "y": y})
            p.line("x", "y", source=src, color=color, line_width=2, alpha=0.9, legend_label=str(s)[:12])
            # Invisible scatter so box-select captures indices
            p.scatter("x", "y", source=src, size=0, alpha=0)
            src.selected.on_change(
                "indices",
                lambda attr, old, new, _src=src: self._on_selection_change(_src)
            )

        # BoxAnnotations for the two regions
        self._signal_ann = BoxAnnotation(
            left=None, right=None,
            fill_color="royalblue", fill_alpha=0.15, line_color=None, level="underlay"
        )
        self._noise_ann = BoxAnnotation(
            left=None, right=None,
            fill_color="gray", fill_alpha=0.15, line_color=None, level="underlay"
        )
        # Restore if already set
        if self.signal_range:
            self._signal_ann.left  = self.signal_range[0]
            self._signal_ann.right = self.signal_range[1]
        if self.noise_range:
            self._noise_ann.left  = self.noise_range[0]
            self._noise_ann.right = self.noise_range[1]
        p.add_layout(self._signal_ann)
        p.add_layout(self._noise_ann)

        if p.legend:
            leg = p.legend[0]
            p.add_layout(leg, "below")
            leg.orientation    = "horizontal"
            leg.location       = "top_left"
            leg.click_policy   = "hide"
            leg.label_text_font_size = "9pt"
            leg.ncols          = ncols

        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        self.plot_pane.object = p

        self.set_signal_btn.disabled = False
        self.set_noise_btn.disabled  = False

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _on_selection_change(self, src: ColumnDataSource):
        indices = src.selected.indices
        if not indices:
            return
        xs = src.data["x"]
        self._current_xrange = (float(xs[min(indices)]), float(xs[max(indices)]))

    def _on_set_signal(self, _=None):
        if self._current_xrange is None:
            self.status.object = warn("Draw a box on the plot first.")
            return
        self.signal_range = self._current_xrange
        if self._signal_ann is not None:
            self._signal_ann.left  = self.signal_range[0]
            self._signal_ann.right = self.signal_range[1]
        self.signal_status.object = (
            f"**Signal region (blue):** {self.signal_range[0]:.3f} – {self.signal_range[1]:.3f} min"
        )
        self.status.object = ok(
            f"Signal region set: [{self.signal_range[0]:.3f}, {self.signal_range[1]:.3f}] min."
        )
        self._maybe_enable_calc()

    def _on_set_noise(self, _=None):
        if self._current_xrange is None:
            self.status.object = warn("Draw a box on the plot first.")
            return
        self.noise_range = self._current_xrange
        if self._noise_ann is not None:
            self._noise_ann.left  = self.noise_range[0]
            self._noise_ann.right = self.noise_range[1]
        self.noise_status.object = (
            f"**Noise region (gray):** {self.noise_range[0]:.3f} – {self.noise_range[1]:.3f} min"
        )
        self.status.object = ok(
            f"Noise region set: [{self.noise_range[0]:.3f}, {self.noise_range[1]:.3f}] min."
        )
        self._maybe_enable_calc()

    def _maybe_enable_calc(self):
        self.calc_btn.disabled = not (
            self.signal_range is not None and self.noise_range is not None
        )

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------

    def _on_calculate(self, _=None):
        if self.signal_range is None or self.noise_range is None:
            self.status.object = warn("Set both Signal and Noise regions first.")
            return

        exclude = set(self.exclude_group.value)
        samples_to_use = [s for s in self._samples if s not in exclude]
        if not samples_to_use:
            self.status.object = warn("No samples selected — all are excluded.")
            return

        sig_lo, sig_hi = self.signal_range
        noi_lo, noi_hi = self.noise_range
        rows = []

        for s in samples_to_use:
            df   = self.input_by_sample[s]
            t    = df["time"].to_numpy()
            y    = df["intensity"].to_numpy()

            sig_mask = (t >= sig_lo) & (t <= sig_hi)
            noi_mask = (t >= noi_lo) & (t <= noi_hi)

            S = float(np.max(y[sig_mask])) if np.any(sig_mask) else np.nan
            if np.any(noi_mask):
                B       = float(np.mean(y[noi_mask]))
                sigma_B = float(np.std(y[noi_mask]))
            else:
                B = sigma_B = np.nan

            snr = (S - B) / sigma_B if (
                np.isfinite(S) and np.isfinite(B) and np.isfinite(sigma_B) and sigma_B > 0
            ) else np.nan

            rows.append({
                "Sample":   s,
                "S":        round(S, 4)       if np.isfinite(S)       else np.nan,
                "B":        round(B, 4)       if np.isfinite(B)       else np.nan,
                "sigma_B":  round(sigma_B, 4) if np.isfinite(sigma_B) else np.nan,
                "SNR":      round(snr, 2)     if np.isfinite(snr)     else np.nan,
            })

        self.results_df = pd.DataFrame(rows).set_index("Sample")
        self.results_table.value = self.results_df.reset_index()
        self.csv_download.disabled = False
        self.status.object = ok(f"SNR calculated for {len(rows)} sample(s).")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _csv_bytes(self):
        if self.results_df is None or self.results_df.empty:
            return io.BytesIO(b"")
        bio = io.BytesIO()
        self.results_df.to_csv(bio)
        bio.seek(0)
        return bio
