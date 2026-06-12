# ============================
# FILE: app/diversity_utils.py
# ============================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import io
import math
import numpy as np
import pandas as pd
import panel as pn

OK = "OK:"; WARN = "Warning:"
def ok(m): return f"{OK} {m}"
def warn(m): return f"{WARN} {m}"

# UI needs Tabulator
pn.extension('tabulator')


# ============================== Data structures ==================================

@dataclass(frozen=True)
class Peak:
    index: int
    x: float
    height: float
    prominence: float
    left_base: int
    right_base: int
    area: float
    left_base_t: float = 0.0
    right_base_t: float = 0.0


# ============================== Utilities ========================================

def _validate_xy(x: np.ndarray, y: np.ndarray, *, require_unit_interval: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x.size != y.size:
        raise ValueError("x and y must be the same length.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")
    return x.astype(float, copy=False), y.astype(float, copy=False)

def _skim_area(x: np.ndarray, y: np.ndarray, i0: int, i1: int) -> float:
    if i1 <= i0:
        return 0.0
    total_area = float(np.trapz(y[i0:i1 + 1], x[i0:i1 + 1]))
    base_area = 0.5 * (y[i0] + y[i1]) * (x[i1] - x[i0])
    return max(0.0, total_area - base_area)

def compute_total_auc(
    x: np.ndarray,
    y: np.ndarray,
    roi: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    Total area under the curve over the ROI (pseudotime).
    """
    x, y = _validate_xy(x, y, require_unit_interval=True)
    lo, hi = max(0.0, float(roi[0])), min(1.0, float(roi[1]))
    if hi <= lo:
        return 0.0
    m = (x >= lo) & (x <= hi)
    if not np.any(m):
        return 0.0
    return float(np.trapz(y[m], x[m]))


# ============================== Peak finding =====================================

def find_electropherogram_peaks(
    x: np.ndarray,
    y: np.ndarray,
    prominence: float,
    *,
    prominence_mode: Literal["absolute", "relative"] = "absolute",
    distance: Optional[int] = None,
    width: Optional[int] = None,
    rel_height: float = 0.5,
    roi: Tuple[float, float] = (0.0, 1.0),
    edge_smooth_window: int = 11,
) -> List[Peak]:
    x, y = _validate_xy(x, y, require_unit_interval=True)

    lo, hi = max(0.0, float(roi[0])), min(1.0, float(roi[1]))
    if hi <= lo:
        return []
        
    if y.size < 3:
        return []

    prom_thresh = float(prominence)
    if prominence_mode == "relative":
        scale = float(np.ptp(y)) or 1.0
        prom_thresh = max(0.0, prom_thresh * scale)
    else:
        prom_thresh = max(0.0, prom_thresh)

    try:
        from scipy.signal import find_peaks as sp_find_peaks, peak_prominences, savgol_filter  # type: ignore

        w = max(3, edge_smooth_window)
        if w % 2 == 0:
            w += 1
        if y.size >= w:
            y_edges = savgol_filter(y, window_length=w, polyorder=2)
        else:
            y_edges = y

        peak_idx, props = sp_find_peaks(
            y_edges,
            distance=distance,
            prominence=prom_thresh if prom_thresh > 0 else None,
            width=width,
        )
        if peak_idx.size == 0:
            return []

        if "prominences" not in props:
            prominences, left_bases, right_bases = peak_prominences(y_edges, peak_idx)
        else:
            prominences = props["prominences"]
            left_bases = props["left_bases"]
            right_bases = props["right_bases"]

        def get_valley(idx1: int, idx2: int) -> int:
            if idx1 >= idx2: return idx1
            return idx1 + int(np.argmin(y_edges[idx1:idx2+1]))

        peaks: List[Peak] = []
        for i, (pk, prom) in enumerate(zip(peak_idx, prominences)):
            if not (lo <= x[pk] <= hi):
                continue
                
            lb = int(left_bases[i])
            if i > 0 and peak_idx[i-1] > lb:
                lb = get_valley(peak_idx[i-1], pk)
                
            rb = int(right_bases[i])
            if i < len(peak_idx) - 1 and peak_idx[i+1] < rb:
                rb = get_valley(pk, peak_idx[i+1])
                
            # Tangent Skimming
            if y_edges[lb] > y_edges[rb]:
                if rb > pk:
                    j_range = np.arange(pk + 1, rb + 1)
                    if len(j_range) > 0:
                        slopes = (y_edges[j_range] - y_edges[lb]) / (x[j_range] - x[lb] + 1e-12)
                        rb = int(j_range[np.argmin(slopes)])
            elif y_edges[rb] > y_edges[lb]:
                if pk > lb:
                    j_range = np.arange(lb, pk)
                    if len(j_range) > 0:
                        slopes = (y_edges[j_range] - y_edges[rb]) / (x[j_range] - x[rb] - 1e-12)
                        lb = int(j_range[np.argmax(slopes)])

            area = _skim_area(x, y, lb, rb)
            peaks.append(
                Peak(
                    index=int(pk),
                    x=float(x[pk]),
                    height=float(y[pk]),
                    prominence=float(prom),
                    left_base=lb,
                    right_base=rb,
                    area=area,
                    left_base_t=float(x[lb]),
                    right_base_t=float(x[rb]),
                )
            )
            
        return peaks

    except Exception:
        # NumPy fallback: simple local maxima + crude prominence & bases
        left = y[1:-1] > y[:-2]
        right = y[1:-1] > y[2:]
        candidate_idx = np.where(left & right)[0] + 1
        if candidate_idx.size == 0:
            return []

        def walk_left(i: int) -> int:
            j = i
            while j > 0 and y[j - 1] <= y[j]:
                j -= 1
            return j

        def walk_right(i: int) -> int:
            j = i
            n = y.size
            while j < n - 1 and y[j + 1] <= y[j]:
                j += 1
            return j

        peaks: List[Peak] = []
        for i in candidate_idx:
            if not (lo <= x[i] <= hi):
                continue
            lb, rb = walk_left(i), walk_right(i)
            left_min = float(np.min(y[lb:i])) if lb < i else float(y[i])
            right_min = float(np.min(y[i + 1:rb + 1])) if i + 1 <= rb else float(y[i])
            prom = float(y[i] - max(left_min, right_min))
            if prom < prom_thresh:
                continue
            peaks.append(
                Peak(
                    index=int(i),
                    x=float(x[i]),
                    height=float(y[i]),
                    prominence=prom,
                    left_base=int(lb),
                    right_base=int(rb),
                    area=_skim_area(x, y, lb, rb),
                    left_base_t=float(x[lb]),
                    right_base_t=float(x[rb]),
                )
            )
            
        return peaks




# ============================== Alpha diversity ==================================

def alpha_diversity(
    abundances: np.ndarray,
    *,
    small_value: float = 0.0,
) -> Dict[str, float]:
    a = np.asarray(abundances, dtype=float)
    if a.ndim != 1:
        raise ValueError("abundances must be 1D.")
    if np.any(a < 0):
        raise ValueError("abundances must be nonnegative.")
    total = float(np.sum(a))
    S = int(np.count_nonzero(a > 0))
    if total <= 0.0:
        return {
            "richness": 0.0,
            "shannon": 0.0,
            "shannon_effective": 0.0,
            "simpson_D": 0.0,
            "gini_simpson": 0.0,
            "hill_q0": 0.0,
            "hill_q1": 0.0,
            "hill_q2": 0.0,
            "pielou_evenness": 0.0,
        }
    p = (a + small_value) / (total + small_value * a.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    D = float(np.sum(p * p))
    N0 = float(S)
    N1 = float(math.exp(H)) if H > 0 else 0.0
    N2 = float(1.0 / D) if D > 0 else 0.0
    J = float(H / math.log(S)) if S > 1 else 0.0
    return {
        "richness": float(S),
        "shannon": H,
        "shannon_effective": N1,
        "simpson_D": D,
        "gini_simpson": float(1.0 - D),
        "hill_q0": N0,
        "hill_q1": N1,
        "hill_q2": N2,
        "pielou_evenness": J,
    }


# ============================== Convenience pipeline =============================

def build_abundance_matrix(
    samples: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    value: Literal["area", "height"] = "area",
    prominence: float = 0.02,
    prominence_mode: Literal["absolute", "relative"] = "absolute",
    distance: Optional[int] = None,
    width: Optional[int] = None,
    baseline_window_t: Optional[float] = 0.1,
    roi: Tuple[float, float] = (0.0, 1.0),
    edge_smooth_window: int = 11,
) -> Tuple[List[List[Peak]], List[Dict[str, float]], List[float]]:
    """
    Peak-detect each sample and compute alpha diversity directly on peak areas/heights.
    Returns
    -------
    (peaks_all, alpha_all, total_aucs)
    """
    peaks_all: List[List[Peak]] = []
    alpha_all: List[Dict[str, float]] = []
    total_aucs: List[float] = []

    for (x, y) in samples:
        if baseline_window_t is not None and baseline_window_t > 0:
            import scipy.ndimage
            dt = np.mean(np.diff(x))
            if dt > 0:
                w_pts = max(1, int(baseline_window_t / dt))
                baseline = scipy.ndimage.minimum_filter1d(y, size=w_pts)
                y = y - baseline
        y = np.maximum(y, 0.0)

        total_auc = compute_total_auc(x, y, roi=roi)
        total_aucs.append(total_auc)

        peaks = find_electropherogram_peaks(
            x, y,
            prominence=prominence,
            prominence_mode=prominence_mode,
            distance=distance,
            width=width,
            rel_height=1.0,
            roi=roi,
            edge_smooth_window=edge_smooth_window,
        )

        abund = np.array([p.area if value == "area" else p.height for p in peaks], dtype=float)

        peaks_all.append(peaks)
        alpha_all.append(alpha_diversity(abund))

    return peaks_all, alpha_all, total_aucs


# ============================== Panel controller =================================

class DiversityController:
    """
    Computes alpha diversity from aligned electropherograms by:
      1) detecting peaks,
      2) assigning peak areas or heights to a shared pseudotime grid,
      3) computing alpha-diversity metrics on the resulting abundance vectors.

    Public API expected by app.py:
      - set_input(pseudotimes_df, norm_df, rows_are_traces=True)
      - on_updated: Optional[Callable[[pd.DataFrame], None]] to push metrics to Viz tab
    """
    def __init__(self):
        # Working data
        self.pseudotimes_df: Optional[pd.DataFrame] = None
        self.norm_df: Optional[pd.DataFrame] = None
        self.rows_are_traces: bool = False
        self.samples: List[str] = []

        # Standalone Uploader
        self.aligned_file = pn.widgets.FileInput(accept=".csv", multiple=False)
        self.aligned_file.param.watch(self._on_load_aligned_csv, "value")

        # outputs
        self.metrics_df: Optional[pd.DataFrame] = None
        self.on_updated = None  # callback(DataFrame)

        # ---- Controls ----
        self.value_kind  = pn.widgets.Select(name="Peak value", options=["area", "height"], value="area", width=160)
        
        self.prominence = pn.widgets.FloatInput(name="Peak prominence", value=0.05, step=0.005, width=180)
        self.prom_mode  = pn.widgets.Select(name="Prominence mode", options=["absolute", "relative"], value="absolute", width=160)
        self.distance = pn.widgets.IntInput(name="Min peak distance (pts)", value=1, start=1, end=999, width=160)
        self.width = pn.widgets.IntInput(name="Min peak width (pts)", value=1, start=1, end=999, width=160)
        self.baseline_window = pn.widgets.FloatInput(name="Sliding window baseline (t)", value=0.1, step=0.01, width=220)

        self.edge_smooth = pn.widgets.IntInput(name="Edge SG Window (must be odd)", value=11, start=3, end=999, step=2, width=220)

        self.roi_lo = pn.widgets.FloatInput(name="ROI start (t)", value=0.0, step=0.01, width=160)
        self.roi_hi = pn.widgets.FloatInput(name="ROI end (t)", value=1.0, step=0.01, width=160)

        self.asinh_toggle = pn.widgets.Checkbox(name="Use asinh transform for plots", value=True)
        self.compute_btn = pn.widgets.Button(name="Compute alpha diversity", button_type="primary", disabled=True)
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")

        self.table = pn.widgets.Tabulator(pd.DataFrame(), show_index=True, selectable=True, height=380, layout="fit_data_stretch")
        self.plots_column = pn.Column(sizing_mode="stretch_width")

        self.csv_name = pn.widgets.TextInput(name="CSV filename", value="alpha_diversity.csv", width=260)
        self.csv_download = pn.widgets.FileDownload(
            label="Download Metrics CSV", filename=self.csv_name.value, button_type="primary",
            embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True
        )
        self.csv_name.param.watch(lambda e: setattr(self.csv_download, "filename", e.new or "alpha_diversity.csv"), "value")

        self.peaks_csv_name = pn.widgets.TextInput(name="Peaks CSV filename", value="peaks.csv", width=260)
        self.peaks_download = pn.widgets.FileDownload(
            label="Download Peaks CSV", filename=self.peaks_csv_name.value, button_type="primary",
            embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True
        )
        self.peaks_csv_name.param.watch(lambda e: setattr(self.peaks_download, "filename", e.new or "peaks.csv"), "value")

        # ---- Layout ----
        self.section = pn.Column(
            pn.pane.Markdown("## 6) Alpha Diversity"),
            pn.pane.Markdown(
                "_Peaks are detected on each aligned trace and their absolute **areas** (or heights) are "
                "used to compute alpha-diversity metrics for each sample independently._",
                styles={"color": "#555"}
            ),
            pn.pane.Markdown("**Upload Pre-Aligned Data (Optional):**", styles={"color":"#555", "margin-top": "10px"}),
            pn.Row(self.aligned_file),
            pn.Row(self.value_kind),
            pn.Row(self.prominence, pn.Spacer(width=12), self.prom_mode, pn.Spacer(width=12), self.distance, pn.Spacer(width=12), self.width),
            pn.Row(self.edge_smooth, pn.Spacer(width=12), self.baseline_window, pn.Spacer(width=12), self.roi_lo, pn.Spacer(width=12), self.roi_hi),
            pn.Row(self.asinh_toggle, pn.Spacer(width=12), self.compute_btn),
            self.status,
            pn.layout.Divider(),
            self.table,
            pn.Row(self.csv_name, pn.Spacer(width=12), self.csv_download),
            pn.Row(self.peaks_csv_name, pn.Spacer(width=12), self.peaks_download),
            pn.layout.Divider(),
            pn.pane.Markdown("### Detected Peaks"),
            self.plots_column,
            sizing_mode="stretch_width",
            visible=False,
        )

        # ---- Wiring ----
        self.compute_btn.on_click(self._on_compute)
        self.csv_download.callback = self._csv_bytes
        self.peaks_download.callback = self._peaks_csv_bytes

    # -------- External API --------
    def set_input(self, pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool = False):
        """
        Receive aligned data from Alignment tab (or from an import step):
        pseudotimes_df: pseudotime values
        norm_df: aligned intensities (same shape)
        rows_are_traces: True if rows=samples, False if columns=samples
        """
        if not isinstance(pseudotimes_df, pd.DataFrame) or not isinstance(norm_df, pd.DataFrame):
            raise TypeError("pseudotimes_df and norm_df must be pandas DataFrames.")
        if pseudotimes_df.shape != norm_df.shape:
            raise ValueError(f"Shape mismatch: pseudotimes {pseudotimes_df.shape} vs norm_df {norm_df.shape}")

        self.pseudotimes_df = pseudotimes_df.copy()
        self.norm_df = norm_df.copy()
        self.rows_are_traces = bool(rows_are_traces)
        self.samples = list(map(str, (pseudotimes_df.index if rows_are_traces else pseudotimes_df.columns).astype(str)))
        self.compute_btn.disabled = not bool(self.samples)

        # Make the tab usable
        self.section.visible = True
        self.status.object = ok(f"Diversity primed with {len(self.samples)} samples.")
        # clear previous results
        self.table.value = pd.DataFrame()
        self.metrics_df = None
        self.csv_download.disabled = True
        self.plots_column.objects = []
        
    def _on_load_aligned_csv(self, event):
        if not self.aligned_file.value:
            return
        try:
            from nmf_utils import _parse_wide_aligned_csv
            P, Y, samples, t = _parse_wide_aligned_csv(self.aligned_file.filename or "aligned.csv", bytes(self.aligned_file.value))
            self.set_input(P, Y, rows_are_traces=False)
            self.status.object = ok(f"Loaded aligned CSV for {len(samples)} samples. Ready to compute diversity.")
        except Exception as e:
            self.status.object = warn(f"Failed to parse aligned CSV: {e}")

    # -------- Internals --------
    def _gather_samples(self) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Returns (sample_names, list_of_(x,y)) using stored orientation.
        """
        assert self.pseudotimes_df is not None and self.norm_df is not None
        P = self.pseudotimes_df
        Y = self.norm_df

        if self.rows_are_traces:
            # rows=samples, columns=timepoints
            names = [str(i) for i in P.index]
            out: List[Tuple[np.ndarray, np.ndarray]] = []
            for s in P.index:
                x = P.loc[s].to_numpy(dtype=float)
                y = Y.loc[s].to_numpy(dtype=float)
                # ensure strictly increasing x — if not, sort by x
                order = np.argsort(x)
                x = x[order]; y = y[order]
                out.append((x, y))
        else:
            # columns=samples, rows=timepoints
            names = [str(c) for c in P.columns]
            out = []
            for s in P.columns:
                x = P[s].to_numpy(dtype=float)
                y = Y[s].to_numpy(dtype=float)
                order = np.argsort(x)
                x = x[order]; y = y[order]
                out.append((x, y))

        return names, out

    def _on_compute(self, _=None):
        try:
            if self.pseudotimes_df is None or self.norm_df is None:
                self.status.object = warn("Aligned data not set.")
                return

            value = "area" if str(self.value_kind.value) == "area" else "height"

            prom = float(self.prominence.value)
            prom_mode = "relative" if str(self.prom_mode.value) == "relative" else "absolute"
            dist = int(self.distance.value) if int(self.distance.value) > 0 else None
            wid  = int(self.width.value) if int(self.width.value) > 0 else None

            base_w = float(self.baseline_window.value)

            lo = float(self.roi_lo.value); hi = float(self.roi_hi.value)
            if not (0.0 <= lo < hi <= 1.0):
                self.status.object = warn("ROI must satisfy 0.0 ≤ start < end ≤ 1.0.")
                return
            roi = (lo, hi)
            
            e_smooth = int(self.edge_smooth.value)

            use_asinh = self.asinh_toggle.value

            # Prepare samples
            names, raw_XY = self._gather_samples()
            
            # Zero out negative values before peak finding
            XY = []
            for x, y in raw_XY:
                y = np.maximum(y, 0.0)
                XY.append((x, y))

            # Run pipeline
            peaks_all, alpha_all, total_aucs = build_abundance_matrix(
                XY,
                value=value,
                prominence=prom,
                prominence_mode=prom_mode,
                distance=dist,
                width=wid,
                baseline_window_t=base_w,
                roi=roi,
                edge_smooth_window=e_smooth,
            )

            # Compose metrics table
            # alpha_all is list of dicts with keys: richness, shannon, shannon_effective, simpson_D, gini_simpson, hill_q0,q1,q2, pielou_evenness
            key_union = set()
            for d in alpha_all:
                key_union.update(d.keys())
            cols = sorted(key_union)
            rows = [{k: float(d.get(k, np.nan)) for k in cols} for d in alpha_all]
            df = pd.DataFrame(rows, index=names, columns=cols)
            df.index.name = "sample"

            self.metrics_df = df
            self.table.value = df.reset_index()
            self.csv_download.disabled = False
            self.status.object = ok(f"Computed alpha diversity for {df.shape[0]} samples across {df.shape[1]} metrics.")

            # Build peaks DataFrame
            peak_rows = []
            for name, peaks in zip(names, peaks_all):
                total_area = sum(p.area for p in peaks)
                for p in peaks:
                    peak_rows.append({
                        "Sample": name,
                        "Peak_Index": p.index,
                        "Pseudotime": p.x,
                        "Height": p.height,
                        "Prominence": p.prominence,
                        "Left_Base_t": p.left_base_t,
                        "Right_Base_t": p.right_base_t,
                        "Area": p.area,
                        "Relative_Area": p.area / total_area if total_area > 0 else 0.0
                    })
            self.peaks_df = pd.DataFrame(peak_rows)
            self.peaks_download.disabled = False

            # Build stacked plots
            from bokeh.plotting import figure
            from bokeh.models import ColumnDataSource, Band
            from bokeh.palettes import Category10
            
            def _get_palette(n: int) -> List[str]:
                base = list(Category10[10])
                return [base[i % 10] for i in range(n)]

            plots = []
            
            for i, (name, (x, y), peaks) in enumerate(zip(names, XY, peaks_all)):
                y_plot = np.arcsinh(y) if use_asinh else y.copy()
                p = figure(
                    height=200, sizing_mode="stretch_width",
                    x_range=(0.0, 1.0),
                    x_axis_label="pseudotime", 
                    y_axis_label=f"{name} (asinh)" if use_asinh else f"{name} (raw)",
                    tools="pan,wheel_zoom,box_zoom,reset,save"
                )
                p.line(x, y_plot, color="black", line_width=1.5)
                
                # Plot peaks
                n_peaks = len(peaks)
                palette = _get_palette(n_peaks) if n_peaks > 0 else []
                
                for pk_idx, pk in enumerate(peaks):
                    # We need the x and y values from left_base to right_base
                    lb, rb = pk.left_base, pk.right_base
                    if lb >= rb:
                        continue
                        
                    pk_left_x = x[lb]
                    pk_right_x = x[rb]
                    pk_center_x = pk.x
                    
                    color = palette[pk_idx]
                    
                    # Draw whisker at y=0
                    p.segment(x0=pk_left_x, y0=0, x1=pk_right_x, y1=0, color=color, line_width=2)
                    
                    # Highlight the peak center at y=0
                    p.scatter([pk_center_x], [0], color=color, size=6, marker="circle")
                
                p.xgrid.grid_line_color = None
                p.ygrid.grid_line_color = None
                plots.append(p)
                
            self.plots_column.objects = [pn.pane.Bokeh(p) for p in plots]

            # Notify Viz tab to add these metrics for coloring
            if callable(self.on_updated):
                try:
                    self.on_updated(df.copy())
                except Exception:
                    pass

        except Exception as e:
            self.metrics_df = None
            self.table.value = pd.DataFrame()
            self.plots_column.objects = []
            self.csv_download.disabled = True
            self.status.object = warn(f"Computation failed: {e}")

    def _csv_bytes(self):
        df = self.metrics_df
        if df is None or df.empty:
            return io.BytesIO(b"")
        bio = io.BytesIO()
        df.to_csv(bio)
        bio.seek(0)
        return bio

    def _peaks_csv_bytes(self):
        df = getattr(self, "peaks_df", None)
        if df is None or df.empty:
            return io.BytesIO(b"")
        bio = io.BytesIO()
        df.to_csv(bio, index=False)
        bio.seek(0)
        return bio


# Public factory
def build_diversity_section():
    ctrl = DiversityController()
    return ctrl.section, ctrl


