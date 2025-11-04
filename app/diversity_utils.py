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

def _trapz_area(x: np.ndarray, y: np.ndarray, i0: int, i1: int) -> float:
    if i1 <= i0:
        return 0.0
    return float(np.trapz(y[i0:i1 + 1], x[i0:i1 + 1]))

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
) -> List[Peak]:
    x, y = _validate_xy(x, y, require_unit_interval=True)

    lo, hi = max(0.0, float(roi[0])), min(1.0, float(roi[1]))
    if hi <= lo:
        return []
    mask = (x >= lo) & (x <= hi)
    if not np.any(mask):
        return []
    x = x[mask]
    y = y[mask]
    if y.size < 3:
        return []

    prom_thresh = float(prominence)
    if prominence_mode == "relative":
        scale = float(np.ptp(y)) or 1.0
        prom_thresh = max(0.0, prom_thresh * scale)
    else:
        prom_thresh = max(0.0, prom_thresh)

    try:
        from scipy.signal import find_peaks as sp_find_peaks, peak_prominences  # type: ignore

        peak_idx, props = sp_find_peaks(
            y,
            distance=distance,
            prominence=prom_thresh if prom_thresh > 0 else None,
            width=width,
        )
        if peak_idx.size == 0:
            return []

        if "prominences" not in props:
            prominences, left_bases, right_bases = peak_prominences(y, peak_idx, wlen=None, rel_height=rel_height)
        else:
            prominences = props["prominences"]
            left_bases = props["left_bases"]
            right_bases = props["right_bases"]

        peaks: List[Peak] = []
        for pk, prom, lb, rb in zip(peak_idx, prominences, left_bases, right_bases):
            area = _trapz_area(x, y, int(lb), int(rb))
            peaks.append(
                Peak(
                    index=int(pk),
                    x=float(x[pk]),
                    height=float(y[pk]),
                    prominence=float(prom),
                    left_base=int(lb),
                    right_base=int(rb),
                    area=area,
                )
            )
        return [p for p in peaks if p.prominence >= prom_thresh]

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
                    area=_trapz_area(x, y, lb, rb),
                )
            )
        return peaks


# ============================== Mapping to shared grid ===========================

def peaks_to_grid(
    peaks: Sequence[Peak],
    x_grid: np.ndarray,
    *,
    tolerance: float,
    value: Literal["area", "height"] = "area",
    normalization_factor: Optional[float] = None,
) -> np.ndarray:
    """
    Map peaks to grid; if `value=="area"` and `normalization_factor` provided,
    each peak area is divided by that factor (i.e., total sample AUC).
    """
    xg = np.asarray(x_grid, dtype=float)
    if xg.ndim != 1 or xg.size == 0 or not np.all(np.diff(xg) > 0):
        raise ValueError("x_grid must be 1D strictly increasing.")
    if xg.min() < -1e-9 or xg.max() > 1 + 1e-9:
        raise ValueError("x_grid must lie within [0,1].")

    out = np.zeros_like(xg)
    if not peaks:
        return out

    px = np.array([p.x for p in peaks], dtype=float)
    if value == "area":
        pv = np.array([p.area for p in peaks], dtype=float)
        if normalization_factor is not None and normalization_factor > 0:
            pv = pv / float(normalization_factor)
    else:
        pv = np.array([p.height for p in peaks], dtype=float)

    idx = np.searchsorted(xg, px)
    idx = np.clip(idx, 1, xg.size - 1)
    left = xg[idx - 1]
    right = xg[idx]
    nearest = np.where(np.abs(px - left) <= np.abs(px - right), idx - 1, idx)
    mask = np.abs(px - xg[nearest]) <= tolerance
    for k, v, m in zip(nearest, pv, mask):
        if m:
            out[k] += v
    return out


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
    x_grid: np.ndarray,
    tolerance: Optional[float] = None,
    value: Literal["area", "height"] = "area",
    prominence: float = 0.02,
    prominence_mode: Literal["absolute", "relative"] = "absolute",
    distance: Optional[int] = None,
    width: Optional[int] = None,
    roi: Tuple[float, float] = (0.0, 1.0),
    normalize_by_total_auc: bool = True,
) -> Tuple[np.ndarray, List[List[Peak]], List[Dict[str, float]], List[float]]:
    """
    Peak-detect each sample then map to the shared grid in [0,1].
    If `normalize_by_total_auc` and `value=="area"`, each peak area is divided
    by the sample's total AUC over ROI before bin assignment.

    Returns
    -------
    (A, peaks_all, alpha_all, total_aucs)
      A: shape (n_samples, len(x_grid))
      peaks_all: list of per-sample Peak lists
      alpha_all: per-sample alpha metrics computed on A[i]
      total_aucs: total AUC per sample (over ROI)
    """
    xg = np.asarray(x_grid, dtype=float)
    if xg.ndim != 1 or xg.size == 0 or not np.all(np.diff(xg) > 0):
        raise ValueError("x_grid must be 1D strictly increasing.")
    if xg.min() < -1e-9 or xg.max() > 1 + 1e-9:
        raise ValueError("x_grid must lie within [0,1].")
    bin_width = float(np.min(np.diff(xg)))
    tol = (bin_width / 2.0) if tolerance is None else float(tolerance)

    A = []
    peaks_all: List[List[Peak]] = []
    alpha_all: List[Dict[str, float]] = []
    total_aucs: List[float] = []

    for (x, y) in samples:
        total_auc = compute_total_auc(x, y, roi=roi)
        total_aucs.append(total_auc)

        peaks = find_electropherogram_peaks(
            x, y,
            prominence=prominence,
            prominence_mode=prominence_mode,
            distance=distance,
            width=width,
            roi=roi,
        )

        norm_factor = (total_auc if (normalize_by_total_auc and value == "area" and total_auc > 0) else None)

        abund = peaks_to_grid(
            peaks, xg,
            tolerance=tol,
            value=value,
            normalization_factor=norm_factor,
        )

        peaks_all.append(peaks)
        alpha_all.append(alpha_diversity(abund))
        A.append(abund)

    return np.vstack(A), peaks_all, alpha_all, total_aucs


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
        # aligned data
        self.pseudotimes_df: Optional[pd.DataFrame] = None   # points x samples (or samples x points)
        self.norm_df: Optional[pd.DataFrame] = None          # same shape as pseudotimes_df
        self.rows_are_traces: bool = True

        # outputs
        self.metrics_df: Optional[pd.DataFrame] = None
        self.on_updated = None  # callback(DataFrame)

        # ---- Controls ----
        self.grid_points = pn.widgets.IntInput(name="Grid points over [0,1]", value=400, start=20, end=2000, step=10, width=220)
        self.value_kind  = pn.widgets.Select(name="Peak value", options=["area", "height"], value="area", width=160)
        self.normalize_auc = pn.widgets.Checkbox(name="Normalize by total AUC (if value='area')", value=True)

        self.prominence = pn.widgets.FloatInput(name="Peak prominence", value=0.05, step=0.005, width=180)
        self.prom_mode  = pn.widgets.Select(name="Prominence mode", options=["absolute", "relative"], value="relative", width=160)
        self.distance   = pn.widgets.IntInput(name="Min peak distance (points, optional)", value=0, start=0, end=100000, width=240)
        self.width      = pn.widgets.IntInput(name="Min peak width (points, optional)", value=0, start=0, end=100000, width=240)

        self.roi_lo = pn.widgets.FloatInput(name="ROI start (t)", value=0.0, step=0.01, width=160)
        self.roi_hi = pn.widgets.FloatInput(name="ROI end (t)", value=1.0, step=0.01, width=160)

        self.compute_btn = pn.widgets.Button(name="Compute alpha diversity", button_type="primary", disabled=True)
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")

        self.table = pn.widgets.Tabulator(pd.DataFrame(), show_index=True, selectable=True, height=380, layout="fit_data_stretch")
        self.csv_name = pn.widgets.TextInput(name="CSV filename", value="alpha_diversity.csv", width=260)
        self.csv_download = pn.widgets.FileDownload(
            label="Download CSV", filename=self.csv_name.value, button_type="primary",
            embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True
        )
        self.csv_name.param.watch(lambda e: setattr(self.csv_download, "filename", e.new or "alpha_diversity.csv"), "value")

        # ---- Layout ----
        self.section = pn.Column(
            pn.pane.Markdown("## 6) Diversity (alpha)"),
            pn.pane.Markdown(
                "_Peaks are detected on each aligned trace, peak **areas** (or heights) are "
                "assigned to a shared pseudotime grid, then alpha-diversity metrics are computed. "
                "If 'Normalize by total AUC' is on, areas are divided by the total AUC per sample._",
                styles={"color": "#555"}
            ),
            pn.Row(self.grid_points, pn.Spacer(width=16), self.value_kind, pn.Spacer(width=16), self.normalize_auc),
            pn.Row(self.prominence, pn.Spacer(width=12), self.prom_mode, pn.Spacer(width=12), self.distance, pn.Spacer(width=12), self.width),
            pn.Row(self.roi_lo, pn.Spacer(width=12), self.roi_hi),
            pn.Row(self.compute_btn),
            self.status,
            pn.layout.Divider(),
            self.table,
            pn.Row(self.csv_name, pn.Spacer(width=12), self.csv_download),
            sizing_mode="stretch_width",
            visible=False,
        )

        # ---- Wiring ----
        self.compute_btn.on_click(self._on_compute)
        self.csv_download.callback = self._csv_bytes

    # -------- External API --------
    def set_input(self, pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool = True):
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

        # Make the tab usable
        self.section.visible = True
        self.compute_btn.disabled = False
        self.status.object = ok("Aligned data connected. Set parameters and click **Compute alpha diversity**.")
        # clear previous results
        self.table.value = pd.DataFrame()
        self.metrics_df = None
        self.csv_download.disabled = True

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

            # Parameters
            n_grid = int(self.grid_points.value)
            if n_grid < 2:
                self.status.object = warn("Grid points must be at least 2.")
                return
            x_grid = np.linspace(0.0, 1.0, n_grid)

            value = "area" if str(self.value_kind.value) == "area" else "height"
            normalize = bool(self.normalize_auc.value)

            prom = float(self.prominence.value)
            prom_mode = "relative" if str(self.prom_mode.value) == "relative" else "absolute"
            dist = int(self.distance.value) if int(self.distance.value) > 0 else None
            wid  = int(self.width.value) if int(self.width.value) > 0 else None

            lo = float(self.roi_lo.value); hi = float(self.roi_hi.value)
            if not (0.0 <= lo < hi <= 1.0):
                self.status.object = warn("ROI must satisfy 0.0 ≤ start < end ≤ 1.0.")
                return
            roi = (lo, hi)

            # Prepare samples
            names, XY = self._gather_samples()

            # Run pipeline
            A, peaks_all, alpha_all, total_aucs = build_abundance_matrix(
                XY,
                x_grid=x_grid,
                tolerance=None,  # auto = half-bin
                value=value,
                prominence=prom,
                prominence_mode=prom_mode,
                distance=dist,
                width=wid,
                roi=roi,
                normalize_by_total_auc=normalize,
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

            # Notify Viz tab to add these metrics for coloring
            if callable(self.on_updated):
                try:
                    self.on_updated(df.copy())
                except Exception:
                    pass

        except Exception as e:
            self.metrics_df = None
            self.table.value = pd.DataFrame()
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


# Public factory
def build_diversity_section():
    ctrl = DiversityController()
    return ctrl.section, ctrl


