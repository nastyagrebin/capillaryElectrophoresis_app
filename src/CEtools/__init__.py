# file: CEtools/__init__.py  (Python 3.8-compatible)
"""
CEtools: Electropherogram utilities, pseudotime transforms, NNLS basis fitting,
and Bokeh visualizations. Works on Python 3.8+.

Quick start:
    import CEtools as cet
    cet.enable_notebook()
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union, Optional, Tuple, Callable, List

import numpy as np
import pandas as pd
from .filters import despike_singletons

__all__ = [
    "find_baseline",
    "subtract_baseline",
    "gaussian",
    "find_glc15_peak",
    "normalize_to_glc15",
    "find_max_peak",
    "group_electropherograms_pca",
    "make_pseudotime_transform",
    "pseudotime_transform",
    "pseudotime_linear",
    "default_gaussian_centers",
    "make_gaussian_basis",
    "heuristic_sigma_from_centers",
    "fit_continuous_basis_loadings_from_dataframes",
    "enable_notebook",
    "plot_loadings_heatmap_bokeh",
    "plot_loadings_heatmap_clustered_bokeh",
    "plot_reconstruction_overlays_bokeh",
    "scatter_pca_from_loadings_bokeh",
    "prepare_features",
    "scatter_embed_bokeh",
    "embed_with_mds",
    "embed_with_tsne",

]

ArrayLike = Union[Sequence[float], np.ndarray]

# ---------------------------- Baseline & peaks --------------------------------

def find_baseline(
    trace: ArrayLike,
    window: int = 501,
    ignore_nans: bool = True,
    pad_mode: str = "reflect",
) -> np.ndarray:
    x = np.asarray(trace, dtype=float)
    n = x.size
    if n == 0:
        return x

    w = int(window)
    if w < 1:
        w = 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w % 2 == 0:
        w -= 1
        if w < 1:
            w = 1

    half = w // 2
    xpad = np.pad(x, (half, half), mode=pad_mode)

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(xpad, window_shape=w)
    except Exception:
        windows = np.vstack([xpad[i:i + n] for i in range(0, 2 * half + 1)])

    baseline = np.nanmedian(windows, axis=1) if ignore_nans else np.median(windows, axis=1)
    return baseline


def subtract_baseline(trace: ArrayLike) -> np.ndarray:
    b = find_baseline(trace)
    return np.asarray(trace, dtype=float) - b




def gaussian(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def find_glc15_peak(
    x: np.ndarray,
    y: np.ndarray,
    range_min: float = 18.0,
    range_max: float = 20.0,
) -> Optional[Tuple[float, float, float]]:
    from scipy.optimize import curve_fit

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x >= range_min) & (x <= range_max)

    x_sub = x[mask]
    y_sub = y[mask]
    if x_sub.size < 3:
        return None

    a0 = float(np.nanmax(y_sub))
    mu0 = float(x_sub[np.nanargmax(y_sub)])
    sigma0 = (range_max - range_min) / 6.0

    try:
        popt, _ = curve_fit(
            gaussian,
            x_sub,
            y_sub,
            p0=[a0, mu0, sigma0],
            bounds=([0, range_min, 0], [np.inf, range_max, np.inf]),
        )
        a_fit, mu_fit, sigma_fit = [float(v) for v in popt]
        area = a_fit * sigma_fit * np.sqrt(2.0 * np.pi)
        return mu_fit, area, a_fit
    except Exception:
        return None


def normalize_to_glc15(times: np.ndarray, spectrum: np.ndarray, min_range=17.5, max_range=20.0) -> np.ndarray:
    res = find_glc15_peak(times, spectrum, range_min=min_range, range_max=max_range)
    if not res:
        return np.asarray(spectrum, dtype=float)
    _, auc, _ = res
    if auc == 0 or not np.isfinite(auc):
        return np.asarray(spectrum, dtype=float)
    return np.asarray(spectrum, dtype=float) / float(auc)


def find_max_peak(
    x: np.ndarray,
    y: np.ndarray,
    range_min: float = 17.4,
    range_max: float = 19.0,
) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x >= range_min) & (x <= range_max)
    x_sub = x[mask]
    y_sub = y[mask]
    if x_sub.size == 0:
        return (np.nan, np.nan, np.nan)
    a0 = float(np.nanmax(y_sub))
    mu0 = float(x_sub[np.nanargmax(y_sub)])
    sigma0 = (range_max - range_min) / 6.0
    area = a0 * sigma0 * np.sqrt(2.0 * np.pi)
    return (mu0, area, a0)


def group_electropherograms_pca(
    peaks: List[np.ndarray],
    n_components: int = 5
) -> Tuple[np.ndarray, "PCA"]:
    from sklearn.decomposition import PCA
    matrix = np.vstack(peaks)
    pca = PCA(n_components=n_components)
    projected_data = pca.fit_transform(matrix)
    return projected_data, pca

# ---------------------------- Pseudotime maps ---------------------------------

def _endpoint_slope(x0, x1, x2, y0, y1, y2) -> float:
    h0 = x1 - x0
    h1 = x2 - x1
    d0 = (y1 - y0) / h0
    d1 = (y2 - y1) / h1
    m = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(m) != np.sign(d0):
        m = 0.0
    elif (np.sign(d0) != np.sign(d1)) and (abs(m) > abs(3 * d0)):
        m = 3 * d0
    return float(m)


def _fritsch_carlson_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 points for PCHIP tangents.")
    d = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = np.zeros(n, dtype=float)
    m[0] = _endpoint_slope(x[0], x[1], x[2], y[0], y[1], y[2])
    m[-1] = _endpoint_slope(x[-1], x[-2], x[-3], y[-1], y[-2], y[-3])
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * (x[i + 1] - x[i]) + (x[i] - x[i - 1])
            w2 = (x[i + 1] - x[i]) + 2 * (x[i] - x[i - 1])
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    for i in range(n - 1):
        if y[i] == y[i + 1]:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            a = m[i] / d[i]
            b = m[i + 1] / d[i]
            if (a * a + b * b) > 9.0:
                tau = 3.0 / np.sqrt(a * a + b * b)
                m[i] = tau * a * d[i]
                m[i + 1] = tau * b * d[i]
    return m


def _pchip_eval(x: np.ndarray, y: np.ndarray, m: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = x.astype(float); y = y.astype(float); m = m.astype(float)
    t = np.asarray(t, dtype=float)
    idx = np.searchsorted(x, t, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    x0 = x[idx]; x1 = x[idx + 1]
    y0 = y[idx]; y1 = y[idx + 1]
    m0 = m[idx]; m1 = m[idx + 1]
    h = x1 - x0
    s = (t - x0) / h
    h00 = (2 * s**3) - (3 * s**2) + 1
    h10 = (s**3) - (2 * s**2) + s
    h01 = (-2 * s**3) + (3 * s**2)
    h11 = (s**3) - (s**2)
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def _validate_increasing(name: str, arr: np.ndarray) -> None:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D.")
    if len(arr) < 3:
        raise ValueError(f"{name} must have at least 3 points.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite floats.")
    if not np.all(arr[1:] > arr[:-1]):
        raise ValueError(f"{name} must be strictly increasing.")


def make_pseudotime_transform(
    anchors: Iterable[float],
    *,
    targets: Iterable[float] = (0.0, 0.1, 0.4, 0.6, 1.0),
    clamp: bool = True,
) -> Callable[[ArrayLike], np.ndarray]:
    x = np.asarray(list(anchors), dtype=float)
    y = np.asarray(list(targets), dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("anchors and targets must have the same length.")
    _validate_increasing("anchors", x)
    _validate_increasing("targets", y)
    m = _fritsch_carlson_slopes(x, y)
    y_min, y_max = float(y[0]), float(y[-1])
    d_left = (y[1] - y[0]) / (x[1] - x[0])
    d_right = (y[-1] - y[-2]) / (x[-1] - x[-2])

    def f(t_true: ArrayLike) -> np.ndarray:
        t_arr = np.asarray(t_true, dtype=float)
        out = _pchip_eval(x, y, m, t_arr)
        left_mask = t_arr < x[0]
        right_mask = t_arr > x[-1]
        if np.any(left_mask):
            out[left_mask] = y[0] + d_left * (t_arr[left_mask] - x[0])
        if np.any(right_mask):
            out[right_mask] = y[-1] + d_right * (t_arr[right_mask] - x[-1])
        if clamp:
            out = np.clip(out, y_min, y_max)
        return out

    return f


def pseudotime_transform(
    true_time: ArrayLike,
    anchors: Iterable[float],
    *,
    targets: Iterable[float] = (0.0, 0.1, 0.4, 0.6, 1.0),
    clamp: bool = True,
) -> np.ndarray:
    f = make_pseudotime_transform(anchors, targets=targets, clamp=clamp)
    return f(true_time)


def pseudotime_linear(
    true_time: ArrayLike,
    anchors: Iterable[float],
    *,
    targets: Iterable[float] = (0.0, 0.1, 0.4, 0.6, 1.0),
    clamp: bool = True,
) -> np.ndarray:
    x = np.asarray(list(anchors), dtype=float)
    y = np.asarray(list(targets), dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("anchors and targets must have the same length.")
    _validate_increasing("anchors", x)
    _validate_increasing("targets", y)

    t = np.asarray(true_time, dtype=float)
    out = np.interp(t, x, y, left=np.nan, right=np.nan)
    left_mask = t < x[0]
    right_mask = t > x[-1]
    if np.any(left_mask):
        d_left = (y[1] - y[0]) / (x[1] - x[0])
        out[left_mask] = y[0] + d_left * (t[left_mask] - x[0])
    if np.any(right_mask):
        d_right = (y[-1] - y[-2]) / (x[-1] - x[-2])
        out[right_mask] = y[-1] + d_right * (t[right_mask] - x[-1])
    if clamp:
        out = np.clip(out, float(y[0]), float(y[-1]))
    return out

# ---------------- Continuous Gaussian basis + NNLS fitting --------------------

def default_gaussian_centers(K: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, max(int(K), 1))


def make_gaussian_basis(centers: np.ndarray, sigma: Union[float, np.ndarray]):
    c = np.asarray(centers, dtype=float).ravel()
    s = np.asarray(sigma, dtype=float)
    if s.ndim == 0:
        s = np.full_like(c, float(s))
    if not (c.ndim == 1 and s.shape == c.shape and np.all(s > 0)):
        raise ValueError("Invalid centers/sigma.")

    def Phi(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).ravel()
        return np.exp(-0.5 * ((t[:, None] - c[None, :]) / s[None, :]) ** 2)

    return Phi


def heuristic_sigma_from_centers(centers: np.ndarray) -> np.ndarray:
    c = np.asarray(centers, dtype=float).ravel()
    d = np.diff(np.r_[c[0], c, c[-1]])
    s = 0.5 * (d[:-1] + d[1:])
    return np.maximum(0.5 * s, 1e-4)


def _try_scipy_nnls(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    try:
        from scipy.optimize import nnls
        x, _ = nnls(A, b)
        return x
    except Exception:
        return None


def nnls_projected_grad(
    A: np.ndarray,
    b: np.ndarray,
    l2: float = 0.0,
    max_iter: int = 300,
    tol: float = 1e-6
) -> np.ndarray:
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float).ravel()
    K_ = A.shape[1]
    G = A.T @ A
    c = A.T @ b
    L = float(np.max(np.linalg.eigvalsh(G))) if K_ > 1 else float(G[0, 0])
    step = 1.0 / (L + l2 + 1e-12)
    h = np.zeros(K_, dtype=float)
    for _ in range(max_iter):
        grad = G @ h - c + l2 * h
        h_new = h - step * grad
        h_new[h_new < 0.0] = 0.0
        if np.linalg.norm(h_new - h) <= tol * (np.linalg.norm(h) + 1e-12):
            h = h_new
            break
        h = h_new
    return h


def fit_continuous_basis_loadings_from_dataframes(
    pseudotimes_df: pd.DataFrame,
    norm_df: pd.DataFrame,
    *,
    K: int = 16,
    centers: Optional[np.ndarray] = None,
    sigma: Optional[Union[float, np.ndarray]] = None,
    l2: float = 1e-3,
    use_scipy: bool = True,
    rows_are_traces: bool = True,
    mask_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[pd.DataFrame, Callable[[np.ndarray], np.ndarray], np.ndarray, pd.DataFrame]:
    P = pseudotimes_df.values if rows_are_traces else pseudotimes_df.values.T
    Y = norm_df.values if rows_are_traces else norm_df.values.T
    if P.shape != Y.shape:
        raise ValueError(f"Shape mismatch: pseudotimes {P.shape} vs norm_df {Y.shape}")
    n_traces, _ = P.shape

    if centers is None:
        centers = default_gaussian_centers(K)
    if sigma is None:
        sigma = heuristic_sigma_from_centers(centers)
    Phi = make_gaussian_basis(centers, sigma)

    lo, hi = mask_range
    H = np.zeros((n_traces, K), dtype=float)
    pseudo_used = np.full_like(P, np.nan, dtype=float)

    for i in range(n_traces):
        t = P[i].astype(float)
        y = Y[i].astype(float)
        m = (t >= lo) & (t <= hi) & np.isfinite(t) & np.isfinite(y)
        if not np.any(m):
            continue
        A = Phi(t[m])
        h = _try_scipy_nnls(A, y[m]) if use_scipy else None
        if h is None:
            h = nnls_projected_grad(A, y[m], l2=float(l2))
        H[i, :] = h
        pseudo_used[i, m] = t[m]

    trace_idx = pseudotimes_df.index if rows_are_traces else pseudotimes_df.columns
    H_cols = [f"basis_{k:02d}" for k in range(K)]
    H_df = pd.DataFrame(H, index=trace_idx, columns=H_cols)

    if rows_are_traces:
        pseudo_used_df = pd.DataFrame(
            pseudo_used, index=pseudotimes_df.index, columns=pseudotimes_df.columns
        )
    else:
        pseudo_used_df = pd.DataFrame(
            pseudo_used.T, index=pseudotimes_df.index, columns=pseudotimes_df.columns
        )

    return H_df, Phi, centers, pseudo_used_df

# --------------------------------- Plotting -----------------------------------

def enable_notebook(bokeh_backend: bool = True, holoviews_backend: bool = True) -> None:
    if bokeh_backend:
        import bokeh.io as _bio
        _bio.output_notebook()
    if holoviews_backend:
        import holoviews as _hv
        _hv.extension("bokeh")


def plot_loadings_heatmap_bokeh(H_df: pd.DataFrame) -> pd.DataFrame:
    import bokeh.models
    import bokeh.plotting
    import bokeh.palettes
    from sklearn.preprocessing import StandardScaler

    Z = pd.DataFrame(
        StandardScaler(with_mean=True, with_std=True).fit_transform(H_df.values),
        index=H_df.index, columns=H_df.columns
    )

    samples = list(Z.index.astype(str))
    comps = list(Z.columns.astype(str))
    n_rows, n_cols = Z.shape

    xs, ys, vals, s_labels, c_labels = [], [], [], [], []
    for i, s in enumerate(samples):
        row_vals = Z.iloc[i].values
        for j, c in enumerate(comps):
            xs.append(j); ys.append(i); vals.append(float(row_vals[j]))
            s_labels.append(s); c_labels.append(c)

    mapper = bokeh.models.LinearColorMapper(
        palette=bokeh.palettes.Viridis256,
        low=float(np.nanmin(vals)),
        high=float(np.nanmax(vals))
    )
    src = bokeh.models.ColumnDataSource(
        dict(x=xs, y=ys, val=vals, sample=s_labels, comp=c_labels)
    )

    width = max(500, min(3 * n_cols, 1400))
    height = max(300, min(24 * n_rows, 1000))
    p = bokeh.plotting.figure(
        height=height, width=width,
        title="Loadings heatmap (z-scored per component)",
        tools="hover,pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom",
        x_range=(-0.5, n_cols - 0.5), y_range=(-0.5, n_rows - 0.5)
    )

    p.rect(x="x", y="y", width=1, height=1, source=src,
           fill_color={"field": "val", "transform": mapper}, line_color=None)

    p.yaxis.ticker = bokeh.models.FixedTicker(ticks=list(range(n_rows)))
    p.yaxis.major_label_overrides = {i: samples[i] for i in range(n_rows)}

    color_bar = bokeh.models.ColorBar(
        title="loading weight",
        color_mapper=mapper,
        ticker=bokeh.models.FixedTicker(),
        formatter=bokeh.models.PrintfTickFormatter(format="%.2f"),
        label_standoff=8, location=(0, 0),
    )
    p.add_layout(color_bar, "right")
    p.hover.tooltips = [("sample", "@sample"), ("component", "@comp"), ("z", "@val{0.00}")]

    bokeh.plotting.show(p)
    return Z


def plot_loadings_heatmap_clustered_bokeh(
    H_df: pd.DataFrame,
    *,
    zscore_cols: bool = True,
    metric_rows: str = "cosine",
    method_rows: str = "average",
    cluster_cols: bool = False,
    metric_cols: str = "cosine",
    method_cols: str = "average",
    title: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[int], List[int]]:
    import bokeh.plotting
    from bokeh.models import (ColumnDataSource, LinearColorMapper, ColorBar,
                              FixedTicker, PrintfTickFormatter)
    from bokeh.palettes import Viridis256
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances

    def _zscore_cols(df: pd.DataFrame) -> pd.DataFrame:
        Z = StandardScaler(with_mean=True, with_std=True).fit_transform(df.values)
        return pd.DataFrame(Z, index=df.index, columns=df.columns)

    def _order_by_hclust(X: np.ndarray, metric: str = "cosine", method: str = "average") -> List[int]:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            D = pairwise_distances(X, metric=metric)
            Z = linkage(squareform(D, checks=False), method=method, optimal_ordering=True)
            order = leaves_list(Z).tolist()
            return order
        except Exception:
            D = pairwise_distances(X, metric=metric)
            n = D.shape[0]
            if n == 0:
                return []
            used = np.zeros(n, dtype=bool)
            order = [int(np.argmin(D.sum(axis=1)))]
            used[order[0]] = True
            for _ in range(1, n):
                last = order[-1]
                d = D[last].copy()
                d[used] = np.inf
                nxt = int(np.argmin(d))
                order.append(nxt)
                used[nxt] = True
            return order

    if not isinstance(H_df, pd.DataFrame):
        raise TypeError("H_df must be a pandas DataFrame.")

    Z_df = _zscore_cols(H_df) if zscore_cols else H_df.copy()
    row_order = _order_by_hclust(Z_df.values, metric=metric_rows, method=method_rows)
    col_order = (
        _order_by_hclust(Z_df.values.T, metric=metric_cols, method=method_cols)
        if cluster_cols else list(range(Z_df.shape[1]))
    )
    Zr = Z_df.iloc[row_order, :].iloc[:, col_order]

    samples = Zr.index.astype(str).tolist()
    comps = Zr.columns.astype(str).tolist()
    n_rows, n_cols = Zr.shape

    xs, ys, vals, s_labels, c_labels = [], [], [], [], []
    for i, s in enumerate(samples):
        row_vals = Zr.iloc[i].values
        for j, c in enumerate(comps):
            xs.append(j); ys.append(i); vals.append(float(row_vals[j]))
            s_labels.append(s); c_labels.append(c)

    vmin = float(np.nanmin(vals)) if len(vals) else 0.0
    vmax = float(np.nanmax(vals)) if len(vals) else 1.0
    if vmin == vmax:
        vmin -= 1e-12; vmax += 1e-12

    mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)
    src = ColumnDataSource(dict(x=xs, y=ys, val=vals, sample=s_labels, comp=c_labels))

    width = max(500, min(3 * n_cols, 1400))
    height = max(300, min(24 * n_rows, 1000))
    p = bokeh.plotting.figure(
        height=height, width=width,
        title=title or f"Loadings heatmap (clustered rows{' & cols' if cluster_cols else ''})",
        tools="hover,pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom",
        x_range=(-0.5, n_cols - 0.5), y_range=(-0.5, n_rows - 0.5),
    )
    p.rect(x="x", y="y", width=1, height=1, source=src,
           fill_color={"field": "val", "transform": mapper}, line_color=None)

    p.yaxis.ticker = FixedTicker(ticks=list(range(n_rows)))
    p.yaxis.major_label_overrides = {i: samples[i] for i in range(n_rows)}

    color_bar = ColorBar(color_mapper=mapper, ticker=FixedTicker(),
                         formatter=PrintfTickFormatter(format="%.2f"),
                         label_standoff=8, location=(0, 0))
    p.add_layout(color_bar, "right")
    p.hover.tooltips = [("sample", "@sample"), ("component", "@comp"), ("z", "@val{0.00}")]

    # bokeh.plotting.show(p)
    return Zr, row_order, col_order, p


def plot_reconstruction_overlays_bokeh(
    sample_name: str,
    H_df: pd.DataFrame,
    Phi: Callable[[np.ndarray], np.ndarray],
    pseudotimes_df: Optional[pd.DataFrame] = None,
    norm_df: Optional[pd.DataFrame] = None,
    rows_are_traces: bool = True,
    n_eval: int = 1000,
    title_prefix: str = "Sample",
):
    import bokeh.plotting

    if sample_name not in H_df.index:
        raise KeyError(f"sample '{sample_name}' not found in H_df.index")

    t_eval = np.linspace(0.0, 1.0, int(n_eval))
    A_eval = Phi(t_eval)

    i = int(H_df.index.get_loc(sample_name))
    h = H_df.iloc[i].values
    yhat = A_eval @ h

    p = bokeh.plotting.figure(
        height=400, width=600,
        title=f"{title_prefix} {sample_name}: reconstruction",
        x_axis_label="Pseudotime", y_axis_label="Intensity",
        x_range=[0.0, 1.0],
    )

    if pseudotimes_df is not None and norm_df is not None:
        if rows_are_traces:
            x_obs = pseudotimes_df.loc[sample_name].values
            y_obs = norm_df.loc[sample_name].values
        else:
            x_obs = pseudotimes_df[sample_name].values
            y_obs = norm_df[sample_name].values
        p.line(x=x_obs, y=y_obs, line_width=2, line_color="red", legend_label="original")

    p.line(x=t_eval, y=yhat, line_width=2, line_color="navy", legend_label="reconstruction")
    p.legend.click_policy = "hide"
    return p

# Add this to CEtools/__all__: "scatter_pca_from_loadings_bokeh",

from typing import Optional, Sequence, Tuple

def scatter_pca_from_loadings_bokeh(
    H_df: pd.DataFrame,
    *,
    labels: Optional[Sequence[str]] = None,
    n_components: int = 5,
    normalize: Optional[str] = "l1",
    metadata: Optional[dict] = None,
    metadata_categorical: bool = True,
    pc_axes: Tuple[int, int] = (1, 2),
):
    """
    PCA scatter of NMF loadings with optional coloring by metadata.

    Parameters
    ----------
    H_df : (samples x components) DataFrame
    labels : optional sequence of point labels (defaults to H_df.index as str)
    n_components : PCA components to compute (may be auto-bumped by pc_axes)
    normalize : {"l1","l2",None} row-normalization before z-scoring
    metadata : dict mapping sample name -> category (str) or value (float)
    metadata_categorical : if True, treat metadata as categorical, else continuous
    pc_axes : tuple[int,int], 1-based principal components to plot (default: (1,2))

    Returns
    -------
    scores : np.ndarray (n_samples x n_components_effective)
    components_ : np.ndarray
    pca : fitted sklearn PCA
    """
    import bokeh.io
    import bokeh.models
    import bokeh.plotting
    from bokeh.palettes import Category10, Category20, Viridis256
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # --- validate/select PCs (1-based -> 0-based) ---
    if not (isinstance(pc_axes, tuple) and len(pc_axes) == 2 and all(isinstance(i, int) for i in pc_axes)):
        raise ValueError("pc_axes must be a tuple of two integers, e.g., (1, 2).")
    if pc_axes[0] == pc_axes[1]:
        raise ValueError("pc_axes must contain two distinct component numbers.")
    if min(pc_axes) < 1:
        raise ValueError("pc_axes are 1-based and must be >= 1.")
    pc_x_1b, pc_y_1b = pc_axes
    pc_x, pc_y = pc_x_1b - 1, pc_y_1b - 1
    n_components_eff = max(int(n_components), pc_x_1b, pc_y_1b)

    # --- prepare matrix ---
    X = H_df.values.copy()
    if normalize == "l1":
        s = np.sum(X, axis=1, keepdims=True) + 1e-12  # why: avoid div-by-zero
        X = X / s
    elif normalize == "l2":
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / n
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca = PCA(n_components=n_components_eff, random_state=0)
    scores = pca.fit_transform(X)

    # labels
    labs = list(labels) if labels is not None else list(H_df.index.astype(str))

    # align metadata
    idx_names = list(H_df.index.astype(str))
    if metadata is None:
        meta_vals = [None] * len(idx_names)
    else:
        def _lookup(name):
            if name in metadata:
                return metadata[name]
            try:
                orig = H_df.index[list(map(str, H_df.index)).index(name)]
                return metadata.get(orig, None)
            except Exception:
                return None
        meta_vals = [_lookup(n) for n in idx_names]

    # coloring
    color_field = "color"
    source_data = dict(
        x=scores[:, pc_x],
        y=scores[:, pc_y],
        label=[str(l) for l in labs],
    )

    title_suffix = ""
    mapper = None
    if metadata is None:
        source_data[color_field] = ["#1f77b4"] * len(idx_names)
    else:
        if metadata_categorical:
            cats = [str(v) if v is not None else "NA" for v in meta_vals]
            unique = sorted(set(cats))
            if len(unique) <= 10:
                palette = list(Category10[10])
            elif len(unique) <= 20:
                palette = list(Category20[20])
            else:
                idxs = np.linspace(0, 255, num=len(unique), dtype=int)
                palette = [Viridis256[i] for i in idxs]
            cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
            source_data[color_field] = [cmap[c] for c in cats]
            source_data["category"] = cats
            title_suffix = " — colored by category"
        else:
            vals = []
            for v in meta_vals:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
            vals = np.asarray(vals, dtype=float)
            vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
            vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0
            mapper = bokeh.models.LinearColorMapper(palette=Viridis256, low=vmin, high=vmax, nan_color="lightgray")
            source_data["meta_value"] = vals
            title_suffix = " — colored by value"

    src = bokeh.models.ColumnDataSource(source_data)

    # figure
    xvar = pca.explained_variance_ratio_[pc_x] * 100.0
    yvar = pca.explained_variance_ratio_[pc_y] * 100.0
    p = bokeh.plotting.figure(
        height=420,
        width=520,
        title=f"PCA of NMF loadings (PC{pc_x_1b} vs PC{pc_y_1b}){title_suffix}",
        x_axis_label=f"PC{pc_x_1b} {xvar:.1f}%",
        y_axis_label=f"PC{pc_y_1b} {yvar:.1f}%",
        tools="hover,pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )

    if metadata is None:
        p.circle("x", "y", size=8, alpha=0.85, fill_color=color_field, line_color=None, source=src)
    else:
        if metadata_categorical:
            p.circle("x", "y", size=8, alpha=0.85, fill_color=color_field, line_color=None,
                     source=src, legend_field="category")
            p.legend.click_policy = "hide"
            p.legend.location = "top_right"
        else:
            p.circle(
                "x", "y", size=8, alpha=0.85,
                fill_color={"field": "meta_value", "transform": mapper},
                line_color=None, source=src,
            )
            color_bar = bokeh.models.ColorBar(
                color_mapper=mapper,
                label_standoff=8,
                location=(0, 0),
                ticker=bokeh.models.BasicTicker(),
                formatter=bokeh.models.PrintfTickFormatter(format="%.3f"),
            )
            p.add_layout(color_bar, "right")

    p.hover.tooltips = [("sample", "@label"),
                        (f"PC{pc_x_1b}", "@x{0.000}"),
                        (f"PC{pc_y_1b}", "@y{0.000}")]

    bokeh.io.show(p)
    return scores, pca.components_, pca

from typing import Dict

def prepare_features(
    H_df: pd.DataFrame,
    *,
    row_norm: str = "l1",
    zscore_cols: bool = True
) -> np.ndarray:
    """
    Prepare (samples x components) loadings for embedding.
    - row_norm: "l1" | "l2" | "none"
    - zscore_cols: if True, z-score columns after row normalization
    """
    from sklearn.preprocessing import StandardScaler, normalize as _normalize

    X = H_df.values.astype(float)
    if row_norm == "l1":
        X = _normalize(X, norm="l1")
    elif row_norm == "l2":
        X = _normalize(X, norm="l2")
    elif row_norm in (None, "none"):
        pass
    else:
        raise ValueError("row_norm must be 'l1', 'l2', or 'none'.")

    if zscore_cols:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    return X


def scatter_embed_bokeh(
    X2: np.ndarray,
    labels: List[str],
    title: str,
    *,
    metadata: Optional[Dict] = None,
    metadata_categorical: bool = True,
):
    """
    2D scatter (Bokeh) with optional metadata-driven coloring.

    - metadata=None  → uniform color.
    - metadata_categorical=True  → unique color per category + legend.
    - metadata_categorical=False → continuous colorbar.

    Returns the Bokeh figure.
    """
    import bokeh.io
    import bokeh.models
    import bokeh.plotting
    from bokeh.palettes import Category10, Category20, Viridis256

    labels = [str(l) for l in labels]
    x, y = X2[:, 0], X2[:, 1]

    src_dict = {"x": x, "y": y, "label": labels}
    title_suffix = ""
    mapper = None

    if metadata is None:
        # Single color
        src_dict["color"] = ["#1f77b4"] * len(labels)
        color_spec = "color"
    else:
        # Align metadata to labels (support original index types or strings)
        aligned = []
        for name in labels:
            if name in metadata:
                aligned.append(metadata[name])
            else:
                aligned.append(metadata.get(name, None))

        if metadata_categorical:
            cats = [str(v) if v is not None else "NA" for v in aligned]
            unique = sorted(set(cats))
            if len(unique) <= 10:
                palette = list(Category10[10])
            elif len(unique) <= 20:
                palette = list(Category20[20])
            else:
                idxs = np.linspace(0, 255, num=len(unique), dtype=int)
                palette = [Viridis256[i] for i in idxs]
            cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
            colors = [cmap[c] for c in cats]
            src_dict["category"] = cats
            src_dict["color"] = colors
            color_spec = "color"
            title_suffix = " — colored by category"
        else:
            vals = []
            for v in aligned:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
            vals = np.asarray(vals, dtype=float)
            vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
            vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0
            mapper = bokeh.models.LinearColorMapper(palette=Viridis256, low=vmin, high=vmax, nan_color="lightgray")
            src_dict["value"] = vals
            color_spec = {"field": "value", "transform": mapper}
            title_suffix = " — colored by value"

    src = bokeh.models.ColumnDataSource(src_dict)

    p = bokeh.plotting.figure(
        height=420, width=520,
        title=f"{title}{title_suffix}",
        tools="hover,pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )

    if metadata is None:
        p.circle("x", "y", size=9, alpha=0.9, line_color=None, fill_color=color_spec, source=src)
    else:
        if metadata_categorical:
            p.circle("x", "y", size=9, alpha=0.9, line_color=None,
                     fill_color=color_spec, source=src, legend_field="category")
            p.legend.click_policy = "hide"
            p.legend.location = "top_right"
        else:
            p.circle("x", "y", size=9, alpha=0.9, line_color=None,
                     fill_color=color_spec, source=src)
            color_bar = bokeh.models.ColorBar(
                color_mapper=mapper,
                label_standoff=8,
                location=(0, 0),
                ticker=bokeh.models.BasicTicker(),
                formatter=bokeh.models.PrintfTickFormatter(format="%.3f"),
            )
            p.add_layout(color_bar, "right")

    p.add_tools(bokeh.models.HoverTool(tooltips=[
        ("sample", "@label"), ("x", "@x{0.000}"), ("y", "@y{0.000}")
    ]))

    bokeh.io.show(p)
    return p


def embed_with_mds(
    H_df: pd.DataFrame,
    *,
    metric: str = "cosine",
    row_norm: str = "l1",
    zscore_cols: bool = True,
    random_state: int = 0,
    metadata: Optional[Dict] = None,
    metadata_categorical: bool = True,
    title: Optional[str] = None,
):
    """
    2D MDS embedding on pairwise distances of prepared features.
    Returns (X2, mds, fig).
    """
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    X = prepare_features(H_df, row_norm=row_norm, zscore_cols=zscore_cols)
    D = pairwise_distances(X, metric=metric)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=int(random_state),
        n_init=10,
        max_iter=1000,
    )
    X2 = mds.fit_transform(D)
    fig = scatter_embed_bokeh(
        X2,
        labels=H_df.index.astype(str).tolist(),
        title=title or f"MDS (metric={metric})",
        metadata=metadata,
        metadata_categorical=metadata_categorical,
    )
    return X2, mds, fig


def embed_with_tsne(
    H_df: pd.DataFrame,
    *,
    perplexity: float = 5.0,
    metric: str = "cosine",
    row_norm: str = "l1",
    zscore_cols: bool = True,
    random_state: int = 0,
    n_iter: int = 2000,
    metadata: Optional[Dict] = None,
    metadata_categorical: bool = True,
    title: Optional[str] = None,
):
    """
    2D t-SNE embedding of prepared features.
    - If metric != 'euclidean', distances are precomputed and t-SNE is run with metric='precomputed'.
    - Validates perplexity < n_samples.
    Returns (X2, tsne, fig).
    """
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances

    X = prepare_features(H_df, row_norm=row_norm, zscore_cols=zscore_cols)
    n = X.shape[0]
    if perplexity >= n:
        raise ValueError(f"perplexity ({perplexity}) must be < n_samples ({n}).")

    if metric.lower() == "euclidean":
        tsne = TSNE(
            n_components=2,
            perplexity=float(perplexity),
            learning_rate="auto",
            init="pca",
            metric="euclidean",
            random_state=int(random_state),
        )
        X2 = tsne.fit_transform(X)
    else:
        D = pairwise_distances(X, metric=metric)
        np.fill_diagonal(D, 0.0)
        tsne = TSNE(
            n_components=2,
            perplexity=float(perplexity),
            learning_rate="auto",
            init="random",     # required for precomputed distances
            metric="precomputed",
            random_state=int(random_state),
        )
        X2 = tsne.fit_transform(D)

    fig = scatter_embed_bokeh(
        X2,
        labels=H_df.index.astype(str).tolist(),
        title=title or f"t-SNE (metric={metric}, perplexity={perplexity})",
        metadata=metadata,
        metadata_categorical=metadata_categorical,
    )
    return X2, tsne, fig

try:
    __all__.append("despike_singletons")
except NameError:
    __all__ = ["despike_singletons"]


# expose the applet via a lazy loader to keep Panel optional
__all__.append("run_anchor_alignment_applet") if "run_anchor_alignment_applet" not in __all__ else None

def run_anchor_alignment_applet(*args, **kwargs):
    """
    Lazy wrapper to avoid importing Panel unless needed.
    """
    from .interactive import run_anchor_alignment_applet as _run  # local import
    return _run(*args, **kwargs)
