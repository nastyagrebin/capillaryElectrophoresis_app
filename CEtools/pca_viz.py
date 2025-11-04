# file: CEtools/pca_viz.py
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import bokeh.plotting
import bokeh.models
import bokeh.io as bio  # <-- needed if you ever want to call show()

def plot_pca_scree_bokeh(pca, *, height: int = 350, width: int = 500, show: bool = False):
    evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
    pcs = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)

    p = bokeh.plotting.figure(height=height, width=width, x_axis_label="PC",
                              y_axis_label="Explained variance", title="Scree plot",
                              tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                              active_scroll=None)
    p.vbar(x=pcs, top=evr, width=0.8, alpha=0.8, legend_label="PC variance")
    p.extra_y_ranges = {"cum": bokeh.models.Range1d(start=0, end=1.0)}
    p.add_layout(bokeh.models.LinearAxis(y_range_name="cum", axis_label="Cumulative"), "right")
    p.line(pcs, cum, y_range_name="cum", line_width=2, legend_label="Cumulative")
    p.circle(pcs, cum, y_range_name="cum", size=5)
    p.hover.tooltips = [("PC", "@x"), ("var", "@top{0.000}")]
    if p.legend:
        p.legend.location = "top_left"
        p.legend.click_policy = "mute"
    if show:
        bio.show(p)
    return p

def plot_pc_loadings_bar_bokeh(
    pca,
    *,
    pc: int = 1,
    feature_names: Optional[Sequence[str]] = None,
    height: int = 400,
    width: int = 900,
    kind: str = "bar",  # "bar" | "line"
    show: bool = False,
):
    comp_idx = int(pc) - 1
    W = np.asarray(pca.components_, dtype=float)  # (n_pc, n_features)
    if comp_idx < 0 or comp_idx >= W.shape[0]:
        raise ValueError(f"pc must be in [1, {W.shape[0]}]")

    v = W[comp_idx]
    n_features = v.size
    x = np.arange(n_features, dtype=int)
    names = [str(i) for i in range(n_features)] if feature_names is None else [str(n) for n in feature_names]
    if len(names) != n_features:
        raise ValueError("feature_names length must match number of features.")

    df = pd.DataFrame({"x": x, "name": names, "weight": v})
    src = bokeh.models.ColumnDataSource(df)

    p = bokeh.plotting.figure(
        height=height, width=width,
        title=f"PC{pc} loadings (ordered by feature index)",
        x_axis_label="Feature index", y_axis_label="Loading weight",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        active_scroll=None,
    )
    p.add_layout(bokeh.models.Span(location=0, dimension="width", line_color="#666", line_alpha=0.6, line_dash="dashed"))

    if kind == "bar":
        p.vbar(x="x", top="weight", bottom=0, width=0.9, alpha=0.9, source=src)
    elif kind == "line":
        p.line("x", "weight", source=src, line_width=2)
        p.circle("x", "weight", source=src, size=4, alpha=0.8)
    else:
        raise ValueError("kind must be 'bar' or 'line'")

    p.hover.tooltips = [("feature", "@name"), ("index", "@x"), ("loading", "@weight{0.000}")]
    if show:
        bio.show(p)
    return src, p

def plot_pc_as_curve_bokeh(
    pca,
    *,
    pc: int = 1,
    Phi: Optional[callable] = None,
    t_eval: Optional[np.ndarray] = None,
    scaler=None,
    height: int = 350,
    width: int = 600,
    title: Optional[str] = None,
    show: bool = False,
):
    comp_idx = int(pc) - 1
    W = np.asarray(pca.components_, dtype=float)
    if comp_idx < 0 or comp_idx >= W.shape[0]:
        raise ValueError(f"pc must be in [1, {W.shape[0]}]")
    v = W[comp_idx].copy()

    def _get_scale(scaler_or_scale, n_features):
        if scaler_or_scale is None:
            return None
        if hasattr(scaler_or_scale, "scale_"):
            return np.asarray(scaler_or_scale.scale_, dtype=float)
        arr = np.asarray(scaler_or_scale, dtype=float).ravel()
        if arr.size != n_features:
            raise ValueError("scale must have length == n_features")
        return arr

    scale = _get_scale(scaler, W.shape[1])
    if scale is not None:
        v = v * scale

    if Phi is not None:
        t = np.linspace(0.0, 1.0, 1000) if t_eval is None else np.asarray(t_eval, dtype=float).ravel()
        A = Phi(t)
        y = A @ v
        p = bokeh.plotting.figure(height=height, width=width,
                                  x_axis_label="Pseudotime", y_axis_label="Component amplitude",
                                  title=title or f"PC{pc} as curve over pseudotime",
                                  tools="pan,wheel_zoom,box_zoom,reset,save,hover", active_scroll=None)
        p.line(t, y, line_width=2)
        p.hover.tooltips = [("t", "$x{0.000}"), ("amp", "$y{0.000}")]
        if show:
            bio.show(p)
        return t, y, p
    else:
        idx = np.arange(v.size)
        p = bokeh.plotting.figure(height=height, width=width,
                                  x_axis_label="Feature index", y_axis_label="Loading",
                                  title=title or f"PC{pc} loadings by feature",
                                  tools="pan,wheel_zoom,box_zoom,reset,save,hover", active_scroll=None)
        p.line(idx, v, line_width=2)
        p.circle(idx, v, size=4, alpha=0.7)
        p.hover.tooltips = [("i", "$x"), ("loading", "$y{0.000}")]
        if show:
            bio.show(p)
        return idx, v, p
