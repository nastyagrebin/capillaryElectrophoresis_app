import io
import os
import re
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Sequence

import numpy as np
import pandas as pd
import panel as pn
import param
import bokeh
import bokeh.plotting as bkp
import bokeh.models
from bokeh.embed import components
from bokeh.resources import CDN, INLINE
from scipy.stats import kruskal, linregress

import common_plot
from common_plot import TitledPlotPane, apply_export_prefix_to_pane
try:
    from version import APP_VERSION
except Exception:
    APP_VERSION = "v2.2.62"


# ---------------- UI Alerts (No Emojis) ----------------
def _ok(msg: str) -> str:
    return f"<div style='color: #155724; background-color: #d4edda; border: 1px solid #c3e6cb; padding: 8px 12px; border-radius: 4px; margin: 6px 0; font-weight: 500;'>[OK] {msg}</div>"

def _warn(msg: str) -> str:
    return f"<div style='color: #856404; background-color: #fff3cd; border: 1px solid #ffeeba; padding: 8px 12px; border-radius: 4px; margin: 6px 0; font-weight: 500;'>[Warning] {msg}</div>"

def _error(msg: str) -> str:
    return f"<div style='color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 8px 12px; border-radius: 4px; margin: 6px 0; font-weight: 500;'>[Error] {msg}</div>"

# ---------------- Helper Utilities ----------------
def _coerce_sid(x: pd.Series) -> pd.Series:
    def _clean(val):
        if pd.isna(val): return np.nan
        s = str(val).strip()
        if s.endswith('.0'): s = s[:-2]
        return s
    return x.apply(_clean)

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out

def extract_bokeh_figure(obj: Any) -> Optional[bokeh.models.LayoutDOM]:
    """Recursively extract a Bokeh figure or LayoutDOM from obj or Panel panes."""
    if obj is None:
        return None
    if isinstance(obj, bokeh.models.LayoutDOM):
        return obj
    if hasattr(obj, "bokeh_pane") and getattr(obj.bokeh_pane, "object", None) is not None:
        return extract_bokeh_figure(obj.bokeh_pane.object)
    if hasattr(obj, "object") and getattr(obj, "object", None) is not None:
        return extract_bokeh_figure(obj.object)
    if hasattr(obj, "objects") and getattr(obj, "objects", None):
        for child in obj.objects:
            fig = extract_bokeh_figure(child)
            if fig is not None:
                return fig
    return None

def find_all_bokeh_figures(obj: Any, max_depth: int = 5) -> List[bokeh.models.LayoutDOM]:
    """Recursively find all Bokeh LayoutDOM instances in a container or pane."""
    figures: List[bokeh.models.LayoutDOM] = []
    if obj is None or max_depth <= 0:
        return figures
    if isinstance(obj, bokeh.models.LayoutDOM):
        figures.append(obj)
        return figures
    if hasattr(obj, "bokeh_pane") and getattr(obj.bokeh_pane, "object", None) is not None:
        figures.extend(find_all_bokeh_figures(obj.bokeh_pane.object, max_depth - 1))
    elif hasattr(obj, "object") and getattr(obj, "object", None) is not None:
        figures.extend(find_all_bokeh_figures(obj.object, max_depth - 1))
    elif hasattr(obj, "objects") and getattr(obj, "objects", None):
        for child in obj.objects:
            figures.extend(find_all_bokeh_figures(child, max_depth - 1))
    return figures

# ---------------- Standalone NMF Group Figures Generator ----------------
def compute_nmf_group_figures(
    metadata_df: pd.DataFrame,
    nmf_loadings_df: pd.DataFrame,
    group_col: str,
    *,
    meta_sample_col: str = "sample",
    nmf_sample_col: str = "sample_id",
    min_group_size: int = 2,
    roi_range: Tuple[float, float] = (0.0, 1.0),
    include_recon: bool = True,
    include_sig: bool = True,
    include_pval: bool = False,
    include_loadings: bool = False,
    loadings_basis_mode: str = "Most Significant Basis",
) -> Dict[str, Any]:
    """
    Compute Kruskal-Wallis stats and generate standalone Bokeh figures for a categorical variable:
    - Sample Reconstruction by Group (asinh)
    - NMF Basis Significance (q-value)
    - NMF Basis Significance (p-value, optional)
    - Basis Loadings Jitter Plot(s) (optional)
    """
    meta = metadata_df.copy()
    H = nmf_loadings_df.copy()

    if group_col not in meta.columns:
        raise KeyError(f"metadata_df missing '{group_col}'")
    if meta_sample_col not in meta.columns:
        for candidate in ["sample", "sample_id", "sample_ID", "Sample", "SampleID", "id"]:
            if candidate in meta.columns:
                meta_sample_col = candidate
                break
        else:
            meta_sample_col = meta.columns[0]

    if nmf_sample_col not in H.columns:
        if "Unnamed: 0" in H.columns:
            nmf_sample_col = "Unnamed: 0"
        elif "sample" in H.columns:
            nmf_sample_col = "sample"
        elif "sample_id" in H.columns:
            nmf_sample_col = "sample_id"
        else:
            H = H.reset_index()
            nmf_sample_col = H.columns[0]

    meta["_sid"] = _coerce_sid(meta[meta_sample_col])
    H["_sid"] = _coerce_sid(H[nmf_sample_col])

    meta = meta.dropna(subset=["_sid", group_col]).copy()
    meta[group_col] = meta[group_col].astype(str)

    merged = pd.merge(meta[["_sid", group_col]], H, on="_sid", how="inner")
    if merged.empty:
        raise ValueError(f"No overlapping samples between metadata and NMF loadings for variable '{group_col}'.")

    # Enforce min group size
    vc = merged[group_col].value_counts()
    keep_groups = vc[vc >= min_group_size].index.tolist()
    if len(keep_groups) < 2 and min_group_size > 1:
        # Fall back to min_group_size = 1
        keep_groups = vc[vc >= 1].index.tolist()

    merged = merged[merged[group_col].isin(keep_groups)].copy()
    groups = sorted(merged[group_col].unique().tolist())
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 distinct groups in '{group_col}' to compute group statistics.")

    basis_cols = [c for c in merged.columns if c not in {"_sid", group_col, nmf_sample_col}]
    if not basis_cols:
        raise ValueError("No basis columns found in NMF loadings.")

    # Kruskal-Wallis stats per basis
    rows = []
    for j, b in enumerate(basis_cols, start=1):
        gvals = []
        means = {}
        ns = {}
        for g in groups:
            v = pd.to_numeric(merged.loc[merged[group_col] == g, b], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            gvals.append(v)
            means[f"mean[{g}]"] = float(np.mean(v)) if v.size else np.nan
            ns[f"n[{g}]"] = int(v.size)
        try:
            stat, p = kruskal(*gvals)
        except Exception:
            stat, p = 0.0, 1.0
        rows.append({"basis": b, "basis_index": j, "stat": float(stat), "p_value": float(p), **means, **ns})

    stats = pd.DataFrame(rows).set_index("basis")
    stats["q_value"] = _bh_fdr(stats["p_value"].to_numpy())
    stats = stats.sort_values(["q_value", "p_value"], ascending=[True, True])

    K = len(basis_cols)
    stats_sorted = stats.reset_index().sort_values("basis_index")
    q_values = stats_sorted["q_value"].to_numpy()
    q_values_clipped = np.clip(q_values, 1e-10, 1.0)
    p_values = stats_sorted["p_value"].to_numpy()
    p_values_clipped = np.clip(p_values, 1e-10, 1.0)

    lo, hi = roi_range
    pseudotimes = np.linspace(lo, hi, K)

    import CEtools as cet
    if lo < hi:
        centers = np.linspace(lo, hi, K)
    else:
        centers = cet.default_gaussian_centers(K)
    sigma = cet.heuristic_sigma_from_centers(centers)
    Phi = cet.make_gaussian_basis(centers, sigma)

    pt_min, pt_max = pseudotimes.min(), pseudotimes.max()
    pt_spacing = (pt_max - pt_min) / max(1, K - 1) if K > 1 else 0.1
    pt_margin = pt_spacing / 2

    recon_fig = None
    sig_fig = None
    pval_fig = None
    jitter_figs: Dict[str, bokeh.models.LayoutDOM] = {}

    # 1. NMF Basis Significance (q-value)
    if include_sig:
        sig_src = bokeh.models.ColumnDataSource(data=dict(
            basis_index=np.arange(1, K + 1),
            q_value=q_values,
            q_value_plot=q_values_clipped,
            pseudotime=pseudotimes,
            basis_name=stats_sorted["basis"].to_numpy()
        ))

        max_q = float(np.max(q_values))
        min_q_gt0 = float(np.min(q_values[q_values > 0]) if np.any(q_values > 0) else 1e-10)
        if min_q_gt0 > 0 and (max_q / min_q_gt0) > 100:
            plot_low = max(1e-10, min(0.01, min_q_gt0))
            cmap = bokeh.models.LogColorMapper(palette=common_plot.get_continuous_palette(256), low=plot_low, high=1.0)
        else:
            cmap = bokeh.models.LinearColorMapper(palette=common_plot.get_continuous_palette(256), low=0, high=max_q if max_q > 0 else 1.0)

        sig_fig = bkp.figure(
            width=800, height=210,
            title=f"NMF Basis Significance: {group_col} (q-value)",
            x_axis_label="Pseudotime",
            tools="hover,save,pan,wheel_zoom,box_zoom,reset",
            toolbar_location="above",
            active_scroll=None,
            x_range=bokeh.models.Range1d(pt_min - pt_margin, pt_max + pt_margin),
            y_range=bokeh.models.Range1d(-1, 1)
        )
        sig_fig.yaxis.visible = False
        sig_fig.ygrid.visible = False

        basis_range = bokeh.models.Range1d(start=0.5, end=K + 0.5)
        sig_fig.extra_x_ranges = {"basis": basis_range}
        sync_code = f"""
            const new_start = (cb_obj.start - {pt_min}) / {pt_spacing} + 1.0;
            const new_end = (cb_obj.end - {pt_min}) / {pt_spacing} + 1.0;
            basis_range.start = new_start;
            basis_range.end = new_end;
        """
        callback = bokeh.models.CustomJS(args=dict(basis_range=basis_range), code=sync_code)
        sig_fig.x_range.js_on_change('start', callback)
        sig_fig.x_range.js_on_change('end', callback)

        basis_axis = bokeh.models.LinearAxis(x_range_name="basis", axis_label="Basis Number")
        sig_fig.add_layout(basis_axis, 'above')

        sig_fig.rect(
            x="pseudotime", y=0, width=pt_spacing * 0.9, height=1.8,
            source=sig_src,
            fill_color={"field": "q_value_plot", "transform": cmap},
            line_color="lightgrey",
            line_width=1
        )

        hover = sig_fig.select(dict(type=bokeh.models.HoverTool))[0]
        hover.tooltips = [
            ("Basis", "@basis_name (@basis_index)"),
            ("q-value", "@q_value{%0.2e}"),
            ("Pseudotime", "@pseudotime{0.00}")
        ]
        hover.formatters = {"@q_value": "printf"}

        cbar = bokeh.models.ColorBar(color_mapper=cmap, title="q-value", orientation="horizontal", padding=0)
        sig_fig.add_layout(cbar, 'below')

    # 2. NMF Basis Significance (p-value)
    if include_pval:
        sig_src_p = bokeh.models.ColumnDataSource(data=dict(
            basis_index=np.arange(1, K + 1),
            p_value=p_values,
            p_value_plot=p_values_clipped,
            pseudotime=pseudotimes,
            basis_name=stats_sorted["basis"].to_numpy()
        ))

        max_p = float(np.max(p_values))
        min_p_gt0 = float(np.min(p_values[p_values > 0]) if np.any(p_values > 0) else 1e-10)
        if min_p_gt0 > 0 and (max_p / min_p_gt0) > 100:
            plot_low_p = max(1e-10, min(0.01, min_p_gt0))
            cmap_p = bokeh.models.LogColorMapper(palette=common_plot.get_continuous_palette(256), low=plot_low_p, high=1.0)
        else:
            cmap_p = bokeh.models.LinearColorMapper(palette=common_plot.get_continuous_palette(256), low=0, high=max_p if max_p > 0 else 1.0)

        pval_fig = bkp.figure(
            width=800, height=210,
            title=f"NMF Basis Significance: {group_col} (p-value)",
            x_axis_label="Pseudotime",
            tools="hover,save,pan,wheel_zoom,box_zoom,reset",
            toolbar_location="above",
            active_scroll=None,
            x_range=bokeh.models.Range1d(pt_min - pt_margin, pt_max + pt_margin),
            y_range=bokeh.models.Range1d(-1, 1)
        )
        pval_fig.yaxis.visible = False
        pval_fig.ygrid.visible = False

        basis_range_p = bokeh.models.Range1d(start=0.5, end=K + 0.5)
        pval_fig.extra_x_ranges = {"basis": basis_range_p}
        callback_p = bokeh.models.CustomJS(args=dict(basis_range=basis_range_p), code=sync_code)
        pval_fig.x_range.js_on_change('start', callback_p)
        pval_fig.x_range.js_on_change('end', callback_p)

        basis_axis_p = bokeh.models.LinearAxis(x_range_name="basis", axis_label="Basis Number")
        pval_fig.add_layout(basis_axis_p, 'above')

        pval_fig.rect(
            x="pseudotime", y=0, width=pt_spacing * 0.9, height=1.8,
            source=sig_src_p,
            fill_color={"field": "p_value_plot", "transform": cmap_p},
            line_color="lightgrey",
            line_width=1
        )

        hover_p = pval_fig.select(dict(type=bokeh.models.HoverTool))[0]
        hover_p.tooltips = [
            ("Basis", "@basis_name (@basis_index)"),
            ("p-value", "@p_value{%0.2e}"),
            ("Pseudotime", "@pseudotime{0.00}")
        ]
        hover_p.formatters = {"@p_value": "printf"}

        cbar_p = bokeh.models.ColorBar(color_mapper=cmap_p, title="p-value", orientation="horizontal", padding=0)
        pval_fig.add_layout(cbar_p, 'below')

    # 3. Sample Reconstruction by Group (asinh)
    if include_recon:
        recon_fig = bkp.figure(
            width=800, height=310,
            title=f"Sample Reconstruction by Group: {group_col} (asinh)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above",
            x_axis_label="Pseudotime",
            x_range=bokeh.models.Range1d(pt_min - pt_margin, pt_max + pt_margin),
            active_scroll=None
        )
        recon_fig.yaxis.visible = False

        t_eval = np.linspace(lo, hi, max(1000, K * 10))
        A_eval = Phi(t_eval)

        n_groups = len(groups)
        if common_plot.is_categorical():
            palette = common_plot._palette(n_groups)
        else:
            palette = common_plot._palette(n_groups, force_name="glasbey")

        group_data = []
        for i, g in enumerate(groups):
            g_color = palette[i % len(palette)]
            g_sids = merged[merged[group_col] == g]["_sid"].astype(str).tolist()
            yhat_list = []
            for s in g_sids:
                row = H[H["_sid"].astype(str) == s]
                if len(row) > 0:
                    h_vals = row.iloc[0][stats_sorted["basis"]].values.astype(float)
                    yhat = A_eval @ h_vals
                    yhat_list.append(np.arcsinh(yhat))
            if len(yhat_list) > 0:
                group_data.append({'g': g, 'g_color': g_color, 'yhat_list': yhat_list})

        # Draw thin gray lines first
        for data in group_data:
            xs = [t_eval for _ in range(len(data['yhat_list']))]
            ys = data['yhat_list']
            recon_fig.multi_line(xs=xs, ys=ys, line_color="gray", line_alpha=0.45, line_width=0.8)

        # Draw thick colored group averages
        for data in group_data:
            y_avg = np.mean(data['yhat_list'], axis=0)
            recon_fig.line(x=t_eval, y=y_avg, line_color=data['g_color'], line_alpha=1.0, line_width=3, legend_label=str(data['g']))

        if recon_fig.legend:
            recon_fig.legend.location = "top_left"
            recon_fig.legend.orientation = "horizontal"
            recon_fig.legend.click_policy = "hide"
            recon_fig.add_layout(recon_fig.legend[0], 'above')

    # 4. Basis Loadings Jitter Plots (Optional)
    if include_loadings:
        n_groups = len(groups)
        palette = common_plot._palette(n_groups, force_name="glasbey") if not common_plot.is_categorical() else common_plot._palette(n_groups)

        # Determine which bases to plot
        if loadings_basis_mode == "Most Significant Basis":
            bases_to_plot = [stats.index[0]]
        elif loadings_basis_mode == "Top 3 Bases":
            bases_to_plot = list(stats.index[:min(3, len(stats))])
        elif loadings_basis_mode == "All Bases":
            bases_to_plot = list(stats_sorted["basis"].to_numpy())
        else:
            bases_to_plot = [stats.index[0]]

        rng = np.random.default_rng(0)
        x_map = {g: i for i, g in enumerate(groups)}

        for b in bases_to_plot:
            d = merged[["_sid", group_col, b]].copy()
            d[b] = pd.to_numeric(d[b], errors="coerce")
            d = d.dropna(subset=[b]).copy()

            xs = np.array([x_map[g] for g in d[group_col].astype(str)], dtype=float) + rng.normal(0, 0.06, size=len(d))
            ys = d[b].to_numpy(dtype=float)

            jitter_src = bokeh.models.ColumnDataSource(data=dict(
                x=xs,
                y=ys,
                group=d[group_col].astype(str).to_numpy(),
                sample_id=d["_sid"].astype(str).to_numpy(),
                color=[palette[x_map[g] % len(palette)] for g in d[group_col].astype(str)]
            ))

            q = float(stats.loc[b, "q_value"]) if b in stats.index else float("nan")
            eff = float(stats.loc[b, "stat"]) if b in stats.index else float("nan")

            j_fig = bkp.figure(
                width=650, height=360,
                x_axis_label=group_col,
                y_axis_label="Loading Value",
                title=f"Loadings by Group: {b} ({group_col}, q={q:.3g}, stat={eff:.3g})",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                active_scroll=None,
            )
            jitter_r = j_fig.circle("x", "y", color="color", source=jitter_src, size=7, fill_alpha=0.75, line_alpha=0.25)
            j_fig.add_tools(bokeh.models.HoverTool(
                renderers=[jitter_r],
                tooltips=[("Sample", "@sample_id"), ("Group", "@group"), ("Loading", "@y{0.000}")]
            ))
            j_fig.xaxis.ticker = list(range(len(groups)))
            j_fig.xaxis.formatter = bokeh.models.CustomJSTickFormatter(code=f"""
                const mapping = {repr(dict(enumerate(groups)))};
                return mapping[tick] || "";
            """)
            jitter_figs[b] = j_fig

    return {
        "group_col": group_col,
        "groups": groups,
        "stats": stats,
        "recon_fig": recon_fig,
        "sig_fig": sig_fig,
        "pval_fig": pval_fig,
        "jitter_figs": jitter_figs
    }


def detach_bokeh_models(models: Union[bokeh.models.Model, Sequence[bokeh.models.Model], Dict[str, bokeh.models.Model]]):
    """Safely detach Bokeh models and their references from any document so they can be embedded repeatedly."""
    if isinstance(models, dict):
        model_list = list(models.values())
    elif isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]

    for m in model_list:
        if m is None:
            continue
        all_models = [m]
        if hasattr(m, "references"):
            all_models.extend(m.references())
        for ref in all_models:
            doc = getattr(ref, "document", None)
            if doc is not None:
                try:
                    if ref in doc.roots:
                        doc.remove_root(ref)
                except Exception:
                    pass
                try:
                    if hasattr(doc, "models") and ref.id in doc.models:
                        del doc.models[ref.id]
                except Exception:
                    pass
            ref._document = None
            if hasattr(ref, "_temp_document"):
                ref._temp_document = None


def set_bokeh_output_backend_svg(models: Union[bokeh.models.Model, Sequence[bokeh.models.Model], Dict[str, bokeh.models.Model]]):
    """Recursively set output_backend = 'svg' on all Bokeh plots and models so Bokeh renders genuine vector elements."""
    if isinstance(models, dict):
        model_list = list(models.values())
    elif isinstance(models, (list, tuple)):
        model_list = list(models)
    else:
        model_list = [models]

    for m in model_list:
        if m is None:
            continue
        all_models = [m]
        if hasattr(m, "references"):
            try:
                all_models.extend(m.references())
            except Exception:
                pass
        for ref in all_models:
            if hasattr(ref, "output_backend"):
                try:
                    ref.output_backend = "svg"
                except Exception:
                    pass


def attach_interactive_offset_slider(fig_or_layout: Any, initial_offset: float = 0.0, max_offset: float = 10.0, step: float = 0.5) -> Any:
    """Attach a client-side Bokeh Slider to a multi-trace plot or paired plots with CustomJS to interactively adjust vertical offset."""
    if not isinstance(fig_or_layout, bokeh.models.LayoutDOM):
        return fig_or_layout

    if isinstance(fig_or_layout, bokeh.models.Plot):
        target_plots = [fig_or_layout]
    else:
        target_plots = [m for m in fig_or_layout.references() if isinstance(m, bokeh.models.Plot)]

    if not target_plots:
        return fig_or_layout

    sources = []
    for p in target_plots:
        line_idx = 0
        for r in p.renderers:
            if hasattr(r, "data_source") and "y" in r.data_source.data and "x" in r.data_source.data:
                src = r.data_source
                if "sample_idx" in src.data and len(src.data["sample_idx"]) > 0:
                    if "y_base" not in src.data or len(src.data["y_base"]) != len(src.data["y"]):
                        idx = src.data["sample_idx"][0]
                        y_curr = np.array(src.data["y"], dtype=float)
                        src.data["y_base"] = y_curr + idx * initial_offset
                    sources.append(src)
                elif hasattr(r, "glyph") and isinstance(r.glyph, bokeh.models.Line):
                    y_curr = np.array(src.data["y"], dtype=float)
                    if "y_base" not in src.data or "sample_idx" not in src.data:
                        src.data["y_base"] = y_curr + line_idx * initial_offset
                        src.data["sample_idx"] = [line_idx] * len(y_curr)
                    sources.append(src)
                    line_idx += 1

    if not sources:
        return fig_or_layout

    slider = bokeh.models.Slider(
        title="Interactive Vertical Offset",
        start=0.0,
        end=max_offset,
        step=step,
        value=initial_offset,
        sizing_mode="stretch_width"
    )
    cb = bokeh.models.CustomJS(args=dict(sources=sources), code="""
        const val = cb_obj.value;
        for (let i = 0; i < sources.length; i++) {
            const src = sources[i];
            if (!src.data['sample_idx'] || src.data['sample_idx'].length === 0) continue;
            const idx = src.data['sample_idx'][0];
            const y_base = src.data['y_base'];
            const new_y = [];
            for (let j = 0; j < y_base.length; j++) {
                new_y.push(y_base[j] - idx * val);
            }
            src.data['y'] = new_y;
            src.change.emit();
        }
    """)
    slider.js_on_change("value", cb)
    return bokeh.layouts.column(slider, fig_or_layout, sizing_mode="stretch_width")


class HTMLReportController:
    """Controller for the HTML Report Export tab."""

    def __init__(self, controllers: Dict[str, Any]):
        self.controllers = controllers
        self._export_prefix = "CE_analysis"
        self._current_html_str: str = ""

        # Nested NMF Group Statistics configurations
        # list of dicts: {col_name, include_recon, include_sig, include_pval, include_loadings, loadings_basis_mode, custom_title}
        self.nmf_group_sections: List[Dict[str, Any]] = []

        # User-added custom items
        self.additional_pipeline_plots: List[Dict[str, Any]] = [] # list of dicts: {id, title, caption, fig, type}
        self.custom_images: List[Dict[str, Any]] = [] # list of dicts: {id, title, caption, data_uri, filename}

        # ---------------- 1. Report Metadata & Config ----------------
        self.report_title_input = pn.widgets.TextInput(
            name="Report Title",
            value=f"{self._export_prefix} Electropherogram Analysis Report",
            sizing_mode="stretch_width"
        )
        self.report_subtitle_input = pn.widgets.TextInput(
            name="Subtitle / Project Description",
            value="",
            placeholder="e.g. Capillary Electrophoresis Quality Control & NMF Decomposition",
            sizing_mode="stretch_width"
        )
        self.author_input = pn.widgets.TextInput(
            name="Author / Analyst Name",
            value="",
            placeholder="e.g. Lab Researcher",
            sizing_mode="stretch_width"
        )
        self.report_date_input = pn.widgets.TextInput(
            name="Report Date",
            value=datetime.now().strftime("%Y-%m-%d"),
            width=160
        )
        self.include_params_chk = pn.widgets.Checkbox(
            name="Include Pipeline Session Parameters",
            value=True,
            width=260
        )
        self.include_toc_chk = pn.widgets.Checkbox(
            name="Include Table of Contents",
            value=True,
            width=200
        )
        self.resource_mode = pn.widgets.RadioButtonGroup(
            name="Resource Mode",
            options=["CDN (Lightweight, ~50 KB)", "Inline (Offline, ~3.5 MB)"],
            value="CDN (Lightweight, ~50 KB)"
        )
        self.filename_input = pn.widgets.TextInput(
            name="HTML File Name",
            value=f"{self._export_prefix}_report.html",
            sizing_mode="stretch_width"
        )

        # ---------------- 2. Standard Plots Selection ----------------
        self.scan_plots_btn = pn.widgets.Button(
            name="Scan & Refresh Plots",
            button_type="primary",
            width=170
        )
        self.scan_plots_btn.on_click(lambda _: self.refresh_available_plots())

        self.select_all_btn = pn.widgets.Button(
            name="Select All Ready",
            button_type="default",
            width=130
        )
        self.select_all_btn.on_click(lambda _: self._set_all_selected(True))

        self.deselect_all_btn = pn.widgets.Button(
            name="Deselect All",
            button_type="default",
            width=120
        )
        self.deselect_all_btn.on_click(lambda _: self._set_all_selected(False))

        self.plots_summary_md = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.standard_plots_layout = pn.Column(sizing_mode="stretch_width")
        self._standard_plot_widgets: Dict[str, Dict[str, Any]] = {}

        # ---------------- 3. NMF Group Statistics (By Categorical Variable) ----------------
        self.metadata_status_md = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.metadata_var_select = pn.widgets.Select(name="Select Categorical Variable", options=[], width=260)
        self.add_category_btn = pn.widgets.Button(name="Add Variable Section", button_type="primary", width=170)
        self.add_category_btn.on_click(lambda _: self._on_add_category_section())
        self.nmf_group_sections_container = pn.Column(sizing_mode="stretch_width")

        # ---------------- 4. Additional Pipeline Sub-Plots ----------------
        # 4A: Sample Reconstruction
        self.sample_recon_select = pn.widgets.Select(name="Select Sample for Reconstruction", options=[], width=260)
        self.sample_recon_asinh = pn.widgets.Checkbox(name="Use asinh transform", value=False)
        self.sample_recon_title = pn.widgets.TextInput(name="Plot Title", value="Sample Reconstruction", width=260)
        self.sample_recon_caption = pn.widgets.TextInput(name="Optional Caption", value="", width=300)
        self.add_sample_recon_btn = pn.widgets.Button(name="Add Sample Reconstruction", button_type="success", width=210)
        self.add_sample_recon_btn.on_click(self._on_add_sample_reconstruction)

        # 4B: Alpha Diversity Peak Detection Sub-Plot
        self.alpha_peak_sample_select = pn.widgets.Select(name="Select Sample", options=[], width=260)
        self.alpha_peak_asinh = pn.widgets.Checkbox(name="Use asinh transform", value=True)
        self.alpha_peak_title = pn.widgets.TextInput(name="Plot Title", value="Sample Peak Detection", width=260)
        self.alpha_peak_caption = pn.widgets.TextInput(name="Optional Caption", value="", width=300)
        self.add_alpha_peak_btn = pn.widgets.Button(name="Add Peak Detection Plot", button_type="success", width=210)
        self.add_alpha_peak_btn.on_click(self._on_add_alpha_peak_plot)
        self.alpha_peak_sample_select.param.watch(
            lambda e: setattr(self.alpha_peak_title, "value", f"Peak Detection: {e.new}" if e.new else "Sample Peak Detection"), "value"
        )

        # 4C: Dimensionality Reduction Custom Plots (PCA, PLS-DA, MDS, t-SNE)
        self.viz_plot_method = pn.widgets.Select(name="Method", options=["PCA", "PLS-DA", "MDS", "t-SNE"], value="PCA", width=130)
        self.viz_color_source = pn.widgets.Select(name="Color Source", options=["Metadata Variable", "Diversity Metric"], value="Metadata Variable", width=160)
        self.viz_color_var = pn.widgets.Select(name="Color By Variable / Metric", options=[], width=220)
        self.viz_color_mode = pn.widgets.RadioButtonGroup(name="Value Type", options=["Categorical", "Continuous"], value="Categorical")

        self.viz_pca_x = pn.widgets.IntInput(name="PC X (1-based)", value=1, start=1, end=50, width=110)
        self.viz_pca_y = pn.widgets.IntInput(name="PC Y (1-based)", value=2, start=1, end=50, width=110)
        self.viz_pls_x = pn.widgets.IntInput(name="LV X (1-based)", value=1, start=1, end=10, width=110, visible=False)
        self.viz_pls_y = pn.widgets.IntInput(name="LV Y (1-based)", value=2, start=1, end=10, width=110, visible=False)
        self.viz_mds_metric = pn.widgets.Select(name="MDS Metric", options=["euclidean", "cosine", "correlation", "cityblock", "braycurtis"], value="euclidean", width=140, visible=False)
        self.viz_tsne_metric = pn.widgets.Select(name="t-SNE Metric", options=["cosine", "euclidean", "correlation", "cityblock"], value="cosine", width=140, visible=False)
        self.viz_tsne_perp = pn.widgets.FloatInput(name="Perplexity", value=30.0, start=1.0, width=110, visible=False)

        self.viz_plot_title = pn.widgets.TextInput(name="Plot Title", value="PCA Score Plot (PC1 vs PC2)", width=280)
        self.viz_plot_caption = pn.widgets.TextInput(name="Optional Caption", value="", width=300)
        self.add_viz_plot_btn = pn.widgets.Button(name="Add Dimensionality Reduction Plot", button_type="success", width=250)
        self.add_viz_plot_btn.on_click(self._on_add_viz_plot)

        def _update_viz_method_controls(event):
            m = event.new
            self.viz_pca_x.visible = (m == "PCA")
            self.viz_pca_y.visible = (m == "PCA")
            self.viz_pls_x.visible = (m == "PLS-DA")
            self.viz_pls_y.visible = (m == "PLS-DA")
            self.viz_mds_metric.visible = (m == "MDS")
            self.viz_tsne_metric.visible = (m == "t-SNE")
            self.viz_tsne_perp.visible = (m == "t-SNE")
            if m == "PLS-DA":
                self.viz_color_mode.value = "Categorical"
            self._auto_update_viz_title()

        self.viz_plot_method.param.watch(_update_viz_method_controls, "value")
        self.viz_pca_x.param.watch(lambda _: self._auto_update_viz_title(), "value")
        self.viz_pca_y.param.watch(lambda _: self._auto_update_viz_title(), "value")
        self.viz_pls_x.param.watch(lambda _: self._auto_update_viz_title(), "value")
        self.viz_pls_y.param.watch(lambda _: self._auto_update_viz_title(), "value")
        self.viz_mds_metric.param.watch(lambda _: self._auto_update_viz_title(), "value")
        self.viz_tsne_metric.param.watch(lambda _: self._auto_update_viz_title(), "value")
        self.viz_color_var.param.watch(lambda _: self._auto_update_viz_title(), "value")
        self.viz_color_source.param.watch(lambda _: self._update_viz_color_options(), "value")

        self.additional_plots_container = pn.Column(
            pn.pane.Markdown("*No additional sub-plots added yet.*"),
            sizing_mode="stretch_width"
        )

        # ---------------- 5. Upload Custom Plot Images ----------------
        self.image_file_input = pn.widgets.FileInput(
            accept=".png,.jpg,.jpeg,.svg,.webp",
            multiple=False,
            sizing_mode="stretch_width"
        )
        self.image_title_input = pn.widgets.TextInput(name="Figure Title", value="External Figure", width=260)
        self.image_section_select = pn.widgets.Select(
            name="Append to Report Section",
            options=[
                "Raw Data & QC",
                "Preprocessing",
                "Alignment & Anchors",
                "Alpha Diversity Analysis",
                "Dimensionality Reduction",
                "NMF Decomposition",
                "Time Series & Transitions",
                "Miscellaneous Analysis"
            ],
            value="Raw Data & QC",
            width=260
        )
        self.image_caption_input = pn.widgets.TextInput(name="Caption / Notes (Optional)", value="", sizing_mode="stretch_width")
        self.add_image_btn = pn.widgets.Button(name="Add Uploaded Image", button_type="success", width=170)
        self.add_image_btn.on_click(self._on_add_custom_image)

        self.custom_images_container = pn.Column(
            pn.pane.Markdown("*No external images added yet.*"),
            sizing_mode="stretch_width"
        )

        # ---------------- 6. Generation & Download ----------------
        self.build_report_btn = pn.widgets.Button(
            name="Build HTML Report",
            button_type="primary",
            sizing_mode="stretch_width",
            height=45
        )
        self.build_report_btn.on_click(self._on_build_report)

        self.download_report_btn = pn.widgets.FileDownload(
            label="Download HTML Report",
            filename=self.filename_input.value,
            button_type="success",
            embed=False,
            auto=False,
            callback=self._get_report_bytes,
            disabled=True,
            sizing_mode="stretch_width",
            height=45
        )
        self.filename_input.param.watch(
            lambda e: setattr(self.download_report_btn, "filename", e.new or "report.html"), "value"
        )

        self.report_status = pn.pane.Markdown("", sizing_mode="stretch_width")

        # ---------------- Layout Construction (Full Width, No Preview) ----------------
        self.section = pn.Column(
            pn.pane.Markdown("## HTML Report Export"),
            pn.pane.Markdown(
                "Generate an interactive, standalone HTML report containing selected pipeline figures, "
                "nested NMF group statistics across categorical metadata variables, custom sample reconstructions, "
                "uploaded external figures, and session execution parameters."
            ),
            pn.layout.Divider(),

            # Card 1: Metadata
            pn.pane.Markdown("### 1. Report Settings"),
            self.report_title_input,
            self.report_subtitle_input,
            pn.Row(self.author_input, self.report_date_input),
            pn.Row(self.include_params_chk, self.include_toc_chk),
            pn.Row(pn.pane.Markdown("**Resource Bundling:**", margin=(5, 5, 0, 0)), self.resource_mode),
            self.filename_input,
            pn.layout.Divider(),

            # Card 2: Standard Plots
            pn.pane.Markdown("### 2. Standard Pipeline Plots"),
            pn.Row(self.scan_plots_btn, self.select_all_btn, self.deselect_all_btn),
            self.plots_summary_md,
            self.standard_plots_layout,
            pn.layout.Divider(),

            # Card 3: NMF Group Statistics (Nested)
            pn.pane.Markdown("### 3. NMF Group Statistics (By Categorical Variable)"),
            pn.pane.Markdown(
                "Specify one or more categorical variables from the metadata. Each variable produces group "
                "reconstructions, basis significance tests, and loadings, which will be integrated into the "
                "NMF Decomposition section of the exported HTML report."
            ),
            self.metadata_status_md,
            pn.Row(self.metadata_var_select, self.add_category_btn, align="end"),
            self.nmf_group_sections_container,
            pn.layout.Divider(),

            # Card 4: Additional Sub-Plots
            pn.pane.Markdown("### 4. Specify Additional Sub-Plots"),
            pn.Tabs(
                ("Sample Reconstruction", pn.Column(
                    pn.Row(self.sample_recon_select, self.sample_recon_asinh),
                    pn.Row(self.sample_recon_title, self.sample_recon_caption),
                    self.add_sample_recon_btn
                )),
                ("Alpha Diversity Peak Detection", pn.Column(
                    pn.Row(self.alpha_peak_sample_select, self.alpha_peak_asinh),
                    pn.Row(self.alpha_peak_title, self.alpha_peak_caption),
                    self.add_alpha_peak_btn
                )),
                ("Dimensionality Reduction", pn.Column(
                    pn.Row(self.viz_plot_method, self.viz_color_source, self.viz_color_var, self.viz_color_mode),
                    pn.Row(self.viz_pca_x, self.viz_pca_y, self.viz_pls_x, self.viz_pls_y, self.viz_mds_metric, self.viz_tsne_metric, self.viz_tsne_perp),
                    pn.Row(self.viz_plot_title, self.viz_plot_caption),
                    self.add_viz_plot_btn
                )),
            ),
            pn.pane.Markdown("**Current Additional Sub-Plots:**", styles={"margin-top": "10px"}),
            self.additional_plots_container,
            pn.layout.Divider(),

            # Card 5: External Figures
            pn.pane.Markdown("### 5. Upload External Figures / Images"),
            self.image_file_input,
            pn.Row(self.image_title_input, self.image_section_select),
            pn.Row(self.image_caption_input, self.add_image_btn, align="end"),
            pn.pane.Markdown("**Current Uploaded Figures:**", styles={"margin-top": "10px"}),
            self.custom_images_container,
            pn.layout.Divider(),

            # Card 6: Generate & Download
            pn.pane.Markdown("### 6. Export Report"),
            pn.Row(self.build_report_btn, self.download_report_btn),
            self.report_status,
            sizing_mode="stretch_width"
        )

        # Initialize available plots
        self._init_standard_plot_definitions()
        self.refresh_available_plots()

    @property
    def export_prefix(self) -> str:
        return getattr(self, "_export_prefix", "CE_analysis")

    @export_prefix.setter
    def export_prefix(self, val: str):
        old_val = getattr(self, "_export_prefix", "")
        self._export_prefix = str(val) if val else "CE_analysis"
        if hasattr(self, "report_title_input") and self.report_title_input is not None:
            curr_title = self.report_title_input.value
            if not curr_title or (old_val and curr_title.startswith(old_val)):
                self.report_title_input.value = f"{self._export_prefix} Electropherogram Analysis Report"
        if hasattr(self, "filename_input") and self.filename_input is not None:
            curr_fn = self.filename_input.value
            if not curr_fn or (old_val and curr_fn.startswith(old_val)):
                self.filename_input.value = f"{self._export_prefix}_report.html"

    # ---------------- Standard Plot Registry ----------------
    def _init_standard_plot_definitions(self):
        """Define the catalog of standard pipeline plots (excluding nested NMF group stats)."""
        self.plot_catalog = [
            # Raw & QC
            {
                "id": "raw_electropherograms",
                "category": "Raw Data & QC",
                "default_title": "Raw Electropherograms Preview",
                "getter": lambda offset=None: self._get_plot_raw(offset=offset),
                "has_offset": True,
                "get_offset": lambda: getattr(getattr(self.controllers.get("state"), "offset_slider", None), "value", 0.0),
                "default_selected": True
            },
            {
                "id": "control_electropherograms",
                "category": "Raw Data & QC",
                "default_title": "Control Electropherograms Preview",
                "getter": lambda offset=None: self._get_plot_controls(offset=offset),
                "has_offset": True,
                "get_offset": lambda: getattr(getattr(self.controllers.get("state"), "control_offset_slider", None), "value", 0.0),
                "default_selected": True
            },
            {
                "id": "snr_data",
                "category": "Raw Data & QC",
                "default_title": "Signal-to-Noise Ratio (SNR)",
                "getter": lambda: self._get_snr_data(),
                "default_selected": True
            },
            # Preprocessing
            {
                "id": "preproc_despike",
                "category": "Preprocessing",
                "default_title": "Despiked Electropherograms (Before and After)",
                "getter": lambda offset=None: self._get_plot_despike(offset=offset),
                "has_offset": True,
                "get_offset": lambda: getattr(getattr(self.controllers.get("despike"), "preproc_offset", None), "value", 0.0),
                "default_selected": False
            },
            {
                "id": "preproc_smooth",
                "category": "Preprocessing",
                "default_title": "Smoothed Electropherograms (Before and After)",
                "getter": lambda offset=None: self._get_plot_smooth(offset=offset),
                "has_offset": True,
                "get_offset": lambda: getattr(getattr(self.controllers.get("smooth"), "preproc_offset", None), "value", 0.0),
                "default_selected": False
            },
            {
                "id": "preproc_baseline",
                "category": "Preprocessing",
                "default_title": "Baseline-Subtracted Electropherograms (Before and After)",
                "getter": lambda offset=None: self._get_plot_baseline(offset=offset),
                "has_offset": True,
                "get_offset": lambda: getattr(getattr(self.controllers.get("baseline"), "preproc_offset", None), "value", 0.0),
                "default_selected": False
            },
            {
                "id": "preproc_norm_integration",
                "category": "Preprocessing",
                "default_title": "Peak Detection & Integration Preview",
                "getter": lambda **kwargs: self._get_plot_norm_integration(),
                "has_offset": False,
                "default_selected": True
            },
            {
                "id": "preproc_norm",
                "category": "Preprocessing",
                "default_title": "Normalized Electropherograms",
                "getter": lambda offset=None: self._get_plot_norm(offset=offset),
                "has_offset": True,
                "get_offset": lambda: getattr(getattr(self.controllers.get("norm"), "post_offset", None), "value", 0.0),
                "default_selected": True
            },
            # Alignment
            {
                "id": "align_anchors",
                "category": "Alignment",
                "default_title": "Anchor Selection Curves",
                "getter": lambda offset=None: self._get_plot_align_anchors(offset=offset),
                "has_offset": True,
                "get_offset": lambda: getattr(getattr(self.controllers.get("alignment"), "offset_slider", None), "value", 0.0),
                "default_selected": False
            },
            {
                "id": "align_aligned",
                "category": "Alignment",
                "default_title": "Aligned Chromatograms",
                "getter": lambda offset=None: self._get_plot_align_aligned(offset=offset),
                "has_offset": True,
                "get_offset": lambda: getattr(getattr(self.controllers.get("alignment"), "aligned_offset_slider", None), "value", 0.0),
                "default_selected": True
            },
            # NMF Decomposition (Basis Regression Matrix moved here as requested)
            {
                "id": "nmf_recon",
                "category": "NMF Decomposition",
                "default_title": "NMF Overall Sample Reconstruction",
                "getter": lambda: self._get_plot_nmf_recon(),
                "default_selected": True
            },
            {
                "id": "nmf_heatmap",
                "category": "NMF Decomposition",
                "default_title": "NMF Loadings Heatmap",
                "getter": lambda: self._get_plot_nmf_heatmap(),
                "default_selected": True
            },
            {
                "id": "nmf_reg_matrix",
                "category": "NMF Decomposition",
                "default_title": "NMF Basis Regression Matrix",
                "getter": lambda: self._get_plot_nmf_reg_matrix(),
                "default_selected": True
            },
            # Alpha Diversity
            {
                "id": "alpha_div_table",
                "category": "Alpha Diversity",
                "default_title": "Alpha Diversity Metrics Table",
                "getter": lambda: self._get_alpha_div_table(),
                "default_selected": True
            },
            # Dimensionality Reduction
            {
                "id": "viz_scree",
                "category": "Dimensionality Reduction",
                "default_title": "PCA Scree Plot (Explained Variance)",
                "getter": lambda: self._get_plot_viz_scree(),
                "default_selected": True
            },
            {
                "id": "viz_loadings_pc1_3",
                "category": "Dimensionality Reduction",
                "default_title": "PCA Loadings for First 3 PCs (PC1, PC2, PC3)",
                "getter": lambda: self._get_plot_viz_pc_loadings_1_3(),
                "default_selected": True
            }
        ]

    # ---------------- Plot Getters from Controllers ----------------
    def _get_plot_raw(self, offset=None):
        state = self.controllers.get("state")
        if not state:
            return None
        if offset is not None and getattr(state, "converted_by_sample", None):
            samples = state.converted_by_sample
            if getattr(state, "sort_by_order_btn", None) and getattr(state.sort_by_order_btn, "value", False) and state.metadata_df is not None and 'Order of Run' in state.metadata_df.columns:
                try:
                    order_map = state.metadata_df.set_index('Original Name')['Order of Run'].dropna().to_dict()
                    sorted_keys = sorted(samples.keys(), key=lambda k: order_map.get(k, 999999))
                    samples = {k: samples[k] for k in sorted_keys}
                except Exception:
                    pass
            min_pref = getattr(getattr(state, "prefer_minutes", None), "value", True)
            ash = getattr(getattr(state, "asinh_toggle", None), "value", True)
            return common_plot.plot_multi(samples, "Chromatograph(s) preview", offset=offset, asinh=ash, minutes=min_pref)

        fig = getattr(state, "last_fig", None)
        if fig is None:
            fig = getattr(state, "last_plot", None)
        if fig is None:
            fig = extract_bokeh_figure(getattr(state, "plot_pane", None))
        return fig

    def _get_plot_controls(self, offset=None):
        state = self.controllers.get("state")
        if not state:
            return None
        if offset is not None and getattr(state, "control_converted_by_sample", None):
            samples = state.control_converted_by_sample
            if getattr(state, "control_sort_by_order_btn", None) and getattr(state.control_sort_by_order_btn, "value", False) and state.control_metadata_df is not None and 'Order of Run' in state.control_metadata_df.columns:
                try:
                    order_map = state.control_metadata_df.set_index('Original Name')['Order of Run'].dropna().to_dict()
                    sorted_keys = sorted(samples.keys(), key=lambda k: order_map.get(k, 999999))
                    samples = {k: samples[k] for k in sorted_keys}
                except Exception:
                    pass
            min_pref = getattr(getattr(state, "control_prefer_minutes", None), "value", True)
            ash = getattr(getattr(state, "control_asinh_toggle", None), "value", True)
            return common_plot.plot_multi(samples, "Control Chromatograph(s) preview", offset=offset, asinh=ash, minutes=min_pref)

        fig = getattr(state, "control_last_fig", None)
        if fig is None:
            fig = extract_bokeh_figure(getattr(state, "control_plot_pane", None))
        return fig

    def _get_snr_data(self):
        ctrl = self.controllers.get("snr")
        if not ctrl:
            return None
        res_df = getattr(ctrl, "results_df", None)
        if res_df is None or (hasattr(res_df, "empty") and res_df.empty):
            return None
        return {
            "type": "snr_table",
            "results_df": res_df,
            "signal_range": getattr(ctrl, "signal_range", None),
            "noise_range": getattr(ctrl, "noise_range", None),
            "excluded_samples": list(getattr(getattr(ctrl, "exclude_group", None), "value", [])),
        }

    def _get_plot_snr(self):
        ctrl = self.controllers.get("snr")
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None)) if ctrl else None

    def _format_snr_html(self, snr_data: Dict[str, Any]) -> str:
        sig_range = snr_data.get("signal_range")
        noi_range = snr_data.get("noise_range")
        excluded = snr_data.get("excluded_samples", [])
        df = snr_data.get("results_df")

        sig_text = f"{sig_range[0]:.3f} &ndash; {sig_range[1]:.3f} min" if sig_range else "Not specified"
        noi_text = f"{noi_range[0]:.3f} &ndash; {noi_range[1]:.3f} min" if noi_range else "Not specified"

        regions_html = f"""
        <div style="display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 12px 16px; flex: 1; min-width: 220px;">
                <div style="font-size: 0.8rem; color: #1e40af; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Chosen Signal Region (Peak S)</div>
                <div style="font-size: 1.15rem; color: #1e3a8a; font-weight: 700;">{sig_text}</div>
                <div style="font-size: 0.8rem; color: #3b82f6; margin-top: 4px;">Tallest peak height S extracted per sample within this window</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px 16px; flex: 1; min-width: 220px;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Chosen Noise Region (Baseline B, std &sigma;<sub>B</sub>)</div>
                <div style="font-size: 1.15rem; color: #334155; font-weight: 700;">{noi_text}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Mean intensity B and baseline standard deviation &sigma;<sub>B</sub> extracted from this window</div>
            </div>
        </div>
        """

        excluded_html = ""
        if excluded:
            ex_str = ", ".join(str(s) for s in excluded)
            excluded_html = f"""
            <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; font-size: 0.85rem; color: #92400e;">
                <strong>Excluded Samples ({len(excluded)}):</strong> {ex_str}
            </div>
            """

        table_html = ""
        if df is not None and not df.empty:
            table_df = df.reset_index() if "Sample" not in df.columns else df.copy()

            col_map = {
                "Sample": "Sample",
                "S": "Signal Peak (S)",
                "B": "Noise Mean (B)",
                "sigma_B": "Noise Std (&sigma;<sub>B</sub>)",
                "SNR": "Signal-to-Noise Ratio (SNR)"
            }
            cols = [c for c in ["Sample", "S", "B", "sigma_B", "SNR"] if c in table_df.columns]
            if not cols:
                cols = list(table_df.columns)

            th_cells = []
            for c in cols:
                align = "left" if c == "Sample" else "right"
                label = col_map.get(c, c)
                th_cells.append(f"<th style='padding: 10px 14px; font-weight: 600; text-align: {align}; border-bottom: 2px solid #cbd5e1; background: #f8fafc; color: #334155;'>{label}</th>")
            thead_html = f"<thead style='position: sticky; top: 0; z-index: 1;'><tr>{''.join(th_cells)}</tr></thead>"

            tr_rows = []
            for idx, row in table_df.iterrows():
                td_cells = []
                bg = "#ffffff" if idx % 2 == 0 else "#f8fafc"
                for c in cols:
                    val = row[c]
                    align = "left" if c == "Sample" else "right"
                    if c == "Sample":
                        val_str = str(val)
                        bold_style = "font-weight: 600; color: #1e293b;"
                    elif pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
                        val_str = "N/A"
                        bold_style = "color: #94a3b8;"
                    elif c == "SNR":
                        val_str = f"{float(val):.2f}"
                        bold_style = "font-weight: 700; color: #0f766e;"
                    else:
                        val_str = f"{float(val):.4f}"
                        bold_style = "color: #334155;"
                    td_cells.append(f"<td style='padding: 8px 14px; text-align: {align}; border-bottom: 1px solid #e2e8f0; {bold_style}'>{val_str}</td>")
                tr_rows.append(f"<tr style='background: {bg};'>{''.join(td_cells)}</tr>")
            tbody_html = f"<tbody>{''.join(tr_rows)}</tbody>"

            table_html = f"""
            <div style="overflow-x: auto; max-height: 480px; border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; font-family: inherit;">
                    {thead_html}
                    {tbody_html}
                </table>
            </div>
            <div style="margin-top: 8px; font-size: 0.8rem; color: #64748b; font-style: italic;">
                SNR calculation formula: (S &minus; B) / &sigma;<sub>B</sub>
            </div>
            """
        else:
            table_html = "<p style='color: #64748b; font-style: italic;'>No SNR results data table available.</p>"

        return f"<div class='snr-report-block'>{regions_html}{excluded_html}{table_html}</div>"

    def _get_plot_despike(self, offset=None):
        ctrl = self.controllers.get("despike")
        if not ctrl:
            return None
        inp = getattr(ctrl, "input_by_sample", None)
        out = getattr(ctrl, "output_by_sample", None)
        ash = getattr(getattr(ctrl, "asinh_toggle", None), "value", True)
        off = offset if offset is not None else getattr(getattr(ctrl, "preproc_offset", None), "value", 0.0)

        if inp:
            p_before = common_plot.plot_multi(inp, "Before: original", offset=off, asinh=ash)
            p_after = common_plot.plot_multi(out if out else inp, "After: despiked", offset=off, asinh=ash)
            p_after.x_range = p_before.x_range
            p_after.y_range = p_before.y_range
            return bokeh.layouts.row(p_before, p_after, sizing_mode="stretch_width")

        f_before = extract_bokeh_figure(getattr(ctrl, "before_pane", None))
        f_after = extract_bokeh_figure(getattr(ctrl, "after_pane", None))
        if f_before and f_after:
            f_after.x_range = f_before.x_range
            f_after.y_range = f_before.y_range
            return bokeh.layouts.row(f_before, f_after, sizing_mode="stretch_width")
        return f_after or f_before

    def _get_plot_smooth(self, offset=None):
        ctrl = self.controllers.get("smooth")
        if not ctrl:
            return None
        inp = getattr(ctrl, "input_by_sample", None)
        out = getattr(ctrl, "output_by_sample", None) or getattr(ctrl, "smoothed_by_sample", None)
        ash = getattr(getattr(ctrl, "asinh_toggle", None), "value", True)
        off = offset if offset is not None else getattr(getattr(ctrl, "preproc_offset", None), "value", 0.0)

        if inp:
            p_before = common_plot.plot_multi(inp, "Before: original", offset=off, asinh=ash)
            p_after = common_plot.plot_multi(out if out else inp, "After: smoothed", offset=off, asinh=ash)
            p_after.x_range = p_before.x_range
            p_after.y_range = p_before.y_range
            return bokeh.layouts.row(p_before, p_after, sizing_mode="stretch_width")

        f_before = extract_bokeh_figure(getattr(ctrl, "before_pane", None))
        f_after = extract_bokeh_figure(getattr(ctrl, "after_pane", None))
        if f_before and f_after:
            f_after.x_range = f_before.x_range
            f_after.y_range = f_before.y_range
            return bokeh.layouts.row(f_before, f_after, sizing_mode="stretch_width")
        return f_after or f_before

    def _get_plot_baseline(self, offset=None):
        ctrl = self.controllers.get("baseline")
        if not ctrl:
            return None
        inp = getattr(ctrl, "input_by_sample", None)
        out = getattr(ctrl, "output_by_sample", None)
        ash = getattr(getattr(ctrl, "asinh_toggle", None), "value", True)
        off = offset if offset is not None else getattr(getattr(ctrl, "preproc_offset", None), "value", 0.0)

        if inp:
            p_before = common_plot.plot_multi(inp, "Before: original", offset=off, asinh=ash)
            p_after = common_plot.plot_multi(out if out else inp, "After: baseline-subtracted", offset=off, asinh=ash)
            p_after.x_range = p_before.x_range
            p_after.y_range = p_before.y_range
            return bokeh.layouts.row(p_before, p_after, sizing_mode="stretch_width")

        f_before = extract_bokeh_figure(getattr(ctrl, "before_pane", None))
        f_after = extract_bokeh_figure(getattr(ctrl, "after_pane", None))
        if f_before and f_after:
            f_after.x_range = f_before.x_range
            f_after.y_range = f_before.y_range
            return bokeh.layouts.row(f_before, f_after, sizing_mode="stretch_width")
        return f_after or f_before

    def _get_plot_norm_integration(self):
        ctrl = self.controllers.get("norm")
        if not ctrl:
            return None
        fig = extract_bokeh_figure(getattr(ctrl, "before_pane", None))
        if fig is None and getattr(ctrl, "current_by_sample", None):
            try:
                ctrl._render_before_fig()
                fig = extract_bokeh_figure(getattr(ctrl, "before_pane", None))
            except Exception:
                pass
        return fig

    def _get_plot_norm(self, offset=None):
        ctrl = self.controllers.get("norm")
        if not ctrl:
            return None
        norm_s = getattr(ctrl, "normalized_by_sample", None)
        ash = getattr(getattr(ctrl, "post_asinh", None), "value", True)
        off = offset if offset is not None else getattr(getattr(ctrl, "post_offset", None), "value", 0.0)
        if offset is not None and norm_s:
            return common_plot.plot_multi(norm_s, title="Normalized Electropherograms", xlab="time", ylab="intensity (normalized)", offset=off, asinh=ash, line_width=2)

        fig = extract_bokeh_figure(getattr(ctrl, "after_pane", None))
        if fig is None and norm_s:
            return common_plot.plot_multi(norm_s, title="Normalized Electropherograms", xlab="time", ylab="intensity (normalized)", offset=off, asinh=ash, line_width=2)
        return fig

    def _get_plot_align_anchors(self, offset=None):
        ctrl = self.controllers.get("alignment")
        if not ctrl:
            return None
        if offset is not None and hasattr(ctrl, "offset_slider") and hasattr(ctrl, "_build_selection_figure"):
            try:
                ctrl.offset_slider.value = offset
                ctrl._build_selection_figure()
                ctrl._refresh_anchor_dots()
            except Exception:
                pass
        fig = extract_bokeh_figure(getattr(ctrl, "selection_pane", getattr(ctrl, "anchor_plot_pane", None)))
        return fig

    def _get_plot_align_aligned(self, offset=None):
        ctrl = self.controllers.get("alignment")
        if not ctrl:
            return None
        al_s = getattr(ctrl, "aligned_by_sample", None)
        ash = getattr(getattr(ctrl, "aligned_asinh_toggle", None), "value", True)
        off = offset if offset is not None else getattr(getattr(ctrl, "aligned_offset_slider", None), "value", 0.0)

        if offset is not None and al_s:
            return common_plot.plot_multi(al_s, title="Aligned Chromatograms", xlab="pseudotime", ylab="intensity (asinh)" if ash else "intensity (raw)", offset=off, asinh=ash, line_width=2)

        fig = extract_bokeh_figure(getattr(ctrl, "aligned_pane", getattr(ctrl, "aligned_plot_pane", None)))
        if fig is None and al_s:
            return common_plot.plot_multi(al_s, title="Aligned Chromatograms", xlab="pseudotime", ylab="intensity (asinh)" if ash else "intensity (raw)", offset=off, asinh=ash, line_width=2)
        return fig

    def _get_plot_nmf_recon(self):
        ctrl = self.controllers.get("nmf")
        if not ctrl:
            return None
        pane = getattr(ctrl, "recon_pane", getattr(ctrl, "recon_plot_pane", None))
        fig = extract_bokeh_figure(pane)
        if fig is not None:
            return fig
        if hasattr(ctrl, "last_recon_fig") and ctrl.last_recon_fig is not None:
            return ctrl.last_recon_fig
        if getattr(ctrl, "H_df", None) is not None and getattr(ctrl, "Phi", None) is not None:
            try:
                ctrl._render_recon_preview()
                return extract_bokeh_figure(getattr(ctrl, "recon_pane", None))
            except Exception:
                pass
        return None

    def _get_plot_nmf_heatmap(self):
        ctrl = self.controllers.get("nmf")
        if not ctrl:
            return None
        pane = getattr(ctrl, "heatmap_pane", getattr(ctrl, "heatmap_plot_pane", None))
        fig = extract_bokeh_figure(pane)
        if fig is not None:
            return fig
        if hasattr(ctrl, "last_heatmap_fig") and ctrl.last_heatmap_fig is not None:
            return ctrl.last_heatmap_fig
        H_df = getattr(ctrl, "H_df", None)
        if H_df is not None and not H_df.empty:
            try:
                import CEtools as cet
                from common_plot import get_continuous_palette
                cmap = get_continuous_palette(256)
                centers = getattr(ctrl, "centers", None)
                if centers is not None:
                    _, _, _, fig = cet.plot_loadings_heatmap_clustered_bokeh(H_df, centers=centers, title="NMF loadings (clustered rows)", cmap=cmap)
                else:
                    fig, _ = cet.plot_loadings_heatmap_bokeh(H_df, cmap=cmap)
                return fig
            except Exception:
                pass
        return None

    def _get_plot_nmf_reg_matrix(self):
        ctrl = self.controllers.get("nmf_meta")
        if ctrl:
            if hasattr(ctrl, "last_reg_fig") and ctrl.last_reg_fig is not None:
                return ctrl.last_reg_fig
            figs = find_all_bokeh_figures(getattr(ctrl, "basis_reg_container", None))
            if figs:
                return figs[0]

        # Generate on demand from H_df if not yet generated
        H_df, _, _, _ = self._get_nmf_and_meta_data()
        if H_df is not None and not H_df.empty:
            cols = [c for c in H_df.columns if c not in ["Unnamed: 0", "_sid", "sample", "sample_id"]]
            if len(cols) >= 2:
                data = []
                for i, v1 in enumerate(cols):
                    for j, v2 in enumerate(cols):
                        if i > j:
                            df_pair = H_df[[v1, v2]].dropna()
                            if len(df_pair) < 3:
                                data.append({'var1': i+1, 'var2': j+1, 'name1': v1, 'name2': v2, 'r': np.nan, 'p': np.nan})
                                continue
                            try:
                                res = linregress(df_pair[v2], df_pair[v1])
                                data.append({'var1': i+1, 'var2': j+1, 'name1': v1, 'name2': v2, 'r': res.rvalue, 'p': res.pvalue})
                            except Exception:
                                data.append({'var1': i+1, 'var2': j+1, 'name1': v1, 'name2': v2, 'r': np.nan, 'p': np.nan})
                plot_df = pd.DataFrame(data)
                cmap = bokeh.models.LinearColorMapper(palette=common_plot.get_divergent_palette(256), low=-1.0, high=1.0, nan_color="lightgray")
                min_var, max_var = 1, len(cols)
                p = bkp.figure(
                    title="Basis Regression Matrix (r-value)",
                    x_range=(0.5, max_var - 0.5), y_range=(max_var + 0.5, 1.5),
                    x_axis_location="above", width=850, height=600,
                    x_axis_label="Basis #", y_axis_label="Basis #",
                    tools="hover,save,pan,wheel_zoom,box_zoom,reset", toolbar_location="right"
                )
                p.rect(x="var2", y="var1", width=1, height=1, source=plot_df,
                       line_color=None, fill_color={"field": "r", "transform": cmap})
                hover = p.select_one(bokeh.models.HoverTool)
                if hover:
                    hover.tooltips = [("Pair", "@name1 vs @name2"), ("r", "@r{0.000}"), ("p-value", "@p{0.00e-0}")]
                cbar = bokeh.models.ColorBar(color_mapper=cmap, width=8, location=(0,0))
                p.add_layout(cbar, 'right')
                return p
        return None

    def _get_alpha_div_table(self):
        ctrl = self.controllers.get("diversity")
        if not ctrl:
            return None
        df = getattr(ctrl, "metrics_df", None)
        if df is None or (hasattr(df, "empty") and df.empty):
            tab_val = getattr(getattr(ctrl, "table", None), "value", None)
            if isinstance(tab_val, pd.DataFrame) and not tab_val.empty:
                df = tab_val
        if df is None or (hasattr(df, "empty") and df.empty):
            return None
        return {
            "type": "alpha_table",
            "metrics_df": df.copy(),
        }

    def _format_alpha_div_table_html(self, alpha_data: Dict[str, Any]) -> str:
        df = alpha_data.get("metrics_df")
        if df is None or df.empty:
            return "<p style='color: #64748b; font-style: italic;'>No alpha diversity metrics available.</p>"

        table_df = df.reset_index() if "sample" not in df.columns and "Sample" not in df.columns else df.copy()
        if "sample" in table_df.columns:
            table_df = table_df.rename(columns={"sample": "Sample"})
        elif "index" in table_df.columns:
            table_df = table_df.rename(columns={"index": "Sample"})

        n_samples = len(table_df)
        metric_cols = [c for c in table_df.columns if c != "Sample"]

        summary_header = f"""
        <div style="display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 12px 16px; flex: 1; min-width: 220px;">
                <div style="font-size: 0.8rem; color: #166534; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Alpha Diversity Cultures / Samples</div>
                <div style="font-size: 1.15rem; color: #14532d; font-weight: 700;">{n_samples} Samples</div>
                <div style="font-size: 0.8rem; color: #22c55e; margin-top: 4px;">{len(metric_cols)} diversity indices evaluated per sample</div>
            </div>
        </div>
        """

        col_labels = {
            "Sample": "Sample",
            "richness": "Richness (S)",
            "shannon": "Shannon (H')",
            "shannon_effective": "Shannon Effective",
            "simpson_D": "Simpson (D)",
            "gini_simpson": "Gini-Simpson (1-D)",
            "pielou_evenness": "Pielou Evenness (J)",
            "hill_q0": "Hill q0",
            "hill_q1": "Hill q1",
            "hill_q2": "Hill q2",
            "total_auc": "Total AUC",
            "roi_auc": "ROI AUC"
        }

        cols = ["Sample"] + metric_cols
        th_cells = []
        for c in cols:
            align = "left" if c == "Sample" else "right"
            label = col_labels.get(c, c)
            th_cells.append(f"<th style='padding: 10px 14px; font-weight: 600; text-align: {align}; border-bottom: 2px solid #cbd5e1; background: #f8fafc; color: #334155; white-space: nowrap;'>{label}</th>")
        thead_html = f"<thead style='position: sticky; top: 0; z-index: 1;'><tr>{''.join(th_cells)}</tr></thead>"

        tr_rows = []
        for idx, row in table_df.iterrows():
            td_cells = []
            bg = "#ffffff" if idx % 2 == 0 else "#f8fafc"
            for c in cols:
                val = row[c]
                align = "left" if c == "Sample" else "right"
                if c == "Sample":
                    val_str = str(val)
                    bold_style = "font-weight: 600; color: #1e293b;"
                elif pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
                    val_str = "N/A"
                    bold_style = "color: #94a3b8;"
                elif isinstance(val, (int, np.integer)):
                    val_str = f"{val}"
                    bold_style = "color: #334155;"
                elif isinstance(val, (float, np.floating)):
                    if c in ["richness", "hill_q0"]:
                        val_str = f"{val:.0f}"
                    elif abs(val) >= 1000 or (abs(val) < 0.001 and val != 0):
                        val_str = f"{val:.3e}"
                    else:
                        val_str = f"{val:.4f}"
                    bold_style = "color: #334155;"
                else:
                    val_str = str(val)
                    bold_style = "color: #334155;"
                td_cells.append(f"<td style='padding: 8px 14px; text-align: {align}; {bold_style} border-bottom: 1px solid #e2e8f0; white-space: nowrap;'>{val_str}</td>")
            tr_rows.append(f"<tr style='background: {bg};'>{''.join(td_cells)}</tr>")
        tbody_html = f"<tbody>{''.join(tr_rows)}</tbody>"

        table_content = f"""
        {summary_header}
        <div style="overflow-x: auto; max-height: 480px; border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 16px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                {thead_html}
                {tbody_html}
            </table>
        </div>
        """
        return table_content

    def _format_markov_table_html(self, markov_data: Dict[str, Any]) -> str:
        df = markov_data.get("df")
        if df is None or (hasattr(df, "empty") and df.empty):
            return "<p style='color: #64748b; font-style: italic;'>No Markov covariate analysis data available.</p>"

        interpretation = markov_data.get("interpretation", "")
        state_col = markov_data.get("state_col", "State")
        cov_col = markov_data.get("cov_col", "Covariate")

        summary_header = f"""
        <div style="display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 12px 16px; flex: 1; min-width: 220px;">
                <div style="font-size: 0.8rem; color: #1e40af; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Markov Covariate Statistical Analysis</div>
                <div style="font-size: 1.15rem; color: #1e3a8a; font-weight: 700;">{state_col} ~ {cov_col}</div>
                <div style="font-size: 0.8rem; color: #3b82f6; margin-top: 4px;">Statistical evaluation of covariate influence on state transitions</div>
            </div>
        </div>
        """

        interp_html = ""
        if interpretation:
            clean_text = interpretation.replace("### Covariate Statistical Analysis", "").replace("**Statistical Association Findings:**", "").strip()
            clean_text = clean_text.replace("\n", "<br>")
            clean_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', clean_text)
            clean_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', clean_text)
            interp_html = f"""
            <div style="background: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 4px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.9em; color: #1e293b; line-height: 1.5;">
                <div style="font-weight: 700; color: #1e40af; margin-bottom: 6px; text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.05em;">Statistical Association Findings</div>
                {clean_text}
            </div>
            """

        headers = list(df.columns)
        th_cells = []
        for h in headers:
            align = "left" if h in ["From", "To", "Significance"] else "right"
            th_cells.append(f"<th style='padding: 10px 14px; font-weight: 600; text-align: {align}; border-bottom: 2px solid #cbd5e1; background: #f8fafc; color: #334155; white-space: nowrap;'>{h}</th>")
        thead_html = f"<thead style='position: sticky; top: 0; z-index: 1;'><tr>{''.join(th_cells)}</tr></thead>"

        tr_rows = []
        for idx, row in df.iterrows():
            td_cells = []
            bg = "#ffffff" if idx % 2 == 0 else "#f8fafc"
            for h in headers:
                val = row[h]
                align = "left" if h in ["From", "To", "Significance"] else "right"
                if pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
                    val_str = "-"
                    bold_style = "color: #94a3b8;"
                elif isinstance(val, (int, np.integer)):
                    val_str = f"{val}"
                    bold_style = "color: #334155;"
                elif isinstance(val, (float, np.floating)):
                    if h in ["p_value", "FDR_p"]:
                        val_str = f"{val:.3e}" if val < 0.001 else f"{val:.4f}"
                    elif h in ["Odds Ratio", "CI_Lower", "CI_Upper", "Stat"]:
                        val_str = f"{val:.3f}"
                    else:
                        val_str = f"{val:.2f}"
                    bold_style = "color: #334155;"
                else:
                    val_str = str(val)
                    bold_style = "color: #334155;"

                if h == "Significance" and "p < 0.05" in str(val):
                    val_str = f"<span style='background: #dcfce7; color: #166534; font-weight: 600; padding: 2px 6px; border-radius: 4px; font-size: 0.85em;'>{val_str}</span>"

                td_cells.append(f"<td style='padding: 8px 14px; text-align: {align}; {bold_style} border-bottom: 1px solid #e2e8f0; white-space: nowrap;'>{val_str}</td>")
            tr_rows.append(f"<tr style='background: {bg};'>{''.join(td_cells)}</tr>")
        tbody_html = f"<tbody>{''.join(tr_rows)}</tbody>"

        table_content = f"""
        {summary_header}
        {interp_html}
        <div style="overflow-x: auto; max-height: 480px; border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 16px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                {thead_html}
                {tbody_html}
            </table>
        </div>
        """
        return table_content

    def _get_plot_diversity(self):
        ctrl = self.controllers.get("diversity")
        if not ctrl:
            return None
        figs = find_all_bokeh_figures(getattr(ctrl, "plots_column", None))
        if figs:
            return figs[0]
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None))

    def _get_plot_viz_scree(self):
        ctrl = self.controllers.get("viz")
        if not ctrl:
            return None
        fig = extract_bokeh_figure(getattr(ctrl, "pca_scree_pane", None))
        if fig is not None:
            return fig
        fig = extract_bokeh_figure(getattr(ctrl, "scree_plot_pane", None))
        if fig is not None:
            return fig
        pca = getattr(ctrl, "last_pca", None)
        if pca is None and hasattr(ctrl, "H_df") and ctrl.H_df is not None and not ctrl.H_df.empty:
            try:
                import CEtools as cet
                from sklearn.decomposition import PCA
                valid_samples = ctrl._get_valid_samples() if hasattr(ctrl, "_get_valid_samples") else ctrl.H_df.index
                H_sub = ctrl.H_df.loc[ctrl.H_df.index.isin(valid_samples)].copy()
                if not H_sub.empty:
                    X = cet.prepare_features(H_sub, row_norm="l1", zscore_cols=True)
                    n_comp = min(X.shape[0], X.shape[1], 10)
                    if n_comp >= 1:
                        pca = PCA(n_components=n_comp, random_state=0)
                        pca.fit(X)
                        ctrl.last_pca = pca
            except Exception:
                pca = None
        if pca is not None:
            try:
                from CEtools import pca_viz as pv
                return pv.plot_pca_scree_bokeh(pca)
            except Exception:
                pass
        return None

    def _get_plot_viz_pc_loadings_1_3(self):
        ctrl = self.controllers.get("viz")
        if not ctrl:
            return None
        pca = getattr(ctrl, "last_pca", None)
        if pca is None and hasattr(ctrl, "H_df") and ctrl.H_df is not None and not ctrl.H_df.empty:
            try:
                import CEtools as cet
                from sklearn.decomposition import PCA
                valid_samples = ctrl._get_valid_samples() if hasattr(ctrl, "_get_valid_samples") else ctrl.H_df.index
                H_sub = ctrl.H_df.loc[ctrl.H_df.index.isin(valid_samples)].copy()
                if not H_sub.empty:
                    X = cet.prepare_features(H_sub, row_norm="l1", zscore_cols=True)
                    n_comp = min(X.shape[0], X.shape[1], 10)
                    if n_comp >= 1:
                        pca = PCA(n_components=n_comp, random_state=0)
                        pca.fit(X)
                        ctrl.last_pca = pca
            except Exception:
                pca = None
        if pca is None:
            return None
        try:
            from CEtools import pca_viz as pv
            from bokeh.layouts import column
            figs = []
            W = np.asarray(pca.components_, dtype=float)
            n_pcs = min(3, W.shape[0])
            for pc in range(1, n_pcs + 1):
                _, p = pv.plot_pc_loadings_bar_bokeh(pca, pc=pc, kind="line", height=220, width=800)
                p.sizing_mode = "stretch_width"
                figs.append(p)
            if figs:
                return column(children=figs, sizing_mode="stretch_width")
        except Exception:
            return None
        return None

    def _get_plot_viz_pca(self):
        ctrl = self.controllers.get("viz")
        fig = extract_bokeh_figure(getattr(ctrl, "pca_pane", None))
        return fig if fig is not None else (extract_bokeh_figure(getattr(ctrl, "pca_plot_pane", None)) if ctrl else None)

    def _get_plot_viz_loadings(self):
        ctrl = self.controllers.get("viz")
        fig = extract_bokeh_figure(getattr(ctrl, "pc_loadings_pane", None))
        return fig if fig is not None else (extract_bokeh_figure(getattr(ctrl, "loadings_plot_pane", None)) if ctrl else None)

    def _get_plot_viz_mds(self):
        ctrl = self.controllers.get("viz")
        fig = extract_bokeh_figure(getattr(ctrl, "mds_pane", None))
        return fig if fig is not None else (extract_bokeh_figure(getattr(ctrl, "mds_plot_pane", None)) if ctrl else None)

    def _get_plot_viz_tsne(self):
        ctrl = self.controllers.get("viz")
        fig = extract_bokeh_figure(getattr(ctrl, "tsne_pane", None))
        return fig if fig is not None else (extract_bokeh_figure(getattr(ctrl, "tsne_plot_pane", None)) if ctrl else None)

    def _get_plot_misc_cvc(self):
        ctrl = self.controllers.get("misc")
        return extract_bokeh_figure(getattr(ctrl, "cvc_plot_pane", None)) if ctrl else None

    def _get_plot_misc_cat(self):
        ctrl = self.controllers.get("misc")
        return extract_bokeh_figure(getattr(ctrl, "cat_plot_pane", None)) if ctrl else None

    def _get_plot_misc_reg(self):
        ctrl = self.controllers.get("misc")
        return extract_bokeh_figure(getattr(ctrl, "reg_plot_pane", None)) if ctrl else None

    def _get_plot_misc_cat_cov(self):
        ctrl = self.controllers.get("misc")
        return extract_bokeh_figure(getattr(ctrl, "cat_cov_plot_pane", None)) if ctrl else None

    def _get_plot_ts_markov(self):
        ctrl = self.controllers.get("timeseries")
        if not ctrl: return None
        fig = extract_bokeh_figure(getattr(ctrl, "markov_plot_pane", None))
        return fig or getattr(ctrl, "last_markov_fig", None)

    def _get_plot_ts_trajectories(self):
        ctrl = self.controllers.get("timeseries")
        if not ctrl: return None
        fig = extract_bokeh_figure(getattr(ctrl, "plot_pane", None))
        if fig is None:
            fig = extract_bokeh_figure(getattr(ctrl, "trajectories_plot_pane", None))
        return fig or getattr(ctrl, "last_traj_fig", None)

    def _get_plot_ts_change(self):
        ctrl = self.controllers.get("timeseries")
        if not ctrl: return None
        fig = extract_bokeh_figure(getattr(ctrl, "change_plot_pane", None))
        return fig or getattr(ctrl, "last_change_fig", None)

    # ---------------- Data Extraction Helpers ----------------
    def _get_nmf_and_meta_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str, Tuple[float, float]]:
        """Retrieve H_df (loadings), metadata_df, sample_col, and roi_range from controllers."""
        nmf_meta_ctrl = self.controllers.get("nmf_meta")
        nmf_ctrl = self.controllers.get("nmf")
        viz_ctrl = self.controllers.get("viz")
        misc_ctrl = self.controllers.get("misc")
        timeseries_ctrl = self.controllers.get("timeseries")

        # 1. H_df
        H_df = getattr(nmf_meta_ctrl, "H_df", None)
        if H_df is None or H_df.empty:
            H_df = getattr(nmf_ctrl, "H_df", None)
        if H_df is not None:
            H_df = H_df.copy()

        # 2. metadata_df
        metadata_df = getattr(nmf_meta_ctrl, "metadata_df", None)
        if (metadata_df is None or metadata_df.empty) and viz_ctrl:
            t = getattr(getattr(viz_ctrl, "meta_table", None), "value", None)
            if t is not None and not t.empty:
                metadata_df = t.copy()
            elif hasattr(viz_ctrl, "metadata_df") and viz_ctrl.metadata_df is not None and not viz_ctrl.metadata_df.empty:
                metadata_df = viz_ctrl.metadata_df.copy()
        if (metadata_df is None or metadata_df.empty) and timeseries_ctrl:
            t = getattr(timeseries_ctrl, "meta_df", None)
            if t is not None and not t.empty:
                metadata_df = t.copy()
        if (metadata_df is None or metadata_df.empty) and misc_ctrl:
            t = getattr(misc_ctrl, "meta_df", None)
            if t is not None and not t.empty:
                metadata_df = t.copy()

        # 3. sample_col
        sample_col = "sample"
        if nmf_meta_ctrl and hasattr(nmf_meta_ctrl, "sample_col_select") and nmf_meta_ctrl.sample_col_select.value:
            sample_col = nmf_meta_ctrl.sample_col_select.value
        elif metadata_df is not None:
            for candidate in ["sample", "sample_id", "sample_ID", "Sample", "SampleID", "id"]:
                if candidate in metadata_df.columns:
                    sample_col = candidate
                    break
            else:
                sample_col = metadata_df.columns[0]

        # 4. roi_range
        roi_range = (0.0, 1.0)
        if nmf_meta_ctrl and hasattr(nmf_meta_ctrl, "roi_lo") and hasattr(nmf_meta_ctrl, "roi_hi"):
            try:
                roi_range = (float(nmf_meta_ctrl.roi_lo.value), float(nmf_meta_ctrl.roi_hi.value))
            except Exception:
                pass

        return H_df, metadata_df, sample_col, roi_range

    def _get_available_categorical_columns(self) -> List[str]:
        _, metadata_df, sample_col, _ = self._get_nmf_and_meta_data()
        if metadata_df is None or metadata_df.empty:
            return []
        cols = []
        for col in metadata_df.columns:
            if col in {sample_col, "_sid", "Unnamed: 0"}:
                continue
            n_unique = metadata_df[col].dropna().nunique()
            if 1 < n_unique <= 100:
                cols.append(col)
        return cols

    # ---------------- Refresh & UI Builders ----------------
    def refresh_available_plots(self):
        """Scan controllers to see which plots are ready and update UI controls."""
        self._standard_plot_widgets.clear()
        categories: Dict[str, List[Dict[str, Any]]] = {}

        ready_count = 0
        for item in self.plot_catalog:
            fig = None
            try:
                fig = item["getter"]()
            except Exception:
                fig = None

            is_ready = fig is not None
            if is_ready:
                ready_count += 1

            cat = item["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "item": item,
                "is_ready": is_ready
            })

        card_objects = []
        for cat_name, cat_items in categories.items():
            header = pn.pane.Markdown(f"#### {cat_name}", margin=(8, 0, 4, 0))
            rows = []
            for entry in cat_items:
                it = entry["item"]
                ready = entry["is_ready"]
                badge_style = "color: #166534; background: #dcfce7; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;" if ready else "color: #64748b; background: #f1f5f9; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;"
                badge_txt = "[Ready]" if ready else "[Not Generated]"
                badge = pn.pane.HTML(f"<span style='{badge_style}'>{badge_txt}</span>", width=110, margin=(10, 0, 0, 0))

                chk = pn.widgets.Checkbox(
                    name=it["default_title"],
                    value=ready and it.get("default_selected", False),
                    disabled=not ready,
                    width=300
                )
                caption_inp = pn.widgets.TextInput(
                    placeholder="Optional caption...",
                    value="",
                    sizing_mode="stretch_width",
                    disabled=not (ready and it.get("default_selected", False))
                )
                chk.param.watch(lambda e, ci=caption_inp, r=ready: setattr(ci, "disabled", not (e.new and r)), "value")

                offset_slider = None
                if it.get("has_offset"):
                    cur_off = 0.0
                    if "get_offset" in it:
                        try:
                            v = it["get_offset"]()
                            if v is not None and isinstance(v, (int, float)):
                                cur_off = float(v)
                        except Exception:
                            cur_off = 0.0
                    max_off = 1.0 if it["id"] == "preproc_norm" else 10.0
                    step_off = 0.05 if it["id"] == "preproc_norm" else 0.5
                    offset_slider = pn.widgets.FloatSlider(
                        name="Vertical offset",
                        start=0.0,
                        end=max_off,
                        step=step_off,
                        value=cur_off,
                        width=180,
                        disabled=not (ready and it.get("default_selected", False))
                    )
                    chk.param.watch(lambda e, sl=offset_slider: setattr(sl, "disabled", not e.new), "value")

                self._standard_plot_widgets[it["id"]] = {
                    "checkbox": chk,
                    "title_input": caption_inp,
                    "caption_input": caption_inp,
                    "offset_slider": offset_slider,
                    "getter": it["getter"],
                    "category": cat_name,
                    "is_ready": ready
                }
                row_items = [chk, badge, caption_inp]
                if offset_slider is not None:
                    row_items.append(offset_slider)
                rows.append(pn.Row(*row_items, sizing_mode="stretch_width", margin=(2, 0, 2, 0)))

            card = pn.Column(header, *rows, margin=(0, 0, 10, 0))
            card_objects.append(card)

        self.standard_plots_layout.objects = card_objects
        self.plots_summary_md.object = f"**Status:** Found **{ready_count}** ready plots out of {len(self.plot_catalog)} standard pipeline plots."

        # Refresh NMF Group Statistics options
        self._update_nmf_group_section_options()

        # Update sub-plot dropdown options
        self._update_dropdowns()

        # Sync any plots sent from pipeline tabs (Misc, Viz, Diversity, TimeSeries)
        for ctrl_name, cat_default in [
            ("misc", "Miscellaneous Analyses"),
            ("viz", "Dimensionality Reduction"),
            ("diversity", "Alpha Diversity"),
            ("timeseries", "Time Series & Pseudotime")
        ]:
            ctrl = self.controllers.get(ctrl_name)
            if ctrl and hasattr(ctrl, "html_plots"):
                existing_ids = {p["id"] for p in self.additional_pipeline_plots}
                for p in ctrl.html_plots:
                    if p["id"] not in existing_ids:
                        entry = {
                            "id": p["id"],
                            "type": p.get("type", "custom_plot"),
                            "category": p.get("category", cat_default),
                            "title": p["title"],
                            "caption": p.get("caption", ""),
                            "figure": p.get("figure"),
                            "details": f"{p.get('category', cat_default)} ({p['title']})"
                        }
                        if "alpha_table_data" in p:
                            entry["alpha_table_data"] = p["alpha_table_data"]
                        if "markov_table_data" in p:
                            entry["markov_table_data"] = p["markov_table_data"]
                        self.additional_pipeline_plots.append(entry)
        self._render_additional_plots_list()

    def _set_all_selected(self, select: bool):
        for data in self._standard_plot_widgets.values():
            if data["is_ready"]:
                data["checkbox"].value = select

    def _update_dropdowns(self):
        """Populate sample reconstruction, basis selection, and peak detection dropdowns."""
        state = self.controllers.get("state")
        nmf_ctrl = self.controllers.get("nmf")
        div_ctrl = self.controllers.get("diversity")

        samples: List[str] = []
        if state and hasattr(state, "current_by_sample") and state.current_by_sample:
            samples = list(state.current_by_sample.keys())
        elif nmf_ctrl and hasattr(nmf_ctrl, "H_df") and nmf_ctrl.H_df is not None and not nmf_ctrl.H_df.empty:
            df = nmf_ctrl.H_df
            scol = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
            samples = df[scol].astype(str).tolist()

        if samples:
            self.sample_recon_select.options = samples
            if not self.sample_recon_select.value or self.sample_recon_select.value not in samples:
                self.sample_recon_select.value = samples[0]

        div_samples: List[str] = []
        if div_ctrl and hasattr(div_ctrl, "sample_peaks_all") and div_ctrl.sample_peaks_all:
            div_samples = list(div_ctrl.sample_peaks_all.keys())
        elif div_ctrl and hasattr(div_ctrl, "samples") and div_ctrl.samples:
            div_samples = list(div_ctrl.samples)
        elif samples:
            div_samples = samples

        if div_samples:
            self.alpha_peak_sample_select.options = div_samples
            if not self.alpha_peak_sample_select.value or self.alpha_peak_sample_select.value not in div_samples:
                self.alpha_peak_sample_select.value = div_samples[0]

        self._update_viz_color_options()

    def _auto_update_viz_title(self):
        m = getattr(self.viz_plot_method, "value", "PCA")
        cvar = (getattr(self.viz_color_var, "value", "") or "").strip()
        c_suffix = f"_coloredBy{cvar}" if cvar and cvar != "None" else ""
        if m == "PCA":
            x_val = getattr(self.viz_pca_x, "value", 1)
            y_val = getattr(self.viz_pca_y, "value", 2)
            self.viz_plot_title.value = f"PCA_PC{x_val}{y_val}{c_suffix}"
        elif m in ("PLS-DA", "PLS"):
            x_val = getattr(self.viz_pls_x, "value", 1)
            y_val = getattr(self.viz_pls_y, "value", 2)
            self.viz_plot_title.value = f"PLS_LV{x_val}{y_val}{c_suffix}"
        elif m == "MDS":
            self.viz_plot_title.value = f"MDS_2D{c_suffix}"
        elif m in ("t-SNE", "tSNE"):
            self.viz_plot_title.value = f"tSNE_2D{c_suffix}"

    def _update_viz_color_options(self):
        source = getattr(self.viz_color_source, "value", "Metadata Variable")
        _, metadata_df, sample_col, _ = self._get_nmf_and_meta_data()
        div_ctrl = self.controllers.get("diversity")
        viz_ctrl = self.controllers.get("viz")

        if source == "Metadata Variable":
            opts = []
            if metadata_df is not None and not metadata_df.empty:
                opts = [c for c in metadata_df.columns if c not in [sample_col, "_sid", "Unnamed: 0"]]
            elif viz_ctrl and hasattr(viz_ctrl, "metadata_df") and viz_ctrl.metadata_df is not None:
                opts = [c for c in viz_ctrl.metadata_df.columns if c not in ["sample", "_sid", "Unnamed: 0"]]
            self.viz_color_var.options = opts
            if opts and (not self.viz_color_var.value or self.viz_color_var.value not in opts):
                self.viz_color_var.value = opts[0]
            if getattr(self.viz_plot_method, "value", "PCA") != "PLS-DA":
                self.viz_color_mode.value = "Categorical"
        else:
            div_df = getattr(div_ctrl, "metrics_df", None)
            if div_df is None or (hasattr(div_df, "empty") and div_df.empty):
                div_df = getattr(viz_ctrl, "diversity_df", None)
            opts = []
            if div_df is not None and not div_df.empty:
                opts = [c for c in div_df.columns if c not in ["sample", "Sample", "index"]]
            self.viz_color_var.options = opts
            if opts and (not self.viz_color_var.value or self.viz_color_var.value not in opts):
                self.viz_color_var.value = opts[0]
            if getattr(self.viz_plot_method, "value", "PCA") != "PLS-DA":
                self.viz_color_mode.value = "Continuous"
        self._auto_update_viz_title()

    # ---------------- NMF Group Statistics Section Management ----------------
    def _update_nmf_group_section_options(self):
        """Update available categorical variables and initialize default section if available."""
        H_df, metadata_df, sample_col, _ = self._get_nmf_and_meta_data()
        avail_cols = self._get_available_categorical_columns()
        self.metadata_var_select.options = avail_cols

        if metadata_df is not None and not metadata_df.empty:
            n_samples = len(metadata_df)
            n_cols = len(metadata_df.columns)
            self.metadata_status_md.object = f"**Metadata Loaded:** {n_samples} samples, {n_cols} columns (Sample ID column: `'{sample_col}'`)."
            self.add_category_btn.disabled = len(avail_cols) == 0
        else:
            self.metadata_status_md.object = "*Metadata not loaded yet. Load metadata in the Upload, NMF Group Stats, or Visualization tab.*"
            self.add_category_btn.disabled = True

        if avail_cols and not self.metadata_var_select.value:
            self.metadata_var_select.value = avail_cols[0]

        # Auto-initialize section from NMF meta dashboard if available and empty
        if not self.nmf_group_sections and metadata_df is not None and H_df is not None:
            nmf_meta_ctrl = self.controllers.get("nmf_meta")
            active_cat = getattr(getattr(nmf_meta_ctrl, "group_col_select", None), "value", None)
            if active_cat and active_cat in avail_cols:
                self._add_category_section(active_cat)

    def _on_add_category_section(self):
        col_name = self.metadata_var_select.value
        if not col_name:
            self.report_status.object = _warn("Please select a categorical variable to add.")
            return
        self._add_category_section(col_name)

    def _add_category_section(self, col_name: str):
        for s in self.nmf_group_sections:
            if s["col_name"] == col_name:
                self.report_status.object = _warn(f"Variable '{col_name}' is already added to NMF Group Statistics.")
                return

        include_recon = pn.widgets.Checkbox(name="Sample Reconstruction by Group (asinh)", value=True, width=320)
        include_sig = pn.widgets.Checkbox(name="NMF Basis Significance Plot (q-value)", value=True, width=320)
        include_pval = pn.widgets.Checkbox(name="NMF Basis Significance Plot (p-value)", value=False, width=320)
        include_loadings = pn.widgets.Checkbox(name="Include Basis Loadings", value=False, width=200)
        loadings_basis_mode = pn.widgets.Select(
            name="Basis Loadings Selection",
            options=["Most Significant Basis", "Top 3 Bases", "All Bases"],
            value="Most Significant Basis",
            width=210,
            visible=False
        )
        custom_title = pn.widgets.TextInput(
            name="Section Title",
            value=f"NMF Group Analysis: {col_name}",
            sizing_mode="stretch_width"
        )

        def _toggle_loadings(event):
            loadings_basis_mode.visible = event.new
        include_loadings.param.watch(_toggle_loadings, "value")

        entry = {
            "col_name": col_name,
            "include_recon": include_recon,
            "include_sig": include_sig,
            "include_pval": include_pval,
            "include_loadings": include_loadings,
            "loadings_basis_mode": loadings_basis_mode,
            "custom_title": custom_title
        }
        self.nmf_group_sections.append(entry)
        self._render_nmf_group_sections_ui()
        self.report_status.object = _ok(f"Added NMF Group Statistics section for '{col_name}'.")

    def _remove_category_section(self, col_name: str):
        self.nmf_group_sections = [s for s in self.nmf_group_sections if s["col_name"] != col_name]
        self._render_nmf_group_sections_ui()
        self.report_status.object = _ok(f"Removed variable '{col_name}'.")

    def _render_nmf_group_sections_ui(self):
        if not self.nmf_group_sections:
            self.nmf_group_sections_container.objects = [
                pn.pane.Markdown("*No categorical variables added yet. Select a variable above and click 'Add Variable Section'.*")
            ]
            return

        cards = []
        for idx, entry in enumerate(self.nmf_group_sections):
            col_name = entry["col_name"]
            del_btn = pn.widgets.Button(name="Remove Variable Section", button_type="default", width=190)
            del_btn.on_click(lambda _, c=col_name: self._remove_category_section(c))

            card = pn.Column(
                pn.pane.Markdown(f"#### Variable: **{col_name}**"),
                entry["custom_title"],
                pn.Row(entry["include_recon"], entry["include_sig"], sizing_mode="stretch_width"),
                pn.Row(entry["include_pval"], entry["include_loadings"], entry["loadings_basis_mode"], sizing_mode="stretch_width"),
                pn.Row(del_btn),
                sizing_mode="stretch_width",
                margin=(5, 0, 12, 0),
                styles={"background": "#f8fafc", "padding": "12px", "border": "1px solid #cbd5e1", "border-radius": "6px"}
            )
            cards.append(card)

        self.nmf_group_sections_container.objects = cards

    # ---------------- Sub-Plot Addition Handlers ----------------
    def _on_add_sample_reconstruction(self, _=None):
        sample = self.sample_recon_select.value
        if not sample:
            self.report_status.object = _warn("Please select a sample for reconstruction.")
            return

        state = self.controllers.get("state")
        nmf_ctrl = self.controllers.get("nmf")
        H_df = getattr(nmf_ctrl, "H_df", None)
        if H_df is None or H_df.empty:
            self.report_status.object = _warn("NMF Loadings (H_df) not found. Please run NMF first.")
            return

        curves_dict = getattr(state, "current_by_sample", {})
        if not curves_dict or sample not in curves_dict:
            self.report_status.object = _warn(f"Electropherogram curve data for sample '{sample}' not found in active session.")
            return

        use_asinh = self.sample_recon_asinh.value
        title = self.sample_recon_title.value or f"Reconstruction: {sample}"
        caption = self.sample_recon_caption.value

        try:
            import CEtools as cet
            fig = cet.plot_reconstruction_overlays_bokeh(
                curves={sample: curves_dict[sample]},
                H_df=H_df,
                use_asinh=use_asinh,
                title=title
            )
            fig.sizing_mode = "stretch_width"

            item_id = f"sub_recon_{sample}_{len(self.additional_pipeline_plots)}"
            self.additional_pipeline_plots.append({
                "id": item_id,
                "type": "sample_reconstruction",
                "title": title,
                "caption": caption,
                "figure": fig,
                "details": f"Sample: {sample} (asinh={use_asinh})"
            })
            self._render_additional_plots_list()
            self.report_status.object = _ok(f"Added reconstruction plot for '{sample}' to report.")
        except Exception as e:
            self.report_status.object = _warn(f"Failed to generate reconstruction: {e}")

    def _on_add_alpha_peak_plot(self, _=None):
        sample = self.alpha_peak_sample_select.value
        if not sample:
            self.report_status.object = _warn("Please select a sample for peak detection.")
            return

        div_ctrl = self.controllers.get("diversity")
        if not div_ctrl:
            self.report_status.object = _warn("Alpha diversity controller not available.")
            return

        use_asinh = self.alpha_peak_asinh.value
        title = self.alpha_peak_title.value or f"Peak Detection: {sample}"
        caption = self.alpha_peak_caption.value

        fig = None
        if hasattr(div_ctrl, "get_sample_peak_plot"):
            try:
                fig = div_ctrl.get_sample_peak_plot(sample, use_asinh=use_asinh)
            except Exception as e:
                self.report_status.object = _warn(f"Failed to generate peak detection plot: {e}")
                return

        if fig is None:
            self.report_status.object = _warn(f"Peak detection data for sample '{sample}' not found. Please run 'Compute alpha diversity' in the Alpha Diversity tab first.")
            return

        fig.sizing_mode = "stretch_width"
        safe_s = re.sub(r'[^a-zA-Z0-9_]', '_', sample)
        item_id = f"sub_alpha_peak_{safe_s}_{len(self.additional_pipeline_plots)}"
        self.additional_pipeline_plots.append({
            "id": item_id,
            "type": "alpha_peak_detection",
            "category": "Alpha Diversity",
            "title": title,
            "caption": caption,
            "figure": fig,
            "details": f"Sample: {sample} (asinh={use_asinh})"
        })
        self._render_additional_plots_list()
        self.report_status.object = _ok(f"Added peak detection plot for '{sample}' to report.")

    def _on_add_viz_plot(self, _=None):
        method = self.viz_plot_method.value
        color_source = self.viz_color_source.value
        color_var = self.viz_color_var.value
        color_mode = self.viz_color_mode.value
        title = self.viz_plot_title.value or f"{method} Score Plot"
        caption = self.viz_plot_caption.value

        try:
            fig = self._compute_custom_viz_plot(
                method=method,
                color_source=color_source,
                color_var=color_var,
                color_mode=color_mode,
                pca_x=self.viz_pca_x.value,
                pca_y=self.viz_pca_y.value,
                pls_x=self.viz_pls_x.value,
                pls_y=self.viz_pls_y.value,
                mds_metric=self.viz_mds_metric.value,
                tsne_metric=self.viz_tsne_metric.value,
                tsne_perp=self.viz_tsne_perp.value,
                title=title
            )
            if fig is None:
                self.report_status.object = _warn(f"Failed to generate {method} plot.")
                return

            safe_method = re.sub(r'[^a-zA-Z0-9_]', '_', method.lower())
            item_id = f"sub_viz_{safe_method}_{len(self.additional_pipeline_plots)}"
            c_desc = f"{color_var} ({color_mode})" if color_var else "default"
            self.additional_pipeline_plots.append({
                "id": item_id,
                "type": "viz_plot",
                "category": "Dimensionality Reduction",
                "title": title,
                "caption": caption,
                "figure": fig,
                "details": f"{method} colored by {c_desc}"
            })
            self._render_additional_plots_list()
            self.report_status.object = _ok(f"Added {method} plot to report.")
        except Exception as e:
            self.report_status.object = _warn(f"Failed to generate {method} plot: {e}")

    def add_pipeline_plot(self, plot_dict: Dict[str, Any]):
        """Receive a plot sent from any tab (Misc, Viz, Diversity, TimeSeries) via 'Send to HTML'."""
        fig = plot_dict.get("figure")
        has_table = ("alpha_table_data" in plot_dict) or ("markov_table_data" in plot_dict)
        if fig is None and not has_table:
            return
        category = plot_dict.get("category", "Miscellaneous Analyses")
        item_id = plot_dict.get("id") or f"sub_{re.sub(r'[^a-zA-Z0-9_]', '_', category.lower())}_{len(self.additional_pipeline_plots)+1}"
        title = plot_dict.get("title", "Analysis Plot")
        caption = plot_dict.get("caption", "")

        for p in self.additional_pipeline_plots:
            if p["id"] == item_id:
                p["figure"] = fig
                p["title"] = title
                p["caption"] = caption
                p["category"] = category
                if "alpha_table_data" in plot_dict:
                    p["alpha_table_data"] = plot_dict["alpha_table_data"]
                if "markov_table_data" in plot_dict:
                    p["markov_table_data"] = plot_dict["markov_table_data"]
                self._render_additional_plots_list()
                self.report_status.object = _ok(f"Updated '{title}' in report ({category}).")
                return

        entry = {
            "id": item_id,
            "type": plot_dict.get("type", "custom_plot"),
            "category": category,
            "title": title,
            "caption": caption,
            "figure": fig,
            "details": f"{category} ({title})"
        }
        if "alpha_table_data" in plot_dict:
            entry["alpha_table_data"] = plot_dict["alpha_table_data"]
        if "markov_table_data" in plot_dict:
            entry["markov_table_data"] = plot_dict["markov_table_data"]
        self.additional_pipeline_plots.append(entry)
        self._render_additional_plots_list()
        self.report_status.object = _ok(f"Added '{title}' to report ({category}).")

    def add_misc_plot(self, plot_dict: Dict[str, Any]):
        self.add_pipeline_plot(plot_dict)

    def _compute_custom_viz_plot(
        self,
        method: str,
        color_source: str,
        color_var: Optional[str],
        color_mode: str,
        pca_x: int = 1,
        pca_y: int = 2,
        pls_x: int = 1,
        pls_y: int = 2,
        mds_metric: str = "euclidean",
        tsne_metric: str = "cosine",
        tsne_perp: float = 30.0,
        title: Optional[str] = None
    ) -> Optional[bokeh.models.LayoutDOM]:
        viz_ctrl = self.controllers.get("viz")
        nmf_ctrl = self.controllers.get("nmf")
        div_ctrl = self.controllers.get("diversity")

        H_df = getattr(viz_ctrl, "H_df", None)
        if H_df is None or H_df.empty:
            H_df = getattr(nmf_ctrl, "H_df", None)
        if H_df is None or H_df.empty:
            raise ValueError("NMF Loadings (H_df) not available. Please run NMF first.")

        H_df = H_df.copy()
        for cand in ["Unnamed: 0", "sample", "sample_id", "_sid"]:
            if cand in H_df.columns:
                H_df = H_df.set_index(cand)
                break

        valid_samples = None
        if viz_ctrl and hasattr(viz_ctrl, "_get_valid_samples"):
            try:
                valid_samples = viz_ctrl._get_valid_samples()
            except Exception:
                valid_samples = None
        if valid_samples:
            H_df = H_df.loc[H_df.index.isin(valid_samples)].copy()
        if H_df.empty:
            raise ValueError("No valid samples found after filtering.")

        import CEtools as cet
        X = cet.prepare_features(H_df, row_norm="l1", zscore_cols=True)
        labels = np.array(list(map(str, H_df.index.astype(str))))

        meta_dict = {}
        _, metadata_df, sample_col, _ = self._get_nmf_and_meta_data()
        if color_source == "Metadata Variable" and color_var:
            if metadata_df is not None and color_var in metadata_df.columns:
                m_df = metadata_df.copy()
                m_df["_sid"] = _coerce_sid(m_df[sample_col])
                val_map = dict(zip(m_df["_sid"], m_df[color_var]))
                for s in labels:
                    clean_s = _coerce_sid(pd.Series([s])).iloc[0]
                    if clean_s in val_map:
                        meta_dict[s] = val_map[clean_s]
        elif color_source == "Diversity Metric" and color_var:
            div_df = getattr(div_ctrl, "metrics_df", None)
            if div_df is None or (hasattr(div_df, "empty") and div_df.empty):
                div_df = getattr(viz_ctrl, "diversity_df", None)
            if div_df is not None and color_var in div_df.columns:
                div_df = div_df.copy()
                if "sample" in div_df.columns:
                    div_df = div_df.set_index("sample")
                val_map = dict(zip(div_df.index.astype(str), div_df[color_var]))
                for s in labels:
                    if s in val_map:
                        meta_dict[s] = val_map[s]

        cat = (color_mode.lower() == "categorical")

        def _add_ellipses_local(p_fig, x_vals, y_vals, c_vals, cmap):
            for c in set(c_vals):
                if c == "NA": continue
                mask = (np.array(c_vals) == c)
                xm = x_vals[mask]
                ym = y_vals[mask]
                if len(xm) < 3: continue
                try:
                    cov = np.cov(xm, ym)
                    mean_x, mean_y = np.mean(xm), np.mean(ym)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = eigvals.argsort()[::-1]
                    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                    angle = np.arctan2(*eigvecs[:, 0][::-1])
                    width, height = 2 * np.sqrt(5.991 * np.maximum(eigvals, 1e-9))
                    p_fig.ellipse(x=mean_x, y=mean_y, width=width, height=height, angle=angle, 
                                  fill_color=cmap[c], line_color=cmap[c], fill_alpha=0.1, line_alpha=0.6,
                                  legend_label=str(c)[:12])
                except Exception:
                    pass

        from bokeh.palettes import Category10, Magma256

        if method == "PCA":
            from sklearn.decomposition import PCA
            pcx = max(1, int(pca_x))
            pcy = max(1, int(pca_y))
            n_comp = min(X.shape[0], X.shape[1], max(pcx, pcy, 2))
            pca = PCA(n_components=n_comp, random_state=0)
            scores = pca.fit_transform(X)
            ix = min(pcx - 1, scores.shape[1] - 1)
            iy = min(pcy - 1, scores.shape[1] - 1)
            x = scores[:, ix]; y = scores[:, iy]

            p = bkp.figure(
                frame_width=400, frame_height=400,
                match_aspect=False, aspect_ratio=None,
                title=title or f"PCA of NMF loadings (PC{pcx} vs PC{pcy})",
                x_axis_label=f"PC{pcx} {pca.explained_variance_ratio_[ix]*100.0:.1f}%",
                y_axis_label=f"PC{pcy} {pca.explained_variance_ratio_[iy]*100.0:.1f}%",
                tools="hover,pan,box_zoom,reset,save",
                active_scroll=None,
            )

            if cat:
                cats = [str(meta_dict.get(s, "NA")) if meta_dict.get(s, None) is not None else "NA" for s in labels]
                unique = sorted(set(cats))
                palette = list(Category10[10]) if len(unique) <= 10 else [Magma256[i] for i in np.linspace(0, 255, num=len(unique), dtype=int)]
                cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
                for c in unique:
                    mask = (np.array(cats) == c)
                    src = bokeh.models.ColumnDataSource(dict(x=x[mask], y=y[mask], label=labels[mask]))
                    p.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color=cmap[c], source=src, legend_label=str(c)[:12])
                _add_ellipses_local(p, x, y, cats, cmap)
                if p.legend:
                    p.legend.click_policy = "hide"
                    p.add_layout(p.legend[0], "right")
            else:
                vals = []
                for s in labels:
                    v = meta_dict.get(s, None)
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(np.nan)
                vals = np.asarray(vals, dtype=float)
                mapper = bokeh.models.LinearColorMapper(
                    palette=Magma256,
                    low=float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0,
                    high=float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0,
                    nan_color="lightgray"
                )
                src = bokeh.models.ColumnDataSource(dict(x=x, y=y, label=labels, value=vals))
                p.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color={"field": "value", "transform": mapper}, source=src)
                c_label = str(color_var)[:20] if color_var else "Value"
                cbar = bokeh.models.ColorBar(color_mapper=mapper, title=c_label, label_standoff=8, location=(0, 0))
                p.add_layout(cbar, "right")

            p.hover.tooltips = [("sample", "@label"), ("x", "@x{0.000}"), ("y", "@y{0.000}")]
            return p

        elif method == "PLS-DA":
            from sklearn.cross_decomposition import PLSRegression
            cats = [str(meta_dict.get(s, "NA")) if meta_dict.get(s, None) is not None else "NA" for s in labels]
            unique = sorted(set(cats))
            unique_valid = [c for c in unique if c != "NA"]
            if len(unique_valid) < 2:
                raise ValueError("PLS-DA requires a categorical variable with at least 2 distinct classes.")

            cat_to_int = {c: i for i, c in enumerate(unique_valid)}
            y_int = np.array([cat_to_int.get(c, -1) for c in cats])
            valid_mask = (y_int != -1)
            if valid_mask.sum() <= 2:
                raise ValueError("Not enough samples with valid categorical labels for PLS-DA.")

            y_onehot = np.zeros((valid_mask.sum(), len(unique_valid)))
            y_onehot[np.arange(valid_mask.sum()), y_int[valid_mask]] = 1
            lv_x = max(1, int(pls_x))
            lv_y = max(1, int(pls_y))
            max_pls = min(valid_mask.sum(), X.shape[1], len(unique_valid))
            ncom_pls = min(max_pls, max(lv_x, lv_y, 2))

            pls = PLSRegression(n_components=ncom_pls)
            pls.fit(X[valid_mask], y_onehot)
            scores_pls = pls.transform(X)
            ix = min(lv_x - 1, scores_pls.shape[1] - 1)
            iy = min(lv_y - 1, scores_pls.shape[1] - 1)
            pls_x_vals, pls_y_vals = scores_pls[:, ix], scores_pls[:, iy]

            p = bkp.figure(
                frame_width=400, frame_height=400,
                match_aspect=False, aspect_ratio=None,
                title=title or f"PLS-DA of NMF loadings (LV{lv_x} vs LV{lv_y})",
                x_axis_label=f"LV{lv_x}", y_axis_label=f"LV{lv_y}",
                tools="hover,pan,box_zoom,reset,save",
                active_scroll=None,
            )
            palette = list(Category10[10]) if len(unique) <= 10 else [Magma256[i] for i in np.linspace(0, 255, num=len(unique), dtype=int)]
            cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
            for c in unique:
                mask = (np.array(cats) == c)
                src = bokeh.models.ColumnDataSource(dict(x=pls_x_vals[mask], y=pls_y_vals[mask], label=labels[mask]))
                p.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color=cmap[c], source=src, legend_label=str(c)[:12])
            _add_ellipses_local(p, pls_x_vals, pls_y_vals, cats, cmap)
            if p.legend:
                p.legend.click_policy = "hide"
                p.add_layout(p.legend[0], "right")
            p.hover.tooltips = [("sample", "@label"), ("x", "@x{0.000}"), ("y", "@y{0.000}")]
            return p

        elif method == "MDS":
            import bokeh.io as _bio
            old_show = _bio.show
            try:
                _bio.show = lambda *a, **k: None
                coords, _, mds_fig = cet.embed_with_mds(
                    H_df,
                    metric=str(mds_metric),
                    row_norm="l1",
                    zscore_cols=True,
                    metadata=meta_dict if meta_dict else None,
                    metadata_categorical=cat,
                    title=title or f"MDS (metric={mds_metric})"
                )
            finally:
                _bio.show = old_show

            mds_fig.frame_width = 400
            mds_fig.frame_height = 400
            mds_fig.match_aspect = False
            mds_fig.aspect_ratio = None
            mds_fig.sizing_mode = None
            mds_fig.width = None
            mds_fig.height = None
            mds_fig.toolbar.active_scroll = None
            if cat and meta_dict:
                mds_fig.renderers = [r for r in mds_fig.renderers if str(type(r).__name__) != 'GlyphRenderer']
                for leg in mds_fig.legend:
                    for pos in [mds_fig.center, mds_fig.right, mds_fig.left, mds_fig.above, mds_fig.below]:
                        if leg in pos: pos.remove(leg)
                x, y = coords[:, 0], coords[:, 1]
                cats = [str(meta_dict.get(s, "NA")) if meta_dict.get(s, None) is not None else "NA" for s in labels]
                unique = sorted(set(cats))
                palette = list(Category10[10]) if len(unique) <= 10 else [Magma256[i] for i in np.linspace(0, 255, num=len(unique), dtype=int)]
                cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
                for c in unique:
                    mask = (np.array(cats) == c)
                    src = bokeh.models.ColumnDataSource(dict(x=x[mask], y=y[mask], label=labels[mask]))
                    mds_fig.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color=cmap[c], source=src, legend_label=str(c)[:12])
                mds_fig.legend.click_policy = "hide"
                mds_fig.add_layout(mds_fig.legend[-1], "right")
                mds_fig.add_tools(bokeh.models.HoverTool(tooltips=[("sample", "@label"), ("x", "@x{0.000}"), ("y", "@y{0.000}")]))
            return mds_fig

        elif method == "t-SNE":
            import bokeh.io as _bio
            old_show = _bio.show
            try:
                _bio.show = lambda *a, **k: None
                coords, _, tsne_fig = cet.embed_with_tsne(
                    H_df,
                    perplexity=float(tsne_perp),
                    metric=str(tsne_metric),
                    row_norm="l1",
                    zscore_cols=True,
                    random_state=0,
                    metadata=meta_dict if meta_dict else None,
                    metadata_categorical=cat,
                    title=title or f"t-SNE (metric={tsne_metric}, perplexity={tsne_perp:.0f})"
                )
            finally:
                _bio.show = old_show

            tsne_fig.frame_width = 400
            tsne_fig.frame_height = 400
            tsne_fig.match_aspect = False
            tsne_fig.aspect_ratio = None
            tsne_fig.sizing_mode = None
            tsne_fig.width = None
            tsne_fig.height = None
            tsne_fig.toolbar.active_scroll = None
            if cat and meta_dict:
                tsne_fig.renderers = [r for r in tsne_fig.renderers if str(type(r).__name__) != 'GlyphRenderer']
                for leg in tsne_fig.legend:
                    for pos in [tsne_fig.center, tsne_fig.right, tsne_fig.left, tsne_fig.above, tsne_fig.below]:
                        if leg in pos: pos.remove(leg)
                x, y = coords[:, 0], coords[:, 1]
                cats = [str(meta_dict.get(s, "NA")) if meta_dict.get(s, None) is not None else "NA" for s in labels]
                unique = sorted(set(cats))
                palette = list(Category10[10]) if len(unique) <= 10 else [Magma256[i] for i in np.linspace(0, 255, num=len(unique), dtype=int)]
                cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
                for c in unique:
                    mask = (np.array(cats) == c)
                    src = bokeh.models.ColumnDataSource(dict(x=x[mask], y=y[mask], label=labels[mask]))
                    tsne_fig.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color=cmap[c], source=src, legend_label=str(c)[:12])
                tsne_fig.legend.click_policy = "hide"
                tsne_fig.add_layout(tsne_fig.legend[-1], "right")
                tsne_fig.add_tools(bokeh.models.HoverTool(tooltips=[("sample", "@label"), ("x", "@x{0.000}"), ("y", "@y{0.000}")]))
            return tsne_fig

        return None

    def _render_additional_plots_list(self):
        if not self.additional_pipeline_plots:
            self.additional_plots_container.objects = [pn.pane.Markdown("*No additional sub-plots added yet.*")]
            return

        rows = []
        for idx, item in enumerate(self.additional_pipeline_plots):
            lbl = pn.pane.Markdown(f"**{idx+1}. {item['title']}** — _{item['details']}_")
            def _make_remover(i=idx):
                b = pn.widgets.Button(name="Remove", button_type="default", width=90)
                b.on_click(lambda _: self._remove_additional_plot(i))
                return b
            rows.append(pn.Row(lbl, pn.Spacer(sizing_mode="stretch_width"), _make_remover(idx), sizing_mode="stretch_width", margin=(2, 0, 2, 0)))
        self.additional_plots_container.objects = rows

    def _remove_additional_plot(self, index: int):
        if 0 <= index < len(self.additional_pipeline_plots):
            removed = self.additional_pipeline_plots.pop(index)
            rem_id = removed.get("id")
            for ctrl in self.controllers.values():
                if hasattr(ctrl, "html_plots"):
                    ctrl.html_plots = [p for p in ctrl.html_plots if p.get("id") != rem_id]
            self._render_additional_plots_list()
            self.report_status.object = _ok("Removed sub-plot.")

    # ---------------- Custom Image Upload Handlers ----------------
    def _on_add_custom_image(self, _=None):
        if not self.image_file_input.value:
            self.report_status.object = _warn("Please select an image file to upload.")
            return

        raw_bytes = self.image_file_input.value
        filename = self.image_file_input.filename or "figure.png"
        ext = os.path.splitext(filename)[1].lower()

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".webp": "image/webp"
        }
        mime = mime_types.get(ext, "image/png")

        b64 = base64.b64encode(raw_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"

        title = self.image_title_input.value or filename
        caption = self.image_caption_input.value or ""
        target_sec = self.image_section_select.value or "Raw Data & QC"

        self.custom_images.append({
            "id": f"img_{len(self.custom_images)+1}",
            "filename": filename,
            "title": title,
            "caption": caption,
            "data_uri": data_uri,
            "section": target_sec,
            "kind": "image"
        })
        self._render_custom_images_list()
        self.report_status.object = _ok(f"Added image '{filename}' to section '{target_sec}'.")
        self.image_file_input.value = None

    def _render_custom_images_list(self):
        if not self.custom_images:
            self.custom_images_container.objects = [pn.pane.Markdown("*No external images added yet.*")]
            return

        rows = []
        for idx, item in enumerate(self.custom_images):
            lbl = pn.pane.Markdown(
                f"**{idx+1}. {item['title']}** <span style='font-size: 0.75em; font-weight: 500; color: #475569; background: #f1f5f9; padding: 2px 8px; border-radius: 4px; border: 1px solid #e2e8f0; margin-left: 6px;'>{item.get('section', 'Raw Data & QC')}</span> ({item['filename']})"
            )
            def _make_remover(i=idx):
                b = pn.widgets.Button(name="Remove", button_type="default", width=90)
                b.on_click(lambda _: self._remove_custom_image(i))
                return b
            rows.append(pn.Row(lbl, pn.Spacer(sizing_mode="stretch_width"), _make_remover(idx), sizing_mode="stretch_width", margin=(2, 0, 2, 0)))
        self.custom_images_container.objects = rows

    def _remove_custom_image(self, index: int):
        if 0 <= index < len(self.custom_images):
            self.custom_images.pop(index)
            self._render_custom_images_list()
            self.report_status.object = _ok("Removed image.")

    # ---------------- Report HTML Assembly & Generation ----------------
    def _on_build_report(self, _=None):
        self.report_status.object = "Generating HTML report..."
        try:
            html_content, plot_count = self._compile_html_document()
            self._current_html_str = html_content
            self.download_report_btn.disabled = False

            size_kb = len(html_content.encode("utf-8")) / 1024.0
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{(size_kb/1024):.2f} MB"
            self.report_status.object = _ok(f"HTML Report generated successfully! Includes {plot_count} plots/figures ({size_str}). Click 'Download HTML Report' below.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.download_report_btn.disabled = True
            self.report_status.object = _warn(f"Failed to generate report: {e}")

    def _get_report_bytes(self) -> io.BytesIO:
        if not self._current_html_str:
            html_content, _ = self._compile_html_document()
            self._current_html_str = html_content
        return io.BytesIO(self._current_html_str.encode("utf-8"))

    def _compile_html_document(self) -> Tuple[str, int]:
        """Compile selected standard plots, nested NMF group sections, subplots, custom images, and parameters into HTML."""
        # 1. Collect standard plots
        selected_standard_plots = []
        for item in self.plot_catalog:
            pid = item["id"]
            if pid in self._standard_plot_widgets:
                w = self._standard_plot_widgets[pid]
                if w["checkbox"].value and w["is_ready"]:
                    user_caption = w["caption_input"].value.strip() if ("caption_input" in w and w["caption_input"].value) else ""
                    offset_val = w["offset_slider"].value if "offset_slider" in w and w["offset_slider"] is not None else None
                    if offset_val is not None:
                        res = w["getter"](offset=offset_val)
                    else:
                        res = w["getter"]()

                    if res is not None:
                        if isinstance(res, dict) and res.get("type") == "snr_table":
                            selected_standard_plots.append({
                                "id": pid,
                                "category": item["category"],
                                "title": item["default_title"],
                                "snr_data": res,
                                "figure": None,
                                "caption": user_caption
                            })
                        elif isinstance(res, dict) and res.get("type") == "alpha_table":
                            selected_standard_plots.append({
                                "id": pid,
                                "category": item["category"],
                                "title": item["default_title"],
                                "alpha_table_data": res,
                                "figure": None,
                                "caption": user_caption
                            })
                        else:
                            fig_to_use = res
                            if item.get("has_offset") and isinstance(res, bokeh.models.LayoutDOM):
                                max_off = 1.0 if pid == "preproc_norm" else 10.0
                                step_off = 0.05 if pid == "preproc_norm" else 0.5
                                off_val = offset_val if offset_val is not None else 0.0
                                fig_to_use = attach_interactive_offset_slider(res, initial_offset=off_val, max_offset=max_off, step=step_off)

                            selected_standard_plots.append({
                                "id": pid,
                                "category": item["category"],
                                "title": item["default_title"],
                                "figure": fig_to_use,
                                "caption": user_caption
                            })

        # 2. Collect nested NMF Group Statistics
        selected_nmf_groups = []
        H_df, metadata_df, sample_col, roi_range = self._get_nmf_and_meta_data()

        for g_sec in self.nmf_group_sections:
            col_name = g_sec["col_name"]
            inc_recon = g_sec["include_recon"].value
            inc_sig = g_sec["include_sig"].value
            inc_pval = g_sec["include_pval"].value
            inc_loadings = g_sec["include_loadings"].value
            loadings_mode = g_sec["loadings_basis_mode"].value
            title_text = g_sec["custom_title"].value or f"NMF Group Analysis: {col_name}"

            if not (inc_recon or inc_sig or inc_pval or inc_loadings):
                continue

            if metadata_df is None or H_df is None:
                continue

            try:
                g_figs = compute_nmf_group_figures(
                    metadata_df=metadata_df,
                    nmf_loadings_df=H_df,
                    group_col=col_name,
                    meta_sample_col=sample_col,
                    roi_range=roi_range,
                    include_recon=inc_recon,
                    include_sig=inc_sig,
                    include_pval=inc_pval,
                    include_loadings=inc_loadings,
                    loadings_basis_mode=loadings_mode
                )
                selected_nmf_groups.append({
                    "col_name": col_name,
                    "title": title_text,
                    "groups": g_figs.get("groups", []),
                    "recon_fig": g_figs.get("recon_fig"),
                    "sig_fig": g_figs.get("sig_fig"),
                    "pval_fig": g_figs.get("pval_fig"),
                    "jitter_figs": g_figs.get("jitter_figs", {})
                })
            except Exception as e:
                # Log error and continue
                print(f"[HTML Report] Error computing NMF stats for '{col_name}': {e}")

        # Sync any plots sent from pipeline tabs (Misc, Viz, Diversity, TimeSeries)
        for ctrl_name, cat_default in [
            ("misc", "Miscellaneous Analyses"),
            ("viz", "Dimensionality Reduction"),
            ("diversity", "Alpha Diversity"),
            ("timeseries", "Time Series & Pseudotime")
        ]:
            ctrl = self.controllers.get(ctrl_name)
            if ctrl and hasattr(ctrl, "html_plots"):
                existing_ids = {p["id"] for p in self.additional_pipeline_plots}
                for p in ctrl.html_plots:
                    if p["id"] not in existing_ids:
                        entry = {
                            "id": p["id"],
                            "type": p.get("type", "custom_plot"),
                            "category": p.get("category", cat_default),
                            "title": p["title"],
                            "caption": p.get("caption", ""),
                            "figure": p.get("figure"),
                            "details": f"{p.get('category', cat_default)} ({p['title']})"
                        }
                        if "alpha_table_data" in p:
                            entry["alpha_table_data"] = p["alpha_table_data"]
                        if "markov_table_data" in p:
                            entry["markov_table_data"] = p["markov_table_data"]
                        self.additional_pipeline_plots.append(entry)

        # 3. Collect additional subplots
        selected_sub_plots = []
        for sp in self.additional_pipeline_plots:
            fig = sp.get("figure", sp.get("fig"))
            has_table = ("alpha_table_data" in sp) or ("markov_table_data" in sp)
            if fig is not None or has_table:
                p_cat = sp.get("category")
                if p_cat in [
                    "Alpha Diversity",
                    "Dimensionality Reduction",
                    "Visualization",
                    "Miscellaneous Analyses",
                    "Miscellaneous",
                    "Time Series & Pseudotime",
                    "Time Series"
                ]:
                    if p_cat == "Visualization":
                        target_cat = "Dimensionality Reduction"
                    elif p_cat == "Miscellaneous":
                        target_cat = "Miscellaneous Analyses"
                    elif p_cat == "Time Series":
                        target_cat = "Time Series & Pseudotime"
                    else:
                        target_cat = p_cat
                    selected_standard_plots.append({
                        "id": sp["id"],
                        "type": sp.get("type", "custom_plot"),
                        "category": target_cat,
                        "title": sp["title"],
                        "figure": fig,
                        "caption": sp.get("caption", ""),
                        "alpha_table_data": sp.get("alpha_table_data"),
                        "markov_table_data": sp.get("markov_table_data")
                    })
                else:
                    selected_sub_plots.append({
                        "id": sp["id"],
                        "category": "Additional Pipeline Sub-Plots",
                        "title": sp["title"],
                        "figure": fig,
                        "caption": sp.get("caption", ""),
                        "alpha_table_data": sp.get("alpha_table_data"),
                        "markov_table_data": sp.get("markov_table_data")
                    })

        # Count total items
        nmf_group_fig_count = 0
        for grp in selected_nmf_groups:
            if grp.get("recon_fig") is not None: nmf_group_fig_count += 1
            if grp.get("sig_fig") is not None: nmf_group_fig_count += 1
            if grp.get("pval_fig") is not None: nmf_group_fig_count += 1
            nmf_group_fig_count += len(grp.get("jitter_figs", {}))

        total_items = len(selected_standard_plots) + nmf_group_fig_count + len(selected_sub_plots) + len(self.custom_images)
        if total_items == 0:
            raise ValueError("No plots or images selected! Please select at least one standard plot, configure an NMF group section, or add an additional figure.")

        # 4. Prepare Bokeh figures dictionary for components()
        figures_dict: Dict[str, bokeh.models.LayoutDOM] = {}

        # Standard plots
        for sp in selected_standard_plots:
            if sp.get("figure") is not None:
                figures_dict[f"std_{sp['id']}"] = sp["figure"]

        # Nested NMF group plots
        for g_idx, grp in enumerate(selected_nmf_groups):
            safe_c = re.sub(r'[^a-zA-Z0-9_]', '_', grp["col_name"])
            if grp.get("recon_fig") is not None:
                figures_dict[f"nmf_grp_{safe_c}_recon_{g_idx}"] = grp["recon_fig"]
            if grp.get("sig_fig") is not None:
                figures_dict[f"nmf_grp_{safe_c}_sig_{g_idx}"] = grp["sig_fig"]
            if grp.get("pval_fig") is not None:
                figures_dict[f"nmf_grp_{safe_c}_pval_{g_idx}"] = grp["pval_fig"]
            for b_name, j_fig in grp.get("jitter_figs", {}).items():
                safe_b = re.sub(r'[^a-zA-Z0-9_]', '_', b_name)
                figures_dict[f"nmf_grp_{safe_c}_jitter_{safe_b}_{g_idx}"] = j_fig

        # Subplots
        for sp in selected_sub_plots:
            figures_dict[f"sub_{sp['id']}"] = sp["figure"]

        # Render Bokeh scripts and divs safely with document detachment
        set_bokeh_output_backend_svg(figures_dict)
        detach_bokeh_models(figures_dict)
        try:
            script, divs = components(figures_dict) if figures_dict else ("", {})
        finally:
            detach_bokeh_models(figures_dict)

        # 5. Build Session Parameters Table
        params_html = ""
        state = self.controllers.get("state")
        if self.include_params_chk.value and state and hasattr(state, "session_log") and state.session_log:
            params_html = "<div class='section-card' id='sec-params'>"
            params_html += "<h2 class='section-title'>Session Execution Parameters</h2>"
            params_html += "<div class='params-grid'>"

            preproc_keys = [
                "Preprocessing: Despike",
                "Preprocessing: Smooth",
                "Preprocessing: Baseline",
                "Preprocessing: Normalization"
            ]
            preproc_items = {}
            other_items = {}

            for stage, pdict in state.session_log.items():
                if stage in preproc_keys or stage.lower().startswith("preprocessing") or "preproc" in stage.lower():
                    preproc_items[stage] = pdict
                else:
                    other_items[stage] = pdict

            # General information card first if present
            if "General" in other_items:
                params_html += f"<div class='param-card'><h4>General</h4><table class='param-table'>"
                for k, v in other_items["General"].items():
                    params_html += f"<tr><td class='param-key'>{k}</td><td class='param-val'>{v}</td></tr>"
                params_html += "</table></div>"

            # Consolidated Preprocessing card (all preprocessing steps in one box)
            if preproc_items:
                params_html += "<div class='param-card'><h4>Preprocessing</h4>"
                params_html += "<div style='display: flex; flex-direction: column; gap: 12px;'>"
                ordered_stages = [k for k in preproc_keys if k in preproc_items] + [k for k in preproc_items if k not in preproc_keys]
                for stage in ordered_stages:
                    pdict = preproc_items[stage]
                    sub_title = stage.replace("Preprocessing:", "").replace("Preprocessing", "").strip(" :")
                    params_html += f"<div><div style='font-weight: 600; color: #1e293b; font-size: 0.88rem; margin-bottom: 4px; border-bottom: 1px dashed #cbd5e1; padding-bottom: 2px;'>{sub_title}</div>"
                    params_html += "<table class='param-table'>"
                    for k, v in pdict.items():
                        params_html += f"<tr><td class='param-key'>{k}</td><td class='param-val'>{v}</td></tr>"
                    params_html += "</table></div>"
                params_html += "</div></div>"

            # Remaining pipeline stages (Alignment, NMF, Alpha Diversity, etc.)
            for stage, pdict in other_items.items():
                if stage == "General":
                    continue
                params_html += f"<div class='param-card'><h4>{stage}</h4><table class='param-table'>"
                for k, v in pdict.items():
                    params_html += f"<tr><td class='param-key'>{k}</td><td class='param-val'>{v}</td></tr>"
                params_html += "</table></div>"

            params_html += "</div></div>"

        # 6. Group and sort all figures/tables into Canonical Report Sections
        REPORT_SECTIONS = [
            "Raw Data & QC",
            "Preprocessing",
            "Alignment & Anchors",
            "Alpha Diversity Analysis",
            "Dimensionality Reduction",
            "NMF Decomposition",
            "Time Series & Transitions",
            "Miscellaneous Analysis"
        ]

        def normalize_section_name(cat: str) -> str:
            cat_lower = (cat or "").lower().strip()
            if "raw" in cat_lower or "qc" in cat_lower:
                return "Raw Data & QC"
            if "preproc" in cat_lower:
                return "Preprocessing"
            if "align" in cat_lower or "anchor" in cat_lower:
                return "Alignment & Anchors"
            if "alpha" in cat_lower or "diversity" in cat_lower:
                return "Alpha Diversity Analysis"
            if "dim" in cat_lower or "reduct" in cat_lower or "viz" in cat_lower:
                return "Dimensionality Reduction"
            if "nmf" in cat_lower:
                return "NMF Decomposition"
            if "time" in cat_lower or "pseudo" in cat_lower or "trans" in cat_lower or "markov" in cat_lower:
                return "Time Series & Transitions"
            if "misc" in cat_lower:
                return "Miscellaneous Analysis"
            return "Miscellaneous Analysis"

        def get_target_section(item: Dict[str, Any]) -> str:
            if "section" in item and item["section"]:
                return normalize_section_name(item["section"])
            item_id = item.get("id", "")
            item_type = item.get("type", "")
            item_cat = item.get("category", "")
            if item.get("kind") == "nmf_group":
                return "NMF Decomposition"
            if item_id.startswith("sub_recon_") or item_type == "sample_reconstruction":
                return "NMF Decomposition"
            if item_id.startswith("sub_alpha_peak_") or item_type == "alpha_peak":
                return "Alpha Diversity Analysis"
            if item_id.startswith("sub_viz_") or item_type in ("PCA", "PLS-DA", "MDS", "t-SNE") or "Dimensionality Reduction" in item_cat:
                return "Dimensionality Reduction"
            if "Time Series" in item_cat or "Pseudotime" in item_cat or "markov" in item_id:
                return "Time Series & Transitions"
            if item_id.startswith("misc_") or "Miscellaneous" in item_cat:
                return "Miscellaneous Analysis"
            return normalize_section_name(item_cat)

        section_items: Dict[str, List[Dict[str, Any]]] = {s: [] for s in REPORT_SECTIONS}

        for sp in selected_standard_plots:
            sec = get_target_section(sp)
            section_items[sec].append(sp)

        for g_idx, grp in enumerate(selected_nmf_groups):
            safe_c = re.sub(r'[^a-zA-Z0-9_]', '_', grp["col_name"])
            section_items["NMF Decomposition"].append({
                **grp,
                "kind": "nmf_group",
                "id": f"nmf_grp_{safe_c}",
                "safe_c": safe_c,
                "g_idx": g_idx
            })

        for sp in selected_sub_plots:
            sec = get_target_section(sp)
            section_items[sec].append(sp)

        for img in self.custom_images:
            sec = get_target_section(img)
            section_items[sec].append({**img, "kind": "image"})

        # Sort Dimensionality Reduction:
        # 1. Scree, 2. Loadings, 3. All PCA, 4. All PLS, 5. All MDS, 6. All t-SNE, 7. Images
        def dim_sort_key(it):
            iid = it.get("id", "").lower()
            title = it.get("title", "").lower()
            m = (it.get("method") or it.get("type") or "").upper()
            if "scree" in iid or "scree" in title:
                return (0, 0, iid)
            if "loadings" in iid or "loadings" in title:
                return (1, 0, iid)
            if m == "PCA" or "pca" in iid or "pca" in title:
                sub = 1 if iid.startswith("sub_") else 0
                return (2, sub, iid)
            if "pls" in m or "pls" in iid or "pls" in title:
                sub = 1 if iid.startswith("sub_") else 0
                return (3, sub, iid)
            if "mds" in m or "mds" in iid or "mds" in title:
                sub = 1 if iid.startswith("sub_") else 0
                return (4, sub, iid)
            if "tsne" in m or "t-sne" in m or "tsne" in iid or "tsne" in title:
                sub = 1 if iid.startswith("sub_") else 0
                return (5, sub, iid)
            if it.get("kind") == "image":
                return (6, 0, iid)
            return (7, 0, iid)

        section_items["Dimensionality Reduction"].sort(key=dim_sort_key)

        # Sort Miscellaneous:
        # 1. Continuous vs Continuous (cvc), 2. Categorical vs Continuous (cat), 3. Regression & Covariance matrices, 4. Images
        def misc_sort_key(it):
            iid = it.get("id", "").lower()
            title = it.get("title", "").lower()
            if "cvc" in iid or "continuous vs continuous" in title or "cvc:" in title:
                return (0, 0, iid)
            if ("cat" in iid and "cov" not in iid) or "categorical vs continuous" in title or "comparison:" in title:
                return (1, 0, iid)
            if "reg" in iid or "regression" in title:
                return (2, 0, iid)
            if "cov" in iid or "covariance" in title:
                return (2, 1, iid)
            if it.get("kind") == "image":
                return (3, 0, iid)
            return (4, 0, iid)

        section_items["Miscellaneous Analysis"].sort(key=misc_sort_key)

        # Sort NMF Decomposition:
        # 1. Standard NMF plots, 2. NMF Group Statistics, 3. Sample Reconstructions, 4. Images
        def nmf_sort_key(it):
            iid = it.get("id", "").lower()
            if "loadings_heatmap" in iid:
                return (0, 0, iid)
            if "recon_heatmap" in iid:
                return (0, 1, iid)
            if "error" in iid:
                return (0, 2, iid)
            if "basis" in iid:
                return (0, 3, iid)
            if it.get("kind") == "nmf_group":
                return (1, 0, iid)
            if "recon" in iid or it.get("type") == "sample_reconstruction":
                return (2, 0, iid)
            if it.get("kind") == "image":
                return (3, 0, iid)
            return (4, 0, iid)

        section_items["NMF Decomposition"].sort(key=nmf_sort_key)

        # Sort Alpha Diversity:
        # 1. Table, 2. Curves, 3. Distributions, 4. Peak Detections, 5. Images
        def alpha_sort_key(it):
            iid = it.get("id", "").lower()
            if "table" in iid or it.get("alpha_table_data") is not None:
                return (0, 0, iid)
            if "curve" in iid:
                return (1, 0, iid)
            if "distrib" in iid:
                return (2, 0, iid)
            if "alpha_peak" in iid or it.get("type") == "alpha_peak":
                return (3, 0, iid)
            if it.get("kind") == "image":
                return (4, 0, iid)
            return (5, 0, iid)

        section_items["Alpha Diversity Analysis"].sort(key=alpha_sort_key)

        # Sort Time Series & Transitions:
        # 1. Trajectory/pseudotime (first, wide solo)
        # 2. Timepoint change plots
        # 3. Transition probability & transition network & covariate heatmap
        # 4. Markov covariate table & corresponding forest plot (side by side)
        # 5. External images / other
        def ts_sort_key(it):
            iid = it.get("id", "").lower()
            itype = (it.get("type") or "").lower()
            title = it.get("title", "").lower()

            # 1. Trajectories
            if "traj" in iid or "traj" in itype or "trajector" in title or "pseudotime" in title:
                return (0, 0, iid)

            # 2. Timepoint change plots
            if "change" in iid or "change" in itype or "timepoint change" in title:
                return (1, 0, iid)

            # 3. Transition probability and transition network
            if "prob" in iid or "prob" in itype:
                return (2, 0, iid)
            if "net" in iid or "net" in itype:
                return (2, 1, iid)
            if ("cov" in iid or "cov" in itype) and ("table" not in iid and "forest" not in iid):
                return (2, 2, iid)

            # 4. Table & Forest plot (paired by suffix or state/cov)
            if it.get("markov_table_data") is not None or "table" in iid or itype == "markov_table":
                pair_tag = iid.replace("ts_markov_table_", "").replace("markov_table_", "")
                return (3, pair_tag, 0, iid)
            if "forest" in iid or itype == "ts_markov_forest":
                pair_tag = iid.replace("ts_markov_forest_", "").replace("markov_forest_", "")
                return (3, pair_tag, 1, iid)

            # 5. External images
            if it.get("kind") == "image":
                return (4, 0, 0, iid)

            return (5, 0, 0, iid)

        section_items["Time Series & Transitions"].sort(key=ts_sort_key)

        # 7. Build HTML Body & Table of Contents
        toc_items = []
        body_sections_html = ""

        if self.include_params_chk.value and params_html:
            toc_items.append("<li><a href='#sec-params'>Session Execution Parameters</a></li>")
            body_sections_html += params_html

        sec_num = 0
        total_rendered_figures = 0

        for sec_name in REPORT_SECTIONS:
            items = section_items.get(sec_name, [])
            if not items:
                continue

            sec_num += 1
            sec_id = f"sec-{re.sub(r'[^a-zA-Z0-9_]', '-', sec_name.lower())}"
            sub_links = []
            grid_cards_html = ""
            skip_ids = set()

            for it in items:
                item_id = it.get("id", f"item_{total_rendered_figures + 1}")
                if item_id in skip_ids:
                    continue

                total_rendered_figures += 1
                card_id = f"card-{item_id}"
                item_title = it.get("title", f"Figure {total_rendered_figures}")
                caption_val = it.get("caption", "").strip() if it.get("caption") else ""
                caption_html = f"<p class='plot-caption'>{caption_val}</p>" if caption_val else ""

                sub_links.append(f"<li><a href='#{card_id}'>{item_title}</a></li>")

                # Trajectories are centered solo and wide (not squished into 2 columns)
                is_trajectory = ("traj" in item_id.lower() or "traj" in (it.get("type") or "").lower() or "trajector" in item_title.lower())

                # Check for Markov table paired with forest plot
                if it.get("markov_table_data") is not None and sec_name == "Time Series & Transitions":
                    pair_tag = item_id.replace("ts_markov_table_", "").replace("markov_table_", "")
                    matching_forest = None
                    for other in items:
                        if other.get("id") not in skip_ids and other.get("id") != item_id:
                            o_id = other.get("id", "")
                            if "forest" in o_id.lower() or other.get("type") == "ts_markov_forest":
                                o_tag = o_id.replace("ts_markov_forest_", "").replace("markov_forest_", "")
                                if o_tag == pair_tag or not matching_forest:
                                    matching_forest = other
                                    if o_tag == pair_tag:
                                        break
                    if matching_forest:
                        skip_ids.add(matching_forest["id"])
                        total_rendered_figures += 1
                        f_id = matching_forest.get("id", f"item_{total_rendered_figures}")
                        f_card_id = f"card-{f_id}"
                        f_title = matching_forest.get("title", "Markov Forest Plot")
                        f_cap_val = matching_forest.get("caption", "").strip() if matching_forest.get("caption") else ""
                        f_caption_html = f"<p class='plot-caption'>{f_cap_val}</p>" if f_cap_val else ""
                        sub_links.append(f"<li><a href='#{f_card_id}'>{f_title}</a></li>")

                        markov_content = self._format_markov_table_html(it["markov_table_data"])
                        f_plot_key = f"std_{f_id}" if f"std_{f_id}" in divs else f"sub_{f_id}"
                        f_div_html = divs.get(f_plot_key, "")

                        grid_cards_html += f"""
                        <div class="grid-card-full table-forest-pair-container" style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; padding: 0; background: transparent; border: none; box-shadow: none;">
                            <div class="grid-card" id="{card_id}">
                                <h3 class="plot-title">{item_title}</h3>
                                {markov_content}
                                {caption_html}
                            </div>
                            <div class="grid-card" id="{f_card_id}">
                                <h3 class="plot-title">{f_title}</h3>
                                <div class="plot-container">{f_div_html}</div>
                                {f_caption_html}
                            </div>
                        </div>
                        """
                        continue

                is_full_width = (
                    is_trajectory
                    or it.get("kind") == "nmf_group"
                    or it.get("snr_data") is not None
                    or it.get("alpha_table_data") is not None
                    or it.get("markov_table_data") is not None
                    or item_id in ("preproc_despike", "preproc_smooth", "preproc_baseline", "raw_electropherograms", "control_electropherograms", "alignment_overlay", "preproc_norm")
                )
                card_class = "grid-card grid-card-full" if is_full_width else "grid-card"

                if it.get("kind") == "nmf_group":
                    safe_c = it["safe_c"]
                    g_idx = it["g_idx"]
                    groups_str = ", ".join([str(g) for g in it.get("groups", [])])
                    meta_sub = f"Differential analysis grouped by <code>{it['col_name']}</code> ({len(it.get('groups', []))} groups: {groups_str})" if groups_str else f"Differential analysis grouped by <code>{it['col_name']}</code>"

                    grp_cards = ""
                    if it.get("recon_fig") is not None:
                        k = f"nmf_grp_{safe_c}_recon_{g_idx}"
                        grp_cards += f"""
                        <div class="grid-card" style="margin-top: 12px;">
                            <h4 class="plot-title">Sample Reconstruction by Group: {it['col_name']} (asinh)</h4>
                            <div class="plot-container">{divs.get(k, '')}</div>
                        </div>"""
                    if it.get("sig_fig") is not None:
                        k = f"nmf_grp_{safe_c}_sig_{g_idx}"
                        grp_cards += f"""
                        <div class="grid-card" style="margin-top: 12px;">
                            <h4 class="plot-title">Basis Significance Across Groups: {it['col_name']} (-log10 q)</h4>
                            <div class="plot-container">{divs.get(k, '')}</div>
                        </div>"""
                    if it.get("pval_fig") is not None:
                        k = f"nmf_grp_{safe_c}_pval_{g_idx}"
                        grp_cards += f"""
                        <div class="grid-card" style="margin-top: 12px;">
                            <h4 class="plot-title">Basis p-value Heatmap Across Groups: {it['col_name']}</h4>
                            <div class="plot-container">{divs.get(k, '')}</div>
                        </div>"""
                    for b_name, j_fig in it.get("jitter_figs", {}).items():
                        safe_b = re.sub(r'[^a-zA-Z0-9_]', '_', b_name)
                        k = f"nmf_grp_{safe_c}_jitter_{safe_b}_{g_idx}"
                        grp_cards += f"""
                        <div class="grid-card" style="margin-top: 12px;">
                            <h4 class="plot-title">Loadings by Group: {b_name} ({it['col_name']})</h4>
                            <div class="plot-container">{divs.get(k, '')}</div>
                        </div>"""

                    grid_cards_html += f"""
                    <div class="{card_class}" id="{card_id}">
                        <h3 class="plot-title">{item_title}</h3>
                        <p class="category-subtitle" style="margin-top: 4px; margin-bottom: 12px;">{meta_sub}</p>
                        <div class="plots-grid" style="margin-top: 0; margin-bottom: 0;">
                            {grp_cards}
                        </div>
                        {caption_html}
                    </div>
                    """
                elif it.get("snr_data") is not None:
                    snr_content = self._format_snr_html(it["snr_data"])
                    grid_cards_html += f"""
                    <div class="{card_class}" id="{card_id}">
                        <h3 class="plot-title">{item_title}</h3>
                        {snr_content}
                        {caption_html}
                    </div>
                    """
                elif it.get("alpha_table_data") is not None:
                    alpha_content = self._format_alpha_div_table_html(it["alpha_table_data"])
                    grid_cards_html += f"""
                    <div class="{card_class}" id="{card_id}">
                        <h3 class="plot-title">{item_title}</h3>
                        {alpha_content}
                        {caption_html}
                    </div>
                    """
                elif it.get("markov_table_data") is not None:
                    markov_content = self._format_markov_table_html(it["markov_table_data"])
                    grid_cards_html += f"""
                    <div class="{card_class}" id="{card_id}">
                        <h3 class="plot-title">{item_title}</h3>
                        {markov_content}
                        {caption_html}
                    </div>
                    """
                elif it.get("data_uri"):
                    grid_cards_html += f"""
                    <div class="{card_class}" id="{card_id}">
                        <h3 class="plot-title">{item_title}</h3>
                        <div class="plot-container" style="text-align: center; margin: 12px 0;">
                            <img src="{it['data_uri']}" alt="{it['title']}" class="responsive-img" style="max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #cbd5e1;" />
                        </div>
                        {caption_html}
                    </div>
                    """
                else:
                    plot_key = f"std_{item_id}" if f"std_{item_id}" in divs else f"sub_{item_id}"
                    div_html = divs.get(plot_key, "")
                    extra_style = " style='display: flex; justify-content: center;'" if is_trajectory else ""
                    grid_cards_html += f"""
                    <div class="{card_class}" id="{card_id}">
                        <h3 class="plot-title">{item_title}</h3>
                        <div class="plot-container"{extra_style}>{div_html}</div>
                        {caption_html}
                    </div>
                    """

            sec_html = f"""
            <div class="section-card" id="{sec_id}">
                <h2 class="section-title">{sec_num}. {sec_name}</h2>
                <div class="plots-grid">
                    {grid_cards_html}
                </div>
            </div>
            """
            body_sections_html += sec_html

            sub_ul = f"<ul class='toc-sublist'>{''.join(sub_links)}</ul>" if sub_links else ""
            toc_items.append(f"<li><a href='#{sec_id}'>{sec_num}. {sec_name}</a>{sub_ul}</li>")

        plot_count = total_rendered_figures

        # Table of Contents HTML
        toc_html = ""
        if self.include_toc_chk.value and toc_items:
            toc_html = f"""
            <div class="toc-container">
                <h3 class="toc-title">Table of Contents</h3>
                <ul class="toc-list">
                    {''.join(toc_items)}
                </ul>
            </div>
            """

        # 8. Report Header & Metadata Pills (No Emojis)
        title = self.report_title_input.value or "CEtools Analysis Report"
        subtitle = self.report_subtitle_input.value or ""
        author = self.author_input.value or ""
        date_str = self.report_date_input.value or datetime.now().strftime("%Y-%m-%d")

        meta_pills = "<div class='meta-pills'>"
        meta_pills += f"<span class='pill'><b>Date:</b> {date_str}</span>"
        if author:
            meta_pills += f"<span class='pill'><b>Analyst:</b> {author}</span>"
        meta_pills += f"<span class='pill'><b>Total Figures:</b> {total_items}</span>"
        meta_pills += f"<span class='pill'><b>Engine:</b> CEtools {APP_VERSION} / Bokeh {bokeh.__version__}</span>"
        meta_pills += "</div>"

        subtitle_html = f"<p class='report-subtitle'>{subtitle}</p>" if subtitle else ""

        # 9. Bokeh Resources
        if "Inline" in self.resource_mode.value:
            bokeh_resources = INLINE.render()
        else:
            bokeh_resources = CDN.render()

        # 10. Complete HTML Document Assembly
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{bokeh_resources}
<style>
  :root {{
    --primary: #1e3a8a;
    --primary-light: #3b82f6;
    --bg-main: #f8fafc;
    --bg-card: #ffffff;
    --text-main: #0f172a;
    --text-muted: #475569;
    --border: #e2e8f0;
    --border-dark: #cbd5e1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-main);
    color: var(--text-main);
    line-height: 1.5;
    margin: 0;
    padding: 30px 15px;
  }}
  .container {{
    max-width: 1200px;
    margin: 0 auto;
  }}
  .header-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  }}
  .header-content {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 20px;
  }}
  .header-info {{
    flex: 1;
    min-width: 320px;
  }}
  .report-export-toolbar {{
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
    min-width: 270px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  }}
  .export-toggle-box {{
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    justify-content: space-between;
  }}
  .export-label {{
    font-size: 0.88rem;
    font-weight: 600;
    color: #334155;
  }}
  .format-toggle-group {{
    display: inline-flex;
    border-radius: 6px;
    border: 1px solid #cbd5e1;
    overflow: hidden;
    background: #ffffff;
  }}
  .format-btn {{
    background: #ffffff;
    border: none;
    padding: 6px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    color: #475569;
    transition: all 0.2s ease;
  }}
  .format-btn:hover {{
    background: #f1f5f9;
  }}
  .format-btn.active {{
    background: var(--primary);
    color: #ffffff;
  }}
  .export-hint {{
    font-size: 0.75rem;
    color: var(--text-muted);
    font-style: italic;
  }}
  .table-forest-pair-container {{
    margin-bottom: 16px;
  }}
  @media (max-width: 960px) {{
    .table-forest-pair-container {{
      grid-template-columns: 1fr !important;
    }}
  }}
  h1 {{
    color: var(--primary);
    font-size: 2.2rem;
    margin: 0 0 8px 0;
    font-weight: 700;
  }}
  .report-subtitle {{
    color: var(--text-muted);
    font-size: 1.15rem;
    margin: 0 0 16px 0;
  }}
  .meta-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
  }}
  .pill {{
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85rem;
    color: var(--text-muted);
  }}
  .toc-container {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  }}
  .toc-title {{
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0 0 12px 0;
    color: var(--primary);
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
  }}
  .toc-list {{
    list-style-type: decimal;
    padding-left: 20px;
    margin: 0;
  }}
  .toc-list > li {{
    margin-bottom: 8px;
    font-weight: 600;
  }}
  .toc-list a {{
    color: var(--primary-light);
    text-decoration: none;
  }}
  .toc-list a:hover {{
    text-decoration: underline;
  }}
  .toc-sublist {{
    list-style-type: circle;
    padding-left: 20px;
    margin-top: 4px;
    font-weight: 400;
  }}
  .toc-sublist li {{
    margin-bottom: 4px;
    font-size: 0.95rem;
  }}
  .toc-nested {{
    list-style-type: disc;
    padding-left: 20px;
    margin-top: 4px;
    font-weight: 500;
  }}
  .toc-nested li {{
    margin-bottom: 4px;
    font-size: 0.95rem;
  }}
  .toc-subnested {{
    list-style-type: circle;
    padding-left: 20px;
    margin-top: 2px;
    font-weight: 400;
  }}
  .toc-subnested li {{
    margin-bottom: 2px;
    font-size: 0.88rem;
    color: var(--text-muted);
  }}
  .section-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  }}
  .section-title {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--primary);
    margin: 0 0 20px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border);
  }}
  .nested-group-container {{
    display: flex;
    flex-direction: column;
    gap: 28px;
  }}
  .category-block {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 24px;
  }}
  .category-title {{
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--primary);
    margin: 0 0 4px 0;
  }}
  .category-subtitle {{
    font-size: 0.95rem;
    color: var(--text-muted);
    margin: 0 0 18px 0;
  }}
  .plots-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 24px;
    margin-top: 16px;
    margin-bottom: 24px;
    align-items: stretch;
  }}
  @media (max-width: 960px) {{
    .plots-grid {{
      grid-template-columns: 1fr;
    }}
  }}
  .grid-card {{
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    display: flex;
    flex-direction: column;
    min-width: 0;
    box-sizing: border-box;
  }}
  .grid-card-full {{
    grid-column: 1 / -1;
  }}
  .plot-card {{
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
  }}
  .plot-card:last-child {{
    margin-bottom: 0;
  }}
  .plot-title {{
    font-size: 1.15rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 12px 0;
  }}
  .plot-container {{
    overflow-x: auto;
    display: flex;
    justify-content: center;
    align-items: center;
    background: white;
    flex: 1 1 auto;
  }}
  .plot-caption {{
    margin-top: 12px;
    color: var(--text-muted);
    font-size: 0.9rem;
    font-style: italic;
    border-top: 1px solid #f1f5f9;
    padding-top: 8px;
    line-height: 1.45;
  }}
  .image-container {{
    text-align: center;
    padding: 10px 0;
  }}
  .responsive-img {{
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    border: 1px solid var(--border);
  }}
  .params-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
  }}
  .param-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 16px;
  }}
  .param-card h4 {{
    margin: 0 0 10px 0;
    color: var(--primary);
    font-size: 1rem;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 4px;
  }}
  .param-table {{
    width: 100%;
    font-size: 0.85rem;
    border-collapse: collapse;
  }}
  .param-key {{ color: var(--text-muted); padding: 4px 0; }}
  .param-val {{ text-align: right; font-weight: 600; color: #1e293b; padding: 4px 0; }}
  .footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }}
  @media print {{
    body {{ background: white; padding: 0; }}
    .container {{ max-width: 100%; }}
    .toc-container {{ display: none; }}
    .section-card {{ box-shadow: none; border: none; padding: 0; margin-bottom: 20px; }}
    .plot-card {{ page-break-inside: avoid; break-inside: avoid; margin-bottom: 24px; box-shadow: none; }}
    .snr-report-block div {{ max-height: none !important; overflow: visible !important; }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header-card">
    <div class="header-content">
      <div class="header-info">
        <h1>{title}</h1>
        {subtitle_html}
        {meta_pills}
      </div>
      <div class="report-export-toolbar">
        <div class="export-toggle-box">
          <span class="export-label">Plot Export Format:</span>
          <div class="format-toggle-group">
            <button type="button" class="format-btn active" id="btn-format-png" onclick="setGlobalPlotFormat('png')">PNG</button>
            <button type="button" class="format-btn" id="btn-format-svg" onclick="setGlobalPlotFormat('svg')">SVG</button>
          </div>
        </div>
        <span class="export-hint" id="export-hint">Default: PNG. Click to switch plot save icons to SVG.</span>
      </div>
    </div>
  </div>

  {toc_html}
  {params_html}
  {body_sections_html}

  <div class="footer">
    Report compiled on {date_str} using CEtools analysis pipeline {APP_VERSION}.
  </div>
</div>
{script}
<script>
let currentPlotFormat = 'png';

function setGlobalPlotFormat(format) {{
  currentPlotFormat = format;
  const btnPng = document.getElementById('btn-format-png');
  const btnSvg = document.getElementById('btn-format-svg');
  const hint = document.getElementById('export-hint');
  if (format === 'svg') {{
    if (btnPng) btnPng.classList.remove('active');
    if (btnSvg) btnSvg.classList.add('active');
    if (hint) hint.textContent = 'SVG selected. Individual plot save icons will download vector SVG files.';
  }} else {{
    if (btnSvg) btnSvg.classList.remove('active');
    if (btnPng) btnPng.classList.add('active');
    if (hint) hint.textContent = 'Default: PNG. Individual plot save icons will download high-DPI PNG files.';
  }}
  attachAllSaveToolOverrides();
}}

async function svgToPngBlob(svgBlob, width, height) {{
  return new Promise((resolve, reject) => {{
    const url = URL.createObjectURL(svgBlob);
    const img = new Image();
    img.onload = () => {{
      URL.revokeObjectURL(url);
      const scale = 2;
      const canvas = document.createElement('canvas');
      const w = width || img.naturalWidth || 800;
      const h = height || img.naturalHeight || 450;
      canvas.width = w * scale;
      canvas.height = h * scale;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0, w, h);
      canvas.toBlob((blob) => {{
        if (blob) resolve(blob);
        else reject(new Error('Failed to create PNG blob'));
      }}, 'image/png');
    }};
    img.onerror = (e) => {{
      URL.revokeObjectURL(url);
      reject(e);
    }};
    img.src = url;
  }});
}}

function attachAllSaveToolOverrides() {{
  if (window.Bokeh && window.Bokeh.index) {{
    const views = Object.values(window.Bokeh.index);
    for (const view of views) {{
      attachSaveToolToView(view);
    }}
  }}
}}

function attachSaveToolToView(view) {{
  if (!view) return;
  if (view.tool_views) {{
    for (const [tool, tool_view] of view.tool_views) {{
      if (tool_view && tool_view.model && (tool_view.model.type === 'SaveTool' || (tool && tool.type === 'SaveTool'))) {{
        overrideSaveToolView(tool_view, view);
      }}
    }}
  }}
  if (view.child_views) {{
    for (const child of view.child_views) {{
      attachSaveToolToView(child);
    }}
  }}
}}

function overrideSaveToolView(tool_view, plot_view) {{
  if (tool_view._custom_save_attached) return;
  tool_view._custom_save_attached = true;
  tool_view.save = async function(filename) {{
    try {{
      const fmt = currentPlotFormat || 'png';
      let blob;
      if (fmt === 'svg') {{
        blob = await plot_view.export('svg').to_blob();
      }} else {{
        const svgBlob = await plot_view.export('svg').to_blob();
        const w = (plot_view.bbox && plot_view.bbox.width) || (plot_view.layout && plot_view.layout.bbox && plot_view.layout.bbox.width) || 800;
        const h = (plot_view.bbox && plot_view.bbox.height) || (plot_view.layout && plot_view.layout.bbox && plot_view.layout.bbox.height) || 450;
        blob = await svgToPngBlob(svgBlob, w, h);
      }}
      let name = filename || tool_view.model.filename || (plot_view.model.title ? plot_view.model.title.text : 'bokeh_plot');
      name = name.replace(/\\.(png|svg)$/i, '');
      name = name.replace(/[^a-zA-Z0-9_-]/g, '_') + '.' + fmt;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.target = '_blank';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    }} catch (e) {{
      console.warn('Custom save failed, falling back:', e);
      if (tool_view.parent && typeof tool_view.parent.export === 'function') {{
        try {{
          const b = await tool_view.parent.export().to_blob();
          const a = document.createElement('a');
          a.href = URL.createObjectURL(b);
          a.download = (filename || 'plot') + '.' + currentPlotFormat;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        }} catch (err) {{
          console.error('Fallback export failed:', err);
        }}
      }}
    }}
  }};
}}

function initPlotExportHooks() {{
  attachAllSaveToolOverrides();
  if (!window.Bokeh || !window.Bokeh.index || Object.keys(window.Bokeh.index).length === 0) {{
    setTimeout(initPlotExportHooks, 250);
  }} else {{
    let checks = 0;
    const interval = setInterval(() => {{
      attachAllSaveToolOverrides();
      checks++;
      if (checks > 10) clearInterval(interval);
    }}, 1000);
  }}
}}
if (document.readyState === 'loading') {{
  document.addEventListener('DOMContentLoaded', initPlotExportHooks);
}} else {{
  initPlotExportHooks();
}}
</script>
</body>
</html>"""
        return full_html, total_items


def build_html_report_section(controllers: Dict[str, Any]) -> Tuple[pn.Column, HTMLReportController]:
    """Factory function to build and return the HTML report export tab layout and controller."""
    ctrl = HTMLReportController(controllers)
    return ctrl.section, ctrl
