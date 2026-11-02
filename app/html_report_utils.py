import io
import os
import re
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import panel as pn
import bokeh
import bokeh.plotting as bkp
import bokeh.models
from bokeh.embed import components
from bokeh.resources import CDN, INLINE
from scipy.stats import kruskal, linregress

import common_plot
from common_plot import TitledPlotPane, apply_export_prefix_to_pane

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

        # 4B: Specific NMF Basis Waveform
        self.basis_select = pn.widgets.Select(name="Select NMF Basis", options=[], width=180)
        self.basis_plot_type = pn.widgets.RadioButtonGroup(
            name="Plot Type",
            options=["Continuous Waveform"],
            value="Continuous Waveform"
        )
        self.basis_title = pn.widgets.TextInput(name="Plot Title", value="NMF Basis Waveform", width=240)
        self.basis_caption = pn.widgets.TextInput(name="Optional Caption", value="", width=300)
        self.add_basis_btn = pn.widgets.Button(name="Add Basis Waveform", button_type="success", width=170)
        self.add_basis_btn.on_click(self._on_add_basis_plot)

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
                "specific NMF basis waveforms, uploaded external figures, and session parameters."
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
                "Specify one or more categorical variables from the metadata. Each variable produces a nested "
                "report section containing its sample reconstruction by group, basis significance plot, "
                "and optional basis loadings."
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
                ("NMF Basis Waveform", pn.Column(
                    pn.Row(self.basis_select, self.basis_plot_type),
                    pn.Row(self.basis_title, self.basis_caption),
                    self.add_basis_btn
                )),
            ),
            pn.pane.Markdown("**Current Additional Sub-Plots:**", styles={"margin-top": "10px"}),
            self.additional_plots_container,
            pn.layout.Divider(),

            # Card 5: External Figures
            pn.pane.Markdown("### 5. Upload External Figures / Images"),
            pn.Row(self.image_file_input),
            pn.Row(self.image_title_input, self.image_caption_input),
            self.add_image_btn,
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
                "getter": lambda: self._get_plot_raw(),
                "default_selected": True
            },
            {
                "id": "snr_plot",
                "category": "Raw Data & QC",
                "default_title": "Signal-to-Noise Ratio (SNR)",
                "getter": lambda: self._get_plot_snr(),
                "default_selected": True
            },
            # Preprocessing
            {
                "id": "preproc_despike",
                "category": "Preprocessing",
                "default_title": "Despiked Traces",
                "getter": lambda: self._get_plot_despike(),
                "default_selected": False
            },
            {
                "id": "preproc_smooth",
                "category": "Preprocessing",
                "default_title": "Smoothed Traces",
                "getter": lambda: self._get_plot_smooth(),
                "default_selected": False
            },
            {
                "id": "preproc_baseline",
                "category": "Preprocessing",
                "default_title": "Baseline Subtracted Traces",
                "getter": lambda: self._get_plot_baseline(),
                "default_selected": False
            },
            {
                "id": "preproc_norm",
                "category": "Preprocessing",
                "default_title": "Normalized Electropherograms",
                "getter": lambda: self._get_plot_norm(),
                "default_selected": True
            },
            # Alignment
            {
                "id": "align_anchors",
                "category": "Alignment",
                "default_title": "Anchor Selection Curves",
                "getter": lambda: self._get_plot_align_anchors(),
                "default_selected": True
            },
            {
                "id": "align_aligned",
                "category": "Alignment",
                "default_title": "Aligned Chromatograms",
                "getter": lambda: self._get_plot_align_aligned(),
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
                "id": "alpha_diversity",
                "category": "Alpha Diversity",
                "default_title": "Alpha Diversity Metrics Comparison",
                "getter": lambda: self._get_plot_diversity(),
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
                "id": "viz_pca",
                "category": "Dimensionality Reduction",
                "default_title": "PCA Score Plot",
                "getter": lambda: self._get_plot_viz_pca(),
                "default_selected": True
            },
            {
                "id": "viz_loadings",
                "category": "Dimensionality Reduction",
                "default_title": "PCA Loadings Plot",
                "getter": lambda: self._get_plot_viz_loadings(),
                "default_selected": False
            },
            {
                "id": "viz_mds",
                "category": "Dimensionality Reduction",
                "default_title": "Metric MDS 2D Embedding",
                "getter": lambda: self._get_plot_viz_mds(),
                "default_selected": False
            },
            {
                "id": "viz_tsne",
                "category": "Dimensionality Reduction",
                "default_title": "t-SNE 2D Embedding",
                "getter": lambda: self._get_plot_viz_tsne(),
                "default_selected": False
            },
            # Miscellaneous
            {
                "id": "misc_cvc",
                "category": "Miscellaneous Analyses",
                "default_title": "Canonical Variate Analysis (CVC)",
                "getter": lambda: self._get_plot_misc_cvc(),
                "default_selected": False
            },
            {
                "id": "misc_cat",
                "category": "Miscellaneous Analyses",
                "default_title": "Categorical Variable Comparisons",
                "getter": lambda: self._get_plot_misc_cat(),
                "default_selected": False
            },
            {
                "id": "misc_reg",
                "category": "Miscellaneous Analyses",
                "default_title": "Linear Regression Loadings Plot",
                "getter": lambda: self._get_plot_misc_reg(),
                "default_selected": False
            },
            {
                "id": "misc_cat_cov",
                "category": "Miscellaneous Analyses",
                "default_title": "Categorical Covariates Comparison",
                "getter": lambda: self._get_plot_misc_cat_cov(),
                "default_selected": False
            },
            # Time Series
            {
                "id": "ts_markov",
                "category": "Time Series & Pseudotime",
                "default_title": "Markov State Transition Probability Heatmap",
                "getter": lambda: self._get_plot_ts_markov(),
                "default_selected": False
            },
            {
                "id": "ts_trajectories",
                "category": "Time Series & Pseudotime",
                "default_title": "Sample Trajectories Along Pseudotime",
                "getter": lambda: self._get_plot_ts_trajectories(),
                "default_selected": False
            },
            {
                "id": "ts_change",
                "category": "Time Series & Pseudotime",
                "default_title": "Timepoint Change by Category",
                "getter": lambda: self._get_plot_ts_change(),
                "default_selected": False
            },
        ]

    # ---------------- Plot Getters from Controllers ----------------
    def _get_plot_raw(self):
        state = self.controllers.get("state")
        return getattr(state, "last_plot", None) if state else None

    def _get_plot_snr(self):
        ctrl = self.controllers.get("snr")
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None)) if ctrl else None

    def _get_plot_despike(self):
        ctrl = self.controllers.get("despike")
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None)) if ctrl else None

    def _get_plot_smooth(self):
        ctrl = self.controllers.get("smooth")
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None)) if ctrl else None

    def _get_plot_baseline(self):
        ctrl = self.controllers.get("baseline")
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None)) if ctrl else None

    def _get_plot_norm(self):
        ctrl = self.controllers.get("norm")
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None)) if ctrl else None

    def _get_plot_align_anchors(self):
        ctrl = self.controllers.get("alignment")
        return extract_bokeh_figure(getattr(ctrl, "anchor_plot_pane", None)) if ctrl else None

    def _get_plot_align_aligned(self):
        ctrl = self.controllers.get("alignment")
        return extract_bokeh_figure(getattr(ctrl, "aligned_plot_pane", None)) if ctrl else None

    def _get_plot_nmf_recon(self):
        ctrl = self.controllers.get("nmf")
        return extract_bokeh_figure(getattr(ctrl, "recon_plot_pane", None)) if ctrl else None

    def _get_plot_nmf_heatmap(self):
        ctrl = self.controllers.get("nmf")
        return extract_bokeh_figure(getattr(ctrl, "heatmap_plot_pane", None)) if ctrl else None

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

    def _get_plot_diversity(self):
        ctrl = self.controllers.get("diversity")
        return extract_bokeh_figure(getattr(ctrl, "plot_pane", None)) if ctrl else None

    def _get_plot_viz_scree(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "scree_plot_pane", None)) if ctrl else None

    def _get_plot_viz_pca(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "pca_plot_pane", None)) if ctrl else None

    def _get_plot_viz_loadings(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "loadings_plot_pane", None)) if ctrl else None

    def _get_plot_viz_mds(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "mds_plot_pane", None)) if ctrl else None

    def _get_plot_viz_tsne(self):
        ctrl = self.controllers.get("viz")
        return extract_bokeh_figure(getattr(ctrl, "tsne_plot_pane", None)) if ctrl else None

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
                    width=320
                )
                title_inp = pn.widgets.TextInput(
                    value=it["default_title"],
                    sizing_mode="stretch_width",
                    disabled=not ready
                )

                self._standard_plot_widgets[it["id"]] = {
                    "checkbox": chk,
                    "title_input": title_inp,
                    "getter": it["getter"],
                    "category": cat_name,
                    "is_ready": ready
                }
                rows.append(pn.Row(chk, badge, title_inp, sizing_mode="stretch_width", margin=(2, 0, 2, 0)))

            card = pn.Column(header, *rows, margin=(0, 0, 10, 0))
            card_objects.append(card)

        self.standard_plots_layout.objects = card_objects
        self.plots_summary_md.object = f"**Status:** Found **{ready_count}** ready plots out of {len(self.plot_catalog)} standard pipeline plots."

        # Refresh NMF Group Statistics options
        self._update_nmf_group_section_options()

        # Update sub-plot dropdown options
        self._update_dropdowns()

    def _set_all_selected(self, select: bool):
        for data in self._standard_plot_widgets.values():
            if data["is_ready"]:
                data["checkbox"].value = select

    def _update_dropdowns(self):
        """Populate sample reconstruction and basis selection dropdowns."""
        state = self.controllers.get("state")
        nmf_ctrl = self.controllers.get("nmf")

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

        bases: List[str] = []
        if nmf_ctrl and hasattr(nmf_ctrl, "H_df") and nmf_ctrl.H_df is not None and not nmf_ctrl.H_df.empty:
            df = nmf_ctrl.H_df
            bases = [c for c in df.columns if c not in ["Unnamed: 0", "sample", "sample_id", "_sid"]]
        elif nmf_ctrl and hasattr(nmf_ctrl, "basis_count") and nmf_ctrl.basis_count.value:
            bases = [f"basis_{i+1}" for i in range(int(nmf_ctrl.basis_count.value))]

        if bases:
            self.basis_select.options = bases
            if not self.basis_select.value or self.basis_select.value not in bases:
                self.basis_select.value = bases[0]

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

    def _on_add_basis_plot(self, _=None):
        basis = self.basis_select.value
        if not basis:
            self.report_status.object = _warn("Please select an NMF basis.")
            return

        nmf_ctrl = self.controllers.get("nmf")
        if not nmf_ctrl:
            self.report_status.object = _warn("NMF controller not available.")
            return

        title = self.basis_title.value or f"NMF {basis} Waveform"
        caption = self.basis_caption.value

        try:
            import CEtools as cet
            K = 10
            if hasattr(nmf_ctrl, "basis_count"):
                K = int(nmf_ctrl.basis_count.value)
            elif hasattr(nmf_ctrl, "H_df") and nmf_ctrl.H_df is not None:
                K = len([c for c in nmf_ctrl.H_df.columns if c not in ["Unnamed: 0", "sample", "_sid"]])

            m = re.search(r'\d+', basis)
            basis_idx = int(m.group(0)) - 1 if m else 0

            centers = cet.default_gaussian_centers(K)
            sigma = cet.heuristic_sigma_from_centers(centers)
            Phi = cet.make_gaussian_basis(centers, sigma)

            t_eval = np.linspace(0.0, 1.0, 1000)
            basis_eval = Phi(t_eval)[:, basis_idx]

            fig = bkp.figure(
                title=title,
                x_axis_label="Normalized Time / Pseudotime",
                y_axis_label="Basis Amplitude",
                width=750, height=300,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above"
            )
            fig.line(t_eval, basis_eval, line_width=2.5, line_color="#1d4ed8")
            fig.sizing_mode = "stretch_width"

            item_id = f"sub_basis_{basis}_{len(self.additional_pipeline_plots)}"
            self.additional_pipeline_plots.append({
                "id": item_id,
                "type": "basis_waveform",
                "title": title,
                "caption": caption,
                "figure": fig,
                "details": f"Basis: {basis} (Gaussian kernel)"
            })
            self._render_additional_plots_list()
            self.report_status.object = _ok(f"Added waveform plot for '{basis}' to report.")
        except Exception as e:
            self.report_status.object = _warn(f"Failed to generate basis waveform: {e}")

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
            self.additional_pipeline_plots.pop(index)
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

        self.custom_images.append({
            "id": f"img_{len(self.custom_images)+1}",
            "filename": filename,
            "title": title,
            "caption": caption,
            "data_uri": data_uri
        })
        self._render_custom_images_list()
        self.report_status.object = _ok(f"Added image '{filename}' to report.")
        self.image_file_input.value = None

    def _render_custom_images_list(self):
        if not self.custom_images:
            self.custom_images_container.objects = [pn.pane.Markdown("*No external images added yet.*")]
            return

        rows = []
        for idx, item in enumerate(self.custom_images):
            lbl = pn.pane.Markdown(f"**{idx+1}. {item['title']}** ({item['filename']})")
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
                    user_title = w["title_input"].value or item["default_title"]
                    fig = w["getter"]()
                    if fig is not None:
                        selected_standard_plots.append({
                            "id": pid,
                            "category": item["category"],
                            "title": user_title,
                            "figure": fig,
                            "caption": ""
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

        # 3. Collect additional subplots
        selected_sub_plots = []
        for sp in self.additional_pipeline_plots:
            fig = sp.get("figure", sp.get("fig"))
            if fig is not None:
                selected_sub_plots.append({
                    "id": sp["id"],
                    "category": "Additional Pipeline Sub-Plots",
                    "title": sp["title"],
                    "figure": fig,
                    "caption": sp.get("caption", "")
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

        # Render Bokeh scripts and divs
        script, divs = components(figures_dict)

        # 5. Build Session Parameters Table
        params_html = ""
        state = self.controllers.get("state")
        if self.include_params_chk.value and state and hasattr(state, "session_log") and state.session_log:
            params_html = "<div class='section-card' id='sec-params'>"
            params_html += "<h2 class='section-title'>Session Execution Parameters</h2>"
            params_html += "<div class='params-grid'>"
            for stage, pdict in state.session_log.items():
                params_html += f"<div class='param-card'><h4>{stage}</h4><table class='param-table'>"
                for k, v in pdict.items():
                    params_html += f"<tr><td class='param-key'>{k}</td><td class='param-val'>{v}</td></tr>"
                params_html += "</table></div>"
            params_html += "</div></div>"

        # 6. Group Standard Plots by Category
        categories_dict: Dict[str, List[Dict[str, Any]]] = {}
        for sp in selected_standard_plots:
            cat = sp["category"]
            if cat not in categories_dict:
                categories_dict[cat] = []
            categories_dict[cat].append(sp)

        # 7. Build HTML Body & Table of Contents
        toc_items = []
        body_sections_html = ""

        if self.include_params_chk.value and params_html:
            toc_items.append("<li><a href='#sec-params'>Session Execution Parameters</a></li>")

        # Standard Sections
        for cat_name, items in categories_dict.items():
            sec_id = f"sec-{re.sub(r'[^a-zA-Z0-9_]', '-', cat_name.lower())}"
            sub_links = []
            sec_html = f"<div class='section-card' id='{sec_id}'>"
            sec_html += f"<h2 class='section-title'>{cat_name}</h2>"

            for it in items:
                plot_key = f"std_{it['id']}"
                div_html = divs.get(plot_key, "")
                card_id = f"card-{it['id']}"
                sub_links.append(f"<li><a href='#{card_id}'>{it['title']}</a></li>")

                sec_html += f"""
                <div class="plot-card" id="{card_id}">
                    <h3 class="plot-title">{it['title']}</h3>
                    <div class="plot-container">{div_html}</div>
                </div>
                """
            sec_html += "</div>"
            body_sections_html += sec_html

            sub_ul = f"<ul class='toc-sublist'>{''.join(sub_links)}</ul>" if sub_links else ""
            toc_items.append(f"<li><a href='#{sec_id}'>{cat_name}</a>{sub_ul}</li>")

        # Nested NMF Group Statistics Section
        if selected_nmf_groups:
            sec_id = "sec-nmf-group-statistics"
            sec_html = f"<div class='section-card' id='{sec_id}'>"
            sec_html += "<h2 class='section-title'>NMF Group Statistics</h2>"
            sec_html += "<div class='nested-group-container'>"

            cat_toc_links = []
            for g_idx, grp in enumerate(selected_nmf_groups):
                safe_c = re.sub(r'[^a-zA-Z0-9_]', '_', grp["col_name"])
                cat_block_id = f"cat-block-{safe_c}-{g_idx}"
                groups_str = ", ".join([str(g) for g in grp.get("groups", [])])
                meta_sub = f"Differential analysis grouped by <code>{grp['col_name']}</code> ({len(grp.get('groups', []))} groups: {groups_str})" if groups_str else f"Differential analysis grouped by <code>{grp['col_name']}</code>"

                sec_html += f"""
                <div class="category-block" id="{cat_block_id}">
                    <h3 class="category-title">{grp['title']}</h3>
                    <p class="category-subtitle">{meta_sub}</p>
                """

                sub_plot_links = []

                # Recon
                if grp.get("recon_fig") is not None:
                    k = f"nmf_grp_{safe_c}_recon_{g_idx}"
                    f_id = f"fig-{safe_c}-recon"
                    sub_plot_links.append(f"<li><a href='#{f_id}'>Sample Reconstruction</a></li>")
                    sec_html += f"""
                    <div class="plot-card" id="{f_id}">
                        <h4 class="plot-title">Sample Reconstruction by Group: {grp['col_name']} (asinh)</h4>
                        <div class="plot-container">{divs.get(k, '')}</div>
                    </div>
                    """

                # Significance (q-value)
                if grp.get("sig_fig") is not None:
                    k = f"nmf_grp_{safe_c}_sig_{g_idx}"
                    f_id = f"fig-{safe_c}-sig"
                    sub_plot_links.append(f"<li><a href='#{f_id}'>Basis Significance (q-value)</a></li>")
                    sec_html += f"""
                    <div class="plot-card" id="{f_id}">
                        <h4 class="plot-title">NMF Basis Significance: {grp['col_name']} (q-value)</h4>
                        <div class="plot-container">{divs.get(k, '')}</div>
                    </div>
                    """

                # Significance (p-value)
                if grp.get("pval_fig") is not None:
                    k = f"nmf_grp_{safe_c}_pval_{g_idx}"
                    f_id = f"fig-{safe_c}-pval"
                    sub_plot_links.append(f"<li><a href='#{f_id}'>Basis Significance (p-value)</a></li>")
                    sec_html += f"""
                    <div class="plot-card" id="{f_id}">
                        <h4 class="plot-title">NMF Basis Significance: {grp['col_name']} (p-value)</h4>
                        <div class="plot-container">{divs.get(k, '')}</div>
                    </div>
                    """

                # Basis Loadings Jitter
                for b_name in grp.get("jitter_figs", {}):
                    safe_b = re.sub(r'[^a-zA-Z0-9_]', '_', b_name)
                    k = f"nmf_grp_{safe_c}_jitter_{safe_b}_{g_idx}"
                    f_id = f"fig-{safe_c}-jitter-{safe_b}"
                    sub_plot_links.append(f"<li><a href='#{f_id}'>Loadings: {b_name}</a></li>")
                    sec_html += f"""
                    <div class="plot-card" id="{f_id}">
                        <h4 class="plot-title">Basis Loadings by Group: {b_name} ({grp['col_name']})</h4>
                        <div class="plot-container">{divs.get(k, '')}</div>
                    </div>
                    """

                sec_html += "</div>" # close category-block

                sub_sub_ul = f"<ul class='toc-subnested'>{''.join(sub_plot_links)}</ul>" if sub_plot_links else ""
                cat_toc_links.append(f"<li><a href='#{cat_block_id}'>{grp['col_name']}</a>{sub_sub_ul}</li>")

            sec_html += "</div></div>" # close nested-group-container and section-card
            body_sections_html += sec_html

            toc_nested_ul = f"<ul class='toc-nested'>{''.join(cat_toc_links)}</ul>" if cat_toc_links else ""
            toc_items.append(f"<li><a href='#{sec_id}'>NMF Group Statistics</a>{toc_nested_ul}</li>")

        # Additional Sub-Plots
        if selected_sub_plots:
            sec_id = "sec-additional-subplots"
            sub_links = []
            sec_html = f"<div class='section-card' id='{sec_id}'>"
            sec_html += "<h2 class='section-title'>Additional Pipeline Sub-Plots</h2>"

            for it in selected_sub_plots:
                plot_key = f"sub_{it['id']}"
                div_html = divs.get(plot_key, "")
                card_id = f"card-{it['id']}"
                sub_links.append(f"<li><a href='#{card_id}'>{it['title']}</a></li>")

                caption_html = f"<p class='plot-caption'>{it['caption']}</p>" if it['caption'] else ""
                sec_html += f"""
                <div class="plot-card" id="{card_id}">
                    <h3 class="plot-title">{it['title']}</h3>
                    <div class="plot-container">{div_html}</div>
                    {caption_html}
                </div>
                """
            sec_html += "</div>"
            body_sections_html += sec_html
            sub_ul = f"<ul class='toc-sublist'>{''.join(sub_links)}</ul>" if sub_links else ""
            toc_items.append(f"<li><a href='#{sec_id}'>Additional Pipeline Sub-Plots</a>{sub_ul}</li>")

        # External Custom Images
        if self.custom_images:
            sec_id = "sec-custom-images"
            sub_links = []
            sec_html = f"<div class='section-card' id='{sec_id}'>"
            sec_html += "<h2 class='section-title'>External Figures & Images</h2>"

            for it in self.custom_images:
                card_id = f"card-{it['id']}"
                sub_links.append(f"<li><a href='#{card_id}'>{it['title']}</a></li>")
                caption_html = f"<p class='plot-caption'>{it['caption']}</p>" if it['caption'] else ""
                sec_html += f"""
                <div class="plot-card" id="{card_id}">
                    <h3 class="plot-title">{it['title']}</h3>
                    <div class="image-container">
                        <img src="{it['data_uri']}" alt="{it['title']}" class="responsive-img" />
                    </div>
                    {caption_html}
                </div>
                """
            sec_html += "</div>"
            body_sections_html += sec_html
            sub_ul = f"<ul class='toc-sublist'>{''.join(sub_links)}</ul>" if sub_links else ""
            toc_items.append(f"<li><a href='#{sec_id}'>External Figures & Images</a>{sub_ul}</li>")

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
        meta_pills += f"<span class='pill'><b>Engine:</b> CEtools / Bokeh {bokeh.__version__}</span>"
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
  .plot-card {{
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
  }}
  .plot-card:last-child {{
    margin-bottom: 0;
  }}
  .plot-title {{
    font-size: 1.15rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 14px 0;
  }}
  .plot-container {{
    overflow-x: auto;
    display: flex;
    justify-content: center;
    background: white;
  }}
  .plot-caption {{
    margin-top: 10px;
    color: var(--text-muted);
    font-size: 0.9rem;
    font-style: italic;
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
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header-card">
    <h1>{title}</h1>
    {subtitle_html}
    {meta_pills}
  </div>

  {toc_html}
  {params_html}
  {body_sections_html}

  <div class="footer">
    Report compiled on {date_str} using CEtools pipeline.
  </div>
</div>
{script}
</body>
</html>"""
        return full_html, total_items


def build_html_report_section(controllers: Dict[str, Any]) -> Tuple[pn.Column, HTMLReportController]:
    """Factory function to build and return the HTML report export tab layout and controller."""
    ctrl = HTMLReportController(controllers)
    return ctrl.section, ctrl
