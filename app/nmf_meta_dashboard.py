from __future__ import annotations

import io
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, Range1d, LinearAxis, LogColorMapper, LinearColorMapper, CustomJSTickFormatter, ColorBar, CustomJS
from bokeh.plotting import figure
from common_plot import TitledPlotPane

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
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def nmf_group_importance_dashboard_stable(
    metadata_df: pd.DataFrame,
    nmf_loadings_df: pd.DataFrame,
    *,
    group_col: str,
    meta_sample_col: str = "sample",
    nmf_sample_col: str = "sample_id",
    min_group_size: int = 3,
    roi_range: tuple[float, float] = (0.0, 1.0),
    svg_export_mode: bool = False,
) -> pn.Column:
    from scipy.stats import kruskal  # type: ignore

    meta = metadata_df.copy()
    H = nmf_loadings_df.copy()

    if group_col not in meta.columns:
        raise KeyError(f"metadata_df missing '{group_col}'")
    if meta_sample_col not in meta.columns:
        raise KeyError(f"metadata_df missing '{meta_sample_col}'")

    if nmf_sample_col not in H.columns:
        # If it's missing, it's likely in the index (like from live memory)
        H = H.reset_index()
        # Rename whatever the index was called to nmf_sample_col
        H = H.rename(columns={H.columns[0]: nmf_sample_col})

    meta["_sid"] = _coerce_sid(meta[meta_sample_col])
    H["_sid"] = _coerce_sid(H[nmf_sample_col])

    meta = meta.dropna(subset=["_sid", group_col]).copy()
    meta[group_col] = meta[group_col].astype(str)

    merged = pd.merge(meta[["_sid", group_col]], H, on="_sid", how="inner")
    if merged.empty:
        raise ValueError("No overlapping samples between metadata and NMF loadings after ID normalization.")

    # enforce min group size
    vc = merged[group_col].value_counts()
    keep_groups = vc[vc >= min_group_size].index.tolist()
    merged = merged[merged[group_col].isin(keep_groups)].copy()
    groups = sorted(merged[group_col].unique().tolist())
    if len(groups) < 2:
        raise ValueError("Need >=2 groups with min_group_size after filtering.")

    basis_cols = [c for c in merged.columns if c not in {"_sid", group_col, nmf_sample_col}]
    if not basis_cols:
        raise ValueError("No basis columns found (expected all columns after sample id).")

    # stats per basis (once)
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
        except ValueError as e:
            if "identical" in str(e).lower():
                stat, p = 0.0, 1.0
            else:
                raise e
        rows.append({"basis": b, "basis_index": j, "stat": float(stat), "p_value": float(p), **means, **ns})
    stats = pd.DataFrame(rows).set_index("basis")
    stats["q_value"] = _bh_fdr(stats["p_value"].to_numpy())
    stats = stats.sort_values(["q_value", "p_value"], ascending=[True, True])

    all_bases_numerical = stats.sort_values("basis_index").index.tolist()
    most_significant = stats.index[0]
    basis_sel = pn.widgets.Select(name="Inspect basis", options=all_bases_numerical, value=most_significant, width=220)

    sio = io.StringIO()
    stats.to_csv(sio)
    sio.seek(0)
    download_btn = pn.widgets.FileDownload(
        file=sio, 
        filename="nmf_group_stats.csv", 
        button_type="success", 
        name="Download Stats CSV",
        width=200
    )

    stats_pane = pn.pane.DataFrame(stats.reset_index(), height=260, sizing_mode="stretch_width")

    import common_plot
    
    K = len(basis_cols)
    basis_indices = np.arange(1, K + 1)
    stats_sorted = stats.reset_index().sort_values("basis_index")
    q_values = stats_sorted["q_value"].to_numpy()
    q_values_clipped = np.clip(q_values, 1e-10, 1.0)
    
    lo, hi = roi_range
    K = len(basis_cols)
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

    sig_src = ColumnDataSource(data=dict(
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
        cmap = LogColorMapper(palette=common_plot.get_continuous_palette(256), low=plot_low, high=1.0)
    else:
        cmap = LinearColorMapper(palette=common_plot.get_continuous_palette(256), low=0, high=max_q if max_q > 0 else 1.0)

    sig_fig = figure(
        width=800, height=200,
        title=f"NMF Basis Significance ({group_col})",
        x_axis_label="Pseudotime",
        tools="hover,save,pan,wheel_zoom,box_zoom,reset",
        toolbar_location="above",
        active_scroll=None,
        x_range=Range1d(pt_min - pt_margin, pt_max + pt_margin),
        y_range=Range1d(-1, 1)
    )
    if svg_export_mode:
        sig_fig.output_backend = "svg"
    sig_fig.yaxis.visible = False
    sig_fig.ygrid.visible = False

    basis_range = Range1d(start=0.5, end=K + 0.5)
    sig_fig.extra_x_ranges = {"basis": basis_range}
    
    sync_code = f"""
        const new_start = (cb_obj.start - {pt_min}) / {pt_spacing} + 1.0;
        const new_end = (cb_obj.end - {pt_min}) / {pt_spacing} + 1.0;
        basis_range.start = new_start;
        basis_range.end = new_end;
    """
    callback = CustomJS(args=dict(basis_range=basis_range), code=sync_code)
    sig_fig.x_range.js_on_change('start', callback)
    sig_fig.x_range.js_on_change('end', callback)
    
    basis_axis = LinearAxis(x_range_name="basis", axis_label="Basis Number")
    sig_fig.add_layout(basis_axis, 'above')
    
    sig_fig.rect(
        x="pseudotime", y=0, width=pt_spacing * 0.9, height=1.8,
        source=sig_src,
        fill_color={"field": "q_value_plot", "transform": cmap},
        line_color="lightgrey",
        line_width=1
    )

    hover = sig_fig.select(dict(type=HoverTool))[0]
    hover.tooltips = [
        ("Basis", "@basis_name (@basis_index)"),
        ("q-value", "@q_value{%0.2e}"),
        ("Pseudotime", "@pseudotime{0.00}")
    ]
    hover.formatters = {"@q_value": "printf"}

    cbar = ColorBar(color_mapper=cmap, title="q-value", orientation="horizontal", padding=0)
    sig_fig.add_layout(cbar, 'below')

    p_values = stats_sorted["p_value"].to_numpy()
    p_values_clipped = np.clip(p_values, 1e-10, 1.0)
    
    sig_src_p = ColumnDataSource(data=dict(
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
        cmap_p = LogColorMapper(palette=common_plot.get_continuous_palette(256), low=plot_low_p, high=1.0)
    else:
        cmap_p = LinearColorMapper(palette=common_plot.get_continuous_palette(256), low=0, high=max_p if max_p > 0 else 1.0)

    pval_fig = figure(
        width=800, height=200,
        title=f"NMF Basis Significance (p-value, {group_col})",
        x_axis_label="Pseudotime",
        tools="hover,save,pan,wheel_zoom,box_zoom,reset",
        toolbar_location="above",
        active_scroll=None,
        x_range=sig_fig.x_range,
        y_range=Range1d(-1, 1)
    )
    if svg_export_mode:
        pval_fig.output_backend = "svg"
    pval_fig.yaxis.visible = False
    pval_fig.ygrid.visible = False
    pval_fig.extra_x_ranges = {"basis": basis_range}
    pval_fig.x_range.js_on_change('start', callback)
    pval_fig.x_range.js_on_change('end', callback)
    
    basis_axis_p = LinearAxis(x_range_name="basis", axis_label="Basis Number")
    pval_fig.add_layout(basis_axis_p, 'above')
    
    pval_fig.rect(
        x="pseudotime", y=0, width=pt_spacing * 0.9, height=1.8,
        source=sig_src_p,
        fill_color={"field": "p_value_plot", "transform": cmap_p},
        line_color="lightgrey",
        line_width=1
    )
    
    hover_p = pval_fig.select(dict(type=HoverTool))[0]
    hover_p.tooltips = [
        ("Basis", "@basis_name (@basis_index)"),
        ("p-value", "@p_value{%0.2e}"),
        ("Pseudotime", "@pseudotime{0.00}")
    ]
    hover_p.formatters = {"@p_value": "printf"}

    cbar_p = ColorBar(color_mapper=cmap_p, title="p-value", orientation="horizontal", padding=0)
    pval_fig.add_layout(cbar_p, 'below')

    # Add Reconstruction Figure
    recon_fig = figure(
        width=800, height=300,
        title="Sample Reconstruction by Group (asinh)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
        x_axis_label="Pseudotime",
        x_range=sig_fig.x_range,
        active_scroll=None
    )
    recon_fig.yaxis.visible = False
    recon_fig.xaxis[0].ticker = sig_fig.xaxis[0].ticker
    if svg_export_mode:
        recon_fig.output_backend = "svg"
        
    t_eval = np.linspace(lo, hi, max(1000, K*10))
    A_eval = Phi(t_eval)
    
    from bokeh.palettes import Category10, Magma256
    
    n_groups = len(groups)
    if n_groups <= 10:
        palette = list(Category10[10])
    else:
        palette = [Magma256[i] for i in np.linspace(0, 255, n_groups, dtype=int)]
        
    # Calculate all data first
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
            group_data.append({
                'g': g,
                'g_color': g_color,
                'yhat_list': yhat_list
            })

    # Pass 1: Draw ALL thin gray lines first (so they are in the background)
    for data in group_data:
        xs = [t_eval for _ in range(len(data['yhat_list']))]
        ys = data['yhat_list']
        # line_width=0.8 is 20% thinner than 1.0
        recon_fig.multi_line(xs=xs, ys=ys, line_color="gray", line_alpha=0.5, line_width=0.8)

    # Pass 2: Draw ALL thick colored averages on top
    for data in group_data:
        y_avg = np.mean(data['yhat_list'], axis=0)
        recon_fig.line(x=t_eval, y=y_avg, line_color=data['g_color'], line_alpha=1.0, line_width=3, legend_label=str(data['g']))

    if recon_fig.legend:
        recon_fig.legend.location = "top_left"
        recon_fig.legend.orientation = "horizontal"
        recon_fig.legend.click_policy = "hide"
        recon_fig.add_layout(recon_fig.legend[0], 'above')

    jitter_src = ColumnDataSource(data=dict(x=[], y=[], group=[], sample_id=[]))
    jitter_fig = figure(
        width=600, height=360,
        x_axis_label="group",
        y_axis_label="loading",
        title="Loadings by group",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll=None,
    )
    if svg_export_mode:
        jitter_fig.output_backend = "svg"
    jitter_r = jitter_fig.circle("x", "y", source=jitter_src, size=7, fill_alpha=0.75, line_alpha=0.25)
    jitter_fig.add_tools(HoverTool(renderers=[jitter_r], tooltips=[("sample", "@sample_id"), ("group", "@group"), ("loading", "@y{0.000}")]))
    jitter_fig.xaxis.ticker = list(range(len(groups)))
    jitter_fig.xaxis.formatter = CustomJSTickFormatter(code=f"""
        const mapping = {repr(dict(enumerate(groups)))};
        return mapping[tick] || "";
    """)

    def _update_jitter():
        b = str(basis_sel.value)
        d = merged[["_sid", group_col, b]].copy()
        d[b] = pd.to_numeric(d[b], errors="coerce")
        d = d.dropna(subset=[b]).copy()

        rng = np.random.default_rng(0)
        x_map = {g: i for i, g in enumerate(groups)}
        xs = np.array([x_map[g] for g in d[group_col].astype(str)], dtype=float) + rng.normal(0, 0.06, size=len(d))
        ys = d[b].to_numpy(dtype=float)

        jitter_src.data = dict(
            x=xs,
            y=ys,
            group=d[group_col].astype(str).to_numpy(),
            sample_id=d["_sid"].astype(str).to_numpy(),
        )

        q = float(stats.loc[b, "q_value"]) if b in stats.index else float("nan")
        eff = float(stats.loc[b, "stat"]) if b in stats.index else float("nan")
        jitter_fig.title.text = f"Loadings by group: {b}   (q={q:.3g}, stat={eff:.3g})"

    basis_sel.param.watch(lambda *_: _update_jitter(), "value")

    # initial render
    _update_jitter()

    return pn.Column(
        pn.pane.Markdown("**Differential basis stats (Kruskal + BH-FDR)**"),
        stats_pane,
        pn.Row(download_btn),
        pn.layout.Divider(),
        pn.Row(pn.pane.Bokeh(recon_fig), sizing_mode="stretch_width"),
        pn.Row(pn.pane.Bokeh(sig_fig), sizing_mode="stretch_width"),
        pn.Row(pn.pane.Bokeh(pval_fig), sizing_mode="stretch_width"),
        pn.layout.Divider(),
        pn.Row(basis_sel),
        pn.Row(pn.pane.Bokeh(jitter_fig), sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


class NMFMetaDashboardController:
    def __init__(self):
        self.H_df: pd.DataFrame | None = None
        self.centers: np.ndarray | None = None
        self.metadata_df: pd.DataFrame | None = None

        self.file_input_meta = pn.widgets.FileInput(accept=".csv,.xlsx", multiple=False)
        self.file_input_nmf = pn.widgets.FileInput(accept=".csv", multiple=False)
        
        self.import_btn = pn.widgets.Button(name="Import Data", button_type="primary")
        self.import_btn.on_click(self._on_import_data)
        
        self.metadata_preview = pn.pane.DataFrame(pd.DataFrame(), max_height=200, sizing_mode="stretch_width", visible=False)
        self.nmf_preview = pn.pane.DataFrame(pd.DataFrame(), max_height=200, sizing_mode="stretch_width", visible=False)

        self.sample_col_select = pn.widgets.Select(name="Sample ID Column", options=[])
        self.group_col_select = pn.widgets.Select(name="Category Column", options=[])
        
        self.calc_basis_btn = pn.widgets.Button(name="Calculate Basis Regression Matrix", button_type="warning")
        self.calc_basis_btn.on_click(self._on_calc_basis)
        self.basis_reg_container = pn.Column(sizing_mode="stretch_width")
        
        self.min_group_size = pn.widgets.IntInput(name="Min Group Size", value=3, start=1)
        self.svg_export = pn.widgets.Checkbox(name="Enable SVG Export Mode", value=False)
        self.svg_export.param.watch(self._on_run, "value")

        self.roi_lo = pn.widgets.FloatInput(name="NMF ROI Start (min)", value=0.0, step=0.01, width=150)
        self.roi_hi = pn.widgets.FloatInput(name="NMF ROI End (max)", value=1.0, step=0.01, width=150)

        self.run_btn = pn.widgets.Button(name="Run Group Importance Analysis", button_type="primary", disabled=True)
        self.run_btn.on_click(self._on_run)

        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.plot_pane = TitledPlotPane(sizing_mode="stretch_both")
        self.dashboard_pane = pn.Column(sizing_mode="stretch_width")

        self.view = pn.Column(
            pn.pane.Markdown("## NMF Group Stats Dashboard\nUpload a metadata CSV/Excel file to find which NMF loadings explain group differences."),
            pn.pane.Markdown("**1. Provide NMF Loadings:**", styles={"color": "#555", "margin-top": "10px"}),
            pn.pane.Markdown("*(If you have already generated NMF loadings in the current session, this will automatically populate. Otherwise, upload a previously exported NMF matrix CSV.)*"),
            pn.Row(self.file_input_nmf),
            self.nmf_preview,
            pn.pane.Markdown("**2. Provide Metadata:**", styles={"color": "#555", "margin-top": "10px"}),
            pn.Row(self.file_input_meta),
            self.metadata_preview,
            pn.Row(self.import_btn),
            pn.pane.Markdown("**3. Set Parameters & Run Analysis:**", styles={"color": "#555", "margin-top": "10px"}),
            pn.pane.Markdown("*(Important: Make sure the pseudotime range below perfectly matches the ROI used to generate your NMF bases!)*", styles={"color": "darkred", "font-style": "italic", "font-size": "0.9em"}),
            pn.Row(self.roi_lo, self.roi_hi, self.min_group_size, pn.Column("Sample ID:", self.sample_col_select), pn.Column("Category:", self.group_col_select)),
            pn.Row(self.svg_export, pn.Spacer(width=12), self.run_btn),
            self.status,
            self.dashboard_pane,
            pn.layout.Divider(),
            pn.Row(self.calc_basis_btn),
            self.basis_reg_container,
            sizing_mode="stretch_width"
        )

    def set_H(self, H_df: pd.DataFrame, centers: np.ndarray | None = None):
        self.H_df = H_df.copy()
        self.centers = centers
        
        if self.centers is not None and len(self.centers) > 0:
            self.roi_lo.value = float(np.min(self.centers))
            self.roi_hi.value = float(np.max(self.centers))
        else:
            self.roi_lo.value = 0.0
            self.roi_hi.value = 1.0
        
        if "Unnamed: 0" not in self.H_df.columns:
            # If coming from live memory, the sample IDs are in the index.
            self.H_df = self.H_df.reset_index()
            self.H_df = self.H_df.rename(columns={self.H_df.columns[0]: "Unnamed: 0"})
            
        self.nmf_preview.object = self.H_df.head(5)
        self.nmf_preview.visible = True
        
        if self.metadata_df is not None:
            self.run_btn.disabled = False
            self.status.object = "NMF loadings loaded. Ready to run analysis."
        else:
            self.status.object = "NMF loadings loaded. Please upload metadata to continue."
        
    def _on_calc_basis(self, event):
        self._update_basis_regression()
        
    def _update_basis_regression(self):
        from bokeh.models import LinearColorMapper, HoverTool, ColorBar
        from bokeh.plotting import figure
        from scipy import stats
        import math
        
        if self.H_df is None or self.H_df.empty:
            self.basis_reg_container.clear()
            return
            
        cols = [c for c in self.H_df.columns if c != "Unnamed: 0"]
        if len(cols) < 2:
            self.basis_reg_container.clear()
            return
            
        data = []
        for i, v1 in enumerate(cols):
            for j, v2 in enumerate(cols):
                if i > j: # Half square (lower triangle)
                    df_pair = self.H_df[[v1, v2]].dropna()
                    if len(df_pair) < 3:
                        data.append({'var1': i+1, 'var2': j+1, 'name1': v1, 'name2': v2, 'r': np.nan, 'p': np.nan})
                        continue
                    try:
                        res = stats.linregress(df_pair[v2], df_pair[v1])
                        data.append({'var1': i+1, 'var2': j+1, 'name1': v1, 'name2': v2, 'r': res.rvalue, 'p': res.pvalue})
                    except Exception:
                        data.append({'var1': i+1, 'var2': j+1, 'name1': v1, 'name2': v2, 'r': np.nan, 'p': np.nan})
                        
        plot_df = pd.DataFrame(data)
        
        import common_plot
        cmap = LinearColorMapper(palette=common_plot.get_divergent_palette(256), low=-1.0, high=1.0, nan_color="lightgray")
        
        # Determine ranges
        min_var = 1
        max_var = len(cols)
        p = figure(title="Basis Regression Matrix (r-value)",
                   x_range=(0.5, max_var - 0.5), y_range=(max_var + 0.5, 1.5), # Reversed y so top-left is (1, 2)
                   x_axis_location="above", width=850, height=600,
                   x_axis_label="Basis #", y_axis_label="Basis #",
                   tools="hover,save,pan,wheel_zoom,box_zoom,reset", toolbar_location="right")
                   
        p.rect(x="var2", y="var1", width=1, height=1, source=plot_df,
               line_color=None, fill_color={"field": "r", "transform": cmap})
               
        hover = p.select_one(HoverTool)
        hover.tooltips = [
            ("Pair", "@name1 vs @name2"),
            ("r", "@r{0.000}"),
            ("p-value", "@p{0.00e-0}")
        ]
        
        color_bar = ColorBar(color_mapper=cmap, width=8, location=(0,0))
        p.add_layout(color_bar, 'right')
        
        if self.svg_export.value:
            p.output_backend = "svg"
            
        pane = TitledPlotPane(p, title="Basis Regression Matrix", sizing_mode="fixed", width=850, height=600)
        self.basis_reg_container.clear()
        self.basis_reg_container.append(pane)

    def _on_import_data(self, event):
        nmf_loaded = False
        meta_loaded = False
        
        if self.file_input_nmf.value:
            try:
                filename = self.file_input_nmf.filename
                content = self.file_input_nmf.value
                df = pd.read_csv(io.BytesIO(content))
                self.set_H(df)
                nmf_loaded = True
            except Exception as e:
                self.status.object = f"**Error parsing NMF file:** {e}"
                return
                
        if self.file_input_meta.value:
            try:
                filename = self.file_input_meta.filename
                content = self.file_input_meta.value
                if filename.endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(content))
                else:
                    df = pd.read_excel(io.BytesIO(content))
                    
                self.metadata_df = df
                cols = list(df.columns)
                self.sample_col_select.options = cols
                self.group_col_select.options = cols
                
                if len(cols) >= 2:
                    self.sample_col_select.value = cols[0]
                    self.group_col_select.value = cols[1]

                self.metadata_preview.object = df.head(5)
                self.metadata_preview.visible = True
                meta_loaded = True

            except Exception as e:
                self.status.object = f"**Error parsing metadata file:** {e}"
                self.metadata_df = None
                self.metadata_preview.visible = False
                self.run_btn.disabled = True
                return
                
        if nmf_loaded and meta_loaded:
            self.status.object = "Loaded both NMF and Metadata. Ready to run analysis."
        elif meta_loaded:
            if self.H_df is not None:
                self.status.object = "Loaded Metadata. Ready to run analysis."
            else:
                self.status.object = "Loaded Metadata, but waiting for NMF decomposition to finish."
        elif nmf_loaded:
            if self.metadata_df is not None:
                self.status.object = "Loaded NMF CSV. Ready to run analysis."
            else:
                self.status.object = "Loaded NMF CSV, but waiting for Metadata."
                
        if self.metadata_df is not None and self.H_df is not None:
            self.run_btn.disabled = False

    def _on_run(self, event):
        if self.H_df is None or self.metadata_df is None:
            self.status.object = "**Error:** Missing NMF loadings or metadata."
            return
            
        if not self.sample_col_select.value or not self.group_col_select.value:
            self.status.object = "**Error:** Must select Sample ID and Category columns."
            return

        self.status.object = "Running Kruskal-Wallis..."
        self.dashboard_pane.clear()
        
        try:
            dash = nmf_group_importance_dashboard_stable(
                self.metadata_df, self.H_df,
                group_col=self.group_col_select.value,
                meta_sample_col=self.sample_col_select.value,
                nmf_sample_col="Unnamed: 0",
                min_group_size=self.min_group_size.value,
                roi_range=(self.roi_lo.value, self.roi_hi.value),
                svg_export_mode=self.svg_export.value,
            )
            self.dashboard_pane.append(dash)
            from common_plot import apply_export_prefix_to_pane, TitledPlotPane
            prefix = getattr(self, "export_prefix", "CE_analysis")
            apply_export_prefix_to_pane(self.dashboard_pane, prefix)
            self.status.object = "Analysis complete."
        except Exception as e:
            self.status.object = f"**Analysis failed:** {e}"

def build_nmf_meta_section() -> tuple[pn.Column, NMFMetaDashboardController]:
    ctrl = NMFMetaDashboardController()
    return ctrl.view, ctrl
