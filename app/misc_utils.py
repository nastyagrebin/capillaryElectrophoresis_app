import io
import numpy as np
import pandas as pd
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, Legend
import scipy.stats as stats

pn.extension()

def no_bokeh_show():
    import bokeh.io
    class DummyContext:
        def __init__(self):
            self.orig = bokeh.io.show
        def __enter__(self):
            bokeh.io.show = lambda *args, **kw: None
        def __exit__(self, *args):
            bokeh.io.show = self.orig
    return DummyContext()

def fdr_bh(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    if n == 0: return pvals
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    qvals = np.zeros(n)
    qvals[-1] = sorted_pvals[-1]
    for i in range(n-2, -1, -1):
        qvals[i] = min(qvals[i+1], sorted_pvals[i] * n / (i + 1))
    orig_qvals = np.zeros(n)
    orig_qvals[sorted_idx] = qvals
    return orig_qvals

class MiscController:
    def __init__(self):
        self.master_df = None
        
        # Uploads
        self.loadings_file = pn.widgets.FileInput(accept=".csv,.xlsx", name="Loadings")
        self.meta_file = pn.widgets.FileInput(accept=".csv,.xlsx", name="Metadata")
        self.div_file = pn.widgets.FileInput(accept=".csv,.xlsx", name="Diversity")
        
        self.merge_btn = pn.widgets.Button(name="Merge Uploaded Data", button_type="primary")
        self.merge_btn.on_click(self._on_merge)
        
        self.svg_export = pn.widgets.Checkbox(name="SVG Export Mode (slower)", value=False)
        
        self.preview_pane = pn.pane.DataFrame(pd.DataFrame(), max_height=200, sizing_mode="stretch_width", visible=False)
        self.status = pn.pane.Markdown("")
        
        # Data Slicing
        self.rules = []
        self.sliced_df = None
        
        self.slice_col = pn.widgets.Select(name="Variable to slice by", options=[], width=200)
        self.slice_type = pn.widgets.RadioButtonGroup(name="Type", options=["Categorical", "Continuous"], value="Categorical")
        
        self.slice_cat_val = pn.widgets.MultiChoice(name="Categories to Keep", options=[], width=400)
        self.slice_cont_min = pn.widgets.FloatInput(name="Min", width=100)
        self.slice_cont_max = pn.widgets.FloatInput(name="Max", width=100)
        
        self.slice_input_area = pn.Row(self.slice_cat_val) # dynamic
        
        self.slice_add_btn = pn.widgets.Button(name="Add Rule", button_type="primary", width=100)
        self.slice_clear_btn = pn.widgets.Button(name="Clear Rules", button_type="danger", width=100)
        self.slice_apply_btn = pn.widgets.Button(name="Apply Filters", button_type="success", width=150)
        
        self.slice_rules_pane = pn.pane.Markdown("**Active Rules:** None", sizing_mode="stretch_width")
        self.slice_status = pn.pane.Markdown("")
        
        self.slice_download_btn = pn.widgets.FileDownload(
            filename="sliced_data.csv",
            button_type="primary",
            name="Download Sliced CSV",
            visible=False,
            width=200
        )
        
        self.slice_col.param.watch(self._update_slicing_ui, "value")
        self.slice_type.param.watch(self._update_slicing_ui, "value")
        self.slice_add_btn.on_click(self._add_rule)
        self.slice_clear_btn.on_click(self._clear_rules)
        self.slice_apply_btn.on_click(self._apply_slicing)

        
        # Continuous vs Continuous
        self.cvc_btn = pn.widgets.Button(name="Visualize", button_type="success")
        self.cvc_btn.on_click(self._update_cvc)
        self.cvc_x = pn.widgets.Select(name="X (Continuous)", options=[], width=150)
        self.cvc_y = pn.widgets.Select(name="Y (Continuous)", options=[], width=150)
        self.cvc_color = pn.widgets.Select(name="Color by", options=[], width=150)
        self.cvc_color_mode = pn.widgets.RadioButtonGroup(name="Color Mode", options=["categorical", "continuous"], value="categorical")
        self.cvc_corr = pn.widgets.RadioButtonGroup(name="Correlation", options=["pearson", "spearman"], value="pearson")
        self.cvc_reg = pn.widgets.Checkbox(name="Show Regression & Envelope", value=True)
        
        self.cvc_plot_pane = pn.pane.Bokeh(sizing_mode="fixed", width=600, height=400)
        self.cvc_stats_pane = pn.pane.Markdown("")
        
        # Categorical vs Continuous
        self.cat_btn = pn.widgets.Button(name="Visualize", button_type="success")
        self.cat_btn.on_click(self._update_cat)
        self.cat_x = pn.widgets.Select(name="X (Categorical)", options=[], width=150)
        self.cat_y = pn.widgets.Select(name="Y (Continuous)", options=[], width=150)
        self.cat_stat = pn.widgets.RadioButtonGroup(name="Test", options=["mann-whitney", "kruskal-wallis"], value="kruskal-wallis")
        self.cat_fdr = pn.widgets.Checkbox(name="FDR Correction", value=True)
        
        self.cat_plot_pane = pn.pane.Bokeh(sizing_mode="fixed", width=600, height=400)

        # Regression Matrix
        self.reg_btn = pn.widgets.Button(name="Visualize", button_type="success")
        self.reg_btn.on_click(self._update_reg)
        self.reg_vars = pn.widgets.MultiChoice(name="Continuous variables to correlate", options=[], width=500)
        self.reg_plot_pane = pn.pane.Bokeh(sizing_mode="fixed", width=850, height=600)
        self.reg_status = pn.pane.Markdown("")

        # Categorical Covariance Matrix
        self.cat_cov_btn = pn.widgets.Button(name="Visualize", button_type="success")
        self.cat_cov_btn.on_click(self._update_cat_cov)
        self.cat_cov_vars = pn.widgets.MultiChoice(name="Categorical variables to correlate", options=[], width=500)
        self.cat_cov_plot_pane = pn.pane.Bokeh(sizing_mode="fixed", width=850, height=600)
        self.cat_cov_status = pn.pane.Markdown("")
        
        self.reg_show_overlay = pn.widgets.Checkbox(name="Show text overlay", value=True)
        self.cat_cov_show_overlay = pn.widgets.Checkbox(name="Show text overlay", value=True)
        self.reg_show_overlay.param.watch(self._update_reg, "value")
        self.cat_cov_show_overlay.param.watch(self._update_cat_cov, "value")
        
        self.section = pn.Column(
            pn.pane.Markdown("## Miscellaneous Visualizations\nUpload standalone files, or let the session automatically bridge generated data here."),
            pn.pane.Markdown("### 1. Manual Data Upload"),
            pn.Row(
                pn.Column("Loadings (CSV/Excel)", self.loadings_file),
                pn.Column("Metadata (CSV/Excel)", self.meta_file),
                pn.Column("Diversity (CSV/Excel)", self.div_file)
            ),
            pn.Row(self.merge_btn, self.status),
            self.preview_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### Data Slicing"),
            pn.Row(self.slice_col, pn.Column("Type:", self.slice_type), self.slice_input_area, pn.Column("", self.slice_add_btn)),
            self.slice_rules_pane,
            pn.Row(self.slice_apply_btn, self.slice_clear_btn, self.slice_download_btn),
            self.slice_status,
            pn.layout.Divider(),
            pn.Row(self.svg_export),
            pn.pane.Markdown("### Continuous vs Continuous"),
            pn.Row(self.cvc_x, self.cvc_y, self.cvc_color, pn.Column("Color mode:", self.cvc_color_mode)),
            pn.Row(self.cvc_corr, self.cvc_reg, pn.Spacer(width=12), self.cvc_btn),
            self.cvc_stats_pane,
            self.cvc_plot_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### Categorical vs Continuous"),
            pn.Row(self.cat_x, self.cat_y),
            pn.Row(self.cat_stat, self.cat_fdr, pn.Spacer(width=12), self.cat_btn),
            self.cat_plot_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### Regression Matrix (Continuous)"),
            pn.Row(self.reg_vars, pn.Spacer(width=12), self.reg_show_overlay, self.reg_btn),
            self.reg_status,
            self.reg_plot_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### Covariance Matrix (Categorical)"),
            pn.Row(self.cat_cov_vars, pn.Spacer(width=12), self.cat_cov_show_overlay, self.cat_cov_btn),
            self.cat_cov_status,
            self.cat_cov_plot_pane,
            sizing_mode="stretch_width"
        )

    def _get_df(self):
        if getattr(self, 'sliced_df', None) is not None:
            return self.sliced_df
        return self.master_df

    def _update_slicing_ui(self, *_):
        if self.master_df is None or not self.slice_col.value: return
        
        col = self.slice_col.value
        if self.slice_type.value == "Categorical":
            unique_vals = [str(x) for x in self.master_df[col].dropna().unique()]
            self.slice_cat_val.options = sorted(unique_vals)
            self.slice_cat_val.value = []
            self.slice_input_area[:] = [self.slice_cat_val]
        else:
            s = pd.to_numeric(self.master_df[col], errors='coerce').dropna()
            if not s.empty:
                self.slice_cont_min.value = float(s.min())
                self.slice_cont_max.value = float(s.max())
            self.slice_input_area[:] = [self.slice_cont_min, self.slice_cont_max]

    def _add_rule(self, *_):
        if not self.slice_col.value: return
        col = self.slice_col.value
        stype = self.slice_type.value
        
        if stype == "Categorical":
            vals = self.slice_cat_val.value
            if not vals: return
            self.rules.append({
                "col": col, "type": "Categorical", "vals": vals,
                "desc": f"`{col}` in {vals}"
            })
        else:
            vmin = self.slice_cont_min.value
            vmax = self.slice_cont_max.value
            if vmin is None or vmax is None: return
            self.rules.append({
                "col": col, "type": "Continuous", "min": vmin, "max": vmax,
                "desc": f"{vmin} <= `{col}` <= {vmax}"
            })
            
        self._update_rules_display()
        
    def _clear_rules(self, *_):
        self.rules = []
        self._update_rules_display()
        
    def _update_rules_display(self):
        if not self.rules:
            self.slice_rules_pane.object = "**Active Rules:** None"
        else:
            lines = ["**Active Rules:**"]
            for i, r in enumerate(self.rules):
                lines.append(f"{i+1}. {r['desc']}")
            self.slice_rules_pane.object = "\n".join(lines)

    def _apply_slicing(self, *_):
        if self.master_df is None: return
        
        df = self.master_df.copy()
        for r in self.rules:
            col = r["col"]
            if r["type"] == "Categorical":
                df = df[df[col].astype(str).isin(r["vals"])]
            else:
                s = pd.to_numeric(df[col], errors='coerce')
                df = df[(s >= r["min"]) & (s <= r["max"])]
                
        self.sliced_df = df.copy()
        self.slice_status.object = f"**Sliced dataframe created:** {len(self.sliced_df)} rows remaining (from {len(self.master_df)})."
        self.preview_pane.object = self.sliced_df.head(10)
        
        sio = io.BytesIO()
        self.sliced_df.to_csv(sio)
        sio.seek(0)
        self.slice_download_btn.file = sio
        self.slice_download_btn.visible = True

    def _read_file(self, file_input):
        if not file_input.value: return None
        if file_input.filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_input.value), index_col=0)
        else:
            return pd.read_excel(io.BytesIO(file_input.value), index_col=0)

    def _on_merge(self, event):
        dfs = []
        l = self._read_file(self.loadings_file)
        m = self._read_file(self.meta_file)
        d = self._read_file(self.div_file)
        if m is not None: dfs.append(m)
        if d is not None: dfs.append(d)
        if l is not None: dfs.append(l)
        
        if not dfs:
            self.status.object = "**No files uploaded to merge.**"
            return
            
        master = dfs[0]
        for df in dfs[1:]:
            master = master.merge(df, left_index=True, right_index=True, how="outer", suffixes=("", "_dup"))
        master = master.loc[:, ~master.columns.duplicated()].copy()
            
        self.bridge_data(master_df=master)
        self.status.object = f"**Successfully merged {len(dfs)} uploaded files.**"

    def bridge_data(self, loadings_df=None, meta_df=None, div_df=None, master_df=None):
        if master_df is not None:
            self.master_df = master_df.loc[:, ~master_df.columns.duplicated()].copy()
        else:
            dfs = []
            if meta_df is not None and not meta_df.empty: dfs.append(meta_df.copy())
            if div_df is not None and not div_df.empty: dfs.append(div_df.copy())
            if loadings_df is not None and not loadings_df.empty: dfs.append(loadings_df.copy())
            if not dfs: return
            
            master = dfs[0]
            for d in dfs[1:]:
                master = master.merge(d, left_index=True, right_index=True, how='outer', suffixes=("", "_dup"))
            self.master_df = master.loc[:, ~master.columns.duplicated()].copy()
            
        self._populate_dropdowns()

    def _populate_dropdowns(self):
        if self.master_df is None: return
        self.preview_pane.object = self.master_df.head(10)
        self.preview_pane.visible = True
        self.sliced_df = None
        self._clear_rules()
        self.slice_status.object = ""
        self.slice_download_btn.visible = False
        
        cols = list(self.master_df.columns)
        num_cols = []
        cat_cols = []
        for c in cols:
            if pd.api.types.is_numeric_dtype(self.master_df[c]):
                num_cols.append(c)
            else:
                try:
                    pd.to_numeric(self.master_df[c])
                    num_cols.append(c)
                except:
                    cat_cols.append(c)
                    
        self.cvc_x.options = num_cols
        self.cvc_y.options = num_cols
        self.cvc_color.options = ["None"] + cols
        
        self.cat_x.options = cols
        self.cat_y.options = num_cols

        self.reg_vars.options = num_cols
        self.reg_vars.value = [c for c in num_cols if "basis" not in str(c).lower()]
        
        self.cat_cov_vars.options = cols
        self.cat_cov_vars.value = [c for c in cols if c not in num_cols]
        
        if num_cols:
            self.cvc_x.value = num_cols[0]
            self.cvc_y.value = num_cols[min(1, len(num_cols)-1)]
            self.cat_y.value = num_cols[0]
        if cols:
            self.cat_x.value = cols[0]
            
        self.slice_col.options = cols
        if cols: self.slice_col.value = cols[0]
        self._update_slicing_ui()
            
        # Do not automatically update plots here. Let user hit Visualize.

    def _update_cvc(self, *_):
        self.cvc_plot_pane.object = None
        df_source = self._get_df()
        if df_source is None or df_source.empty:
            self.cvc_stats_pane.object = "**Error: No data available.**"
            return
        if not self.cvc_x.value or not self.cvc_y.value:
            return
            
        x_col = self.cvc_x.value
        y_col = self.cvc_y.value
        color_col = self.cvc_color.value
        
        x_s = pd.to_numeric(df_source[x_col], errors='coerce')
        if isinstance(x_s, pd.DataFrame): x_s = x_s.iloc[:, 0]
        y_s = pd.to_numeric(df_source[y_col], errors='coerce')
        if isinstance(y_s, pd.DataFrame): y_s = y_s.iloc[:, 0]
        
        df = pd.DataFrame({x_col: x_s, y_col: y_s})
        
        if color_col is not None and color_col != "None" and color_col in df_source.columns:
            c_s = df_source[color_col]
            if isinstance(c_s, pd.DataFrame): c_s = c_s.iloc[:, 0]
            df[color_col] = c_s
        else:
            color_col = "None"
        
        df = df.dropna(subset=[x_col, y_col])
        if df.empty:
            return
            
        x = df[x_col].values
        y = df[y_col].values
        
        if self.cvc_corr.value == "pearson":
            r, p = stats.pearsonr(x, y)
            stat_txt = f"**Pearson r:** {r:.3f} (p-value: {p:.3e})"
        else:
            r, p = stats.spearmanr(x, y)
            stat_txt = f"**Spearman r:** {r:.3f} (p-value: {p:.3e})"
        self.cvc_stats_pane.object = stat_txt
        
        p_fig = figure(width=400, height=400, title=f"{y_col} vs {x_col}",
                       x_axis_label=x_col, y_axis_label=y_col,
                       tools="pan,wheel_zoom,box_zoom,reset,save",
                       output_backend="svg" if self.svg_export.value else "canvas")
                       
        if color_col == "None":
            p_fig.circle(x, y, size=8, alpha=0.7)
        else:
            c_vals = df[color_col]
            is_cont = False
            if self.cvc_color_mode.value == "continuous":
                c_vals = pd.to_numeric(c_vals, errors='coerce')
                is_cont = True
                
            if is_cont:
                from bokeh.palettes import Magma256
                valid_c = c_vals.dropna()
                c_min = valid_c.min() if len(valid_c) > 0 else 0
                c_max = valid_c.max() if len(valid_c) > 0 else 1
                mapper = LinearColorMapper(palette=Magma256, low=c_min, high=c_max)
                src = ColumnDataSource(dict(x=x, y=y, c=c_vals))
                p_fig.circle("x", "y", size=8, alpha=0.7, fill_color={"field": "c", "transform": mapper}, line_color=None, source=src)
                bar = ColorBar(color_mapper=mapper, title=color_col)
                p_fig.add_layout(bar, "right")
            else:
                c_vals = c_vals.astype(str)
                unique = sorted(c_vals.unique())
                from bokeh.palettes import Category10, Magma256
                palette = list(Category10[10]) if len(unique) <= 10 else [Magma256[i] for i in np.linspace(0, 255, len(unique), dtype=int)]
                cmap = {c: palette[i%len(palette)] for i, c in enumerate(unique)}
                
                from bokeh.models import Legend, LegendItem
                color_items = []
                for c in unique:
                    mask = (c_vals == c).values
                    src = ColumnDataSource(dict(x=x[mask], y=y[mask]))
                    renderer = p_fig.circle("x", "y", size=8, alpha=0.7, fill_color=cmap[c], line_color=None, source=src)
                    color_items.append(LegendItem(label=str(c)[:12], renderers=[renderer]))
                
                if color_items:
                    color_legend = Legend(items=color_items, click_policy="hide")
                    p_fig.add_layout(color_legend, "right")

        if self.cvc_reg.value and len(x) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_seq = np.linspace(x.min(), x.max(), 100)
            y_seq = intercept + slope * x_seq
            if self.cvc_corr.value == "pearson":
                reg_label = f"Regression (r={r:.3f}, p={p:.3e})"
            else:
                reg_label = f"Regression (rho={r:.3f}, p={p:.3e})"
            reg_renderer = p_fig.line(x_seq, y_seq, color='red', line_width=2)
            
            n = len(x)
            if n > 2:
                y_pred = intercept + slope * x
                sse = np.sum((y - y_pred)**2)
                s_e = np.sqrt(sse / (n - 2))
                t_val = stats.t.ppf(0.975, n - 2)
                x_mean = np.mean(x)
                me = t_val * s_e * np.sqrt(1/n + (x_seq - x_mean)**2 / np.sum((x - x_mean)**2))
                y_ci_lower = y_seq - me
                y_ci_upper = y_seq + me
                
                p_fig.patch(
                    np.append(x_seq, x_seq[::-1]),
                    np.append(y_ci_lower, y_ci_upper[::-1]),
                    color='red', alpha=0.2, line_width=0
                )
            
            from bokeh.models import Legend, LegendItem
            reg_legend = Legend(items=[LegendItem(label=reg_label, renderers=[reg_renderer])], click_policy="hide")
            p_fig.add_layout(reg_legend, "below")

        self.cvc_plot_pane.object = p_fig
        from common_plot import apply_export_prefix_to_pane
        prefix = getattr(self, "export_prefix", "CE_analysis")
        apply_export_prefix_to_pane(self.cvc_plot_pane, prefix)

    def _update_cat(self, *_):
        self.cat_plot_pane.object = None
        df_source = self._get_df()
        if df_source is None or df_source.empty:
            return
        if not self.cat_x.value or not self.cat_y.value:
            return
            
        x_col = self.cat_x.value
        y_col = self.cat_y.value
        
        x_s = df_source[x_col]
        if isinstance(x_s, pd.DataFrame): x_s = x_s.iloc[:, 0]
        y_s = pd.to_numeric(df_source[y_col], errors='coerce')
        if isinstance(y_s, pd.DataFrame): y_s = y_s.iloc[:, 0]
        
        df = pd.DataFrame({x_col: x_s.astype(str), y_col: y_s})
        df = df.dropna(subset=[y_col])
        
        if df.empty:
            self.cat_plot_pane.object = None
            return
            
        groups = sorted(df[x_col].unique())
        if len(groups) < 2:
            self.cat_plot_pane.object = None
            return
            
        import itertools
        pairs = list(itertools.combinations(groups, 2))
        pvals = []
        for g1, g2 in pairs:
            v1 = df[df[x_col] == g1][y_col].values
            v2 = df[df[x_col] == g2][y_col].values
            if len(v1) == 0 or len(v2) == 0:
                pvals.append(1.0)
                continue
            if self.cat_stat.value == "mann-whitney":
                try:
                    _, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                except:
                    p = 1.0
            else:
                try:
                    _, p = stats.kruskal(v1, v2)
                except:
                    p = 1.0
            pvals.append(p)
            
        if self.cat_fdr.value and len(pvals) > 0:
            pvals = fdr_bh(pvals)
            
        def get_asterisks(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            return "n.s."
            
        import math
        p_fig = figure(width=600, height=400, title=f"{y_col} by {x_col}",
                       tools="pan,wheel_zoom,box_zoom,reset,save",
                       output_backend="svg" if self.svg_export.value else "canvas")
                       
        p_fig.xaxis.ticker = list(range(len(groups)))
        p_fig.xaxis.major_label_overrides = {i: g for i, g in enumerate(groups)}
        p_fig.xaxis.major_label_orientation = math.pi / 4
        
        rng = np.random.default_rng(42)
        for i, g in enumerate(groups):
            vals = df[df[x_col] == g][y_col].values
            if len(vals) == 0: continue
            xs = i + rng.uniform(-0.15, 0.15, size=len(vals))
            p_fig.circle(xs, vals, size=6, alpha=0.6)
            med = np.median(vals)
            p_fig.line([i-0.2, i+0.2], [med, med], color='black', line_width=2)
            
        max_y = df[y_col].max()
        y_range = max_y - df[y_col].min()
        if y_range == 0: y_range = 1
        step = y_range * 0.1
        current_y = max_y + step
        
        for (g1, g2), p in zip(pairs, pvals):
            ast = get_asterisks(p)
            if ast == "n.s.": continue
            
            i1 = groups.index(g1)
            i2 = groups.index(g2)
            
            p_fig.line([i1, i1, i2, i2], [current_y - step*0.2, current_y, current_y, current_y - step*0.2], color='black', line_width=1.5)
            p_fig.text(x=[(i1+i2)/2], y=[current_y], text=[ast], text_align="center", text_baseline="bottom")
            current_y += step
            
        self.cat_plot_pane.object = p_fig
        from common_plot import apply_export_prefix_to_pane
        prefix = getattr(self, "export_prefix", "CE_analysis")
        apply_export_prefix_to_pane(self.cat_plot_pane, prefix)

    def _update_reg(self, *_):
        from bokeh.models import LinearColorMapper, HoverTool, ColorBar
        from bokeh.plotting import figure
        from scipy import stats
        import math
        
        df_source = self._get_df()
        if df_source is None or df_source.empty:
            self.reg_status.object = "**Error: No data available.**"
            return
            
        vars = self.reg_vars.value
        if not vars or len(vars) < 2:
            self.reg_status.object = "**Error: Please select at least two variables.**"
            return
            
        data = []
        for i, v1 in enumerate(vars):
            for j, v2 in enumerate(vars):
                if i > j: # Half square (lower triangle)
                    df_pair = df_source[[v1, v2]].dropna()
                    if len(df_pair) < 3:
                        data.append({'var1': v1, 'var2': v2, 'r': np.nan, 'p': np.nan})
                        continue
                    try:
                        res = stats.linregress(df_pair[v2], df_pair[v1])
                        data.append({'var1': v1, 'var2': v2, 'r': res.rvalue, 'p': res.pvalue})
                    except Exception:
                        data.append({'var1': v1, 'var2': v2, 'r': np.nan, 'p': np.nan})
                        
        plot_df = pd.DataFrame(data)
        def format_text(row):
            if pd.isna(row['r']): return ""
            p = row['p']
            if pd.isna(p): return f"r={row['r']:.2f}\np=NaN"
            p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
            return f"r={row['r']:.2f}\np={p_str}"
        plot_df['text'] = plot_df.apply(format_text, axis=1)
        
        from bokeh.palettes import RdBu
        cmap = LinearColorMapper(palette=list(reversed(RdBu[11])), low=-1.0, high=1.0, nan_color="lightgray")
        
        p = figure(title="Regression Matrix (r-value)",
                   x_range=vars[:-1], y_range=list(reversed(vars[1:])),
                   x_axis_location="above", width=850, height=600,
                   tools="hover,save,pan,wheel_zoom,box_zoom,reset", toolbar_location="right")
                   
        p.xaxis.major_label_orientation = math.pi / 4
        
        p.rect(x="var2", y="var1", width=1, height=1, source=plot_df,
               line_color="white", fill_color={"field": "r", "transform": cmap})
               
        if self.reg_show_overlay.value:
            p.text(x="var2", y="var1", text="text", text_color="gray", 
                   text_align="center", text_baseline="middle", 
                   text_font_size="10pt", source=plot_df)
               
        hover = p.select_one(HoverTool)
        hover.tooltips = [
            ("Pair", "@var1 vs @var2"),
            ("r", "@r{0.000}"),
            ("p-value", "@p{0.00e-0}")
        ]
        
        color_bar = ColorBar(color_mapper=cmap, width=8, location=(0,0))
        p.add_layout(color_bar, 'right')
        
        if self.svg_export.value:
            p.output_backend = "svg"
            
        self.reg_plot_pane.object = p
        from common_plot import apply_export_prefix_to_pane
        prefix = getattr(self, "export_prefix", "CE_analysis")
        apply_export_prefix_to_pane(self.reg_plot_pane, prefix)
        self.reg_status.object = ""


    def _update_cat_cov(self, *_):
        from bokeh.models import LinearColorMapper, HoverTool, ColorBar
        from bokeh.plotting import figure
        import scipy.stats as ss
        import math
        
        def cramers_v_and_p(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2, p_value, _, _ = ss.chi2_contingency(confusion_matrix)
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            if n == 0 or min(r, k) == 1:
                return np.nan, np.nan
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            denom = min((kcorr-1), (rcorr-1))
            if denom == 0: return np.nan, np.nan
            v = np.sqrt(phi2corr / denom)
            return v, p_value

        df_source = self._get_df()
        if df_source is None or df_source.empty:
            self.cat_cov_status.object = "**Error: No data available.**"
            return
            
        vars = self.cat_cov_vars.value
        if not vars or len(vars) < 2:
            self.cat_cov_status.object = "**Error: Please select at least two variables.**"
            return
            
        data = []
        for i, v1 in enumerate(vars):
            for j, v2 in enumerate(vars):
                if i > j:
                    df_pair = df_source[[v1, v2]].dropna()
                    if len(df_pair) < 3:
                        data.append({'var1': v1, 'var2': v2, 'v': np.nan, 'p': np.nan})
                        continue
                    try:
                        v_val, p_val = cramers_v_and_p(df_pair[v1], df_pair[v2])
                        data.append({'var1': v1, 'var2': v2, 'v': v_val, 'p': p_val})
                    except Exception:
                        data.append({'var1': v1, 'var2': v2, 'v': np.nan, 'p': np.nan})
                        
        plot_df = pd.DataFrame(data)
        def format_text(row):
            if pd.isna(row['v']):
                return ""
            p = row['p']
            if pd.isna(p): return f"V={row['v']:.2f}\np=NaN"
            p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
            return f"V={row['v']:.2f}\np={p_str}"
        plot_df['text'] = plot_df.apply(format_text, axis=1)
        
        from bokeh.palettes import Blues
        cmap = LinearColorMapper(palette=list(reversed(Blues[9])), low=0.0, high=1.0, nan_color="lightgray")
        
        p = figure(title="Categorical Covariance Matrix (Cramér's V)",
                   x_range=vars[:-1], y_range=list(reversed(vars[1:])),
                   x_axis_location="above", width=850, height=600,
                   tools="hover,save,pan,wheel_zoom,box_zoom,reset", toolbar_location="right")
                   
        p.xaxis.major_label_orientation = math.pi / 4
        
        p.rect(x="var2", y="var1", width=1, height=1, source=plot_df,
               line_color="white", fill_color={"field": "v", "transform": cmap})
               
        if self.cat_cov_show_overlay.value:
            p.text(x="var2", y="var1", text="text", text_color="gray", 
                   text_align="center", text_baseline="middle", 
                   text_font_size="10pt", source=plot_df)
               
        hover = p.select_one(HoverTool)
        hover.tooltips = [
            ("Pair", "@var1 vs @var2"),
            ("Cramér's V", "@v{0.000}"),
            ("p-value", "@p{0.00e-0}")
        ]
        
        color_bar = ColorBar(color_mapper=cmap, width=8, location=(0,0))
        p.add_layout(color_bar, 'right')
        
        if self.svg_export.value:
            p.output_backend = "svg"
            
        self.cat_cov_plot_pane.object = p
        from common_plot import apply_export_prefix_to_pane
        prefix = getattr(self, "export_prefix", "CE_analysis")
        apply_export_prefix_to_pane(self.cat_cov_plot_pane, prefix)
        self.cat_cov_status.object = ""

def build_misc_section():

    ctrl = MiscController()
    return ctrl.section, ctrl
