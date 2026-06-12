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
        
        self.visualize_btn = pn.widgets.Button(name="Visualize", button_type="success")
        self.visualize_btn.on_click(self._on_visualize)
        
        self.preview_pane = pn.pane.DataFrame(pd.DataFrame(), max_height=200, sizing_mode="stretch_width", visible=False)
        self.status = pn.pane.Markdown("")
        
        # Continuous vs Continuous
        self.cvc_x = pn.widgets.Select(name="X (Continuous)", options=[], width=150)
        self.cvc_y = pn.widgets.Select(name="Y (Continuous)", options=[], width=150)
        self.cvc_color = pn.widgets.Select(name="Color by", options=[], width=150)
        self.cvc_color_mode = pn.widgets.RadioButtonGroup(name="Color Mode", options=["categorical", "continuous"], value="categorical")
        self.cvc_corr = pn.widgets.RadioButtonGroup(name="Correlation", options=["pearson", "spearman"], value="pearson")
        self.cvc_reg = pn.widgets.Checkbox(name="Show Regression & Envelope", value=True)
        
        self.cvc_plot_pane = pn.pane.Bokeh(sizing_mode="fixed", width=600, height=400)
        self.cvc_stats_pane = pn.pane.Markdown("")
        # Categorical vs Continuous
        self.cat_x = pn.widgets.Select(name="X (Categorical)", options=[], width=150)
        self.cat_y = pn.widgets.Select(name="Y (Continuous)", options=[], width=150)
        self.cat_stat = pn.widgets.RadioButtonGroup(name="Test", options=["mann-whitney", "kruskal-wallis"], value="kruskal-wallis")
        self.cat_fdr = pn.widgets.Checkbox(name="FDR Correction", value=True)
        
        self.cat_plot_pane = pn.pane.Bokeh(sizing_mode="fixed", width=600, height=400)
        
        # Removed param.watch bindings to allow manual Visualize button trigger
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
            pn.Row(self.visualize_btn),
            pn.pane.Markdown("### Continuous vs Continuous"),
            pn.Row(self.cvc_x, self.cvc_y, self.cvc_color, pn.Column("Color mode:", self.cvc_color_mode)),
            pn.Row(self.cvc_corr, self.cvc_reg),
            self.cvc_stats_pane,
            self.cvc_plot_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### Categorical vs Continuous"),
            pn.Row(self.cat_x, self.cat_y),
            pn.Row(self.cat_stat, self.cat_fdr),
            self.cat_plot_pane,
            sizing_mode="stretch_width"
        )

    def _on_visualize(self, event):
        self._update_cvc()
        self._update_cat()

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
        
        if num_cols:
            self.cvc_x.value = num_cols[0]
            self.cvc_y.value = num_cols[min(1, len(num_cols)-1)]
            self.cat_y.value = num_cols[0]
        if cols:
            self.cat_x.value = cols[0]
            
        # Do not automatically update plots here. Let user hit Visualize.

    def _update_cvc(self, *_):
        self.cvc_plot_pane.object = None
        if self.master_df is None or not self.cvc_x.value or not self.cvc_y.value:
            return
            
        x_col = self.cvc_x.value
        y_col = self.cvc_y.value
        color_col = self.cvc_color.value
        
        x_s = pd.to_numeric(self.master_df[x_col], errors='coerce')
        if isinstance(x_s, pd.DataFrame): x_s = x_s.iloc[:, 0]
        y_s = pd.to_numeric(self.master_df[y_col], errors='coerce')
        if isinstance(y_s, pd.DataFrame): y_s = y_s.iloc[:, 0]
        
        df = pd.DataFrame({x_col: x_s, y_col: y_s})
        
        if color_col is not None and color_col != "None" and color_col in self.master_df.columns:
            c_s = self.master_df[color_col]
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
                       tools="pan,wheel_zoom,box_zoom,reset,save")
                       
        if color_col == "None":
            p_fig.circle(x, y, size=8, alpha=0.7)
        else:
            c_vals = df[color_col]
            is_cont = False
            if self.cvc_color_mode.value == "continuous":
                c_vals = pd.to_numeric(c_vals, errors='coerce')
                is_cont = True
                
            if is_cont:
                from bokeh.palettes import Viridis256
                valid_c = c_vals.dropna()
                c_min = valid_c.min() if len(valid_c) > 0 else 0
                c_max = valid_c.max() if len(valid_c) > 0 else 1
                mapper = LinearColorMapper(palette=Viridis256, low=c_min, high=c_max)
                src = ColumnDataSource(dict(x=x, y=y, c=c_vals))
                p_fig.circle("x", "y", size=8, alpha=0.7, fill_color={"field": "c", "transform": mapper}, line_color=None, source=src)
                bar = ColorBar(color_mapper=mapper, title=color_col)
                p_fig.add_layout(bar, "right")
            else:
                c_vals = c_vals.astype(str)
                unique = sorted(c_vals.unique())
                from bokeh.palettes import Category10, Viridis256
                palette = list(Category10[10]) if len(unique) <= 10 else [Viridis256[i] for i in np.linspace(0, 255, len(unique), dtype=int)]
                cmap = {c: palette[i%len(palette)] for i, c in enumerate(unique)}
                
                for c in unique:
                    mask = (c_vals == c).values
                    src = ColumnDataSource(dict(x=x[mask], y=y[mask]))
                    p_fig.circle("x", "y", size=8, alpha=0.7, fill_color=cmap[c], line_color=None, source=src, legend_label=str(c)[:12])

        if self.cvc_reg.value and len(x) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_seq = np.linspace(x.min(), x.max(), 100)
            y_seq = intercept + slope * x_seq
            p_fig.line(x_seq, y_seq, color='red', line_width=2, legend_label='Regression')
            
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

        if p_fig.legend:
            p_fig.legend.click_policy = "hide"
            try:
                p_fig.add_layout(p_fig.legend[0], "right")
            except Exception:
                pass

        self.cvc_plot_pane.object = p_fig

    def _update_cat(self, *_):
        if self.master_df is None or not self.cat_x.value or not self.cat_y.value:
            return
            
        x_col = self.cat_x.value
        y_col = self.cat_y.value
        
        x_s = self.master_df[x_col]
        if isinstance(x_s, pd.DataFrame): x_s = x_s.iloc[:, 0]
        y_s = pd.to_numeric(self.master_df[y_col], errors='coerce')
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
                       tools="pan,wheel_zoom,box_zoom,reset,save")
                       
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

def build_misc_section():
    ctrl = MiscController()
    return ctrl.section, ctrl
