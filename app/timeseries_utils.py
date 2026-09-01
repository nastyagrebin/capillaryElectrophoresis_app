import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar
from bokeh.palettes import Category10, Magma256

class TimeSeriesController:
    def __init__(self):
        self.master_df = None
        self.sliced_df = None
        self.rules = []
        
        # Uploads
        self.loadings_file = pn.widgets.FileInput(accept=".csv,.xlsx", name="Loadings")
        self.meta_file = pn.widgets.FileInput(accept=".csv,.xlsx", name="Metadata")
        self.div_file = pn.widgets.FileInput(accept=".csv,.xlsx", name="Diversity")
        
        self.merge_btn = pn.widgets.Button(name="Merge Uploaded Data", button_type="primary")
        self.merge_btn.on_click(self._on_merge)
        
        self.visualize_btn = pn.widgets.Button(name="Visualize", button_type="success")
        self.visualize_btn.on_click(self._on_visualize)
        
        self.svg_export = pn.widgets.Checkbox(name="SVG Export Mode (slower)", value=False)
        
        self.preview_pane = pn.pane.DataFrame(pd.DataFrame(), max_height=200, sizing_mode="stretch_width", visible=False)
        self.status = pn.pane.Markdown("")
        
        # Controls
        self.ts_x = pn.widgets.Select(name="Time Column (X-axis)", options=[], width=150)
        self.ts_x_type = pn.widgets.RadioButtonGroup(name="Time Type", options=["Continuous", "Categorical"], value="Continuous")
        self.ts_x_order = pn.widgets.TextInput(name="Categorical Order (comma separated)", placeholder="e.g. Day1, Day2, Day3", visible=False, width=300)
        
        self.ts_y = pn.widgets.Select(name="Y-axis", options=[], width=150)
        self.ts_y_mode = pn.widgets.Select(name="Y-axis Mode", options=["Actual Value", "Net Change from Initial", "Change from Previous"], width=200)
        self.ts_patient = pn.widgets.Select(name="Patient ID", options=[], width=150)
        self.ts_color = pn.widgets.Select(name="Color by (Categorical)", options=[], width=150)
        
        self.plot_pane = pn.pane.Bokeh(sizing_mode="fixed", width=800, height=600)
        
        # Markov UI
        self.markov_state = pn.widgets.Select(name="State Variable", options=[], width=150)
        self.markov_cov = pn.widgets.Select(name="Covariate to Test", options=[], width=150)
        self.markov_timing = pn.widgets.Select(name="Covariate Timing", options=["Time A (Baseline)", "Time B (Outcome)", "Change (B - A)"], width=150)
        self.markov_btn = pn.widgets.Button(name="Run Markov Analysis", button_type="primary")
        self.markov_btn.on_click(self._on_markov_run)
        self.markov_status = pn.pane.Markdown("")
        self.markov_plot = pn.pane.Bokeh(sizing_mode="fixed", width=500, height=400, visible=False)
        self.markov_table = pn.pane.DataFrame(pd.DataFrame(), sizing_mode="stretch_width", visible=False)
        
        # Slicing UI
        self.slice_col = pn.widgets.Select(name="Variable to Slice By", options=[])
        self.slice_type = pn.widgets.RadioButtonGroup(name="Type", options=["Categorical", "Continuous"], value="Categorical")
        self.slice_val_cat = pn.widgets.MultiChoice(name="Categories to Include", options=[], visible=True)
        self.slice_val_cont_min = pn.widgets.FloatInput(name="Min Value", value=0.0, visible=False)
        self.slice_val_cont_max = pn.widgets.FloatInput(name="Max Value", value=100.0, visible=False)
        self.slice_add_btn = pn.widgets.Button(name="Add Rule", button_type="primary")
        self.slice_clear_btn = pn.widgets.Button(name="Clear Rules", button_type="danger")
        self.slice_apply_btn = pn.widgets.Button(name="Apply Filters", button_type="success")
        self.slice_rules_md = pn.pane.Markdown("**Active Rules:** None", sizing_mode="stretch_width")
        
        self.slice_download = pn.widgets.FileDownload(
            filename="sliced_timeseries_data.csv",
            callback=self._get_sliced_csv,
            button_type="success",
            visible=False
        )
        
        # Watchers
        self.ts_x_type.param.watch(self._toggle_x_order, 'value')
        self.slice_col.param.watch(self._update_slicing_ui, 'value')
        self.slice_type.param.watch(self._update_slicing_ui, 'value')
        self.slice_add_btn.on_click(self._add_rule)
        self.slice_clear_btn.on_click(self._clear_rules)
        self.slice_apply_btn.on_click(self._apply_slicing)
        
        self.section = pn.Column(
            pn.pane.Markdown("## Time Series Plot\nUpload standalone files, or let the session automatically bridge generated data here."),
            pn.pane.Markdown("### 1. Manual Data Upload"),
            pn.Row(
                pn.Column("Loadings (CSV/Excel)", self.loadings_file),
                pn.Column("Metadata (CSV/Excel)", self.meta_file),
                pn.Column("Diversity (CSV/Excel)", self.div_file)
            ),
            pn.Row(self.merge_btn, self.status),
            self.preview_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### 2. Subset Data (Optional)"),
            pn.Row(self.slice_col, pn.Column("Type:", self.slice_type)),
            pn.Row(self.slice_val_cat, self.slice_val_cont_min, self.slice_val_cont_max),
            pn.Row(self.slice_add_btn, self.slice_clear_btn, self.slice_apply_btn, self.slice_download),
            self.slice_rules_md,
            pn.layout.Divider(),
            pn.pane.Markdown("### 3. Time Series Configuration"),
            pn.Row(self.ts_x, pn.Column("Time variable type:", self.ts_x_type), self.ts_x_order),
            pn.Row(self.ts_y, self.ts_y_mode, self.ts_patient, self.ts_color),
            pn.Row(self.svg_export, pn.Spacer(width=12), self.visualize_btn),
            self.plot_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### 4. Markov State Transitions"),
            pn.Row(self.markov_state, self.markov_cov, self.markov_timing),
            pn.Row(self.markov_btn, self.markov_status),
            pn.Row(self.markov_plot, self.markov_table),
            sizing_mode="stretch_width"
        )

    def _toggle_x_order(self, event):
        self.ts_x_order.visible = (self.ts_x_type.value == "Categorical")

    def _get_sliced_csv(self):
        if self.sliced_df is not None:
            sio = io.BytesIO()
            self.sliced_df.to_csv(sio)
            sio.seek(0)
            return sio
        return None

    def _update_slicing_ui(self, event=None):
        if self.master_df is None or not self.slice_col.value: return
        is_cat = (self.slice_type.value == "Categorical")
        self.slice_val_cat.visible = is_cat
        self.slice_val_cont_min.visible = not is_cat
        self.slice_val_cont_max.visible = not is_cat
        
        col = self.slice_col.value
        if is_cat:
            cats = [str(x) for x in self.master_df[col].dropna().unique()]
            self.slice_val_cat.options = cats
            self.slice_val_cat.value = []
        else:
            try:
                numeric_series = pd.to_numeric(self.master_df[col], errors='coerce').dropna()
                self.slice_val_cont_min.value = float(numeric_series.min()) if not numeric_series.empty else 0.0
                self.slice_val_cont_max.value = float(numeric_series.max()) if not numeric_series.empty else 100.0
            except:
                pass

    def _add_rule(self, event):
        if not self.slice_col.value: return
        
        is_cat = (self.slice_type.value == "Categorical")
        if is_cat:
            if not self.slice_val_cat.value: return
            self.rules.append({
                "col": self.slice_col.value,
                "type": "categorical",
                "categories": list(self.slice_val_cat.value)
            })
        else:
            self.rules.append({
                "col": self.slice_col.value,
                "type": "continuous",
                "min": self.slice_val_cont_min.value,
                "max": self.slice_val_cont_max.value
            })
        self._update_rules_display()

    def _clear_rules(self, event):
        self.rules = []
        self._update_rules_display()
        
    def _update_rules_display(self):
        if not self.rules:
            self.slice_rules_md.object = "**Active Rules:** None"
            return
            
        md = "**Active Rules (AND Logic):**\n"
        for i, r in enumerate(self.rules):
            if r["type"] == "categorical":
                md += f"- **{r['col']}** is in `{r['categories']}`\n"
            else:
                md += f"- **{r['col']}** between `{r['min']}` and `{r['max']}`\n"
        self.slice_rules_md.object = md

    def _apply_slicing(self, event):
        if self.master_df is None: return
        df = self.master_df.copy()
        
        for r in self.rules:
            if r["type"] == "categorical":
                df = df[df[r["col"]].astype(str).isin(r["categories"])]
            else:
                df[r["col"]] = pd.to_numeric(df[r["col"]], errors='coerce')
                df = df[(df[r["col"]] >= r["min"]) & (df[r["col"]] <= r["max"])]
                
        self.sliced_df = df.copy()
        self.slice_download.visible = True
        self.status.object = f"**Data Sliced:** {len(self.sliced_df)} samples remain."
        self._on_visualize(None)

    def _get_df(self) -> pd.DataFrame:
        if self.sliced_df is not None:
            return self.sliced_df
        return self.master_df

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
        self.slice_col.options = cols
        
        self.ts_x.options = cols
        self.ts_y.options = cols
        self.ts_patient.options = cols
        self.ts_color.options = ["None"] + cols
        self.markov_state.options = cols
        self.markov_cov.options = ["None"] + cols
        
        if cols:
            self.slice_col.value = cols[0]
            self.ts_x.value = cols[0]
            self.ts_y.value = cols[0]
            self.ts_patient.value = cols[0]
        self._update_slicing_ui()

    def _on_visualize(self, event):
        self.plot_pane.object = None
        current_df = self._get_df()
        
        if current_df is None or not self.ts_x.value or not self.ts_y.value or not self.ts_patient.value:
            return
            
        x_col = self.ts_x.value
        y_col = self.ts_y.value
        y_mode = self.ts_y_mode.value
        p_col = self.ts_patient.value
        color_col = self.ts_color.value
        
        df = current_df[[x_col, y_col, p_col]].copy()
        if color_col != "None":
            df[color_col] = current_df[color_col]
        df = df.dropna()
        if df.empty:
            self.status.object = "**Warning:** Plotting data is empty after removing NaNs."
            return

        # Handle Y-axis
        # Try to convert to numeric, if fails, treat as categorical
        y_is_cat = False
        try:
            df[y_col] = pd.to_numeric(df[y_col])
            y_range = None
        except ValueError:
            y_is_cat = True
            y_cats = list(df[y_col].astype(str).unique())
            y_range = y_cats

        # Handle X-axis
        x_is_cat = (self.ts_x_type.value == "Categorical")
        x_order = []
        if x_is_cat:
            user_order = [s.strip() for s in self.ts_x_order.value.split(",") if s.strip()]
            df[x_col] = df[x_col].astype(str)
            if user_order:
                # Filter to only ordered
                df = df[df[x_col].isin(user_order)]
                x_order = [s for s in user_order if s in df[x_col].unique()]
                # Map to numeric values internally so we can draw lines across categories easily in Bokeh
                # Bokeh can draw lines over categorical axes natively too.
            else:
                x_order = list(df[x_col].unique())
            x_range = x_order
        else:
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df = df.dropna(subset=[x_col])
            x_range = None
            
        if df.empty:
            self.status.object = "**Warning:** Plotting data is empty."
            return

        # Sort data for connecting lines
        if x_is_cat:
            order_map = {val: i for i, val in enumerate(x_order)}
            df["_x_sort"] = df[x_col].map(order_map)
            df = df.sort_values(by=[p_col, "_x_sort"])
            
            # Apply Jitter evenly to categories
            df["_jitter"] = 0.0
            for cat in x_order:
                mask = df[x_col] == cat
                n = mask.sum()
                if n > 0:
                    if n == 1:
                        df.loc[mask, "_jitter"] = 0.0
                    else:
                        df.loc[mask, "_jitter"] = np.linspace(-0.25, 0.25, n)
            
            df["x_plot"] = list(zip(df[x_col], df["_jitter"]))
        else:
            df = df.sort_values(by=[p_col, x_col])
            df["x_plot"] = df[x_col]

        if not y_is_cat and y_mode != "Actual Value":
            new_dfs = []
            for p in df[p_col].unique():
                pdf = df[df[p_col] == p].copy()
                if len(pdf) < 2: 
                    if y_mode == "Net Change from Initial":
                        pdf[y_col] = 0.0
                        new_dfs.append(pdf)
                    continue
                
                if y_mode == "Net Change from Initial":
                    base_val = pdf.iloc[0][y_col]
                    pdf[y_col] = pdf[y_col] - base_val
                    new_dfs.append(pdf)
                elif y_mode == "Change from Previous":
                    pdf[y_col] = pdf[y_col].diff()
                    pdf = pdf.dropna(subset=[y_col])
                    new_dfs.append(pdf)
                    
            if new_dfs:
                df = pd.concat(new_dfs, ignore_index=True)
            else:
                df = pd.DataFrame(columns=df.columns)
            y_range = None

        # Prepare Bokeh Plot
        kwargs = {
            "width": 800, "height": 600, "title": "Time Series Plot",
            "x_axis_label": x_col, "y_axis_label": f"{y_col} ({y_mode})",
            "tools": "pan,wheel_zoom,box_zoom,reset,save",
            "output_backend": "svg" if self.svg_export.value else "canvas"
        }
        if x_range is not None:
            kwargs["x_range"] = x_range
        if y_range is not None:
            kwargs["y_range"] = y_range

        p_fig = figure(**kwargs)

        from bokeh.models import CustomJS
        patients = df[p_col].unique()
        line_palette = list(Category10[10])

        # MultiLine for connecting dots (drawn first)
        xs = []
        ys = []
        pats = []
        colors = []
        for i, p in enumerate(patients):
            pdf = df[df[p_col] == p]
            if len(pdf) > 1:
                xs.append(pdf["x_plot"].values)
                ys.append(pdf[y_col].values)
                pats.append(p)
                colors.append(line_palette[i % 10])

        lines_src = None
        if xs:
            lines_src = ColumnDataSource({'xs': xs, 'ys': ys, 'patient': pats, 'color': colors})
            lines_renderer = p_fig.multi_line(xs='xs', ys='ys', color='color', line_width=1.5, alpha=0.6, 
                                              hover_line_width=3, hover_line_alpha=1.0, hover_line_color='color', source=lines_src)

        # Draw dots colored by variable
        circle_srcs = []
        circles = []
        if color_col == "None":
            src = ColumnDataSource(df)
            circle_srcs.append(src)
            r = p_fig.circle("x_plot", y_col, size=8, alpha=0.8, fill_color=line_palette[0], line_color="black", 
                             source=src, legend_label="Data",
                             selection_line_color="black", selection_line_width=3, selection_fill_alpha=1.0,
                             nonselection_alpha=0.8, nonselection_line_color="black")
            circles.append(r)
        else:
            c_vals = df[color_col].astype(str)
            unique = sorted(c_vals.unique())
            palette = list(Category10[10]) if len(unique) <= 10 else [Magma256[i] for i in np.linspace(0, 255, len(unique), dtype=int)]
            cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
            
            for c in unique:
                mask = (df[color_col] == c)
                src = ColumnDataSource(df[mask])
                circle_srcs.append(src)
                r = p_fig.circle("x_plot", y_col, size=8, alpha=0.8, fill_color=cmap[c], line_color="black", 
                                 source=src, legend_label=str(c)[:12],
                                 selection_line_color="black", selection_line_width=3, selection_fill_alpha=1.0,
                                 nonselection_alpha=0.8, nonselection_line_color="black")
                circles.append(r)

        if p_fig.legend:
            p_fig.legend.click_policy = "hide"
            p_fig.add_layout(p_fig.legend[0], "right")

        # Standard hover tool for individual points
        p_fig.add_tools(HoverTool(renderers=circles, tooltips=[
            ("Patient", f"@{p_col}"),
            ("Time", f"@{x_col}"),
            ("Value", f"@{y_col}")
        ]))

        if lines_src:
            callback = CustomJS(args=dict(circle_srcs=circle_srcs, lines_src=lines_src, p_col=p_col), code="""
                var line_indices = cb_data.index.line_indices;
                if (!line_indices && cb_data.index.1d) line_indices = cb_data.index.1d.indices;
                if (line_indices && line_indices.length > 0) {
                    var line_idx = line_indices[0];
                    var hovered_patient = lines_src.data['patient'][line_idx];
                    
                    for (var j = 0; j < circle_srcs.length; j++) {
                        var src = circle_srcs[j];
                        var circle_indices = [];
                        var all_patients = src.data[p_col];
                        if (all_patients) {
                            for (var i = 0; i < all_patients.length; i++) {
                                if (all_patients[i] == hovered_patient) {
                                    circle_indices.push(i);
                                }
                            }
                        }
                        src.selected.indices = circle_indices;
                        src.change.emit();
                    }
                } else {
                    for (var j = 0; j < circle_srcs.length; j++) {
                        circle_srcs[j].selected.indices = [];
                        circle_srcs[j].change.emit();
                    }
                }
            """)
            
            line_hover = HoverTool(renderers=[lines_renderer], tooltips=[("Patient", "@patient")], callback=callback, mode='mouse')
            p_fig.add_tools(line_hover)

        self.plot_pane.object = p_fig
        from common_plot import apply_export_prefix_to_pane
        prefix = getattr(self, "export_prefix", "CE_analysis")
        apply_export_prefix_to_pane(self.plot_pane, prefix)
        self.status.object = "Plot updated."


    def _on_markov_run(self, event):
        self.markov_status.object = "Running..."
        self.markov_plot.visible = False
        self.markov_table.visible = False
        
        current_df = self._get_df()
        if current_df is None or current_df.empty:
            self.markov_status.object = "**Error:** No data available."
            return
            
        p_col = self.ts_patient.value
        t_col = self.ts_x.value
        state_col = self.markov_state.value
        cov_col = self.markov_cov.value
        timing = self.markov_timing.value
        
        if not p_col or not t_col or not state_col:
            self.markov_status.object = "**Error:** Must select Patient, Time, and State variables."
            return
            
        df = current_df.copy()
        
        # Sort by patient and time
        # If time is categorical, we assume it's already sorted by user order in ts_x_order, 
        # or we just sort alphabetically. Let's use the UI order if provided.
        if self.ts_x_type.value == "Categorical":
            user_order = [s.strip() for s in self.ts_x_order.value.split(",") if s.strip()]
            if user_order:
                df = df[df[t_col].isin(user_order)]
                order_map = {val: i for i, val in enumerate(user_order)}
                df["_t_sort"] = df[t_col].astype(str).map(order_map)
                df = df.sort_values(by=[p_col, "_t_sort"])
            else:
                df = df.sort_values(by=[p_col, t_col])
        else:
            df[t_col] = pd.to_numeric(df[t_col], errors='coerce')
            df = df.dropna(subset=[t_col])
            df = df.sort_values(by=[p_col, t_col])
            
        df = df.dropna(subset=[state_col])
        df['Next_State'] = df.groupby(p_col)[state_col].shift(-1)
        
        if cov_col and cov_col != "None":
            df['Next_Cov'] = df.groupby(p_col)[cov_col].shift(-1)
            
        # Filter to valid transitions
        trans_df = df.dropna(subset=['Next_State']).copy()
        
        if trans_df.empty:
            self.markov_status.object = "**Error:** No valid step-to-step transitions found."
            return
            
        # 1. Baseline Transition Matrix Heatmap
        # Count transitions A -> B
        state_counts = trans_df.groupby([state_col, 'Next_State']).size().reset_index(name='Count')
        total_from_A = trans_df.groupby(state_col).size().reset_index(name='Total')
        state_counts = pd.merge(state_counts, total_from_A, on=state_col)
        state_counts['Prob'] = state_counts['Count'] / state_counts['Total']
        
        states = sorted(list(set(trans_df[state_col].unique()) | set(trans_df['Next_State'].unique())))
        state_counts[state_col] = state_counts[state_col].astype(str)
        state_counts['Next_State'] = state_counts['Next_State'].astype(str)
        states_str = [str(s) for s in states]
        
        p = figure(width=400, height=400, title="Transition Probabilities",
                   x_range=states_str, y_range=states_str[::-1],
                   toolbar_location="right", tools="hover,save")
                   
        p.xaxis.axis_label = "Next State"
        p.yaxis.axis_label = "Current State"
        
        cmap = LinearColorMapper(palette="Blues8", low=0, high=1.0)
        src = ColumnDataSource(state_counts)
        p.rect(x="Next_State", y=state_col, width=1, height=1, source=src,
               line_color="white", fill_color={"field": "Prob", "transform": cmap})
               
        p.text(x="Next_State", y=state_col, text="Prob", text_color="black",
               text_align="center", text_baseline="middle", source=src)
               
        hover = p.select_one(HoverTool)
        hover.tooltips = [
            ("Transition", f"@{state_col} -> @Next_State"),
            ("Probability", "@Prob{0.00}"),
            ("N", "@Count / @Total")
        ]
        
        self.markov_plot.object = p
        self.markov_plot.visible = True
        
        # 2. Covariate Statistical Analysis
        if cov_col == "None":
            self.markov_status.object = "Baseline transitions calculated (No covariate selected)."
            return
            
        # Construct X
        if timing == "Time A (Baseline)":
            trans_df['X'] = trans_df[cov_col]
        elif timing == "Time B (Outcome)":
            trans_df['X'] = trans_df['Next_Cov']
        else: # Change
            trans_df['X'] = pd.to_numeric(trans_df['Next_Cov'], errors='coerce') - pd.to_numeric(trans_df[cov_col], errors='coerce')
            
        trans_df = trans_df.dropna(subset=['X'])
        if trans_df.empty:
            self.markov_status.object = "**Error:** Covariate data is empty for transitions."
            return
            
        # Is X continuous or categorical?
        try:
            trans_df['X'] = pd.to_numeric(trans_df['X'])
            x_is_cont = True
        except:
            x_is_cont = False
            
        results = []
        # For each starting state A, test if X predicts going to B
        for stateA in states:
            df_A = trans_df[trans_df[state_col] == stateA]
            if len(df_A) < 5: continue # Need minimum samples
            
            # What are the next states from A?
            next_states = df_A['Next_State'].unique()
            for stateB in next_states:
                df_A_to_B = df_A[df_A['Next_State'] == stateB]
                df_A_to_Other = df_A[df_A['Next_State'] != stateB]
                
                n_B = len(df_A_to_B)
                n_Other = len(df_A_to_Other)
                
                if n_B < 3 or n_Other < 3:
                    continue # Not enough data to run logit or tests reliably
                    
                y = (df_A['Next_State'] == stateB).astype(int)
                X = df_A['X']
                
                if x_is_cont:
                    # Logistic Regression
                    try:
                        import statsmodels.api as sm
                        X_sm = sm.add_constant(X)
                        logit = sm.Logit(y, X_sm)
                        res = logit.fit(disp=0)
                        pval = res.pvalues['X']
                        coef = res.params['X']
                        or_val = np.exp(coef)
                        results.append({
                            'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                            'Odds Ratio': or_val, 'Coef': coef, 'p-value': pval, 'Test': 'Logit'
                        })
                    except:
                        # Fallback to Mann-Whitney
                        from scipy.stats import mannwhitneyu
                        stat, pval = mannwhitneyu(X[y==1], X[y==0], alternative='two-sided')
                        results.append({
                            'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                            'Odds Ratio': np.nan, 'Coef': np.nan, 'p-value': pval, 'Test': 'MWU'
                        })
                else:
                    # Categorical X -> Chi-square or Fisher
                    from scipy.stats import fisher_exact, chi2_contingency
                    contingency = pd.crosstab(y, X)
                    if contingency.shape == (2,2):
                        oddsr, pval = fisher_exact(contingency)
                        results.append({
                            'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                            'Odds Ratio': oddsr, 'Coef': np.nan, 'p-value': pval, 'Test': 'Fisher'
                        })
                    else:
                        try:
                            chi2, pval, dof, ex = chi2_contingency(contingency)
                            results.append({
                                'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                                'Odds Ratio': np.nan, 'Coef': np.nan, 'p-value': pval, 'Test': 'Chi2'
                            })
                        except:
                            pass
                            
        if results:
            res_df = pd.DataFrame(results)
            # FDR correction
            from statsmodels.stats.multitest import multipletests
            pvals = res_df['p-value'].values
            valid_idx = ~np.isnan(pvals)
            if valid_idx.any():
                _, pvals_fdr, _, _ = multipletests(pvals[valid_idx], method='fdr_bh')
                res_df['FDR_p'] = np.nan
                res_df.loc[valid_idx, 'FDR_p'] = pvals_fdr
                
            res_df = res_df.sort_values('p-value')
            # Format numeric columns for display
            for col in ['Odds Ratio', 'Coef', 'p-value', 'FDR_p']:
                if col in res_df.columns:
                    res_df[col] = res_df[col].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else "")
                    
            self.markov_table.object = res_df
            self.markov_table.visible = True
            self.markov_status.object = "Analysis complete."
        else:
            self.markov_status.object = "Analysis complete, but no valid tests could be run (insufficient N)."

def build_timeseries_section():
    ctrl = TimeSeriesController()
    return ctrl.section, ctrl
