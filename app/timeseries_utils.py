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
        self.markov_patient = pn.widgets.Select(name="Patient ID", options=[], width=150)
        self.markov_time = pn.widgets.Select(name="Time Variable", options=[], width=150)
        self.markov_time_type = pn.widgets.RadioButtonGroup(name="Time Type", options=["Continuous", "Categorical"], value="Continuous")
        self.markov_time_order = pn.widgets.TextInput(name="Categorical Order (comma separated)", placeholder="e.g. Day1, Day2, Day3", visible=False, width=200)
        
        self.markov_time_type.param.watch(self._toggle_markov_time_order, 'value')
        
        self.markov_svg_export = pn.widgets.Checkbox(name="SVG Export Mode (slower)", value=False)
        self.markov_btn = pn.widgets.Button(name="Run Markov Analysis", button_type="primary")
        self.markov_btn.on_click(self._on_markov_run)
        self.markov_status = pn.pane.Markdown("")
        self.markov_results_container = pn.Column(sizing_mode="stretch_width")
        self.markov_preview = pn.pane.DataFrame(pd.DataFrame(), max_height=250, sizing_mode="stretch_width", visible=False)
        
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
            pn.Row(self.markov_patient, self.markov_time, pn.Column("Time variable type:", self.markov_time_type), self.markov_time_order),
            pn.Row(self.markov_state, self.markov_cov, self.markov_timing),
            pn.Row(self.markov_svg_export, pn.Spacer(width=12), self.markov_btn, self.markov_status),
            self.markov_results_container,
            sizing_mode="stretch_width"
        )

    def _toggle_x_order(self, event):
        self.ts_x_order.visible = (self.ts_x_type.value == "Categorical")
        
    def _toggle_markov_time_order(self, event):
        self.markov_time_order.visible = (self.markov_time_type.value == "Categorical")

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
        self.markov_time.options = cols
        self.markov_patient.options = cols
        if cols:
            self.markov_time.value = cols[0]
            self.markov_patient.value = cols[0]
        
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
        self.markov_results_container.clear()
        
        current_df = self._get_df()
        if current_df is None or current_df.empty:
            self.markov_status.object = "**Error:** No data available."
            return
            
        p_col = self.markov_patient.value
        t_col = self.markov_time.value
        state_col = self.markov_state.value
        cov_col = self.markov_cov.value
        timing = self.markov_timing.value
        
        if not p_col or not t_col or not state_col:
            self.markov_status.object = "**Error:** Must select Patient, Time, and State variables."
            return
            
        df = current_df.copy()
        debug_log = [f"Initial rows: {len(df)}"]
        
        # Sort by patient and time
        if self.markov_time_type.value == "Categorical":
            user_order = [s.strip() for s in self.markov_time_order.value.split(",") if s.strip()]
            if user_order:
                # Robust matching between strings and floats
                t_str = df[t_col].astype(str).str.replace(r'\.0$', '', regex=True)
                order_clean = [s.replace('.0', '') for s in user_order]
                df = df[t_str.isin(order_clean)]
                debug_log.append(f"After filtering to Categorical Order: {len(df)} rows")
                if df.empty:
                    self.markov_status.object = "<br>".join(debug_log) + f"<br>**Error:** Time order '{self.markov_time_order.value}' did not match any values in time column '{t_col}'."
                    return
                order_map = {val: i for i, val in enumerate(order_clean)}
                df["_t_sort"] = t_str.map(order_map)
                df = df.sort_values(by=[p_col, "_t_sort"])
            else:
                df = df.sort_values(by=[p_col, t_col])
        else:
            orig_len = len(df)
            df[t_col] = pd.to_numeric(df[t_col], errors='coerce')
            df = df.dropna(subset=[t_col])
            debug_log.append(f"After dropping NA in time '{t_col}': {len(df)} rows")
            if df.empty and orig_len > 0:
                self.markov_status.object = f"**Error:** Time variable '{t_col}' was parsed as Continuous but all values became missing. Try switching to Categorical."
                return
            df = df.sort_values(by=[p_col, t_col])
            
        orig_len = len(df)
        df = df.dropna(subset=[state_col])
        debug_log.append(f"After dropping NA in state '{state_col}': {len(df)} rows")
        if df.empty and orig_len > 0:
            self.markov_status.object = f"**Error:** State variable '{state_col}' is completely empty."
            return
            
        # Exclude patients with duplicate time points
        patient_time_counts = df.groupby([p_col, t_col]).size()
        patients_with_dupes = patient_time_counts[patient_time_counts > 1].index.get_level_values(0).unique()
        if len(patients_with_dupes) > 0:
            df = df[~df[p_col].isin(patients_with_dupes)]
            debug_log.append(f"Excluded {len(patients_with_dupes)} patients due to duplicate time points. Remaining rows: {len(df)}")
            if df.empty:
                self.markov_status.object = "<br>".join(debug_log) + "<br>**Error:** All patients were excluded because they had multiple readings at the exact same time point."
                return
            self.markov_status.object = "<br>".join(debug_log) + "<br>"
        else:
            self.markov_status.object = "<br>".join(debug_log) + "<br>"
            
        # Exclude patients with < 2 timepoints
        patient_counts = df[p_col].value_counts()
        
        # Let's add debug info about the patient counts
        top_patients = patient_counts.head(5).to_dict()
        msg_patients = f"<br>Patient ID column selected: '{p_col}'. <br>Top patients and their row counts: {top_patients}."
        
        valid_patients = patient_counts[patient_counts >= 2].index
        df = df[df[p_col].isin(valid_patients)]
        
        # Remove the preview table
        self.markov_preview.visible = False
        
        if df.empty:
            msg = f"<br>**Error:** No patients left with >= 2 valid timepoints." + msg_patients
            self.markov_status.object += msg
            return
            
        self.markov_status.object += msg_patients
            
        df['Next_State'] = df.groupby(p_col)[state_col].shift(-1)
        
        if cov_col and cov_col != "None":
            df['Next_Cov'] = df.groupby(p_col)[cov_col].shift(-1)
            
        # Filter to valid transitions
        trans_df = df.dropna(subset=['Next_State']).copy()
        
        if trans_df.empty:
            if len(df) > 0:
                self.markov_status.object += f"<br>**Error:** Found {len(df)} rows, but NO patient had 2 or more consecutive timepoints. Check that you selected the correct Patient ID (currently '{p_col}')."
            else:
                self.markov_status.object += "<br>**Error:** No valid step-to-step transitions found."
            return
            
        # 1. Baseline Transition Matrix Heatmap
        # Count transitions A -> B
        state_counts = trans_df.groupby([state_col, 'Next_State']).size().reset_index(name='Count')
        total_from_A = trans_df.groupby(state_col).size().reset_index(name='Total')
        state_counts = pd.merge(state_counts, total_from_A, on=state_col)
        state_counts['Prob'] = state_counts['Count'] / state_counts['Total']
        
        states = sorted(list(set(trans_df[state_col].unique()) | set(trans_df['Next_State'].unique())))
        states_str = [str(s) for s in states]
        
        state_counts[state_col] = state_counts[state_col].astype(str)
        state_counts['Next_State'] = state_counts['Next_State'].astype(str)
        
        import itertools
        full_grid = pd.DataFrame(list(itertools.product(states_str, states_str)), columns=[state_col, 'Next_State'])
        state_counts = pd.merge(full_grid, state_counts, on=[state_col, 'Next_State'], how='left')
        
        state_counts['Prob'] = state_counts['Prob'].fillna(0.0)
        state_counts['Count'] = state_counts['Count'].fillna(0)
        state_counts['Total'] = state_counts['Total'].fillna(0)
        state_counts['Prob_Str'] = state_counts['Prob'].apply(lambda x: f"{x:.2f}")
        
        from bokeh.palettes import Blues8
        from bokeh.models import ColorBar
        
        # User requested: X-axis = "initial state", Y-axis = "next state"
        p = figure(width=450, height=400, title=f"Transition Probabilities ({state_col})",
                   x_range=states_str, y_range=states_str[::-1],
                   toolbar_location="right", tools="hover,save")
                   
        import math
        p.xaxis.axis_label = "Initial State"
        p.yaxis.axis_label = "Next State"
        p.xaxis.major_label_orientation = math.pi/4
        
        if self.markov_svg_export.value:
            p.output_backend = "svg"
        
        cmap = LinearColorMapper(palette=Blues8[::-1], low=0, high=1.0)
        src = ColumnDataSource(state_counts)
        p.rect(x=state_col, y="Next_State", width=1, height=1, source=src,
               line_color="black", fill_color={"field": "Prob", "transform": cmap})
               
        p.text(x=state_col, y="Next_State", text="Prob_Str", text_color="gray",
               text_font_size="9pt", text_align="center", text_baseline="middle", source=src)
               
        hover = p.select_one(HoverTool)
        hover.tooltips = [
            ("Transition", f"@{state_col} -> @Next_State"),
            ("Probability", "@Prob{0.00}"),
            ("N", "@Count / @Total")
        ]
        
        cbar = ColorBar(color_mapper=cmap, width=8, location=(0,0))
        p.add_layout(cbar, 'right')
        
        # 1b. Transition Network Graph
        import numpy as np
        from bokeh.models import Arrow, NormalHead
        
        N = len(states_str)
        p2 = figure(width=650, height=650, title=f"Transition Network ({state_col})",
                    x_range=(-1.8, 1.8), y_range=(-1.8, 1.8),
                    toolbar_location="right", tools="save")
        p2.axis.visible = False
        p2.grid.visible = False
        
        nodes = {}
        for i, s in enumerate(states_str):
            angle = 2 * np.pi * i / N - np.pi / 2
            nodes[s] = (np.cos(angle), np.sin(angle), angle)
            
        # Draw edges first so they are behind nodes
        for idx, row in state_counts.iterrows():
            u = row[state_col]
            v = row['Next_State']
            prob = row['Prob']
            if prob <= 0: continue
            
            thick = max(1, int(prob * 10))
            if u == v:
                angle = nodes[u][2]
                cx = nodes[u][0] + 0.15 * np.cos(angle)
                cy = nodes[u][1] + 0.15 * np.sin(angle)
                p2.circle([cx], [cy], radius=0.2, fill_color=None, line_color="black", line_width=thick)
                tx = nodes[u][0] + 0.45 * np.cos(angle)
                ty = nodes[u][1] + 0.45 * np.sin(angle)
                p2.text([tx], [ty], text=[f"{prob:.2f}"], text_align="center", text_baseline="middle", text_font_size="8pt", text_color="gray", level="overlay")
            else:
                xA, yA, _ = nodes[u]
                xB, yB, _ = nodes[v]
                dx = xB - xA
                dy = yB - yA
                length = np.hypot(dx, dy)
                ux, uy = dx/length, dy/length
                nx, ny = uy, -ux
                
                offset = 0.1
                gap = 0.2
                sx = xA + nx*offset + ux*gap
                sy = yA + ny*offset + uy*gap
                ex = xB + nx*offset - ux*gap
                ey = yB + ny*offset - uy*gap
                
                arrow = Arrow(end=NormalHead(size=10, fill_color="black", line_color="black"),
                              x_start=sx, y_start=sy, x_end=ex, y_end=ey,
                              line_color="black", line_width=thick)
                p2.add_layout(arrow)
                
                tx = (sx + ex)/2 + nx*0.1
                ty = (sy + ey)/2 + ny*0.1
                p2.text([tx], [ty], text=[f"{prob:.2f}"], text_align="center", text_baseline="middle", text_font_size="8pt", text_color="gray", level="overlay")
                
        if self.markov_svg_export.value:
            p2.output_backend = "svg"
            
        # Draw nodes on top
        node_x = [nodes[s][0] for s in states_str]
        node_y = [nodes[s][1] for s in states_str]
        states_capped = [s[:12] for s in states_str]
        p2.circle(node_x, node_y, size=80, color="lightblue", line_color="black")
        p2.text(x=node_x, y=node_y, text=states_capped, text_align="center", text_baseline="middle", text_font_style="bold", text_font_size="10pt")
                
        plot_col = pn.Column(pn.pane.Bokeh(p), pn.pane.Bokeh(p2))
        self.markov_results_container.append(plot_col)
        
        # 2. Covariate Statistical Analysis
        if cov_col == "None":
            msg = "Baseline transitions calculated (No covariate selected)."
            if "Excluded" in self.markov_status.object:
                self.markov_status.object += msg
            else:
                self.markov_status.object = msg
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
            next_states = df_A['Next_State'].unique()
            
            if len(df_A) < 5:
                for stateB in next_states:
                    n_B = len(df_A[df_A['Next_State'] == stateB])
                    results.append({
                        'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                        'Odds Ratio': float('nan'), 'CI_Lower': float('nan'), 'CI_Upper': float('nan'), 'Coef': float('nan'), 'p-value': float('nan'), 'Test': 'Insufficient N (Total < 5)'
                    })
                continue
                
            for stateB in next_states:
                df_A_to_B = df_A[df_A['Next_State'] == stateB]
                df_A_to_Other = df_A[df_A['Next_State'] != stateB]
                
                n_B = len(df_A_to_B)
                n_Other = len(df_A_to_Other)
                
                if n_B < 3 or n_Other < 3:
                    results.append({
                        'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                        'Odds Ratio': float('nan'), 'CI_Lower': float('nan'), 'CI_Upper': float('nan'), 'Coef': float('nan'), 'p-value': float('nan'), 'Test': 'Insufficient N (<3)'
                    })
                    continue
                    
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
                        ci = res.conf_int().loc['X']
                        results.append({
                            'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                            'Odds Ratio': or_val, 'CI_Lower': np.exp(ci[0]), 'CI_Upper': np.exp(ci[1]), 'Coef': coef, 'p-value': pval, 'Test': 'Logit'
                        })
                    except:
                        # Fallback to Mann-Whitney
                        from scipy.stats import mannwhitneyu
                        stat, pval = mannwhitneyu(X[y==1], X[y==0], alternative='two-sided')
                        results.append({
                            'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                            'Odds Ratio': np.nan, 'CI_Lower': np.nan, 'CI_Upper': np.nan, 'Coef': np.nan, 'p-value': pval, 'Test': 'MWU'
                        })
                else:
                    # Categorical X -> Chi-square or Fisher
                    from scipy.stats import fisher_exact, chi2_contingency
                    contingency = pd.crosstab(y, X)
                    if contingency.shape == (2,2):
                        oddsr, pval = fisher_exact(contingency)
                        try:
                            a, b = contingency.iloc[1,1], contingency.iloc[1,0]
                            c, d = contingency.iloc[0,1], contingency.iloc[0,0]
                            se = np.sqrt(1/(a+0.5) + 1/(b+0.5) + 1/(c+0.5) + 1/(d+0.5))
                            log_or = np.log(oddsr) if oddsr > 0 else np.nan
                            cil = np.exp(log_or - 1.96*se)
                            ciu = np.exp(log_or + 1.96*se)
                        except:
                            cil, ciu = np.nan, np.nan
                            
                        results.append({
                            'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                            'Odds Ratio': oddsr, 'CI_Lower': cil, 'CI_Upper': ciu, 'Coef': np.nan, 'p-value': pval, 'Test': 'Fisher'
                        })
                    else:
                        try:
                            chi2, pval, dof, ex = chi2_contingency(contingency)
                            results.append({
                                'From': stateA, 'To': stateB, 'N (Event/Total)': f"{n_B}/{len(df_A)}",
                                'Odds Ratio': np.nan, 'CI_Lower': np.nan, 'CI_Upper': np.nan, 'Coef': np.nan, 'p-value': pval, 'Test': 'Chi2'
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
                
            if 'FDR_p' in res_df.columns:
                res_df = res_df.sort_values(['FDR_p', 'p-value'])
            else:
                res_df = res_df.sort_values('p-value')
                
            # Format numeric columns for display
            for col in ['Odds Ratio', 'Coef', 'p-value', 'FDR_p']:
                if col in res_df.columns:
                    res_df[col] = res_df[col].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else "")
                    
            interpretation = """
**Statistical Tests Performed:**
- **Continuous Covariates**: Logistic Regression is used to test if the numeric value of the covariate predicts the binary outcome of transitioning to the "To" state (vs all other available states). If regression fails, it falls back to a non-parametric Mann-Whitney U test.
- **Categorical Covariates**: Fisher's Exact Test (for 2x2) or Chi-Square Test is used.
- **FDR**: p-values are corrected for multiple comparisons using the Benjamini-Hochberg method.

**How to interpret these results:**
- **Odds Ratio > 1 (Positive Coef)**: Higher covariate values increase the likelihood of transitioning to the "To" state instead of any other state.
- **Odds Ratio < 1 (Negative Coef)**: Higher covariate values decrease the likelihood.
- **FDR_p < 0.05**: The association is statistically significant after correcting for multiple comparisons.
"""
            # Build plotting df (keep raw values)
            plot_df = res_df.copy()
            for col in ['Odds Ratio', 'Coef', 'p-value', 'FDR_p']:
                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                
            # Replace infinity with NaN to prevent Bokeh JS renderer crashes
            import numpy as np
            plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
            
            try:
                # Covariate Heatmap
                import itertools
                from bokeh.models import Span
                
                full_grid = pd.DataFrame(list(itertools.product(states_str, states_str)), columns=['From', 'To'])
                plot_df['From'] = plot_df['From'].astype(str)
                plot_df['To'] = plot_df['To'].astype(str)
                heat_df = pd.merge(full_grid, plot_df, on=['From', 'To'], how='left')
                
                heat_df['Score'] = 0.0
                mask = heat_df['FDR_p'].notnull() & heat_df['Odds Ratio'].notnull()
                
                fdr_safe = np.where(heat_df['FDR_p'] == 0, 1e-10, heat_df['FDR_p'])
                heat_df.loc[mask, 'Score'] = -np.log10(fdr_safe[mask]) * np.sign(np.log(heat_df.loc[mask, 'Odds Ratio'].astype(float) + 1e-10))
                
                def make_label(row):
                    if pd.isna(row['Odds Ratio']):
                        return ""
                    fdr_str = f"FDR: {row['FDR_p']:.3f}" if not pd.isna(row['FDR_p']) else ""
                    or_str = f"OR: {row['Odds Ratio']:.2f}"
                    return f"{or_str}\n{fdr_str}"
                    
                heat_df['Label'] = heat_df.apply(make_label, axis=1)
                
                p3 = figure(width=400, height=400, title=f"Covariate Effect Heatmap ({cov_col})",
                           x_range=states_str, y_range=states_str[::-1],
                           toolbar_location="right", tools="hover,save")
                
                p3.xaxis.axis_label = "Initial State"
                p3.yaxis.axis_label = "Next State"
                p3.xaxis.major_label_orientation = math.pi/4
                
                from bokeh.palettes import RdBu10
                cmap3 = LinearColorMapper(palette=RdBu10[::-1], low=-3, high=3)
                src3 = ColumnDataSource(heat_df)
                p3.rect(x="From", y="To", width=1, height=1, source=src3,
                       line_color="white", fill_color={"field": "Score", "transform": cmap3})
                       
                p3.text(x="From", y="To", text="Label", text_color="gray",
                       text_align="center", text_baseline="middle", text_font_size="9pt", source=src3)
                
                cbar3 = ColorBar(color_mapper=cmap3, width=8, location=(0,0), title="-log(FDR) * dir")
                p3.add_layout(cbar3, 'right')
                
                # Forest Plot
                forest_df = plot_df.dropna(subset=['Odds Ratio', 'CI_Lower']).copy()
                if not forest_df.empty:
                    forest_df['Transition'] = forest_df['From'] + " -> " + forest_df['To']
                    if 'FDR_p' in forest_df.columns:
                        forest_df = forest_df.sort_values('FDR_p', ascending=False)
                    
                    trans_list = forest_df['Transition'].tolist()
                    
                    p4 = figure(width=500, height=max(200, len(forest_df)*40 + 50), title="Forest Plot (Odds Ratios)",
                                y_range=trans_list, x_axis_type="log", toolbar_location="right", tools="hover,save")
                    p4.xaxis.axis_label = "Odds Ratio (log scale)"
                    
                    # Safe min/max for x-axis to prevent bokeh crashes
                    min_x = max(0.01, forest_df['CI_Lower'].min() * 0.5)
                    max_x = min(100, forest_df['CI_Upper'].max() * 1.5)
                    if min_x >= max_x:
                        min_x, max_x = 0.01, 100
                    p4.x_range.start = min_x
                    p4.x_range.end = max_x
                    
                    src4 = ColumnDataSource(forest_df)
                    p4.segment(x0="CI_Lower", x1="CI_Upper", y0="Transition", y1="Transition", source=src4, line_width=2, line_color="black")
                    p4.circle(x="Odds Ratio", y="Transition", source=src4, size=8, color="blue")
                    
                    vline = Span(location=1, dimension='height', line_dash='dashed', line_color='red', line_width=2)
                    p4.add_layout(vline)
                    
                    if self.markov_svg_export.value:
                        p3.output_backend = "svg"
                        p4.output_backend = "svg"
                        
                    viz_row = pn.Row(pn.pane.Bokeh(p3), pn.pane.Bokeh(p4))
                else:
                    if self.markov_svg_export.value:
                        p3.output_backend = "svg"
                    viz_row = pn.Row(pn.pane.Bokeh(p3))
                    
                self.markov_status.object += "<br>✅ Covariate Heatmap and Forest Plot successfully rendered."
            except Exception as e:
                viz_row = pn.pane.Markdown(f"**Error plotting covariate visuals:** {str(e)}")
                self.markov_status.object += f"<br>❌ Error plotting visuals: {str(e)}"
                
            table_col = pn.Column(
                pn.pane.Markdown(interpretation, sizing_mode="stretch_width"),
                pn.widgets.Tabulator(res_df, sizing_mode="stretch_width", height=300, pagination=None, show_index=False),
                pn.pane.Markdown("### Covariate Visualizations", sizing_mode="stretch_width"),
                viz_row,
                sizing_mode="stretch_width"
            )
            self.markov_results_container.append(table_col)
            self.markov_status.object += "<br>Analysis complete."
        else:
            self.markov_status.object += "<br>Analysis complete, but no valid tests could be run (insufficient N)."

def build_timeseries_section():
    ctrl = TimeSeriesController()
    return ctrl.section, ctrl
