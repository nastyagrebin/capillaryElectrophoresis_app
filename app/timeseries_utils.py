import io
import numpy as np
import pandas as pd
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar
from bokeh.palettes import Category10, Magma256

class TimeSeriesController:
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
        
        # Watchers
        self.ts_x_type.param.watch(self._toggle_x_order, 'value')
        
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
            pn.pane.Markdown("### 2. Time Series Configuration"),
            pn.Row(self.ts_x, pn.Column("Time variable type:", self.ts_x_type), self.ts_x_order),
            pn.Row(self.ts_y, self.ts_y_mode, self.ts_patient, self.ts_color),
            pn.Row(self.svg_export, pn.Spacer(width=12), self.visualize_btn),
            self.plot_pane,
            sizing_mode="stretch_width"
        )

    def _toggle_x_order(self, event):
        self.ts_x_order.visible = (self.ts_x_type.value == "Categorical")

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
        self.ts_x.options = cols
        self.ts_y.options = cols
        self.ts_patient.options = cols
        self.ts_color.options = ["None"] + cols
        
        if cols:
            self.ts_x.value = cols[0]
            self.ts_y.value = cols[0]
            self.ts_patient.value = cols[0]

    def _on_visualize(self, event):
        self.plot_pane.object = None
        if self.master_df is None or not self.ts_x.value or not self.ts_y.value or not self.ts_patient.value:
            return
            
        x_col = self.ts_x.value
        y_col = self.ts_y.value
        y_mode = self.ts_y_mode.value
        p_col = self.ts_patient.value
        color_col = self.ts_color.value
        
        df = self.master_df[[x_col, y_col, p_col]].copy()
        if color_col != "None":
            df[color_col] = self.master_df[color_col]
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
        else:
            df = df.sort_values(by=[p_col, x_col])

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

        # Connect sequential dots for same patient
        patients = df[p_col].unique()
        line_palette = list(Category10[10])
        for i, p in enumerate(patients):
            pdf = df[df[p_col] == p]
            if len(pdf) > 1:
                p_color = line_palette[i % 10]
                p_fig.line(pdf[x_col].values, pdf[y_col].values, color=p_color, line_width=1.5, alpha=0.6)

        # Draw dots colored by variable
        if color_col == "None":
            src = ColumnDataSource(df)
            p_fig.circle(x_col, y_col, size=8, alpha=0.8, source=src)
        else:
            c_vals = df[color_col].astype(str)
            unique = sorted(c_vals.unique())
            palette = list(Category10[10]) if len(unique) <= 10 else [Magma256[i] for i in np.linspace(0, 255, len(unique), dtype=int)]
            cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
            
            for c in unique:
                mask = (df[color_col] == c)
                src = ColumnDataSource(df[mask])
                p_fig.circle(x_col, y_col, size=8, alpha=0.8, fill_color=cmap[c], line_color="black", source=src, legend_label=str(c)[:12])

            if p_fig.legend:
                p_fig.legend.click_policy = "hide"
                p_fig.add_layout(p_fig.legend[0], "below")

        # Hover Tool
        p_fig.add_tools(HoverTool(tooltips=[
            ("Patient", f"@{p_col}"),
            ("Time", f"@{x_col}"),
            ("Value", f"@{y_col}")
        ]))

        self.plot_pane.object = p_fig
        self.status.object = "Plot updated."

def build_timeseries_section():
    ctrl = TimeSeriesController()
    return ctrl.section, ctrl
