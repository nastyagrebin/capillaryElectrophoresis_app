from __future__ import annotations
from typing import Dict, Optional, List
import io
from contextlib import contextmanager

import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting
import bokeh.models
import bokeh.io as _bio

OK = "OK:"; WARN = "Warning:"
def ok(m): return f"{OK} {m}"
def warn(m): return f"{WARN} {m}"

from common_plot import TitledPlotPane

pn.extension('tabulator')

@contextmanager
def no_bokeh_show():
    old_show = _bio.show
    try:
        _bio.show = lambda *a, **k: None
        yield
    finally:
        _bio.show = old_show

def _read_metadata_table_from_bytes(name: str, data: bytes) -> pd.DataFrame:
    name = (name or "").lower()
    bio = io.BytesIO(data or b"")
    if name.endswith(".csv"):
        df = pd.read_csv(bio)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(bio)
    else:
        raise ValueError("Unsupported file format (use .csv or .xlsx).")
    if df.shape[1] < 2:
        raise ValueError("Expected at least two columns.")
    df = df.rename(columns={df.columns[0]: "sample"})
    df["sample"] = df["sample"].astype(str)
    return df

class VizController:
    def __init__(self):
        self.H_df: Optional[pd.DataFrame] = None
        self.samples: List[str] = []
        self.metadata_df: Optional[pd.DataFrame] = None  # sample, metadata
        self.enabled: bool = False

        self.diversity_df = None  # NEW: rows=samples, cols=metrics
        self.last_pca = None
        
        self.rules = []
        self.sliced_df = None
        
        # ---- Slicing UI ----
        self.slice_col = pn.widgets.Select(name="Variable to Slice By", options=[])
        self.slice_type = pn.widgets.RadioButtonGroup(name="Type", options=["Categorical", "Continuous"], value="Categorical")
        self.slice_val_cat = pn.widgets.MultiChoice(name="Categories to Include", options=[], visible=True)
        self.slice_val_cont_min = pn.widgets.FloatInput(name="Min Value", value=0.0, visible=False)
        self.slice_val_cont_max = pn.widgets.FloatInput(name="Max Value", value=100.0, visible=False)
        self.slice_add_btn = pn.widgets.Button(name="Add Rule", button_type="primary")
        self.slice_clear_btn = pn.widgets.Button(name="Clear Rules", button_type="danger")
        self.slice_apply_btn = pn.widgets.Button(name="Apply Filters", button_type="success")
        self.slice_rules_md = pn.pane.Markdown("**Active Rules:** None", sizing_mode="stretch_width")


        # ---- Standalone NMF Upload UI ----
        self.nmf_file = pn.widgets.FileInput(accept=".csv", multiple=False)
        self.nmf_file.param.watch(self._on_load_nmf_csv, "value")

        # ---- Standalone Diversity Upload UI ----
        self.div_file = pn.widgets.FileInput(accept=".csv", multiple=False)
        self.div_file.param.watch(self._on_load_div_csv, "value")

        # ---- Metadata entry/upload UI ----
        self.meta_help = pn.pane.Markdown(
            "Upload a CSV/XLSX with two columns **sample, metadata**, or edit manually below.",
            sizing_mode="stretch_width"
        )
        self.meta_file = pn.widgets.FileInput(accept=".csv,.xlsx,.xls", multiple=False)
        self.meta_load_btn = pn.widgets.Button(name="Load metadata from file", button_type="primary")
        
        self.meta_col_select = pn.widgets.Select(name="Metadata Column", options=[], width=220)

        _columns = [
            {"field": "sample", "title": "sample", "editable": False},
            {"field": "metadata", "title": "metadata", "editor": "input"},
        ]
        self.meta_table = pn.widgets.Tabulator(
            pd.DataFrame({"sample": [], "metadata": []}),
            editable=True, selectable=False, show_index=False,
            height=300, min_width=600, layout="fit_data_stretch", columns=_columns,
        )

        # Coloring controls
        self.color_source = pn.widgets.Select(
            name="Color by", options=["manual metadata", "diversity metric"], value="manual metadata", width=220
        )
        self.meta_mode = pn.widgets.RadioButtonGroup(
            name="Value type", options=["categorical", "continuous"], value="categorical"
        )
        self.div_metric_select = pn.widgets.Select(name="Metric", options=[], value=None, width=220)  # populated later
        
        self.svg_export = pn.widgets.Checkbox(name="SVG Export Mode (slower)", value=False)

        self.html_plots = []
        self.on_send_to_html = None

        # ---- PCA controls/plot ----
        self.pca_caption = pn.widgets.TextInput(placeholder="Optional caption...", value="", width=250)
        self.pca_send_btn = pn.widgets.Button(name="Send to HTML", button_type="primary", disabled=True, width=120)
        self.pca_send_status = pn.pane.Markdown("", margin=(5, 0, 0, 10))
        self.last_pca_fig = None
        self.pca_send_btn.on_click(self._on_send_pca_to_html)

        self.pls_caption = pn.widgets.TextInput(placeholder="Optional caption...", value="", width=250)
        self.pls_send_btn = pn.widgets.Button(name="Send to HTML", button_type="primary", disabled=True, width=120)
        self.pls_send_status = pn.pane.Markdown("", margin=(5, 0, 0, 10))
        self.last_pls_fig = None
        self.pls_send_btn.on_click(self._on_send_pls_to_html)

        self.scree_caption = pn.widgets.TextInput(placeholder="Optional caption...", value="", width=250)
        self.scree_send_btn = pn.widgets.Button(name="Send to HTML", button_type="primary", disabled=True, width=120)
        self.scree_send_status = pn.pane.Markdown("", margin=(5, 0, 0, 10))
        self.last_scree_fig = None
        self.scree_send_btn.on_click(self._on_send_scree_to_html)

        self.loadings_caption = pn.widgets.TextInput(placeholder="Optional caption...", value="", width=250)
        self.loadings_send_btn = pn.widgets.Button(name="Send to HTML", button_type="primary", disabled=True, width=120)
        self.loadings_send_status = pn.pane.Markdown("", margin=(5, 0, 0, 10))
        self.last_loadings_fig = None
        self.loadings_send_btn.on_click(self._on_send_loadings_to_html)

        # ---- MDS controls/plot ----
        self.mds_metric = pn.widgets.Select(name="MDS metric", options=["cosine", "euclidean"], value="euclidean", width=160)
        self.mds_caption = pn.widgets.TextInput(placeholder="Optional caption...", value="", width=250)
        self.mds_send_btn = pn.widgets.Button(name="Send to HTML", button_type="primary", disabled=True, width=120)
        self.mds_send_status = pn.pane.Markdown("", margin=(5, 0, 0, 10))
        self.last_mds_fig = None
        self.mds_send_btn.on_click(self._on_send_mds_to_html)
        self.mds_pane = TitledPlotPane(sizing_mode="stretch_width")

        # ---- t-SNE controls/plot ----
        self.tsne_perp = pn.widgets.FloatSlider(name="t-SNE perplexity", start=5.0, end=50.0, step=1.0, value=9.0, width=280)
        self.tsne_metric = pn.widgets.Select(name="t-SNE metric", options=["cosine", "euclidean"], value="cosine", width=160)
        self.tsne_caption = pn.widgets.TextInput(placeholder="Optional caption...", value="", width=250)
        self.tsne_send_btn = pn.widgets.Button(name="Send to HTML", button_type="primary", disabled=True, width=120)
        self.tsne_send_status = pn.pane.Markdown("", margin=(5, 0, 0, 10))
        self.last_tsne_fig = None
        self.tsne_send_btn.on_click(self._on_send_tsne_to_html)
        self.tsne_pane = TitledPlotPane(sizing_mode="stretch_width")

        self.visualize_btn = pn.widgets.Button(name="Visualize", button_type="success", disabled=True)
        self.status = pn.pane.Markdown("", sizing_mode="fixed")

        # ---- PCA controls/plot ----
        self.pca_n = pn.widgets.IntInput(name="# PCs for PCA", value=10, start=2, end=50, width=140)
        self.pca_x = pn.widgets.IntInput(name="PC X (1-based)", value=1, start=1, end=50, width=140)
        self.pca_y = pn.widgets.IntInput(name="PC Y (1-based)", value=2, start=1, end=50, width=140)
        self.pls_x = pn.widgets.IntInput(name="LV X (1-based)", value=1, start=1, end=10, width=140)
        self.pls_y = pn.widgets.IntInput(name="LV Y (1-based)", value=2, start=1, end=10, width=140)
        self.pca_pane = TitledPlotPane(sizing_mode="stretch_width")
        self.pls_pane = TitledPlotPane(sizing_mode="stretch_width")
        self.pca_scree_pane = TitledPlotPane(sizing_mode="stretch_width")
        self.pc_loadings_pane = TitledPlotPane(sizing_mode="stretch_width")

        # Layout
        self.section = pn.Column(
            pn.pane.Markdown("## 5) Visualization"),
            pn.pane.Markdown("_Populate metadata, click **Visualize** to enable, then tweak each section independently._",
                             styles={"color": "#555"}),
            pn.pane.Markdown("**1. Upload Pre-computed NMF Loadings (Optional):**", styles={"color":"#555", "margin-top": "10px"}),
            pn.Row(self.nmf_file),
            pn.pane.Markdown("**1.5. Upload Pre-computed Diversity Metrics (Optional):**", styles={"color":"#555", "margin-top": "10px"}),
            pn.Row(self.div_file),
            pn.pane.Markdown("**2. Upload Metadata:**", styles={"color":"#555", "margin-top": "10px"}),
            pn.Row(self.meta_file, pn.Spacer(width=10), self.meta_load_btn, pn.Spacer(width=20),
                   self.color_source, pn.Spacer(width=10), self.meta_col_select, pn.Spacer(width=10), self.div_metric_select, pn.Spacer(width=20), self.meta_mode),
            self.meta_help,

            self.meta_table,
            pn.Row(self.svg_export, pn.Spacer(width=12), self.visualize_btn),
            pn.layout.Divider(),
            pn.pane.Markdown("### 3. Subset Data (Optional)"),
            pn.Row(self.slice_col, pn.Column("Type:", self.slice_type)),
            pn.Row(self.slice_val_cat, self.slice_val_cont_min, self.slice_val_cont_max),
            pn.Row(self.slice_add_btn, self.slice_clear_btn, self.slice_apply_btn),
            self.slice_rules_md,
            pn.layout.Divider(),

            pn.pane.Markdown("### PCA / PLS-DA of NMF loadings (colored)"),
            pn.Row(self.pca_n, pn.Spacer(width=10), self.pca_x, pn.Spacer(width=10), self.pca_y, pn.Spacer(width=10), self.pls_x, pn.Spacer(width=10), self.pls_y),
            pn.Row(self.pca_caption, self.pca_send_btn, self.pca_send_status),
            self.pca_pane,
            pn.Row(self.pls_caption, self.pls_send_btn, self.pls_send_status),
            self.pls_pane,
            pn.pane.Markdown("#### PCA scree & PC loadings"),
            pn.Row(self.scree_caption, self.scree_send_btn, self.scree_send_status),
            self.pca_scree_pane,
            pn.Row(self.loadings_caption, self.loadings_send_btn, self.loadings_send_status),
            self.pc_loadings_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### MDS (2D)"),
            pn.Row(self.mds_metric, pn.Spacer(width=12), self.mds_caption, self.mds_send_btn, self.mds_send_status),
            self.mds_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### t-SNE (2D)"),
            pn.Row(self.tsne_perp, pn.Spacer(width=20), self.tsne_metric, pn.Spacer(width=12), self.tsne_caption, self.tsne_send_btn, self.tsne_send_status),
            self.tsne_pane,
            self.status,
            sizing_mode="stretch_width",
            visible=False,
        )

        # Wire

        self.meta_load_btn.on_click(self._on_load_meta_file)
        self.visualize_btn.on_click(self._on_visualize_enable)
        
        self.slice_col.param.watch(self._update_slicing_ui, 'value')
        self.slice_type.param.watch(self._update_slicing_ui, 'value')
        self.slice_add_btn.on_click(self._add_rule)
        self.slice_clear_btn.on_click(self._clear_rules)
        self.slice_apply_btn.on_click(self._apply_slicing)


        # Independent watchers
        self.color_source.param.watch(self._on_color_source_change, "value")
        self.meta_col_select.param.watch(lambda *_: self._refresh_all(), "value")
        self.div_metric_select.param.watch(lambda *_: self._refresh_all(), "value")
        self.meta_mode.param.watch(lambda *_: self._refresh_all(), "value")

        self.pca_n.param.watch(lambda *_: self._update_pca(), "value")
        self.pca_x.param.watch(lambda *_: self._update_pca(), "value")
        self.pca_y.param.watch(lambda *_: self._update_pca(), "value")
        self.pls_x.param.watch(lambda *_: self._update_pca(), "value")
        self.pls_y.param.watch(lambda *_: self._update_pca(), "value")

        self.mds_metric.param.watch(lambda *_: self._update_mds(), "value")
        self.tsne_perp.param.watch(lambda *_: self._update_tsne(), "value")
        self.tsne_metric.param.watch(lambda *_: self._update_tsne(), "value")

        self._on_color_source_change()


    # ---------- Slicing ----------
    def _get_master_df(self) -> pd.DataFrame:
        dfs = []
        if self.H_df is not None and not self.H_df.empty:
            dfs.append(self.H_df.copy())
            
        try:
            meta = pd.DataFrame(self.meta_table.value)
            if not meta.empty and "sample" in meta.columns:
                m = meta.set_index("sample")
                m.index = m.index.astype(str)
                dfs.append(m)
        except Exception:
            pass
            
        if self.diversity_df is not None and not self.diversity_df.empty:
            dfs.append(self.diversity_df.copy())
            
        if not dfs: return None
        master = dfs[0]
        for d in dfs[1:]:
            master = master.merge(d, left_index=True, right_index=True, how='outer', suffixes=("", "_dup"))
        return master.loc[:, ~master.columns.duplicated()].copy()

    def _update_slice_options(self):
        mdf = self._get_master_df()
        if mdf is not None:
            cols = list(mdf.columns)
            self.slice_col.options = cols
            if cols and self.slice_col.value not in cols:
                self.slice_col.value = cols[0]
            self._update_slicing_ui()

    def _update_slicing_ui(self, event=None):
        mdf = self._get_master_df()
        if mdf is None or not self.slice_col.value: return
        is_cat = (self.slice_type.value == "Categorical")
        self.slice_val_cat.visible = is_cat
        self.slice_val_cont_min.visible = not is_cat
        self.slice_val_cont_max.visible = not is_cat
        
        col = self.slice_col.value
        if is_cat:
            cats = [str(x) for x in mdf[col].dropna().unique()]
            self.slice_val_cat.options = cats
            self.slice_val_cat.value = []
        else:
            try:
                numeric_series = pd.to_numeric(mdf[col], errors='coerce').dropna()
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
        self.sliced_df = None
        self._update_rules_display()
        
    def _update_rules_display(self):
        if not self.rules:
            self.slice_rules_md.object = "**Active Rules:** None"
            return
            
        md = "**Active Rules (AND Logic):**\\n"
        for i, r in enumerate(self.rules):
            if r["type"] == "categorical":
                md += f"- **{r['col']}** is in `{r['categories']}`\\n"
            else:
                md += f"- **{r['col']}** between `{r['min']}` and `{r['max']}`\\n"
        self.slice_rules_md.object = md

    def _apply_slicing(self, event):
        df = self._get_master_df()
        if df is None: return
        
        for r in self.rules:
            if r["type"] == "categorical":
                df = df[df[r["col"]].astype(str).isin(r["categories"])]
            else:
                df[r["col"]] = pd.to_numeric(df[r["col"]], errors='coerce')
                df = df[(df[r["col"]] >= r["min"]) & (df[r["col"]] <= r["max"])]
                
        self.sliced_df = df.copy()
        self.status.object = f"**Data Sliced:** {len(self.sliced_df)} samples remain."
        self._refresh_all()

    def _get_valid_samples(self):
        if self.sliced_df is not None:
            return list(map(str, self.sliced_df.index))
        return self.samples

    # ---------- External API ----------

    def set_input(self, H_df: pd.DataFrame) -> None:
        self.H_df = H_df.copy()
        self.H_df.index = self.H_df.index.astype(str)
        self.samples = list(self.H_df.index)
        df = pd.DataFrame({"sample": self.samples, "metadata": [None] * len(self.samples)})
        self.meta_table.value = df
        self.metadata_df = df
        self.visualize_btn.disabled = False
        self.enabled = False
        self.section.visible = True
        self.status.object = ok("NMF loadings received. Enter/upload metadata, then click **Visualize**.")
        self.pca_pane.object = None; self.mds_pane.object = None; self.tsne_pane.object = None
        self._update_slice_options()
        # if we already have diversity metrics, populate choices
        self._populate_div_metric_choices()
        self._update_slice_options()

    def _on_load_nmf_csv(self, event):
        if not self.nmf_file.value:
            return
        try:
            filename = self.nmf_file.filename or "nmf_loadings.csv"
            df = pd.read_csv(io.BytesIO(self.nmf_file.value), index_col=0)
            self.set_input(df)
            self.status.object = ok(f"Loaded NMF CSV: {filename}. Now upload metadata.")
        except Exception as e:
            self.status.object = warn(f"Failed to parse NMF CSV: {e}")

    def _on_load_div_csv(self, event):
        if not self.div_file.value:
            return
        try:
            filename = self.div_file.filename or "diversity.csv"
            df = pd.read_csv(io.BytesIO(self.div_file.value), index_col=0)
            self.set_diversity(df)
            self.status.object = ok(f"Loaded Diversity CSV: {filename}.")
        except Exception as e:
            self.status.object = warn(f"Failed to parse Diversity CSV: {e}")

    def set_diversity(self, df: pd.DataFrame) -> None:
        """Accept external diversity metrics (rows=samples, cols=metrics)."""
        if df is None or df.empty:
            self.diversity_df = None
        else:
            # ensure index are strings matching sample strings
            df_ = df.copy()
            df_.index = list(map(str, df_.index))
            self.diversity_df = df_
        self._populate_div_metric_choices()
        self._update_slice_options()

    # ---------- internals ----------
    def _populate_div_metric_choices(self):
        metrics = []
        if self.diversity_df is not None:
            metrics = list(self.diversity_df.columns.astype(str))
        self.div_metric_select.options = metrics
        if metrics and (self.div_metric_select.value not in metrics):
            self.div_metric_select.value = metrics[0]
        if not metrics:
            self.div_metric_select.value = None

    def _on_color_source_change(self, *_):
        use_div = (self.color_source.value == "diversity metric")
        self.div_metric_select.visible = use_div
        self.meta_col_select.visible = not use_div

    def _metadata_dict(self) -> Dict[str, object]:
        if self.metadata_df is None:
            return {}
        try:
            self.metadata_df = pd.DataFrame(self.meta_table.value)
        except Exception:
            pass
        
        col = str(self.meta_col_select.value)
        if col not in self.metadata_df.columns:
            return {s: None for s in self.samples}
            
        vals: Dict[str, object] = {}
        for _, r in self.metadata_df.iterrows():
            s = str(r["sample"])
            if s in self.samples:
                vals[s] = r[col]
        for s in self.samples:
            vals.setdefault(s, None)
        return vals

    def _current_color_dict(self) -> Dict[str, object]:
        """Return mapping sample->value/category based on current color_source."""
        if self.color_source.value == "diversity metric":
            if self.diversity_df is None or self.div_metric_select.value is None:
                return {s: None for s in self.samples}
            col = str(self.div_metric_select.value)
            series = self.diversity_df[col] if col in self.diversity_df.columns else None
            if series is None:
                return {s: None for s in self.samples}
            d = {str(idx): series.loc[idx] for idx in series.index}
            # Ensure all samples present
            return {s: d.get(s, None) for s in self.samples}
        else:
            return self._metadata_dict()

    def _on_visualize_enable(self, _=None):
        if self.H_df is None or self.H_df.empty:
            self.status.object = warn("No NMF loadings available.")
            return
        self.enabled = True
        self._refresh_all()
        self.status.object = ok("Visualization enabled. Adjust controls to update each plot.")
        
    def _on_load_meta_file(self, *_):
        # backwards-compat shim
        return self._on_load_file()


    def _refresh_all(self):
        if not self.enabled: return
        self._update_pca()
        self._update_mds()
        self._update_tsne()
        
        from common_plot import apply_export_prefix_to_pane
        prefix = getattr(self, "export_prefix", "CE_analysis")
        apply_export_prefix_to_pane(self.pca_pane, prefix)
        apply_export_prefix_to_pane(self.pls_pane, prefix)
        apply_export_prefix_to_pane(self.pca_scree_pane, prefix)
        apply_export_prefix_to_pane(self.pc_loadings_pane, prefix)
        apply_export_prefix_to_pane(self.mds_pane, prefix)
        apply_export_prefix_to_pane(self.tsne_pane, prefix)
        
    def _on_load_file(self, _=None):
        self.status.object = "Loading metadata..."
        if not self.meta_file.value:
            self.status.object = warn("Choose a metadata file (.csv or .xlsx) first.")
            return
        try:
            df = _read_metadata_table_from_bytes(self.meta_file.filename or "", bytes(self.meta_file.value))
        except Exception as e:
            self.status.object = warn(f"Failed to parse file: {e}")
            return
    
        opts = [c for c in df.columns if c != "sample"]
        self.meta_col_select.options = opts
        if opts: self.meta_col_select.value = opts[0]
        
        df["sample"] = df["sample"].astype(str)
        if self.samples:
            df_samples = pd.DataFrame({"sample": self.samples})
            df = df_samples.merge(df, on="sample", how="left")
            
        self.meta_table.value = df
        self.metadata_df = df
        
        # update tabulator columns
        cols = [{"field": "sample", "title": "sample", "editable": False}]
        for c in opts:
            cols.append({"field": c, "title": c, "editor": "input"})
        self.meta_table.columns = cols
        
        self.status.object = ok("Metadata loaded. You can edit cells directly in the table.")
        self._update_slice_options()


    def _update_pca(self):
        if not self.enabled or self.H_df is None or self.H_df.empty:
            return
        meta = self._current_color_dict()
        cat = (self.meta_mode.value == "categorical")
        try:
            import CEtools as cet
            from CEtools import pca_viz as pv
            from sklearn.decomposition import PCA

            valid_samples = self._get_valid_samples()
            H_subset = self.H_df.loc[self.H_df.index.isin(valid_samples)].copy()
            if H_subset.empty: return
            
            X = cet.prepare_features(H_subset, row_norm="l1", zscore_cols=True)
            ncom = int(self.pca_n.value)
            pcx  = int(self.pca_x.value)
            pcy  = int(self.pca_y.value)
            max_comp = min(X.shape[0], X.shape[1])
            ncom_eff = min(max_comp, max(ncom, pcx, pcy))

            pca = PCA(n_components=ncom_eff, random_state=0)
            scores = pca.fit_transform(X)
            self.last_pca = pca
            ix = min(pcx - 1, scores.shape[1] - 1)
            iy = min(pcy - 1, scores.shape[1] - 1)
            x = scores[:, ix]; y = scores[:, iy]
            labels = np.array(list(map(str, H_subset.index.astype(str))))

            def _add_ellipses(p_fig, x_vals, y_vals, c_vals, cmap):
                for c in set(c_vals):
                    if c == "NA": continue
                    mask = (np.array(c_vals) == c)
                    xm = x_vals[mask]
                    ym = y_vals[mask]
                    if len(xm) < 3: continue
                    cov = np.cov(xm, ym)
                    mean_x, mean_y = np.mean(xm), np.mean(ym)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = eigvals.argsort()[::-1]
                    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                    angle = np.arctan2(*eigvecs[:, 0][::-1])
                    width, height = 2 * np.sqrt(5.991 * eigvals)
                    p_fig.ellipse(x=mean_x, y=mean_y, width=width, height=height, angle=angle, 
                                  fill_color=cmap[c], line_color=cmap[c], fill_alpha=0.1, line_alpha=0.6,
                                  legend_label=str(c)[:12])

            # calculate dynamic width for legend
            calc_width = 620 if cat else 520
            calc_height = 420

            p = bokeh.plotting.figure(
                width=calc_width, height=calc_height,
                title=f"PCA of NMF loadings (PC{pcx} vs PC{pcy})",
                x_axis_label=f"PC{pcx} {pca.explained_variance_ratio_[ix]*100.0:.1f}%",
                y_axis_label=f"PC{pcy} {pca.explained_variance_ratio_[iy]*100.0:.1f}%",
                tools="hover,pan,box_zoom,reset,save",
                active_scroll=None,
                output_backend="svg" if self.svg_export.value else "canvas",
            )
            pls_p = None

            if cat:
                cats = [str(meta.get(s, "NA")) if meta.get(s, None) is not None else "NA" for s in labels]
                unique = sorted(set(cats))
                from bokeh.palettes import Category10, Magma256
                if len(unique) <= 10:
                    palette = list(Category10[10])
                else:
                    idxs = np.linspace(0, 255, num=len(unique), dtype=int)
                    palette = [Magma256[i] for i in idxs]
                cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
                
                # Plot PCA categorical
                for c in unique:
                    mask = (np.array(cats) == c)
                    src = bokeh.models.ColumnDataSource(dict(x=x[mask], y=y[mask], label=labels[mask]))
                    p.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color=cmap[c],
                             source=src, legend_label=str(c)[:12])
                _add_ellipses(p, x, y, cats, cmap)
                p.legend.click_policy = "hide"
                p.add_layout(p.legend[0], "right")
                p.title.text += " — colored by category"

                # PLS-DA
                from sklearn.cross_decomposition import PLSRegression
                unique_valid = [c for c in unique if c != "NA"]
                if len(unique_valid) >= 2:
                    cat_to_int = {c: i for i, c in enumerate(unique_valid)}
                    y_int = np.array([cat_to_int.get(c, -1) for c in cats])
                    valid_mask = (y_int != -1)
                    if valid_mask.sum() > 2:
                        y_onehot = np.zeros((valid_mask.sum(), len(unique_valid)))
                        y_onehot[np.arange(valid_mask.sum()), y_int[valid_mask]] = 1
                        lv_x = int(self.pls_x.value)
                        lv_y = int(self.pls_y.value)
                        ncom_pls = max(lv_x, lv_y)

                        pls = PLSRegression(n_components=ncom_pls)
                        pls.fit(X[valid_mask], y_onehot)
                        scores_pls = pls.transform(X)
                        pls_x_vals, pls_y_vals = scores_pls[:, lv_x - 1], scores_pls[:, lv_y - 1]
                        
                        pls_p = bokeh.plotting.figure(
                            width=calc_width, height=calc_height,
                            title="PLS-DA of NMF loadings",
                            x_axis_label=f"LV{lv_x}", y_axis_label=f"LV{lv_y}",
                            tools="hover,pan,box_zoom,reset,save",
                            active_scroll=None,
                            output_backend="svg" if self.svg_export.value else "canvas",
                        )
                        for c in unique:
                            mask = (np.array(cats) == c)
                            src = bokeh.models.ColumnDataSource(dict(x=pls_x_vals[mask], y=pls_y_vals[mask], label=labels[mask]))
                            pls_p.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color=cmap[c],
                                         source=src, legend_label=str(c)[:12])
                        _add_ellipses(pls_p, pls_x_vals, pls_y_vals, cats, cmap)
                        pls_p.legend.click_policy = "hide"
                        pls_p.add_layout(pls_p.legend[0], "right")
                        pls_p.add_tools(bokeh.models.HoverTool(tooltips=[("sample", "@label"), ("x", "@x{0.000}"), ("y", "@y{0.000}")]))

            else:
                vals = []
                for s in labels:
                    v = meta.get(s, None)
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(np.nan)
                vals = np.asarray(vals, dtype=float)
                from bokeh.palettes import Magma256
                mapper = bokeh.models.LinearColorMapper(
                    palette=Magma256,
                    low=float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0,
                    high=float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0,
                    nan_color="lightgray"
                )
                src = bokeh.models.ColumnDataSource(dict(x=x, y=y, label=labels, value=vals))
                color_spec = {"field": "value", "transform": mapper}
                p.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color=color_spec, source=src)
                color_var = str(self.div_metric_select.value) if self.color_source.value == "diversity metric" else str(self.meta_col_select.value)
                color_bar = bokeh.models.ColorBar(color_mapper=mapper, title=str(color_var)[:20], label_standoff=8, location=(0, 0))
                p.add_layout(color_bar, "right")
                p.title.text += f" — colored by {str(color_var)[:20]}"

            p.add_tools(bokeh.models.HoverTool(tooltips=[("sample", "@label"), ("x", "@x{0.000}"), ("y", "@y{0.000}")]))
            
            self.pca_pane.object = p
            self.pls_pane.object = pls_p
            self.last_pca_fig = p
            self.pca_send_btn.disabled = False
            if pls_p is None:
                self.pls_pane.visible = False
                self.pls_send_btn.disabled = True
            else:
                self.pls_pane.visible = True
                self.last_pls_fig = pls_p
                self.pls_send_btn.disabled = False

            # CEtools pca_viz subplots, suppress external show
            scree_fig = None
            try:
                from CEtools import pca_viz as pv
                with no_bokeh_show():
                    scree_fig = pv.plot_pca_scree_bokeh(pca)
                if scree_fig.legend:
                    scree_fig.legend.location = "center_right"
                self.pca_scree_pane.object = scree_fig
                self.last_scree_fig = scree_fig
                self.scree_send_btn.disabled = False
            except Exception as e:
                self.pca_scree_pane.object = None
                self.last_scree_fig = None
                self.scree_send_btn.disabled = True
                self.status.object = warn(f"PCA scree failed: {e}")

            load_fig = None
            try:
                from CEtools import pca_viz as pv
                with no_bokeh_show():
                    _, load_fig = pv.plot_pc_loadings_bar_bokeh(
                        pca, pc=int(self.pca_x.value), kind="line"
                    )
                self.pc_loadings_pane.object = load_fig
                self.last_loadings_fig = load_fig
                self.loadings_send_btn.disabled = False
            except Exception as e:
                self.pc_loadings_pane.object = None
                self.last_loadings_fig = None
                self.loadings_send_btn.disabled = True
                self.status.object = warn(f"PC loadings plot failed: {e}")

            from common_plot import apply_export_prefix_to_pane
            prefix = getattr(self, "export_prefix", "CE_analysis")
            apply_export_prefix_to_pane(self.pca_pane, prefix)
            apply_export_prefix_to_pane(self.pls_pane, prefix)
            apply_export_prefix_to_pane(self.pca_scree_pane, prefix)
            apply_export_prefix_to_pane(self.pc_loadings_pane, prefix)

        except Exception as e:
            self.pca_pane.object = None
            self.pca_scree_pane.object = None
            self.pc_loadings_pane.object = None
            self.status.object = warn(f"PCA failed: {e}")

    def _update_mds(self):
        if not self.enabled or self.H_df is None or self.H_df.empty:
            return
        meta = self._current_color_dict()
        cat = (self.meta_mode.value == "categorical")
        try:
            import CEtools as cet
            
            valid_samples = self._get_valid_samples()
            H_subset = self.H_df.loc[self.H_df.index.isin(valid_samples)].copy()
            if H_subset.empty: return

            with no_bokeh_show():
                coords, _, mds_fig = cet.embed_with_mds(
                    H_subset,
                    metric=str(self.mds_metric.value),
                    row_norm="l1",
                    zscore_cols=True,
                    metadata=meta,
                    metadata_categorical=cat,
                    title=f"MDS (metric={self.mds_metric.value})",
                )
            mds_fig.output_backend = "svg" if self.svg_export.value else "canvas"
            mds_fig.toolbar.active_scroll = None
            if cat:
                mds_fig.renderers = [r for r in mds_fig.renderers if str(type(r).__name__) != 'GlyphRenderer']
                for leg in mds_fig.legend:
                    for pos in [mds_fig.center, mds_fig.right, mds_fig.left, mds_fig.above, mds_fig.below]:
                        if leg in pos: pos.remove(leg)
                
                x, y = coords[:, 0], coords[:, 1]
                labels = np.array(list(map(str, H_subset.index.astype(str))))
                cats = [str(meta.get(s, "NA")) if meta.get(s, None) is not None else "NA" for s in labels]
                unique = sorted(set(cats))
                from bokeh.palettes import Category10, Magma256
                if len(unique) <= 10:
                    palette = list(Category10[10])
                else:
                    idxs = np.linspace(0, 255, num=len(unique), dtype=int)
                    palette = [Magma256[i] for i in idxs]
                cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
                
                for c in unique:
                    mask = (np.array(cats) == c)
                    src = bokeh.models.ColumnDataSource(dict(x=x[mask], y=y[mask], label=labels[mask]))
                    mds_fig.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color=cmap[c],
                                   source=src, legend_label=str(c)[:12])
                
                mds_fig.legend.click_policy = "hide"
                mds_fig.add_layout(mds_fig.legend[-1], "right")
                mds_fig.width = (mds_fig.width or 520) + 100
                mds_fig.add_tools(bokeh.models.HoverTool(tooltips=[("sample", "@label"), ("x", "@x{0.000}"), ("y", "@y{0.000}")]))
            else:
                color_var = str(self.div_metric_select.value) if self.color_source.value == "diversity metric" else str(self.meta_col_select.value)
                for rb in mds_fig.right:
                    if isinstance(rb, bokeh.models.ColorBar):
                        rb.title = str(color_var)[:20]
                mds_fig.title.text += f" — colored by {str(color_var)[:20]}"
                
            self.mds_pane.object = mds_fig
            self.last_mds_fig = mds_fig
            self.mds_send_btn.disabled = False
            from common_plot import apply_export_prefix_to_pane
            prefix = getattr(self, "export_prefix", "CE_analysis")
            apply_export_prefix_to_pane(self.mds_pane, prefix)
        except Exception as e:
            self.mds_pane.object = None
            self.last_mds_fig = None
            self.mds_send_btn.disabled = True
            self.status.object = warn(f"MDS failed: {e}")

    def _update_tsne(self):
        if not self.enabled or self.H_df is None or self.H_df.empty:
            return
        meta = self._current_color_dict()
        cat = (self.meta_mode.value == "categorical")
        try:
            import CEtools as cet
            
            valid_samples = self._get_valid_samples()
            H_subset = self.H_df.loc[self.H_df.index.isin(valid_samples)].copy()
            if H_subset.empty: return

            with no_bokeh_show():
                coords, _, tsne_fig = cet.embed_with_tsne(
                    H_subset,
                    perplexity=float(self.tsne_perp.value),
                    metric=str(self.tsne_metric.value),
                    row_norm="l1",
                    zscore_cols=True,
                    random_state=0,
                    metadata=meta,
                    metadata_categorical=cat,
                    title=f"t-SNE (metric={self.tsne_metric.value}, perplexity={self.tsne_perp.value:.0f})",
                )
            tsne_fig.output_backend = "svg" if self.svg_export.value else "canvas"
            tsne_fig.toolbar.active_scroll = None
            if cat:
                tsne_fig.renderers = [r for r in tsne_fig.renderers if str(type(r).__name__) != 'GlyphRenderer']
                for leg in tsne_fig.legend:
                    for pos in [tsne_fig.center, tsne_fig.right, tsne_fig.left, tsne_fig.above, tsne_fig.below]:
                        if leg in pos: pos.remove(leg)
                
                x, y = coords[:, 0], coords[:, 1]
                labels = np.array(list(map(str, H_subset.index.astype(str))))
                cats = [str(meta.get(s, "NA")) if meta.get(s, None) is not None else "NA" for s in labels]
                unique = sorted(set(cats))
                from bokeh.palettes import Category10, Magma256
                if len(unique) <= 10:
                    palette = list(Category10[10])
                else:
                    idxs = np.linspace(0, 255, num=len(unique), dtype=int)
                    palette = [Magma256[i] for i in idxs]
                cmap = {c: palette[i % len(palette)] for i, c in enumerate(unique)}
                
                for c in unique:
                    mask = (np.array(cats) == c)
                    src = bokeh.models.ColumnDataSource(dict(x=x[mask], y=y[mask], label=labels[mask]))
                    tsne_fig.circle("x", "y", size=8, alpha=0.9, line_color=None, fill_color=cmap[c],
                                    source=src, legend_label=str(c)[:12])
                
                tsne_fig.legend.click_policy = "hide"
                tsne_fig.add_layout(tsne_fig.legend[-1], "right")
                tsne_fig.width = (tsne_fig.width or 520) + 100
                tsne_fig.add_tools(bokeh.models.HoverTool(tooltips=[("sample", "@label"), ("x", "@x{0.000}"), ("y", "@y{0.000}")]))
            else:
                color_var = str(self.div_metric_select.value) if self.color_source.value == "diversity metric" else str(self.meta_col_select.value)
                for rb in tsne_fig.right:
                    if isinstance(rb, bokeh.models.ColorBar):
                        rb.title = str(color_var)[:20]
                tsne_fig.title.text += f" — colored by {str(color_var)[:20]}"
                
            self.tsne_pane.object = tsne_fig
            self.last_tsne_fig = tsne_fig
            self.tsne_send_btn.disabled = False
            from common_plot import apply_export_prefix_to_pane
            prefix = getattr(self, "export_prefix", "CE_analysis")
            apply_export_prefix_to_pane(self.tsne_pane, prefix)
        except Exception as e:
            self.tsne_pane.object = None
            self.last_tsne_fig = None
            self.tsne_send_btn.disabled = True
            self.status.object = warn(f"t-SNE failed: {e}")

    def _send_plot_to_html(self, entry: dict, status_pane: pn.pane.Markdown):
        existing_idx = None
        for i, p in enumerate(self.html_plots):
            if p.get("id") == entry.get("id"):
                existing_idx = i
                break
        if existing_idx is not None:
            self.html_plots[existing_idx] = entry
        else:
            self.html_plots.append(entry)

        status_pane.object = f"**Sent '{entry.get('title', 'plot')}' to HTML Report export.**"
        if callable(self.on_send_to_html):
            try:
                self.on_send_to_html(entry)
            except Exception as e:
                print(f"[VizController] Failed to send plot to HTML report controller: {e}")

    def _on_send_pca_to_html(self, _=None):
        if self.last_pca_fig is None:
            return
        fig = self.last_pca_fig
        fig.frame_width = 400
        fig.frame_height = 400
        fig.match_aspect = False
        fig.aspect_ratio = None
        fig.sizing_mode = None
        fig.width = None
        fig.height = None
        pcx = int(self.pca_x.value)
        pcy = int(self.pca_y.value)
        cvar = str(self.div_metric_select.value) if self.color_source.value == "diversity metric" else str(self.meta_col_select.value or "")
        c_suffix = f"_coloredBy{cvar}" if cvar and cvar != "None" else ""
        title = f"PCA_PC{pcx}{pcy}{c_suffix}"
        entry = {
            "id": f"viz_pca_pc{pcx}_pc{pcy}",
            "type": "viz_pca",
            "category": "Dimensionality Reduction",
            "title": title,
            "figure": fig,
            "caption": self.pca_caption.value.strip() if self.pca_caption.value else ""
        }
        self._send_plot_to_html(entry, self.pca_send_status)

    def _on_send_pls_to_html(self, _=None):
        if self.last_pls_fig is None:
            return
        fig = self.last_pls_fig
        fig.frame_width = 400
        fig.frame_height = 400
        fig.match_aspect = False
        fig.aspect_ratio = None
        fig.sizing_mode = None
        fig.width = None
        fig.height = None
        lv_x = int(self.pls_x.value)
        lv_y = int(self.pls_y.value)
        cvar = str(self.div_metric_select.value) if self.color_source.value == "diversity metric" else str(self.meta_col_select.value or "")
        c_suffix = f"_coloredBy{cvar}" if cvar and cvar != "None" else ""
        title = f"PLS_LV{lv_x}{lv_y}{c_suffix}"
        entry = {
            "id": f"viz_pls_lv{lv_x}_lv{lv_y}",
            "type": "viz_pls",
            "category": "Dimensionality Reduction",
            "title": title,
            "figure": fig,
            "caption": self.pls_caption.value.strip() if self.pls_caption.value else ""
        }
        self._send_plot_to_html(entry, self.pls_send_status)

    def _on_send_scree_to_html(self, _=None):
        if self.last_scree_fig is None:
            return
        entry = {
            "id": "viz_scree_plot",
            "type": "viz_scree",
            "category": "Dimensionality Reduction",
            "title": "PCA Scree Plot (Explained Variance)",
            "figure": self.last_scree_fig,
            "caption": self.scree_caption.value.strip() if self.scree_caption.value else ""
        }
        self._send_plot_to_html(entry, self.scree_send_status)

    def _on_send_loadings_to_html(self, _=None):
        if self.last_loadings_fig is None:
            return
        pc = int(self.pca_x.value)
        entry = {
            "id": f"viz_loadings_pc{pc}",
            "type": "viz_loadings",
            "category": "Dimensionality Reduction",
            "title": f"PC Loadings (PC{pc})",
            "figure": self.last_loadings_fig,
            "caption": self.loadings_caption.value.strip() if self.loadings_caption.value else ""
        }
        self._send_plot_to_html(entry, self.loadings_send_status)

    def _on_send_mds_to_html(self, _=None):
        if self.last_mds_fig is None:
            return
        fig = self.last_mds_fig
        fig.frame_width = 400
        fig.frame_height = 400
        fig.match_aspect = False
        fig.aspect_ratio = None
        fig.sizing_mode = None
        fig.width = None
        fig.height = None
        metric = str(self.mds_metric.value)
        cvar = str(self.div_metric_select.value) if self.color_source.value == "diversity metric" else str(self.meta_col_select.value or "")
        c_suffix = f"_coloredBy{cvar}" if cvar and cvar != "None" else ""
        title = f"MDS_2D{c_suffix}"
        entry = {
            "id": f"viz_mds_{metric}",
            "type": "viz_mds",
            "category": "Dimensionality Reduction",
            "title": title,
            "figure": fig,
            "caption": self.mds_caption.value.strip() if self.mds_caption.value else ""
        }
        self._send_plot_to_html(entry, self.mds_send_status)

    def _on_send_tsne_to_html(self, _=None):
        if self.last_tsne_fig is None:
            return
        fig = self.last_tsne_fig
        fig.frame_width = 400
        fig.frame_height = 400
        fig.match_aspect = False
        fig.aspect_ratio = None
        fig.sizing_mode = None
        fig.width = None
        fig.height = None
        metric = str(self.tsne_metric.value)
        perp = float(self.tsne_perp.value)
        cvar = str(self.div_metric_select.value) if self.color_source.value == "diversity metric" else str(self.meta_col_select.value or "")
        c_suffix = f"_coloredBy{cvar}" if cvar and cvar != "None" else ""
        title = f"tSNE_2D{c_suffix}"
        entry = {
            "id": f"viz_tsne_{metric}_{int(perp)}",
            "type": "viz_tsne",
            "category": "Dimensionality Reduction",
            "title": title,
            "figure": fig,
            "caption": self.tsne_caption.value.strip() if self.tsne_caption.value else ""
        }
        self._send_plot_to_html(entry, self.tsne_send_status)

# Public factory
def build_viz_section():
    ctrl = VizController()
    return ctrl.section, ctrl
