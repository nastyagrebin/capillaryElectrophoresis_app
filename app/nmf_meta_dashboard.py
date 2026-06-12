from __future__ import annotations

import io
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

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

    # ---------------- UI ----------------
    basis_sel = pn.widgets.Select(name="Inspect basis", options=list(stats.index[: min(25, len(stats))]), width=220)

    # panes/figures created ONCE
    stats_pane = pn.pane.DataFrame(stats.reset_index(), height=260, sizing_mode="stretch_width")

    # per-basis loadings jitter figure
    jitter_src = ColumnDataSource(data=dict(x=[], y=[], group=[], sample_id=[]))
    jitter_fig = figure(
        width=600, height=360,
        x_axis_label="group",
        y_axis_label="loading",
        title="Loadings by group",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll=None,
    )
    jitter_r = jitter_fig.circle("x", "y", source=jitter_src, size=7, fill_alpha=0.75, line_alpha=0.25)
    jitter_fig.add_tools(HoverTool(renderers=[jitter_r], tooltips=[("sample", "@sample_id"), ("group", "@group"), ("loading", "@y{0.000}")]))
    jitter_fig.xaxis.ticker = list(range(len(groups)))
    jitter_fig.xaxis.major_label_overrides = {i: g for i, g in enumerate(groups)}

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
        pn.layout.Divider(),
        pn.Row(basis_sel),
        pn.Row(pn.pane.Bokeh(jitter_fig), sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


class NMFMetaDashboardController:
    def __init__(self):
        self.H_df: pd.DataFrame | None = None
        self.metadata_df: pd.DataFrame | None = None

        self.file_input_meta = pn.widgets.FileInput(accept=".csv,.xlsx", multiple=False)
        self.file_input_meta.param.watch(self._on_meta_upload, "value")
        
        self.file_input_nmf = pn.widgets.FileInput(accept=".csv", multiple=False)
        self.file_input_nmf.param.watch(self._on_nmf_upload, "value")
        
        self.metadata_preview = pn.pane.DataFrame(pd.DataFrame(), max_height=200, sizing_mode="stretch_width", visible=False)
        self.nmf_preview = pn.pane.DataFrame(pd.DataFrame(), max_height=200, sizing_mode="stretch_width", visible=False)

        self.sample_col_select = pn.widgets.Select(name="Sample ID Column", options=[])
        self.group_col_select = pn.widgets.Select(name="Category Column", options=[])
        
        self.min_group_size = pn.widgets.IntInput(name="Min Group Size", value=3, start=1)

        self.run_btn = pn.widgets.Button(name="Run Group Importance Analysis", button_type="primary", disabled=True)
        self.run_btn.on_click(self._on_run)

        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")
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
            pn.Row(self.sample_col_select, self.group_col_select, self.min_group_size),
            pn.Row(self.run_btn),
            self.status,
            pn.layout.Divider(),
            self.dashboard_pane,
            sizing_mode="stretch_width"
        )

    def set_H(self, H_df: pd.DataFrame):
        self.H_df = H_df.copy()
        
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

    def _on_nmf_upload(self, event):
        if not self.file_input_nmf.value:
            return
            
        try:
            filename = self.file_input_nmf.filename
            content = self.file_input_nmf.value
            
            df = pd.read_csv(io.BytesIO(content))
            self.set_H(df)
            self.status.object = f"Loaded standalone NMF CSV: {filename}."

        except Exception as e:
            self.status.object = f"**Error parsing NMF file:** {e}"

    def _on_meta_upload(self, event):
        if not self.file_input_meta.value:
            return
            
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

            if self.H_df is not None:
                self.run_btn.disabled = False
                self.status.object = f"Loaded {filename}. Ready to run analysis."
            else:
                self.status.object = f"Loaded {filename}, but waiting for NMF decomposition to finish."

        except Exception as e:
            self.status.object = f"**Error parsing file:** {e}"
            self.metadata_df = None
            self.metadata_preview.visible = False
            self.run_btn.disabled = True

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
                metadata_df=self.metadata_df,
                nmf_loadings_df=self.H_df,
                group_col=self.group_col_select.value,
                meta_sample_col=self.sample_col_select.value,
                nmf_sample_col="Unnamed: 0",
                min_group_size=self.min_group_size.value,
            )
            self.dashboard_pane.append(dash)
            self.status.object = "Analysis complete."
        except Exception as e:
            self.status.object = f"**Analysis failed:** {e}"

def build_nmf_meta_section() -> tuple[pn.Column, NMFMetaDashboardController]:
    ctrl = NMFMetaDashboardController()
    return ctrl.view, ctrl
