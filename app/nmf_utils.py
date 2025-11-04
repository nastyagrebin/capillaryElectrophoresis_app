# ============================
# FILE: app/nmf_utils.py
# ============================
from __future__ import annotations
from typing import Optional, Tuple, List, Callable

import io
import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting
import bokeh.models

OK = "OK:"; WARN = "Warning:"
def ok(m): return f"{OK} {m}"
def warn(m): return f"{WARN} {m}"

pn.extension('tabulator')

def _parse_wide_aligned_csv(name: str, data: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Optional[np.ndarray]]:
    """
    Parse 'pseudotimes_wide.csv' exported by Alignment.

    Columns expected:
      - optional 'time'
      - for each sample S: '{S}_pt' (pseudotime) and '{S}' (intensity)

    Returns:
      pseudotimes_df (rows=timepoints, cols=samples),
      norm_df       (rows=timepoints, cols=samples),
      samples       (list),
      time          (np.ndarray | None)
    """
    df = pd.read_csv(io.BytesIO(data or b""))
    cols = list(df.columns)
    samples = []
    for c in cols:
        if c == "time" or c.endswith("_pt"):  # intensity column candidates
            continue
        s = c
        if f"{s}_pt" in cols:
            samples.append(s)
    if not samples:
        raise ValueError("No valid sample columns found (expect '{sample}_pt' and '{sample}')")

    P = pd.DataFrame({s: df[f"{s}_pt"].to_numpy() for s in samples})
    Y = pd.DataFrame({s: df[s].to_numpy() for s in samples})
    P.index = df.index
    Y.index = df.index
    t = df["time"].to_numpy() if "time" in df.columns else None
    return P, Y, samples, t


class NMFController:
    """
    NMF tab controller using CEtools' continuous Gaussian basis NNLS routines.

    Public hooks:
      - set_input(pseudotimes_df, norm_df, rows_are_traces=False)
      - on_done: Optional[Callable[[pd.DataFrame], None]]  -> called after Calculate NMF Loadings
    """
    def __init__(self):
        # Data
        self.pseudotimes_df: Optional[pd.DataFrame] = None
        self.norm_df: Optional[pd.DataFrame] = None
        self.rows_are_traces: bool = False
        self.samples: List[str] = []

        # Results
        self.H_df: Optional[pd.DataFrame] = None
        self.Phi = None
        self.centers = None
        self.pseudo_used_df: Optional[pd.DataFrame] = None

        # Optional callback for app.py to unlock Viz/Diversity
        self.on_done: Optional[Callable[[pd.DataFrame], None]] = None

        # ---------------- Import aligned CSV (now at top of NMF tab) ----------------
        self.aligned_file = pn.widgets.FileInput(accept=".csv", multiple=False)
        self.aligned_load_btn = pn.widgets.Button(name="Load aligned pseudotimes_wide.csv", button_type="primary")
        self.aligned_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.aligned_preview = pn.widgets.Tabulator(pd.DataFrame(), height=180, show_index=False)

        # ---------------- Controls ----------------
        self.k_slider = pn.widgets.IntSlider(name="K (number of basis)", start=20, end=500, value=250, step=5, width=260)
        self.l2_input = pn.widgets.FloatInput(name="L2 (ridge)", value=1e-5, step=1e-5, start=0.0, width=160)
        self.sample_select = pn.widgets.Select(name="Sample for preview", options=[], value=None, width=260)
        self.preview_btn = pn.widgets.Button(name="Preview reconstruction", button_type="success", disabled=True)
        self.calc_btn = pn.widgets.Button(name="Calculate NMF Loadings", button_type="success", disabled=True)
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")

        # ---------------- Plots ----------------
        self.recon_pane = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")
        self.heatmap_pane = pn.pane.Bokeh(height=520, sizing_mode="stretch_width")

        # ---------------- Export (loadings CSV) ----------------
        self.csv_name = pn.widgets.TextInput(name="Loadings CSV filename", value="nmf_loadings.csv", width=260)
        self.csv_download = pn.widgets.FileDownload(
            label="Download loadings CSV", filename=self.csv_name.value,
            button_type="primary", embed=False, auto=False, callback=lambda: io.BytesIO(b""),
            disabled=True
        )
        self.csv_name.param.watch(lambda e: setattr(self.csv_download, "filename", e.new or "nmf_loadings.csv"), "value")
        # Optional callback: notify app when aligned CSV is imported
        self.on_aligned_imported = None  # type: Optional[Callable[[pd.DataFrame, pd.DataFrame, bool], None]]


        # Layout
        self.section = pn.Column(
            pn.pane.Markdown("## 4) NMF"),
            pn.pane.Markdown("_Load aligned data here **or** arrive from the Alignment tab._", styles={"color":"#555"}),
            pn.Row(self.aligned_file, pn.Spacer(width=8), self.aligned_load_btn),
            self.aligned_preview,
            self.aligned_status,
            pn.layout.Divider(),
            pn.Row(self.k_slider, pn.Spacer(width=12), self.l2_input, pn.Spacer(width=12), self.sample_select),
            pn.Row(self.preview_btn, pn.Spacer(width=12), self.calc_btn),
            self.recon_pane,
            pn.layout.Divider(),
            pn.pane.Markdown("### NMF loadings heatmap"),
            self.heatmap_pane,
            pn.Row(self.csv_name, pn.Spacer(width=10), self.csv_download),
            self.status,
            sizing_mode="stretch_width",
            visible=True,   # visible so users can load aligned CSV without running Alignment
        )

        # Wire
        self.aligned_load_btn.on_click(self._on_load_aligned_csv)
        self.preview_btn.on_click(self._on_preview)
        self.calc_btn.on_click(self._on_calculate)
        self.k_slider.param.watch(lambda *_: self._maybe_enable_preview(), "value")
        self.l2_input.param.watch(lambda *_: self._maybe_enable_preview(), "value")
        self.sample_select.param.watch(lambda *_: self._maybe_enable_preview(), "value")
        self.csv_download.callback = self._csv_bytes

    # ---------------- External API ----------------
    def set_input(self, pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool) -> None:
        """
        Accept aligned data (from Alignment or from CSV import).
        """
        self.pseudotimes_df = pseudotimes_df.copy()
        self.norm_df = norm_df.copy()
        self.rows_are_traces = bool(rows_are_traces)

        if rows_are_traces:
            self.samples = list(map(str, self.pseudotimes_df.index.astype(str)))
        else:
            self.samples = list(map(str, self.pseudotimes_df.columns.astype(str)))

        # Prime controls
        self.sample_select.options = self.samples
        self.sample_select.value = self.samples[0] if self.samples else None
        self.preview_btn.disabled = False
        self.calc_btn.disabled = False
        self.status.object = ok(f"NMF input set: {len(self.samples)} samples; ready to preview or calculate.")
        # clear old panes
        self.recon_pane.object = None
        self.heatmap_pane.object = None
        self.csv_download.disabled = True

    # ---------------- Import aligned CSV ----------------
    def _on_load_aligned_csv(self, _=None):
        if not self.aligned_file.value:
            self.aligned_status.object = warn("Pick a CSV file exported by Alignment.")
            return
        try:
            P, Y, samples, t = _parse_wide_aligned_csv(self.aligned_file.filename or "aligned.csv",
                                                       bytes(self.aligned_file.value))
        except Exception as e:
            self.aligned_status.object = warn(f"Failed to parse: {e}")
            return

        # Short preview: time + first sample’s pt & intensity (first 8 rows)
        try:
            show = samples[:1]
            d = {}
            if t is not None:
                d["time"] = t
            for s in show:
                d[f"{s}_pt"] = P[s].to_numpy()
                d[s] = Y[s].to_numpy()
            self.aligned_preview.value = pd.DataFrame(d).head(8)
        except Exception:
            self.aligned_preview.value = pd.DataFrame({"samples": samples})

        self.aligned_status.object = ok(f"Loaded aligned CSV for {len(samples)} samples; priming NMF...")
        # Prime NMF with columns-as-samples data (rows_are_traces=False)
        self.set_input(P, Y, rows_are_traces=False)
        
        # Tell the app we now have aligned data from CSV, so downstream tabs can use it
        if callable(self.on_aligned_imported):
            try:
                self.on_aligned_imported(P, Y, False)  # rows_are_traces=False for wide CSV (columns=samples)
            except Exception:
                pass

    # ---------------- Actions ----------------
    def _maybe_enable_preview(self):
        ok_ready = (self.pseudotimes_df is not None) and (self.norm_df is not None) and (self.sample_select.value is not None)
        self.preview_btn.disabled = not ok_ready

    def _on_preview(self, _=None):
        if self.pseudotimes_df is None or self.norm_df is None or self.sample_select.value is None:
            self.status.object = warn("Load aligned data first.")
            return
        try:
            import CEtools as cet
            # Quick fit for preview
            K = int(self.k_slider.value)
            l2 = float(self.l2_input.value)
            H_df, Phi, centers, pseudo_used_df = cet.fit_continuous_basis_loadings_from_dataframes(
                self.pseudotimes_df, self.norm_df,
                K=K, l2=l2, rows_are_traces=self.rows_are_traces,
            )
            # Plot reconstruction of selected sample
            fig = cet.plot_reconstruction_overlays_bokeh(
                str(self.sample_select.value), H_df, Phi,
                pseudotimes_df=self.pseudotimes_df,
                norm_df=self.norm_df,
                rows_are_traces=self.rows_are_traces,
                title_prefix="Sample"
            )
            # Don't pop new windows; just show in-pane
            try:
                fig.toolbar.active_scroll = None
            except Exception:
                pass
            self.recon_pane.object = fig
            self.status.object = ok("Preview updated.")
        except Exception as e:
            self.recon_pane.object = None
            self.status.object = warn(f"Preview failed: {e}")

    def _on_calculate(self, _=None):
        if self.pseudotimes_df is None or self.norm_df is None:
            self.status.object = warn("Load aligned data first.")
            return
        try:
            import CEtools as cet
            K = int(self.k_slider.value)
            l2 = float(self.l2_input.value)
            H_df, Phi, centers, pseudo_used_df = cet.fit_continuous_basis_loadings_from_dataframes(
                self.pseudotimes_df, self.norm_df,
                K=K, l2=l2, rows_are_traces=self.rows_are_traces,
            )
            self.H_df = H_df
            self.Phi = Phi
            self.centers = centers
            self.pseudo_used_df = pseudo_used_df

            # Heatmap (stay in-pane)
            try:
                # clustered version returns (Zr, row_order, col_order, fig)
                _, _, _, fig = cet.plot_loadings_heatmap_clustered_bokeh(H_df, title="NMF loadings (clustered rows)")
            except Exception:
                # fallback to simple heatmap if clustered fails
                _ = cet.plot_loadings_heatmap_bokeh(H_df)
                fig = bokeh.plotting.gcf()
            try:
                fig.toolbar.active_scroll = None
            except Exception:
                pass
            self.heatmap_pane.object = fig

            self.csv_download.disabled = False
            self.status.object = ok("NMF loadings calculated.")
            # Notify app to unlock Viz/Diversity
            if callable(self.on_done):
                try:
                    self.on_done(self.H_df)
                except Exception:
                    pass
        except Exception as e:
            self.heatmap_pane.object = None
            self.csv_download.disabled = True
            self.status.object = warn(f"NMF failed: {e}")

    # ---------------- Export ----------------
    def _csv_bytes(self):
        if self.H_df is None or self.H_df.empty:
            return io.BytesIO(b"")
        bio = io.BytesIO()
        self.H_df.to_csv(bio)
        bio.seek(0)
        return bio


def build_nmf_section():
    ctrl = NMFController()
    return ctrl.section, ctrl


