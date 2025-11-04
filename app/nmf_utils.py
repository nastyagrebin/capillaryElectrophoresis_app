# ============================
# FILE: app/nmf_utils.py
# ============================
from __future__ import annotations
from typing import Optional, Tuple, List, Callable, Literal

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
    df = pd.read_csv(io.BytesIO(data or b""))
    cols = list(df.columns)
    samples = []
    for c in cols:
        if c == "time" or c.endswith("_pt"):
            continue
        if f"{c}_pt" in cols:
            samples.append(c)
    if not samples:
        raise ValueError("No valid sample columns found (need '{sample}_pt' + '{sample}')")
    P = pd.DataFrame({s: df[f"{s}_pt"].to_numpy() for s in samples}, index=df.index)
    Y = pd.DataFrame({s: df[s].to_numpy() for s in samples}, index=df.index)
    t = df["time"].to_numpy() if "time" in df.columns else None
    return P, Y, samples, t


class NMFController:
    """
    NMF tab controller using CEtools continuous-basis NNLS.

    Public:
      - set_input(pseudotimes_df, norm_df, rows_are_traces=False)   # back-compat
      - set_alignment_input(pseudotimes_df, norm_df, rows_are_traces=False)
      - on_request_alignment: Optional[Callable[[], Optional[Tuple[pd.DataFrame, pd.DataFrame, bool]]]]
      - on_aligned_imported: Optional[Callable[[pd.DataFrame, pd.DataFrame, bool], None]]
      - on_done: Optional[Callable[[pd.DataFrame], None]]
    """
    def __init__(self):
        # Active working data
        self.pseudotimes_df: Optional[pd.DataFrame] = None
        self.norm_df: Optional[pd.DataFrame] = None
        self.rows_are_traces: bool = False
        self.samples: List[str] = []

        # Stored sources
        self._alignment_input: Optional[Tuple[pd.DataFrame, pd.DataFrame, bool, List[str]]] = None
        self._csv_input: Optional[Tuple[pd.DataFrame, pd.DataFrame, bool, List[str]]] = None

        # Results
        self.H_df: Optional[pd.DataFrame] = None
        self.Phi = None
        self.centers = None
        self.pseudo_used_df: Optional[pd.DataFrame] = None

        # Callbacks
        self.on_done: Optional[Callable[[pd.DataFrame], None]] = None
        self.on_aligned_imported: Optional[Callable[[pd.DataFrame, pd.DataFrame, bool], None]] = None
        self.on_request_alignment: Optional[Callable[[], Optional[Tuple[pd.DataFrame, pd.DataFrame, bool]]]] = None

        # Source toggle (dict mapping label->token)
        self.source_select = pn.widgets.RadioButtonGroup(
            name="Source",
            options={"Alignment (current session)": "alignment", "CSV (pseudotimes_wide.csv)": "csv"},
            value="alignment",
            button_type="primary",
        )

        # CSV import
        self.aligned_file = pn.widgets.FileInput(accept=".csv", multiple=False)
        self.aligned_load_btn = pn.widgets.Button(name="Load aligned pseudotimes_wide.csv", button_type="primary")
        self.aligned_status = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.aligned_preview = pn.widgets.Tabulator(pd.DataFrame(), height=180, show_index=False)

        # Controls
        self.k_slider = pn.widgets.IntSlider(name="K (number of basis)", start=20, end=500, value=250, step=5, width=260)
        self.l2_input = pn.widgets.FloatInput(name="L2 (ridge)", value=1e-5, step=1e-5, start=0.0, width=160)
        self.sample_select = pn.widgets.Select(name="Sample for preview", options=[], value=None, width=260)
        self.preview_btn = pn.widgets.Button(name="Preview reconstruction", button_type="success", disabled=True)
        self.calc_btn = pn.widgets.Button(name="Calculate NMF Loadings", button_type="success", disabled=True)
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Plots
        self.recon_pane = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")
        self.heatmap_pane = pn.pane.Bokeh(height=520, sizing_mode="stretch_width")

        # Export
        self.csv_name = pn.widgets.TextInput(name="Loadings CSV filename", value="nmf_loadings.csv", width=260)
        self.csv_download = pn.widgets.FileDownload(
            label="Download loadings CSV", filename=self.csv_name.value,
            button_type="primary", embed=False, auto=False,
            callback=lambda: io.BytesIO(b""),
            disabled=True
        )
        self.csv_name.param.watch(lambda e: setattr(self.csv_download, "filename", e.new or "nmf_loadings.csv"), "value")

        # Layout
        self._csv_row = pn.Row(self.aligned_file, pn.Spacer(width=8), self.aligned_load_btn)
        self.section = pn.Column(
            pn.pane.Markdown("## 4) NMF"),
            pn.pane.Markdown("_Choose your NMF input: **Alignment (current session)** or **CSV import**._", styles={"color": "#555"}),
            self.source_select,
            self._csv_row,
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
            visible=True,
        )

        # Wire
        self.source_select.param.watch(self._on_source_changed, "value")
        self.aligned_load_btn.on_click(self._on_load_aligned_csv)
        self.preview_btn.on_click(self._on_preview)
        self.calc_btn.on_click(self._on_calculate)
        self.k_slider.param.watch(lambda *_: self._maybe_enable_preview(), "value")
        self.l2_input.param.watch(lambda *_: self._maybe_enable_preview(), "value")
        self.sample_select.param.watch(lambda *_: self._maybe_enable_preview(), "value")
        self.csv_download.callback = self._csv_bytes

        # Default UI
        self._apply_csv_visibility(show=False)

    # -------- External API --------
    def set_input(self, pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool) -> None:
        current = self._normalize_source_token(self.source_select.value)
        if current == "alignment":
            self.set_alignment_input(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
        else:
            self._set_source_input("csv", pseudotimes_df, norm_df, rows_are_traces)
            self._apply_source("csv")

    def set_alignment_input(self, pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool) -> None:
        self._set_source_input("alignment", pseudotimes_df, norm_df, rows_are_traces)
        if self._normalize_source_token(self.source_select.value) != "alignment":
            self.source_select.value = "alignment"
        self._apply_source("alignment")

    # -------- CSV import --------
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

        # Small preview
        try:
            show = samples[:1]
            d = {}
            if t is not None:
                d["time"] = t
            for s in show:
                d[f"{s}_pt"] = P[s].to_numpy()
            for s in show:
                d[s] = Y[s].to_numpy()
            self.aligned_preview.value = pd.DataFrame(d).head(8)
        except Exception:
            self.aligned_preview.value = pd.DataFrame({"samples": samples})

        self.aligned_status.object = ok(f"Loaded aligned CSV for {len(samples)} samples; priming NMF...")
        self._set_source_input("csv", P, Y, rows_are_traces=False)
        if self._normalize_source_token(self.source_select.value) != "csv":
            self.source_select.value = "csv"
        self._apply_source("csv")

        if callable(self.on_aligned_imported):
            try:
                self.on_aligned_imported(P, Y, False)
            except Exception:
                pass

    # -------- Internals --------
    @staticmethod
    def _normalize_source_token(val) -> Literal["alignment", "csv"]:
        if val in ("alignment", "csv"):
            return val
        s = str(val).lower()
        return "csv" if "csv" in s else "alignment"

    def _set_source_input(self, kind: Literal["alignment", "csv"], P: pd.DataFrame, Y: pd.DataFrame, rows_are_traces: bool):
        samples = list(map(str, (P.index if rows_are_traces else P.columns).astype(str)))
        blob = (P.copy(), Y.copy(), bool(rows_are_traces), samples)
        if kind == "alignment":
            self._alignment_input = blob
        else:
            self._csv_input = blob

    def _apply_source(self, kind: Literal["alignment", "csv"]) -> None:
        tpl = self._alignment_input if kind == "alignment" else self._csv_input

        # NEW: pull-based fallback if Alignment slot is empty
        if tpl is None and kind == "alignment" and callable(self.on_request_alignment):
            try:
                pulled = self.on_request_alignment()
            except Exception:
                pulled = None
            if pulled:
                P, Y, rows_are_traces = pulled
                self._set_source_input("alignment", P, Y, rows_are_traces)
                tpl = self._alignment_input

        if tpl is None:
            self._clear_active()
            which = "Alignment" if kind == "alignment" else "CSV"
            self.status.object = warn(f"Selected source **{which}** has no data yet.")
            return

        P, Y, rows_are_traces, samples = tpl
        self.pseudotimes_df = P.copy()
        self.norm_df = Y.copy()
        self.rows_are_traces = bool(rows_are_traces)
        self.samples = list(samples)

        self.sample_select.options = self.samples
        self.sample_select.value = self.samples[0] if self.samples else None
        has = bool(self.samples)
        self.preview_btn.disabled = not has
        self.calc_btn.disabled = not has
        self.status.object = ok(f"NMF input set from **{ 'Alignment' if kind=='alignment' else 'CSV' }**: {len(self.samples)} samples.")
        self.recon_pane.object = None
        self.heatmap_pane.object = None
        self.csv_download.disabled = True

    def _clear_active(self):
        self.pseudotimes_df = None
        self.norm_df = None
        self.rows_are_traces = False
        self.samples = []
        self.sample_select.options = []
        self.sample_select.value = None
        self.preview_btn.disabled = True
        self.calc_btn.disabled = True
        self.recon_pane.object = None
        self.heatmap_pane.object = None
        self.csv_download.disabled = True

    def _apply_csv_visibility(self, show: bool) -> None:
        self._csv_row.visible = show
        self.aligned_preview.visible = show
        self.aligned_status.visible = show

    def _on_source_changed(self, *_):
        kind = self._normalize_source_token(self.source_select.value)
        self._apply_csv_visibility(show=(kind == "csv"))
        self._apply_source(kind)
        self._maybe_enable_preview()

    # -------- Actions --------
    def _maybe_enable_preview(self):
        ok_ready = (self.pseudotimes_df is not None) and (self.norm_df is not None) and (self.sample_select.value is not None)
        self.preview_btn.disabled = not ok_ready

    def _on_preview(self, _=None):
        if self.pseudotimes_df is None or self.norm_df is None or self.sample_select.value is None:
            self.status.object = warn("Provide NMF input first (Alignment or CSV).")
            return
        try:
            import CEtools as cet
            K = int(self.k_slider.value); l2 = float(self.l2_input.value)
            H_df, Phi, centers, pseudo_used_df = cet.fit_continuous_basis_loadings_from_dataframes(
                self.pseudotimes_df, self.norm_df, K=K, l2=l2, rows_are_traces=self.rows_are_traces,
            )
            fig = cet.plot_reconstruction_overlays_bokeh(
                str(self.sample_select.value), H_df, Phi,
                pseudotimes_df=self.pseudotimes_df, norm_df=self.norm_df,
                rows_are_traces=self.rows_are_traces, title_prefix="Sample"
            )
            try: fig.toolbar.active_scroll = None
            except Exception: pass
            self.recon_pane.object = fig
            self.status.object = ok("Preview updated.")
        except Exception as e:
            self.recon_pane.object = None
            self.status.object = warn(f"Preview failed: {e}")

    def _on_calculate(self, _=None):
        if self.pseudotimes_df is None or self.norm_df is None:
            self.status.object = warn("Provide NMF input first (Alignment or CSV).")
            return
        try:
            import CEtools as cet
            K = int(self.k_slider.value); l2 = float(self.l2_input.value)
            H_df, Phi, centers, pseudo_used_df = cet.fit_continuous_basis_loadings_from_dataframes(
                self.pseudotimes_df, self.norm_df, K=K, l2=l2, rows_are_traces=self.rows_are_traces,
            )
            self.H_df, self.Phi, self.centers, self.pseudo_used_df = H_df, Phi, centers, pseudo_used_df

            try:
                _, _, _, fig = cet.plot_loadings_heatmap_clustered_bokeh(H_df, title="NMF loadings (clustered rows)")
            except Exception:
                _ = cet.plot_loadings_heatmap_bokeh(H_df); fig = bokeh.plotting.gcf()
            try: fig.toolbar.active_scroll = None
            except Exception: pass
            self.heatmap_pane.object = fig

            self.csv_download.disabled = False
            self.status.object = ok("NMF loadings calculated.")
            if callable(self.on_done):
                try: self.on_done(self.H_df)
                except Exception: pass
        except Exception as e:
            self.heatmap_pane.object = None
            self.csv_download.disabled = True
            self.status.object = warn(f"NMF failed: {e}")

    def _csv_bytes(self):
        if self.H_df is None or self.H_df.empty:
            return io.BytesIO(b"")
        bio = io.BytesIO(); self.H_df.to_csv(bio); bio.seek(0); return bio


def build_nmf_section():
    ctrl = NMFController()
    return ctrl.section, ctrl




