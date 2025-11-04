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

    Public:
      - set_input(pseudotimes_df, norm_df, rows_are_traces=False)  # back-compat
      - set_alignment_input(pseudotimes_df, norm_df, rows_are_traces=False)
      - on_done: Optional[Callable[[pd.DataFrame], None]]
      - on_aligned_imported: Optional[Callable[[pd.DataFrame, pd.DataFrame, bool], None]]
    """
    def __init__(self):
        # Active working data (drives preview/calc)
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

        # Callbacks into app
        self.on_done: Optional[Callable[[pd.DataFrame], None]] = None
        self.on_aligned_imported: Optional[Callable[[pd.DataFrame, pd.DataFrame, bool], None]] = None

        # ---------------- Source toggle ----------------
        self.source_select = pn.widgets.RadioButtonGroup(
            name="Source",
            value="alignment",
            options=[("Alignment (current session)", "alignment"), ("CSV (pseudotimes_wide.csv)", "csv")],
            button_type="primary"
        )

        # ---------------- Import aligned CSV ----------------
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

        # Layout
        self._csv_row = pn.Row(self.aligned_file, pn.Spacer(width=8), self.aligned_load_btn)
        self.section = pn.Column(
            pn.pane.Markdown("## 4) NMF"),
            pn.pane.Markdown("_Choose your NMF input: **Alignment (current session)** or **CSV import**._", styles={"color":"#555"}),
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
        self.source_select.param.watch(lambda *_: self._on_source_changed(), "value")
        self.aligned_load_btn.on_click(self._on_load_aligned_csv)
        self.preview_btn.on_click(self._on_preview)
        self.calc_btn.on_click(self._on_calculate)
        self.k_slider.param.watch(lambda *_: self._maybe_enable_preview(), "value")
        self.l2_input.param.watch(lambda *_: self._maybe_enable_preview(), "value")
        self.sample_select.param.watch(lambda *_: self._maybe_enable_preview(), "value")
        self.csv_download.callback = self._csv_bytes

        # Default UI state
        self._apply_csv_visibility(show=False)

    # ---------------- External API ----------------
    def set_input(self, pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool) -> None:
        """
        Back-compat: set whichever source is currently selected.
        """
        if (self.source_select.value or "alignment") == "alignment":
            self.set_alignment_input(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
        else:
            # rarely used; CSV usually comes via loader
            self._set_source_input("csv", pseudotimes_df, norm_df, rows_are_traces)

    def set_alignment_input(self, pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool) -> None:
        self._set_source_input("alignment", pseudotimes_df, norm_df, rows_are_traces)
        if self.source_select.value != "alignment":
            # Users arriving from Alignment should see Alignment by default.
            self.source_select.value = "alignment"
        else:
            self._apply_source("alignment")

    # ---------------- CSV import path ----------------
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

        # Short preview
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
        # Store as CSV source, columns-as-samples
        self._set_source_input("csv", P, Y, rows_are_traces=False)
        if self.source_select.value == "csv":
            self._apply_source("csv")

        # Notify app so downstream tabs can also use it
        if callable(self.on_aligned_imported):
            try:
                self.on_aligned_imported(P, Y, False)
            except Exception:
                pass

    # ---------------- Internal helpers ----------------
    def _set_source_input(self, kind: Literal["alignment","csv"], P: pd.DataFrame, Y: pd.DataFrame, rows_are_traces: bool):
        if rows_are_traces:
            samples = list(map(str, P.index.astype(str)))
        else:
            samples = list(map(str, P.columns.astype(str)))
        if kind == "alignment":
            self._alignment_input = (P.copy(), Y.copy(), bool(rows_are_traces), samples)
        else:
            self._csv_input = (P.copy(), Y.copy(), bool(rows_are_traces), samples)

    def _apply_source(self, kind: Literal["alignment","csv"]) -> None:
        tpl = self._alignment_input if kind == "alignment" else self._csv_input
        if tpl is None:
            self._clear_active()
            self.status.object = warn("Selected source has no data yet.")
            return
        P, Y, rows_are_traces, samples = tpl
        self.pseudotimes_df = P.copy()
        self.norm_df = Y.copy()
        self.rows_are_traces = bool(rows_are_traces)
        self.samples = list(samples)

        # Prime controls
        self.sample_select.options = self.samples
        self.sample_select.value = self.samples[0] if self.samples else None
        self.preview_btn.disabled = not bool(self.samples)
        self.calc_btn.disabled = not bool(self.samples)
        self.status.object = ok(f"NMF input set from **{ 'Alignment' if kind=='alignment' else 'CSV' }**: {len(self.samples)} samples.")
        # Clear panes & export
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

    def _on_source_changed(self):
        is_csv = self.source_select.value == "csv"
        self._apply_csv_visibility(show=is_csv)
        self._apply_source("csv" if is_csv else "alignment")
        self._maybe_enable_preview()

    # ---------------- Actions ----------------
    def _maybe_enable_preview(self):
        ok_ready = (self.pseudotimes_df is not None) and (self.norm_df is not None) and (self.sample_select.value is not None)
        self.preview_btn.disabled = not ok_ready

    def _on_preview(self, _=None):
        if self.pseudotimes_df is None or self.norm_df is None or self.sample_select.value is None:
            self.status.object = warn("Provide NMF input first (Alignment or CSV).")
            return
        try:
            import CEtools as cet
            K = int(self.k_slider.value)
            l2 = float(self.l2_input.value)
            H_df, Phi, centers, pseudo_used_df = cet.fit_continuous_basis_loadings_from_dataframes(
                self.pseudotimes_df, self.norm_df,
                K=K, l2=l2, rows_are_traces=self.rows_are_traces,
            )
            fig = cet.plot_reconstruction_overlays_bokeh(
                str(self.sample_select.value), H_df, Phi,
                pseudotimes_df=self.pseudotimes_df,
                norm_df=self.norm_df,
                rows_are_traces=self.rows_are_traces,
                title_prefix="Sample"
            )
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
            self.status.object = warn("Provide NMF input first (Alignment or CSV).")
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

            try:
                _, _, _, fig = cet.plot_loadings_heatmap_clustered_bokeh(H_df, title="NMF loadings (clustered rows)")
            except Exception:
                _ = cet.plot_loadings_heatmap_bokeh(H_df)
                fig = bokeh.plotting.gcf()
            try:
                fig.toolbar.active_scroll = None
            except Exception:
                pass
            self.heatmap_pane.object = fig

            self.csv_download.disabled = False
            self.status.object = ok("NMF loadings calculated.")
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


# ============================
# FILE: app.py
# ============================
from __future__ import annotations
import io, zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting

# Local utils
from upload_utils import (
    sanitize_name, convert_cdf_bytes_to_df, try_merge_same_time,
)
from common_plot import make_preview_plot

# Preprocess controllers/sections
from preprocess_despike_utils import build_despike_section
from preprocess_smooth_utils import build_smoothing_section
from preprocess_norm_utils import build_normalization_section, NormalizationController

try:
    from preprocess_baseline_utils import build_baseline_section, BaselineController
except Exception:
    from app.preprocess_baseline_utils import build_baseline_section, BaselineController

# Alignment + NMF + Viz + Diversity
from alignment_utils import build_alignment_section
from nmf_utils import build_nmf_section
from viz_utils import build_viz_section
from diversity_utils import build_diversity_section

OK = "OK:"; WARN = "Warning:"
def ok(m): return f"{OK} {m}"
def warn(m): return f"{WARN} {m}"

def _ensure_panel():
    if not getattr(pn.state, "_ce_panel_ready", False):
        pn.extension('tabulator')
        pn.state._ce_panel_ready = True
_ensure_panel()

# ---------- Session state ----------
class SessionState:
    def __init__(self):
        self.uploads: List[Tuple[str, bytes]] = []
        self.sample_names: Dict[str, str] = {}
        self.converted_by_sample: Dict[str, pd.DataFrame] = {}
        self.merged_df: Optional[pd.DataFrame] = None
        self.last_fig: Optional[bokeh.plotting.Figure] = None
        self.current_by_sample: Dict[str, pd.DataFrame] = {}
        self.aligned_pseudotimes_df: Optional[pd.DataFrame] = None
        self.aligned_norm_df: Optional[pd.DataFrame] = None
        self.rows_are_traces_aligned: bool = False
        self.H_df: Optional[pd.DataFrame] = None

state = SessionState()

# ---------- Upload tab ----------
upload = pn.widgets.FileInput(accept=".cdf,.CDF", multiple=True)
prefer_minutes = pn.widgets.Checkbox(name="Convert time to minutes (from seconds)", value=False)
upload_status = pn.pane.Markdown(
    "Upload one or more `.cdf` files; convert to CSV using scalar timing.",
    sizing_mode="stretch_width"
)
convert_btn = pn.widgets.Button(name="Convert to CSV", button_type="primary", disabled=True)

# Renamer
rename_status = pn.pane.Markdown("", sizing_mode="stretch_width")
rename_box = pn.Column(sizing_mode="stretch_both")

def _unique_names(names: List[str]) -> Tuple[bool, str]:
    lower = [n.lower() for n in names]
    return (True, "OK") if len(set(lower)) == len(lower) else (False, "Duplicate names detected.")

def _build_rename_panel():
    rename_box.objects = []
    if not state.uploads:
        rename_status.object = ""
        return
    if not state.sample_names:
        state.sample_names = {orig: sanitize_name(orig) for orig, _ in state.uploads}
    for orig, _ in state.uploads:
        left = pn.pane.Markdown(f"**Original**: `{Path(orig).name}`", width=380)
        edit = pn.widgets.TextInput(name="", value=state.sample_names[orig], width=260)
        def _on_change(event, orig=orig, edit=edit):
            val = sanitize_name(event.new)
            edit.value = val
            state.sample_names[orig] = val
            names = [state.sample_names[o] for o, _ in state.uploads]
            ok_flag, msg = _unique_names(names)
            rename_status.object = ok("Names are valid.") if ok_flag else warn(msg)
            convert_btn.disabled = not ok_flag
        edit.param.watch(_on_change, "value")
        rename_box.append(pn.Row(left, edit))
    names = [state.sample_names[o] for o, _ in state.uploads]
    ok_flag, msg = _unique_names(names)
    rename_status.object = ok("Names are valid.") if ok_flag else warn(msg)
    convert_btn.disabled = not ok_flag

# Downloads (per-sample ZIP, merged CSV)
zip_name = pn.widgets.TextInput(name="ZIP filename (per-sample CSVs)", value="converted_csvs.zip")
zip_download = pn.widgets.FileDownload(
    label="Download ZIP", filename=zip_name.value,
    button_type="primary", embed=False, auto=False,
    callback=lambda: io.BytesIO(b""), disabled=True
)
zip_name.param.watch(lambda e: setattr(zip_download, "filename", e.new or "converted_csvs.zip"), "value")

merge_status = pn.pane.Markdown("", sizing_mode="stretch_width")
merge_name = pn.widgets.TextInput(name="Merged CSV filename", value="merged.csv")
merge_download = pn.widgets.FileDownload(
    label="Download Merged CSV", filename=merge_name.value,
    button_type="primary", embed=False, auto=False,
    callback=lambda: io.BytesIO(b""), disabled=True
)
merge_name.param.watch(lambda e: setattr(merge_download, "filename", e.new or "merged.csv"), "value")

# Preview plot + slider
plot_pane = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")
offset_slider = pn.widgets.FloatSlider(
    name="Vertical offset", start=0.0, end=10.0, step=0.5, value=0.0,
    sizing_mode="stretch_width"
)

# Grouped sections (hidden until conversion)
downloads_group = pn.Column(
    pn.pane.Markdown("### Downloads"),
    pn.Row(zip_name, zip_download),
    pn.Row(merge_name, merge_download),
    merge_status,
    visible=False,
    sizing_mode="stretch_width",
)
preview_group = pn.Column(
    pn.pane.Markdown("### Chromatograph(s) preview"),
    plot_pane,
    offset_slider,
    visible=False,
    sizing_mode="stretch_width",
)

def _on_upload_change(event):
    files = []
    if isinstance(upload.value, (bytes, bytearray)):
        if upload.value:
            files = [(upload.filename or "uploaded.cdf", bytes(upload.value))]
    elif isinstance(upload.value, list):
        names = upload.filename if isinstance(upload.filename, list) else [upload.filename] * len(upload.value)
        for nm, by in zip(names, upload.value):
            if by:
                files.append((nm or "uploaded.cdf", bytes(by)))
    else:
        val = getattr(upload, "value", None)
        if isinstance(val, dict):
            files = list(val.items())

    state.uploads = files
    state.converted_by_sample.clear()
    state.merged_df = None
    state.last_fig = None
    state.current_by_sample = {}

    # reset UI
    zip_download.disabled = True
    merge_download.disabled = True
    merge_status.object = ""
    plot_pane.object = None
    downloads_group.visible = False
    preview_group.visible = False

    try:
        alignment_section.visible = True
    except Exception:
        pass

    if not files:
        upload_status.object = warn("No files selected.")
        state.sample_names.clear()
        rename_box.objects = []
        rename_status.object = ""
        convert_btn.disabled = True
        return

    state.sample_names = {orig: sanitize_name(orig) for orig, _ in state.uploads}
    _build_rename_panel()
    upload_status.object = ok(f"Queued {len(files)} CDF file(s). Edit names on the right, then click 'Convert to CSV'.")

upload.param.watch(_on_upload_change, "value")

def _render_plot(*_):
    if not state.converted_by_sample:
        plot_pane.object = None
        state.last_fig = None
        return
    fig = make_preview_plot(
        state.converted_by_sample,
        minutes=prefer_minutes.value,
        offset=offset_slider.value,
        title="Chromatograph(s) preview"
    )
    plot_pane.object = fig
    state.last_fig = fig

offset_slider.param.watch(_render_plot, "value")
prefer_minutes.param.watch(_render_plot, "value")

def _on_convert_click(event):
    if not state.uploads:
        upload_status.object = warn("Upload CDF files first.")
        return
    names = [state.sample_names.get(orig, sanitize_name(orig)) for orig, _ in state.uploads]
    ok_flag, msg = _unique_names(names)
    if not ok_flag:
        upload_status.object = warn(msg)
        convert_btn.disabled = True
        return

    errors = []
    state.converted_by_sample.clear()
    state.merged_df = None
    state.last_fig = None
    state.current_by_sample = {}

    merge_download.disabled = True
    merge_status.object = ""
    plot_pane.object = None
    downloads_group.visible = False
    preview_group.visible = False

    for orig, by in state.uploads:
        try:
            df = convert_cdf_bytes_to_df(orig, by, prefer_minutes=prefer_minutes.value)
            sample = state.sample_names.get(orig, sanitize_name(orig))
            state.converted_by_sample[sample] = df
        except Exception as e:
            errors.append(f"{Path(orig).name}: {e}")

    if state.converted_by_sample:
        state.current_by_sample = {k: v.copy() for k, v in state.converted_by_sample.items()}
        _render_plot()

        _init_preprocess_from_current()
        _unlock_despike()

        downloads_group.visible = True
        preview_group.visible = True

        def _zip_bytes():
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for sample, df in state.converted_by_sample.items():
                    zf.writestr(f"{sample}.csv", df.to_csv(index=False))
            bio.seek(0); return bio
        zip_download.callback = _zip_bytes
        zip_download.disabled = False

        merged, reason = try_merge_same_time(state.converted_by_sample)
        if merged is not None:
            state.merged_df = merged
            def _merged_bytes():
                bio = io.BytesIO()
                state.merged_df.to_csv(bio, index=False)
                bio.seek(0); return bio
            merge_download.callback = _merged_bytes
            merge_download.disabled = False
            merge_status.object = ok("Time stamps match across all files. You can download the merged CSV.")
        else:
            merge_download.disabled = True
            merge_status.object = warn(f"Cannot merge into one CSV: {reason}")

        msg = ok(f"Converted {len(state.converted_by_sample)} file(s) to CSV.")
        if errors:
            msg += f" {WARN} {len(errors)} failed: " + "; ".join(errors[:3]) + (" ..." if len(errors) > 3 else "")
        upload_status.object = msg

        try:
            TABS.active = 1  # jump to Preprocess
        except Exception:
            pass
    else:
        upload_status.object = warn("Conversion failed.")

convert_btn.on_click(_on_convert_click)

# Upload layout
left_col = pn.Column(
    pn.pane.Markdown("## 1) Upload & Convert (.cdf → .csv, scalar timing)"),
    pn.Row(upload, pn.Spacer(width=12), prefer_minutes),
    pn.Row(convert_btn),
    pn.Spacer(height=8),
    upload_status,
    pn.layout.Divider(),
    downloads_group,
    pn.layout.Divider(),
    preview_group,
)
right_col = pn.Column(
    pn.pane.Markdown("### Sample Renamer"),
    pn.pane.Markdown("Edit each sample name (must be unique)."),
    rename_status,
    pn.layout.Divider(),
    rename_box,
    sizing_mode="stretch_both",
    width=520,
)
upload_tab = pn.Column(
    pn.Row(left_col, pn.layout.HSpacer(width=16), right_col, sizing_mode="stretch_width"),
    sizing_mode="stretch_width"
)

# ---------- Preprocess tab ----------
despike_section, despike_ctrl = build_despike_section()
smooth_section,  smooth_ctrl  = build_smoothing_section()
baseline_section, baseline_ctrl = build_baseline_section()
norm_section,    norm_ctrl    = build_normalization_section()

smooth_section.visible = False
baseline_section.visible = False
norm_section.visible = False

# ---------- Alignment tab ----------
alignment_section, alignment_ctrl = build_alignment_section()
alignment_section.visible = True

# ---------- NMF tab ----------
nmf_section, nmf_ctrl = build_nmf_section()  # includes CSV import & source toggle
nmf_section.visible = True

def _nmf_aligned_imported(P: pd.DataFrame, Y: pd.DataFrame, rows_are_traces: bool):
    state.aligned_pseudotimes_df = P.copy()
    state.aligned_norm_df = Y.copy()
    state.rows_are_traces_aligned = bool(rows_are_traces)
nmf_ctrl.on_aligned_imported = _nmf_aligned_imported

# ---------- Viz tab ----------
viz_section, viz_ctrl = build_viz_section()
viz_section.visible = True

# ---------- Diversity tab ----------
diversity_section, diversity_ctrl = build_diversity_section()
diversity_section.visible = True

# ===================== UNLOCK HELPERS =====================

def _unlock_despike():
    if not state.converted_by_sample:
        return
    despike_ctrl.input_by_sample = {k: v.copy() for k, v in state.converted_by_sample.items()}
    try:
        despike_section.visible = True
        despike_ctrl.preview_btn.disabled = False
        despike_ctrl.skip_btn.disabled = False
        despike_ctrl.apply_btn.disabled = True
        despike_ctrl.before_pane.object = None
        despike_ctrl.after_pane.object = None
        despike_ctrl.export_btn.disabled = True
        despike_ctrl.export_status.object = ""
        despike_ctrl.status.object = "Set parameters, then click 'Show despiking preview' to refresh."
    except Exception:
        pass

def _unlock_smoothing():
    if not state.current_by_sample:
        return
    smooth_ctrl.input_by_sample = {k: v.copy() for k, v in state.current_by_sample.items()}
    smooth_section.visible = True
    try:
        smooth_ctrl.preview_btn.disabled = False
        smooth_ctrl.skip_btn.disabled = False
        smooth_ctrl.apply_btn.disabled = True
        smooth_ctrl.before_pane.object = None
        smooth_ctrl.after_pane.object = None
        smooth_ctrl.export_btn.disabled = True
        smooth_ctrl.export_status.object = ""
        smooth_ctrl.status.object = "Set parameters, then click 'Show smoothing preview' to refresh."
    except Exception:
        pass

def _unlock_baseline():
    if not state.current_by_sample:
        return
    baseline_ctrl.input_by_sample = {k: v.copy() for k, v in state.current_by_sample.items()}
    baseline_section.visible = True
    try:
        baseline_ctrl.preview_btn.disabled = False
        baseline_ctrl.skip_btn.disabled = False
        baseline_ctrl.apply_btn.disabled = True
        baseline_ctrl.before_pane.object = None
        baseline_ctrl.after_pane.object = None
        baseline_ctrl.export_btn.disabled = True
        baseline_ctrl.export_status.object = ""
        baseline_ctrl.status.object = "Set parameters, then click 'Show baseline subtraction preview' to refresh."
    except Exception:
        pass

def _unlock_normalization():
    if not state.current_by_sample:
        return
    norm_ctrl.current_by_sample = {k: v.copy() for k, v in state.current_by_sample.items()}
    norm_section.visible = True
    try:
        norm_ctrl._render_before_fig()
        norm_ctrl.apply_btn.disabled = False
        norm_ctrl.skip_btn.disabled = False
    except Exception:
        pass

def _set_working_dataset(d: Dict[str, pd.DataFrame]) -> None:
    state.current_by_sample = {k: v.copy() for k, v in d.items()}

def _unlock_alignment():
    if not state.current_by_sample:
        return
    alignment_ctrl.set_input(state.current_by_sample)
    alignment_section.visible = True

def _force_unlock_nmf_controls():
    for attr in ("import_btn", "preview_btn", "calc_btn", "apply_btn", "run_btn"):
        btn = getattr(nmf_ctrl, attr, None)
        try:
            if btn is not None:
                btn.disabled = False
        except Exception:
            pass
    for attr in ("status", "info", "message"):
        pane = getattr(nmf_ctrl, attr, None)
        try:
            if pane is not None and hasattr(pane, "object"):
                pane.object = ok("Aligned data connected from Alignment tab. NMF is ready.")
        except Exception:
            pass

def _unlock_nmf_from_alignment(pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool):
    # Persist
    state.aligned_pseudotimes_df = pseudotimes_df.copy()
    state.aligned_norm_df = norm_df.copy()
    state.rows_are_traces_aligned = bool(rows_are_traces)

    # Feed into NMF, select Alignment as source
    try:
        # New API: stash as "alignment" and activate
        nmf_ctrl.set_alignment_input(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
        try:
            nmf_ctrl.source_select.value = "alignment"
        except Exception:
            pass
    except Exception:
        # Back-compat: older controllers
        try:
            nmf_ctrl.set_input(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
        except Exception:
            pass

    _force_unlock_nmf_controls()
    try:
        nmf_section.visible = True
        # Optionally: TABS.active = 3
    except Exception:
        pass

def _unlock_viz_diversity_after_nmf(H_df: pd.DataFrame):
    state.H_df = H_df.copy()
    try:
        viz_ctrl.set_input(H_df)
        viz_section.visible = True
    except Exception:
        pass
    try:
        if state.aligned_pseudotimes_df is not None and state.aligned_norm_df is not None:
            diversity_ctrl.set_input(
                state.aligned_pseudotimes_df,
                state.aligned_norm_df,
                rows_are_traces=state.rows_are_traces_aligned
            )
        diversity_ctrl.set_H(H_df)
        diversity_section.visible = True
    except Exception:
        pass

# ===================== WIRING: APPLY/SKIP ADVANCE =====================

def _wire_despike_apply():
    def _apply(_=None):
        out = getattr(despike_ctrl, "output_by_sample", None)
        if not out:
            despike_ctrl.status.object = warn("Generate a despiking preview first.")
            return
        _set_working_dataset(out)
        despike_ctrl.status.object = ok("Despiked data applied.")
        _unlock_smoothing()
    despike_ctrl.apply_btn.on_click(_apply)

def _wire_despike_skip():
    def _skip(_=None):
        inp = getattr(despike_ctrl, "input_by_sample", None)
        if not inp:
            despike_ctrl.status.object = warn("No input data to skip.")
            return
        _set_working_dataset(inp)
        despike_ctrl.status.object = ok("Skipped despiking. Using input data.")
        _unlock_smoothing()
    despike_ctrl.skip_btn.on_click(_skip)

def _wire_smooth_apply():
    def _apply(_=None):
        out = getattr(smooth_ctrl, "output_by_sample", None) or getattr(smooth_ctrl, "smoothed_by_sample", None)
        if not out:
            try:
                smooth_ctrl._on_preview()
            except Exception:
                pass
            out = getattr(smooth_ctrl, "output_by_sample", None) or getattr(smooth_ctrl, "smoothed_by_sample", None)
        if not out:
            smooth_ctrl.status.object = warn("Please click 'Show smoothing preview' first.")
            return
        _set_working_dataset(out)
        smooth_ctrl.status.object = ok("Smoothed data applied.")
        _unlock_baseline()
    smooth_ctrl.apply_btn.on_click(_apply)

def _wire_smooth_skip():
    def _skip(_=None):
        inp = getattr(smooth_ctrl, "input_by_sample", None)
        if not inp:
            smooth_ctrl.status.object = warn("No input data to skip.")
            return
        _set_working_dataset(inp)
        smooth_ctrl.status.object = ok("Skipped smoothing. Using input data.")
        _unlock_baseline()
    smooth_ctrl.skip_btn.on_click(_skip)

def _wire_baseline_apply():
    def _apply(_=None):
        out = getattr(baseline_ctrl, "output_by_sample", None)
        if not out:
            baseline_ctrl.status.object = warn("Generate a baseline preview first.")
            return
        _set_working_dataset(out)
        baseline_ctrl.status.object = ok("Baseline-subtracted data applied.")
        _unlock_normalization()
    baseline_ctrl.apply_btn.on_click(_apply)

def _wire_baseline_skip():
    def _skip(_=None):
        inp = getattr(baseline_ctrl, "input_by_sample", None)
        if not inp:
            baseline_ctrl.status.object = warn("No input data to skip.")
            return
        _set_working_dataset(inp)
        baseline_ctrl.status.object = ok("Skipped baseline subtraction. Using input data.")
        _unlock_normalization()
    baseline_ctrl.skip_btn.on_click(_skip)

def _init_preprocess_from_current():
    if not state.current_by_sample:
        return
    despike_ctrl.input_by_sample = {k: v.copy() for k, v in state.current_by_sample.items()}
    try:
        despike_ctrl.preview_btn.disabled = False
        despike_ctrl.skip_btn.disabled = False
        despike_ctrl.apply_btn.disabled = True
        despike_ctrl.before_pane.object = None
        despike_ctrl.after_pane.object = None
        despike_ctrl.export_btn.disabled = True
        despike_ctrl.export_status.object = ""
        despike_ctrl.status.object = ok("Loaded data into Preprocessing → Despiking.")
    except Exception:
        pass
    smooth_section.visible = False
    baseline_section.visible = False
    norm_section.visible = False
    alignment_section.visible = True

_wire_despike_apply()
_wire_despike_skip()
_wire_smooth_apply()
_wire_smooth_skip()
_wire_baseline_apply()
_wire_baseline_skip()

def _wire_norm_apply_skip():
    def _apply(_=None):
        out = getattr(norm_ctrl, "normalized_by_sample", None) or getattr(norm_ctrl, "output_by_sample", None)
        if not out:
            norm_ctrl.status.object = warn("Generate a normalization preview first.")
            return
        state.current_by_sample = {k: v.copy() for k, v in out.items()}
        norm_ctrl.status.object = ok("Normalized data applied. Later tabs will use normalized traces.")
        alignment_ctrl.set_input(state.current_by_sample)
        alignment_section.visible = True
    norm_ctrl.apply_btn.on_click(_apply)

    def _skip(_=None):
        norm_ctrl.status.object = ok("Skipped normalization. Current dataset unchanged.")
        if not state.current_by_sample:
            state.current_by_sample = {k: v.copy() for k, v in state.converted_by_sample.items()}
        alignment_ctrl.set_input(state.current_by_sample)
        alignment_section.visible = True
    norm_ctrl.skip_btn.on_click(_skip)

_wire_norm_apply_skip()

# ---------- Bridge Alignment → NMF ----------
try:
    def _alignment_done_callback(pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool):
        _unlock_nmf_from_alignment(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
    alignment_ctrl.on_aligned = _alignment_done_callback
except Exception:
    pass

# ---------- Bridge NMF → Viz/Diversity ----------
def _nmf_done_callback(H_df: pd.DataFrame):
    _unlock_viz_diversity_after_nmf(H_df)

try:
    nmf_ctrl.on_done = _nmf_done_callback
except Exception:
    pass

# ---------- Tabs ----------
preproc_tab = pn.Column(
    pn.pane.Markdown("## 2) Preprocessing"),
    despike_section,
    pn.layout.Divider(),
    smooth_section,
    pn.layout.Divider(),
    baseline_section,
    pn.layout.Divider(),
    norm_section,
    sizing_mode="stretch_width",
)

TABS = pn.Tabs(
    ("Upload", upload_tab),
    ("Preprocess", preproc_tab),
    ("Alignment", alignment_section),
    ("NMF", nmf_section),
    ("Diversity", diversity_section),
    ("Visualization", viz_section),
    dynamic=True,
)

HEADER = pn.pane.Markdown("# CEtools — Electropherogram Pipeline", sizing_mode="stretch_width")
app = pn.Column(HEADER, TABS, sizing_mode="stretch_width")
app.servable(title="CEtools Pipeline")

if __name__ == "__main__":
    pn.serve(app, title="CEtools Pipeline", show=True)


