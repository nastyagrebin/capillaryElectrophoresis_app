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

# ===== app.py (optional: after _ensure_panel, before building tabs) =====
import uuid
from copy import deepcopy

def _snapshot_state_to_cache(key: str):
    pn.state.cache[key] = dict(
        converted_by_sample={k: v.copy() for k, v in state.converted_by_sample.items()},
        current_by_sample={k: v.copy() for k, v in state.current_by_sample.items()},
        aligned_pseudotimes_df=None if state.aligned_pseudotimes_df is None else state.aligned_pseudotimes_df.copy(),
        aligned_norm_df=None if state.aligned_norm_df is None else state.aligned_norm_df.copy(),
        rows_are_traces=state.rows_are_traces_aligned,
        H_df=None if state.H_df is None else state.H_df.copy(),
    )

def _restore_state_from_cache(key: str):
    snap = pn.state.cache.get(key)
    if not snap: 
        return False
    state.converted_by_sample = {k: v.copy() for k, v in snap["converted_by_sample"].items()}
    state.current_by_sample   = {k: v.copy() for k, v in snap["current_by_sample"].items()}
    state.aligned_pseudotimes_df = None if snap["aligned_pseudotimes_df"] is None else snap["aligned_pseudotimes_df"].copy()
    state.aligned_norm_df        = None if snap["aligned_norm_df"] is None else snap["aligned_norm_df"].copy()
    state.rows_are_traces_aligned = bool(snap["rows_are_traces"])
    state.H_df = None if snap["H_df"] is None else snap["H_df"].copy()
    return True

def _ensure_session_token():
    if pn.state.location is None:
        return None
    qp = dict(pn.state.location.query_params)
    sid = qp.get("sid", [None])[0]
    if not sid:
        sid = uuid.uuid4().hex[:8]
        qp["sid"] = [sid]
        pn.state.location.update(query_params=qp, replace=True)
    return sid

# On page load, try to restore:
def _onload():
    sid = _ensure_session_token()
    if sid and _restore_state_from_cache(sid):
        bridge_status.object = ok("Session restored from cache.")
        # Optionally re-wire Alignment/NMF UI with restored data:
        try:
            if state.current_by_sample:
                alignment_ctrl.set_input(state.current_by_sample)
        except Exception:
            pass

pn.state.onload(_onload)

# Call _snapshot_state_to_cache(sid) after key transitions:
def _maybe_snapshot(note=""):
    if pn.state.location is None:
        return
    sid = pn.state.location.query_params.get("sid", [None])[0]
    if sid:
        _snapshot_state_to_cache(sid)
        bridge_status.object = ok(f"State saved {note}".strip())

# Example call sites:
# - end of _on_convert_click(), after building current_by_sample
# - end of _wire_norm_apply_skip() _apply/_skip handlers
# - end of _unlock_nmf_from_alignment()
# - end of _nmf_done_callback()



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
        # pipeline dataset (what the next stage should see)
        self.current_by_sample: Dict[str, pd.DataFrame] = {}

        # aligned data (from Alignment or CSV import via NMF tab)
        self.aligned_pseudotimes_df: Optional[pd.DataFrame] = None
        self.aligned_norm_df: Optional[pd.DataFrame] = None
        self.rows_are_traces_aligned: bool = False

        # NMF loadings
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

# Preview plot + slider (no PNG/SVG exports)
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

    # keep alignment section visible
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

        # Initialize preprocessing with current dataset and unlock despike
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

bridge_status = pn.pane.Markdown("", sizing_mode="stretch_width")

# ---------- Preprocess tab (assemble controllers) ----------
despike_section, despike_ctrl = build_despike_section()
smooth_section,  smooth_ctrl  = build_smoothing_section()
baseline_section, baseline_ctrl = build_baseline_section()
norm_section,    norm_ctrl    = build_normalization_section()

# hide downstream initially (revealed via unlockers)
smooth_section.visible = False
baseline_section.visible = False
norm_section.visible = False

# ---------- Alignment tab ----------
alignment_section, alignment_ctrl = build_alignment_section()
alignment_section.visible = True  # keep anchors UI present

# ---------- NMF tab ----------
# ---------- NMF tab ----------
nmf_section, nmf_ctrl = build_nmf_section()
nmf_section.visible = True

# Provide a pull-based alignment provider to NMF
def _provide_alignment_for_nmf():
    P = state.aligned_pseudotimes_df
    Y = state.aligned_norm_df
    if P is None or Y is None:
        return None
    return (P.copy(), Y.copy(), bool(state.rows_are_traces_aligned))

nmf_ctrl.on_request_alignment = _provide_alignment_for_nmf

def _nmf_aligned_imported(P: pd.DataFrame, Y: pd.DataFrame, rows_are_traces: bool):
    state.aligned_pseudotimes_df = P.copy()
    state.aligned_norm_df = Y.copy()
    state.rows_are_traces_aligned = bool(rows_are_traces)
nmf_ctrl.on_aligned_imported = _nmf_aligned_imported

# When Alignment finishes, still push into NMF eagerly (push path)
def _unlock_nmf_from_alignment(pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool):
    # Persist
    state.aligned_pseudotimes_df = pseudotimes_df.copy()
    state.aligned_norm_df = norm_df.copy()
    state.rows_are_traces_aligned = bool(rows_are_traces)

    # Feed into NMF, select Alignment as source
    try:
        nmf_ctrl.set_alignment_input(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
        try:
            nmf_ctrl.source_select.value = "alignment"  # ensure correct source is active
        except Exception:
            pass
        # --- New: auto-generate preview using default params & default sample
        try:
            nmf_ctrl.preview_now()
        except Exception:
            # why: never let preview errors break the doc; NMF tab will show the warning
            pass
    except Exception:
        # Back-compat fallback
        try:
            nmf_ctrl.set_input(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
            try:
                nmf_ctrl.preview_now()
            except Exception:
                pass
        except Exception:
            pass

    _force_unlock_nmf_controls()
    try:
        nmf_section.visible = True
        # optional: don't auto-switch tabs; user stays on Alignment and will find NMF pre-populated
        # TABS.active = 3
    except Exception:
        pass

# Hook alignment completion (if the controller exposes on_aligned)
try:
    def _alignment_done_callback(pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool):
        _unlock_nmf_from_alignment(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
    alignment_ctrl.on_aligned = _alignment_done_callback
except Exception:
    pass

# EXTRA: When user switches to the NMF tab, push alignment data if available
def _on_tabs_active_changed(event):
    try:
        if event.new == 3:  # 0:Upload,1:Preprocess,2:Alignment,3:NMF
            if state.aligned_pseudotimes_df is not None and state.aligned_norm_df is not None:
                nmf_ctrl.set_alignment_input(
                    state.aligned_pseudotimes_df, state.aligned_norm_df,
                    rows_are_traces=state.rows_are_traces_aligned
                )
                try: nmf_ctrl.source_select.value = "alignment"
                except Exception: pass
    except Exception:
        pass

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
    """Try to enable all common NMF UI bits regardless of internal names."""
    for attr in ("import_btn", "preview_btn", "calc_btn", "apply_btn", "run_btn"):
        btn = getattr(nmf_ctrl, attr, None)
        try:
            if btn is not None:
                btn.disabled = False
        except Exception:
            pass
    # If controller exposes a status/message pane, note the source
    for attr in ("status", "info", "message"):
        pane = getattr(nmf_ctrl, attr, None)
        try:
            if pane is not None and hasattr(pane, "object"):
                pane.object = ok("Aligned data connected from Alignment tab. NMF is ready.")
        except Exception:
            pass

def _unlock_nmf_from_alignment(pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool):
    # Persist to session
    state.aligned_pseudotimes_df = pseudotimes_df.copy()
    state.aligned_norm_df = norm_df.copy()
    state.rows_are_traces_aligned = bool(rows_are_traces)

    # Feed into NMF controller and unlock UI
    try:
        nmf_ctrl.set_input(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
    except Exception:
        # Back-compat: some versions use set_aligned_input(...)
        try:
            nmf_ctrl.set_aligned_input(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
        except Exception:
            pass

    _force_unlock_nmf_controls()

    try:
        nmf_section.visible = True
        # Optionally switch to NMF so user sees it's ready:
        # TABS.active = 3
    except Exception:
        pass

def _unlock_viz_diversity_after_nmf(H_df: pd.DataFrame):
    state.H_df = H_df.copy()
    # Viz
    try:
        viz_ctrl.set_input(H_df)
        viz_section.visible = True
    except Exception:
        pass
    # Diversity: connect aligned inputs so user can compute either from aligned or from H
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
    
def _goto_alignment_with_current(note: str = "") -> None:
    """
    Safely hand current dataset to the Alignment tab and focus it.
    Shows an inline status if anything goes wrong instead of blanking tabs.
    """
    try:
        if not state.current_by_sample:
            raise ValueError("No current dataset to pass to Alignment.")
        alignment_ctrl.set_input(state.current_by_sample)
        alignment_section.visible = True
        try:
            TABS.active = 2  # focus "Alignment"
        except Exception:
            pass
        bridge_status.object = ok(f"Alignment input updated from Normalization. {note}".strip())
    except Exception as e:
        # Never let this throw; surface the error instead
        bridge_status.object = warn(f"Failed to pass data to Alignment: {e!s}")

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
    """Push the current dataset (from Upload tab) into the first preprocess stage."""
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
    # hide downstream until advanced
    smooth_section.visible = False
    baseline_section.visible = False
    norm_section.visible = False
    # alignment visible but waiting for data
    alignment_section.visible = True

# Bind wiring for preprocess
_wire_despike_apply()
_wire_despike_skip()
_wire_smooth_apply()
_wire_smooth_skip()
_wire_baseline_apply()
_wire_baseline_skip()

# ===== app.py (replace _wire_norm_apply_skip with this version) =====
def _wire_norm_apply_skip():
    def _apply(_=None):
        out = getattr(norm_ctrl, "normalized_by_sample", None) or getattr(norm_ctrl, "output_by_sample", None)
        if not out:
            norm_ctrl.status.object = warn("Generate a normalization preview first.")
            return
        state.current_by_sample = {k: v.copy() for k, v in out.items()}
        norm_ctrl.status.object = ok("Normalized data applied. Later tabs will use normalized traces.")
        _goto_alignment_with_current(note="(apply)")

    norm_ctrl.apply_btn.on_click(_apply)

    def _skip(_=None):
        norm_ctrl.status.object = ok("Skipped normalization. Current dataset unchanged.")
        if not state.current_by_sample:
            # fall back to converted data if needed
            state.current_by_sample = {k: v.copy() for k, v in state.converted_by_sample.items()}
        _goto_alignment_with_current(note="(skip)")

    norm_ctrl.skip_btn.on_click(_skip)


_wire_norm_apply_skip()

# ---------- Bridge Alignment → NMF ----------
try:
    def _alignment_done_callback(pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool):
        _unlock_nmf_from_alignment(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
    # alignment controller should call this when "Align curves" completes
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
    ("NMF", nmf_section),               # users may also import aligned CSV here
    ("Diversity", diversity_section),
    ("Visualization", viz_section),
    dynamic=True,
)

TABS.param.watch(_on_tabs_active_changed, "active")

HEADER = pn.pane.Markdown("# CEtools — Electropherogram Pipeline", sizing_mode="stretch_width")
app = pn.Column(
    HEADER,
    bridge_status,            # <-- new: shows safe-guarded errors/success messages
    TABS,
    sizing_mode="stretch_width"
)
app.servable(title="CEtools Pipeline")

if __name__ == "__main__":
    pn.serve(app, title="CEtools Pipeline", show=True)

