# ============================
# FILE: app.py
# ============================
from __future__ import annotations
import sys
import io, zipfile, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import panel as pn
import bokeh.plotting

# Local utils
from upload_utils import sanitize_name, convert_cdf_bytes_to_df, try_merge_same_time, load_csv_bytes_to_dfs
from common_plot import make_preview_plot, apply_export_prefix_to_pane, TitledPlotPane

# Preprocess controllers/sections
from preprocess_despike_utils import build_despike_section
from preprocess_smooth_utils import build_smoothing_section
from preprocess_norm_utils import build_normalization_section, NormalizationController
try:
    from preprocess_baseline_utils import build_baseline_section, BaselineController
except Exception:
    from app.preprocess_baseline_utils import build_baseline_section, BaselineController

from alignment_utils import build_alignment_section
from nmf_utils import build_nmf_section
from nmf_meta_dashboard import build_nmf_meta_section
from viz_utils import build_viz_section
from diversity_utils import build_diversity_section
from misc_utils import build_misc_section
from timeseries_utils import build_timeseries_section
from snr_utils import SNRController

# ---------------- Common UI helpers ----------------
OK = "OK:"; WARN = "Warning:"
def ok(m): return f"{OK} {m}"
def warn(m): return f"{WARN} {m}"

def _ensure_panel():
    if not getattr(pn.state, "_ce_panel_ready", False):
        pn.extension("tabulator", "bokeh")
        pn.state._ce_panel_ready = True
_ensure_panel()

bridge_status = pn.pane.Markdown("", sizing_mode="stretch_width")

# ---------- Session state ----------
class SessionState:
    def __init__(self):
        self.uploads: List[Tuple[str, bytes]] = []
        self.sample_names: Dict[str, str] = {}
        self.converted_by_sample: Dict[str, pd.DataFrame] = {}
        self.merged_df: Optional[pd.DataFrame] = None
        self.last_fig: Optional[bokeh.plotting.Figure] = None
        self.current_by_sample: Dict[str, pd.DataFrame] = {}
        # Aligned data
        self.aligned_pseudotimes_df: Optional[pd.DataFrame] = None
        self.aligned_norm_df: Optional[pd.DataFrame] = None
        
        self.metadata_df: Optional[pd.DataFrame] = None
        self.rows_are_traces_aligned: bool = False
        # NMF loadings
        self.H_df: Optional[pd.DataFrame] = None
        # Session parameters logging
        self.session_log: Dict[str, dict] = {}

state = SessionState()

# ---------- Minimal session persistence ----------
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
    if getattr(pn.state, "location", None) is None:
        return None
    qp = dict(pn.state.location.query_params)
    sid = qp.get("sid", [None])[0]
    if not sid:
        sid = uuid.uuid4().hex[:8]
        try:
            if hasattr(pn.state.location, "update"):
                pn.state.location.update(query_params={"sid": sid})
            else:
                pn.state.location.search = f"?sid={sid}"
        except Exception:
            pass
    return sid

def _onload():
    sid = _ensure_session_token()
    if sid and _restore_state_from_cache(sid):
        bridge_status.object = ok("Session restored from cache.")
        try:
            if state.current_by_sample:
                alignment_ctrl.set_input(state.current_by_sample)
        except Exception:
            pass
pn.state.onload(_onload)

def _maybe_snapshot(note=""):
    if pn.state.location is None:
        return
    sid = pn.state.location.query_params.get("sid", [None])[0]
    if sid:
        _snapshot_state_to_cache(sid)
        bridge_status.object = ok(f"State saved {note}".strip())

# ===================== Upload tab =====================
upload = pn.widgets.FileInput(accept=".cdf,.CDF,.csv,.CSV", multiple=True)
prefer_minutes = pn.widgets.Checkbox(name="Convert time to minutes (from seconds)", value=True)
asinh_toggle = pn.widgets.Checkbox(name="Use asinh transform", value=True)
upload_status = pn.pane.Markdown("Upload one or more `.cdf` or `.csv` files.", sizing_mode="stretch_width")
convert_btn = pn.widgets.Button(name="Process Files", button_type="primary", disabled=True)

rename_status = pn.pane.Markdown("", sizing_mode="stretch_width")
rename_box = pn.Column(sizing_mode="stretch_width")

def _unique_names(names: List[str]) -> Tuple[bool, str]:
    lower = [n.lower() for n in names]
    return (True, "OK") if len(set(lower)) == len(lower) else (False, "Duplicate names detected.")

def _build_rename_panel():
    if not state.uploads:
        rename_status.object = ""
        rename_box.clear()
        return
    if not state.sample_names:
        state.sample_names = {orig: sanitize_name(orig) for orig, _ in state.uploads}
        
    rows = []
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
        rows.append(pn.Row(left, edit))
        
    rename_box.objects = rows
    
    names = [state.sample_names[o] for o, _ in state.uploads]
    ok_flag, msg = _unique_names(names)
    rename_status.object = ok("Names are valid.") if ok_flag else warn(msg)
    convert_btn.disabled = not ok_flag

zip_name = pn.widgets.TextInput(name="ZIP filename (per-sample CSVs)", value="CE_analysis_converted_csvs.zip")
zip_download = pn.widgets.FileDownload(label="Download ZIP", filename=zip_name.value, button_type="primary", embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True)
zip_download._manually_prefixed = True
zip_name.param.watch(lambda e: setattr(zip_download, "filename", e.new or "converted_csvs.zip"), "value")

merge_status = pn.pane.Markdown("", sizing_mode="stretch_width")
merge_name = pn.widgets.TextInput(name="Merged CSV filename", value="CE_analysis_merged.csv")
merge_download = pn.widgets.FileDownload(label="Download Merged CSV", filename=merge_name.value, button_type="primary", embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True)
merge_download._manually_prefixed = True
merge_name.param.watch(lambda e: setattr(merge_download, "filename", e.new or "merged.csv"), "value")

plot_pane = TitledPlotPane(sizing_mode="stretch_width")
offset_slider = pn.widgets.FloatSlider(name="Vertical offset", start=0.0, end=10.0, step=0.5, value=0.0, sizing_mode="stretch_width")
upload_svg_export = pn.widgets.Checkbox(name="Enable SVG Export Mode", value=False)
sort_by_order_btn = pn.widgets.Checkbox(name="Sort by Order of Run", value=False)

downloads_group = pn.Column(pn.pane.Markdown("### Downloads"), pn.Row(zip_name, zip_download), pn.Row(merge_name, merge_download), merge_status, visible=False, sizing_mode="stretch_width")
preview_group = pn.Column(pn.pane.Markdown("### Chromatograph(s) preview"), pn.Row(offset_slider, asinh_toggle, upload_svg_export, sort_by_order_btn), plot_pane, visible=False, sizing_mode="stretch_width")

import importlib.metadata
try:
    from version import APP_VERSION
except Exception:
    APP_VERSION = "vUnknown"

def _get_pkg_version(pkg_name: str) -> str:
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "Not installed"

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
    
    import datetime
    import sys
    state.session_log.clear()
    state.session_log["General"] = {
        "App Version": APP_VERSION,
        "Python Version": sys.version.split()[0],
        "Panel Version": _get_pkg_version("panel"),
        "Bokeh Version": _get_pkg_version("bokeh"),
        "Pandas Version": _get_pkg_version("pandas"),
        "Numpy Version": _get_pkg_version("numpy"),
        "SciPy Version": _get_pkg_version("scipy"),
        "Session Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Uploaded Files": [f[0] for f in files]
    }

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
    upload_status.object = ok(f"Queued {len(files)} file(s). Edit names on the right, then click 'Process Files'.")
upload.param.watch(_on_upload_change, "value")

def _render_plot(*_):
    if not state.converted_by_sample:
        plot_pane.object = None
        state.last_fig = None
        return
        
    samples = state.converted_by_sample
    if sort_by_order_btn.value and state.metadata_df is not None and 'Order of Run' in state.metadata_df.columns:
        # Sort based on Order of Run
        try:
            order_map = state.metadata_df.set_index('Original Name')['Order of Run'].dropna().to_dict()
            # The keys in state.converted_by_sample might be renamed, wait
            # state.metadata_df has '_sample_id' which matches keys if they weren't renamed?
            # Actually, `Original Name` in metadata_df matches `state.sample_names.items()` value.
            # When we convert, we store in state.converted_by_sample using `final_name = state.sample_names.get(orig, s_name)`.
            # So `final_name` is exactly what `Original Name` stores in metadata_df!
            # Let's use Original Name to map to Order of Run
            sorted_keys = sorted(samples.keys(), key=lambda k: order_map.get(k, 999999))
            samples = {k: samples[k] for k in sorted_keys}
        except Exception:
            pass
            
    fig = make_preview_plot(samples, minutes=prefer_minutes.value, offset=offset_slider.value, asinh=asinh_toggle.value, title="Chromatograph(s) preview")
    fig.output_backend = "svg" if upload_svg_export.value else "canvas"
    plot_pane.object = fig
    apply_export_prefix_to_pane(plot_pane, global_export_prefix.value, "")
    state.last_fig = fig

offset_slider.param.watch(_render_plot, "value")
prefer_minutes.param.watch(_render_plot, "value")
asinh_toggle.param.watch(_render_plot, "value")
upload_svg_export.param.watch(_render_plot, "value")
sort_by_order_btn.param.watch(_render_plot, "value")

def _on_convert_click(event):
    if not state.uploads:
        upload_status.object = warn("Upload CDF files first.")
        return
    names = [state.sample_names.get(orig, sanitize_name(orig)) for orig, _ in state.uploads]
    ok_flag, msg = _unique_names(names)
    if not ok_flag:
        upload_status.object = warn(msg); convert_btn.disabled = True; return

    errors = []
    state.converted_by_sample.clear()
    state.merged_df = None
    state.last_fig = None
    state.current_by_sample = {}
    merge_download.disabled = True; merge_status.object = ""
    plot_pane.object = None; downloads_group.visible = False; preview_group.visible = False

    for orig, by in state.uploads:
        try:
            if orig.lower().endswith(".cdf"):
                df = convert_cdf_bytes_to_df(orig, by, prefer_minutes=prefer_minutes.value)
                sample = state.sample_names.get(orig, sanitize_name(orig))
                state.converted_by_sample[sample] = df
            else:
                # Load CSV (single or merged)
                dfs = load_csv_bytes_to_dfs(orig, by)
                for s_name, s_df in dfs.items():
                    # If single sample in file, respect the renamer; else use column name
                    final_name = state.sample_names.get(orig, s_name) if len(dfs) == 1 else s_name
                    state.converted_by_sample[final_name] = s_df
        except Exception as e:
            errors.append(f"{Path(orig).name}: {e}")

    if not state.converted_by_sample:
        upload_status.object = warn("Conversion failed."); return

    state.current_by_sample = {k: v.copy() for k, v in state.converted_by_sample.items()}
    _render_plot()
    snr_ctrl.set_input(state.converted_by_sample)
    _init_preprocess_from_current()
    _unlock_despike()

    downloads_group.visible = True; preview_group.visible = True

    def _zip_bytes():
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for sample, df in state.converted_by_sample.items():
                zf.writestr(f"{sample}.csv", df.to_csv(index=False))
        bio.seek(0); return bio
    zip_download.callback = _zip_bytes; zip_download.disabled = False

    merged, reason = try_merge_same_time(state.converted_by_sample)
    if merged is not None:
        state.merged_df = merged
        def _merged_bytes():
            bio = io.BytesIO(); state.merged_df.to_csv(bio, index=False); bio.seek(0); return bio
        merge_download.callback = _merged_bytes; merge_download.disabled = False
        merge_status.object = ok("Time stamps match across all files. You can download the merged CSV.")
    else:
        merge_download.disabled = True
        merge_status.object = warn(f"Cannot merge into one CSV: {reason}")

    msg = ok(f"Processed {len(state.converted_by_sample)} sample(s).")
    if errors:
        msg += f" {WARN} {len(errors)} failed: " + "; ".join(errors[:3]) + (" ..." if len(errors) > 3 else "")
    upload_status.object = msg
    try:
        _maybe_snapshot("(after process)")
    except Exception:
        pass
convert_btn.on_click(_on_convert_click)

# ===================== Metadata Extractor =====================
meta_pattern_input = pn.widgets.TextInput(name="Extraction Pattern", value="[*]-[M]-[DD]-[YYYY]_[HR]-[MIN]-[SEC]_[AM]dat-LIF_[*].cdf", sizing_mode="stretch_width")
meta_extract_btn = pn.widgets.Button(name="Extract Metadata", button_type="primary", width=150)
meta_status = pn.pane.Markdown("", sizing_mode="stretch_width")
meta_table = pn.widgets.Tabulator(pd.DataFrame(), height=200, show_index=False, sizing_mode="stretch_width")
meta_csv_name = pn.widgets.TextInput(name="Metadata CSV", value="metadata.csv", width=200)
meta_csv_btn = pn.widgets.FileDownload(
    label="Download Metadata", filename=meta_csv_name.value, button_type="success",
    embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True
)
meta_csv_name.param.watch(lambda e: setattr(meta_csv_btn, "filename", e.new or "metadata.csv"), "value")

import re
from datetime import datetime

def _on_meta_extract_click(event):
    if not state.sample_names:
        meta_status.object = warn("No samples uploaded.")
        return
        
    pattern = meta_pattern_input.value
    if not pattern:
        meta_status.object = warn("Please enter a pattern.")
        return
        
    # Strip extension from pattern since sample names are sanitized without it
    pattern = re.sub(r'(?i)\.cdf$', '', pattern)
    pattern = re.sub(r'(?i)\.csv$', '', pattern)
        
    regex_parts = []
    variables = []
    
    parts = re.split(r'(\[[^\]]+\])', pattern)
    for p in parts:
        if p.startswith('[') and p.endswith(']'):
            var_name = p[1:-1]
            if var_name == '*':
                regex_parts.append('.*?')
            else:
                regex_parts.append(f"(?P<{var_name}>.*?)")
                variables.append(var_name)
        else:
            regex_parts.append(re.escape(p))
            
    regex_str = "".join(regex_parts)
    
    try:
        compiled_re = re.compile(regex_str, re.IGNORECASE)
    except Exception as e:
        meta_status.object = warn(f"Invalid pattern regex: {str(e)}")
        return
        
    extracted = []
    for s_id, s_name in state.sample_names.items():
        match = compiled_re.search(s_name)
        if match:
            d = match.groupdict()
            d['_sample_id'] = s_id
            d['Original Name'] = s_name
            extracted.append(d)
        else:
            extracted.append({'_sample_id': s_id, 'Original Name': s_name})
            
    df = pd.DataFrame(extracted)
    
    # Try to parse Date/Time
    has_year = 'YYYY' in df.columns or 'YY' in df.columns
    has_month = 'MM' in df.columns or 'M' in df.columns
    has_day = 'DD' in df.columns or 'D' in df.columns
    
    if len(df) > 0 and has_year and has_month and has_day:
        parsed_dates = []
        for i, row in df.iterrows():
            try:
                y = row.get('YYYY') or row.get('YY') or '2000'
                m = row.get('M') or row.get('MM') or '1'
                d = row.get('DD') or row.get('D') or '1'
                date_str = f"{y}-{m}-{d}"
                
                time_str = "00:00:00"
                
                h = row.get('HR') or row.get('HH') or row.get('HH24')
                min_val = row.get('MIN') or row.get('MM_min')
                if not min_val and 'M' in df.columns and 'MM' in df.columns:
                    min_val = row.get('MM')
                min_val = min_val or '00'
                s = row.get('SEC') or row.get('SS') or '00'
                
                if h is not None:
                    time_str = f"{h}:{min_val}:{s}"
                    pm = row.get('AM') or row.get('PM')
                    if pm:
                        dt = datetime.strptime(f"{date_str} {time_str} {pm}", "%Y-%m-%d %I:%M:%S %p")
                    else:
                        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                else:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                parsed_dates.append(dt)
            except Exception:
                parsed_dates.append(None)
        
        df['Parsed Datetime'] = parsed_dates
        # Compute order of run
        df['Order of Run'] = df['Parsed Datetime'].rank(method='min').astype('Int64')
        
    state.metadata_df = df
    meta_table.value = df.drop(columns=['_sample_id']) if '_sample_id' in df.columns else df
    
    def _csv_bytes():
        bio = io.BytesIO()
        df.to_csv(bio, index=False)
        bio.seek(0)
        return bio
        
    meta_csv_btn.callback = _csv_bytes
    meta_csv_btn.disabled = False
    meta_status.object = ok(f"Extracted metadata for {len(df)} samples.")
    
    # Trigger preview update
    _render_plot()

meta_extract_btn.on_click(_on_meta_extract_click)

left_col = pn.Column(
    pn.pane.Markdown("## 1) Upload & Process (.cdf or .csv)"),
    pn.Row(upload, pn.Spacer(width=12), prefer_minutes),
    pn.Row(convert_btn),
    pn.Spacer(height=8),
    upload_status,
    pn.layout.Divider(),
    downloads_group,
)
mid_col = pn.Column(
    pn.pane.Markdown("### Sample Renamer"),
    pn.pane.Markdown("Edit each sample name (must be unique)."),
    rename_status,
    pn.layout.Divider(),
    rename_box,
    sizing_mode="stretch_both",
    width=380,
)
right_col = pn.Column(
    pn.pane.Markdown("### Metadata Extractor"),
    pn.pane.Markdown("Extract variables from filenames using brackets. E.g. `[sample]_[*]-[YYYY]-[MM]-[DD].cdf`"),
    meta_pattern_input,
    meta_extract_btn,
    meta_status,
    meta_table,
    pn.Row(meta_csv_name, meta_csv_btn),
    sizing_mode="stretch_both",
    width=420,
)
upload_tab = pn.Column(
    preview_group,
    pn.Row(left_col, pn.layout.HSpacer(width=16), mid_col, pn.layout.HSpacer(width=16), right_col, sizing_mode="stretch_width"),
    sizing_mode="stretch_width"
)

# ===================== SNR tab =====================
snr_ctrl = SNRController()

# ===================== Preprocess tab =====================
despike_section,  despike_ctrl  = build_despike_section()
smooth_section,   smooth_ctrl   = build_smoothing_section()
baseline_section, baseline_ctrl = build_baseline_section()
norm_section,     norm_ctrl     = build_normalization_section()
smooth_section.visible = False; baseline_section.visible = False; norm_section.visible = False

# ===================== Alignment tab =====================
alignment_section, alignment_ctrl = build_alignment_section()
alignment_section.visible = True

nmf_section, nmf_ctrl = build_nmf_section()
nmf_section.visible = True
nmf_meta_section, nmf_meta_ctrl = build_nmf_meta_section()
nmf_meta_section.visible = True

def _nmf_aligned_imported(P: pd.DataFrame, Y: pd.DataFrame, rows_are_traces: bool):
    state.aligned_pseudotimes_df = P.copy()
    state.aligned_norm_df = Y.copy()
    state.rows_are_traces_aligned = bool(rows_are_traces)
    bridge_status.object = ok("Aligned data manually supplied. Proceeding to NMF / Diversity...")
    
    # Try unlocking diversity
    try:
        diversity_ctrl.set_input(
            state.aligned_pseudotimes_df,
            state.aligned_norm_df,
            rows_are_traces=state.rows_are_traces_aligned
        )
    except Exception:
        pass
    try: _maybe_snapshot("(after CSV loaded)")
    except Exception: pass

nmf_ctrl.on_aligned_imported = _nmf_aligned_imported

# ===================== Viz & Diversity =====================
viz_section, viz_ctrl = build_viz_section()
viz_section.visible = True

diversity_section, diversity_ctrl = build_diversity_section()
misc_section, misc_ctrl = build_misc_section()
timeseries_section, timeseries_ctrl = build_timeseries_section()
diversity_section.visible = True

# CRITICAL WIRING: feed Diversity metrics into Viz for coloring
# Removed duplicate diversity binding here; handled below

# ===================== UNLOCK HELPERS =====================
def _unlock_despike():
    if not state.converted_by_sample: return
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
    except Exception: pass

def _unlock_smoothing():
    if not state.current_by_sample: return
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
    except Exception: pass

def _unlock_baseline():
    if not state.current_by_sample: return
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
    except Exception: pass

def _unlock_normalization():
    if not state.current_by_sample: return
    norm_ctrl.current_by_sample = {k: v.copy() for k, v in state.current_by_sample.items()}
    norm_section.visible = True
    try:
        norm_ctrl._render_before_fig()
        norm_ctrl.apply_btn.disabled = False
        norm_ctrl.skip_btn.disabled = False
    except Exception: pass

def _set_working_dataset(d: Dict[str, pd.DataFrame]) -> None:
    state.current_by_sample = {k: v.copy() for k, v in d.items()}

def _unlock_alignment():
    if not state.current_by_sample: return
    alignment_ctrl.set_input(state.current_by_sample)
    alignment_section.visible = True

def _unlock_viz_after_nmf(H_df: pd.DataFrame, centers: Optional[np.ndarray] = None):
    """
    Called after either NMF tab completes. Makes Viz ready.
    """
    state.H_df = H_df.copy()
    # Viz gets H
    try:
        viz_ctrl.set_input(H_df)
        viz_section.visible = True
    except Exception:
        pass
        
    # NMF Meta Dashboard gets H
    try:
        nmf_meta_ctrl.set_H(H_df, centers)
        nmf_meta_section.visible = True
    except Exception:
        pass

def _goto_alignment_with_current(note: str = "") -> None:
    try:
        if not state.current_by_sample:
            raise ValueError("No current dataset to pass to Alignment.")
        alignment_ctrl.set_input(state.current_by_sample)
        alignment_section.visible = True
        bridge_status.object = ok(f"Alignment input updated from Normalization. {note}".strip())
    except Exception as e:
        bridge_status.object = warn(f"Failed to pass data to Alignment: {e!s}")

# ===================== Wiring =====================
def _wire_despike_apply():
    def _apply(_=None):
        out = getattr(despike_ctrl, "output_by_sample", None)
        if not out: despike_ctrl.status.object = warn("Generate a despiking preview first."); return
        _set_working_dataset(out)
        despike_ctrl.status.object = ok("Despiked data applied.")
        state.session_log["Preprocessing: Despike"] = {
            "Status": "Applied",
            "Window": despike_ctrl.window.value,
            "Z-Score Threshold": despike_ctrl.z_thresh.value
        }
        _unlock_smoothing()
    despike_ctrl.apply_btn.on_click(_apply)

def _wire_despike_skip():
    def _skip(_=None):
        inp = getattr(despike_ctrl, "input_by_sample", None)
        if not inp: despike_ctrl.status.object = warn("No input data to skip."); return
        _set_working_dataset(inp)
        despike_ctrl.status.object = ok("Skipped despiking. Using input data.")
        state.session_log["Preprocessing: Despike"] = {"Status": "Skipped"}
        _unlock_smoothing()
    despike_ctrl.skip_btn.on_click(_skip)

def _wire_smooth_apply():
    def _apply(_=None):
        out = getattr(smooth_ctrl, "output_by_sample", None) or getattr(smooth_ctrl, "smoothed_by_sample", None)
        if not out:
            try: smooth_ctrl._on_preview()
            except Exception: pass
            out = getattr(smooth_ctrl, "output_by_sample", None) or getattr(smooth_ctrl, "smoothed_by_sample", None)
        if not out: smooth_ctrl.status.object = warn("Please click 'Show smoothing preview' first."); return
        _set_working_dataset(out)
        smooth_ctrl.status.object = ok("Smoothed data applied.")
        state.session_log["Preprocessing: Smooth"] = {
            "Status": "Applied",
            "Window": smooth_ctrl.window.value,
            "Polyorder": smooth_ctrl.poly.value,
            "Derivative order": smooth_ctrl.deriv_slider.value
        }
        _unlock_baseline()
    smooth_ctrl.apply_btn.on_click(_apply)

def _wire_smooth_skip():
    def _skip(_=None):
        inp = getattr(smooth_ctrl, "input_by_sample", None)
        if not inp: smooth_ctrl.status.object = warn("No input data to skip."); return
        _set_working_dataset(inp)
        smooth_ctrl.status.object = ok("Skipped smoothing. Using input data.")
        state.session_log["Preprocessing: Smooth"] = {"Status": "Skipped"}
        _unlock_baseline()
    smooth_ctrl.skip_btn.on_click(_skip)

def _wire_baseline_apply():
    def _apply(_=None):
        out = getattr(baseline_ctrl, "output_by_sample", None)
        if not out: baseline_ctrl.status.object = warn("Generate a baseline preview first."); return
        _set_working_dataset(out)
        baseline_ctrl.status.object = ok("Baseline-subtracted data applied.")
        state.session_log["Preprocessing: Baseline"] = {
            "Status": "Applied",
            "Baseline Start": baseline_ctrl.baseline_start.value,
            "Baseline End": baseline_ctrl.baseline_end.value
        }
        _unlock_normalization()
    baseline_ctrl.apply_btn.on_click(_apply)

def _wire_baseline_skip():
    def _skip(_=None):
        inp = getattr(baseline_ctrl, "input_by_sample", None)
        if not inp: baseline_ctrl.status.object = warn("No input data to skip."); return
        _set_working_dataset(inp)
        baseline_ctrl.status.object = ok("Skipped baseline subtraction. Using input data.")
        state.session_log["Preprocessing: Baseline"] = {"Status": "Skipped"}
        _unlock_normalization()
    baseline_ctrl.skip_btn.on_click(_skip)

def _init_preprocess_from_current():
    if not state.current_by_sample: return
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
    except Exception: pass
    smooth_section.visible = False; baseline_section.visible = False; norm_section.visible = False
    alignment_section.visible = True

_wire_despike_apply()
_wire_despike_skip()
_wire_smooth_apply()
_wire_smooth_skip()
_wire_baseline_apply()
_wire_baseline_skip()

def _wire_norm_apply_skip():
    def _apply(event=None):
        out = getattr(norm_ctrl, "normalized_by_sample", None) or getattr(norm_ctrl, "output_by_sample", None)
        if not out: norm_ctrl.status.object = warn("Generate a normalization preview first."); return
        state.current_by_sample = {k: v.copy() for k, v in out.items()}
        norm_ctrl.status.object = ok("Normalized data applied. Later tabs will use normalized traces.")
        
        # Log norm parameters
        try:
            xmin = float(norm_ctrl._sel_state.data['xmin'][0])
            xmax = float(norm_ctrl._sel_state.data['xmax'][0])
        except Exception:
            xmin, xmax = "Unknown", "Unknown"
            
        method = "Area" if (event and event.obj.name == "Normalize by Area") else ("Height" if (event and event.obj.name == "Normalize by Height") else "Applied")
            
        state.session_log["Preprocessing: Normalization"] = {
            "Status": method,
            "Reference Peak Start": xmin,
            "Reference Peak End": xmax
        }
        
        _goto_alignment_with_current(note="(apply)")
        try: _maybe_snapshot("(after norm apply)")
        except Exception: pass
    norm_ctrl.apply_btn.on_click(_apply)
    norm_ctrl.apply_height_btn.on_click(_apply)

    def _skip(_=None):
        norm_ctrl.status.object = ok("Skipped normalization. Current dataset unchanged.")
        if not state.current_by_sample:
            state.current_by_sample = {k: v.copy() for k, v in state.converted_by_sample.items()}
        state.session_log["Preprocessing: Normalization"] = {"Status": "Skipped"}
        _goto_alignment_with_current(note="(skip)")
        try: _maybe_snapshot("(after norm skip)")
        except Exception: pass
    norm_ctrl.skip_btn.on_click(_skip)
_wire_norm_apply_skip()

# ---------- Alignment → NMF (current) bridge ----------
def _alignment_done_callback(pseudotimes_df: pd.DataFrame, norm_df: pd.DataFrame, *, rows_are_traces: bool):
    state.aligned_pseudotimes_df = pseudotimes_df.copy()
    state.aligned_norm_df        = norm_df.copy()
    state.rows_are_traces_aligned = bool(rows_are_traces)
    bridge_status.object = ok("Alignment completed. Proceed to NMF.")
    
    state.session_log["Alignment"] = {
        "Status": "Completed",
        "Mode": alignment_ctrl.align_mode.value,
        "Reference Sample": alignment_ctrl.reference_sample.value
    }
    
    # Prime current NMF and focus its tab
    try:
        nmf_ctrl.set_alignment_input(pseudotimes_df, norm_df, rows_are_traces=rows_are_traces)
        bridge_status.object = ok("Alignment complete → NMF and Diversity primed.")
    except Exception as e:
        bridge_status.object = warn(f"Failed to pass aligned data to NMF: {e}")
        
    # Unlock Diversity tab
    try:
        if state.aligned_pseudotimes_df is not None and state.aligned_norm_df is not None:
            diversity_ctrl.set_input(
                state.aligned_pseudotimes_df,
                state.aligned_norm_df,
                rows_are_traces=state.rows_are_traces_aligned
            )
        diversity_section.visible = True
    except Exception:
        pass

    try: _maybe_snapshot("(after alignment)")
    except Exception: pass

try:
    alignment_ctrl.on_aligned = _alignment_done_callback
except Exception:
    pass

# ---------- When either NMF finishes, unlock viz ----------
def _nmf_done_callback(H_df: pd.DataFrame, centers: Optional[np.ndarray] = None):
    state.session_log["NMF"] = {
        "Status": "Completed",
        "Num Components": nmf_ctrl.k_slider.value,
        "L2 Penalty": nmf_ctrl.l2_input.value,
        "ROI Start": nmf_ctrl.roi_lo.value,
        "ROI End": nmf_ctrl.roi_hi.value
    }
    _unlock_viz_after_nmf(H_df, centers)
    bridge_status.object = ok("NMF loadings set for this session.")
    try: _maybe_snapshot("(after NMF)")
    except Exception: pass

try:
    nmf_ctrl.on_done  = _nmf_done_callback
except Exception:
    pass

def _diversity_updated_callback(div_df: pd.DataFrame):
    state.session_log["Alpha Diversity"] = {
        "Status": "Completed",
        "Edge Smooth Window": diversity_ctrl.edge_smooth.value,
        "Baseline Window": diversity_ctrl.baseline_window.value,
        "ROI Start": diversity_ctrl.roi_lo.value,
        "ROI End": diversity_ctrl.roi_hi.value
    }
    viz_ctrl.set_diversity(div_df)

try:
    diversity_ctrl.on_updated = _diversity_updated_callback
except Exception:
    pass

# ===================== Tabs & Layout =====================
preproc_tab = pn.Column(
    pn.pane.Markdown("## 2) Preprocessing"),
    despike_section, pn.layout.Divider(),
    smooth_section, pn.layout.Divider(),
    baseline_section, pn.layout.Divider(),
    norm_section,
    sizing_mode="stretch_width",
)

from report_utils import build_report_section
report_section, report_ctrl = build_report_section()

TABS = pn.Tabs(
    ("Upload", upload_tab),
    ("SNR", snr_ctrl.section),
    ("Preprocess", preproc_tab),
    ("Alignment", alignment_section),
    ("NMF", nmf_section),
    ("NMF Group Stats", nmf_meta_section),
    ("Alpha Diversity", diversity_section),
    ("Visualization", viz_section),
    ("Miscellaneous", misc_section),
    ("Time Series", timeseries_section),
    ("Final Report", report_section),
    dynamic=True,
)

def _on_tab_change(event):
    if event.new in [8, 9]: # Miscellaneous (8) or Time Series (9)
        # Try to get loadings from NMF stats (might be uploaded) or NMF
        loadings = getattr(nmf_meta_ctrl, "H_df", None)
        if loadings is None or loadings.empty:
            loadings = getattr(nmf_ctrl, "H_df", None)
        if loadings is not None and not loadings.empty:
            loadings = loadings.copy()
            if "Unnamed: 0" in loadings.columns:
                loadings.set_index("Unnamed: 0", inplace=True)

        # Try to get metadata from visualization or NMF stats
        m = None
        if hasattr(viz_ctrl, "meta_table") and viz_ctrl.meta_table.value is not None and not viz_ctrl.meta_table.value.empty:
            tmp = viz_ctrl.meta_table.value
            if "sample" in tmp.columns:
                m = tmp.copy()
                m.set_index("sample", inplace=True)
        
        if (m is None or m.empty) and hasattr(nmf_meta_ctrl, "metadata_df") and nmf_meta_ctrl.metadata_df is not None and not nmf_meta_ctrl.metadata_df.empty:
            tmp = nmf_meta_ctrl.metadata_df.copy()
            scol = getattr(nmf_meta_ctrl.sample_col_select, "value", None)
            if scol and scol in tmp.columns:
                tmp.set_index(scol, inplace=True)
                m = tmp

        div = getattr(diversity_ctrl, "metric_df", None)
        
        if event.new == 8:
            misc_ctrl.bridge_data(loadings_df=loadings, meta_df=m, div_df=div)
        elif event.new == 9:
            timeseries_ctrl.bridge_data(loadings_df=loadings, meta_df=m, div_df=div)
            
    elif event.new == 10: # Final Report
        report_ctrl.update_preview(state.session_log)

TABS.param.watch(_on_tab_change, "active")

global_export_prefix = pn.widgets.TextInput(name="Global Export Prefix", value="CE_analysis", width=250)

def _update_export_prefixes(event):
    p = getattr(event, "new", event)
    old_p = getattr(event, "old", "") if hasattr(event, "old") else ""
    for w in [zip_name, merge_name, meta_csv_name]:
        val = w.value
        if old_p and val.startswith(f"{old_p}_"):
            w.value = f"{p}_{val[len(old_p)+1:]}"
        elif not val.startswith(f"{p}_"):
            w.value = f"{p}_{val}"

    for ctrl in [viz_ctrl, diversity_ctrl, misc_ctrl, timeseries_ctrl, nmf_ctrl, nmf_meta_ctrl]:
        if hasattr(ctrl, "export_prefix"):
            ctrl.export_prefix = p
        else:
            setattr(ctrl, "export_prefix", p)
    apply_export_prefix_to_pane(TABS, p, old_p)

global_export_prefix.param.watch(_update_export_prefixes, "value")
_update_export_prefixes(global_export_prefix.value)

import common_plot

global_colormap = pn.widgets.Select(name="Colormap", options=common_plot.get_available_palettes(), value="glasbey", width=200)

def _update_colormap(event):
    common_plot.CURRENT_PALETTE_NAME = event.new
    # Trigger a replot of the preview if it exists
    if len(state.current_by_sample) > 0:
        _render_plot()

global_colormap.param.watch(_update_colormap, "value")

HEADER = pn.pane.Markdown(f"# CEtools — Electropherogram Pipeline ({APP_VERSION})", sizing_mode="stretch_width")
HEADER_ROW = pn.Row(HEADER, pn.Spacer(sizing_mode='stretch_width'), global_colormap, global_export_prefix, sizing_mode="stretch_width")

COLORMAP_INFO = pn.pane.Markdown("""
***Colormap Types**: **Continuous** creates a smooth gradient (great for many samples or time-series). **Categorical** uses distinct, high-contrast colors (great for discrete groups). **Divergent** transitions between two contrasting colors. 
Colormaps provided by [Crameri (Scientific Colour Maps)](https://www.fabiocrameri.ch/colourmaps/) and [Colorcet (Glasbey)](https://colorcet.holoviz.org/)*
""", sizing_mode="stretch_width", margin=(0, 15, 10, 15))

app = pn.Column(HEADER_ROW, COLORMAP_INFO, bridge_status, TABS, sizing_mode="stretch_width")
app.servable(title="CEtools Pipeline")

if __name__ == "__main__":
    pn.serve(app, title="CEtools Pipeline", show=True, websocket_max_message_size=104857600)





