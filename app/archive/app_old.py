# app/app.py — Upload/rename + preview; Preprocess tab: Despike → Smooth → Normalize (selection + AUC fill)

from __future__ import annotations
import io, re, zipfile, tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import panel as pn
import bokeh, bokeh.plotting
from bokeh.palettes import Category20, Turbo256
from bokeh.models import BoxAnnotation, ColumnDataSource
from scipy.signal import find_peaks, peak_widths


# Optional exports (PNG/SVG)
try:
    from bokeh.io import export_png, export_svgs
except Exception:
    export_png = None  # type: ignore[assignment]
    export_svgs = None  # type: ignore[assignment]

# ---------------------- UX helpers ----------------------
OK = "OK:"
WARN = "Warning:"

def ok(msg: str) -> str:
    return f"{OK} {msg}"

def warn(msg: str) -> str:
    return f"{WARN} {msg}"

# ---------------------- Panel init ----------------------
def _ensure_panel():
    if not getattr(pn.state, "_ce_panel_ready", False):
        pn.extension()
        pn.state._ce_panel_ready = True
_ensure_panel()

# ====================== Core: CDF → CSV (scalar timing) ======================
def _open_cdf(path: str):
    try:
        import netCDF4
        return netCDF4.Dataset(path, "r"), "netcdf4"
    except Exception:
        import xarray as xr
        try:
            return xr.open_dataset(path, engine="scipy", decode_times=False), "xarray"
        except Exception:
            return xr.open_dataset(path, engine="netcdf4", decode_times=False), "xarray"

def _get_var(ds, name: str):
    return ds.variables[name] if hasattr(ds, "variables") else ds[name]

def _as_float_scalar(obj) -> float:
    if hasattr(obj, "values"):
        arr = np.asarray(obj.values)
    else:
        arr = np.asarray(obj[:]) if hasattr(obj, "__getitem__") else np.asarray(obj)
    return float(arr.reshape(()))

def convert_cdf_scalar_timing_to_df(path: str, *, prefer_minutes: bool = False) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    ds, _ = _open_cdf(path)
    try:
        names = list(ds.variables.keys()) if hasattr(ds, "variables") else list(ds.variables)
        if "ordinate_values" not in names:
            raise ValueError("CDF missing 'ordinate_values'.")
        var_y = _get_var(ds, "ordinate_values")
        y = np.asarray(var_y.values if hasattr(var_y, "values") else var_y[:], dtype=float)

        if "actual_sampling_interval" not in names or "actual_run_time_length" not in names:
            raise ValueError("CDF must contain 'actual_sampling_interval' and 'actual_run_time_length'.")
        dt  = _as_float_scalar(_get_var(ds, "actual_sampling_interval"))
        T   = _as_float_scalar(_get_var(ds, "actual_run_time_length"))
        t0  = _as_float_scalar(_get_var(ds, "actual_delay_time")) if "actual_delay_time" in names else 0.0

        n_expected = int(round(T / dt)) + 1
        t = t0 + dt * np.arange(y.size if y.size != n_expected else n_expected, dtype=float)
        if prefer_minutes:
            t = t / 60.0

        n = min(t.size, y.size)
        t, y = t[:n], y[:n]
        mask = np.isfinite(t) & np.isfinite(y)
        t, y = t[mask], y[mask]
        return pd.DataFrame({"time": t, "intensity": y})
    finally:
        try:
            ds.close()
        except Exception:
            pass

def convert_cdf_bytes_to_df(file_name: str, data: bytes, *, prefer_minutes: bool = False) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / (Path(file_name).stem + ".cdf")
        p.write_bytes(data)
        return convert_cdf_scalar_timing_to_df(str(p), prefer_minutes=prefer_minutes)

# ====================== Merge helper ======================
def _sanitize_name(s: str) -> str:
    s = Path(s).stem
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s or "untitled"

def try_merge_same_time(
    dfs_by_sample: Dict[str, pd.DataFrame],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> tuple[Optional[pd.DataFrame], str]:
    if not dfs_by_sample:
        return None, "No data to merge."
    names = list(dfs_by_sample.keys())
    t0 = dfs_by_sample[names[0]]["time"].to_numpy()
    n0 = t0.size
    for nm in names[1:]:
        df = dfs_by_sample[nm]
        if not {"time","intensity"}.issubset(df.columns):
            return None, f"{nm} missing required columns."
        ti = df["time"].to_numpy()
        if ti.size != n0:
            return None, f"Time length mismatch: {names[0]}={n0}, {nm}={ti.size}."
        if not np.allclose(t0, ti, rtol=rtol, atol=atol):
            return None, f"Time stamps differ between {names[0]} and {nm}; cannot merge."
    out = pd.DataFrame({"time": t0})
    for nm in names:
        out[nm] = dfs_by_sample[nm]["intensity"].to_numpy()
    return out, "OK"

# ====================== App session ======================
class SessionState:
    def __init__(self):
        self.uploads: List[Tuple[str, bytes]] = []
        self.sample_names: Dict[str, str] = {}
        self.converted_by_sample: Dict[str, pd.DataFrame] = {}
        self.merged_df: Optional[pd.DataFrame] = None
        self.last_fig: Optional[bokeh.plotting.Figure] = None

        # Preprocess pipeline datasets
        self.current_by_sample: Dict[str, pd.DataFrame] = {}
        self.despiked_by_sample: Dict[str, pd.DataFrame] = {}
        self.smoothed_by_sample: Dict[str, pd.DataFrame] = {}
        self.normalized_by_sample: Dict[str, pd.DataFrame] = {}
        self.norm_auc: Dict[str, float] = {}

state = SessionState()

# ====================== Renamer UI ======================
rename_status = pn.pane.Markdown("", sizing_mode="stretch_width")
rename_box = pn.Column(sizing_mode="stretch_both")

def _unique_names(names: List[str]) -> tuple[bool, str]:
    lower = [n.lower() for n in names]
    return (True, "OK") if len(set(lower)) == len(lower) else (False, "Duplicate names detected.")

def _build_rename_panel():
    rename_box.objects = []
    if not state.uploads:
        rename_status.object = ""
        return
    if not state.sample_names:
        state.sample_names = {orig: _sanitize_name(orig) for orig, _ in state.uploads}

    for orig, _ in state.uploads:
        left = pn.pane.Markdown(f"**Original**: `{Path(orig).name}`", width=380)
        edit = pn.widgets.TextInput(name="", value=state.sample_names[orig], width=260)
        def _on_change(event, orig=orig, edit=edit):
            val = _sanitize_name(event.new)
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

# ====================== Upload/Convert widgets ======================
upload = pn.widgets.FileInput(accept=".cdf,.CDF", multiple=True)
prefer_minutes = pn.widgets.Checkbox(name="Convert time to minutes (from seconds)", value=False)
upload_status = pn.pane.Markdown("Upload one or more `.cdf` files; convert to CSV using scalar timing.", sizing_mode="stretch_width")
convert_btn = pn.widgets.Button(name="Convert to CSV", button_type="primary", disabled=True)

# Downloads
zip_name = pn.widgets.TextInput(name="ZIP filename (per-sample CSVs)", value="converted_csvs.zip")
zip_download = pn.widgets.FileDownload(label="Download ZIP", filename=zip_name.value, button_type="primary", embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True)
zip_name.param.watch(lambda e: setattr(zip_download, "filename", e.new or "converted_csvs.zip"), "value")

merge_status = pn.pane.Markdown("", sizing_mode="stretch_width")
merge_name = pn.widgets.TextInput(name="Merged CSV filename", value="merged.csv")
merge_download = pn.widgets.FileDownload(label="Download Merged CSV", filename=merge_name.value, button_type="primary", embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True)
merge_name.param.watch(lambda e: setattr(merge_download, "filename", e.new or "merged.csv"), "value")

# Preview plot + slider + exports
plot_pane = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")
offset_slider = pn.widgets.FloatSlider(name="Vertical offset", start=0.0, end=10.0, step=0.5, value=0.0, sizing_mode="stretch_width")
export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
png_name = pn.widgets.TextInput(name="PNG filename", value="preview.png", width=300)
svg_name = pn.widgets.TextInput(name="SVG filename", value="preview.svg", width=300)
png_download = pn.widgets.FileDownload(label="Export PNG", filename=png_name.value, button_type="primary", embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True)
svg_download = pn.widgets.FileDownload(label="Export SVG (vector)", filename=svg_name.value, button_type="primary", embed=False, auto=False, callback=lambda: io.BytesIO(b""), disabled=True)
png_name.param.watch(lambda e: setattr(png_download, "filename", e.new or "preview.png"), "value")
svg_name.param.watch(lambda e: setattr(svg_download, "filename", e.new or "preview.svg"), "value")

# ====================== Upload Callbacks ======================
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

    # reset preprocess pipeline & UI
    state.current_by_sample = {}
    state.despiked_by_sample.clear()
    state.smoothed_by_sample.clear()
    state.normalized_by_sample.clear()
    state.norm_auc.clear()

    zip_download.disabled = True
    merge_download.disabled = True
    merge_status.object = ""
    plot_pane.object = None
    png_download.disabled = True
    svg_download.disabled = True
    export_status.object = ""

    preproc_status.object = ""
    before_despike_pane.object = None
    after_despike_pane.object = None
    apply_despike_btn.disabled = True
    despike_export_btn.disabled = True
    despike_export_status.object = ""

    smoothing_section.visible = False
    preview_smooth_btn.disabled = True
    apply_smooth_btn.disabled = True
    smooth_export_btn.disabled = True
    before_smooth_pane.object = None
    after_smooth_pane.object = None
    smooth_export_status.object = ""

    normalization_section.visible = False
    preview_norm_btn.disabled = True
    apply_norm_btn.disabled = True
    skip_norm_btn.disabled = True
    norm_export_btn.disabled = True
    norm_export_status.object = ""
    norm_before_pane.object = None
    norm_after_pane.object = None
    _norm_clear_selection_state()

    if not files:
        upload_status.object = warn("No files selected.")
        state.sample_names.clear()
        rename_box.objects = []
        rename_status.object = ""
        convert_btn.disabled = True
        return

    state.sample_names = {orig: _sanitize_name(orig) for orig, _ in state.uploads}
    _build_rename_panel()
    upload_status.object = ok(f"Queued {len(files)} CDF file(s). Edit names on the right, then click 'Convert to CSV'.")

upload.param.watch(_on_upload_change, "value")

def _make_preview_plot(samples_to_df: Dict[str, pd.DataFrame], *, minutes: bool, offset: float, title: str = "Chromatograph(s) preview") -> bokeh.plotting.Figure:
    n = len(samples_to_df)
    if n <= 20:
        colors = list(Category20[20])[:n]
    else:
        idxs = np.linspace(0, 255, num=n, dtype=int)
        colors = [Turbo256[i] for i in idxs]

    p = bokeh.plotting.figure(
        title=title,
        height=400,
        sizing_mode="stretch_width",
        x_axis_label="time (min)" if minutes else "time (s)",
        y_axis_label="TIC (raw)",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        active_scroll="wheel_zoom",
    )
    for i, (sample, df) in enumerate(samples_to_df.items()):
        t = df["time"].to_numpy()
        y = df["intensity"].to_numpy() - i * offset
        p.line(x=t, y=y, legend_label=sample, color=colors[i % len(colors)])
    p.legend.click_policy = "hide"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p

def _render_plot(*_):
    if not state.converted_by_sample:
        plot_pane.object = None
        state.last_fig = None
        png_download.disabled = True
        svg_download.disabled = True
        return
    fig = _make_preview_plot(state.converted_by_sample, minutes=prefer_minutes.value, offset=offset_slider.value)
    plot_pane.object = fig
    state.last_fig = fig
    png_download.disabled = False
    svg_download.disabled = False
    export_status.object = ok("Ready to export current preview.")

offset_slider.param.watch(_render_plot, "value")
prefer_minutes.param.watch(_render_plot, "value")

def _on_convert_click(event):
    if not state.uploads:
        upload_status.object = warn("Upload CDF files first.")
        return
    names = [state.sample_names.get(orig, _sanitize_name(orig)) for orig, _ in state.uploads]
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
    state.despiked_by_sample.clear()
    state.smoothed_by_sample.clear()
    state.normalized_by_sample.clear()
    state.norm_auc.clear()

    merge_download.disabled = True
    merge_status.object = ""
    plot_pane.object = None
    png_download.disabled = True
    svg_download.disabled = True
    export_status.object = ""

    preproc_status.object = ""
    before_despike_pane.object = None
    after_despike_pane.object = None
    apply_despike_btn.disabled = True
    despike_export_btn.disabled = True
    despike_export_status.object = ""

    smoothing_section.visible = False
    preview_smooth_btn.disabled = True
    apply_smooth_btn.disabled = True
    smooth_export_btn.disabled = True
    before_smooth_pane.object = None
    after_smooth_pane.object = None
    smooth_export_status.object = ""

    normalization_section.visible = False
    preview_norm_btn.disabled = True
    apply_norm_btn.disabled = True
    skip_norm_btn.disabled = True
    norm_export_btn.disabled = True
    norm_export_status.object = ""
    norm_before_pane.object = None
    norm_after_pane.object = None
    _norm_clear_selection_state()

    for orig, by in state.uploads:
        try:
            df = convert_cdf_bytes_to_df(orig, by, prefer_minutes=prefer_minutes.value)
            sample = state.sample_names.get(orig, _sanitize_name(orig))
            state.converted_by_sample[sample] = df
        except Exception as e:
            errors.append(f"{Path(orig).name}: {e}")

    if state.converted_by_sample:
        state.current_by_sample = {k: v.copy() for k, v in state.converted_by_sample.items()}
        _render_plot()

        def _zip_bytes():
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for sample, df in state.converted_by_sample.items():
                    zf.writestr(f"{sample}.csv", df.to_csv(index=False))
            bio.seek(0)
            return bio
        zip_download.callback = _zip_bytes
        zip_download.disabled = False

        merged, reason = try_merge_same_time(state.converted_by_sample)
        if merged is not None:
            state.merged_df = merged
            def _merged_bytes():
                bio = io.BytesIO()
                state.merged_df.to_csv(bio, index=False)
                bio.seek(0)
                return bio
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
    else:
        upload_status.object = warn("Conversion failed.")

convert_btn.on_click(_on_convert_click)

# --------- Export callbacks (PNG/SVG) ---------
def _export_png_bytes():
    fig = state.last_fig
    if fig is None:
        export_status.object = warn("No figure to export.")
        return io.BytesIO(b"")
    if export_png is None:
        export_status.object = warn("PNG export dependencies not available (bokeh export).")
        return io.BytesIO(b"")
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preview.png"
            export_png(fig, filename=str(p))
            data = p.read_bytes()
            return io.BytesIO(data)
    except Exception as e:
        export_status.object = warn(f"PNG export failed: {e}")
        return io.BytesIO(b"")

def _export_svg_bytes():
    fig = state.last_fig
    if fig is None:
        export_status.object = warn("No figure to export.")
        return io.BytesIO(b"")
    if export_svgs is None:
        export_status.object = warn("SVG export dependencies not available (bokeh export).")
        return io.BytesIO(b"")
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "preview.svg"
            orig_backend = getattr(fig, "output_backend", "canvas")
            fig.output_backend = "svg"
            export_svgs(fig, filename=str(p))
            fig.output_backend = orig_backend
            data = p.read_bytes()
            return io.BytesIO(data)
    except Exception as e:
        export_status.object = warn(f"SVG export failed: {e}")
        return io.BytesIO(b"")

png_download.callback = _export_png_bytes
svg_download.callback = _export_svg_bytes

# ====================== LAYOUT: Upload tab ======================
export_row = pn.Row(png_name, png_download, pn.Spacer(width=12), svg_name, svg_download, sizing_mode="stretch_width")
export_status_row = pn.Row(export_status, sizing_mode="stretch_width")

left_col = pn.Column(
    pn.pane.Markdown("## 1) Upload & Convert (.cdf → .csv, scalar timing)"),
    pn.Row(upload, pn.Spacer(width=12), prefer_minutes),
    pn.Row(convert_btn),
    pn.Spacer(height=8),
    upload_status,
    pn.layout.Divider(),
    pn.pane.Markdown("### Downloads"),
    pn.Row(zip_name, zip_download),
    pn.Row(merge_name, merge_download),
    merge_status,
    pn.layout.Divider(),
    plot_pane,
    offset_slider,
    export_row,
    export_status_row,
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

top_row = pn.Row(left_col, pn.layout.HSpacer(width=16), right_col, sizing_mode="stretch_width")
upload_tab = pn.Column(top_row, sizing_mode="stretch_width")

# ====================== PREPROCESS Tab ======================
preproc_status = pn.pane.Markdown("", sizing_mode="stretch_width")

# --- Despiking ---
window_slider = pn.widgets.IntSlider(name="Despike window (odd)", start=3, end=21, step=2, value=5)
z_slider = pn.widgets.FloatSlider(name="Z-threshold", start=3.0, end=12.0, step=0.5, value=6.0)
preview_despike_btn = pn.widgets.Button(name="Show despiking preview", button_type="primary")
apply_despike_btn   = pn.widgets.Button(name="Apply despiked data", button_type="success", disabled=True)

before_despike_pane = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")
after_despike_pane  = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")

despike_export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
despike_export_name = pn.widgets.TextInput(name="Despiked filename", value="despiked_merged.csv", width=300)
despike_export_btn = pn.widgets.FileDownload(
    label="Export despiked",
    filename=despike_export_name.value,
    button_type="primary",
    embed=False,
    auto=False,
    callback=lambda: io.BytesIO(b""),
    disabled=True,
)
despike_export_name.param.watch(lambda e: setattr(despike_export_btn, "filename", e.new or "despiked_merged.csv"), "value")

def _plot_multi(samples_to_df: Dict[str, pd.DataFrame], title: str) -> bokeh.plotting.Figure:
    n = len(samples_to_df)
    colors = (list(Category20[20])[:n]) if n <= 20 else [Turbo256[i] for i in np.linspace(0,255,n,dtype=int)]
    p = bokeh.plotting.figure(
        title=title,
        height=400,
        sizing_mode="stretch_width",
        x_axis_label="time",
        y_axis_label="intensity",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        active_scroll="wheel_zoom",
    )
    for i, (sample, df) in enumerate(samples_to_df.items()):
        p.line(df["time"], df["intensity"], color=colors[i % len(colors)], legend_label=sample)
    p.legend.click_policy = "hide"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p

def _on_preview_despike(_):
    if not state.converted_by_sample:
        preproc_status.object = warn("No converted data found. Complete uploads and conversion in the Upload tab.")
        return
    try:
        from CEtools.filters import remove_single_point_spikes  # type: ignore
    except Exception:
        try:
            from filters import remove_single_point_spikes  # type: ignore
        except Exception as e:
            preproc_status.object = warn(f"Could not import remove_single_point_spikes: {e}")
            return

    win = int(window_slider.value)
    zt  = float(z_slider.value)

    state.despiked_by_sample.clear()
    for sample, df in state.converted_by_sample.items():
        y = df["intensity"].to_numpy(dtype=float)
        y_ds = remove_single_point_spikes(y, window=win, z_thresh=zt)
        state.despiked_by_sample[sample] = pd.DataFrame({"time": df["time"].to_numpy(), "intensity": y_ds})

    before_fig = _plot_multi(state.converted_by_sample, title="Before (raw)")
    after_fig  = _plot_multi(state.despiked_by_sample,  title="After (despiked)")
    after_fig.x_range = before_fig.x_range
    after_fig.y_range = before_fig.y_range

    before_despike_pane.object = before_fig
    after_despike_pane.object  = after_fig
    preproc_status.object = ok("Despiking preview generated.")
    despike_export_btn.disabled = False
    despike_export_status.object = ok("Despiked data ready to export.")
    apply_despike_btn.disabled = False

preview_despike_btn.on_click(_on_preview_despike)

def _despiked_export_bytes():
    if not state.despiked_by_sample:
        despike_export_status.object = warn("No despiked data to export. Generate a preview first.")
        return io.BytesIO(b"")

    merged, reason = try_merge_same_time(state.despiked_by_sample)
    if merged is not None:
        if not despike_export_name.value.lower().endswith(".csv"):
            despike_export_name.value = "despiked_merged.csv"
        bio = io.BytesIO()
        merged.to_csv(bio, index=False)
        bio.seek(0)
        despike_export_status.object = ok("Exporting merged despiked CSV.")
        return bio

    if not despike_export_name.value.lower().endswith(".zip"):
        despike_export_name.value = "despiked_csvs.zip"
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sample, df in state.despiked_by_sample.items():
            zf.writestr(f"{sample}.csv", df.to_csv(index=False))
    bio.seek(0)
    despike_export_status.object = warn(f"Time stamps differ; exporting per-sample CSVs as ZIP. Reason: {reason}")
    return bio

despike_export_btn.callback = _despiked_export_bytes

def _unlock_normalization_stage():
    normalization_section.visible = True
    preview_norm_btn.disabled = False
    apply_norm_btn.disabled = True
    skip_norm_btn.disabled = False
    norm_export_btn.disabled = True
    norm_export_status.object = ""
    norm_before_pane.object = None
    norm_after_pane.object = None
    _norm_clear_selection_state()

def _on_apply_despike(_):
    if not state.despiked_by_sample:
        preproc_status.object = warn("Generate a despiking preview first.")
        return
    state.current_by_sample = {k: v.copy() for k, v in state.despiked_by_sample.items()}
    preproc_status.object = ok("Despiked data applied. Smoothing is now available (and normalization unlocked).")
    smoothing_section.visible = True
    preview_smooth_btn.disabled = False
    apply_smooth_btn.disabled = True
    smooth_export_btn.disabled = True
    before_smooth_pane.object = None
    after_smooth_pane.object = None
    smooth_export_status.object = ""
    _unlock_normalization_stage()

apply_despike_btn.on_click(_on_apply_despike)

# --- Smoothing ---
smooth_win = pn.widgets.IntSlider(name="Savgol window (odd)", start=5, end=101, step=2, value=9)
smooth_poly = pn.widgets.IntSlider(name="Polyorder", start=2, end=5, step=1, value=3)
preview_smooth_btn = pn.widgets.Button(name="Show smoothing preview", button_type="primary", disabled=True)
apply_smooth_btn   = pn.widgets.Button(name="Apply smoothed data", button_type="success", disabled=True)

before_smooth_pane = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")
after_smooth_pane  = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")

smooth_export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
smooth_export_name = pn.widgets.TextInput(name="Smoothed filename", value="smoothed_merged.csv", width=300)
smooth_export_btn = pn.widgets.FileDownload(
    label="Export smoothed",
    filename=smooth_export_name.value,
    button_type="primary",
    embed=False,
    auto=False,
    callback=lambda: io.BytesIO(b""),
    disabled=True,
)
smooth_export_name.param.watch(lambda e: setattr(smooth_export_btn, "filename", e.new or "smoothed_merged.csv"), "value")

def _savgol_smooth(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = y.size
    if n == 0:
        return y.copy()
    w = int(window)
    if w % 2 == 0:
        w -= 1
    minw = poly + 2 if (poly + 2) % 2 == 1 else (poly + 3)
    w = max(w, minw)
    w = min(w, n - 1 if (n - 1) % 2 == 1 else (n - 2))
    if w < 3:
        w = 3
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    from scipy.signal import savgol_filter
    return savgol_filter(y, window_length=int(w), polyorder=int(poly), mode="interp")

def _on_preview_smooth(_):
    if not state.current_by_sample:
        preproc_status.object = warn("No input for smoothing. Apply despiked data first.")
        return
    w = int(smooth_win.value)
    p = int(smooth_poly.value)

    state.smoothed_by_sample.clear()
    for sample, df in state.current_by_sample.items():
        y = df["intensity"].to_numpy(dtype=float)
        ys = _savgol_smooth(y, window=w, poly=p)
        state.smoothed_by_sample[sample] = pd.DataFrame({"time": df["time"].to_numpy(), "intensity": ys})

    before_fig = _plot_multi(state.current_by_sample, title="Before smoothing (input)")
    after_fig  = _plot_multi(state.smoothed_by_sample,  title="After smoothing")
    after_fig.x_range = before_fig.x_range
    after_fig.y_range = before_fig.y_range

    before_smooth_pane.object = before_fig
    after_smooth_pane.object  = after_fig
    preproc_status.object = ok("Smoothing preview generated.")
    smooth_export_btn.disabled = False
    smooth_export_status.object = ok("Smoothed data ready to export.")
    apply_smooth_btn.disabled = False

preview_smooth_btn.on_click(_on_preview_smooth)

def _smoothed_export_bytes():
    if not state.smoothed_by_sample:
        smooth_export_status.object = warn("No smoothed data to export. Generate a preview first.")
        return io.BytesIO(b"")

    merged, reason = try_merge_same_time(state.smoothed_by_sample)
    if merged is not None:
        if not smooth_export_name.value.lower().endswith(".csv"):
            smooth_export_name.value = "smoothed_merged.csv"
        bio = io.BytesIO()
        merged.to_csv(bio, index=False)
        bio.seek(0)
        smooth_export_status.object = ok("Exporting merged smoothed CSV.")
        return bio

    if not smooth_export_name.value.lower().endswith(".zip"):
        smooth_export_name.value = "smoothed_csvs.zip"
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sample, df in state.smoothed_by_sample.items():
            zf.writestr(f"{sample}.csv", df.to_csv(index=False))
    bio.seek(0)
    smooth_export_status.object = warn(f"Time stamps differ; exporting per-sample CSVs as ZIP. Reason: {reason}")
    return bio

smooth_export_btn.callback = _smoothed_export_bytes

def _on_apply_smooth(_):
    if not state.smoothed_by_sample:
        preproc_status.object = warn("Generate a smoothing preview first.")
        return
    state.current_by_sample = {k: v.copy() for k, v in state.smoothed_by_sample.items()}
    preproc_status.object = ok("Smoothed data applied. Normalization is available.")
    _unlock_normalization_stage()

apply_smooth_btn.on_click(_on_apply_smooth)

smoothing_section = pn.Column(
    pn.pane.Markdown("### Smoothing"),
    pn.Row(preview_smooth_btn, pn.Spacer(width=12), smooth_win, smooth_poly),
    pn.Row(before_smooth_pane, pn.Spacer(width=12), after_smooth_pane, sizing_mode="stretch_width"),
    pn.Row(smooth_export_name, smooth_export_btn),
    smooth_export_status,
    pn.Row(apply_smooth_btn),
)
smoothing_section.visible = False

# --- Normalization (Box Select AUC) ---
# state for selection UI
_norm_scatter_sources: Dict[str, ColumnDataSource] = {}
_norm_area_sources: Dict[str, ColumnDataSource] = {}
_norm_sel_box: Optional[BoxAnnotation] = None

def _norm_clear_selection_state():
    _norm_scatter_sources.clear()
    _norm_area_sources.clear()

preview_norm_btn = pn.widgets.Button(name="Show normalization preview (select AUC)", button_type="primary", disabled=True)
apply_norm_btn   = pn.widgets.Button(name="Apply normalized data", button_type="success", disabled=True)
skip_norm_btn    = pn.widgets.Button(name="Skip normalization", button_type="warning", disabled=True)

norm_export_status = pn.pane.Markdown("", sizing_mode="stretch_width")
norm_export_name = pn.widgets.TextInput(name="Normalized filename", value="normalized_merged.csv", width=300)
norm_export_btn = pn.widgets.FileDownload(
    label="Export normalized",
    filename=norm_export_name.value,
    button_type="primary",
    embed=False,
    auto=False,
    callback=lambda: io.BytesIO(b""),
    disabled=True,
)
norm_export_name.param.watch(lambda e: setattr(norm_export_btn, "filename", e.new or "normalized_merged.csv"), "value")

norm_before_pane = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")
norm_after_pane  = pn.pane.Bokeh(height=420, sizing_mode="stretch_width")

def _norm_build_before_fig(samples_to_df: Dict[str, pd.DataFrame], title: str = "Select normalization peak (Box Select)") -> bokeh.plotting.Figure:
    n = len(samples_to_df)
    colors = (list(Category20[20])[:n]) if n <= 20 else [Turbo256[i] for i in np.linspace(0,255,n,dtype=int)]
    fig = bokeh.plotting.figure(
        title=title,
        height=400,
        sizing_mode="stretch_width",
        x_axis_label="time",
        y_axis_label="intensity",
        tools="pan,wheel_zoom,box_zoom,box_select,reset,save,hover",
        active_drag="box_select",
        active_scroll="wheel_zoom",
    )
    fig.legend.click_policy = "hide"
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None

    # selection box — GREY (not red)
    global _norm_sel_box
    _norm_sel_box = BoxAnnotation(
        left=None, right=None, fill_alpha=0.2, fill_color="gray",
        line_color=None, level="underlay"
    )
    fig.add_layout(_norm_sel_box)

    _norm_scatter_sources.clear()
    _norm_area_sources.clear()

    for i, (sample, df) in enumerate(samples_to_df.items()):
        t = df["time"].to_numpy()
        y = df["intensity"].to_numpy()
        fig.line(t, y, color=colors[i % len(colors)], legend_label=sample)

        # scatter to capture selections
        cds = ColumnDataSource(dict(x=t, y=y, s=[sample]*len(t)))
        fig.circle("x", "y", source=cds, size=4, alpha=0.001, line_alpha=0.001,
                   fill_color=colors[i % len(colors)], line_color=colors[i % len(colors)])
        _norm_scatter_sources[sample] = cds

        # area fill source (initially empty) — will show *only* the tallest peak AUC in red
        area_src = ColumnDataSource(dict(x=[], y=[]))
        fig.patch("x", "y", source=area_src, fill_alpha=0.35, fill_color="red", line_color=None)
        _norm_area_sources[sample] = area_src

    return fig


def _norm_compute_selected_xrange() -> Optional[Tuple[float, float]]:
    xs = []
    for cds in _norm_scatter_sources.values():
        sel = cds.selected.indices
        if sel:
            xs.extend(np.asarray(cds.data["x"])[sel].tolist())
    if not xs:
        return None
    return float(np.min(xs)), float(np.max(xs))

def _norm_update_sel_box(xr: Optional[Tuple[float,float]]):
    if _norm_sel_box is None:
        return
    if xr is None:
        _norm_sel_box.left = None
        _norm_sel_box.right = None
    else:
        _norm_sel_box.left, _norm_sel_box.right = xr[0], xr[1]

def _norm_on_selection_change(attr, old, new):
    xr = _norm_compute_selected_xrange()
    _norm_update_sel_box(xr)
    # clear fills on change
    for src in _norm_area_sources.values():
        src.data = dict(x=[], y=[])

def _norm_auc(x: np.ndarray, y: np.ndarray, xmin: float, xmax: float) -> float:
    m = (x >= xmin) & (x <= xmax)
    if not np.any(m):
        return np.nan
    xs = x[m]; ys = y[m]
    if xs.size < 2:
        return np.nan
    return float(np.trapz(ys, xs))

def _on_preview_norm(_):
    if not state.current_by_sample:
        preproc_status.object = warn("No input for normalization. Apply despiked/smoothed data first.")
        return
    # build before fig with scatter/areas & selection handlers
    before_fig = _norm_build_before_fig(state.current_by_sample)
    # attach selection watchers
    for cds in _norm_scatter_sources.values():
        cds.selected.on_change("indices", _norm_on_selection_change)

    norm_before_pane.object = before_fig
    norm_after_pane.object = None
    preproc_status.object = ok("Select the normalization peak range with Box Select, then click the button again to compute preview.")

    # If a range already selected, compute preview immediately
    _norm_preview_compute()

def _norm_preview_compute():
    xr = _norm_compute_selected_xrange()
    _norm_update_sel_box(xr)
    if xr is None:
        return  # wait for user selection

    xmin, xmax = xr
    state.norm_auc.clear()
    state.normalized_by_sample.clear()

    for sample, df in state.current_by_sample.items():
        x = df["time"].to_numpy()
        y = df["intensity"].to_numpy()

        # Restrict to selection window
        m = (x >= xmin) & (x <= xmax)
        if not np.any(m):
            _norm_area_sources[sample].data = dict(x=[], y=[])
            state.norm_auc[sample] = np.nan
            state.normalized_by_sample[sample] = df.copy()
            continue

        xs = x[m]
        ys = y[m]

        # Find peaks within the selection; choose the tallest by height
        peaks, _ = find_peaks(ys)
        if peaks.size == 0:
            # no peak detected: clear fill and keep original
            _norm_area_sources[sample].data = dict(x=[], y=[])
            state.norm_auc[sample] = np.nan
            state.normalized_by_sample[sample] = df.copy()
            continue

        # Tallest peak by amplitude
        tallest_idx = peaks[np.argmax(ys[peaks])]

        # Use base width bounds via prominence-based width at rel_height=1.0 (approx "to baseline")
        try:
            widths, left_ips, right_ips, _ = peak_widths(ys, [tallest_idx], rel_height=1.0)
            li = int(np.floor(left_ips[0]))
            ri = int(np.ceil(right_ips[0]))
        except Exception:
            # Fallback: expand to nearest local minima around the peak
            li = tallest_idx
            while li > 0 and ys[li-1] < ys[li]:
                li -= 1
            ri = tallest_idx
            while ri < ys.size - 1 and ys[ri+1] < ys[ri]:
                ri += 1

        # Clamp bounds to selection segment and ensure at least 2 points
        li = max(0, min(li, ys.size - 2))
        ri = max(li + 1, min(ri, ys.size - 1))

        xs_seg = xs[li:ri+1]
        ys_seg = ys[li:ri+1]

        # Compute AUC for this peak only
        auc = float(np.trapz(ys_seg, xs_seg)) if xs_seg.size >= 2 else np.nan
        state.norm_auc[sample] = auc

        # Draw red filled polygon under the chosen peak (baseline at 0)
        if xs_seg.size >= 2 and np.isfinite(auc) and auc > 0:
            poly_x = np.r_[xs_seg, xs_seg[::-1]]
            poly_y = np.r_[ys_seg, np.zeros_like(ys_seg)[::-1]]
            _norm_area_sources[sample].data = dict(x=poly_x, y=poly_y)
        else:
            _norm_area_sources[sample].data = dict(x=[], y=[])

        # Build normalized series (divide by this peak's AUC if valid)
        if np.isfinite(auc) and auc > 0:
            yy = y / auc
            state.normalized_by_sample[sample] = pd.DataFrame({"time": x, "intensity": yy})
        else:
            state.normalized_by_sample[sample] = df.copy()

    # After-plot: link to before
    after_fig = _plot_multi(state.normalized_by_sample, title="After normalization (intensity / peak AUC)")
    if isinstance(norm_before_pane.object, bokeh.plotting.Figure):
        after_fig.x_range = norm_before_pane.object.x_range
        after_fig.y_range = norm_before_pane.object.y_range
    norm_after_pane.object = after_fig

    apply_norm_btn.disabled = False
    norm_export_btn.disabled = False
    norm_export_status.object = ok("Normalization preview generated. You can export or apply.")


# Reuse the preview button to either create fig or (if existing + selection present) compute
def _on_preview_norm_click(_):
    if norm_before_pane.object is None:
        _on_preview_norm(_)
    else:
        _norm_preview_compute()
        if norm_after_pane.object is None:
            preproc_status.object = warn("Select a range with Box Select before computing AUC.")

preview_norm_btn.on_click(_on_preview_norm_click)

def _norm_export_bytes():
    if not state.normalized_by_sample:
        norm_export_status.object = warn("No normalized data to export. Generate a preview first.")
        return io.BytesIO(b"")

    merged, reason = try_merge_same_time(state.normalized_by_sample)
    if merged is not None:
        if not norm_export_name.value.lower().endswith(".csv"):
            norm_export_name.value = "normalized_merged.csv"
        bio = io.BytesIO()
        merged.to_csv(bio, index=False)
        bio.seek(0)
        norm_export_status.object = ok("Exporting merged normalized CSV.")
        return bio

    if not norm_export_name.value.lower().endswith(".zip"):
        norm_export_name.value = "normalized_csvs.zip"
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sample, df in state.normalized_by_sample.items():
            zf.writestr(f"{sample}.csv", df.to_csv(index=False))
    bio.seek(0)
    norm_export_status.object = warn(f"Time stamps differ; exporting per-sample CSVs as ZIP. Reason: {reason}")
    return bio

norm_export_btn.callback = _norm_export_bytes

def _on_apply_norm(_):
    if not state.normalized_by_sample:
        preproc_status.object = warn("Generate a normalization preview first.")
        return
    state.current_by_sample = {k: v.copy() for k, v in state.normalized_by_sample.items()}
    preproc_status.object = ok("Normalized data applied. Later tabs will use normalized traces.")

apply_norm_btn.on_click(_on_apply_norm)

def _on_skip_norm(_):
    # Keep current_by_sample as-is
    preproc_status.object = ok("Skipped normalization. Current dataset unchanged.")
    # Still allow export of 'current' as normalized if user wants — disable here:
    apply_norm_btn.disabled = True
    # No other action needed.

skip_norm_btn.on_click(_on_skip_norm)

normalization_section = pn.Column(
    pn.pane.Markdown("### Normalization (select a peak; AUC will be filled in red)"),
    pn.Row(preview_norm_btn, pn.Spacer(width=12), apply_norm_btn, pn.Spacer(width=12), skip_norm_btn),
    pn.Row(norm_before_pane, pn.Spacer(width=12), norm_after_pane, sizing_mode="stretch_width"),
    pn.Row(norm_export_name, norm_export_btn),
    norm_export_status,
)
normalization_section.visible = False

# ====================== Assemble PREPROCESS tab ======================
preproc_tab = pn.Column(
    pn.pane.Markdown("## 2) Preprocessing"),
    pn.pane.Markdown("### Despiking"),
    pn.Row(preview_despike_btn, pn.Spacer(width=12), window_slider, z_slider),
    pn.Row(before_despike_pane, pn.Spacer(width=12), after_despike_pane, sizing_mode="stretch_width"),
    pn.Row(despike_export_name, despike_export_btn),
    despike_export_status,
    pn.Row(apply_despike_btn),
    pn.layout.Divider(),
    smoothing_section,
    pn.layout.Divider(),
    normalization_section,
    preproc_status,
    sizing_mode="stretch_width",
)

# ====================== Other tabs (placeholders) ======================
alignment_tab = pn.Column(pn.pane.Markdown("## 3) Alignment (placeholder)"), sizing_mode="stretch_width")
nmf_tab = pn.Column(pn.pane.Markdown("## 4) NMF (placeholder)"), sizing_mode="stretch_width")
viz_tab = pn.Column(pn.pane.Markdown("## 5) Visualization (placeholder)"), sizing_mode="stretch_width")

# ====================== App shell ======================
TABS = pn.Tabs(
    ("Upload", upload_tab),
    ("Preprocess", preproc_tab),
    ("Alignment", alignment_tab),
    ("NMF", nmf_tab),
    ("Visualization", viz_tab),
    dynamic=True,
)
HEADER = pn.pane.Markdown("# CEtools — Electropherogram Pipeline (Prototype)", sizing_mode="stretch_width")
app = pn.Column(HEADER, TABS, sizing_mode="stretch_width")
app.servable(title="CEtools Pipeline")

if __name__ == "__main__":
    pn.serve(app, title="CEtools Pipeline", show=True)




