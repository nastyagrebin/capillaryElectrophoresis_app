# ============================
# FILE: app/upload_utils.py
# ============================
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re
import tempfile

import numpy as np
import pandas as pd

# ---- sanitization ----
def sanitize_name(s: str) -> str:
    s = Path(s).stem
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s or "untitled"

# ---- CDF → DataFrame (scalar timing) ----
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
        n = y.size if y.size != n_expected else n_expected
        t = t0 + dt * np.arange(n, dtype=float)
        if prefer_minutes:
            t = t / 60.0

        n2 = min(t.size, y.size)
        t, y = t[:n2], y[:n2]
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

# ---- Merge helper (only when identical timestamps) ----
def try_merge_same_time(
    dfs_by_sample: Dict[str, pd.DataFrame],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> Tuple[Optional[pd.DataFrame], str]:
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
