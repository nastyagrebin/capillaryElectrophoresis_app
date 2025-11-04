from __future__ import annotations
from typing import Optional, Tuple, Literal
import numpy as np

__all__ = ["remove_single_point_spikes"]

def despike_singletons(
    y: np.ndarray,
    *,
    x: Optional[np.ndarray] = None,
    window: int = 11,
    zscore_thresh: float = 6.0,
    neighbor_ratio: float = 0.6,
    neighbor_gap_factor: float = 4.0,
    max_iter: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove isolated 1-sample spikes (detector glitches) from a 1D trace.

    A point i is flagged as a spike if ALL hold:
      1) Strict local maximum: y[i] > y[i-1] and y[i] > y[i+1]
      2) Robust z-score vs local median >= zscore_thresh
      3) y[i] - mean(y[i-1], y[i+1]) >= neighbor_gap_factor * MAD_i
      4) max(y[i-1], y[i+1]) <= y[i] * (1 - neighbor_ratio)

    Replacement: linear interpolation between immediate finite neighbors (in x if provided;
    otherwise average of neighbors). Falls back to local median if neighbors are invalid.

    Parameters
    ----------
    y : np.ndarray
        1D signal.
    x : np.ndarray, optional
        1D coordinates (same length as y). Only used to interpolate linearly in x.
    window : int, default 11
        Odd window length for local median/MAD (>=5).
    zscore_thresh : float, default 6.0
        Robust z-score threshold (higher = stricter).
    neighbor_ratio : float, default 0.6
        Each neighbor must be at least this fraction below the spike (0.6 => ≥60% lower).
    neighbor_gap_factor : float, default 4.0
        Required gap, in MAD units, between spike and mean(neighbors).
    max_iter : int, default 1
        Number of passes (use >1 if multiple isolated glitches remain after first pass).

    Returns
    -------
    y_clean : np.ndarray
        Copy of y with spikes replaced.
    spike_idx : np.ndarray
        Sorted indices that were replaced.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return y.copy(), np.array([], dtype=int)

    if x is not None:
        x = np.asarray(x, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x must have the same shape as y.")

    if window < 5 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 5.")
    half = window // 2

    def _rolling_median_mad(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        med = np.empty_like(arr)
        mad = np.empty_like(arr)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            w = arr[lo:hi]
            w = w[np.isfinite(w)]
            if w.size == 0:
                med[i] = np.nan
                mad[i] = np.nan
            else:
                m = np.median(w)
                med[i] = m
                mad[i] = np.median(np.abs(w - m))
        mad = np.where(np.isfinite(mad) & (mad > 0), mad, np.nan)
        return med, mad

    def _interp_between_neighbors(i: int, yc: np.ndarray) -> float:
        left = i - 1
        right = i + 1
        while left >= 0 and not np.isfinite(yc[left]):
            left -= 1
        while right < n and not np.isfinite(yc[right]):
            right += 1
        if left >= 0 and right < n:
            if x is None:
                return 0.5 * (yc[left] + yc[right])  # minimal bias
            dx = (x[i] - x[left]) / (x[right] - x[left])
            return yc[left] + dx * (yc[right] - yc[left])
        # fallback: local median
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = yc[lo:hi]
        w = w[np.isfinite(w)]
        return float(np.median(w)) if w.size else yc[i]

    y_clean = y.copy()
    all_spikes: list[int] = []

    for _ in range(int(max_iter)):
        med, mad = _rolling_median_mad(y_clean)
        scale = 1.4826 * mad  # MAD->sigma
        z = (y_clean - med) / np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)

        idx = np.arange(1, n - 1)
        y_im1 = y_clean[idx - 1]
        y_i   = y_clean[idx]
        y_ip1 = y_clean[idx + 1]

        finite_triplet = np.isfinite(y_im1) & np.isfinite(y_i) & np.isfinite(y_ip1)
        local_max = (y_i > y_im1) & (y_i > y_ip1)
        high_z = z[idx] >= zscore_thresh

        mean_neighbors = 0.5 * (y_im1 + y_ip1)
        gap_ok = (y_i - mean_neighbors) >= (neighbor_gap_factor * scale[idx])
        rel_ok = np.maximum(y_im1, y_ip1) <= (y_i * (1.0 - neighbor_ratio))

        mask = finite_triplet & local_max & high_z & gap_ok & rel_ok
        spike_idx = idx[mask]
        if spike_idx.size == 0:
            break

        for i in spike_idx:
            y_clean[i] = _interp_between_neighbors(i, y_clean)

        all_spikes.extend(spike_idx.tolist())

    all_spikes = np.array(sorted(set(all_spikes)), dtype=int)
    return y_clean, all_spikes

def remove_single_point_spikes(
    y: np.ndarray,
    *,
    window: int = 5,
    z_thresh: float = 6.0,
    replace: Literal["median", "interp"] = "median",
) -> np.ndarray:
    """
    Remove isolated one-sample spikes using robust z-scores against a rolling median.
    window: odd >= 3; z_thresh: threshold on |z|; replace: 'median' or 'interp'
    """
    x = np.asarray(y, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()
    w = int(window)
    if w < 3 or w % 2 == 0:
        w = max(3, w | 1)  # force odd

    half = w // 2
    xp = np.pad(x, (half, half), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(xp, w)
    med = np.median(windows, axis=1)

    resid = x - med
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    z = resid / (1.4826 * mad)

    spikes = np.abs(z) >= float(z_thresh)
    y_out = x.copy()
    if not np.any(spikes):
        return y_out

    if replace == "median":
        y_out[spikes] = med[spikes]
    else:
        idx = np.where(spikes)[0]
        for i in idx:
            if 0 < i < n - 1:
                y_out[i] = 0.5 * (y_out[i - 1] + y_out[i + 1])
            else:
                y_out[i] = med[i]
    return y_out