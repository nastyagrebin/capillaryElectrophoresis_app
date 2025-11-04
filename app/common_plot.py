# app/common_plot.py

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import bokeh.plotting
from bokeh.palettes import Category10, Turbo256

def _palette(n: int):
    if n <= 10:
        return list(Category10[10])[:n]
    idxs = np.linspace(0, 255, num=n, dtype=int)
    return [Turbo256[i] for i in idxs]

def make_preview_plot(samples_to_df: Dict[str, pd.DataFrame], *, minutes: bool, offset: float, title: str) -> bokeh.plotting.Figure:
    colors = _palette(len(samples_to_df))
    p = bokeh.plotting.figure(
        title=title,
        height=400,
        sizing_mode="stretch_width",
        x_axis_label="time (min)" if minutes else "time (s)",
        y_axis_label="TIC (raw)",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        # active_scroll not set -> wheel zoom NOT active by default
        # active_drag not set -> default drag (no forced tool)
    )
    for i, (sample, df) in enumerate(samples_to_df.items()):
        t = df["time"].to_numpy()
        y = df["intensity"].to_numpy() - i * offset
        p.line(x=t, y=y, legend_label=sample, color=colors[i % len(colors)])
    p.legend.click_policy = "hide"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p

def plot_multi(samples_to_df: Dict[str, pd.DataFrame], title: str, *, xlab="time", ylab="intensity") -> bokeh.plotting.Figure:
    colors = _palette(len(samples_to_df))
    p = bokeh.plotting.figure(
        title=title,
        height=400,
        sizing_mode="stretch_width",
        x_axis_label=xlab, y_axis_label=ylab,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        # Do NOT set active_scroll
    )
    for i, (sample, df) in enumerate(samples_to_df.items()):
        p.line(df["time"], df["intensity"], color=colors[i % len(colors)], legend_label=sample)
    p.legend.click_policy = "hide"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p
