# app/common_plot.py
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import bokeh.plotting
from bokeh.palettes import Category10, Turbo256
import panel as pn

# scrollable legend logic removed for simpler approach

def _palette(n: int):
    if n <= 10:
        return list(Category10[10])[:n]
    idxs = np.linspace(0, 255, num=n, dtype=int)
    return [Turbo256[i] for i in idxs]

def make_preview_plot(samples_to_df: Dict[str, pd.DataFrame], *, minutes: bool, offset: float, title: str, asinh: bool = False) -> bokeh.plotting.Figure:
    n_samples = len(samples_to_df)
    ncols = 6
    # 500px for the chart + 30px per legend row below it
    legend_rows = math.ceil(n_samples / ncols) if n_samples > 0 else 1
    total_height = 500 + legend_rows * 30
    
    colors = _palette(n_samples)
    p = bokeh.plotting.figure(
        title=title,
        height=total_height,
        sizing_mode="stretch_width",
        x_axis_label="time (min)" if minutes else "time (s)",
        y_axis_label="fluorescence (asinh)" if asinh else "fluorescence (raw)",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
    )
    
    for i, (sample, df) in enumerate(samples_to_df.items()):
        t = df["time"].to_numpy()
        y = df["intensity"].to_numpy()
        if asinh:
            y = np.arcsinh(y)
        y = y - i * offset
        p.line(x=t, y=y, color=colors[i % len(colors)], line_width=2, legend_label=str(sample)[:12])
    
    if p.legend:
        leg = p.legend[0]
        p.add_layout(leg, "below")
        leg.orientation = "horizontal"
        leg.location = "top_left"
        leg.click_policy = "hide"
        leg.label_text_font_size = "9pt"
        leg.ncols = ncols

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p

def plot_multi(samples_to_df: Dict[str, pd.DataFrame], title: str, *, xlab="time", ylab=None, offset=0.0, asinh=False, line_width=1, minutes=True) -> bokeh.plotting.Figure:
    n_samples = len(samples_to_df)
    if ylab is None:
        ylab = "fluorescence (asinh)" if asinh else "fluorescence (raw)"
    
    if xlab == "time":
        xlab = "time (min)" if minutes else "time (s)"

    ncols = 6
    # 500px for the chart + 30px per legend row below it
    legend_rows = math.ceil(n_samples / ncols) if n_samples > 0 else 1
    total_height = 500 + legend_rows * 30

    colors = _palette(n_samples)
    p = bokeh.plotting.figure(
        title=title,
        height=total_height,
        sizing_mode="stretch_width",
        x_axis_label=xlab, y_axis_label=ylab,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
    )
    for i, (sample, df) in enumerate(samples_to_df.items()):
        t = df["time"].to_numpy()
        y = df["intensity"].to_numpy()
        if asinh:
            y = np.arcsinh(y)
        y = y - i * offset
        p.line(t, y, color=colors[i % len(colors)], line_width=line_width, legend_label=str(sample)[:12])
    
    if p.legend:
        leg = p.legend[0]
        p.add_layout(leg, "below")
        leg.orientation = "horizontal"
        leg.location = "top_left"
        leg.click_policy = "hide"
        leg.label_text_font_size = "9pt"
        leg.ncols = ncols

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p
