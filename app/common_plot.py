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

import param

class TitledPlotPane(pn.viewable.Viewer):
    object = param.Parameter(default=None)
    sizing_mode = param.String(default="stretch_width")
    
    def __init__(self, object=None, default_title="Plot", **params):
        super().__init__(object=object, **params)
        self.title_input = pn.widgets.TextInput(name="Plot Title", value=default_title, sizing_mode="stretch_width")
        self.bokeh_pane = pn.pane.Bokeh(object=self.object)
        self._layout = pn.Column(self.title_input, self.bokeh_pane, sizing_mode=self.sizing_mode)
        self.title_input.param.watch(self._update_title, "value")

    def __panel__(self):
        return self._layout

    def select(self, selector=None):
        return self._layout.select(selector)

    @param.depends("object", watch=True)
    def _update_object(self):
        self.bokeh_pane.object = self.object
        if self.object is not None:
            title = getattr(self.object, "title", None)
            # If the user hasn't typed anything yet, or we're initializing, use the plot's default
            if title and hasattr(title, "text"):
                if not getattr(self, "_user_edited", False) and self.title_input.value == "Plot":
                    self.title_input.value = title.text
            self._update_title()

    def _update_title(self, event=None):
        if event is not None:
            self._user_edited = True
        fig = self.bokeh_pane.object
        if fig and getattr(fig, "title", None) and hasattr(fig.title, "text"):
            fig.title.text = self.title_input.value
            
            # Sync the SaveTool filename with the current title
            if hasattr(fig, "tools"):
                for t in fig.tools:
                    if type(t).__name__ == "SaveTool":
                        t_str = self.title_input.value or "plot"
                        t.filename = t_str.replace(' ', '_').replace('/', '_')

def apply_export_prefix_to_pane(pane_obj, prefix: str, old_prefix: str = ""):
    if pane_obj is None: return
    if hasattr(pane_obj, "objects"):
        for child in pane_obj.objects:
            apply_export_prefix_to_pane(child, prefix, old_prefix)
    
    if isinstance(pane_obj, TitledPlotPane):
        val = pane_obj.title_input.value
        if old_prefix and val.startswith(f"{old_prefix}_"):
            pane_obj.title_input.value = f"{prefix}_{val[len(old_prefix)+1:]}"
        elif not val.startswith(f"{prefix}_"):
            pane_obj.title_input.value = f"{prefix}_{val}"
    elif isinstance(pane_obj, pn.widgets.FileDownload):
        if getattr(pane_obj, "_manually_prefixed", False):
            return
        if not hasattr(pane_obj, "_base_filename"):
            pane_obj._base_filename = pane_obj.filename or "export"
        pane_obj.filename = f"{prefix}_{pane_obj._base_filename}"
    elif isinstance(pane_obj, pn.widgets.TextInput) and any(x in (pane_obj.name or "").lower() for x in ["filename", "csv", "zip"]):
        val = pane_obj.value
        if old_prefix and val.startswith(f"{old_prefix}_"):
            pane_obj.value = f"{prefix}_{val[len(old_prefix)+1:]}"
        elif not val.startswith(f"{prefix}_"):
            pane_obj.value = f"{prefix}_{val}"
    elif hasattr(pane_obj, "object") and pane_obj.object is not None:
        fig = pane_obj.object
        if hasattr(fig, "tools"):
            for t in fig.tools:
                if type(t).__name__ == "SaveTool":
                    title = getattr(fig, "title", None)
                    t_str = title.text if title and hasattr(title, "text") else "plot"
                    if not t_str.startswith(prefix + "_"):
                        t.filename = f"{prefix}_{t_str.replace(' ', '_').replace('/', '_')}"
import colorcet as cc
import cmcrameri.cm as cmc

CURRENT_PALETTE_NAME = "glasbey"

def get_available_palettes() -> Dict[str, str]:
    # Hardcode some well-known divergent maps in cmcrameri
    divergent_bases = {"broc", "cork", "vik", "lisbon", "tofino", "berlin", "roma", "bam", "vanimo"}
    
    palettes = {}
    
    # 1. Add colorcet glasbey maps
    palettes["glasbey (Categorical)"] = "glasbey"
    palettes["glasbey_cool (Categorical)"] = "glasbey_cool"
    palettes["glasbey_warm (Categorical)"] = "glasbey_warm"
    
    # 2. Add cmcrameri maps (only continuous and divergent, no categorical)
    for name in dir(cmc):
        if name.startswith("_") or name.endswith("_r") or name.endswith("S"):
            continue
            
        base = name
        is_divergent = base in divergent_bases
        
        type_str = "Continuous"
        div_str = ", Divergent" if is_divergent else ""
        
        display_name = f"{name} ({type_str}{div_str})"
        palettes[display_name] = name
        
    # Sort the dictionary by display name
    return {k: palettes[k] for k in sorted(palettes.keys())}

def _palette(n: int, force_name: str = None):
    # Try to find the palette in cmcrameri first
    palette_data = None
    target_name = force_name or CURRENT_PALETTE_NAME
    
    if hasattr(cmc, target_name):
        cmap = getattr(cmc, target_name)
        import matplotlib.colors as mcolors
        import numpy as np
        
        # Categorical: use explicit distinct colors
        if target_name.endswith("S") and hasattr(cmap, "colors"):
            palette_data = [mcolors.to_hex(c) for c in cmap.colors]
        else:
            # Continuous: sample exactly n colors across the gradient
            if n == 1:
                return [mcolors.to_hex(cmap(0.5))]
            return [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, max(n, 2))]
            
    elif hasattr(cc, target_name):
        palette_data = getattr(cc, target_name)
        
    if palette_data is None:
        # Fallback to Category10 / Turbo256
        if n <= 10:
            palette_data = list(Category10[10])
        else:
            idxs = np.linspace(0, 255, num=n, dtype=int)
            return [Turbo256[i] for i in idxs]

    # For categorical maps, cycle if we have more samples than distinct colors
    return [palette_data[i % len(palette_data)] for i in range(n)]

def is_categorical(name: str = None) -> bool:
    name = name or CURRENT_PALETTE_NAME
    return name.startswith("glasbey") or name.endswith("S")

def is_divergent(name: str = None) -> bool:
    name = name or CURRENT_PALETTE_NAME
    divergent_bases = {"broc", "cork", "vik", "lisbon", "tofino", "berlin", "roma", "bam", "vanimo"}
    return name in divergent_bases

def get_continuous_palette(n: int = 256) -> List[str]:
    if is_categorical():
        return _palette(n, force_name="lajolla")
    return _palette(n)

def get_divergent_palette(n: int = 256) -> List[str]:
    if not is_divergent():
        return _palette(n, force_name="vik")
    return _palette(n)

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
