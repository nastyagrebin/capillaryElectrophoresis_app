import pandas as pd
import numpy as np
import panel as pn
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, LogColorMapper, LinearColorMapper, ColorBar
from bokeh.palettes import Reds256

K = 16
lo, hi = 0.1, 1.0
pseudotimes = np.linspace(lo, hi, K)
q_values = np.random.rand(K) * 0.01

pt_min, pt_max = pseudotimes.min(), pseudotimes.max()
pt_spacing = (pt_max - pt_min) / max(1, K - 1) if K > 1 else 0.1
pt_margin = pt_spacing / 2

sig_src = ColumnDataSource(data=dict(
    basis_index=np.arange(1, K + 1),
    q_value=q_values,
    q_value_plot=np.clip(q_values, 1e-10, 1.0),
    pseudotime=pseudotimes,
))

# Decide mapper
max_q, min_q = np.max(q_values), np.min(q_values)
if min_q > 0 and (max_q / min_q) > 100:
    cmap = LogColorMapper(palette=Reds256, low=max(1e-10, min_q), high=1.0)
else:
    cmap = LinearColorMapper(palette=Reds256, low=0, high=np.max(q_values))

sig_fig = figure(
    width=800, height=200,
    title="NMF Basis Significance",
    x_axis_label="Pseudotime",
    tools="hover,save,pan,wheel_zoom,box_zoom,reset",
    toolbar_location="above",
    active_scroll=None,
    x_range=Range1d(pt_min - pt_margin, pt_max + pt_margin),
    y_range=Range1d(-1, 1)
)
sig_fig.yaxis.visible = False
sig_fig.ygrid.visible = False

sig_fig.extra_x_ranges = {"basis": Range1d(start=0.5, end=K + 0.5)}
basis_axis = LinearAxis(x_range_name="basis", axis_label="Basis Number")
sig_fig.add_layout(basis_axis, 'above')

sig_fig.rect(
    x="pseudotime", y=0, width=pt_spacing * 0.9, height=1.8,
    source=sig_src,
    fill_color={"field": "q_value_plot", "transform": cmap},
    line_color="lightgrey",
    line_width=1
)
cbar = ColorBar(color_mapper=cmap, title="q-value", orientation="horizontal", padding=0)
sig_fig.add_layout(cbar, 'below')

# Reconstruction
recon_fig = figure(
    width=800, height=250,
    title="Sample Reconstruction",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    x_axis_label="Pseudotime", y_axis_label="Intensity (asinh)",
    x_range=sig_fig.x_range,
    active_scroll=None
)

t_eval = np.linspace(lo, hi, 1000)
yhat = np.arcsinh(np.sin(t_eval * 10) * 10 + 10)
recon_src = ColumnDataSource(dict(t=t_eval, y=yhat))
recon_fig.line('t', 'y', source=recon_src, line_color="black", line_width=2)

pn.Column(recon_fig, sig_fig).save('test.html')
