import panel as pn
from bokeh.plotting import figure
pn.extension()

p = figure()
p.line([1, 2, 3], [4, 5, 6])
pane = pn.pane.Bokeh(p)

checkbox = pn.widgets.Checkbox(name="Enable SVG mode")
def toggle_svg(event):
    p.output_backend = "svg" if event.new else "canvas"
    pane.param.trigger('object')

checkbox.param.watch(toggle_svg, 'value')
pn.Column(checkbox, pane).save("test_svg.html")
