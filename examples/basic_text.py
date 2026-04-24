"""Demonstrates rendering text."""

import cmap

import scenex as snx

view = snx.View(
    scene=snx.Scene(
        children=[snx.Text(text="Hello, Scenex!", color=cmap.Color("cyan"), size=24)]
    ),
    camera=snx.Camera(),
)


# Show and position camera
canvas = snx.show(view)
ci = snx.CanvasInteractor(canvas)
ci.set_controller(view, snx.PanZoom())
snx.run()
