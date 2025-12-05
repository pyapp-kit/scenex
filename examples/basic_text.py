import cmap

import scenex as snx

view = snx.View(
    scene=snx.Scene(
        children=[snx.Text(text="Hello, Scenex!", color=cmap.Color("cyan"), size=24)]
    ),
    camera=snx.Camera(mouse=snx.PanZoomMouseStrategy(), interactive=True),
)


# Show and position camera
snx.use("pygfx")
snx.show(view)
snx.run()
