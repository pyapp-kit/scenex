import cmap
import numpy as np

import scenex as snx
from scenex.app.events import (
    Event,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
)
from scenex.utils.controllers import PanZoomController


# Create a more complex mesh - a grid of vertices
def create_grid_mesh(
    size: int = 10, spacing: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Create a grid mesh with given size and spacing."""
    vertices = []
    faces = []

    # Create vertices in a grid
    for i in range(size):
        for j in range(size):
            x = i * spacing
            y = j * spacing
            z = 0.0
            vertices.append([x, y, z])

    # Create triangular faces
    for i in range(size - 1):
        for j in range(size - 1):
            # Current vertex indices
            v0 = i * size + j
            v1 = i * size + (j + 1)
            v2 = (i + 1) * size + j
            v3 = (i + 1) * size + (j + 1)

            # Two triangles per grid square
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    return np.array(vertices), np.array(faces)


# Create the mesh
original_vertices, original_faces = create_grid_mesh(size=15, spacing=0.15)

mesh = snx.Mesh(
    vertices=original_vertices, faces=original_faces, color=cmap.Color("cyan")
)

view = snx.View(
    scene=snx.Scene(children=[mesh]),
    camera=snx.Camera(controller=PanZoomController(), interactive=True),
)


def event_filter(event: Event) -> bool:
    """Interactive mesh manipulation based on mouse events."""
    if isinstance(event, MouseMoveEvent):
        if intersections := event.world_ray.intersections(view.scene):
            # Find mesh intersection
            for node, _distance in intersections:
                if isinstance(node, snx.Mesh):
                    # Remove the intersected face
                    indices = [i for i, _d in node.intersecting_faces(event.world_ray)]
                    node.faces = np.delete(node.faces, indices, axis=0)
            return True
    elif isinstance(event, MousePressEvent):
        if event.buttons & MouseButton.LEFT:
            # Reset the mesh on click
            mesh.vertices = original_vertices.copy()
            mesh.faces = original_faces.copy()
            return True

    return False


# Set up the event filter
view.set_event_filter(event_filter)

# Show and position camera
snx.use("vispy")
snx.show(view)

print("Interactive Mesh Demo:")
print("- Move mouse over mesh to delete intersected faces")
print("- Left click to reset all faces")
print("- Use mouse to pan/zoom the camera")

snx.run()
