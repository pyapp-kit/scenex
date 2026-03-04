"""An interactive histogram."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cmap
import numpy as np
import numpy.typing as npt

import scenex as snx
from scenex.app import CursorType, app, events
from scenex.model._layout import fr, px
from scenex.utils import projections

if TYPE_CHECKING:
    from collections.abc import Sequence


def gaussian_dataset(
    n: int = 100000,
    mean: float = 32767.5,
    std: float = 80.0,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Generate a gaussian-distributed dataset clipped to the given dtype range."""
    if dtype is None:
        dtype = np.dtype(np.uint16)
    info = np.iinfo(dtype)
    data = np.random.normal(mean, std, n)
    return np.clip(data, info.min, info.max).astype(dtype)


def _calc_hist_bins(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxval = np.iinfo(data.dtype).max
    counts = np.bincount(data.flatten(), minlength=maxval + 1)
    bin_edges = np.arange(maxval + 2) - 0.5
    return counts, bin_edges


_AXIS = 40  # pixels reserved for each axis strip


class Histogram:
    """A simple interactive histogram view with adjustable clims and gamma."""

    def __init__(self) -> None:
        self._clims: tuple[float, float] = (0, 65535)
        self._gamma = 1.0
        self._grabbed: snx.Node | None = None

        self._containers: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._values: np.ndarray | None = None
        self._bins: np.ndarray | None = None
        self._log_base: float | None = None
        self._max_bin: float | None = None

        self.x_view = snx.View(
            scene=snx.Scene(name="x axis"),
            camera=snx.Camera(),
        )
        self.view = snx.View(
            scene=snx.Scene(name="main scene"),
            camera=snx.Camera(interactive=True),
        )
        self.y_view = snx.View(
            scene=snx.Scene(name="y axis"),
            camera=snx.Camera(),
        )
        # Create canvas early so it's available before set_data
        self.canvas = snx.Canvas(
            width=600,
            height=600,
            views=[self.x_view, self.y_view, self.view],
            visible=True,
        )

        # Lay out views using pixel-anchored strategies so resizing works correctly
        self.y_view.layout.x = px(0), px(_AXIS)
        self.y_view.layout.y = px(0), px(-_AXIS)

        self.x_view.layout.y = px(-_AXIS), fr(1)

        self.view.layout.x = px(_AXIS), fr(1)
        self.view.layout.y = px(0), px(-_AXIS)

        self._tick_objects: list[snx.Text] = []
        self._init_x_view()
        self._init_y_view()
        self._init_main_view()

    def _init_x_view(self) -> None:
        """Populate the x-axis view scene."""
        self.x_axis = snx.Line(
            vertices=np.array([[0, 0, 0], [1, 0, 0]]),
            width=2,
            color=snx.UniformColor(color=cmap.Color("white")),
        )
        self.x_view.scene.add_child(self.x_axis)

        # Pre-create 10 tick objects with line children (enough for min, max, and ticks)
        for _ in range(10):
            tick_line = snx.Line(
                vertices=np.array([[0, 0, 0], [0, -0.1, 0]]),
                width=1,
                color=snx.UniformColor(color=cmap.Color("white")),
                transform=snx.Transform().translated((0, 0.4, 0)),
            )
            tick_text = snx.Text(text="0", children=[tick_line], antialias=True)  # type: ignore
            self._tick_objects.append(tick_text)

    def _init_y_view(self) -> None:
        """Populate the y-axis view scene."""
        self.y_axis = snx.Line(
            # vertices=np.array([[0, 0.2, 0], [0, 0.8, 0]]),
            vertices=np.array([[0, 0, 0], [0, 1, 0]]),
            width=2,
            color=snx.UniformColor(color=cmap.Color("white")),
        )
        self.y_max = snx.Text(
            text="1", transform=snx.Transform().translated((-0.5, 0.95)), antialias=True
        )
        self.y_view.scene.add_child(self.y_axis)
        self.y_view.scene.add_child(self.y_max)

    def _init_main_view(self) -> None:
        """Populate the main histogram view scene and connect event handlers."""
        self.mesh = snx.Mesh(
            vertices=np.zeros((1, 3), dtype=np.float32),
            faces=np.zeros((1, 3), dtype=np.uint16),
            color=snx.UniformColor(color=cmap.Color("steelblue")),
            order=0,
        )

        self.highlight_line = snx.Line(
            vertices=np.array([[0, 0, 0], [0, 1, 0]]),
            width=2,
            color=snx.UniformColor(color=cmap.Color("yellow")),
            visible=False,  # Start hidden
        )

        # Split LUT line into three interactive components
        self.left_clim = snx.Line(
            name="left clim",
            interactive=True,
            order=1,
        )
        self.gamma_curve = snx.Line(
            name="gamma curve",
            interactive=False,
            order=1,
        )
        self.right_clim = snx.Line(
            name="right clim",
            interactive=True,
            order=1,
        )
        self.gamma_handle = snx.Points(
            name="gamma handle",
            vertices=np.array([[0.5, 0.5, 0]]),
            size=8,
            scaling="fixed",
            face_color=snx.UniformColor(color=cmap.Color("white")),
            edge_color=snx.UniformColor(color=cmap.Color("black")),
            interactive=True,
            order=2,
        )

        self._create_static_clim_lines()
        self._update_lut_line()

        self.controls = snx.Scene(
            name="controls scene",
            children=[
                self.left_clim,
                self.gamma_curve,
                self.right_clim,
                self.gamma_handle,
            ],
            interactive=True,
        )

        # Draw order (from bottom to top):
        # 0: histogram mesh
        self.mesh.order = 0
        self.view.scene.add_child(self.mesh)
        # 1: controls (clim lines, gamma curve, handle)
        self.controls.order = 1
        self.view.scene.add_child(self.controls)
        # 2: highlight line
        self.highlight_line.order = 2
        self.view.scene.add_child(self.highlight_line)

        # Set up event handlers and controllers
        self.view.camera.controller = snx.PanZoom(lock_y=True)

        self.view.camera.events.transform.connect(self._update_x_axis)
        self.view.camera.events.projection.connect(self._update_x_axis)
        self.canvas.events.width.connect(self._update_x_axis)
        self.view.set_event_filter(self._on_main_view)

        # self.set_clims(self._clims)

    def _on_main_view(self, event: events.Event) -> bool:
        if isinstance(event, events.MousePressEvent):
            intersections = [
                node
                for node, _dist in event.world_ray.intersections(self.controls)
                if node.interactive
            ]
            if len(intersections):
                self._grabbed = intersections[0]
                self.view.camera.interactive = False
        elif isinstance(event, events.MouseDoublePressEvent):
            intersections = [
                node
                for node, _dist in event.world_ray.intersections(self.controls)
                if node.interactive
            ]
            if self.gamma_handle in intersections:
                self.set_gamma(1.0)
        if isinstance(event, events.MouseMoveEvent):
            if self._grabbed is self.left_clim:
                # The left clim must stay to the left of the right clim
                new_left = min(event.world_ray.origin[0], self._clims[1])
                # ...and no less than the minimum value
                if self._bins is not None:
                    new_left = max(new_left, self._bins[0])
                # Set it
                self.set_clims((new_left, self._clims[1]))
            elif self._grabbed is self.right_clim:
                # The right clim must stay to the right of the left clim
                new_right = max(self._clims[0], event.world_ray.origin[0])
                # ...and no more than the minimum value
                if self._bins is not None:
                    new_right = min(new_right, self._bins[-1])
                # Set it
                self.set_clims((self._clims[0], new_right))
            elif self._grabbed is self.gamma_handle:
                # Set it
                self.set_gamma(-np.log2(event.world_ray.origin[1]))
            elif self._grabbed is None:
                intersections = [
                    node
                    for node, _dist in event.world_ray.intersections(self.controls)
                    if node.interactive
                ]
                if self.right_clim in intersections or self.left_clim in intersections:
                    app().set_cursor(self.canvas, CursorType.H_ARROW)
                elif self.gamma_handle in intersections:
                    app().set_cursor(self.canvas, CursorType.V_ARROW)
                else:
                    app().set_cursor(self.canvas, CursorType.DEFAULT)

        if isinstance(event, events.MouseReleaseEvent | events.MouseLeaveEvent):
            self._grabbed = None
            self.view.camera.interactive = True
        return False

    def set_clims(self, clims: tuple[float, float]) -> None:
        """Set the histogram clims."""
        self._clims = clims
        if self.controls is not None:
            self.controls.transform = (
                snx.Transform()
                .scaled((self._clims[1] - self._clims[0], 1, 1))
                .translated((self._clims[0], 0, 0))
            )

    def set_gamma(self, gamma: float) -> None:
        """Set the gamma."""
        self._gamma = gamma
        self._update_lut_line()

    def _create_static_clim_lines(self) -> None:
        """Create the static left and right clim lines that don't change with gamma."""
        # Left clim line (vertical line)
        left_x = np.array([0, 0, 0])
        left_y = np.array([1, 0.5, 0])
        left_z = np.zeros(3)
        if line := self.left_clim:
            line.vertices = np.column_stack((left_x, left_y, left_z))

        # Right clim line (vertical line)
        right_x = np.array([1, 1, 1])
        right_y = np.array([1, 0.5, 0])
        right_z = np.zeros(3)
        if line := self.right_clim:
            line.vertices = np.column_stack((right_x, right_y, right_z))

        # Color the clim lines
        dark_clim_color = cmap.Color((0.4, 0.4, 0.4))
        light_clim_color = cmap.Color((0.7, 0.7, 0.7))
        if line := self.left_clim:
            line.color = snx.VertexColors(
                color=[dark_clim_color, light_clim_color, dark_clim_color],
            )
        if line := self.right_clim:
            line.color = snx.VertexColors(
                color=[dark_clim_color, light_clim_color, dark_clim_color],
            )

    def _update_lut_line(self) -> None:
        """Updates the gamma curve vertices and colors."""
        if self.gamma_curve is None or self.gamma_handle is None:
            return

        npoints = 256
        # Gamma curve (non-interactive) - updates when gamma changes
        gamma_x = np.linspace(0, 1, npoints)
        gamma_y = np.linspace(0, 1, npoints) ** (
            self._gamma if self._gamma is not None else 1
        )
        gamma_z = np.zeros(npoints)
        self.gamma_curve.vertices = np.column_stack((gamma_x, gamma_y, gamma_z))

        # Gamma curve gets gradient colors
        gamma_colors = [
            cmap.Color(c)
            for c in np.linspace(0.2, 0.8, npoints).repeat(3).reshape(-1, 3)
        ]
        self.gamma_curve.color = snx.VertexColors(color=gamma_colors)
        self.gamma_handle.transform = snx.Transform().translated(
            (0, 0.5**self._gamma - 0.5)
        )

    def set_data(self, source: np.ndarray) -> None:
        """Set the histogram data."""
        values, bin_edges = _calc_hist_bins(source)
        uninitialized = self._values is None
        # Update the histogram mesh
        if self._values is None:
            self._values = np.copy(values)
            self._bins = np.copy(bin_edges)
            self._containers[id(source)] = (values, bin_edges)
        else:
            if id(source) in self._containers:
                old_values, _old_bins = self._containers[id(source)]
                self._values -= old_values

            self._values += values
            self._containers[id(source)] = (values, bin_edges)

        self._max_bin = np.max(self._values)
        if mesh := self.mesh:
            mesh.vertices, mesh.faces = self._hist_counts_to_mesh(
                self._values,
                self._bins,  # type: ignore
                False,
            )
        # Reapply log scaling if necessary
        if log := self._log_base:
            self._log_base = None
            self.set_log_base(log)

        # Rescale the y axis
        self._update_y_axis()

        if uninitialized:
            self.set_range()

    def _has_data(self) -> bool:
        if self.mesh is None:
            return False
        return bool(np.asarray(self.mesh.vertices).size)

    # ---- LutView interface implementations ----

    def set_log_base(self, base: float | None) -> None:
        """Sets the log base for the histogram y-axis."""
        if self.mesh is None:
            return

        old_log, new_log = self._log_base, base
        verts = np.zeros_like(self.mesh.vertices)
        verts[:, :] = self.mesh.vertices[:, :]
        if old_log is not None:
            verts[:, 1] = np.power(old_log, verts[:, 1]) - 1
        # use a count+1 histogram to gracefully handle 0, 1
        self._log_base = base
        if new_log is not None:
            verts[:, 1] = np.log(verts[:, 1] + 1) / np.log(new_log)
        # FIXME: Just telling scenex to refresh would be great
        verts[:, 0] = self.mesh.vertices[:, 0]
        self.mesh.vertices = verts

        self._update_y_axis()

    # ---- Viewable interface implementations ----

    def set_range(self) -> None:
        """Sets the range of the x axis."""
        projections.zoom_to_fit(self.view, "orthographic", zoom_factor=1)
        self.x_view.camera.projection = projections.orthographic(1, 1, 1)
        self.y_view.camera.projection = projections.orthographic(1, 1, 1)
        # FIXME: Vispy doesn't render the lines if they're on the edge.
        self.x_view.camera.transform = snx.Transform().translated((0.5, -0.499, 0))
        self.y_view.camera.transform = snx.Transform().translated((-0.499, 0.5, 0))

    def highlight(self, value: float | None) -> None:
        """Highlight a specific value on the histogram."""
        if self.highlight_line is None:
            return
        self.highlight_line.visible = value is not None
        self.highlight_line.transform = (
            self.highlight_line.transform
            if value is None
            else snx.Transform().translated((value, 0, 0))
        )

    def _calculate_tick_step(
        self, min_val: float, max_val: float, target_ticks: int = 5
    ) -> float:
        """Calculate a nice tick step for the given range."""
        from math import floor, log10

        if max_val <= min_val:
            return 1.0

        range_val = max_val - min_val
        approx_step = range_val / target_ticks

        # Find a "nice" step size
        power10 = 10.0 ** floor(log10(approx_step))
        for multiplier in [1.0, 2.0, 2.5, 5.0, 10.0]:
            step = multiplier * power10
            if step >= approx_step:
                return step

        return power10

    def _get_tick_positions(
        self, min_val: float, max_val: float, step: float
    ) -> list[float]:
        """Get tick positions within range, including min/max and culling overlaps."""
        from math import ceil, floor

        if step <= 0:
            return [min_val, max_val]

        # Calculate intermediate tick positions
        first_tick = ceil(min_val / step) * step
        last_tick = floor(max_val / step) * step

        intermediate_ticks: list[float] = []
        current = first_tick
        while current <= last_tick and len(intermediate_ticks) < 20:  # Safety limit
            intermediate_ticks.append(current)
            current += step

        # Filter out ticks that are too close to min/max to avoid overlap
        min_distance = step * 0.15  # Minimum distance between ticks
        filtered_ticks = []
        for tick_val in intermediate_ticks:
            if (
                abs(tick_val - min_val) >= min_distance
                and abs(tick_val - max_val) >= min_distance
            ):
                filtered_ticks.append(tick_val)

        # Always include min and max values, plus filtered intermediate ticks
        all_ticks = [min_val, *filtered_ticks, max_val]

        # Remove duplicates while preserving order
        seen = set()
        unique_ticks: list[float] = []
        for tick in all_ticks:
            if tick not in seen:
                seen.add(tick)
                unique_ticks.append(tick)

        return unique_ticks

    def _clear_ticks(self) -> None:
        """Remove all existing tick marks and labels from the scene."""
        for tick_obj in self._tick_objects:
            if tick_obj in self.x_view.scene.children:
                self.x_view.scene.remove_child(tick_obj)

    def _update_x_axis(self) -> None:
        # Update the x-axis labels based on the current camera projection
        cam = self.view.camera
        left, *_others = cam.transform.map(cam.projection.imap((-1, 0)))
        right, *_others = cam.transform.map(cam.projection.imap((1, 0)))

        # Clear existing ticks and labels
        self._clear_ticks()

        # Calculate tick positions (includes min/max and culling logic)
        tick_step = self._calculate_tick_step(left, right)
        unique_positions = self._get_tick_positions(left, right, tick_step)

        _x, _y, w, _h = self.canvas.rect_for(self.x_view)
        start = _AXIS / w

        # Use cached tick objects for all positions
        tick_idx = 0
        for tick_val in unique_positions:
            # Don't exceed our pre-created tick objects
            if tick_idx >= len(self._tick_objects):
                break

            # Calculate normalized position (0.1 to 0.95 maps to left to right)
            if right != left:
                norm_pos = start + (tick_val - left) / (right - left) * (1 - start)
            else:
                norm_pos = 0.5  # Default to center if no range

            # Reuse pre-created tick object
            tick_obj = self._tick_objects[tick_idx]
            tick_obj.text = f"{tick_val:.0f}"
            tick_obj.transform = snx.Transform().translated((norm_pos, -0.5, 0))

            # Add to scene
            self.x_view.scene.add_child(tick_obj)
            tick_idx += 1

    def _update_y_axis(self) -> None:
        if self.mesh is None or self.y_max is None:
            return

        max_val = self.mesh.bounding_box[1][1]
        # Scale the y-axis to [0, 1]
        self.mesh.transform = snx.Transform().scaled((1, 0.95 / max(max_val, 1), 1))
        # Resize the y-axis against the new data
        self.y_max.text = f"{max_val:.2f}"

    def _hist_counts_to_mesh(
        self,
        values: Sequence[float] | npt.NDArray,
        bin_edges: Sequence[float] | npt.NDArray,
        vertical: bool = False,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
        """Convert histogram counts to mesh vertices and faces for plotting."""
        n_edges = len(bin_edges)
        X, Y = (1, 0) if vertical else (0, 1)

        #   4-5
        #   | |
        # 1-2/7-8
        # |/| | |
        # 0-3-6-9
        # construct vertices
        # TODO: Reusing the arrays would be nice.
        vertices = np.zeros((3 * n_edges - 2, 3), np.float32)
        vertices[:, X] = np.repeat(bin_edges, 3)[1:-1]
        vertices[1::3, Y] = values
        vertices[2::3, Y] = values
        vertices[vertices == float("-inf")] = 0

        # construct triangles
        faces = np.zeros((2 * n_edges - 2, 3), np.uint32)
        offsets = 3 * np.arange(n_edges - 1, dtype=np.uint32)[:, np.newaxis]
        faces[::2] = np.array([0, 2, 1]) + offsets
        faces[1::2] = np.array([2, 0, 3]) + offsets
        return vertices, faces


histogram = Histogram()
data = gaussian_dataset(n=10000)
histogram.set_data(data)
histogram.set_clims((data.min(), data.max()))
snx.show(histogram.canvas)
histogram.set_range()
snx.run()
