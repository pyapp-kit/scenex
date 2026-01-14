from __future__ import annotations

from pydantic import ConfigDict, Field

from ._evented_list import EventedList
from ._layout import Layout
from ._view import View  # noqa: TC001


class GridAssignment(Layout):
    """Assignment of a view to a specific position in a grid layout.

    A GridAssignment associates a View with a grid cell location, specifying its
    row, column, and optional spanning across multiple cells. This is used internally
    by the Grid to manage view placement.

    Attributes
    ----------
    view : View
        The view being assigned to this grid position.
    row : int
        The starting row index (0-based) where the view is placed.
    col : int
        The starting column index (0-based) where the view is placed.
    rowspan : int
        The number of rows the view spans (default 1).
    colspan : int
        The number of columns the view spans (default 1).
    """

    view: View
    row: int = Field(default=0, description="The starting row index (0-based)")
    col: int = Field(default=0, description="The starting column index (0-based)")
    rowspan: int = Field(default=1, description="Number of rows the view spans")
    colspan: int = Field(default=1, description="Number of columns the view spans")

    model_config = ConfigDict(extra="forbid")


class Grid(Layout):
    """A flexible grid layout system for arranging views.

    The Grid divides a rectangular area into rows and columns, allowing views to be
    positioned at specific grid locations. Row and column sizes are specified as
    relative weights, which are normalized to fit the available space. Views can span
    multiple rows and columns.

    Attributes
    ----------
    grid : EventedList[GridAssignment]
        The list of view assignments in this grid. Each assignment specifies a view
        and its position (row, col, rowspan, colspan).
    row_sizes : tuple[float, ...]
        Relative weights determining the height of each row. Each row's height is
        proportional to its weight divided by the sum of all row weights. If empty,
        rows are equally sized.
    col_sizes : tuple[float, ...]
        Relative weights determining the width of each column. Each column's width is
        proportional to its weight divided by the sum of all column weights. If empty,
        columns are equally sized.

    Examples
    --------
    Create a grid with two views side-by-side with equal widths:
        >>> grid = Grid()
        >>> view1 = View()
        >>> view2 = View()
        >>> grid.add(view1, row=0, col=0)
        >>> grid.add(view2, row=0, col=1)

    Create a grid with custom column sizes (2:1 ratio):
        >>> grid = Grid(col_sizes=(2.0, 1.0))
        >>> grid.add(view1, row=0, col=0)
        >>> grid.add(view2, row=0, col=1)

    Create a view spanning multiple cells:
        >>> main_view = View()
        >>> side_view = View()
        >>> grid = Grid()
        >>> grid.add(main_view, row=0, col=0, rowspan=2, colspan=2)
        >>> grid.add(side_view, row=0, col=2)
    """

    grid: EventedList[GridAssignment] = Field(
        default_factory=EventedList,
        description="List of view assignments specifying view positions in the grid",
    )

    row_sizes: tuple[float, ...] = Field(
        default_factory=tuple,
        description="Relative weights for row heights (proportional to weight sum)",
    )
    col_sizes: tuple[float, ...] = Field(
        default_factory=tuple,
        description="Relative weights for column widths (proportional to weight sum)",
    )

    def add(
        self,
        view: View,
        row: int | None = None,
        col: int | None = None,
        rowspan: int = 1,
        colspan: int = 1,
    ) -> None:
        """Add a view to the grid at the specified position."""
        if row is None and col is None:
            row = 0
            col = 0
        if row is None:
            views_in_col = [a for a in self.grid if a.col == col]
            if len(views_in_col) == 0:
                row = 0
            else:
                row = max(a.row + a.rowspan for a in views_in_col)
        if col is None:
            views_in_row = [a for a in self.grid if a.row == row]
            if len(views_in_row) == 0:
                col = 0
            else:
                col = max(a.col + a.colspan for a in views_in_row)

        # Ensure row_sizes length - pad additional rows with previous average size.
        # e.g. if there were 5 rows and the addition requires 7 rows, 2 more rows will
        #     be added with the average of the existing 5 row sizes.
        # This is done first so that the grid layout computation that happens on
        # grid.append can use the correct row/col sizes.
        if len(self.row_sizes) < row + rowspan:
            avg_row_size = (
                sum(self.row_sizes) / len(self.row_sizes) if self.row_sizes else 1.0
            )
            for _ in range(len(self.row_sizes), row + rowspan):
                self.row_sizes += (avg_row_size,)
        # Ensure col_sizes length - pad additional columns with previous average size.
        if len(self.col_sizes) < col + colspan:
            avg_col_size = (
                sum(self.col_sizes) / len(self.col_sizes) if self.col_sizes else 1.0
            )
            for _ in range(len(self.col_sizes), col + colspan):
                self.col_sizes += (avg_col_size,)

        # Add the assignment (and recompute the layout)
        self.grid.append(
            GridAssignment(
                view=view, row=row, col=col, rowspan=rowspan, colspan=colspan
            )
        )
