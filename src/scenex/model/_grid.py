from __future__ import annotations

from pydantic import ConfigDict, Field

from ._evented_list import EventedList
from ._layout import Layout
from ._view import View  # noqa: TC001


class GridAssignment(Layout):
    """Assignment of a view to a grid cell (lightweight Evented model).

    This holds the reference to the view and its grid placement.
    """

    view: View
    row: int = Field(default=0)
    col: int = Field(default=0)
    rowspan: int = Field(default=1)
    colspan: int = Field(default=1)

    model_config = ConfigDict(extra="forbid")


class Grid(Layout):
    """Grid layout.

    The grid divides the layout's content rectangle into `rows` x `cols` cells.
    Row and column sizes are specified as positive numbers which are interpreted
    as weights. The final pixel size for each row/column is computed by
    normalizing the weights to the available content size. If `row_sizes` or
    `col_sizes` is empty, the rows/columns are treated as equally weighted.

    Views can be assigned to grid cells via `assignments`, using `GridAssignment`
    objects. Spanning is supported via `rowspan` and `colspan`.
    """

    grid: EventedList[GridAssignment] = Field(default_factory=EventedList)

    row_sizes: tuple[float, ...] = Field(
        default_factory=tuple,
        description="""
            Weights forming the height of each row in the grid.
            Each row i will have height=(row_sizes[i] / sum(row_sizes) * total_height)
        """,
    )
    col_sizes: tuple[float, ...] = Field(
        default_factory=tuple,
        description="""
            Weights forming the width of each column in the grid.
            Each column i will have width=(col_sizes[i] / sum(col_sizes) * total_width)
        """,
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
