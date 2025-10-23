from __future__ import annotations

from pydantic import ConfigDict, Field, PrivateAttr, computed_field

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

    # Private attributes for row and column sizes
    # Allows for setting via EventedList or regular list
    _col_sizes: EventedList[float] = PrivateAttr(default_factory=EventedList)
    _row_sizes: EventedList[float] = PrivateAttr(default_factory=EventedList)

    @computed_field()  # type: ignore
    @property
    def row_sizes(self) -> EventedList[float]:
        """Weights forming the height of each row in the grid.

        Each row i will have height=(row_sizes[i] / sum(row_sizes) * total_height)
        """
        return self._row_sizes

    @row_sizes.setter
    def row_sizes(self, value: list[float] | EventedList) -> None:
        if isinstance(value, EventedList):
            self._row_sizes = value
        self._row_sizes = EventedList(value)

    @computed_field()  # type: ignore
    @property
    def col_sizes(self) -> EventedList[float]:
        """Weights forming the width of each column in the grid.

        Each column i will have width=(col_sizes[i] / sum(col_sizes) * total_width)
        """
        return self._col_sizes

    @col_sizes.setter
    def col_sizes(self, value: list[float] | EventedList) -> None:
        if isinstance(value, EventedList):
            self._col_sizes = value
        self._col_sizes = EventedList(value)

    def add(
        self,
        view: View,
        row: int | None = None,
        col: int | None = None,
        rowspan: int = 1,
        colspan: int = 1,
    ) -> None:
        """Add a view to the grid at the specified position."""
        # Ensure row_sizes length - pad additional rows with previous average size.
        # e.g. if there were 5 rows and the addition requires 7 rows, 2 more rows will
        #     be added with the average of the existing 5 row sizes.
        # This is done first so that the grid layout computation that happens on
        # grid.append can use the correct row/col sizes.
        if row is None:
            row = 0
        if col is None:
            col = len(self.col_sizes)

        if len(self.row_sizes) < row + rowspan:
            avg_row_size = (
                sum(self.row_sizes) / len(self.row_sizes) if self.row_sizes else 1.0
            )
            for _ in range(len(self.row_sizes), row + rowspan):
                self.row_sizes.append(avg_row_size)
        # Ensure col_sizes length - pad additional columns with previous average size.
        if len(self.col_sizes) < col + colspan:
            avg_col_size = (
                sum(self.col_sizes) / len(self.col_sizes) if self.col_sizes else 1.0
            )
            for _ in range(len(self.col_sizes), col + colspan):
                self.col_sizes.append(avg_col_size)

        # Add the assignment (and recompute the layout)
        self.grid.append(
            GridAssignment(
                view=view, row=row, col=col, rowspan=rowspan, colspan=colspan
            )
        )
