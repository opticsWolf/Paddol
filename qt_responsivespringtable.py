# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
import numpy as np
from typing import Any, Union, Literal, Optional
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QResizeEvent, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QTableView,
    QHeaderView,
    QVBoxLayout,
    QWidget
)

# ==========================================
# 1. The High-Performance Model
# ==========================================
class NumpyTableModel(QAbstractTableModel):
    """A high-performance Qt Table Model backed by a NumPy array.
    
    This model provides efficient read-only access to a 2D NumPy array
    and supports dynamic header updates and column sorting.
    """

    def __init__(self, data: np.ndarray, headers: list[str]) -> None:
        """Initializes the model.

        Args:
            data (np.ndarray): The 2D NumPy array containing the table data.
            headers (list[str]): Initial list of column header strings.
        """
        super().__init__()
        self._data = data
        self._headers = headers

    def set_headers(self, headers: list[str]) -> None:
        """Updates the column headers dynamically and notifies the view.

        Args:
            headers (list[str]): New list of header strings.
        """
        self._headers = headers
        # Notify the view that the header data has changed
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(headers) - 1)

    def rowCount(self, parent=QModelIndex()) -> int:
        """Returns the number of rows in the dataset.

        Args:
            parent (QModelIndex, optional): Parent index (unused).

        Returns:
            int: Number of rows.
        """
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()) -> int:
        """Returns the number of columns in the dataset.

        Args:
            parent (QModelIndex, optional): Parent index (unused).

        Returns:
            int: Number of columns.
        """
        return self._data.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Retrieves data for a specific index.

        Args:
            index (QModelIndex): The index of the item.
            role (int, optional): The data role. Defaults to DisplayRole.

        Returns:
            Any: The string representation of the data, or None.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data[index.row(), index.column()])
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Retrieves header data for columns or rows.

        Args:
            section (int): The row or column index.
            orientation (Qt.Orientation): Horizontal or Vertical.
            role (int, optional): The data role. Defaults to DisplayRole.

        Returns:
            Any: The header string or None.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                # Safety check for index out of bounds if headers list is short
                if section < len(self._headers):
                    return self._headers[section]
                return str(section)
            elif orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        return None

    def sort(self, column: int, order: Qt.SortOrder) -> None:
        """Sorts the data based on a specific column.

        Uses stable sorting. Falls back to string conversion if types are mixed
        or non-numeric.

        Args:
            column (int): The column index to sort by.
            order (Qt.SortOrder): Ascending or Descending.
        """
        self.layoutAboutToBeChanged.emit()
        col_data = self._data[:, column]
        try:
            indices = np.argsort(col_data, kind='stable')
        except TypeError:
            col_string = col_data.astype(str)
            indices = np.argsort(col_string, kind='stable')
        
        if order == Qt.SortOrder.DescendingOrder:
            indices = indices[::-1]
            
        self._data = self._data[indices]
        self.layoutChanged.emit()


# ==========================================
# 2. The Configurable Responsive View
# ==========================================
class ResponsiveSpringTableView(QTableView):
    """A customized QTableView with proportional resizing and strict width constraints.

    Features:
    - Proportional column resizing on window resize.
    - Strict min/max width enforcement during manual and auto resizing.
    - A "spring" column that occupies remaining space.
    - Batch configuration of headers and constraints.
    """

    def __init__(self, 
                 enable_sorting: bool = False, 
                 show_headers: bool = True,
                 show_row_labels: bool = True,
                 alternating_rows: bool = True,
                 alternate_color: Union[str, QColor, Literal["lighter", "darker"], None] = "lighter") -> None:
        """Initializes the view settings.

        Args:
            enable_sorting (bool): Enable column sorting.
            show_headers (bool): Show horizontal headers.
            show_row_labels (bool): Show vertical row numbers.
            alternating_rows (bool): Enable alternating row colors.
            alternate_color (Union[str, QColor...]): Color strategy for alternating rows.
        """
        super().__init__()
        
        # Internal storage for width constraints
        self._min_widths: dict[int, int] = {}
        self._max_widths: dict[int, int] = {}

        # 1. Styling
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(alternating_rows)
        if alternating_rows and alternate_color:
            self.set_alternating_color(alternate_color)
        
        # 2. Header Visibility
        self.horizontalHeader().setVisible(show_headers)
        self.verticalHeader().setVisible(show_row_labels)

        # 3. Resize Logic
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().sectionResized.connect(self.on_column_resized)
        
        # 4. Sorting
        if enable_sorting and show_headers:
            self.setSortingEnabled(True)

    def configure_columns(self, config: list[dict[str, Any]]) -> None:
        """Configures column headers and width constraints in batch.

        This method updates the model's headers and stores min/max width
        constraints for the view.

        Args:
            config (list[dict[str, Any]]): A list of dictionaries where each dict
                corresponds to a column (by index).
                Expected keys:
                    - "label" (str): The column header text.
                    - "min_width" (int): Minimum pixel width.
                    - "max_width" (int): Maximum pixel width.
        """
        # 1. Update Headers
        labels = []
        current_headers = self.model()._headers if hasattr(self.model(), "_headers") else []
        
        for i in range(self.model().columnCount()):
            if i < len(config) and "label" in config[i]:
                labels.append(config[i]["label"])
            elif i < len(current_headers):
                labels.append(current_headers[i])
            else:
                labels.append(str(i))
        
        if hasattr(self.model(), "set_headers"):
            self.model().set_headers(labels)

        # 2. Update Constraints
        self._min_widths.clear()
        self._max_widths.clear()

        for i, settings in enumerate(config):
            if "min_width" in settings:
                self._min_widths[i] = settings["min_width"]
            if "max_width" in settings:
                self._max_widths[i] = settings["max_width"]
        
        # Force layout update
        if self.isVisible():
            self.resizeEvent(QResizeEvent(self.size(), self.size()))

    def set_alternating_color(self, color_spec: Union[str, QColor, Literal["lighter", "darker"]]) -> None:
        """Sets the alternating row color based on a specification.

        Args:
            color_spec (Union[str, QColor, Literal["lighter", "darker"]]):
                The color or derivation mode to use.
        """
        if not self.alternatingRowColors(): return

        palette = self.palette()
        base_color = palette.color(QPalette.ColorRole.Base)
        target_color = None

        if color_spec == "lighter":
            target_color = self._derive_color(base_color, "lighter", 110)
        elif color_spec == "darker":
            target_color = self._derive_color(base_color, "darker", 105)
        elif isinstance(color_spec, (str, QColor)):
            target_color = QColor(color_spec)
        
        if target_color and target_color.isValid():
            palette.setColor(QPalette.ColorRole.AlternateBase, target_color)
            self.setPalette(palette)

    @staticmethod
    def _derive_color(base: QColor, mode: Literal["lighter", "darker"], factor: int = 125) -> QColor:
        """Derives a lighter or darker color from a base color.

        Args:
            base (QColor): Base color.
            mode (Literal["lighter", "darker"]): Derivation mode.
            factor (int): Intensity factor (default 125).

        Returns:
            QColor: The new color.
        """
        if mode == "lighter": return base.lighter(factor)
        elif mode == "darker": return base.darker(factor)
        return base

    def showEvent(self, event) -> None:
        """Handles the show event to trigger initial layout calculation."""
        super().showEvent(event)
        self.update_spring_column()

    def on_column_resized(self, index: int, old_size: int, new_size: int) -> None:
        """Handles manual user resizing with strict constraint enforcement.

        If the user attempts to resize beyond min/max bounds, this method
        forcibly resets the column width to the limit.

        Args:
            index (int): Column index.
            old_size (int): Previous width.
            new_size (int): New width attempted by user.
        """
        header = self.horizontalHeader()
        last_index = header.count() - 1

        # Ignore the spring column (it does what it wants)
        if index == last_index: 
            return

        # 1. Check Constraints
        min_w = self._min_widths.get(index, 30)
        max_w = self._max_widths.get(index, 999999)

        corrected_width = new_size
        
        # 2. Clamp
        if new_size < min_w:
            corrected_width = min_w
        elif new_size > max_w:
            corrected_width = max_w

        # 3. Enforce (if user went out of bounds)
        if corrected_width != new_size:
            # CRITICAL: Block signals to prevent infinite recursion loop
            header.blockSignals(True)
            self.setColumnWidth(index, corrected_width)
            header.blockSignals(False)

        # 4. Update the spring to fill the remaining space
        self.update_spring_column()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handles window resizing with proportional scaling and constraints.

        Scales all columns (except the spring) proportionally to the window
        width change, while ensuring no column violates its min/max width.

        Args:
            event (QResizeEvent): The resize event.
        """
        old_width = event.oldSize().width()
        new_width = event.size().width()
        super().resizeEvent(event)

        if new_width <= 0: return
        if old_width <= 0:
            self.update_spring_column()
            return

        header = self.horizontalHeader()
        count = header.count()
        ratio = new_width / old_width
        rounding_error = 0.0
        
        for i in range(count - 1):
            current_width = self.columnWidth(i)
            target_width = (current_width * ratio) + rounding_error
            new_width_int = int(round(target_width))
            
            # Constraints
            min_w = self._min_widths.get(i, 30)
            max_w = self._max_widths.get(i, 999999)
            
            if new_width_int < min_w:
                new_width_int = min_w
            elif new_width_int > max_w:
                new_width_int = max_w
            
            rounding_error = target_width - new_width_int
            if abs(rounding_error) > 10: rounding_error = 0.0
            
            header.blockSignals(True)
            self.setColumnWidth(i, new_width_int)
            header.blockSignals(False)

        self.update_spring_column()

    def update_spring_column(self) -> None:
        """Calculates and updates the width of the last 'spring' column.

        The spring column absorbs all remaining horizontal space in the viewport.
        """
        header = self.horizontalHeader()
        count = header.count()
        if count == 0: return
        last_index = count - 1
        
        used_width = sum(self.columnWidth(i) for i in range(count - 1))
        viewport_width = self.viewport().width()
        new_spring_width = max(0, viewport_width - used_width)
        
        header.blockSignals(True)
        self.setColumnWidth(last_index, new_spring_width)
        header.blockSignals(False)

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    
    def generate_data(rows: int) -> np.ndarray:
        """Generates mock data for testing."""
        data = np.empty((rows, 4), dtype=object)
        data[:, 0] = np.arange(rows)
        data[:, 1] = "Standard"
        data[:, 2] = np.random.randint(100, 999, size=rows)
        data[:, 3] = "Spring"
        return data

    def main() -> None:
        """Main entry point used to test the component."""
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        window = QWidget()
        window.setWindowTitle("Configurable Table with Max Width")
        window.resize(900, 500)
        
        layout = QVBoxLayout()
        
        # Initial Data
        data = generate_data(100)
        # Placeholder headers, will be overwritten by configure_columns
        model = NumpyTableModel(data, ["A", "B", "C", "D"])
        
        table = ResponsiveSpringTableView(
            enable_sorting=True,
            show_headers=True,
            show_row_labels=False,  
            alternating_rows=True,
            alternate_color="darker" #keywords: darker or lighter, hex colors e.g. #f0f0f0 or colornames e.g. darkgrey
        )
        
        table.setModel(model)
        
        # --- NEW: Configure Columns with List of Dicts ---
        column_config = [
            # Col 0: ID - Min 40, Max 80 (Cannot grow huge)
            {"label": "ID", "min_width": 40, "max_width": 80},
            
            # Col 1: Category - Min 100, No Max
            {"label": "Category", "min_width": 100},
            
            # Col 2: Value - Min 50, Max 100
            {"label": "Value", "min_width": 50, "max_width": 100},
            
            # Col 3: Description (Spring) - Constraints ignored
            {"label": "Description (Spring)", "min_width": 100, "max_width": 50} 
        ]
        
        table.configure_columns(column_config)
        
        # Set initial sizes to respect the config immediately
        table.setColumnWidth(0, 60) # Fits within 40-80
        table.setColumnWidth(1, 150)
        table.setColumnWidth(2, 80)
        
        layout.addWidget(table)
        window.setLayout(layout)
        window.show()
        
        sys.exit(app.exec())
    

    main()
