# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
import numpy as np
from typing import Any, Union, Literal, Optional
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QMimeData, QByteArray, QDataStream, QIODevice
from PySide6.QtGui import QResizeEvent, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QTableView,
    QHeaderView,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QAbstractItemView,
    QLabel,
    QFrame
)

# ==========================================
# 1. The High-Performance Model (UNCHANGED)
# ==========================================
class NumpyTableModel(QAbstractTableModel):
    """
    A high-performance Qt Table Model backed by a NumPy array.
    Supports sorting, row reordering via drag-and-drop, and efficient data access.
    """
    def __init__(self, data: np.ndarray, headers: list[str]) -> None:
        """
        Initializes the model with data and headers.

        Args:
            data (np.ndarray): The 2D numpy array containing table data.
            headers (list[str]): A list of string headers corresponding to columns.
        """
        super().__init__()
        self._data = data
        self._headers = headers

    def set_headers(self, headers: list[str]) -> None:
        """
        Updates the column headers and emits the headerDataChanged signal.

        Args:
            headers (list[str]): The new list of column headers.
        """
        self._headers = headers
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(headers) - 1)

    def rowCount(self, parent=QModelIndex()) -> int:
        """Returns the number of rows in the dataset."""
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()) -> int:
        """Returns the number of columns in the dataset."""
        return self._data.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Returns data for a specific index if the role is DisplayRole."""
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data[index.row(), index.column()])
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Returns the header label for the given section (row or column)."""
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if section < len(self._headers):
                    return self._headers[section]
                return str(section)
            elif orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        return None

    def reorder_rows(self, new_indices: list[int] | np.ndarray) -> None:
        """
        Reorders the internal data array based on a list of new indices.

        Args:
            new_indices (list[int] | np.ndarray): An array of integers representing
                the new order of rows. Must match current row count.

        Raises:
            ValueError: If the length of new_indices does not match the row count.
        """
        if len(new_indices) != self.rowCount():
            raise ValueError(f"Index length {len(new_indices)} does not match row count.")
        self.layoutAboutToBeChanged.emit()
        idx_array = np.array(new_indices, dtype=int)
        self._data = self._data[idx_array]
        self.layoutChanged.emit()

    def sort(self, column: int, order: Qt.SortOrder) -> None:
        """
        Sorts the NumPy array based on the specified column and order.

        This method attempts to sort the data natively using the column's data type.
        If a TypeError occurs (mixed types), it falls back to converting the column
        to strings and sorting lexicographically. Uses a stable sort algorithm.

        Args:
            column (int): The index of the column to sort by.
            order (Qt.SortOrder): Ascending or Descending order.
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

    def mimeTypes(self) -> list[str]:
        """Returns the MIME types allowed for drag and drop operations."""
        return ["application/x-paddol-row"]

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """
        Returns item flags for the given index.
        
        Enables Drag, Drop, Selection, and general availability if the index is valid.
        Returns only DropEnabled if the index is invalid (dropping onto empty space).
        """
        if not index.isValid():
            return Qt.ItemFlag.ItemIsDropEnabled
        return (Qt.ItemFlag.ItemIsDragEnabled | 
                Qt.ItemFlag.ItemIsDropEnabled | 
                Qt.ItemFlag.ItemIsSelectable | 
                Qt.ItemFlag.ItemIsEnabled)

    def supportedDropActions(self) -> Qt.DropAction:
        """Returns the supported drop actions (MoveAction only)."""
        return Qt.DropAction.MoveAction

    def mimeData(self, indexes: list[QModelIndex]) -> QMimeData:
        """
        Encodes the row index of the dragged item into MIME data.

        Encodes the row integer into a QByteArray using QDataStream for
        the custom MIME type 'application/x-paddol-row'.

        Args:
            indexes (list[QModelIndex]): List of indices being dragged.

        Returns:
            QMimeData: The data containing the source row index.
        """
        mime = QMimeData()
        encoded_data = QByteArray()
        stream = QDataStream(encoded_data, QIODevice.OpenModeFlag.WriteOnly)
        if indexes:
            stream.writeInt32(indexes[0].row())
        mime.setData("application/x-paddol-row", encoded_data)
        return mime

    def dropMimeData(self, data: QMimeData, action: Qt.DropAction, row: int, column: int, parent: QModelIndex) -> bool:
        """
        Handles the dropping of data to reorder rows within the NumPy array.

        Decodes the source row from MIME data, calculates the destination row,
        and performs a NumPy insertion/deletion to move the row. Emits
        beginMoveRows and endMoveRows to notify the view of the change.

        Args:
            data (QMimeData): The source data.
            action (Qt.DropAction): The action type (Move, Copy, etc.).
            row (int): The drop row ( -1 if dropped on a parent).
            column (int): The drop column.
            parent (QModelIndex): The parent index.

        Returns:
            bool: True if the drop was successful, False otherwise.
        """
        if not data.hasFormat("application/x-paddol-row"): return False
        if action == Qt.DropAction.IgnoreAction: return True

        encoded_data = data.data("application/x-paddol-row")
        stream = QDataStream(encoded_data, QIODevice.OpenModeFlag.ReadOnly)
        src_row = stream.readInt32()
        
        dst_row = row
        if dst_row == -1:
            dst_row = parent.row() if parent.isValid() else self.rowCount()

        if src_row == dst_row: return False

        self.beginMoveRows(QModelIndex(), src_row, src_row, QModelIndex(), dst_row)
        row_data = self._data[src_row].copy()
        self._data = np.delete(self._data, src_row, axis=0)
        insert_idx = dst_row if src_row > dst_row else dst_row - 1
        self._data = np.insert(self._data, insert_idx, row_data, axis=0)
        self.endMoveRows()
        return True
        
    def set_data(self, data: np.ndarray) -> None:
        """Resets the model with a completely new dataset."""
        self.beginResetModel()
        self._data = data
        self.endResetModel()

# ==========================================
# 2. The Configurable Responsive View (FIXED)
# ==========================================
class ResponsiveSpringTableView(QTableView):
    """
    A specialized QTableView that supports proportional column resizing, 
    persistent layout states, and a 'spring' column to fill empty space.
    """
    def __init__(self, 
                 enable_sorting: bool = False, 
                 enable_row_drag_drop: bool = False,
                 show_headers: bool = True,
                 show_row_labels: bool = True,
                 alternating_rows: bool = True,
                 alternate_color: Union[str, QColor, Literal["lighter", "darker"], None] = "lighter",
                 rows_resizable: bool = False) -> None:
        """
        Initializes the Responsive Table View with extensive configuration options.

        Args:
            enable_sorting (bool): If True, enables column header sorting.
            enable_row_drag_drop (bool): If True, enables row reordering.
            show_headers (bool): Visibility of horizontal column headers.
            show_row_labels (bool): Visibility of vertical row indices.
            alternating_rows (bool): Enables alternating row background colors.
            alternate_color (str|QColor|Literal): Defines the alternate color style.
            rows_resizable (bool): If True, allows manual row height resizing.
        """
        super().__init__()
        
        self._min_widths: dict[int, int] = {}
        self._max_widths: dict[int, int] = {}
        self._draggable_column_indices: set[int] = set()

        self._saved_state: Union[QByteArray, None] = None
        self._default_state: Union[QByteArray, None] = None
        self._is_initialized: bool = False

        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(alternating_rows)
        if alternating_rows and alternate_color:
            self.set_alternating_color(alternate_color)
        
        self.horizontalHeader().setVisible(show_headers)
        self.verticalHeader().setVisible(show_row_labels)
        self.set_rows_resizable(rows_resizable)

        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setSectionsMovable(True) 
        self.horizontalHeader().sectionResized.connect(self.on_column_resized)
        self.horizontalHeader().sectionMoved.connect(self.on_column_moved)
        
        # FIX 1: Explicitly initialize the Sort Indicator to Ascending immediately.
        # This prevents Qt from assuming "Descending" (1) if it encounters an uninitialized state.
        self.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)

        if enable_row_drag_drop:
            self.switch_to_drag()
        else:
            self.switch_to_sort()
            if not enable_sorting:
                self.setSortingEnabled(False)

    # --- INTEGRATED CONTROL FUNCTIONS ---

    def capture_default_state(self) -> None:
        """Captures the current layout (widths/positions) as the 'Factory Default'."""
        self._default_state = self.horizontalHeader().saveState()

    def save_layout(self, state: Optional[QByteArray] = None) -> None:
        """
        Saves the current table layout state.
        
        Args:
            state (Optional[QByteArray]): If provided, saves this specific state.
                Otherwise, captures the current state of the header.
        """
        if state is not None:
            self._saved_state = state
            print(">> External Layout State Stored!")
        else:
            self._saved_state = self.horizontalHeader().saveState()
            print(">> Current Layout Saved!")

    def restore_layout(self) -> None:
        """Restores previously saved column widths and order and recalculates spring."""
        if self._saved_state:
            self.horizontalHeader().restoreState(self._saved_state)
            self.update_spring_column()
            print(">> Layout Restored!")
        else:
            print(">> No saved layout found.")

    def reset_table(self) -> None:
        """
        Resets table to default layout and state.
        
        Restores the 'Factory Default' header state and recalculates the spring column.
        Crucially, layout restoration happens before any other logic to ensure 
        consistency.
        """
        
        # FIX 3: Restore Layout FIRST. 
        # If we sorted first, restoring the layout (which contains a sort state) 
        # might override our sort.
        if self._default_state:
            self.horizontalHeader().restoreState(self._default_state)
        
        self.update_spring_column()
        print(">> Table Reset to Defaults.")

    def switch_to_sort(self) -> None:
        """
        Enables sorting mode and disables row dragging.
        
        Ensures the sort indicator is initialized to Ascending if unset 
        to prevent automatic descending sorts on mode switch.
        """
        self.set_row_drag_drop(False)
        
        # FIX 2: Set indicator BEFORE enabling sorting logic.
        # This ensures that when sorting wakes up, it sees "Ascending" 
        # and doesn't trigger an automatic "Descending" sort.
        if self.horizontalHeader().sortIndicatorSection() == -1:
            self.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
            
        self.setSortingEnabled(True)
        self.horizontalHeader().setSectionsClickable(True)

    def switch_to_drag(self) -> None:
        """Enables row dragging mode and disables sorting capabilities."""
        self.set_row_drag_drop(True)

    def showEvent(self, event) -> None:
        """
        Handles the widget show event to initialize default states.
        
        Ensures that the default layout state is captured exactly once when the 
        table is first shown.
        """
        super().showEvent(event)
        
        if not self._is_initialized:
            # Capture the default state NOW, which includes the Ascending flag set in __init__
            self.capture_default_state()
            
            if self._saved_state is None:
                self.save_layout()
            
            self._is_initialized = True
            
        self.update_spring_column()

    # --- Internal Helpers ---

    def set_row_drag_drop(self, enable: bool) -> None:
        """Sets internal flags for drag/drop mode, disabling sorting if enabled."""
        self.setDragEnabled(enable)
        self.setAcceptDrops(enable)
        if enable:
            self.setSortingEnabled(False)
            self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.setDragDropOverwriteMode(False)
            self.setDropIndicatorShown(True)
        else:
            self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)

    def set_rows_resizable(self, enable: bool) -> None:
        """Toggles the ability for the user to resize row heights manually."""
        mode = QHeaderView.ResizeMode.Interactive if enable else QHeaderView.ResizeMode.Fixed
        self.verticalHeader().setSectionResizeMode(mode)

    def configure_columns(self, config: list[dict[str, Any]]) -> None:
        """
        Configures column properties based on a dictionary list.

        Iterates through the provided configuration to set headers, min/max widths,
        and draggability per column. Updates the underlying model's headers automatically.

        Args:
            config (list[dict]): A list of dicts. Example:
                {"label": "ID", "min_width": 50, "max_width": 100, "draggable": False}
        """
        labels = []
        current_headers = self.model()._headers if hasattr(self.model(), "_headers") else []
        self._draggable_column_indices.clear()

        for i in range(self.model().columnCount()):
            if i < len(config):
                if "label" in config[i]: labels.append(config[i]["label"])
                elif i < len(current_headers): labels.append(current_headers[i])
                else: labels.append(str(i))
                
                if config[i].get("draggable", True):
                    self._draggable_column_indices.add(i)
            else:
                labels.append(str(i))
                self._draggable_column_indices.add(i)
        
        if hasattr(self.model(), "set_headers"):
            self.model().set_headers(labels)

        self._min_widths.clear()
        self._max_widths.clear()
        for i, settings in enumerate(config):
            if "min_width" in settings: self._min_widths[i] = settings["min_width"]
            if "max_width" in settings: self._max_widths[i] = settings["max_width"]
        
        if self.isVisible():
            self.resizeEvent(QResizeEvent(self.size(), self.size()))

    def set_alternating_color(self, color_spec: Union[str, QColor, Literal["lighter", "darker"]]) -> None:
        """
        Sets the background color for alternating rows.

        Args:
            color_spec (Union[str, QColor, Literal]): 
                - "lighter": Lightens the base color.
                - "darker": Darkens the base color.
                - str/QColor: Sets a specific color.
        """
        if not self.alternatingRowColors(): return
        palette = self.palette()
        base_color = palette.color(QPalette.ColorRole.Base)
        target_color = None
        if color_spec == "lighter": target_color = self._derive_color(base_color, "lighter", 125)
        elif color_spec == "darker": target_color = self._derive_color(base_color, "darker", 125)
        elif isinstance(color_spec, (str, QColor)): target_color = QColor(color_spec)
        if target_color and target_color.isValid():
            palette.setColor(QPalette.ColorRole.AlternateBase, target_color)
            self.setPalette(palette)

    @staticmethod
    def _derive_color(base: QColor, mode: Literal["lighter", "darker"], factor: int = 125) -> QColor:
        """Static helper to lighten or darken a QColor by a specific factor."""
        if mode == "lighter": return base.lighter(factor)
        elif mode == "darker": return base.darker(factor)
        return base

    def on_column_moved(self, logicalIndex: int, oldVisualIndex: int, newVisualIndex: int) -> None:
        """
        Slot triggered when a column is moved.
        
        Enforces non-draggable column constraints (reverting the move if illegal)
        and recalculates the spring column width.
        """
        header = self.horizontalHeader()
        if logicalIndex not in self._draggable_column_indices:
            header.blockSignals(True) 
            header.moveSection(newVisualIndex, oldVisualIndex)
            header.blockSignals(False)
            return
        self.update_spring_column()

    def on_column_resized(self, index: int, old_size: int, new_size: int) -> None:
        """
        Slot triggered when a column is resized.
        
        Enforces minimum and maximum widths defined in configuration. If the 
        new size is out of bounds, it programmatically corrects the width.
        """
        header = self.horizontalHeader()
        last_visual_index = header.count() - 1
        last_logical_index = header.logicalIndex(last_visual_index)
        if index == last_logical_index: return
        
        min_w = self._min_widths.get(index, 30)
        max_w = self._max_widths.get(index, 999999)
        corrected = max(min_w, min(new_size, max_w))
        
        if corrected != new_size:
            header.blockSignals(True)
            self.setColumnWidth(index, corrected)
            header.blockSignals(False)
        self.update_spring_column()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Handles widget resizing to proportionally scale all columns.

        Calculates the ratio between new width and old width and applies it to
        all columns (except the spring column). It handles rounding errors by 
        accumulating the float delta and applying it when it exceeds 1 pixel,
        preventing columns from slowly shrinking or growing due to integer truncation.

        Args:
            event (QResizeEvent): The resize event containing old and new sizes.
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
        
        for visual_i in range(count - 1):
            logical_i = header.logicalIndex(visual_i)
            current_width = self.columnWidth(logical_i)
            target_width = (current_width * ratio) + rounding_error
            new_width_int = int(round(target_width))
            
            min_w = self._min_widths.get(logical_i, 30)
            max_w = self._max_widths.get(logical_i, 999999)
            
            if new_width_int < min_w: new_width_int = min_w
            elif new_width_int > max_w: new_width_int = max_w
            
            rounding_error = target_width - new_width_int
            if abs(rounding_error) > 10: rounding_error = 0.0
            
            header.blockSignals(True)
            self.setColumnWidth(logical_i, new_width_int)
            header.blockSignals(False)
        self.update_spring_column()

    def update_spring_column(self) -> None:
        """
        Calculates and updates the width of the last visible column (spring).
        
        The last column fills the remaining viewport width. If the total width
        of other columns exceeds the viewport, the spring column collapses to 0.
        """
        header = self.horizontalHeader()
        count = header.count()
        if count == 0: return
        last_visual_index = count - 1
        last_logical_index = header.logicalIndex(last_visual_index)
        used_width = 0
        for i in range(count):
            if i != last_logical_index:
                used_width += self.columnWidth(i)
        viewport_width = self.viewport().width()
        new_spring_width = max(0, viewport_width - used_width)
        header.blockSignals(True)
        self.setColumnWidth(last_logical_index, new_spring_width)
        header.blockSignals(False)

# ==========================================
# 3. Main Execution (Complex Example)
# ==========================================
class ComplexTableController(QWidget):
    """
    Main controller widget demonstrating the PADDOL table system.
    Combines the NumpyTableModel and ResponsiveSpringTableView with a control panel.
    """
    def __init__(self) -> None:
        """Initializes the demo application, generating data and setting up UI layout."""
        super().__init__()
        self.setWindowTitle("Complex Table Demo")
        self.resize(1000, 700)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 1. Data & Model
        self.data = self.generate_data(50)
        self.headers = ["ID", "Category", "Value", "Notes", "Spring"]
        self.model = NumpyTableModel(self.data, self.headers)

        # 2. Table
        self.table = ResponsiveSpringTableView(
            enable_sorting=True,          
            enable_row_drag_drop=False, 
            show_headers=True,
            show_row_labels=True,
            alternating_rows=True,
            alternate_color="darker",
            rows_resizable=True
        )
        self.table.setModel(self.model)

        # 3. Config
        self.config = [
            {"label": "ID", "min_width": 50, "max_width": 100, "draggable": False},
            {"label": "Category", "min_width": 120},
            {"label": "Value", "min_width": 80, "max_width": 150},
            {"label": "Notes (Draggable)", "min_width": 150},
            {"label": "Description (Spring)", "min_width": 100} 
        ]
        self.table.configure_columns(self.config)
        self.layout.addWidget(self.table)
        
        # 4. Controls
        self.create_controls()

        # 5. Capture Defaults
        self.show()
        QApplication.processEvents() 
        
        # The table now handles its own default state capture via showEvent()
        # but we also need to store the data default for reset purposes.
        self._default_data = self.data.copy()

    def generate_data(self, rows: int) -> np.ndarray:
        """Generates a dummy NumPy dataset for testing purposes."""
        data = np.empty((rows, 5), dtype=object)
        data[:, 0] = np.arange(rows)
        data[:, 1] = np.random.choice(["Alpha", "Beta", "Gamma"], size=rows)
        data[:, 2] = np.random.randint(100, 9999, size=rows)
        data[:, 3] = [f"Note {i}" for i in range(rows)]
        data[:, 4] = "Spring Filler"
        return data

    def create_controls(self) -> None:
        """Creates and links the UI control buttons for Mode and Layout management."""
        mode_frame = QFrame()
        mode_frame.setFrameShape(QFrame.Shape.StyledPanel)
        mode_layout = QHBoxLayout(mode_frame)
        
        self.btn_sort = QPushButton("Sorting Mode")
        self.btn_drag = QPushButton("Row Drag Mode")
        self.btn_sort.setCheckable(True)
        self.btn_drag.setCheckable(True)
        self.btn_sort.setChecked(True) 
        
        self.btn_sort.clicked.connect(self.on_click_sort)
        self.btn_drag.clicked.connect(self.on_click_drag)
        
        mode_layout.addWidget(QLabel("<b>Mode:</b>"))
        mode_layout.addWidget(self.btn_sort)
        mode_layout.addWidget(self.btn_drag)
        
        state_frame = QFrame()
        state_frame.setFrameShape(QFrame.Shape.StyledPanel)
        state_layout = QHBoxLayout(state_frame)
        
        btn_save = QPushButton("Save Layout")
        btn_restore = QPushButton("Restore Layout")
        btn_reset = QPushButton("Factory Reset")
        
        # Connect to Table Methods directly using lambda to support optional args if needed
        btn_save.clicked.connect(lambda: self.table.save_layout())
        btn_restore.clicked.connect(self.table.restore_layout)
        btn_reset.clicked.connect(self.on_click_reset)
        
        btn_reset.setStyleSheet("background-color: #ffcccc; color: darkred;")
        
        state_layout.addWidget(QLabel("<b>Layout:</b>"))
        state_layout.addWidget(btn_save)
        state_layout.addWidget(btn_restore)
        state_layout.addWidget(btn_reset)

        control_layout = QHBoxLayout()
        control_layout.addWidget(mode_frame)
        control_layout.addWidget(state_frame)
        self.layout.addLayout(control_layout)

    def on_click_sort(self) -> None:
        """Switches the UI and Table to Sorting Mode."""
        self.btn_sort.setChecked(True)
        self.btn_drag.setChecked(False)
        self.table.switch_to_sort()

    def on_click_drag(self) -> None:
        """Switches the UI and Table to Drag Mode."""
        self.btn_sort.setChecked(False)
        self.btn_drag.setChecked(True)
        self.table.switch_to_drag()

    def on_click_reset(self) -> None:
        """Performs a full Factory Reset on table layout and sorting order."""
        # Reset the table internals (Sort + Layout)
        self.table.reset_table()
        # Ensure UI buttons match the new state (Sort Mode)
        self.on_click_sort()

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    
    demo = ComplexTableController()
    demo.show()
    
    sys.exit(app.exec())
