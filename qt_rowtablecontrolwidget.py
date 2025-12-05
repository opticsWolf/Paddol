# -*- coding: utf-8 -*-
"""
PADDOL: Row-Based Table Control Widget
Copyright (c) 2025 opticsWolf

An optimized transposed version of the TableControlWidget where:
- Rows = Properties
- Columns = Items (Entities)
"""
from __future__ import annotations
import sys
from typing import Any, List, Optional, Union

from PySide6.QtCore import Qt, QModelIndex, QPoint, QRegularExpression
from PySide6.QtGui import QPainter, QIntValidator, QDoubleValidator, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTableWidgetItem,
    QAbstractItemView, QStyleOptionViewItem, QHeaderView, QMenu
)

# Import from user-provided files
from qt_tablecontrolwidget import PropertyDelegate, PropertyDescriptor, TableControlWidget
from qt_responsivespringtable import ResponsiveSpringTableWidget

# ──────────────────────────────────────────────────────────────
# 1. Transposed Delegate
# ──────────────────────────────────────────────────────────────

class RowBasedPropertyDelegate(PropertyDelegate):
    """
    A delegate that maps 'Rows' to properties instead of 'Columns'.
    Inherits static formatting logic from PropertyDelegate but overrides
    state-dependent methods to flip the axis.
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        """Overridden to fetch property from index.row() instead of index.column()."""
        if not index.isValid():
            return
        
        # KEY CHANGE: Property is determined by ROW, not COLUMN
        prop = self._properties[index.row()]

        if prop.get("always_visible", False):
            self._draw_empty_cell(painter, option, index)
            return
        
        # Color Button Logic
        if prop.get("widget_type") is sys.modules['PySide6.QtWidgets'].QPushButton: # Dynamic check
            val = index.data(Qt.EditRole)
            # Use base class static helpers where possible, or reimplement lightweight logic
            from PySide6.QtGui import QColor
            color = QColor(val) if val else QColor(Qt.transparent)
            if color.isValid():
                painter.save()
                rect = option.rect.adjusted(4, 4, -4, -4)
                painter.setBrush(color)
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(rect, 2, 2)
                if prop.get("with_label", True):
                    brightness = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
                    text_color = Qt.black if brightness > 128 else Qt.white
                    painter.setPen(text_color)
                    painter.drawText(rect, Qt.AlignCenter, color.name())
                painter.restore()
                return

        # Standard Text Painting
        super().paint(painter, option, index)

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex) -> QWidget:
        """Overridden to fetch property from index.row()."""
        if not index.isValid():
            return None
        
        # KEY CHANGE: Property from ROW
        prop = self._properties[index.row()]
        
        if not prop.get("editable", True):
            return None
        
        # Reuse the robust creation logic from the base class, 
        # but we must trick it or copy the logic. 
        # Since 'createEditor' in base uses index.column(), we cannot call super().
        # We must duplicate the logic but map the prop correctly.
        
        # Optimization: We can temporarily wrap the index or properties list, 
        # but copying the switch-case is faster and cleaner for performance than object proxying.
        
        t = prop.get("widget_type")
        obj = index.data(Qt.UserRole)
        current_val = self.get_value(obj, prop) if obj else index.data(Qt.EditRole)

        # -- Exact logic from PropertyDelegate.createEditor, mapped to 'prop' --
        from PySide6.QtWidgets import (
            QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox, 
            QSlider, QTimeEdit, QDateEdit, QPushButton
        )
        if t is QCheckBox: return QCheckBox(parent)
        if t is QDoubleSpinBox:
            editor = QDoubleSpinBox(parent)
            editor.setDecimals(prop.get("decimals", 9))
            if "value_range" in prop: editor.setRange(*prop["value_range"])
            else: editor.setRange(*self._calculate_dynamic_limits(current_val))
            if "step_size" in prop: editor.setSingleStep(prop["step_size"])
            if "unit" in prop: editor.setSuffix(f" {prop['unit']}")
            return editor
        if t is QSpinBox:
            editor = QSpinBox(parent)
            if "value_range" in prop: editor.setRange(int(prop["value_range"][0]), int(prop["value_range"][1]))
            else: 
                rng = self._calculate_dynamic_limits(current_val)
                editor.setRange(int(rng[0]), int(rng[1]))
            if "step_size" in prop: editor.setSingleStep(int(prop["step_size"]))
            if "unit" in prop: editor.setSuffix(f" {prop['unit']}")
            return editor
        if t is QLineEdit:
            editor = QLineEdit(parent)
            vt = prop.get("value_type")
            if vt == "numeric": editor.setValidator(QDoubleValidator(editor))
            elif vt == "integer": editor.setValidator(QIntValidator(editor))
            elif vt == "alphabetic": editor.setValidator(QRegularExpressionValidator(QRegularExpression("^[a-zA-Z]*$"), editor))
            elif vt == "alphanumeric": editor.setValidator(QRegularExpressionValidator(QRegularExpression("^[a-zA-Z0-9_()-]*$"), editor))
            return editor
        if t is QComboBox:
            editor = QComboBox(parent)
            editor.addItems(prop.get("items", []))
            return editor
        if t is QSlider: return self._create_slider_widget(parent, prop, current_val, None)
        if t is QTimeEdit:
            editor = QTimeEdit(parent)
            editor.setDisplayFormat(self._get_time_fmt(prop))
            return editor
        if t is QDateEdit:
            editor = QDateEdit(parent)
            editor.setCalendarPopup(True)
            editor.setDisplayFormat("yyyy-MM-dd")
            return editor
        if t is QPushButton:
            editor = QPushButton(parent)
            self._setup_color_button(editor, current_val, None, is_editor=True, prop=prop)
            return editor
        
        return None

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        """Overridden to fetch property from index.row()."""
        prop = self._properties[index.row()] # KEY CHANGE
        obj = index.data(Qt.UserRole)
        value = self.get_value(obj, prop) if obj is not None else index.data(Qt.EditRole)
        
        # Reuse base class logic patterns
        from PySide6.QtWidgets import (
            QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox, 
            QTimeEdit, QDateEdit, QPushButton
        )

        try:
            if isinstance(editor, QCheckBox): editor.setChecked(bool(value))
            elif isinstance(editor, QDoubleSpinBox): editor.setValue(float(value) if value is not None else 0.0)
            elif isinstance(editor, QSpinBox): editor.setValue(int(float(value)) if value is not None else 0)
            elif isinstance(editor, QLineEdit): editor.setText(str(value))
            elif isinstance(editor, QComboBox):
                idx = editor.findText(str(value))
                if idx >= 0: editor.setCurrentIndex(idx)
            elif hasattr(editor, "slider"):
                decimals = prop.get("decimals", 0)
                multiplier = 10 ** decimals
                editor.slider.setValue(int(float(value) * multiplier) if value is not None else 0)
            elif isinstance(editor, QTimeEdit): editor.setTime(self._to_qtime(value))
            elif isinstance(editor, QDateEdit): editor.setDate(self._to_qdate(value))
            elif isinstance(editor, QPushButton): self._update_color_button_face(editor, value)
        except ValueError: pass

    def setModelData(self, editor: QWidget, model: Any, index: QModelIndex) -> None:
        """Overridden to support row-based batch editing."""
        from PySide6.QtWidgets import (
            QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox, 
            QTimeEdit, QDateEdit, QPushButton
        )

        # Extract value
        if isinstance(editor, QCheckBox): val = editor.isChecked()
        elif isinstance(editor, QDoubleSpinBox): val = editor.value()
        elif isinstance(editor, QSpinBox): val = editor.value()
        elif isinstance(editor, QLineEdit): val = editor.text()
        elif isinstance(editor, QComboBox): val = editor.currentText() 
        elif hasattr(editor, "slider"): 
            prop = self._properties[index.row()] # KEY CHANGE
            val = editor.slider.value() / (10 ** prop.get("decimals", 0))
        elif isinstance(editor, QTimeEdit): val = editor.time() 
        elif isinstance(editor, QDateEdit): val = editor.date() 
        elif isinstance(editor, QPushButton): val = editor.property("color_value")
        else: val = None

        # Route to parent table for batch application
        table_widget = self.parent()
        if table_widget and hasattr(table_widget, "apply_batch_edit"):
             # In transposed view, we pass (row, col) as is, but the table logic handles it
             table_widget.apply_batch_edit(index.row(), index.column(), val)
        else:
             model.setData(index, val, Qt.EditRole)

    def destroyEditor(self, editor: QWidget, index: QModelIndex) -> None:
        """Overridden to format display text using index.row()."""
        if index.isValid():
            dock = self.parent()
            if dock and hasattr(dock, "_is_updating"): dock._is_updating = True
            
            prop = self._properties[index.row()] # KEY CHANGE
            val = index.data(Qt.EditRole)
            unit = prop.get("unit", "")
            formatted_val = self.format_value(val, prop)
            display_text = formatted_val if unit and formatted_val.endswith(unit) else f"{formatted_val} {unit}".strip()
            
            index.model().setData(index, display_text, Qt.DisplayRole)
            if dock and hasattr(dock, "_is_updating"): dock._is_updating = False
        
        # We can call super here because QStyledItemDelegate.destroyEditor manages the widget lifecycle,
        # which is independent of the property logic we just handled above.
        super(PropertyDelegate, self).destroyEditor(editor, index)


# ──────────────────────────────────────────────────────────────
# 2. Row-Based Control Widget
# ──────────────────────────────────────────────────────────────
class RowTableControlWidget(ResponsiveSpringTableWidget):
    """
    A transposed TableControlWidget where:
    - Vertical Header = Property Names
    - Horizontal Header = Item IDs
    - Rows = Properties
    - Columns = Items
    """

    def __init__(
        self, 
        parent: Optional[QWidget] = None, 
        properties_list: List[PropertyDescriptor] = None, 
        item_list: List[Any] = None,
        # Inherit styling options
        show_headers: bool = True,
        alternating_rows: bool = True,
        alternate_color: str = "lighter",
        enable_row_drag_drop: bool = True # This now enables dragging PROPERTIES (Rows)
    ):
        super().__init__(
            parent=parent,
            enable_sorting=False, # Sorting properties alphabetically is usually not desired
            enable_row_drag_drop=enable_row_drag_drop,
            show_headers=show_headers,
            show_row_labels=True,
            alternating_rows=alternating_rows,
            alternate_color=alternate_color,
            rows_resizable=True
        )

        self.properties_list = properties_list or []
        self.item_list = item_list or []
        self._is_updating = False

        # --- TRANSPOSED DIMENSIONS ---
        # Rows = Number of Properties
        # Cols = Number of Items
        self.setRowCount(len(self.properties_list))
        self.setColumnCount(len(self.item_list))

        # Setup Headers
        self.setVerticalHeaderLabels([p.get("label", str(i)) for i, p in enumerate(self.properties_list)])
        
        # Try to get item names/IDs for horizontal headers
        col_labels = []
        for i, item in enumerate(self.item_list):
            if hasattr(item, "id_code"): col_labels.append(str(item.id_code))
            elif hasattr(item, "name"): col_labels.append(str(item.name))
            else: col_labels.append(f"Item {i+1}")
        self.setHorizontalHeaderLabels(col_labels)

        # -----------------------------------------------------------
        # FIX APPLIED HERE
        # -----------------------------------------------------------
        # Previous: QAbstractItemView.SelectColumns (This caused your issue)
        # New: QAbstractItemView.SelectItems (Allows single cell selection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        
        # Standard extended selection allows CTRL+Click and Drag selection
        self.setSelectionMode(QAbstractItemView.ExtendedSelection) 
        # -----------------------------------------------------------

        # Apply Configuration to Columns (Items)
        # In a transposed view, we need to ensure the columns (Items) are wide enough 
        # for the widest Property widget (e.g., a Slider).
        self._apply_transposed_configuration()

        # Install Transposed Delegate
        # We assume RowBasedPropertyDelegate is defined in your file as previously shown
        # If it's in the same file, pass it directly. If imported, ensure import is correct.
        # For this snippet, I assume the delegate class is available in scope.
        self._delegate = RowBasedPropertyDelegate(self.properties_list, self)
        
        # Apply delegate to ALL rows/cols (since layout is uniform grid)
        self.setItemDelegate(self._delegate)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        self.populate_table()
        self.itemChanged.connect(self.on_item_changed)

    def _apply_transposed_configuration(self) -> None:
        """
        Calculates column configuration.
        Since columns are Items, we check ALL properties to find the maximum required width.
        """
        if not self.properties_list:
            return

        # Find the max 'min_width' among all properties to ensure sliders fit
        global_min_width = 80
        for prop in self.properties_list:
            p_min = prop.get("min_width", 0)
            if p_min > global_min_width:
                global_min_width = p_min

        # Generate config for Item Columns
        config = []
        for i in range(len(self.item_list)):
            col_cfg = {
                # Items are draggable columns if desired, but we default to standard
                "draggable": True,
                "min_width": global_min_width,
            }
            config.append(col_cfg)
        
        self.configure_columns(config)

    def populate_table(self) -> None:
        """Populates the table in Transposed (Row=Property) order."""
        self.blockSignals(True)
        self.clearContents()
        
        # Dimensions already set in __init__, but safe to reset if data changed
        self.setRowCount(len(self.properties_list))
        self.setColumnCount(len(self.item_list))
        
        # LOOP: Outer = Rows (Properties), Inner = Cols (Items)
        for r, prop in enumerate(self.properties_list):
            for c, obj in enumerate(self.item_list):
                self._create_cell(r, c, obj, prop)
                
        self.blockSignals(False)
        self.update_spring_column()

    def _create_cell(self, row: int, col: int, obj: Any, prop: PropertyDescriptor) -> None:
        val = PropertyDelegate.get_value(obj, prop)
        is_editable = prop.get("editable", True)

        # Persistent Widgets (Sliders, Buttons)
        if prop.get("always_visible", False):
            def _persist_cb(v):
                self.apply_batch_edit(row, col, v)
            
            # Use the delegate to create the widget
            w = self._delegate.create_persistent_widget(prop, val, _persist_cb)
            if w:
                if not is_editable: w.setEnabled(False)
                self.setCellWidget(row, col, w)
                
                # We still need an item to hold the UserRole data
                item = QTableWidgetItem()
                item.setData(Qt.UserRole, obj)
                self.setItem(row, col, item)
                return

        # Standard Items
        unit = prop.get("unit", "")
        formatted_val = PropertyDelegate.format_value(val, prop)
        display_text = f"{formatted_val} {unit}".strip()

        item = QTableWidgetItem(display_text)
        item.setData(Qt.EditRole, val)
        item.setData(Qt.DisplayRole, display_text)
        item.setData(Qt.UserRole, obj)
        
        if not is_editable or prop.get("readonly", False):
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            
        self.setItem(row, col, item)

    def apply_batch_edit(self, row: int, col: int, value: Any) -> None:
        """
        Applies a value change.
        Logic is adjusted for Transposed View:
        - 'row' identifies the Property.
        - 'col' identifies the Item.
        - Batch editing applies to the Property (row) across selected Items (columns).
        """
        if self._is_updating: return
        self._is_updating = True
        self.blockSignals(True)
        
        try:
            prop = self.properties_list[row]
            if not prop.get("editable", True): return

            # Get selected columns (Items)
            selected_indices = self.selectionModel().selectedIndexes()
            
            # Determine target columns (Items) that are on the SAME ROW (Property)
            target_cols = set()
            
            # Check if the specific cell is selected, or we fall back to just the edited cell
            cell_is_selected = False
            for idx in selected_indices:
                if idx.row() == row and idx.column() == col:
                    cell_is_selected = True
                    break
            
            if not selected_indices or not cell_is_selected:
                target_cols.add(col)
            else:
                for idx in selected_indices:
                    # Only apply to this property row
                    if idx.row() == row:
                        target_cols.add(idx.column())

            # Apply Loop
            for c in target_cols:
                if c >= len(self.item_list): continue
                
                obj = self.item_list[c]
                PropertyDelegate.set_value(obj, prop, value)
                
                # Update Table Item
                item = self.item(row, c)
                if item:
                    item.setData(Qt.EditRole, value)
                    item.setData(Qt.UserRole, obj)
                    if not prop.get("always_visible", False):
                        unit = prop.get("unit", "")
                        formatted_val = PropertyDelegate.format_value(value, prop)
                        item.setData(Qt.DisplayRole, f"{formatted_val} {unit}".strip())
                
                # Update Persistent Widget
                if prop.get("always_visible", False):
                    widget = self.cellWidget(row, c)
                    if widget: self._update_persistent_widget_value(widget, value, prop)

        finally:
            self.blockSignals(False)
            self._is_updating = False

    def on_item_changed(self, item: QTableWidgetItem) -> None:
        if not item or self._is_updating: return
        
        # Transposed mapping: Row=Prop, Col=Item
        row, col = item.row(), item.column()
        
        # Validate indices
        if row >= len(self.properties_list) or col >= len(self.item_list):
            return

        self._is_updating = True
        try:
            prop = self.properties_list[row]
            obj = item.data(Qt.UserRole) or self.item_list[col]
            val = item.data(Qt.EditRole)
            
            PropertyDelegate.set_value(obj, prop, val)
            
            if not prop.get("always_visible", False):
                unit = prop.get("unit", "")
                formatted_val = PropertyDelegate.format_value(val, prop)
                item.setData(Qt.DisplayRole, f"{formatted_val} {unit}".strip())
        finally:
            self._is_updating = False

    def _update_persistent_widget_value(self, widget: QWidget, val: Any, prop: PropertyDescriptor) -> None:
        """Helper to update widget state without triggering signals."""
        widget.blockSignals(True)
        try:
            # Need access to internal calc
            # Note: Accessing protected member _calculate_dynamic_limits of delegate
            # Ideally this should be a public static or utility method.
            if hasattr(self._delegate, '_calculate_dynamic_limits'):
                 new_min, new_max = self._delegate._calculate_dynamic_limits(val)
            else:
                 new_min, new_max = (0, 100) # Fallback

            decimals = prop.get("decimals", 0)
            multiplier = 10 ** decimals
            
            from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit, QTimeEdit, QDateEdit, QPushButton

            if "value_range" not in prop:
                if hasattr(widget, "slider"): widget.slider.setRange(int(new_min * multiplier), int(new_max * multiplier))
                elif isinstance(widget, QDoubleSpinBox): widget.setRange(new_min, new_max)
                elif isinstance(widget, QSpinBox): widget.setRange(int(new_min), int(new_max))

            if hasattr(widget, "input_widget"): widget.input_widget.setChecked(bool(val))
            elif hasattr(widget, "slider"): 
                widget.slider.setValue(int(float(val) * multiplier))
                if hasattr(widget, "label"):
                    unit = prop.get("unit", "")
                    widget.label.setText(f"{float(val):.{decimals}f} {unit}".strip())
            elif isinstance(widget, QDoubleSpinBox): widget.setValue(float(val))
            elif isinstance(widget, QSpinBox): widget.setValue(int(float(val)) if val is not None else 0)
            elif isinstance(widget, QComboBox): widget.setCurrentText(str(val))
            elif isinstance(widget, QLineEdit): widget.setText(str(val))
            elif isinstance(widget, QTimeEdit): widget.setTime(PropertyDelegate._to_qtime(val))
            elif isinstance(widget, QDateEdit): widget.setDate(PropertyDelegate._to_qdate(val))
            elif isinstance(widget, QPushButton): self._delegate._update_color_button_face(widget, val)

        finally:
            widget.blockSignals(False)

    def _show_context_menu(self, pos: QPoint) -> None:
        item = self.itemAt(pos)
        if item is None: return
        
        # Transposed: Row is Property
        row = item.row()
        prop = self.properties_list[row]
        
        if not prop.get("editable", True): return
        if "default" not in prop: return
        
        menu = QMenu(self)
        action_text = f"Reset '{prop['label']}' to {prop['default']}"
        action = menu.addAction(action_text)
        
        if menu.exec(self.viewport().mapToGlobal(pos)) == action:
            # Apply to this cell (Item) for this Row (Prop)
            self.apply_batch_edit(row, item.column(), prop['default'])

# ──────────────────────────────────────────────────────────────
# Example Usage
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from PySide6.QtCore import QTime, QDate
    from PySide6.QtWidgets import QDockWidget, QLineEdit, QDoubleSpinBox, QSpinBox, QSlider, QComboBox, QCheckBox, QPushButton, QTimeEdit, QDateEdit
    
    # Mock Data Class
    class SimulationEntity:
        def __init__(self, name, mass, material):
            self.id_code = name
            self.mass = mass
            self.velocity = 0.0
            self.material = material
            self.color = "#3498db"
            self.visible = True
        
        def set_momentum(self, v): pass # Mock
        def get_momentum(self): return self.mass * self.velocity

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    # Property Definitions
    props: List[PropertyDescriptor] = [
        {"attr": "id_code", "label": "ID", "widget_type": QLineEdit, "value_type": "alphanumeric", "default": "ENT_001", "min_width": 80},
        {"attr": "mass", "label": "Mass", "widget_type": QDoubleSpinBox, "unit": "kg", "value_range": (0.1, 1000.0), "step_size": 0.5, "default": 10.0, "min_width": 70},
        {"attr": "velocity", "label": "Velocity", "widget_type": QDoubleSpinBox, "unit": "m/s", "decimals": 4, "step_size": 0.01, "default": 0.0, "min_width": 70},
        {"attr": "material", "label": "Material", "widget_type": QComboBox, "items": ["Silicon", "Gallium", "Vacuum", "Gold"], "always_visible": True, "default": "Vacuum", "min_width": 120},
        {"attr": "color", "label": "Color", "widget_type": QPushButton, "always_visible": True, "with_label": True, "default": "#000000", "min_width": 100},
        {"attr": "visible", "label": "Visible", "widget_type": QCheckBox, "always_visible": True, "default": True, "min_width": 60},
        # Calculated Property
        {"label": "Momentum", "widget_type": QDoubleSpinBox, "unit": "kg⋅m/s", "getter": lambda obj: obj.get_momentum(), "setter": lambda obj, val: obj.set_momentum(val), "readonly": True, "min_width": 100}
    ]

    # Data Items (Columns)
    items = [
        SimulationEntity("Entity_A", 50.0, "Silicon"),
        SimulationEntity("Entity_B", 120.5, "Gold"),
        SimulationEntity("Entity_C", 0.5, "Vacuum"),
        SimulationEntity("Entity_D", 12.0, "Gallium"),
    ]

    win = QMainWindow()
    win.setWindowTitle("PADDOL - Row Based Control")
    
    # Init Row-Based Widget
    row_table = RowTableControlWidget(
        properties_list=props,
        item_list=items,
        show_headers=True,
        alternating_rows=True,
        alternate_color="lighter"
    )
    
    dock = QDockWidget("Transposed Attributes", win)
    dock.setWidget(row_table)
    win.addDockWidget(Qt.BottomDockWidgetArea, dock)
    
    win.resize(800, 400)
    win.show()

    sys.exit(app.exec())