# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later

Optimized TableDockWidget (v23) - Slider Precision & Float Support.

Features:
1. Slider Precision: 'decimals' property now controls QSlider precision 
   (e.g., decimals=2 allows 0.00-1.00 range via internal integer scaling).
2. Robust Paint Logic: Retains v22 visual hiding fix.
3. Escape Key Integrity: Retains v21 data integrity fix.
4. Dynamic Range: Auto-scales limits for persistent widgets.
"""
from __future__ import annotations
import sys
from typing import Any, List, Optional, Callable, TypedDict, Union, Literal

from PySide6.QtCore import (
    Qt, 
    QPoint, 
    QAbstractItemModel, 
    QModelIndex,
    QTime,
    QDate,
    QRegularExpression
)
from PySide6.QtGui import (
    QIntValidator, 
    QDoubleValidator, 
    QRegularExpressionValidator,
    QPainter,
    QColor,
    QPalette
)
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QHeaderView,
    QStyledItemDelegate,
    QMenu,
    QTableWidgetItem,
    QHBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QLabel,
    QStyle,
    QStyleOptionViewItem,
    QSlider,
    QMainWindow,
    QTimeEdit,
    QDateEdit,
    QPushButton,
    QColorDialog,
    QAbstractItemView
)

# ──────────────────────────────────────────────────────────────
# Type Definitions
# ──────────────────────────────────────────────────────────────

class PropertyDescriptor(TypedDict, total=False):
    """Schema for property definitions to ensure type safety."""
    attr: str
    label: str
    widget_type: type[QWidget]
    value_type: str 
    readonly: bool
    unit: str
    value_range: tuple[float, float]
    step_size: float
    decimals: int # Controls precision for QDoubleSpinBox AND QSlider      
    items: List[str]  
    always_visible: bool 
    orientation: Qt.Orientation
    getter: Callable[[Any], Any]
    setter: Callable[[Any, Any], None]
    default: Any
    precision: Literal["min", "secs"]
    with_label: bool

# ──────────────────────────────────────────────────────────────
# Optimized Delegate
# ──────────────────────────────────────────────────────────────

class PropertyDelegate(QStyledItemDelegate):
    """
    High-performance delegate handling painting and editing.
    """

    def __init__(self, properties: List[PropertyDescriptor], parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._properties = properties

    # ------------------------------------------------------------------ #
    # Data Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_value(obj: Any, prop: PropertyDescriptor) -> Any:
        if obj is None: return None
        if "getter" in prop: return prop["getter"](obj)
        attr = prop.get("attr")
        if not attr or not hasattr(obj, attr): return None
        val = getattr(obj, attr, None)
        return val() if callable(val) else val

    @staticmethod
    def _get_time_fmt(prop: PropertyDescriptor) -> str:
        return "HH:mm:ss" if prop.get("precision") == "secs" else "HH:mm"

    @staticmethod
    def format_value(val: Any, prop: PropertyDescriptor) -> str:
        if val is None: return ""
        
        if isinstance(val, QTime) or prop.get("widget_type") is QTimeEdit:
            qtime = PropertyDelegate._to_qtime(val)
            return qtime.toString(PropertyDelegate._get_time_fmt(prop))
            
        if isinstance(val, QDate) or prop.get("widget_type") is QDateEdit:
            qdate = PropertyDelegate._to_qdate(val)
            return qdate.toString("yyyy-MM-dd")
        
        # Respect decimals for float display if specified
        if isinstance(val, (float, int)) and "decimals" in prop:
            return f"{val:.{prop['decimals']}f}"

        return PropertyDelegate.format_float(val)

    @staticmethod
    def format_float(val: Any) -> str:
        if isinstance(val, bool): return str(val)
        if isinstance(val, (QTime, QDate)): return val.toString()
        if not isinstance(val, (float, int)): return str(val)
        if val == 0: return "0"
        abs_val = abs(val)
        if abs_val < 0.001 or abs_val >= 100000:
            return f"{val:.3e}"
        text = f"{val:.3f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    @staticmethod
    def set_value(obj: Any, prop: PropertyDescriptor, value: Any) -> None:
        if obj is None: return
        if "setter" in prop:
            prop["setter"](obj, value)
            return
        attr = prop.get("attr")
        target = getattr(obj, attr, None)
        if callable(target):
            try: target(value)
            except TypeError: pass 
        else:
            setattr(obj, attr, value)

    @staticmethod
    def _calculate_dynamic_limits(val: Any, current_widget: Optional[QWidget] = None) -> tuple[float, float]:
        try:
            v = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            v = 0.0
        
        if abs(v) < 1e-9:
            if current_widget is not None:
                if hasattr(current_widget, "minimum") and hasattr(current_widget, "maximum"):
                    return (float(current_widget.minimum()), float(current_widget.maximum()))
                if hasattr(current_widget, "slider"): 
                     # Note: Sliders return ints, but this function returns logical floats
                     # We cannot trust raw slider min/max if it's scaled
                     pass 
            return (0.0, 100.0)

        lower = v * 0.05
        upper = v * 2.0
        return (min(lower, upper), max(lower, upper))

    @staticmethod
    def _to_qtime(val: Any) -> QTime:
        if isinstance(val, QTime): return val
        if isinstance(val, str): return QTime.fromString(val)
        return QTime.currentTime()

    @staticmethod
    def _to_qdate(val: Any) -> QDate:
        if isinstance(val, QDate): return val
        if isinstance(val, str): return QDate.fromString(val, Qt.ISODate)
        return QDate.currentDate()

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        if not index.isValid(): return
        
        prop = self._properties[index.column()]

        # 1. Handle Persistent Widgets (Always Hide Text)
        if prop.get("always_visible", False):
            self._draw_empty_cell(painter, option, index)
            return

        # 2. Handle Color Buttons
        if prop.get("widget_type") is QPushButton:
            val = index.data(Qt.EditRole)
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

        # 3. Handle Active Editing (Robust Double-Check)
        is_editing = bool(option.state & QStyle.State_Editing)
        
        if not is_editing and option.widget:
            view = option.widget
            if not isinstance(view, QAbstractItemView) and view.parent():
                view = view.parent()
            
            if isinstance(view, QAbstractItemView):
                if view.state() == QAbstractItemView.EditingState and view.currentIndex() == index:
                    is_editing = True

        if is_editing:
            self._draw_empty_cell(painter, option, index)
            return

        super().paint(painter, option, index)

    def _draw_empty_cell(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Helper to draw the cell background/focus without the text."""
        opt_copy = QStyleOptionViewItem(option)
        self.initStyleOption(opt_copy, index)
        opt_copy.text = "" 
        widget = option.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt_copy, painter)

    # ------------------------------------------------------------------ #
    # Editor Creation
    # ------------------------------------------------------------------ #
    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex) -> QWidget:
        if not index.isValid(): return None
        
        prop = self._properties[index.column()]
        t = prop.get("widget_type")
        obj = index.data(Qt.UserRole)
        current_val = self.get_value(obj, prop) if obj else index.data(Qt.EditRole)

        if t is QCheckBox: return QCheckBox(parent)
        
        if t is QDoubleSpinBox:
            editor = QDoubleSpinBox(parent)
            editor.setDecimals(prop.get("decimals", 9))
            if "value_range" in prop:
                editor.setRange(*prop["value_range"])
            else:
                rng = self._calculate_dynamic_limits(current_val, current_widget=None)
                editor.setRange(*rng)
            if "step_size" in prop: editor.setSingleStep(prop["step_size"])
            if "unit" in prop: editor.setSuffix(f" {prop['unit']}")
            return editor

        if t is QSpinBox:
            editor = QSpinBox(parent)
            if "value_range" in prop:
                editor.setRange(int(prop["value_range"][0]), int(prop["value_range"][1]))
            else:
                rng = self._calculate_dynamic_limits(current_val, current_widget=None)
                editor.setRange(int(rng[0]), int(rng[1]))
            if "step_size" in prop: editor.setSingleStep(int(prop["step_size"]))
            if "unit" in prop: editor.setSuffix(f" {prop['unit']}")
            return editor

        if t is QLineEdit:
            editor = QLineEdit(parent)
            vt = prop.get("value_type")
            if vt == "numeric": 
                editor.setValidator(QDoubleValidator(editor))
            elif vt == "integer": 
                editor.setValidator(QIntValidator(editor))
            elif vt == "alphabetic":
                rx = QRegularExpression("^[a-zA-Z]*$")
                editor.setValidator(QRegularExpressionValidator(rx, editor))
            elif vt == "alphanumeric":
                rx = QRegularExpression("^[a-zA-Z0-9_()-]*$")
                editor.setValidator(QRegularExpressionValidator(rx, editor))
            return editor
            
        if t is QComboBox:
            editor = QComboBox(parent)
            editor.addItems(prop.get("items", []))
            return editor
            
        if t is QSlider:
            return self._create_slider_widget(parent, prop, current_val, None)

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

        return super().createEditor(parent, option, index)

    # ------------------------------------------------------------------ #
    # Editor Destruction
    # ------------------------------------------------------------------ #
    def destroyEditor(self, editor: QWidget, index: QModelIndex) -> None:
        """Called when editing finishes OR is cancelled (Escape)."""
        if index.isValid():
            dock = self.parent()
            if dock: dock._is_updating = True

            prop = self._properties[index.column()]
            val = index.data(Qt.EditRole)
            unit = prop.get("unit", "")
            
            formatted_val = PropertyDelegate.format_value(val, prop)
            
            if unit and formatted_val.endswith(unit):
                display_text = formatted_val
            else:
                display_text = f"{formatted_val} {unit}".strip()

            index.model().setData(index, display_text, Qt.DisplayRole)
            
            if dock: dock._is_updating = False

        super().destroyEditor(editor, index)

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        prop = self._properties[index.column()]
        obj = index.data(Qt.UserRole)
        value = self.get_value(obj, prop) if obj is not None else index.data(Qt.EditRole)
        try:
            if isinstance(editor, QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QDoubleSpinBox):
                editor.setValue(float(value) if value is not None else 0.0)
            elif isinstance(editor, QSpinBox):
                editor.setValue(int(float(value)) if value is not None else 0)
            elif isinstance(editor, QLineEdit):
                editor.setText(str(value))
            elif isinstance(editor, QComboBox):
                idx = editor.findText(str(value))
                if idx >= 0: editor.setCurrentIndex(idx)
            elif hasattr(editor, "slider"):
                # v23: Apply decimal multiplier
                decimals = prop.get("decimals", 0)
                multiplier = 10 ** decimals
                editor.slider.setValue(int(float(value) * multiplier) if value is not None else 0)
            elif isinstance(editor, QTimeEdit):
                editor.setTime(self._to_qtime(value))
            elif isinstance(editor, QDateEdit):
                editor.setDate(self._to_qdate(value))
            elif isinstance(editor, QPushButton):
                self._update_color_button_face(editor, value)
        except ValueError:
            pass

    def setModelData(self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex) -> None:
        if isinstance(editor, QCheckBox): val = editor.isChecked()
        elif isinstance(editor, QDoubleSpinBox): val = editor.value()
        elif isinstance(editor, QSpinBox): val = editor.value()
        elif isinstance(editor, QLineEdit): val = editor.text()
        elif isinstance(editor, QComboBox): val = editor.currentText() 
        elif hasattr(editor, "slider"): 
            # v23: Reverse decimal multiplier
            prop = self._properties[index.column()]
            decimals = prop.get("decimals", 0)
            multiplier = 10 ** decimals
            val = editor.slider.value() / multiplier
        elif isinstance(editor, QTimeEdit): val = editor.time() 
        elif isinstance(editor, QDateEdit): val = editor.date() 
        elif isinstance(editor, QPushButton): val = editor.property("color_value")
        else: val = None

        dock_widget = self.parent()
        if dock_widget and hasattr(dock_widget, "apply_batch_edit"):
             dock_widget.apply_batch_edit(index.row(), index.column(), val)
        else:
             model.setData(index, val, Qt.EditRole)

    # ------------------------------------------------------------------ #
    # Persistent Widget Helpers
    # ------------------------------------------------------------------ #
    def create_persistent_widget(self, prop: PropertyDescriptor, val: Any, callback: Callable) -> QWidget:
        t = prop.get("widget_type")
        if t is QSlider: return self._create_slider_widget(None, prop, val, callback)
            
        widget = None
        if t is QCheckBox:
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            widget = QCheckBox()
            widget.setChecked(bool(val))
            widget.toggled.connect(callback)
            container.input_widget = widget
            layout.addWidget(widget)
            return container

        elif t is QDoubleSpinBox:
            widget = QDoubleSpinBox()
            widget.setDecimals(prop.get("decimals", 9))
            if "value_range" in prop:
                widget.setRange(*prop["value_range"])
            else:
                rng = self._calculate_dynamic_limits(val, current_widget=None)
                widget.setRange(*rng)
            if "step_size" in prop: widget.setSingleStep(prop["step_size"])
            if "unit" in prop: widget.setSuffix(f" {prop['unit']}")
            widget.setValue(float(val) if val is not None else 0.0)
            widget.editingFinished.connect(lambda: callback(widget.value()))

        elif t is QSpinBox:
            widget = QSpinBox()
            if "value_range" in prop:
                widget.setRange(int(prop["value_range"][0]), int(prop["value_range"][1]))
            else:
                rng = self._calculate_dynamic_limits(val, current_widget=None)
                widget.setRange(int(rng[0]), int(rng[1]))
            if "step_size" in prop: widget.setSingleStep(int(prop["step_size"]))
            if "unit" in prop: widget.setSuffix(f" {prop['unit']}")
            widget.setValue(int(float(val)) if val is not None else 0)
            widget.editingFinished.connect(lambda: callback(widget.value()))
            
        elif t is QComboBox:
            widget = QComboBox()
            widget.addItems(prop.get("items", []))
            widget.setCurrentText(str(val))
            widget.currentTextChanged.connect(callback)
            
        elif t is QLineEdit:
            widget = QLineEdit()
            widget.setText(str(val) if val is not None else "")
            widget.editingFinished.connect(lambda: callback(widget.text()))

        elif t is QTimeEdit:
            widget = QTimeEdit()
            widget.setDisplayFormat(self._get_time_fmt(prop))
            widget.setTime(self._to_qtime(val))
            widget.timeChanged.connect(callback)

        elif t is QDateEdit:
            widget = QDateEdit()
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("yyyy-MM-dd")
            widget.setDate(self._to_qdate(val))
            widget.dateChanged.connect(callback)

        elif t is QPushButton:
            widget = QPushButton()
            self._setup_color_button(widget, val, callback, is_editor=False, prop=prop)

        return widget

    def _create_slider_widget(self, parent: Optional[QWidget], prop: PropertyDescriptor, val: Any, callback: Optional[Callable]) -> QWidget:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)
        slider = QSlider(prop.get("orientation", Qt.Horizontal))
        
        # v23: Calculate Multiplier
        decimals = prop.get("decimals", 0)
        multiplier = 10 ** decimals

        # Determine limits (logical)
        if "value_range" in prop:
            logical_rng = prop["value_range"]
        else:
            logical_rng = self._calculate_dynamic_limits(val, current_widget=None)
        
        # Set Slider Integer Range
        slider.setRange(int(logical_rng[0] * multiplier), int(logical_rng[1] * multiplier))
        
        # Set Slider Value
        try: float_val = float(val) if val is not None else float(logical_rng[0])
        except ValueError: float_val = float(logical_rng[0])
        slider.setValue(int(float_val * multiplier))
        
        unit = prop.get("unit", "")
        # Use format_value to respect decimal setting in label
        init_text = PropertyDelegate.format_value(val if val is not None else 0, prop)
        label = QLabel(f"{init_text} {unit}".strip())
        label.setFixedWidth(50)
        
        layout.addWidget(slider)
        layout.addWidget(label)
        container.slider = slider
        container.label = label
        
        def update_lbl(slider_val: int) -> None:
            # v23: Convert back to float for callback and label
            logical_val = slider_val / multiplier
            
            # Format label specifically for decimals
            txt = f"{logical_val:.{decimals}f}"
            label.setText(f"{txt} {unit}".strip())
            
            if callback: callback(logical_val)
        
        slider.sliderReleased.connect(lambda: update_lbl(slider.value()))
        # Live label update
        slider.valueChanged.connect(lambda v: label.setText(f"{v / multiplier:.{decimals}f} {unit}".strip()))
        
        return container

    # ------------------------------------------------------------------ #
    # Color Picker Logic
    # ------------------------------------------------------------------ #
    def _setup_color_button(self, button: QPushButton, val: Any, callback: Optional[Callable], is_editor: bool, prop: Optional[PropertyDescriptor] = None) -> None:
        show_label = True
        if prop and "with_label" in prop:
            show_label = prop["with_label"]
        button.setProperty("show_label", show_label)

        self._update_color_button_face(button, val)

        def on_click():
            initial = QColor(button.property("color_value"))
            if not initial.isValid(): initial = QColor(Qt.white)
            c = QColorDialog.getColor(initial, button, "Select Color")
            if c.isValid():
                hex_color = c.name()
                self._update_color_button_face(button, hex_color)
                if is_editor:
                    self.commitData.emit(button)
                    self.closeEditor.emit(button)
                elif callback:
                    callback(hex_color)

        button.clicked.connect(on_click)

    def _update_color_button_face(self, button: QPushButton, val: Any) -> None:
        c = QColor(val) if val else QColor(Qt.gray)
        if not c.isValid(): c = QColor(Qt.gray)
        
        button.setProperty("color_value", c.name())
        show_label = button.property("show_label")
        if show_label is None: show_label = True
        
        brightness = c.red() * 0.299 + c.green() * 0.587 + c.blue() * 0.114
        text_color = "black" if brightness > 128 else "white"
        
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {c.name()};
                color: {text_color};
                border: 1px solid #555;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border: 1px solid {text_color};
            }}
        """)
        
        if show_label:
            button.setText(c.name())
        else:
            button.setText("")


# ──────────────────────────────────────────────────────────────
# Optimized Dock Widget
# ──────────────────────────────────────────────────────────────

class TableDockWidget(QDockWidget):
    def __init__(
        self, 
        parent: Optional[QWidget] = None, 
        title: str = "", 
        properties_list: List[PropertyDescriptor] = None, 
        item_list: List[Any] = None,
        show_headers: bool = True,
        show_row_numbers: bool = True,
        alternating_rows: bool = True,
        alternate_color: Optional[Union[QColor, str, Literal["lighter", "darker"]]] = "darker"
    ):
        super().__init__(title, parent)
        self.properties_list = properties_list or []
        self.item_list = item_list or []
        
        self._show_headers = show_headers
        self._show_row_numbers = show_row_numbers
        self._alternating_rows = alternating_rows
        self._alternate_color = alternate_color
        
        self._is_updating = False 
        
        self._setup_ui()
        self._delegate = PropertyDelegate(self.properties_list, self)
        
        for col, prop in enumerate(self.properties_list):
            self.table.setItemDelegateForColumn(col, self._delegate)

        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)

        self.populate_table()
        self.table.itemChanged.connect(self.on_item_changed)

    @staticmethod
    def _derive_color(base: QColor, mode: Literal["lighter", "darker"], factor: int = 105) -> QColor:
        if mode == "lighter":
            return base.lighter(factor)
        elif mode == "darker":
            return base.darker(factor)
        return base

    def _setup_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.properties_list))
        
        self.table.setHorizontalHeaderLabels([p.get("label", "") for p in self.properties_list])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        if not self._show_headers:
            header.hide()
            
        if not self._show_row_numbers:
            self.table.verticalHeader().hide()

        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(self._alternating_rows)
        
        if self._alternating_rows and self._alternate_color:
            palette = self.table.palette()
            base_color = palette.color(QPalette.ColorRole.Base)
            target_color = None
            if self._alternate_color == "lighter":
                target_color = self._derive_color(base_color, "lighter", 105)
            elif self._alternate_color == "darker":
                target_color = self._derive_color(base_color, "darker", 105)
            elif isinstance(self._alternate_color, (str, QColor)):
                target_color = QColor(self._alternate_color)
            
            if target_color and target_color.isValid():
                palette.setColor(QPalette.ColorRole.AlternateBase, target_color)
                self.table.setPalette(palette)

        layout.addWidget(self.table)
        self.setWidget(container)

    def populate_table(self) -> None:
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(len(self.item_list))
        for r, obj in enumerate(self.item_list):
            for c, prop in enumerate(self.properties_list):
                self._create_cell(r, c, obj, prop)
        self.table.blockSignals(False)

    def _create_cell(self, row: int, col: int, obj: Any, prop: PropertyDescriptor) -> None:
        val = PropertyDelegate.get_value(obj, prop)
        
        if prop.get("always_visible", False):
            def _persist_cb(v):
                self.apply_batch_edit(row, col, v)
                
            w = self._delegate.create_persistent_widget(prop, val, _persist_cb)
            if w:
                self.table.setCellWidget(row, col, w)
                item = QTableWidgetItem()
                item.setData(Qt.UserRole, obj) 
                self.table.setItem(row, col, item)
                return

        unit = prop.get("unit", "")
        formatted_val = PropertyDelegate.format_value(val, prop)
        display_text = f"{formatted_val} {unit}".strip()

        item = QTableWidgetItem(display_text)
        item.setData(Qt.EditRole, val)
        item.setData(Qt.DisplayRole, display_text)
        item.setData(Qt.UserRole, obj)
        if prop.get("readonly", False):
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(row, col, item)
    
    def apply_batch_edit(self, origin_row: int, col: int, value: Any) -> None:
        if self._is_updating: return
        self._is_updating = True
        self.table.blockSignals(True) 
        
        try:
            prop = self.properties_list[col]
            selected_indices = self.table.selectionModel().selectedRows()
            selected_rows = {idx.row() for idx in selected_indices}
            
            if origin_row in selected_rows or not selected_rows:
                 selected_rows.add(origin_row)
            else:
                 selected_rows = {origin_row}

            for r in selected_rows:
                obj = self.item_list[r]
                PropertyDelegate.set_value(obj, prop, value)
                
                item = self.table.item(r, col)
                if item:
                    item.setData(Qt.EditRole, value)
                    item.setData(Qt.UserRole, obj)
                    
                    if not prop.get("always_visible", False):
                        unit = prop.get("unit", "")
                        formatted_val = PropertyDelegate.format_value(value, prop)
                        item.setData(Qt.DisplayRole, f"{formatted_val} {unit}".strip())
                
                if prop.get("always_visible", False):
                    widget = self.table.cellWidget(r, col)
                    if widget: 
                        self._update_persistent_widget_value(widget, value, prop)

        finally:
            self.table.blockSignals(False)
            self._is_updating = False

    def on_item_changed(self, item: QTableWidgetItem) -> None:
        if not item or self._is_updating: return
        
        self._is_updating = True
        try:
            row, col = item.row(), item.column()
            prop = self.properties_list[col]
            obj = item.data(Qt.UserRole) or self.item_list[row]
            val = item.data(Qt.EditRole)
            
            PropertyDelegate.set_value(obj, prop, val)
            
            if not prop.get("always_visible", False):
                unit = prop.get("unit", "")
                formatted_val = PropertyDelegate.format_value(val, prop)
                item.setData(Qt.DisplayRole, f"{formatted_val} {unit}".strip())
        finally:
            self._is_updating = False

    def _update_persistent_widget_value(self, widget: QWidget, val: Any, prop: PropertyDescriptor) -> None:
        """
        Updates a persistent widget's value and optionally its dynamic range 
        if no fixed range is provided in the property descriptor.
        """
        widget.blockSignals(True)
        try:
            # v23: Calculate Multiplier for sliders
            decimals = prop.get("decimals", 0)
            multiplier = 10 ** decimals

            if "value_range" not in prop:
                new_min, new_max = self._delegate._calculate_dynamic_limits(val)
                if hasattr(widget, "slider"):
                    # Apply multiplier to dynamic range limits
                    widget.slider.setRange(int(new_min * multiplier), int(new_max * multiplier))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setRange(new_min, new_max)
                elif isinstance(widget, QSpinBox):
                    widget.setRange(int(new_min), int(new_max))

            if hasattr(widget, "input_widget"): 
                widget.input_widget.setChecked(bool(val))
            elif hasattr(widget, "slider"): 
                # Apply multiplier to value
                widget.slider.setValue(int(float(val) * multiplier))
                # Update the label manually
                if hasattr(widget, "label"):
                    unit = prop.get("unit", "")
                    widget.label.setText(f"{float(val):.{decimals}f} {unit}".strip())
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(val))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(float(val)) if val is not None else 0)
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(val))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(val))
            elif isinstance(widget, QTimeEdit):
                widget.setTime(PropertyDelegate._to_qtime(val))
            elif isinstance(widget, QDateEdit):
                widget.setDate(PropertyDelegate._to_qdate(val))
            elif isinstance(widget, QPushButton):
                self._delegate._update_color_button_face(widget, val)
        finally:
            widget.blockSignals(False)

    def _show_context_menu(self, pos: QPoint) -> None:
        item = self.table.itemAt(pos)
        if item is None: return
        col = item.column()
        prop = self.properties_list[col]
        if "default" not in prop: return
        default_val = prop["default"]
        
        menu = QMenu(self.table)
        action_text = f"Reset to {default_val}"
        if "unit" in prop: action_text += f" {prop['unit']}"
        action = menu.addAction(action_text)
        
        if menu.exec(self.table.viewport().mapToGlobal(pos)) == action:
            self.apply_batch_edit(item.row(), col, default_val)


# ──────────────────────────────────────────────────────────────
# Example Usage
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    class Layer:
        def __init__(self, name, priority=1, color="#FF0000"):
            self.material_name = name
            self.thickness = 50.0 
            self.opacity = 0.5
            self.priority = priority
            self.rotation = 90
            self.color = color
            self.start_time = QTime(9, 30)
            self.duration = QTime(1, 45, 30) 
            self.creation_date = QDate.currentDate()
            self.coherent = "Incoherent"
            self.is_active = True

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion") 

    props: List[PropertyDescriptor] = [
        {"attr": "material_name", "label": "Material", "widget_type": QLineEdit, "value_type": "alphabetic", "default": "Vacuum"},
        
        {"attr": "thickness", "label": "Thickness (Fixed)", "widget_type": QDoubleSpinBox, "unit": "um", "value_range": (0.0, 100.0), "default": 50.0},
        
        # v23: New Slider Precision Test
        # Range 0.0 to 1.0, with 2 decimals (100 steps on slider)
        {"attr": "opacity", "label": "Opacity (0.00-1.00)", "widget_type": QSlider, "decimals": 2,"always_visible": True, "default": 0.5},

        {"attr": "priority", "label": "Priority (Dynamic)", "widget_type": QSpinBox, "default": 1, "always_visible": True},
        
        {"attr": "rotation", "label": "Rotation (Dynamic)", "widget_type": QSlider, "unit": "deg", "always_visible": True, "default": 90},
        
        {"attr": "is_active", "label": "Active (Checkbox)", "widget_type": QCheckBox, "always_visible": True, "default": True},
        {"attr": "color", "label": "Color", "widget_type": QPushButton, "always_visible": True, "with_label": False, "default": "#FFFFFF"},
        
        {"attr": "start_time", "label": "Start (HH:mm)", "widget_type": QTimeEdit, "precision": "min", "default": "09:00"},
        {"attr": "duration", "label": "Duration (HH:mm:ss)", "widget_type": QTimeEdit, "precision": "secs", "default": "01:45:30"},
        
        {"attr": "creation_date", "label": "Date", "widget_type": QDateEdit, "default": "2023-01-01"},
    ]

    layers = [
        Layer("Silicon", 1, "#FF5733"), 
        Layer("Gold", 5, "#FFC300"), 
        Layer("Air", 10, "#DAF7A6"), 
    ]
    
    win = QMainWindow()
    dock = TableDockWidget(
        title="Layer Manager (v23 - Slider Precision)", 
        properties_list=props, 
        item_list=layers,
        alternating_rows=True,
        alternate_color="darker"
    )
    
    win.setCentralWidget(dock)
    win.resize(1000, 400)
    win.show()

    sys.exit(app.exec())