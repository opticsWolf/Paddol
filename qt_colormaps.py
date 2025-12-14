# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
# import json
from typing import List, Optional, Union, Dict, Any

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QPushButton, QHBoxLayout, QGroupBox, QScrollArea,
    QRadioButton, QButtonGroup, QSizePolicy, QComboBox,
    QLineEdit, QMessageBox, QDialog, QFrame, QDialogButtonBox
)
from PySide6.QtGui import (
    QImage, QPixmap, QColor, QPainterPath,
    QPainter, QIcon, QPaintEvent
)
from PySide6.QtCore import Qt, Signal, QRectF, QTimer, QSize


import numpy as np
from colorengine import OklabEngine, ColorMapsPresetManager
from qt_icons import ICON_DICT
from qt_colorwheel import ColorGeneratorDialog

# ------------------------------------------------------------
def numpy_to_qpixmap(data: np.ndarray, height: int = 100) -> QPixmap:
    """
    Converts an (N, 3) NumPy array of RGB float values (0.0-1.0) into a QPixmap.

    This conversion is optimized for speed by creating a QImage directly
    from the array buffer, avoiding unnecessary copying where possible.

    Args:
        data (np.ndarray): The input NumPy array of shape (N, 3) where N is the
            number of color steps, and each row is [R, G, B] floats (0.0-1.0).
        height (int): The target height for the resulting QPixmap (default 100).
            Note: The generated QImage internally has a height of 1 pixel before
            being converted to a stretchable QPixmap.

    Returns:
        QPixmap: A QPixmap ready for display.

    Raises:
        ValueError: If the input data is not an (N, 3) array.
    """
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Input data must be an (N, 3) array.")

    # Ensure contiguous array for QImage, cast to uint8 [0-255]
    rgb_int = np.clip(data * 255.0, 0, 255).astype(np.uint8)

    # Check for C-contiguous to avoid copying if possible
    if not rgb_int.flags['C_CONTIGUOUS']:
        rgb_int = np.ascontiguousarray(rgb_int)

    height_img, width_img = 1, rgb_int.shape[0]
    bytes_per_line = width_img * 3

    # Create QImage directly from data buffer
    q_image = QImage(
        rgb_int.data,
        width_img,
        height_img,
        bytes_per_line,
        QImage.Format_RGB888
    )

    # Copy to decouple from numpy array memory
    pixmap = QPixmap.fromImage(q_image.copy())
    return pixmap


# --- UI Components ---
class ColorRow(QWidget):
    """
    A single widget row displaying a color point in the gradient list,
    with buttons for changing the color, inserting a new color below, and removing itself.

    Signals:
        colorChanged: Emitted when the color of this row is modified.
        removeRequested (object): Emits self when the remove button is clicked.
        insertRequested (object): Emits self when the insert button is clicked.
    """

    colorChanged = Signal()
    removeRequested = Signal(object)  # Emits self
    insertRequested = Signal(object)  # Emits self (New Signal)

    ICON_BUTTON_STYLE = """
        QPushButton {
            background-color: #333;
            color: #ccc;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 0px; /* CRITICAL: Remove padding to fit inside 28px */
            font-weight: bold;
            font-size: 16px; /* Larger font for the symbol */
        }
        QPushButton:hover { 
            background-color: #444; 
            color: white;
            border: 1px solid #666; 
        }
        QPushButton:pressed { 
            background-color: #222; 
            border: 1px solid #3daee9; 
            color: #3daee9; 
        }
    """

    def __init__(self, color: QColor, can_remove: bool = True) -> None:
        """
        Initializes the ColorRow widget.

        Args:
            color (QColor): The initial color for this row.
            can_remove (bool): Whether the remove button should be visible initially.
        """
        super().__init__()
        self._color = color

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(5)

        # 1. Color Button
        self.btn_color = QPushButton()
        self.btn_color.setFixedHeight(28)
        self.btn_color.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_color.clicked.connect(self.open_color_dialog)
        self.update_button_style()

        # 2. Insert Button (+)
        self.btn_insert = QPushButton("")
        self.btn_insert.setIcon(QIcon(ICON_DICT['plus_']))
        self.btn_insert.setStyleSheet(self.ICON_BUTTON_STYLE)
        self.btn_insert.setFixedSize(28, 28)
        self.btn_insert.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_insert.setToolTip("Insert new color below")
        # Emit self so parent knows which row triggered the insert
        self.btn_insert.clicked.connect(lambda: self.insertRequested.emit(self))

        # 3. Remove Button (x)
        self.btn_remove = QPushButton("")
        self.btn_remove.setIcon(QIcon(ICON_DICT['x']))
        self.btn_remove.setFixedSize(28, 28)
        self.btn_remove.setStyleSheet(self.ICON_BUTTON_STYLE)
        self.btn_remove.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_remove.setToolTip("Remove this color")
        self.btn_remove.clicked.connect(lambda: self.removeRequested.emit(self))
        self.btn_remove.setVisible(can_remove)

        # 4. Add to Layout
        layout.addWidget(self.btn_color, 1)  # Stretch factor 1
        layout.addWidget(self.btn_insert, 0) # Fixed size
        layout.addWidget(self.btn_remove, 0) # Fixed size
        self.setLayout(layout)

    def color(self) -> QColor:
        """
        Returns the QColor currently set for this row.

        Returns:
            QColor: The current color.
        """
        return self._color

    def set_remove_enabled(self, enabled: bool) -> None:
        """
        Sets the visibility of the remove button.

        Args:
            enabled (bool): True to show the remove button, False to hide it.
        """
        self.btn_remove.setVisible(enabled)

    def open_color_dialog(self) -> None:
        """Opens the Color Picker Tool Dialog to select a new color."""
        dlg = ColorGeneratorDialog(
            parent=self,
            seed_color=self._color.name(),
            color_rule="Single",
            # allowed_rules=["Single"]  # NEW: Restrict to only Single mode
        )

        if dlg.exec() == QDialog.DialogCode.Accepted:
            colors, _ = dlg.get_colors()
            if colors:
                self._color = colors[0]
                self.update_button_style()
                self.colorChanged.emit()

    def update_button_style(self) -> None:
        """Updates the button's background color and text based on the current color."""
        # Determine contrast color
        text_col = "black" if self._color.lightness() > 128 else "white"
        hex_code = self._color.name().upper()

        self.btn_color.setText(f"{hex_code}")
        self.btn_color.setStyleSheet(
            f"background-color: {hex_code}; "
            f"color: {text_col}; "
            "font-weight: bold; "
            "border: 1px solid #555; border-radius: 4px;"
        )


class RoundedGradientWidget(QWidget):
    """
    A custom widget that renders a QPixmap with dynamic rounded corners.

    The widget uses a QPainterPath clipping mask to ensure the gradient
    texture is displayed with fixed, rounded edges regardless of the widget's size.
    """

    def __init__(self, parent: Optional[QWidget] = None, radius: int = 12):
        """
        Initializes the RoundedGradientWidget.

        Args:
            parent (Optional[QWidget]): The parent widget.
            radius (int): The radius for the rounded corners in pixels.
        """
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._radius = radius

        # Performance: Set a minimum size so it doesn't collapse
        self.setMinimumHeight(64)
        # Policy: Expand horizontally, fixed-ish vertically (or change as needed)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def set_gradient(self, pixmap: QPixmap) -> None:
        """
        Updates the stored gradient texture and triggers a repaint.

        Args:
            pixmap (QPixmap): The QPixmap containing the gradient texture.
        """
        self._pixmap = pixmap
        self.update()  # Schedule a paintEvent

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Renders the gradient using a clipping mask for rounded corners.

        Args:
            event (QPaintEvent): The paint event triggered by the Qt framework.
        """
        if not self._pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Create the clipping path based on CURRENT widget size
        path = QPainterPath()
        rect = QRectF(self.rect())
        path.addRoundedRect(rect, self._radius, self._radius)

        # 2. Activate Clipping
        painter.setClipPath(path)

        # 3. Draw the pixmap, stretching it to fill the current rect
        # The clipping ensures the corners are cut off cleanly
        painter.drawPixmap(self.rect(), self._pixmap)


class ColormapGenerator(QWidget):
    """
    A widget for creating, managing, and visualizing gradient colormaps.

    Includes preset management, Oklab interpolation support, and state validation
    to check if the current settings match the loaded preset.

    Signals:
        validityChanged (bool): Emitted when the 'dirty' state changes (True if clean/saved).
    """

    validityChanged = Signal(bool)

    GROUPBOX_STYLE = """
        QGroupBox {
            border: 1px solid #555;
            border-radius: 10px;
            margin-top: 10px;
            padding: 4px;
            padding-top: 8px;
            font-weight: bold;
            color: #ddd;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            left: 10px;
            background-color: #282828;
        }
    """

    COMBO_STYLE = """
        QComboBox {
            padding: 8px 15px;
            background: #333;
            color: white;
            border-radius: 8px;
            border: 1px solid #444;
            selection-background-color: #3daee9;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 30px;
            border-left-width: 0px;
            border-top-right-radius: 12px;
            border-bottom-right-radius: 12px;
        }
        QComboBox QAbstractItemView {
            background-color: #333;
            color: white;
            border: 1px solid #444;
            selection-background-color: #3daee9;
            selection-color: white;
            outline: none;
            padding: 4px;
        }
    """

    BUTTON_STYLE = """
        QPushButton {
            background-color: #333;
            color: white;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 8px 15px;
            font-weight: bold;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #444;
            border: 1px solid #555;
        }
        QPushButton:pressed {
            background-color: #222;
            border: 1px solid #3daee9;
            color: #3daee9;
        }
        QPushButton:disabled {
            background-color: #252525;
            color: #555;
            border: 1px solid #333;
        }
    """

    def __init__(self, current_cmap_name: str = "Viridis") -> None:
        """
        Initializes the ColormapGenerator.

        Args:
            current_cmap_name (str): The name of the preset to load initially.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Window configuration
        # ------------------------------------------------------------------
        self.setWindowTitle("Colormap Tool")
        self.resize(1024, 640)
        self.setWindowIcon(QIcon(ICON_DICT.get("colormap", "")))

        # ------------------------------------------------------------------
        # State tracking
        # ------------------------------------------------------------------
        # Stores the state of the currently loaded preset for 'dirty' checking
        self._clean_state: Dict[str, Any] = {"colors": [], "mode": "strict"}

        # Engine that does the colour math
        self.cmap_engine = OklabEngine()

        # ------------------------------------------------------------------
        # Preset handling
        # ------------------------------------------------------------------
        self.cmap_manager = ColorMapsPresetManager()
        load_error = self.cmap_manager.load_custom_presets_from_file()
        if load_error:
            print(f"Warning: Could not load custom presets: {load_error}")

        # Expose the manager’s data for convenience
        self.custom_presets = self.cmap_manager.custom_presets

        # Combine default and custom presets
        self.colormap_presets = self.cmap_manager.default_presets.copy()
        self.all_presets = self.colormap_presets.copy()
        self.all_presets.update(self.custom_presets)

        # ------------------------------------------------------------------
        # UI components
        # ------------------------------------------------------------------
        self.color_rows: List[ColorRow] = []
        self.color_rule: Optional[str] = None

        self._setup_ui()

        # ------------------------------------------------------------------
        # Preset selection (Fixed Logic)
        # ------------------------------------------------------------------
        if current_cmap_name not in self.all_presets:
            current_cmap_name = "Viridis"  # Fallback to default

        self._current_preset_name = current_cmap_name

        # CRITICAL FIX: Populate the list BEFORE trying to set the text
        self.refresh_preset_list(select_item=self._current_preset_name)

        # Use Timer to allow UI to settle before heavy lifting/layout calculation
        QTimer.singleShot(10, lambda: self.load_selected_preset(current_cmap_name))

    def _setup_ui(self) -> None:
        """Builds the main widget layout, consisting of a control panel and a visualizer."""
        main_h_layout = QHBoxLayout()
        main_h_layout.setContentsMargins(10, 10, 10, 10)
        main_h_layout.setSpacing(10)

        left_panel = QWidget()
        left_panel.setFixedWidth(360)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Preset Manager
        group_presets = QGroupBox("Preset Manager")
        group_presets.setStyleSheet(self.GROUPBOX_STYLE)
        layout_presets = QVBoxLayout()

        self.combo_presets = QComboBox()
        self.combo_presets.setStyleSheet(self.COMBO_STYLE)
        self.combo_presets.currentTextChanged.connect(self.load_selected_preset)
        layout_presets.addWidget(self.combo_presets)

        h_preset_btns = QHBoxLayout()
        self.input_preset_name = QLineEdit()
        self.input_preset_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.input_preset_name.setPlaceholderText("New preset name...")
        self.input_preset_name.setStyleSheet("""
            QLineEdit {
                padding: 8px 8px 8px 8px; 
                border: 1px solid #cccccc;
                border-radius: 8px;
            }
        """)

        self.btn_save_preset = self._make_btn(" Save", "Saves custom preset", ICON_DICT.get("save2"))
        self.btn_save_preset.clicked.connect(self.save_current_as_preset)

        self.btn_del_preset = self._make_btn(" Delete", "Removes custom preset", ICON_DICT.get("delete"))
        self.btn_del_preset.clicked.connect(self.delete_selected_preset)

        h_preset_btns.addWidget(self.input_preset_name)
        h_preset_btns.addWidget(self.btn_save_preset)
        h_preset_btns.addWidget(self.btn_del_preset)
        layout_presets.addLayout(h_preset_btns)
        group_presets.setLayout(layout_presets)

        left_layout.addWidget(group_presets)

        # 2. Mode Selection
        group_modes = QGroupBox("Interpolation Mode")
        group_modes.setStyleSheet(self.GROUPBOX_STYLE)
        layout_modes = QVBoxLayout()

        self.radio_strict = QRadioButton("Strict (Use exact colors)")
        self.radio_balanced = QRadioButton("Balanced (50% Mix)")
        self.radio_luma = QRadioButton("Luminosity (Linearize Lightness)")

        self.radio_strict.setChecked(True)
        self.radio_balanced.setToolTip("Averages your chosen brightness with a perfectly linear gradient.")

        self.btn_group_modes = QButtonGroup(self)
        self.btn_group_modes.addButton(self.radio_strict)
        self.btn_group_modes.addButton(self.radio_balanced)
        self.btn_group_modes.addButton(self.radio_luma)

        self.btn_group_modes.buttonToggled.connect(self._on_mode_changed)

        layout_modes.addWidget(self.radio_strict)
        layout_modes.addWidget(self.radio_balanced)
        layout_modes.addWidget(self.radio_luma)
        group_modes.setLayout(layout_modes)

        left_layout.addWidget(group_modes)

        # 3. Color List
        group_colors = QGroupBox("Gradient Colors")
        group_colors.setStyleSheet(self.GROUPBOX_STYLE)
        layout_colors = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.color_list_widget = QWidget()
        self.color_list_layout = QVBoxLayout()
        self.color_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.color_list_layout.setContentsMargins(0, 0, 0, 0)
        self.color_list_widget.setLayout(self.color_list_layout)

        self.scroll_area.setWidget(self.color_list_widget)
        layout_colors.addWidget(self.scroll_area)

        action_layout = QHBoxLayout()
        self.btn_pick_screen = self._make_btn(" Color Wheel", "Get harmonic colors", ICON_DICT.get("color_wheel"))
        self.btn_pick_screen.clicked.connect(self.open_harmony_generator)

        self.btn_invert = self._make_btn(" Reverse", "Reverse order", ICON_DICT.get("reverse"))
        self.btn_invert.clicked.connect(self.reverse_colors)

        action_layout.addWidget(self.btn_pick_screen)
        action_layout.addWidget(self.btn_invert)
        layout_colors.addLayout(action_layout)

        group_colors.setLayout(layout_colors)
        left_layout.addWidget(group_colors)

        left_panel.setLayout(left_layout)

        # --- Right Panel (Visualizer) ---
        group_visualizer = QGroupBox("Generated Colormap")
        group_visualizer.setStyleSheet(self.GROUPBOX_STYLE)
        group_visualizer_layout = QVBoxLayout()
        group_visualizer_layout.setContentsMargins(10, 10, 10, 10)
        group_visualizer_layout.setSpacing(5)

        # Assuming radius logic handled internally or via property
        self.image_label = RoundedGradientWidget()
        self.image_label.setMinimumHeight(100)

        group_visualizer_layout.addWidget(self.image_label, 1)

        group_visualizer.setLayout(group_visualizer_layout)

        main_h_layout.addWidget(left_panel)
        main_h_layout.addWidget(group_visualizer, 1)
        self.setLayout(main_h_layout)

    def _make_btn(self, text: str, tooltip: str, icon_path: Optional[str]) -> QPushButton:
        """
        Helper to create standardized buttons with pre-applied styling.

        Args:
            text (str): The text displayed on the button.
            tooltip (str): The tooltip text.
            icon_path (Optional[str]): Path to the icon image.

        Returns:
            QPushButton: The standardized button instance.
        """
        btn = QPushButton()
        btn.setText(text)
        if icon_path:
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(24, 24))
        btn.setToolTip(tooltip)
        btn.setStyleSheet(self.BUTTON_STYLE)
        return btn

    def _on_mode_changed(self, button: QRadioButton, checked: bool) -> None:
        """Triggers colormap generation when an interpolation mode radio button is toggled."""
        if checked:
            self.generate_colormap()

    # --- PRESET LOGIC ---

    def refresh_preset_list(self, select_item: Optional[str] = None) -> None:
        """
        Rebuilds the contents of the preset combobox, including separators for custom presets.

        Args:
            select_item (Optional[str]): The item name to select after refresh. If None,
                it defaults to the first item.
        """
        # 1. Rebuild list safely
        self.combo_presets.blockSignals(True)

        self.combo_presets.clear()

        all_presets_keys = sorted(self.colormap_presets.keys())
        custom_keys = sorted(self.custom_presets.keys())

        self.combo_presets.addItems(all_presets_keys)

        if custom_keys:
            # Add separator at the current end of the list
            self.combo_presets.insertSeparator(self.combo_presets.count())
            self.combo_presets.addItems(custom_keys)

        self.combo_presets.blockSignals(False)

        if select_item is None:
            self.combo_presets.setCurrentIndex(0)
            select_item = self.combo_presets.currentText()

        # 3. Restore/Set Selection
        idx = self.combo_presets.findText(select_item)
        if idx >= 0:
            self.combo_presets.setCurrentIndex(idx)
        else:
            # Fallback: select 0 if exists
            if self.combo_presets.count() > 0:
                self.combo_presets.setCurrentIndex(0)

        # 4. Manually trigger load because we blocked signals
        current_text = self.combo_presets.currentText()
        if current_text:
            self.load_selected_preset(current_text)

    def load_selected_preset(self, preset_name: str) -> None:
        """
        Loads the color points and interpolation mode from a selected preset into the UI.

        Args:
            preset_name (str): The name of the preset to load.
        """
        if not preset_name or preset_name.startswith("---"):
            return

        # 1. Retrieve Data
        if preset_name in self.colormap_presets:
            data = self.colormap_presets[preset_name]
            self.btn_del_preset.setEnabled(False)
        elif preset_name in self.custom_presets:
            data = self.custom_presets[preset_name]
            self.btn_del_preset.setEnabled(True)
        else:
            return

        # 2. Update Internal Clean State
        self._current_preset_name = preset_name
        self._clean_state = {
            "colors": list(data.get("colors", [])),
            "mode": data.get("mode", "strict")
        }

        # 3. Update UI (Block signals to prevent intermediate dirty checks)
        self.color_list_widget.setUpdatesEnabled(False)
        self.blockSignals(True)
        try:
            mode = self._clean_state["mode"]
            if mode == "luma": self.radio_luma.setChecked(True)
            elif mode == "balanced": self.radio_balanced.setChecked(True)
            else: self.radio_strict.setChecked(True)

            while self.color_rows:
                row = self.color_rows.pop()
                self.color_list_layout.removeWidget(row)
                row.deleteLater()

            for hex_c in self._clean_state["colors"]:
                self.add_color_point_internal(QColor(hex_c))
        finally:
            self.blockSignals(False)

        self.update_ui_state()
        self.color_list_widget.setUpdatesEnabled(True)

        # 4. Generate visual and VALIDATE
        QTimer.singleShot(10, self.generate_colormap)
        # Explicitly valid since we just loaded
        self.validityChanged.emit(True)

    def save_current_as_preset(self) -> None:
        """Saves the current gradient configuration as a new custom preset."""
        name = self.input_preset_name.text().strip()
        if not name or name in self.colormap_presets:
            QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty or match a default preset.")
            return

        # 1. Gather current state
        colors = [row.color().name() for row in self.color_rows]
        if self.radio_luma.isChecked(): mode = "luma"
        elif self.radio_balanced.isChecked(): mode = "balanced"
        else: mode = "strict"

        # 2. Save to dict and file
        self.custom_presets[name] = {"colors": colors, "mode": mode}
        self.cmap_manager.custom_presets = self.custom_presets
        err = self.cmap_manager.save_custom_presets_to_file()
        if err: QMessageBox.critical(self, "Error", f"Save failed: {err}")

        # 3. Update UI cleanly
        self.input_preset_name.clear()

        # 4. Refresh List AND Select New Item
        self.refresh_preset_list(select_item=name)

    def delete_selected_preset(self) -> None:
        """Deletes the currently selected custom preset."""
        name = self.combo_presets.currentText()
        if name in self.custom_presets:
            reply = QMessageBox.question(
                self, 'Confirm Delete',
                f"Are you sure you want to delete '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                # 1. Delete
                del self.custom_presets[name]
                self.cmap_manager.custom_presets = self.custom_presets
                self.cmap_manager.save_custom_presets_to_file()

                # 2. Refresh (Defaults to index 0 if name not found)
                self.refresh_preset_list(select_item=None)

    # --- COLOR LOGIC ---

    def check_dirty_state(self) -> None:
        """
        Compares the current UI state with the last loaded 'clean' state.

        Emits `validityChanged(True)` if the state is clean (matches the loaded preset)
        and `validityChanged(False)` otherwise.
        """
        if self.signalsBlocked(): return

        # 1. Get Current UI State
        current_colors = [row.color().name() for row in self.color_rows]

        current_mode = "strict"
        if self.radio_balanced.isChecked(): current_mode = "balanced"
        elif self.radio_luma.isChecked(): current_mode = "luma"

        # 2. Compare with Clean State
        clean_colors = self._clean_state.get("colors", [])
        clean_mode = self._clean_state.get("mode", "strict")

        is_clean = (current_colors == clean_colors) and (current_mode == clean_mode)

        # 3. Emit Result
        self.validityChanged.emit(is_clean)

    def open_harmony_generator(self) -> None:
        """Opens a dialog to generate a new list of harmonizing colors based on a seed color."""
        seed_color = QColor(128, 128, 128)
        if self.color_rows:
            seed_color = self.color_rows[0].color()

        # Determine valid start rule (fallback if previous was "Single")
        start_rule = self.color_rule
        if not start_rule or start_rule == "Single":
            start_rule = "Complementary"

        dlg = ColorGeneratorDialog(
            parent=self,
            seed_color=seed_color.name(),
            color_rule=start_rule,
            icon=ICON_DICT.get("color_wheel"),
            excluded_rules=["Single"]  # NEW: Remove Single option from harmonies
        )

        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_colors, color_rule = dlg.get_colors()
            if not new_colors: return

            self.blockSignals(True)
            self.color_list_widget.setUpdatesEnabled(False)

            self.color_rule = color_rule

            while self.color_rows:
                row = self.color_rows.pop()
                self.color_list_layout.removeWidget(row)
                row.deleteLater()

            for c in new_colors:
                self.add_color_point_internal(c)

            self.blockSignals(False)
            self.update_ui_state()
            self.color_list_widget.setUpdatesEnabled(True)
            self.generate_colormap()

    def add_color_point(self, color: Union[QColor, bool] = None) -> None:
        """
        Adds a new color point row to the list with a default or specified color.

        Args:
            color (Union[QColor, bool], optional): The color to add. Defaults to gray if None or invalid.
        """
        if not isinstance(color, QColor) or not color.isValid():
            color = QColor(128, 128, 128)

        if len(self.color_rows) >= 10:
            return

        self.add_color_point_internal(color)

        if not self.signalsBlocked():
            self.update_ui_state()
            self.generate_colormap()

    def add_color_point_internal(self, color: QColor) -> None:
        """
        Creates and connects a new ColorRow, then adds it to the layout.

        Args:
            color (QColor): The color for the new row.
        """
        row = ColorRow(color)
        row.colorChanged.connect(self.generate_colormap)
        row.removeRequested.connect(self.remove_color_point)
        row.insertRequested.connect(self.insert_color_point)

        self.color_rows.append(row)
        self.color_list_layout.addWidget(row)

    def insert_color_point(self, target_row: ColorRow) -> None:
        """Inserts a new color row immediately after the target row."""
        if len(self.color_rows) >= 10: return
        try:
            idx = self.color_rows.index(target_row)
        except ValueError:
            return

        new_row = ColorRow(target_row.color())
        new_row.colorChanged.connect(self.generate_colormap)
        new_row.removeRequested.connect(self.remove_color_point)
        new_row.insertRequested.connect(self.insert_color_point)

        insert_idx = idx + 1
        self.color_rows.insert(insert_idx, new_row)
        self.color_list_layout.insertWidget(insert_idx, new_row)

        self.update_ui_state()
        self.generate_colormap()

    def remove_color_point(self, row: ColorRow) -> None:
        """Removes the specified color row from the list."""
        if len(self.color_rows) <= 2: return
        self.color_list_layout.removeWidget(row)
        row.deleteLater()
        self.color_rows.remove(row)
        self.update_ui_state()
        self.generate_colormap()

    def reverse_colors(self) -> None:
        """Reverses the order of the colors in the gradient list."""
        if len(self.color_rows) < 2: return
        self.color_rows.reverse()
        self.color_list_widget.setUpdatesEnabled(False)
        for row in self.color_rows:
            self.color_list_layout.addWidget(row)
        self.color_list_widget.setUpdatesEnabled(True)
        self.generate_colormap()

    def update_ui_state(self) -> None:
        """Updates UI elements based on the number of color points (e.g., disables remove if only 2 colors remain)."""
        can_remove = len(self.color_rows) > 2
        for row in self.color_rows:
            row.set_remove_enabled(can_remove)

    def generate_colormap(self) -> None:
        """Generates the gradient using the Oklab engine and updates the visualizer."""
        if self.signalsBlocked(): return

        # 1. Generate Gradient
        if len(self.color_rows) < 2:
            self.image_label.set_gradient(None)
            return

        colors = [row.color().name() for row in self.color_rows]

        mode = "strict"
        if self.radio_balanced.isChecked(): mode = "balanced"
        elif self.radio_luma.isChecked(): mode = "luma"

        try:
            # Assuming this returns a numpy array or similar
            rgb_array = self.cmap_engine.generate_gradient(
                colors=colors,
                mode=mode,
                n_steps=1024
            )
            # Convert to pixmap for display
            raw_pixmap = numpy_to_qpixmap(rgb_array, height=1)
            self.image_label.set_gradient(raw_pixmap)

        except ValueError:
            self.image_label.set_gradient(None)

        # 2. CRITICAL: Check Dirty State
        self.check_dirty_state()

    def get_cmaps(self) -> Dict[str, Any]:
        """
        Returns the combined dictionary of all colormap data, including presets and active name.

        Returns:
            Dict[str, Any]: A dictionary containing 'active', 'names', and 'cmaps' keys.
        """
        all_presets = {**self.custom_presets, **self.colormap_presets}
        cmap_dict = self.cmap_manager.return_cmaps(all_presets)

        all_preset_names = (sorted(self.colormap_presets.keys()) + sorted(self.custom_presets.keys()))

        return {
            "active": self._current_preset_name,
            "names": all_preset_names,
            "cmaps": cmap_dict
        }

class ColorMapDialog(QDialog):
    """
    A dialog wrapper for the ColormapGenerator widget, providing OK/Cancel buttons
    and handling the 'dirty' state to conditionally enable the OK button.
    """

    BUTTON_STYLE = """
        QDialogButtonBox { dialogbuttonbox-buttons-have-icons: 0; }
        QPushButton {
            background-color: #333; color: white;
            border: 1px solid #444; border-radius: 8px;
            padding: 8px 15px; font-weight: bold; min-width: 80px;
        }
        QPushButton:hover { background-color: #444; border: 1px solid #555; }
        QPushButton:pressed { background-color: #222; border: 1px solid #3daee9; color: #3daee9; }
        QPushButton:disabled { background-color: #252525; color: #555; border: 1px solid #333; }
    """

    def __init__(self, parent=None, current_cmap_name='Viridis', icon_path=None):
        """
        Initializes the ColorMapDialog.

        Args:
            parent (Optional[QWidget]): The parent widget for the dialog.
            current_cmap_name (str): The name of the preset to load initially in the generator.
            icon_path (Optional[str]): Path to the dialog window icon.
        """
        super().__init__(parent)
        self.setWindowTitle("Colormap Generator")
        self.resize(1040, 650)
        if icon_path: self.setWindowIcon(QIcon(icon_path))

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        try:
            self._tool = ColormapGenerator(current_cmap_name=current_cmap_name)
            self._tool.validityChanged.connect(self._on_validity_changed)
            layout.addWidget(self._tool)
        except Exception as e:
            layout.addWidget(QLabel(f"Error initializing tool: {e}"))

        button_container = QWidget()
        button_container.setStyleSheet("background-color: #282828; border-top: 1px solid #444;")
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(15, 15, 15, 15)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.setStyleSheet(self.BUTTON_STYLE)
        self._ok_button = self._buttons.button(QDialogButtonBox.Ok)

        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self._buttons)
        button_container.setLayout(button_layout)

        layout.addWidget(button_container)

        # Initial Check
        self._on_validity_changed(True)

    def _on_validity_changed(self, is_valid: bool) -> None:
        """Handles the 'validity' signal from the generator, enabling or disabling the OK button."""
        if self._ok_button:
            self._ok_button.setEnabled(is_valid)
            if is_valid:
                self._ok_button.setToolTip("Select this preset")
            else:
                self._ok_button.setToolTip("Unsaved changes. Save as a new preset to continue.")


    def get_cmaps(self) -> dict:
        """
        Retrieves the colormap data from the internal generator tool.

        Returns:
            dict: The dictionary containing 'active', 'names', and 'cmaps'.
        """
        if hasattr(self, '_tool'):
            return self._tool.get_cmaps()
        return {}

    @staticmethod
    def get_colormaps(parent=None, current_cmap_name="Viridis", icon_path=None) -> dict:
        """
        Static helper to run the dialog and return the full result dict.

        Args:
            parent (Optional[QWidget]): The parent widget for the dialog.
            current_cmap_name (str): The name of the preset to load initially.
            icon_path (Optional[str]): Path to the dialog window icon.

        Returns:
            dict: The colormap data (containing 'active', 'names', 'cmaps') if accepted, else empty dict.
        """
        dialog = ColorMapDialog(parent, current_cmap_name, icon_path)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_cmaps()
        return {}


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    dlg = ColorMapDialog(current_cmap_name="Turbo")
    if dlg.exec() == QDialog.DialogCode.Accepted:
        print("Selected:", dlg.get_cmaps()['active'])
    sys.exit(0)