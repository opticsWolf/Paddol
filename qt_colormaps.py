# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
import json
from pathlib import Path
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Any

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QPushButton, QHBoxLayout, QGroupBox, QScrollArea,
    QColorDialog, QRadioButton, QButtonGroup,
    QSizePolicy, QComboBox, QLineEdit, QMessageBox,
    QDialog, QFrame, QToolButton, QDialogButtonBox
)
from PySide6.QtGui import (
    QGuiApplication, QImage, QPixmap, QColor, QPainterPath,
    QPainter, QAction, QResizeEvent, QIcon, QPaintEvent
)
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QTimer, QSize, QRectF

# Assumed external modules (Mocked for context if missing, but using imports as requested)
from qt_icons import ICON_DICT
from qt_colorwheel import ColorGeneratorDialog

from colorengine import OklabEngine, ColorMapsPresetManager

def numpy_to_qpixmap(data: np.ndarray, height: int = 100) -> QPixmap:
    """
    Optimized conversion from NumPy array to QPixmap.
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
    """A single row representing a color, with Insert and Remove actions."""
    
    colorChanged = Signal()
    removeRequested = Signal(object) # Emits self
    insertRequested = Signal(object) # Emits self (New Signal)

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
        return self._color
        
    def set_remove_enabled(self, enabled: bool) -> None: 
        self.btn_remove.setVisible(enabled)

    def open_color_dialog(self) -> None:
        """Opens the Color Picker Tool in a Dialog to replace current color."""
        # Note: Ensure ColorGeneratorDialog is imported/available in this scope
        dlg = ColorGeneratorDialog(
            parent=self, 
            seed_color=self._color.name(), 
            color_rule="Single"
        )
        
        if dlg.exec() == QDialog.DialogCode.Accepted:
            colors, _ = dlg.get_colors()
            
            if colors:
                self._color = colors[0]
                self.update_button_style()
                self.colorChanged.emit()

    def update_button_style(self) -> None:
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
    """A custom widget that renders a QPixmap with dynamic rounded corners.
    
    This widget ensures that even when resized, the border radius remains 
    fixed and circular, while the content scales to fill the space.
    """

    def __init__(self, parent: Optional[QWidget] = None, radius: int = 12):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._radius = radius
        
        # Performance: Set a minimum size so it doesn't collapse
        self.setMinimumHeight(64)
        # Policy: Expand horizontally, fixed-ish vertically (or change as needed)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def set_gradient(self, pixmap: QPixmap) -> None:
        """Update the stored gradient texture and trigger a repaint."""
        self._pixmap = pixmap
        self.update()  # Schedule a paintEvent

    def paintEvent(self, event: QPaintEvent) -> None:
        """Render the gradient with a clipping mask."""
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
    
    # 1. New Signal
    validityChanged = Signal(bool)
    
    GROUPBOX_STYLE = """
        QGroupBox {
            border: 1px solid #555;
            border-radius: 10px;
            margin-top: 10px; /* Leave space for the title */
            padding: 4px;
            padding-top: 8px;
            font-weight: bold;
            color: #ddd;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            left: 10px; /* Indent the title */
            background-color: #282828; /* Match window background to hide border behind text */
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
            background-color: #444; /* Slightly lighter on hover */
            border: 1px solid #555;
        }
        QPushButton:pressed {
            background-color: #222; /* Darker when clicked */
            border: 1px solid #3daee9; /* Blue border match */
            color: #3daee9;
        }
        QPushButton:disabled {
            background-color: #252525;
            color: #555;
            border: 1px solid #333;
        }
    """
    
    def __init__(self, current_cmap_name: str = "Viridis") -> None:
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
        self._clean_state: dict = {"colors": [], "mode": "strict"}

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
        self.color_rule = None

        self._setup_ui()

        # ------------------------------------------------------------------
        # Preset selection
        # ------------------------------------------------------------------
        if current_cmap_name not in self.all_presets:
            current_cmap_name = "Viridis"  # Fallback to default
        self._current_preset_name = current_cmap_name

        self.combo_presets.setCurrentText(self._current_preset_name)
        QTimer.singleShot(10, lambda: self.load_selected_preset(current_cmap_name))

    def _setup_ui(self) -> None:
        main_h_layout = QHBoxLayout()
        main_h_layout.setContentsMargins(10, 10, 10, 10)
        main_h_layout.setSpacing(10)

        # --- Left Panel (Controls) ---
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
        self.refresh_preset_list()
        self.combo_presets.currentTextChanged.connect(self.load_selected_preset)
        layout_presets.addWidget(self.combo_presets)
        
        h_preset_btns = QHBoxLayout()
        self.input_preset_name = QLineEdit()
        self.input_preset_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input_preset_name.setPlaceholderText("New preset name...")
        self.input_preset_name.setStyleSheet(
            """
            QLineEdit {
                /* Padding set to 5px top/bottom to match QPushButton height */
                padding: 8px 8px 8px 8px; 
                border: 1px solid #cccccc;
                border-radius: 8px;
            }
            """
        )     

        self.btn_save_preset = self._make_btn(" Save", "Saves custome preset to template", ICON_DICT.get("save2"))
        self.btn_save_preset.clicked.connect(self.save_current_as_preset)
        
        self.btn_del_preset = self._make_btn(" Delete", "Removes custome preset", ICON_DICT.get("delete"))
        #self.btn_del_preset.setStyleSheet("color: red;")
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

        self.btn_group_modes = QButtonGroup(self) # Parent to self for GC safety
        self.btn_group_modes.addButton(self.radio_strict)
        self.btn_group_modes.addButton(self.radio_balanced)
        self.btn_group_modes.addButton(self.radio_luma)
        
        # Connect to a single handler instead of three lambdas
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
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        self.color_list_widget = QWidget()
        self.color_list_layout = QVBoxLayout()
        self.color_list_layout.setAlignment(Qt.AlignTop)
        self.color_list_layout.setContentsMargins(0, 0, 0, 0)
        self.color_list_widget.setLayout(self.color_list_layout)
        
        self.scroll_area.setWidget(self.color_list_widget)
        layout_colors.addWidget(self.scroll_area)
        
        action_layout = QHBoxLayout()
        #self.btn_add = self._make_btn(" Add Color", "Add a new color to the gradient", ICON_DICT.get("plus"))
        #self.btn_add.clicked.connect(self.add_color_point)
        
        self.btn_pick_screen = self._make_btn(" Color Wheel", "Get harmonic colors with a color wheel", ICON_DICT.get("color_wheel"))
        self.btn_pick_screen.clicked.connect(self.open_harmony_generator)

        self.btn_invert = self._make_btn(" Reverse", "Reverse order", ICON_DICT.get("reverse"))
        self.btn_invert.clicked.connect(self.reverse_colors)
        
        #action_layout.addWidget(self.btn_add)
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
        
        # The image_label setup remains critical for performance and scaling
        self.image_label = RoundedGradientWidget(radius=8)
        # CRITICAL: Ignored for width/height preference, Expanding for vertical space
        #self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        #self.image_label.setScaledContents(True) # Ensures the fixed-res pixmap stretches
        self.image_label.setMinimumHeight(100)
        
        # Add the image label with a stretch factor to ensure it expands vertically
        # This is generally cleaner than relying on addStretch() at the end
        group_visualizer_layout.addWidget(self.image_label, 1) # Set stretch factor to 1 (takes all space)
        # Removed the unnecessary group_visualizer_layout.addStretch(1) from the previous iteration.
        
        group_visualizer.setLayout(group_visualizer_layout)
        
        # --- Main Layout Integration ---
        main_h_layout.addWidget(left_panel)
        main_h_layout.addWidget(group_visualizer, 1) # Set stretch factor to 1 (takes all remaining horizontal space)
        self.setLayout(main_h_layout)

    def _make_btn(self, text: str, tooltip: str, icon_path: Optional[str]) -> QPushButton:
        """
        Creates an optimized QPushButton, ensuring both the icon and text are visible 
        and includes spacing between the icon and the button edge via stylesheet.
    
        Args:
            self: Reference to the parent widget/object.
            text: The visible text for the button.
            tooltip: The tooltip text.
            icon_path: Optional path to an icon file.
    
        Returns:
            QPushButton: The configured button instance.
        """
        btn = QPushButton()
        
        # 1. Set the text unconditionally
        btn.setText(text)
        
        if icon_path:
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(24, 24))
            
        btn.setToolTip(tooltip)
        
        # 2. FIX: Add a stylesheet to introduce padding.
        # We use padding-left to push the icon/text away from the left edge.
        # The default layout engine then naturally centers the text relative to the icon.
        # Adjust '8px' to control the final spacing.
        
        btn.setStyleSheet(self.BUTTON_STYLE)
        
        return btn

    def _on_mode_changed(self, button, checked):
        if checked:
            self.generate_colormap()

    def load_custom_presets_from_file(self) -> None:
        """
        Delegates loading of custom presets to the ColorMapsPresetManager.
        
        The manager handles the file I/O and updates self.custom_presets (which 
        is aliased to the manager's internal storage).
        """
        # The manager handles all file I/O and dictionary updating.
        return self.cmap_manager.load_custom_presets_from_file()

    def save_custom_presets_to_file(self) -> None:
        """
        Delegates saving of custom presets to the ColorMapsPresetManager.
        
        Handles any exception returned by the manager by displaying a critical 
        QMessageBox to the user.
        """
        # 1. Update manager's internal state (already aliased, but good practice 
        #    if you ever copy the dict instead of aliasing)
        self.cmap_manager.custom_presets = self.custom_presets
        
        # 2. Delegate save and check the return value
        error = self.cmap_manager.save_custom_presets_to_file()
        
        # 3. Handle the returned exception at the UI level
        if error is not None:
            QMessageBox.critical(self, "Error", f"Save failed: {error}")

    def refresh_preset_list(self) -> None:
        current = self.combo_presets.currentText()
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()
        
        all_presets = sorted(self.colormap_presets.keys())
        custom_keys = sorted(self.custom_presets.keys())
        
        self.combo_presets.addItems(all_presets)
        if custom_keys:
            self.combo_presets.insertSeparator(self.combo_presets.count())
            self.combo_presets.addItems(custom_keys)

        self.combo_presets.blockSignals(False)
        if current:
            idx = self.combo_presets.findText(current)
            if idx >= 0:
                self.combo_presets.setCurrentIndex(idx)

    def load_selected_preset(self, preset_name: str) -> None:
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

        # 2. Update Internal Tracking
        self._current_preset_name = preset_name
        # Deep copy the lists to ensure reference safety
        self._clean_state = {
            "colors": list(data.get("colors", [])),
            "mode": data.get("mode", "strict")
        }

        # 3. UI Updates (Standard logic)
        self.color_list_widget.setUpdatesEnabled(False)
        self.blockSignals(True) # Block signals to prevent dirty check during build
        
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
            
        self.blockSignals(False) # Unblock
        self.update_ui_state()
        self.color_list_widget.setUpdatesEnabled(True)
        
        # 4. Generate and Explicitly mark as valid
        QTimer.singleShot(10, self.generate_colormap)
        # Force a validity emit because generate_colormap might not trigger it 
        # correctly if signals were blocked during setup
        self.validityChanged.emit(True) 

    def save_current_as_preset(self) -> None:
        name = self.input_preset_name.text().strip()
        if not name or name in self.colormap_presets: 
            QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty or match a default preset.")
            return

        colors = [row.color().name() for row in self.color_rows]
        if self.radio_luma.isChecked(): mode = "luma"
        elif self.radio_balanced.isChecked(): mode = "balanced"
        else: mode = "strict"
        
        # Save to dict
        self.custom_presets[name] = {"colors": colors, "mode": mode}
        self.save_custom_presets_to_file()
        
        # Update Tracking
        self._current_preset_name = name
        self._clean_state = {"colors": list(colors), "mode": mode}
        
        self.refresh_preset_list()
        self.combo_presets.setCurrentText(name)
        self.input_preset_name.clear()
        
        # Explicitly validate
        self.validityChanged.emit(True)

    def delete_selected_preset(self) -> None:
        name = self.combo_presets.currentText()
        if name in self.custom_presets:
            reply = QMessageBox.question(
                self, 'Confirm Delete',
                f"Are you sure you want to delete '{name}'?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                del self.custom_presets[name]
                self.save_custom_presets_to_file()
                self.refresh_preset_list()
                if self.combo_presets.count() > 0:
                    self.combo_presets.setCurrentIndex(0)
    
    # --- Color Stop Management ---
    def open_harmony_generator(self) -> None:
            """Opens the Color Picker Tool in a Dialog to replace current colors."""
            # 1. Seed with current Primary Color (first in list)
            seed_color = QColor(128, 128, 128)
            if self.color_rows:
                seed_color = self.color_rows[0].color()
            
            if self.color_rule:
                color_rule = self.color_rule
            else:
                color_rule = "Single"
            
            # 2. Open Dialog
            try:
                # UPDATE: Use keyword arguments to match the new constructor signature:
                # __init__(self, parent=None, seed_color='#FF0000', color_rule="Complementary")
                dlg = ColorGeneratorDialog(
                    parent=self, 
                    seed_color=seed_color.name(), # Convert QColor to Hex String
                    color_rule=color_rule, icon = ICON_DICT.get("color_wheel")   # Set a default rule or pass dynamically
                )
                
                if dlg.exec() == QDialog.Accepted:
                    
                    new_colors, color_rule = dlg.get_colors()
                    if not new_colors: return
                    
                    # 3. Replace Logic
                    self.blockSignals(True)
                    self.color_list_widget.setUpdatesEnabled(False)
                    
                    self.color_rule = color_rule
                    
                    # Clear existing
                    while self.color_rows:
                        row = self.color_rows.pop()
                        self.color_list_layout.removeWidget(row)
                        row.deleteLater()
                    
                    # Add new
                    for c in new_colors:
                        self.add_color_point_internal(c)
                        
                    self.blockSignals(False)
                    self.update_ui_state()
                    self.color_list_widget.setUpdatesEnabled(True)
                    self.generate_colormap()
                    
            except Exception as e:
                # Avoid crashing the entire app if the dialog fails
                print(f"Error in Harmony Generator: {str(e)}")


    def add_color_point(self, color: Union[QColor, bool] = None) -> None:
        """Public slot to add a color point and trigger updates."""
        if not isinstance(color, QColor) or not color.isValid(): 
            color = QColor(128, 128, 128)
        
        if len(self.color_rows) >= 10:
            return
            
        self.add_color_point_internal(color)
        
        if not self.signalsBlocked():
            self.update_ui_state()
            self.generate_colormap()

    def add_color_point_internal(self, color: QColor) -> None:
        """Internal helper to add row without side effects (updates/generation)."""
        row = ColorRow(color)
        
        # Connect signals
        row.colorChanged.connect(self.generate_colormap)
        row.removeRequested.connect(self.remove_color_point)
        # NEW: Connect the insert request signal
        row.insertRequested.connect(self.insert_color_point)
        
        self.color_rows.append(row)
        self.color_list_layout.addWidget(row)

    def insert_color_point(self, target_row: ColorRow) -> None:
        """Inserts a new color row immediately after the target row."""
        # 1. Enforce Max Limit
        if len(self.color_rows) >= 10:
            return

        # 2. Find the position of the row that was clicked
        try:
            idx = self.color_rows.index(target_row)
        except ValueError:
            return # Safety check

        # 3. Create new row with the SAME color as the target
        new_color = target_row.color()
        new_row = ColorRow(new_color)
        
        # 4. Connect signals
        new_row.colorChanged.connect(self.generate_colormap)
        new_row.removeRequested.connect(self.remove_color_point)
        new_row.insertRequested.connect(self.insert_color_point)

        # 5. Insert into List and Layout at index + 1
        insert_idx = idx + 1
        self.color_rows.insert(insert_idx, new_row)
        self.color_list_layout.insertWidget(insert_idx, new_row)

        # 6. Update UI
        self.update_ui_state()
        self.generate_colormap()

    def remove_color_point(self, row: ColorRow) -> None:
        if len(self.color_rows) <= 2:
            return
        self.color_list_layout.removeWidget(row)
        row.deleteLater()
        self.color_rows.remove(row)
        self.update_ui_state()
        self.generate_colormap()

    def reverse_colors(self) -> None:
        """
        Reverses the order of the color stops (both logical list and UI layout).
        """
        if len(self.color_rows) < 2:
            return

        # 1. Reverse the logical list in O(N)
        self.color_rows.reverse()

        # 2. Update the Layout (Visuals)
        # Optimization: Instead of removing/creating widgets, we just re-add 
        # existing widgets to the layout in the new order. Qt moves them 
        # to the end, effectively sorting the layout.
        self.color_list_widget.setUpdatesEnabled(False)
        for row in self.color_rows:
            self.color_list_layout.addWidget(row)
        self.color_list_widget.setUpdatesEnabled(True)

        # 3. Update the Preview
        self.generate_colormap()

    def update_ui_state(self) -> None:
        #self.btn_add.setEnabled(len(self.color_rows) < 10)
        can_remove = len(self.color_rows) > 2
        for row in self.color_rows:
            row.set_remove_enabled(can_remove)

    # --- Gradient Generation and Display ---

    def generate_colormap(self) -> None:
        """Generates the gradient and updates the custom widget."""
        if self.signalsBlocked():
            return

        # 1. Handle Empty State
        if len(self.color_rows) < 2:
            # Use set_gradient(None) if your widget doesn't have a .clear() method
            if hasattr(self.image_label, 'clear'):
                self.image_label.clear()
            else:
                self.image_label.set_gradient(None)
            return
            
        colors = [row.color().name() for row in self.color_rows]
        
        mode = "luma" if self.radio_luma.isChecked() else "balanced" if self.radio_balanced.isChecked() else "strict"
        
        # Performance: 1024 steps for a high-quality horizontal lookup
        n_steps = 1024 
        
        try:
            rgb_array = self.cmap_engine.generate_gradient(
                colors=colors,
                mode=mode,
                n_steps=n_steps
            )

            # 2. Optimization: Generate 1px Height
            # We generate a 1x1024 strip. The RoundedGradientWidget's paintEvent
            # will automatically stretch this vertically to fill the rounded rect.
            raw_pixmap = numpy_to_qpixmap(rgb_array, height=1)
            
            # 3. Update the Widget
            # Pass the raw data; the widget handles the rounding and stretching.
            self.image_label.set_gradient(raw_pixmap)
            
        except ValueError as e:
            QMessageBox.critical(self, "Gradient Error", str(e))
            # Clear on error
            if hasattr(self.image_label, 'clear'):
                self.image_label.clear()
            else:
                self.image_label.set_gradient(None)

    def get_cmaps(self) -> dict:
        """
        Retrieves the active preset name, a sorted list of all names, and the combined data.

        Returns:
            dict: A collection containing:
                - 'active': (str) The currently selected preset name.
                - 'names': (list[str]) All preset names sorted alphabetically.
                - 'presets': (dict) The combined dictionary of default and custom data.
        """
        # 1. Safely merge dictionaries into a new object using unpacking
        # This prevents modifying self.colormap_presets permanently
        #Default presets overwrite custom presets with identical names
        all_presets = {**self.custom_presets, **self.colormap_presets}
        cmap_dict = {}
        
        # 2. Generate the sorted list of names for UI consistency
        all_preset_names = sorted([
            k for k in self.custom_presets.keys() 
            if k not in self.colormap_presets.keys()
        ])
        
        cmap_dict = self.cmap_manager.return_cmaps(all_presets)
    
        # 3. Return as a structured dictionary for cleaner access in the Dialog
        return {
            "active": self._current_preset_name,
            "names": all_preset_names,
            "cmaps": cmap_dict
        }

class ColorMapDialog(QDialog):
    """A dialog wrapper for the ColormapGenerator widget."""

    BUTTON_STYLE = """
        QDialogButtonBox {
            dialogbuttonbox-buttons-have-icons: 0;
        }
        QPushButton {
            background-color: #333;
            color: white;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 8px 15px;
            font-weight: bold;
            font-size: 12px;
            min-width: 80px;
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
    """

    WINDOW_WIDTH = 1040 # Slightly wider to accommodate the tool
    WINDOW_HEIGHT = 650

    def __init__(self, parent=None, current_cmap_name='Viridis', icon_path=None):
        super().__init__(parent)
        self.setWindowTitle("Colormap Generator")
        self.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        
        # Set Window Icon if provided
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        # --- Layout Setup ---
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        # --- Embed the Tool ---
        try:
            # Note: Your ColormapGenerator __init__ does not take arguments 
            # like seed_color or color_rule anymore.
            self._tool = ColormapGenerator(current_cmap_name=current_cmap_name)
            
            # CRITICAL: Connect the validity signal to the handler
            self._tool.validityChanged.connect(self._on_validity_changed)
            
            layout.addWidget(self._tool)
        except Exception as e:
            # Fallback for debugging
            label = QLabel(f"Error initializing tool: {e}")
            label.setStyleSheet("color: red; padding: 20px;")
            layout.addWidget(label)
            print(f"Error initializing ColormapGenerator: {e}")

        # --- Dialog Buttons (Ok/Cancel) ---
        button_container = QWidget()
        button_container.setStyleSheet("background-color: #282828; border-top: 1px solid #444;")
        button_layout = QHBoxLayout() 
        button_layout.setContentsMargins(15, 15, 15, 15)
        
        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.setStyleSheet(self.BUTTON_STYLE) # Apply style defined in class
        
        self._ok_button = self._buttons.button(QDialogButtonBox.Ok) # Reference to OK btn
        
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        
        button_layout.addStretch() # Push buttons to the right
        button_layout.addWidget(self._buttons)
        button_container.setLayout(button_layout)
        
        layout.addWidget(button_container)

        # Initialize button state (Should match tool's initial state)
        # Usually starts valid (loading default preset)
        self._on_validity_changed(True)

    def _on_validity_changed(self, is_valid: bool):
        """Slot to enable/disable OK button based on tool state."""
        if self._ok_button:
            self._ok_button.setEnabled(is_valid)
            
            # Optional: Add visual feedback
            if is_valid:
                self._ok_button.setToolTip("Select this preset")
            else:
                self._ok_button.setToolTip("You have unsaved changes. Save as a new preset to continue.")

    def get_cmaps(self) -> dict:
        if hasattr(self, '_tool'):
            return self._tool.get_cmaps() # Call the new method name
        return {}

    @staticmethod
    def get_colormaps(parent=None, icon_path=None):
        """
        Static helper to run the dialog and return result directly.
        Returns:
            dict or None: The colormap data if accepted, else None.
        """
        dialog = ColorMapDialog(parent, icon_path)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_cmaps()
        return None


# --------------------------------------------------------------------------- #
# 1. Application entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Create (or reuse) the QApplication instance.
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    # Optional: provide a path to an icon file if you have one.
    ICON_PATH = None  # e.g. "resources/colormap_icon.png"

    # -----------------------------------------------------------------------
    # 2. Create the dialog
    # -----------------------------------------------------------------------
    # NOTE: The signature is (parent=None, current_cmap_name="Viridis", icon_path=None)
    dialog = ColorMapDialog(
        parent=None,
        current_cmap_name="Inferno",
        icon_path=ICON_PATH,
    ) 

    # -----------------------------------------------------------------------
    # 3. Run the modal dialog
    # -----------------------------------------------------------------------
    if dialog.exec_() == QDialog.Accepted:
        # The instance method returns a dict with 'active', 'names' and 'cmaps'
        cmap_data = dialog.get_cmaps()

        print("--- Selection Confirmed ---")
        print(f"Preset Name: {cmap_data['active']}")
        print("All Preset Names:", ", ".join(cmap_data["names"]))
        print("Colors:")
        for name, colors in cmap_data["cmaps"].items():
            # Print only the first 3 colours as a short preview
            print(f"  {name}: {colors[:3]} ...")
    else:
        print("--- Dialog Canceled ---")

    # -----------------------------------------------------------------------
    # 4. Clean exit (0 = success)
    # -----------------------------------------------------------------------
    sys.exit(0)