# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
import numpy as np
from typing import List, Tuple, Optional
from PySide6.QtWidgets import (QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, 
                               QSizePolicy, QLabel, QComboBox, QGroupBox, QDialogButtonBox,
                               QPushButton, QGridLayout, QStyle)
from PySide6.QtGui import (QPainter, QImage, QPixmap, QPaintEvent, QColor, QIcon,
                           QPen, QBrush, QMouseEvent, QPalette, QGuiApplication, QWheelEvent)
from PySide6.QtCore import Qt, Signal, Slot, QPointF, QRectF

from qt_icons import ICON_DICT
from colorengine import ColorMath, HarmonyEngine
from qt_colorpipette import PixelColorPicker

# --- UI Components ---
class ClickableLabel(QLabel):
    """A QLabel that copies its text to clipboard on double click."""
    def mouseDoubleClickEvent(self, event):
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.text())
        print(f"Copied to clipboard: {self.text()}")

class GradientSlider(QWidget):
    """A slider with a gradient background representing Saturation or Brightness."""
    valueChanged = Signal(int)

    def __init__(self, mode='saturation'):
        super().__init__()
        self.mode = mode
        self.value = 255
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._pixmap = None
        self._hue, self._sat, self._val = 0.0, 1.0, 1.0
        self._is_dragging = False

    def setValue(self, val: int, block_signals: bool = False):
        """Sets the slider value programmatically with bounds checking."""
        val = max(0, min(255, val))
        if self.value == val:
            return
            
        self.value = val
        if not block_signals:
            self.valueChanged.emit(self.value)
        self.update()  # Force repaint

    def set_color_state(self, h, s, v):
        self._hue, self._sat, self._val = h, s, v
        self._generate_texture()
        self.update()

    def _generate_texture(self):
        w = max(256, self.width())
        steps = np.linspace(0, 1, w)
        if self.mode == 'saturation':
            r, g, b = ColorMath.hsv_to_rgb_vectorized(self._hue, steps, self._val)
        else: 
            r, g, b = ColorMath.hsv_to_rgb_vectorized(self._hue, self._sat, steps)
        rgba = np.zeros((1, w, 4), dtype=np.uint8)
        rgba[0, :, 0], rgba[0, :, 1], rgba[0, :, 2], rgba[0, :, 3] = r, g, b, 255
        img = QImage(rgba.data, w, 1, 4 * w, QImage.Format.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(img.copy())

    def _update_value_from_pos(self, x):
        padding = self.height() // 2
        eff_w = self.width() - 2 * padding
        if eff_w <= 0: return
        pct = max(0, min(1, (x - padding) / eff_w))
        val = int(pct * 255)
        self.setValue(val)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._update_value_from_pos(e.position().x())

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._is_dragging: self._update_value_from_pos(e.position().x())

    def mouseReleaseEvent(self, e: QMouseEvent): self._is_dragging = False

    def wheelEvent(self, e: QWheelEvent):
        """Handle mouse wheel events to increment/decrement value."""
        delta = e.angleDelta().y()
        if delta == 0:
            return

        increment = 1 if delta > 0 else -1
        self.setValue(self.value + increment)
        e.accept()
    
    def resizeEvent(self, e):
        self._generate_texture()
        super().resizeEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h, w = self.height(), self.width()
        
        # 1. Taller Gradient Track
        track_h = 20  
        
        if self._pixmap:
            rect = QRectF(0, (h - track_h)/2, w, track_h)
            p.setBrush(QBrush(self._pixmap.scaled(w, int(track_h))))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(rect, track_h/2, track_h/2)

        # 2. Handle Calculation
        padding = h // 2
        x_pos = padding + (self.value / 255.0) * (w - 2 * padding)
        center = QPointF(x_pos, h/2)
        
        # Determine Color Under Handle
        if self.mode == 'saturation':
            r,g,b = ColorMath.hsv_to_rgb_vectorized(self._hue, self.value/255.0, self._val)
        else:
            r,g,b = ColorMath.hsv_to_rgb_vectorized(self._hue, self._sat, self.value/255.0)
            
        handle_color = QColor(int(r), int(g), int(b))
        
        # 3. Dynamic Border Contrast
        border_color = ColorMath.get_contrast_color(r, g, b)

        # 4. Draw Handle
        handle_radius = 7 
        
        p.setBrush(handle_color)
        p.setPen(QPen(border_color, 2)) 
        p.drawEllipse(center, handle_radius, handle_radius)

class ColorCard(QWidget):
    """
    Displays color swatch and values. 
    Clicking sets it as active color.
    Double clicking swatch copies ALL data.
    Double clicking text copies SPECIFIC data.
    """
    clicked = Signal(float)  # Emits hue when clicked

    def __init__(self):
        super().__init__()
        self.setFixedWidth(110)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Swatch
        self.swatch = QWidget()
        self.swatch.setFixedSize(106, 56)
        self.swatch.paintEvent = self._paint_swatch
        self.swatch.mouseDoubleClickEvent = self._copy_all_data
        # Forward mouse press from swatch to card
        self.swatch.mousePressEvent = self.mousePressEvent 
        
        # Data
        self._color = QColor(100, 100, 100)
        self._hue_val = 0.0
        self._is_primary = False
        self.rgb_str = ""
        self.hex_str = ""
        self.hsv_str = ""
        
        # Labels
        self.lbl_hex = ClickableLabel("HEX")
        self.lbl_rgb = ClickableLabel("RGB")
        self.lbl_hsv = ClickableLabel("HSV")
        
        for lbl in [self.lbl_hex, self.lbl_rgb, self.lbl_hsv]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-family: monospace; font-size: 12px; color: #ccc;")
            lbl.setToolTip("Double-click to copy")

        layout.addWidget(self.swatch, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(4)
        layout.addWidget(self.lbl_hex)
        layout.addWidget(self.lbl_rgb)
        layout.addWidget(self.lbl_hsv)

    def set_data(self, c: QColor, h, s, v, is_primary=False):
        self._color = c
        self._hue_val = h
        self._is_primary = is_primary
        
        # Format strings
        self.hex_str = c.name().upper()
        self.rgb_str = f"{c.red()}, {c.green()}, {c.blue()}"
        self.hsv_str = f"{int(h*360)}°, {int(s*100)}%, {int(v*100)}%"
        
        self.lbl_hex.setText(self.hex_str)
        self.lbl_rgb.setText(self.rgb_str)
        self.lbl_hsv.setText(self.hsv_str)
        
        self.swatch.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._hue_val)
            super().mousePressEvent(event)

    def _paint_swatch(self, event):
        p = QPainter(self.swatch)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Inset the base rectangle
        margin = 6
        r = self.swatch.rect().adjusted(margin, margin, -margin, -margin)
        
        # 1. Draw Background Color
        p.setBrush(QBrush(self._color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 8, 8)
        
        if self._is_primary:
            p.setBrush(Qt.BrushStyle.NoBrush)
            
            # 2. Draw Thick White Border (4px)
            p.setPen(QPen(Qt.GlobalColor.white, 2))
            p.drawRoundedRect(r.adjusted(-3,-3, 3, 3), 11, 11)
            
            # 3. Draw Thinner Black Border (2px)
            p.setPen(QPen(Qt.GlobalColor.black, 2))
            p.drawRoundedRect(r.adjusted(-6, -6, 6, 6), 16, 16)

    def _copy_all_data(self, event):
        """Copies full data packet."""
        data = f"HEX: {self.hex_str}\nRGB: {self.rgb_str}\nHSV: {self.hsv_str}"
        QGuiApplication.clipboard().setText(data)
        print("All color data copied!")

class ColorWheelWidget(QWidget):
    hueChanged = Signal(float)

    def __init__(self, size: int = 400, thickness_pct: float = 0.40):
        super().__init__()
        self.wheel_size = size
        self.thickness_pct = thickness_pct
        
        # State
        self._hue = 0.0
        self._saturation = 1.0
        self._brightness = 1.0
        self._harmony_hues = []
        self._is_dragging = False
        
        # Caching & Rendering
        self._pixmap = None
        self._lut_size = 1024  # Precision of the gradient
        self._cache_valid = False
        self._cached_indices = None # Stores hue index for every pixel
        self._cached_alpha = None   # Stores alpha mask for every pixel
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(250, 250)
        
        # Initial render
        self._update_render()

    def set_saturation_brightness(self, sat, bri):
        # Only trigger repaint if values actually change
        if self._saturation != sat or self._brightness != bri:
            self._saturation = sat
            self._brightness = bri
            self._update_render() # Uses cached geometry, only updates color LUT

    def set_harmony_hues(self, hues):
        self._harmony_hues = hues
        self.update()

    def set_hue(self, hue):
        self._hue = hue % 1.0
        self.update()

    def _update_hue_from_mouse(self, pos: QPointF):
        dx = pos.x() - self.width() / 2
        dy = pos.y() - self.height() / 2
        hue = (np.arctan2(dy, dx) + np.pi / 2) / (2 * np.pi)
        self._hue = hue % 1.0
        self.hueChanged.emit(self._hue)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._update_hue_from_mouse(e.position())

    def mouseMoveEvent(self, e):
        if self._is_dragging: self._update_hue_from_mouse(e.position())

    def mouseReleaseEvent(self, e): self._is_dragging = False

    def wheelEvent(self, e: QWheelEvent):
        delta = e.angleDelta().y()
        if delta == 0: return
        hue_shift = (delta * 0.75) / 129600 
        self._hue = (self._hue + hue_shift) % 1.0
        self.hueChanged.emit(self._hue)
        self.update()
        e.accept()

    def resizeEvent(self, e):
        side = min(self.width(), self.height())
        if side != self.wheel_size:
            self.wheel_size = side
            self._cache_valid = False # Invalidate geometry cache
            self._update_render()
        super().resizeEvent(e)

    def _ensure_geometry_cache(self):
        """
        Expensive Step: Calculates angles and distances.
        Only runs when widget is resized.
        """
        if self._cache_valid and self._cached_indices is not None:
            return

        super_sample_factor = 1.25
        size = int(self.wheel_size * super_sample_factor)
        radius = size // 2
        
        # 1. Grid Generation
        y, x = np.ogrid[-radius : size - radius, -radius : size - radius]
        
        # 2. Masking (Geometry)
        hypot = np.hypot(x, y)
        mask_bool = (hypot >= radius * (1 - self.thickness_pct)) & (hypot <= radius * 0.99)
        self._cached_alpha = (mask_bool * 255).astype(np.uint8)
        
        # 3. Hue Calculation -> LUT Indices (Geometry)
        # Convert angle (-pi to pi) to normalized hue (0.0 to 1.0)
        hue_grid = ((np.arctan2(y, x) + np.pi / 2) / (2 * np.pi)) % 1.0
        
        # Map float hue 0.0-1.0 to int indices 0-1023
        self._cached_indices = (hue_grid * (self._lut_size - 1)).astype(np.int16)
        
        self._cache_valid = True

    def _update_render(self):
        """
        Fast Step: Generates color LUT and maps it to cached geometry.
        Runs on S/B change or Resize.
        """
        self._ensure_geometry_cache()
        
        size = self._cached_indices.shape[0]
        
        # 1. Generate Look-Up Table (LUT)
        # Only calculate HSV->RGB for 'lut_size' (1024) pixels, not 'size*size' (250k+)
        lut_hues = np.linspace(0, 1, self._lut_size)
        r, g, b = ColorMath.hsv_to_rgb_vectorized(lut_hues, self._saturation, self._brightness)
        
        # Stack into (1024, 3) array
        lut_rgb = np.column_stack((r, g, b)).astype(np.uint8)
        
        # 2. Apply LUT to Indices (Advanced Numpy Indexing)
        # This replaces the pixels with the looked-up RGB values
        rgb_grid = lut_rgb[self._cached_indices]
        
        # 3. Buffer Packing
        # We need (H, W, 4). We have (H, W, 3) in rgb_grid and (H, W) in alpha
        rgba = np.empty((size, size, 4), dtype=np.uint8)
        rgba[..., :3] = rgb_grid
        rgba[..., 3] = self._cached_alpha
        
        img = QImage(rgba.data, size, size, 4 * size, QImage.Format.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(img.copy())
        
        self.update()

    def paintEvent(self, e):
        if not self._pixmap: return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Draw Wheel
        dest_size = self.wheel_size
        x = (self.width() - dest_size) // 2
        y = (self.height() - dest_size) // 2
        p.drawPixmap(QRectF(x, y, dest_size, dest_size), self._pixmap, QRectF(self._pixmap.rect()))
        
        # Geometry
        rad_out = dest_size // 2
        rad_in = rad_out * (1 - self.thickness_pct)
        rad_mid = (rad_out * 0.98 + rad_in) / 2
        cx, cy = x + rad_out, y + rad_out
        marker_rad = ((rad_out * 0.98) - rad_in) * 0.55 / 3

        # --- Uniform Contrast Calculation ---
        mr, mg, mb = ColorMath.hsv_to_rgb_vectorized(self._hue, self._saturation, self._brightness)
        main_border_col = ColorMath.get_contrast_color(mr, mg, mb)
        
        def draw_marker(h_val, radius, is_primary):
            theta = h_val * (2 * np.pi) - (np.pi / 2)
            mx = cx + rad_mid * np.cos(theta)
            my = cy + rad_mid * np.sin(theta)
            
            r, g, b = ColorMath.hsv_to_rgb_vectorized(h_val, self._saturation, self._brightness)
            p.setBrush(QColor(int(r), int(g), int(b)))
            p.setPen(QPen(main_border_col, 4 if is_primary else 2))
            p.drawEllipse(QPointF(mx, my), radius, radius)

        # Draw Harmonies
        for h in self._harmony_hues:
            draw_marker(h, marker_rad, False)
        
        # Draw Main
        draw_marker(self._hue, marker_rad * 1.667, True)

# --- Main Window ---

class ColorWheelTool(QWidget):
    """
    A comprehensive tool for selecting colors and generating harmonic palettes.
    
    Supports both vertical and horizontal layouts.
    """

    GROUPBOX_STYLE = """
        QGroupBox {
            border: 1px solid #444;
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
    
    def __init__(self, 
                 seed_color: Optional[str] = None, 
                 color_rule: str = "Complementary", 
                 orientation: str = "horizontal",
                 allowed_rules: Optional[List[str]] = None,  # NEW
                 excluded_rules: Optional[List[str]] = None  # NEW
                 ):
        """
        Initialize the Color Picker Tool.

        Args:
            seed_color (str, optional): Hex code or name to initialize the color state.
            color_rule (str): The initial harmony rule to apply.
            orientation (str): Layout mode, either "vertical" or "horizontal".
            allowed_rules (List[str]): If provided, only these rules will appear.
            excluded_rules (List[str]): If provided, these rules will be hidden.
        """
        super().__init__()
        self.hue, self.sat, self.bri = 0.0, 1.0, 1.0
        self.initial_rule = color_rule
        self.orientation = orientation.lower()
        self.allowed_rules = allowed_rules
        self.excluded_rules = excluded_rules
        
        # Parse seed color if provided
        if seed_color:
            c = QColor(seed_color)
            if c.isValid():
                self.hue = c.hueF()
                if self.hue < 0: self.hue = 0.0 
                self.sat = c.saturationF()
                self.bri = c.valueF()

        self.init_ui()

    def init_ui(self):
        """Builds the UI elements based on the specified orientation."""
        
        # 1. Create the Main Layout based on orientation
        if self.orientation == "horizontal":
            main_layout = QHBoxLayout(self)
        else:
            main_layout = QVBoxLayout(self)
            
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # --- Section 1: Color Wheel (Now wrapped in GroupBox) ---
        self.group_wheel = QGroupBox("Color Wheel")
        self.group_wheel.setStyleSheet(self.GROUPBOX_STYLE)
        
        # Use QGridLayout to allow overlay/corner positioning of the button
        wheel_layout = QGridLayout(self.group_wheel)
        wheel_layout.setContentsMargins(12, 12, 12, 12)
        
        self.wheel = ColorWheelWidget(size=400, thickness_pct=0.30)
        # Ensure wheel expands correctly
        self.wheel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.wheel.setMinimumWidth(300)
        self.wheel.setMinimumHeight(300)
        
        # Add Wheel to grid (0,0)
        wheel_layout.addWidget(self.wheel, 0, 0)

        # Pipette Button Construction
        self.btn_pipette = QPushButton("")
        # Use standard icon as fallback if ICON_DICT is missing in context
        # Replace with: QIcon(ICON_DICT['pipette']) if available
        self.btn_pipette.setIcon(QIcon(ICON_DICT['pipette']) )
        self.btn_pipette.setStyleSheet(self.ICON_BUTTON_STYLE)
        self.btn_pipette.setFixedSize(28, 28)
        self.btn_pipette.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pipette.setToolTip("Pick color from screen")
        
        # Add Button to grid (0,0) but aligned Bottom-Right
        wheel_layout.addWidget(self.btn_pipette, 0, 0, 
                               Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        # --- Section 2: Controls ---
        self.group_controls = QGroupBox("Adjustments")
        self.group_controls.setStyleSheet(self.GROUPBOX_STYLE)
        controls_layout = QVBoxLayout(self.group_controls)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(12, 12, 12, 12)

        self.combo = QComboBox()
        
        # --- NEW: Filter Logic for ComboBox ---
        all_rules = list(HarmonyEngine.RELATIONSHIPS.keys())
        
        if self.allowed_rules:
            all_rules = [r for r in all_rules if r in self.allowed_rules]
        
        if self.excluded_rules:
            all_rules = [r for r in all_rules if r not in self.excluded_rules]
            
        self.combo.addItems(all_rules)
        
        # Smart Selection: Try initial rule, fallback to index 0
        if self.initial_rule in all_rules:
            self.combo.setCurrentText(self.initial_rule)
        elif all_rules:
            self.combo.setCurrentIndex(0)
        # -------------------------------------

        self.combo.setStyleSheet(self.COMBO_STYLE)
        
        self.slider_sat = GradientSlider(mode='saturation')
        self.slider_bri = GradientSlider(mode='brightness')

        controls_layout.addWidget(QLabel("Color Rule"))
        controls_layout.addWidget(self.combo)
        controls_layout.addWidget(QLabel("Saturation"))
        controls_layout.addWidget(self.slider_sat)
        controls_layout.addWidget(QLabel("Brightness"))
        controls_layout.addWidget(self.slider_bri)

        # --- Section 3: Harmony Palette ---
        self.group_harmony = QGroupBox("Color Palette")
        self.group_harmony.setStyleSheet(self.GROUPBOX_STYLE)
        self.cards_layout = QHBoxLayout(self.group_harmony)
        self.cards_layout.setSpacing(10)
        self.cards_layout.setContentsMargins(0, 8, 0, 8)

        # --- Assembly Phase ---
        
        if self.orientation == "horizontal":
            # Left Side: The Wheel Group
            main_layout.addWidget(self.group_wheel, 2) # Stretch factor 2

            # Right Side: Container for Controls + Palette
            right_container = QWidget()
            right_layout = QVBoxLayout(right_container)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(10)
            
            right_layout.addWidget(self.group_controls)
            right_layout.addWidget(self.group_harmony)
            # Add spacer to push content up if needed, or let it stretch
            right_layout.addStretch() 
            
            main_layout.addWidget(right_container, 1) # Stretch factor 1
        else:
            # Vertical Mode: Stack everything
            main_layout.addWidget(self.group_wheel, 1)
            main_layout.addWidget(self.group_controls)
            main_layout.addWidget(self.group_harmony)

        # Connections
        self.wheel.hueChanged.connect(self._on_hue)
        self.slider_sat.valueChanged.connect(self._on_sat)
        self.slider_bri.valueChanged.connect(self._on_bri)
        self.combo.currentTextChanged.connect(self._update)
        self.btn_pipette.clicked.connect(self._launch_pipette)

        # Sync UI with initial state (from seed)
        self.wheel.set_hue(self.hue)
        self.wheel.set_saturation_brightness(self.sat, self.bri)
        
        # Update sliders to match seed using setValue (blocks signals to avoid redundant update loop)
        self.slider_sat.setValue(int(self.sat * 255), block_signals=True)
        self.slider_bri.setValue(int(self.bri * 255), block_signals=True)

        # Initial Update
        self._update()
    
    @Slot(float)
    def _on_hue(self, h: float): 
        self.hue = h
        self._update()

    @Slot(int)
    def _on_sat(self, v: int): 
        self.sat = v/255.0
        self.wheel.set_saturation_brightness(self.sat, self.bri)
        self._update()

    @Slot(int)
    def _on_bri(self, v: int): 
        self.bri = v/255.0
        self.wheel.set_saturation_brightness(self.sat, self.bri)
        self._update()

    def _on_card_clicked(self, h: float):
        """Sets the clicked hue as the primary hue and refreshes palette."""
        self.hue = h
        self.wheel.set_hue(h)
        self._update()
    
    @Slot()
    def _launch_pipette(self):
        """Launches the pixel picker and updates the tool with the result."""
        c = PixelColorPicker.pick()
        if c and c.isValid():
            self.hue = c.hueF()
            # Handle grayscale case where hue is -1
            if self.hue < 0: self.hue = 0.0
                
            self.sat = c.saturationF()
            self.bri = c.valueF()
            
            # Update visual state
            self.wheel.set_hue(self.hue)
            self.wheel.set_saturation_brightness(self.sat, self.bri)
            
            # Update sliders (blocking signals to prevent recursion)
            self.slider_sat.setValue(int(self.sat * 255), block_signals=True)
            self.slider_bri.setValue(int(self.bri * 255), block_signals=True)
            
            self._update()

    def _update(self):
        self.slider_sat.set_color_state(self.hue, self.sat, self.bri)
        self.slider_bri.set_color_state(self.hue, self.sat, self.bri)
        
        # Get raw hues
        raw_hues = HarmonyEngine.get_harmonies(self.hue, self.combo.currentText())
        
        # Update Wheel Visuals (pass non-primary hues)
        self.wheel.set_harmony_hues(raw_hues[1:])

        # Rebuild Palette
        while self.cards_layout.count():
            w = self.cards_layout.takeAt(0).widget()
            if w: w.deleteLater()
        
        # Create Cards
        for i, h in enumerate(raw_hues):
            r, g, b = ColorMath.hsv_to_rgb_vectorized(h, self.sat, self.bri)
            card = ColorCard()
            
            # The first item is always the primary (active) one
            is_active = (i == 0)
            card.set_data(QColor(int(r), int(g), int(b)), h, self.sat, self.bri, is_active)
            
            # Connect the click signal to re-order the palette
            card.clicked.connect(self._on_card_clicked)
            
            self.cards_layout.addWidget(card)
    
    def get_colors(self) -> Tuple[List[QColor], str]:
        """Calculates and returns the current list of harmonious colors and the selected rule."""
        rule = self.combo.currentText()
        raw_hues = HarmonyEngine.get_harmonies(self.hue, rule)
        colors = []
        for h in raw_hues:
            r, g, b = ColorMath.hsv_to_rgb_vectorized(h, self.sat, self.bri)
            colors.append(QColor(int(r), int(g), int(b)))
        return colors, rule



class ColorGeneratorDialog(QDialog):

    """A dialog for generating color harmonies based on a seed color."""

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

    WINDOW_WIDTH = 400
    WINDOW_HEIGHT = 400

    def __init__(self, parent=None, seed_color='#FF0000', color_rule="Complementary",
                 icon=None, orientation: str = "horizontal",
                 allowed_rules: Optional[List[str]] = None,  # NEW
                 excluded_rules: Optional[List[str]] = None  # NEW
                 ):
        super().__init__(parent)
        self.setWindowTitle("Generate Color Harmonies")
        self.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.setWindowFlags(self.windowFlags())

        if icon:
            self.setWindowIcon(QIcon(icon))

        # Initialize layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        try:
            # Pass new arguments to the internal tool
            self._tool = ColorWheelTool(
                seed_color=seed_color, 
                color_rule=color_rule, 
                orientation=orientation,
                allowed_rules=allowed_rules,
                excluded_rules=excluded_rules
            )
            layout.addWidget(self._tool)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize color picker tool: {str(e)}") from e

        # Initialize buttons
        button_container_layout = QHBoxLayout()
        button_container_layout.setContentsMargins(10, 10, 10, 10) # Args: left, top, right, bottom
        
        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.apply_button_style()
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        button_container_layout.addWidget(self._buttons)
        layout.addLayout(button_container_layout)

    def apply_button_style(self):
        self._buttons.setStyleSheet(self.BUTTON_STYLE)
    
    def get_colors(self) -> List[QColor]:
        """Returns the list of selected colors from the inner tool."""
        return self._tool.get_colors()


if __name__ == "__main__":
    # 1. Safe Application Instance Check
    app = QApplication.instance() or QApplication(sys.argv)

    # 2. Execution
    # Ensure ColorGeneratorDialog is imported or defined
    dlg = ColorGeneratorDialog(seed_color="#3daee9", color_rule="Triadic")
    
    if dlg.exec():
        # Correctly unpack the Tuple returned by get_colors
        generated_colors, rule_name = dlg.get_colors()
        
        print(f"Accepted Colors (Rule: {rule_name}):")
        
        for c in generated_colors:
            # f-strings for clean, efficient formatting
            print(f" - {c.name()} (RGB: {c.red()}, {c.green()}, {c.blue()})")
    
    sys.exit()