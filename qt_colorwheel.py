import sys
import numpy as np
from typing import List, Tuple
from PySide6.QtWidgets import (QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, 
                               QSizePolicy, QLabel, QComboBox, QGroupBox, QDialogButtonBox)
from PySide6.QtGui import (QPainter, QImage, QPixmap, QPaintEvent, QColor, QIcon,
                           QPen, QBrush, QMouseEvent, QPalette, QGuiApplication, QWheelEvent)
from PySide6.QtCore import Qt, Signal, Slot, QPointF, QRectF

# --- Core Math & Logic ---

class ColorMath:
    """High-performance vectorized color conversion and utility methods."""
    
    @staticmethod
    def hsv_to_rgb_vectorized(h, s, v):
        """Converts HSV to RGB using NumPy vectorization.
        
        Args:
            h: Hue (0.0 - 1.0), scalar or numpy array.
            s: Saturation (0.0 - 1.0), scalar or numpy array.
            v: Value (0.0 - 1.0), scalar or numpy array.
            
        Returns:
            Tuple of (r, g, b) where values are 0-255.
        """
        h = np.asarray(h)
        h6 = h * 6.0
        r_base = np.clip(np.abs(h6 - 3) - 1, 0, 1)
        g_base = np.clip(2 - np.abs(h6 - 2), 0, 1)
        b_base = np.clip(2 - np.abs(h6 - 4), 0, 1)
        s_inv = 1.0 - s
        red   = v * (s_inv + s * r_base) * 255
        green = v * (s_inv + s * g_base) * 255
        blue  = v * (s_inv + s * b_base) * 255
        if h.ndim == 0: return red, green, blue
        return red, green, blue

    @staticmethod
    def get_contrast_color(r, g, b) -> QColor:
        """Calculates luminance to return optimal text contrast color (Black/White)."""
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return Qt.GlobalColor.black if lum > 140 else Qt.GlobalColor.white


class HarmonyEngine:
    """Logic for calculating color harmony relationships."""
    
    RELATIONSHIPS = {
        "Single":               [0],
        "Analogous":            [0, 30, 60],
        "Complementary":        [0, 180],
        "Split-Complementary":  [0, 150, 210],
        "Triadic":              [0, 120, 240],
        "Double-Complementary": [0, 30, 180, 210],
        "Square":               [0, 90, 180, 270],
        "Tetradic":             [0, 60, 180, 240],
        
        # --- NEW 5-Point Methods ---
        
        # 1. Pentagonal: Perfectly balanced, high contrast, vibrant.
        # Spaced every 72 degrees (360 / 5)
        "Pentagonal":           [0, 72, 144, 216, 288],
        
        # 2. 5-Tone Analogous: Very low contrast, unified look.
        # Extends the analogous run further
        "Analogous 5-Tone":     [0, 30, 60, 90, 120],
        
        # 3. Star (Split-Analogous): Complex, rich harmony.
        # Base (0), two split complements (150, 210), and two wide accents (72, 288 approx)
        # Or simpler: Base, +/- 30 (Analogous), +/- 150 (Split Comp)
        "Star 5-Tone":          [0, 30, 330, 150, 210], # 330 is -30 normalized
    }

    @staticmethod
    def get_harmonies(base_hue: float, mode: str) -> List[float]:
        """Generates a list of hue values based on the selected harmony mode.
        
        Args:
            base_hue: A float between 0.0 and 1.0 representing the starting color.
            mode: The dictionary key for the harmony rule.
            
        Returns:
            List[float]: A list of hue values (0.0-1.0).
        """
        offsets = HarmonyEngine.RELATIONSHIPS.get(mode, [0])
        
        # Calculate hues and ensure they wrap around 1.0 using modulo
        return [(base_hue + (deg / 360.0)) % 1.0 for deg in offsets]


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
        self._hue = 0.0
        self._saturation = 1.0
        self._brightness = 1.0
        self._harmony_hues = []
        self._pixmap = None
        self._is_dragging = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(250, 250)
        self._generate_wheel_pixmap()

    def set_saturation_brightness(self, sat, bri):
        self._saturation = sat
        self._brightness = bri
        self._generate_wheel_pixmap()
        self.update()

    def set_harmony_hues(self, hues):
        self._harmony_hues = hues
        self.update()

    def set_hue(self, hue):
        """External setter that updates internal state and redraws."""
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
        """
        Handles mouse wheel rotation for precise hue adjustment.
        """
        delta = e.angleDelta().y()
        if delta == 0: return

        # Sensitivity: 1 full rotation (360 degrees) = 1.0 hue unit
        hue_shift = (delta * 0.75) / 129600 

        self._hue = (self._hue + hue_shift) % 1.0
        self.hueChanged.emit(self._hue)
        self.update()
        e.accept()

    def resizeEvent(self, e):
        side = min(self.width(), self.height())
        if side != self.wheel_size:
            self.wheel_size = side
            self._generate_wheel_pixmap()
        super().resizeEvent(e)

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
        # Calculate the RGB of the MAIN color
        mr, mg, mb = ColorMath.hsv_to_rgb_vectorized(self._hue, self._saturation, self._brightness)
        main_border_col = ColorMath.get_contrast_color(mr, mg, mb)
        
        # Helper
        def draw_marker(h_val, radius, is_primary):
            theta = h_val * (2 * np.pi) - (np.pi / 2)
            mx = cx + rad_mid * np.cos(theta)
            my = cy + rad_mid * np.sin(theta)
            
            # Fill color is specific to the marker's hue
            r, g, b = ColorMath.hsv_to_rgb_vectorized(h_val, self._saturation, self._brightness)
            
            p.setBrush(QColor(int(r), int(g), int(b)))
            # Thicker border for primary
            p.setPen(QPen(main_border_col, 4 if is_primary else 2))
            p.drawEllipse(QPointF(mx, my), radius, radius)

        # Draw Harmonies
        for h in self._harmony_hues:
            draw_marker(h, marker_rad, False)
        
        # Draw Main
        draw_marker(self._hue, marker_rad * 1.667, True)

    def _generate_wheel_pixmap(self):
        super_sample_factor = 1.25
        size = int(self.wheel_size * super_sample_factor)
        radius = size // 2
        
        y, x = np.ogrid[-radius : size - radius, -radius : size - radius]
        
        # 1. Masking
        hypot = np.hypot(x, y)
        mask = (hypot >= radius * (1 - self.thickness_pct)) & (hypot <= radius * 0.99)
        
        # 2. Hue Calculation
        hue = ((np.arctan2(y, x) + np.pi / 2) / (2 * np.pi)) % 1.0
        
        # 3. Color Vectorization
        r, g, b = ColorMath.hsv_to_rgb_vectorized(hue, self._saturation, self._brightness)
        
        # 4. Buffer Packing
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[..., 0] = r.astype(np.uint8)
        rgba[..., 1] = g.astype(np.uint8)
        rgba[..., 2] = b.astype(np.uint8)
        rgba[..., 3] = (mask * 255).astype(np.uint8)
        
        img = QImage(rgba.data, size, size, 4 * size, QImage.Format.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(img.copy())

# --- Main Window ---
class ColorPickerTool(QWidget):
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
    
    def __init__(self, seed_color=None, color_rule="Complementary"):
        super().__init__()
        self.hue, self.sat, self.bri = 0.0, 1.0, 1.0
        self.initial_rule = color_rule
        
        # Parse seed color if provided
        if seed_color:
            c = QColor(seed_color)
            if c.isValid():
                self.hue = c.hueF()
                if self.hue < 0: self.hue = 0.0 # handle grayscale
                self.sat = c.saturationF()
                self.bri = c.valueF()

        self.init_ui()

    def init_ui(self):
        # Main Window Layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Color Wheel
        self.wheel = ColorWheelWidget(size=400, thickness_pct=0.30)
        layout.addWidget(self.wheel, 1)

        # 2. Controls GroupBox
        self.group_controls = QGroupBox("Adjustments")
        self.group_controls.setStyleSheet(self.GROUPBOX_STYLE)
        controls_layout = QVBoxLayout(self.group_controls)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(12, 12, 12, 12)

        self.combo = QComboBox()
        self.combo.addItems(list(HarmonyEngine.RELATIONSHIPS.keys()))
        self.combo.setCurrentText(self.initial_rule)
        
        # Robust check: If text didn't set (e.g. invalid rule name), default to 0
        if self.combo.currentIndex() == -1:
            self.combo.setCurrentIndex(0)
            
        self.combo.setStyleSheet(self.COMBO_STYLE)
        
        self.slider_sat = GradientSlider(mode='saturation')
        self.slider_bri = GradientSlider(mode='brightness')

        controls_layout.addWidget(QLabel("Color Rule"))
        controls_layout.addWidget(self.combo)
        controls_layout.addWidget(QLabel("Saturation"))
        controls_layout.addWidget(self.slider_sat)
        controls_layout.addWidget(QLabel("Brightness"))
        controls_layout.addWidget(self.slider_bri)

        layout.addWidget(self.group_controls)

        # 3. Harmony Palette GroupBox
        self.group_harmony = QGroupBox("Color Palette")
        self.group_harmony.setStyleSheet(self.GROUPBOX_STYLE)
        self.cards_layout = QHBoxLayout(self.group_harmony)
        self.cards_layout.setSpacing(10)
        self.cards_layout.setContentsMargins(0, 8, 0, 8)
        
        layout.addWidget(self.group_harmony)

        # Connections
        self.wheel.hueChanged.connect(self._on_hue)
        self.slider_sat.valueChanged.connect(self._on_sat)
        self.slider_bri.valueChanged.connect(self._on_bri)
        self.combo.currentTextChanged.connect(self._update)

        # Sync UI with initial state (from seed)
        self.wheel.set_hue(self.hue)
        self.wheel.set_saturation_brightness(self.sat, self.bri)
        
        # Update sliders to match seed using setValue (blocks signals to avoid redundant update loop)
        self.slider_sat.setValue(int(self.sat * 255), block_signals=True)
        self.slider_bri.setValue(int(self.bri * 255), block_signals=True)

        # Initial Update
        self._update()
    
    @Slot(float)
    def _on_hue(self, h): 
        self.hue = h
        self._update()

    @Slot(int)
    def _on_sat(self, v): 
        self.sat = v/255.0
        self.wheel.set_saturation_brightness(self.sat, self.bri)
        self._update()

    @Slot(int)
    def _on_bri(self, v): 
        self.bri = v/255.0
        self.wheel.set_saturation_brightness(self.sat, self.bri)
        self._update()

    def _on_card_clicked(self, h):
        """Sets the clicked hue as the primary hue and refreshes palette."""
        self.hue = h
        self.wheel.set_hue(h)
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

    WINDOW_WIDTH = 540
    WINDOW_HEIGHT = 840

    def __init__(self, parent=None, seed_color='#FF0000', color_rule="Complementary", icon=None):
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
            self._tool = ColorPickerTool(seed_color=seed_color, color_rule=color_rule)
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
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(40, 40, 40))
    p.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    app.setPalette(p)
    
    dlg = ColorGeneratorDialog(seed_color="#3daee9", color_rule="Triadic")
    if dlg.exec():
        print("Accepted Colors:")
        for c in dlg.get_colors():
            print(f" - {c.name()} (RGB: {c.red()}, {c.green()}, {c.blue()})")
    
    sys.exit()