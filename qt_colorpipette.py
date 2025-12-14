# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import sys
from typing import Optional, Callable
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import (
    Qt, QPoint, QRect, QTimer, Signal, QEventLoop, QObject, QSize
)
from PySide6.QtGui import (
    QPainter, QPen, QColor, QPixmap, QCursor, QPainterPath,
    QImage, QFont, QPaintEvent, QKeyEvent, QMouseEvent
)


class ZoomPipette(QWidget):
    """Visual overlay widget that follows the mouse and provides a zoomed view of underlying pixels.

    This widget takes a screenshot and dynamically updates its magnified view
    as the user moves the cursor.

    Attributes:
        diameter (int): The total pixel width/height of the widget.
        zoom_outer (float): Zoom level for the outer ring (e.g., 4.0x).
        zoom_inner (float): Zoom level for the center reticle (e.g., 8.0x).
    """

    # Signals to communicate with the wrapper
    color_selected = Signal(QColor)
    cancelled = Signal()

    def __init__(self, diameter: int = 180):
        """Initializes the ZoomPipette widget."""
        super().__init__()
        self.diameter = diameter

        # Zoom Configuration
        self.zoom_outer: float = 4.0
        self.zoom_inner: float = 8.0
        self.inner_diameter: int = 56

        # State
        self._screen_pixmap: Optional[QPixmap] = None
        self._screen_geo: QRect = QRect()
        self._device_pixel_ratio: float = 1.0
        self.current_color: QColor = QColor(0, 0, 0)
        self._last_global_pos: QPoint = QPoint(0, 0)

        # UI Timer - 60 FPS update loop
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)

        # Window Setup
        # Frameless + Tool + OnTop ensures it floats above everything
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        
        # Ensure widget receives mouse events despite transparency
        self.setMouseTracking(True)
        self.setFixedSize(self.diameter, self.diameter)

    def start_session(self) -> None:
        """Prepares the screen snapshot, shows the widget, and starts the update loop."""
        self.refresh_snapshot()
        self.update_position(QCursor.pos())
        self.show()
        self.timer.start()

    def refresh_snapshot(self) -> None:
        """Takes a screenshot of the current screen.

        The widget is momentarily hidden before capturing to avoid self-capture.
        Also updates the screen geometry and device pixel ratio.
        """
        pos = QCursor.pos()
        screen = QApplication.screenAt(pos)
        if not screen:
            screen = QApplication.primaryScreen()
        
        if not screen:
            return  # Safety fallback

        # Temporarily hide to grab clean screenshot
        was_visible = self.isVisible()
        if was_visible:
            self.hide()

        # Grab window 0 (desktop)
        self._screen_pixmap = screen.grabWindow(0)
        self._screen_geo = screen.geometry()
        self._device_pixel_ratio = screen.devicePixelRatio()

        if was_visible:
            self.show()

    def update_position(self, global_mouse_pos: QPoint) -> None:
        """Moves the widget to center on the mouse and updates the internal color sample.

        Args:
            global_mouse_pos (QPoint): The current global mouse coordinates.
        """
        self._last_global_pos = global_mouse_pos

        # Multi-Monitor Check: If we moved to a new screen, refresh the screenshot
        if not self._screen_geo.contains(global_mouse_pos):
            self.refresh_snapshot()

        # Center on mouse
        center_offset = QPoint(self.width() // 2, self.height() // 2)
        top_left = global_mouse_pos - center_offset
        self.move(top_left)

        # Sample the pixel color at the center
        self._sync_sample_color()
        self.update()

    def _calculate_source_rect(self, zoom_factor: float, diameter_px: int) -> QRect:
        """Calculates the rectangle on the screenshot corresponding to the zoomed view.

        Args:
            zoom_factor (float): The zoom level (e.g., 4.0 or 8.0).
            diameter_px (int): The pixel diameter of the target draw area (e.g., 180 or 56).

        Returns:
            QRect: The source rectangle on the QPixmap, scaled by devicePixelRatio.
        """
        view_w = diameter_px / zoom_factor
        view_h = diameter_px / zoom_factor
        
        rel_x = self._last_global_pos.x() - self._screen_geo.x()
        rel_y = self._last_global_pos.y() - self._screen_geo.y()
        
        src_x = int((rel_x - (view_w / 2)) * self._device_pixel_ratio)
        src_y = int((rel_y - (view_h / 2)) * self._device_pixel_ratio)
        src_w = int(view_w * self._device_pixel_ratio)
        src_h = int(view_h * self._device_pixel_ratio)
        
        return QRect(src_x, src_y, src_w, src_h)

    def _sync_sample_color(self) -> None:
        """Extracts the pixel color at the exact center of the current lens and updates current_color."""
        if not self._screen_pixmap:
            return

        source_rect = self._calculate_source_rect(self.zoom_inner, self.inner_diameter)
        center_x = source_rect.x() + (source_rect.width() // 2)
        center_y = source_rect.y() + (source_rect.height() // 2)
        
        img: QImage = self._screen_pixmap.toImage()
        if img.valid(center_x, center_y):
            self.current_color = img.pixelColor(center_x, center_y)

    def _tick(self) -> None:
        """Timer callback to synchronize the widget position with the cursor."""
        self.update_position(QCursor.pos())

    # --- Event Handlers ---

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handles mouse press events.

        Left click emits ``color_selected``, right click emits ``cancelled``.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.color_selected.emit(self.current_color)
        elif event.button() == Qt.MouseButton.RightButton:
            self.cancelled.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handles key press events.

        Escape key emits the ``cancelled`` signal.
        """
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paints the zoom pipette overlay.

        Renders the outer zoomed area, the inner high-zoom reticle, the
        UI overlays (rings, crosshair), and the color information box.

        Args:
            event (QPaintEvent): The paint event generated by Qt.
        """
        if not self._screen_pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        
        lens_rect = self.rect()

        def draw_zoom_layer(target_rect: QRect, zoom_factor: float, diameter: int) -> None:
            source_rect = self._calculate_source_rect(zoom_factor, diameter)
            painter.drawPixmap(target_rect, self._screen_pixmap, source_rect)

        # 1. Outer Lens (Background)
        outer_path = QPainterPath()
        outer_path.addEllipse(lens_rect)
        painter.setClipPath(outer_path)
        
        # Fill black first to ensure opacity behind the image
        painter.fillRect(lens_rect, Qt.GlobalColor.black)
        draw_zoom_layer(lens_rect, self.zoom_outer, self.width())

        # 2. Inner Lens (High Zoom)
        d_inner = self.inner_diameter
        inner_rect = QRect(
            (self.width() - d_inner) // 2,
            (self.height() - d_inner) // 2,
            d_inner, d_inner
        )
        inner_path = QPainterPath()
        inner_path.addEllipse(inner_rect)
        painter.setClipPath(inner_path, Qt.ClipOperation.ReplaceClip)
        
        draw_zoom_layer(inner_rect, self.zoom_inner, d_inner)

        # 3. UI Overlays (Rings)
        painter.setClipping(False)
        
        # Thick dark border
        painter.setPen(QPen(QColor(30, 30, 30), 6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(3, 3, self.width() - 6, self.height() - 6)

        # Thin light border
        painter.setPen(QPen(QColor(220, 220, 220), 2))
        painter.drawEllipse(3, 3, self.width() - 6, self.height() - 6)

        # Inner highlight ring
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        painter.drawEllipse(inner_rect)

        # Crosshair (Dynamic Contrast)
        lum = 0.2126 * self.current_color.red() + 0.7152 * self.current_color.green() + 0.0722 * self.current_color.blue()
        cross_color = QColor(Qt.GlobalColor.black) if lum > 140 else QColor(Qt.GlobalColor.white)
        painter.setPen(QPen(cross_color, 1))

        cx, cy = self.width() // 2, self.height() // 2
        gap = 4
        length = 6
        painter.drawLine(cx, cy - gap - length, cx, cy - gap)
        painter.drawLine(cx, cy + gap, cx, cy + gap + length)
        painter.drawLine(cx - gap - length, cy, cx - gap, cy)
        painter.drawLine(cx + gap, cy, cx + gap + length, cy)

        # 4. Text Info Box
        font = QFont("Monospace")
        font.setBold(True)
        font.setPixelSize(11)
        painter.setFont(font)

        hex_text = self.current_color.name().upper()
        rgb_text = f"{self.current_color.red()}, {self.current_color.green()}, {self.current_color.blue()}"
        
        box_w, box_h = 110, 34
        box_x = (self.width() - box_w) // 2
        box_y = cy + 32

        # Semi-transparent Background Box
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(30, 30, 30, 180))
        painter.drawRoundedRect(QRect(box_x, box_y, box_w, box_h), 6, 6)

        # Text Labels
        painter.setPen(QColor(230, 230, 230))
        painter.drawText(QRect(box_x, box_y + 2, box_w, box_h // 2), Qt.AlignmentFlag.AlignCenter, hex_text)
        painter.drawText(QRect(box_x, box_y + (box_h // 2) - 2, box_w, box_h // 2), Qt.AlignmentFlag.AlignCenter, rgb_text)


class PixelColorPicker(QObject):
    """Static wrapper for launching the ZoomPipette in a modal event loop.

    This class provides a synchronous interface to launch the screen color picker
    and retrieve the selected color.
    """

    @staticmethod
    def pick() -> Optional[QColor]:
        """Launches the modal screen color picker and blocks until selection or cancellation.

        Returns:
            QColor or None: The selected QColor object if picked, or None if cancelled
                (via ESC or right-click).
        """
        lens = ZoomPipette(diameter=180)
        
        # --- CRITICAL FIX: MODALITY ---
        # If the parent application has a Modal Dialog open (like ColorGeneratorDialog),
        # this new window will be BLOCKED unless it is also ApplicationModal.
        # This overrides the previous modal lock.
        lens.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        loop = QEventLoop()
        result: Optional[QColor] = None

        # Callbacks
        def on_selected(color: QColor):
            nonlocal result
            result = color
            loop.quit()

        def on_cancel():
            loop.quit()

        lens.color_selected.connect(on_selected)
        lens.cancelled.connect(on_cancel)

        # Start Session
        lens.start_session()
        
        # Ensure the window is physically active to receive keyboard events (Esc)
        lens.raise_()
        lens.activateWindow()
        
        # Grab Inputs
        # NOTE: grabMouse is safe here because the window follows the cursor.
        # It ensures clicks are registered even if the user flicks the mouse fast.
        lens.grabMouse()
        lens.grabKeyboard()
        lens.setFocus()

        QApplication.setOverrideCursor(Qt.CursorShape.BlankCursor)
        
        # Block Main Thread
        loop.exec()

        # Cleanup
        lens.releaseMouse()
        lens.releaseKeyboard()
        lens.hide()
        lens.deleteLater()
        QApplication.restoreOverrideCursor()

        return result


if __name__ == "__main__":
    # --- Example Usage (Pseudo "Other Tool") ---
    from PySide6.QtWidgets import QVBoxLayout, QLabel, QPushButton

    class DemoTool(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("My External Tool")
            self.resize(300, 150)
            
            layout = QVBoxLayout(self)
            
            self.lbl = QLabel("No Color Selected")
            self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
            layout.addWidget(self.lbl)
            
            btn = QPushButton("Launch Pipette")
            btn.clicked.connect(self.launch_picker)
            layout.addWidget(btn)

        def launch_picker(self):
            # This is how you call it from your tool:
            color = PixelColorPicker.pick()
            
            if color:
                self.lbl.setText(f"Selected: {color.name().upper()}")
                self.lbl.setStyleSheet(f"background-color: {color.name()}; font-size: 14px; font-weight: bold; color: {'white' if color.lightness() < 128 else 'black'}")
            else:
                self.lbl.setText("Selection Cancelled")
                self.lbl.setStyleSheet("background-color: none; font-size: 14px; font-weight: bold;")
    
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    demo = DemoTool()
    demo.show()
    
    app.exec()