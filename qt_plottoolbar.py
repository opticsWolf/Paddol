# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import io
import pyqtgraph as pg
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QImage, QPainter, QColor
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QToolButton, QComboBox, 
    QLabel, QSizePolicy, QFileDialog, QApplication, QMessageBox
)
from typing import List, Dict, Any

try:
    from qt_icons import ICON_DICT
except ImportError:
    ICON_DICT = {}


from qt_colormaps import ColorMapDialog, OklabEngine, ColorMapsPresetManager

class DirectionalZoomViewBox(pg.ViewBox):
    """
    Custom ViewBox logic:
    - Left Drag (Left->Right): Rect Zoom In.
    - Left Drag (Right->Left): Zoom Out (2x).
    - Middle Drag: Pan.
    - Right Drag: Disabled (prevents accidental scaling).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseMode(self.RectMode)
        self.rbScaleBox.setPen(pg.mkPen(color=(128, 128, 128, 128), width=2))
        self.rbScaleBox.setBrush(pg.mkBrush(color=(164, 164, 164, 78)))
        
    def apply_theme(self, theme_dict):
        pen_color = theme_dict.get('zoombox_color', '#A4A4A44E' )
        brush_color = theme_dict.get('zoombox_border', '#80808080')
        
        #print (pen_color, brush_color)
        
        self.rbScaleBox.setPen(pg.mkPen(color=pen_color, width=2))
        self.rbScaleBox.setBrush(pg.mkBrush(color=brush_color))
        
    def mouseDragEvent(self, ev, axis=None):
        # --- Disable Right Mouse Drag (Standard Zoom) ---
        if ev.button() == Qt.RightButton:
            ev.accept()
            return

        # --- Handle Middle Mouse Pan ---
        if ev.button() == Qt.MiddleButton:
            ev.accept()
            if not ev.isStart():
                # Calculate delta in View Coordinates
                tr = self.mapToView(ev.pos()) - self.mapToView(ev.lastPos())
                # Apply translation (negative because moving camera opposite to drag)
                self.translateBy(-tr)
            return

        # --- Handle Left Mouse Rect Zoom / Directional Zoom Out ---
        if self.state['mouseMode'] == self.RectMode and ev.button() == Qt.LeftButton:
            if ev.isFinish():
                start_pos = ev.buttonDownPos()
                end_pos = ev.pos()

                # Right-to-Left: ZOOM OUT
                if end_pos.x() < start_pos.x():
                    self.rbScaleBox.hide()
                    center = self.mapToView(ev.buttonDownPos())
                    self.scaleBy((2.0, 2.0), center=center)
                    ev.accept()
                    return
        
        # Pass standard Left-to-Right Rect Zoom to super
        super().mouseDragEvent(ev, axis)


class CSVPlottoolbar(QWidget):
    """
    Toolbar for controlling a PyQtGraph PlotWidget.
    """
    def __init__(self, plot_widget: pg.PlotWidget, current_cmap_name: str = 'Viridis', parent: QWidget | None = None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        self.parent_widget = parent  # Reference to PolarsPlotWidget
        
        # Enforce Static Interaction Mode
        self.plot_widget.plotItem.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        
        self.cmap_dialog = ColorMapDialog()
        #self.cmap_presets = self.cmap_dialog.all_presets
        cmap_manager = ColorMapsPresetManager()
        self.cmap_list, cmap_presets = cmap_manager.return_presets()
        self.cmap_dict  = cmap_manager.return_cmaps(cmap_presets)
        self._on_cmap_changed(current_cmap_name)
        self._build_ui()
        
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, 
                                  rateLimit=60, slot=self._update_coordinates)
        

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 4)
        layout.setSpacing(8)

        # --- View Controls ---
        self.btn_home = self._make_btn("Reset", "Reset View", ICON_DICT.get("home"))
        self.btn_zoom_in = self._make_btn("In", "Zoom In", ICON_DICT.get("zoom_in"))
        self.btn_zoom_out = self._make_btn("Out", "Zoom Out", ICON_DICT.get("zoom_out"))
        
        self.btn_home.clicked.connect(lambda: self.plot_widget.plotItem.getViewBox().enableAutoRange())
        self.btn_zoom_in.clicked.connect(lambda: self.plot_widget.plotItem.getViewBox().scaleBy((0.8, 0.8)))
        self.btn_zoom_out.clicked.connect(lambda: self.plot_widget.plotItem.getViewBox().scaleBy((1.2, 1.2)))

        layout.addWidget(self.btn_home)
        layout.addWidget(self.btn_zoom_in)
        layout.addWidget(self.btn_zoom_out)

        # --- Legend Toggle (New) ---
        self.btn_legend = self._make_btn("Leg", "Show/Hide Legend", ICON_DICT.get("legend"))
        self.btn_legend.setCheckable(True)
        self.btn_legend.setChecked(True)
        self.btn_legend.clicked.connect(self._toggle_legend)
        layout.addWidget(self.btn_legend)

        # Separator
        line = QWidget()
        line.setFixedWidth(1)
        line.setStyleSheet("background-color: #CCCCCC;")
        layout.addWidget(line)

        # --- Data & Export Controls ---
        self.btn_copy_data = self._make_btn("Copy Data", "Copy Table Data", ICON_DICT.get("copy2"))
        self.btn_copy_data.clicked.connect(self.copy_data_safe)

        self.btn_copy_img = self._make_btn("Copy Graph", "Copy Plot Image", ICON_DICT.get("copy"))
        self.btn_copy_img.clicked.connect(self.copy_plot_image)

        self.btn_save = self._make_btn("Save", "Save Image...", ICON_DICT.get("save"))
        self.btn_save.clicked.connect(self.save_plot)

        layout.addWidget(self.btn_copy_data)
        layout.addWidget(self.btn_copy_img)
        layout.addWidget(self.btn_save)

        # --- Info & Settings ---
        spacer1 = QWidget()
        spacer1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(spacer1)

        self.xy_label = QLabel("")
        self.xy_label.setMinimumWidth(150)
        self.xy_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.xy_label)

        spacer2 = QWidget()
        spacer2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(spacer2)

        layout.addWidget(QLabel("Colormap:"))
        self.cmap_selector = QComboBox()
        self.cmap_selector.addItems(self.cmap_list)
        # Set the selector to the active name (which might be the newly saved preset)
        idx = self.cmap_selector.findText(self.current_cmap_name)
        if idx >= 0:
            self.cmap_selector.setCurrentIndex(idx)
        
        self.cmap_selector.currentTextChanged.connect(self._on_cmap_changed)
        layout.addWidget(self.cmap_selector)
        
        self.cmap_button = self._make_btn("Colormaps", "Colormaps generator", ICON_DICT.get("colormap"))
        self.cmap_button.clicked.connect(self.open_cmap_dialog)
        layout.addWidget(self.cmap_button)

    def _make_btn(self, text, tooltip, icon_path) -> QToolButton:
        btn = QToolButton()
        if icon_path:
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(20, 20))
        else:
            btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        return btn

    def _toggle_legend(self, checked: bool):
        """Passes the toggle command to the parent widget."""
        if hasattr(self.parent_widget, "toggle_legend"):
            self.parent_widget.toggle_legend(checked)

    # ... [Keep previous _update_coordinates, copy_data_safe, copy_plot_image, save_plot methods] ...
    # (Ensure you keep the "Stable" QPainter implementation of copy_plot_image from the previous step)
    def _update_coordinates(self, evt):
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.xy_label.setText(f"X: {mouse_point.x():.2f}  Y: {mouse_point.y():.2f}")

    def _on_cmap_changed(self, text):
        if hasattr(self.parent_widget, "set_new_colormap"):
            self.current_cmap_name = text
            self.parent_widget.set_new_colormap(self.current_cmap_name , self.cmap_dict.get(self.current_cmap_name, 'Viridis'))
            
    def copy_data_safe(self) -> None:
        """Safely copies the parent's dataframe to clipboard as TSV."""
        if not self.parent_widget or self.parent_widget.dataframe is None:
            return

        df = self.parent_widget.dataframe
        if df.is_empty():
            return

        try:
            if df.height > 500_000:
                QMessageBox.warning(self, "Data Too Large", 
                                  "Dataset > 500k rows. Please use Save to File.")
                return

            buffer = io.BytesIO()
            df.write_csv(buffer, separator="\t")
            csv_text = buffer.getvalue().decode('utf-8')

            clipboard = QApplication.clipboard()
            clipboard.setText(csv_text)
            
            print(f"Copied {df.height} rows to clipboard.")

        except Exception as e:
            QMessageBox.critical(self, "Copy Error", f"Failed to copy data:\n{str(e)}")

    def copy_plot_image(self):
        """Render the plot widget to a QImage and copy to clipboard."""
        try:
            widget = self.plot_widget
            target_w = 1920
            rect = widget.rect()
            if rect.width() == 0: return
            target_h = int(target_w * (rect.height() / rect.width()))

            qimage = QImage(target_w, target_h, QImage.Format_ARGB32)
            qimage.fill(Qt.transparent)

            painter = QPainter(qimage)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.scale(target_w / rect.width(), target_h / rect.height())
            widget.render(painter)
            painter.end()

            QApplication.clipboard().setImage(qimage)
            print(f"Graph copied ({target_w}x{target_h}).")
        except Exception as e:
            QMessageBox.critical(self, "Image Copy Error", f"Failed to copy image:\n{str(e)}")

    def save_plot(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "Images (*.png *.jpg *.svg)")
        if not fname: return
        try:
            if fname.endswith('.svg'):
                import pyqtgraph.exporters
                exporter = pg.exporters.SVGExporter(self.plot_widget.plotItem)
                exporter.export(fname)
            else:
                target_w = 1920
                rect = self.plot_widget.rect()
                target_h = int(target_w * (rect.height() / rect.width()))
                qimage = QImage(target_w, target_h, QImage.Format_ARGB32)
                qimage.fill(Qt.transparent)
                painter = QPainter(qimage)
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.scale(target_w / rect.width(), target_h / rect.height())
                self.plot_widget.render(painter)
                painter.end()
                qimage.save(fname)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")

    def open_cmap_dialog(self):
        """
        Opens the Colormap Dialog, processes the result (which contains all presets 
        and the active name), and updates the plot and the selector control.
        """
        # CRITICAL: We call ColorMapDialog.get_colormap which returns the dictionary 
        # from get_cmap_dict (which includes the 'active' name and all 'presets').
        result_dict = ColorMapDialog.get_colormaps(parent=self, icon_path=ICON_DICT.get("colormap"))

        if result_dict:
            # 1. Extract the returned structure
            active_preset: str = result_dict.get('active', 'Viridis')
            cmap_names: List[str] = result_dict.get('names', [])
            cmap_dict: Dict[str, Dict[str, Any]] = result_dict.get('cmaps', {})
             
            if cmap_names:
                self.cmap_list = cmap_names
            
            if cmap_dict:
                self.cmap_dict = cmap_dict
            
            # --- Update the Toolbar UI and Cache ---
            
            # The result_dict already contains the fully refreshed list and presets, 
            # so we update the toolbar's internal state directly.
            self.cmap_list = cmap_names

            # Update the QComboBox list
            self.cmap_selector.blockSignals(True)
            self.cmap_selector.clear()
            self.cmap_selector.addItems(self.cmap_list)
            self.cmap_selector.blockSignals(False)
            # Set the selector to the active name (which might be the newly saved preset)
            idx = self.cmap_selector.findText(active_preset)
            if idx >= 0:
                self.cmap_selector.setCurrentIndex(idx)
        
            print(f"Applied colormap: '{active_preset}.")