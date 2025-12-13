# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
from typing import List, Dict, Any, Optional

import math

import polars as pl
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import QItemSelectionModel, Qt, QItemSelection, Slot, QPointF, QTimer
from PySide6.QtGui import (QStandardItemModel, QStandardItem, QIcon, 
                         QPixmap, QColor, QPainter, QPen)
from PySide6.QtWidgets import (QWidget, QListView, QVBoxLayout,
                             QApplication, QAbstractItemView, 
                             QSplitter, QGroupBox)

from qt_plottoolbar import CSVPlottoolbar, DirectionalZoomViewBox
from plotstyler import PlotStyler

# Global configuration
pg.setConfigOptions(antialias=True)
#pg.setConfigOption('background', 'w')
#pg.setConfigOption('foreground', 'k')

# ------------------------------------------------------------------------------
# 2. Main Widget
# ------------------------------------------------------------------------------


class PolarsPlotWidget(QWidget):
    """
    A widget that displays a single Polars DataFrame using PyQtGraph.
    """

    def __init__(self, parent: Any = None, current_map_name: str = "Viridis", plot_style: str = "dark") -> None:
        super().__init__(parent)

        self.dataframe: Optional[pl.DataFrame] = None
        self.x_array: np.ndarray = np.array([])
        self.column_colors = {}
        self.plot_items: Dict[str, pg.PlotDataItem] = {}
       
        self.current_cmap_name = current_map_name
        self.active_cmap = None
        
        self.current_style = plot_style
        self.plot_styler = PlotStyler(self.current_style)

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setOpaqueResize(True)

        # --- Left Pane: Columns ---
        col_group = QGroupBox("Data Columns")
        col_layout = QVBoxLayout(col_group)
        col_layout.setContentsMargins(4, 8, 4, 4)

        self.col_list_view = QListView()
        self.col_list_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.col_model = QStandardItemModel()
        self.col_list_view.setModel(self.col_model)
        self.col_list_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        
        col_layout.addWidget(self.col_list_view)
        
        left_wrapper = QWidget()
        left_layout = QVBoxLayout(left_wrapper)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(col_group)
        left_wrapper.setMinimumWidth(150)
        
        main_splitter.addWidget(left_wrapper)

        # --- Right Pane: Plot & Toolbar ---
        plot_group = QGroupBox("Plot View")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(4, 8, 4, 0)

        # ----------------------------------------------------------------------
        # INJECT CUSTOM VIEWBOX HERE
        # ----------------------------------------------------------------------
        self.custom_vb = DirectionalZoomViewBox()
        self.custom_vb.apply_theme(self.plot_styler.current_style)
        self.plot_widget = pg.PlotWidget(viewBox=self.custom_vb)
        
        self.plot_widget.addLegend(offset=(30, 30))
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setClipToView(True)
        self.plot_item.setDownsampling(mode='peak')

        # Toolbar (Handles Pan via Middle Click automatically due to ViewBox defaults)
        self.toolbar = CSVPlottoolbar(self.plot_widget, self.current_cmap_name, self)

        plot_layout.addWidget(self.plot_widget, 1)
        plot_layout.addWidget(self.toolbar, 0)

        right_wrapper = QWidget()
        right_layout = QVBoxLayout(right_wrapper)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(plot_group)
        
        main_splitter.addWidget(right_wrapper)
        main_splitter.setStretchFactor(1, 1)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(main_splitter)

        self._apply_modern_style()

    # ------------------------------------------------------------------
    # Data Handling
    # ------------------------------------------------------------------
    def set_dataframe(self, df: pl.DataFrame) -> None:
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame")

        self.dataframe = df
        self.plot_items.clear()
        self.plot_widget.clear()
        
        if df.is_empty():
            self.x_array = np.array([])
            self.col_model.clear()
            return

        self.x_array = df[df.columns[0]].to_numpy()
        self.plot_item.setLabel('bottom', df.columns[0])
        self.plot_item.setLabel('left', "Photometric Value")

        self._update_column_list()
        self.plot_item.autoRange()

    def _update_column_list(self) -> None:
        if self.dataframe is None or self.dataframe.is_empty():
            return

        cols = sorted(self.dataframe.columns[1:])
        self.col_model.clear()

        n_columns = len(cols)
        cmap = self.active_cmap
        
        self.column_colors = {}
        # 1. Vectorized Index Calculation
        # Map columns to 0-255 indices in one step
        if n_columns > 1:
            indices = np.linspace(0, 255, n_columns).astype(np.uint8)
        else:
            indices = np.array([128], dtype=np.uint8)
        
        # 2. Batch Fetch RGBA Values
        # shape: (n_columns, 4). Much faster than looking up one-by-one.
        rgba_batch = cmap[indices]
        
        # 3. Block signals to prevent UI flickering/lag during mass insertion
        self.col_model.blockSignals(True)
        
        try:
            # Zip allows us to iterate the name and the pre-fetched color together
            for col_name, rgba in zip(cols, rgba_batch):
                # --- Color Creation ---
                # Unpack the numpy row directly into fromRgbF
                qcolor = QColor.fromRgbF(*rgba)
                self.column_colors[col_name] = qcolor
        
                # --- Item Creation ---
                item = QStandardItem(col_name)
                
                # --- Icon Drawing ---
                # 18x12 canvas
                pix = QPixmap(18, 12)
                pix.fill(Qt.GlobalColor.transparent)
                
                p = QPainter(pix)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                
                # Create pen with specific color
                pen = QPen(qcolor, 3)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                
                # Draw line (x1, y1, x2, y2)
                p.drawLine(2, 6, 16, 6)
                p.end() # Critical: Must end painter to apply changes to pixmap
                
                item.setIcon(QIcon(pix))
                self.col_model.appendRow(item)

        finally:
            # 4. Re-enable signals to trigger a single UI refresh
            self.col_model.blockSignals(False)
            # Force a layout update just in case the view needs it
            if hasattr(self.col_model, 'layoutChanged'):
                self.col_model.layoutChanged.emit()

        if cols:
            sel_model = self.col_list_view.selectionModel()
            first = self.col_model.index(0, 0)
            last = self.col_model.index(len(cols) - 1, 0)
            sel_model.select(QItemSelection(first, last), 
                           QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    @Slot()
    def _on_selection_changed(self, *_) -> None:
        if self.dataframe is None: return

        selected_indexes = self.col_list_view.selectionModel().selectedIndexes()
        selected_cols = {self.col_model.data(idx) for idx in selected_indexes}
        
        active_plots = set(self.plot_items.keys())
        
        # Remove
        for col in active_plots - selected_cols:
            self.plot_widget.removeItem(self.plot_items[col])
            del self.plot_items[col]

        # Add
        cols_to_add = sorted(list(selected_cols - active_plots))
        
        for col in cols_to_add:
            if col not in self.dataframe.columns: continue
            
            y_data = self.dataframe[col].to_numpy()
            color = self.column_colors.get(col, QColor("white"))
            pen = pg.mkPen(color=color, width=1.5)
            
            item = self.plot_widget.plot(self.x_array, y_data, pen=pen, name=col, autoDownsample=True)
            item.setCurveClickable(True)
            self.plot_items[col] = item
        
        # Apply Item Styling and Reflow Legend
        # The new items now exist, so we can reliably update their colors.
        self._update_legend_item_styles() 
        self._reflow_legend()

    # ------------------------------------------------------------------
    # Legend Management
    # ------------------------------------------------------------------
    def toggle_legend(self, visible: bool):
        """Show or hide the legend."""
        if self.plot_item.legend:
            self.plot_item.legend.setVisible(visible)

    # ------------------------------------------------------------------
    # Dynamic Resizing Logic
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        """
        Triggered whenever the widget is resized.
        """
        super().resizeEvent(event)
        
        # Critical: We use a 0ms single-shot timer to defer the reflow 
        # until the Qt Event Loop has finished processing the layout changes.
        # This ensures self.plot_item.vb.height() returns the NEW height, 
        # not the OLD height.
        QTimer.singleShot(0, self._reflow_legend)

    def _reflow_legend(self):
        """
        Dynamically arranges legend items into columns.
        Fixes the background box rendering by forcing layout activation.
        """
        legend = self.plot_item.legend
        if not legend or not legend.items:
            return

        items = legend.items
        n_items = len(items)
        
        layout = legend.layout

        # ------------------------------------------------------------
        # 1. Compact Style with Safety Margins
        # ------------------------------------------------------------
        # We need a small margin (e.g. 5px) so the background box 
        # doesn't look like it's cutting off the text.
        layout.setContentsMargins(5, 5, 5, 5)
        
        layout.setVerticalSpacing(2)
        layout.setHorizontalSpacing(10)

        # ------------------------------------------------------------
        # 2. Viewport Constraints
        # ------------------------------------------------------------
        curr_h = self.plot_widget.height()
        # Safety default if initializing
        if curr_h < 50:
            curr_h = 400.0
            
        # Available height minus margins
        available_height = max(100.0, curr_h - 60.0)

        # ------------------------------------------------------------
        # 3. Calculate Grid Dimensions
        # ------------------------------------------------------------
        row_height = 30.0 
        
        max_rows_per_col = int(available_height // row_height)
        max_rows_per_col = max(3, max_rows_per_col) # Minimum 3 rows

        n_cols = math.ceil(n_items / max_rows_per_col)
        rows_per_col = math.ceil(n_items / n_cols)

        # ------------------------------------------------------------
        # 4. Rebuild Grid Layout
        # ------------------------------------------------------------
        for i in reversed(range(layout.count())):
            layout.removeAt(i)

        for idx, (sample, label) in enumerate(items):
            col = idx // rows_per_col
            row = idx % rows_per_col
            
            layout.addItem(sample, row, col * 2)
            layout.addItem(label, row, col * 2 + 1)

        # ------------------------------------------------------------
        # 5. Fix Background Box Rendering (Critical Step)
        # ------------------------------------------------------------
        # 1. Force the layout to calculate new positions NOW
        layout.activate()
        
        # 2. Calculate the exact size needed by the new grid
        #    We use Qt.PreferredSize to respect the text width
        new_size = layout.effectiveSizeHint(Qt.SizeHint.PreferredSize)
        
        # 3. Force the LegendItem (the box) to adopt this size
        legend.resize(new_size)
        
        # 4. Trigger a full repaint of the scene to redraw the box border
        legend.update()
        
    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def set_new_colormap(self, cmap_name: str, cmap) -> None:
        self.current_cmap_name = cmap_name
        self.active_cmap = cmap
        current_selection = [i.row() for i in self.col_list_view.selectionModel().selectedIndexes()]
        self._update_column_list()
        
        sel_model = self.col_list_view.selectionModel()
        sel_model.clearSelection()
        for row in current_selection:
            idx = self.col_model.index(row, 0)
            sel_model.select(idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def _apply_modern_style(self) -> None:
        """
        Applies the current style configuration to the plot and its legend container.
        """
        style_conf = self.plot_styler.current_style
        
        # 1. Main Plot Background
        self.plot_widget.setBackground(style_conf['background'])
        
        # 2. Grid & Axes
        self.plot_item.showGrid(x=True, y=True, alpha=style_conf['grid_alpha'])
        
        axis_pen = pg.mkPen(color=style_conf['axis_color'], width=1)
        text_pen = pg.mkPen(color=style_conf['text_color'])
        
        for axis in ['left', 'bottom']:
            ax = self.plot_item.getAxis(axis)
            ax.setPen(axis_pen)
            ax.setTextPen(text_pen)

        # 3. Legend Container Styling (Background and Border)
        if self.plot_item.legend:
            legend = self.plot_item.legend
            
            # A. Background (Brush)
            bg_color = QColor(style_conf['legend_bg'])
            bg_color.setAlpha(style_conf.get('legend_alpha', 230))
            legend.setBrush(pg.mkBrush(bg_color))
            
            # B. Border (Pen)
            border_color = style_conf.get('legend_border', style_conf['axis_color'])
            legend.setPen(pg.mkPen(color=border_color, width=1))
        
        # NOTE: Text color is now handled by _update_legend_item_styles when items are added.
        # If this method is called after items exist (e.g., on a theme switch),
        # we still need to apply the text color.
        self._update_legend_item_styles()

    def _update_legend_item_styles(self) -> None:
        """
        Applies theme-specific styling (like text color) to individual legend items.
        Must be called AFTER items have been added to the legend.
        """
        if not self.plot_item.legend:
            return

        legend = self.plot_item.legend
        style_conf = self.plot_styler.current_style
        
        text_color_hex = style_conf.get('legend_text', style_conf['text_color'])
        q_text_color = QColor(text_color_hex)
        
        for sample, label in legend.items:
            # 1. Clean the text to remove old HTML artifacts (nested spans)
            clean_text = ""
            
            if isinstance(label, pg.LabelItem):
                if hasattr(label, 'item') and hasattr(label.item, 'toPlainText'):
                    clean_text = label.item.toPlainText()
                else:
                    clean_text = label.text
                
                # Apply using pg's native color argument
                label.setText(clean_text, color=text_color_hex)
                
                # Double-ensure via the native Qt item (for persistence)
                if hasattr(label, 'item'):
                    label.item.setDefaultTextColor(q_text_color)
                    
            elif hasattr(label, 'setDefaultTextColor'):
                # Fallback for raw QGraphicsTextItems
                label.setDefaultTextColor(q_text_color)
        
        # Force a redraw
        legend.update()

# ----------------------------------------------------------------------
# Usage Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)

    # 1. Generate Dummy Data
    wavelengths = np.linspace(400, 800, 2000)
    data = {}
    for i, col_name in zip(range(1, 21), [chr(c) for c in range(66, 86)]):
        noise = np.random.normal(0, 0.5, len(wavelengths))
        signal = np.sin(wavelengths * 0.01 * i) * 10 + i * 5
        data[col_name] = signal + noise
    
    df_large = pl.DataFrame({"Wavelength / nm": wavelengths, **data})

    # 2. Run Widget
    widget = PolarsPlotWidget()
    widget.set_dataframe(df_large)
    widget.resize(1000, 600)
    widget.show()

    sys.exit(app.exec())