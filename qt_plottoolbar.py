# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 22:36:03 2025

@author: Frank
"""

"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional

import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QToolButton,
    QComboBox,
    QLabel,
    QSizePolicy,
    QFileDialog,
    QApplication,
    QMenu
)

COLORMAPS = (
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
    "Managua", "Berlin", "Vanimo", "Turbo", "Terrain", "Rainbow"
)
from qt_icons import ICON_DICT, scan_icons_folder

class CSVPlottoolbar(QWidget):
    """
    Lightweight custom Matplotlib toolbar without using NavigationToolbar2QT.

    Features:
        • Home / Reset
        • Zoom In / Zoom Out
        • Pan (middle mouse)
        • Scroll-wheel zoom
        • Copy to clipboard
        • Save figure to PNG
        • Colormap selector
        • XY coordinate readout
    """

    def __init__(self, canvas: FigureCanvasQTAgg, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.canvas = canvas
        self.figure = canvas.figure
        self.active_axes = self.canvas.figure.get_axes()[0]
        self._hovering = False

        self.selector = RectangleSelector(
            self.active_axes,
            self.on_select,
            #drawtype='box',
            useblit=True,
            button=[1],        # left mouse
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=False
        )

        # Current pan state
        self._pan_start = None

        self._build_ui()
        self._connect_canvas_events()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 4)
        layout.setSpacing(8)

        # Create buttons
        self.btn_home = self._make_btn("Reset view", "Reset view", QIcon(ICON_DICT["Home"]))
        self.btn_zoom_in = self._make_btn("Zoom in", "Zoom in", QIcon(ICON_DICT["zoom_in"]))
        self.btn_zoom_out = self._make_btn("Zoom out", "Zoom out", QIcon(ICON_DICT["zoom_out"]))
        self.btn_copy = self._make_btn("Copy plot", "Copy plot to clipboard",  QIcon(ICON_DICT["Copy"]))
        self.btn_save = self._make_btn("Save plot", "Save plot", QIcon(ICON_DICT["Save"]))

        self.btn_home.clicked.connect(self.reset_view)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_copy.clicked.connect(self.copy_plot)
        self.btn_save.clicked.connect(self.save_plot)

        # Add buttons
        layout.addWidget(self.btn_home)
        layout.addWidget(self.btn_zoom_in)
        layout.addWidget(self.btn_zoom_out)
        layout.addWidget(self.btn_copy)
        layout.addWidget(self.btn_save)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(spacer)

        # XY label
        self.xy_label = QLabel(" ")
        self.xy_label.setMinimumWidth(150)
        layout.addWidget(self.xy_label)

        # Colormap selector
        layout.addWidget(QLabel("Colormap:"))
        self.cmap_selector = QComboBox()
        for name in COLORMAPS:
            self.cmap_selector.addItem(name)
        layout.addWidget(self.cmap_selector)

    def _make_btn(self, text: str, tooltip: str, icon) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.setIcon(icon)
        btn.setIconSize(QSize(24, 24))
        return btn

    # ------------------------------------------------------------------
    # Canvas events
    # ------------------------------------------------------------------
    def _connect_canvas_events(self) -> None:
        c = self.canvas
        c.mpl_connect("button_press_event", self._on_mouse_press)
        c.mpl_connect("button_release_event", self._on_mouse_release)
        c.mpl_connect("motion_notify_event", self._on_mouse_move)
        c.mpl_connect("scroll_event", self._on_scroll)
        c.mpl_connect("motion_notify_event", self._update_coordinates)

        # Track enter / leave
        c.enterEvent = self._on_enter
        c.leaveEvent = self._on_leave

        # Context menu
        c.setContextMenuPolicy(Qt.CustomContextMenu)
        c.customContextMenuRequested.connect(self._show_context_menu)

    # ------------------------------------------------------------------
    # Interaction logic
    # ------------------------------------------------------------------
    def _on_enter(self, event):
        self._hovering = True

    def _on_leave(self, event):
        self._hovering = False
        self.xy_label.setText(" ")

    def _update_coordinates(self, event):
        if not self._hovering or event.xdata is None or event.ydata is None:
            self.xy_label.setText("")
            return

        ax = event.inaxes
        if ax is None:
            self.xy_label.setText("")
            return

        xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]

        xd = max(1, -int(np.floor(np.log10(xrange))) + 2)
        yd = max(1, -int(np.floor(np.log10(yrange))) + 2)

        xlabel = ax.get_xlabel() or "x"
        ylabel = ax.get_ylabel() or "y"

        self.xy_label.setText(
            f"{xlabel}: {event.xdata:.{xd}f}, {ylabel}: {event.ydata:.{yd}f}"
        )

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------
    def _on_mouse_press(self, event):
        if not self.active_axes:
            return
    
        # Middle-button drag start
        if event.button == MouseButton.MIDDLE:
            if event.x is None or event.y is None:
                return
    
            self._pan_start_px = (event.x, event.y)
            ax = self.active_axes
            self._pan_original_xlim = ax.get_xlim()
            self._pan_original_ylim = ax.get_ylim()
            return
    
        qt_event = getattr(event, "guiEvent", None)
        modifiers = qt_event.modifiers() if qt_event else Qt.NoModifier
    
        # Ctrl + double-left → zoom out
        if event.dblclick and event.button == MouseButton.LEFT and (modifiers & Qt.ControlModifier):
            self.zoom_out()
            return
    
        # Double-left → zoom in
        if event.dblclick and event.button == MouseButton.LEFT:
            self.zoom_in()
            return
    
    def _on_mouse_release(self, event):
        if event.button == MouseButton.MIDDLE:
            self._pan_start = None
            self._pan_original_xlim = None
            self._pan_original_ylim = None


    def on_select(self, eclick, erelease):
        """Zoom to the rectangle selected by the RectangleSelector.
    
        Parameters
        ----------
        eclick : MouseEvent
            Mouse down event.
        erelease : MouseEvent
            Mouse release event.
        """
        ax = self.active_axes
        if ax is None:
            return
    
        # Validate data coordinates
        if (
            eclick is None or erelease is None
            or eclick.xdata is None or erelease.xdata is None
            or eclick.ydata is None or erelease.ydata is None
        ):
            return
    
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
    
        # Prevent zero-area selection
        if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12:
            return
    
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
    
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()


    def _on_mouse_move(self, event):
        # Hard guard against half-initialized state
        if (
            not hasattr(self, "_pan_start_px")
            or not hasattr(self, "_pan_original_xlim")
            or not hasattr(self, "_pan_original_ylim")
            or self._pan_start_px is None
            or self._pan_original_xlim is None
            or self._pan_original_ylim is None
            or event.x is None
            or event.y is None
        ):
            return
    
        ax = self.active_axes
        if ax is None:
            return
    
        # Pixel delta
        x0_px, y0_px = self._pan_start_px
        dx_px = event.x - x0_px
        dy_px = event.y - y0_px
    
        # Convert pixel delta → data delta
        inv = ax.transData.inverted()
        x0_data, y0_data = inv.transform((0, 0))
        x1_data, y1_data = inv.transform((dx_px, dy_px))
    
        dx_data = x1_data - x0_data
        dy_data = y1_data - y0_data
    
        xlim0, xlim1 = self._pan_original_xlim
        ylim0, ylim1 = self._pan_original_ylim
    
        # Update limits
        ax.set_xlim(xlim0 - dx_data, xlim1 - dx_data)
        ax.set_ylim(ylim0 - dy_data, ylim1 - dy_data)
    
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        ax = event.inaxes
        if not ax:
            return

        scale = 1.25
        if event.button == "up":
            scale = 1 / scale

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xmid = event.xdata if event.xdata is not None else 0.5 * (xlim[0] + xlim[1])
        ymid = event.ydata if event.ydata is not None else 0.5 * (ylim[0] + ylim[1])

        new_xrange = (xlim[1] - xlim[0]) * scale
        new_yrange = (ylim[1] - ylim[0]) * scale

        ax.set_xlim(xmid - new_xrange / 2, xmid + new_xrange / 2)
        ax.set_ylim(ymid - new_yrange / 2, ymid + new_yrange / 2)

        self.canvas.draw()

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------
    def zoom_in(self):
        if not self.active_axes:
            return

        xlim = self.active_axes.get_xlim()
        ylim = self.active_axes.get_ylim()
        self.active_axes.set_xlim(
            [xlim[0] + 0.1 * (xlim[1] - xlim[0]), xlim[1] - 0.1 * (xlim[1] - xlim[0])]
        )
        self.active_axes.set_ylim(
            [ylim[0] + 0.1 * (ylim[1] - ylim[0]), ylim[1] - 0.1 * (ylim[1] - ylim[0])]
        )
        self.canvas.draw()

    def zoom_out(self):
        if not self.active_axes:
            return

        xlim = self.active_axes.get_xlim()
        ylim = self.active_axes.get_ylim()
        self.active_axes.set_xlim(
            [xlim[0] - 0.1 * (xlim[1] - xlim[0]), xlim[1] + 0.1 * (xlim[1] - xlim[0])]
        )
        self.active_axes.set_ylim(
            [ylim[0] - 0.1 * (ylim[1] - ylim[0]), ylim[1] + 0.1 * (ylim[1] - ylim[0])]
        )
        self.canvas.draw()


    def reset_view(self):
        if self.active_axes is None:
            return

        xvals = []
        yvals = []
        for line in self.active_axes.lines:
            x, y = line.get_data()
            xvals.extend(x)
            yvals.extend(y)

        if not xvals or not yvals:
            return

        x0, x1 = min(xvals), max(xvals)
        y0, y1 = min(yvals), max(yvals)

        xm = 0.05 * (x1 - x0) or 0.1
        ym = 0.05 * (y1 - y0) or 0.1

        self.active_axes.set_xlim(x0 - xm, x1 + xm)
        self.active_axes.set_ylim(y0 - ym, y1 + ym)
        self.canvas.draw()

    def copy_plot(self):
        buf = io.BytesIO()
        self.canvas.figure.savefig(buf, format="png")
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        QApplication.clipboard().setPixmap(pixmap)

    def save_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png)"
        )
        if file_path:
            self.canvas.figure.savefig(file_path)

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------
    def _show_context_menu(self, pos):
        menu = QMenu(self)

        menu.addAction(QIcon(ICON_DICT["Home"]), "Reset view", self.reset_view)
        menu.addAction(QIcon(ICON_DICT["zoom_in"]), "Zoom in", self.zoom_in)
        menu.addAction(QIcon(ICON_DICT["zoom_out"]), "Zoom out", self.zoom_out)
        menu.addAction(QIcon(ICON_DICT["Copy"]), "Copy plot", self.copy_plot)
        menu.addAction(QIcon(ICON_DICT["Save"]), "Save plot", self.save_plot)

        global_pos = self.canvas.mapToGlobal(pos)
        menu.exec_(global_pos)