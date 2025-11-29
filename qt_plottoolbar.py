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
    """Lightweight custom Matplotlib toolbar for interactive plot control.

    Provides essential navigation and visualization tools without using the full
    NavigationToolbar2QT. Designed for quick access to common operations while
    maintaining a minimal footprint.

    Features:
        • Home / Reset view button
        • Zoom In / Zoom Out buttons
        • Pan functionality (middle mouse drag)
        • Scroll-wheel zoom support
        • Copy plot to clipboard
        • Save figure as PNG
        • Colormap selector dropdown
        • XY coordinate readout when hovering

    Attributes:
        canvas: The FigureCanvasQTAgg instance this toolbar controls.
        figure: The matplotlib Figure object being displayed.
        active_axes: Currently active Axes for operations (defaults to first axes).
        _hovering: Boolean tracking whether mouse is over the canvas.

    Args:
        canvas: FigureCanvasQTAgg instance that contains the plot to be controlled
        parent: Optional parent QWidget (default None)
    """

    def __init__(self, canvas: FigureCanvasQTAgg, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.canvas = canvas
        self.figure = canvas.figure
        self.active_axes = self.canvas.figure.get_axes()[0]
        self._hovering = False

        self.parent = parent
        
        # Current pan state
        self._pan_start = None

        self._build_ui()
        self._make_rect_selector()
        self._connect_canvas_events()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Constructs the user interface for the plot controller.

        Creates and arranges several control buttons (home/reset, zoom in/out,
        copy, save), a coordinate display label, and a colormap selector.
        Connects button click events to their respective handler methods.
        
        Side effects:
            Sets up the main QHBoxLayout for the widget with appropriate margins and spacing.
            Creates and stores button instances (self.btn_home, self.btn_zoom_in, etc.).
            Creates and connects the colormap selector combo box.
            Configures the coordinate display label (self.xy_label).
        """
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

        # Spacer 1
        spacer1 = QWidget()
        spacer1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(spacer1)

        # XY label
        self.xy_label = QLabel(" ")
        self.xy_label.setMinimumWidth(150)
        layout.addWidget(self.xy_label)

        # Spacer 2
        spacer2 = QWidget()
        spacer2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(spacer2)

        # Colormap selector
        layout.addWidget(QLabel("Colormap:"))
        self.cmap_selector = QComboBox()
        for name in COLORMAPS:
            self.cmap_selector.addItem(name)
        self.cmap_selector.currentTextChanged.connect(self._set_new_colormap)
        layout.addWidget(self.cmap_selector)

    def _make_btn(self, text: str, tooltip: str, icon) -> QToolButton:
        """Creates a consistently styled tool button with given properties.

        This is a helper method to standardize the appearance of control buttons
        throughout the interface. Buttons created this way will have consistent
        size, tooltip behavior, and icon placement.
        
        Args:
            text: The display text for the button.
            tooltip: Help text shown on hover.
            icon: QIcon instance to use for the button's visual representation.
        
        Returns:
            QToolButton: A configured button ready to be added to a layout.
        """
        btn = QToolButton(self)
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.setIcon(icon)
        btn.setIconSize(QSize(24, 24))
        return btn
    
    def _make_rect_selector(self) -> None:
        """
        Create a RectangleSelector for the current active axes.
        Checks:
        - Validates that self.parent._return_rect_selector_style exists and returns proper properties
        """
        if not hasattr(self.parent, '_return_rect_selector_style'):
            self.rect_props = dict(facecolor='lightblue', alpha=0.3, edgecolor='blue') 
        else:
            self.rect_props = self.parent._return_rect_selector_style()
        self.rect_selector = RectangleSelector(
            self.active_axes,
            self.on_select,
            #drawtype='box',
            useblit=True,
            button=[1],        # left mouse
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=False,
            props = self.rect_props
        )
        
    # ------------------------------------------------------------------
    # Canvas events
    # ------------------------------------------------------------------
    def _connect_canvas_events(self) -> None:
        """Connects various canvas events to their handler methods.

        This method sets up event handlers for mouse press, release, movement,
        scrolling, and context menu requests. It also configures the hover state
        tracking when mouse enters or leaves the canvas area.
        
        Side effects:
            Connects several event handlers to the matplotlib canvas.
            Sets custom context menu policy and connects it to the context menu handler.
        """
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
        """Sets the hovering state to True when mouse enters canvas.

        Args:
            event: The enter event object (not used in this implementation).
        """
        self._hovering = True

    def _on_leave(self, event):
        """Sets the hovering state to False and clears coordinate label when mouse leaves canvas.

        Args:
            event: The leave event object (not used in this implementation).

        Side effects:
            Sets `_hovering` to False.
            Clears the text of `self.xy_label`.
        """
        self._hovering = False
        self.xy_label.setText(" ")

    def _update_coordinates(self, event):
        """Updates the coordinate display label with current mouse position.

        Formats coordinates based on axis ranges for appropriate decimal places.
        Only updates when hovering and valid data coordinates are available.
        
        Args:
            event: A motion notify event containing xdata, ydata, inaxes attributes.
        
        Side effects:
            Sets the text of `self.xy_label` with formatted coordinate information
            or clears it if conditions aren't met (not hovering or no valid axes).
        """
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
        """Handles mouse press events for panning and zooming.

        When middle button is pressed, stores the starting position for potential panning.
        When left button is double-clicked with Ctrl modifier, triggers zoom out.
        When left button is double-clicked without Ctrl, triggers zoom in.
        
        Args:
            event: A mouse event object containing information about the press, including
                   button (MouseButton), coordinates (x, y), and possibly a Qt GUI event for modifiers.
        
        Side effects:
            May store pan start position in `_pan_start_px` when middle button is pressed.
            May store original x/y limits of active axes when middle button is pressed.
            Calls `zoom_out()` or `zoom_in()` on appropriate double-click events.
        """
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
        """Handles mouse release events to clean up panning state.

        Specifically clears the pan start position and original axis limits when the middle button is released.

        Args:
            event: A mouse event object containing information about the release, including
                   which button was released.
        """
        if event.button == MouseButton.MIDDLE:
            self._pan_start = None
            self._pan_original_xlim = None
            self._pan_original_ylim = None


    def on_select(self, eclick, erelease) -> None:
        """Handles zoom to rectangle selection when the user draws a box with the mouse.

        This method takes the start and end points of a rectangle selector,
        validates them, and sets the axes limits to match the selected area.
        If the selection has zero area or invalid coordinates, it does nothing.

        Args:
            eclick: Mouse down event that started the selection
            erelease: Mouse release event that ended the selection
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


    def _on_mouse_move(self, event) -> None:
        """Handles panning when mouse moves with a button pressed.

        This method tracks pixel movement and converts it to data coordinate movement,
        updating the view by translating the original axes limits. If any of the required
        state variables are not initialized, it does nothing.
        
        Note: This is an internal method that should only be called during mouse drag events.
        """
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

    def _on_scroll(self, event) -> None:
        """Handles zoom in/out when scrolling with the mouse wheel.

        The zoom center is either at the cursor position (if xdata/ydata available)
        or at the current view center. Zooming uses a scale factor of 1.25 per step
        for both up and down scrolls.
        
        Args:
            event (MouseEvent): The scroll event containing axis information
        """
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
    def zoom_in(self) -> None:
        """Zooms in on the active axes by reducing the visible range by 20%.

        The view is reduced equally in all directions - 10% from each side.
        If no active axes are present, does nothing and returns immediately.
        
        This method modifies the x and y limits of the axes and triggers a redraw.
        """
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

    def zoom_out(self) -> None:
        """Zooms out on the active axes by increasing the visible range by 20%.

        The view is expanded equally in all directions - 10% from each side.
        If no active axes are present, does nothing and returns immediately.

        This method modifies the x and y limits of the axes and triggers a redraw.
        """
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


    def reset_view(self) -> None:
        """Resets the view of the active axes to fit all plotted data with margin.

        Calculates the minimum and maximum x/y values from all line objects,
        then sets the axes limits to include these values with a 5% margin
        on each side (or at least 10% if the data range is very small).
        If no lines are present or no active axes, does nothing.

        This method modifies the x and y limits of the axes and triggers a redraw.
        """
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

    def copy_plot(self) -> None:
        """Copies the current plot to clipboard as an image.

        This function captures the current figure as PNG format in memory, converts it to
        a QPixmap, and places this image on the system clipboard for pasting elsewhere.
        """
        buf = io.BytesIO()
        self.canvas.figure.savefig(buf, format="png")
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        QApplication.clipboard().setPixmap(pixmap)

    def save_plot(self) -> None:
        """Saves the current plot to a file.

        Opens a file dialog to let the user select a location and filename. The file is
        saved in PNG format by default.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png)"
        )
        if file_path:
            self.canvas.figure.savefig(file_path)

    def _set_new_colormap(self, name: str) -> None:
        """
        Look up the colormap from the predefined dictionary and pass
        it back to the parent widget.
        """
        if name in COLORMAPS:
            # return the colormap to parent if it implements setter
            #parent = self.parent()
            if hasattr(self.parent, "set_new_colormap"):
                self.parent.set_new_colormap(name.lower())

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------
    def _show_context_menu(self, pos) -> None:
        """Displays a context menu at the specified position with plot actions.

        The menu includes actions for resetting view, zooming in/out,
        copying and saving the plot. The menu is shown at the global coordinates
        corresponding to the given canvas-relative position.
        
        Args:
            pos (QPoint): The position where the context menu should be shown,
                 relative to this widget's coordinate system.
        """
        menu = QMenu(self)

        menu.addAction(QIcon(ICON_DICT["Home"]), "Reset view", self.reset_view)
        menu.addAction(QIcon(ICON_DICT["zoom_in"]), "Zoom in", self.zoom_in)
        menu.addAction(QIcon(ICON_DICT["zoom_out"]), "Zoom out", self.zoom_out)
        menu.addAction(QIcon(ICON_DICT["Copy"]), "Copy plot", self.copy_plot)
        menu.addAction(QIcon(ICON_DICT["Save"]), "Save plot", self.save_plot)

        global_pos = self.canvas.mapToGlobal(pos)
        menu.exec_(global_pos)