# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

# -*- coding: utf-8 -*-

import sys
import io
from pathlib import Path
from typing import List, Dict, Any, Optional

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtCore import QItemSelectionModel, QPoint, Qt, QItemSelection

from PyQt5.QtGui import (QStandardItemModel, QStandardItem, QIcon, 
                         QImage, QPixmap, QColor, QPainter, QPen
                         )

from PyQt5.QtWidgets import (QWidget, QListView, QHBoxLayout, QVBoxLayout,
                             QApplication, QAbstractItemView, QMenu, QAction,
                             QSplitter, QGroupBox, QSizePolicy, QComboBox,
                             QLabel
                             )

from qt_icons import ICON_DICT, scan_icons_folder


COLORMAPS = ("Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Managua", "Berlin", "Vanimo", "Turbo", "Terrain", "Rainbow")

class CustomNavigationToolbar(NavigationToolbar):
    """
    NavigationToolbar with custom icons, layout, spacers and XY readout.

    The constructor expects the path to a directory that contains PNG files
    named exactly after the button names that should be replaced.
    For example, ``icons/Home.png`` will replace the default “Home” icon.

    Parameters
    ----------
    canvas : matplotlib.figure.FigureCanvasBase
        Matplotlib canvas that the toolbar controls.
    parent : QWidget | None
        Optional Qt parent widget.
    icons_dir : str | None
        Path to a directory with PNG files.  If ``None`` (default),
        the toolbar keeps the original icons.
    """

    def __init__(self, canvas, parent=None, icons_dir=None):
        # Custom icon handling (your existing logic)
        self._custom_icons = scan_icons_folder(icons_dir) if icons_dir else {}
        original_items = list(NavigationToolbar.toolitems)
        new_items = []
        
        skip = {"Back", "Forward", "Subplots", "Customize"}
        for name, tooltip, image, command in original_items:
            if name in self._custom_icons and name not in skip:
                icon_path = Path(self._custom_icons[name])
                image = str(icon_path.parent / icon_path.stem)
                new_items.append((name, tooltip, image, command))

        self.__class__.toolitems = new_items
        super().__init__(canvas, parent)
        self.__class__.toolitems = original_items

        # ===========================
        # Create Custom Toolbar Row
        # ===========================
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        # Spacer left of tools
        left_spacer = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(left_spacer, 2)

        self.parent = parent

        # Add colormap selector
        label_cmap = QLabel("Colormap:")
        self.cmap_selector = QComboBox()
        for name in COLORMAPS:
            self.cmap_selector.addItem(name)
        self.cmap_selector.currentTextChanged.connect(self._set_new_colormap)

        layout.addWidget(label_cmap)
        layout.addWidget(self.cmap_selector)

        # Spacer between controls and XY
        middle_spacer = QWidget()
        middle_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(middle_spacer)

        # Create XY label but don't show coordinates initially
        self.xy_label = QLabel(" ", self)
        self.addWidget(self.xy_label)

        # Connect mouse move
        canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        # Connect enter/leave events
        canvas.enterEvent = self._on_enter
        canvas.leaveEvent = self._on_leave
        
        self._hovering = False

        # Add custom widget to the toolbar
        self.addWidget(container)

        # Connect signals
        canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    # -------------------------------------------------------
    def set_message(self, s: str) -> None:
        """Suppress the default Matplotlib XY coordinates."""
        # Prevent the inherited toolbar from showing coordinates.
        return

    def _on_enter(self, event):
        """Mouse entered the canvas: start showing coordinates."""
        self._hovering = True
        #self.xy_label.setText("x: –, y: –")
        self.xy_label.setText("")

    def _on_leave(self, event):
        """Mouse left the canvas: hide coordinates."""
        self._hovering = False
        self.xy_label.setText(" ")

    def _on_mouse_move(self, event):
        """Update XY label with dynamic decimal precision based on zoom level using NumPy."""
        if not self._hovering or event.xdata is None or event.ydata is None:
            #self.xy_label.setText("x: –, y: –")
            self.xy_label.setText("")
            return
    
        ax = event.inaxes
        if ax is None:
            #self.xy_label.setText("x: –, y: –")
            self.xy_label.setText("")
            return
    
        # Compute dynamic precision based on axis range
        xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    
        xdigits = max(1, -int(np.floor(np.log10(xrange))) + 2)
        ydigits = max(1, -int(np.floor(np.log10(yrange))) + 2)
    
        xlabel = ax.get_xlabel() or "x"
        ylabel = ax.get_ylabel() or "y"
    
        self.xy_label.setText(
            f"{xlabel}: {event.xdata:.{xdigits}f}, {ylabel}: {event.ydata:.{ydigits}f}"
        )


    def home(self):
        """Overwriting the default Matplotlib Home Function."""
        if not self.canvas.figure.get_axes():
            return
    
        # Get data range from all lines in the plot
        x_data = []
        y_data = []
        for ax in self.canvas.figure.get_axes():
            for line in ax.lines:
                x, y = line.get_data()
                x_data.extend(x)
                y_data.extend(y)
    
        # Compute min and max values
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
    
        # Add a small margin (5% of the data range)
        x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
        y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    
        # Apply limits with margins
        for ax in self.canvas.figure.get_axes():
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
        # Redraw the canvas
        self.canvas.draw()
        
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

        
class PolarsPlotWidget(QWidget):
    """
    A widget that displays a single Polars DataFrame as a Matplotlib plot.

    The first column in the DataFrame is used for the x-axis. Columns can be selected
    from the list view to include them in the plot.

    Attributes
    ----------
    dataframe (Optional[pl.DataFrame]): The current Polars DataFrame being displayed.
    x_array (np.ndarray): NumPy array of x values (first column).
    y_arrays (Dict[str, np.ndarray]): Mapping of column names to their NumPy arrays for plotting.
    """

    def __init__(self, parent: Any = None, current_map: str = "viridis") -> None:
        """Initialize the widget with a horizontal split view for columns and plot.

        The layout consists of two main areas in group boxes arranged horizontally using QSplitter.
        Each group box is wrapped in an additional QVBoxLayout for better spacing control and
        future extensibility. The left area shows available columns to select from, while the right
        area displays the plot with navigation toolbar below it. Both panes are resizable by
        dragging the splitter handle.

        Args:
            parent: Optional parent widget. Defaults to None.
            current_map: Name of the colormap to use initially for plots. Defaults to "viridis".
        """
        super().__init__(parent)

        self.dataframe: Optional[pl.DataFrame] = None
        self.x_array: np.ndarray = np.array([])
        self.column_colors = {}
        self.current_cmap = current_map

        # Initialize UI components
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the main user interface layout."""
        # Create horizontal splitter for resizable panes
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setOpaqueResize(True)  # Optimize for resizing performance
    
        # Left pane: Column selection group box with outer vbox layout
        col_group = QGroupBox("Data Columns")
        col_layout = QVBoxLayout(col_group)
        col_layout.setContentsMargins(8, 8, 8, 8)  # Margins: left, top, right, bottom
    
        self.col_list_view = QListView()
        self.col_list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.col_model = QStandardItemModel()
        self.col_list_view.setModel(self.col_model)
        self.col_list_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        col_layout.addWidget(self.col_list_view)
    
        # Set size policy for the list view to take available space
        self.col_list_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
        # Wrap the group box in an outer vbox layout for more control
        left_wrapper = QWidget()
        left_layout = QVBoxLayout(left_wrapper)
        left_layout.setContentsMargins(0, 0, 0, 0)  # No margins since col_group has its own
        left_layout.addWidget(col_group)
    
        # Set size policy for the wrapper to take available space but not more than needed
        left_wrapper.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        left_wrapper.setMinimumWidth(150)
    
        main_splitter.addWidget(left_wrapper)
    
        # Right pane: Plot group box with outer vbox layout
        plot_group = QGroupBox("Plot View")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(8, 8, 8, 8)
    
        # Initialize matplotlib components
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = CustomNavigationToolbar(self.canvas, self, icons_dir="src/icons")
        self._init_context_menu()
    
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(self.toolbar)
    
        # Wrap the group box in an outer vbox layout
        right_wrapper = QWidget()
        right_layout = QVBoxLayout(right_wrapper)
        right_layout.setContentsMargins(0, 0, 0, 0)  # No margins since plot_group has its own
        right_layout.addWidget(plot_group)
    
        # Set size policy for the wrapper to take available space
        right_wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
        main_splitter.addWidget(right_wrapper)
    
        # Set up main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(main_splitter)
    
        # Set initial sizes (column list 25% / plot 75%) using relative ratios
        #main_splitter.setSizes([1, 5])  # Ratio of left to right
    
        self._apply_modern_style()

    # ------------------------------------------------------------------
    def set_dataframe(self, df: pl.DataFrame) -> None:
        """Load a single Polars DataFrame for display.

        Args:
            df: A Polars DataFrame where the first column will be used as x-axis.
                Subsequent columns can be selected for plotting on y-axis.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame")

        self.dataframe = df

        # Handle empty or single-column DataFrames gracefully
        if len(df.columns) == 0:
            self.x_array = np.array([])
            self.col_model.clear()
            return

        try:
            # Convert first column to numpy array for x values
            self.x_array = df[df.columns[0]].to_numpy()

        except Exception as e:
            print(f"Error converting DataFrame to numpy arrays: {e}")
            self.x_array = np.array([])
            
            return

        # Update column selection UI with available columns (excluding x-axis)
        self._update_column_list()

    # ------------------------------------------------------------------   
    def _update_column_list(self) -> None:
        if self.dataframe is not None:
            if self.dataframe.is_empty():
                return
        else:
            return
    
        cols = sorted(self.dataframe.columns[1:])
    
        self.col_model.clear()

        # Build color map
        n_columns = len(self.dataframe.columns)
        max_colors = min(n_columns, 256)
        cmap = plt.get_cmap(self.current_cmap, max_colors)
    
        self.column_colors = {}
        for i, col in enumerate(self.dataframe.columns):
            norm_i = i / (max_colors - 1) if max_colors > 1 else 0.5
            self.column_colors[col] = cmap(norm_i)  # RGBA floats (0-1)
    
        # ---- Add items with icons ----
        icon_size = 12  # small square icon
    
        for col in cols:
            item = QStandardItem(col)
    
            r, g, b, a = self.column_colors[col]
            qcolor = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))

            '''
            #Round icon
            pix = QPixmap(icon_size, icon_size)
            pix.fill(Qt.transparent)
    
            pix = QPixmap(icon_size, icon_size)
            pix.fill(Qt.transparent)
            
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setBrush(qcolor)
            p.setPen(Qt.NoPen)
            p.drawEllipse(0, 0, icon_size, icon_size)
            p.end()
            '''
            #Line Icon
            icon_w, icon_h = 18, 12
            pix = QPixmap(icon_w, icon_h)
            pix.fill(Qt.transparent)
            
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing, True)
            
            pen = QPen(qcolor, 3)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            
            y = icon_h // 2
            p.drawLine(2, y, icon_w - 2, y)
            p.end()
            
            item.setIcon(QIcon(pix))
            
            self.col_model.appendRow(item)

        # ---- Select all rows ----
        sel_model = self.col_list_view.selectionModel()
        if not cols:
            return
    
        first_idx = self.col_model.index(0, 0)
        last_idx = self.col_model.index(len(cols) - 1, 0)
        selection = QItemSelection(first_idx, last_idx)
    
        sel_model.select(selection, QItemSelectionModel.Select | QItemSelectionModel.Rows)
    
    # ------------------------------------------------------------------
    def set_new_colormap(self, cmap):
        self.current_cmap = cmap
        # Optionally refresh drawings, recolor lines, etc.
        self._update_column_list()
        
    # ------------------------------------------------------------------
    def _on_selection_changed(self, *_):
        """Refresh plot when column selection changes."""
        self._update_plot()

    # ------------------------------------------------------------------
    def _apply_modern_style(self) -> None:
        """Apply a tight, modern, minimal look to the plot."""
        self.figure.set_facecolor("#fafafa")
        self.ax.set_facecolor("#dadada")
    
        # Simplified, neutral axes
        for spine in self.ax.spines.values():
            spine.set_color("#bcbcbc")
            spine.set_linewidth(0.6)
        #self.ax.spines["top"].set_visible(False)
        #self.ax.spines["right"].set_visible(False)
    
        # Fine grid
        self.ax.grid(True, color="#bcbcbc", linewidth=0.5, alpha=0.5)
        self.ax.tick_params(
            colors="#555555",
            labelsize=6.5,
            width=0.6,
            length=3,
            pad=2,
        )
        self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        self.ax.tick_params(
            which='minor',
            width=0.4,      # Slightly thinner than major ticks
            length=1.5,     # Shorter than major ticks
            color="#999999" # Lighter color for subticks
            )
    
        self.ax.xaxis.label.set_color("#555555")
        self.ax.yaxis.label.set_color("#555555")
        self.ax.xaxis.label.set_size(7)
        self.ax.yaxis.label.set_size(7)
        self.ax.title.set_color("#222222")
        self.ax.title.set_size(8)

    # ------------------------------------------------------------------
    def _update_plot(self) -> None:
        """
        Plot selected columns from the DataFrame with modern style.

        This method clears the current plot, applies styling, and plots all selected
        columns. It also sets up interactive tooltips for hovering over lines.
        """
        if len(self.x_array) == 0 or not hasattr(self, 'ax'):
            return

        sel_col_model = self.col_list_view.selectionModel()
        selected_cols = [self.col_model.data(idx) for idx in sel_col_model.selectedIndexes()]

        # Clear and set up plot
        self.ax.clear()
        
        self._apply_modern_style()
        
        plotted_any = False
        plotted_lines: List[Any] = []
        full_labels: List[str] = []
        max_label_len = 25
        i = 0
        for col in selected_cols:
            if not hasattr(self, 'dataframe') or self.dataframe is None:
                continue
            if col not in self.dataframe.columns:
                continue
    
            try:
                y = self.dataframe[col]#.to_numpy()
            except Exception as e:
                print(f"Error accessing column {col}: {e}")
                continue
    
            ## Check for length mismatch (shouldn't happen with Polars, but good to verify)
            #if len(self.x_array) != len(y):
            #    print(f"Length mismatch in column {col}, skipping")
            #    continue
    
            full_label = f"{col}"
            short_label = (
                full_label if len(full_label) <= max_label_len
                else full_label[:max_label_len - 1] + "…"
            )

            color = self.column_colors.get(col, '#000000')
            line, = self.ax.plot(
                self.x_array, y,
                label=short_label,
                color=color,
                linewidth=1.25,
                alpha=0.9,
            )
            plotted_lines.append(line)
            full_labels.append(full_label)
            plotted_any = True
            i += 1

        if plotted_any:
            legend = self.ax.legend(
                fontsize=7,
                frameon=False,
                labelcolor="#f0f0f0",
                handlelength=2.8,
                handletextpad=0.6,
                borderaxespad=0.2,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderpad=0.0,
            )
            for text in legend.get_texts():
                text.set_color("#333333")

            # Axis label use first column from dataframe
            x_axes_label = self.dataframe.columns[0]
                      
            # Allocate space for the right legend
            self.figure.subplots_adjust(right=0.80)
        else:
            # Axis label use defaults if no data is plotted
            x_axes_label = "x-axes"
            self.figure.subplots_adjust(right=0.98)
        y_axes_label = "photometric value"
        self.ax.set_xlabel(self.dataframe.columns[0])
        self.ax.set_ylabel(y_axes_label)

        # --- Hover tooltip ---
        annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="#333333", ec="#CCCCCC", lw=0.5),
            color="#FFFFFF",
            fontsize=8,
            visible=False,
        )

        def on_hover(event) -> None:
            """
            Handle hover events to display tooltips.

            This function checks if the mouse is over a line, and if so, displays
            an annotation with the column name.
            """
            if event.inaxes != self.ax:
                annot.set_visible(False)
                self.canvas.draw_idle()
                return

            visible = False
            for line, label in zip(plotted_lines, full_labels):
                cont, ind = line.contains(event)
                if cont:
                    x_data, y_data = line.get_data()
                    idx = ind["ind"][0]
                    x, y = x_data[idx], y_data[idx]
                    annot.xy = (x, y)
                    annot.set_text(label)
                    line_color = line.get_color()
                    r, g, b, a = line_color[:4]
                    
                    # Create darker edge by reducing lightness while preserving hue
                    # Using a simple method: reduce each channel by 30%
                    dark_r = max(0, r - 0.1)
                    dark_g = max(0, g - 0.1)
                    dark_b = max(0, b - 0.1)  # Slightly more reduction for blue
                    edge_color = (dark_r, dark_g, dark_b, 1.0)
                    bg_color = (dark_r, dark_g, dark_b, 0.66)
                    annot.set_color("#FFFFFF")  # For background if needed
                    annot.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=bg_color, edgecolor = edge_color, lw=0.5))  # Slightly transparent
                    
                    annot.set_visible(True)                   
                    visible = True
                    break

            if not visible and annot.get_visible():
                annot.set_visible(False)

            self.canvas.draw_idle()

        # Disconnect previous event handler to avoid multiple connections
        if hasattr(self, '_hover_cid'):
            self.figure.canvas.mpl_disconnect(self._hover_cid)
        self._hover_cid = self.figure.canvas.mpl_connect("motion_notify_event", on_hover)

        self.figure.tight_layout(pad=1.0, w_pad=2.5, h_pad=3.0)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _create_context_menu(self) -> QMenu:
        """
        Create and return a context menu for right-click actions.

        Returns:
            QMenu: A context menu with actions to copy the image, save it, or reset/zoom home.
        """
        context_menu = QMenu(self)

        # Copy action
        copy_action = QAction(QIcon(ICON_DICT["Copy"]), "Copy Image", self)
        copy_action.triggered.connect(self._copy_image)
        context_menu.addAction(copy_action)

        # Save action
        save_action = QAction(QIcon(ICON_DICT["Save"]), "Save Image...", self)
        save_action.triggered.connect(self._save_image)
        context_menu.addAction(save_action)

        return context_menu

    def _init_context_menu(self) -> None:
        """
        Initialize and connect the right-click context menu for the canvas.
        """
        # Create menu once
        self.context_menu = self._create_context_menu()
    
        # Connect mouse button event to show the menu
        if not hasattr(self, "_context_cid"):
            self._context_cid = self.canvas.mpl_connect("button_press_event", self._show_context_menu)
    

    def _show_context_menu(self, event) -> None:
        """
        Show the context menu on right-click exactly at the mouse position.
    
        Args:
            event: Matplotlib mouse event.
        """
        if event.button == 3:  # Right-click
            # Matplotlib coordinates (origin bottom-left)
            x = int(event.x)
            y = int(event.y)
    
            # Convert to Qt coordinates (origin top-left)
            y = self.canvas.height() - y
            global_pos = self.canvas.mapToGlobal(QPoint(x, y))
    
            # Show menu
            self.context_menu.exec_(global_pos)

    def _copy_image(self) -> None:
        """
        Copy the current Matplotlib figure to the system clipboard.
    
        This method renders the figure to a PNG in memory, converts it to a QPixmap,
        and sets it on the Qt clipboard. Works reliably with FigureCanvasQTAgg.
        """
        if not hasattr(self, 'figure') or self.figure is None:
            return  # No figure available
    
        try:
            # Save figure to a bytes buffer in PNG format
            buffer = io.BytesIO()
            self.figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_data = buffer.getvalue()
    
            # Convert bytes to QImage
            qimage = QImage.fromData(img_data)
            if qimage.isNull():
                return  # Invalid image
    
            # Convert QImage to QPixmap and copy to clipboard
            pixmap = QPixmap.fromImage(qimage)
            clipboard = QApplication.clipboard()
            clipboard.setPixmap(pixmap)
        except Exception as e:
            print(f"Failed to copy image to clipboard: {e}")

    def _save_image(self) -> None:
        """
        Save the current figure to a file.

        This action displays a save dialog and saves the current image with high quality.
        """
        from PyQt5.QtWidgets import QFileDialog

        # Set up file dialog options
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)",
            options=options
        )

        if file_name:
            # Extract extension from file name
            ext = file_name.split('.')[-1].lower()
            if ext not in ['png', 'jpg', 'jpeg']:
                file_name += '.png'  # Default to PNG

            # Save with high DPI for quality
            self.figure.savefig(file_name, dpi=300)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    wavelengths = np.arange(400, 801, 1)

    data = {}
    for i, col_name in zip(range(1, 21), [chr(c) for c in range(66, 86)]):  # B to V
        data[col_name] = np.random.random(size=len(wavelengths))*np.random.randint(0, 3, 1)+np.random.randint(0, 50, 1)
    
    # Create DataFrame with Wavelength as first column
    df_large = pl.DataFrame({
        "Wavelength / nm": wavelengths,
        **data
    })

    widget = PolarsPlotWidget()
    widget.set_dataframe(df_large)
    widget.resize(1000, 600)
    widget.show()

    sys.exit(app.exec())
