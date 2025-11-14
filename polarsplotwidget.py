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
#import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtCore import QItemSelectionModel, QPoint, Qt

from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon, QImage, QPixmap

from PyQt5.QtWidgets import (
    QWidget,
    QListView,
    QHBoxLayout,
    QVBoxLayout,
    QApplication,
    QAbstractItemView,
    QMenu,
    QAction,
    QSplitter,
    QGroupBox,
    QSizePolicy
    )

from qt_icons import ICON_DICT, scan_icons_folder

class CustomNavigationToolbar(NavigationToolbar):
    """
    Matplotlib navigation toolbar with user‑supplied icons.

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

    def __init__(self, canvas: Any, parent: Any = None, icons_dir: str | None = None) -> None:
        self._custom_icons: Dict[str, str] = (
            scan_icons_folder(icons_dir) if icons_dir else {}
        )

        # Copy original toolitems so we don't mutate the class attribute globally
        original_items = list(NavigationToolbar.toolitems)  # type: ignore[attr-defined]
        new_items = []

        for name, tooltip, image, command in original_items:
            if name in self._custom_icons:
                # Use the custom icon name (without extension)
                image = str(Path(self._custom_icons[name]).parent / Path(self._custom_icons[name]).stem)
            new_items.append((name, tooltip, image, command))

        # Temporarily override the class attribute for this instance only
        self.__class__.toolitems = new_items  # type: ignore[assignment]
        super().__init__(canvas, parent)
        self.__class__.toolitems = original_items  # restore

        # Tell the toolbar where to look for icons
        if icons_dir:
            self.icon_dir = str(Path(icons_dir).resolve())

        # Store initial view limits when first shown
        self._initial_limits: Dict[str, List[float]] = {}

    def home(self):
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

    def __init__(self, parent: Any = None) -> None:
        """Initialize the widget with a horizontal split view for columns and plot.
    
        The layout consists of two main areas in group boxes arranged horizontally using QSplitter.
        Each group box is wrapped in an additional QVBoxLayout for better spacing control and
        future extensibility. The left area shows available columns to select from, while the right
        area displays the plot with navigation toolbar below it. Both panes are resizable by
        dragging the splitter handle.
    
        Args:
            parent: Optional parent widget. Defaults to None.
        """
        super().__init__(parent)
    
        self.dataframe: Optional[pl.DataFrame] = None
        self.x_array: np.ndarray = np.array([])
    
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
        """Update the column list view to show all available columns (excluding x-axis)."""
        if not hasattr(self, 'dataframe') or self.dataframe is None:
            return

        cols = sorted([col for col in self.dataframe.columns[1:]])

        self.col_model.clear()
        for col in cols:
            self.col_model.appendRow(QStandardItem(col))

        # Select all columns initially
        col_sel_model = self.col_list_view.selectionModel()
        for row in range(len(cols)):
            idx = self.col_model.index(row, 0)
            col_sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)

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
            colors="#444444",
            labelsize=8,
            width=0.6,
            length=3,
            pad=2,
        )
    
        self.ax.xaxis.label.set_color("#333333")
        self.ax.yaxis.label.set_color("#333333")
        self.ax.xaxis.label.set_size(8)
        self.ax.yaxis.label.set_size(8)
        self.ax.title.set_color("#222222")
        self.ax.title.set_size(9)

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
        
        # Gradient color map
        cmap = plt.colormaps["viridis"]
        total_series = sum(
            1 for c in selected_cols if c in self.dataframe.columns
            )
        norm = mcolors.Normalize(vmin=0, vmax=max(total_series - 1, 1))
        
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
            color = cmap(norm(i))
            line, = self.ax.plot(
                self.x_array, y,
                label=short_label,
                color=color,
                linewidth=1.4,
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

            # Use first column name as x-axis label
            self.ax.set_xlabel(self.dataframe.columns[0], fontsize=8)
            self.ax.set_ylabel("photometric value", fontsize=8)

            # Allocate space for the right legend
            self.figure.subplots_adjust(right=0.80)
        else:
            self.figure.subplots_adjust(right=0.98)

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
        "Wavelength": wavelengths,
        **data
    })

    widget = PolarsPlotWidget()
    widget.set_dataframe(df_large)
    widget.resize(1000, 600)
    widget.show()

    sys.exit(app.exec())
