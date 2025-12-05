# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
from typing import List, Dict, Any, Optional

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# Switch to backend_qtagg for PySide6 compatibility
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


from PySide6.QtCore import QItemSelectionModel, Qt, QItemSelection

from PySide6.QtGui import (QStandardItemModel, QStandardItem, QIcon, 
                         QPixmap, QColor, QPainter, QPen
                         )

from PySide6.QtWidgets import (QWidget, QListView, QVBoxLayout,
                             QApplication, QAbstractItemView, 
                             QSplitter, QGroupBox, QSizePolicy, 
                             )

#from qt_icons import ICON_DICT
from qt_plottoolbar import CSVPlottoolbar
from plotstyler import PlotStyler

COLORMAPS = ("Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Managua", "Berlin", "Vanimo", "Turbo", "Terrain", "Rainbow")

        
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

    def __init__(self, parent: Any = None, current_map: str = "viridis", plot_style: str = "grey") -> None:
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

        self.current_style = plot_style  # Default to light theme light or dark
        self.plot_styler = PlotStyler(self.current_style)

        # Initialize UI components
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the main user interface layout."""
        # Create horizontal splitter for resizable panes
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setOpaqueResize(True)  # Optimize for resizing performance
    
        # Left pane: Column selection group box with outer vbox layout
        col_group = QGroupBox("Data Columns")
        col_layout = QVBoxLayout(col_group)
        col_layout.setContentsMargins(8, 8, 8, 8)  # Margins: left, top, right, bottom
    
        self.col_list_view = QListView()
        self.col_list_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.col_model = QStandardItemModel()
        self.col_list_view.setModel(self.col_model)
        self.col_list_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        col_layout.addWidget(self.col_list_view)
    
        # Set size policy for the list view to take available space
        self.col_list_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
        # Wrap the group box in an outer vbox layout for more control
        left_wrapper = QWidget()
        left_layout = QVBoxLayout(left_wrapper)
        left_layout.setContentsMargins(4, 4, 4, 4)  # No margins since col_group has its own
        left_layout.addWidget(col_group)
    
        # Set size policy for the wrapper to take available space but not more than needed
        left_wrapper.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        left_wrapper.setMinimumWidth(150)
    
        main_splitter.addWidget(left_wrapper)
    
        # Right pane: Plot group box with outer vbox layout
        plot_group = QGroupBox("Plot View")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(8, 8, 8, 0)
    
        # Initialize matplotlib components
        self.figure, self.ax = plt.subplots(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = CSVPlottoolbar(self.canvas, self)#, icons_dir="src/icons")
        
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.toolbar, 0)
    
        # Wrap the group box in an outer vbox layout
        right_wrapper = QWidget()
        right_layout = QVBoxLayout(right_wrapper)
        right_layout.setContentsMargins(4, 4, 4, 4)  # No margins since plot_group has its own
        right_layout.addWidget(plot_group)
    
        # Set size policy for the wrapper to take available space
        right_wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
        main_splitter.addWidget(right_wrapper)
    
        # Set up main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(main_splitter)
     
    
        self.setup_auto_tightlayout()
    
        self._apply_modern_style()

        # ===== Style =====
        self.setStyleSheet("""
           QGroupBox {
               font-weight: bold;
               border: 1px solid #AAAAAA;
               border-radius: 6px;
               margin-top: 8px;
           }
           QGroupBox::title {
               subcontrol-origin: margin;
               left: 10px;
               padding: 0 4px;
           }
           QPushButton {
               padding: 4px 10px;
           }
           QLabel {
               min-width: 80px;
           }
           QListWidget{
               border: 1px solid #AAAAAA;
               border: 1px solid #AAAAAA;
           }
       """)

    def get_plot_settings(self) -> Dict[str, str]:
        """
        Get the current plot settings.
        
        Returns a dictionary containing:
        - "style": Current style name (str)
        - "colormap": Current colormap name (str)
        
        Returns:
            Dictionary with current plot settings
        """
        settings_dict = {}
        settings_dict["style"] = self.current_style
        settings_dict["colormap"] = self.current_cmap
        return settings_dict
        
    def set_plot_settings(self, settings_dict: Dict[str, Any]) -> None:
        """
        Set plot settings from a dictionary.
        
        Args:
            settings_dict: Dictionary containing plot settings. Expected keys:
                - "style": Style name (str)
                - "colormap": Colormap name (str)
        
        The method will use default values ("grey" for style, "viridis" for colormap)
        if the corresponding keys are missing in the input dictionary.
        """
        self.current_style = settings_dict.get("style", "grey")
        self.current_cmap = settings_dict.get("colormap", "viridis")
        self.setup_auto_tightlayout()
        self._apply_modern_style()
        
    def _return_rect_selector_style(self) -> Dict[str, Any]:
        """
        Get rectangle selector style from the plot styler.
        
        Returns:
            Dictionary containing rectangle selector properties, or empty dict if not available.
        """
        return self.plot_styler.get_rect_selector_style()    
    
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
        self._apply_modern_style()
        self._update_column_list()

    # ------------------------------------------------------------------   
    def _update_column_list(self) -> None:
        """Updates the column list display with colored icons representing each dataframe column.

        This method populates the column selection model with visual representations of each
        column, using a colormap to assign unique colors. It creates line-style icons for each
        column and selects all columns by default when the list is populated.
        
        Preconditions:
            - self.dataframe must be non-empty for updates to occur (empty or None skips update)
        
        Side effects:
            - Clears and repopulates self.col_model with QStandardItems representing columns
            - Stores column color mappings in self.column_colors dictionary
            - Sets selection of all rows in the column list view
        """
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
            pix.fill(Qt.GlobalColor.transparent)
    
            pix = QPixmap(icon_size, icon_size)
            pix.fill(Qt.GlobalColor.transparent)
            
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setBrush(qcolor)
            p.setPen(Qt.NoPen)
            p.drawEllipse(0, 0, icon_size, icon_size)
            p.end()
            '''
            #Line Icon
            icon_w, icon_h = 18, 12
            pix = QPixmap(icon_w, icon_h)
            pix.fill(Qt.GlobalColor.transparent)
            
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            
            pen = QPen(qcolor, 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
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
    
        sel_model.select(selection, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    
    # ------------------------------------------------------------------
    def set_new_colormap(self, cmap):
        """
        Set a new colormap to be used for plotting.
        
        Args:
            cmap: The name of the colormap (str) or None. If None is provided,
                  it will revert to the default colormap ("viridis").
        This method updates the current colormap and triggers an update of related
        elements that depend on the colormap
        """
        self.current_cmap = cmap
        # Optionally refresh drawings, recolor lines, etc.
        self._update_column_list()
            
    # ------------------------------------------------------------------
    def _on_selection_changed(self, *_):
        """
        Refresh the plot when column selection changes.
        
        This method is typically connected to a signal that triggers on selection change.
        It ensures the visualization is updated to reflect the current data selection.
        
        Args:
            *_ : Any additional arguments are ignored as this is connected to signals
        """
        self._update_plot()

    def _apply_modern_style(self) -> None:
        """
        Apply modern styling to the current plot.
        
        This sets a clean, professional look with appropriate spacing and formatting.
        The actual style is determined by self.current_style and applied through
        the plot_styler.
        """
        self.plot_styler.apply_style(self.current_style, self.canvas, self.ax)

    def setup_auto_tightlayout(self) -> None:
        """Set up automatic tight layout updates for a Matplotlib figure in PyQt5.
    
        Args:
            widget: Any QWidget that might be resized (e.g., QMainWindow or container)
            fig: The matplotlib Figure object to keep in tight layout.
        """
        def on_resize(event=None):
            try:
                #plt.tight_layout()
                self.figure.tight_layout(pad=1.0, w_pad=2.5, h_pad=3.0)
                if hasattr(self.figure.canvas, 'draw_idle'):
                    self.figure.canvas.draw_idle()
            except Exception as e:
                print(f"Error updating tight layout: {str(e)}")
    
        # Save the original resizeEvent
        original_resize = self.resizeEvent
    
        def new_resize(event):
            # First call the original resize handler to maintain all existing behavior
            if original_resize:
                original_resize(event)
            on_resize()
    
        # Replace the widget's resizeEvent with our modified version
        self.resizeEvent = new_resize

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
            # legend = self.ax.legend(
            #     fontsize=7,
            #     frameon=False,
            #     labelcolor="#f0f0f0",
            #     handlelength=2.8,
            #     handletextpad=0.6,
            #     borderaxespad=0.2,
            #     loc="upper left",
            #     bbox_to_anchor=(1.02, 1.0),
            #     borderpad=0.0,
            # )
            # for text in legend.get_texts():
            #     text.set_color("#333333")

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
            fontsize=12,
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