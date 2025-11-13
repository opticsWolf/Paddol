# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 22:26:19 2025

@author: Frank
"""

# -*- coding: utf-8 -*-

import sys
import io
from pathlib import Path
from typing import List, Dict, Any, Union

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtCore import Qt, QModelIndex, QItemSelectionModel, QPoint, QMimeData, Qt    

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
    )

def scan_icons_folder(folder_path: Union[str, Path]) -> Dict[str, str]:
    """Scan *folder_path* for icon files and return a mapping of base names to paths.

    The function looks only at regular files whose extensions are ``.png`` or
    ``.svg`` (case‑insensitive).  For each matching file the key in the returned
    dictionary is the filename without its extension (*stem*), while the value
    is the absolute path as a string, which can be supplied directly to PyQt
    widgets.

    Args:
        folder_path: Directory containing icon assets. Can be a ``str`` or
            :class:`pathlib.Path`.

    Returns:
        dict[str, str]: Mapping from icon base name to absolute file path.
    """
    folder = Path(folder_path)
    icon_dict: Dict[str, str] = {}
    if not folder.is_dir():
        return icon_dict
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in {".png", ".svg"}:
            icon_dict[file.stem] = str(file.resolve())
    return icon_dict

ICON_DICT = scan_icons_folder(Path("src/icons"))

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

class PolarsPlotWidget(QWidget):
    """
    A widget that displays one or more Polars DataFrames as a Matplotlib plot.

    The first column in each DataFrame is used for the x-axis. Multiple DataFrames can be plotted together,
    with the legend using "dict_key: column_name" format. Column names appear in QListViews; selecting
    items updates the plot dynamically.

    Attributes
    ----------
    dataframes (Dict[str, pl.DataFrame]): A dictionary mapping names to Polars DataFrames.
    x_arrays (Dict[str, np.ndarray]): Mapping from DataFrame names to their first column's NumPy array.
    y_arrays (Dict[str, Dict[str, np.ndarray]]): Nested mapping of DataFrame names to column names
            and corresponding NumPy arrays for plotting.
    """

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)

        self.dataframes: Dict[str, pl.DataFrame] = {}
        self.x_arrays: Dict[str, np.ndarray] = {}
        self.y_arrays: Dict[str, Dict[str, np.ndarray]] = {}

        # ----- UI layout -----
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left side: data source selection (dict keys) and column list
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        # DataFrame list view
        self.df_list_view = QListView()
        self.df_list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.df_model = QStandardItemModel()
        self.df_list_view.setModel(self.df_model)
        self.df_list_view.selectionModel().selectionChanged.connect(
            self._on_df_selection_changed
        )
        left_layout.addWidget(self.df_list_view, stretch=1)

        # Column list view
        self.col_list_view = QListView()
        self.col_list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.col_model = QStandardItemModel()
        self.col_list_view.setModel(self.col_model)
        self.col_list_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        left_layout.addWidget(self.col_list_view, stretch=2)

        # Right: Matplotlib canvas
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = CustomNavigationToolbar(self.canvas, self, icons_dir="src/icons")
        self._init_context_menu() 

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)
        plot_layout.addWidget(self.toolbar)

        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(plot_widget, stretch=3)
        
    # ------------------------------------------------------------------
    def set_data_dict(self, data_dict: Dict[str, pl.DataFrame]) -> None:
        """Load a dictionary of Polars DataFrames."""
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict must be a dict of Polars DataFrames")

        self.dataframes.clear()
        self.x_arrays.clear()
        self.y_arrays.clear()

        for key, df in data_dict.items():
            if not isinstance(df, pl.DataFrame):
                continue
            self.dataframes[key] = df
            self.x_arrays[key] = df[df.columns[0]].to_numpy()
            self.y_arrays[key] = {
                col: df[col].to_numpy() for col in df.columns[1:]
            }

        # Populate df list view
        self.df_model.clear()
        for key in self.dataframes.keys():
            self.df_model.appendRow(QStandardItem(key))

        # Automatically select all dataframes
        sel_model = self.df_list_view.selectionModel()
        for row in range(len(self.dataframes)):
            idx = self.df_model.index(row, 0)
            sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)

        # Update combined column list
        self._update_column_list()

    # ------------------------------------------------------------------
    def _update_column_list(self) -> None:
        """Update the column list view to show all unique columns across selected DataFrames."""
        sel_model = self.df_list_view.selectionModel()
        selected_keys = [
            self.df_model.data(idx)
            for idx in sel_model.selectedIndexes()
        ]
        all_cols = set()
        for key in selected_keys:
            all_cols.update(self.dataframes[key].columns[1:])  # skip x-axis
        all_cols = sorted(all_cols)

        self.col_model.clear()
        for col in all_cols:
            self.col_model.appendRow(QStandardItem(col))

        # Select all columns initially
        col_sel_model = self.col_list_view.selectionModel()
        for row in range(len(all_cols)):
            idx = self.col_model.index(row, 0)
            col_sel_model.select(idx, QItemSelectionModel.Select | QItemSelectionModel.Rows)

        self._update_plot()

    # ------------------------------------------------------------------
    def _on_df_selection_changed(self, *_):
        """Update available columns when DataFrame selection changes."""
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

        # Compact layout
        #self.figure.subplots_adjust(left=0.12, right=0.94, top=0.94, bottom=0.12)

    # ------------------------------------------------------------------
    def _update_plot(self) -> None:
        """
        Plot selected columns from all selected DataFrames with modern style, with truncation and tooltips.

        This method clears the current plot, applies a modern style, and plots the selected
        data series. It also sets up interactive tooltips for hovering over lines.
        """
        sel_df_model = self.df_list_view.selectionModel()
        selected_dfs = [self.df_model.data(idx) for idx in sel_df_model.selectedIndexes()]
        sel_col_model = self.col_list_view.selectionModel()
        selected_cols = [self.col_model.data(idx) for idx in sel_col_model.selectedIndexes()]

        # Ensure figure and axes are set
        if not hasattr(self, 'figure') or self.figure is None:
            self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.clear()
        self._apply_modern_style()

        # Gradient color map
        cmap = plt.colormaps["viridis"]
        total_series = sum(
            1 for k in selected_dfs for c in selected_cols if c in self.y_arrays[k]
        )
        norm = mcolors.Normalize(vmin=0, vmax=max(total_series - 1, 1))

        plotted_any = False
        i = 0
        plotted_lines: List[Any] = []
        full_labels: List[str] = []
        max_label_len = 25

        for key in selected_dfs:
            df = self.dataframes[key]
            x = self.x_arrays[key]
            for col in selected_cols:
                if col not in self.y_arrays[key]:
                    continue
                y = self.y_arrays[key][col]
                full_label = f"{key}: {col}"
                short_label = (
                    full_label if len(full_label) <= max_label_len
                    else full_label[:max_label_len - 1] + "…"
                )

                color = cmap(norm(i))
                line, = self.ax.plot(
                    x, y,
                    label=short_label,
                    color=color,
                    linewidth=1.4,
                    alpha=0.9,
                )
                plotted_lines.append(line)
                full_labels.append(full_label)
                i += 1
                plotted_any = True

        if plotted_any:
            legend = self.ax.legend(
                fontsize=7,
                frameon=False,
                labelcolor="#f0f0f0",
                handlelength=2.8,
                handletextpad=0.6,
                borderaxespad=0.2,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),  # right top
                borderpad=0.0,
            )
            for text in legend.get_texts():
                text.set_color("#333333")

            self.ax.set_xlabel("wavelength", fontsize=8)
            self.ax.set_ylabel("photometric value", fontsize=8)

            # Allocate space for the right legend
            self.figure.subplots_adjust(right=0.80)
        else:
            self.figure.subplots_adjust(right=0.98)

        # --- Hover tooltip (no mplcursors) ---
        annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
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
            an annotation with the label of the line.
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
    
    # ------------------------------------------------------------------
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

    df1 = pl.DataFrame({
        "Wavelength": [400, 500, 600, 700],
        "A": [1, 2, 3, 4],
        "B": [2, 3, 4, 5],
    })
    df2 = pl.DataFrame({
        "Wavelength": [400, 500, 600, 700],
        "B": [1, 2, 1, 3],
        "C": [3, 3, 2, 1],
    })

    widget = PolarsPlotWidget()
    widget.set_data_dict({"df1": df1, "df2": df2})
    widget.resize(1000, 600)
    widget.show()

    sys.exit(app.exec())