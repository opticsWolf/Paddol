# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Generator

# === PySide6 Imports ===
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QListWidget,
    QListWidgetItem, QMenu, QFileDialog, QLabel, QHBoxLayout,
    QComboBox, QSplitter, QTextEdit, QTableView, QMessageBox, QSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QSizePolicy, QLayout, QLayoutItem, QSpacerItem
)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QPoint

import polars as pl

# === Custom Imports ===
# Assumed to be available in the user's environment
from qt_icons import ICON_DICT
from qt_collapsible import CollapsibleSectionMixin
from csvdata import CSVData

CONFIG_FILE = Path.cwd() / "config" / "csv_importer_config.json"


class CSVPolarsModel(QAbstractTableModel):
    """QAbstractTableModel that wraps a :class:`polars.DataFrame`.

    The model exposes the DataFrame’s rows and columns to Qt view widgets such as
    ``QTableView``. Implements minimal read-only methods for performance.

    Notes:
        The implementation does not support editing or selection changes via the
        model itself; it merely forwards the underlying DataFrame to Qt’s view layer.
        For performance, the model caches column names and uses vectorised access
        where possible.

    Args:
        data (pl.DataFrame | None): Initial DataFrame to display.
            If ``None``, an empty model is created.
    """

    def __init__(self, data: pl.DataFrame | None = None):
        super().__init__()
        self._data: pl.DataFrame = data if data is not None else pl.DataFrame()

    def setDataFrame(self, dataframe: pl.DataFrame) -> None:
        """Replace the current DataFrame and reset the model.

        The method signals Qt that the underlying data has changed by calling
        :meth:`beginResetModel` before swapping in the new frame and
        :meth:`endResetModel` afterwards. All attached views will be refreshed
        to reflect the new contents.

        Args:
            dataframe (pl.DataFrame): The new DataFrame to display.
        """
        self.beginResetModel()
        self._data = dataframe
        self.endResetModel()

    def rowCount(self, index: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows in the current DataFrame."""
        if index.isValid():
            return 0
        return len(self._data)

    def columnCount(self, index: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns in the current DataFrame."""
        if index.isValid():
            return 0
        return len(self._data.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return the string representation of a cell value for display.

        This method optimizes for direct access performance.

        Args:
            index (QModelIndex): The index of the item.
            role (int): The role for which data is requested. Only
                ``Qt.ItemDataRole.DisplayRole`` is supported.

        Returns:
            Any: The string representation of the value, or None if invalid.
        """
        if role != Qt.ItemDataRole.DisplayRole or not index.isValid():
            return None

        # Guard against empty frames or OOB access
        if self._data.is_empty():
            return None

        row, col = index.row(), index.column()

        try:
            # Polars efficient access
            val = self._data[row, col]
            return str(val)
        except IndexError:
            return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return column names or row numbers for the view headers.

        Args:
            section (int): The section index (row or column number).
            orientation (Qt.Orientation): Vertical (rows) or Horizontal (columns).
            role (int): The role (DisplayRole usually).

        Returns:
            Any: The header label (column name or row index) or None.
        """
        if role != Qt.ItemDataRole.DisplayRole or self._data.is_empty():
            return None

        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self._data.columns[section])
            except IndexError:
                return None
        elif orientation == Qt.Orientation.Vertical:
            return str(section + 1)
        return None


class CSVImporterWindow(QMainWindow, CollapsibleSectionMixin):
    """Main window for importing and displaying multiple CSV files via Polars.

    This window handles:
    1.  Selection of single or multiple CSV files.
    2.  Recursive directory scanning for CSVs.
    3.  Configuration of import parameters (separators, decimals, skipping rows).
    4.  Manual or Automatic scaling of data.
    5.  Previewing raw text content vs parsed Polars DataFrames.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Multi-CSV → Polars Importer")
        self.resize(1100, 700)

        self.csv_data = CSVData()
        self.files: List[tuple[str, bool]] = []  # [(path, valid:bool)]
        self.current_file: str | None = None
        self.dataframe_dict: Dict[str, pl.DataFrame] = {}

        # Default state
        self.last_directory: str = ""

        # ===== Central Layout =====
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ===== File Selection Group =====
        file_group = QGroupBox("Loaded CSV Files")
        file_layout = QVBoxLayout(file_group)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.select_button = QPushButton(" Select CSV Files")
        self.select_button.setIcon(QIcon(ICON_DICT['open_folder']))
        self.select_button.clicked.connect(self.select_files)

        self.import_button_folder = QPushButton(" Load CSV-Files from Folders")
        icon_key = ICON_DICT.get('import_folder', 'open_folder')
        self.import_button_folder.setIcon(QIcon(ICON_DICT[icon_key]))
        self.import_button_folder.clicked.connect(self.select_files_from_folder)

        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.import_button_folder)
        file_layout.addLayout(button_layout)

        self.file_list = QListWidget()
        self.file_list.itemChanged.connect(self.on_file_checked)
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_file_context_menu)
        self.file_list.setMaximumHeight(120)
        file_layout.addWidget(self.file_list, 1)

        main_layout.addWidget(file_group)

        # ===== Import Options =====
        self.options_group = QGroupBox("Import Options")
        self.options_group.setCheckable(True)
        options_layout = QFormLayout(self.options_group)
        main_layout.addWidget(self.options_group)

        self.auto_detect_checkbox = QCheckBox("Enable Auto-Detection")
        self.auto_detect_checkbox.setChecked(True)
        self.auto_detect_checkbox.stateChanged.connect(self.toggle_auto_detection)
        options_layout.addRow(self.auto_detect_checkbox)

        sep_dec_layout = QHBoxLayout()
        self.separator_combo = QComboBox()
        self.separator_combo.addItems([",", ";", "\t", "|"])
        self.separator_combo.setEnabled(False)
        self.separator_combo.currentIndexChanged.connect(self.on_seperator_changed)

        self.decimal_combo = QComboBox()
        self.decimal_combo.addItems([".", ","])
        self.decimal_combo.setEnabled(False)
        self.decimal_combo.currentIndexChanged.connect(self.on_seperator_changed)

        sep_dec_layout.addWidget(QLabel("Separator:"), stretch=1)
        sep_dec_layout.addWidget(self.separator_combo, stretch=3)
        sep_dec_layout.addStretch(1)
        sep_dec_layout.addWidget(QLabel("Decimal:"), stretch=1)
        sep_dec_layout.addWidget(self.decimal_combo, stretch=3)
        options_layout.addRow(sep_dec_layout)

        skip_layout = QHBoxLayout()
        self.skip_initial_spin = QSpinBox()
        self.skip_initial_spin.setRange(0, 100)
        self.skip_initial_spin.setEnabled(False)
        self.skip_initial_spin.valueChanged.connect(self.import_data)

        self.skip_end_spin = QSpinBox()
        self.skip_end_spin.setRange(0, 100)
        self.skip_end_spin.setEnabled(False)
        self.skip_end_spin.valueChanged.connect(self.import_data)

        skip_layout.addWidget(QLabel("Skip Start:"), stretch=1)
        skip_layout.addWidget(self.skip_initial_spin, stretch=3)
        skip_layout.addStretch(1)
        skip_layout.addWidget(QLabel("Skip End:"), stretch=1)
        skip_layout.addWidget(self.skip_end_spin, stretch=3)
        options_layout.addRow(skip_layout)

        # ────── Checkbox to enable/disable scaling ──────
        self.manual_scale_checkbox = QCheckBox("Enable Manual Scaling")
        self.manual_scale_checkbox.stateChanged.connect(self.toggle_manual_scaling)
        options_layout.addRow(self.manual_scale_checkbox)

        # ────── Combos (inside a horizontal layout) ──────
        manual_layout = QHBoxLayout()
        self.wavelength_unit_combo = QComboBox()
        self.wavelength_unit_combo.addItems(["Å", "nm", "µm", "mm"])
        self.wavelength_unit_combo.setCurrentIndex(1)

        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["Percentage (%)", "Normalized (0–1)", "Radians (°→rad)"])
        self.data_type_combo.setCurrentIndex(1)

        self.wavelength_unit_combo.setEnabled(False)
        self.data_type_combo.setEnabled(False)

        manual_layout.addWidget(QLabel("Wavelength:"), stretch=1)
        manual_layout.addWidget(self.wavelength_unit_combo, stretch=3)
        manual_layout.addStretch(1)
        manual_layout.addWidget(QLabel("Data Type:"), stretch=1)
        manual_layout.addWidget(self.data_type_combo, stretch=3)
        options_layout.addRow(manual_layout)

        self.wavelength_unit_combo.currentIndexChanged.connect(self.on_scaling_changed)
        self.data_type_combo.currentIndexChanged.connect(self.on_scaling_changed)

        self.auto_sort_checkbox = QCheckBox("Auto Sort by Wavelength")
        self.auto_sort_checkbox.setChecked(True)
        self.auto_sort_checkbox.stateChanged.connect(self.update_auto_sorting)
        options_layout.addRow(self.auto_sort_checkbox)

        self.interp_mode_combo = QComboBox()
        self.interp_mode_combo.addItems(["Crop to overlap", "Extend unique wavelengths"])
        self.interp_mode_combo.currentIndexChanged.connect(self.on_interp_mode_change)
        options_layout.addRow(QLabel("Interpolation Mode:"), self.interp_mode_combo)

        # Enable collapsible behavior
        self.enable_collapsible(
            container=self.options_group,
            root_item=self.options_group
        )

        # ===== Import Button =====
        self.import_button_all = QPushButton(" Import Files → Polars")
        self.import_button_all.setIcon(QIcon(ICON_DICT['file_import']))
        self.import_button_all.clicked.connect(self.import_all_files)
        main_layout.addWidget(self.import_button_all)

        # ===== Splitter =====
        dataview_group = QGroupBox("Preview Data")
        dataview_layout = QVBoxLayout(dataview_group)
        dataview_layout.setContentsMargins(6, 6, 6, 6)
        dataview_layout.setSpacing(6)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(4)

        # Raw Data Preview
        raw_widget = QWidget()
        raw_layout = QVBoxLayout(raw_widget)
        raw_layout.setContentsMargins(4, 4, 4, 4)
        raw_layout.setSpacing(2)
        raw_layout.addWidget(QLabel("Raw Data Preview"))
        self.raw_display = QTextEdit()
        self.raw_display.setReadOnly(True)
        self.raw_display.setPlaceholderText("Raw preview of the selected file...")
        raw_layout.addWidget(self.raw_display)
        self.splitter.addWidget(raw_widget)

        # Table View Preview
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(4, 4, 4, 4)
        table_layout.setSpacing(2)
        table_layout.addWidget(QLabel("Parsed Table Preview"))
        self.table_view = QTableView()
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.setAlternatingRowColors(True)
        table_layout.addWidget(self.table_view)
        self.splitter.addWidget(table_widget)

        self.load_import_options()
        self.splitter.setSizes([500, 700])
        dataview_layout.addWidget(self.splitter)
        main_layout.addWidget(dataview_group, 3)

        # ===== Model =====
        self.model = CSVPolarsModel()
        self.table_view.setModel(self.model)

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
            QListWidget, QTextEdit, QTableView {
                border: 1px solid #AAAAAA;
            }
        """)

    # ====================================================================
    # UTILITIES
    # ====================================================================
    @contextmanager
    def bulk_signal_blocker(self, widgets: List[QWidget]) -> Generator[None, None, None]:
        """Context manager to temporarily block signals for a list of widgets.

        This is useful during bulk updates (e.g., loading config) to prevent
        triggering redundant event cascades. It ensures signals are restored
        to their previous state even if an error occurs.

        Args:
            widgets (List[QWidget]): List of widgets to block signals for.

        Yields:
            None: Control returns to the caller with signals blocked.
        """
        states = [w.signalsBlocked() for w in widgets]
        for w in widgets:
            w.blockSignals(True)
        try:
            yield
        finally:
            for w, state in zip(widgets, states):
                w.blockSignals(state)

    def load_import_options(self) -> None:
        """Load import options from config file using JSON.

        Suppresses signals via :meth:`bulk_signal_blocker` while updating widgets
        to prevent event cascades. If the config file doesn't exist or is invalid,
        default values will be used automatically.
        """
        if not CONFIG_FILE.exists():
            return

        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                if not isinstance(config, dict):
                    raise ValueError("Config file must be a JSON object")

            widgets_to_block = [
                self.auto_detect_checkbox, self.separator_combo, self.decimal_combo,
                self.skip_initial_spin, self.skip_end_spin, self.manual_scale_checkbox,
                self.wavelength_unit_combo, self.data_type_combo,
                self.auto_sort_checkbox, self.interp_mode_combo
            ]

            with self.bulk_signal_blocker(widgets_to_block):
                self.auto_detect_checkbox.setChecked(config.get('auto_detect', True))
                self.separator_combo.setCurrentText(str(config.get('separator', ",")))
                self.decimal_combo.setCurrentText(str(config.get('decimal', ".")))
                self.skip_initial_spin.setValue(int(config.get('skip_start', 0)))
                self.skip_end_spin.setValue(int(config.get('skip_end', 0)))

                manual_scale = config.get('manual_scale', False)
                self.manual_scale_checkbox.setChecked(manual_scale)
                self.wavelength_unit_combo.setEnabled(manual_scale)
                self.data_type_combo.setEnabled(manual_scale)

                self.wavelength_unit_combo.setCurrentText(str(config.get('wavelength_unit', "nm")))
                self.data_type_combo.setCurrentText(str(config.get('data_type', "Normalized (0–1)")))
                self.auto_sort_checkbox.setChecked(config.get('auto_sort', True))
                self.interp_mode_combo.setCurrentText(str(config.get('interp_mode', "Crop to overlap")))

                expanded = bool(config.get("import_options_expanded", True))
                self.options_group.setChecked(expanded)
                self.last_directory = str(config.get('last_directory', ""))

        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"Warning: Could not load config - using defaults. Error: {e}")

    def save_import_options(self) -> None:
        """Save current import options and last directory to config file.

        Creates parent directories if they don't exist. This is a non-critical
        operation; failures will be logged but will not raise exceptions.
        """
        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

            config = {
                "auto_detect": self.auto_detect_checkbox.isChecked(),
                "separator": self.separator_combo.currentText(),
                "decimal": self.decimal_combo.currentText(),
                "skip_start": self.skip_initial_spin.value(),
                "skip_end": self.skip_end_spin.value(),
                "manual_scale": self.manual_scale_checkbox.isChecked(),
                "wavelength_unit": self.wavelength_unit_combo.currentText(),
                "data_type": self.data_type_combo.currentText(),
                "auto_sort": self.auto_sort_checkbox.isChecked(),
                "interp_mode": self.interp_mode_combo.currentText(),
                "import_options_expanded": self.options_group.isChecked(),
                "last_directory": self.last_directory
            }

            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)

        except OSError as e:
            print(f"Warning: Could not save config. Error: {e}")

    def closeEvent(self, event) -> None:
        """Save current options when dialog is closed."""
        self.save_import_options()
        super().closeEvent(event)

    # ====================================================================
    # FILE HANDLING
    # ====================================================================
    def select_files(self) -> None:
        """Allow user to select multiple CSV files and preview the first one.

        Uses the saved last directory for the file dialog if it exists, otherwise
        defaults to the home directory. Only adds new files that aren't already
        in the list. Auto-selects and previews the first new file added.
        """
        start_dir = self.last_directory if (self.last_directory and Path(self.last_directory).exists()) else str(Path.home())

        files, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV Files", start_dir, "CSV Files (*.csv);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if not files:
            return

        self._process_new_files([Path(f) for f in files])

    def select_files_from_folder(self) -> None:
        """Recursively scan a selected folder for CSV files.

        Opens a directory selection dialog. If a directory is chosen, it recursively
        searches for ``.csv`` files and adds non-duplicates to the list.
        """
        start_dir = self.last_directory if (self.last_directory and Path(self.last_directory).exists()) else str(Path.home())

        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if not dir_path:
            return

        root = Path(dir_path)
        # Use generator to avoid holding massive list if unnecessary
        new_files = [p for p in root.rglob('*.csv') if p.is_file()]
        
        self.last_directory = str(root)
        self._process_new_files(new_files)

    def _process_new_files(self, paths: List[Path]) -> None:
        """Helper to add new files to the list widget and update state.

        Args:
            paths (List[Path]): List of file paths to add.
        """
        first_new_item = None
        current_paths = {f[0] for f in self.files}

        for path in paths:
            str_path = str(path)
            if str_path in current_paths:
                continue

            self.files.append((str_path, True))
            item = QListWidgetItem(str_path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Checked)
            self.file_list.addItem(item)
            current_paths.add(str_path)

            if first_new_item is None:
                first_new_item = item

        if not self.files:
            return

        self.skip_initial_spin.setEnabled(True)
        self.skip_end_spin.setEnabled(True)

        if first_new_item:
            self.file_list.setCurrentItem(first_new_item)
            self.preview_and_display_file(first_new_item.text())

        if paths:
             # Update last dir to the parent of the first new file
            self.last_directory = str(paths[0].parent)

    def on_file_selected(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        """Triggered when user selects a file in the list."""
        if not current:
            return
        if current.checkState() == Qt.CheckState.Checked:
            self.preview_and_display_file(current.text())

    def on_file_checked(self, item: QListWidgetItem) -> None:
        """Triggered when user checks/unchecks a file.

        If a file is unchecked while currently displayed, the preview is cleared.
        """
        file_path = item.text()
        if item.checkState() == Qt.CheckState.Checked:
            self.preview_and_display_file(file_path)
        else:
            if self.current_file == file_path:
                self.raw_display.clear()
                self.model.setDataFrame(pl.DataFrame())
                self.current_file = None

    def preview_and_display_file(self, file_path: str) -> None:
        """Load and display both raw text and DataFrame preview.

        1. Loads raw CSV content via :class:`CSVData`.
        2. Auto-detects separators if the option is enabled.
        3. Parses the content into a Polars DataFrame and updates the table model.

        Args:
            file_path (str): Path to the CSV file.
        """
        self.current_file = file_path
        try:
            self.csv_data.interp_mode = "extend" if self.interp_mode_combo.currentIndex() == 1 else "crop"
            self.csv_data.load_csv(file_path)
            self.raw_display.setText(self.csv_data.raw_content)

            if self.auto_detect_checkbox.isChecked():
                self.detect_separators_and_decimal()

            # Attempt to parse and preview as DataFrame
            df = self.csv_data.import_data()
            self.model.setDataFrame(df)

        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))
            self._uncheck_file(file_path)
            self.model.setDataFrame(pl.DataFrame())
            self.raw_display.clear()

    def _uncheck_file(self, file_path: str) -> None:
        """Helper to uncheck a file in the list widget."""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text() == file_path:
                item.setCheckState(Qt.CheckState.Unchecked)
                break

    # ====================================================================
    # RIGHT CLICK MENU
    # ====================================================================
    def show_file_context_menu(self, pos: QPoint) -> None:
        """Display a context menu that allows removal of selected or all files.

        The menu includes:
            - "Remove File": Removes the file at the clicked position.
            - "Remove All Files": Removes all files and clears the preview.

        Args:
            pos (QPoint): Position relative to ``file_list`` where the menu is shown.
        """
        item = self.file_list.itemAt(pos)
        has_items = self.file_list.count() > 0

        menu = QMenu(self)
        remove_action = None

        if item:
            remove_action = QAction(QIcon(ICON_DICT['delete_file']), "Remove File", self)
            menu.addAction(remove_action)

        if has_items:
            remove_all_action = QAction(
                QIcon(ICON_DICT.get('delete_all', ICON_DICT['delete_file'])),
                "Remove All Files", self
            )
            menu.addAction(remove_all_action)

        action = menu.exec(self.file_list.mapToGlobal(pos))

        if not action:
            return

        if item and action == remove_action:
            self._remove_file(item.text())
        elif has_items and action == remove_all_action:
            self._remove_all_files()

    def _remove_file(self, file_path: str) -> None:
        """Remove a specific file from the list and update UI state."""
        self.files = [f for f in self.files if f[0] != file_path]

        for i in range(self.file_list.count()):
            if self.file_list.item(i).text() == file_path:
                self.file_list.takeItem(i)
                break

        if self.current_file == file_path:
            self.raw_display.clear()
            self.model.setDataFrame(pl.DataFrame())
            self.current_file = None

    def _remove_all_files(self) -> None:
        """Remove all files from the list."""
        self.files.clear()
        self.file_list.clear()
        self.raw_display.clear()
        self.model.setDataFrame(pl.DataFrame())
        self.current_file = None

    # ====================================================================
    # SETTINGS AND IMPORT LOGIC
    # ====================================================================
    def toggle_auto_detection(self, state: int) -> None:
        """Enable or disable manual separator/decimal controls.

        When checked, the UI controls are disabled and separators are auto‑detected
        (if a file is loaded). When unchecked, the controls are enabled and
        reset to defaults or user values.

        Args:
            state (int): Qt.CheckState value.
        """
        is_checked = (state == Qt.CheckState.Checked.value)
        self.separator_combo.setEnabled(not is_checked)
        self.decimal_combo.setEnabled(not is_checked)

        if is_checked and self.current_file:
            self.detect_separators_and_decimal()
        elif not is_checked:
            # Trigger update via index change 0
            self.on_seperator_changed(0)

    def on_seperator_changed(self, index: int) -> None:
        """Apply a new separator/decimal configuration when the user changes it.

        The update occurs only if manual scaling is off and a file has been
        loaded. ``index`` is unused but kept for signal compatibility.

        Args:
            index (int): Index of the selected item in ``separator_combo``.
        """
        if not self.manual_scale_checkbox.isChecked() and self.current_file:
            self.csv_data.separator = self.separator_combo.currentText()
            self.csv_data.decimal_sep = self.decimal_combo.currentText()
            self.preview_and_display_file(self.current_file)

    def detect_separators_and_decimal(self) -> None:
        """Auto‑detect the field separator and decimal sign from the loaded CSV.

        The detected values are applied to :attr:`csv_data` and, if present,
        selected in the corresponding combo boxes. No action is taken when
        no content has been loaded.
        """
        if not self.csv_data.raw_content:
            return

        detected_sep = self.csv_data.detect_separator(self.csv_data.raw_content)
        detected_decimal = self.csv_data.detect_decimal_sign(self.csv_data.raw_content)

        # Update Combos without triggering redundant signals if possible,
        # but here we want the data object to update, so standard set is fine.
        if (idx := self.separator_combo.findText(detected_sep)) != -1:
            self.separator_combo.setCurrentIndex(idx)
        
        if (idx := self.decimal_combo.findText(detected_decimal)) != -1:
            self.decimal_combo.setCurrentIndex(idx)

        self.csv_data.separator = detected_sep
        self.csv_data.decimal_sep = detected_decimal

    def update_auto_sorting(self, state: int) -> None:
        """Enable or disable automatic sorting of the CSV data.

        Sets :attr:`csv_data.auto_sorting` based on ``state``. The preview is
        refreshed afterwards if a file is loaded.

        Args:
            state (int): Qt.CheckState value.
        """
        if self.current_file:
            self.csv_data.auto_sorting = (state == Qt.CheckState.Checked.value)
            self.preview_and_display_file(self.current_file)

    def toggle_manual_scaling(self, state: int) -> None:
        """Switch between automatic and manual scaling for the data.

        * Checked: Enable unit/type selectors and set :attr:`csv_data.manual_scaling` to ``True``.
        * Unchecked: Disable the selectors and set the flag to ``False``.

        The preview is refreshed if a file is currently loaded.

        Args:
            state (int): Qt.CheckState value.
        """
        is_checked = (state == Qt.CheckState.Checked.value)
        self.wavelength_unit_combo.setEnabled(is_checked)
        self.data_type_combo.setEnabled(is_checked)
        self.csv_data.manual_scaling = is_checked

        if self.current_file:
            self.preview_and_display_file(self.current_file)

    def on_scaling_changed(self, index: int) -> None:
        """Apply a new unit/data‑type combination when manual scaling is active.

        Connected to the *unit* and *data type* combo boxes; updates
        :attr:`csv_data.wavelength_unit` and :attr:`csv_data.data_type_unit`
        only if ``manual_scale_checkbox`` is checked.

        Args:
            index (int): Index of the selected item in the combo box.
        """
        if self.manual_scale_checkbox.isChecked():
            self.csv_data.wavelength_unit = self.wavelength_unit_combo.currentText()
            self.csv_data.data_type_unit = self.data_type_combo.currentText()
            if self.current_file:
                self.preview_and_display_file(self.current_file)

    def on_interp_mode_change(self, index: int) -> None:
        """Re‑render the preview when the interpolation mode changes.

        Triggered by the interpolation mode combo box; simply refreshes the
        current file’s display to use the new setting.

        Args:
            index (int): Index of the chosen interpolation mode.
        """
        if self.current_file:
            self.preview_and_display_file(self.current_file)

    def _apply_csv_settings(self) -> None:
        """Helper to sync UI settings to the csv_data object."""
        self.csv_data.separator = self.separator_combo.currentText()
        self.csv_data.decimal_sep = self.decimal_combo.currentText()
        self.csv_data.skip_initial_rows = self.skip_initial_spin.value()
        self.csv_data.skip_end_rows = self.skip_end_spin.value()
        self.csv_data.interp_mode = (
            "extend" if self.interp_mode_combo.currentIndex() == 1 else "crop"
        )

    def import_data(self) -> None:
        """Import all user‑selected CSV files and display the last successfully read file.

        The method reads configuration options from the UI (separator, decimal sign,
        row‑skipping, interpolation mode), then iterates over the list widget items.
        Each checked file is loaded via :meth:`csv_data.load_csv` and converted to a
        Polars DataFrame by :meth:`csv_data.import_data`.

        Files that raise an exception are un‑checked and trigger a warning dialog.
        """
        checked_items = [
            self.file_list.item(i) for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.CheckState.Checked
        ]

        if not checked_items:
            QMessageBox.warning(self, "No Files Selected", "Please check at least one file.")
            return

        self._apply_csv_settings()
        valid_dfs: List[pl.DataFrame] = []

        for item in checked_items:
            path = item.text()
            try:
                self.csv_data.load_csv(path)
                df = self.csv_data.import_data()
                valid_dfs.append(df)
            except Exception as e:
                item.setCheckState(Qt.CheckState.Unchecked)
                QMessageBox.warning(self, "Import Failed", f"{path}\n{e}")

        if not valid_dfs:
            QMessageBox.warning(self, "No Valid Files", "All selected files failed to load.")
            return

        # Show last successfully processed file
        self.model.setDataFrame(valid_dfs[-1])
        self.raw_display.setText(self.csv_data.raw_content)

    def import_all_files(self) -> None:
        """Load every checked CSV file into ``self.dataframe_dict``.

        For each selected file the method:
        1.  Loads the raw content via :meth:`csv_data.load_csv`.
        2.  If *Auto‑Detect* is enabled, updates separator/decimal settings.
        3.  Converts the CSV into a Polars DataFrame with :meth:`csv_data.import_data`
            and stores it in ``self.dataframe_dict``, keyed by the file’s base name.

        Files that raise an exception are un‑checked and trigger a warning dialog.
        After processing, a message box reports the success count.
        """
        self.dataframe_dict = {}
        self._apply_csv_settings()
        
        count = self.file_list.count()
        success_count = 0
        
        for i in range(count):
            item = self.file_list.item(i)
            if item.checkState() != Qt.CheckState.Checked:
                continue

            # Keep UI responsive during batch import
            QApplication.processEvents()

            file_path = item.text()
            base_name = Path(file_path).name

            try:
                self.csv_data.load_csv(file_path)
                if self.auto_detect_checkbox.isChecked():
                    self.detect_separators_and_decimal()
                
                df = self.csv_data.import_data()
                self.dataframe_dict[base_name] = df
                success_count += 1
            except Exception as e:
                QMessageBox.warning(self, "Import Warning", f"Failed to import {base_name}:\n{e}")
                item.setCheckState(Qt.CheckState.Unchecked)

        if not self.dataframe_dict:
            QMessageBox.information(self, "No Data", "No valid CSV files imported.")
        else:
            QMessageBox.information(self, "Import Complete", f"Imported {success_count} files successfully.")


if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = CSVImporterWindow()
    window.show()

    sys.exit(app.exec())