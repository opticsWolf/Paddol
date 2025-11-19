# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QListWidget, 
    QListWidgetItem, QMenu, QAction, QFileDialog, QLabel, QHBoxLayout, 
    QComboBox, QSplitter, QTextEdit, QTableView, QMessageBox, QSpinBox, 
    QCheckBox, QGroupBox, QFormLayout, QSizePolicy
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QAbstractTableModel

import polars as pl

from typing import Dict, Union

from qt_icons import ICON_DICT, scan_icons_folder
from csvdata import CSVData


#ICON_DICT = scan_icons_folder(Path("src/icons"))
CONFIG_FILE = str(Path.cwd() / "config" / "csv_importer_config.json")
print (CONFIG_FILE)

class PolarsModel(QAbstractTableModel):
    """QAbstractTableModel that wraps a :class:`polars.DataFrame`.

    The model exposes the DataFrame’s rows and columns to Qt view widgets such as
    ``QTableView``.  Only the minimal set of read‑only methods is implemented:
    :meth:`rowCount`, :meth:`columnCount`, :meth:`data` and
    :meth:`headerData`.

    Args:
        data (polars.DataFrame, optional): Initial DataFrame to display.
            If ``None`` an empty model is created.

    Notes:
        The implementation does not support editing or selection changes; it
        merely forwards the underlying DataFrame to Qt’s view layer.  For
        performance, the model caches column names and uses vectorised access
        where possible.
    """
    def __init__(self, data=None):
        super().__init__()
        self._data = None
        if data is not None:
            self.setDataFrame(data)
    
    def setDataFrame(self, dataframe):
        """Replace the current DataFrame and reset the model.

        The method signals Qt that the underlying data has changed by calling
        :meth:`beginResetModel` before swapping in the new frame and
        :meth:`endResetModel` afterwards.  All attached views will be refreshed
        to reflect the new contents.

        Args:
            dataframe (polars.DataFrame): The new DataFrame to display.
        """
        self.beginResetModel()
        self._data = dataframe
        self.endResetModel()

    def rowCount(self, index):
        """Return the number of rows in the current DataFrame."""
        return len(self._data) if self._data is not None else 0

    def columnCount(self, index):
        """Return the number of columns in the current DataFrame."""
        return len(self._data.columns) if self._data is not None else 0

    # ----------------------------------------------------------------------
    # Data accessors
    # ----------------------------------------------------------------------
    def data(self, index, role=Qt.DisplayRole):
        """Return the string representation of a cell value for display."""
        if not index.isValid() or self._data is None or role != Qt.DisplayRole:
            return None
        try:
            # Direct access for better performance
            value = self._data[index.row(), index.column()]
            return str(value)
        except Exception:
            return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Return column names or row numbers for the view headers."""
        if role != Qt.DisplayRole or self._data is None:
            return None
        if orientation == Qt.Horizontal:
            return str(self._data.columns[section])
        elif orientation == Qt.Vertical:
            return str(section + 1)
        return None

class CSVImporterWindow(QMainWindow):
    """Main window for importing and displaying multiple CSV files."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-CSV → Polars Importer")
        self.resize(1100, 700)

        self.csv_data = CSVData()
        self.files = []  # [(path, valid:bool)]
        self.current_file = None
        self.interp_mode = "crop"

        self.last_directory = ""

        # ===== Central Layout =====
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ===== File Selection Group =====
        file_group = QGroupBox("Loaded CSV Files")
        file_layout = QVBoxLayout(file_group)
        
        # Create horizontal layout for the two buttons with proper spacing
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)  # Add some space between buttons
        
        self.select_button = QPushButton(" Select CSV Files")
        self.select_button.setIcon(QIcon(ICON_DICT['open_folder']))
        self.select_button.clicked.connect(self.select_files)
        
        self.import_button = QPushButton(" Load CSV-Files from Folders")
        # Use same icon as folder open if 'import_folder' not defined
        icon_key = ICON_DICT.get('import_folder', 'open_folder')
        self.import_button.setIcon(QIcon(ICON_DICT[icon_key]))
        self.import_button.clicked.connect(self.select_files_from_folder)
        
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.import_button)
        file_layout.addLayout(button_layout)  # Add the horizontal layout to VBox
        
        # File list with checkboxes (unchanged from original)
        self.file_list = QListWidget()
        self.file_list.itemChanged.connect(self.on_file_checked)
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_file_context_menu)
        self.file_list.setMaximumHeight(120)
        file_layout.addWidget(self.file_list, 1)
        
        main_layout.addWidget(file_group)

        # ===== Import Options =====
        options_group = QGroupBox("Import Options")
        options_group.setCheckable(True)
        options_layout = QFormLayout(options_group)
        main_layout.addWidget(options_group)

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
        self.data_type_combo.addItems(["Percentage (%)", "Normalized (0 – 1)", "Radians (°→rad)"])
        self.data_type_combo.setCurrentIndex(1)
        # Initially disabled – they will be enabled when the checkbox is checked
        self.wavelength_unit_combo.setEnabled(False)
        self.data_type_combo.setEnabled(False)

        manual_layout.addWidget(QLabel("Wavelength:"), stretch=1)
        manual_layout.addWidget(self.wavelength_unit_combo, stretch=3)
        manual_layout.addStretch(1)
        manual_layout.addWidget(QLabel("Data Type:"), stretch=1)
        manual_layout.addWidget(self.data_type_combo, stretch=3)
        options_layout.addRow(manual_layout)

        # ────── Connect the two combo boxes to one slot ──────
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

        # ────── Build a flat list of all child widgets that should hide/show together ──────        
        self._import_option_items = []
        
        for i in range(options_layout.count()):
            item = options_layout.itemAt(i)
        
            if item.widget():
                self._import_option_items.append(item.widget())
            elif item.layout():
                # Collect all widgets inside nested layouts
                for k in range(item.layout().count()):
                    sub_item = item.layout().itemAt(k)
                    if sub_item.widget():
                        self._import_option_items.append(sub_item.widget())
        
        def toggle_import_options_visibility(checked):
            """Show or hide all child widgets when the QGroupBox is toggled."""
            if not hasattr(toggle_import_options_visibility, 'original'):
                toggle_import_options_visibility.original = {
                    "height": options_group.height(),
                    "margins": options_layout.contentsMargins()
                }
        
            # Toggle visibility of all collected widgets
            for w in self._import_option_items:
                w.setVisible(checked)
        
            if checked:
                # Restore original margins and height
                m = toggle_import_options_visibility.original["margins"]
                options_layout.setContentsMargins(m.left(), m.top(), m.right(), m.bottom())
                options_group.setMinimumHeight(toggle_import_options_visibility.original["height"])
            else:
                # Collapse fully
                options_layout.setContentsMargins(0, 0, 0, 0)
                options_group.setMinimumHeight(0)
        
        # Connect the group box's state change to the visibility toggle function
        options_group.toggled.connect(toggle_import_options_visibility)
        
        self.options_group = options_group
        
        # ===== Import Button =====
        self.import_button = QPushButton(" Import Files → Polars")
        self.import_button.setIcon(QIcon(ICON_DICT['file_import']))
        self.import_button.clicked.connect(self.import_all_files)
        main_layout.addWidget(self.import_button)

        # ===== Splitter =====
        dataview_group = QGroupBox("Preview Data")
        dataview_layout = QVBoxLayout(dataview_group)
        dataview_layout.setContentsMargins(6, 6, 6, 6)
        dataview_layout.setSpacing(6)
        
        # Horizontal splitter for side-by-side views
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(4)
        #self.splitter.setStyleSheet("""
        #    QSplitter { background-color: white; }
        #    QSplitter::handle { background-color: #f1f3f5; }
        #""")
        
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
    
        # Load import options        
        self.load_import_options()
        
        # Optional: set initial sizes
        self.splitter.setSizes([500, 700])
        
        # Add splitter to group layout
        dataview_layout.addWidget(self.splitter)
        
        # Add group to main layout
        main_layout.addWidget(dataview_group, 3)

        # ===== Model =====
        self.model = PolarsModel()
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
            QListWidget{
                border: 1px solid #AAAAAA;
            }
            QTextEdit {
                border: 1px solid #AAAAAA;
            }
            QTableView {
                border: 1px solid #AAAAAA;
            }
        """)


    # ====================================================================
    def load_import_options(self):
        """Load import options from config file if it exists using JSON.

        Suppresses signals while updating widgets to prevent event cascades.
        If the config file doesn't exist or is invalid, default values will be used.
        """
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                if not isinstance(config, dict):
                    raise ValueError("Config file must be a JSON object")

            # List of widgets that need signal blocking
            widget_list = [
                self.auto_detect_checkbox,
                self.separator_combo,
                self.decimal_combo,
                self.skip_initial_spin,
                self.skip_end_spin,
                self.manual_scale_checkbox,
                self.wavelength_unit_combo,
                self.data_type_combo,
                self.auto_sort_checkbox,
                self.interp_mode_combo
            ]

            # Block signals for all specified widgets
            signal_states = [widget.signalsBlocked() for widget in widget_list]
            for widget in widget_list:
                widget.blockSignals(True)

            try:
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
                self.data_type_combo.setCurrentText(str(config.get('data_type',
                                                                   "Normalized (0 – 1)")))
                self.auto_sort_checkbox.setChecked(config.get('auto_sort', True))
                self.interp_mode_combo.setCurrentText(str(config.get('interp_mode',
                                                                    "Crop to overlap")))

                expanded = bool(config.get("import_options_expanded", True))
                self.options_group.setChecked(expanded)

                # last_directory doesn't trigger signals
                self.last_directory = str(config.get('last_directory', ""))
            finally:
                # Restore original signal blocking states
                for i, widget in enumerate(widget_list):
                    widget.blockSignals(signal_states[i])

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not load config - using defaults. Error: {e}")
            pass  # Default values are used automatically due to .get() defaults

    def save_import_options(self):
        """Save current import options and last directory to config file using JSON.

        Creates parent directories if they don't exist. Non-critical operation -
        failures will be logged but not raise exceptions.
        """
        try:
            # Create parent directories for the config file
            Path(CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)

            config = {
                "auto_detect": self.auto_detect_checkbox.isChecked(),
                "separator": str(self.separator_combo.currentText()),
                "decimal": str(self.decimal_combo.currentText()),
                "skip_start": int(self.skip_initial_spin.value()),
                "skip_end": int(self.skip_end_spin.value()),
                "manual_scale": self.manual_scale_checkbox.isChecked(),
                "wavelength_unit": str(self.wavelength_unit_combo.currentText()),
                "data_type": str(self.data_type_combo.currentText()),
                "auto_sort": self.auto_sort_checkbox.isChecked(),
                "interp_mode": str(self.interp_mode_combo.currentText()),
                "import_options_expanded": bool(self.options_group.isChecked()),
                "last_directory": str(self.last_directory)
            }

            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)  # Indent for human-readable output
        except (IOError, OSError) as e:
            print(f"Warning: Could not save config. Error: {e}")
            pass  # Silently fail - this is non-critical functionality

    def closeEvent(self, event):
        """Save current options when dialog is closed."""
        self.save_import_options()
        super().closeEvent(event)


    # ====================================================================
    # FILE HANDLING
    # ====================================================================  
    def select_files(self):
        """Allow user to select multiple CSV files and preview the first one.
    
        Uses saved last directory for the file dialog if it exists, otherwise defaults to home directory.
        Only adds new files that aren't already in the list.
        Auto-selects and previews the first new file added.
        """
        # Determine starting directory
        start_dir = ""
        try:
            if self.last_directory and Path(self.last_directory).exists():
                start_dir = str(Path(self.last_directory))
            else:
                start_dir = str(Path.home())  # Fall back to home directory
        except Exception:
            start_dir = str(Path.home())
    
        options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
    
        if not files:
            return
        
        first_new_item = None
        for path in files:
            # Skip duplicate paths (case-sensitive)
            if any(path == f[0] for f in self.files):
                continue
    
            # Add new file to the list and UI
            self.files.append((path, True))
            item = QListWidgetItem(path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked)
            self.file_list.addItem(item)
    
            if first_new_item is None:
                first_new_item = item
    
        self.skip_initial_spin.setEnabled(True)
        self.skip_end_spin.setEnabled(True)
    
        # Auto-select and preview the first new file
        if first_new_item:
            self.file_list.setCurrentItem(first_new_item)
            self.preview_and_display_file(first_new_item.text())
    
        # Update last_directory to the directory of the first selected file
        if files:
            self.last_directory = str(Path(files[0]).parent)

    def select_files_from_folder(self, file_extension='.csv'):
        """Selects a folder, recursively scans for files with the given extension,
        and adds non-duplicate paths to self.files and the UI list widget.
    
        Args:
            file_extension (str): File extension including dot (e.g., '.csv').
        """
        file_extension='.csv'
        # Determine starting directory
        try:
            if getattr(self, 'last_directory', None) and Path(self.last_directory).exists():
                start_dir = str(Path(self.last_directory))
            else:
                start_dir = str(Path.home())
        except Exception:
            start_dir = str(Path.home())
    
        # Folder selection
        dir_path = QFileDialog.getExistingDirectory(
            self,
            f"Select Folder to Import {file_extension[1:].upper()} Files",
            start_dir
        )
        if not dir_path:
            return
    
        root = Path(dir_path)
        ext_lower = file_extension.lower()
        new_files_to_add = []
    
        # Pure pathlib recursive scan inlined here
        try:
            for path in root.rglob(f'*{ext_lower}'):
                if path.is_file():
                    # Case-sensitive duplicate check
                    if not any(path == p[0] for p in self.files):
                        new_files_to_add.append(str(path))
        except OSError:
            pass  # Skip inaccessible directories silently
    
        # Add new files to data model and UI
        first_new_item = None
        for filepath in new_files_to_add:
            self.files.append((filepath, True))
    
            item = QListWidgetItem(filepath)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked)
            self.file_list.addItem(item)
    
            if first_new_item is None:
                first_new_item = item
    
        # Enable skip controls
        self.skip_initial_spin.setEnabled(True)
        self.skip_end_spin.setEnabled(True)
    
        # Auto-select first new file
        if first_new_item:
            self.file_list.setCurrentItem(first_new_item)
            self.preview_and_display_file(first_new_item.text())
    
        # Persist last used directory
        self.last_directory = str(root)

    
    def on_file_selected(self, current, previous):
        """Triggered when user selects a file in the list."""
        if not current:
            return
        file_path = current.text()
        if current.checkState() == Qt.Checked:
            self.preview_and_display_file(file_path)
    
    def on_file_checked(self, item):
        """Triggered when user checks/unchecks a file."""
        file_path = item.text()
        if item.checkState() == Qt.Checked:
            self.preview_and_display_file(file_path)
        else:
            if self.current_file == file_path:
                self.raw_display.clear()
                self.model.setDataFrame(pl.DataFrame())
     
    def preview_file(self, file_path: str):
        """Load and display the raw CSV preview for the given file."""
        self.current_file = file_path
        try:
            self.csv_data.load_csv(file_path)
            self.raw_display.setText(self.csv_data.raw_content)
            if self.auto_detect_checkbox.isChecked():
                self.detect_separators_and_decimal()
        except Exception as e:
            QMessageBox.critical(self, "File Error", str(e))
            # Deselect invalid file
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.text() == file_path:
                    item.setCheckState(Qt.Unchecked)
                    break
                
    def preview_and_display_file(self, file_path: str):
        """Load and display both raw text and DataFrame preview."""
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
            # Uncheck invalid file
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.text() == file_path:
                    item.setCheckState(Qt.Unchecked)
                    break
            self.model.setDataFrame(pl.DataFrame())
            self.raw_display.clear()
        
    # ====================================================================
    # RIGHT CLICK MENU
    # ====================================================================
    def _remove_file(self, file_path):
        """Remove a specific file from the list and update UI state."""
        self.files = [(p, checked) for p, checked in self.files if p != file_path]
        
        # Find and remove the item from QListWidget
        for i in range(self.file_list.count()):
            if self.file_list.item(i).text() == file_path:
                self.file_list.takeItem(i)
                break
        
        # Clear preview if this file was shown
        if self.current_file == file_path:
            self.raw_display.clear()
            self.model.setDataFrame(pl.DataFrame())
            self.current_file = None
    
    def _remove_all_files(self):
        """Remove all files from the list and update UI state."""
        self.files = []
        self.file_list.clear()
        
        if self.current_file is not None:
            self.raw_display.clear()
            self.model.setDataFrame(pl.DataFrame())
            self.current_file = None
    
    def show_file_context_menu(self, pos):
        """Display a context menu that allows removal of selected or all files.
    
        The menu includes:
            - "Remove File": Removes the file at the clicked position.
            - "Remove All Files" (if there are any files): Removes all files and clears the preview.
    
        Parameters:
            pos (QPoint): Position relative to ``file_list`` where the menu is shown.
        """
        item = self.file_list.itemAt(pos)
        has_items = bool(self.file_list.count())
    
        # ---- Build the menu -------------------------------------------------
        menu = QMenu(self)
    
        if item:
            remove_action = QAction(QIcon(ICON_DICT['delete_file']), "Remove File", self)
            menu.addAction(remove_action)
    
        if has_items:
            remove_all_action = QAction(
                QIcon(ICON_DICT.get('delete_all', ICON_DICT['delete_file'])),
                "Remove All Files",
                self
            )
            menu.addAction(remove_all_action)
    
        # ---- Execute the menu -----------------------------------------------
        action = menu.exec_(self.file_list.mapToGlobal(pos))
        
        if not action:
            return
    
        if item and action == remove_action:
            file_path = item.text()
            self._remove_file(file_path)
        
        elif has_items and action == menu.actions()[-1]:  # The last action added is "Remove All Files"
            self._remove_all_files()

    # ====================================================================
    # SETTINGS AND IMPORT LOGIC
    # ====================================================================
    def toggle_auto_detection(self, state):
        """Enable or disable manual separator/decimal controls.

        When checked the UI controls are disabled and separators are auto‑detected
        (if a file is loaded).  When unchecked the controls are enabled and
        reset to defaults.

        Args:
            state (int): Qt.CheckState value.
        """
        self.separator_combo.setEnabled(state == Qt.Unchecked)
        self.decimal_combo.setEnabled(state == Qt.Unchecked)
        
        if state == Qt.Checked and self.current_file:
            self.detect_separators_and_decimal()
        elif state == Qt.Unchecked:
            self.on_seperator_changed(0)

    def on_seperator_changed(self, index):
        """Apply a new separator/decimal configuration when the user changes it.

        The update occurs only if manual scaling is off and a file has been
        loaded.  *index* is unused but kept for signal compatibility.

        Args:
            index (int): Index of the selected item in ``separator_combo``.
        """
        if self.manual_scale_checkbox.isChecked() is not True and self.current_file:
            self.csv_data.separator = self.separator_combo.currentText()
            self.csv_data.decimal_sep = self.decimal_combo.currentText()
            self.preview_and_display_file(self.current_file)
            
    def detect_separators_and_decimal(self):
        """Auto‑detect the field separator and decimal sign from the loaded CSV.

        The detected values are applied to :attr:`csv_data` and, if present,
        selected in the corresponding combo boxes.  No action is taken when
        no content has been loaded.
        """
        if not self.csv_data.raw_content:
            return
        detected_sep = self.csv_data.detect_separator(self.csv_data.raw_content)
        detected_decimal = self.csv_data.detect_decimal_sign(self.csv_data.raw_content)
        idx_sep = self.separator_combo.findText(detected_sep)
        if idx_sep != -1:
            self.separator_combo.setCurrentIndex(idx_sep)
        idx_dec = self.decimal_combo.findText(detected_decimal)
        if idx_dec != -1:
            self.decimal_combo.setCurrentIndex(idx_dec)
        self.csv_data.separator = detected_sep
        self.csv_data.decimal_sep = detected_decimal

    def update_auto_sorting(self, state):
        """Enable or disable automatic sorting of the CSV data.

        Sets :attr:`csv_data.auto_sorting` based on *state* (``Qt.Checked`` → ``True``).
        The preview is refreshed afterwards if a file is loaded.

        Args:
            state (int): Qt.CheckState value.
        """
        if self.current_file:
            self.csv_data.auto_sorting = (state == Qt.Checked)
            self.preview_and_display_file(self.current_file)

    def toggle_manual_scaling(self, state):
        """Switch between automatic and manual scaling for the data.

        * Checked → enable unit/type selectors and set
          :attr:`csv_data.manual_scaling` to ``True``.
        * Unchecked → disable the selectors and set the flag to ``False``.

        The preview is refreshed if a file is currently loaded.

        Args:
            state (int): Qt.CheckState value.
        """
        self.wavelength_unit_combo.setEnabled(state == Qt.Checked)
        self.data_type_combo.setEnabled(state == Qt.Checked)
        self.csv_data.manual_scaling = (state == Qt.Checked)
        if self.current_file:
            self.preview_and_display_file(self.current_file)

    def on_scaling_changed(self, index):
        """Apply a new unit / data‑type combination when manual scaling is active.

        Connected to the *unit* and *data type* combo boxes; updates
        :attr:`csv_data.wavelength_unit` and :attr:`csv_data.data_type_unit`
        only if ``manual_scale_checkbox`` is checked.  The preview is refreshed
        afterwards.

        Args:
            index (int): Index of the selected item in the combo box
                (unused, kept for signal compatibility).
        """
        if self.manual_scale_checkbox.isChecked():
            #self.csv_data.manual_scaling = self.manual_scale_checkbox.isChecked()
            self.csv_data.wavelength_unit = self.wavelength_unit_combo.currentText()
            self.csv_data.data_type_unit = self.data_type_combo.currentText()
        
            if self.current_file:
                self.preview_and_display_file(self.current_file)
    
    def on_interp_mode_change(self, index):
        """Re‑render the preview when the interpolation mode changes.

        Triggered by the interpolation mode combo box; simply refreshes the
        current file’s display to use the new setting.

        Args:
            index (int): Index of the chosen interpolation mode
                (unused in this slot).
        """
        if self.current_file:
            self.preview_and_display_file(self.current_file)

    def import_data(self):
        """
        Import all user‑selected CSV files and display the last successfully read file.
    
        The method reads configuration options from the UI (separator, decimal sign,
        row‑skipping, interpolation mode), then iterates over the list widget items.
        Each checked file is loaded via :meth:`csv_data.load_csv` and converted to a
        pandas DataFrame by :meth:`csv_data.import_data`.  Files that raise an exception
        are un‑checked and trigger a warning dialog.
    
        After processing, if at least one file succeeded the table model is updated
        with the last DataFrame and the raw text area shows the raw CSV content.
        If no files were selected or all imports failed, appropriate warnings are shown.
        """
        checked_items = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == Qt.Checked
        ]
        if not checked_items:
            QMessageBox.warning(self, "No Files Selected", "Please check at least one file.")
            return

        self.csv_data.separator = self.separator_combo.currentText()
        self.csv_data.decimal_sep = self.decimal_combo.currentText()
        self.csv_data.skip_initial_rows = self.skip_initial_spin.value()
        self.csv_data.skip_end_rows = self.skip_end_spin.value()
        self.csv_data.interp_mode = (
            "extend" if self.interp_mode_combo.currentIndex() == 1 else "crop"
        )

        valid_dfs = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            path = item.text()
            try:
                self.csv_data.load_csv(path)
                df = self.csv_data.import_data()
                valid_dfs.append(df)
                item.setCheckState(Qt.Checked)
            except Exception as e:
                item.setCheckState(Qt.Unchecked)
                QMessageBox.warning(self, "Import Failed", f"{path}\n{e}")

        if not valid_dfs:
            QMessageBox.warning(self, "No Valid Files", "All selected files failed to load.")
            return

        # Show last successfully processed file
        final_df = valid_dfs[-1]
        self.model.setDataFrame(final_df)
        self.raw_display.setText(self.csv_data.raw_content)

    def import_all_files(self):
        """
        Load every checked CSV file into ``self.dataframe_dict``.
        
        For each selected file the method:
        
        1. Loads the raw content via :meth:`csv_data.load_csv`.
        2. If *Auto‑Detect* is enabled, calls :meth:`detect_separators_and_decimal`
           to update separator/decimal settings.
        3. Converts the CSV into a pandas DataFrame with
           :meth:`csv_data.import_data` and stores it in ``self.dataframe_dict``,
           keyed by the file’s base name.
        
        Files that raise an exception are un‑checked and trigger a warning dialog.
        After processing, a message box reports how many files were imported or
        informs the user if none succeeded.
        """
        self.dataframe_dict = {}
    
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
    
            file_path = item.text()
            base_name = file_path.split("/")[-1]
    
            try:
                self.csv_data.load_csv(file_path)
                if self.auto_detect_checkbox.isChecked():
                    self.detect_separators_and_decimal()
                df = self.csv_data.import_data()
                self.dataframe_dict[base_name] = df
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Import Warning",
                    f"Failed to import {base_name}:\n{e}"
                )
                item.setCheckState(Qt.Unchecked)
    
        if not self.dataframe_dict:
            QMessageBox.information(self, "No Data", "No valid CSV files imported.")
        else:
            QMessageBox.information(
                self,
                "Import Complete",
                f"Imported {len(self.dataframe_dict)} files successfully."
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVImporterWindow()
    window.show()
    sys.exit(app.exec_())
