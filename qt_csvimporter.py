# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QListWidget, QListWidgetItem, QMenu,
    QFileDialog, QLabel, QHBoxLayout, QComboBox, QSplitter, QTextEdit, QTableView, QMessageBox, QSpinBox, QCheckBox, QGroupBox, QFormLayout
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QAbstractTableModel

import polars as pl

from csvdata import CSVData

def scan_icons_folder(folder_path):
    icon_dict = {}
    folder = Path(folder_path)
    # Get all files in the directory that end with .png or .svg (case insensitive)
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in ('.png', '.svg'):
            key = file.stem  # This is filename without extension
            icon_dict[key] = str(file)  # Convert to string path for PyQt compatibility
    return icon_dict

ICON_DICT = scan_icons_folder(Path("src/icons"))

class PolarsModel(QAbstractTableModel):
    """Qt model to display a Polars DataFrame in QTableView."""
    def __init__(self, data=None):
        super().__init__()
        self._data = None
        if data is not None:
            self.setDataFrame(data)
    
    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._data = dataframe
        self.endResetModel()

    def rowCount(self, index):
        return len(self._data) if self._data is not None else 0

    def columnCount(self, index):
        return len(self._data.columns) if self._data is not None else 0

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._data is None or role != Qt.DisplayRole:
            return None
        try:
            # Direct access for better performance
            value = self._data[index.row(), index.column()]
            return str(value)
        except Exception:
            return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
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

        # ===== Central Layout =====
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ===== File Selection Group =====
        file_group = QGroupBox("Loaded CSV Files")
        file_layout = QVBoxLayout(file_group)
        self.select_button = QPushButton(" Select CSV Files")
        self.select_button.setIcon(QIcon(ICON_DICT['open_folder']))
        self.select_button.clicked.connect(self.select_files)
        file_layout.addWidget(self.select_button)

        # File list with checkboxes
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
        sep_dec_layout.addWidget(QLabel("Separator:"))
        sep_dec_layout.addWidget(self.separator_combo)
        sep_dec_layout.addWidget(QLabel("Decimal:"))
        sep_dec_layout.addWidget(self.decimal_combo)
        options_layout.addRow(sep_dec_layout)

        skip_layout = QHBoxLayout()
        self.skip_initial_spin = QSpinBox()
        self.skip_initial_spin.setRange(0, 100)
        self.skip_initial_spin.valueChanged.connect(self.import_data)
        self.skip_end_spin = QSpinBox()
        self.skip_end_spin.setRange(0, 100)
        self.skip_end_spin.valueChanged.connect(self.import_data)
        skip_layout.addWidget(QLabel("Skip Start:"))
        skip_layout.addWidget(self.skip_initial_spin)
        skip_layout.addWidget(QLabel("Skip End:"))
        skip_layout.addWidget(self.skip_end_spin)
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

        manual_layout.addWidget(QLabel("Wavelength:"))
        manual_layout.addWidget(self.wavelength_unit_combo)

        manual_layout.addWidget(QLabel("Data Type:"))
        manual_layout.addWidget(self.data_type_combo)
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
    # FILE HANDLING
    # ====================================================================
    def select_files(self):
        """Allow user to select multiple CSV files and preview the first one."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not file_paths:
            return
    
        first_new_item = None
        for path in file_paths:
            if path not in [f[0] for f in self.files]:
                self.files.append((path, True))
                item = QListWidgetItem(path)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Checked)
                self.file_list.addItem(item)
                if first_new_item is None:
                    first_new_item = item
    
        # Auto-select and preview the first new file
        if first_new_item:
            self.file_list.setCurrentItem(first_new_item)
            self.preview_and_display_file(first_new_item.text())
    
    
    #def on_file_selected(self, current, previous):
    #    """Triggered when user selects a file in the list."""
    #    if current and current.checkState() == Qt.Checked:
    #        self.preview_file(current.text())
    
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
    def show_file_context_menu(self, pos):
        """Show context menu to remove files from the list."""
        item = self.file_list.itemAt(pos)
        if not item:
            return
    
        menu = QMenu(self)
        remove_action = menu.addAction("Remove File")
        action = menu.exec_(self.file_list.mapToGlobal(pos))
    
        if action == remove_action:
            file_path = item.text()
    
            # Remove from internal list
            self.files = [(p, checked) for p, checked in self.files if p != file_path]
    
            # Remove from QListWidget
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
    
            # Clear preview if this file was shown
            if self.current_file == file_path:
                self.raw_display.clear()
                self.model.setDataFrame(pl.DataFrame())
                self.current_file = None

    # ====================================================================
    # SETTINGS AND IMPORT LOGIC
    # ====================================================================
    def toggle_auto_detection(self, state):
        self.separator_combo.setEnabled(state == Qt.Unchecked)
        self.decimal_combo.setEnabled(state == Qt.Unchecked)
        
        if state == Qt.Checked and self.current_file:
            self.detect_separators_and_decimal()
        elif state == Qt.Unchecked:
            self.on_seperator_changed(0)

    def on_seperator_changed(self, index):
        if self.manual_scale_checkbox.isChecked() is not True and self.current_file:
            self.csv_data.separator = self.separator_combo.currentText()
            self.csv_data.decimal_sep = self.decimal_combo.currentText()
            self.preview_and_display_file(self.current_file)
            
    def detect_separators_and_decimal(self):
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
        if self.current_file:
            self.csv_data.auto_sorting = (state == Qt.Checked)
            self.preview_and_display_file(self.current_file)

    def toggle_manual_scaling(self, state):
        self.wavelength_unit_combo.setEnabled(state == Qt.Checked)
        self.data_type_combo.setEnabled(state == Qt.Checked)
        self.csv_data.manual_scaling = (state == Qt.Checked)
        if self.current_file:
            self.preview_and_display_file(self.current_file)

    def on_scaling_changed(self, index):
        if self.manual_scale_checkbox.isChecked():
            #self.csv_data.manual_scaling = self.manual_scale_checkbox.isChecked()
            self.csv_data.wavelength_unit = self.wavelength_unit_combo.currentText()
            self.csv_data.data_type_unit = self.data_type_combo.currentText()
        
            if self.current_file:
                self.preview_and_display_file(self.current_file)
    
    def on_interp_mode_change(self, index):
        if self.current_file:
            self.preview_and_display_file(self.current_file)

    def import_data(self):
        """Import all checked files and display the last processed one."""
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
        """Import all checked files into dataframe_dict keyed by filename."""
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
