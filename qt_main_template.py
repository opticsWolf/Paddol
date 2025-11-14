# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

from PySide6.QtWidgets import (QApplication, QMainWindow, QDockWidget,
                               QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                               QLabel, QCheckBox, QSlider, QSpinBox, QTabWidget,
                               QGroupBox, QRadioButton, QStatusBar, QSpacerItem,
                               QSizePolicy, QProgressBar, QButtonGroup, QMenu)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QSize, Signal, QTimer, QDateTime

import time #only needed for progress bar examples

class CustomStatusBar(QStatusBar):
    """Custom status bar with centered message and action buttons."""

    button1_clicked = Signal()
    button2_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create the main layout
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)

        # Create right-aligned message label with minimum size
        self.message_label = QLabel("Ready")
        self.message_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.message_label.setMinimumSize(200, 20)
        #self.message_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Initialize time tracking variables
        self.start_time = None
        self.total_duration = 0.0
        self.elapsed = 0.0
        self.last_update_time = None
        self.time_update_timer = QTimer()
        self.time_update_timer.setInterval(100)  # Update every 100ms for smoother updates
        self.time_update_timer.timeout.connect(self.update_time_display)

        # Set up the left layout (message and progress bar)
        self.left_layout = QHBoxLayout()
        self.left_layout.setContentsMargins(8, 0, 0, 0)

        # Progress bar - left aligned
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setFixedWidth(250)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 6px;
                background-color: #e0e0e0;
                color: #616161;
                font-size: 8pt;
                text-align: center;
                padding-left: 0px;
            }
            QProgressBar::chunk {
                border: none;
                background-color: #1E90FF;
                border-radius: 6px;
            }
        """)
        self.progress_bar.hide()

        # Add widgets directly (no spacer)
        self.left_layout.addWidget(self.message_label, stretch=1)
        self.left_layout.addWidget(self.progress_bar)
        self.left_layout.addStretch(4)  # push everything left

        # Add left layout to main layout
        self.main_layout.addLayout(self.left_layout)

        # Button container (already right-aligned due to margins)
        self.button_container = QWidget()
        self.button_layout = QHBoxLayout(self.button_container)
        self.button_layout.setContentsMargins(0, 0, 0, 0)  # Left/right margins

        icon_size = 20
        self.button1 = QPushButton()
        self.button1.setIcon(QIcon.fromTheme("help-about"))
        self.button1.setIconSize(QSize(icon_size, icon_size))
        self.button1.setToolTip("Help")
        self.button1.setStyleSheet("padding: 0px; border: none; background: transparent;")
        self.button1.clicked.connect(self.emit_button1_clicked)

        self.button2 = QPushButton()
        self.button2.setIcon(QIcon.fromTheme("dialog-warning"))
        self.button2.setIconSize(QSize(icon_size, icon_size))
        self.button2.setToolTip("About")
        self.button2.setStyleSheet("padding: 6px; border: none; background: transparent;")
        self.button2.clicked.connect(self.emit_button2_clicked)

        self.button_layout.addWidget(self.button1)
        self.button_layout.addWidget(self.button2)

        # Add button container to main layout
        self.main_layout.addWidget(self.button_container)

        # Set the layout for our widget and add to status bar
        self.setContentsMargins(0, 0, 0, 0)  # Remove default margins
        self.addPermanentWidget(self.main_widget, stretch=1)

        # Timer for auto-hiding progress bar
        self.progress_timer = QTimer()
        self.progress_timer.setInterval(5000)  # 5 seconds
        self.progress_timer.timeout.connect(self.hide_progress_bar)

    def emit_button1_clicked(self):
        """Emit signal when first button is clicked."""
        self.button1_clicked.emit()

    def emit_button2_clicked(self):
        """Emit signal when second button is clicked."""
        self.button2_clicked.emit()

    def showMessage(self, text: str):
        """Set the status message."""
        self.message_label.setText(text)
    
    def start_timing(self, total_duration_seconds: float):
        """Start timing an operation with specified total duration."""
        self.total_duration = float(total_duration_seconds)
        self.start_time = QDateTime.currentMSecsSinceEpoch() / 1000.0
        self.last_update_time = self.start_time
        self.elapsed = 0.0
    
        # Start timer to periodically refresh label
        self.time_update_timer.start()
        self.update_time_display()
    
    def update_time_display(self):
        """Update displayed elapsed and remaining time based on progress."""
        if self.start_time is None:
            return
    
        progress = self.progress_bar.value() / max(1, self.progress_bar.maximum())
    
        if progress <= 0.0:
            elapsed = self.elapsed
            remaining = self.total_duration
            status = f"Elapsed: {self.format_seconds(elapsed)} | Remaining: {self.format_seconds(remaining)}"
    
        elif progress >= 1.0:
            # Completed
            elapsed = self.elapsed
            status = f"Elapsed: {self.format_seconds(elapsed)} | Done"
            self.stop_timing()
    
        else:
            # Estimate total time based on progress ratio
            estimated_total = self.elapsed / progress if progress > 0 else self.total_duration
            remaining = max(0.0, estimated_total - self.elapsed)
            elapsed = self.elapsed
            status = f"Elapsed: {self.format_seconds(elapsed)} | Remaining: {self.format_seconds(remaining)}"
    
        self.message_label.setText(status)
        self.main_widget.updateGeometry()

    def stop_timing(self):
        """Stop the timing and reset display."""
        if self.time_update_timer.isActive():
            self.time_update_timer.stop()
        self.start_time = None
        self.last_update_time = None
      
    def format_seconds(self, total_seconds: float) -> str:
        """Format elapsed time adaptively as S.ms, M:SS, or H:MM:SS.
    
        The function formats time in one of three ways based on magnitude:
        - Less than a minute: seconds.milliseconds (e.g., "5.123")
        - At least a minute but less than an hour: minutes:seconds.milliseconds (e.g., "1:05.123")
        - One hour or more: hours:minutes:seconds (e.g., "1:02:05")
    
        Millisecond precision is always maintained where applicable.
    
        Args:
            total_seconds (float): Elapsed time in seconds (with millisecond precision).
    
        Returns:
            str: Formatted time string.
        """
        if total_seconds < 0:
            return "0.000"
    
        # Calculate components with millisecond precision
        total_ms = round(total_seconds * 1000)  # Convert to integer milliseconds, rounding
    
        hours = total_ms // 3_600_000
        remaining_after_hours = total_ms % 3_600_000
        minutes = remaining_after_hours // 60_000
        remaining_after_minutes = remaining_after_hours % 60_000
        seconds = remaining_after_minutes // 1000
        milliseconds = remaining_after_minutes % 1000
    
        # Get the first digit of milliseconds (tenths place)
        ms_digit = milliseconds // 100
    
        # Format adaptively
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        elif minutes > 0:
            return f"{minutes:02d}:{seconds:02d}"
        else:
            # For less than a minute, show seconds.millisecond
            if seconds >= 1:
                return f"{seconds}.{ms_digit}s"
            else:
                return f"0.{ms_digit}s"

    
    def show_progress_bar(self, value: int = 0):
        """Show the progress bar and start timing if not running."""
        self.progress_bar.setValue(value)
        self.progress_bar.show()
    
        if not self.time_update_timer.isActive():
            # Default total duration placeholder (can be customized)
            self.start_timing(60.0)
    
        self.main_widget.updateGeometry()
    
    def hide_progress_bar(self):
        """Hide the progress bar and stop timing."""
        if self.progress_bar.isVisible():
            self.progress_bar.hide()
            self.stop_timing()
            self.main_widget.updateGeometry()
    
    def update_progress(self, value: int):
        """Update progress bar and compute elapsed time incrementally."""
        if not self.progress_bar.isVisible():
            self.show_progress_bar(value)
    
        current_time = QDateTime.currentMSecsSinceEpoch() / 1000.0
    
        if self.start_time is None:
            # Initialize timing if missing
            self.start_timing(60.0)
            self.last_update_time = current_time
    
        # Compute delta since last update
        interval = max(0.0, current_time - self.last_update_time)
        self.last_update_time = current_time
        self.elapsed += interval
    
        # Update progress value and refresh display
        self.progress_bar.setValue(value)
        self.update_time_display()
    
        # Restart hide timer (auto-hide after 5s inactivity)
        self.progress_timer.start()
    
        # Stop timing when complete
        if value >= self.progress_bar.maximum():
            self.stop_timing()
    
    def set_range(self, minimum: int = 0, maximum: int = 100):
        """Set range for progress bar."""
        self.progress_bar.setRange(minimum, maximum)
    
    def is_progress_visible(self) -> bool:
        """Check if progress bar is currently visible."""
        return self.progress_bar.isVisible()


class CustomDockWidget(QDockWidget):
    """Custom dock widget with two group boxes: 
    one with a label and button, another with two radio buttons."""

    def __init__(self, title: str = "Dock - Right Only", parent=None):
        super().__init__(title, parent)

        # Enable custom context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_custom_menu)

        # Restrict docking to the right
        self.setAllowedAreas(Qt.RightDockWidgetArea)

        # Main content widget
        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # ----- First GroupBox -----
        group_box1 = QGroupBox("Group Box 1", content)
        vbox1 = QVBoxLayout(group_box1)

        label = QLabel("Label in GroupBox", group_box1)
        vbox1.addWidget(label)

        button = QPushButton("Click me", group_box1)
        button.clicked.connect(self.on_button_clicked)
        vbox1.addWidget(button)

        group_box1.setLayout(vbox1)
        layout.addWidget(group_box1)

        # ----- Second GroupBox (Radio Buttons) -----
        group_box2 = QGroupBox("Options", content)
        vbox2 = QVBoxLayout(group_box2)

        self.radio1 = QRadioButton("Option 1", group_box2)
        self.radio2 = QRadioButton("Option 2", group_box2)

        # Optional: group the radio buttons
        self.radio_group = QButtonGroup(group_box2)
        self.radio_group.addButton(self.radio1)
        self.radio_group.addButton(self.radio2)

        vbox2.addWidget(self.radio1)
        vbox2.addWidget(self.radio2)

        group_box2.setLayout(vbox2)
        layout.addWidget(group_box2)

        # Set main layout to content and assign to dock
        content.setLayout(layout)
        self.setWidget(content)

    def show_custom_menu(self, pos):
        """Show a hybrid context menu (default dock actions + custom actions)."""
        menu = QMenu(self)
    
        # Standard float action
        if self.features() & QDockWidget.DockWidgetFloatable:
            float_action = menu.addAction("Float")
            float_action.setCheckable(True)
            float_action.setChecked(self.isFloating())
            float_action.triggered.connect(lambda: self.setFloating(not self.isFloating()))
            menu.addAction(float_action)
    
        # Standard close action
        if self.features() & QDockWidget.DockWidgetClosable:
            close_action = menu.addAction("Close")
            close_action.triggered.connect(self.close)
            menu.addAction(close_action)
    
        menu.addSeparator()
    
        # Custom actions
        custom_action = menu.addAction("Custom Action")
        custom_action.triggered.connect(lambda: print("Custom action clicked!"))
    
        # **Convert local pos to global**
        menu.exec(self.mapToGlobal(pos))


    def on_button_clicked(self):
        """Handle button click event."""
        print("Button clicked inside custom dock widget!")
   

class MainWindow(QMainWindow):
    """
    A PySide6 application demonstrating five dockable widgets with different
    arrangements and features. Some are tabbed together, others are standalone.
    All content is contained in dock widgets; no central widget is fixed.

    Attributes:
        dock1 (QDockWidget): Standalone dock widget with vertical layout.
        dock2 (QDockWidget): Part of a tabbed group with horizontal layout.
        dock3 (QDockWidget): Tabbed with dock2, using vertical layout.
        dock4 (QDockWidget): Part of another tabbed group with nested layouts.
        dock5 (QDockWidget): Tabbed with dock4, using horizontal layout.
    """

    def __init__(self):
        """Initialize the main window and set up all dock widgets."""
        super().__init__()
        self.setWindowTitle("Dock Widget Example")
        self.resize(800, 600)

        # Set up custom status bar
        self.status_bar = CustomStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")  # Initial message

        # Connect signals from status bar to handlers with new names
        self.status_bar.button1_clicked.connect(self.on_button1_clicked)
        self.status_bar.button2_clicked.connect(self.on_button2_clicked)

        # Set tab position to top for all dock areas
        self.setTabPosition(Qt.LeftDockWidgetArea, QTabWidget.North)
        self.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.South)
        self.setTabPosition(Qt.TopDockWidgetArea, QTabWidget.North)
        self.setTabPosition(Qt.BottomDockWidgetArea, QTabWidget.South)
    
        # Create and configure DockWidget 1 (limited to left side only)
        dock1 = QDockWidget("Dock 1 - Left Only", self)
        content1 = QWidget()
        layout1 = QVBoxLayout()
        button1 = QPushButton("Button 1")
        label1 = QLabel("This can only be on the left side")
        button2 = QPushButton("Another Button")
        layout1.addWidget(button1)
        layout1.addWidget(label1)
        layout1.addWidget(button2)
        content1.setLayout(layout1)
        dock1.setWidget(content1)
    
        allowed_areas_dock1 = Qt.LeftDockWidgetArea | Qt.TopDockWidgetArea
        dock1.setAllowedAreas(allowed_areas_dock1)
    
        self.addDockWidget(Qt.LeftDockWidgetArea, dock1)
    
        # Create and configure DockWidget 2 (first tabbed group - limited to top)
        dock2 = QDockWidget("Dock 2 - Top Only", self)
        content2 = QWidget()
        layout2 = QHBoxLayout()
        checkbox1 = QCheckBox("Option A")
        checkbox2 = QCheckBox("Option B")
        layout2.addWidget(checkbox1)
        layout2.addWidget(checkbox2)
        content2.setLayout(layout2)
        dock2.setWidget(content2)
    
        # Create and configure DockWidget 3 (tabbed with dock2 - limited to top)
        dock3 = QDockWidget("Dock 3 - Top Only", self)
        content3 = QWidget()
        layout3 = QVBoxLayout()
        slider = QSlider(Qt.Horizontal)
        spinbox = QSpinBox()
        layout3.addWidget(slider)
        layout3.addWidget(spinbox)
        content3.setLayout(layout3)
        dock3.setWidget(content3)
    
        # Restrict dock 3 both to top side only
        allowed_areas_dock3 = Qt.LeftDockWidgetArea | Qt.TopDockWidgetArea
        dock3.setAllowedAreas(allowed_areas_dock3)
    
        self.addDockWidget(Qt.TopDockWidgetArea, dock2)
        self.tabifyDockWidget(dock2, dock3)
    
        # Create and configure DockWidget 4 (second tabbed group - limited to right)
        dock4 = CustomDockWidget("Dock 4 - Right Only", self)
        dock4.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
    
        # Create and configure DockWidget 5 (tabbed with dock4 - limited to right)
        dock5 = QDockWidget("Dock 5 - Right Only", self)
        content5 = QWidget()
        layout5 = QHBoxLayout()
        radio3 = QRadioButton("Choice X")
        radio4 = QRadioButton("Choice Y")
        label2 = QLabel("This can only be on the right side")
        layout5.addWidget(radio3)
        layout5.addWidget(radio4)
        layout5.addWidget(label2)
        content5.setLayout(layout5)
        dock5.setWidget(content5)
    
        # Restrict both to right side only
        allowed_areas_dock4 = Qt.TopDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        allowed_areas_dock5 = Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        dock4.setAllowedAreas(allowed_areas_dock4)
        dock5.setAllowedAreas(allowed_areas_dock5)
    
        self.addDockWidget(Qt.RightDockWidgetArea, dock4)
        self.tabifyDockWidget(dock4, dock5)


    def on_button1_clicked(self):
        """Handle first button click."""
        print("First button clicked")
        # Show progress for an operation
        self.status_bar.set_range(0, 100)
        self.status_bar.showMessage("In progress")
        for i in range(1, 101):
            self.status_bar.update_progress(i)
            time.sleep(0.05)  # Simulate work


    def on_button2_clicked(self):
        """Handle second button click."""
        print("Second button clicked")
        # Show progress for a quick operation that might finish before timer
        self.status_bar.set_range(0, 100)
        self.status_bar.showMessage("About In progress")
        for i in range(1, 101):  # Will show longer than the actual work
            self.status_bar.update_progress(i)
            time.sleep(1.0)  # Simulate work


# The rest of your code (dock widget setup, main function, etc.) remains the same
from PySide6.QtGui import QGuiApplication

QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Light)

def main():
    """Entry point for the application."""
    # Check if application already exists (for environments like Spyder)
    app = QApplication.instance()
    if not app:
        app = QApplication([])
        #app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    # For environments where we need to explicitly exit
    try:
        app.exec()
    except SystemExit:
        pass

if __name__ == "__main__":

    main()
