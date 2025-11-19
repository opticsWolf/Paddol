# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
from PyQt5.QtWidgets import QApplication, QWidget, QLayout, QSplitter, QSpacerItem, QLayoutItem
from PyQt5.QtCore import QMargins, QPropertyAnimation, QEasingCurve, QObject

class CollapsibleSectionMixin(QObject):
    """Adds collapsible-section behavior to any QWidget with a layout."""

    def collect_qt_children(self, root):
        """Recursively collect all descendant widgets, layouts, and layout items.
    
        Args:
            root (QWidget | QLayout | QLayoutItem): The starting element to traverse.
    
        Returns:
            list: All descendant elements including widgets, layouts, and items.
                  Excludes None values and the root itself if not a widget/container.
        """
        descendants = []
    
        def _collect(node):
            if node is None:
                return
            descendants.append(node)
    
            # Handle QWidget case (including those with layout or splitter)
            if isinstance(node, QWidget):
                # Process layout if exists
                layout = node.layout()
                if layout is not None:
                    _collect(layout)
    
                # Handle special case for splitters
                if isinstance(node, QSplitter):
                    for i in range(node.count()):
                        widget = node.widget(i)
                        _collect(widget)
    
            # Handle QLayout case (process all items)
            elif isinstance(node, QLayout):
                for i in range(node.count()):
                    item = node.itemAt(i)
                    _collect(item)
    
            # Handle QLayoutItem cases
            elif isinstance(node, QLayoutItem):
                # Process widget if exists
                widget = node.widget()
                _collect(widget)
    
                # Process nested layout if exists
                layout = node.layout()
                _collect(layout)
    
        _collect(root)
        return descendants[1:]  # Exclude the root itself

    def enable_collapsible(self, container, root_item):
        """
        Enable expand/collapse with animation for a section.

        Args:
            container (QWidget): Must have QLayout.
            root_item (QWidget or QLayout): root to collect descendants from.
        """
        items = list(self.collect_qt_children(root_item))

        self._toggle_fn = self.make_animated_toggle(container, items)
        if hasattr(container, "toggled"):
            container.toggled.connect(self._toggle_fn)
        else:
            # Allow manual control:
            # self._toggle_fn(True / False)
            pass

        return self._toggle_fn  # optional


    def make_animated_toggle(self, container, items, duration=220):
        """Return toggle(bool) that animates collapsible/expandable sections.
    
        Args:
            container (QWidget): widget owning a QLayout
            items (list): descendants; only QWidget + QSpacerItem are used
            duration (int): animation duration in ms
    
        Returns:
            callable: toggle(bool)
        """
        layout = container.layout()
    
        # Extract only real widgets and spacers
        widgets = [w for w in items if isinstance(w, QWidget)]
        spacers = [s for s in items if isinstance(s, QSpacerItem)]
    
        # Cache original layout geometry
        original_margins = layout.contentsMargins() if layout else QMargins(0, 0, 0, 0)
        original_spacing = layout.spacing() if layout else -1
        original_height = container.sizeHint().height() - 10
    
        # Configure animation
        anim = QPropertyAnimation(container, b"minimumHeight")
        anim.setDuration(duration)
        anim.setEasingCurve(QEasingCurve.InOutCubic)
    
        # Ensure finished signal is not stacked multiple times
        def clear_hidden_widgets():
            for w in widgets:
                if not w.isVisible():
                    w.setVisible(False)
        anim.finished.connect(clear_hidden_widgets)
    
        def toggle(checked: bool):
            # Toggle widget visibility
            for w in widgets:
                w.setVisible(checked)
    
            # Collapse spacers by adjusting their sizeHint/size
            for s in spacers:
                if checked:
                    s.changeSize(s.sizeHint().width(), s.sizeHint().height())
                else:
                    s.changeSize(0, 0)
    
            if not layout:
                container.updateGeometry()
                return
    
            if checked:
                layout.setContentsMargins(
                    original_margins.left(),
                    original_margins.top(),
                    original_margins.right(),
                    original_margins.bottom(),
                )
                layout.setSpacing(original_spacing)
    
                # animate expand
                anim.stop()
                anim.setStartValue(container.minimumHeight())
                anim.setEndValue(original_height)
                anim.start()
    
            else:
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(0)
    
                # animate collapse
                anim.stop()
                anim.setStartValue(container.minimumHeight())
                anim.setEndValue(0)
                anim.start()
    
            container.updateGeometry()
            QApplication.processEvents()  # Ensure UI updates immediately

        return toggle