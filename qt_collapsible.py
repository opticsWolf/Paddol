# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
from PySide6.QtWidgets import QApplication, QWidget, QLayout, QSplitter, QSpacerItem, QLayoutItem
from PySide6.QtCore import QMargins, QPropertyAnimation, QEasingCurve

class CollapsibleSectionMixin():
    """Mixin that provides collapsible section functionality to Qt widgets.

    This class is designed to be mixed into a class that eventually inherits from
    QWidget (or QMainWindow). It provides logic to recursively traverse a layout
    tree and animate the visibility of widgets and spacers.

    Note:
        This is a pure Python class and does not inherit from QObject or QWidget
        to avoid "Diamond Inheritance" / Meta-Object conflicts in PySide6.
    """

    def collect_qt_children(
        self, 
        root: QWidget | QLayout | QLayoutItem | None
    ) -> list[QWidget | QLayout | QLayoutItem]:
        """Recursively collects all descendant widgets, layouts, and layout items.

        Traverses the Qt object tree starting from the given root node to identify
        all elements that need to be hidden or shown during a collapse animation.
        It handles nested layouts, splitters, and individual layout items.

        Args:
            root (QWidget | QLayout | QLayoutItem | None): The starting element 
                to traverse. If None, the method returns immediately.

        Returns:
            list[QWidget | QLayout | QLayoutItem]: A flat list of all descendant 
                elements found in the subtree. The list excludes the `root` itself 
                (if it was passed as the starting point) and any None values.
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
        # Calculate a safe height or use sizeHint
        sh = container.sizeHint()
        original_height = sh.height() - int(sh.height() / 10)
    
        # Configure animation
        # Note: PySide6 QPropertyAnimation usually expects bytes for property name
        anim = QPropertyAnimation(container, b"minimumHeight")
        anim.setDuration(duration)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
    
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
                    # Retrieve cached hint if possible or current hint
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