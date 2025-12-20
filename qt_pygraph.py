import sys
from typing import Optional, Union, Tuple, Dict, Any, Literal, List
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPen, QColor, QFont, QPainter
from PySide6.QtCore import Qt, QRectF, QPointF

# --- Type Aliases ---
ColorType = Union[str, Tuple[int, int, int], Tuple[int, int, int, int], QColor]
TickDirection = Literal['in', 'out']
LineStyleType = Union[Qt.PenStyle, str]
StyleDictType = Dict[str, Union[str, int, float, Dict[str, Any]]]
ViewMode = Literal['left', 'right']

# --- Custom Synchronization Class ---
class SyncViewBox(pg.ViewBox):
    """
    A custom ViewBox that intercepts interaction events and synchronizes them
    proportionally with a peer ViewBox.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.peer_view: Optional[pg.ViewBox] = None
        self._is_updating_peer = False 

    def set_peer_view(self, view: Optional[pg.ViewBox]) -> None:
        """Sets the target ViewBox to synchronize with."""
        self.peer_view = view

    def scaleBy(self, s=None, center=None, x=None, y=None) -> None:
        """
        Intercepts zoom events to apply proportional scaling to the peer.
        """
        # 1. Apply zoom to self (the overlay/master)
        super().scaleBy(s=s, center=center, x=x, y=y)

        # 2. Sync Logic (Apply to peer/underlay)
        if self.peer_view and not self._is_updating_peer:
            self._is_updating_peer = True
            try:
                # Normalize Scale Factor 'sy'
                sy = 1.0
                if y is not None:
                    sy = y
                elif s is not None:
                    # Robust check for vector vs scalar
                    if hasattr(s, '__getitem__'):
                        sy = s[1]
                    else:
                        sy = s
                
                # If sy is effectively 1.0, no vertical zoom happened
                if sy == 1.0: return

                # Normalize Center
                peer_center = None
                if center is not None:
                    # Map: Data(Self) -> Scene -> Data(Peer)
                    pt_scene = self.mapViewToScene(QPointF(center[0], center[1]))
                    pt_peer = self.peer_view.mapSceneToView(pt_scene)
                    peer_center = (pt_peer.x(), pt_peer.y())

                # Apply scale to peer. 
                # Note: X is 1.0 because X-axis is usually linked via standard setXLink
                self.peer_view.scaleBy(s=(1.0, sy), center=peer_center)
            finally:
                self._is_updating_peer = False

    def translateBy(self, t=None, x=None, y=None, axis=None) -> None:
        """
        Intercepts pan events to apply proportional panning to the peer.
        """
        # 1. Apply translation to self
        super().translateBy(t=t, x=x, y=y)

        # 2. Sync Logic
        if self.peer_view and not self._is_updating_peer:
            self._is_updating_peer = True
            try:
                # Normalize Vertical Translation 'dy'
                dy = 0.0
                
                if y is not None:
                    dy = y
                elif t is not None:
                    # Use pg.Point to safely handle QPointF, tuple, or list
                    pt = pg.Point(t)
                    dy = pt.y()
                elif axis == 1 and t is not None:
                    # Fallback for scalar t where axis is explicitly Y
                    dy = t
                
                if dy == 0: return

                # Calculate Percentage Shift
                # We move the peer by the same *percentage* of its view range
                current_range = self.viewRange()[1] 
                height = current_range[1] - current_range[0]
                
                if height == 0: return 

                percent_shift = dy / height
                
                # Calculate Peer Shift
                peer_range = self.peer_view.viewRange()[1]
                peer_height = peer_range[1] - peer_range[0]
                peer_dy = percent_shift * peer_height
                
                # Apply translation (X is handled by setXLink)
                self.peer_view.translateBy(x=0, y=peer_dy)
                
            finally:
                self._is_updating_peer = False


class MatplotlibStylePlot(pg.PlotWidget):
    """
    A unified PlotWidget with CSS-like styling, supporting distinct backgrounds
    and interactive dual-axis (Left/Right) focus switching via Z-Order swapping.
    """

    class _GridItem(pg.GraphicsObject):
        """
        Optimized internal item for drawing custom Grids and Inner Backgrounds.
        Rendered at Z-Value -100 to stay behind all ViewBoxes.
        """
        def __init__(self, plot_item: pg.PlotItem):
            super().__init__()
            self.plot_item = plot_item
            self.pen_major = QPen(Qt.GlobalColor.black)
            self.pen_minor = QPen(Qt.GlobalColor.black)
            self.bg_color: Optional[QColor] = None
            self.setZValue(-100) # Ensure it is behind everything

        def update_pens(self, major: QPen, minor: QPen) -> None:
            self.pen_major = major
            self.pen_minor = minor
            self.update()

        def set_background(self, color: Optional[QColor]) -> None:
            self.bg_color = color
            self.update()

        def boundingRect(self) -> QRectF:
            if self.plot_item.vb is None:
                return QRectF()
            return self.plot_item.vb.viewRect()

        def paint(self, p: QPainter, *args) -> None:
            if self.plot_item.vb is None: 
                return
            rect = self.plot_item.vb.viewRect()
            
            # 1. Paint Background
            if self.bg_color is not None:
                p.fillRect(rect, self.bg_color)

            # 2. Paint Grid
            self._draw_axis_grid(p, 'bottom', rect, is_vertical=True)
            self._draw_axis_grid(p, 'left', rect, is_vertical=False)

        def _draw_axis_grid(self, p: QPainter, axis_name: str, rect: QRectF, is_vertical: bool) -> None:
            axis = self.plot_item.getAxis(axis_name)
            if not axis.isVisible(): 
                return
            
            canvas_len = self.plot_item.vb.width() if is_vertical else self.plot_item.vb.height()
            
            if is_vertical:
                min_val, max_val = sorted((rect.left(), rect.right()))
            else:
                min_val, max_val = sorted((rect.top(), rect.bottom()))

            try:
                tick_levels = axis.tickValues(min_val, max_val, canvas_len)
            except Exception:
                return
            
            tick_levels.sort(key=lambda x: x[0], reverse=True)

            for i, (_, values) in enumerate(tick_levels):
                p.setPen(self.pen_major if i == 0 else self.pen_minor)
                for val in values:
                    if is_vertical:
                        p.drawLine(QPointF(val, rect.top()), QPointF(val, rect.bottom()))
                    else:
                        p.drawLine(QPointF(rect.left(), val), QPointF(rect.right(), val))

    def __init__(
        self, 
        parent=None, 
        background: ColorType = 'w', 
        title: Optional[str] = None, 
        highlight_factor: float = 2.0,
        **kargs
    ):
        # --- Injection Strategy ---
        # We manually construct the PlotItem with our custom SyncViewBox 
        # so the Primary View (Left) also supports proportional syncing.
        self.main_vb = SyncViewBox()
        self.plot_item = pg.PlotItem(viewBox=self.main_vb)
        
        # Initialize PlotWidget with our custom PlotItem
        super().__init__(parent=parent, background=background, plotItem=self.plot_item, **kargs)
        
        if title:
            self.plot_item.setTitle(title)
        
        self.vb_right: Optional[SyncViewBox] = None 
        self._vb_right_connection = None 
        self._axis_style_config: Dict[str, Any] = {} 
        self._active_view_mode: ViewMode = 'left'
        
        # Scaling & Coupling Configurations
        self._highlight_factor = highlight_factor
        self._link_x: bool = True
        self._link_y: bool = False
        
        # 1. Initialize Pens
        self._pen_spine = QPen(QColor(0, 0, 0, 255), 1.5)
        self._pen_spine.setCosmetic(True)
        self._pen_tick = QPen(QColor(0, 0, 0, 255), 1.5)
        self._pen_tick.setCosmetic(True)

        self._pen_grid_major = QPen(QColor(160, 160, 160, 255), 1.0)
        self._pen_grid_major.setCosmetic(True)
        self._pen_grid_minor = QPen(QColor(220, 220, 220, 255), 0.8)
        self._pen_grid_minor.setStyle(Qt.PenStyle.DotLine)
        self._pen_grid_minor.setCosmetic(True)

        # 2. Setup Grid & Inner Background Item
        self.plot_item.showGrid(x=False, y=False)
        self._grid_item = self._GridItem(self.plot_item)
        self._grid_item.update_pens(self._pen_grid_major, self._pen_grid_minor)
        self.plot_item.addItem(self._grid_item, ignoreBounds=True)

        # 3. Init Axes
        self._init_axes()
        
        # 4. Connect Click Logic
        self.scene().sigMouseClicked.connect(self._on_scene_clicked)

    def _init_axes(self) -> None:
        self.plot_item.setContentsMargins(15, 15, 15, 15)
        for axis_name in ['left', 'bottom', 'right', 'top']:
            self.plot_item.showAxis(axis_name)
            is_main = (axis_name in ['left', 'bottom'])
            self._apply_axis_style(axis_name, show_ticks=is_main, show_values=is_main)

    def _to_qcolor(self, color: ColorType, alpha: Optional[float] = None) -> QColor:
        c = pg.mkColor(color)
        if alpha is not None:
            c.setAlphaF(max(0.0, min(1.0, alpha)))
        return c

    def _resolve_pen_style(self, style: LineStyleType) -> Qt.PenStyle:
        if isinstance(style, Qt.PenStyle): return style
        s = str(style).lower().strip()
        mapping = {
            'solid': Qt.PenStyle.SolidLine, '-': Qt.PenStyle.SolidLine, 
            'dashed': Qt.PenStyle.DashLine, '--': Qt.PenStyle.DashLine,
            'dotted': Qt.PenStyle.DotLine, ':': Qt.PenStyle.DotLine,
            'dash-dot': Qt.PenStyle.DashDotLine, '-.': Qt.PenStyle.DashDotLine,
            'none': Qt.PenStyle.NoPen, '': Qt.PenStyle.NoPen
        }
        return mapping.get(s, Qt.PenStyle.SolidLine)
        
    def _parse_css_style(self, style_dict: StyleDictType) -> Tuple[QFont, Dict, Dict, bool]:
        weight = style_dict.get('font-weight', 'normal')
        style = style_dict.get('font-style', 'normal')
        decoration = style_dict.get('text-decoration', 'none')
        size_raw = style_dict.get('font-size', '10pt')
        color = style_dict.get('color', None)
        
        qfont = QFont()
        if isinstance(size_raw, str):
            try:
                size_val = int(''.join(filter(str.isdigit, size_raw)))
                qfont.setPointSize(size_val)
            except ValueError:
                qfont.setPointSize(10)
        elif isinstance(size_raw, int):
            qfont.setPointSize(size_raw)

        if weight == 'bold': qfont.setBold(True)
        if style == 'italic': qfont.setItalic(True)
        if decoration == 'underline': qfont.setUnderline(True)

        label_opts = {}
        if size_raw: label_opts['font-size'] = str(size_raw)
        if weight == 'bold': label_opts['font-weight'] = 'bold'
        if style == 'italic': label_opts['font-style'] = 'italic'
        if color: label_opts['color'] = self._to_qcolor(color).name()

        title_opts = {}
        if size_raw: title_opts['size'] = str(size_raw)
        if weight == 'bold': title_opts['bold'] = True
        if style == 'italic': title_opts['italic'] = True
        if color: title_opts['color'] = self._to_qcolor(color).name()

        should_underline = (decoration == 'underline')

        return qfont, label_opts, title_opts, should_underline

    def _wrap_html(self, text: str, underline: bool) -> str:
        if not underline:
            if text.startswith('<u>') and text.endswith('</u>'):
                return text[3:-4]
            return text
        if not (text.startswith('<u>') and text.endswith('</u>')):
            return f"<u>{text}</u>"
        return text

    # --- Styling API ---

    def set_grid_style(
        self,
        major_color: ColorType = "gray",
        major_width: float = 1.0,
        major_alpha: float = 1.0,
        major_style: LineStyleType = Qt.PenStyle.SolidLine,
        minor_color: ColorType = "lightgray",
        minor_width: float = 0.8,
        minor_alpha: float = 1.0,
        minor_style: LineStyleType = Qt.PenStyle.DotLine
    ) -> None:
        qt_major_style = self._resolve_pen_style(major_style)
        qt_minor_style = self._resolve_pen_style(minor_style)

        c_major = self._to_qcolor(major_color, major_alpha)
        self._pen_grid_major = QPen(c_major, major_width)
        self._pen_grid_major.setStyle(qt_major_style)
        self._pen_grid_major.setCosmetic(True)

        c_minor = self._to_qcolor(minor_color, minor_alpha)
        self._pen_grid_minor = QPen(c_minor, minor_width)
        self._pen_grid_minor.setStyle(qt_minor_style)
        self._pen_grid_minor.setCosmetic(True)

        self._grid_item.update_pens(self._pen_grid_major, self._pen_grid_minor)

    def set_axis_style(
        self, 
        color: ColorType = "black", 
        width: float = 1.5,
        spine_width: Optional[float] = None,
        tick_width: Optional[float] = None,
        tick_label_style: Optional[StyleDictType] = None,
        axis_label_style: Optional[StyleDictType] = None,
        show_ticks: Union[bool, List[str]] = True,
        show_tick_labels: Union[bool, List[str]] = ['left', 'bottom'],
        tick_length: int = 5,
        tick_direction: TickDirection = "out",
        tick_alpha: Optional[float] = None,
        spine_alpha: Optional[float] = None
    ) -> None:
        t_style = tick_label_style if tick_label_style else {}
        a_style = axis_label_style if axis_label_style else {}

        final_spine_width = spine_width if spine_width is not None else width
        final_tick_width = tick_width if tick_width is not None else width

        self._axis_style_config = {
            'color': color, 
            'width': width,                   
            'spine_width': final_spine_width, 
            'tick_width': final_tick_width,   
            'tick_label_style': t_style,
            'axis_label_style': a_style,
            'show_ticks': show_ticks, 'show_tick_labels': show_tick_labels,
            'tick_length': tick_length, 'tick_direction': tick_direction,
            'tick_alpha': tick_alpha, 'spine_alpha': spine_alpha
        }

        main_c = self._to_qcolor(color)
        spine_c = QColor(main_c)
        if spine_alpha is not None: spine_c.setAlphaF(spine_alpha)
        tick_c = QColor(main_c)
        if tick_alpha is not None: tick_c.setAlphaF(tick_alpha)

        self._pen_spine = QPen(spine_c)
        self._pen_spine.setWidthF(final_spine_width)
        self._pen_spine.setCosmetic(True)
        
        self._pen_tick = QPen(tick_c)
        self._pen_tick.setWidthF(final_tick_width)
        self._pen_tick.setCosmetic(True)
        
        target_ticks = set(show_ticks) if isinstance(show_ticks, list) else ({'left', 'bottom', 'right', 'top'} if show_ticks else set())
        target_labels = set(show_tick_labels) if isinstance(show_tick_labels, list) else ({'left', 'bottom', 'right', 'top'} if show_tick_labels else set())

        for axis_name in ['left', 'bottom', 'right', 'top']:
            self._apply_axis_style(
                axis_name, 
                show_ticks=(axis_name in target_ticks),
                show_values=(axis_name in target_labels),
                tick_length=tick_length, tick_direction=tick_direction,
                tick_style_dict=t_style,
                label_style_dict=a_style
            )

    def _apply_axis_style(
        self, 
        axis_name: str, 
        show_ticks: bool, 
        show_values: bool,
        tick_length: int = 5,
        tick_direction: TickDirection = 'in',
        tick_style_dict: dict = {},
        label_style_dict: dict = {},
        pen_spine: Optional[QPen] = None,
        pen_tick: Optional[QPen] = None
    ) -> None:
        axis = self.plot_item.getAxis(axis_name)
        
        # Relative Positioning: Ensure Axis is above grid/plot items
        axis.setZValue(1000)

        axis.setPen(pen_spine if pen_spine is not None else self._pen_spine)
        axis.setTickPen(pen_tick if pen_tick is not None else self._pen_tick)
        
        tick_qfont, _, _, _ = self._parse_css_style(tick_style_dict)
        axis.setTickFont(tick_qfont)
        
        if 'color' in tick_style_dict:
            axis.setTextPen(self._to_qcolor(tick_style_dict['color']))

        _, label_css_opts, _, _ = self._parse_css_style(label_style_dict)
        if label_css_opts:
            axis.setLabel(**label_css_opts)

        final_length = 0 if not show_ticks else (abs(tick_length) if tick_direction == 'out' else -abs(tick_length))
        axis.setStyle(tickLength=final_length, showValues=show_values)

    def set_title_style(self, style: StyleDictType = {}, bottom_spacing: int = 0) -> None:
        title_text = self.plot_item.titleLabel.text
        _, _, title_kwargs, underline = self._parse_css_style(style)
        title_text = self._wrap_html(title_text, underline)
        self.plot_item.setTitle(title_text, **title_kwargs)
        if self.plot_item.layout:
            self.plot_item.layout.setRowSpacing(0, float(bottom_spacing))
            self.plot_item.titleLabel.setContentsMargins(0, 0, 0, 0)

    # --- Functionality: Coupling ---
    def set_view_coupling(self, link_x: bool = True, link_y: bool = False) -> None:
        """
        Configure mouse interaction coupling.
        Args:
            link_x (bool): Standard X-Linkage (identical ranges).
            link_y (bool): If True, enables *proportional* Y-synchronization via SyncViewBox.
                           If False, Y axes are independent.
        """
        self._link_x = link_x
        self._link_y = link_y
        
        if self.vb_right is not None:
            # 1. Handle X-Link (Standard Mechanism)
            # Both views share the exact same X-Range
            target = self.plot_item.vb
            self.vb_right.setXLink(target if link_x else None)
            
            # 2. Handle Y-Link (Proportional Sync Mechanism)
            # We do NOT use setYLink (which forces identical ranges).
            # Instead, we set the 'peer_view' on our custom SyncViewBoxes.
            if link_y:
                # Bi-directional sync
                if isinstance(self.plot_item.vb, SyncViewBox):
                    self.plot_item.vb.set_peer_view(self.vb_right)
                self.vb_right.set_peer_view(self.plot_item.vb)
            else:
                # Disable sync
                if isinstance(self.plot_item.vb, SyncViewBox):
                    self.plot_item.vb.set_peer_view(None)
                self.vb_right.set_peer_view(None)

    def _init_right_view(self) -> None:
        """Helper to initialize the secondary ViewBox if it doesn't exist."""
        if self.vb_right is not None:
            return

        # Create the secondary view using our CUSTOM SyncViewBox
        self.vb_right = SyncViewBox()
        self.vb_right.setBackgroundColor(None)
        
        # Apply coupling immediately
        self.set_view_coupling(self._link_x, self._link_y)
        
        self.plot_item.scene().addItem(self.vb_right)
        self.plot_item.getAxis('right').linkToView(self.vb_right)
        self.plot_item.showAxis('right')
        
        # Relative Positioning: Ensure vb_right geometry tracks main view exactly
        update_geo = lambda: self.vb_right.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self._vb_right_connection = self.plot_item.vb.sigResized.connect(update_geo)
        update_geo()
        
        self._set_active_view('left')

    def set_axis_label(self, axis_key: str, text: str, units: Optional[str] = None, **style_kwargs) -> None:
        mapping = {'y-axis': 'left', 'x-axis': 'bottom', 'y2-axis': 'right', 'x-axis flipped': 'top',
                   'left': 'left', 'bottom': 'bottom', 'right': 'right', 'top': 'top'}
        if axis_key not in mapping: raise ValueError(f"Unknown axis: {axis_key}")
        target_axis = mapping[axis_key]
        
        base_style = self._axis_style_config.get('axis_label_style', {}).copy()
        base_style.update(style_kwargs) 
        _, label_css_opts, _, underline = self._parse_css_style(base_style)
        text = self._wrap_html(text, underline)

        if axis_key == 'x-axis flipped':
            self.plot_item.showAxis('top', True)
            self.plot_item.getAxis('bottom').setStyle(showValues=False)
            self.plot_item.setLabel('bottom', '')
            self.plot_item.getAxis('top').setStyle(showValues=True)
            
        if axis_key == 'y2-axis':
            # Use shared initialization logic
            self._init_right_view()
            
            if 'color' in base_style:
                c = self._to_qcolor(base_style['color'])
                current_spine_width = self._axis_style_config.get('spine_width', 1.5)
                current_tick_width = self._axis_style_config.get('tick_width', 1.5)
                
                pen_spine = QPen(c)
                pen_spine.setWidthF(current_spine_width)
                pen_spine.setCosmetic(True)

                pen_tick = QPen(c)
                pen_tick.setWidthF(current_tick_width)
                pen_tick.setCosmetic(True)

                self.plot_item.getAxis('right').setPen(pen_spine)
                self.plot_item.getAxis('right').setTickPen(pen_tick)
                self.plot_item.getAxis('right').setZValue(1000)

        self.plot_item.setLabel(target_axis, text, units=units, **label_css_opts)

    def disable_secondary_axis(self) -> None:
        if self.vb_right is None:
            return
        if self._vb_right_connection:
            try:
                self.plot_item.vb.sigResized.disconnect(self._vb_right_connection)
            except (TypeError, RuntimeError):
                pass
            self._vb_right_connection = None
        
        # Cleanup coupling
        if isinstance(self.plot_item.vb, SyncViewBox):
            self.plot_item.vb.set_peer_view(None)
            
        right_axis = self.plot_item.getAxis('right')
        right_axis.linkToView(None)
        right_axis.hide()
        self.plot_item.scene().removeItem(self.vb_right)
        self.vb_right = None
        self._set_active_view('left')

    # --- INTERACTION / SELECTION LOGIC ---
    
    def _set_active_view(self, view_name: ViewMode) -> None:
        """Switches focus between left and right axes."""
        self._active_view_mode = view_name
        
        if view_name == 'right' and self.vb_right is None:
            self._active_view_mode = 'left'
    
        # 1. Visual Feedback: Proportional Scaling
        left_axis = self.plot_item.getAxis('left')
        right_axis = self.plot_item.getAxis('right')
        
        base_width = self._axis_style_config.get('spine_width')
        if base_width is None:
             base_width = self._axis_style_config.get('width', 1.5)
            
        # Proportional Scaling: Use user-defined factor
        active_width: float = base_width * self._highlight_factor
        
        left_pen = QPen(left_axis.pen())
        left_pen.setWidthF(active_width if self._active_view_mode == 'left' else base_width)
        left_pen.setCapStyle(Qt.PenCapStyle.FlatCap if self._active_view_mode == 'left' else Qt.PenCapStyle.SquareCap)
        left_axis.setPen(left_pen)
    
        if self.vb_right is not None:
            right_pen = QPen(right_axis.pen())
            right_pen.setWidthF(active_width if self._active_view_mode == 'right' else base_width)
            right_pen.setCapStyle(Qt.PenCapStyle.FlatCap if self._active_view_mode == 'right' else Qt.PenCapStyle.SquareCap)
            right_axis.setPen(right_pen)
    
        left_axis.update()
        right_axis.update()
    
        # 2. Event Capture: Relative Positioning (Z-Ordering)
        # We ensure the active view is strictly 'above' the inactive one in the scene stack.
        primary_vb = self.plot_item.vb
        
        if self._active_view_mode == 'left':
            primary_vb.setZValue(10)
            primary_vb.setMouseEnabled(x=True, y=True)
            if self.vb_right:
                self.vb_right.setZValue(0) # Relative: 0 < 10
                self.vb_right.setMouseEnabled(x=False, y=False)
                
        elif self._active_view_mode == 'right' and self.vb_right:
            self.vb_right.setZValue(10)
            self.vb_right.setMouseEnabled(x=True, y=True)
            primary_vb.setZValue(0) # Relative: 0 < 10
            primary_vb.setMouseEnabled(x=False, y=False)

    def _on_scene_clicked(self, event) -> None:
        if event.isAccepted(): 
            return
        pos = event.scenePos()
        left_axis = self.plot_item.getAxis('left')
        right_axis = self.plot_item.getAxis('right')
        if left_axis.sceneBoundingRect().contains(pos):
            self._set_active_view('left')
            event.accept()
        elif right_axis.isVisible() and right_axis.sceneBoundingRect().contains(pos):
            self._set_active_view('right')
            event.accept()

    def _on_curve_clicked(self, curve) -> None:
        if curve.getViewBox() == self.vb_right:
            self._set_active_view('right')
        else:
            self._set_active_view('left')
            
    def plot(self, *args, **kargs):
        kargs.setdefault('clickable', True)
        curve = self.plot_item.plot(*args, **kargs)
        curve.sigClicked.connect(self._on_curve_clicked)
        return curve
    
    def _add_item_to_right(self, item) -> None:
        """Internal method to add an item to the right ViewBox."""
        if self.vb_right is None:
            raise RuntimeError("Right axis not enabled.")
        self.vb_right.addItem(item)
        if hasattr(item, 'setClickable'):
            item.setClickable(True)
        if hasattr(item, 'sigClicked'):
            item.sigClicked.connect(self._on_curve_clicked)

    def add_plot_item(
        self, 
        x: np.ndarray | List[float], 
        y: np.ndarray | List[float], 
        color: ColorType, 
        width: float, 
        linestyle: LineStyleType, 
        name: Optional[str] = None, 
        axis: Literal["y-axis", "y2-axis"] = "y-axis"
    ) -> None:
        """
        Creates and adds a new PlotCurveItem with specified styling to the requested axis.

        Args:
            x (np.ndarray | List[float]): X-axis data.
            y (np.ndarray | List[float]): Y-axis data.
            color (ColorType): Color of the line (hex, name, or tuple).
            width (float): Width of the line.
            linestyle (LineStyleType): Style of the line ('solid', 'dashed', etc.).
            name (Optional[str]): Name of the curve for legends. Defaults to None.
            axis (Literal["y-axis", "y2-axis"]): Target axis. Defaults to "y-axis".
        """
        # 1. Resolve Pen
        qt_style = self._resolve_pen_style(linestyle)
        c = self._to_qcolor(color)
        pen = QPen(c)
        pen.setWidthF(width)
        pen.setStyle(qt_style)
        pen.setCosmetic(True) # Optimization for zooming/scaling

        # 2. Create Item
        # We explicitly enable clickability to support the active view switching logic
        item = pg.PlotCurveItem(x=x, y=y, pen=pen, name=name, clickable=True)
        item.sigClicked.connect(self._on_curve_clicked)

        # 3. Add to View
        if axis == "y2-axis":
            # Automatically activate secondary axis if not present
            if self.vb_right is None:
                self._init_right_view()
            self._add_item_to_right(item)
        else:
            # Add to primary (default) view
            self.plot_item.addItem(item)

    def set_plot_style(self, style_dict: dict) -> None:
        if 'background' in style_dict:
            bg_conf = style_dict['background']
            if isinstance(bg_conf, dict):
                if 'canvas-color' in bg_conf:
                    self.setBackground(self._to_qcolor(bg_conf['canvas-color']))
                elif 'color' in bg_conf: 
                    self.setBackground(self._to_qcolor(bg_conf['color']))
                if 'plot-color' in bg_conf:
                    self._grid_item.set_background(self._to_qcolor(bg_conf['plot-color']))
                else:
                    self._grid_item.set_background(None)
            elif isinstance(bg_conf, (str, tuple)):
                self.setBackground(self._to_qcolor(bg_conf))
                self._grid_item.set_background(None)

        if 'axis' in style_dict:
            self.set_axis_style(**style_dict['axis'])

        if 'grid' in style_dict:
            self.set_grid_style(**style_dict['grid'])
        
        if 'title' in style_dict:
            conf = style_dict['title']
            font_style = {k:v for k,v in conf.items() if k != 'bottom_spacing'}
            spacing = conf.get('bottom_spacing', 0)
            self.set_title_style(style=font_style, bottom_spacing=spacing)

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    
    try: 
        from pygraph_plotstyle import PlotStyle
    except:
        # User's custom style module import (simulated for standalone execution)
        class PlotStyle:
           def __init__(self, theme): self.current_style = {}

    
    app = QApplication.instance() or QApplication(sys.argv)

    widget = MatplotlibStylePlot(title="Grid Style Demo")
    widget.resize(1000, 700)
    
    style_config = {
            # 1. Background Config
            'background': {
                'canvas-color': '#e8e8e8', # Light Grey (Widget area)
                'plot-color': '#ffffff'    # Pure White (Data area)
                #'color': 'blue' #alternatively single keyword can be given
            },
    
            # 2. Title Config
            'title': {
                # CSS Font Properties
                'font-size': '22pt',
                'font-weight': 'normal',
                'font-style': 'italic',
                'text-decoration': 'underline', # Triggers HTML underline
                'color': '#2c3e50',
                
                # Layout Property
                'bottom_spacing': 25
            },
    
            # 3. Axis Config
            'axis': {
                # Spine / Line Geometry
                'color': '#34495e',
                'spine_alpha': 0.8,  # Transparency of the axis line
                'spine_width': 3.0,      # Axis Spine Width (Thicker) -> Decoupled!
                #'width': 1.0,           # Optional, can be used for axis and ticks
                
                
                # Tick Geometry
                'tick_length': 15,
                'tick_width': 1.0, 
                'tick_direction': 'in', # 'in' or 'out'
                'tick_alpha': 0.6,       # Transparency of the tick marks
                'show_ticks': True,      # Or specific list ['left', 'bottom']
                'show_tick_labels': ['left', 'bottom'],
    
                # Tick Text CSS Style
                'tick_label_style': {
                    'font-size': '10pt',
                    'font-weight': 'bold',
                    'font-style': 'normal',
                    'text-decoration': 'none',
                    'color': '#7f8c8d'
                },
    
                # Axis Label Text CSS Style
                'axis_label_style': {
                    'font-size': '14pt',
                    'font-weight': 'bold',
                    'font-style': 'normal',
                    'text-decoration': 'none', # Default, can be overridden per axis
                    'color': '#2c3e50'
                }
            },
            
            # 4. Grid Config
            'grid': {
                # Major Grid Lines
                'major_color': '#bbbbbb',
                'major_width': 1.5,
                'major_alpha': 0.8,
                'major_style': '--', # Dashed
    
                # Minor Grid Lines
                'minor_color': '#bbbbbb',
                'minor_width': 1.0,
                'minor_alpha': 0.6,
                'minor_style': ':'   # Dotted
            }
        }
    
    plot_styler = PlotStyle('dark')
    style_config = plot_styler.current_style
    widget.set_plot_style(style_config)

    widget.set_plot_style(style_config)

    # 3. Add Data (Primary Axis) via standard plot
    x = np.linspace(0, 10, 200)
    widget.plot(x, np.sin(x), pen=pg.mkPen('#1f77b4', width=3), name="Sine (Left)")

    # 4. Axis Labels
    widget.set_axis_label("x-axis", "Wavelength")
    widget.set_axis_label("y-axis", "Amplitude (Primary)")
    
    # 5. Add Data using NEW add_plot_item function
    # This automatically activates y2-axis
    y2 = np.cos(x) * 20 + 100 
    widget.add_plot_item(
        x=x, 
        y=y2, 
        color='#d62728', 
        width=5.0, 
        linestyle='--', # Dashed
        name="Cosine (Right - New Method)",
        axis="y2-axis"
    )
    
    # We can still label the axis afterwards
    widget.set_axis_label("y2-axis", "Secondary Scale (Right)", color='#d62728')
    
    widget.set_view_coupling(link_x=True, link_y=True)
        
    widget.show()
    sys.exit(app.exec())