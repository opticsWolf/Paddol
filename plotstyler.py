# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 00:27:44 2025

@author: Frank
"""

from matplotlib.ticker import AutoMinorLocator

class PlotStyler:
    def __init__(self):
        """Initialize the plot styler with default light theme."""
        self.current_style = 'light'
        # Define all available styles in a dictionary of dictionaries
        self.styles = {
            'light': {
                'figure.facecolor': 'white',
                'figure.edgecolor': 'white',
                'figure.labelcolor': 'black',
                'figure.titlesize': 14,
                'figure.labelsize': 12,

                'axes.facecolor': 'white',
                'axes.edgecolor': 'darkgray',
                'axes.grid': True,
                'axes.labelcolor': 'black',
                'axes.titlesize': 12,
                'axes.labelsize': 12,

                'grid.color': 'gray',
                'grid.linestyle': '--',

                'xtick.color': 'black',
                'ytick.color': 'black',
                'xtick.minor.visible': True,
                'ytick.minor.visible': True,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'xtick.minor.color': None,  # Default to same as tick color
                'ytick.minor.color': None,

                'legend.facecolor': 'white',
                'legend.edgecolor': 'black'
            },
            
            'grey': {
                'figure.facecolor': '#e7e7e7',
                'figure.edgecolor': '#e7e7e7',
                'figure.labelcolor': '#555555',
                'figure.titlesize': 14,
                'figure.labelsize': 12,

                'axes.facecolor': '#e9e9e9',
                'axes.edgecolor': '#a3a3a3',
                'axes.grid': True,
                'axes.labelcolor': '#555555',
                'axes.titlesize': 12,
                'axes.labelsize': 12,

                'grid.color': '#a3a3a3',
                'grid.linestyle': '-',

                'xtick.color': '#555555',
                'ytick.color': '#555555',
                'xtick.minor.visible': True,
                'ytick.minor.visible': True,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'xtick.minor.color': None,  # Default to same as tick color
                'ytick.minor.color': None,

                'legend.facecolor': '#e9e9e9',
                'legend.edgecolor': '#e0e0e0'
            },
            
            'dark': {
                'figure.facecolor': '#222222',
                'figure.edgecolor': '#222222',
                'figure.labelcolor': 'white',
                'figure.titlesize': 14,
                'figure.labelsize': 12,

                'axes.facecolor': '#222222',
                'axes.edgecolor': '#dddddd',
                'axes.grid': True,
                'axes.labelcolor': 'white',
                'axes.titlesize': 12,
                'axes.labelsize': 12,

                'grid.color': '#555555',
                'grid.linestyle': '--',

                'xtick.color': 'white',
                'ytick.color': 'white',
                'xtick.minor.visible': True,
                'ytick.minor.visible': True,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'xtick.minor.color': None,  # Default to same as tick color
                'ytick.minor.color': None,

                'legend.facecolor': '#333333',
                'legend.edgecolor': '#dddddd'
            },
            
            'modern': {
                'figure.facecolor': '#fafafa',
                'figure.edgecolor': None,
                'figure.labelcolor': '#222222',  # From title color in original
                'figure.titlesize': 14,  # Not directly in modern style, using default
                'figure.labelsize': 12,

                'axes.facecolor': '#dadada',
                'axes.edgecolor': '#bcbcbc',
                'axes.grid': True,
                'axes.labelcolor': '#555555',
                'axes.titlesize': 10,  # From original
                'axes.labelsize': 10,   # From original

                'grid.color': '#bcbcbc',
                'grid.linestyle': '-',

                'xtick.color': '#555555',
                'ytick.color': '#555555',
                'xtick.minor.visible': True,
                'ytick.minor.visible': True,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.labelsize': 10,  # From original
                'ytick.labelsize': 10,   # From original
                'xtick.minor.color': '#999999',  # Added from code
                'ytick.minor.color': '#999999',

                'legend.facecolor': '#f7f7f7',
                'legend.edgecolor': '#e0e0e0'
            }
        }

    def get_style(self, style_name='light'):
        """Get the style dictionary for a given theme.

        Args:
            style_name: Name of the style to retrieve (default: 'light')

        Returns:
            dict: The requested style dictionary
        """
        return self.styles.get(style_name, self.styles['light'])


    def _apply_plot_style(self, canvas, axes, style_dict):
        """Apply a given style dictionary to the plot.

        Args:
            canvas: The matplotlib figure object.
            axes: The matplotlib axes object(s) to be styled (can be single or list of axes).
            style_dict: A dictionary containing style settings.
        """
        # Figure properties
        current_facecolor = canvas.figure.get_facecolor()
        canvas.figure.set_facecolor(style_dict.get('figure.facecolor', current_facecolor))
        
        current_edgecolor = canvas.figure.get_edgecolor()
        canvas.figure.set_edgecolor(style_dict.get('figure.edgecolor', current_edgecolor))

        # Handle suptitle if it exists
        if hasattr(canvas, '_suptitle') and canvas._suptitle:
            current_title = canvas._suptitle.get_text()
            canvas.suptitle(
                text=current_title,
                color=style_dict.get('figure.labelcolor', '#222222'),
                fontsize=style_dict.get('figure.titlesize', 18)
            )
            # Apply figure-level label size if present
            fig_label_size = style_dict.get('figure.labelsize')
            if fig_label_size:
                canvas._suptitle.set_fontsize(fig_label_size)

        # Ensure axes is a list
        axes_list = [axes] if not isinstance(axes, (list, tuple)) else axes

        for ax in axes_list:
            # Axes background and spines
            current_facecolor = ax.get_facecolor()
            ax.set_facecolor(style_dict.get('axes.facecolor', current_facecolor))

            edge_color = style_dict.get('axes.edgecolor')
            linewidth = 0.6

            for spine in ax.spines.values():
                if edge_color is not None:
                    spine.set_edgecolor(edge_color)
                spine.set_linewidth(linewidth)

            # Grid - only show if both axes.grid and grid.color are specified, otherwise use default
            ax.grid(
                visible=style_dict.get('axes.grid', True),
                color=style_dict.get('grid.color', '#cccccc'),
                linestyle=style_dict.get('grid.linestyle', '-'),
                linewidth=0.5,
                alpha=0.5
            )

            # Tick parameters for both x and y axes (major ticks)
            for axis in ['x', 'y']:
                direction = style_dict.get(f'{axis}tick.direction', 'in')
                params = {
                    'colors': style_dict.get(f'{axis}tick.color') or style_dict.get('axes.labelcolor'),
                    'labelcolor': style_dict.get('axes.labelcolor') or style_dict.get(f'{axis}tick.color'),
                    'labelsize': style_dict.get(f'{axis}tick.labelsize') or style_dict.get('axes.labelsize', 12),
                    'width': 0.6,
                    'length': 3,
                    'pad': 2,
                    'direction': direction
                }
                ax.tick_params(axis=axis, which='major',
                             **{k: v for k, v in params.items() if v is not None})

            # Minor ticks - handle visibility and styling separately
            show_x_minor = style_dict.get('xtick.minor.visible', True)
            show_y_minor = style_dict.get('ytick.minor.visible', True)

            x_direction = style_dict.get('xtick.direction', 'in')
            y_direction = style_dict.get('ytick.direction', 'in')

            if show_x_minor:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(axis='x', which='minor',
                             width=0.4,
                             length=1.5,
                             colors=style_dict.get('xtick.minor.color') or '#999999',
                             direction=x_direction)

            if show_y_minor:
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(axis='y', which='minor',
                             width=0.4,
                             length=1.5,
                             colors=style_dict.get('ytick.minor.color') or '#999999',
                             direction=y_direction)

            # Axis labels
            for axis in ['x', 'y']:
                label = getattr(ax, f'{axis}axis').label
                if not hasattr(label, 'get_color'):
                    continue  # In case there's no label text

                current_color = label.get_color()
                current_size = label.get_size()

                label.set_color(
                    style_dict.get('axes.labelcolor') or
                    style_dict.get(f'{axis}tick.color', current_color)
                )
                label.set_size(
                    style_dict.get(f'{axis}tick.labelsize') or
                    style_dict.get('axes.labelsize', current_size)
                )

            # Axes title if it exists
            if hasattr(ax, 'title') and ax.title:
                current_title_color = ax.title.get_color()
                current_title_size = ax.title.get_size()

                ax.title.set_color(
                    style_dict.get('axes.labelcolor') or
                    style_dict.get('figure.labelcolor', '#222222')
                )
                ax.title.set_size(style_dict.get('axes.titlesize', current_title_size))

            # Legend (if present)
            legend = ax.get_legend()
            if legend:
                frame = legend.get_frame()
                facecolor = style_dict.get('legend.facecolor') or 'white'
                edgecolor = style_dict.get('legend.edgecolor') or 'black'

                frame.set_facecolor(facecolor)
                frame.set_edgecolor(edgecolor)

    def get_styles(self):
        return self.styles.keys()

    def apply_style(self, style_key, canvas, axes) -> None:
        """Apply a tight, modern, minimal look to the plot."""
        self.current_style = style_key
        self.current_style_dict = self.styles.get(style_key, "modern")
        self._apply_plot_style(canvas, axes, self.current_style_dict)
