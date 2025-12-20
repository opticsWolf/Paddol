# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

from typing import Dict, List, Any

# Define a TypeAlias for the nested style dictionary structure
StyleConfig = Dict[str, Any]

class PlotStyle:
    """
    Manages translation of styles to PyQtGraph configurations.
    
    Updates:
    - [2025-12-19] Refactored to support nested dictionary schema.
    - [2025-12-19] Updated colors for Qt6 Fusion compatibility and Modern design.
    - [2025-12-20] Decoupled background (canvas/plot) and axis geometry (spine/tick).
    """

    STYLES: Dict[str, StyleConfig] = {
        # -------------------------------------------------------------------------
        # 1. Light Theme (Fusion Compatible)
        # Matches standard Qt Fusion Light palette (#efefef window, #000000 text)
        # -------------------------------------------------------------------------
        'light': {
            'background': {
                'canvas-color': '#efefef',  # Standard Fusion Window Color
                'plot-color': '#ffffff'     # Pure White Data Area
            },
            'title': {
                'font-size': '14pt',
                'font-weight': 'normal',
                'font-style': 'normal',
                'text-decoration': 'none',
                'color': '#353535', # Soft Black
                'bottom_spacing': 25
            },
            'axis': {
                'color': '#353535', # Matches Window Text
                'spine_alpha': 0.8,
                'spine_width': 2.0,
                'tick_length': 8,
                'tick_width': 1.0,
                'tick_direction': 'out',
                'tick_alpha': 0.6,
                'show_ticks': True,
                'show_tick_labels': ['left', 'bottom'],
                'tick_label_style': {
                    'font-size': '10pt',
                    'font-weight': 'bold',
                    'font-style': 'normal',
                    'text-decoration': 'none',
                    'color': '#505050' # Slightly lighter than axis line
                },
                'axis_label_style': {
                    'font-size': '12pt',
                    'font-weight': 'bold',
                    'font-style': 'normal',
                    'text-decoration': 'none',
                    'color': '#353535'
                }
            },
            'grid': {
                'major_color': '#b0b0b0',
                'major_width': 1.5,
                'major_alpha': 0.6,
                'major_style': '--',
                'minor_color': '#d0d0d0',
                'minor_width': 1.0,
                'minor_alpha': 0.4,
                'minor_style': ':'
            }
        },

        # -------------------------------------------------------------------------
        # 2. Dark Theme (Fusion Compatible)
        # Matches standard Qt Fusion Dark palette (#353535 window, #ffffff text)
        # -------------------------------------------------------------------------
        'dark': {
            'background': {
                'canvas-color': '#353535',  # Standard Fusion Dark Window Color
                'plot-color': '#2b2b2b'     # Slightly darker for plot contrast
            },
            'title': {
                'font-size': '14pt',
                'font-weight': 'bold',
                'font-style': 'normal',
                'text-decoration': 'none',
                'color': '#ffffff', # Pure White
                'bottom_spacing': 15
            },
            'axis': {
                'color': '#ffffff',
                'spine_alpha': 0.9,
                'spine_width': 3.0,
                'tick_length': 8,
                'tick_width': 1.0,
                'tick_direction': 'in',
                'tick_alpha': 0.8,
                'show_ticks': ['left', 'bottom'],
                'show_tick_labels': ['left', 'bottom'],
                'tick_label_style': {
                    'font-size': '10pt',
                    'font-weight': 'normal',
                    'font-style': 'normal',
                    'text-decoration': 'none',
                    'color': '#dddddd' # Off-white for ticks
                },
                'axis_label_style': {
                    'font-size': '12pt',
                    'font-weight': 'bold',
                    'font-style': 'normal',
                    'text-decoration': 'none',
                    'color': '#ffffff'
                }
            },
            'grid': {
                'major_color': '#ffffff',
                'major_width': 1.0,
                'major_alpha': 0.15,
                'major_style': '--',
                'minor_color': '#ffffff',
                'minor_width': 1.0,
                'minor_alpha': 0.05,
                'minor_style': ':'
            }
        },

        # -------------------------------------------------------------------------
        # 3. Grey Theme (Modern Tones)
        # Uses "Flat UI" Palette: Platinum background with Gunmetal/Dim Gray text
        # -------------------------------------------------------------------------
        'grey': {
            'background': {
                'canvas-color': '#dfe6e9',  # "City Lights" Platinum
                'plot-color': '#f5f6fa'     # Near White for clean data look
            },
            'title': {
                'font-size': '14pt',
                'font-weight': 'bold',
                'font-style': 'italic',
                'text-decoration': 'none',
                'color': '#2d3436', # "Dracula Orchid" Dark Grey
                'bottom_spacing': 25
            },
            'axis': {
                'color': '#636e72', # "River City" Dim Grey
                'spine_alpha': 1.0,
                'spine_width': 2.0,
                'tick_length': 6,
                'tick_width': 1.0,
                'tick_direction': 'in',
                'tick_alpha': 0.8,
                'show_ticks': True,
                'show_tick_labels': ['left', 'bottom'],
                'tick_label_style': {
                    'font-size': '11pt',
                    'font-weight': 'normal',
                    'font-style': 'normal',
                    'text-decoration': 'none',
                    'color': '#636e72'
                },
                'axis_label_style': {
                    'font-size': '12pt',
                    'font-weight': 'bold',
                    'font-style': 'normal',
                    'text-decoration': 'none',
                    'color': '#2d3436'
                }
            },
            'grid': {
                'major_color': '#b2bec3', # "Soothing Breeze" Grey
                'major_width': 1.2,
                'major_alpha': 0.5,
                'major_style': '-', 
                'minor_color': '#b2bec3',
                'minor_width': 1.0,
                'minor_alpha': 0.2,
                'minor_style': '-'
            }
        },

        # -------------------------------------------------------------------------
        # 4. Modern Theme (Bold Contrast + Sprinkle of Color)
        # High contrast Charcoal background with Electric Blue & Emerald Green accents
        # -------------------------------------------------------------------------
        'modern': {
            'background': {
                'canvas-color': '#121212',  # Deep Matte Black
                'plot-color': '#1e1e1e'     # Dark Grey "Card" color
            },
            'title': {
                'font-size': '14pt',
                'font-weight': 'bold',
                'font-style': 'normal',
                'text-decoration': 'underline',
                'color': '#3498db', # SPRINKLE: Electric Blue Title
                'bottom_spacing': 30
            },
            'axis': {
                'color': '#ecf0f1', # Stark White
                'spine_alpha': 1.0,
                'spine_width': 2.5,
                'tick_length': 10,
                'tick_width': 1.5,  # Thicker ticks for modern look
                'tick_direction': 'out',
                'tick_alpha': 1.0,
                'show_ticks': True,
                'show_tick_labels': ['left', 'bottom'],
                'tick_label_style': {
                    'font-size': '11pt',
                    'font-weight': 'bold',
                    'font-style': 'normal',
                    'text-decoration': 'none',
                    'color': '#bdc3c7' # Silver
                },
                'axis_label_style': {
                    'font-size': '13pt',
                    'font-weight': '900', # Extra Bold
                    'font-style': 'normal',
                    'text-decoration': 'none',
                    'color': '#2ecc71' # SPRINKLE: Emerald Green Labels
                }
            },
            'grid': {
                'major_color': '#34495e', # Dark Blue-Grey Grid
                'major_width': 1.5,
                'major_alpha': 0.4,
                'major_style': '--',
                'minor_color': '#34495e',
                'minor_width': 1.0,
                'minor_alpha': 0.2,
                'minor_style': ':'
            }
        }
    }

    def __init__(self, current_style: str = "light") -> None:
        """
        Initialize the PlotStyler.
        """
        self._current_style_name: str = (
            current_style if current_style in self.STYLES else "light"
        )

    @property
    def current_style(self) -> StyleConfig:
        return self.STYLES[self._current_style_name]

    def set_style(self, style_name: str) -> None:
        if style_name in self.STYLES:
            self._current_style_name = style_name
        else:
            print(f"Warning: Style '{style_name}' not found. "
                  f"Keeping '{self._current_style_name}'.")

    def get_available_styles(self) -> List[str]:
        return list(self.STYLES.keys())