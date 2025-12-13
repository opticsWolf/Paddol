# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 23:03:37 2025

@author: Frank
"""

import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
from typing import List

__all__ = ["ColorMath", "HarmonyEngine"]

class ColorMath:
    """High-performance vectorized color conversion and utility methods."""
    
    @staticmethod
    def hsv_to_rgb_vectorized(h, s, v):
        """Converts HSV to RGB using NumPy vectorization.
        
        Args:
            h: Hue (0.0 - 1.0), scalar or numpy array.
            s: Saturation (0.0 - 1.0), scalar or numpy array.
            v: Value (0.0 - 1.0), scalar or numpy array.
            
        Returns:
            Tuple of (r, g, b) where values are 0-255.
        """
        h = np.asarray(h)
        h6 = h * 6.0
        r_base = np.clip(np.abs(h6 - 3) - 1, 0, 1)
        g_base = np.clip(2 - np.abs(h6 - 2), 0, 1)
        b_base = np.clip(2 - np.abs(h6 - 4), 0, 1)
        s_inv = 1.0 - s
        red   = v * (s_inv + s * r_base) * 255
        green = v * (s_inv + s * g_base) * 255
        blue  = v * (s_inv + s * b_base) * 255
        if h.ndim == 0: return red, green, blue
        return red, green, blue

    @staticmethod
    def get_contrast_color(r, g, b) -> QColor:
        """Calculates luminance to return optimal text contrast color (Black/White)."""
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return Qt.GlobalColor.black if lum > 140 else Qt.GlobalColor.white


class HarmonyEngine:
    """Logic for calculating color harmony relationships."""
    
    RELATIONSHIPS = {
        "Single":               [0],
        "Analogous":            [0, 30, 60],
        "Complementary":        [0, 180],
        "Split-Complementary":  [0, 150, 210],
        "Triadic":              [0, 120, 240],
        "Double-Complementary": [0, 30, 180, 210],
        "Square":               [0, 90, 180, 270],
        "Tetradic":             [0, 60, 180, 240],
        
        # --- NEW 5-Point Methods ---
        
        # 1. Pentagonal: Perfectly balanced, high contrast, vibrant.
        # Spaced every 72 degrees (360 / 5)
        "Pentagonal":           [0, 72, 144, 216, 288],
        
        # 2. 5-Tone Analogous: Very low contrast, unified look.
        # Extends the analogous run further
        "Analogous 5-Tone":     [0, 30, 60, 90, 120],
        
        # 3. Star (Split-Analogous): Complex, rich harmony.
        # Base (0), two split complements (150, 210), and two wide accents (72, 288 approx)
        # Or simpler: Base, +/- 30 (Analogous), +/- 150 (Split Comp)
        "Star 5-Tone":          [0, 30, 330, 150, 210], # 330 is -30 normalized
    }

    @staticmethod
    def get_harmonies(base_hue: float, mode: str) -> List[float]:
        """Generates a list of hue values based on the selected harmony mode.
        
        Args:
            base_hue: A float between 0.0 and 1.0 representing the starting color.
            mode: The dictionary key for the harmony rule.
            
        Returns:
            List[float]: A list of hue values (0.0-1.0).
        """
        offsets = HarmonyEngine.RELATIONSHIPS.get(mode, [0])
        
        # Calculate hues and ensure they wrap around 1.0 using modulo
        return [(base_hue + (deg / 360.0)) % 1.0 for deg in offsets]