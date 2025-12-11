"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import json
from pathlib import Path
import numpy as np
from typing import List, Tuple, Union, Dict, Any, Optional

__all__ = ["COLORMAP_PRESETS", "JSON_FILENAME", "OklabEngine", "ColorMapsPresetManager", "COLORMAPS", "COLORMAP_DICT" ]
    
COLORMAP_PRESETS = {
        "Viridis": {"colors": ["#440154", "#21918c", "#fde725"], "mode": "strict"},
        "Plasma":  {"colors": ["#0d0887", "#cc4778", "#f0f921"], "mode": "strict"},
        "Inferno": {"colors": ["#000004", "#bb3754", "#fca50a", "#fcffa4"], "mode": "strict"},
        "Magma":   {"colors": ["#000004", "#51127c", "#b73779", "#fcfdbf"], "mode": "strict"},
        "Cividis": {"colors": ["#002051", "#757633", "#fdea45"], "mode": "strict"},
        "Turbo":   {"colors": ["#30123b", "#4686fb", "#1ae4b6", "#a2fc3c", "#fbb41a", "#e34509", "#7a0403"], "mode": "strict"},
    }

JSON_FILENAME = "presets.json"

# --- Color Math Library (Oklab) ---
class OklabEngine:
    """
    Implements vectorized sRGB <-> Oklab color space conversions and
    perceptually uniform gradient generation using NumPy.
    """
    # M1: sRGB Linear -> LMS
    M1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ]).T
    # M2: LMS Cube Root -> Oklab
    M2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ]).T
    
    # Pre-calculated inverse matrices for performance
    M2_INV = np.linalg.inv(M2.T).T
    M1_INV = np.linalg.inv(M1.T).T

    @staticmethod
    def srgb_to_oklab(rgb_array: np.ndarray) -> np.ndarray:
        linear_rgb = np.where(
            rgb_array <= 0.04045, 
            rgb_array / 12.92, 
            np.power((rgb_array + 0.055) / 1.055, 2.4)
        )
        lms = np.dot(linear_rgb, OklabEngine.M1)
        return np.dot(np.cbrt(lms), OklabEngine.M2)

    @staticmethod
    def oklab_to_srgb(lab_array: np.ndarray) -> np.ndarray:
        lms = np.power(np.dot(lab_array, OklabEngine.M2_INV), 3)
        linear_rgb = np.dot(lms, OklabEngine.M1_INV)
        linear_rgb_clamped = np.clip(linear_rgb, 0.0, 1.1)
        srgb = np.where(
            linear_rgb_clamped <= 0.0031308, 
            12.92 * linear_rgb_clamped, 
            1.055 * np.power(linear_rgb_clamped, 1.0 / 2.4) - 0.055
        )
        return np.clip(srgb, 0.0, 1.0)

    @staticmethod
    def _normalize_input(colors: Union[List[str], List[Tuple[float, float, float]]]) -> np.ndarray:
        parsed_rgb = []
        for c in colors:
            # 1. Handle Hex String
            if isinstance(c, str):
                c = c.lstrip('#')
                if len(c) == 6:
                    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                    parsed_rgb.append([r / 255.0, g / 255.0, b / 255.0])
                else:
                    raise ValueError(f"Invalid Hex format: {c}")
            
            # 2. Handle Tuple/List
            elif isinstance(c, (tuple, list, np.ndarray)):
                c_arr = np.array(c, dtype=np.float32)
                if np.any(c_arr > 1.0):
                    c_arr /= 255.0
                parsed_rgb.append(c_arr)
            else:
                raise ValueError(f"Unsupported color format: {type(c)}")

        return np.array(parsed_rgb, dtype=np.float32)

    @staticmethod
    def generate_gradient(
        colors: Union[List[str], List[Tuple[float, float, float]]], 
        mode: str = "strict", 
        n_steps: int = 1024
    ) -> np.ndarray:
        if len(colors) < 2:
            raise ValueError("At least 2 colors are required to form a gradient.")

        rgb_anchors = OklabEngine._normalize_input(colors)
        lab_anchors = OklabEngine.srgb_to_oklab(rgb_anchors)

        if mode == "luma":
            lab_anchors[:, 0] = np.linspace(
                lab_anchors[0, 0], lab_anchors[-1, 0], len(lab_anchors)
            )
        elif mode == "balanced":
            l_strict = lab_anchors[:, 0]
            l_linear = np.linspace(l_strict[0], l_strict[-1], len(l_strict))
            lab_anchors[:, 0] = (0.5 * l_strict) + (0.5 * l_linear)

        x = np.linspace(0.0, 1.0, len(lab_anchors))
        t = np.linspace(0.0, 1.0, n_steps)
        
        lab_final = np.stack([
            np.interp(t, x, lab_anchors[:, 0]), # L
            np.interp(t, x, lab_anchors[:, 1]), # a
            np.interp(t, x, lab_anchors[:, 2])  # b
        ], axis=1)

        return OklabEngine.oklab_to_srgb(lab_final)

    @staticmethod
    def generate_matplotlib_cmap(
        colors: Union[List[str], List[Tuple[float, float, float]]], 
        mode: str = "strict", 
        n_steps: int = 256,
    ) -> np.ndarray:
        """
        Generates a NumPy array formatted for direct use by matplotlib.colors.ListedColormap.

        This function generates a perceptually uniform color gradient, adds the 
        alpha channel (full opacity), and returns the raw data structure.
        Crucially, this function has NO external dependencies beyond NumPy.

        Args:
            colors: List of anchor colors.
            mode: Interpolation mode ("strict", "luma", or "balanced").
            n_steps: The number of steps (color resolution) in the output array.

        Returns:
            A NumPy array of shape (n_steps, 4) with normalized float values 
            [R, G, B, A] in the range [0.0, 1.0].
        """
        # 1. Call the high-performance gradient generator
        # Output is (N, 3) normalized sRGB floats [0.0, 1.0]
        rgb_normalized = OklabEngine.generate_gradient(colors, mode, n_steps)
        
        # 2. Add the alpha channel (full opacity: 1.0)
        # Performance: Vectorized creation and stacking is fast.
        alpha_channel = np.ones((n_steps, 1), dtype=np.float32)
        
        # 3. Combine and return the final (N, 4) array
        # Output format is (N, 4) [R, G, B, A] normalized floats [0.0, 1.0]
        return np.hstack([rgb_normalized, alpha_channel])

# Assuming this method belongs to a class, let's represent the class structure
class ColorMapsPresetManager:
    """
    Manages custom color map presets by loading, validating, and saving 
    them to a JSON file.

    Attributes:
        JSON_FILENAME: The string name/path of the preset file.
        custom_presets: Dictionary storing the loaded presets ({name: {colors, mode}}).
    """
    def __init__(self, filename: str = "presets.json"):
        self.JSON_FILENAME: str = filename
        self.custom_presets: Dict[str, Any] = {}

    def load_custom_presets_from_file(self) -> Optional[Exception]:
        """
        Loads custom presets from the specified JSON file if it exists.

        Returns:
            Optional[Exception]: Returns the caught Exception object if a read or 
            decoding error occurs. Returns None on successful completion or 
            if the file does not exist.
        """
        presets_path = Path(self.JSON_FILENAME)
        if presets_path.exists():
            try:
                with presets_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, dict):
                        self.custom_presets = data
                        return self.custom_presets  # Success
                    
                    # If file read successfully but content is not a dict
                    return ValueError("Preset file content is not a valid dictionary.")
                        
            except (OSError, json.JSONDecodeError) as e:
                # Return the specific exception object
                return e

    def save_custom_presets_to_file(self) -> Optional[Exception]:
        """
        Writes the current contents of self.custom_presets to the JSON file.

        The file is written with an indentation of 4.

        Returns:
            Optional[Exception]: Returns the caught Exception object if the 
            save operation fails (e.g., permission error, I/O error). 
            Returns None on successful completion.
        """
        presets_path = Path(self.JSON_FILENAME)
        try:
            with presets_path.open('w', encoding='utf-8') as f:
                json.dump(self.custom_presets, f, indent=4)
            return None
        except Exception as e:
            # Return the exception object for the caller to handle
            return e
        
try:
    cmap_mnager = ColorMapsPresetManager(JSON_FILENAME)
    custom_colormaps_presets = cmap_mnager.load_custom_presets_from_file()
    COLORMAP_PRESETS.update(custom_colormaps_presets)
    cmap_generator = OklabEngine()
    COLORMAP_DICT = {}
    COLORMAPS = []
    for key, value in COLORMAP_PRESETS.items():
        COLORMAPS.append(key)
        COLORMAP_DICT[key] = cmap_generator.generate_matplotlib_cmap(value["colors"], value["mode"])
except Exception as e:
    print(f"Error initializing colormaps: {e}")