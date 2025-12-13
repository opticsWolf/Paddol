"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import json
from pathlib import Path
import numpy as np
from typing import List, Tuple, Union, Dict, Any, Optional, Final

# Define exports explicitly
__all__ = [
    "OklabEngine",
    "ColorMapsPresetManager",
]

# Immutable default configuration.
# We use a private constant for the source of truth.
_DEFAULT_PRESETS: Final[Dict[str, Dict[str, Any]]] = {
    "Viridis": {"colors": ["#440154", "#21918c", "#fde725"], "mode": "strict"},
    "Plasma":  {"colors": ["#0d0887", "#cc4778", "#f0f921"], "mode": "strict"},
    "Inferno": {"colors": ["#000004", "#bb3754", "#fca50a", "#fcffa4"], "mode": "strict"},
    "Magma":   {"colors": ["#000004", "#51127c", "#b73779", "#fcfdbf"], "mode": "strict"},
    "Cividis": {"colors": ["#002051", "#757633", "#fdea45"], "mode": "strict"},
    "Turbo":   {"colors": ["#30123b", "#4686fb", "#1ae4b6", "#a2fc3c",
                           "#fbb41a", "#e34509", "#7a0403"],"mode": "strict"},
}

CMAP_PRESET_JSON = "colorengine/presets.json"

# The public, mutable dictionary that will hold defaults + custom loaded presets
COLORMAP_PRESETS: Dict[str, Dict[str, Any]] = _DEFAULT_PRESETS.copy()

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
    them to a JSON file. Features deep validation and result caching.

    Attributes:
        json_filename: The string name/path of the preset file.
        custom_presets: Dictionary storing the loaded presets.
        combined_presets: Cached dictionary of Defaults + Customs.
    """
    def __init__(self):
        self.json_filename: str = CMAP_PRESET_JSON
        self.custom_presets: Dict[str, Any] = None
        # default_presets is for reference, but we merge into the global COLORMAP_PRESETS externally
        self.default_presets: Dict[str, Any] = COLORMAP_PRESETS
        
        # Cache for the merged dictionary
        self.combined_presets: Optional[Dict[str, Any]] = None

    def _create_or_reset_file(self) -> Optional[Exception]:
        path = Path(self.json_filename)
        try:
            with path.open('w', encoding='utf-8') as f:
                json.dump({}, f, indent=4)
            self.custom_presets = {}
            print(f"[{self.__class__.__name__}] Note: Created/Reset {self.json_filename}")
            return None
        except OSError as e:
            return e

    def _is_valid_hex_string(self, s: str) -> bool:
        """
        Validates if a string is a strict '#RRGGBB' hex code.
        """
        if not s.startswith("#"):
            return False
        if len(s) != 7:
            return False
        try:
            # Attempt to interpret the substring as a base-16 integer
            int(s[1:], 16)
            return True
        except ValueError:
            return False

    def _validate_and_filter_presets(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Iterates through loaded data, performs deep validation on colors, 
        and keeps only fully valid presets.
        """
        valid_presets = {}
        
        for name, content in raw_data.items():
            # 1. Structure Checks
            if not isinstance(content, dict):
                print(f"[{self.__class__.__name__}] Warning: Skipping '{name}' (Expected dict)")
                continue
            
            if "colors" not in content:
                print(f"[{self.__class__.__name__}] Warning: Skipping '{name}' (Missing 'colors')")
                continue
                
            colors = content["colors"]
            if not isinstance(colors, list):
                print(f"[{self.__class__.__name__}] Warning: Skipping '{name}' ('colors' not a list)")
                continue

            if len(colors) < 2:
                print(f"[{self.__class__.__name__}] Warning: Skipping '{name}' (Need at least 2 colors)")
                continue

            # 2. Deep Color Validation
            colors_are_valid = True
            for i, c in enumerate(colors):
                is_valid = False
                
                # Check A: Hex String
                if isinstance(c, str):
                    if self._is_valid_hex_string(c):
                        is_valid = True
                
                # Check B: List/Tuple (RGB/RGBA)
                elif isinstance(c, (list, tuple)):
                    # Basic check: length 3 or 4, and numeric
                    if len(c) in (3, 4) and all(isinstance(x, (int, float)) for x in c):
                        is_valid = True

                if not is_valid:
                    print(f"[{self.__class__.__name__}] Warning: Skipping '{name}' (Invalid color format at index {i}: {c})")
                    colors_are_valid = False
                    break # Stop checking this preset

            # Only add if all colors passed
            if colors_are_valid:
                valid_presets[name] = content
            
        return valid_presets

    def load_custom_presets_from_file(self) -> Optional[Exception]:
        """
        Loads custom presets with self-healing and deep validation.
        """
        presets_path = Path(self.json_filename)
        
        if not presets_path.exists():
            return self._create_or_reset_file()

        try:
            with presets_path.open('r', encoding='utf-8') as f:
                content = f.read().strip()

                if not content:
                    return self._create_or_reset_file()
                
                data = json.loads(content)
                
                if not isinstance(data, dict):
                    print(f"[{self.__class__.__name__}] Error: Root JSON element is not a dict. Resetting.")
                    return self._create_or_reset_file()

                # Filter partial corruption with deep inspection
                self.custom_presets = self._validate_and_filter_presets(data)
                return None

        except json.JSONDecodeError:
            print(f"[{self.__class__.__name__}] Error: JSON Syntax is broken. Resetting file.")
            return self._create_or_reset_file()
        except OSError as e:
            return e
        
    def return_presets(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        Returns the cached combined presets with a priority-sorted key list.
        
        The key list is sorted such that all Default Presets appear first (alphabetically),
        followed by all Custom Presets (alphabetically). Custom presets that conflict
        with default names are suppressed in favor of the default.
    
        Returns:
            Tuple[Dict[str, Any], List[str]]: 
                - A dictionary of all combined presets.
                - A list of preset names ordered by [Sorted Defaults] + [Sorted Customs].
        """
        # 1. Trigger lazy loading of custom presets if necessary
        if self.custom_presets is None:
            self.load_custom_presets_from_file()
    
        # 2. Rebuild the combined dictionary cache if invalidated
        if getattr(self, "combined_presets", None) is None:
            # Start with custom presets as base
            # Default presets overwrite custom presets with identical names
            self.combined_presets = {**self.custom_presets, **self.default_presets}
        
        # 3. Generate the Priority Sorted Key List
        # Step A: Sort default keys alphabetically
        default_keys_sorted = sorted(self.default_presets.keys())
    
        # Step B: Sort custom keys alphabetically, excluding those that are 
        # already defined in defaults (since defaults take precedence)
        custom_keys_sorted = sorted([
            k for k in self.custom_presets.keys() 
            if k not in self.default_presets
        ])
    
        # Step C: Concatenate to maintain priority grouping
        priority_sorted_keys = default_keys_sorted + custom_keys_sorted
    
        return priority_sorted_keys, self.combined_presets

    @staticmethod
    def return_cmaps(
        colormap_presets: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates Matplotlib Colormap objects from a dictionary of preset configurations.

        Args:
            colormap_presets: A dictionary where keys are the colormap names (str) 
                and values are configuration dictionaries. Each config dict is 
                expected to contain a 'colors' key and optionally a 'mode' key.

        Returns:
            Dict[str, GeneratedColormap]: A new dictionary mapping the original 
            preset names to their newly generated Matplotlib Colormap objects.
        """
        # Modern Standards: Use a dictionary comprehension for conciseness and clarity.
        # This replaces the need for initializing 'cmap_dict = {}' and the 'for' loop.
        cmap_dict = {
            key: OklabEngine.generate_matplotlib_cmap(
                item['colors'],
                # item.get() provides a safe fallback for the 'mode' argument
                item.get("mode", "strict")
            )
            for key, item in colormap_presets.items()
        }
        
        return cmap_dict
    def save_custom_presets_to_file(self) -> Optional[Exception]:
        presets_path = Path(self.json_filename)
        try:
            with presets_path.open('w', encoding='utf-8') as f:
                json.dump(self.custom_presets, f, indent=4)
            return None
        except Exception as e:
            return e

