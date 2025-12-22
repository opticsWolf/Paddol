import numpy as np
import json
from numba import njit, float64
from typing import Tuple, Dict, Union, Any
from pathlib import Path
import sys

# Type aliases
NDArrayFloat = np.typing.NDArray[np.float64]

@njit(cache=True, fastmath=True)
def _mix_mesopic_curves(
    photopic: NDArrayFloat,
    scotopic: NDArrayFloat,
    m: float
) -> NDArrayFloat:
    """
    Blends V(lambda) and V'(lambda) based on adaptation coefficient m.
    Formula: V_mes(lambda) = normalized( m * V + (1-m) * V' )
    """
    n = len(photopic)
    result = np.empty(n, dtype=np.float64)
    max_val = 0.0

    # 1. Linear combination
    for i in range(n):
        val = m * photopic[i] + (1.0 - m) * scotopic[i]
        result[i] = val
        if val > max_val:
            max_val = val

    # 2. Normalize to Peak=1.0
    if max_val > 1e-9:
        inv_max = 1.0 / max_val
        for i in range(n):
            result[i] *= inv_max
            
    return result

@njit(cache=True, fastmath=True)
def _integrate_spectrum(
    spectral_data: NDArrayFloat, 
    lef: NDArrayFloat, 
    km: float, 
    wl_step: float = 1.0
) -> float:
    """
    Calculates Luminous Flux: Phi_v = Km * sum(Phi_e(lambda) * V(lambda) * d_lambda)
    """
    total = 0.0
    n = len(spectral_data)
    # Assumes inputs are already aligned
    for i in range(n):
        total += spectral_data[i] * lef[i]
    return total * km * wl_step

class CIEDataHandler:
    """
    Parses specific CIE JSON formats, robustly handling missing metadata by 
    falling back to the 'lambda' definition.
    """
    def __init__(self, master_wl_start: int = 360, master_wl_end: int = 830):
        self.wl_start = master_wl_start
        self.wl_end = master_wl_end
        self.count = self.wl_end - self.wl_start + 1
        self.wavelengths = np.linspace(self.wl_start, self.wl_end, self.count)
        
    def load_aligned_spectrum(self, json_content: dict, quantity_key: str) -> NDArrayFloat:
        """
        Loads spectral data (e.g. V(lambda)).
        
        ROBUSTNESS: 
        1. Checks the specific quantity node for 'wavelength_first'.
        2. If missing, falls back to 'data.lambda.wavelength_first'.
        3. Validates against ':unap'.
        """
        data_section = json_content.get('data', {})
        data_node = data_section.get(quantity_key)
        
        if not data_node:
            raise KeyError(f"Key '{quantity_key}' not found in JSON data.")

        # 1. Try getting metadata from the quantity itself
        wl_first = data_node.get('wavelength_first')
        
        # 2. Fallback: Try getting metadata from the 'lambda' node (common in CIE files)
        if wl_first is None:
            lambda_node = data_section.get('lambda')
            if lambda_node:
                wl_first = lambda_node.get('wavelength_first')
                
        # 3. Final validation
        if wl_first is None:
             raise ValueError(f"Could not find 'wavelength_first' for '{quantity_key}' or in 'lambda' node.")
        
        if isinstance(wl_first, str) and wl_first == ":unap":
            raise ValueError(
                f"Data for '{quantity_key}' is marked ':unap' (scalar). "
                "Use 'load_lookup_table' instead."
            )
            
        raw_values = np.array(data_node['values'], dtype=np.float64)
        
        # Align to master grid (pad with zeros)
        aligned = np.zeros(self.count, dtype=np.float64)
        
        # Calculate offsets
        start_idx = int(wl_first) - self.wl_start
        end_idx = start_idx + len(raw_values)
        
        # Intersection logic
        target_start = max(0, start_idx)
        target_end = min(self.count, end_idx)
        source_start = max(0, -start_idx)
        source_end = source_start + (target_end - target_start)
        
        if target_end > target_start:
            aligned[target_start:target_end] = raw_values[source_start:source_end]
            
        return aligned

    def load_lookup_table(self, json_content: dict, x_key: str, y_key: str) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """
        Loads scalar lookup tables (e.g. m vs Km). 
        Ignores wavelength metadata.
        """
        try:
            x_data = json_content['data'][x_key]['values']
            y_data = json_content['data'][y_key]['values']
        except KeyError as e:
            raise KeyError(f"Lookup table keys not found: {e}")

        return (
            np.array(x_data, dtype=np.float64),
            np.array(y_data, dtype=np.float64)
        )
class MesopicEngine:
    """
    Principal calculation engine for Photopic, Scotopic, and Mesopic vision.
    """
    def __init__(self, 
                 photopic_json: dict, 
                 scotopic_json: dict, 
                 max_sle_json: dict):
        
        self.handler = CIEDataHandler(360, 830)
        
        # 1. Load Spectral Curves (Strict Validation)
        self.v_photopic = self.handler.load_aligned_spectrum(photopic_json, 'V(lambda)')
        self.v_scotopic = self.handler.load_aligned_spectrum(scotopic_json, "V'(lambda)")
        
        # 2. Load Scalar Efficacy Table (Safe for ':unap')
        #
        self.m_grid, self.km_grid = self.handler.load_lookup_table(
            max_sle_json, 'm', 'K_m,mes;m'
        )
        
        # Cache standard constants using interpolation for precision
        self.km_photopic = np.interp(1.0, self.m_grid, self.km_grid) # ~683
        self.km_scotopic = np.interp(0.0, self.m_grid, self.km_grid) # ~1700

    def calculate_mesopic_flux(self, spectral_radiance: NDArrayFloat, m: float) -> float:
        """
        Calculates Luminous Flux (Lumens) for a given adaptation coefficient m.
        """
        # Optimized hot-paths for standard observers
        if m >= 1.0:
            return _integrate_spectrum(spectral_radiance, self.v_photopic, self.km_photopic)
        if m <= 0.0:
            return _integrate_spectrum(spectral_radiance, self.v_scotopic, self.km_scotopic)
            
        # Mesopic case
        curve = _mix_mesopic_curves(self.v_photopic, self.v_scotopic, m)
        km = np.interp(m, self.m_grid, self.km_grid)
        return _integrate_spectrum(spectral_radiance, curve, km)

    # Convenience wrappers
    def calculate_photopic_flux(self, spectral_radiance: NDArrayFloat) -> float:
        return self.calculate_mesopic_flux(spectral_radiance, 1.0)

    def calculate_scotopic_flux(self, spectral_radiance: NDArrayFloat) -> float:
        return self.calculate_mesopic_flux(spectral_radiance, 0.0)

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # 1. Construct Path: Up one level -> src -> CIE -> lef
    # Path(__file__) is the script file. .parent is the dir. .parent.parent is the root.
    base_path = Path(__file__).resolve().parent.parent / "src" / "CIE" / "lef"
    
    # Define files
    files = {
        "photopic": base_path / "CIE_sle_photopic_V.json",
        "scotopic": base_path / "CIE_sle_scotopic_V.json",
        "mesopic":  base_path / "CIE_max_sle_mesopic.json"
    }

    # 2. Check existence
    missing = [f.name for f in files.values() if not f.exists()]
    if missing:
        print(f"Error: Missing files in {base_path}: {missing}")
        sys.exit(1)

    # 3. Load JSONs
    #try:
    data = {}
    for key, path in files.items():
        with open(path, 'r', encoding='utf-8') as f:
            data[key] = json.load(f)
            
    # 4. Initialize Engine
    engine = MesopicEngine(data["photopic"], data["scotopic"], data["mesopic"])
    print(f"Engine Initialized. Loaded data from: {base_path}")
    
    # 5. Run Example: 1 Watt uniform spectrum (360-830nm)
    # 471 points from 360 to 830 inclusive
    dummy_spectrum = np.ones(471, dtype=np.float64) 
    
    p_lumens = engine.calculate_photopic_flux(dummy_spectrum)
    s_lumens = engine.calculate_scotopic_flux(dummy_spectrum)
    m_lumens = engine.calculate_mesopic_flux(dummy_spectrum, 0.5)
    
    print("\n--- Performance Test Results ---")
    print(f"Spectrum: Uniform 1.0 W/nm (360-830nm)")
    print(f"Photopic (m=1.0): {p_lumens:,.2f} lm  (Standard Daylight)")
    print(f"Scotopic (m=0.0): {s_lumens:,.2f} lm  (Darkness)")
    print(f"Mesopic  (m=0.5): {m_lumens:,.2f} lm  (Twilight)")
        
    #except Exception as e:
    #    print(f"Runtime Error: {e}")