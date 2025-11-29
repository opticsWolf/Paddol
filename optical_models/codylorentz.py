# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from typing import Dict, Union, Optional, List

# Try importing from local structure, mock if missing for standalone usage
try:
    from .material import Material, compute_energy
except ImportError:
    # Mock for standalone testing if package structure isn't present
    class Material:
        def __init__(self, params, wavelength=None): 
            self.A = params.get('A', 0)
            if wavelength is not None: self.set_wavelength_range(wavelength)
        def set_wavelength_range(self, wl): self.wavelength = np.asarray(wl, dtype=np.float64)
        def get_params(self): return {'A': self.A}
        
    @njit(cache=True)
    def compute_energy(wavelength, h_c): return h_c / np.maximum(wavelength, 1e-9)

__all__ = ["CodyLorentz"]


# --- Numba Accelerated Helper Functions ---

@njit(cache=True, parallel=True, fastmath=True)
def kramers_kronig_maclaurin(E: np.ndarray, 
                             eps2: np.ndarray, 
                             eps_inf: float) -> np.ndarray:
    """
    Computes eps1 via Maclaurin's Method (Subtract-and-Add).
    
    This replaces the 'exclusion' method. It mathematically removes the 
    singularity at E' = E, allowing integration across the entire range 
    without skipping bins, which is crucial for accuracy on coarse grids.
    
    Formula:
        P int ... = int [ (E'*e2(E') - E*e2(E)) / (E'^2 - E^2) ]
    
    Args:
        E: Energy array [eV] (source and target are the same).
        eps2: Imaginary dielectric function array.
        eps_inf: High-frequency offset.
    """
    n = E.shape[0]
    eps1 = np.full(n, eps_inf, dtype=np.float64)
    factor = 2.0 / np.pi

    # Parallel loop for speed
    for i in prange(n):
        Ei = E[i]
        Ei_sq = Ei * Ei
        val_i = eps2[i] # eps2 at the singularity
        
        integral = 0.0
        
        for j in range(n - 1):
            Ej = E[j]
            Ej1 = E[j+1]
            
            # --- Point j ---
            denom_j = Ej**2 - Ei_sq
            # Handle Singularity Limit
            if np.abs(denom_j) < 1e-12:
                # Use neighbor approximation for the finite limit
                term_j = (E[j-1] * eps2[j-1] - Ei * val_i) / (E[j-1]**2 - Ei_sq) if j > 0 else 0.0
            else:
                term_j = (Ej * eps2[j] - Ei * val_i) / denom_j

            # --- Point j+1 ---
            denom_j1 = Ej1**2 - Ei_sq
            if np.abs(denom_j1) < 1e-12:
                term_j1 = (E[j+2] * eps2[j+2] - Ei * val_i) / (E[j+2]**2 - Ei_sq) if j+2 < n else 0.0
            else:
                term_j1 = (Ej1 * eps2[j+1] - Ei * val_i) / denom_j1
            
            # Trapezoidal Rule on smoothed function
            dE = Ej1 - Ej
            integral += 0.5 * (term_j + term_j1) * dE
            
        eps1[i] += factor * integral
        
    return eps1


@njit(cache=True)
def _compute_eps2_cody_lorentz(E: np.ndarray,
                               Eg: float, 
                               Et: float, 
                               Ep: float, 
                               A: float, 
                               Eu: float, 
                               Gamma: float) -> np.ndarray:
    """Internal helper to calculate eps2 for the Single Cody-Lorentz Model."""
    eps2 = np.zeros_like(E, dtype=np.float64)
    E_sq = E**2

    # --- 1. Band-to-Band Region (E >= Et) ---
    mask_band = E >= Et
    
    # Cody Factor: (E - Eg)^2 / E^2
    cody_factor = np.zeros_like(E)
    safe_E_sq = np.maximum(E_sq, 1e-12)
    
    mask_cody = (E > Eg) & mask_band
    cody_factor[mask_cody] = (E[mask_cody] - Eg)**2 / safe_E_sq[mask_cody]
    
    # Lorentz Term: E * Gamma / D(E)
    denom_band = (E_sq - Ep**2)**2 + (E * Gamma)**2
    lorentz_term = (E * Gamma) / np.maximum(denom_band, 1e-12)
    
    eps2[mask_band] = A * cody_factor[mask_band] * lorentz_term[mask_band]
    
    # --- 2. Urbach Tail Region (E < Et) ---
    Et_sq = Et**2
    if Et > Eg:
        cf_Et = (Et - Eg)**2 / Et_sq
    else:
        cf_Et = 0.0
        
    den_Et = (Et_sq - Ep**2)**2 + (Et * Gamma)**2
    lb_Et = (Et * Gamma) / np.maximum(den_Et, 1e-12)
    A_t = A * cf_Et * lb_Et
    
    mask_tail = E < Et
    E_tail = E[mask_tail]
    
    if E_tail.size > 0:
        valid_tail = E_tail > 1e-6
        E_valid = E_tail[valid_tail]
        
        val_tail = A_t * (Et / E_valid) * np.exp((E_valid - Et) / Eu)
        
        temp_tail = np.zeros_like(E_tail)
        temp_tail[valid_tail] = val_tail
        eps2[mask_tail] = temp_tail

    return np.maximum(eps2, 0.0)


@njit(cache=True)
def compute_nk_cody_lorentz(target_E: np.ndarray,
                             params_arr: np.ndarray,
                             eps_inf: float) -> np.ndarray:
    """
    Numba driver to compute complex refractive index with Grid Extrapolation.
    
    Automatically expands the integration grid to 0-80 eV to capture 
    UV poles correctly, then interpolates back to target_E.
    """
    Eg, Et, Ep, A, Eu, Gamma = params_arr[0], params_arr[1], params_arr[2], params_arr[3], params_arr[4], params_arr[5]
    
    # 1. Create Extended Grid (0 to 80 eV)
    # This is critical. If we only integrate 1.5-3.0 eV, we miss the Si/SiO2 poles at 4-11 eV.
    extrap_max = 80.0
    extrap_step = 0.05
    base_grid = np.arange(0.01, extrap_max, extrap_step)
    
    # Merge with target grid to ensure precision at requested points
    combined_E = np.concatenate((base_grid, target_E))
    full_E = np.sort(combined_E)
    
    # Remove duplicates to prevent div/0 (naive approach for Numba)
    # We construct a mask of unique values
    unique_mask = np.empty(full_E.shape[0], dtype=np.bool_)
    unique_mask[0] = True
    unique_mask[1:] = np.diff(full_E) > 1e-6
    calc_E = full_E[unique_mask]
    
    # 2. Calculate eps2 on FULL grid
    eps2_full = _compute_eps2_cody_lorentz(calc_E, Eg, Et, Ep, A, Eu, Gamma)
    
    # 3. Calculate eps1 via Maclaurin KK on FULL grid
    eps1_full = kramers_kronig_maclaurin(calc_E, eps2_full, eps_inf)
    
    # 4. Interpolate eps1 back to target grid
    eps1_target = np.interp(target_E, calc_E, eps1_full)
    
    # 5. Calculate eps2 exactly on target grid
    eps2_target = _compute_eps2_cody_lorentz(target_E, Eg, Et, Ep, A, Eu, Gamma)
    
    # 6. Convert to Refractive Index
    eps_complex = eps1_target + 1j * eps2_target
    return np.sqrt(eps_complex)


# --- Main Class ---

class CodyLorentz(Material):
    """
    Simple Continuous Cody-Lorentz (CCL) Model.
    """

    def __init__(self, 
                 params: Dict[str, Union[float, int]], 
                 wavelength: Optional[np.ndarray] = None):
        super().__init__({'A': params.get('A', 0.0)}, wavelength)
        
        self.epsilon_inf = float(params.get('epsilon_inf', 1.0))
        self.Eg = float(params.get('Eg', 0.0))
        self.Et = float(params.get('Et', 0.0))
        self.Ep = float(params.get('Ep', 0.0))
        self.A = float(params.get('A', 0.0))
        self.Eu = float(params.get('Eu', 0.1))
        self.Gamma = float(params.get('Gamma', 0.0))

        self._params_arr = np.array([
            self.Eg, self.Et, self.Ep, self.A, self.Eu, self.Gamma
        ], dtype=np.float64)
        
        self.h_c_by_eV_nm = 1239.8419843320028

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm) 

    def complex_refractive_index(self, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        if not hasattr(self, 'E'):
             raise AttributeError("Wavelength range must be set.")

        self.nk = compute_nk_cody_lorentz(
            self.E,
            self._params_arr,
            self.epsilon_inf
        )

        return self.nk

    def get_params(self) -> Dict[str, float]:
        return {
            'epsilon_inf': self.epsilon_inf,
            'Eg': self.Eg,
            'Et': self.Et,
            'Ep': self.Ep,
            'A': self.A,
            'Eu': self.Eu,
            'Gamma': self.Gamma
        }


# --- Execution Block: Examples and Plotting ---

if __name__ == "__main__":
    # 1. Define Spectral Range
    # 200 nm to 1200 nm
    wavelengths = np.linspace(200, 1200, 1001)

    # 2. Define Material Parameters
    
    # a-Si: Bandgap ~1.64 eV
    si_params = {
        'Eg': 1.64,
        'Et': 1.80,
        'Ep': 3.40,
        'A': 60.0,
        'Eu': 0.15,
        'Gamma': 2.4,
        'epsilon_inf': 1.0
    }

    # SiO2: Peak ~11 eV (Far UV)
    # NOTE: With grid extrapolation to 80eV, the KK integral will now SEE the 11eV peak
    # even though our calc range stops at 200nm (6.2eV). 
    # We can lower epsilon_inf closer to 1.0 (physical) because the integral provides the index.
    sio2_params = {
        'Eg': 8.0,
        'Et': 8.0,
        'Ep': 11.0,
        'A': 100.0,
        'Eu': 0.05,
        'Gamma': 0.5,
        'epsilon_inf': 1.0 # The UV pole integration will lift 'n' naturally!
    }

    # 3. Instantiate and Compute
    si_model = CodyLorentz(si_params, wavelengths)
    si_nk = si_model.complex_refractive_index()
    si_n = si_nk.real
    si_k = si_nk.imag

    sio2_model = CodyLorentz(sio2_params, wavelengths)
    sio2_nk = sio2_model.complex_refractive_index()
    sio2_n = sio2_nk.real
    sio2_k = sio2_nk.imag

    # 4. Print n and k every 100 nm
    print(f"{'Wavelength (nm)':<20} | {'Si (n)':<10} {'Si (k)':<10} | {'SiO2 (n)':<10} {'SiO2 (k)':<10}")
    print("-" * 75)
    
    target_wls = np.arange(200, 1201, 100)
    
    for target in target_wls:
        idx = (np.abs(wavelengths - target)).argmin()
        wl = wavelengths[idx]
        print(f"{wl:<20.1f} | {si_n[idx]:<10.3f} {si_k[idx]:<10.3f} | {sio2_n[idx]:<10.3f} {sio2_k[idx]:<10.3f}")

    # 5. Plotting in Subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Silicon
    ax1 = axes[0]
    ax1.plot(wavelengths, si_n, 'b-', label='n')
    ax1.plot(wavelengths, si_k, 'r--', label='k')
    ax1.set_title("Silicon (a-Si) Model")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Optical Constants")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot SiO2
    ax2 = axes[1]
    ax2.plot(wavelengths, sio2_n, 'b-', label='n')
    ax2.plot(wavelengths, sio2_k, 'r--', label='k')
    ax2.set_title("Silicon Dioxide (SiO2) Model")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Optical Constants")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 2.5) 

    plt.tight_layout()
    plt.show()