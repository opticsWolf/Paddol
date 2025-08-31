# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from numba import njit #, prange
from typing import List, Tuple, Dict, Union, Optional
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator

from .material import Material, compute_energy

__all__ = ["CodyLorentz", "ContinousCodyLorentz"]

def get_num_points_for_kk(E_min: float, E_max: float, points_per_eV: int = 100) -> int:
    """
    Calculate the number of points for uniform energy grid in Kramers-Kronig integration.

    Args:
        E_min : float Minimum energy (eV) of the grid.
        E_max : float Maximum energy (eV) of the grid.
        points_per_eV : int, optional Number of points per electronvolt (default is 100).

    Returns:
        int Total number of points for the energy grid, with a minimum of 200.
    """
    E_range = E_max - E_min
    return max(200, int(E_range * points_per_eV))


@njit(cache=True)
def calc_Eu(E0: float, Et: float, Gamma: float, Ep: float, Eg: float) -> float:
    """
    Calculate Urbach energy parameter Eu based on Cody-Lorentz model parameters.    
    Returns:
        Urbach energy parameter Eu in units of eV

    Args:        
        E0    : Lorentz resonance energy [eV]
        Et    : Urbach transition energy [eV]
        Gamma : Lorentz broadening (damping) [eV]
        Ep    : Cody gap transition energy [eV]
        Eg    : Optical bandgap energy [eV]

    Returns:
        Urbach energy parameter Eu in units of eV
    """
    num = E0**4 - Et**4
    denom = ((E0**2 - Et**2)**2 +
             Gamma**2 * Et**2 +
             Ep**2 * (Et - Eg)**2 +
             Ep**2 * Et / (Et - Eg))
    return num / denom

@njit(cache=True)
def _CL_Gc(Ei, Eg, Ep):
    numerator = (Ei - Eg) ** 2
    return numerator / (numerator + Ep**2)

@njit(cache=True)
def _CL_L(Ei, A, E0, Gamma):
    # Lorentz oscillator contribution
    Ei2 = Ei ** 2
    GaEi = Gamma * Ei
    return A * E0 * GaEi / (E0**2 - Ei2)**2 + GaEi**2

@njit(cache=True)
def _epsilon2_cody_lorentz(E: np.ndarray, Eg: float, Ep: float,
                          A: float, E0: float, Gamma: float,
                          Et: float, Eu: float) -> np.ndarray:
    """
    Compute the imaginary part of the dielectric function ε₂(E) using the Cody–Lorentz model.
    
    Args:
        E     : Photon energies [eV], 1D array
        Eg    : Optical bandgap energy [eV]
        Ep    : Cody gap transition energy [eV]
        A     : Lorentz oscillator strength (amplitude)
        E0    : Lorentz resonance energy [eV]
        Gamma : Lorentz broadening (damping) [eV]
        Et    : Urbach transition energy [eV]
        Eu    : Urbach decay width [eV]
    
    Returns:
        eps2 : Imaginary part of dielectric function, same shape as E
    """

    eps2 = np.zeros(E.shape, dtype=np.float64)

    # Avoid division by zero or negative sqrt later
    #if Et <= Eg or Eu <= 0.0:
    #    return eps2  # Unphysical model, return zero
    
    E1 = Et * _CL_L(Et, A, E0, Gamma) * _CL_Gc(Et, Eg, Ep)
    #E_2 = E**2
    
    for i in range(E.shape[0]):
        if E[i] >= Et:
            #L(E) Lorentz function
            L = _CL_L(E[i], A, E0, Gamma)
            #Gc(E) Coady gap function
            Gc = _CL_Gc(E[i], Eg, Ep)
            eps2[i] = L * Gc
            
        elif 0 < E[i] < Et:
            eps2[i] = E1/E[i] * np.exp((E[i] - Et) / Eu)
            
    return eps2

@njit(cache=True)
def _eps2_c_cody_lorentz(E: np.ndarray, A: float, Eg: float, Ep: float,
                      E0: float, Gamma: float, Et: float, Eu: float) -> np.ndarray:
    """
    Calculate the imaginary part of the dielectric function ε₂(E) based on 
    the Continuous-Cody–Lorentz optical model.
    
    The function models absorption in three regions:
    - For photon energies E ≤ Eg (bandgap energy), absorption is zero.
    - For Eg < E < Et (transition energy), absorption follows an Urbach tail with exponential decay.
    - For E ≥ Et, absorption follows the Lorentz oscillator form ensuring continuity and physical consistency.
    
    Parameters:
        E (np.ndarray): Photon energy array (in eV) at which ε₂(E) is evaluated.
        A (float): Amplitude coefficient controlling the strength of absorption.
        Eg (float): Bandgap energy (in eV), below which absorption is zero.
        Ep (float): Parameter related to oscillator strength or broadening in the model.
        E0 (float): Resonance energy of the oscillator (in eV).
        Gamma (float): Broadening (damping) parameter of the oscillator (in eV).
        Et (float): Transition energy (in eV) between the Urbach tail and Lorentz oscillator regions.
        Eu (float): Urbach decay energy (in eV), controlling the exponential tail slope.
        
    Returns:
        np.ndarray: Imaginary part of the dielectric function ε₂(E) evaluated at energies E.
    """
    eps2 = np.zeros_like(E)
    for i in range(E.shape[0]):
        Ei = E[i]
        if Ei <= Eg:
            eps2[i] = 0.0
        elif Eg < Ei < Et:
            eps2[i] = A * Ei / ((Ei - Eg)**2 + Ep**2) * np.exp((Ei - Et) / Eu)
        else:
            numerator = A * Gamma * Ei * (E0**2 - Ei**2)**2
            denominator = ((E0**2 - Ei**2)**2 + Gamma**2 * Ei**2) * (Ei**2 - Eg**2 + Ep**2)
            eps2[i] = numerator / denominator
    return eps2

@njit(cache=True, fastmath=True)
def _eps1_kramer_kronig(E: np.ndarray, eps2: np.ndarray, eps_inf: float) -> np.ndarray:
    """
    Compute the real part of the dielectric function ε₁(E) using the Kramers–Kronig relation.
    
    Vectorized Hilbert transform for ε1(E), assuming:
    - uniform spacing in E
    - sorted ascending E
    Assumes uniform E spacing and sorted input.
    """
    N = E.shape[0]
    dE = E[1] - E[0]
    eps1 = np.zeros(N)

    for i in range(N):
        Ei = E[i]
        sum_val = 0.0
        for j in range(N):
            if i == j:
                continue
            Ej = E[j]
            sum_val += Ej * eps2[j] / (Ej**2 - Ei**2)
        eps1[i] = eps_inf + (2.0 / np.pi) * dE * sum_val

    return eps1

@njit(cache=True)
def epsilon1_from_epsilon2_kramers_kronig(
    E: np.ndarray,
    eps2: np.ndarray,
    epsilon_inf: float = 1.0
) -> np.ndarray:
    """
    Compute the real part of the dielectric function ε₁(E) using the Kramers–Kronig relation
    with the trapezoidal rule.

    Parameters:
        E           : Photon energies [eV], 1D array (must be sorted ascending)
        eps2        : Imaginary part ε₂(E), same shape as E
        epsilon_inf : High-frequency dielectric constant

    Returns:
        eps1        : Real part ε₁(E), same shape as E
    """
    n = E.shape[0]
    eps1 = np.zeros(n)

    for i in range(n):
        Ei = E[i]
        sum_val = 0.0

        for j in range(n - 1):
            Ej1 = E[j]
            Ej2 = E[j + 1]
            dE = Ej2 - Ej1

            denom1 = Ej1**2 - Ei**2
            denom2 = Ej2**2 - Ei**2

            if np.abs(denom1) < 1e-10 or np.abs(denom2) < 1e-10:
                continue  # Skip near singularities

            integrand1 = Ej1 * eps2[j]     / denom1
            integrand2 = Ej2 * eps2[j + 1] / denom2

            sum_val += 0.5 * (integrand1 + integrand2) * dE

        eps1[i] = epsilon_inf + (2.0 / np.pi) * sum_val

    return eps1



@njit(cache=True)
def _epsilon1_trapezoidal_kramers_kronig(E: np.ndarray, E_full: np.ndarray, eps2_full: np.ndarray, epsilon_inf: float) -> np.ndarray:
    """
    Compute the real part of the dielectric function ε₁(E) using the Kramers–Kronig relation
    with proper treatment of the singularity at E = E'.
    """
    eps1 = np.zeros_like(E)

    for i in range(E.shape[0]):
        Ei = E[i]
        integral = 0.0

        for j in range(E_full.shape[0] - 1):
            Ej1 = E_full[j]
            Ej2 = E_full[j + 1]
            dE = Ej2 - Ej1

            # Skip points where E_j == E_i (singularity)
            if abs(Ej1 - Ei) < 1e-10 or abs(Ej2 - Ei) < 1e-10:
                continue

            denom1 = Ej1**2 - Ei**2
            denom2 = Ej2**2 - Ei**2

            val1 = Ej1 * eps2_full[j] / denom1
            val2 = Ej2 * eps2_full[j + 1] / denom2

            integral += 0.5 * (val1 + val2) * dE

        eps1[i] = epsilon_inf + (2.0 / np.pi) * integral

    return eps1

@njit(cache=True)
def compute_cody_lorentz_complex_nk(E: np.ndarray, Eg: float, Ep: float, A: float, E0: float, Gamma: float, Et: float, Eu: float, E_full: np.ndarray, epsilon_inf: float):
    """
    Compute complex refractive index (n + ik) using Cody-Lorentz model.
    """

    eps2 = _epsilon2_cody_lorentz(E, Eg, Ep, A, E0, Gamma, Et, Eu)
    print (eps2[::5])
    #eps1 = _eps1_kramer_kronig(E, eps2, epsilon_inf)           
    eps1 = _epsilon1_trapezoidal_kramers_kronig(E, E_full, eps2, epsilon_inf)
    print (eps1[::5])
    
    eps_abs = np.sqrt(eps1**2 + eps2**2)
    return np.sqrt((eps_abs + eps1) / 2) + 1j * np.sqrt((eps_abs - eps1) / 2)

@njit(cache=True)
def compute_c_cody_lorentz_complex_nk(E: np.ndarray, A: float, Eg: float, Ep: float, E0: float, Gamma: float, Et: float, epsilon_inf: float):
    """
    Compute complex refractive index (n + ik) using Continous-Cody-Lorentz model with Kramers-Kronig relations.

    """
    Eu = calc_Eu(E0, Et, Gamma, Ep, Eg)
    eps2 = _eps2_c_cody_lorentz(E, A, Eg, Ep, E0, Gamma, Et, Eu)
    eps1 = _eps1_kramer_kronig(E, eps2, epsilon_inf)                            
    #eps1 = _eps1_kramer_kronig_optimized(E, eps2, epsilon_inf)           
    eps_abs = np.sqrt(eps1**2 + eps2**2)
    return np.sqrt((eps_abs + eps1) / 2) + 1j * np.sqrt((eps_abs - eps1) / 2)


class CodyLorentz(Material):
    """Cody–Lorentz model with Numba acceleration.

    This class models the dielectric response of materials using the Cody-Lorentz model,
    which combines a Lorentzian oscillator for interband transitions with an Urbach-like
    absorption edge below the bandgap.
        
    Attributes:
        A (float): Amplitude coefficient controlling the strength of absorption.
        E0 (float): Energy parameter in eV, representing the peak energy position.
        Gamma (float): Broadening parameter in eV, representing the width of the absorption edge.
        Eg (float): Bandgap energy in eV.
        epsilon_inf (float): High-frequency dielectric constant. Default is 1.0.
        h_c_by_e_nm (float): Pre-calculated value of h * c / e in nm for conversion purposes.
    
    Args:
        params (Dict[str, Union[float, int]]): Dictionary containing the model parameters.
            Required keys: 'E0', 'Gamma', 'Eg'.
            Optional key: 'epsilon_inf' (default is 1.0).
            wavelength (Optional[np.ndarray]): Array of wavelengths in nm for which to evaluate
            the dielectric function. If provided, the wavelength range will be set.
            
    Example:
        >>> # Create Cody-Lorentz model for silicon-like material
        >>> params = {
        ...     'E0': 3.2,      # Resonance energy position [eV]
        ...     'Gamma': 0.15,   # Damping constant [eV]
        ...     'Eg': 1.1       # Band gap energy [eV] (similar to Si at room temperature)
        ... }
        >>>
        >>> # With wavelength range (visible spectrum: 300-800 nm)
        >>> wavelengths = np.linspace(300, 800, 500)  # nm from 300 to 800nm
        >>> model = CodyLorentz(params, wavelengths)
    """

    def __init__(self,
                 params: Dict[str, Union[float, int]],
                 wavelength: Optional[np.ndarray] = None):
        
        """Initialize the Cody–Lorentz model.

        Args:
            params (Dict[str, Union[float, int]]): Dictionary containing the model parameters.
                Required keys: 'A', 'E0', 'Gamma', 'Eg'.
                Optional key: 'epsilon_inf' (default is 1.0).
            wavelength (Optional[np.ndarray]): Array of wavelengths in nm for which to evaluate
                the dielectric function. If provided, the wavelength range will be set.
        
        Raises:
            KeyError: If any required parameter ('E0', 'Gamma', 'Eg') is missing from `params`.
        """
        self.Eg = params['Eg']
        self.Ep = params['Ep']
        self.A = params['A']
        self.E0 = params['E0']
        self.Gamma = params['Gamma']
        self.Et = params['Eg']
        self.Eu = params['Eu']
        self.epsilon_inf = params.get('epsilon_inf', 1.0)
        #super().__init__(params, wavelength)

        self.h_c_by_eV_nm = 1239.8419843320028 # ready calculated h * c / eV in nm

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Set the wavelength range for calculations."""        
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm)  # Convert to energy in eV
        e_min = max(1e-3, np.min(self.E))
        e_max = max(10.0, np.max(self.E) + 2.0)  # Add buffer above target range
        N_samples = get_num_points_for_kk(e_min, e_max)
        self.E_full = np.linspace(e_min, e_max, N_samples)

    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
        print (len(self.wavelength), len(self.E))
        self.nk = compute_cody_lorentz_complex_nk(self.E, self.Eg, self.Ep, self.A, self.E0, self.Gamma, self.Et, self.Eu, self.E_full, self.epsilon_inf)
        return self.nk


    def get_params(self) -> Dict[str, float]:
        """Return material parameters."""
        base_params = super().get_params()
        base_params.update({
            'E0': self.E0,
            'Gamma': self.Gamma,
            'Eg': self.Eg,
            'epsilon_inf': self.epsilon_inf,
        })
        return base_params

class ContinousCodyLorentz(Material):
    """Continous-Cody–Lorentz model with Numba acceleration.

    This class implements the Continous-Cody–Lorentz dispersion model for optical materials.
    The model is used to describe the complex dielectric function of a material over a range
    of wavelengths. The implementation includes optional acceleration using Numba for improved performance.
    
    "New dispersion model for band gap tracking": https://doi.org/10.1016/j.tsf.2015.10.024
    
    Attributes:
        A (float): Amplitude coefficient controlling the strength of absorption.
        Eg (float): Bandgap energy (in eV), below which absorption is zero.
        Ep (float): Parameter related to oscillator strength or broadening in the model.
        E0 (float): Resonance energy of the oscillator (in eV).
        Gamma (float): Broadening (damping) parameter of the oscillator (in eV).
        Et (float): Transition energy (in eV) between the Urbach tail and Lorentz oscillator regions.
        epsilon_inf (float): High-frequency dielectric constant. Default is 1.0.
        h_c_by_e_nm (float): Pre-calculated value of h * c / e in nm for conversion purposes.
    
    Args:
        params (Dict[str, Union[float, int]]): Dictionary containing the model parameters.
            Required keys: 'E0', 'Gamma', 'Eg'.
            Optional key: 'epsilon_inf' (default is 1.0).
        wavelength (Optional[np.ndarray]): Array of wavelengths in nm for which to evaluate
            the dielectric function. If provided, the wavelength range will be set.
            
    Example:
        >>> # Create a Cody-Lorentz material model
        >>> params = {
        ...     'E0': 3.5,      # Resonance energy [eV]
        ...     'Gamma': 0.4,   # Damping constant [eV]
        ...     'Eg': 2.0       # Band gap energy [eV]
        ... }
        >>>
        >>> # With wavelength range
        >>> wavelengths = np.linspace(150, 700, 300)  # nm from 150 to 700nm
        >>> material = CodyLorentz(params, wavelengths)
        >>>
        >>> # Without wavelength range (use default or set later)
        >>> material = CodyLorentz(params)
    """

    def __init__(self,
                 params: Dict[str, Union[float, int]],
                 wavelength: Optional[np.ndarray] = None):
        
        """Initialize the Cody–Lorentz model.

        Args:
            params (Dict[str, Union[float, int]]): Dictionary containing the model parameters.
                Required keys: 'A', 'Eg', 'Ep', 'E0', 'Gamma', 'Et',
                Optional key: 'epsilon_inf' (default is 1.0).
            wavelength (Optional[np.ndarray]): Array of wavelengths in nm for which to evaluate
                the dielectric function. If provided, the wavelength range will be set.
        
        Raises:
            KeyError: If any required parameter ('E0', 'Gamma', 'Eg') is missing from `params`.
        """
        self.A = params['A']
        self.Eg = params['Eg']
        self.Ep = params['Ep']
        self.E0 = params['E0']
        self.Gamma = params['Gamma']
        self.Et = params['Et']
        self.epsilon_inf = params.get('epsilon_inf', 1.0)
        #super().__init__(params, wavelength)

        self.h_c_by_eV_nm = 1239.8419843320028 # ready calculated h * c / eV in nm

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Set the wavelength range for calculations."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm) # Convert to energy in eV

    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
        
        self.nk = compute_c_cody_lorentz_complex_nk(self.E, self.A, self.Eg, self.Ep, self.E0, self.Gamma, self.Et, self.epsilon_inf)
        return self.nk

    def get_params(self) -> Dict[str, float]:
        """Return material parameters."""
        base_params = super().get_params()
        base_params.update({
            'E0': self.E0,
            'Gamma': self.Gamma,
            'Eg': self.Eg,
            'epsilon_inf': self.epsilon_inf,
        })
        return base_params
