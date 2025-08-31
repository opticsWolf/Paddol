# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from numba import njit #, prange
from typing import List, Tuple, Dict, Union, Optional

from .material import Material, compute_energy

__all__ = ["LorentzOscillator"]

@njit(cache=True)
def compute_lorentz_complex_nk(E: np.ndarray,
                               lorentz_params: np.ndarray,
                               eps_inf: float = 1.0) -> np.ndarray:
    """
    Compute complex refractive index (n + ik) for Lorentz oscillators.

    Args:
        E:Array of photon energies in electron volts [eV]
        lorentz_params: Oscillator parameters as array of shape (N, 3) of (E0, Gamma, f0), where:
            - E0: Resonance energy position [eV]
            - Gamma: Damping constant [eV]
            - f0: Oscillator strength parameter
        eps_inf: High-frequency dielectric constant (default 1.0)

    Returns:
        Complex refractive index array where real parts are n and imaginary parts are k
    """
    eps = np.full(E.shape, eps_inf, dtype=np.complex128)
    
    E_2 = E**2
    
    for j in range(lorentz_params.shape[0]):
        E0, Gamma, f0 = lorentz_params[j]
        E0_2 = E0**2
        eps += (f0 * E0_2) / ( (E0_2 - E_2) - 1j * E * Gamma )

    return np.sqrt(eps)

class LorentzOscillator (Material):
    """
    Material class implementing the Lorentz-Lorentz oscillator dispersion model.

    This class represents a material with optical properties described by multiple Lorentz oscillators.
    The model is suitable for materials where the dielectric response can be represented as a sum
    of oscillator contributions. Each oscillator is characterized by its resonance energy, damping,
    and oscillator strength parameters.

    Attributes:
        n_oscillators (int): Number of oscillators in the model
        epsilon_inf (float): High-frequency dielectric constant (default: 1.0)
        h_c_by_e_nm (float): Pre-calculated value of h * c / e in nm for energy conversion

    Args:
        params (Dict[str, Union[float, int]]): Dictionary containing model parameters.
            Required keys: 'n_oscillators' and a list of oscillator parameters as tuples.
            Optional key: 'epsilon_inf' (default 1.0)
        wavelength (Optional[np.ndarray]): Array of wavelengths in nm for evaluation

    Example:
        >>> params = {
        ...     'n_oscillators': 2,
        ...     'osc_params': [
        ...         (3.0, 0.2, 0.5),   # E0=3eV, Gamma=0.2eV, f0=0.5
        ...         (4.5, 0.1, 0.7)    # Second oscillator
        ...     ],
        ...     'epsilon_inf': 1.0
        ... }
        >>> material = LorentzLorentz(params)
    """

    def __init__(self,
                 params: Dict[str, Union[float, int, List[Tuple[float, float, float]]]],
                 wavelength: Optional[np.ndarray] = None):
        """
        Initialize the Lorentz-Lorentz model.

        Args:
            params (Dict): Dictionary containing model parameters
            wavelength (Optional[np.ndarray]): Array of wavelengths in nm

        Raises:
            KeyError: If required parameters are missing or invalid
        """
        self.n_oscillators = params.get('n_oscillators', 2)
        self.epsilon_inf = params.get('epsilon_inf', 1.0)
        self.osc_params = params['osc_params']

        # Energy conversion constants
        self.h_c_by_eV_nm = 1239.8419843320028 # ready calculated h * c / eV in nm
                
        # Validate oscillator parameters
        if len(self.osc_params) != self.n_oscillators:
            raise ValueError("Number of oscillators does not match provided parameters")
        for osc in self.osc_params:
            if len(osc) != 3:
                raise ValueError("Each oscillator parameter must be a tuple of (E0, Gamma, f0)")
            E0, Gamma, f0 = osc
            if E0 <= 0 or Gamma < 0 or f0 <= 0:
                raise ValueError("E0 and f0 must be positive and Gamma non-negative")

        # Convert osc_params list to NumPy array for Numba
        self._lorentz_params = np.array(self.osc_params, dtype=np.float64)

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

        # Use optimized Numba function to compute n+ik
        self.nk = compute_lorentz_complex_nk(self.E,
                                            self._lorentz_params,
                                            self.epsilon_inf)
        return self.nk

    def get_params(self) -> Dict[str, float]:
        """Return material parameters."""
        base_params = super().get_params()
        base_params.update({
            'n_oscillators': self.n_oscillators,
            'epsilon_inf': self.epsilon_inf,
            'oscillator_parameters': self.osc_params
        })
        return base_params