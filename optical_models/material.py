# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from numba import njit #, prange
from typing import Dict, Union, Optional


@njit(cache=True)
def compute_energy(wavelength: np.ndarray,
                  h_c: float) -> np.ndarray:
    """
    Compute photon energy from wavelength.

    Args:
        wavelength: Array of wavelengths in nanometers
        h_c: Product of Planck's constant (h) and speed of light (c), either as:
            - h*c (returns energies in Joules)
            - h*c/eV (returns energies in electron volts)

    Returns:
        Photon energy corresponding to each wavelength value

    Note:
        Energy = h*c / λ where λ is the wavelength
    """
    return h_c / (wavelength)

class Material:
    """Base class for optical materials with complex refractive index."""

    def __init__(self,
                 params: Dict[str, Union[float, int]], 
                 wavelength: Optional[np.ndarray] = None):
        self.A = params['A']

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """
        Set the wavelength range for calculations.

        Args:
            wavelength (np.ndarray): Array of wavelengths in nm
        """
        self.wavelength = np.asarray(wavelength, dtype=np.float64)

    def complex_refractive_index(self,
                               wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        # Compute real and imaginary parts
        n = self.real_part()
        k = self.imag_part()
        self.nk = n + 1j * k
        return self.nk

    def get_params(self) -> Dict[str, float]:
        """Return material parameters."""
        return {
            'A': self.A,
        }

    def set_param(self, param_name: str, value: Union[float, int]) -> None:
        """Set a material parameter.

        Args:
            param_name: Name of the parameter
            value: Parameter value

        Raises:
            AttributeError if parameter does not exist.
        """
        if hasattr(self, param_name):
            setattr(self, param_name, float(value))
        else:
            raise AttributeError(f"Parameter {param_name} does not exist.")