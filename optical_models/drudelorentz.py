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

__all__ = ["Drude", "DrudeLorentz"]

@njit(cache=True)
def compute_drude_complex_nk(E: np.ndarray,
                            omega_p: float,
                            gamma_drude: float,
                            eps_inf: float) -> np.ndarray:
    """Compute complex refractive index (n + ik) for Drude model only.

    Args:
        E: Array of photon energies in electron volts [eV]
        omega_p: Plasma frequency parameter for Drude term [eV]
        gamma_drude: Damping coefficient for Drude term [eV]
        eps_inf: High-frequency dielectric constant (default 1.0)

    Returns:
        Complex refractive index array where real parts are n and imaginary parts are k.

    Notes:
        - The function assumes proper units for all parameters (energies in eV)
        - For best performance, input energy array should be sorted
        - Only Drude term is computed without Lorentz oscillators
        - Square root calculation preserves physical consistency between n and k
    """
    # Initialize dielectric function with high-frequency component
    eps = np.full(E.shape, eps_inf + 0j)
    
    # Drude term: -omega_p^2 / (E^2 + i * gamma * E)
    drude_denominator = E**2 + 1j * gamma_drude * E
    eps -= omega_p**2 / drude_denominator
    
    # Convert dielectric function ε to refractive index n + ik
    eps_abs = np.sqrt(eps.real**2 + eps.imag**2)
    n = np.sqrt((eps_abs + eps.real) / 2)
    k = np.sqrt((eps_abs - eps.real) / 2)

    return n + 1j * k

@njit(cache=True)
def compute_drude_lorentz_complex_nk(E: np.ndarray,
                                     omega_p: float,
                                     gamma_drude: float,
                                     eps_inf: float,
                                     lorentz_params: np.ndarray) -> np.ndarray:
    """Compute complex refractive index (n + ik) for Drude-Lorentz model.

    Args:
        E: Array of photon energies in electron volts [eV]
        omega_p: Plasma frequency parameter for Drude term [eV]
        gamma_drude: Damping coefficient for Drude term [eV]
        eps_inf: High-frequency dielectric constant (default 1.0)
        lorentz_params: Oscillator parameters as array of shape (N, 3) with
            - E0: Resonance energy position [eV]
            - Gamma: Damping constant [eV]
            - f0: Oscillator strength parameter

    Returns:
        Complex refractive index array where real parts are n and imaginary parts are k.

    Notes:
        - The function assumes proper units for all parameters (energies in eV)
        - For best performance, input energy array should be sorted
        - Lorentz terms are added to the Drude response with proper sign handling
        - Square root calculation preserves physical consistency between n and k
    """
    eps = np.full(E.shape, eps_inf + 0j)  # complex array initialized with eps_inf

    # Drude term: -omega_p^2 / (E^2 + i * gamma * E)
    drude_denominator = E**2 + 1j * gamma_drude * E
    eps -= omega_p**2 / drude_denominator
    
    E_2 = E**2
    
    # Lorentz terms
    for i in range(lorentz_params.shape[0]):
        E0, Gamma, f0 = lorentz_params[i]
        E0_2 = E0**2
        eps += (f0 * E0_2) / ((E0_2 - E_2) - 1j * E * Gamma)

    # Convert dielectric function ε to refractive index n + ik
    eps_abs = np.sqrt(eps.real**2 + eps.imag**2)
    n = np.sqrt((eps_abs + eps.real) / 2)
    k = np.sqrt((eps_abs - eps.real) / 2)

    return n + 1j * k


class Drude(Material):
    """
    Implementation of the Drude dispersion model for free-electron response only.

    This model describes dielectric properties using only a Drude term representing
    intraband electronic transitions without any Lorentz oscillators.

    Attributes:
        omega_p: Plasma frequency for Drude term [eV]
        gamma_drude: Damping constant for Drude term [eV]
        eps_inf: High-frequency dielectric constant
        h_c_by_e_nm: Pre-calculated value of h*c/e in nm

    Args:
        params: Dictionary containing model parameters with keys:
            - 'omega_p': Plasma frequency [eV]
            - 'gamma_drude': Drude damping constant [eV]
            - Optional: 'eps_inf' High-frequency dielectric constant (default 1.0)
        wavelength: Optional array of wavelengths in nm

    Raises:
        ValueError: If invalid parameters are provided

    Example:
        >>> # Initialize with Drude parameters
        >>> params = {
        ...     'omega_p': 2.5,                # Plasma frequency [eV]
        ...     'gamma_drude': 0.3,            # Damping constant [eV]
        ...     'eps_inf': 3.5                 # High-frequency dielectric constant
        ... }
        >>>
        >>> # Initialize with wavelength range
        >>> wavelengths = np.linspace(150, 600)  # nm from 150 to 600nm
        >>> material = Drude(params, wavelengths)
    """

    def __init__(self,
                params: Dict[str, Union[float, int]],
                wavelength: Optional[np.ndarray] = None):
        """
        Initialize the Drude model.

        Args:
            params: Dictionary of parameters
            wavelength: Optional array of wavelengths in nm

        Raises:
            ValueError: If invalid parameters are provided
        """
        # Validate and extract Drude parameters
        self.omega_p = float(params['omega_p'])
        if self.omega_p <= 0:
            raise ValueError("Plasma frequency must be positive")

        self.gamma_drude = float(params['gamma_drude'])
        if self.gamma_drude < 0:
            raise ValueError("Drude damping constant must be non-negative")

        # High-frequency dielectric constant
        self.eps_inf = float(params.get('eps_inf', 1.0))

        # Energy conversion constants
        self.h_c_by_eV_nm = 1239.8419843320028  # h * c / eV in nm

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """
        Set the wavelength range for calculations and convert to energies.

        Args:
            wavelength: Array of wavelengths in nm
        """
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm)  # Convert to energy in eV

    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
            
        self.nk = compute_drude_complex_nk(
            E=self.E,
            omega_p=self.omega_p,
            gamma_drude=self.gamma_drude,
            eps_inf=self.eps_inf
        )
        return self.nk

    def get_params(self) -> Dict[str, Union[float, int]]:
        """
        Return all model parameters as a dictionary.

        Returns:
            Dictionary containing all model parameters
        """
        return {
            'omega_p': self.omega_p,
            'gamma_drude': self.gamma_drude,
            'eps_inf': self.eps_inf
        }


class DrudeLorentz(Material):
    """
    Implementation of the Drude-Lorentz dispersion model combining free-electron response
    with multiple Lorentz oscillators.

    This model describes dielectric properties by combining:
    - A Drude term representing intraband electronic transitions
    - Multiple Lorentz oscillators representing interband transitions

    Each Lorentz oscillator is characterized by its resonance energy (E0), damping constant (Gamma),
    and oscillator strength parameter (f0).

    Attributes:
        n_oscillators (int): Number of oscillators in the model
        omega_p: Plasma frequency for Drude term [eV]
        gamma_drude: Damping constant for Drude term [eV]
        eps_inf: High-frequency dielectric constant
        lorentz_params: Array of shape (N,3) containing oscillator parameters (E0, Gamma, f0)
        h_c_by_e_nm: Pre-calculated value of h*c/e in nm

    Args:
        params: Dictionary containing model parameters with keys:
            - 'omega_p': Plasma frequency [eV]
            - 'gamma_drude': Drude damping constant [eV]
            - Either:
                'lorentz_params': List of tuples (E0, Gamma, f0) for all oscillators
            - Optional: 'eps_inf' High-frequency dielectric constant (default 1.0)
        wavelength: Optional array of wavelengths in nm

    Raises:
        ValueError: If invalid parameters are provided or no oscillators specified

    Example:
        >>> # Initialize with Lorentz oscillator parameters
        >>> params = {
        ...     'omega_p': 2.5,                # Plasma frequency [eV]
        ...     'gamma_drude': 0.3,            # Damping constant [eV]
        ...     'lorentz_params': [
        ...         (4.0, 0.5, 1.0),           # Oscillator parameters
        ...         (6.0, 0.2, 0.8)
        ...     ],
        ...     'eps_inf': 3.5                 # High-frequency dielectric constant
        ... }
        >>>
        >>> # Initialize with wavelength range
        >>> wavelengths = np.linspace(150, 600)  # nm from 150 to 600nm
        >>> material = DrudeLorentz(params, wavelengths)
    """

    def __init__(self,
                params: Dict[str, Union[float, int, List[Tuple[float, float, float]]]],
                wavelength: Optional[np.ndarray] = None):
        """
        Initialize the Drude-Lorentz model.

        Args:
            params: Dictionary of parameters
            wavelength: Optional array of wavelengths in nm

        Raises:
            ValueError: If invalid parameters are provided or no oscillators are specified
        """
        self.n_oscillators = params.get('n_oscillators', 1)
        self.omega_p = float(params['omega_p'])
        self.gamma_drude = float(params['gamma_drude'])
        self.lorentz_params = params['lorentz_params']
        # Validate and extract Drude parameters
        self.omega_p = params['omega_p']
        if self.omega_p <= 0:
            raise ValueError("Plasma frequency must be positive")

        self.gamma_drude = params['gamma_drude']
        if self.gamma_drude < 0:
            raise ValueError("Drude damping constant must be non-negative")

        if self.n_oscillators > 0:
            self._oscillators = params['lorentz_params']
            # Validate oscillator parameters
            if len(self.lorentz_params) != self.n_oscillators:
                raise ValueError("Number of oscillators does not match provided parameters")
            for osc in self.lorentz_params:
                if len(osc) != 3:
                    raise ValueError("Each oscillator parameter must be a tuple of (E0, Gamma, f0)")
                E0, Gamma, f0 = osc
                if E0 <= 0 or Gamma < 0 or f0 <= 0:
                    raise ValueError("E0 and f0 must be positive and Gamma non-negative")
    
            # Convert to NumPy array for efficient computation
            self.lorentz_params = np.array(self._oscillators, dtype=np.float64)
    
            # Validate at least one oscillator exists
            if not self._oscillators:
                raise ValueError("At least one Lorentz oscillator must be specified")
        else:
            self.lorentz_params = np.array([])

        # High-frequency dielectric constant
        self.eps_inf = float(params.get('eps_inf', 1.0))

        # Energy conversion constants
        self.h_c_by_eV_nm = 1239.8419843320028 # ready calculated h * c / eV in nm

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """
        Set the wavelength range for calculations and convert to energies.

        Args:
            wavelength: Array of wavelengths in nm
        """
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm) # Convert to energy in eV

    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
            
        self.nk = compute_drude_lorentz_complex_nk(
            E=self.E,
            omega_p=self.omega_p,
            gamma_drude=self.gamma_drude,
            eps_inf=self.eps_inf,
            lorentz_params=self.lorentz_params
        )
        return self.nk

    def get_params(self) -> Dict[str, Union[float, int]]:
        """
        Return all model parameters as a dictionary.

        Returns:
            Dictionary containing all model parameters with oscillator parameters converted back to lists of tuples
        """
        return {
            'omega_p': self.omega_p,
            'gamma_drude': self.gamma_drude,
            'eps_inf': self.eps_inf,
            'lorentz_params': [tuple(params) for params in self.lorentz_params]
        }
