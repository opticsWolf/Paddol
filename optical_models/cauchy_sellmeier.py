# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from numba import njit #, prange
from typing import Dict, Union, Optional

from .material import Material, compute_energy

__all__ = ["Cauchy", "CauchyUrbach", "Sellmeier", "SellmeierUrbach" ]


@njit(cache=True)
def compute_cauchy_n_part(wavelength_µm_2: np.ndarray,
                           A: float,
                           B: float,
                           C: float) -> np.ndarray:
    """
    Compute real part of refractive index from Cauchy model.

    Args:
        wavelength_µm_2: Array of wavelengths in µm squared
        A: Coefficient for the constant term
        B: Coefficient for 1/wavelength^2 term
        C: Coefficient for 1/wavelength^4 term

    Returns:
        Refractive index (real part) for given wavelengths
    """
    return A + B / wavelength_µm_2 + C / (wavelength_µm_2 ** 2)

@njit(cache=True)
def compute_urbach_k_part(wavelength_m: np.ndarray,
                            E: np.ndarray,
                            alpha0: float,
                            Eu: float,
                            lambda_g: float,
                            h_c: float) -> np.ndarray:
    """
    Compute Urbach extinction coefficient k for given wavelengths and energies.

    Args:
        wavelength_m: Array of wavelengths in m
        E: Corresponding array of photon energies
        alpha0: Absorption coefficient at band gap energy (1/cm)
        Eu: Urbach energy parameter (eV)
        lambda_g: Band gap wavelength (nm)
        h_c: Product of Planck's constant and speed of light

    Returns:
        Extinction coefficient k for the given wavelengths
    """
    k_part = np.zeros_like(E)
    E_g = h_c / (lambda_g)

    for i in range(len(E)):
        #if wavelength[i] < lambda_g:
        if E[i] < E_g:
            exponent = (E[i] - E_g) / Eu
            absorption_coeff = alpha0 * np.exp(exponent)
            k_part[i] = absorption_coeff * wavelength_m[i] / (4 * np.pi)
    return k_part

@njit(cache=True)
def compute_cauchy_complex_nk(wavelength_µm_2: np.ndarray,
                             A: float,
                             B: float,
                             C: float) -> np.ndarray:
    """
    Compute complex refractive index (n + ik) from Cauchy model and sets k as 0.

    Args:
        wavelength_µm_2: Array of wavelengths in µm squared
        A, B, C: Cauchy model coefficients for real part (n)

    Returns:
        Complex refractive index array where real part is from Cauchy model
        and imaginary part is set to 0.
    """
    n = compute_cauchy_n_part(wavelength_µm_2, A, B, C)
    k = np.zeros_like(wavelength_µm_2)
    return n + 1j * k

@njit(cache=True)
def compute_cauchy_urbach_complex_nk(wavelength_m: np.ndarray, 
                             wavelength_µm_2: np.ndarray,
                             E: np.ndarray,
                             A: float,
                             B: float,
                             C: float,
                             alpha0: float,
                             Eu: float,
                             lambda_g: float,
                             h_c) -> np.ndarray:
    """
    Compute complex refractive index (n + ik) from Cauchy and Urbach models.

    Args:
        wavelength_m: Array of wavelengths in m
        wavelength_µm_2: Array of wavelengths in µm squared
        E: Corresponding photon energies in eV or Joules
        A, B, C: Cauchy model coefficients for real part (n)
        alpha0: Urbach absorption coefficient at band gap energy (1/cm)
        Eu: Urbach energy parameter (eV)
        lambda_g: Band gap wavelength (nm)
        h_c: Product of Planck's constant and speed of light

    Returns:
        Complex refractive index array where real part is from Cauchy model
        and imaginary part is from Urbach model extinction coefficient.
    """
    n = compute_cauchy_n_part(wavelength_µm_2, A, B, C)
    k = compute_urbach_k_part(wavelength_m, E, alpha0, Eu, lambda_g, h_c)
    return n + 1j * k

@njit(cache=True)
def compute_sellmeier_n_part(wavelength_µm_2: np.ndarray,
                        B1: float, C1: float,
                        B2: float, C2: float,
                        B3: float, C3: float) -> np.ndarray:
    """Compute refractive index from Sellmeier equation with Numba acceleration.

    Args:
        wavelength_µm_2: Array of wavelengths in µm squared
        B1-C3: Coefficients for the Sellmeier equation
        C1-C3: Denominator coefficients

    Returns:
        Refractive index (n) as numpy array
    """
    
    c1_sq = C1# * C1
    c2_sq = C2# * C2
    c3_sq = C3# * C3

    # Compute terms
    term1 = (B1 * wavelength_µm_2) / (wavelength_µm_2 - C1)
    term2 = (B2 * wavelength_µm_2) / (wavelength_µm_2 - C2)

    # Conditional term calculation
    if B3 != 0.0:
        term3 = (B3 * wavelength_µm_2) / (wavelength_µm_2 - C3)
    else:
        term3 = np.zeros_like(wavelength_µm_2)

    # Calculate n squared and take square root for n
    n_squared = 1.0 + term1 + term2 + term3

    ## For now, k=0 (no absorption)
    #k = np.zeros_like(n_squared)

    return np.sqrt(n_squared)

@njit(cache=True)
def compute_sellmeier_complex_nk(wavelength_µm_2: np.ndarray,
                             B1: float, C1: float,
                             B2: float, C2: float,
                             B3: float, C3: float) -> np.ndarray:
    """Compute complex refractive index (n + ik) for Sellmeier model, setting k to 0.
    Args:
        wavelength_m: Array of wavelengths in m
        wavelength_µm_2: Array of wavelengths in µm squared
        B1-C3: Coefficients for the Sellmeier equation
        C1-C3: Denominator coefficients

    Returns:
        Complex refractive index array where real part is from Sellmeier model
        and imaginary part is set to 0.
    """   
    n = compute_sellmeier_n_part(wavelength_µm_2, B1, C1, B2, C2, B3, C3)
    k = np.zeros_like(wavelength_µm_2)
    return n + 1j * k


@njit(cache=True)
def compute_sellmeier_urbach_complex_nk(wavelength_m: np.ndarray,
                             wavelength_µm_2: np.ndarray,
                             E: np.ndarray,
                             B1: float, C1: float,
                             B2: float, C2: float,
                             B3: float, C3: float,
                             alpha0: float,
                             Eu: float,
                             lambda_g: float,
                             h_c: float) -> np.ndarray:
    """Compute complex refractive index (n + ik) for Sellmeier model using Urbach model for k.
    Args:
        wavelength_m: Array of wavelengths in m
        wavelength_µm_2: Array of wavelengths in µm squared
        B1-C3: Coefficients for the Sellmeier equation
        C1-C3: Denominator coefficients
        E: Corresponding array of photon energies
        alpha0: Absorption coefficient at band gap energy (1/cm)
        Eu: Urbach energy parameter (eV)
        lambda_g: Band gap wavelength (nm)
        h_c: Product of Planck's constant and speed of light
    Returns:
        Complex refractive index array where real part is from Sellmeier model
        and imaginary part is from Urbach model extinction coefficient.
    """   
    n = compute_sellmeier_n_part(wavelength_µm_2, B1, C1, B2, C2, B3, C3)
    k = compute_urbach_k_part(wavelength_m, E, alpha0, Eu, lambda_g, h_c)
    return n + 1j * k


class Cauchy(Material):
    """Cauchy dispersion model with optional wavelength range.

    This class implements the Cauchy dispersion model for optical materials.
    The Cauchy model describes how the refractive index of a material varies with
    wavelength. The implementation includes support for specifying a wavelength range
    during initialization.
    
    Attributes:
        A (float): Coefficient for the first term in the Cauchy equation (1/lambda^2).
        B (float): Coefficient for the second term in the Cauchy equation (1/lambda^4).
        C (float): Constant phase offset or additional correction term.
        
    Args:
        params (Dict[str, Union[float, int]]): Dictionary containing the model parameters.
            Required keys: 'A', 'B', 'C'.
        wavelength (Optional[np.ndarray]): Array of wavelengths in nm for which to evaluate
            the dielectric function. If provided, the wavelength range will be set.
    
    Raises:
        KeyError: If any required parameter ('A', 'B', 'C') is missing from `params`.

    Examples:
        >>> params = {
        ...     'A': 2.0, 'B': 3.5, 'C': 1.8
        ... }
        >>> material = CauchyUrbach(params)
        >>> # Initialize with a specific wavelength range
        >>> wavelengths = np.linspace(200, 600, 100)
        >>> material.set_wavelength_range(wavelengths)
    """
    
    def __init__(self, params, wavelength=None):
        """Initialize the Cauchy dispersion model.

        Args:
            params (Dict[str, Union[float, int]]): Dictionary containing the model parameters.
                Required keys: 'A', 'B', 'C'.
            wavelength (Optional[np.ndarray]): Array of wavelengths in nm for which to evaluate
                the dielectric function. If provided, the wavelength range will be set.
        
        Raises:
            KeyError: If any required parameter ('A', 'B', 'C') is missing from `params`.
        """
        self.A = float(params['A'])
        self.B = float(params['B'])
        self.C = float(params['C'])

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Convert wavelength array to energy values for calculations."""
        self.wavelength = np.asarray(wavelength)
        self.wavelength_µm_2 = (self.wavelength * 1E-3) ** 2

    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik) with k=0."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
        
        self.nk = compute_cauchy_complex_nk(self.wavelength_µm_2, self.A, self.B, self.C)

        return self.nk

    def get_params(self) -> Dict[str, float]:
        """Return material parameters."""
        return {
            'A': self.A,
            'B': self.B,
            'C': self.C,
        }


class CauchyUrbach(Material):
    """
    A class to represent and compute the Cauchy-Urbach dispersion model for optical materials.
    
    The Cauchy-Urbach model describes the refractive index n(λ) of a material as a function
    of wavelength λ. It consists of two parts: a Cauchy term that dominates in the transparent
    region, and an Urbach tail that becomes significant near the absorption edge.
    
    Attributes:
        A (float): Coefficient for the Cauchy term.
        B (float): Coefficient for the Urbach tail.
        C (float): Coefficient for the Urbach tail.
        alpha0 (float): Absorption coefficient at the band gap energy.
        Eu (float): Urbach energy, which characterizes the width of the band tail.
        lambda_g (float): Band gap wavelength.
    
    Parameters:
        params (dict): Dictionary containing the parameters for the Cauchy-Urbach model. Keys are:
            - 'A': Coefficient for the Cauchy term.
            - 'B': Coefficient for the Urbach tail.
            - 'C': Coefficient for the Urbach tail.
            - 'alpha0': Absorption coefficient at the band gap energy.
            - 'Eu': Urbach energy.
            - 'lambda_g': Band gap wavelength.
        wavelength (np.ndarray or None, optional): Array of wavelengths in nanometers to set
            the range. If provided, the class will be initialized with this range.
    
    Notes:
        - The constants h_c_nm represent the product of Planck's constant h and the speed of light c,
          divided by 1 nanometer (1e-9 meters), i.e., h*c / lambda where lambda is in nanometers.
          This conversion factor is used to convert between wavelength in nanometers and energy.

    Examples:
        >>> params = {
        ...     'A': 2.0, 'B': 3.5, 'C': 1.8,
        ...     'alpha0': 1e3, 'Eu': 50e-3, 'lambda_g': 400
        ... }
        >>> material = CauchyUrbach(params)
        >>> # Initialize with a specific wavelength range
        >>> wavelengths = np.linspace(200, 600, 100)
        >>> material.set_wavelength_range(wavelengths)
    """
    
    def __init__(self, params, wavelength=None):
        """
        Initialize the Cauchy_Urbach class with parameters and optional wavelength range.
        
        Args:
            params (dict): Dictionary containing the parameters for the Cauchy-Urbach model.
            wavelength (np.ndarray or None, optional): Array of wavelengths in nanometers to set
                the range. If provided, the class will be initialized with this range.
        """
        self.A = float(params['A'])
        self.B = float(params['B'])
        self.C = float(params['C'])
        self.alpha0 = float(params['alpha0'])
        self.Eu = float(params['Eu'])
        self.lambda_g = float(params['lambda_g'])


        # Energy conversion constants
        self.h_c_by_eV_nm = 1239.8419843320028 # ready calculated h * c / eV in nm
        #self.c = 2.99792458e+08
        #self.h = 6.62607015e-34
        self.h_c_nm = 1.9864458571489287e-16

        if wavelength is not None:
            self.set_wavelength_range(wavelength)
        
    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Set the wavelength range for calculations."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.wavelength_m = self.wavelength * 1E-9
        self.wavelength_µm_2 = (self.wavelength * 1E-3) ** 2
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm) # Convert to energy in Joules


    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        self.nk = compute_cauchy_urbach_complex_nk(
            self.wavelength_m,
            self.wavelength_µm_2, self.E,
            self.A, self.B, self.C,
            self.alpha0, self.Eu, self.lambda_g,
            self.h_c_by_eV_nm
        )
        return self.nk

    def get_params(self) -> Dict[str, float]:
        """Return material parameters."""
        return {
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'alpha0': self.alpha0,
            'Eu': self.Eu,
            'lambda_g': self.lambda_g
        }


class Sellmeier(Material):
    """
    A class representing a material following the Sellmeier dispersion model using Schott convention.

    The Sellmeier equation is commonly used to describe the chromatic dispersion of optical materials.
    This class extends the Material base class and implements the Sellmeier dispersion model.

    Attributes:
        B1, B2, B3 (float): Schott-style coefficients for the Sellmeier equation.
        C1, C2, C3 (float): Schott-style coefficients for the Sellmeier equation.

    Parameters:
        params (Dict[str, Union[float, int]]): Dictionary containing the parameters for the Sellmeier model.
            Keys are expected to include 'B1', 'B2', 'B3', 'C1', 'C2', and 'C3'. If any of these keys
            are missing, they will default to 0.0.
        wavelength (np.ndarray or None, optional): Array of wavelengths in nanometers to set the range.
            If provided, the class will be initialized with this range.

    Notes:
        - The Sellmeier equation is given by:
          n^2 = B1 + B2*lambda^2 / (lambda^2 - C1) + B3*lambda^2 / (lambda^2 - C2)
          where lambda is the wavelength, and B1-B3, C1-C3 are material-specific coefficients.
              
    Examples:
    >>> params = {
    ...     'B1': 1.03961212, 'C1': 0.00600069867,
    ...     'B2': 0.231792344, 'C2': 0.0200179144,
    ...     'B3': 1.01046945, 'C3': 103.560653    
    ... }
    >>> material = Sellmeier(params)
    >>> # Initialize with a specific wavelength range
    >>> wavelengths = np.linspace(300, 500, 100)
    >>> material.set_wavelength_range(wavelengths)
    """

    def __init__(self,
                 params: Dict[str, Union[float, int]],
                 wavelength: Optional[np.ndarray] = None):
        """
        Initialize the Sellmeier class with parameters and optional wavelength range.
        
        Args:
            params (Dict[str, Union[float, int]]): Dictionary containing the parameters for the Sellmeier model.
                Keys are expected to include 'B1', 'B2', 'B3', 'C1', 'C2', and 'C3'. If any of these keys
                are missing, they will default to 0.0.
            wavelength (np.ndarray or None, optional): Array of wavelengths in nanometers to set the range.
                If provided, the class will be initialized with this range.
        """
        self.B1 = float(params.get('B1', 0.0))
        self.B2 = float(params.get('B2', 0.0))
        self.B3 = float(params.get('B3', 0.0))

        self.C1 = float(params.get('C1', 0.0))
        self.C2 = float(params.get('C2', 0.0))
        self.C3 = float(params.get('C3', 0.0))

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Set the wavelength range for calculations."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.wavelength_µm_2 = (self.wavelength * 1E-3) ** 2

    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        # Use Numba-accelerated function for calculation
        self.nk = compute_sellmeier_complex_nk(self.wavelength_µm_2,
            self.B1, self.C1,
            self.B2, self.C2,
            self.B3, self.C3)

        return self.nk

    def get_params(self) -> Dict[str, float]:
        """Return material parameters in Schott convention."""
        return {
            'B1': self.B1,
            'B2': self.B2,
            'B3': self.B3,
            'C1': self.C1,
            'C2': self.C2,
            'C3': self.C3
        }

class SellmeierUrbach(Material):
    """
    A class representing a material following the Sellmeier-Urbach dispersion model.

    This class combines the Sellmeier dispersion model with an Urbach tail to describe the absorption
    near the band gap. The Sellmeir part describes the chromatic dispersion in the transparent region,
    while the Urbach part accounts for the exponential increase of absorption near the band gap.

    Attributes:
        B1, B2, B3 (float): Schott-style coefficients for the Sellmeier equation.
        C1, C2, C3 (float): Schott-style coefficients for the Sellmeier equation.
        alpha0 (float): Absorption coefficient at the band gap energy.
        Eu (float): Urbach energy, which characterizes the width of the band tail.
        lambda_g (float): Band gap wavelength in nanometers.
        h_c_nm (float): Product of Planck's constant and speed of light divided by 1 nm,
            used for energy conversion.

    Parameters:
        params (Dict[str, Union[float, int]]): Dictionary containing the parameters for the Sellmeier-Urbach model.
            Required keys are 'alpha0', 'Eu', 'lambda_g'. Optional keys include 'B1', 'B2', 'B3', 'C1', 'C2', and 'C3'.
        wavelength (np.ndarray or None, optional): Array of wavelengths in nanometers to set the range.
            If provided, the class will be initialized with this range.

    Notes:
        - The Sellmeier equation is given by:
          n² = 1 + Σ[Bi*λ²/(λ²-Ci)]
          where lambda is the wavelength, and B1-B3, C1-C2 are material-specific coefficients.
        - The Urbach tail is described by an exponential function with characteristic energy Eu,
          which determines the width of the band tail. alpha0 is the absorption coefficient at
          the band gap energy.
          
    Examples:
    >>> params = {
    ...     'B1': 1.03961212, 'C1': 0.00600069867,
    ...     'B2': 0.231792344, 'C2': 0.0200179144,
    ...     'B3': 1.01046945, 'C3': 103.560653,
    ...     'alpha0': 1e3, 'Eu': 50e-3, 'lambda_g': 400
    ... }
    >>> material = SellmeierUrbach(params)
    >>> # Initialize with a specific wavelength range
    >>> wavelengths = np.linspace(300, 500, 100)
    >>> material.set_wavelength_range(wavelengths)
    """

    def __init__(self,
                 params: Dict[str, Union[float, int]],
                 wavelength: Optional[np.ndarray] = None):
        """
        Initialize the Sellmeier_Urbach class with parameters and optional wavelength range.

        Args:
            params (Dict[str, Union[float, int]]): Dictionary containing the parameters for the Sellmeier-Urbach model.
                Required keys are 'alpha0', 'Eu', 'lambda_g'. Optional keys include 'B1', 'B2', 'B3', 'C1', 'C2', and 'C3'.
            wavelength (np.ndarray or None, optional): Array of wavelengths in nanometers to set the range.
                If provided, the class will be initialized with this range.
        """
        self.B1 = float(params.get('B1', 0.0))
        self.B2 = float(params.get('B2', 0.0))
        self.B3 = float(params.get('B3', 0.0))

        self.C1 = float(params.get('C1', 0.0))
        self.C2 = float(params.get('C2', 0.0))
        self.C3 = float(params.get('C3', 0.0))

        self.alpha0 = float(params['alpha0'])
        self.Eu = float(params['Eu'])
        self.lambda_g = float(params['lambda_g'])

        # Energy conversion constants
        self.h_c_by_eV_nm = 1239.8419843320028 # ready calculated h * c / eV in nm

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Set the wavelength range for calculations."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.wavelength_m = self.wavelength * 1E-9
        self.wavelength_µm_2 = (self.wavelength * 1E-3) ** 2
        self.E = compute_energy(self.wavelength, self.h_c_by_eV_nm) # Convert to energy in Joules


    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        # Use Numba-accelerated function for calculation
        self.nk = compute_sellmeier_urbach_complex_nk(self.wavelength_m,
                                               self.wavelength_µm_2,
                                               self.E,
                                               self.B1, self.C1,
                                               self.B2, self.C2,
                                               self.B3, self.C3,
                                               self.alpha0, self.Eu,
                                               self.lambda_g, self.h_c_by_eV_nm
                                                )
        return self.nk

    def get_params(self) -> Dict[str, float]:
        """Return material parameters in Schott convention."""
        return {
            'B1': self.B1,
            'B2': self.B2,
            'B3': self.B3,
            'C1': self.C1,
            'C2': self.C2,
            'C3': self.C3
        }

