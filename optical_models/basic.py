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

from .material import Material

__all__ = ["Konstant", "TableMaterial"]

class Konstant(Material):
    """
    Material with constant real (n) and imaginary (k) refractive indices.

    Useful for simple media or reference materials where optical properties
    don't vary with wavelength. Requires positive 'n' and non-negative 'k'.

    Attributes:
        n: Constant real part of complex refractive index
        k: Constant imaginary part

    Parameters:
        params: Dictionary with required keys 'n', 'k'
            - n (float): >0 for physical materials
            - k (float): >=0, 0 for non-absorbing media
        wavelength: Optional nm array for calculations

    Example:
        >>> # Non-absorbing glass
        >>> params = {'n': 1.52, 'k': 0.0}
        >>> mat = Konstant(params)
        >>>
        >>> # Air (refractive index ~1.0)
        >>> air = Konstant({'n': 1.0, 'k': 0.0})
    """

    def __init__(self,
                 params: Dict[str, Union[float, int]],
                 wavelength: Optional[np.ndarray] = None):
        """
        Initialize the constant refractive index material.
    
        Args:
            params: Dictionary with required keys 'n' and 'k'
                - n (float): Must be positive
                - k (float): Must be non-negative
            wavelength: Optional nm array for calculations
            
        Raises:
            KeyError: If either required parameter is missing from params
            ValueError: If any parameter has invalid value (negative or non-numeric)
        """
        # Validate numeric type
        for param in ['n', 'k']:
            if not isinstance(params[param], (int, float)):
                raise ValueError(f"{param} must be numeric")
    
        self.n = float(params['n'])
        self.k = float(params['k'])

        # Optional wavelength range for calculations
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Convert wavelength array to energy values for calculations."""
        self.wavelength = np.asarray(wavelength)

    def complex_refractive_index(self,
                               wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)

        # Compute real and imaginary parts
        n = self.n * np.ones(self.wavelength.shape)
        k = self.k * np.ones(self.wavelength.shape)
        self.nk = n + 1j * k
        return self.nk

    def get_params(self) -> Dict[str, float]:
        """Return material parameters."""
        return {
            'n': self.n,
            'k': self.k,
        }

class TableMaterial(Material):
    """Material with tabulated refractive index and extinction coefficient data."""

    def __init__(self,
                 params: Dict[str, Union[float, int]],
                 n_data: np.ndarray,
                 k_data: np.ndarray = None,
                 interpolation_type_n: str = 'linear',
                 interpolation_type_k: str = 'linear',
                 wavelength: Optional[np.ndarray] = None):
               
        """
        Initialize a material with tabulated refractive index and extinction coefficient.

        Args:
            params: Dictionary of material parameters (not used for TabulatedMaterial)
            wavelength_range: Array of wavelengths corresponding to n_data and k_data
            n_data: Refractive index values at given wavelengths
            k_data: Extinction coefficient values at given wavelengths (optional, default 0)
            interpolation_type_n: Type of interpolation for refractive index n ('linear' or 'spline')
            interpolation_type_k: Type of interpolation for absoprtion coefficient k('linear' or 'spline')
            wavelength: Optional initial wavelength range for calculations
        """
        self.n_factor = params.get('n', 1.0)
        self.k_factor = params.get('k', 1.0)

        self.interpolation_type_n = interpolation_type_n
        self.interpolation_type_k = interpolation_type_k
        
        self.n_data = np.asarray(n_data, dtype=np.float64)
        if k_data is None:
            self.k_data = k_data
        else:
            self.k_data = np.asarray(k_data, dtype=np.float64)

        if wavelength is not None:
            self.set_wavelength_range(wavelength)

    def _interpolate_data(self,
                        data_type: str,
                        data: Optional[np.ndarray],
                        interpolation_type: str = 'linear') -> np.ndarray:
        """
        Interpolates or generates data based on the specified type and interpolation method.
    
        This function handles two primary tasks:
        1. Interpolating given data points to match the current wavelength.
        2. Generating arrays of ones ('n' type) or zeros ('k' type) if no input data is provided.
    
        Args:
            data_type (str): Specifies the type of data to process or generate.
                            'n' for refractive index (real part), 'k' for extinction coefficient (imaginary part).
            data (Optional[np.ndarray]): 2D array with shape (2, N) where data[0] contains wavelengths and
                                       data[1] contains corresponding values (either n or k). If None,
                                       generates an array of ones or zeros based on data_type.
            interpolation_type (str): The method used for interpolation. Options are:
                                     'linear', 'pchip', 'akima', 'makima'. Defaults to 'linear'.
    
        Returns:
            np.ndarray: Interpolated data values at the current wavelength, or generated array of ones/zeros.
    
        Raises:
            ValueError: If self.wavelength is not set, or if an invalid interpolation_type is provided.
        """
        if self.wavelength is None:
            raise ValueError("self.wavelength must be set before interpolation.")


        if data:
            wvl, vals = data
            # Create a mapping for interpolation methods to avoid repetitive code
            interpolation_methods = {
                'linear': lambda w, v: np.interp(self.wavelength, w, v),
                'cubicspline': lambda w, v: CubicSpline(w, v, extrapolate=True)(self.wavelength),
                'pchip': lambda w, v: PchipInterpolator(w, v, extrapolate=True)(self.wavelength),
                'akima': lambda w, v: Akima1DInterpolator(w, v, method="akima", extrapolate=True)(self.wavelength),
                'makima': lambda w, v: Akima1DInterpolator(w, v, method="makima", extrapolate=True)(self.wavelength)
            }
    
            # Use the mapping to get the appropriate interpolation function
            if interpolation_type in interpolation_methods:
                return interpolation_methods[interpolation_type](wvl, vals)
            else:
                raise ValueError(f"Unknown interpolation type '{interpolation_type}'. Choose 'linear', 'cubicspline', 'pchip', 'akima' or 'makima'.")
        else:
            # Handle the case where data is None
            if data_type == 'n':
                return np.ones_like(self.wavelength)
            elif data_type == 'k':
                return np.zeros_like(self.wavelength)

    # Override the base class methods to prevent modification
    def set_wavelength_range(self, wavelength: np.ndarray) -> None:
        """Set the wavelength range for calculations, storing but not sorting."""
        self.wavelength = np.asarray(wavelength, dtype=np.float64)
        self.n_interp = None
        self.k_interp = None
        self.complex_refractive_index()

    def complex_refractive_index(self,
                              wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the complex refractive index (n + ik)."""
        if wavelength is not None:
            self.set_wavelength_range(wavelength)
            
        if self.n_interp is None:
            self.n_interp = self._interpolate_data('n', self.n_data, self.interpolation_type_n)
        if self.k_interp is None:
            self.k_interp = self._interpolate_data('k', self.k_data, self.interpolation_type_k)
        self.nk = self.n_factor * self.n_interp + 1j * self.k_factpr * self.k_interp
        return self.nk

    def get_params(self) -> Dict[str, float]:
        """Return material parameters. For tabulated data, we'll just return empty dict since no specific parameters."""
        return {}

    def set_param(self, param_name: str, value: Union[float, int]) -> None:
        """This class doesn't support parameter setting as it uses tabulated data."""
        raise NotImplementedError("TabulatedMaterial does not support parameter setting")