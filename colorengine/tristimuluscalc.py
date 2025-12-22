# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 22:16:31 2025

@author: Frank
"""

import numpy as np
from numba import njit, float64
from dataclasses import dataclass
from typing import Tuple, Optional

# Type aliases for clarity
ArrayFloat = np.ndarray

@njit(fastmath=True, cache=True)
def _fast_calculate_xyz(
    spectra_values: ArrayFloat, 
    weighting_vector: ArrayFloat
) -> ArrayFloat:
    """
    Numba-optimized dot product for single or batched spectra.
    
    Args:
        spectra_values: Shape (N,) or (M, N)
        weighting_vector: Shape (N, 3) pre-computed weights.
        
    Returns:
        XYZ values (3,) or (M, 3).
    """
    # np.dot is effectively BLAS dgemm/dgemv, highly optimized.
    # Numba removes the python call overhead.
    return np.dot(spectra_values, weighting_vector)

class FastCIEIntegrator:
    """
    High-performance CIE Tristimulus calculator.
    
    This class pre-computes the integration weights combining CMFs, Illuminant,
    and normalization factors. It assumes spectral data is aligned to the
    wavelengths defined during initialization.
    """

    def __init__(
        self,
        cmfs: ArrayFloat,
        illuminant: ArrayFloat,
        interval: float = 1.0,
        k_normalization: Optional[float] = None
    ):
        """
        Initialize the integrator.

        Args:
            cmfs (np.ndarray): Color Matching Functions, shape (N, 3).
                Columns should be [x_bar, y_bar, z_bar].
            illuminant (np.ndarray): Illuminant Spectral Power Distribution, shape (N,).
            interval (float): The wavelength interval (step size) in nm (e.g., 1.0, 5.0).
            k_normalization (float, optional): Manual normalization factor k.
                If None, calculated so Y=100 for a perfect reflector.
        """
        self._validate_inputs(cmfs, illuminant)

        # 1. Calculate the normalization factor 'k' (ASTM E308)
        # k = 100 / sum(S(lambda) * y_bar(lambda) * delta_lambda)
        if k_normalization is None:
            # Column 1 of CMFs is usually Y (CIE 1931 XYZ)
            y_bar = cmfs[:, 1]
            sum_s_y = np.sum(illuminant * y_bar * interval)
            self.k = 100.0 / sum_s_y
        else:
            self.k = k_normalization

        # 2. Pre-calculate the combined Weighting Matrix (W)
        # W_i = k * S_i * CMF_i * delta_lambda
        # We broadcast S (N,) to (N, 1) to multiply against CMFs (N, 3)
        self.weights: ArrayFloat = (
            cmfs * illuminant[:, np.newaxis] * self.k * interval
        ).astype(np.float64)

    def _validate_inputs(self, cmfs: ArrayFloat, illuminant: ArrayFloat) -> None:
        """Internal validation of shapes."""
        if cmfs.ndim != 2 or cmfs.shape[1] != 3:
            raise ValueError(f"CMFs must be shape (N, 3), got {cmfs.shape}")
        if illuminant.ndim != 1:
            raise ValueError(f"Illuminant must be shape (N,), got {illuminant.shape}")
        if cmfs.shape[0] != illuminant.shape[0]:
            raise ValueError(
                f"Dimension mismatch: CMFs {cmfs.shape[0]} vs Illuminant {illuminant.shape[0]}"
            )

    def sd_to_xyz(self, spectra: ArrayFloat) -> ArrayFloat:
        """
        Convert spectral data to XYZ using pre-calculated weights.
        
        This is the method to call inside your optimization loop.

        Args:
            spectra (np.ndarray): Reflectance or Transmittance data.
                - Shape (N,): Single sample.
                - Shape (M, N): Batch of M samples.
                Must match the wavelength length N used in __init__.

        Returns:
            np.ndarray: XYZ tristimulus values.
                - Shape (3,) if input was (N,)
                - Shape (M, 3) if input was (M, N)
        """
        # We delegate to the JIT compiled function for minimal overhead
        return _fast_calculate_xyz(spectra, self.weights)

    def sd_to_xyz_batch_gen(self, spectra_generator):
        """
        Helper for very large datasets that don't fit in memory.
        """
        for spectrum in spectra_generator:
            yield _fast_calculate_xyz(spectrum, self.weights)


# ==========================================
# Usage Example & Verification
# ==========================================

if __name__ == "__main__":
    import time

    # --- 1. Setup Mock Data (Standard CIE 1931 2 degree, D65, 360-780nm) ---
    # In a real scenario, load this from the colour library or files
    wavelengths = np.arange(360, 781, 1)
    N = len(wavelengths)
    
    # Random mock CMFs (x, y, z)
    mock_cmfs = np.abs(np.sin(wavelengths[:, None] / 50.0)) 
    
    # Random mock Illuminant (D65-ish)
    mock_illuminant = np.ones(N) * 100.0
    
    # --- 2. Initialize the Optimized Class ---
    # This happens ONCE outside your loop
    converter = FastCIEIntegrator(
        cmfs=mock_cmfs,
        illuminant=mock_illuminant,
        interval=1.0
    )

    print(f"Normalization factor k: {converter.k:.4f}")

    # --- 3. Simulation of Optimization Loop ---
    print("\n--- Starting Performance Benchmark ---")
    
    # Create a batch of 100,000 random spectra (Optimization candidates)
    # Shape: (100000, 421)
    num_iterations = 100_000
    batch_spectra = np.random.rand(num_iterations, N).astype(np.float64)

    # Measure pure calculation time
    start_time = time.perf_counter()
    
    # Option A: Batch processing (Fastest for vectorization)
    xyz_results = converter.sd_to_xyz(batch_spectra)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"Processed {num_iterations} spectra.")
    print(f"Total time: {duration:.4f} seconds.")
    print(f"Time per spectrum: {duration/num_iterations*1e6:.4f} microseconds.")
    print(f"XYZ Shape: {xyz_results.shape}")
    print(f"First XYZ: {xyz_results[0]}")
    
    # Check consistency (Perfect White Reflector)
    perfect_white = np.ones(N)
    white_xyz = converter.sd_to_xyz(perfect_white)
    print(f"\nPerfect White Y value (Target 100.0): {white_xyz[1]:.4f}")