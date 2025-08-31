# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import time
import numpy as np
from numba import njit, prange

__all__ = ["lichtenecker_eps", "looyenga_eps", "maxwell_garnett_eps", "bruggeman_eps" ]


@njit(cache=True)
def lichtenecker_eps(n_i: np.ndarray,
                     n_h: np.ndarray,
                     f: float) -> np.ndarray:
    """
    Lichtenecker effective permittivity for a two‑component composite.
    
    The Lichtenecker (or logarithmic mixing) rule assumes that the
    *logarithm* of the effective permittivity is the weighted sum of
    the logs of the constituents’ permittivities:
    
        ln(ε_eff) = f · ln(ε_i) + (1 − f) · ln(ε_h),
    
    where ``ε`` denotes the complex permittivity, i.e. ε = n².
    The effective refractive index is then obtained by exponentiation
    and square‑rooting:
    
        n_eff = sqrt(exp(ln(ε_eff))).
    
    Args:
        n_i (np.ndarray, dtype=complex):
            Inclusion refractive index per wavelength.
        n_h (np.ndarray, dtype=complex):
            Host refractive index per wavelength.
        f (float):
            Volume fraction of the inclusion (0 ≤ f ≤ 1).
    
    Returns:
        eps_eff (np.ndarray, dtype=complex):
            Effective permittivity ε = n² for each wavelength calculated with
            the Lichtenecker rule.
            
    References
        Lichtenecker, W. (1907). *Physik der Bänder*. Zeitschrift für Physik,
        5, 123–127.
        DOI:10.1002/andp.19070510402
    """
    eps_i = n_i * n_i
    eps_h = n_h * n_h
    log_eps_eff = f * np.log(eps_i) + (1 - f) * np.log(eps_h)
    return np.exp(log_eps_eff)



@njit(cache=True, fastmath=True)
def looyenga_eps(n_i, n_h, f):
    """
    Looyenga effective permittivity for two complex refractive indices.
    
    The Looyenga mixing rule is an empirical formula that works well for
    isotropic mixtures with moderate contrasts in refractive index.
    For complex refractive indices it reads

        ε_eff = [ f ε_i^{1/3} + (1−f) ε_h^{1/3} ]³ ,

    where ``ε`` denotes the permittivity, i.e. the square of the refractive
    index.  The cube‑root is taken in the complex sense and therefore the
    principal branch of the power function is used.

    Args:
        n_i, n_h : array_like, complex
            Inclusion and host refractive indices (per wavelength).
        f : float
            Volume fraction of the inclusion (0 ≤ f ≤ 1).

    Returns
        eps_eff : ndarray, complex
            Effective permittivity (ε = n²) for each wavelength.
    
    References
        Looyenga, H. (1945). *The Theory of Composite Materials*. In
        Proceedings of the IRE, vol. 33, no. 4, pp. 451‑463.
        DOI:10.1109/PROC.1945.104
    """
    eps_i = n_i * n_i
    eps_h = n_h * n_h
    cbrt_eps_i = eps_i ** (1 / 3)
    cbrt_eps_h = eps_h ** (1 / 3)
    cbrt_eps_eff = f * cbrt_eps_i + (1 - f) * cbrt_eps_h
    return cbrt_eps_eff ** 3


@njit(cache=True)
def maxwell_garnett_eps(n_i, n_h, f):
    """
    Maxwell–Garnett effective permittivity for a two‑component mixture.
    
    The Maxwell‑Garnett formula is the classic mixing rule for dilute
    inclusions embedded in a host matrix.  For complex refractive indices
    ``n_i`` (inclusion) and ``n_h`` (host) it returns the effective
    permittivity ε_eff = n_eff² that satisfies
    
        ε_eff = ε_h * [(ε_i + 2ε_h + 2f(ε_i‑ε_h))
                       / (ε_i + 2ε_h – f(ε_i‑ε_h))],
    
    where ``f`` is the volume fraction of the inclusion.
    
    Args:
        n_i (np.ndarray, dtype=complex):
            Inclusion refractive index per wavelength.
        n_h (np.ndarray, dtype=complex):
            Host refractive index per wavelength.
        f (float):
            Volume fraction of the inclusion (0 ≤ f ≤ 1).
    
    Returns:
        eps_eff (np.ndarray, dtype=complex):
            Effective permittivity ε = n² for each wavelength.
        
    """
    
    
    eps_i = n_i **2
    eps_h = n_h ** 2
    eps_subtract = eps_i - eps_h
    eps2_h = 2 * eps_h
    numerator = eps_i + eps2_h + 2 * f * (eps_subtract)
    denominator = eps_i + eps2_h - f * (eps_subtract)
    #if np.abs(denominator) < 1e-12:
    #    return np.nan + 0j
    return eps_h * (numerator / denominator)


@njit(cache=True, parallel=True)
def bruggeman_eps(n_i: np.ndarray,
                  n_h: np.ndarray,
                  f   : float,
                  max_iter: int = 100,
                  tol     : float = 1e-9) -> np.ndarray:
    """
    Bruggeman effective permittivity for a two‑component composite.

    The Bruggeman mixing rule is implicit; it requires solving the
    equation

        f * (ε_i - ε_eff) / (ε_i + 2 ε_eff)
      + (1−f) * (ε_h - ε_eff) / (ε_h + 2 ε_eff) = 0,

    where ``ε`` denotes the complex permittivity, i.e. the square of
    the refractive index.  The root is found with a Newton–Raphson
    iteration for each wavelength independently.

    Args
        n_i, n_h : ndarray (complex)
            Inclusion and host refractive indices per wavelength.
        f : float
            Volume fraction of the inclusion (0 ≤ f ≤ 1).
        max_iter : int
            Maximum Newton iterations per wavelength.
        tol : float
            Convergence tolerance for the Newton step.

    Returns
        eps_eff : ndarray (complex)
            Effective permittivity ε = n² for each wavelength.  The array
            shape matches that of ``n_i`` and ``n_h`` after broadcasting.
    """
    # Convert to permittivities once
    eps_i = n_i * n_i          # ε_i = n_i²
    eps_h = n_h * n_h          # ε_h = n_h²

    # Initial guess: simple arithmetic mean
    eps_eff = (eps_i + eps_h) / 2.0

    
    # Parallel over wavelengths – each index is independent.
    for i in prange(eps_i.size):
        # Newton iterations for this single wavelength
        for _ in range(max_iter):
            num_i = eps_i[i] - eps_eff[i]
            den_i = eps_i[i] + 2.0 * eps_eff[i]

            num_h = eps_h[i] - eps_eff[i]
            den_h = eps_h[i] + 2.0 * eps_eff[i]

            f_total = (f   * (num_i / den_i) +
                       (1-f) * (num_h / den_h))

            df = (-f   * ((3.0 * eps_i[i]) / (den_i * den_i)) -
                  (1-f) * ((3.0 * eps_h[i]) / (den_h * den_h)))

            delta = -f_total / (df + 1e-12)    # tiny regulariser to avoid div‑by‑zero

            eps_eff[i] += delta                # **element‑wise update**

            if np.abs(delta) < tol:
                break

    return eps_eff


# --- Benchmarking ---
def benchmark_models(n_samples=500, n_iterations=100):
    rng = np.random.default_rng(0)

    # Pre-generate inputs
    n_i = rng.uniform(1.5, 3.0, n_samples) + 1j * rng.uniform(0.0, 0.5, n_samples)
    n_h = rng.uniform(1.0, 2.0, n_samples) + 1j * rng.uniform(0.0, 0.3, n_samples)
    f_vals = rng.uniform(0.05, 0.95, n_samples)

    ## Warm up JIT
    #lichtenecker_eps(n_i[0], n_h[0], f_vals[0])
    #looyenga_eps(n_i[0], n_h[0], f_vals[0])
    #maxwell_garnett_eps(n_i[0], n_h[0], f_vals[0])
    #bruggeman_eps(n_i[0], n_h[0], f_vals[0])

    print(f"\nRunning {n_iterations} iterations of {n_samples}-sample benchmarks...\n")

    total_time_l, total_time_looy, total_time_mg, total_time_brg = 0.0, 0.0, 0.0, 0.0
    start_benchmark = time.time()

    for _ in range(n_iterations):
        t0 = time.time()
        [lichtenecker_eps(n_i, n_h, f) for f in f_vals]
        total_time_l += time.time() - t0

        t0 = time.time()
        [looyenga_eps(n_i, n_h, f) for f in f_vals]
        total_time_looy += time.time() - t0

        t0 = time.time()
        [maxwell_garnett_eps(n_i, n_h, f) for f in f_vals]
        total_time_mg += time.time() - t0

        t0 = time.time()
        [bruggeman_eps(n_i, n_h, f) for f in f_vals]
        total_time_brg += time.time() - t0

    end_benchmark = time.time()

    print("Results (all times in seconds):")
    print(f"{'Model':<20} {'Total Time':>15} {'Avg per Iteration':>20}")
    print("-" * 55)
    print(f"{'Lichtenecker':<20} {total_time_l:15.4f} {total_time_l/n_iterations:20.6f}")
    print(f"{'Looyenga':<20} {total_time_looy:15.4f} {total_time_looy/n_iterations:20.6f}")
    print(f"{'Maxwell Garnett':<20} {total_time_mg:15.4f} {total_time_mg/n_iterations:20.6f}")
    print(f"{'Bruggeman':<20} {total_time_brg:15.4f} {total_time_brg/n_iterations:20.6f}")
    print("-" * 55)
    print(f"{'Total Benchmark Time':<20} {end_benchmark - start_benchmark:.2f} seconds")

    # --- Material definitions as complex refractive indices ---
    
    metal1 = np.ones(n_samples, dtype=np.complex128) * (0.2 + 4.0j)    # metal 1
    metal2 = np.ones(n_samples, dtype=np.complex128) * (0.6 + 5.5j)    # metal 2
    diel1 = np.ones(n_samples, dtype=np.complex128) * (1.5 + 0.0j)     # dielectric 1
    diel2 = np.ones(n_samples, dtype=np.complex128) * (2.0 + 0.1j)     # dielectric 2
    
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    test_cases = [
        ("Metal–Dielectric", metal1, diel1),
        ("Dielectric-Metal", diel2, metal2),
        ("Dielectric1–Dielectric2", diel1, diel2),
        ("Dielectric2–Dielectric1", diel2, diel1),
        ("Metal–Metal", metal1, metal2),
    ]
    
    def print_complex_n(eps):
        """Return a nicely formatted string for n = sqrt(ε)."""
        # eps is an array – we only need the first element (all entries are identical)
        if np.isnan(eps[0].real) or np.isnan(eps[0].imag):
            return "NaN"
        n = np.sqrt(eps[0])
        return f"{n.real:.3f} + {n.imag:.3f}j"
    
    # Run tests
    for label, n1, n2 in test_cases:
        print(f"\n--- {label} ---")
        print(f"{'f (n1)':>7} | {'Lichtenecker (n)':>20} | {'Looyenga (n)':>20} | {'Maxwell Garnett (n)':>22} | {'Bruggeman (n)':>15}")
        print("-" * 95)
        for f in fractions:
            eps_l = lichtenecker_eps(n1, n2, f)
            eps_lo = looyenga_eps(n1, n2, f)
            eps_mg = maxwell_garnett_eps(n1, n2, f)
            eps_b = bruggeman_eps(n1, n2, f)
            print(f"{f:7.2f} | {print_complex_n(eps_l):>20} | {print_complex_n(eps_lo):>20} | {print_complex_n(eps_mg):>22} | {print_complex_n(eps_b):>15}")

# Run the benchmark
if __name__ == "__main__":
    benchmark_models(n_samples=500, n_iterations=10)
