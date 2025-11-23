# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""
import numpy as np
from numba import njit, prange, complex128, float64, int32, boolean

# --- Constants ---
POL_S = 0
POL_P = 1

# --- JIT Compiled Core Functions ---

@njit(cache=True, inline='always')
def mat_mul_2x2_complex(a, b):
    """
    Multiplies two 2x2 complex matrices representing transfer operators for coherent wave propagation.

    This function is essential for calculating the combined effect of multiple layers in stratified media,
    such as thin films or multilayers. In X-ray and neutron reflectometry, these matrices encode Fresnel
    reflection/transmission coefficients and phase shifts due to layer thicknesses. The complex numbers
    are necessary because wave fields have both amplitude (magnitude) and phase (imaginary component).

    Args:
        a: First 2x2 complex matrix (dtype=complex128) representing the first transfer operator.
        b: Second 2x2 complex matrix (dtype=complex128) representing the second transfer operator.

    Returns:
        The product matrix R = A·B as a 2x2 complex matrix. This represents how the wave field
        transforms when passing through two sequential layers.

    Physical Context:
    In coherent scattering theories, the relationship between incoming and outgoing waves at an interface is described by:

        [E⁺_j]   = M₁·M₂·...·[E⁺_0]
        [E⁻_j]          [E⁻_0]

    where each 2x2 matrix Mᵢ represents one layer's transfer operator, and (E⁺, E⁻) are the
    complex wave amplitudes in opposite directions. The product of matrices accumulates these effects.
    """
    res = np.empty((2, 2), dtype=complex128)
    res[0, 0] = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0]
    res[0, 1] = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1]
    res[1, 0] = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0]
    res[1, 1] = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1]
    return res

@njit(cache=True, inline='always')
def mat_mul_2x2_real(a, b):
    """
    Multiplies two 2x2 real matrices representing intensity transfer operators for incoherent systems.

    This function is used when modeling systems where wave phase information is lost (due to short coherence length,
    absorption, or diffuse scattering), and only intensities matter. The matrix elements typically represent
    probabilities, absorption coefficients, or intensity factors that combine linearly through addition/multiplication.

    Args:
        a: First 2x2 real matrix (dtype=float64) representing the first intensity transfer operator.
        b: Second 2x2 real matrix (dtype=float64) representing the second intensity transfer operator.

    Returns:
        The product matrix R = A·B as a 2x2 real matrix. This represents how the intensity transforms
        when passing through two sequential incoherent processes.

    Physical Context:
    In incoherent scattering models, the relationship between incoming and outgoing intensities at an interface is described by:

        [I_j]   = N₁·N₂·...·[I_0]
        [Q_j]          [Q_0]

    where each 2x2 matrix Nᵢ represents one layer's intensity operator. I and Q typically represent
    the reflected/transmitted intensity and some auxiliary quantity (e.g., polarization), and
    the real-valued matrices contain absorption coefficients or diffuse scattering factors.
    """
    res = np.empty((2, 2), dtype=float64)
    res[0, 0] = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0]
    res[0, 1] = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1]
    res[1, 0] = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0]
    res[1, 1] = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1]
    return res

@njit(cache=True, inline='always')
def w_function(q, rough_type):
    """
    Calculates the roughness factor W(q), also known as the Debye-Waller-like factor or
    interfacial form factor, for optical thin films with various interface profiles.

    This function models how surface roughness affects reflectivity by providing a
    decay factor that depends on the momentum transfer q and the assumed interface profile.
    The implemented models are commonly used in X-ray and neutron reflectometry to account
    for non-ideal interfaces.

    Args:
        q (float or complex): Momentum transfer perpendicular to the interface, typically
            denoted as q_z with units of inverse length (Å⁻¹).
        rough_type (int): Integer specifying the type of roughness model. Supported values:
            - 0: No roughness (sharp interface)
            - 1: Linear profile
            - 2: Step function
            - 3: Exponential decay
            - 4: Gaussian profile

    Returns:
        complex: The calculated W(q) factor, which is real-valued in all cases (imaginary part = 0).

    Roughness Models:

    Type 0: None / Ideal Interface
        Profile: Sharp step function
        Formula: W(q) = **1.0 + 0j**
        Explanation: Represents a perfectly sharp interface with no roughness. This is the baseline case where reflectivity follows Fresnel equations.

    Type 1: Linear Profile (Triangle Form Factor)
        Profile: Linear change in scattering length density over finite thickness
        Formula: W(q) = **sin(√3 q) / (√3 q)**
        Explanation: Derived from Fourier transform of rectangular derivative profile. The √3 factor normalizes the width parameter to standard deviation σ (L = 2σ√3).

    Type 2: Step Profile (Cosine Form Factor)
        Profile: Heaviside step function derivative
        Formula: W(q) = **cos(q)**
        Explanation: Less common for intrinsic roughness, sometimes used in specific approximations or composite structures.

    Type 3: Exponential Profile
        Profile: Exponential decay of scattering length density
        Formula: W(q) = **1 / (1 + q²/2)**
        Explanation: Results from Lorentzian derivative profile. Used for interfaces with long-range mixing, like degraded surfaces.

    Type 4: Gaussian Profile (Debye-Waller Factor)
        Profile: Error function (integrated Gaussian), height fluctuations follow Gaussian statistics
        Formula: W(q) = **e^(-q²/2)**
        Explanation: Classic form assuming surface heights are normally distributed. Often written as e^(-q²σ²/2) with σ=1 here.
    """

    if rough_type == 0: # None
        return 1.0 + 0j

    # Linear (Default in Code 2)
    if rough_type == 1: 
        factor = 1.73205080757 # sqrt(3)
        val = q * factor
        # Avoid div by zero
        if np.abs(val) < 1e-9:
            return 1.0 + 0j
        return np.sin(val) / val

    # Step
    elif rough_type == 2: 
        return np.cos(q)

    # Exponential
    elif rough_type == 3: 
        return 1.0 / (1.0 + (q**2) / 2.0)

    # Gaussian / Error
    elif rough_type == 4: 
        return np.exp(-(q**2) / 2.0)
        
    return 1.0 + 0j
    if rough_type == 0: # None
        return 1.0 + 0j
    
    # Linear (Default in Code 2)
    if rough_type == 1: 
        factor = 1.73205080757 # sqrt(3)
        val = q * factor
        # Avoid div by zero
        if np.abs(val) < 1e-9:
            return 1.0 + 0j
        return np.sin(val) / val
    
    # Step
    elif rough_type == 2: 
        return np.cos(q)
    
    # Exponential
    elif rough_type == 3: 
        return 1.0 / (1.0 + (q**2) / 2.0)
    
    # Gaussian / Error
    elif rough_type == 4: 
        return np.exp(-(q**2) / 2.0)
        
    return 1.0 + 0j

@njit(cache=True, inline='always')
def mod_matr(S):
    """
    Converts a coherent scattering matrix to an intensity matrix for incoherent light propagation.

    This function transforms a 2x2 complex scattering matrix S (representing field amplitudes)
    into a real-valued modified intensity matrix M that describes how intensities propagate
    in systems where wave phase information is lost, such as thick layers or with incoherent sources.

    Mathematical Definitions:

    Input:  The coherent scattering matrix relates complex fields:
        [E_out⁺]   = S * [E_in⁺]
        [E_in⁻]           [E_out⁻]

    Output: The intensity matrix maps intensities (I ∝ |E|²):
        [I_trans]   = M * [I_inc]
        [I_back]            [I_ref]

    where I_trans is transmitted intensity, I_back is back-reflected intensity,
    I_inc is incident intensity, and I_ref is reflected intensity from the layer below.

    Physical Context:
    In systems with thick layers or incoherent light sources, phase information is lost. This
    function converts the coherent field description to an incoherent intensity description using
    the Katsidis & Siapkas formalism. The matrix elements are constructed from the squared magnitudes
    of S's elements and its determinant, ensuring energy conservation.

    Args:
        S: Complex 2x2 numpy array (dtype=complex128) representing the coherent scattering matrix.

    Returns:
        M: Real 2x2 numpy array (dtype=float64) with elements derived from |S_ij|² and det(S).

    References:
        Katsidis, C. C., & Siapkas, D. I. (2002). Applied Optics, 41(19), 3978-3987.
        Optimized Modified Matrix calculation.
    1. Removes sqrt() calls for speed (using manual norm squared).
    2. Stabilizes M[1, 1] calculation for absorbing/lossy media.
    """
    M = np.empty((2, 2), dtype=float64)
    
    # Extract components
    s00 = S[0, 0]
    s01 = S[0, 1]
    s10 = S[1, 0]
    s11 = S[1, 1]
    
    # helper: fast complex norm squared (avoids sqrt)
    # |z|^2 = Re(z)^2 + Im(z)^2
    mag_s00_sq = s00.real*s00.real + s00.imag*s00.imag
    mag_s01_sq = s01.real*s01.real + s01.imag*s01.imag
    mag_s10_sq = s10.real*s10.real + s10.imag*s10.imag
    
    # M00 = |1/t|^2
    M[0, 0] = mag_s00_sq
    
    # M01 = -|r_back/t|^2
    M[0, 1] = -mag_s01_sq
    
    # M10 = |r/t|^2
    M[1, 0] = mag_s10_sq
    
    # M11 calculation using determinant identity
    # Formula: (|det S|^2 - |s01 * s10|^2) / |s00|^2
    # This handles absorption correctly where |det S| != 1
    
    det_real = s00.real*s11.real - s00.imag*s11.imag - (s01.real*s10.real - s01.imag*s10.imag)
    det_imag = s00.real*s11.imag + s00.imag*s11.real - (s01.real*s10.imag + s01.imag*s10.real)
    mag_det_sq = det_real*det_real + det_imag*det_imag
    
    prod_s01s10 = s01 * s10
    mag_prod_sq = prod_s01s10.real*prod_s01s10.real + prod_s01s10.imag*prod_s01s10.imag
    
    numerator = mag_det_sq - mag_prod_sq
    
    # Guard against division by zero (though S[0,0] -> 0 implies infinite transmission)
    if mag_s00_sq > 1e-16:
        M[1, 1] = numerator / mag_s00_sq
    else:
        M[1, 1] = 0.0
        
    return M

@njit(cache=True)
def solve_coherent_chunk(start_idx, end_idx, n_arr, d_arr, rough_arr, rough_type_arr, 
                         lam, NSinFi, pol):
    """
    Calculates the scattering matrix for a stack of coherent layers using Transfer Matrix Method (TMM).

    This function computes the combined effect of interfaces and layer propagations within
    a specified range of layers. It accounts for material properties, layer thicknesses,
    interfacial roughness, and wave polarization to determine how light interacts with stratified media.

    Physical Context:
    The TMM is fundamental in optics and reflectometry, modeling coherent wave propagation through layered materials.
    The solution is built by sequentially multiplying 2×2 complex matrices that represent:
    - Interface effects (reflection/transmission) between layers
    - Phase shifts as waves propagate through layer thicknesses

    Parameters:
        start_idx: First layer index in the stack (inclusive)
        end_idx: Last layer index + 1 (exclusive)
        n_arr: Array of complex refractive indices or scattering length densities for each layer
        d_arr: Array of real layer thicknesses (in same units as wavelength)
        rough_arr: Array of roughness parameters σ (standard deviations) for interfaces
        rough_type_arr: Array of integers specifying roughness model types (0-5)
        lam: Wavelength of incident radiation (same units as thickness, typically Å or nm)
        NSinFi: Snell's law invariant = N₀ sin(φ₀), with N₀ = incident medium index
        pol: Polarization state (P=1 for parallel, S=0 for perpendicular to plane of incidence)

    Returns:
        A 2×2 complex scattering matrix S that relates incoming and outgoing wave amplitudes:
            [E⁺_out]   = S₁₁·E⁺_in + S₁₂·E⁻_out
            [E⁻_in]     S₂₁·E⁺_in + S₂₂·E⁻_out
        From this, the reflectivity R = |S₂₁/S₁₁|² and transmittivity T = 1-|r|² can be calculated.

    Implementation Details:
    - Matrices are accumulated from bottom (substrate) to top for numerical stability.
    - Fresnel coefficients r and t are computed for each interface, accounting for polarization.
    - Roughness factors α, β, γ modify these coefficients based on the specified model (0-5):
        * Models 0-4 use q-vector dependent factors via w_function()
        * Model 5 uses Nevot-Croce exponential decay factor uniformly
    - Phase shifts are applied to waves propagating through each layer thickness.
    - Numerical safeguards limit imaginary parts of phases to ±50 to prevent overflow.
    """
    # Initialize Identity
    S = np.eye(2, dtype=complex128)
    
    # Loop Interfaces Backwards (standard TMM accumulation)
    # Range: end_idx-1 down to start_idx
    # An interface K exists between Layer K and K+1
    
    count = end_idx - start_idx
    
    for k in range(count - 1, -1, -1):
        idx = start_idx + k
        
        # --- Interface Calculation (Layer idx -> Layer idx+1) ---
        N1 = n_arr[idx]
        N2 = n_arr[idx+1]
        sigma = rough_arr[idx+1]
        rtype = rough_type_arr[idx+1]
        
        # Cosines derived from Snell's invariant NSinFi
        # cos = sqrt(1 - (N0sinFi/N)^2)
        cos1 = np.sqrt(1.0 - (NSinFi/N1)**2)
        cos2 = np.sqrt(1.0 - (NSinFi/N2)**2)
        
        # Fresnel Coefficients
        if pol == POL_P:
            num_r = N2 * cos1 - N1 * cos2
            den_r = N2 * cos1 + N1 * cos2
            num_t = 2.0 * N1 * cos1
            # den_t = den_r
        else: # POL_S
            num_r = N1 * cos1 - N2 * cos2
            den_r = N1 * cos1 + N2 * cos2
            num_t = 2.0 * N1 * cos1
            
        r = num_r / den_r
        t = num_t / den_r
        
        # Roughness Factors (w_function)
        two_pi_lam = (2.0 * np.pi) / lam
        
        # --- NEW ROUGHNESS FACTOR CALCULATION ---
        if rtype == 5: # Nevot-Croce Factor (Simplified)
            k0 = two_pi_lam # 2*pi/lambda
            kz1 = N1 * cos1
            kz2 = N2 * cos2
            
            # The Nevot-Croce factor: exp(-2.0 * (k0 * sigma)**2 * kz1 * kz2)
            # This is typically used to replace the gamma^2 term, 
            # or is applied more generally. Assuming it applies to all factors 
            # for a simplified implementation.
            nc_factor = np.exp(-2.0 * (k0 * sigma)**2 * kz1 * kz2)
            
            al = nc_factor
            be = nc_factor
            ga = nc_factor # Applying it to all factors
            
        else: # Existing Code 2 Roughness Models (including Gaussian)
            # q vectors
            q_al = two_pi_lam * 2 * N1 * cos1 * sigma
            q_be = two_pi_lam * 2 * N2 * cos2 * sigma
            q_ga = two_pi_lam * (N1 * cos1 - N2 * cos2) * sigma
            
            al = w_function(q_al, rtype)
            be = w_function(q_be, rtype)
            ga = w_function(q_ga, rtype)
        
        # Interface Matrix I
        gat = ga * t
        inv_gat = 1.0 / gat
        
        I_mat = np.empty((2, 2), dtype=complex128)
        I_mat[0, 0] = inv_gat
        I_mat[0, 1] = be * r * inv_gat
        I_mat[1, 0] = al * r * inv_gat
        I_mat[1, 1] = (ga**2 * (1.0 - r**2) + al * be * r**2) * inv_gat
        
        S = mat_mul_2x2_complex(I_mat, S)
        
        # --- Phase Matrix Calculation ---
        # Apply phase of layer 'idx' IF it is not the very first layer of this chunk
        if k > 0:
            d = d_arr[idx]
            # beta = 2pi/lam * d * N * cos
            beta = two_pi_lam * d * N1 * cos1
            
            # Overflow protection (from Code 2)
            if beta.imag < -50.0: beta = complex(beta.real, -50.0)
            elif beta.imag > 50.0: beta = complex(beta.real, 50.0)
            
            P_mat = np.zeros((2, 2), dtype=complex128)
            P_mat[0, 0] = np.exp(-1j * beta)
            P_mat[1, 1] = np.exp(1j * beta)
            
            S = mat_mul_2x2_complex(P_mat, S)
            
    return S

@njit(parallel=True, fastmath=True, cache=True)
def core_engine(wavls, theta_rad, n_layers, indices, thicknesses, 
                incoherent_flags, rough_types, rough_vals, 
                calc_s, calc_p):
    """
    Computes polarized reflectance (R) and transmittance (T) spectra for a multilayer structure.

    This function handles both coherent and incoherent light propagation through layered materials
    using a hybrid matrix method that combines:
    - Transfer Matrix Method (TMM) for coherent chunks of thin layers
    - Modified intensity matrices for incoherent boundaries

    The calculation is parallelized across wavelengths, making it efficient for spectral analysis.

    Parameters:
        wavls: float array, wavelengths in [length units] to calculate spectra for
        theta_rad: float, angle of incidence in radians
        n_layers: int, total layers including ambient and substrate
        indices: complex ndarray (n_layers, len(wavls)), refractive indices per layer at each wavelength
        thicknesses: float array (n_layers), physical thickness of each layer [length units]
        incoherent_flags: bool array (n_layers), True for layers that break coherence
        rough_types: int array (n_layers), roughness model type for interfaces below each layer
        rough_vals: float array (n_layers), roughness σ values [length units] for interfaces
        calc_s (bool): True if S-polarization should be calculated.
        calc_p (bool): True if P-polarization should be calculated.

    Returns:
        Rs_out, Rp_out: float arrays of same length as wavls, reflectance for S and P polarizations
        Ts_out, Tp_out: float arrays of same length as wavls, transmittance for S and P polarizations

    Physical Context:
    The function models real-world systems where some layers maintain coherence (thin films)
    while others break it (thick substrates). It combines coherent field calculations with
    incoherent intensity transport to accurately predict optical properties.

    Parallelization: The wavelength loop is parallelized for efficient broadband calculations.
    """
    num_wavs = len(wavls)
    
    # Initialize all arrays to zero. They will be filled only if requested.
    Rs_out = np.zeros(num_wavs, dtype=float64)
    Rp_out = np.zeros(num_wavs, dtype=float64)
    Ts_out = np.zeros(num_wavs, dtype=float64)
    Tp_out = np.zeros(num_wavs, dtype=float64)
    
    # Identify Layer 0 (Ambient) and Layer N (Final Medium)
    idx_N = n_layers - 1
    
    for w in prange(num_wavs):
        lam = wavls[w]
        N0 = indices[0, w]      # Ambient Refractive Index
        NN = indices[idx_N, w]  # Final Medium Refractive Index
        
        # Snell's Invariant: N₀ sin(φ₀)
        NSinFi = N0 * np.sin(theta_rad)
        
        # Pre-calculate angular factors for Ambient (Layer 0) and Final Medium (Layer N)
        cos0 = np.sqrt(1.0 - (NSinFi/N0)**2)
        cosN = np.sqrt(1.0 - (NSinFi/NN)**2)
        
        # --- 1. S POLARIZATION PATH (Perpendicular) ---
        if calc_s:
            M_total_s = np.eye(2, dtype=float64)
            start_node = 0
            
            for i in range(n_layers):
                if incoherent_flags[i] or i == idx_N:
                    
                    # Calculate Coherent S-Matrix for chunk [start_node ... i]
                    S_coh = solve_coherent_chunk(start_node, i, indices[:, w], thicknesses, 
                                                 rough_vals, rough_types, lam, NSinFi, POL_S)
                    M_coh = mod_matr(S_coh)
                    M_total_s = mat_mul_2x2_real(M_coh, M_total_s)
                    
                    # Propagate through the Incoherent Layer 'i' (if it exists)
                    if i < idx_N:
                        N_inc = indices[i, w]
                        d_inc = thicknesses[i]
                        cos_inc = np.sqrt(1.0 - (NSinFi/N_inc)**2)
                        
                        beta = (2.0 * np.pi * d_inc / lam) * N_inc * cos_inc
                        term1 = np.abs(np.exp(-1j * beta))**2
                        term2 = np.abs(np.exp(1j * beta))**2
                        
                        M_inc = np.zeros((2, 2), dtype=float64)
                        M_inc[0, 0] = term1; M_inc[1, 1] = term2
                        M_total_s = mat_mul_2x2_real(M_inc, M_total_s)
                    
                    start_node = i
            
            # --- Extract S results ---
            m00 = M_total_s[0, 0]
            m10 = M_total_s[1, 0]
            
            if m00 > 1e-15:
                R_val = m10 / m00
                T_raw = 1.0 / m00
                
                # T Normalization: Re(N_N * cosN) / Re(N0 * cos0)
                # This factor correctly scales the intensity to energy flux (Poynting vector)
                # for any medium (absorbing or non-absorbing).
                numer = np.real(NN * cosN)
                denom = np.real(N0 * cos0)
                
                # Check for zero denominator (e.g., TIR boundary condition)
                if denom > 1e-12:
                    factor = numer / denom
                else:
                    factor = 0.0

                Rs_out[w] = R_val
                Ts_out[w] = T_raw * factor


        # --- 2. P POLARIZATION PATH (Parallel) ---
        if calc_p:
            M_total_p = np.eye(2, dtype=float64)
            start_node = 0
            
            for i in range(n_layers):
                if incoherent_flags[i] or i == idx_N:
                    
                    # Calculate Coherent S-Matrix for chunk [start_node ... i]
                    S_coh = solve_coherent_chunk(start_node, i, indices[:, w], thicknesses, 
                                                 rough_vals, rough_types, lam, NSinFi, POL_P)
                    M_coh = mod_matr(S_coh)
                    M_total_p = mat_mul_2x2_real(M_coh, M_total_p)
                    
                    # Propagate through the Incoherent Layer 'i' (if it exists)
                    if i < idx_N:
                        N_inc = indices[i, w]
                        d_inc = thicknesses[i]
                        cos_inc = np.sqrt(1.0 - (NSinFi/N_inc)**2)
                        
                        beta = (2.0 * np.pi * d_inc / lam) * N_inc * cos_inc
                        term1 = np.abs(np.exp(-1j * beta))**2
                        term2 = np.abs(np.exp(1j * beta))**2
                        
                        M_inc = np.zeros((2, 2), dtype=float64)
                        M_inc[0, 0] = term1; M_inc[1, 1] = term2
                        M_total_p = mat_mul_2x2_real(M_inc, M_total_p)
                    
                    start_node = i
            
            # --- Extract P results ---
            m00 = M_total_p[0, 0]
            m10 = M_total_p[1, 0]
            
            if m00 > 1e-15:
                R_val = m10 / m00
                T_raw = 1.0 / m00

                # T Normalization: Re(N_N * cosN) / Re(N0 * cos0)
                numer = np.real(NN * cosN)
                denom = np.real(N0 * cos0)
                
                if denom > 1e-12:
                    factor = numer / denom
                else:
                    factor = 0.0

                Rp_out[w] = R_val
                Tp_out[w] = T_raw * factor
                
    return Rs_out, Rp_out, Ts_out, Tp_out

# --- Python Class Wrapper ---

class FastScatterMatrix:
    def __init__(self, layer_indices, thicknesses, incoherent_flags, roughness_params, wavls, theta):
        """
        Prepares physical parameters for fast spectral calculations of multilayer optical structures.

        This class preprocesses and validates input data for the optimized `core_engine`
        function, which uses a hybrid coherent-incoherent matrix method to calculate
        polarized reflectance (R) and transmittance (T) spectra.

        Parameters:
            layer_indices: complex ndarray (n_layers × n_wavs)
                Complex refractive indices [N = n + ik] for each layer at all wavelengths
            thicknesses: float ndarray (n_layers)
                Physical thickness of layers [length units, same as wavls]
                Ambient and substrate should have thickness=0
            incoherent_flags: bool ndarray (n_layers)
                True for layers that break phase coherence (e.g., thick substrates)
            roughness_params: list of tuples [(type, sigma), ...]
                For each interface below a layer:
                - type: int, roughness model identifier (0-5)
                - sigma: float, roughness magnitude [length units]
            wavls: float ndarray
                Wavelengths at which to calculate spectra [same length units as thicknesses]
            theta: float
                Angle of incidence in degrees

        Attributes:
            theta: converted angle of incidence in radians for internal calculations
            n_layers: total number of layers including ambient and substrate
            indices, thicknesses, inc_flags: validated input arrays for core engine
            r_types, r_vals: extracted roughness parameters as contiguous arrays
        """
        self.wavls = np.ascontiguousarray(wavls, dtype=np.float64)
        self.theta = np.radians(theta)
        self.n_layers = len(thicknesses)
        
        # Arrays
        self.indices = np.ascontiguousarray(layer_indices, dtype=np.complex128)
        self.thicknesses = np.ascontiguousarray(thicknesses, dtype=np.float64)
        self.inc_flags = np.ascontiguousarray(incoherent_flags, dtype=np.bool_)
        
        # Parse Roughness
        r_types = []
        r_vals = []
        for r_t, r_v in roughness_params:
            r_types.append(r_t)
            r_vals.append(r_v)
            
        self.r_types = np.ascontiguousarray(r_types, dtype=np.int32)
        self.r_vals = np.ascontiguousarray(r_vals, dtype=np.float64)
        
    def compute(self, mode='u'):
        """
        Calculates polarized reflectance (R) and transmittance (T) spectra based 
        on requested polarization mode..
        
        This method executes the Numba-optimized `core_engine` with preprocessed parameters,
        returning spectral results for S (perpendicular) or P (parallel) polarizations
        as well as unpolarized averages.
        
        Args:
            mode (str): 's', 'p', or 'u' (default).
                        'unpolarized' computes both S and P and averages them
        
        Returns:
            dict: Contains calculated spectral quantities, each an array of same length as wavls:
                - Rs, Rp: Reflectance for S and P polarization respectively
                - Ts, Tp: Transmittance for S and P polarization respectively
                - Ru, Tu: Unpolarized reflectance and transmittance averages
        
        Note: The computation is performed in parallel across wavelengths by core_engine.
        """
        calc_s = False
        calc_p = False
        
        if mode.lower() == 's':
            calc_s = True
        elif mode.lower() == 'p':
            calc_p = True
        else: # Unpolarized or Both
            calc_s = True
            calc_p = True

        Rs, Rp, Ts, Tp = core_engine(
            self.wavls, self.theta, self.n_layers, self.indices, 
            self.thicknesses, self.inc_flags, self.r_types, self.r_vals,
            calc_s, calc_p
        )
        
        res = {}
        if calc_s:
            res['Rs'] = Rs
            res['Ts'] = Ts
        if calc_p:
            res['Rp'] = Rp
            res['Tp'] = Tp
        if calc_s and calc_p:
            res['Ru'] = (Rs + Rp)/2.0
            res['Tu'] = (Ts + Tp)/2.0
            
        return res

if __name__ == "__main__":
    import time
    
    print("--- Setting up Mock Data ---")
    
    # 1. Mock Data Setup
    n_wavs = 1000
    wavls = np.linspace(300, 800, n_wavs) # Wavelengths in nm
    
    # Target wavelength for Quarter-Wave Stack design
    lambda_0 = 550.0 
    
    # Ambient and Substrate
    n_air = np.full(n_wavs, 1.0 + 0.0j, dtype=np.complex128)      # Air
    n_bk7 = np.full(n_wavs, 1.515 + 0.0j, dtype=np.complex128)    # Substrate (BK7 Glass)
    
    # Layer Materials
    n_sio2 = np.full(n_wavs, 1.46 + 0.0j, dtype=np.complex128)   # Low Index (L)
    n_ta2o5 = np.full(n_wavs, 2.10 + 0.0j, dtype=np.complex128)  # High Index (H)
    
    # Thicknesses for Quarter-Wave Stack at lambda_0=550nm
    # d = lambda_0 / (4 * n)
    d_sio2 = lambda_0 / (4.0 * 1.46)
    d_ta2o5 = lambda_0 / (4.0 * 2.10)
    
    # 2. Build Structure List
    # Format: [Thickness, N_Array, Incoherent_Bool, Roughness_Float]
    structure = []
    
    # Layer 0: Ambient (Air, Semi-infinite)
    structure.append([0.0, n_air, False, 0.0])

    # Layers 1 to 20: Alternating Ta2O5 (H) and SiO2 (L)
    num_pairs = 10
    
    for i in range(num_pairs):
        # Ta2O5 (High Index Layer)
        structure.append([d_ta2o5, n_ta2o5, False, 0.1]) # Layer H
        # SiO2 (Low Index Layer)
        structure.append([d_sio2, n_sio2, False, 0.1])   # Layer L
        
    # Layer 21: Substrate (BK7, Semi-infinite)
    structure.append([0.0, n_bk7, False, 0.0])
    
    # 3. Simulation Parameters
    fi_rad = np.deg2rad(45.0) # 45 degrees incidence angle
    # --- BRIDGE: Convert User List to Class Inputs ---
    indices_list = []
    thick_list = []
    rough_list = []
    inc_list = [] # <--- Added: Need to extract incoherent flags for new class
    
    for layer in structure:
        thick_list.append(layer[0])
        indices_list.append(layer[1])
        inc_list.append(layer[2]) # <--- Extract Incoherent Bool
        
        # Map float roughness to (Type 1=Linear, Value)
        r_val = layer[3]
        if r_val > 0:
            rough_list.append((1, r_val)) 
        else:
            rough_list.append((0, 0.0))
            
    indices_arr = np.vstack(indices_list)
    thick_arr = np.array(thick_list)
    inc_arr = np.array(inc_list, dtype=np.bool_) # <--- Convert to numpy array
    # --------------------------------------------------

    # Instantiate the Class
    # Note: New class calculates all polarizations, so pol_type arg is removed
    solver = FastScatterMatrix(indices_arr, thick_arr, inc_arr, rough_list, wavls, np.degrees(fi_rad))

    print("--- Starting Numba JIT Compilation (First Run) ---")
    t_compile_start = time.time()
    _ = solver.compute()
    print(f"First run (Compile + Exec) time: {time.time() - t_compile_start:.4f}s")

    print("\n--- Starting Benchmark (Second Run) ---")
    t0 = time.time()
    
    # Calculate
    for i in range(1000):
        res = solver.compute()
    
        # Map Dictionary results to usage example variables
        Rs, Rp, R_avg = res['Rs'], res['Rp'], res['Ru']
        Ts, Tp, T_avg = res['Ts'], res['Tp'], res['Tu']
    
    dt = time.time() - t0
    print(f"Calculation for {n_wavs} wavelengths took: {dt:.6f}s")
    print(f"Speed: {n_wavs / dt:.2f} wavelengths/sec")

    # 5. Display Sample Results
    print("\n--- Sample Results (Every 100th point) ---")
    print(f"{'Wavel (nm)':<12} | {'Rs':<10} | {'Rp':<10} | {'Trans (Avg)':<10}")
    print("-" * 50)
    for i in range(0, n_wavs, 100):
        print(f"{wavls[i]:<12.1f} | {Rs[i]:<10.5f} | {Rp[i]:<10.5f} | {T_avg[i]:<10.5f}")